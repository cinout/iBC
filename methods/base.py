import copy
import os
import time
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from bcu.distillation import distillation
from mimic.model_train import mimic_model_train
from mimic.scheduler import weight_scheduler
from warmup_scheduler import GradualWarmupScheduler
from collections import Counter
from networks.resnet_org import model_dict
from networks.resnet_cifar import model_dict as model_dict_cifar
from utils.util import (
    AverageMeter,
    extract_backbone,
    get_feat_dim,
    save_model,
    update_seed,
)
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from networks.mask_batchnorm import MaskBatchNorm2d
import PIL
import random
from kornia import augmentation as aug
from rnp.maskprune import (
    test_maskprune,
    evaluate_by_threshold,
    read_data,
    save_mask_scores,
    refill_unlearned_model,
    train_step_recovering,
    train_step_unlearning,
)
from torch.utils.data import Subset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"


"""
The core function for calculating spectral signature

Input:
    visual_features: shape [bs*n_views, C=512], in numpy format

Return:
    max_indices_at_channel: the indices of channels with highest contribution to SS. In numpy format, shape [bs, n_view*take_channel]
"""


def ss_statistics(visual_features, bs, feat_dim, args):
    u, s, v = np.linalg.svd(
        visual_features - np.mean(visual_features, axis=0, keepdims=True),
        full_matrices=False,
    )

    # get top eigenvector
    eig_for_indexing = v[0:1]  # [1, C]

    corrs = np.matmul(
        eig_for_indexing, np.transpose(visual_features)
    )  # [1, bs*n_view], not .abs() yet.

    coeff_adjust = np.where(corrs > 0, 1, -1)  # [1, bs*n_view]
    coeff_adjust = np.transpose(coeff_adjust)  # [bs*n_view, 1]
    elementwise = (
        eig_for_indexing * visual_features * coeff_adjust
    )  # [bs*n_view, C]; if corrs is negative, then adjust its elements to reverse sign

    # get contributing indices sorted from low to high
    max_indices = np.argsort(
        elementwise, axis=1
    )  # [bs*n_view, C], C are indices, sorted by value from low to high

    max_indices = max_indices.reshape(bs, args.num_views, feat_dim)  # [bs, n_view, C]

    take_channel = args.voted_channel_num

    max_indices_at_channel = max_indices[
        :, :, -take_channel:
    ]  # [bs, n_view, take_channel]
    max_indices_at_channel = max_indices_at_channel.reshape(
        bs, -1
    )  # [bs, n_view*take_channel]

    return max_indices_at_channel


"""
To augment image into N views
"""


def generate_view_tensors(input, ss_transform):
    # input.shape: [total, 3, 32, 32]; value range: [0, 1]
    input = torch.permute(input, (0, 2, 3, 1))
    input = input * 255.0
    input = torch.clamp(input, 0, 255)
    input = np.array(
        input.cpu(), dtype=np.uint8
    )  # shape: [total, 32, 32, 3]; value range: [0, 255]

    view_tensors = []
    for img in input:
        img = PIL.Image.fromarray(img)  # in PIL format now
        views = ss_transform(
            img
        )  # a list of args.num_views elements, each one is a PIL image

        tensors_of_an_image = []
        for view in views:
            view = np.asarray(view).astype(np.float32) / 255.0
            view = torch.tensor(view)
            view = torch.permute(view, (2, 0, 1))  # shape: [c=3, h, w], value: [0, 1]
            tensors_of_an_image.append(view)
        tensors_of_an_image = torch.stack(
            tensors_of_an_image, dim=0
        )  # [num_views, c, h, w]
        view_tensors.append(tensors_of_an_image)

    view_tensors = torch.stack(
        view_tensors, dim=0
    )  # [total, num_views, c, h, w], value in [0,1]

    return view_tensors


"""
Return the estimated trigger channels.

Return:
    essential_indices: tensor of shape (#max_removed_channels,)
"""


def find_trigger_channels(
    args, data_loader, backbone, ss_transform, normalize_transform
):

    # store votes information
    all_votes = []

    # sample a few poisoned/clean images
    dataset = data_loader.dataset
    poisoned_indices = [
        i
        for i, (_, train_is_poisoned, _, _) in enumerate(dataset)
        if train_is_poisoned == 1
    ]
    clean_indices = [
        i
        for i, (_, train_is_poisoned, _, _) in enumerate(dataset)
        if train_is_poisoned == 0
    ]
    random_poisoned_indices = random.sample(
        poisoned_indices, args.find_channels_from_n_poison_samples
    )
    random_clean_indices = random.sample(
        clean_indices, args.find_channels_from_n_clean_samples
    )
    all_indices = [*random_poisoned_indices, *random_clean_indices]
    subset = Subset(dataset, all_indices)
    data_loader = DataLoader(
        subset, batch_size=args.linear_probe_batch_size, shuffle=False
    )

    # votes
    for i, content in tqdm(enumerate(data_loader)):
        images = content[0]
        images = images.to(device)

        if args.num_views == 1:
            views = images.clone()
            views = views.unsqueeze(1)
        else:
            views = generate_view_tensors(images, ss_transform)

        views = views.to(device)
        bs, n_views, c, h, w = views.shape
        views = views.reshape(-1, c, h, w)  # [bs*n_views, c, h, w]
        views = normalize_transform(views)

        with torch.no_grad():
            vision_features = backbone(views)  # [bs*n_views, 512]

        _, C = vision_features.shape

        max_indices_at_channel = ss_statistics(
            vision_features.detach().cpu().numpy(), bs, C, args
        )

        all_votes.append(max_indices_at_channel)

    # find the most frequently voted channels (identified channels)
    all_votes = np.concatenate(all_votes, axis=0)  # [#dataset, n_view*take_channel]
    essential_indices = Counter(all_votes.flatten()).most_common(
        args.removed_channel_num
    )
    essential_indices = torch.tensor([idx for (idx, occ_count) in essential_indices])
    return essential_indices


"""
Get representations from encoder
"""


def get_feats(loader, model, normalize_transform):

    model.eval()
    feats, ptr = None, 0

    with torch.no_grad():
        for i, content in enumerate(loader):
            images = content[0]
            images = images.to(device)
            images = normalize_transform(images)

            output = model(images)

            cur_feats = F.normalize(output, dim=1).cpu()
            B, D = cur_feats.shape

            inds = torch.arange(B) + ptr  # [0, 1, ..., B-1] + prt

            if not ptr:
                # arrive only when ptr is 0 (i.e. first iteration)
                feats = torch.zeros(
                    (len(loader.dataset), D)
                ).float()  # len(loader.dataset) is the whole dataset's size, not just batch size

            feats.index_copy_(0, inds, cur_feats)  # (dim, index, tensor)
            ptr += B
    return feats


"""
Train the linear classifier
"""


def train_linear_classifier(
    probe_loader, backbone, linear, optimizer, normalize_transform
):
    backbone.eval()
    linear.train()

    for i, content in enumerate(probe_loader):
        images, target, _ = content

        images = images.to(device)
        images = normalize_transform(images)
        target = target.to(device)

        with torch.no_grad():
            output = backbone(images)

        output = linear(output)
        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()  # update params of linear
        optimizer.step()


"""
Helper function for function eval_linear_classifier()
"""


def produces_evaluation_results(linear, output, target, acc1_accumulator, total_count):
    output = linear(output)
    _, pred = output.topk(
        1, 1, True, True
    )  # k=1, dim=1, largest, sorted; pred is the indices of largest class
    # pred.shape: [bs, k=1]
    pred = pred.squeeze(1)  # shape: [bs, ]

    total_count += target.shape[0]
    acc1_accumulator += (pred == target).float().sum().item()
    return acc1_accumulator, total_count


"""
Evaluate the performance of linear probing (linear classifier)
"""


def eval_linear_classifier(
    val_loader,
    backbone,
    linear,
    args,
    normalize_transform,
    val_mode,
    use_ss_detector,
    contributing_indices,
):
    with torch.no_grad():

        acc1_accumulator = 0.0
        total_count = 0

        for i, content in enumerate(val_loader):
            if val_mode == "poison":
                images, target, original_label, _ = content
                original_label = original_label.to(device)
            elif val_mode == "clean":
                images, target, _ = content
            else:
                raise Exception(f"unimplemented val_mode {val_mode}")

            images = images.to(device)
            images = normalize_transform(images)
            target = target.to(device)

            if val_mode == "poison":
                valid_indices = original_label != args.target_class
                if torch.all(~valid_indices):
                    # all inputs are from target class, skip this iteration
                    continue

                images = images[valid_indices]
                target = target[valid_indices]

            # compute output
            output = backbone(images)

            if use_ss_detector:
                indices_toremove = contributing_indices[0 : args.removed_channel_num]
                output[:, indices_toremove] = 0.0

            acc1_accumulator, total_count = produces_evaluation_results(
                linear, output, target, acc1_accumulator, total_count
            )

        return acc1_accumulator / total_count * 100.0


"""
Normalization for linear classifier
"""


class Normalize(nn.Module):
    def forward(self, x):
        return x / x.norm(2, dim=1, keepdim=True)


class FullBatchNorm(nn.Module):
    def __init__(self, var, mean):
        super(FullBatchNorm, self).__init__()
        self.register_buffer("inv_std", (1.0 / torch.sqrt(var + 1e-5)))
        self.register_buffer("mean", mean)

    def forward(self, x):
        return (x - self.mean) * self.inv_std


"""
Used as the base class for BYOL and SimCLR
"""


class CLModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.method = args.method
        self.arch = args.arch
        self.dataset = args.dataset

        if self.arch == "vit_b_16":
            model_fun, feat_dim = model_dict[self.arch]
            self.mlp_layers = 3
        else:
            if "cifar" in self.dataset:
                # CIFAR-variant Resnet is loaded
                model_fun, feat_dim = model_dict_cifar[self.arch]
                self.mlp_layers = 2
            else:
                # Original Resnet is loaded
                model_fun, feat_dim = model_dict[self.arch]
                self.mlp_layers = 3

        self.model_generator = model_fun
        self.backbone = model_fun()
        self.feat_dim = feat_dim

    def forward(self, x):
        pass

    def loss(self, reps):
        pass


class CLTrainer:

    def __init__(self, args):
        self.args = args
        self.args.warmup_epoch = 10
        self.normalize_transform = T.Compose(
            [
                T.Normalize(args.mean, args.std),
            ]
        )

    """
    Baseline: BCU (Backdoor Cleansing with Unlabeled Data, CVPR 2023)
    """

    def bcu(self, teacher, student, poison):
        teacher.eval()

        # create aug pipeline
        bcu_aug = nn.Sequential(
            aug.RandomCrop(
                size=(self.args.image_size, self.args.image_size), padding=4
            ),
            aug.RandomHorizontalFlip(),
            aug.Normalize(mean=self.args.mean, std=self.args.std),
        )

        # adaptive layer-wise weight re-initialization
        teacher_state_dict = copy.deepcopy(teacher.state_dict())
        student_state_dict = student.state_dict()
        for key in teacher_state_dict.keys():
            if (
                key.find("bn") != -1
                or key.find("shortcut.1") != -1
                or key.find("scalar_label") != -1
                or key.find("queue") != -1
                or key.find("queue_ptr") != -1
            ):
                # ignore
                continue
            if key.endswith(".weight") or key.endswith(".bias"):
                p = self.args.bcu_layerwise_ratio[0]
                if key.startswith("layer1"):
                    p = self.args.bcu_layerwise_ratio[1]
                elif key.startswith("layer2"):
                    p = self.args.bcu_layerwise_ratio[2]
                elif key.startswith("layer3"):
                    p = self.args.bcu_layerwise_ratio[3]
                elif key.startswith("layer4"):
                    p = self.args.bcu_layerwise_ratio[4]
                elif key.startswith("fc"):
                    p = self.args.bcu_layerwise_ratio[5]

                mask_one = torch.ones(teacher_state_dict[key].shape) * (1 - p)
                mask = torch.bernoulli(mask_one).to(device)
                masked_weight = teacher_state_dict[key] * mask + student_state_dict[
                    key
                ] * (
                    1 - mask
                )  # 1 copy, 0 random
                teacher_state_dict[key] = masked_weight
        student.load_state_dict(teacher_state_dict, strict=False)
        student.to(device)

        optimizer = torch.optim.SGD(
            student.parameters(), lr=self.args.bcu_lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = getattr(torch.optim.lr_scheduler, "CosineAnnealingLR")(
            optimizer, T_max=100
        )
        student.train()
        for i in range(self.args.bcu_epochs):
            distillation(
                self.args,
                teacher,
                student,
                optimizer,
                scheduler,
                i,
                poison.train_probe_loader,
                bcu_aug,
                device,
            )

    """
    Baseline: MIMIC (Mutual Information Guided Backdoor Mitigation for Pre-trained Encoders, IEEE Transactions on Information Forensics and Security 2024), called when args.use_mimic==True
    """

    def mimic(self, teacher, student, poison, train_transform):
        teacher.eval()
        train_transform = train_transform.to(device)

        # optimizer
        optimizer = torch.optim.Adam(
            student.parameters(), lr=self.args.mimic_lr, weight_decay=1e-6
        )

        scheduler = weight_scheduler(
            base_opt=[
                self.args.opt1,
                self.args.opt2,
                self.args.opt3,
                self.args.opt4,
            ],  # base value, to be modified into params[0], ...
            args=self.args,
            momentum_opt=10000,
            EPOCHS=200,
        )

        # set params by mi
        # their memory_loader, which is tensorized, and normalized (input, target pair)
        mi, student_hook_info = scheduler.estimate_mi(
            student, poison.train_probe_loader, device
        )
        params = scheduler.update_weight(
            mi
        )  # weighted four MI values, one for each layer
        self.args.opt1, self.args.opt2, self.args.opt3, self.args.opt4 = (
            params[0],
            params[1],
            params[2],
            params[3],
        )
        print("Estimated weight: ", params[0], params[1], params[2], params[3])

        epoch_start = 1

        # Training loop
        for epoch in range(epoch_start, self.args.mimic_epochs + 1):
            print("=================================================")
            train_loss = mimic_model_train(
                student,
                teacher,
                poison.train_probe_loader,
                optimizer,
                epoch,
                self.args,
                train_transform,
                student_hook_info,
            )
            if epoch % 1000 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": student.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(self.args.saved_path)
                    + "mimic_student_model_ep"
                    + str(epoch)
                    + ".pth",
                )

    """
    Baseline: RNP (ICML 2023), called when args.use_rnp==True
    """

    def mask_prune(self, backbone, poison, trained_linear):

        new_linear = copy.deepcopy(trained_linear)
        new_linear.train()

        new_backbone = copy.deepcopy(backbone)
        new_backbone.train()

        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(
            list(new_backbone.parameters()) + list(new_linear.parameters()),
            lr=self.args.unlearning_lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.args.schedule, gamma=0.1
        )

        #### stage 1: model unlearing
        print(f">>>>>>>> start model unlearning")
        for epoch in range(0, self.args.unlearning_epochs + 1):
            # UNLEARNING: use copied backbone and linear, both are learnable
            train_acc = train_step_unlearning(
                args=self.args,
                model=new_backbone,
                linear=new_linear,
                criterion=criterion,
                optimizer=optimizer,
                data_loader=poison.train_probe_loader,
            )

            scheduler.step()
            print(f">>>>>>>> at epoch {epoch}, the train_acc is {train_acc}")

            if train_acc <= self.args.clean_threshold:
                print(
                    f">>>>>>>> arrive at early break of stage 1 unlearning at epoch {epoch}"
                )
                # end stage 1
                break

        #### stage 2: model recovering
        print(f">>>>>>>> start model recovering")
        if self.args.method == "mocov2":
            feat_dim = get_feat_dim(self.args)
            unlearned_model = models.__dict__[self.args.arch](
                num_classes=feat_dim,
                norm_layer=MaskBatchNorm2d,
            )
            unlearned_model.fc = nn.Sequential()
        else:
            if "cifar" in self.args.dataset:
                model_fun, _ = model_dict_cifar[self.args.arch]
            else:
                model_fun, _ = model_dict[self.args.arch]
            unlearned_model = model_fun(norm_layer=MaskBatchNorm2d)

        # initialze it with the weights of unlearned model new_backbone
        refill_unlearned_model(
            unlearned_model, orig_state_dict=new_backbone.state_dict()
        )

        unlearned_model = unlearned_model.to(device)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        parameters = list(unlearned_model.named_parameters())
        mask_params = [
            v for n, v in parameters if "neuron_mask" in n
        ]  # only update neuron_mask ones
        mask_optimizer = torch.optim.SGD(
            mask_params, lr=self.args.recovering_lr, momentum=0.9
        )

        for epoch in range(1, self.args.recovering_epochs + 1):
            train_step_recovering(
                args=self.args,
                unlearned_model=unlearned_model,  # use the unlearnt backbone, plus learnable masks injected
                linear=new_linear,  # use the unlearnt linear from previous step
                criterion=criterion,
                data_loader=poison.train_probe_loader,
                mask_opt=mask_optimizer,
            )

        save_mask_scores(
            unlearned_model.state_dict(),
            os.path.join(self.args.saved_path, "mask_values.txt"),
        )

        del unlearned_model, new_linear, new_backbone

        #### stage 3: model pruning
        print(f">>>>>>>> start model pruning")

        backbone = copy.deepcopy(backbone)  # unimpacted backbone
        linear = copy.deepcopy(trained_linear)  # unimpacted linear

        criterion = torch.nn.CrossEntropyLoss().to(device)
        mask_file = os.path.join(self.args.saved_path, "mask_values.txt")
        mask_values = read_data(mask_file)
        mask_values = sorted(mask_values, key=lambda x: float(x[2]))
        print(
            "No. \t Layer Name \t Neuron Idx \t Mask \t knn_acc \t knn_asr \t linear_acc \t linear_asr"
        )

        # unimpacted kNN performance
        clean_acc, back_acc = self.knn_monitor_fre(
            backbone,
            poison.memory_loader,
            poison.test_clean_loader,
            poison.test_pos_loader,
            self.args,
            classes=self.args.num_classes,
        )

        # unimpacted linear performance
        cl_loss, cl_acc = test_maskprune(
            args=self.args,
            model=backbone,
            linear=linear,
            criterion=criterion,
            data_loader=poison.test_clean_loader,
            val_mode="clean",
        )
        po_loss, po_acc = test_maskprune(
            args=self.args,
            model=backbone,
            linear=linear,
            criterion=criterion,
            data_loader=poison.test_pos_loader,
            val_mode="poison",
        )

        print(
            "0 \t None     \t None  \t None   \t {:.1f} \t {:.1f} \t {:.1f} \t {:.1f}".format(
                # knn acc
                clean_acc,
                # knn asr
                back_acc,
                # linear acc
                cl_acc * 100,
                # linear asr
                po_acc * 100,
            )
        )  # this records the backdoored model's initial results

        # masking with changing threshold
        if self.args.pruning_by == "threshold":
            evaluate_by_threshold(
                self.args,
                backbone,
                linear,
                mask_values,
                pruning_max=self.args.pruning_max,
                pruning_step=self.args.pruning_step,
                criterion=criterion,
                clean_loader=poison.test_clean_loader,
                poison_loader=poison.test_pos_loader,
                memory_loader=poison.memory_loader,
                knn_evaluator=self.knn_monitor_fre,
            )
        else:
            raise Exception("Not implemented yet")

    """
    Linear classifier training and evalaution
    """

    def linear_probing(
        self,
        backbone,
        poison,
        force_training=False,
    ):
        backbone.eval()

        feat_dim = get_feat_dim(self.args)

        # initialize linear mode, including normalization module
        train_probe_feats = get_feats(
            poison.train_probe_loader, backbone, self.normalize_transform
        )  # shape: [N, D]
        train_var, train_mean = torch.var_mean(train_probe_feats, dim=0)
        linear = nn.Sequential(
            Normalize(),  # L2 norm
            FullBatchNorm(train_var, train_mean),
            nn.Linear(feat_dim, self.args.num_classes),
        )

        # if pretrained model available
        if self.args.pretrained_linear_model != "":
            pretrained_state_dict = torch.load(
                self.args.pretrained_linear_model, map_location=device
            )
            linear.load_state_dict(pretrained_state_dict, strict=True)
        linear = linear.to(device)

        # training required
        if self.args.pretrained_linear_model == "" or force_training:
            optimizer = torch.optim.SGD(
                linear.parameters(),
                lr=0.06,
                momentum=0.9,
                weight_decay=1e-4,
            )
            sched = [15, 30, 40]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=sched
            )
            linear_probing_epochs = 40

            for epoch in range(linear_probing_epochs):
                print(f"training linear classifier, epoch: {epoch}")
                train_linear_classifier(
                    poison.train_probe_loader,
                    backbone,
                    linear,
                    optimizer,
                    self.normalize_transform,
                )
                lr_scheduler.step()

            save_model(
                linear.state_dict(),
                filename=os.path.join(self.args.saved_path, "linear.pth.tar"),
            )

        # evaluation of linear (uncleansed)
        backbone.eval()
        linear.eval()
        clean_acc1 = eval_linear_classifier(
            poison.test_clean_loader,
            backbone,
            linear,
            self.args,
            self.normalize_transform,
            val_mode="clean",
            use_ss_detector=False,
            contributing_indices=None,
        )
        poison_acc1 = eval_linear_classifier(
            poison.test_pos_loader,
            backbone,
            linear,
            self.args,
            self.normalize_transform,
            val_mode="poison",
            use_ss_detector=False,
            contributing_indices=None,
        )

        print(
            f"for linear classifier, the ACC on clean val is: {np.round(clean_acc1,1)}, the ASR on poisoned val is: {np.round(poison_acc1,1)}"
        )

        return linear

    """
    Compute adaptive attack loss to bypass iBC defense
    
    The adaptive attack tries to distribute the trigger across many channels
    to avoid being detected by spectral signature analysis (which looks for
    concentrated trigger signals in few channels).
    
    Args:
        features: Output features from SSL model, shape [bs, feat_dim]
        mode: 'entropy' (maximize channel entropy), 'l2_spread' (spread L2 norm),
              or 'adversarial' (make features robust to channel removal)
        epoch: Current training epoch
    
    Returns:
        loss: Scalar loss value
    """

    def compute_adaptive_loss(self, features, mode="entropy", epoch=0):
        if mode == "entropy":
            # Maximize entropy of feature distribution across channels
            # This encourages the trigger to activate many channels equally
            # L2 normalize to get probability-like distribution
            feat_norm = F.normalize(features, p=2, dim=1)  # [bs, feat_dim]

            # Compute Shannon entropy for each sample
            # Higher entropy means more uniform activation across channels
            feat_abs = torch.abs(feat_norm)
            feat_probs = feat_abs / (feat_abs.sum(dim=1, keepdim=True) + 1e-8)

            # Shannon entropy: -sum(p * log(p))
            entropy = -torch.sum(feat_probs * torch.log(feat_probs + 1e-8), dim=1)

            # We want to MINIMIZE entropy loss (negative entropy), which means
            # we want to maximize entropy to spread activation
            loss = -entropy.mean()  # Negative because we want to maximize entropy

        elif mode == "l2_spread":
            # Encourage L2 norm to be spread across channels evenly
            # by minimizing the standard deviation of channel magnitudes
            feat_norm = F.normalize(features, p=2, dim=0)  # Normalize across batch
            channel_magnitudes = torch.norm(features, dim=0)  # [feat_dim]

            # Minimize variance of channel magnitudes for even spread
            loss = torch.var(channel_magnitudes)

        elif mode == "adversarial":
            # Simulate the defense: if top-k channels are removed,
            # the attack should still succeed on remaining channels
            # This encourages redundancy in trigger encoding
            feat_dim = features.shape[1]
            k = self.args.removed_channel_num

            # Get top-k channels by magnitude
            top_k_vals, top_k_indices = torch.topk(torch.abs(features).mean(dim=0), k)

            # Create mask for remaining channels after defense
            remaining_mask = torch.ones(feat_dim, device=features.device)
            remaining_mask[top_k_indices] = 0

            # Loss: minimize the magnitude of top-k channels while keeping others
            # This way, when top-k are removed, remaining channels still contain info
            masked_features = features * remaining_mask.unsqueeze(0)
            loss = -torch.norm(masked_features, p=2) / (
                torch.norm(features, p=2) + 1e-8
            )
        elif mode == "svd_correlation":
            if epoch < self.args.svd_start_epoch:
                # Don't apply SVD-based loss until certain epoch to allow stable SVD computation
                # create a zero scalar that matches `features` dtype/device
                return None

            # Center features along channel dim
            X = features - features.mean(dim=0, keepdim=True)

            # sanitize NaN/Inf early
            if not torch.isfinite(X).all():
                # either return zero loss or nan-to-num; returning zero is safest for stability
                # (log here if you want to trace occurrences)
                return None

            # try SVD with safe fallbacks
            try:
                _, _, V = torch.linalg.svd(X, full_matrices=False)
                eig_for_indexing = V[0:1]
                corrs = eig_for_indexing.matmul(features.t())  # [1, bs*n_view]

                # minimize the variance of corrs to avoid concentrated correlation in few channels
                loss = torch.var(corrs)
            except Exception:
                return None
                # try:
                #     # CPU fallback (more stable though slower)
                #     X_cpu = X.detach().cpu()
                #     _, _, V_cpu = torch.linalg.svd(X_cpu, full_matrices=False)
                #     eig_for_indexing = V_cpu[0:1].to(X.device)
                #     corrs = eig_for_indexing.matmul(features.t())  # [1, bs*n_view]
                #     loss = torch.var(corrs)
                # except Exception:
                #     # # final fallback: eigendecomposition of X^T X with tiny jitter
                #     # C = X.t().matmul(X)
                #     # jitter = 1e-6 * torch.eye(
                #     #     C.shape[0], device=C.device, dtype=C.dtype
                #     # )
                #     # try:
                #     #     vals, vecs = torch.linalg.eigh(C + jitter)
                #     #     eig_for_indexing = vecs[:, -1:].t()
                #     # except Exception:
                #     #     # give up for this batch
                #     return None

        else:
            raise ValueError(f"Unknown adaptive attack mode: {mode}")

        return loss

    """

        Train the SSL encoder and then perform kNN classifier evalution
    """

    def train_freq(
        self, model, optimizer, train_transform, poison, force_training=False
    ):
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.args.pretrain_epochs
        )
        warmup_scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=self.args.warmup_epoch,
            after_scheduler=cosine_scheduler,
        )

        train_loader = poison.train_pos_loader  # poisoned train set
        test_clean_loader = poison.test_clean_loader  # clean val set
        test_back_loader = poison.test_pos_loader  # poisoned val set

        clean_acc = 0.0
        back_acc = 0.0

        training_required = self.args.pretrained_ssl_model == "" or force_training

        for epoch in range(self.args.start_epoch, self.args.pretrain_epochs):
            losses = AverageMeter()
            cl_losses = AverageMeter()

            train_transform = train_transform.to(device)

            start = time.time()

            # SSL TRAIN
            if training_required:
                for i, content in enumerate(train_loader):
                    images = content[0]
                    images = images.to(device)
                    model.train()

                    # SSL aug
                    v1 = train_transform(images)
                    v2 = train_transform(images)

                    features = model(v1, v2)

                    if self.args.method == "simclr":
                        loss = model.supConLoss(features)
                    elif self.args.method == "byol":
                        loss = model.negcos(*features)
                    elif self.args.method == "mocov2":
                        loss = model.loss(*features)

                    # Add adaptive loss term for bypassing iBC defense
                    adaptive_loss = None
                    if self.args.use_adaptive_attack:
                        # Extract base features for adaptive loss computation
                        # For different SSL methods, features structure differs
                        if self.args.method == "simclr":
                            # features shape: [bs, 2, C]; extract both views
                            f1 = features[:, 0, :]
                            f2 = features[:, 1, :]
                            feat_for_loss = torch.cat([f1, f2], dim=0)
                        elif self.args.method == "byol":
                            # features is tuple of (online_proj, target_proj, online_pred, target_pred)
                            feat_for_loss = torch.cat([features[0], features[1]], dim=0)
                        elif self.args.method == "mocov2":
                            # features is tuple (q, k) of query and key embeddings
                            feat_for_loss = torch.cat([features[0], features[1]], dim=0)

                        adaptive_loss = self.compute_adaptive_loss(
                            feat_for_loss,
                            mode=self.args.adaptive_attack_mode,
                            epoch=epoch,
                        )

                    # keep base loss (before adaptive addition) for diagnostics
                    base_loss = loss

                    # combine losses safely
                    if adaptive_loss is not None:
                        try:
                            adaptive_loss = adaptive_loss.to(base_loss.device).type_as(
                                base_loss
                            )
                        except Exception:
                            adaptive_loss = base_loss.new_tensor(adaptive_loss)
                        loss = (
                            base_loss + self.args.adaptive_attack_lambda * adaptive_loss
                        )
                    else:
                        loss = base_loss

                    # update meters using correct batch size
                    try:
                        batch_n = images.size(0)
                    except Exception:
                        batch_n = 1
                    losses.update(float(loss.detach().cpu().item()), batch_n)
                    cl_losses.update(float(loss.detach().cpu().item()), batch_n)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                warmup_scheduler.step()

            # EVAL
            if epoch + 1 == self.args.pretrain_epochs or (
                (training_required) and epoch % self.args.knn_eval_freq == 0
            ):
                model.eval()
                backbone = extract_backbone(self.args.method, model)

                clean_acc, back_acc = self.knn_monitor_fre(
                    backbone,
                    poison.memory_loader,
                    test_clean_loader,
                    test_back_loader,
                    self.args,
                    classes=self.args.num_classes,
                )
                print(
                    "[{}-epoch] time:{:.1f} | clean acc: {:.1f} | back acc: {:.1f} | loss:{:.3f} | cl_loss:{:.3f}".format(
                        epoch + 1,
                        time.time() - start,
                        clean_acc,
                        back_acc,
                        losses.avg,
                        cl_losses.avg,
                    )
                )
            if (epoch + 1) % self.args.model_save_freq == 0 and training_required:
                save_model(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    filename=os.path.join(
                        self.args.saved_path, f"encoder_epoch_{epoch + 1}.pth.tar"
                    ),
                )

        # save final model
        if training_required:
            save_model(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                filename=os.path.join(self.args.saved_path, "encoder.pth.tar"),
            )

        return model

    """
    Use iBC defense
    """

    def trigger_channel_removal(self, model, poison, trained_linear):

        backbone = extract_backbone(self.args.method, model)
        backbone.eval()
        trained_linear.eval()

        # Esimate poisoned triggers
        if self.args.use_randomdrop:
            feat_dim = get_feat_dim(self.args)
            contributing_indices = torch.randperm(feat_dim)[
                : self.args.removed_channel_num
            ]
        else:
            contributing_indices = find_trigger_channels(
                self.args,
                poison.train_pos_loader,  # poisoned training set
                backbone,
                poison.ss_transform,
                self.normalize_transform,
            )
        print(f"predicted trigger channels are: {contributing_indices}")

        ############# KNN
        clean_acc_SSDETECTOR, back_acc_SSDETECTOR = self.knn_monitor_fre(
            backbone,
            poison.memory_loader,
            poison.test_clean_loader,
            poison.test_pos_loader,
            self.args,
            classes=self.args.num_classes,
            use_SS_detector=True,
            contributing_indices=contributing_indices,
        )

        print(
            f"In kNN classification, by replacing top-{self.args.removed_channel_num} channels, clean acc: {clean_acc_SSDETECTOR:.1f} | back acc: {back_acc_SSDETECTOR:.1f}"
        )

        ########### Linear Probe
        # Clean Validation Set
        clean_acc1 = eval_linear_classifier(
            poison.test_clean_loader,
            backbone,
            trained_linear,
            self.args,
            self.normalize_transform,
            val_mode="clean",
            use_ss_detector=True,
            contributing_indices=contributing_indices,
        )
        poison_acc1 = eval_linear_classifier(
            poison.test_pos_loader,
            backbone,
            trained_linear,
            self.args,
            self.normalize_transform,
            val_mode="poison",
            use_ss_detector=True,
            contributing_indices=contributing_indices,
        )

        print(
            f"In linear probe, by replacing {self.args.removed_channel_num} channels, the ACC on clean val is: {np.round(clean_acc1,1)}, the ASR on poisoned val is: {np.round(poison_acc1,1)}"
        )

        return (clean_acc_SSDETECTOR, back_acc_SSDETECTOR, clean_acc1, poison_acc1)

    """
    kNN classifier evaluation (label prediction).
    """

    @torch.no_grad()
    def knn_monitor_fre(
        self,
        net,
        memory_data_loader,
        test_clean_loader,
        test_poi_loader,
        args,
        k=130,  # or 200 / 70
        t=0.1,
        hide_progress=True,
        classes=-1,
        use_SS_detector=False,
        contributing_indices=None,
    ):

        net.eval()

        feature_bank = []
        # generate feature bank
        for data, target, _ in tqdm(
            memory_data_loader,
            desc="Feature extracting",
            leave=False,
            disable=hide_progress,
        ):
            data = data.to(device)
            # If using ViT backbone and images are small (e.g., CIFAR 32), upsample to expected size
            if hasattr(args, "arch") and "vit" in args.arch.lower():
                expected_size = args.image_size
                _, _, h, w = data.shape
                if h != expected_size or w != expected_size:
                    data = F.interpolate(data, size=(expected_size, expected_size), mode="bilinear", align_corners=False)
            data = self.normalize_transform(data)

            with torch.no_grad():
                feature = net(data)

            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)

        # feature_bank: [dim, total num]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()

        # feature_labels: [total num]
        feature_labels = (
            memory_data_loader.dataset[:][1].clone().detach().to(feature_bank.device)
        )

        """
        Evaluate clean KNN
        """

        clean_val_top1, clean_val_total_num = 0.0, 0

        test_bar = tqdm(test_clean_loader, desc="kNN", disable=hide_progress)
        for content in test_bar:

            data, target, _ = content

            data, target = data.to(device), target.to(device)
            if hasattr(args, "arch") and "vit" in args.arch.lower():
                expected_size = args.image_size
                _, _, h, w = data.shape
                if h != expected_size or w != expected_size:
                    data = F.interpolate(data, size=(expected_size, expected_size), mode="bilinear", align_corners=False)
            data = self.normalize_transform(data)

            with torch.no_grad():
                feature = net(data)

            if use_SS_detector:

                indices_toremove = contributing_indices[0 : args.removed_channel_num]
                feature[:, indices_toremove] = 0.0

            feature = F.normalize(feature, dim=1)
            pred_labels = self.knn_predict(
                feature, feature_bank, feature_labels, classes, k, t
            )

            clean_val_total_num += data.size(0)
            clean_val_top1 += (pred_labels[:, 0] == target).float().sum().item()

        """
        Evaluate poison KNN
        """

        backdoor_val_top1, backdoor_val_total_num = 0.0, 0

        backdoor_test_bar = tqdm(test_poi_loader, desc="kNN", disable=hide_progress)

        for content in backdoor_test_bar:
            data, target, original_label, _ = content

            data, target, original_label = (
                data.to(device),
                target.to(device),
                original_label.to(device),
            )

            if hasattr(args, "arch") and "vit" in args.arch.lower():
                expected_size = args.image_size
                _, _, h, w = data.shape
                if h != expected_size or w != expected_size:
                    data = F.interpolate(data, size=(expected_size, expected_size), mode="bilinear", align_corners=False)
            data = self.normalize_transform(data)

            valid_indices = original_label != args.target_class
            if torch.all(~valid_indices):
                # all inputs are from target class, skip this iteration
                continue

            data = data[valid_indices]
            target = target[valid_indices]

            with torch.no_grad():
                feature = net(data)

            if use_SS_detector:

                indices_toremove = contributing_indices[0 : args.removed_channel_num]
                feature[:, indices_toremove] = 0.0

            feature = F.normalize(feature, dim=1)
            # feature: [bsz, dim]
            pred_labels = self.knn_predict(
                feature, feature_bank, feature_labels, classes, k, t
            )

            backdoor_val_total_num += data.size(0)
            backdoor_val_top1 += (pred_labels[:, 0] == target).float().sum().item()

        return (
            clean_val_top1 / clean_val_total_num * 100,
            backdoor_val_top1 / backdoor_val_total_num * 100,
        )

    """
    Helper function for kNN classifier label prediction.
    """

    def knn_predict(self, feature, feature_bank, feature_labels, classes, knn_k, knn_t):
        # feature: [bsz, dim]
        # feature_bank: [dim, clean_val_total_num]
        # feature_labels: [clean_val_total_num]

        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, feature_bank)
        # sim_matrix: [bsz, K]
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)

        # sim_labels: [bsz, K]
        sim_labels = torch.gather(
            feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
        )
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(
            feature.size(0) * knn_k, classes, device=sim_labels.device
        )

        # one_hot_label: [bsz*K, C]
        one_hot_label = one_hot_label.scatter(
            dim=-1, index=sim_labels.view(-1, 1), value=1.0
        )  # for each row, only one column is 1, which is the label of k-nearest this neighbor

        # weighted score ---> [bsz, C]
        pred_scores = torch.sum(
            one_hot_label.view(feature.size(0), -1, classes)  # [bs, k, C=Classes]
            * sim_weight.unsqueeze(dim=-1),  # [bs, k, 1]
            dim=1,
        )  # [bs, C], where each column means the SCORE (weight) of the sample to the class at this column index

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels  # [bs, C], where the first column is the index (class) of nearest cluster
