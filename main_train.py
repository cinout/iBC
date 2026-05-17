import os
import argparse
import random
import torch.optim as optim
from datetime import datetime
from data_prepare.data_prepare import set_aug_diff, PoisonAgent
from methods import set_model
from methods.base import CLTrainer
from utils.util import *
from utils.frequency import PoisonFre
from utils.htba import PoisonHTBA
from ssl_cleanse.ssl_cleanse import (
    trigger_inversion,
    trigger_mitigation,
)
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser()

"""
Pretrained Models
"""
parser.add_argument(
    "--pretrained_ssl_model",
    type=str,
    default="",
    help="path for pretrained ssl model (stage 1)",
)
parser.add_argument(
    "--pretrained_linear_model",
    type=str,
    default="",
    help="path for pretrained linear model (stage 2)",
)


"""
Image Size
"""
parser.add_argument("--window_size", default=32, type=int)
parser.add_argument(
    "--image_size",
    type=int,
    default=32,
    help="64 for ImageNet-100, 32 for CIFAR-10/100",
)

"""
Batch Size
"""
parser.add_argument("--pretrain_batch_size", default=128, type=int)
parser.add_argument("--linear_probe_batch_size", default=128, type=int)


"""
Dataset
"""
parser.add_argument("--data_path", default="./datasets/")
parser.add_argument(
    "--dataset", choices=["cifar10", "cifar100", "imagenet100"], required=True
)


"""
Architecture & Optimization
"""
parser.add_argument(
    "--arch",
    default="resnet18",
    type=str,
)


"""
SSL
"""
parser.add_argument("--method", default="simclr", choices=["simclr", "byol", "mocov2"])
parser.add_argument("--temp", default=0.5, type=float)
parser.add_argument("--lr", default=0.06, type=float)
parser.add_argument("--wd", default=5e-4, type=float)
parser.add_argument("--cos", action="store_true", default=True)
parser.add_argument("--byol-m", default=0.996, type=float)

"""
Basics
"""
parser.add_argument(
    "--timestamp",
    type=str,
    default=datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_"
    + str(random.randint(0, 100))
    + "_"
    + str(random.randint(0, 100)),
)
parser.add_argument("--ssl_pretrain_seed", default=42, type=int)
parser.add_argument("--num_workers", default=0, type=int)


"""
Epochs
"""
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--pretrain_epochs", default=800, type=int)


"""
Logging & Model Saving
"""
parser.add_argument(
    "--log_path", default="Experiments", type=str, help="parent directory"
)
parser.add_argument("--saved_path", default="none", type=str, help="created ad hoc")


"""
Evaluation
"""
parser.add_argument("--knn_eval_freq", default=5, type=int)


"""
Trigger / Poisoning
"""
parser.add_argument(
    "--trigger_type", default="ftrojan", choices=["ftrojan", "htba", "none"]
)
parser.add_argument("--target_class", default=0, type=int)
parser.add_argument("--poison_ratio", default=0.01, type=float)
parser.add_argument("--probe_set_percent", default=0.01, type=float)
parser.add_argument(
    "--trigger_position", nargs="+", type=int, default=[15, 31], help="for FTrojan"
)
parser.add_argument("--magnitude_train", default=50.0, type=float, help="for FTrojan")
parser.add_argument("--magnitude_val", default=100.0, type=float, help="for FTrojan")
parser.add_argument("--ftrojan_channel", nargs="+", type=int, default=[1, 2])

"""
Image Augmentation
"""
# TODO: update this
parser.add_argument(
    "--num_views",
    type=int,
    default=64,
    help="how many views are generated for each image.",
)
parser.add_argument(
    "--rrc_scale_min",
    type=float,
    default=0.3,
)
parser.add_argument(
    "--rrc_scale_max",
    type=float,
    default=0.95,
)

"""
iBC Options
"""
parser.add_argument(
    "--use_ibc",
    action="store_true",
    help="apply channel removal strategy",
)
parser.add_argument("--trigger_channel_removal_seed", default=42, type=int)
# TODO: update this
parser.add_argument(
    "--removed_channel_num",
    type=int,
    default=70,
    help="model-level backdoor estimation channel",
)
# TODO: update this
parser.add_argument(
    "--voted_channel_num",
    type=int,
    default=30,
    help="view-level backdoor estimation channel",
)
parser.add_argument(
    "--find_channels_from_n_poison_samples",
    type=int,
    default=0,
    help="If >0, sample from limited number of images for trigger channel",
)
parser.add_argument(
    "--find_channels_from_n_clean_samples",
    type=int,
    default=0,
    help="If >0, sample some clean images as well",
)


"""
Defense Baseline: RandomDrop
"""
parser.add_argument(
    "--use_randomdrop",
    action="store_true",
    help="a baseline: randomly drop out some channels",
)
parser.add_argument(
    "--randomdrop_seed",
    type=int,
    default=42,
)


"""
Defense Baseline: RNP
"""
parser.add_argument(
    "--use_rnp",
    action="store_true",
    help="apply mask pruning (RNP paper)",
)
parser.add_argument("--rnp_seed", default=42, type=int)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument(
    "--clean_threshold",
    type=float,
    default=0.20,
    help="threshold of unlearning accuracy",
)
parser.add_argument(
    "--unlearning_lr",
    type=float,
    default=0.01,
    help="the learning rate for neuron unlearning",
)
parser.add_argument(
    "--recovering_lr",
    type=float,
    default=0.2,
    help="the learning rate for mask optimization",
)
parser.add_argument(
    "--unlearning_epochs",
    type=int,
    default=20,
    help="the number of epochs for unlearning",
)
parser.add_argument(
    "--recovering_epochs",
    type=int,
    default=20,
    help="the number of epochs for recovering",
)
parser.add_argument(
    "--pruning-by", type=str, default="threshold", choices=["number", "threshold"]
)
parser.add_argument(
    "--pruning-max",
    type=float,
    default=0.90,
    help="the maximum number/threshold for pruning",
)
parser.add_argument(
    "--pruning-step",
    type=float,
    default=0.05,
    help="the step size for evaluating the pruning",
)
parser.add_argument(
    "--schedule",
    type=int,
    nargs="+",
    default=[10, 20],
    help="Decrease learning rate at these epochs.",
)


"""
Defense Baseline: SSL-Cleanse
"""
parser.add_argument(
    "--use_ssl_cleanse",
    action="store_true",
    help="use the method from ECCV2024 paper: ssl-cleanse",
)
parser.add_argument(
    "--ssl_cleanse_seed",
    type=int,
    default=10,
)
parser.add_argument(
    "--attack_succ_threshold",
    type=float,
    default=0.99,
    help="",
)
parser.add_argument(
    "--lam",
    type=float,
    default=0.1,
    help="",
)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--lam_multiplier_up", type=float, default=1.5)
parser.add_argument("--ratio", type=float, default=0.05)
parser.add_argument(
    "--num_clusters",
    type=int,
    default=12,
)
parser.add_argument(
    "--trigger_path",
    default="",
    type=str,
)
parser.add_argument(
    "--mitigate_epochs",
    type=int,
    default=5,
)
parser.add_argument(
    "--draw_local_trigger_by", type=str, choices=["global", "local"], default="global"
)
parser.add_argument(
    "--drop",
    type=int,
    nargs="*",
    default=[50, 25],
    help="milestones for learning rate decay (0 = last epoch)",
)
parser.add_argument(
    "--drop_gamma",
    type=float,
    default=0.2,
    help="multiplicative factor of learning rate decay",
)
parser.add_argument("--eval_every", type=int, default=20, help="how often to evaluate")
parser.add_argument("--cj0", default=0.4, help="color jitter brightness")
parser.add_argument("--cj1", default=0.4, help="color jitter contrast")
parser.add_argument("--cj2", default=0.4, help="color jitter saturation")
parser.add_argument("--cj3", default=0.1, help="color jitter hue")
parser.add_argument("--cj_p", default=0.8, help="color jitter probability")
parser.add_argument("--gs_p", default=0.1, help="grayscale probability")
parser.add_argument("--crop_s0", default=0.2, help="crop size from")
parser.add_argument("--crop_s1", default=1.0, help="crop size to")
parser.add_argument("--crop_r0", default=0.75, help="crop ratio from")
parser.add_argument("--crop_r1", default=(4 / 3), help="crop ratio to")
parser.add_argument("--hf_p", default=0.5, help="horizontal flip probability")
parser.add_argument("--trigger_width", type=int, default=6)
parser.add_argument("--trigger_location", type=float, default=0.9)


"""
Defense Baseline: MIMIC
"""
parser.add_argument(
    "--use_mimic",
    action="store_true",
    help="use the method from Mutual Information Guided Backdoor Mitigation for Pre-trained Encoders",
)
parser.add_argument("--mimic_seed", default=42, type=int)
parser.add_argument(
    "--mimic_lr", default=1e-2, type=float, help="initial learning rate"
)
parser.add_argument("--mimic_batch_size", default=128, type=int, help="")
parser.add_argument("--mimic_epochs", default=1000, type=int, help="")
parser.add_argument("--opt1", default=1000, type=int, help="opt1")
parser.add_argument("--opt2", default=1000, type=int, help="opt2")
parser.add_argument("--opt3", default=1000, type=int, help="opt3")
parser.add_argument("--opt4", default=1000, type=int, help="opt4")
parser.add_argument("--opt5", default=1, type=int, help="opt5")


"""
Defense Baseline: BCU
"""
parser.add_argument(
    "--use_bcu",
    action="store_true",
    help="use the method from Backdoor Cleansing with Unlabeled Data (CVPR 2023)",
)
parser.add_argument("--bcu_seed", default=42, type=int)
parser.add_argument(
    "--bcu_layerwise_ratio",
    type=float,
    nargs="+",
    default=[0.01, 0.01, 0.03, 0.09, 0.27, 0.10],
)
parser.add_argument(
    "--bcu_lr",
    type=float,
    default=0.01,
)
parser.add_argument(
    "--bcu_epochs",
    type=int,
    default=100,
)


"""
Adaptive Attack (bypasses iBC defense)
"""
parser.add_argument(
    "--use_adaptive_attack",
    action="store_true",
    help="Use adaptive attack that tries to bypass iBC defense by distributing trigger across channels",
)
parser.add_argument(
    "--adaptive_attack_lambda",
    type=float,
    default=0.1,
    help="Weight for adaptive attack loss term (higher = more emphasis on distribution)",
)
parser.add_argument(
    "--adaptive_attack_mode",
    choices=["entropy", "l2_spread", "adversarial"],
    default="entropy",
    help="Strategy for adaptive attack: entropy (maximize channel entropy), l2_spread (spread L2 norm), or adversarial (adversarial robustness)",
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    update_seed(args.ssl_pretrain_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """
    Create Model
    """
    print("=> creating model '{}'".format(args.arch))

    model = set_model(args)

    if args.pretrained_ssl_model != "":
        pretrained_state_dict = torch.load(
            args.pretrained_ssl_model, map_location=device
        )
        model.load_state_dict(pretrained_state_dict["state_dict"], strict=True)
    model = model.to(device)

    """
    Create Dataset/DataLoader
    """
    (
        train_dataset,
        test_dataset,
        memory_loader,
        train_transform,
    ) = set_aug_diff(args)

    """
    Construct Trainer
    """
    trainer = CLTrainer(args)

    """
    Create Poisoning Dataset
    """
    if args.trigger_type == "ftrojan":
        poison_agent = PoisonFre(args)
    elif args.trigger_type == "htba":
        poison_agent = PoisonHTBA(args)
    elif args.trigger_type == "none":
        poison_agent = None

    poison = PoisonAgent(
        args,
        poison_agent,
        train_dataset,
        test_dataset,
        memory_loader,
    )

    """
    Print All Args
    """
    all_args = "\n".join(
        "%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())
    )
    print(all_args)

    """
    Train and Evaluate
    """

    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
    )

    # SSL attack and KNN Evaluation [Poisoned Model]
    trainer.train_freq(model, optimizer, train_transform, poison)
    backbone = extract_backbone(args.method, model)

    # Linear Probe and Evaluation [Poisoned Model]
    trained_linear = trainer.linear_probing(backbone, poison)

    """
    iBC Defense
    """
    if args.use_ibc:
        update_seed(args.trigger_channel_removal_seed)
        knn_clean_acc_list = []
        knn_back_asr_list = []
        linear_clean_acc_list = []
        linear_back_asr_list = []

        for _ in range(10):
            knn_clean, knn_back, linear_clean, linear_back = (
                trainer.trigger_channel_removal(model, poison, trained_linear)
            )
            knn_clean_acc_list.append(knn_clean)
            knn_back_asr_list.append(knn_back)
            linear_clean_acc_list.append(linear_clean)
            linear_back_asr_list.append(linear_back)
        print("============= Overall =============")
        print(
            f"knn_clean_acc: {np.round(np.mean(knn_clean_acc_list),1)}±{np.round(np.std(knn_clean_acc_list),1)}"
        )
        print(
            f"knn_back_asr: {np.round(np.mean(knn_back_asr_list),1)}±{np.round(np.std(knn_back_asr_list),1)}"
        )
        print(
            f"linear_clean_acc: {np.round(np.mean(linear_clean_acc_list),1)}±{np.round(np.std(linear_clean_acc_list),1)}"
        )
        print(
            f"linear_back_asr: {np.round(np.mean(linear_back_asr_list),1)}±{np.round(np.std(linear_back_asr_list),1)}"
        )

    """
    Baseline 1: Use SSL-CLeanse (Ssl-cleanse: Trojan detection and mitigation in self-supervised learning, ECCV 2024)
    """
    if args.use_ssl_cleanse:
        update_seed(args.ssl_cleanse_seed)

        backbone = extract_backbone(args.method, model)

        trainset_data = trigger_inversion(
            args, backbone, poison, model.feat_dim
        )  # trainset_data is a tuple of (x_untransformed, y)

        cleansed_backbone = trigger_mitigation(args, backbone, trainset_data)

        new_trainer = CLTrainer(args)

        clean_acc, back_acc = new_trainer.knn_monitor_fre(
            cleansed_backbone,
            poison.memory_loader,
            poison.test_clean_loader,
            poison.test_pos_loader,
            args,
            classes=args.num_classes,
        )
        print(
            f">>>> With SSL-cleanse model, for kNN classifier, clean acc: {clean_acc:.1f}, back acc: {back_acc:.1f}",
        )

        _ = new_trainer.linear_probing(cleansed_backbone, poison, force_training=True)

    """
    Baseline 2: Mask Pruning Strategy (Reconstructive neuron pruning for backdoor defense, ICML 2023)
    """
    if args.use_rnp:
        backbone = extract_backbone(args.method, model)
        update_seed(args.rnp_seed)
        trainer.mask_prune(backbone, poison, trained_linear)

    """
    Baseline 3: Random Channel Removal
    """
    if args.use_randomdrop:
        update_seed(args.randomdrop_seed)
        trainer.trigger_channel_removal(model, poison, trained_linear)

    """
    Baseline 4: MIMIC (Mutual information guided backdoor mitigation for pretrained encoders, IEEE Transactions on Information Forensics and Security, 2025)
    """
    if args.use_mimic:
        update_seed(args.mimic_seed)
        student = set_model(args)
        student = student.to(device)
        trainer.mimic(model, student, poison, train_transform)

        student.eval()
        for p in student.parameters():
            p.requires_grad = False

        if args.method == "mocov2":
            student_backbone = student.encoder_q
            student_backbone.fc = nn.Sequential()
        else:
            student_backbone = student.backbone

        new_trainer = CLTrainer(args)
        clean_acc, back_acc = new_trainer.knn_monitor_fre(
            student_backbone,
            poison.memory_loader,
            poison.test_clean_loader,
            poison.test_pos_loader,
            args,
            classes=args.num_classes,
        )
        print(
            f">>>> With MIMIC model, for kNN classifier, clean acc: {clean_acc:.1f}, back acc: {back_acc:.1f}",
        )
        _ = new_trainer.linear_probing(student_backbone, poison, force_training=True)

    """
    Baseline 5: BCU (Backdoor cleansing with unlabeled data, CVPR 2023)
    """
    if args.use_bcu:
        update_seed(args.bcu_seed)
        student = set_model(args)
        student = student.to(device)
        trainer.bcu(model, student, poison)

        student.eval()

        for p in student.parameters():
            p.requires_grad = False

        if args.method == "mocov2":
            student_backbone = student.encoder_q
            student_backbone.fc = nn.Sequential()
        else:
            student_backbone = student.backbone

        new_trainer = CLTrainer(args)
        clean_acc, back_acc = new_trainer.knn_monitor_fre(
            student_backbone,
            poison.memory_loader,
            poison.test_clean_loader,
            poison.test_pos_loader,
            args,
            classes=args.num_classes,
        )
        print(
            f">>>> With BCU model, for kNN classifier, clean acc: {clean_acc:.1f}, back acc: {back_acc:.1f}",
        )
        _ = new_trainer.linear_probing(student_backbone, poison, force_training=True)


if __name__ == "__main__":
    args = parser.parse_args()

    args.saved_path = os.path.join(
        f"./{args.log_path}/{args.timestamp}_{args.dataset}_{args.trigger_type}_{args.method}_sd{args.ssl_pretrain_seed}"
    )

    # Defense Baseline: SSL-Cleanse generated triggers
    if args.trigger_path == "":
        args.trigger_path = f"{args.timestamp}_trigger_estimation_{args.method}_{args.dataset}_{args.trigger_type}_SD{args.ssl_cleanse_seed}"

    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)

    main(args)
