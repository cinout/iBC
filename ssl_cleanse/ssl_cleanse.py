import random
import torch.nn.functional as F
import torch
import copy
import torchvision.transforms as T
import torchvision
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import torch.optim as optim
import numpy as np
import os
from ssl_cleanse.inversion import (
    DatasetEval,
    DatasetInit,
    dataloader_cluster,
    eval_knn,
    get_data,
)

from ssl_cleanse.mitigation import ds_train, get_scheduler, outlier

device = "cuda" if torch.cuda.is_available() else "cpu"


def draw_global(base, mean, std, mask, delta):
    delta_norm = torchvision.transforms.functional.normalize(delta, mean, std)
    img = torch.mul(base, 1 - mask) + torch.mul(delta_norm, mask)
    return img


def draw_local(base, mean, std, mask, delta, image_size):
    # trigger_width = random.randint(4, 10)
    trigger_width = int(image_size * random.uniform(0.05, 0.3))

    trigger_location_x = random.uniform(0.1, 0.9)
    trigger_location_y = random.uniform(0.1, 0.9)

    location_x = int((image_size - trigger_width) * trigger_location_x)
    location_y = int((image_size - trigger_width) * trigger_location_y)

    mask = F.interpolate(mask, size=(trigger_width, trigger_width))
    delta = T.functional.normalize(delta, mean, std)
    delta = F.interpolate(delta, size=(trigger_width, trigger_width))

    img = base.clone()

    img[
        :,
        :,
        location_x : location_x + trigger_width,
        location_y : location_y + trigger_width,
    ] = torch.mul(
        base[
            :,
            :,
            location_x : location_x + trigger_width,
            location_y : location_y + trigger_width,
        ],
        1 - mask,
    ) + torch.mul(
        delta, mask
    )
    return base


def norm_mse_loss(x0, x1):
    x0 = F.normalize(x0)
    x1 = F.normalize(x1)
    return 2 - 2 * (x0 * x1).sum(dim=-1).mean()


def evaluate_trigger_during_inversion(
    trigger_type,
    ep,
    args,
    x,
    rep,
    y,
    target,
    mask_tanh,
    delta_tanh,
    backbone,
    feat_dim,
    avg_loss,
    avg_loss_reg,
    statistics,
):

    # apply the learned trigger to all images
    if trigger_type == "local":
        if args.draw_local_trigger_by == "local":
            x_trigger = (
                draw_local(
                    x.to(device),
                    args.mean,
                    args.std,
                    mask_tanh,
                    delta_tanh,
                    args.image_size,
                )
                .detach()
                .to("cpu")
            )
        elif args.draw_local_trigger_by == "global":
            x_trigger = (
                draw_global(
                    x.to(device),
                    args.mean,
                    args.std,
                    mask_tanh,
                    delta_tanh,
                )
                .detach()
                .to("cpu")
            )
    elif trigger_type == "global":
        x_trigger = (
            draw_global(
                x.to(device),
                args.mean,
                args.std,
                mask_tanh,
                delta_tanh,
            )
            .detach()
            .to("cpu")
        )

    # shuffle, and pick 1000 images
    dataloader_eval = DataLoader(
        dataset=DatasetEval(x_trigger, 1000),
        batch_size=100,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # return the percentage of triggered images that are predictd to be the current cluster, aka, attack success rate
    asr_knn = eval_knn(
        device,
        backbone,
        dataloader_eval,
        rep,  # ALL clean images' latent representation
        torch.tensor(y),  # ALL predicted cluster ids
        target,  # current cluster id
        feat_dim,
    )

    if trigger_type == "local":
        print(
            f"ep: {ep}, [trigger 1 local], asr_knn: {asr_knn:.3f}, avg_loss: {avg_loss:.3f}, avg_loss_reg: {avg_loss_reg:.3f}"
        )
    elif trigger_type == "global":
        print(
            f"ep: {ep}, [trigger 2 global], asr_knn: {asr_knn:.3f}, avg_loss: {avg_loss:.3f}, avg_loss_reg: {avg_loss_reg:.3f}"
        )

    if asr_knn > args.attack_succ_threshold and avg_loss_reg < statistics["reg_best"]:
        statistics["mask_best"] = mask_tanh
        statistics["delta_best"] = delta_tanh
        statistics["reg_best"] = avg_loss_reg

    """
    adjusting lambda
    """
    if statistics["lam"] == 0 and asr_knn >= args.attack_succ_threshold:
        statistics["cost_set_counter"] += 1
        if statistics["cost_set_counter"] >= args.patience:  # >=5 patience is 5
            statistics["lam"] = args.lam  # reset to initial value
            statistics["cost_up_counter"] = 0
            statistics["cost_down_counter"] = 0
    else:
        statistics["cost_set_counter"] = 0

    if asr_knn >= args.attack_succ_threshold:
        statistics["cost_up_counter"] += 1
        statistics["cost_down_counter"] = 0
    else:
        statistics["cost_up_counter"] = 0
        statistics["cost_down_counter"] += 1

    if statistics["lam"] != 0 and statistics["cost_up_counter"] >= args.patience:
        # boost up
        statistics["cost_up_counter"] = 0
        statistics["lam"] *= args.lam_multiplier_up

    elif statistics["lam"] != 0 and statistics["cost_down_counter"] >= args.patience:
        # bring down
        statistics["cost_down_counter"] = 0
        statistics["lam"] /= args.lam_multiplier_up


def trigger_inversion(args, backbone, poison, feat_dim):

    backbone = backbone.eval()
    # poisoned encoder f is frozen
    for param in backbone.parameters():
        param.requires_grad = False

    with torch.no_grad():
        """
        prepare dataset
        """

        dataloader = DataLoader(
            dataset=DatasetInit(poison.train_probe_loader),
            batch_size=100,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        transform = T.Compose(
            [
                T.Normalize(args.mean, args.std),
            ]
        )
        # rep: [#total_images, feat_dim]
        # x: [#total_images, 3, image_size, image_size], tensored (value range in 0-1), and transformed by above
        # x_untransformed: same shape as above, tensored (value range in 0-1), but no transformed
        # _ is gt label

        rep, x, x_untransformed, _ = get_data(
            device, backbone, dataloader, args.image_size, feat_dim, transform
        )

        """
        Clustering
        """
        kmeans = KMeans(n_clusters=args.num_clusters, random_state=0, n_init=30).fit(
            rep
        )
        y = kmeans.labels_  # predicted cluster ids

        counts_label = {}  # # of images belonging to cluster i
        for i in range(np.unique(y).shape[0]):
            mask_for_cluster = y == i
            counts_label[i] = mask_for_cluster.sum()  # #images belonging to cluster i

    # estimate trigger for each cluster
    for target in np.unique(y):  # for each cluster

        print(f"Now estimating triggers for cluster {target}...")

        if not os.path.exists(
            os.path.join(args.trigger_path, f"{target}.pth")
        ):  # if trigger is not available yet
            """
            set up data of target cluster and other clusters
            """
            rep_target = rep[y == target]  # [#target_cluster_size, rep_dim]
            x_other = x[y != target]  # [#other_images, 3, image_size, image_size]
            x_other_indices = torch.randperm(x_other.shape[0])[
                : x.shape[0] - max(counts_label.values())
            ]
            x_other_sample = x_other[
                x_other_indices
            ]  # other clusters' images in a shuffled order, [~#other_images, 3, image_size, image_size]

            """
            initialize mask and delta
            """
            mask1 = torch.arctanh(
                (torch.rand([1, 1, args.image_size, args.image_size]) - 0.5) * 2
            ).to(
                device
            )  # value range [-1, 1] -> arctanh -> (-inf, inf)
            delta1 = torch.arctanh(
                (torch.rand([1, 3, args.image_size, args.image_size]) - 0.5) * 2
            ).to(device)

            mask2 = torch.arctanh(
                (torch.rand([1, 1, args.image_size, args.image_size]) - 0.5) * 2
            ).to(
                device
            )  # value range [-1, 1] -> arctanh -> (-inf, inf)
            delta2 = torch.arctanh(
                (torch.rand([1, 3, args.image_size, args.image_size]) - 0.5) * 2
            ).to(device)

            mask1.requires_grad = True
            delta1.requires_grad = True
            mask2.requires_grad = True
            delta2.requires_grad = True

            opt = optim.Adam([delta1, mask1, delta2, mask2], lr=1e-1, betas=(0.5, 0.9))

            trigger_1_statistics = {
                "reg_best": torch.inf,  # records optimal regression loss
                "lam": 0,
                "cost_set_counter": 0,
                "cost_up_counter": 0,
                "cost_down_counter": 0,
                "mask_best": torch.tanh(mask1) / 2 + 0.5,
                "delta_best": torch.tanh(delta1) / 2 + 0.5,
            }
            trigger_2_statistics = {
                "reg_best": torch.inf,  # records optimal regression loss
                "lam": 0,
                "cost_set_counter": 0,
                "cost_up_counter": 0,
                "cost_down_counter": 0,
                "mask_best": torch.tanh(mask2) / 2 + 0.5,
                "delta_best": torch.tanh(delta2) / 2 + 0.5,
            }

            dataloader_train = dataloader_cluster(args, rep_target, x_other_sample)

            for ep in range(1000):
                """
                train and learn triggers
                """
                loss_reg_list_1, loss_list_1 = [], []
                loss_reg_list_2, loss_list_2 = [], []

                for images, target_reps in dataloader_train:

                    images = images.to(device)  # image from another cluster
                    target_reps = target_reps.to(
                        device
                    )  # target cluster image representation

                    """
                    trigger 1: patch-based (size-based, local)
                    """
                    mask1_tanh = torch.tanh(mask1) / 2 + 0.5  # value range (0, 1)
                    delta1_tanh = torch.tanh(delta1) / 2 + 0.5  # value range (0, 1)

                    if args.draw_local_trigger_by == "local":
                        X_R = draw_local(
                            images,
                            args.mean,
                            args.std,
                            mask1_tanh,
                            delta1_tanh,
                            args.image_size,
                        )  # draw trigger mask1 onto the image
                    elif args.draw_local_trigger_by == "global":
                        X_R = draw_global(
                            images,
                            args.mean,
                            args.std,
                            mask1_tanh,
                            delta1_tanh,
                        )  # draw trigger mask1 onto the image

                    loss_asr = norm_mse_loss(target_reps, backbone(X_R))
                    loss_reg = torch.mean(mask1_tanh)

                    loss = loss_asr + trigger_1_statistics["lam"] * loss_reg

                    opt.zero_grad()
                    loss.backward(retain_graph=True)
                    opt.step()

                    # loss_asr_list.append(loss_asr.item())
                    loss_reg_list_1.append(loss_reg.item())
                    loss_list_1.append(loss.item())

                    """
                    trigger 2: magnitude-based (global)
                    """

                    mask2_tanh = torch.tanh(mask2) / 2 + 0.5  # value range (0, 1)
                    delta2_tanh = torch.tanh(delta2) / 2 + 0.5  # value range (0, 1)
                    X_R = draw_global(
                        images, args.mean, args.std, mask2_tanh, delta2_tanh
                    )
                    loss_asr = norm_mse_loss(target_reps, backbone(X_R))
                    loss_reg = torch.mean(mask2_tanh * delta2_tanh)

                    loss = loss_asr + trigger_2_statistics["lam"] * loss_reg

                    opt.zero_grad()
                    loss.backward(retain_graph=True)
                    opt.step()

                    # loss_asr_list.append(loss_asr.item())
                    loss_reg_list_2.append(loss_reg.item())
                    loss_list_2.append(loss.item())

                # avg_loss_asr = torch.tensor(loss_asr_list).mean()
                avg_loss_reg_1 = torch.tensor(loss_reg_list_1).mean()
                avg_loss_1 = torch.tensor(loss_list_1).mean()
                avg_loss_reg_2 = torch.tensor(loss_reg_list_2).mean()
                avg_loss_2 = torch.tensor(loss_list_2).mean()

                """
                evaluate trigger 1
                """

                evaluate_trigger_during_inversion(
                    "local",
                    ep,
                    args,
                    x,
                    rep,
                    y,
                    target,
                    mask1_tanh,
                    delta1_tanh,
                    backbone,
                    feat_dim,
                    avg_loss_1,
                    avg_loss_reg_1,
                    trigger_1_statistics,
                )

                """
                evaluate trigger 2
                """
                evaluate_trigger_during_inversion(
                    "global",
                    ep,
                    args,
                    x,
                    rep,
                    y,
                    target,
                    mask2_tanh,
                    delta2_tanh,
                    backbone,
                    feat_dim,
                    avg_loss_2,
                    avg_loss_reg_2,
                    trigger_2_statistics,
                )

            os.makedirs(args.trigger_path, exist_ok=True)
            torch.save(
                {
                    "mask1": trigger_1_statistics["mask_best"],
                    "delta1": trigger_1_statistics["delta_best"],
                    "reg1": (
                        trigger_1_statistics["reg_best"]
                        if trigger_1_statistics["reg_best"] != torch.inf
                        else 1
                    ),
                    "mask2": trigger_2_statistics["mask_best"],
                    "delta2": trigger_2_statistics["delta_best"],
                    "reg2": (
                        trigger_2_statistics["reg_best"]
                        if trigger_2_statistics["reg_best"] != torch.inf
                        else 1
                    ),
                },
                os.path.join(args.trigger_path, f"{target}.pth"),
            )

    return (x_untransformed, y)


def trigger_mitigation(args, backbone, trainset_data):
    """
    setup frozen triggered encoder and learnable encoder
    """

    backbone_unlearn_trigger = copy.deepcopy(backbone)
    backbone_unlearn_trigger = backbone_unlearn_trigger.train()
    for param in backbone_unlearn_trigger.parameters():
        param.requires_grad = True

    backbone = backbone.eval()

    """
    set up optimizer and scheduler
    """
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, backbone_unlearn_trigger.parameters()),
        lr=3e-3,
        weight_decay=1e-6,
    )
    scheduler = get_scheduler(args, optimizer)
    lr_warmup = 0
    torch.backends.cudnn.benchmark = True

    """
    setup dataloader
    """
    dataloader = DataLoader(
        dataset=ds_train(args, trainset_data),
        batch_size=128,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    trigger_masks1, trigger_deltas1, trigger_regs1 = [], [], []
    trigger_masks2, trigger_deltas2, trigger_regs2 = [], [], []

    for target in range(args.num_clusters):
        trigger_path = os.path.join(args.trigger_path, f"{target}.pth")
        trigger = torch.load(trigger_path, map_location=device)

        trigger_masks1.append(trigger["mask1"].detach())
        trigger_deltas1.append(trigger["delta1"].detach())
        trigger_regs1.append(trigger["reg1"])
        trigger_masks2.append(trigger["mask2"].detach())
        trigger_deltas2.append(trigger["delta2"].detach())
        trigger_regs2.append(trigger["reg2"])

    trigger_masks1 = torch.cat(
        trigger_masks1, dim=0
    )  # [#clusters, 1, imgsize, imgsize]
    trigger_deltas1 = torch.cat(
        trigger_deltas1, dim=0
    )  # [#clusters, 3, imgsize, imgsize]
    trigger_masks2 = torch.cat(trigger_masks2, dim=0)
    trigger_deltas2 = torch.cat(trigger_deltas2, dim=0)

    trigger_regs1 = torch.tensor(trigger_regs1)  # [#clusters,]
    trigger_regs2 = torch.tensor(trigger_regs2)

    trigger1_top_indices = outlier(trigger_regs1)  # [#clusters,] list, local
    trigger2_top_indices = outlier(trigger_regs2)  # global

    for ep in range(args.mitigate_epochs):

        for clean_view_1, clean_view_2, cluster_ids in dataloader:
            clean_view_1 = clean_view_1.to(device)  # [bs, 3, img_size, img_size]
            clean_view_2 = clean_view_2.to(device)  # [bs, 3, img_size, img_size]
            # clean_view_3 = clean_view_3.to(device)  # [bs, 3, img_size, img_size]
            cluster_ids = cluster_ids.to(
                device
            )  # [bs], the cluster id of current image

            if lr_warmup < 500:
                lr_scale = (lr_warmup + 1) / 500
                for pg in optimizer.param_groups:
                    pg["lr"] = 3e-3 * lr_scale
                lr_warmup += 1
            optimizer.zero_grad()

            with torch.no_grad():
                clean_view_1_feature = backbone(clean_view_1)

            compare_views = []
            for idx, view2 in enumerate(clean_view_2):
                # view2.shape: [3, img_size, img_size]
                view2 = view2.unsqueeze(0)  # [1, 3, img_size, img_size]

                use_clean_view = random.random() < 0.5

                if use_clean_view:
                    """
                    # no trigger added
                    """
                    compare_views.append(view2)
                else:
                    """
                    # let's add trigger
                    """

                    use_local_trigger = random.random() < 0.5
                    cid = cluster_ids[idx]  # cluster id of the image

                    if use_local_trigger:
                        """
                        # ADD LOCAL TRIGGER
                        """
                        trigger_index = random.choice(
                            [index for index in trigger1_top_indices if index != cid]
                        )
                        mask = trigger_masks1[trigger_index].unsqueeze(
                            0
                        )  # [1, 1, imgsize, imgsize]
                        delta = trigger_deltas1[trigger_index].unsqueeze(
                            0
                        )  # [1, 3, imgsize, imgsize]

                        if args.draw_local_trigger_by == "local":

                            new_view = draw_local(
                                view2,
                                args.mean,
                                args.std,
                                mask,
                                delta,
                                args.image_size,
                            )
                        elif args.draw_local_trigger_by == "global":
                            new_view = draw_global(
                                view2, args.mean, args.std, mask, delta
                            )
                    else:
                        """
                        # ADD GLOBAL TRIGGER
                        """
                        trigger_index = random.choice(
                            [index for index in trigger2_top_indices if index != cid]
                        )
                        mask = trigger_masks2[trigger_index].unsqueeze(
                            0
                        )  # [1, 1, imgsize, imgsize]
                        delta = trigger_deltas2[trigger_index].unsqueeze(
                            0
                        )  # [1, 3, imgsize, imgsize]
                        new_view = draw_global(view2, args.mean, args.std, mask, delta)

                    compare_views.append(new_view)

            compare_views = torch.cat(compare_views, dim=0)

            loss_sum = norm_mse_loss(
                clean_view_1_feature, backbone_unlearn_trigger(compare_views)
            )

            loss_sum.backward()

            optimizer.step()

        scheduler.step()
        print(f"epoch {ep}, loss: {loss_sum.item()}")

    return backbone_unlearn_trigger
