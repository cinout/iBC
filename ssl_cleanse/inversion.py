from torch.utils import data
import torch


class DatasetInit(data.Dataset):
    def __init__(self, train_probe_loader):

        train_probe_dataset = train_probe_loader.dataset
        self.original_length = len(train_probe_dataset)

        self.file_list = train_probe_dataset[:][
            :2
        ]  # tuple, image [500, 3, 32, 32] and label [500]

    def __getitem__(self, idx):
        image, target = self.file_list[0][idx], self.file_list[1][idx]

        return image, target

    def __len__(self):
        return self.original_length


class DatasetEval(data.Dataset):
    def __init__(self, x, sample_size):
        x_indices = torch.randint(0, x.shape[0], (sample_size,))
        self.x = x[x_indices]  # [sample_size, 3, image_size, image_size]

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return self.x.shape[0]


class DatasetCluster(data.Dataset):
    def __init__(self, rep_target, x_other_sample):

        self.rep_target = rep_target  # [#target_cluster_size, rep_dim]
        self.x_other_sample = (
            x_other_sample  # [~#other_images, 3, image_size, image_size]
        )

        # this means that each imgae in other cluster is related to the representation of one random targer cluster image
        self.rep_target_indices = torch.randint(
            0, rep_target.shape[0], (x_other_sample.shape[0],)
        )  # [~#other_images,], values in range [0, #target_cluster_size]

    def __getitem__(self, idx):
        image = self.x_other_sample[idx]  # why the image is from another cluster?
        rep_target = self.rep_target[self.rep_target_indices[idx]]

        return image, rep_target

    def __len__(self):
        return self.x_other_sample.shape[0]


def dataloader_cluster(args, rep_target, x_other_sample):
    return data.DataLoader(
        dataset=DatasetCluster(rep_target, x_other_sample),
        batch_size=32,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_data(device, encoder, loader, image_size, output_size, transform):
    # output_size: feat dim

    input_size = (3, image_size, image_size)
    xs = torch.empty(
        len(loader), loader.batch_size, *input_size, dtype=torch.float32, device=device
    )
    xs_untransformed = torch.empty(
        len(loader), loader.batch_size, *input_size, dtype=torch.float32, device=device
    )
    ys = torch.empty(len(loader), loader.batch_size, dtype=torch.long, device=device)
    reps = torch.empty(
        len(loader), loader.batch_size, output_size, dtype=torch.float32, device=device
    )

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            xs_untransformed[i] = x
            x = transform(x)

            reps[i] = encoder(x)
            xs[i] = x
            ys[i] = y
    xs = xs.view(-1, *input_size)  # [#total_images, 3, image_size, image_size]
    xs_untransformed = xs_untransformed.view(
        -1, *input_size
    )  # [#total_images, 3, image_size, image_size]
    ys = ys.view(-1)  # [#total_images]
    reps = reps.view(-1, output_size)  # [#total_images, feat_dim]
    return reps.to("cpu"), xs.to("cpu"), xs_untransformed.to("cpu"), ys.to("cpu")


def eval_knn(device, encoder, loader, rep_center, y, target, output_size, k=1):
    # loader: triggered images
    # rep_center: ALL representation of clean images
    # y: ALL predicted cluster ids
    # target: current cluster id

    rep_center, y = rep_center.to(device), y.to(device)

    with torch.no_grad():
        rep = torch.empty(
            (len(loader), loader.batch_size, output_size),
            dtype=torch.float,
            device=device,
        )
        for i, x in enumerate(loader):
            x = x.to(device)
            rep[i] = encoder(x)
        rep = rep.view(
            (-1, output_size)
        )  # representation of triggered images, [#triggered_images=1000, feat_dim]
        d_t = torch.cdist(
            rep, rep_center
        )  # Computes batched the p-norm distance between each pair of the two collections
        topk_t = torch.topk(
            d_t, k=k, dim=1, largest=False
        )  # smallest dist to clean reps, [#triggered_images=1000,]
        labels_t = y[
            topk_t.indices
        ]  # predicted cluster ids of the K closest reps, [#triggered_images=1000, k=1]

        pred_t = torch.empty(
            rep.shape[0], device=device
        )  # [#triggered_images,], which records the cluster id of most closest reps [WE want the triggered image to have the closest representation to clean images of the current cluster, as that is the goal of attack]
        for i in range(len(labels_t)):
            x = labels_t[i].unique(
                return_counts=True
            )  # tuple, first item is unique elements, second item is their counts
            pred_t[i] = x[0][x[1].argmax()]  # choose the element with highest count
        asr = (
            (pred_t == target).float().mean().item()
        )  # [Attack Success Rate] what percentage of triggered images are predictd to be current cluster

    return asr
