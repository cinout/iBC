import os
import PIL
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as T
import random
from PIL import ImageFilter, Image
import torchvision.transforms.functional as F
import torch
from torch.utils.data import Dataset


CONSISTENCY = 1.4826


def outlier(l1_norm_list, combined=False):
    ##### (Option 1) return all indices
    return list(range(len(l1_norm_list)))

    median = torch.median(l1_norm_list)  # median of the list
    median_dist_to_median = CONSISTENCY * torch.median(torch.abs(l1_norm_list - median))
    scores = torch.abs(l1_norm_list - median) / median_dist_to_median

    #### (Option 2) use ||>2 as indicated in the paper, but we need to be aware of potential zero set issue
    # indices = torch.nonzero(scores > 2).flatten()
    # print(f"indices.shape: {indices.shape}")

    #### (Option 3) return top 2
    if combined:
        _, indices = torch.topk(scores, k=4, largest=True, sorted=True)
    else:
        _, indices = torch.topk(scores, k=1, largest=True, sorted=True)

    return indices.tolist()


def get_scheduler(args, optimizer):
    m = [args.mitigate_epochs - a for a in args.drop]
    return MultiStepLR(optimizer, milestones=m, gamma=args.drop_gamma)


def aug_transform(args):
    """augmentation transform generated from config"""
    return T.Compose(
        [
            T.RandomResizedCrop(
                args.image_size,
                scale=(args.crop_s0, args.crop_s1),
                ratio=(args.crop_r0, args.crop_r1),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomApply(
                [T.ColorJitter(args.cj0, args.cj1, args.cj2, args.cj3)], p=args.cj_p
            ),
            T.RandomGrayscale(p=args.gs_p),
            T.RandomApply([RandomBlur(0.1, 2.0)], p=0.5),
            T.RandomHorizontalFlip(p=args.hf_p),
            T.ToTensor(),
        ]
    )


class RandomBlur:
    def __init__(self, r0, r1):
        self.r0, self.r1 = r0, r1

    def __call__(self, image):
        r = random.uniform(self.r0, self.r1)
        return image.filter(ImageFilter.GaussianBlur(radius=r))


class FileListDataset(Dataset):
    def __init__(
        self,
        args,
        trainset_data,  # a tuple of (x_untransformed, y), where y is cluster id, x_untransformed values are in [0,1]
    ):
        self.basic_transform = T.Compose(
            [aug_transform(args), T.Normalize(mean=args.mean, std=args.std)]
        )
        self.num_clusters = args.num_clusters
        self.image_list = trainset_data[
            0
        ]  # [#total=1% trainset, 3, image_size, image_size], values in [0,1]
        self.cluster_list = trainset_data[1]  # [#total=1% trainset]

    def __getitem__(self, idx):

        image = self.image_list[idx]
        image = torch.permute(image, (1, 2, 0))
        image = image * 255.0
        image = torch.clamp(image, 0, 255)
        image = np.array(image.cpu(), dtype=np.uint8)
        image = PIL.Image.fromarray(image)  # PIL format
        clean_view_1 = self.basic_transform(image)  # tensor, [3, img_size, img_size]
        clean_view_2 = self.basic_transform(image)  # tensor, [3, img_size, img_size]
        # clean_view_3 = self.basic_transform(image)  # tensor

        cluster_id = self.cluster_list[idx]

        # valid_trigger_indices = [
        #     index for index in range(self.num_clusters) if index != cluster_id
        # ]
        # trigger_index = random.choice(valid_trigger_indices)

        return clean_view_1, clean_view_2, cluster_id

    def __len__(self):
        return self.cluster_list.shape[0]


def ds_train(args, trainset_data):

    return FileListDataset(args, trainset_data)
