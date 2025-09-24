from torch.utils.data import DataLoader, TensorDataset
import random, os
import numpy as np
from PIL import Image
from torch import Tensor
from typing import Callable
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from kornia import augmentation as aug
import PIL
from PIL import ImageFilter

device = "cuda" if torch.cuda.is_available() else "cpu"


# class Subset(torch.utils.data.Subset):
#     """Overwrite subset class to provide class methods of main class."""

#     def __getattr__(self, name):
#         """Call this only if all attributes of Subset are exhausted."""
#         return getattr(self.dataset, name)


def get_data_and_label(paths, size):
    images = []
    targets = []
    for i, item in enumerate(paths):
        if i % 1000 == 0:
            print(f"transform data to {i}/{len(paths)}")

        img = Image.open(item.split()[0]).convert("RGB")
        img = img.resize((size, size))
        img = np.asarray(img).astype(np.float32) / 255.0  # tensorised
        img = torch.tensor(img)
        img = torch.permute(img, (2, 0, 1))  # shape: [c=3, h, w], value: [0, 1]
        images.append(img)

        target = int(item.split()[1])
        target = torch.tensor(target, dtype=torch.long)
        targets.append(target)

        # remove later
        # if i % 200 == 0:
        #     break

    return images, targets


# def tensor_back_to_PIL(input):
#     input = torch.permute(input, (1, 2, 0))
#     input = input * 255.0
#     input = torch.clamp(input, 0, 255)
#     input = np.array(input, dtype=np.uint8)
#     input = PIL.Image.fromarray(input)

#     return input


class NCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n):
        self.base_transform = base_transform
        self.n = n

    def __call__(self, x):
        aug_image = []

        for _ in range(self.n):
            aug_image.append(self.base_transform(x))

        return aug_image


class PoisonAgent:
    def __init__(
        self,
        args,
        fre_agent,
        trainset,
        validset,
        memory_loader,
        magnitude_train,
        magnitude_val,
    ):
        self.args = args
        self.trainset = trainset
        self.validset = validset
        self.memory_loader = memory_loader
        self.poison_num = int(
            len(trainset) * self.args.poison_ratio
        )  #  determine how many to be poisoned
        self.fre_poison_agent = fre_agent  # who does the poisoning work

        self.magnitude_train = magnitude_train
        self.magnitude_val = magnitude_val

        ss_views_aug = [
            transforms.RandomResizedCrop(
                self.args.image_size,
                scale=(self.args.rrc_scale_min, self.args.rrc_scale_max),
                ratio=(0.2, 5),
            ),
            transforms.RandomPerspective(p=0.5),
        ]

        # used in find_trigger_channels_or_poisoned_images. for augment an image into multiple views, and finding trigger channels
        self.ss_transform = NCropsTransform(
            transforms.Compose(ss_views_aug), self.args.num_views
        )

        print(
            f"Initializing Poison data (chosen images, examples, sources, labels) with random seed {self.args.ssl_pretrain_seed}"
        )

        (
            self.train_pos_loader,
            self.test_clean_loader,
            self.test_pos_loader,
            self.memory_loader,
            self.train_probe_loader,
        ) = self.choose_poisons_randomly()

    def choose_poisons_randomly(self):
        """
        basic data manipulation
        """
        if self.args.dataset == "imagenet100":
            ###### NOTICE: quick_fetch_tensors_imagenet100_XXX.pth are generated with SD42 and should be used with SSL poisoned model with SD42
            # TODO: handle this quickfetch
            if os.path.exists(
                f"quick_fetch_tensors_imagenet100_{self.args.trigger_type}.pth"
            ):
                # tensors are saved to local disk already
                pass
            else:

                train_paths = self.trainset
                val_paths = self.validset

                print("transform training data")

                x_train_tensor, y_train_tensor = get_data_and_label(
                    train_paths, self.args.image_size
                )

                x_train_tensor = torch.stack(x_train_tensor)
                y_train_tensor = torch.stack(y_train_tensor)

                print("transform validation data")

                x_test_tensor, y_test_tensor = get_data_and_label(
                    val_paths, self.args.image_size
                )

                x_test_tensor = torch.stack(x_test_tensor)
                y_test_tensor = torch.stack(y_test_tensor)

                # memory

                x_memory_tensor = x_train_tensor.clone().detach()
                y_memory_tensor = y_train_tensor.clone().detach()

        else:
            # CIFAR-10/100

            # get image data
            x_train_np, x_test_np = (
                self.trainset.data.astype(np.float32)
                / 255.0,  # .data returns numpy array; value range: 0-254; shape: [50000, 32, 32, 3]
                self.validset.data.astype(np.float32) / 255.0,
            )
            x_memory_np = self.memory_loader.dataset.data.astype(np.float32) / 255.0

            # get labels
            y_train_np, y_test_np = np.array(self.trainset.targets), np.array(
                self.validset.targets
            )
            y_memory_np = np.array(self.memory_loader.dataset.targets)

            # turn from np to torch tensor [keep all y]
            x_train_tensor, y_train_tensor = torch.tensor(x_train_np), torch.tensor(
                y_train_np, dtype=torch.long
            )
            x_test_tensor, y_test_tensor = torch.tensor(x_test_np), torch.tensor(
                y_test_np, dtype=torch.long
            )
            x_memory_tensor = torch.tensor(x_memory_np)
            y_memory_tensor = torch.tensor(y_memory_np, dtype=torch.long)

            # shift image data into [bs, c=3, h, w] shape, [update all x_]
            x_train_tensor = x_train_tensor.permute(
                0, 3, 1, 2
            )  # shape: [50000, 3, 32, 32]; value range: [0, 1]
            x_test_tensor = x_test_tensor.permute(0, 3, 1, 2)
            x_memory_tensor = x_memory_tensor.permute(0, 3, 1, 2)

        """
        # POISONed Validation Set
        """
        if self.args.dataset == "imagenet100" and os.path.exists(
            f"quick_fetch_tensors_imagenet100_{self.args.trigger_type}.pth"
        ):
            pass
        else:
            # test set (poison all images)
            if self.args.trigger_type == "ftrojan":
                x_test_pos_tensor, y_test_pos_tensor = (
                    self.fre_poison_agent.Poison_Frequency_Diff(
                        x_test_tensor.clone().detach(),
                        y_test_tensor.clone().detach(),
                        self.magnitude_val,
                    )
                )
            elif self.args.trigger_type == "htba":
                x_test_pos_tensor, y_test_pos_tensor = (
                    self.fre_poison_agent.Poison_HTBA(
                        x_test_tensor.clone().detach(),
                        y_test_tensor.clone().detach(),
                    )
                )

            # why? is it because above code does not assign correct label to poisoned images?
            # [YES], the Poison_Frequency_Diff() function only poisons image data, but does not pollute label.
            y_test_pos_tensor = (
                torch.ones_like(y_test_pos_tensor, dtype=torch.long)
                * self.args.target_class
            )

            # uncomment to show poisoned image example
            # tensor_back_to_PIL(x_test_pos_tensor[0])

        """
        # POISONed Train Set (for stage 1 attack)
        """
        if self.args.dataset == "imagenet100" and os.path.exists(
            f"quick_fetch_tensors_imagenet100_{self.args.trigger_type}.pth"
        ):
            pass
        else:
            poison_index = torch.where(y_train_tensor == self.args.target_class)[0]
            poison_index = poison_index[: self.poison_num]

            # train set (poison only a portion of train images)
            if self.args.trigger_type == "ftrojan":

                x_train_tensor[poison_index], y_train_tensor[poison_index] = (
                    self.fre_poison_agent.Poison_Frequency_Diff(
                        x_train_tensor[poison_index],
                        y_train_tensor[poison_index],
                        self.magnitude_train,
                    )
                )
            elif self.args.trigger_type == "htba":
                x_train_tensor[poison_index], y_train_tensor[poison_index] = (
                    self.fre_poison_agent.Poison_HTBA(
                        x_train_tensor[poison_index],
                        y_train_tensor[poison_index],
                    )
                )

        # for convenience (saved in disk)
        if self.args.dataset == "imagenet100" and os.path.exists(
            f"quick_fetch_tensors_imagenet100_{self.args.trigger_type}.pth"
        ):
            tensor_dict = torch.load(
                f"quick_fetch_tensors_imagenet100_{self.args.trigger_type}.pth",
                # map_location=device,
            )

            x_train_tensor = tensor_dict["x_train_tensor"]
            y_train_tensor = tensor_dict["y_train_tensor"]
            x_test_tensor = tensor_dict["x_test_tensor"]
            y_test_tensor = tensor_dict["y_test_tensor"]
            x_test_pos_tensor = tensor_dict["x_test_pos_tensor"]
            y_test_pos_tensor = tensor_dict["y_test_pos_tensor"]
            x_memory_tensor = tensor_dict["x_memory_tensor"]
            y_memory_tensor = tensor_dict["y_memory_tensor"]
            poison_index = tensor_dict["poison_index"]

        """
        Create dataloaders
        """
        train_is_poisoned = torch.zeros_like(y_train_tensor)
        train_is_poisoned[poison_index] = 1

        # for image indexing, used for input-filtering methods
        train_index = torch.tensor(list(range(len(self.trainset))), dtype=torch.long)
        test_index = torch.tensor(list(range(len(self.validset))), dtype=torch.long)
        memory_index = torch.tensor(list(range(len(x_memory_tensor))), dtype=torch.long)

        # contain both CLEAN and a portion of POISONED images
        train_loader = DataLoader(
            (
                TensorDataset(
                    x_train_tensor,
                    train_is_poisoned,  # FIXME: can be removed later, for debug purpose only
                    y_train_tensor,
                    train_index,
                )
            ),
            batch_size=self.args.pretrain_batch_size,
            sampler=None,
            shuffle=True,
        )

        # clean validation set (used in knn eval only, in base.py)
        test_clean_loader = DataLoader(
            TensorDataset(x_test_tensor, y_test_tensor, test_index),
            batch_size=self.args.linear_probe_batch_size,
            shuffle=False,
        )

        # poisoned validation set (used in knn eval only, in base.py)
        test_pos_loader = DataLoader(
            TensorDataset(
                x_test_pos_tensor, y_test_pos_tensor, y_test_tensor, test_index
            ),  # y_test_tensor serves as the original label tensor (for correcting ASR)
            batch_size=self.args.linear_probe_batch_size,
            shuffle=False,
        )

        # memory set is never poisoned (used in knn eval only, in base.py)
        memory_loader = DataLoader(
            TensorDataset(x_memory_tensor, y_memory_tensor, memory_index),
            batch_size=self.args.linear_probe_batch_size,
            shuffle=False,
        )

        # create 1% train probe set for linear classifier training
        id_and_label = dict()  # choose 1% images for each label to achieve balance
        for i, label in enumerate(y_memory_tensor.cpu().detach().numpy()):
            if label in id_and_label.keys():
                id_and_label[label].append(i)
            else:
                id_and_label[label] = [i]
        x_probe_tensor = []
        y_probe_tensor = []
        for label, indices in id_and_label.items():
            # for each label (class)
            random.shuffle(indices)
            indices = torch.tensor(
                indices[: int(len(indices) * self.args.probe_set_percent)]
            )

            x_probe_tensor.append(x_memory_tensor[indices])
            y_probe_tensor.append(y_memory_tensor[indices])
        x_probe_tensor = torch.cat(x_probe_tensor, dim=0)
        y_probe_tensor = torch.cat(y_probe_tensor, dim=0)
        probe_index = torch.tensor(
            list(range(len(x_probe_tensor))), dtype=torch.long
        )  # indexed based on x_probe_tensor, not on the whole trainset

        train_probe_loader = DataLoader(
            TensorDataset(x_probe_tensor, y_probe_tensor, probe_index),
            batch_size=self.args.linear_probe_batch_size,
            shuffle=True,
        )

        return (
            train_loader,
            test_clean_loader,
            test_pos_loader,
            memory_loader,  # for kNN eval
            train_probe_loader,  # 1% clean images
        )


class RandomApply(nn.Module):
    def __init__(self, fn: Callable, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x if random.random() > self.p else self.fn(x)


def set_aug_diff(args):
    if args.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        args.mean = mean
        args.std = std
        args.num_classes = 10

    elif args.dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        args.mean = mean
        args.std = std
        args.num_classes = 100

    # used for imagenet100
    elif args.dataset == "imagenet100":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        args.mean = mean
        args.std = std
        args.num_classes = 100

    else:
        raise ValueError(args.dataset)

    ####################### Define Diff Transforms #######################

    # class GaussianBlur(object):
    #     def __call__(self, x):
    #         sigma = np.random.uniform(0.1, 2.0)
    #         x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
    #         return x

    # if "cifar" in args.dataset or args.dataset == "imagenet100":
    if True:
        # this is applied during training, not during poison generation

        #  use different train_transform for different SSL methods
        # if args.method == "byol":
        #     # transform_1 = transforms.Compose(
        #     #     [
        #     #         aug.RandomResizedCrop(
        #     #             size=(args.image_size, args.image_size), scale=(0.2, 1.0)
        #     #         ),
        #     #         aug.RandomHorizontalFlip(),
        #     #         RandomApply(aug.ColorJitter(0.4, 0.4, 0.2, 0.1), p=0.8),
        #     #         aug.RandomGrayscale(p=0.2),

        #     #         transforms.RandomApply(
        #     #             [transforms.GaussianBlur(kernel_size=(3, 7))], p=1.0
        #     #         ),
        #     #         normalize,
        #     #     ]
        #     # )
        #     # transform_2 = transforms.Compose(
        #     #     [
        #     #         aug.RandomResizedCrop(
        #     #             size=(args.image_size, args.image_size), scale=(0.2, 1.0)
        #     #         ),
        #     #         aug.RandomHorizontalFlip(),
        #     #         RandomApply(aug.ColorJitter(0.4, 0.4, 0.2, 0.1), p=0.8),
        #     #         aug.RandomGrayscale(p=0.2),

        #     #         transforms.RandomApply(
        #     #             [transforms.GaussianBlur(kernel_size=(3, 7))], p=0.1
        #     #         ),
        #     #         aug.RandomSolarize(p=0.2),
        #     #         normalize,
        #     #     ]
        #     # )

        #     transform_1 = nn.Sequential(
        #         aug.RandomResizedCrop(
        #             size=(args.image_size, args.image_size), scale=(0.2, 1.0)
        #         ),
        #         aug.RandomHorizontalFlip(),
        #         RandomApply(aug.ColorJitter(0.4, 0.4, 0.2, 0.1), p=0.8),
        #         aug.RandomGrayscale(p=0.2),
        #         aug.RandomGaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=1.0),
        #         normalize,
        #     )
        #     transform_2 = nn.Sequential(
        #         aug.RandomResizedCrop(
        #             size=(args.image_size, args.image_size), scale=(0.2, 1.0)
        #         ),
        #         aug.RandomHorizontalFlip(),
        #         RandomApply(aug.ColorJitter(0.4, 0.4, 0.2, 0.1), p=0.8),
        #         aug.RandomGrayscale(p=0.2),
        #         aug.RandomGaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=0.1),
        #         aug.RandomSolarize(p=0.2),
        #         normalize,
        #     )

        #     train_transform = (transform_1, transform_2)

        # elif args.method == "simclr":
        #     # transform = transforms.Compose(
        #     #     [
        #     #         aug.RandomResizedCrop(
        #     #             size=(args.image_size, args.image_size), scale=(0.2, 1.0)
        #     #         ),
        #     #         aug.RandomHorizontalFlip(),
        #     #         RandomApply(aug.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
        #     #         aug.RandomGrayscale(p=0.2),
        #     #         transforms.RandomApply(
        #     #             [transforms.GaussianBlur(kernel_size=(3, 7))], p=0.5
        #     #         ),
        #     #         normalize,
        #     #     ]
        #     # )

        #     transform_1 = nn.Sequential(
        #         aug.RandomResizedCrop(
        #             size=(args.image_size, args.image_size), scale=(0.2, 1.0)
        #         ),
        #         aug.RandomHorizontalFlip(),
        #         RandomApply(aug.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
        #         aug.RandomGrayscale(p=0.2),
        #         aug.RandomGaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=0.5),
        #         normalize,
        #     )
        #     transform_2 = nn.Sequential(
        #         aug.RandomResizedCrop(
        #             size=(args.image_size, args.image_size), scale=(0.2, 1.0)
        #         ),
        #         aug.RandomHorizontalFlip(),
        #         RandomApply(aug.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
        #         aug.RandomGrayscale(p=0.2),
        #         aug.RandomGaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=0.5),
        #         normalize,
        #     )

        #     train_transform = (transform_1, transform_2)

        # elif args.method == "mocov2":
        #     # transform = transforms.Compose(
        #     #     [
        #     #         aug.RandomResizedCrop(
        #     #             size=(args.image_size, args.image_size), scale=(0.2, 1.0)
        #     #         ),
        #     #         aug.RandomHorizontalFlip(),
        #     #         RandomApply(aug.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
        #     #         aug.RandomGrayscale(p=0.2),
        #     #         transforms.RandomApply(
        #     #             [transforms.GaussianBlur(kernel_size=(3, 7))], p=0.5
        #     #         ),
        #     #         normalize,
        #     #     ]
        #     # )

        #     transform_1 = nn.Sequential(
        #         aug.RandomResizedCrop(
        #             size=(args.image_size, args.image_size), scale=(0.2, 1.0)
        #         ),
        #         aug.RandomHorizontalFlip(),
        #         RandomApply(aug.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
        #         aug.RandomGrayscale(p=0.2),
        #         aug.RandomGaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=0.5),
        #         normalize,
        #     )
        #     transform_2 = nn.Sequential(
        #         aug.RandomResizedCrop(
        #             size=(args.image_size, args.image_size), scale=(0.2, 1.0)
        #         ),
        #         aug.RandomHorizontalFlip(),
        #         RandomApply(aug.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
        #         aug.RandomGrayscale(p=0.2),
        #         aug.RandomGaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=0.5),
        #         normalize,
        #     )
        #     train_transform = (transform_1, transform_2)

        # train_transform assumes input images are already tensorized, so we use kornia.augmentation to augment images, which accepts tensoirized inputs
        train_transform = nn.Sequential(
            aug.RandomResizedCrop(
                size=(args.image_size, args.image_size), scale=(0.2, 1.0)
            ),
            aug.RandomHorizontalFlip(),
            RandomApply(aug.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
            aug.RandomGrayscale(p=0.2),
            aug.Normalize(mean=mean, std=std),
        )

        # applied to a PIL image (NEVER used?)
        transform_load = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]  # arrive here
        )

    else:
        raise NotImplementedError

    ####################### Define Datasets #######################
    if args.dataset == "cifar10":

        train_dataset = CIFAR10(
            root=args.data_path, train=True, transform=transform_load, download=True
        )
        # ft_dataset = CIFAR10(
        #     root=args.data_path, transform=transform_load, download=False
        # )
        test_dataset = CIFAR10(
            root=args.data_path, train=False, transform=transform_load, download=True
        )
        memory_dataset = CIFAR10(
            root=args.data_path, train=True, transform=transform_load, download=False
        )

    elif args.dataset == "cifar100":
        train_dataset = CIFAR100(
            root=args.data_path, train=True, transform=transform_load, download=True
        )
        # ft_dataset = CIFAR100(
        #     root=args.data_path, transform=transform_load, download=False
        # )
        test_dataset = CIFAR100(
            root=args.data_path, train=False, transform=transform_load, download=True
        )
        memory_dataset = CIFAR100(
            root=args.data_path, train=True, transform=transform_load, download=False
        )
    elif args.dataset == "imagenet100":
        train_file_path = "./datasets/imagenet100_train_clean_filelist.txt"
        val_file_path = "./datasets/imagenet100_val_clean_filelist.txt"
        with open(train_file_path, "r") as f:
            train_file_list = f.readlines()
            train_file_list = [row.rstrip() for row in train_file_list]
        f.close()
        with open(val_file_path, "r") as f:
            val_file_list = f.readlines()
            val_file_list = [row.rstrip() for row in val_file_list]
        f.close()

        train_dataset = train_file_list
        memory_dataset = train_file_list
        test_dataset = val_file_list
        # ft_dataset = val_file_list  # dummy placeholder here, not used anyway
    else:
        raise NotImplementedError

    # memory loader is train set without shuffle
    memory_loader = torch.utils.data.DataLoader(
        memory_dataset,
        args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return (
        train_dataset,  # [double check] used as PoisonAgent's train_dataset
        test_dataset,  # [double check] used as PoisonAgent's val_dataset
        memory_loader,  #  [double check] used as PoisonAgent's memory_loader
        train_transform,  #  [double check] used in train_loader iteration, SSL methods' augmentation pipeline
    )


class CIFAR10(datasets.CIFAR10):
    """Super-class CIFAR10 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        # NEVER GETS CALLED, ignore
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index


class CIFAR100(datasets.CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index
