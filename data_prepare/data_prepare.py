from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
from PIL import Image
from torch import Tensor
from typing import Callable
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from kornia import augmentation as aug

device = "cuda" if torch.cuda.is_available() else "cpu"


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

    return images, targets


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
    ):
        self.args = args
        self.trainset = trainset
        self.validset = validset
        self.memory_loader = memory_loader
        self.poison_num = int(
            len(trainset) * self.args.poison_ratio
        )  #  determine how many to be poisoned
        self.fre_poison_agent = fre_agent  # who does the poisoning work

        self.magnitude_train = args.magnitude_train
        self.magnitude_val = args.magnitude_val

        ss_views_aug = [
            transforms.RandomResizedCrop(
                self.args.image_size,
                scale=(self.args.rrc_scale_min, self.args.rrc_scale_max),
                ratio=(0.2, 5),
            ),
            transforms.RandomPerspective(p=0.5),
        ]

        # for spectral signature step
        self.ss_transform = NCropsTransform(
            transforms.Compose(ss_views_aug), self.args.num_views
        )

        print(
            f"Initializing Poison data with random seed {self.args.ssl_pretrain_seed}"
        )

        (
            self.train_pos_loader,
            self.test_clean_loader,
            self.test_pos_loader,
            self.memory_loader,  # new
            self.train_probe_loader,
        ) = self.choose_poisons_randomly()

    def choose_poisons_randomly(self):
        """
        basic data manipulation
        """
        if self.args.dataset == "imagenet100":
            train_paths = self.trainset
            val_paths = self.validset

            print("prepare training data")
            x_train_tensor, y_train_tensor = get_data_and_label(
                train_paths, self.args.image_size
            )
            x_train_tensor = torch.stack(x_train_tensor)
            y_train_tensor = torch.stack(y_train_tensor)

            print("prepare validation data")
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
            # self.memory_loader is the one from set_aug_diff()
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
        # Poisoned Validation Set
        """
        # test set (poison all images) -- support no-poison option
        if self.args.trigger_type == "ftrojan":
            x_test_pos_tensor, y_test_pos_tensor = (
                self.fre_poison_agent.Poison_Frequency_Diff(
                    x_test_tensor.clone().detach(),
                    y_test_tensor.clone().detach(),
                    self.magnitude_val,
                )
            )
        elif self.args.trigger_type == "htba":
            x_test_pos_tensor, y_test_pos_tensor = self.fre_poison_agent.Poison_HTBA(
                x_test_tensor.clone().detach(),
                y_test_tensor.clone().detach(),
            )
        elif self.args.trigger_type == "none":
            # No poisoning: keep validation images unchanged
            x_test_pos_tensor = x_test_tensor.clone().detach()
            y_test_pos_tensor = y_test_tensor.clone().detach()

        # assign correct label to poisoned images, as the Poison_Frequency_Diff() function only poisons image data, but does not pollute label
        y_test_pos_tensor = (
            torch.ones_like(y_test_pos_tensor, dtype=torch.long)
            * self.args.target_class
        )

        """
        # Poisoned Train Set (poison only a portion of train images)
        """
        poison_index = torch.where(y_train_tensor == self.args.target_class)[0]
        poison_index = poison_index[: self.poison_num]

        # If trigger_type is 'none', skip poisoning (empty index)
        if self.args.trigger_type == "none":
            poison_index = torch.tensor([], dtype=torch.long)

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

        """
        Create dataloaders
        """
        train_is_poisoned = torch.zeros_like(y_train_tensor)
        train_is_poisoned[poison_index] = 1

        # image indexing for input-filtering methods
        train_index = torch.tensor(list(range(len(self.trainset))), dtype=torch.long)
        test_index = torch.tensor(list(range(len(self.validset))), dtype=torch.long)
        memory_index = torch.tensor(list(range(len(x_memory_tensor))), dtype=torch.long)

        # contain both CLEAN and a portion of poisoned images
        train_loader = DataLoader(
            (
                TensorDataset(
                    x_train_tensor,
                    train_is_poisoned,
                    y_train_tensor,
                    train_index,
                )
            ),
            batch_size=self.args.pretrain_batch_size,
            sampler=None,
            shuffle=True,
        )

        # clean validation set
        test_clean_loader = DataLoader(
            TensorDataset(x_test_tensor, y_test_tensor, test_index),
            batch_size=self.args.linear_probe_batch_size,
            shuffle=False,
        )

        # poisoned validation set
        test_pos_loader = DataLoader(
            TensorDataset(
                x_test_pos_tensor, y_test_pos_tensor, y_test_tensor, test_index
            ),  # y_test_tensor serves as the original label tensor (for correcting ASR)
            batch_size=self.args.linear_probe_batch_size,
            shuffle=False,
        )

        # memory set is clean
        memory_loader = DataLoader(
            TensorDataset(x_memory_tensor, y_memory_tensor, memory_index),
            batch_size=self.args.linear_probe_batch_size,
            shuffle=False,
        )

        # create 1% train probe (reference) set
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
            memory_loader,  # for kNN classifier
            train_probe_loader,  # for linear classifier and baselines
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
    elif args.dataset == "imagenet100":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        args.mean = mean
        args.std = std
        args.num_classes = 100

    else:
        raise ValueError(args.dataset)

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

    ####################### Define Datasets #######################
    if args.dataset == "cifar10":
        train_dataset = CIFAR10(
            root=args.data_path, train=True, transform=None, download=True
        )
        test_dataset = CIFAR10(
            root=args.data_path, train=False, transform=None, download=True
        )
        memory_dataset = CIFAR10(
            root=args.data_path, train=True, transform=None, download=False
        )

    elif args.dataset == "cifar100":
        train_dataset = CIFAR100(
            root=args.data_path, train=True, transform=None, download=True
        )

        test_dataset = CIFAR100(
            root=args.data_path, train=False, transform=None, download=True
        )
        memory_dataset = CIFAR100(
            root=args.data_path, train=True, transform=None, download=False
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
    else:
        raise NotImplementedError

    # memory loader is clean train set without shuffle, replaced later in the PoisonAgent step
    memory_loader = torch.utils.data.DataLoader(
        memory_dataset,
        512,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return (
        train_dataset,
        test_dataset,
        memory_loader,
        train_transform,
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
