import copy, random
import torch
import torch.nn as nn
import numpy as np
from networks.resnet_org import model_dict
from networks.resnet_cifar import model_dict as model_dict_cifar

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_backbone(method, model):
    # unwrap DistributedDataParallel if necessary
    if hasattr(model, "module"):
        model_module = model.module
    else:
        model_module = model

    if method == "mocov2":
        backbone = copy.deepcopy(model_module.encoder_q)
        backbone.fc = nn.Sequential()
    else:
        backbone = copy.deepcopy(model_module.backbone)
    return backbone


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


def update_seed(seed):
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (all devices)
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU (if using multi-GPU)
    np.random.seed(seed)
    random.seed(seed)


def get_feat_dim(args):
    # Prefer CIFAR-specific registry when available, otherwise fall back
    # to the general model registry. This allows using backbones like ViT
    # even when training on CIFAR datasets.
    if args.arch in model_dict_cifar:
        _, feat_dim = model_dict_cifar[args.arch]
    elif args.arch in model_dict:
        _, feat_dim = model_dict[args.arch]
    else:
        raise KeyError(f"Unknown architecture '{args.arch}' for feat dim lookup")
    return feat_dim
