import copy, random
import torch
import torch.nn as nn
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_backbone(method, model):
    if method == "mocov2":
        backbone = copy.deepcopy(model.encoder_q)
        backbone.fc = nn.Sequential()
    else:
        backbone = copy.deepcopy(model.backbone)
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
