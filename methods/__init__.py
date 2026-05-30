from utils.util import get_feat_dim
from .SimCLR.simclr import SimCLRModel
from .BYOL.byol import BYOL
from .MoCoV2.mocov2 import MoCo
import torchvision.models as models
import torch.nn as nn


def set_model(args):
    if args.method == "simclr":
        return SimCLRModel(args)
    elif args.method == "byol":
        return BYOL(args)
    elif args.method == "mocov2":
        feat_dim = get_feat_dim(args)

        base_encoder = models.__dict__[args.arch]
        if "vit_b_16" == args.arch.lower():

            def vit_base_encoder(**kwargs):
                m = base_encoder(**kwargs)
                m.heads = nn.Identity()
                return m

            return MoCo(
                vit_base_encoder,
                args,
                dim=feat_dim,
                K=8192 if args.arch.lower() == "vit_b_16" else 65536,
                m=0.999,
                contr_tau=0.2,
                mlp=True,
            )
        else:
            return MoCo(
                base_encoder,
                args,
                dim=feat_dim,
                K=65536,
                m=0.999,
                contr_tau=0.2,
                mlp=True,
            )
    else:
        raise NotImplementedError
