from utils.util import get_feat_dim
from .SimCLR.simclr import SimCLRModel
from .BYOL.byol import BYOL
from .MoCoV2.mocov2 import MoCo
import torchvision.models as models


def set_model(args):
    if args.method == "simclr":
        return SimCLRModel(args)
    elif args.method == "byol":
        return BYOL(args)
    elif args.method == "mocov2":
        feat_dim = get_feat_dim(args)

        return MoCo(
            models.__dict__[args.arch],
            args,
            dim=feat_dim,
            K=65536,
            m=0.999,
            contr_tau=0.2,
            mlp=True,
        )
    else:
        raise NotImplementedError
