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
        return MoCo(
            models.__dict__[args.arch],
            args,
            dim=512,  # FIXME: 512, read from resnet18
            K=65536,
            m=0.999,
            contr_tau=0.2,
            mlp=True,
        )
    else:
        raise NotImplementedError
