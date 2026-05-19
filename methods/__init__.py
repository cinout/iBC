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

        # For Vision Transformer backbones, torchvision returns a model with
        # a `heads` (not `fc`) attribute. MoCo implementation expects an
        # encoder constructor that produces a model with an `fc` attribute.
        # Wrap the torchvision constructor for ViT to alias `fc` -> `heads`.
        base_encoder = models.__dict__[args.arch]
        if "vit_b_16" == args.arch.lower():

            def vit_base_encoder(**kwargs):
                m = base_encoder(**kwargs)
                # alias `fc` to the classification head for compatibility
                if hasattr(m, "heads"):
                    # torchvision ViT `heads` may be a Sequential(LayerNorm, Linear).
                    # Find the final Linear module inside and use it as `fc` so
                    # MoCo can access `.weight` as expected.
                    heads = m.heads
                    if isinstance(heads, nn.Sequential):
                        # find last Linear in the sequential
                        linear_layer = None
                        for mod in reversed(list(heads)):
                            if isinstance(mod, nn.Linear):
                                linear_layer = mod
                                break
                        if linear_layer is not None:
                            m.fc = linear_layer
                        else:
                            m.fc = nn.Identity()
                    elif isinstance(heads, nn.Linear):
                        m.fc = heads
                    else:
                        m.fc = nn.Identity()
                elif hasattr(m, "classifier"):
                    cls = m.classifier
                    if isinstance(cls, nn.Linear):
                        m.fc = cls
                    elif isinstance(cls, nn.Sequential):
                        linear_layer = None
                        for mod in reversed(list(cls)):
                            if isinstance(mod, nn.Linear):
                                linear_layer = mod
                                break
                        if linear_layer is not None:
                            m.fc = linear_layer
                        else:
                            m.fc = nn.Identity()
                else:
                    m.fc = nn.Identity()
                return m

            return MoCo(
                vit_base_encoder,
                args,
                dim=feat_dim,
                K=65536,
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
