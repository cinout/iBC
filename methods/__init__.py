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
            # TODO: remove all tghe prints
            def vit_base_encoder(**kwargs):
                m = base_encoder(**kwargs)
                # alias `fc` to the classification head for compatibility
                if hasattr(m, "heads"):
                    print(">>> arrive here, vit_base_encoder, m.heads: ", m.heads)
                    # torchvision ViT `heads` may be a Sequential(LayerNorm, Linear).
                    # Find the final Linear module inside and use it as `fc` so
                    # MoCo can access `.weight` as expected.
                    heads = m.heads
                    if isinstance(heads, nn.Sequential):
                        print(">>> heads is nn.Sequential, heads: ", heads)
                        # find last Linear in the sequential
                        linear_layer = None
                        for mod in reversed(list(heads)):
                            if isinstance(mod, nn.Linear):
                                print(">>> find linear layer in heads: ", mod)
                                linear_layer = mod
                                break
                        if linear_layer is not None:
                            print(
                                ">>> set m.fc to linear layer in heads: ", linear_layer
                            )
                            m.fc = linear_layer
                        else:
                            print(
                                ">>> no linear layer found in heads, set m.fc to Identity"
                            )
                            m.fc = nn.Identity()
                    elif isinstance(heads, nn.Linear):
                        print(">>> heads is nn.Linear, set m.fc to heads: ", heads)
                        m.fc = heads
                    else:
                        print(
                            ">>> heads is neither nn.Sequential nor nn.Linear, set m.fc to Identity"
                        )
                        m.fc = nn.Identity()
                elif hasattr(m, "classifier"):
                    print(
                        ">>> arrive here, vit_base_encoder, m.classifier: ",
                        m.classifier,
                    )
                    cls = m.classifier
                    if isinstance(cls, nn.Linear):
                        print(
                            ">>> vit_base_encoder, m.classifier is nn.Linear, set m.fc to classifier: ",
                            cls,
                        )
                        m.fc = cls
                    elif isinstance(cls, nn.Sequential):
                        linear_layer = None
                        for mod in reversed(list(cls)):
                            if isinstance(mod, nn.Linear):
                                print(
                                    ">>> vit_base_encoder, find linear layer in classifier: ",
                                    mod,
                                )
                                linear_layer = mod
                                break
                        if linear_layer is not None:
                            print(
                                ">>> vit_base_encoder, found linear layer in classifier: ",
                                linear_layer,
                            )
                            m.fc = linear_layer
                        else:
                            print(
                                ">>> vit_base_encoder, no linear layer found in classifier, set m.fc to Identity"
                            )
                            m.fc = nn.Identity()
                else:
                    print(
                        ">>> arrive here, vit_base_encoder, no heads or classifier found, set m.fc to Identity"
                    )
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
