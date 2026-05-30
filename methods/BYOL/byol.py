import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.base import CLModel


class BYOL(CLModel):
    """
    Build a BYOL model. https://arxiv.org/abs/2006.07733
    """

    def __init__(self, args):
        """
        encoder_q: online network
        encoder_k: target network
        dim: feature dimension (default: 4096)
        pred_dim: hidden dimension of the predictor (default: 256)
        """
        super(BYOL, self).__init__(args)
        self.args = args
        self.backbone_k = self.model_generator()
        self.m = args.byol_m

        self.projector_q = nn.Sequential(
            nn.Linear(self.feat_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 256),
        )

        self.projector_k = nn.Sequential(
            nn.Linear(self.feat_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 256),
        )

        self.predictor = nn.Sequential(
            nn.Linear(256, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 256),
        )

        self.encoder_q = nn.Sequential(self.backbone, self.projector_q)
        self.encoder_k = nn.Sequential(self.backbone_k, self.projector_k)

        # Momentum encoder (encoder_k and projector_k) should not receive gradients
        # and must be excluded from optimizer/DDP reductions. Mark their params
        # as not requiring grad.
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        for p in self.projector_k.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        """

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

        # Explicitly get backbone outputs, then projector, so backbone features
        # are available without re-running the backbone separately.
        feat_q1 = self.backbone(x1)
        proj_q1 = self.projector_q(feat_q1)
        p1 = self.predictor(proj_q1)  # NxC

        feat_q2 = self.backbone(x2)
        proj_q2 = self.projector_q(feat_q2)
        p2 = self.predictor(proj_q2)  # NxC

        # target encoder (momentum) outputs (compute backbone_k then projector_k)
        feat_k2 = self.backbone_k(x2)
        z2 = self.projector_k(feat_k2)

        feat_k1 = self.backbone_k(x1)
        z1 = self.projector_k(feat_k1)

        if self.args.use_adaptive_loss:
            return (feat_q1, feat_q2, feat_k1, feat_k2), (p1, p2, z1, z2)
        else:

            return p1, p2, z1, z2

    def negcos(self, p1, p2, z1, z2, mean=True):

        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        if mean:
            standard_byol_loss = -0.5 * (
                F.cosine_similarity(p1, z2.detach(), dim=-1).mean()
                + F.cosine_similarity(p2, z1.detach(), dim=-1).mean()
            )

            return standard_byol_loss
        else:
            # NOT USED
            return -0.5 * (
                F.cosine_similarity(p1, z2.detach(), dim=-1)
                + F.cosine_similarity(p2, z1.detach(), dim=-1)
            )
