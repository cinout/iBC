import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.base import CLModel

device = "cuda" if torch.cuda.is_available() else "cpu"


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class SimCLRModel(CLModel):
    def __init__(self, args):
        super().__init__(args)

        self.args = args

        self.proj_dim = 128  # C
        self.hidden_dim = 2048  # C

        if self.mlp_layers == 2:
            self.proj_head = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim, self.proj_dim),
            )
        elif self.mlp_layers == 3:
            if args.arch.lower().startswith("vit"):
                # DDP for ViT pretraining, make sure BatchNorm1d layers in the MLP projection head
                self.proj_head = nn.Sequential(
                    nn.Linear(self.feat_dim, self.hidden_dim, bias=False),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden_dim, self.proj_dim, bias=False),
                    BatchNorm1dNoBias(self.proj_dim),
                )
            else:
                self.proj_head = nn.Sequential(
                    nn.Linear(self.feat_dim, self.feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.feat_dim, self.feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.feat_dim, self.proj_dim),
                )

    @torch.no_grad()
    def moving_average(self):
        """
        Momentum update of the key encoder
        """
        m = 0.5
        for param_q, param_k in zip(
            self.distill_backbone.parameters(), self.backbone.parameters()
        ):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

    def forward(self, v1, v2):

        x = torch.cat([v1, v2], dim=0)
        x = self.backbone(x)  # includes two views' features
        reps = F.normalize(self.proj_head(x), dim=1)

        bsz = reps.shape[0] // 2
        f1, f2 = torch.split(reps, [bsz, bsz], dim=0)  # each [bs, C]

        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        if self.args.use_adaptive_attack:
            x1, x2 = torch.split(x, [bsz, bsz], dim=0)  # each [bs, C]
            backbone_features = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1)], dim=1)
            return backbone_features, features  # [bs, 2, C], [bs, 2, C]
        else:
            return features  # [bs, 2, C]

    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    
    Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """

    def supConLoss(self, features, labels=None, mask=None):
        # features.shape [bs, 2, C]
        temperature = self.args.temp
        contrast_mode = "all"
        base_temperature = 0.07

        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            # arrive here
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # [bs, bs]
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [bs*2, C]

        if contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif contrast_mode == "all":
            # arrive here
            anchor_feature = contrast_feature  # [bs*2, C]
            anchor_count = contrast_count  # 2
        else:
            raise ValueError("Unknown mode: {}".format(contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask  # [2bs, 2bs]

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(temperature / base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)  # [2, bs]

        standard_simclr_loss = loss.mean()
        return standard_simclr_loss
