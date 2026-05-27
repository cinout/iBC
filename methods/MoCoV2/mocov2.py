import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class MoCo(nn.Module):
    r"""
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
        self,
        base_encoder,  # resnet18
        args,
        dim=512,
        K=65536,
        m=0.999,
        contr_tau=0.2,
        mlp=True,
    ):
        r"""
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        contr_tau: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.args = args
        self.feat_dim = dim
        self.K = K
        self.m = m
        self.contr_tau = contr_tau

        if contr_tau is not None:
            self.register_buffer("scalar_label", torch.zeros((), dtype=torch.long))
        else:
            self.register_parameter("scalar_label", None)

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(
            num_classes=dim
        )  # has [conv1, bn1, relu, maxpool, layer1-4, avgpool, fc]
        self.encoder_k = base_encoder(num_classes=dim)

        # TODO: remove later
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]  # in_feautere=512
            print(">>> mlp is True, dim_mlp: ", dim_mlp)

            # resnet18's fc is replaced with a MLP, with two linear layers, 512 -> 512 -> 1000
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        r"""
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)

        if ptr + batch_size > self.K:
            self.queue[:, ptr : self.K] = keys.T[:, : self.K - ptr]
            self.queue[:, : ptr + batch_size - self.K] = keys.T[:, self.K - ptr :]
        else:
            self.queue[:, ptr : ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        r"""
        Batch shuffle, for making use of BatchNorm.
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).to(device)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = 0
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        r"""
        Undo batch shuffle.
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = 0
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def loss(self, q, k):

        # lazyily computed & cached!
        def get_q_bdot_k():
            if not hasattr(get_q_bdot_k, "result"):
                get_q_bdot_k.result = (q * k).sum(dim=1)
            assert get_q_bdot_k.result._version == 0
            return get_q_bdot_k.result

        # lazyily computed & cached!
        def get_q_dot_queue():
            if not hasattr(get_q_dot_queue, "result"):
                get_q_dot_queue.result = q @ self.queue.clone().detach()
            assert get_q_dot_queue.result._version == 0
            return get_q_dot_queue.result

        # l_contrastive
        if self.contr_tau is not None:
            # compute logits

            # positive logits: Nx1
            l_pos = get_q_bdot_k().unsqueeze(-1)
            # negative logits: NxK
            l_neg = get_q_dot_queue()

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            # apply temperature
            logits /= self.contr_tau

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        standard_mocov2_loss = F.cross_entropy(
            logits, self.scalar_label.expand(logits.shape[0])
        )

        return standard_mocov2_loss

    def forward(self, im_q, im_k):
        r"""
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # both are flattened, normalized projector (.fc) output
        return q, k


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    return tensor
