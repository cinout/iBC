from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
from mimic.HookTool import get_feas_by_hook

device = "cuda" if torch.cuda.is_available() else "cpu"


"""
AT with sum of absolute values with power p
code from: https://github.com/AberHu/Knowledge-Distillation-Zoo
"""


class AT(nn.Module):
    """
    Paying More Attention to Attention: Improving the Performance of Convolutional
    Neural Netkworks wia Attention Transfer
    https://arxiv.org/pdf/1612.03928.pdf
    """

    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    """
    Inputs:
      fm_s:
      fm_t:
    """

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)
        return am


"""
Input:
    data_loader: their train_loader, which is paired augmented, tensorized, and normalized inputs
"""


def mimic_model_train(
    snet,
    tnet,
    data_loader,
    train_optimizer,
    epoch,
    args,
    train_transform,
    student_hook_info,
):
    student_hooks, layers, backbone_output_layer = student_hook_info
    teacher_hooks = get_feas_by_hook(
        tnet,
        layer_names=layers + [backbone_output_layer],
    )

    criterionAT = AT(2)
    snet.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    for content in train_bar:
        (images, _, _) = content
        images = images.to(device)  # tensorized, and normalized
        v1 = train_transform(images)
        v2 = train_transform(images)

        # feature_1 & feature_3: flattneed and normalized BACKBONE output from v1, using student & teacher,
        ## out_1 & out_2: flattened and normalized projector output from student, v1 and v2 separately
        # feature_1, out_1 = snet(v1)
        # _, out_2 = snet(v2)
        # feature_3, _ = tnet(v1)

        student_proj_out = snet(v1, v2)
        teacher_proj_out = tnet(v1, v2)

        feature_1 = student_hooks[backbone_output_layer].fea_out
        feature_3 = teacher_hooks[backbone_output_layer].fea_out
        feature_1 = F.normalize(feature_1, dim=-1)
        feature_3 = F.normalize(feature_3, dim=-1)

        # simclr/moco: use v1;
        # byol: use v2, as the hook naturally overwrites v1
        if args.method == "simclr":
            bs = feature_1.shape[0] // 2
            feature_1, _ = torch.split(feature_1, [bs, bs], dim=0)
            bs = feature_3.shape[0] // 2
            feature_3, _ = torch.split(feature_3, [bs, bs], dim=0)

        # SSL loss, for student only
        if args.method == "simclr":
            conloss = snet.supConLoss(student_proj_out)
        elif args.method == "byol":
            conloss = snet.negcos(*student_proj_out)
        elif args.method == "mocov2":
            conloss = snet.loss(*student_proj_out)

        # out = torch.cat([out_1, out_2], dim=0)  # [2*B, D]
        # # [2*B, 2*B]
        # sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.knn_t)
        # mask = (
        #     torch.ones_like(sim_matrix)
        #     - torch.eye(2 * args.mimic_batch_size, device=sim_matrix.device)
        # ).bool()
        # # [2*B, 2*B-1]
        # sim_matrix = sim_matrix.masked_select(mask).view(2 * args.mimic_batch_size, -1)

        # # SSL loss
        # pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.knn_t)
        # # [2*B]
        # pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        # conloss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        # Distillation loss
        cloneloss = -torch.sum(feature_3 * feature_1, dim=-1).mean()

        # Attention Loss
        student_layer1_featmap = student_hooks[layers[0]].fea_map
        student_layer2_featmap = student_hooks[layers[1]].fea_map
        student_layer3_featmap = student_hooks[layers[2]].fea_map
        student_layer4_featmap = student_hooks[layers[3]].fea_map
        teacher_layer1_featmap = teacher_hooks[layers[0]].fea_map
        teacher_layer2_featmap = teacher_hooks[layers[1]].fea_map
        teacher_layer3_featmap = teacher_hooks[layers[2]].fea_map
        teacher_layer4_featmap = teacher_hooks[layers[3]].fea_map

        at4_loss = (
            criterionAT(student_layer4_featmap, teacher_layer4_featmap.detach())
            * args.opt4
        )
        at3_loss = (
            criterionAT(student_layer3_featmap, teacher_layer3_featmap.detach())
            * args.opt3
        )
        at2_loss = (
            criterionAT(student_layer2_featmap, teacher_layer2_featmap.detach())
            * args.opt2
        )
        at1_loss = (
            criterionAT(student_layer1_featmap, teacher_layer1_featmap.detach())
            * args.opt1
        )

        # tnet_train_list = []
        # for name, module in tnet.f.f._modules.items():
        #     if name == "0":
        #         # conv1, store its output
        #         tnet_train_list.append(module(v1))
        #     else:
        #         # recursively store each module's output
        #         tnet_train_list.append(module(tnet_train_list[int(name) - 1]))
        # # for the last module, normalize it
        # tnet_train_list[-1] = F.normalize(tnet_train_list[-1], dim=-1)

        # snet_train_list = []
        # for name, module in snet.f.f._modules.items():
        #     if name == "0":
        #         snet_train_list.append(module(v1))
        #     else:
        #         snet_train_list.append(module(snet_train_list[int(name) - 1]))
        # snet_train_list[-1] = F.normalize(snet_train_list[-1], dim=-1)

        # at4_loss = (
        #     criterionAT(snet_train_list[6], tnet_train_list[6].detach()) * args.opt4
        # )
        # at3_loss = (
        #     criterionAT(snet_train_list[5], tnet_train_list[5].detach()) * args.opt3
        # )
        # at2_loss = (
        #     criterionAT(snet_train_list[4], tnet_train_list[4].detach()) * args.opt2
        # )
        # at1_loss = (
        #     criterionAT(snet_train_list[3], tnet_train_list[3].detach()) * args.opt1
        # )

        # SUM of Losses
        loss = (
            conloss + at1_loss + at2_loss + at3_loss + at4_loss + cloneloss * args.opt5
        )

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description(
            "Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}".format(
                epoch,
                args.mimic_epochs,
                train_optimizer.param_groups[0]["lr"],
                total_loss / total_num,
            )
        )

    return total_loss / total_num
