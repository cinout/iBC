import torch
import torch.nn as nn
import torch.nn.functional as F


def CrossEntropy(outputs, targets, T=3):
    log_softmax_outputs = F.log_softmax(outputs / T, dim=1)
    softmax_targets = F.softmax(targets / T, dim=1)
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)
    output = kl_loss(log_softmax_outputs, softmax_targets)
    return output


def distillation(
    args, teacher, student, optimizer, scheduler, epoch, trainloader, bcu_aug, device
):
    for content in trainloader:
        (images, _, _) = content
        images = images.to(device)  # tensorized and /255.0ed
        v1 = bcu_aug(images)
        v2 = bcu_aug(images)

        with torch.no_grad():
            teacher_outputs = teacher(v1, v2)

        student_outputs = student(v1, v2)

        # the shape of output. The original code's last layer is fc (512-> num_class)
        if args.method == "simclr":
            teacher_outputs = teacher_outputs.flatten(start_dim=1)  # [B, 2c]
            student_outputs = student_outputs.flatten(start_dim=1)
        elif args.method == "byol":
            teacher_outputs = torch.cat(
                [teacher_outputs[0], teacher_outputs[1]], dim=1
            )  # [B, 2c]
            student_outputs = torch.cat([student_outputs[0], student_outputs[1]], dim=1)
        elif args.method == "mocov2":
            teacher_outputs = teacher_outputs[0]
            student_outputs = student_outputs[0]

        teacher_outputs = teacher_outputs.detach()

        loss = CrossEntropy(
            student_outputs,
            teacher_outputs,
            1 + (3 / args.bcu_epochs) * float(1 + epoch),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch:{} | Loss: {:.3f}".format(epoch, loss.item()))
    scheduler.step()
