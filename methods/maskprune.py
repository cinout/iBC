import torch
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict
import torch.nn as nn
import torchvision.transforms as T

device = "cuda" if torch.cuda.is_available() else "cpu"


# clip value to be witihin 0 and 1
def clip_mask(unlearned_model, lower=0.0, upper=1.0):
    params = [
        param
        for name, param in unlearned_model.named_parameters()
        if "neuron_mask" in name
    ]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def refill_unlearned_model(net, orig_state_dict):
    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = "{}.{}".format(neuron[0], "weight")
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)


# for evaluating performances at different stages
def test_maskprune(args, model, linear, criterion, data_loader, val_mode):
    transform = T.Compose(
        [
            T.Normalize(args.mean, args.std),
        ]
    )

    model.eval()
    linear.eval()

    total_correct = 0
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for content in data_loader:
            if val_mode == "poison":
                (images, labels, original_label, _) = content
                original_label = original_label.to(device)
            elif val_mode == "clean":
                (images, labels, _) = content
            else:
                raise Exception(f"unimplemented val_mode {val_mode}")

            images = images.to(device)
            images = transform(images)
            labels = labels.to(device)
            if val_mode == "poison":
                valid_indices = original_label != args.target_class
                if torch.all(~valid_indices):
                    # all inputs are from target class, skip this iteration
                    continue

                images = images[valid_indices]
                labels = labels[valid_indices]

            output = model(images)
            output = linear(output)

            total_loss += criterion(output, labels).item()

            _, pred = output.topk(
                1, 1, True, True
            )  # k=1, dim=1, largest, sorted; pred is the indices of largest class
            # pred.shape: [bs, k=1]
            pred = pred.squeeze(1)  # shape: [bs, ]
            total_count += labels.shape[0]
            total_correct += (pred == labels).float().sum().item()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / total_count
    return loss, acc


# called at 3rd pruning stage
def evaluate_by_threshold(
    args,
    model,  # backbone
    linear,
    mask_values,  # sorted by [2], from low to high.
    pruning_max,  # 0.9
    pruning_step,  # 0.05
    criterion,
    clean_loader,
    poison_loader,
    memory_loader,
    knn_evaluator,
):

    model.eval()
    linear.eval()

    thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
    start = 0  # prune from which idx in mask_values

    for threshold in thresholds:
        idx = start
        for idx in range(start, len(mask_values)):
            if float(mask_values[idx][2]) <= threshold:
                # pruning action here
                pruning(model, mask_values[idx])
                start += 1
            else:
                break
        layer_name, neuron_idx, value = (
            mask_values[idx][0],
            mask_values[idx][1],
            mask_values[idx][2],
        )

        # kNN performance
        clean_acc, back_acc = knn_evaluator(
            model,
            memory_loader,
            clean_loader,
            args,
            classes=args.num_classes,
            backdoor_loader=poison_loader,
        )

        # linear performance
        cl_loss, cl_acc = test_maskprune(
            args=args,
            model=model,
            linear=linear,
            criterion=criterion,
            data_loader=clean_loader,
            val_mode="clean",
        )
        po_loss, po_acc = test_maskprune(
            args=args,
            model=model,
            linear=linear,
            criterion=criterion,
            data_loader=poison_loader,
            val_mode="poison",
        )
        print(
            "{} \t {} \t {} \t {:.2f} \t {:.1f} \t {:.1f} \t {:.1f} \t {:.1f}".format(
                start,  # prune from which idx in mask_values
                layer_name,  # idx's layer name
                neuron_idx,  # idx's indice inn the layer
                threshold,
                # knn acc
                clean_acc,
                # knn asr
                back_acc,
                # linear acc
                cl_acc * 100,
                # linear asr
                po_acc * 100,
            )
        )


# called at 3rd stage to read mask (use mask_values.txt as reference)
def read_data(file_name):
    tempt = pd.read_csv(file_name, sep="\s+", skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values


# called at the end of 2nd stage
def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if "neuron_mask" in name:
            for idx in range(param.size(0)):
                neuron_name = ".".join(name.split(".")[:-1])
                mask_values.append(
                    "{} \t {} \t {} \t {:.4f} \n".format(
                        count, neuron_name, idx, param[idx].item()
                    )
                )
                count += 1
    with open(file_name, "w") as f:
        f.write("No \t Layer Name \t Neuron Idx \t Mask Score \n")
        f.writelines(mask_values)


def train_step_recovering(
    args, unlearned_model, linear, criterion, mask_opt, data_loader
):
    unlearned_model.train()
    linear.train()

    for content in data_loader:

        images, labels, _ = content

        images, labels = images.to(device), labels.to(device)

        mask_opt.zero_grad()
        output = unlearned_model(images)
        output = linear(output)

        loss = criterion(output, labels)
        loss = args.alpha * loss

        loss.backward()
        mask_opt.step()
        clip_mask(unlearned_model)


def train_step_unlearning(args, model, linear, criterion, optimizer, data_loader):
    model.train()
    linear.train()
    total_correct = 0
    total_count = 0
    for content in data_loader:
        images, labels, _ = content

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        output = linear(output)

        loss = criterion(output, labels)

        _, pred = output.topk(
            1, 1, True, True
        )  # k=1, dim=1, largest, sorted; pred is the indices of largest class
        # pred.shape: [bs, k=1]
        pred = pred.squeeze(1)  # shape: [bs, ]

        total_correct += (pred == labels).float().sum().item()
        total_count += labels.shape[0]

        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(linear.parameters()),
            max_norm=20,
            norm_type=2,
        )
        (-loss).backward()
        optimizer.step()

    acc = float(total_correct) / total_count
    return acc
