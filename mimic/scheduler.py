import torch
from tqdm import tqdm
import numpy as np
import math
from torch import nn
from mimic.DV import DV
from mimic.HookTool import get_feas_by_hook
from mimic.TNet import TNet
import torch.nn.functional as F
from torchvision import transforms


class weight_scheduler:
    def __init__(self, base_opt, args, momentum_opt, EPOCHS=100):
        # final_opt = base_opt + momentum_opt
        self.EPOCHS = EPOCHS
        self.base_opt = base_opt
        self.momentum_opt = momentum_opt
        self.args = args
        self.transform = transforms.Compose(
            [
                transforms.Normalize(self.args.mean, self.args.std),
            ]
        )  # no augmentation

    @staticmethod
    def test_mi(
        model,
        estimator,
        test_loader,
        device,
        layer,
        transform,
        student_hooks,
        backbone_output_layer,
        method,
    ):
        e_mi = 0

        for content in test_loader:
            (images, _, _) = content
            images = images.to(device)  # tensorized and /255.0ed
            v1 = transform(images)

            estimator.eval()
            model.eval()

            with torch.no_grad():
                proj_outputs = model(v1, v1)
                # features is backbone_ouput
                features = student_hooks[backbone_output_layer].fea_out
                features = F.normalize(features, dim=-1)
                # internal maps @ layer1/2/3/4. flattened
                fea_layers = student_hooks[layer].fea_out

                if method == "simclr":
                    # for backbone features
                    bs = features.shape[0] // 2
                    features, _ = torch.split(features, [bs, bs], dim=0)
                    # for internal layers
                    bs = fea_layers.shape[0] // 2
                    fea_layers, _ = torch.split(fea_layers, [bs, bs], dim=0)

            mi = estimator(fea_layers, features).item()
            mi = 0 if math.isnan(mi) or mi < 0 else mi
            e_mi += mi
        return e_mi / len(test_loader)

    def estimate_mi(self, model, train_loader, device):
        if self.args.method == "mocov2":
            layers = [
                "encoder_q.layer1",
                "encoder_q.layer2",
                "encoder_q.layer3",
                "encoder_q.layer4",
            ]
            backbone_output_layer = "encoder_q.avgpool"
        elif self.args.method in ["byol", "simclr"]:
            layers = [
                "backbone.layer1",
                "backbone.layer2",
                "backbone.layer3",
                "backbone.layer4",
            ]
            backbone_output_layer = "backbone.avgpool"
        else:
            raise Exception(f"not implmented method: {self.args.method}")

        results_mi = {}
        for layer in layers:
            results_mi[layer] = []
        LR = 1e-3

        # initialize the estimators
        Estimators = []  # one for each layer in layers
        opts = []  # one for each layer in layers

        # hook student's backbone output
        student_hooks = get_feas_by_hook(
            model,
            layer_names=layers + [backbone_output_layer],
        )

        for index in range(len(layers)):
            # just to get the feature map size
            model.eval()
            with torch.no_grad():
                inputs_1 = torch.rand(
                    100, 3, self.args.image_size, self.args.image_size
                ).to(device)
                inputs_2 = torch.rand(
                    100, 3, self.args.image_size, self.args.image_size
                ).to(device)

                proj_outputs = model(inputs_1, inputs_2)

                # features is backbone_ouput
                features = student_hooks[backbone_output_layer].fea_out
                features = F.normalize(features, dim=-1)
                # internal maps @ layer1/2/3/4. Retain H W shape
                feature_maps = student_hooks[layers[index]].fea_map

                # beware that simclr concatenates two view's features into one
                if self.args.method == "simclr":
                    # for backbone features
                    bs = features.shape[0] // 2
                    features, _ = torch.split(features, [bs, bs], dim=0)
                    # for internal layers
                    bs = feature_maps.shape[0] // 2
                    feature_maps, _ = torch.split(feature_maps, [bs, bs], dim=0)

            T = TNet(
                feature_map_size=feature_maps.size(2),
                feature_map_channels=feature_maps.size(1),
                latent_dim=features.shape[-1],
            ).to(device)

            MI_estimator = DV(T)

            Estimators.append(MI_estimator)
            opt = torch.optim.Adam(MI_estimator.parameters(), lr=LR, weight_decay=1e-5)
            opts.append(opt)

        for index in range(len(layers)):
            estimator = Estimators[index].to(device)
            opt = opts[index]

            estimator.train()
            train_epochs = tqdm(range(self.EPOCHS))
            print(
                f"------------------------------- MI-Esti-Layer-{layers[index]}-------------------------------"
            )
            for t in train_epochs:
                for batch, content in enumerate(train_loader):
                    (images, _, _) = content
                    images = images.to(device)  # tensorized, and normalized
                    v1 = self.transform(images)

                    with torch.no_grad():
                        proj_outputs = model(v1, v1)

                        # features is backbone_ouput
                        features = student_hooks[backbone_output_layer].fea_out
                        features = F.normalize(features, dim=-1)
                        # internal maps @ layer1/2/3/4. flattened
                        fea_layers = student_hooks[layers[index]].fea_out

                        if self.args.method == "simclr":
                            # for backbone features
                            bs = features.shape[0] // 2
                            features, _ = torch.split(features, [bs, bs], dim=0)
                            # for internal layers
                            bs = fea_layers.shape[0] // 2
                            fea_layers, _ = torch.split(fea_layers, [bs, bs], dim=0)

                    estimator.train()

                    loss = estimator.learning_loss(fea_layers, features)
                    assert not loss.isnan()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                # estimate mi
                e_mi_clean = self.test_mi(
                    model,
                    estimator,
                    train_loader,
                    device,
                    layers[index],
                    self.transform,
                    student_hooks,
                    backbone_output_layer,
                    self.args.method,
                )
                # epoch by epoch
                results_mi[layers[index]].append(e_mi_clean)
                train_epochs.set_description(
                    f"Esti MI[{t}/{self.EPOCHS}]: lower bound of layer-{layers[index]}: "
                    f"clean: {e_mi_clean:.6f}-Epoch-{t}"
                )

        for layer in results_mi:  # for each key in dict
            results_mi[layer] = np.mean(
                results_mi[layer][-10:]
            )  # choose last 10 epochs' mean value as MI for each layer

        # results_mi: still a dict (layer:string-> MI_value:float)
        return results_mi, (
            student_hooks,
            layers,
            backbone_output_layer,
        )

    def update_weight(self, mi):
        weight = np.array(list(mi.values()))
        weight /= weight.sum()
        return self.base_opt + self.momentum_opt * weight
