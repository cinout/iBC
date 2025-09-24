import torch.nn as nn
import torch


class TNet(nn.Module):
    """Global statistics network

    Args:
        feature_map_size (int): Size of input feature maps
        feature_map_channels (int): Number of channels in the input feature maps
        latent_dim (int): flattened output feature dims (C*H*W)
    """

    def __init__(
        self, feature_map_size: int, feature_map_channels: int, latent_dim: int
    ):

        super().__init__()
        self.dense1 = nn.Linear(
            in_features=(feature_map_size**2 * feature_map_channels) + latent_dim,
            out_features=512,
        )
        self.dense2 = nn.Linear(in_features=512, out_features=512)
        self.dense3 = nn.Linear(in_features=512, out_features=1)
        self.relu = nn.ReLU()

    """
    Input:
        feature_maps: flattned internal hooks outputs, shape: [bs, dim_A]
        repres: backbone_ouput + torch.flatten(x, start_dim=1) and then F.normalize(feature, dim=-1), shape: [bs, dim_B]
    """

    def forward(
        self,
        feature_map: torch.Tensor,
        representation: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([feature_map, representation], dim=-1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        global_statistics = self.dense3(x)
        return global_statistics
