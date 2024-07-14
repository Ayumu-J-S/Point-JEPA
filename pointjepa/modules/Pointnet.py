import torch
from torch import nn

class PointNetSmall(nn.Module):
    def __init__(self, channels: int, feature_dim: int):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(channels, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, feature_dim, 1),
        )

    def forward(self, points) -> torch.Tensor:
        # points: (B, N, C)
        feature = self.first_conv(points.transpose(2, 1))  # (B, 256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True).values  # (B, 256, 1)
        # concating global features to each point features
        feature = torch.cat(
            [feature_global.expand(-1, -1, feature.shape[2]), feature], dim=1
        )  # (B, 512, N)
        feature = self.second_conv(feature)  # (B, feature_dim, N)
        feature_global = torch.max(feature, dim=2).values  # (B, feature_dim)
        return feature_global


class PointNetLarge(nn.Module):
    """
    Larger PointNet encoder taken from PointGPT.
    """

    def __init__(self, channels: int, feature_dim: int):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(channels, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(2048, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, feature_dim, 1),
        )

    def forward(self, points):
        """
        point_groups : B, N, C
        -----------------
        feature_global : B G C
        """
        # encoder
        feature = self.first_conv(points.transpose(2, 1))  # (B, 256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat(
            [feature_global.expand(-1, -1, feature.shape[2]), feature], dim=1
        )
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global


class PointNetEncoder(nn.Module):
    """
    Class takes output dim and small/large as input and returns the corresponding PointNet encoder
    """

    def __init__(self, channels: int, feature_dim: int, name: str):
        super().__init__()
        if name == "small":
            self.encoder = PointNetSmall(channels, feature_dim)
        elif name == "large":
            self.encoder = PointNetLarge(channels, feature_dim)
        else:
            raise ValueError(f"Unknown PointNet encoder name {name}")

    def forward(self, points):
        return self.encoder(points)
