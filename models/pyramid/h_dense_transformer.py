import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resampler import Resampler


class HorizontalDenseTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        resolution,
        grid_extents,
        ymin,
        ymax,
        xmin,
        xmax,
        zmin,
        zmax,
        focal_length,
        groups=1,
    ):
        super().__init__()

        # Initial convolution to reduce feature dimensions
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.GroupNorm(16, out_channels)

        # Resampler transforms perspective features to BEV
        self.resampler = Resampler(resolution, [xmin, zmin, xmax, zmax])

        # Compute input width based on region of image covered by grid
        self.zmin, self.zmax = grid_extents[1], grid_extents[3]
        self.xmin, self.xmax = grid_extents[0], grid_extents[2]
        self.in_width = math.ceil(focal_length * (xmax - xmin) / zmax * resolution)

        # Compute number of output cells required
        self.out_depth = math.ceil((zmax - zmin) / resolution)

        # Dense layer which maps UV features to UZ
        self.fc = nn.Conv1d(
            out_channels * self.in_width,
            out_channels * self.out_depth,
            1,
            groups=groups,
        )
        self.out_channels = out_channels

    def forward(self, features, calib, *args):
        # Reduce feature dimension to minimize memory usage
        features = F.relu(self.bn(self.conv(features)))

        # Bottlenecks
        B, C, H, W = features.shape
        flat_feats = features.permute(0, 1, 3, 2).flatten(1, 2)  # (B, C*W, H)

        bev_feats = self.fc(flat_feats).view(B, C, H, -1)

        # Resample to orthographic grid
        return self.resampler(bev_feats, calib)
