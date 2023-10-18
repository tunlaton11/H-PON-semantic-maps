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
        in_width,
        resolution,
        grid_extents,
        xmin,
        xmax,
        ymin,
        ymax,
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
        self.in_width = in_width
        
        # Compute number of output cells required
        self.out_depth = math.ceil((zmax - zmin) / resolution)
        self.out_width = math.ceil((xmax - xmin) / resolution)  # ***

        # Dense layer which maps UV features to UZ
        self.fc = nn.Conv1d(
            out_channels * self.in_width,
            out_channels * self.out_width,
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


class MultiscaleHDenseTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        resolution,
        grid_extents,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        focal_length,
        groups=1,
    ):
        super().__init__()

        self.focal_length = focal_length

        # Initial convolution to reduce feature dimensions
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.GroupNorm(16, out_channels)

        # Resampler transforms perspective features to BEV
        self.resampler = Resampler(resolution, grid_extents)

        # Compute input height based on region of image covered by grid
        self.zmin, self.zmax = grid_extents[1], grid_extents[3]
        self.xmin, self.xmax = grid_extents[0], grid_extents[2]
        self.in_height = math.ceil(focal_length * (ymax - ymin) / self.zmin)

        self.in_width = math.ceil(resolution / 2 * focal_length * (xmax - xmin) / zmax)

        self.ymid = (ymin + ymax) / 2

        # Compute number of output cells required
        self.out_depth = math.ceil((self.zmax - self.zmin) / resolution)

        # Dense layer which maps UV features to UZ
        self.fc = nn.Conv1d(
            out_channels * self.in_width,
            out_channels * self.out_depth,
            1,
            groups=groups,
        )
        self.out_channels = out_channels

    def forward(self, features, calib, *args):
        # Crop feature maps to a fixed input height
        features = torch.stack(
            [self._crop_feature_map(fmap, cal) for fmap, cal in zip(features, calib)]
        )

        # Reduce feature dimension to minimize memory usage
        features = F.relu(self.bn(self.conv(features)))

        # Bottlenecks
        B, C, H, W = features.shape
        flat_feats = features.permute(0, 1, 3, 2).flatten(1, 2)  # (B, C*W, H)

        # expand
        bev_feats = self.fc(flat_feats).view(B, C, H, -1)

        # Resample to orthographic grid
        return self.resampler(bev_feats, calib)

    def _crop_feature_map(self, fmap, calib):
        # Compute upper and lower bounds of visible region
        focal_length, img_offset = calib[1, 1:]
        vmid = self.ymid * focal_length / self.zmin + img_offset
        vmin = math.floor(vmid - self.in_height / 2)
        vmax = math.floor(vmid + self.in_height / 2)

        # Pad or crop input tensor to match dimensions
        return F.pad(fmap, [0, 0, -vmin, vmax - fmap.shape[-2]])
