import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from .h_dense_transformer import HorizontalDenseTransformer, MultiscaleHDenseTransformer


class HorizontalTransformerPyramid(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        htfm_out_channels,
        resolution,
        extents,
        ymin,
        ymax,
        focal_length,
        img_width,
        method: Literal["collage", "stack"],
    ):
        super().__init__()

        self.h_dense_tfms = nn.ModuleList()

        self.method = method
        if self.method == "stack":
            self.final_conv = nn.Conv2d(htfm_out_channels * 5, out_channels, kernel_size=1)

        for i in range(5):
            # Scaled focal length and extents for each transformer
            scale_factor = pow(2, i + 3)
            focal = focal_length / scale_factor
            xmin = extents[0] / scale_factor
            zmin = extents[1] / scale_factor
            xmax = extents[2] / scale_factor
            zmax = extents[3] / scale_factor

            subset_extents = [xmin, zmin, xmax, zmax]

            self.in_width = math.ceil(img_width/scale_factor)

            # Build transformers
            dense_tfm = HorizontalDenseTransformer(
                in_channels=in_channels,
                out_channels=htfm_out_channels,
                in_width=self.in_width,
                resolution=resolution,
                grid_extents=subset_extents,
                xmin=extents[0],
                xmax=extents[2],
                ymin=ymin,
                ymax=ymax,
                zmin=extents[1],
                zmax=extents[3],
                focal_length=focal,
            )

            self.h_dense_tfms.append(dense_tfm)

    def forward(self, feature_maps, calib):
        h_bev_feats = list()
        for i, fmap in enumerate(feature_maps):
            # Scale calibration matrix to account for downsampling
            scale = 8 * 2**i
            calib_downsamp = calib.clone()
            calib_downsamp[:, :2] = calib[:, :2] / scale

            # Apply orthographic transformation to each feature map separately
            h_bev_feats.append(self.h_dense_tfms[i](fmap, calib_downsamp))

        if self.method == "stack":
            final_h_bev_feat = self.final_conv(torch.cat(h_bev_feats, dim=1))
        elif self.method == "collage":
            final_h_bev_feat = arrange_h_bev_feats(h_bev_feats)

        return final_h_bev_feat


def arrange_h_bev_feats(h_bev_feats):
    W = h_bev_feats[0].shape[3]
    num_layers = len(h_bev_feats)
    slice_width = W // (num_layers * 2)

    left_slices = []
    right_slices = []

    for i, feature in enumerate(h_bev_feats[::-1]):
        # arrange left side
        left_lower_bound, left_uppper_bound = i * slice_width, (i + 1) * slice_width
        left_slice = feature[:, :, :, left_lower_bound:left_uppper_bound]
        left_slices.append(left_slice)

        # arange right side
        if i == num_layers - 1:
            right_lower_bound, right_upper_bound = (i + 1) * slice_width, W - (
                slice_width * i
            )
        else:
            right_lower_bound, right_upper_bound = W - (slice_width * (i + 1)), W - (
                slice_width * i
            )

        right_slice = feature[:, :, :, right_lower_bound:right_upper_bound]
        right_slices.append(right_slice)

    final_h_bev_feats = torch.cat((*left_slices, *right_slices[::-1]), dim=3)

    return final_h_bev_feats



class MultiscaleHDenseTransformerPyramid(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        resolution,
        extents,
        ymin,
        ymax,
        focal_length,
    ):
        super().__init__()
        self.h_dense_tfms = nn.ModuleList()
        for i in range(5):
            # Scaled focal length for each transformer
            focal = focal_length / pow(2, i + 3)

            # Compute grid bounds for each transformer
            zmax = min(math.floor(focal * 2) * resolution, extents[3])
            zmin = math.floor(focal) * resolution if i < 4 else extents[1]
            subset_extents = [extents[0], zmin, extents[2], zmax]

            # Build transformers
            tfm = MultiscaleHDenseTransformer(
                in_channels,
                out_channels,
                resolution,
                subset_extents,
                xmin=extents[0],
                xmax=extents[2],
                ymin=ymin, 
                ymax=ymax,
                zmin=extents[1],
                zmax=extents[3],
                focal_length=focal
            )
            self.h_dense_tfms.append(tfm)

    def forward(self, feature_maps, calib):
        bev_feats = list()
        for i, fmap in enumerate(feature_maps):
            # Scale calibration matrix to account for downsampling
            scale = 8 * 2**i
            calib_downsamp = calib.clone()
            calib_downsamp[:, :2] = calib[:, :2] / scale

            # Resize fmap
            fmap = F.interpolate(
                fmap, size=(fmap.shape[2], self.h_dense_tfms[i].in_width)
            )

            # Apply orthographic transformation to each feature map separately
            bev_feats.append(self.h_dense_tfms[i](fmap, calib_downsamp))

        # Combine birds-eye-view feature maps along the depth axis
        return torch.cat(bev_feats[::-1], dim=-2)
