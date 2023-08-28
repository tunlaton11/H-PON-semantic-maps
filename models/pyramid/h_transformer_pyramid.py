import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .h_dense_transformer import HorizontalDenseTransformer


class HorizontalTransformerPyramid(nn.Module):
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
            # Scaled focal length and extents for each transformer
            focal = focal_length / pow(2, i + 3)
            xmin = extents[0] / pow(2, i + 3)
            zmin = extents[1] / pow(2, i + 3)
            xmax = extents[2] / pow(2, i + 3)
            zmax = extents[3] / pow(2, i + 3)

            subset_extents = [xmin, zmin, xmax, zmax]

            # Build transformers
            dense_tfm = HorizontalDenseTransformer(
                in_channels=in_channels,
                out_channels=out_channels,
                resolution=resolution,
                grid_extents=subset_extents,
                ymin=ymin,
                ymax=ymax,
                xmin=extents[0],
                xmax=extents[2],
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

            # B, C, H, W = fmap.shape
            # in_width = self.h_dense_tfms[i].in_width
            # fmap = fmap.detach().resize_(B, C, H, in_width)

            # Resize fmap
            fmap = F.interpolate(
                fmap, size=(fmap.shape[2], self.h_dense_tfms[i].in_width)
            )

            # Apply orthographic transformation to each feature map separately
            h_bev_feats.append(self.h_dense_tfms[i](fmap, calib_downsamp))

        # Combine birds-eye-view feature maps along the width axis
        return arrange_h_bev_feats(h_bev_feats)


def arrange_h_bev_feats(h_bev_feats):
    W = h_bev_feats[0].shape[3]
    num_layers = len(h_bev_feats)
    slice_width = W // (num_layers * 2)

    left_slices = []
    right_slices = []

    for i, feature in enumerate(h_bev_feats):
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
