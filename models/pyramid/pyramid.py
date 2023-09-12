import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidOccupancyNetwork(nn.Module):
    def __init__(self, frontend, v_transformer, topdown, classifier):
        super().__init__()

        self.frontend = frontend
        self.v_transformer = v_transformer
        self.topdown = topdown
        self.classifier = classifier

    def forward(self, image, calib, *args):
        # Extract multiscale feature maps
        feature_maps = self.frontend(image)

        # Transform image features to birds-eye-view
        v_bev_feats = self.v_transformer(feature_maps, calib)

        # Apply topdown network
        td_feats = self.topdown(v_bev_feats)

        # Predict individual class log-probabilities
        logits = self.classifier(td_feats)
        return logits


class HorizontallyAwarePyramidOccupancyNetwork(nn.Module):
    """Horizontally-aware Pyramid Occupancy Network - the original
    Pyramid Occupancy Network is extended with a new component called
    horizontal transformer pyramid.
    """

    def __init__(self, frontend, v_transformer, h_transformer, topdown, classifier):
        super().__init__()

        self.frontend = frontend
        self.v_transformer = v_transformer
        self.h_transformer = h_transformer
        self.topdown = topdown
        self.classifier = classifier

    def forward(self, image, calib, *args):
        # Extract multiscale feature maps
        feature_maps = self.frontend(image)

        # Transform image features to birds-eye-view
        v_bev_feats = self.v_transformer(feature_maps, calib)
        h_bev_feats = self.h_transformer(feature_maps, calib)
        bev_feats = torch.cat([v_bev_feats, h_bev_feats], dim=1)  # stack both bev feats

        # Apply topdown network
        td_feats = self.topdown(bev_feats)

        # Predict individual class log-probabilities
        logits = self.classifier(td_feats)
        return logits
