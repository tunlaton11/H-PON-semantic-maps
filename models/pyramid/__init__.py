import torch
from .pyramid import PyramidOccupancyNetwork, HorizontallyAwarePyramidOccupancyNetwork, HorizontallyAwarePyramidOccupancyNetworkV2
from .fpn import FPN50
from .topdown import TopdownNetwork
from .v_transformer_pyramid import VerticalTransformerPyramid
from .h_transformer_pyramid import (
    HorizontalTransformerPyramid,
    MultiscaleHDenseTransformerPyramid,
)
from .classifier import LinearClassifier, BayesianClassifier
from operator import mul
from functools import reduce
from typing import Literal


def build_pon(config):
    # Build frontend
    frontend = FPN50()

    # Build transformer pyramid
    tfm_resolution = config.map_resolution * reduce(mul, config.topdown.strides)

    v_transformer = VerticalTransformerPyramid(
        256,
        config.tfm_channels,
        tfm_resolution,
        config.map_extents,
        config.ymin,
        config.ymax,
        config.focal_length,
    )

    # Build topdown network
    topdown = TopdownNetwork(
        config.tfm_channels,
        config.topdown.channels,
        config.topdown.layers,
        config.topdown.strides,
        config.topdown.blocktype,
    )

    # Build classifier
    if config.bayesian:
        classifier = BayesianClassifier(topdown.out_channels, config.num_class)
    else:
        classifier = LinearClassifier(topdown.out_channels, config.num_class)
    classifier.initialise(config.prior)

    # Assemble Pyramid Occupancy Network
    return PyramidOccupancyNetwork(
        frontend,
        v_transformer,
        topdown,
        classifier,
    )


def build_hpon(
    config,
    htfm_method: Literal["collage", "stack", "multiscale"],
):
    # Build frontend
    frontend = FPN50()

    # Build transformer pyramid
    tfm_resolution = config.map_resolution * reduce(mul, config.topdown.strides)

    v_transformer = VerticalTransformerPyramid(
        256,
        config.tfm_channels,
        tfm_resolution,
        config.map_extents,
        config.ymin,
        config.ymax,
        config.focal_length,
    )

    if htfm_method == "multiscale":
        h_transformer = MultiscaleHDenseTransformerPyramid(
            256,
            config.tfm_channels,
            tfm_resolution,
            config.map_extents,
            config.ymin,
            config.ymax,
            config.focal_length,
        )
    else:
        h_transformer = HorizontalTransformerPyramid(
            256,
            config.tfm_channels,
            config.htfm_channels,
            tfm_resolution,
            config.map_extents,
            config.ymin,
            config.ymax,
            config.focal_length,
            config.img_size[0],
            htfm_method,
        )

    # Build topdown network
    topdown = TopdownNetwork(
        config.tfm_channels * 2,
        config.topdown.channels,
        config.topdown.layers,
        config.topdown.strides,
        config.topdown.blocktype,
    )

    # Build classifier
    if config.bayesian:
        classifier = BayesianClassifier(topdown.out_channels, config.num_class)
    else:
        classifier = LinearClassifier(topdown.out_channels, config.num_class)
    classifier.initialise(config.prior)

    # Assemble Pyramid Occupancy Network
    return HorizontallyAwarePyramidOccupancyNetwork(
        frontend,
        v_transformer,
        h_transformer,
        topdown,
        classifier,
    )

def build_hponv2(
    config,
    htfm_method: Literal["collage", "stack", "multiscale"],
):
    # Build frontend
    frontend = FPN50()

    # Build transformer pyramid
    tfm_resolution = config.map_resolution * reduce(mul, config.topdown.strides)

    v_transformer = VerticalTransformerPyramid(
        256,
        config.tfm_channels,
        tfm_resolution,
        config.map_extents,
        config.ymin,
        config.ymax,
        config.focal_length,
    )

    if htfm_method == "multiscale":
        h_transformer = MultiscaleHDenseTransformerPyramid(
            256,
            config.tfm_channels,
            tfm_resolution,
            config.map_extents,
            config.ymin,
            config.ymax,
            config.focal_length,
        )
    else:
        h_transformer = HorizontalTransformerPyramid(
            256,
            config.tfm_channels,
            config.htfm_channels,
            tfm_resolution,
            config.map_extents,
            config.ymin,
            config.ymax,
            config.focal_length,
            config.img_size[0],
            htfm_method,
        )

    # Build topdown network
    v_topdown = TopdownNetwork(
        config.tfm_channels * 2,
        config.topdown.channels,
        config.topdown.layers,
        config.topdown.strides,
        config.topdown.blocktype,
    )

    h_topdown = TopdownNetwork(
        config.tfm_channels * 2,
        config.topdown.channels,
        config.topdown.layers,
        config.topdown.strides,
        config.topdown.blocktype,
    )

    # Build classifier
    if config.bayesian:
        classifier = BayesianClassifier(v_topdown.out_channels + h_topdown.out_channels, config.num_class)
    else:
        classifier = LinearClassifier(v_topdown.out_channels + h_topdown.out_channels, config.num_class)
    classifier.initialise(config.prior)

    # Assemble Pyramid Occupancy Network
    return HorizontallyAwarePyramidOccupancyNetworkV2(
        frontend,
        v_transformer,
        h_transformer,
        v_topdown,
        h_topdown,
        classifier,
    )