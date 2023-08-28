import torch
from .pyramid import PyramidOccupancyNetwork, ExtendedPyramidOccupancyNetwork
from .fpn import FPN50
from .topdown import TopdownNetwork
from .v_transformer_pyramid import VerticalTransformerPyramid
from .h_transformer_pyramid import HorizontalTransformerPyramid
from .classifier import LinearClassifier, BayesianClassifier
from operator import mul
from functools import reduce


def build_pyramid_occupancy_network(config):
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
    # classifier.initialise(config.prior)

    # Assemble Pyramid Occupancy Network
    return PyramidOccupancyNetwork(
        frontend,
        v_transformer,
        topdown,
        classifier,
    )


def build_extended_pyramid_occupancy_network(config):
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

    h_transformer = HorizontalTransformerPyramid(
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
    # classifier.initialise(config.prior)

    # Assemble Pyramid Occupancy Network
    return ExtendedPyramidOccupancyNetwork(
        frontend,
        v_transformer,
        h_transformer,
        topdown,
        classifier,
    )
