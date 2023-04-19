from .pyramid import PyramidOccupancyNetwork
from .fpn import FPN50
from .topdown import TopdownNetwork
from .transformer_pyramid import TransformerPyramid
from .classifier import LinearClassifier, BayesianClassifier
from operator import mul
from functools import reduce


class DefaultTopdownConfig:
    def __init__(self):

        # Number of feature channels at each layer of the topdown network
        self.channels = 128

        # Number of blocks in each layer
        self.layers = [4, 4]

        # Upsampling factor in each stage of the topdown network
        self.strides = [1, 2]

        # Type of residual block to use [ basic | bottleneck ]
        self.blocktype = "bottleneck"


class DefaultConfig:
    def __init__(self):
        
        self.topdown = DefaultTopdownConfig()
        self.map_extents = [-25.0, 1.0, 25.0, 50.0]
        self.map_resolution = 0.25
        self.bayesian = False
        self.ymin = -2
        self.ymax = 4
        self.focal_length = 630
        self.tfm_channels = 64
        self.num_class = 14
        self.prior = [0.44679, 0.02407, 0.14491, 0.02994, 0.02086, 0.00477, 0.00156, 0.00189, 0.00084, 0.00119, 0.00019, 0.00012, 0.00031, 0.00176]


def build_pyramid_occupancy_network(config=DefaultConfig()):

    # Build frontend
    frontend = FPN50()

    # Build transformer pyramid
    tfm_resolution = config.map_resolution * reduce(mul, config.topdown.strides)
    transformer = TransformerPyramid(256, config.tfm_channels, tfm_resolution,
                                     config.map_extents, config.ymin, 
                                     config.ymax, config.focal_length)

    # Build topdown network
    topdown = TopdownNetwork(config.tfm_channels, config.topdown.channels,
                             config.topdown.layers, config.topdown.strides,
                             config.topdown.blocktype)
    
    # Build classifier
    if config.bayesian:
        classifier = BayesianClassifier(topdown.out_channels, config.num_class)
    else:
        classifier = LinearClassifier(topdown.out_channels, config.num_class)
    # classifier.initialise(config.prior)
    
    # Assemble Pyramid Occupancy Network
    return PyramidOccupancyNetwork(frontend, transformer, topdown, classifier)


if __name__=="__main__":
    network = build_pyramid_occupancy_network()
    print(type(network))