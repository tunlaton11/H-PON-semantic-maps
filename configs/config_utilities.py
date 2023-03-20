import os
from yacs.config import CfgNode


def load_config() -> CfgNode:
    """Load configurations from a YAML file.

    Returns
    -------
    CfgNode
        Configurations
    """
    
    config_path = os.path.join(os.getcwd(), "configs\configs.yml")
    with open(config_path) as f:
        return CfgNode.load_cfg(f)
