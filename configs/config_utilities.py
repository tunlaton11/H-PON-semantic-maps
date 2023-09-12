import os
from yacs.config import CfgNode


def load_config(config_path: str) -> CfgNode:
    """Load configurations from a YAML file.

    Parameters
    ----------
    config_path : str
        Path of configuration YAML file

    Returns
    -------
    CfgNode
        Configurations
    """
    with open(config_path) as f:
        return CfgNode.load_cfg(f)
