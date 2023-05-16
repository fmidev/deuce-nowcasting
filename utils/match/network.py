"""
Match neural network name to Pytorch Module
"""

from networks import *

def match_network(network_config : dict):
    if network_config.name == "unet":
        return UNet(**network_config.kwargs)
    else:
        raise NotImplementedError(f"Model name {network_config.name} undefined.")