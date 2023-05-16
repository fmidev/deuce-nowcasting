"""
Match loss function name to callable
"""

from costfunctions import *
import torch.nn as nn

def match_loss(loss_config : dict):
    if loss_config.name == "gaussian_nll":
        return GaussianNLL(**loss_config.kwargs)
    else:
        raise NotImplementedError(f"Loss {loss_config.name} not implemented!")