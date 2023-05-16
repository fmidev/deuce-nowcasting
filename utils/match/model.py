"""
Match pl.LightningModule model name to Pytorch Module
"""

from models import *

def match_model(model_config : dict):
    if model_config.name == "bcnn":
        return SingleScaleBCNN(model_config.kwargs)
    else:
        raise NotImplementedError(f"Model name {model_config.name} undefined.")
