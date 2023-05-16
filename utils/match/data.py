"""
Match pl.LightningDataModule datamodule name to Pytorch Module
"""

from datamodules import *

def match_datamodule(data_config : dict):
    if data_config.name == "fmi":
        return FMICompositeDataModule(data_config.kwargs)
    else:
        raise NotImplementedError(f"Datamodule name {data_config.name} undefined.")