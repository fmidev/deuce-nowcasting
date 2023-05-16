"""
Match prediction writer name to appropriate implementation
"""

from utils.prediction_writers import *

def match_prediction_writer(writer_config : dict):
    if writer_config.name == "bcnn_hetero_hdf5_writer":
        return BCNNHeteroHDF5Writer(**writer_config.kwargs)
    else:
        raise NotImplementedError(f"Model name {writer_config.name} undefined.")