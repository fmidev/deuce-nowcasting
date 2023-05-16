"""
Save measurements as HDF5
"""
from addict import Dict
from skimage.measure import block_reduce

import numpy as np
import h5py
from pysteps.utils import conversion, dimension

from .. import io_tools
from ..prediction_builder import PredictionBuilder


class MeasurementBuilder(PredictionBuilder):
    def __init__(self, config: Dict):
        super().__init__(config)

    def read_input(self, timestamp: str, num_next_files: int = 0):
        return super().read_input(timestamp, num_next_files)

    def save(self, nowcast: np.ndarray, group: h5py.Group, save_parameters: Dict):
        """Save the nowcast into the hdf5 file,
        in the group "group".

        Args:
            nowcast (np.ndarray): (n_timesteps,h,w) shaped predictions
            group (h5py.Group): parent group (eg.
            timestamp/method) that will contain the saved nowcast
            save_parameters (Dict): parameters regarding
            saving nowcasts to the hdf5 file.
        """
        what_attrs = save_parameters.what_attrs

        io_tools.write_data_to_h5_group(
            group=group, ds_name="data", data=nowcast, what_attrs=what_attrs
        )

    def run(self, timestamp: str):
        return super().run(timestamp)

    def preprocessing(self, data: np.ndarray, metadata: dict, params: Dict = None):
        """
        All the processing of data before nowcasting
        in : data, metadata, params
        out: data, metadata
        """

        if params is None:
            params = Dict(
                {
                    "bbox": [125, 637, 604, 1116],
                    "nan_to_zero": True,
                    "downsampling": False,
                    "threshold": None,
                }
            )

        bbox = params.bbox
        bbox = (
            bbox[0] * metadata["xpixelsize"],
            bbox[1] * metadata["xpixelsize"],
            bbox[2] * metadata["ypixelsize"],
            bbox[3] * metadata["ypixelsize"],
        )

        data = data.squeeze()
        assert data.ndim == 2

        metadata["yorigin"] = "lower"
        data, metadata = dimension.clip_domain(R=data, metadata=metadata, extent=bbox)

        data, metadata = conversion.to_rainrate(data, metadata)

        if params.downsampling:
            # Upsample by averaging
            data = block_reduce(data, func=np.nanmean, cval=0, block_size=(2, 2))

        if params.threshold is not None:
            data[data < float(params.threshold)] = 0

        data, metadata = conversion.to_reflectivity(data, metadata)

        if params.nan_to_zero:
            data[~np.isfinite(data)] = -32
        data[data < -32] = -32

        return data, metadata

    def nowcast(self, data: np.ndarray, params: Dict = None):
        return data

    def postprocessing(self, nowcast, metadata, params: Dict = None):
        return nowcast
