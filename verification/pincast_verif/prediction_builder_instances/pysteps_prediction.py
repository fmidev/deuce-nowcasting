"""
    Class for computing PYSTEPS nowcasts based on advection extrapolation
"""

from addict import Dict
import numpy as np
import h5py
from pysteps import motion, nowcasts
from pysteps.utils import dimension

from pincast_verif.conversion_tools import *
from pincast_verif.prediction_builder import PredictionBuilder


class PystepsPrediction(PredictionBuilder):
    "Easy interface for running PySTEPS predictions with a list of timestamps"

    def __init__(self, config: Dict):
        super().__init__(config)

    def read_input(self, timestamp: str, num_next_files: int = 0):
        return super().read_input(timestamp, num_next_files)

    def save(self, nowcast: np.ndarray, group: h5py.Group, save_parameters: Dict):
        return super().save(nowcast, group, save_parameters)

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
                    "transformation": "db",  # ['db','db_inverse']
                    "conversion": "reflectivity",  # ['reflectivity', 'rainrate']
                    "threshold": 0.1,
                    "zerovalue": 15.0,
                    "nan_to_zero": True,
                    "downscaling": 1.0,
                }
            )

        bbox = params.bbox
        bbox = (
            bbox[0] * metadata["xpixelsize"],
            bbox[1] * metadata["xpixelsize"],
            bbox[2] * metadata["ypixelsize"],
            bbox[3] * metadata["ypixelsize"],
        )
        metadata["yorigin"] = "lower"
        data, metadata = dimension.clip_domain(R=data, metadata=metadata, extent=bbox)

        if params.downscaling != 1.0:
            metadata["xpixelsize"] = metadata["ypixelsize"]
            data, metadata = dimension.aggregate_fields_space(
                data, metadata, metadata["xpixelsize"] * params.downscaling
            )

        unit_conversion_method = get_unit_conversion_method(name=params.conversion)
        transformation_method = get_transformation_method(name=params.transformation)

        data, metadata = unit_conversion_method(R=data, metadata=metadata)
        data, metadata = transformation_method(
            R=data,
            metadata=metadata,
            threshold=params.threshold,
            zerovalue=params.zerovalue,
        )
        data[data < params.threshold] = params.zerovalue
        if params.nan_to_zero:
            data[~np.isfinite(data)] = params.zerovalue

        return data, metadata

    def nowcast(self, data: np.ndarray, params: Dict = None):
        "Advection extrapolation, S-PROG, LINDA feasible"

        if params is None:
            params = Dict(
                {
                    "nowcast_method": "advection",
                    "sample_slice": [None, -1, None],
                    "oflow_slice": [0, -1, 1],
                    "n_leadtimes": 36,
                    "oflow_params": {"oflow_method": "lucaskanade"},
                    "nowcast_params": {},
                }
            )

        oflow_name = params.oflow_method
        oflow_fun = motion.get_method(oflow_name)
        nowcast_name = params.nowcast_method
        nowcast_fun = nowcasts.get_method(nowcast_name)
        sample_slice = slice(*params.sample_slice)
        oflow = oflow_fun(data, **params.oflow_params)

        nowcast = nowcast_fun(
            data[sample_slice, ...].squeeze(),
            oflow,
            params.n_leadtimes,
            **params.nowcast_params
        )
        return nowcast

    def postprocessing(self, nowcast, metadata, params):
        if params is None:
            params = Dict(
                {
                    "threshold": -10,
                    "zerovalue": 0,
                    "nan_to_zero": True,
                    "transformation": "db",  # ['db','db_inverse']
                    "conversion": "reflectivity",  # ['reflectivity', 'rainrate']
                }
            )
        # hard constrain on physically possible predictions.
        # needed for some nowcasting methods that can output negative rain rates.
        if metadata["unit"] == "mm/h":
            nowcast[nowcast < 0.0] = 0.0

        unit_conversion_method = get_unit_conversion_method(name=params.conversion)
        transformation_method = get_transformation_method(name=params.transformation)

        nowcast, metadata = transformation_method(
            R=nowcast,
            metadata=metadata,
            threshold=params.threshold,
            zerovalue=params.zerovalue,
        )
        nowcast, metadata = unit_conversion_method(R=nowcast, metadata=metadata)

        nowcast[nowcast < params.threshold] = params.zerovalue
        if params.nan_to_zero:
            nowcast[~np.isfinite(nowcast)] = params.zerovalue

        if nowcast.ndim == 4:  # S,T,W,H case
            nowcast = nowcast.transpose(1, 0, 2, 3)
            # putting T axis first for saving 1 lt S,W,H preds together

        return nowcast
