"""
Uncertainty composition metric calculation script
for Aleatoric/Epistemic uncertainty modeling nowcasts,
with aleatoric uncertainty estimates saved as a separate channel. 

- Takes as arguments:
    1. prediction HDF5 database path
    2. observation HDF5 database path
    3. result directory path

- A few constants related to the data are defined globally. 
    - TIMESTEP [int/min]
    - Z_BIN_EDGES [float/dbz]
    - T,S,W,H,C dimension numbers [int]
    - DBZ_ZEROVALUE [float/dbz]

- Statistics are averaged over in-image pixels and ensemble members, but are saved for each:
    1. Prediction lead time
    2. Ground truth radar reflectivity, binned according to Z_BIN_EDGES. 
    3. timestamp of the prediction HDF5 data

- The statistics calculated are:
    - STD of the prediction error
    - Mean aleatoric STD
    - Mean Epistemic STD

The script saves each statistic in a separate netcdf file, corresponding to a xarray dataset, with arrays for each timestamp.

Bent Harnist (FMI) 2023
"""
import argparse
import os

import numpy as np
import xarray as xr
from scipy.stats import binned_statistic_2d
from tqdm import tqdm

from pincast_verif import io_tools as io

TIMESTEP = 5
Z_BIN_EDGES = np.arange(5, 65, 5)
T, S, W, H, C = 0, 1, 2, 3, 4
DBZ_ZEROVALUE = -32.0


def run(**kwargs):
    prediction_path = kwargs.get("prediction")
    observation_path = kwargs.get("observation")
    result_path = kwargs.get("result_path")

    observation_metadata = io.get_hdf5_metadata(observation_path)
    prediction_metadata = io.get_hdf5_metadata(prediction_path)

    try:
        os.makedirs(result_path)
        me_std = xr.Dataset(data_vars=None)
        al_std = xr.Dataset(data_vars=None)
        ep_std = xr.Dataset(data_vars=None)
    except FileExistsError:
        me_std = xr.open_dataset(os.path.join(result_path, "me_std.nc"))
        al_std = xr.open_dataset(os.path.join(result_path, "al_std.nc"))
        ep_std = xr.open_dataset(os.path.join(result_path, "ep_std.nc"))

    remaining_timestamps = list(
        set(prediction_metadata["timestamps"]) - set(ep_std.keys())
    )

    for ts in tqdm(remaining_timestamps):
        prediction = io.load_data(
            db_path=prediction_path,
            data_location_getter=io.get_prediction_data_locs,
            time_0=ts,
            leadtimes=prediction_metadata["leadtimes"],
            method_name=prediction_metadata["method_name"],
        )

        observation = io.load_data(
            db_path=observation_path,
            data_location_getter=io.get_observation_data_locs,
            time_0=ts,
            leadtimes=prediction_metadata["leadtimes"],
            method_name=observation_metadata["method_name"],
        )

        # coords
        radar_coords = {
            "t": TIMESTEP * np.arange(1, prediction.shape[T] + 1),
            "w": np.arange(prediction.shape[W]),
            "h": np.arange(prediction.shape[H]),
        }
        pred_coords = {
            **radar_coords,
            "s": np.arange(prediction.shape[S]),
            "c": np.arange(2),
        }

        # np -> xr
        x_obs = xr.DataArray(observation, dims=["t", "w", "h"], coords=radar_coords)
        prediction[..., 1] = prediction[..., 1] - DBZ_ZEROVALUE
        x_pred = xr.DataArray(
            prediction, dims=["t", "s", "w", "h", "c"], coords=pred_coords
        )

        # quantities of interest average over ensemble
        me_arr = (x_obs - x_pred[..., 0]).mean(dim="s")
        al_std_arr = x_pred[..., 1].mean(dim="s")
        ep_std_arr = x_pred[..., 0].std(dim="s")

        me_bin, _, _, _ = binned_statistic_2d(
            x=x_obs.values.ravel(),
            y=x_obs.t.broadcast_like(x_obs).values.ravel(),
            values=me_arr.values.ravel(),
            bins=[Z_BIN_EDGES, radar_coords["t"]],
            statistic="std",
        )
        al_bin, _, _, _ = binned_statistic_2d(
            x=x_obs.values.ravel(),
            y=x_obs.t.broadcast_like(x_obs).values.ravel(),
            values=al_std_arr.values.ravel(),
            bins=[Z_BIN_EDGES, radar_coords["t"]],
            statistic="mean",
        )
        ep_bin, _, _, _ = binned_statistic_2d(
            x=x_obs.values.ravel(),
            y=x_obs.t.broadcast_like(x_obs).values.ravel(),
            values=ep_std_arr.values.ravel(),
            bins=[Z_BIN_EDGES, radar_coords["t"]],
            statistic="mean",
        )

        me_std = me_std.assign(
            {
                ts: xr.DataArray(
                    data=me_bin,
                    dims=["z", "t"],
                    coords={
                        "z": bin_center(Z_BIN_EDGES),
                        "t": bin_center(x_obs.t.values),
                    },
                    attrs={"z_bin_edges": Z_BIN_EDGES, "t_bin_edges": x_obs.t.values},
                )
            }
        )
        al_std = al_std.assign(
            {
                ts: xr.DataArray(
                    data=al_bin,
                    dims=["z", "t"],
                    coords={
                        "z": bin_center(Z_BIN_EDGES),
                        "t": bin_center(x_obs.t.values),
                    },
                    attrs={"z_bin_edges": Z_BIN_EDGES, "t_bin_edges": x_obs.t.values},
                )
            }
        )
        ep_std = ep_std.assign(
            {
                ts: xr.DataArray(
                    data=ep_bin,
                    dims=["z", "t"],
                    coords={
                        "z": bin_center(Z_BIN_EDGES),
                        "t": bin_center(x_obs.t.values),
                    },
                    attrs={"z_bin_edges": Z_BIN_EDGES, "t_bin_edges": x_obs.t.values},
                )
            }
        )
        me_std.to_netcdf(os.path.join(result_path, "me_std.nc"))
        al_std.to_netcdf(os.path.join(result_path, "al_std.nc"))
        ep_std.to_netcdf(os.path.join(result_path, "ep_std.nc"))


def bin_center(bin: np.ndarray):
    return (bin[:-1] + bin[1:]) / 2


def bin_limits(bin: np.ndarray):
    return np.vstack([bin[:-1], bin[1:]]).T


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "-p",
        "--prediction",
        type=str,
        help="Input HDF5 prediction archive path.",
    )
    argparser.add_argument(
        "-o",
        "--observation",
        type=str,
        help="Input HDF5 observation archive path.",
    )
    argparser.add_argument(
        "-r",
        "--result-path",
        type=str,
        help="Number of aleatoric uncertainty draws for ensemble members.",
    )
    args = argparser.parse_args()

    run(
        prediction=args.prediction,
        observation=args.observation,
        result_path=args.result_path,
    )
