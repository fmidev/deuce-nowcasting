"""
Script to combine epistemic, aleatoric uncertainties for 
heteroscedastic BCNN predictions. 

Arguments: 
    -i : input file path
    -n : Number of aleatoric uncertainty draw per ensemble member.
    
Outputs:
    $({-i}.stem)_combined.hdf5 : 
        - saved in the running folder
        - has the same timestamps and leadtimes as the input
        - has $(-n) x N ensemble member, if $(-i) input has N member.
"""

import argparse
from pathlib import Path
import logging

import numpy as np
from tqdm import tqdm
import h5py
from pysteps.noise import get_method

import pincast_verif.io_tools as io


def combine_uncertainties(
        input_path: str,
        output_path: str,
        obs_path: str,
        n_samples: int,
        noise_method: str
        ):
    "inner function for combining aleatoric and epistemic uncertainties"
    meta = io.get_hdf5_metadata(input_path)
    obs_meta = io.get_hdf5_metadata(obs_path)
    filter_method, generator_method = get_method(noise_method)

    for timestamp in tqdm(meta["timestamps"]):
        locs = io.get_prediction_data_locs(
            time_0=timestamp,
            leadtimes=meta["leadtimes"],
            method_name=meta["method_name"]
            )
        with h5py.File(output_path,'r') as f:
            if all([loc in f for loc in locs]):
                continue

        observations = io.load_observations(
            db_path=obs_path,
            time_0=timestamp,
            leadtimes=range(-12,0),
            method_name=obs_meta["method_name"]
        )
        if observations is None:
            logging.warning(f"Reading observations failed at {timestamp}")
            continue
        filter = filter_method(observations)
        del observations
        corr_noise = np.stack(
            [generator_method(filter)for _ in range(n_samples)],
            axis=0
        )
        preds = io._load_hdf5_data_at(input_path, locs) # T x S x H x W x C
        preds[...,1] = (preds[...,1] + 32.0)**2 # make al STDs to variances starting at 0.0. 
        pred_mean = preds[...,0].mean(axis=1)
        pred_var = ((preds[...,1]).mean(axis=1)) + preds[...,0].var(axis=1) #mean of al var + ep var

        pred_combined = np.stack(
                [
                    pred_mean + corr_noise[i] * np.sqrt(pred_var)
                    for i in range(n_samples)
                ],
                axis=1
            )
        pred_combined[pred_combined < -32.0] = -32.0
        pred_combined[pred_combined >= 95.0] = 95.0
        io._write_hdf5_data_to(pred_combined, output_path, locs)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    argparser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Input HDF5 prediction archive path.",
    )
    argparser.add_argument(
        "-o",
        "--observations",
        type=str,
        default=None,
        help="HDF5 observation archive path for noise correlation estimation.",
    )
    argparser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        default=48,
        help="Number of monte carlo samples to draw from the predictive distribution.",
    )
    argparser.add_argument(
        "-m",
        "--noise-method",
        type=str,
        default="nonparameter",
        help="Pysteps correlated noise generation method name.",
    )

    args = argparser.parse_args()

    input_path = Path(args.input)
    obs_path = (Path(args.observations))
    output_path = input_path.with_name("deuce-combined.hdf5")
    combine_uncertainties(
        input_path=input_path,
        output_path=output_path,
        obs_path=obs_path,
        n_samples=args.n_samples,
        noise_method=args.noise_method
    )
