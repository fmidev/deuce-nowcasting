"""
Script to fetch the means for BCNN predictions,
averaging epistemic uncertainties and discarding aleatoric ones. 

Arguments: 
    -i : input file path
    
Outputs:
    $({-i}.stem)_combined.hdf5 : 
        - saved in the running folder
        - has the same timestamps and leadtimes as the input
        - has $(-n) x N ensemble member, if $(-i) input has N member.
"""

import argparse
from pathlib import Path

import numpy as np
import h5py
from tqdm import tqdm

import pincast_verif.io_tools as io


def run(input_path: str, output_path: str):
    "inner function for extracting the mean prediction."
    meta = io.get_hdf5_metadata(input_path)

    for timestamp in tqdm(meta["timestamps"]):
        locs = io.get_prediction_data_locs(
            time_0=timestamp,
            leadtimes=meta["leadtimes"],
            method_name=meta["method_name"]
            )
        preds = io._load_hdf5_data_at(input_path, locs) # T x S x H x W x C
        pred_mean = preds[...,0].mean(axis=1)
        io._write_hdf5_data_to(pred_mean, output_path, locs)


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

    args = argparser.parse_args()

    input_path = Path(args.input)
    output_path = input_path.with_name("deuce-mean.hdf5")
    run(input_path=input_path, output_path=output_path)
