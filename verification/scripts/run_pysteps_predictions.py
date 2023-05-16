"""
    This script will run nowcasting predictions
    for advection based deterministic methods implemented in pysteps, with multiple different configurations
    
    Working (tested) prediction types: 
    - extrapolation
    - S-PROG
    - LINDA
    - ANVIL
    - STEPS

    Usage requires: 
    1) Having a workable pysteps installation with
    .pystepsrc configured. 
    2) (Optionally) modifying your AdvectionPrediction class to
    satisfy requirements.
    3) Setting configuration files for each prediction experiment
    to be run, putting them in the folder passed as an argument
"""
import argparse
import sys
import os
from typing import Sequence
from pathlib import Path

import h5py
from tqdm import tqdm

from pincast_verif.prediction_builder_instances import PystepsPrediction
from pincast_verif import io_tools

# temp for except handling
import dask


def run(builders: Sequence[PystepsPrediction]) -> None:
    date_paths = [builder.date_path for builder in builders]
    if any(path != date_paths[0] for path in date_paths):
        raise ValueError(
            "The datelists used must be the same for all runs,\
                        Please check that the paths given match."
        )

    print(date_paths)
    timesteps = io_tools.read_timestamp_txt_file(date_paths[0])
    output_dbs = [builder.hdf5_path for builder in builders]

    for t in tqdm(timesteps):
        for i, builder in enumerate(builders):
            print(f"sample {t} ongoing...")
            group_name = builder.save_params.group_format.format(
                timestamp=io_tools.get_neighboring_timestamps(
                    time=t, distance=builder.input_params.num_next_files
                ),
                method=builder.save_params.method_name,
            )
            with h5py.File(output_dbs[i], "a") as f:
                group = f.require_group(group_name)
                if len(group.keys()) == builder.nowcast_params.n_leadtimes:
                    continue
            print(
                f"Running predictions for {builder.nowcast_params.nowcast_method} method."
            )
            sys.stdout = open(os.devnull, "w")
            # try:
            nowcast = builder.run(t)
            # except:  # indexError, or other error (mainly with LINDA and/or dask)
            #    sys.stdout = sys.__stdout__
            #    continue
            sys.stdout = sys.__stdout__
            with h5py.File(output_dbs[i], "a") as f:
                group = f.require_group(group_name)
                builder.save(
                    nowcast=nowcast, group=group, save_parameters=builder.save_params
                )


if __name__ == "__main__":
    # Handle command line argument
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "yaml_config_path",
        type=str,
        help="Path to either (1) a configuration folder containing \
            one YAML configuration file per nowcast that is to be computed \
            OR (2) a single YAML configuration file for the nowcast that is to be computed",
    )
    args = argparser.parse_args()

    # loading YAML configuration(s) into dict
    config_path = Path(args.yaml_config_path)
    if config_path.is_dir():
        config_filenames = config_path.glob("*.yaml")
        configurations = [
            io_tools.load_yaml_config(filename) for filename in config_filenames
        ]
    else:
        configurations = [io_tools.load_yaml_config(config_path)]

    # Initialize and run
    predictor_builders = [PystepsPrediction(config=config) for config in configurations]
    run(builders=predictor_builders)
