"""
Bent Harnist 2022 (FMI)

Version 0.2

Script for calculating prediction skill metrics for wanted methods / models
and saving them to npy files for fast access. Predictions and measurements are read
from an hdf5 file and advancement of calculations is saved
in the "done" csv file, enabling us to continue calculations if they are once stopped
without having to redo them all.
"""

import os
import argparse
import logging
from datetime import datetime

from addict import Dict
from tqdm import tqdm
from dask.diagnostics import ProgressBar

from pincast_verif.metrics_calculator import MetricsCalculator
from pincast_verif.io_tools import load_yaml_config


def run(config: Dict, config_path: str):
    metrics_calculator = MetricsCalculator(config=config, config_path=config_path)

    if config.debugging:
        import random

        random.seed(12345)
        samples_left = random.sample(
            population=metrics_calculator.timestamps, k=config.debugging
        )
    else:
        samples_left = metrics_calculator.samples_left

    # Main Loop
    if config.accumulate:
        # No parallelization
        if config.n_chunks == 0:
            for sample in tqdm(samples_left):
                done_df, metrics_df = metrics_calculator.accumulate(
                    sample=sample,
                    done_df=metrics_calculator.done_df,
                    metrics_df=metrics_calculator.metrics_df,
                )
                metrics_calculator.update_state(done_df=done_df, metrics_df=metrics_df)
                metrics_calculator.save_metric_states()
                metrics_calculator.save_done_df()
        # parallelized code
        # does not allow for state saving in-between samples or chunks
        else:
            pbar = ProgressBar()
            pbar.register()
            metrics_calculator.parallel_accumulation()
            metrics_calculator.save_metric_states()
            metrics_calculator.save_done_df()

    metrics_data = metrics_calculator.compute()
    metrics_calculator.save_metrics(computed_metrics_df=metrics_data)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("config_path", type=str, help="Configuration file path")
    args = argparser.parse_args()

    config = load_yaml_config(args.config_path)

    if config["accumulate"]:
        try:
            os.makedirs(config.path.root.format(id=config["exp_id"]), exist_ok=False)
        except FileExistsError:
            config["exp_id"] = config["exp_id"] + datetime.now().strftime("-%Y%m%d-%H%M%S")
            os.makedirs(config.path.root.format(id=config["exp_id"]), exist_ok=False)
            logging.basicConfig(
                filename=config.path.logging.format(id=config["exp_id"]),
                level=config.logging_level,
                format="%(asctime)-15s %(levelname)-8s %(message)s",
            )
            logging.warning(
                f"Root directory for experiment id existed, hence created a new unique exp id and directory {config.path.root.format(id=config['exp_id'])}."
            )
    else:
        if not os.path.exists(config.path.root.format(id=config["exp_id"])):
            raise ValueError("The folder for the experiment id given must exist if 'accumulate' is set to false.")


    run(config, args.config_path)
