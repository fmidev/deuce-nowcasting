"""
This script will run nowcasting prediction 
for the model and dataset of choice
"""
import argparse
from pathlib import Path
import os
import shutil
import random

from attrdict import AttrDict
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from utils import load_config, setup_logging

from utils.match.data import match_datamodule
from utils.match.model import match_model
from utils.match.prediction_writer import match_prediction_writer


def run(checkpoint_path, config_path, platform, model_name : str,
    data_name : str, seed=0) -> None:

    cfg_path = Path(config_path)
    platform_cfg = load_config(cfg_path / Path("platform") / Path(platform) / Path("fmi_platform.yaml"))
    data_cfg = load_config(cfg_path / Path("data") / Path(f"{data_name}_dataset.yaml"))
    data_cfg.update(platform_cfg)
    model_cfg = load_config(cfg_path / Path("model") / Path(f"{model_name}.yaml"))
    experimental_cfg = load_config(cfg_path / Path("experimental.yaml"))


    exp_id = experimental_cfg.experiment_id
    out_path = experimental_cfg.save_path_fmt.format(
        id=exp_id
    )
    os.makedirs(out_path, exist_ok=True)
    if not os.path.exists(os.path.join(out_path, f"{exp_id}_config")):
        shutil.copytree(
            src=cfg_path,
            dst=os.path.join(out_path, f"{exp_id}_config"),
            symlinks=True
        )

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    setup_logging(experimental_cfg.logging)
    
    datamodule = match_datamodule(data_config=AttrDict({"name" : data_name, "kwargs" : data_cfg}))
    model_cfg.update({"dataset_size" : datamodule.get_train_size()})
    model = match_model(model_config=AttrDict({"name" : model_name, "kwargs" : model_cfg}))

    model.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=torch.device(model_cfg.nowcasting_params.device)
        )

    output_writer = match_prediction_writer(experimental_cfg.prediction_output)

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(out_path, "logs"),
        name=f"predict_{cfg_path}"
        )
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[output_writer],
        **experimental_cfg.trainer_kwargs
    )

    # Predictions are written in HDF5 file
    trainer.predict(model, datamodule, return_predictions=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    argparser.add_argument(
        "config",
        type=str,
        help="Configuration folder path",
    )
    argparser.add_argument("platform", type=str, help="platform name")
    argparser.add_argument("-m", "--model", type=str, required=True,
                            help="Name of the Nowcasting model to use, as described in utils/match/....")
    argparser.add_argument("-d", "--data", type=str, required=True,
                            help="Name of the dataset to use, as described in utils/match/....")
    argparser.add_argument("-s", "--seed", type=str, default=0,
                            help="Seed for the RNG.")

    args = argparser.parse_args()

    run(args.checkpoint, args.config, args.platform, args.model, args.data, args.seed)
