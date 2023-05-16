"""
This script will train the model given 
with the dataset and configuration folder given.
"""
from pathlib import Path
import argparse
import random
import shutil
import os
import sys

from attrdict import AttrDict
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from utils.config import load_config
from utils.logging import setup_logging

from utils.match.callback import match_callback
from utils.match.data import match_datamodule
from utils.match.model import match_model


def main(
    config_path : str,
    platform : str,
    model_name : str,
    data_name : str,
    callback_name : str = None,
    checkpoint=None,
    seed=0):

    cfg_path = Path(config_path)
    platform_cfg = load_config(cfg_path / Path("platform") / Path(platform) / Path(f"{data_name}_platform.yaml"))
    data_cfg = load_config(cfg_path / Path("data") / Path(f"{data_name}_dataset.yaml"))
    data_cfg.update(platform_cfg)
    model_cfg = load_config(cfg_path / Path("model") / Path(f"{model_name}.yaml"))
    experimental_cfg = load_config(cfg_path / Path("experimental.yaml"))
    
    callbacks = []
    if callback_name is not None:
        callbacks_cfg = load_config(cfg_path / Path("callback") / Path(f"{callback_name}.yaml"))
        for callback_config in callbacks_cfg.values():
            callbacks.append(match_callback(callback_config=AttrDict(callback_config)))

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

    # callbacks
    
    # tensorboard logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(out_path, "logs"),
        name=f"train_{cfg_path}"
        )

    trainer = pl.Trainer(
        logger = tb_logger,
        callbacks=callbacks,
        **experimental_cfg.trainer_kwargs
        )

    if checkpoint is not None:
        model.load_from_checkpoint(checkpoint, map_location=torch.device(model_cfg.nowcasting_params.device))
    
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)

    torch.save(
        obj = model.state_dict(),
        f = os.path.join(
            out_path,
            f"{exp_id}_state_dict_{experimental_cfg.savefile}.ckpt")
            )
    trainer.save_checkpoint(
        filepath = os.path.join(
            out_path,
            f"{exp_id}_{experimental_cfg.savefile}.ckpt"
        )
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("config", type=str, help="Configuration folder")
    argparser.add_argument("platform", type=str, help="platform name")
    argparser.add_argument("-m", "--model", type=str, required=True,
                            help="Name of the Nowcasting model to use, as described in utils/match/....")
    argparser.add_argument("-d", "--data", type=str, required=True,
                            help="Name of the dataset to use, as described in utils/match/....")
    argparser.add_argument("-a", "--callback", type=str, required=False,
                            help="Name of the callback param YAML file to use")
    argparser.add_argument("-c", "--continue_training", type=str, default=None,
                            help="Path to checkpoint for model that is continued.")
    argparser.add_argument("-s", "--seed", type=str, default=0,
                            help="Seed for the RNG.")

    args = argparser.parse_args()
    main(args.config, args.platform, args.model, args.data, callback_name=args.callback, checkpoint=args.continue_training, seed=args.seed)

