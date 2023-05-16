"""Datamodule for FMI radar composite."""
import h5py
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import FMIDatasetHDF5, FMIDatasetPGM
from datamodules.transformation import NowcastingTransformation


class FMICompositeDataModule(pl.LightningDataModule):
    def __init__(self, config_dm : dict):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_config = config_dm.dataset_config
        self.train_config = config_dm.train_config

        self.importer = config_dm.importer
        if self.importer == "hdf5":
            self.db_path = config_dm.hdf5_path
            self.db = h5py.File(self.db_path, 'r')
            self.ds_cls = FMIDatasetHDF5
        elif self.importer == "pgm_gzip":
            self.pgm_path = config_dm.path
            self.pgm_fname = config_dm.filename
            self.ds_cls = FMIDatasetPGM
        else:
            raise NotImplementedError(f"importer {self.importer} unavailable.")

        self.train_transform = NowcastingTransformation(self.train_config.train_transformation)
#        self.valid_transform = NowcastingTransformation(self.train_config.valid_transformation)
        

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self, stage):
        # called on every GPU
        if self.importer == "hdf5":
            self.train_dataset = self.ds_cls(split="train", db=self.db, **self.dataset_config)
            self.valid_dataset = self.ds_cls(split="valid", db=self.db,  **self.dataset_config)
            self.test_dataset = self.ds_cls(split="test", db=self.db, **self.dataset_config)
            self.predict_dataset = self.ds_cls(split="predict",db = self.db, **self.dataset_config)
        elif self.importer == "pgm_gzip":
            self.train_dataset = self.ds_cls(split="train", path=self.pgm_path, filename=self.pgm_fname, **self.dataset_config)
            self.valid_dataset = self.ds_cls(split="valid", path=self.pgm_path, filename=self.pgm_fname, **self.dataset_config)
            self.test_dataset = self.ds_cls(split="test", path=self.pgm_path, filename=self.pgm_fname, **self.dataset_config)
            self.predict_dataset = self.ds_cls(split="predict", path=self.pgm_path, filename=self.pgm_fname, **self.dataset_config)
    
    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_config.train_batch_size,
            num_workers=self.train_config.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.train_config.valid_batch_size,
            num_workers=self.train_config.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.train_config.test_batch_size,
            num_workers=self.train_config.num_workers,
            shuffle=False,
        )
        return test_loader
    
    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.predict_dataset,
            batch_size=self.train_config.predict_batch_size,
            num_workers=self.train_config.num_workers,
            shuffle=False,
            collate_fn=_collate_fn,
        )
        return predict_loader
    
    def teardown(self, stage=None):
        if self.importer == "hdf5":
            self.db.close()

    def get_train_size(self):
        if hasattr(self, "train_dataset"):
            return len(self.train_dataset)
        else:
            dummy_train_ds = (
                self.ds_cls(split="train", db=self.db, **self.dataset_config)
                if self.importer == "hdf5" else self.ds_cls(split="train", path=self.pgm_path, filename=self.pgm_fname, **self.dataset_config)
            )
            train_size = len(dummy_train_ds)
            del dummy_train_ds
            return train_size

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        "Apply training data transformation"
        if self.trainer.state.fn != "predict":
            return self.train_transform(batch)
#        elif self.trainer.validating:
#            return self.valid_transform(batch)
        else:
            return batch


def _collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


