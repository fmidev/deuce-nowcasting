"""
FMI lowest reflectivity composite datasets
"""
import logging
import gzip
from pathlib import Path

import numpy as np
import torch
from matplotlib.pyplot import imread
import h5py
import pandas as pd

from datasets import AbstractRadarDataset

class PgmInterfaceFMI:
    "PGM GZIP storage interface for FMI composite data"
    def __init__(
        self,
        path=None,
        filename=None,
        ) -> None:
        assert path is not None, "No path for FMI composites given!"
        assert filename is not None, "No filename format for FMI composites given!"
        self.path = path
        self.filename = filename

    def _read_pgm_composite(self, filename, no_data_value=-32):
            """Read uint8 PGM composite, convert to dBZ."""
            data = imread(gzip.open(filename, "r"))
            mask = data == 255
            data = data.astype(np.float64)
            data = (data - 64.0) / 2.0
            data[mask] = no_data_value

            return data

    def _read_data_from_storage(self, storage_idx):
        """Use the storage retrieval identifier data to read the data into memory"""
        window = storage_idx

        data = np.empty((self.num_frames, *self.image_size))

        # Check that window has correct length
        if (window[-1] - window[0]).seconds / (self.timestep * 60) != (
            self.num_frames - 1
        ):
            logging.info(f"Window {window[0]} - {window[-1]} wrong!")

        for i, date in enumerate(window):
            fn = Path(
                self.path.format(
                    year=date.year,
                    month=date.month,
                    day=date.day,
                    hour=date.hour,
                    minute=date.minute,
                    second=date.second,
                )
            ) / Path(
                self.filename.format(
                    year=date.year,
                    month=date.month,
                    day=date.day,
                    hour=date.hour,
                    minute=date.minute,
                    second=date.second,
                )
            )
            data[i, ...] = self._read_pgm_composite(fn, no_data_value=-32)
        
        return torch.Tensor(data)

class HDF5InterfaceFMI:
    "HDF5 storage interface for FMI composite data"
    def __init__(
        self,
        db : h5py.File = None,
        ) -> None:
        assert db is not None
        self.db = db

    def read_hdf5_db(self, filename, db):
        """Read composite reflectivity (dBZ) from the hdf5 database"""
        return db[filename][...]

    def _read_data_from_storage(self, storage_idx):
        """Use the storage retrieval identifier data to read the data into memory"""
        window = storage_idx
        data = np.empty((self.num_frames, *self.image_size))

        # Check that window has correct length
        if (window[-1] - window[0]).seconds / (self.timestep * 60) != (
            self.num_frames - 1
        ):
            logging.info(f"Window {window[0]} - {window[-1]} wrong!")
        
        for i, date in enumerate(window):
            fn = date.strftime("%Y-%m-%d %H:%M:%S")
            data[i, ...] = self.read_hdf5_db(fn, self.db)

        return torch.Tensor(data)

class AbstractFMIDataset(AbstractRadarDataset):
    "Abstract base class for FMI composite datasets"
    def __init__(
        self,
        split="train",
        date_list=None,
        predict_date_list=None,
        len_date_block = 48,
        **kwargs
        ) -> None:

        super().__init__(**kwargs)
        
        assert date_list is not None, "No date list for FMI composites given!"

        # get filename where data timestamps reside
        if split != "predict" or predict_date_list is None:
            date_list_fn = date_list.format(split=split)
        else:
            date_list_fn = predict_date_list
        # Then get data timestamps from that file ...
        self.date_list = pd.read_csv(
            date_list_fn, header=None, parse_dates=[0]
        )
        self.date_list.set_index(self.date_list.columns[0], inplace=True)

        self.len_date_block = len_date_block
        self.date_list_pdt = self.date_list.index.to_pydatetime()
        self.windows = self.make_windows()

    def __len__(self):
        """Mandatory property for Dataset."""
        return self.windows.shape[0]

    def make_windows(self):
        # Get windows
        num_blocks = int(len(self.date_list) / self.len_date_block)
        blocks = np.array_split(self.date_list.index.to_pydatetime(), num_blocks)
        windows = pd.DataFrame(
            np.concatenate(
                [
                    np.lib.stride_tricks.sliding_window_view(b, self.num_frames)
                    for b in blocks
                ]
            )
        )
        return windows

    def ds_idx_to_storage_idx(self, ds_index):
        """Use the dataset batch idx to fetch the corresponding storage retrieval identifier"""
        if isinstance(ds_index, int):
            return self.windows.iloc[ds_index].dt.to_pydatetime()
        elif ds_index.numel() == 1:
            return self.windows.iloc[ds_index.item()].dt.to_pydatetime()
        else:
            return np.stack(
                [
                    self.windows.iloc[ds_index].iloc[i].dt.to_pydatetime()
                    for i in range(len(ds_index))
                ]
            )

    def get_common_time(self, index):
        window = self.get_window(index)
        return window[self.num_frames_input - 1]

    def storage_to_input_conversion(self, data: torch.Tensor):
        if self.normalization_method == "unit":
            return torch.Tensor(((data + 32.0) / 127.0)).type(torch.float)
        elif self.normalization_method == "none":
            return data
        else:
            raise ValueError

    def output_to_storage_conversion(self, data: torch.Tensor):
        if self.normalization_method == "unit":
            return ((data * 127) - 32)
        elif self.normalization_method == "none":
            return data
        else:
            raise ValueError


class FMIDatasetPGM(AbstractFMIDataset, PgmInterfaceFMI):
    "FMI composite dataset with PGM GZIP storage interface"
    def __init__(
        self,
        path=None,
        filename=None,
        **kwargs
        ) -> None:
        AbstractFMIDataset.__init__(self, **kwargs)
        PgmInterfaceFMI.__init__(self, path, filename)

    def _read_data_from_storage(self, storage_idx):
        return PgmInterfaceFMI._read_data_from_storage(self,storage_idx)


class FMIDatasetHDF5(AbstractFMIDataset, HDF5InterfaceFMI):
    "FMI composite dataset with HDF5 storage interface"
    def __init__(
        self,
        db : h5py.File,
        **kwargs
        ) -> None:
        HDF5InterfaceFMI.__init__(self, db)
        AbstractFMIDataset.__init__(self, **kwargs)

    def _read_data_from_storage(self, storage_idx):
        return HDF5InterfaceFMI._read_data_from_storage(self, storage_idx)
