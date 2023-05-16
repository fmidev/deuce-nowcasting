"""
Bent Harnist (2022) FMI
"""
from abc import ABC, abstractmethod

from torch.utils.data import Dataset
from skimage.measure import block_reduce
import torch
import numpy as np

class AbstractRadarDataset(Dataset, ABC):
    """
    Base class of Datasets for fetching sequences of radar reflectivity scans or composites.
    """
    def __init__(
        self,
        input_block_length=None,
        prediction_block_length=None,
        timestep=None,
        bbox=None,
        image_size=None,
        bbox_image_size=None,
        input_image_size=None,
        apply_threshold = True,
        threshold = 8.0,
        zerovalue = -10,
        zr_relationship = (200,1.6),
        upsampling_method="average",
        input_reflectivity_unit = "dBZ", #dBZ
        normalization_method = "unit", # unit
        return_idx_with_batch = True, 
        force_channel_dimension = False,
        ) -> None:

        Dataset.__init__(self)

        self.image_size = image_size
        self.bbox_image_size = bbox_image_size
        self.input_image_size = input_image_size
        
        if bbox is None or self.image_size == self.bbox_image_size:
            self.use_bbox = False
        else:
            self.use_bbox = True
            self.bbox_x_slice = slice(bbox[0], bbox[1])
            self.bbox_y_slice = slice(bbox[2], bbox[3])


        self.num_frames_input = input_block_length
        self.num_frames_output = prediction_block_length
        self.num_frames = input_block_length + prediction_block_length

        self.return_idx_with_batch = return_idx_with_batch
        self.force_channel_dimension = force_channel_dimension

        self.timestep = timestep
        self.apply_threshold = apply_threshold
        self.threshold = threshold
        self.zerovalue = zerovalue
        self.zr_relationship = zr_relationship

        if normalization_method not in ["unit"]:
            raise NotImplementedError(f"data normalization method {normalization_method} not implemented, please choose from ['unit', 'none']")
        else:    
            self.normalization_method = normalization_method

        if input_reflectivity_unit not in ["dBZ"]:
            raise NotImplementedError(f"Radar echo unit {input_reflectivity_unit} not implemented, please choose from ['dBZ']")
        else:    
            self.input_reflectivity_unit = input_reflectivity_unit

        if upsampling_method not in ["average"]:
            raise NotImplementedError(
                f"Upsampling {self.upsampling_method} not yet implemented, please choose from ['average']"
            )
        else:
            self.upsampling = np.nanmean

    def __len__(self):
        """Mandatory property for Dataset."""
        raise NotImplementedError()

    @abstractmethod
    def ds_idx_to_storage_idx(self, ds_idx):
        """Use the dataset batch idx to fetch the corresponding storage retrieval identifier"""
        pass

    @abstractmethod
    def _read_data_from_storage(self, storage_idx):
        """Use the storage retrieval identifier data to read the data into memory"""
        pass

    def get_window(self, idx):
        "alias"
        return self.ds_idx_to_storage_idx(idx)

    def __getitem__(self, idx):
        """Get one sequence from the storage using the dataloader idx."""

        storage_idx = self.ds_idx_to_storage_idx(idx)
        raw_data = self._read_data_from_storage(storage_idx)
        if self.force_channel_dimension and raw_data.ndim == 3:
            raw_data = raw_data[:,None,:,:]
        elif self.force_channel_dimension and raw_data.ndim != 4:
            raise ValueError(f"input data is of bad dimension with shape {raw_data.shape}")
        else:
            if not raw_data.ndim in [3,4]:
                raise ValueError(f"input data is of bad dimension with shape {raw_data.shape}")
        
        if self.use_bbox:
            raw_data = raw_data[..., self.bbox_x_slice, self.bbox_y_slice]

        block_x = int(self.bbox_image_size[0] / self.input_image_size[0])
        block_y = int(self.bbox_image_size[1] / self.input_image_size[1])
        if raw_data.ndim == 3:
            block_size = (1,block_x, block_y)
        else:
            block_size = (1,1,block_x,block_y)
        raw_data = block_reduce(
                raw_data, func=np.nanmean, cval=0, block_size=block_size
            )
        
        if self.apply_threshold:
            raw_data[raw_data < self.threshold] = self.zerovalue

        data = self.storage_to_input_conversion(raw_data)

        inputs, outputs = (
            data[:self.num_frames_input],
            data[self.num_frames_input:]
        )

        return {
            "inputs" : inputs,
            "outputs": outputs, 
            "idx": idx
        }


    ####################################
    # unit conversion and tresholding

    @abstractmethod
    def storage_to_input_conversion(self, data: torch.Tensor):
        """Unit conversion from storage unit to model units"""
        pass
            
    @abstractmethod
    def output_to_storage_conversion(self, data: torch.Tensor):
        """Unit conversion from model units unit to storage units"""
        pass

    @classmethod
    def dbz_to_mmh(self, data_dbz : torch.Tensor, zr_relationship=(200, 1.6)):
        zr_a, zr_b = zr_relationship
        data_Z = 10 ** (data_dbz * 0.1)
        data_mmh = ( data_Z / zr_a) ** (1/zr_b)
        del data_Z, data_dbz
        return data_mmh

    @classmethod
    def mmh_to_dbz(
        self,
        data_mmh: torch.Tensor,
        threshold_dbz : float = None,
        zerovalue_dbz : float = -32,
        zr_relationship=(200, 1.6)
        ):
        zr_a, zr_b = zr_relationship
        data_mmh[data_mmh < 0] = 0
        data_z = zr_a * data_mmh ** zr_b
        data_dbz = 10 * torch.log10(data_z)
        data_dbz[data_dbz < threshold_dbz] = zerovalue_dbz
        del data_z, data_mmh
        return data_dbz


