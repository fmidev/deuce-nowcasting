import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from typing import Any, List
import numpy as np
from datetime import timedelta
import h5py
from pathlib import Path

class BCNNHeteroHDF5Writer(BasePredictionWriter):
    def __init__(
        self,
	    db_name: str,
        group_format: str,
        method_name: str,
        what_attrs: dict = {},
        write_interval: str = "batch",
        combine: bool = True,
        **kwargs,
    ):
        super().__init__(write_interval)
        self.db_name = db_name
        self.what_attrs = what_attrs
        self.group_format = group_format
        self.method_name = method_name
        self.combine = combine

    def write_on_batch_end(
        self,
        trainer,
        pl_module : 'LightningModule',
        prediction: Any,
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
    
        batch_indices_ = batch["idx"]
        batch_indices_ = batch_indices_.tolist()

        prediction = prediction.detach()

        prediction_mu = prediction[...,0] # means
        prediction_std = torch.sqrt(torch.exp(prediction[...,1])) # aleatoric stds

        if self.combine:
            eps = torch.broadcast_to(torch.randn(prediction.shape[:2], device=prediction.device)[...,None,None,None], prediction.shape[:-1])
            prediction = prediction_mu + eps * prediction_std
            #prediction[prediction < -31.5] = -31.5 # cut off too small, big values in dBZ
            #prediction[prediction > 90.0] = 90.0 # ...
        else:
            prediction = torch.stack([prediction_mu, prediction_std], dim=-1)

        prediction = prediction.cpu().numpy()


        with h5py.File(self.db_name, 'a') as db:

            for bi, b_idx in enumerate(batch_indices_):
                common_time = trainer.datamodule.predict_dataset.get_common_time(b_idx)
                group_name = self.group_format.format(
                    timestamp=common_time,
                    method = self.method_name)
                group = db.require_group(group_name)

                n_leadtimes = prediction.shape[2] # S, B, T, W, H, C

                for i in range(n_leadtimes):
                    date = common_time + timedelta(
                        minutes= i * trainer.datamodule.predict_dataset.timestep
                    )
                    dname = f"{i + 1}"
                    ds_group = group.require_group(dname)
                    what_attrs = self.what_attrs.copy()
                    what_attrs["validtime"] = np.string_(f"{date:%Y-%m-%d %H:%M:%S}")
                    packed = self.arr_compress_uint8(prediction[:,bi,i], missing_val = 255)
                    write_image(
                        group=ds_group,
                        ds_name="data",
                        data=packed,
                        what_attrs=what_attrs,
                    )

    def write_on_epoch_end(
        self,
	    trainer,
        pl_module: "LightningModule",
        predictions: List[Any],
        batch_indices: List[Any],
    ):
        pass

    def arr_compress_uint8(
        self, array: np.ndarray, missing_val: np.uint8 = 255, threshold=0,
    ) -> np.ndarray:
        masked = np.ma.masked_where(~np.isfinite(array), array)
        max_value = 0.995
        mask_big_values = array[...] >= max_value
        masked[masked[...] < 0.0] = 0.0
        arr = (masked * 255).astype(np.uint8)
        arr[arr < threshold] = 0
        arr[arr.mask] = missing_val
        arr[mask_big_values] = 254
        return arr.data

