"""
Log probabilistic nowcast images to the model logger.
Bent Harnist (FMI) 2022

Currently implemented: 
    - Plot inputs, ground truth, prediction means, their epistemic and aleatoric uncertainties
"""
from pytorch_lightning.callbacks import Callback
from utils import uncertainty_estimation
from utils import uncertainty_plot, input_plot
import torch

class LogProbabilisticNowcast(Callback):
    """
    Log nowcast images to the model logger.
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.verif_display = config.verif_display
        self.train_display = config.train_display
        self.quantity = config.quantity
        self.prediction_cmap = config.prediction_cmap
        self.uncertainty_cmap = config.uncertainty_cmap
        self.n_stds = config.n_stds


    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs : dict,
        batch : tuple,
        batch_idx: int,
        ) -> None:
        if ((pl_module.global_step - 1) % self.train_display) != 0:
            return
        self.plot_routine(
            batch=batch,
            outputs=outputs,
            pl_module=pl_module,
            dataset_object=trainer.datamodule.train_dataset,
            split_name="train"
        )
        
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: dict,
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if batch_idx not in self.verif_display:
            return
        self.plot_routine(
                batch=batch,
                outputs=outputs,
                pl_module=pl_module,
                dataset_object=trainer.datamodule.valid_dataset,
                split_name="valid"
            )

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: dict,
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if batch_idx not in self.verif_display:
            return
        self.plot_routine(
                batch=batch,
                outputs=outputs,
                pl_module=pl_module,
                dataset_object=trainer.datamodule.test_dataset,
                split_name="test"
            )

    def plot_routine(
        self,
        batch,
        outputs,
        pl_module : "pl.LightningModule",
        dataset_object,
        split_name : str
        ):

        x = batch["inputs"]
        y = batch["outputs"]
        idx = batch["idx"]

        if pl_module.mode == "regression" or pl_module.mode == "segmentation":
            # B,T,S,W,H gotten with "regression"
            y_hat = outputs["prediction"]
            ep = torch.std(y_hat, dim=0) * self.n_stds
            y_mean = torch.mean(y_hat, dim=0)
            al = torch.zeros_like(y_mean) * self.n_stds
            plot_al = False
            pl_module.log("mean_epistemic_uncertainty", torch.mean(ep), prog_bar=True)
        
        elif "heteroscedastic" in pl_module.mode:
            # B,T,S,C,W,H gotten with heteroscedastic regression
            y_hat = outputs["prediction"][...,0]
            sigma = outputs["prediction"][...,1]
            al, ep = uncertainty_estimation(y_hat=y_hat, sigma=sigma, n_stds=self.n_stds)
            if y_hat.ndim == 5:
                y_mean = torch.mean(y_hat, dim=0)
            elif y_hat.ndim == 4:
                y_mean = y_hat
            plot_al = True
            pl_module.log("mean_epistemic_uncertainty", torch.mean(ep), prog_bar=True)
            pl_module.log("mean_aleatoric_uncertainty", torch.mean(al), prog_bar=True)

        else:
            raise NotImplementedError(f"Nowcast plot not implemented for mode {pl_module.mode}")

        if self.quantity == "DBZH":
            transform_data = dataset_object.scaled_to_dbz
        elif self.quantity == "RR":
            transform_data = dataset_object.inverse_transform
        else:
            transform_data = lambda x:x
        
        #time_list = dataset_object.ds_idx_to_storage_idx(idx[0])
        time_list = dataset_object.get_window(idx[0])
        input_time_list = time_list[:x.shape[1]]
        pred_time_list = time_list[x.shape[1]: x.shape[1] + y_mean.shape[1]]

        input_fig = input_plot(
            inputs=transform_data(x).cpu(),
            time_list=input_time_list,
            quantity=self.quantity,
            cmap=self.prediction_cmap
        )

        uncertainty_fig = uncertainty_plot(
            ground_truth=transform_data(y).detach().cpu(),
            output_mean=transform_data(y_mean).detach().cpu(),
            output_aleatoric=transform_data(al).detach().cpu(),
            output_epistemic=transform_data(ep).detach().cpu(),
            time_list=pred_time_list,
            quantity=self.quantity,
            cmaps=(self.prediction_cmap, self.uncertainty_cmap),
            plot_aleatoric=plot_al
        )
        
        pl_module.logger.experiment.add_figure(
            f"{split_name}_sample_input_images", input_fig, global_step=pl_module.global_step)
        pl_module.logger.experiment.add_figure(
            f"{split_name}_sample_prediction_images", uncertainty_fig, global_step=pl_module.global_step)
        pl_module.logger.experiment.flush()
