"""
Ensemble nowcasting using Bayesian Neural Networks with Variational inference.
Bent Harnist (FMI) 2022
"""
import tyxe
import torch

from models import EnsembleModel, VIModel

class EnsembleVIModel(EnsembleModel, VIModel):
    """
    Ensemble nowcasting using Bayesian Neural Networks with Variational inference.
    """
    def __init__(self, config : dict) -> None:
        EnsembleModel.__init__(self, config)
        VIModel.__init__(self, config)

        self.bnn = tyxe.PytorchBNN(
            self.nn,
            self.prior,
            self.guide,
            name="bnn",
            closed_form_kl=config.vi_params.closed_form_kldiv
            )
        self.dummy_data = torch.empty((1,self.nn.n_input_timesteps,32,32),device=self.personal_device)
        self.bnn.to(self.personal_device)

    def configure_optimizers(self):
        return super().configure_optimizers(parameters=self.bnn.pytorch_parameters(self.preprocess_inputs(self.dummy_data)))

    def setup(self, stage = None) -> None:
        return VIModel.setup(self, stage=None)

    def ensemble_step(self, batch, batch_idx, leadtimes: int, samples: int, return_predictions: bool):
        #return super().ensemble_step(batch, batch_idx, leadtimes, samples, return_predictions)
        x = batch["inputs"]
        y = batch["outputs"]

        x_preprocessed = self.preprocess_inputs(x)

        data_loss = 0
        kl_loss = 0

        y_seq = torch.empty((samples, *y[:,:leadtimes].shape), device=self.device)

        if "heteroscedastic" in self.mode:
            y_seq = y_seq[...,None].repeat(*[1]*y_seq.ndim,2)

        for s in range(samples):
            with self.fit_ctxt():
                y_hat_raw = self.bnn(x_preprocessed, leadtimes)
                y_hat = self.postprocess_outputs(y_hat_raw)

            y_seq[s] = y_hat.detach()
            next_data_loss = self.calculate_loss(y, y_hat_raw, leadtimes)
            data_loss += (sum(next_data_loss) if isinstance(next_data_loss, list) else next_data_loss)
            kl_loss += self.bnn.cached_kl_loss * self.kl_weight(
                batch_idx = self.global_step, current_epoch = self.current_epoch
                )

        data_loss /= samples
        kl_loss /= samples
        total_loss = data_loss + kl_loss

        self.log_dict({
            f"{self.trainer.state.stage}_loss": total_loss,
            f"{self.trainer.state.stage}_data_loss": data_loss,
            f"{self.trainer.state.stage}_kl_loss": self.bnn.cached_kl_loss
            })
        if return_predictions:
            return (
            {"prediction" : y_seq,
            "loss" : total_loss}
            )
        else:
            return {"loss" : total_loss}

    #def validation_step_end(self, step_output):
    #    return step_output["loss"]

    def predict_step(self, batch, batch_idx: int, dataloader_idx = None):
        x  = batch["inputs"]
        x_preprocessed = self.preprocess_inputs(x)
        y_hat = torch.empty((self.predict_samples, x.shape[0], self.predict_leadtimes, *x.shape[-2:]), device=self.device)
        if "heteroscedastic" in self.mode:
            y_hat = y_hat[...,None].repeat(*[1]*y_hat.ndim,2)

        for s in range(self.predict_samples):
            y_hat_raw = self.bnn(x_preprocessed, self.predict_leadtimes)
            y_hat[s] = self.postprocess_outputs(y_hat_raw)

        if "heteroscedastic" not in self.mode:
            return self.trainer.datamodule.predict_dataset.scaled_to_dbz(y_hat)
        else:
            return y_hat # return log-normalized version, needed for post-processing

    def on_load_checkpoint(self, checkpoint):
        VIModel.on_load_checkpoint(self, checkpoint)
        return

    def on_save_checkpoint(self, checkpoint):
        VIModel.on_save_checkpoint(self, checkpoint)
        return
