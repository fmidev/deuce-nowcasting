"""
Deterministic nowcasting neural network base class.
Inherits from NowcastingModel.
Bent Harnist (FMI) 2022
"""
import torch

from models.nowcasting_model import NowcastingModel

class DeterministicModel(NowcastingModel):
    """
    Deterministic nowcasting neural network base class.
    Inherits from NowcastingModel.
    Bent Harnist (FMI) 2022
    """
    def __init__(self, config: dict):
        super().__init__(config)

    def setup(self, stage = None) -> None:
        return super().setup(stage)

    def configure_optimizers(self):
        return NowcastingModel.configure_optimizers(self, parameters=self.parameters())

    def deterministic_step(self, batch, batch_idx, leadtimes : int, return_predictions : bool):
        x,y,_ = batch
        x_preprocessed = self.preprocess_inputs(x)
        y_hat_raw = self.nn(x_preprocessed, leadtimes)
        y_hat = self.postprocess_outputs(y_hat_raw)
        loss = self.calculate_loss(y, y_hat_raw, leadtimes=leadtimes)
        loss = sum(loss) if isinstance(loss, list) else loss
        self.log_dict({f"{self.trainer.state.stage}_loss": loss})
        if return_predictions:
            return (
            {"prediction" : y_hat,
            "loss" : loss}
            )
        else:
            return {"loss" : loss}

    def predict_step(self, batch, batch_idx: int, dataloader_idx = None):
        x,_,_ = batch
        y_hat = self.nn(x, self.predict_leadtimes)
        if "heteroscedastic" not in self.mode:
            return self.trainer.datamodule.predict_dataset.scaled_to_dbz(y_hat)
        else:
            return y_hat # return log-normalized version, needed for post-processing

    def training_step(self, batch, batch_idx):
        return self.deterministic_step(
            batch,
            batch_idx,
            self.train_leadtimes,
            return_predictions = (self.global_step) % self.train_display == 0
            )
        
    def validation_step(self, batch, batch_idx):
        return self.deterministic_step(
            batch,
            batch_idx,
            self.verif_leadtimes,
            True
            )

    def test_step(self, batch, batch_idx):
        return self.deterministic_step(
            batch,
            batch_idx,
            self.verif_leadtimes,
            True
            )
