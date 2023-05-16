"""
Ensemble nowcasting neural network base class.
Inherits from NowcastingModel.
Bent Harnist (FMI) 2022
"""
import torch

from models.nowcasting_model import NowcastingModel

class EnsembleModel(NowcastingModel):
    """
    Ensemble nowcasting neural network base class.
    Inherits from NowcastingModel.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.train_samples = config.ensemble_params.train_samples
        self.verif_samples = config.ensemble_params.verif_samples
        self.predict_samples = config.ensemble_params.predict_samples

    def configure_optimizers(self, parameters):
        return super().configure_optimizers(parameters)

    def setup(self, stage = None) -> None:
        return super().setup(stage)

    def ensemble_step(self, batch, batch_idx, leadtimes : int, samples : int, return_predictions : bool):
        raise NotImplementedError("Inheriting ensemble model must implement this method")

    def predict_step(self, batch, batch_idx: int, dataloader_idx = None):
        raise NotImplementedError("Inheriting ensemble model must implement this method")

    def training_step(self, batch, batch_idx):
        return self.ensemble_step(
            batch,
            batch_idx,
            self.train_leadtimes,
            self.train_samples,
            return_predictions = self.global_step % self.train_display == 0
            )
        
    def validation_step(self, batch, batch_idx):
        #self.bnn.train(False)
        return self.ensemble_step(
            batch,
            batch_idx,
            self.verif_leadtimes,
            self.verif_samples,
            True
            )

    def test_step(self, batch, batch_idx):
        #self.bnn.train(False)
        return self.ensemble_step(
            batch,
            batch_idx,
            self.verif_leadtimes,
            self.verif_samples,
            True
            )
