"""
Nowcasting Deep Learning model base class.
Bent Harnist (FMI) 2022
"""
import pytorch_lightning as pl
import torch

from utils.match.loss import match_loss
from utils.match.optimizer import match_optimizer, match_lr_scheduler
from utils.match.network import match_network


class NowcastingModel(pl.LightningModule):

    def __init__(self, config : dict):
        super(NowcastingModel, self).__init__()
        self.save_hyperparameters()
        self.mode = config.nn.kwargs.mode
        self.train_display = config.nowcasting_params.train_display
        self.personal_device = torch.device(config.nowcasting_params.device)

        self.nn = match_network(config.nn)
        self.criterion = match_loss(config.loss)
        self.optimizer = match_optimizer(config.optim)
            
        #leadtimes
        self.train_leadtimes = config.nowcasting_params.train_leadtimes
        self.verif_leadtimes = config.nowcasting_params.verif_leadtimes
        self.predict_leadtimes = config.nowcasting_params.predict_leadtimes

        self.automatic_optimization = True
        self.lr_sch_params = config.lr_scheduler


    def configure_optimizers(self, parameters):
        optimizer = self.optimizer(parameters)
        if self.lr_sch_params.name is None:
            return optimizer
        else:
            lr_scheduler_instance = match_lr_scheduler(lr_scheduler_config=self.lr_sch_params)
            lr_scheduler = lr_scheduler_instance(optimizer=optimizer)
            return [optimizer],{"scheduler" : lr_scheduler, "monitor" : self.lr_sch_params.monitor}


    def calculate_loss(self, batch, y_hat, leadtimes, **kwargs):
        batch["outputs"] = batch["outputs"][:,:leadtimes]

        if "heteroscedastic" in self.mode:
            uncertainty = y_hat[:,:leadtimes,:,:,1] # as log variance
            y_hat = y_hat[:,:leadtimes,:,:,0]
            loss = self.criterion(batch, y_hat, torch.exp(uncertainty))
        else:
            y_hat = y_hat[:,:leadtimes,:,:]
            loss = self.criterion(batch, y_hat)

        return loss

    def preprocess_inputs(self, x):
        return x

    def postprocess_outputs(self, y_hat):
        return y_hat
