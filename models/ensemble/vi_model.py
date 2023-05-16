"""
Basic Variational Inference, Bayesian NN components for nowcasting neural networks.
Bent Harnist (FMI) 2022
"""
from functools import partial
import contextlib

import torch
import tyxe
import pyro
import pyro.distributions as dist

from utils.uncertainty.kl_weighting import *

class VIModel(object):
    """
    Basic Variational Inference components for nowcasting neural networks.
    """
    def __init__(self, config : dict) -> None:
        self.dataset_size = config.dataset_size

        if not hasattr(self, "personal_device"):
            self.personal_device = torch.device(config.nowcasting_params.device)

        self.epochs_at_start = config.vi_params.epochs_at_start

        # prior definition
        if config.vi_params.prior.name == "IID_normal":
            self.prior = tyxe.priors.IIDPrior(
                dist.Normal(
                    torch.tensor(float(config.vi_params.prior.kwargs.loc), device=self.personal_device), 
                    torch.tensor(float(config.vi_params.prior.kwargs.scale), device=self.personal_device)
                )
            )
        elif config.vi_params.prior.name == "IID_gaussian_scale_mixture":
            probs = torch.as_tensor(config.vi_params.prior.kwargs.probs, device=self.personal_device)
            locs = torch.zeros_like(probs, device=self.personal_device)
            scales = torch.as_tensor(config.vi_params.prior.kwargs.scale, device=self.personal_device)
            self.prior = tyxe.priors.IIDPrior(
                GaussianScaleMixture(
                    mixture_distribution=torch.distributions.Categorical(probs),
                    component_distribution=torch.distributions.Normal(locs, scales)
                )
            )
        else:
            raise NameError(f"prior {config.vi_params.prior.name} undefined")

        # fit context definition
        if config.vi_params.fit_context == 'flipout':
            self.fit_ctxt = tyxe.poutine.flipout
        else:
            self.fit_ctxt = contextlib.nullcontext

        # guide definition
        self.guide = partial(tyxe.guides.AutoNormal, init_scale = float(config.vi_params.guide.init_scale))

        # optimization parameters
        self.closed_form_kldiv = config.vi_params.closed_form_kldiv
        #self.optim_params = config.vi_params.optim
        self.kl_weighting_scheme = config.vi_params.kl_weighting


    def setup(self, stage=None):
        if self.kl_weighting_scheme == "blundell":
            kl_weight_fn = kl_weight_blundell
        elif self.kl_weighting_scheme == "equal":
            kl_weight_fn = kl_weight_equal
        elif self.kl_weighting_scheme == "epoch":
            kl_weight_fn = kl_weight_epoch

        self.kl_weight = partial(kl_weight_fn, 
            num_batches = self.trainer.datamodule.get_train_size(),
            epochs_at_start = self.epochs_at_start
            )

    def remove_dict_entry_startswith(self, dictionary, string):
        """Used to remove entries with 'bnn' in checkpoint state dict"""
        n = len(string)
        for key in dictionary:
            if string == key[:n]:
                dict2 = dictionary.copy()
                dict2.pop(key)
                dictionary = dict2
        return dictionary


    def on_save_checkpoint(self, checkpoint):
        """Saving Pyro's param_store for the bnn's parameters"""
        checkpoint["param_store"] = pyro.get_param_store().get_state()
        

    def on_load_checkpoint(self, checkpoint):
        pyro.get_param_store().set_state(checkpoint["param_store"])
        checkpoint['state_dict'] = self.remove_dict_entry_startswith(
            checkpoint['state_dict'], 'bnn')
