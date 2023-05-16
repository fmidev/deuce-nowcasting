"""
Methods for decomposing Variational predictive variance into 
Epistemic and Aleatoric uncertainties for nowcasts.

Bent Harnist (FMI) 2022
"""
import torch
import numpy as np


def _kendall_estimate_npy(y_hat : np.ndarray, sigma : np.ndarray, n_stds = 1, logvar = True):
    """Estimate aleatoric and epistemic uncertainty for numpy array predictions 
    using the method from Kendall and Gal (2017).

    Args:
        y_hat (np.ndarray): Predicted precipitation values, first axis is samples. 
        sigma (np.ndarray): Predicted precipitation log-variance, first axis is samples.
        n_stds (int, optional): Number of standard deviations
        for output uncertainties. Defaults to 2.

    Returns:
        (np.ndarray, np.ndarray): (aleatoric, epistemic) uncertainties
    """
    # epistemic uncertainty
    epistemic = np.std(y_hat, axis=0) * n_stds
    # aleatoric uncertainty
    if logvar:
        aleatoric = np.mean(np.sqrt(np.exp(sigma)), axis=0) * n_stds
    else: 
        aleatoric = np.mean(np.sqrt(sigma), axis=0) * n_stds

    return aleatoric, epistemic


def _kendall_estimate_torch(y_hat : torch.Tensor, sigma : torch.Tensor, n_stds = 1, logvar = True):
    """Estimate aleatoric and epistemic uncertainty for torch.Tensor predictions 
    using the method from Kendall and Gal (2017).

    Args:
        y_hat (torch.Tensor): Predicted precipitation values, first axis is samples. 
        sigma (torch.Tensor): Predicted precipitation log-variance, first axis is samples.
        n_stds (int, optional): Number of standard deviations
        for output uncertainties. Defaults to 2.

    Returns:
        (torch.Tensor, torch.Tensor): (aleatoric, epistemic) uncertainties
    """
    # epistemic uncertainty
    epistemic = torch.std(y_hat, dim=0) * n_stds
    # aleatoric uncertainty
    if logvar:
        aleatoric = torch.mean(torch.sqrt(torch.exp(sigma)), dim=0) * n_stds
    else: 
        aleatoric = torch.mean(torch.sqrt(sigma), dim=0) * n_stds

    return aleatoric, epistemic


def uncertainty_estimation(
    y_hat : "np.ndarray|torch.tensor",
    sigma : "np.ndarray|torch.tensor",
    n_stds : int = 1,
    method : str = "gaussian_likelihood",
    ):
    """Interface for aleatoric and epistemic uncertainty estimations.

    Args:
        y_hat (np.ndarray|torch.tensor): Predicted precipitation values, first axis is samples. 
        sigma (np.ndarray|torch.tensor): Predicted precipitation log-variance, first axis is samples.
        n_stds (int, optional): Number of standard deviations
        for output uncertainties. Defaults to one (1).
        method (str, optional): Uncertainty estimation method. Defaults to "kendall".

    Returns:
        (np.ndarray|torch.tensor, np.ndarray|torch.tensor): (aleatoric, epistemic) uncertainties
    """

    if y_hat.ndim == 4 and sigma.ndim == 4:
        # assuming logvar is true here
        al = torch.sqrt(torch.exp(sigma)) * n_stds
        ep = torch.zeros_like(al)
        return al, ep

    if y_hat.ndim != 5:
        raise ValueError(f"y_hat should be 5-dimensional (Sample, Batch, Timesteps, Width, Height),\
             but is {y_hat.ndim}-dimensional with shape {y_hat.shape}.")
    if sigma.ndim != 5:
        raise ValueError(f"sigma should be 5-dimensional (Sample, Batch, Timesteps, Width, Height),\
             but is {sigma.ndim}-dimensional with shape {sigma.shape}.")

    if method == "gaussian_likelihood":
        if isinstance(y_hat, np.ndarray):
            return _kendall_estimate_npy(y_hat, sigma, n_stds=n_stds)
        elif isinstance(y_hat, torch.Tensor):
            return _kendall_estimate_torch(y_hat, sigma, n_stds=n_stds)
        else:
            raise NotImplementedError(f"Method {method} not implemented for\
                input type {type(y_hat)}.")
    else:
        raise NotImplementedError(f"Method {method} not implemented.")
