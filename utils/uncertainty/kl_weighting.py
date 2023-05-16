"""
Strategies for weighting the (KL) complexity term in (pseudo)bayesian NN.
"""
import torch
import pyro.distributions as dist

def kl_weight_blundell(num_batches, batch_idx, **kwargs):
    "From weight uncertainty in neural networks paper by Blundell et al (bayes by backprop paper)"
    try:
        return 2 ** (num_batches - batch_idx - 1) / (2 ** num_batches - 1)
    except: # overflow error countering ...
        return 0

def kl_weight_equal(num_batches, batch_idx, **kwargs):
    "Simpler scheme"
    return 1 / num_batches

def kl_weight_epoch(current_epoch, epochs_at_start, num_batches, batch_idx, **kwargs):
    "Own scheme for small batch sizes, should work best" 
    return (1 / num_batches) * (1.0 / (2.0 ** (current_epoch + epochs_at_start)))

class GaussianScaleMixture(torch.distributions.MixtureSameFamily, dist.torch_distribution.TorchDistributionMixin):
    pass
