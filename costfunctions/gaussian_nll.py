"""
Wrapper around Pytorch's Gaussian Negative log likelihood
with the beta-NLL extension (https://arxiv.org/pdf/2203.09168.pdf)
"""

import torch
import torch.nn	as nn

class GaussianNLL(nn.Module):

    def	__init__(
        self,
        mode : str = "homoscedastic",
        learnable : bool = False,
        sigma : float = None,
        beta : float = 0.0, #0.0 is pure NLL loss
        loss_kwargs : dict = {},
        ):

        super(GaussianNLL, self).__init__()
        self.learnable = learnable
        self.mode = mode
        
        if learnable and mode == "homoscedastic":
            raise NotImplementedError("Learnable homoscedastic likelihood uncertainty unimplemented yet")
            
        if not learnable:
            assert sigma is not None
            self.var = sigma ** 2
        self.loss_function = nn.GaussianNLLLoss(reduction='none', **loss_kwargs)
        assert (beta >= 0.0) and (beta <= 1.0), "beta should take values between zero and one"
        self.beta = beta
        
    def forward(self, batch, target, var = None, beta = None):
        input = batch["outputs"]
        weights = batch["weights"]
        assert input.shape == target.shape

        # to allow potentially dynamically modifying beta
        if beta is None:
            beta = self.beta

        if self.learnable:
            assert var is not None
            if self.mode == "homoscedastic":
                if isinstance(var, float):
                    var = torch.as_tensor(var)
                assert var.ndim == 0
                var = torch.ones_like(input) * var
            else: 
                assert var.shape == input.shape
        else:
            var = torch.ones_like(input) * self.var
            
        loss = (var.detach() ** beta) * self.loss_function(input, target, var)

        if weights is None:
            return loss.mean()
        else:
            return (weights * loss).sum()

