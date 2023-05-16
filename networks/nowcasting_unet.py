"""
Implementation of U-Net variants.
Bent Harnist (FMI) 2022
"""

import torch
import torch.nn as nn

from .nowcasting_unet_components import *

class UNet(NowcastingUNet, SingleScaleStructure):
    """
    Single-scale U-Net definition.
    Encompasses:
        - Iterative and non-iterative versions
        - Possible Heteroscedastic modeling of uncertainty as second output channel
    """

    def __init__(
        self,
        n_input_timesteps: int = 4,
        n_output_timesteps: int = 1,
        n_input_channels: int = 1,
        n_output_channels: int = 1,
        kernel_size: tuple = [3,3],
        dropout_probability: float = 0.5,
        dropout_mode : str = "bridge", # bridge|all
        use_partial_convolution : bool = True,
        mode = "regression",
        iterative = True,
        encoder_shape = [
            ["1", [4,64]],
            ["2" , [64,128]],
            ["3" , [128,256]],
            ["4" , [256,512]],
            ["5" , [512,1024]]],
        decoder_shape =
            [["6" , [1536,512]],
            ["7" , [768,256]],
            ["8" , [384,128]],
            ["9" , [192,64]]],
        last_shape = {"last" : [[64,2], [2,1]]}
        ) -> None:
        super().__init__(n_input_timesteps, n_output_timesteps, n_input_channels, n_output_channels, kernel_size, dropout_probability, dropout_mode, use_partial_convolution)
        if mode in ["regression", "heteroscedastic_regression", "heteroscedastic_regression_ez"]:
            self.mode = mode
        else:
            raise NameError(f"mode {mode} unknown")
        self.iterative = iterative
        self.make(encoder_shape, decoder_shape, last_shape)

    def make(self, encoder_shape, decoder_shape, last_shape):
        self.encoder = self.make_encoder(encoder_shape=encoder_shape)
        self.decoder = self.make_decoder(decoder_shape=decoder_shape)
        self.output = self.make_output(last_shape=last_shape, last_activation=nn.Identity())
        if self.mode == "heteroscedastic_regression":
            self.var_decoder = self.make_decoder(decoder_shape=decoder_shape)
            self.var_output = self.make_output(last_shape=last_shape, last_activation=nn.Identity())

    def forward(self, x, n_timesteps : int = 1):
        "complete prediction, e.g. for one hour..."
        # encoding
        x_enc, xns_enc = self.forward_encoder(self.encoder, x)
        #decoding
        if self.mode == "regression":
            x = self.forward_decoder(self.decoder, self.output["last"], x_enc, *xns_enc)
        elif self.mode == "heteroscedastic_regression":
            prediction = self.forward_decoder(self.decoder, self.output["last"], x_enc, *xns_enc)
            variance = self.forward_decoder(self.var_decoder, self.var_output["last"], x_enc, *xns_enc)
            x = torch.stack([prediction, variance], dim=-1)
        elif self.mode == "heteroscedastic_regression_ez":
            x = self.forward_decoder(self.decoder, self.output["last"], x_enc, *xns_enc)
            prediction, variance = torch.split(x,int(x.shape[1]/2),dim=1)
            x = torch.stack([prediction, variance], dim=-1)

        #assert x.shape[1] == n_timesteps, f"n_timesteps is {n_timesteps} but x.shape is {x.shape}!"
        return x
