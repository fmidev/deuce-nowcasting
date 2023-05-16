"""
Base U-Net components to build from.
Contains:
    - NowcastingUNet(Base components from which to build variants)
    - SingleScaleStructure (Mixin with single-scale implementation methods)
    - MultiScaleStructure (Mixin with multi-scale implementation methods)
    - iterative_unet : Decorator wrapping a U-Net implementation to make iterative
    predictions. Only for SingleScaleStructure childs.

Bent Harnist (FMI) 2022
"""
import torch 
import torch.nn as nn

from .nowcasting_nn import NowcastingNN
from .components.partial_conv2d import PartialConv2d

def _cat_if_list(input):
    "utility for handling late-fusion in the decoder"
    if isinstance(input, torch.Tensor):
        return input
    elif isinstance(input, list):
        return torch.cat(*input, dim=1)
    else: 
        raise ValueError(f"Type of input should be torch.Tensor or list, but is {type(input)}")

class NowcastingUNet(NowcastingNN):
    "Base components for UNet based nowcasting  CNN architectures"

    def __init__(
        self,
        n_input_timesteps: int,
        n_output_timesteps: int,
        n_input_channels : int,
        n_output_channels : int,
        kernel_size : tuple = (3,3),
        dropout_probability : float = 0.5,
        dropout_mode : str = "bridge", # bridge|all
        use_partial_convolution : bool = True,
        ) -> None:
        super().__init__(n_input_timesteps, n_output_timesteps, n_input_channels, n_output_channels)
        
        self.kernel_size = kernel_size


        self.pool = nn.MaxPool2d(kernel_size = (2,2))
        self.upsample = nn.Upsample(scale_factor=(2,2))
        
        self.conv_layer = PartialConv2d if use_partial_convolution else nn.Conv2d
        self.dropout_probability = dropout_probability
        self.drop = (
            nn.Dropout(p=self.dropout_probability)
            if self.dropout_probability > 0.0
            else nn.Identity()
            )
        self.dropout_mode = dropout_mode
        self.dropout_bridge = self.drop if self.dropout_mode == "bridge" else nn.Identity()
        self.dropout_block = self.drop if self.dropout_mode == "block" else nn.Identity()
    
    def make_conv_block(self, in_ch, out_ch, kernel_size):
        "Make a conv2d -> ReLU -> conv2d -> ReLU U-net component"
        return nn.Sequential(
            self.conv_layer(in_ch, out_ch, kernel_size, padding='same'),
            nn.ReLU(),
            self.conv_layer(out_ch, out_ch, kernel_size, padding='same'),
            nn.ReLU(),
            self.dropout_block
            )

    def make_encoder(self, encoder_shape):
        "Make and return a RainNet encoder module"
        encoder_module = nn.ModuleDict()
        
        for name, (in_ch, out_ch) in encoder_shape:
            encoder_module[name] = nn.Sequential(
                self.make_conv_block(in_ch, out_ch, self.kernel_size)
            )
        return encoder_module


    def make_decoder(self, decoder_shape : list):
        "Make and return a RainNet decoder module"
        decoder_module = nn.ModuleDict()
        for name, (in_ch, out_ch) in decoder_shape:
            decoder_module[name] = self.make_conv_block(in_ch,
                                                       out_ch ,self.kernel_size)
        return decoder_module


    def make_output(self, last_shape : dict, last_activation : nn.Module):
        "Make and return a RainNet output module"
        output_module = nn.ModuleDict()
        for name in last_shape.keys():
            output_module[str(name)] = nn.Sequential(
                self.conv_layer(last_shape[name][0][0], last_shape[name][0][1], kernel_size=3, padding='same'),
                self.conv_layer(last_shape[name][1][0], last_shape[name][1][1], kernel_size=1, padding = 'valid'),
                last_activation
            )
        return output_module


class SingleScaleStructure:
    """
    (Mixin) Encoder, decoder definitions for single-scale U-Net 
    """

    def forward_encoder(self, encoder_dict, x):
        "Forward x through an encoder module"
        x1s = encoder_dict["1"](x.float()) # conv1s
        x2s = encoder_dict["2"](self.pool(x1s)) # conv2s
        x3s = encoder_dict["3"](self.pool(x2s)) # conv3s
        x4s = encoder_dict["4"](self.pool(x3s)) # conv4s
        x = encoder_dict["5"](self.pool(self.dropout_bridge(x4s))) # conv5s

        return (x, (x1s, x2s, x3s, x4s))

    def forward_decoder(self, decoder_dict, last_module, x, x1s, x2s, x3s, x4s):
        "Forward x through a decoder module."
        x = torch.cat((self.upsample(self.dropout_bridge(_cat_if_list(x))), _cat_if_list(x4s)), dim=1) # up6
        x = torch.cat((self.upsample(decoder_dict["6"](x)), _cat_if_list(x3s)), dim=1) # up7
        x = torch.cat((self.upsample(decoder_dict["7"](x)), _cat_if_list(x2s)), dim=1) # up8
        x = torch.cat((self.upsample(decoder_dict["8"](x)), _cat_if_list(x1s)), dim=1) # up9
        x = decoder_dict["9"](x) #conv9
        x = last_module(x) #outputs
        return x