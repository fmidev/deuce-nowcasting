"""
Base class for all neural networks used for precipitation nowcasting.
Bent Harnist (FMI) 2022
"""
import torch 
import torch.nn as nn

class NowcastingNN(nn.Module):

    def __init__(
        self,
        n_input_timesteps : int,
        n_output_timesteps : int,
        n_input_channels : int,
        n_output_channels : int,
        ) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_input_timesteps = n_input_timesteps
        self.n_output_timesteps = n_output_timesteps
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels

    

