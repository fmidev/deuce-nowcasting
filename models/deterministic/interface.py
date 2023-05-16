"""
Interface for ready-to-use deterministic nowcasting models
Bent Harnist (FMI) 2022
"""
from models import DeterministicModel

class SingleScaleUNet(DeterministicModel):
    "UNet/RainNet implementations by varying parameters"