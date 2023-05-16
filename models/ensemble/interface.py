"""
Interface for ready-to-use ensemble nowcasting models
Bent Harnist (FMI) 2022
"""
from models import EnsembleVIModel

class SingleScaleBCNN(EnsembleVIModel):
    "Bayesian CNN model without scale decomposition"
    pass
