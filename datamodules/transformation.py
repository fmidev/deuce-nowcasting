"""
Data Augmentation for precipitation nowcasting
Bent Harnist (FMI) 11.11.2022
Version 0.1
"""

import random
import torch
import torchvision.transforms as tf

class ChoiceRotationTransformation:
    """Rotate by one of the given angles."""

    def __init__(self,angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return tf.functional.rotate(x, angle)

class NowcastingTransformation:
    """
    Compose all needed transformations for augmenting radar images
    used for precipitation nowcasting.
    """

    def __init__(self, transformation_cfg):
        self.transformation_cfg = transformation_cfg 
        self.transform = tf.Compose([
            self._match_transformation(name, kwargs)
            for name, kwargs in self.transformation_cfg.items()
        ])

    def _match_transformation(self, name : str, kwargs : dict):
        if name == "rotate":
            return ChoiceRotationTransformation(**kwargs)
        elif name == "horizontal_flip":
            return tf.RandomHorizontalFlip(**kwargs)
        elif name == "vertical_flip":
            return tf.RandomVerticalFlip(**kwargs)
        elif name == "random_crop":
            return tf.RandomCrop(**kwargs)

    def __call__(self, batch):
        batch_input_size = batch["inputs"].size()
        x = batch["inputs"]
        y = batch["outputs"]
        if len(batch_input_size) == 5 and batch_input_size[2] != 1 and "rotate" in self.transformation_cfg:
            raise NotImplementedError("Nowcasting transformation not implemented for 3D data yet,")
        elif len(batch_input_size) == 5 and "rotate" in self.transformation_cfg:
            x = x[:,:,0]
            y = y[:,:,0]
        transformed = self.transform(torch.cat([x,y], dim=1))
        batch_transformed = batch.copy()
        if len(batch_input_size) == 5:
            transformed = transformed[:,:,None,...]
        batch_transformed["inputs"] = transformed[:,:x.shape[1]]
        batch_transformed["outputs"] = transformed[:,x.shape[1]:]
        return batch_transformed
