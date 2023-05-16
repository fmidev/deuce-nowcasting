"""
Match optimizer name to Pytorch Optimizer class,
partially initiallizing it with given kwargs, and Learning rate scheduler
to given pytorch implementation.
Bent Harnist (FMI) 2022
"""
from torch.optim import *
from functools import partialmethod


def partial_instance(cls,**kwds):

    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, **kwds)

    return NewCls

def match_optimizer(optim_config):

    if optim_config.name == "adam":
        return partial_instance(Adam, **optim_config.kwargs)
    else:
        raise NotImplementedError(f"Given optimizer name {optim_config.name} not available (yet?)")

def match_lr_scheduler(lr_scheduler_config):
    if lr_scheduler_config.name == "reduce_lr_on_plateau":
        return partial_instance(lr_scheduler.ReduceLROnPlateau, **lr_scheduler_config.kwargs)
    else:
        raise NotImplementedError(f"Given LR scheduler name {lr_scheduler_config.name} not available (yet?)")