"""
Match callback name to Pytorch Lightning Callback object
"""

from callbacks import *
from pytorch_lightning.callbacks import *

def match_callback(callback_config : dict):
    if callback_config.name == "log_probabilistic_nowcast":
        return LogProbabilisticNowcast(callback_config.kwargs)
    elif callback_config.name == "nowcast_metrics":
        return NowcastMetrics(callback_config.kwargs)
    elif callback_config.name == "early_stopping":
        return EarlyStopping(**callback_config.kwargs)
    elif callback_config.name == "model_checkpoint":
        return ModelCheckpoint(**callback_config.kwargs)
    elif callback_config.name == "learning_rate_monitor":
        return LearningRateMonitor(**callback_config.kwargs)
    elif callback_config.name == "device_stats_monitor":
        return DeviceStatsMonitor(**callback_config.kwargs)
    else:
        raise NotImplementedError(f"Callback name {callback_config.name} undefined.")
