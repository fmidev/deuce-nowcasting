import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr


from pincast_verif.metrics import Metric


class DataQuality(Metric):
    """
    Records how much data is missing from the dataset used
    """

    def __init__(self):
        super().__init__()

    def accumulate(self, x_pred, x_obs):
        raise NotImplementedError("DQ accumulate unimplemented as of yet.")

    def compute(self):
        raise NotImplementedError("DQ compute unimplemented as of yet.")

    @staticmethod
    def get_done_df_stats(done_df: pd.DataFrame, i: str) -> list:
        stats = []
        for col in done_df:
            col_mod = i + "/" + col
            if "unnamed" not in col.lower():
                perc = done_df[col].value_counts(normalize=True)
                stats.append([col_mod, 1 - perc[False]])
        return stats
