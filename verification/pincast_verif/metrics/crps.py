import numpy as np
from pysteps import verification
import xarray as xr

from pincast_verif.metrics import Metric


class CrpsMetric(Metric):
    def __init__(self, leadtimes, tables: dict = None, **kwargs) -> None:
        self.name_template = "CRPS_l_{leadtime}"
        self.leadtimes = leadtimes
        if tables is None:
            self.tables = {}
            for lt in leadtimes:
                name = self.name_template.format(leadtime=lt)
                self.tables[name] = verification.probscores.CRPS_init()
            self.is_empty = True
        else:
            self.tables = tables
            self.is_empty = False

    def accumulate(self, x_pred, x_obs):
        if x_pred.ndim != 4:
            raise ValueError(
                f"Prediction array must be 4-dimensional with (T,S,W,H), but instead is of shape {x_pred.shape}"
            )
        for i, lt in enumerate(self.leadtimes):
            name = self.name_template.format(leadtime=lt)
            verification.probscores.CRPS_accum(
                CRPS=self.tables[name], X_f=x_pred[i], X_o=x_obs[i]
            )
        self.is_empty = False

    def compute(self):
        values = []
        metric_values = np.empty(len(self.leadtimes))
        for i, lt in enumerate(self.leadtimes):
            in_name = self.name_template.format(leadtime=lt)
            metric_values[i] = verification.probscores.CRPS_compute(
                self.tables[in_name]
            )
        values.append(metric_values)
        return xr.DataArray(
            data = np.array(values).squeeze(),
            dims=["leadtime"],
            coords={
            "leadtime" : self.leadtimes
            },
            attrs={
            "metric" : "CRPS"
            }
        )

    @staticmethod
    def merge_tables(table_self, table_other):
        "CRPS tables do not have a merge function, but contain only 'n' and 'sum' fields."
        return {key: table_self[key] + table_other[key] for key in table_self.keys()}

    def merge(self, crps_other):
        self.tables = {
            name: CrpsMetric.merge_tables(table, crps_other.tables[name])
            for name, table in self.tables.items()
        }
