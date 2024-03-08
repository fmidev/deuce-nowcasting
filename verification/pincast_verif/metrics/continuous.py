import numpy as np
from pysteps import verification
import xarray as xr

from pincast_verif.metrics import Metric

class ContinuousMetric(Metric):
    def __init__(self, leadtimes, cont_metrics, tables: dict = None, **kwargs) -> None:
        self.name_template = "CONT_l_{leadtime}"
        self.leadtimes = leadtimes
        self.cont_metrics = cont_metrics
        if tables is None:
            self.tables = {}
            for lt in leadtimes:
                name = self.name_template.format(leadtime=lt)
                self.tables[name] = verification.det_cont_fct_init()
            self.is_empty = True
        else:
            self.tables = tables
            self.is_empty = False

    def accumulate(self, x_pred, x_obs):
        if x_pred.ndim == 4:
            x_pred = x_pred.mean(axis=1)
        for i, lt in enumerate(self.leadtimes):
            name = self.name_template.format(leadtime=lt)
            verification.det_cont_fct_accum(
                err=self.tables[name], pred=x_pred[i], obs=x_obs[i]
            )
        self.is_empty = False

    def compute(self):
        values = []
        for metric in self.cont_metrics:
            metric_values = np.empty(len(self.leadtimes))
            for i, lt in enumerate(self.leadtimes):
                in_name = self.name_template.format(leadtime=lt)
                metric_values[i] = verification.det_cont_fct_compute(
                    self.tables[in_name], scores=metric
                )[metric]
            values.append(metric_values)
        return xr.DataArray(
            data = np.array(values),
            dims=["cont_metric", "leadtime"],
            coords={
            "cont_metric" : self.cont_metrics,
            "leadtime" : self.leadtimes
            },
            attrs={
            "metric" : "CONT"
            }
        )

    def merge(self, continuous_other):
        self.tables = {
            name: verification.det_cont_fct_merge(table, continuous_other.tables[name])
            for name, table in self.tables.items()
        }
