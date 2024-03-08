import numpy as np
from pysteps import verification
import xarray as xr


from pincast_verif.metrics import Metric


class FssMetric(Metric):
    def __init__(
        self, leadtimes, thresh, scales, tables: dict = None, **kwargs
    ) -> None:
        self.name_template = "FSS_s_{scale}_t_{thresh}_l_{leadtime}"
        self.leadtimes = leadtimes
        self.thresholds = thresh
        self.scales = scales
        if tables is None:
            self.tables = {}
            for lt in leadtimes:
                for thr in thresh:
                    for scale in scales:
                        name = self.name_template.format(
                            leadtime=lt, thresh=thr, scale=scale
                        )
                        self.tables[name] = verification.spatialscores.fss_init(
                            thr=thr, scale=scale
                        )
            self.is_empty = True
        else:
            self.tables = tables
            self.is_empty = False

    def accumulate(self, x_pred, x_obs):
        if x_pred.ndim == 4:
            x_pred = x_pred.mean(axis=1)
        for i, lt in enumerate(self.leadtimes):
            for thr in self.thresholds:
                for scale in self.scales:
                    name = self.name_template.format(
                        leadtime=lt, thresh=thr, scale=scale
                    )
                    verification.spatialscores.fss_accum(
                        fss=self.tables[name], X_f=x_pred[i], X_o=x_obs[i]
                    )
        self.is_empty = False

    def compute(self):
        values = np.empty((len(self.scales),len(self.thresholds),len(self.leadtimes)))
        for i,scl in enumerate(self.scales):
            for j,thr in enumerate(self.thresholds):
                for k, lt in enumerate(self.leadtimes):
                    in_name = self.name_template.format(
                        scale=scl, thresh=thr, leadtime=lt
                    )
                    values[i,j,k] = verification.fss_compute(self.tables[in_name])
        return xr.DataArray(
            data=values,
            dims=["scale", "threshold", "leadtime"],
            coords={
            "scale": self.scales,
            "threshold": self.thresholds,
            "leadtime": self.leadtimes
            },
            attrs={
            "metric" : "FSS"
            }
        )

    def merge(self, fss_other):
        self.tables = {
            name: verification.spatialscores.fss_merge(table, fss_other.tables[name])
            for name, table in self.tables.items()
        }
