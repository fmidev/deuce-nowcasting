from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from addict import Dict
from pysteps import verification
from pysteps.postprocessing.ensemblestats import excprob

from pincast_verif.metrics import Metric
from pincast_verif.misc import enumerated_product



class ROCMetric(Metric):
    "ROC forecast metrics (for ensemble based forecasts)"

    def __init__(
        self,
        leadtimes: list,
        thresholds: list,
        roc_kwargs: dict,
        tables: dict = None,
        **kwargs
    ):
        super().__init__()
        self.leadtimes = leadtimes
        self.thresholds = thresholds
        self.roc_kwargs = roc_kwargs

        if tables is None:
            self.tables = Dict()
            for lt, thr in product(self.leadtimes, self.thresholds):
                self.tables[lt][thr] = verification.ROC_curve_init(
                    X_min=thr,
                    **self.roc_kwargs
                )
                self.prob_thrs = self.tables[lt][thr]["prob_thrs"]
            self.is_empty = True
        else:
            assert isinstance(tables, Dict)
            self.tables = tables
            self.is_empty = False

    def accumulate(self, x_pred, x_obs) -> None:
        if x_pred.ndim != 4:
            raise ValueError(
                f"Prediction array must be 4-dimensional with (T,S,W,H), but instead is of shape {x_pred.shape}"
            )
        probs = np.stack(excprob(X=x, X_thr=self.thresholds) for x in x_pred)
        for (i,j),(lt, thr) in enumerated_product(self.leadtimes, self.thresholds):
            verification.ROC_curve_accum(self.tables[lt][thr], P_f=probs[i, j], X_o=x_obs[i])
        self.is_empty = False

    def compute(self):
        n_prob_thrs = self.roc_kwargs.get("n_prob_thrs", 10)
        values = np.zeros((len(self.leadtimes), len(self.thresholds), 3, n_prob_thrs))
        for (i,j),(lt, thr) in enumerated_product(self.leadtimes, self.thresholds):
            POFD, POD, area = verification.ROC_curve_compute(self.tables[lt][thr], compute_area=True)
            values[i,j] = np.stack([POD, POFD, np.ones(n_prob_thrs)*area],axis=0)

        return xr.DataArray(
            data = np.array(values),
            dims=["leadtime", "threshold", "quantity", "prob_thrs"],
            coords={
            "threshold": self.thresholds,
            "leadtime": self.leadtimes,
            "quantity": ["POD","POFD","AUC"],
            "prob_thrs": self.prob_thrs,
            },
            attrs={
            "metric" : "ROC",
            }
        )


    @staticmethod
    def merge_tables(table_self, table_other):
        "ROC, Reliability diag. tables do not have a built-in merge function either.. :("
        return {
            key: (
                table_self[key] + table_other[key]
                if key
                in [
                    "hits",
                    "misses",
                    "false_alarms",
                    "corr_neg",
                ]
                else table_self[key]
            )  # prob_thrs, X_min, bin_edges, n_bins, min_count
            for key in table_self.keys()
        }

    def merge(self, roc_other):
        for lt, thr in product(self.leadtimes, self.thresholds):
                self.tables[lt][thr] = ROCMetric.merge_tables(
                    self.tables[lt][thr],
                    roc_other.tables[lt][thr]
                )
