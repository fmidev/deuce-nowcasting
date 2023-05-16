from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from addict import Dict
from pysteps import verification
from pysteps.postprocessing.ensemblestats import excprob

from pincast_verif.metrics import Metric
from pincast_verif.misc import enumerated_product

def expected_calibration_error(rd: xr.DataArray):
    return sum(
        [
        rd.sel(bins=bin, quantity="sample_size") * 
        np.abs(
            rd.sel(bins=bin, quantity="forecast_probability") - 
            rd.sel(bins=bin, quantity="observed_relative_frequency")
        )
        for bin in rd.bins
        ]
    ) / sum(rd.sel(quantity="sample_size"))

class ReliabilityDiagramMetric(Metric):
    "ROC forecast metrics (for ensemble based forecasts)"

    def __init__(
        self,
        leadtimes: list,
        thresholds: list,
        reldiag_kwargs: dict,
        tables: dict = None,
        **kwargs
        ):
        super().__init__()
        self.leadtimes = leadtimes
        self.thresholds = thresholds
        self.reldiag_kwargs = reldiag_kwargs

        if tables is None:
            self.tables = Dict()
            for lt, thr in product(self.leadtimes, self.thresholds):
                self.tables[lt][thr] = verification.reldiag_init(
                    X_min=thr,
                    **self.reldiag_kwargs
                )
                self.bin_edges = self.tables[lt][thr]["bin_edges"]
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
            verification.reldiag_accum(self.tables[lt][thr], P_f=probs[i, j], X_o=x_obs[i])
        self.is_empty = False

    def compute(self):
        n_bins = self.reldiag_kwargs.get("n_bins", 10)
        values = np.zeros((len(self.leadtimes), len(self.thresholds), 3, n_bins))
        for (i,j),(lt, thr) in enumerated_product(self.leadtimes, self.thresholds):
            r, f = verification.reldiag_compute(self.tables[lt][thr])
            values[i,j] = np.stack([r, f, self.tables[lt][thr]["sample_size"]],axis=0)

        return xr.DataArray(
            data = np.array(values),
            dims=["leadtime", "threshold", "quantity", "bins"],
            coords={
            "threshold": self.thresholds,
            "leadtime": self.leadtimes,
            "quantity": [
                "forecast_probability",
                "observed_relative_frequency",
                "sample_size"
                ],
            "bins": range(n_bins),
            },
            attrs={
            "metric" : "ROC",
            "bin_edges": self.bin_edges
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
                    "X_sum",
                    "Y_sum",
                    "num_idx",
                    "sample_size",
                ]
                else table_self[key]
            )  # prob_thrs, X_min, bin_edges, n_bins, min_count
            for key in table_self.keys()
        }

    def merge(self, roc_other):
        for lt, thr in product(self.leadtimes, self.thresholds):
                self.tables[lt][thr] = ReliabilityDiagramMetric.merge_tables(
                    self.tables[lt][thr],
                    roc_other.tables[lt][thr]
                )
