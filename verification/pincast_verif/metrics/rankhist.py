import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray as xr
from pysteps import verification


from pincast_verif.metrics import Metric
from pincast_verif.plot_tools import plot_1d


class RankHistogramMetric(Metric):
    "Rank Histogram skill metric for ensemble forecasts"

    def __init__(
        self,
        leadtimes: list,
        num_ens_member: int,
        X_min: float = None,
        tables: dict = None,
    ):
        self.name_template = "rankhist_l_{leadtime}"
        self.leadtimes = leadtimes
        self.num_ens_member = num_ens_member

        if tables is None:
            self.tables = {}
            for lt in leadtimes:
                name = self.name_template.format(leadtime=lt)
                self.tables[name] = verification.rankhist_init(
                    num_ens_members=self.num_ens_member, X_min=X_min
                )
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
            verification.rankhist_accum(
                rankhist=self.tables[name], X_f=x_pred[i], X_o=x_obs[i]
            )
        self.is_empty = False

    def compute(self):
        values = []
        for lt in self.leadtimes:
            name = self.name_template.format(leadtime=lt)
            values.append(verification.rankhist_compute(self.tables[name]))
        return xr.DataArray(
            data = np.array(values),
            dims=["leadtime", "rank"],
            coords={
            "leadtime" : self.leadtimes,
            "rank": range(self.num_ens_member + 1)
            },
            attrs={
            "metric" : "RANK_HISTOGRAM"
            }
        )

    @staticmethod
    def merge_tables(table_self, table_other):
        "Rank histogram tables do not have a merge function"
        return {
            key: (
                table_self[key] + table_other[key] if key == "n" else table_self[key]
            )  # num_ens_member, X_min
            for key in table_self.keys()
        }

    def merge(self, rankhist_other):
        self.tables = {
            name: RankHistogramMetric.merge_tables(table, rankhist_other.tables[name])
            for name, table in self.tables.items()
        }
