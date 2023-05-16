import logging

import numpy as np
from pysteps.utils.spectral import rapsd
from pysteps.visualization.spectral import plot_spectrum1d
from pysteps.utils.transformation import dB_transform
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr


from pincast_verif.metrics import Metric

# not storing fft method in class makes object picklable
FFT_METHOD = np.fft

class RapsdMetric(Metric):
    
    def __init__(
        self,
        leadtimes: list,
        im_size: tuple = (512, 512),
        return_freq: bool = True,
        normalize: bool = True,
        d: float = 1.0,
        ensemble_mode: str = "rapsd_of_mean", #rapsd_of_mean or mean_of_rapsd
        tables: dict = None,
        **kwargs,
    ) -> None:
        self.leadtimes = leadtimes
        self.name_template = "RAPSD_l_{lts}"
        self.name = self.name_template.format(
            lts="_".join([str(lt * 5) for lt in self.leadtimes])
        )
        self.n_freqs = int(max(im_size) / 2)

        self.normalize = normalize  # does power spectrum sum to one
        self.return_freq = return_freq
        self.d = d  # 1 / sampling rate
        if ensemble_mode in ["rapsd_of_mean", "mean_of_rapsd"]:
            self.ensemble_mode = ensemble_mode
        else:
            raise ValueError(f"Ensemble handling mode not in \
                            ['rapsd_of_mean', 'mean_of_rapsd'],\
                             but is {ensemble_mode} !")

        if tables is None:
            self.tables = {self.name: {}}
            self.tables[self.name].update({"n": 0})
            self.tables[self.name].update(
                {
                    "values": np.zeros((len(self.leadtimes), self.n_freqs)),
                    "obs_values": np.zeros((len(self.leadtimes), self.n_freqs)),
                }
            )
            self.is_empty = True
        else:
            self.tables = tables
            self.is_empty = False


    def accumulate(self, x_pred: np.ndarray, x_obs: np.ndarray) -> dict:
        if not isinstance(x_pred, np.ndarray):
            x_pred = np.array(x_pred)
        assert(np.isclose(np.nanmin(x_pred), np.nanmin(x_obs))), \
            f"min (zerovalue) of pred and obs should be equal,\
             but were {(np.nanmin(x_pred), np.nanmin(x_obs))}"

        x_pred[~np.isfinite(x_pred)] = np.nanmin(x_pred)
        x_obs[~np.isfinite(x_obs)] = np.nanmin(x_obs)

        def _acc(pred, obs):
            "inner accumulate logic"
            for i, lt in enumerate(self.leadtimes):
                result, freq = rapsd(
                    field=pred[lt - 1],
                    fft_method=FFT_METHOD,
                    return_freq=True,
                    normalize=self.normalize,
                    d=self.d,
                )
                obs_result = rapsd(
                    field=obs[lt - 1],
                    fft_method=FFT_METHOD,
                    return_freq=False,
                    normalize=self.normalize,
                    d=self.d,
                )

                if np.isnan(result).any():
                    logging.error(
                        "Some NaN values were found in RAPSD accumulated power spectra. Aborting."
                    )
                    raise ValueError

                self.tables[self.name]["values"][i] += result
                self.tables[self.name]["obs_values"][i] += obs_result
                if self.return_freq and "freq" not in self.tables[self.name]:
                    self.tables[self.name]["freq"] = freq
            self.tables[self.name]["n"] += 1

        if x_pred.ndim == 3:
            _acc(x_pred,x_obs)
        elif x_pred.ndim == 4 and self.ensemble_mode == "rapsd_of_mean":
            _acc(x_pred.mean(axis=1),x_obs)
        elif x_pred.ndim == 4 and self.ensemble_mode == "mean_of_rapsd":
            for i in range(x_pred.shape[1]):
                _acc(x_pred[:,i],x_obs)

        self.is_empty = False


    def compute(self) -> np.ndarray:
        # Forecast
        values = self.tables[self.name]["values"] / self.tables[self.name]["n"]
        # Observations
        obs_values = self.tables[self.name]["obs_values"] / self.tables[self.name]["n"]
        if self.return_freq:
            freq = self.tables[self.name]["freq"]
        else:
            freq = range(self.n_freqs)

        return xr.DataArray(
            data=np.stack([values,obs_values],axis=0),
            dims=["type","leadtime","freq"],
            coords={
            "leadtime" : self.leadtimes,
            "type" : ["prediction", "observation"],
            "freq" : freq
            },
            attrs={"metric" : "RAPSD"}
        )


    def _merge_tables(self, table_self, table_other):
        return {
            key: (
                table_self[key] + table_other[key] if key != "freq" else table_self[key]
            )
            for key in table_self.keys()
        }

    def merge(self, other_rapsd):
        self.tables = {
            self.name: self._merge_tables(
                self.tables[self.name], other_rapsd.tables[other_rapsd.name]
            )
        }
        if (
            self.return_freq
            and "freq" not in self.tables[self.name]
            and "freq" in other_rapsd.tables[other_rapsd.name]
        ):
            self.tables[self.name]["freq"] = other_rapsd.tables[other_rapsd.name][
                "freq"
            ]
