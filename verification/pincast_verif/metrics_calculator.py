import os
import shutil
import logging
import pickle

from addict import Dict
import numpy as np
import pandas as pd
import dask

from pincast_verif import metric_tools
from pincast_verif import io_tools
from pincast_verif import conversion_tools
from pincast_verif.masking import mask


class MetricsCalculator(object):
    """
    Class providing an interface for metrics calculation for nowcasts.

    Implicit assumptions:
        - Metric calculated for possibly multiple nowcast methods
        - Metric depends possibly on observation too
        - Parameters (such as thresholds) are stored inside metrics
        - !!! Metric states are saved as dictionaries of np.ndarray!!!
            (no support for pd df, xarray)
        - computed metrics return np.ndarray, names as list
        - name should allow retrieval of metric part relevant
            (but this is not always the case, as with RAPSD)


    Methods :
    # to call at the start of the script #
    - __init__()
        calls -> _init_contincency_tables()
        calls -> _init_done_df()
    - get_samples_left()

    # to call inside the main calculation loop #
    - accumulate()
    - save_done_df()
    - save_contingency_tables()

    # to call at the end of the script, once metrics are accumulated #
    - compute()
    - save_metrics()

    A usage example is given in scripts/calculate_metrics.py
    """

    def __init__(
        self,
        config: Dict,
        config_path: str = None,
        logger=None,
    ) -> None:
        if logger is None:
            self.logger = logging.getLogger("metrics")
        else:
            self.logger = logger

        self.id = config.exp_id

        # init paths
        self.path = config.path

        # init config
        self.methods = config.methods
        self.measurements = config.measurements
        self.n_leadtimes = config.n_leadtimes
        self.leadtimes = range(1, self.n_leadtimes + 1)
        self.common_mask = config.common_mask
        self.metrics = config.metrics
        self.n_chunks = config.n_chunks
        self.n_workers = config.n_workers

        # preprocessing config
        self.data_convert_mmh = config.preprocessing.convert_mmh
        self.data_threshold = config.preprocessing.threshold
        self.data_zerovalue = config.preprocessing.zerovalue

        # init working lists and dicts
        self.metrics_df, self.has_loaded_metrics = self._init_metric_states(
            attempt_data_load=True
        )
        self.timestamps = io_tools.read_timestamp_txt_file(
            self.path.timestamps, start_idx=1
        )
        self.done_df = self._init_done_df()

        self.logger.info(f"Initializing Metrics calculator")
        self.logger.info(f"Metrics: {self.metrics.keys()}")
        self.logger.info(f"Prediction methods: {self.methods.keys()}")

        if config.debugging:
            import random

            random.seed(12345)
            self.samples_left = random.sample(
                population=self.timestamps, k=config.debugging
            )
        else:
            self.samples_left = self.get_samples_left()

        if self.n_chunks > 0:
            self.chunks = np.array_split(np.array(self.samples_left), self.n_chunks)

        # init directory structure
        self.logger.info("Initializing directory structure")
        for path in self.path:
            for method in self.methods:
                for metric in self.metrics:
                    path_formatted = self.path[path].format(
                        id=self.id, method=method, metric=metric
                    )
                    os.makedirs(os.path.dirname(path_formatted), exist_ok=True)
                    self.logger.debug(f"Initialized path to: {path_formatted}")

        # copy config to results folder if source path specified
        if config_path is not None:
            shutil.copyfile(
                src=config_path, dst=self.path.config_copy.format(id=self.id)
            )

    def _init_metric_states(self, attempt_data_load: bool) -> dict:
        has_loaded_data = False
        metric_data = pd.DataFrame(index=self.methods.keys(), columns=self.metrics.keys())
        for method in self.methods:
            for metric in self.metrics:
                state_path = self.path.states.format(method=method, metric=metric, id=self.id)
                if os.path.exists(state_path) and attempt_data_load:
                    has_loaded_data = True
                    self.logger.info(
                        "existing output file found, loading {}".format(state_path)
                    )
                    with open(state_path, 'rb') as f:
                        metric_data[metric][method] = pickle.load(f)
                else:
                    metric_data[metric][method] = metric_tools.get_metric(
                        metric_name=metric,
                        metric_params=self.metrics[metric]["init_kwargs"]
                    )
        return metric_data, has_loaded_data

    def save_metric_states(self) -> None:
        for metric in self.metrics:
            for method in self.methods:
                out_path = self.path.states.format(method=method, metric=metric, id=self.id)
                with open(out_path, 'wb') as f:
                    pickle.dump(self.metrics_df[metric][method], f, protocol=pickle.HIGHEST_PROTOCOL)

    def _init_done_df(self) -> pd.DataFrame:
        self.logger.info("Initializing done df")

        if os.path.exists(self.path.done.format(id=self.id)) and self.has_loaded_metrics:
            done = pd.read_csv(
                self.path.done.format(id=self.id), sep=",", header=0, index_col=0
            )
            for method in self.methods:
                if method not in done:
                    done[method] = False
            if "bad" not in done:
                raise ValueError("bad column not found in existing CSV, aborting")
        else:
            done = pd.DataFrame(
                index=self.timestamps, columns=self.methods.keys(), dtype=bool, data=False
            )
            done["bad"] = False
        return done

    def save_done_df(self) -> None:
        self.logger.info(f"Saving done df to {self.path.done.format(id=self.id)}")
        self.done_df.to_csv(self.path.done.format(id=self.id))

    def get_samples_left(self) -> list:
        indices_set = set(self.timestamps)
        existing_set = set(
            [
                self.timestamps[i]
                for i in range(len(self.timestamps))
                if all(self.done_df.iloc[i])
            ]
        )
        samples_left = list(indices_set - existing_set)
        return samples_left

    def accumulate(self, sample, done_df, metrics_df):
        self.logger.debug("\n Sample {} ongoing \n Loading data".format(sample))

        observations = io_tools.load_observations(
            db_path=self.measurements.path,
            time_0=sample,
            leadtimes=self.leadtimes,
            method_name=self.measurements.name,
        )
        if observations is None:
            done_df.loc[sample]["bad"] = True
            self.logger.debug(
                f"Sample containing missing observation found, skipping sample {sample}"
            )
            return done_df, metrics_df

        if self.data_convert_mmh:
            observations = conversion_tools.dbz_to_rainrate(observations)
        observations[observations < self.data_threshold] = self.data_zerovalue

        preds = io_tools.load_predictions_dict(
            self.methods, time_0=sample, leadtimes=self.leadtimes
        )
        if isinstance(preds, str):
            done_df.loc[sample]["bad"] = True
            self.logger.debug(
                f"Sample containing missing prediction {preds} found, skipping sample {sample}"
            )
            return done_df, metrics_df

        if self.common_mask:
            preds = mask(predictions=preds, n_leadtimes=self.n_leadtimes)

        for method in self.methods:
            self.logger.debug(
                "Accumulating metrics for {method} predictions".format(method=method)
            )
            if self.data_convert_mmh:
                pred = conversion_tools.dbz_to_rainrate(preds[method])
            else:
                pred = preds[method]
            pred[pred < self.data_threshold] = self.data_zerovalue

            for metric in self.metrics:
                self.logger.debug(f"Metric {metric} ongoing...")
                metrics_df[metric][method].accumulate(x_pred=pred, x_obs=observations)
            done_df.loc[sample][method] = True

        del observations, preds

        return done_df, metrics_df

    def update_state(self, done_df, metrics_df):
        self.done_df = done_df
        self.metrics_df = metrics_df

    def accumulate_chunk(self, chunk_index):
        _done_df, _metrics_df = (
            self._init_done_df(),
            self._init_metric_states(attempt_data_load=False)[0],
        )
        for sample in self.chunks[chunk_index]:
            _done_df, _metrics_df = self.accumulate(
                sample=sample, done_df=_done_df, metrics_df=_metrics_df
            )
        self.logger.info(f"Done with chunk {chunk_index}")
        return _done_df, _metrics_df

    def merge_done_dfs(self):
        partial_done_dfs = [data[0] for data in self.chunked_data]
        if self.has_loaded_metrics:
            partial_done_dfs.insert(0, self.done_df)
        self.done_df = metric_tools.merge_boolean_df_list(partial_done_dfs)

    def merge_metrics_dfs(self):
        partial_metrics_dfs = [data[1] for data in self.chunked_data]
        if self.has_loaded_metrics:
            partial_metrics_dfs.insert(0, self.metrics_df)
        self.metrics_df = metric_tools.merge_metrics_df_list(partial_metrics_dfs)

    def compute(self) -> dict:
        self.logger.info("Computing metrics from tables...")
        computed_metrics = self.metrics_df.applymap(lambda x : x.compute())
        for metric in self.metrics:
            for method in self.methods:
                computed_metrics[metric][method] = computed_metrics[metric][method].assign_attrs(
                    {"method" : method}
                )
                computed_metrics[metric][method] = computed_metrics[metric][method].rename(method)
        return computed_metrics

    def save_metrics(self, computed_metrics_df) -> None:
        self.logger.info("Saving computed metrics to the file system.")
        """
        for method, metric_i_dict in metrics_dict.items():
            for metric_i, (value, name) in metric_i_dict.items():
                npy_path = self.path.metrics.format(
                    id=self.id, method=method, metric=metric_i
                )
                name_path = self.path.name.format(
                    id=self.id, method=method, metric=metric_i
                )
                np.save(file=npy_path, arr=value)
                with open(name_path, "w") as name_file:
                    for n in name:
                        name_file.write(n + "\n")
        """
        for metric in self.metrics:
            for method in self.methods:
                metrics_save_path = self.path.metrics.format(
                    id=self.id, method=method, metric=metric
                )
                computed_metrics_df[metric][method].to_netcdf(metrics_save_path)

    def parallel_accumulation(self):
        res = []
        for chunk_idx in range(self.n_chunks):
            # y = self.accumulate_chunk(chunk_index=chunk_idx)
            y = dask.delayed(self.accumulate_chunk)(chunk_index=chunk_idx)
            res.append(y)

        scheduler = "processes" if self.n_workers > 1 else "single-threaded"
        self.logger.info(
            f"Starting metric accumulation with {scheduler} DASK scheduler."
        )
        self.chunked_data = dask.compute(
            *res, num_workers=self.n_workers, scheduler=scheduler, traverse=False
        )
        self.merge_metrics_dfs()
        self.merge_done_dfs()
