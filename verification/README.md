# verification
Scripts and tools for computing verification metrics.

- `scripts` contains callable end-user scripts
- `pincast_verif` contains reusable components that are used by above scripts

## How to use this code ? 

1. First and foremost: Clone/add as a submodule the repository and make sure all the dependencies are installed.
2. Install the package locally with running`pip install -e .` in your conda/virtual environment from the repository root folder. 
3. Setup your Pysteps configuration locally as instructed in [the Pysteps documentation](https://pysteps.readthedocs.io/en/stable/user_guide/set_pystepsrc.html).
4. Proceed further now ...

### I want to make predictions.

1. Build your own prediction builder object inheriting `PredictionBuilder` or use one defined in `pincast_verif/prediction_builder_instances` if your use case is covered there.
2. Create/find and use a script calling that builder such as `scripts/run_pysteps_predictions.py` or a jupyter notebook depending on your needs.
3. With a script such as `scripts/run_pysteps_predictions.py`, you will get out an HDF5 file containing predictions usable for metrics calculation. Functions for reading/writing nowcasts in HDF5 archives are found in `pincast_verif.io_tools`.

### I want to calculate metrics.

1. If some metric you want to use is lacking from `pincast_verif/metrics`, take example on the other metrics and add it there, to `pincast_verif/metrics/__init__.py`, and as a case in the `get_metric_class()` function under `pincast_verif/metric_tools.py`.
2. Run `scripts/calculate_metrics.py` with a YAML configuration containing your metric parameters. 
3. Intermediate results are saved as in the Metric objects, and are saved to disk after each prediction is processed *if no parallelization is used*. Otherwise the results are only saved after all predictions are processed.
4. Information on samples calculated and samples with missing data is saved in a "done" pandas dataFrame saved to disk in `.csv` format with the same condition as for metric states.  
5. Final calculated metrics are saved to disk in NetCDF4 format containing relevant values and metadata. These are readable with xarray into `xarray.DataArray` objects.    

### I want to plot metrics. 

- Using the script:
  1. Run `scripts/plot.py` with appropriate YAML configuration and potentially modified to plot your custom metrics, or call the plot method of your metric in some other way, like in a jupyter notebook.
  2. Plots will be saved to disk in specified folders.
- Using Notebooks or your own scripts:
  1. Load a metric for a single prediction method with `xarray.open_dataarray(path)` or a multiple predictions with for example:
  ```python
  metric = [xarray.open_dataarray(path) for p in BASE_DIR.glob("*.nc") if "METRIC_ID" in str(path)]
  metric = xarray.Dataset(data_vars={arr.name : arr for arr in metric}) 
  ```
  2. Freely plot using the `plot` or `plot_multiple` static methods of metric classes, or building your own custom visualization.
