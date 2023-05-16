"""
Utility function to plot aleatoric and epistemic
uncertainties of probabilistic nowcasts.
"""
import matplotlib as mpl
mpl.use("agg")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import utils.radar_image_plots

def uncertainty_plot(
    ground_truth,
    output_mean, 
    output_aleatoric, 
    output_epistemic,
    time_list : list, 
    write_fig : bool = False,
    quantity : str = "RR",
    fig_filename : str = "prediction_uncertainty.png",
    cmaps : tuple = ("viridis", "plasma"),
    plot_kwargs : dict = {},
    gamma : float = 0.35,
    plot_aleatoric = True
    ) : 
    """
    Assuming B,T,W,H inputs, So no channels (yet).
    """
    (cmap_mean, norm_mean), (cmap_var, norm_var) = get_colors(
        quantity=quantity,
        cmaps=cmaps, 
        get_variance=True,
        gamma=gamma
    )
    n_timesteps = output_mean.shape[1]
    n_cols=4 if plot_aleatoric else 3
    fig, axs = plt.subplots(nrows=n_timesteps, ncols=n_cols, figsize=(5*n_cols, 5*n_timesteps))
    if n_timesteps == 1:
        axs = axs[np.newaxis, ...]
    for t in range(n_timesteps):
        if t == 0:
            axs[t][0].set_title("Observation", fontsize=20)
            axs[t][1].set_title("Prediction Mean", fontsize=20)
            axs[t][2].set_title("Predictive \n Uncertainty", fontsize=20)
            if plot_aleatoric:
                axs[t][2].set_title("Epistemic \n Uncertainty", fontsize=20)
                axs[t][3].set_title("Aleatoric \n Uncertainty", fontsize=20)

        if time_list is not None:
            axs[t][0].set_ylabel(str(time_list[t]), fontsize=15)
        else:
            axs[t][0].set_ylabel(str(t), fontsize=15)

        axs[t][0].imshow(ground_truth[0,t], cmap=cmap_mean, norm=norm_mean, **plot_kwargs)
        im = axs[t][1].imshow(output_mean[0,t], cmap=cmap_mean, norm=norm_mean, **plot_kwargs)
        im2 = axs[t][2].imshow(output_epistemic[0,t], cmap=cmap_var, norm=norm_var, **plot_kwargs)
        if plot_aleatoric:
            axs[t][3].imshow(output_aleatoric[0,t], cmap=cmap_var, norm=norm_var, **plot_kwargs)

    if plot_aleatoric:
        plt.subplots_adjust(
            left=0.125,
            bottom=0.2, 
            right=0.8, 
            top=0.7, 
            wspace=0.2, 
            hspace=0.25)

    add_colorbar_outside(im,axs=axs[n_timesteps-1][:2], height=0.03/n_timesteps, eps=0.12/n_timesteps)
    add_colorbar_outside(im2,axs=axs[n_timesteps-1][2:], height=0.03/n_timesteps, eps=0.12/n_timesteps)

    if write_fig:
        fig.savefig(fig_filename)
    return fig

def input_plot(
    inputs,
    time_list : list, 
    quantity : str,
    cmap : str = "viridis",
    plot_kwargs : dict = {},
    gamma : float = 0.35
    ) : 
    """
    Assuming B,T,W,H inputs, So no channels (yet).
    """
    cmap, norm = get_colors(quantity=quantity, cmaps=cmap, get_variance=False, gamma=gamma)

    n_timesteps = inputs.shape[1]
    fig, axs = plt.subplots(ncols=n_timesteps, figsize=(5*n_timesteps, 5))

    for t in range(n_timesteps):
        im = axs[t].imshow(inputs[0,t],cmap=cmap, norm=norm, **plot_kwargs)
        axs[t].set_title(str(time_list[t]), fontsize=15)

    add_colorbar_outside(im,axs=[axs[0], axs[n_timesteps-1]], eps=0.15, height=0.03)        
    return fig

def get_colors(
    quantity : str, 
    cmaps : tuple,
    get_variance : bool, 
    gamma = 0.35
    ):
    if isinstance(cmaps, str):
        cmap_mean, cmap_var = cmaps, None
    elif isinstance(cmaps, tuple) and len(cmaps) == 2:
        cmap_mean, cmap_var = cmaps
    else:
        raise NotImplementedError(f"cmaps type must be str or tuple of length 2,\
             but is {type(cmaps)} of length {len(cmaps)}")

    if quantity == "DBZH":
        norm_var = mpl.colors.Normalize(vmin=0, vmax=50)
        if cmap_mean is None:
            cmap_mean, norm_mean = utils.radar_image_plots.get_colormap(quantity="DBZH")
        else:
            norm_mean = mpl.colors.Normalize(vmin=-15, vmax=60)
    elif quantity == "RR":
        norm_mean = mpl.colors.PowerNorm(gamma=gamma,vmin=0,vmax=100)
        norm_var = mpl.colors.Normalize(vmin=0,vmax=30)
    elif quantity == "logRR":
        norm_var = mpl.colors.Normalize(vmin=0, vmax=2.5)
        norm_mean = mpl.colors.Normalize(vmin=-3.5, vmax=3.5)
    elif quantity == "logunitRR":
        norm_var = mpl.colors.Normalize(vmin=0, vmax=0.3)
        norm_mean = mpl.colors.Normalize(vmin=0, vmax=1)
    else:
        raise NotImplementedError()

    if get_variance:
        return (
        (cmap_mean, norm_mean),
        (cmap_var, norm_var)
    )
    else:
        return (
            cmap_mean, norm_mean
        )

def add_colorbar_outside(im,axs, eps=0.04, height=0.01):
    if len(axs) == 1:
        fig = axs[0].get_figure()
        bbox0 = axs[0].get_position()
        height = height
        eps =eps #margin between plot and colorbar
        # [left most position, bottom position, width, height] of color bar.
        cax = fig.add_axes([bbox0.x0, bbox0.y0 - eps, (bbox0.x1 - bbox0.x0), height])
        fig.colorbar(im, cax=cax, orientation="horizontal")
    else:
        fig = axs[0].get_figure()
        bbox0 = axs[0].get_position() #bbox contains the [x0 (left), y0 (bottom), x1 (right), y1 (top)] of the axis.
        bbox1 = axs[1].get_position()
        height = height
        eps =eps #margin between plot and colorbar
        # [left most position, bottom position, width, height] of color bar.
        cax = fig.add_axes([bbox0.x0, bbox0.y0 - eps, (bbox1.x1 - bbox0.x0), height])
        fig.colorbar(im, cax=cax, orientation="horizontal")