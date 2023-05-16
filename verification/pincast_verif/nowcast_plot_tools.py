"""
Collection of functions making it easier to visualize precipitation nowcasts.

Bent Harnist (FMI), 2022-2023
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pysteps.visualization import plot_precip_field
from pysteps.visualization import get_colormap
from IPython.display import HTML


def return_im_not_ax(fn: callable, idx: int = 0, *fn_args, **fn_kwargs):
    """
    Wrapper for returning the idx:th image from a plot function
    that would otherwise return the whole mpl.axes.Axes object.

    Args:
        fn (callable): Visualization function returning an mpl.axes.Axes object.
        idx (int): index corresponding to which image of the axis to get. Defaults to 0.

    Returns:
        AxesImage: Object corresponding to the image drawn on the axis
    """
    ""
    ax = fn(*fn_args, **fn_kwargs)
    return [
        obj for obj in ax.get_children() if isinstance(obj, matplotlib.image.AxesImage)
    ][idx]


def plot_precip_field_im(*args, **kwargs):
    """
    Wrapped PySTEPS plot_precip_field function,
    which makes it return an AxesImage object instead of an mpl.axes.Axes object,
    making it consistent with the behaviour of e.g. `matplotlib.pyplot.imshow`.
    """
    return return_im_not_ax(plot_precip_field, 0, *args, **kwargs)


def animate_image_sequence(
    data: np.ndarray,
    show_function: callable,
    fig: matplotlib.figure.Figure = None,
    show_function_args: list = [],
    show_function_kwargs: dict = {},
    save_path: bool = None,
    show: bool = True,
    fps=5,
    func_animation_kwargs: dict = {},
):
    """
    Animate arbitrary image sequences, such as nowcasts or radar images,
    using a user defined drawing function.
    Supports only one animation per matplotlib figure.

    Args:
        data (np.ndarray): image sequence, time x width x height dimensions
        fig (matplotlib.figure.Figure): figure upon wihch to show the animation
        show_function (callable): function for drawing frames, which has to return an AxesImage object.
        show_function_args (list, optional): positional arguments to show_function. Defaults to [].
        show_function_kwargs (dict, optional): keyword arguments to show_function. Defaults to {}.
        save_path (bool, optional): None (no saving) or string indicating saving path and filename
            for the animation. Defaults to None.
        show (bool, optional): whether to return the animation as HTML for showing in notebooks. Defaults to True.
        fps (int, optional): Frames per second for the animation. Defaults to 5.
        func_animation_kwargs (dict, optional): keyword arguments for the matplotlib `animation.FuncAnimation`
           function used as the animation backend. Defaults to {}.

    Returns:
        HTMLDisplayObject / None: Animation in HTML format / None.
    """
    if fig is None:
        fig, ax = plt.subplots()

    n_frames = data.shape[0]

    im = show_function(data[0], *show_function_args, **show_function_kwargs)

    #def init():
    #    im.set_data(data[0, :, :])
    #    return (im,)

    # animation function. This is called sequentially
    def animate(i):
        data_slice = data[i, :, :]
        im.set_data(data_slice)
        return (im,)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=n_frames,
        interval=1000 / fps,
        **func_animation_kwargs,
    )

    if save_path is not None:
        anim.save(save_path)

    plt.close()

    if show:
        return HTML(anim.to_html5_video())


def plot_image_grid(
    data: np.ndarray,
    colormap_kwargs: str = None,
    subplots_kwargs: dict = {},
):
    """
    Plot a gridded overview of a collection
    of (nowcast/radar) images labeled with their corresponding index.

    Args:
        data (np.ndarray): Collection of images to plot, of shape ith_image x width x height
        unit (str): Unit identifier passable to the 'unit' argument of `pysteps.visualization.plot_precip_field`.
            Setting this will use that function for visualization, while leaving it to None will
            default to using imshow with default arguments. Defaults to None.
        subplot_kwargs (dict): dictionary of keyword arguments for the plt.subplots object generated for the grid.

    Returns:
        matplotlib.figure.Figure: Figure of the collection overview
    """
    n_rows = int(np.sqrt(data.shape[0]))
    n_cols = int(data.shape[0] / n_rows) + 1
    
    if colormap_kwargs is None:
        vmin=np.nanmin(data)
        vmax=np.nanmax(data)
    else:
        cmap, norm, _, _ = get_colormap(**colormap_kwargs)

    assert (
        n_rows > 1 and n_cols > 1
    ), "both number of rows and columns must be superior to one"

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, **subplots_kwargs)

    counter = 0
    for axrow in axes:
        for ax in axrow:
            ax.axis("off")
            try:
                if colormap_kwargs is None:
                    ax.imshow(data[counter], vmin=vmin, vmax=vmax)
                else:
                    ax.imshow(data[counter], cmap=cmap, norm=norm)
                ax.set_title(counter)
                counter += 1
            except IndexError:
                continue
    plt.tight_layout()
    plt.close()
    return fig
