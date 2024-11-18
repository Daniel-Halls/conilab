import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def define_new_cmap(cm_name) -> object:
    """
    Funciton to define new cmaps

    Parameters
    ----------
    cm_name: str
        name of colormap

    Returns
    -------
    object: matplotlib.colors.ListedColormap
    """
    cmap_col = plt.cm.get_cmap(cm_name)
    cmap = cmap_col(np.arange(cmap_col.N))
    cmap[:, -1] = np.linspace(0.5, 1, cmap_col.N)
    return ListedColormap(cmap)


def plot_time_series(time_series: np.ndarray) -> None:
    """
    Function to plot time series,
    auto correlation and partial correlation
    plots

    Parameters
    ----------
    time_series: np.ndarray
        array of time series

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_acf(time_series, lags=40, ax=ax1)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_pacf(time_series, lags=40, ax=ax2)
    ax3 = fig.add_subplot(gs[1, :])
    plt.plot(time_series)
    plt.title("Time series")
    plt.xlabel("Scan number")
    plt.ylabel("Signal")
    plt.show()
