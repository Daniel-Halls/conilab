import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
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
    _ = fig.add_subplot(gs[1, :])
    plt.plot(time_series)
    plt.title("Time series")
    plt.xlabel("Scan number")
    plt.ylabel("Signal")
    plt.show()


def plot_xtract_corr(
    df: object, threshold_value: float, size_x: int = 16, size_y: int = 8
) -> None:
    """
    Function to plot heat map one unthresholded
    the other thresholded

    Parameters
    ----------
    df: object
        dataframe to plot
    threshold_value: float
        threshold value

    Returns
    -------
    None
    """
    _, ax = plt.subplots(1, 2, figsize=(size_x, size_y))
    sns.heatmap(df, ax=ax[0])
    sns.heatmap(
        df[df > threshold_value]
        .dropna(axis=1, how="all")
        .dropna(axis=0, how="all")
        .fillna(0),
        ax=ax[1],
    )
