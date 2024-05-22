import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


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
    cmap[:,-1] = np.linspace(0.5,1, cmap_col.N)
    return ListedColormap(cmap)