import numpy as np


def bf_upper_bound(p: float) -> float:
    """
    Function to calculate the bayes factor upper bound

    1/-ep log p

    where e is natural base
    p is p value
    and log is natural log

    Parameters
    ----------
    p: float
        p value

    Returns
    ------
    float: bayes factor upper bound
    """
    return 1 / ((-np.e * p) * np.log(p))
