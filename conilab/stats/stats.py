import numpy as np


def calculate_dice(mask_1_array: np.array, mask_2_path: np.array) -> float:
    """
    Function to calculate the dice
    score for two arrays. Expects arrays
    to be of 0 and 1

    Parameters
    ----------
    mask_1_array: np.array
        array of mask 1
    mask_2_array: np.array
        array of mask 2

    Returns
    -------
    dice_score: float
        float of dice score
    """

    mask1 = mask_1_array.astype(bool)
    mask2 = mask_2_path.astype(bool)
    intersection = np.sum(mask1 & mask2)
    size_sum = np.sum(mask1) + np.sum(mask2)
    return float(2 * intersection / size_sum)