import nibabel as nib
import numpy as np
from sklearn.preprocessing import StandardScaler


def save_nifti(data: np.array, affine: np.array, filename: str) -> None:
    """
    Function to save nifit file

    Parameters
    ---------
    data: np.array
        array of data to save as image
    affine: np.array
        affine of image
    filename: str
        filename of image to save
    """
    new_img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(new_img, filename)


def normalization(img_data: np.array) -> np.array:
    """
    Function to normalise an image

    Parameters
    ----------
    img_data: np.array
        array of data to normalise

    Returns
    -------
    zscores: np.array
        array of zscores reshaped
        to be saved as a imge
    """
    n_voxels = np.prod(img_data.shape[:-1])
    n_vol = img_data.shape[-1]
    reshaped_data = img_data.reshape(n_voxels, n_vol)
    z_scores = StandardScaler().fit_transform(reshaped_data)
    return z_scores.reshape(img_data.shape)


def remove_negative(img_path: str, img_name: str) -> None:
    """
    Function to remove all negative values
    from an image.

    Parameters
    ----------
    img_path: str
        str to path
    img_name: str
        str of img name

    Returns
    -------
    None
    """
    img_comp = nib.load(img_path)
    data_img = img_comp.get_fdata()
    data_img[data_img < 0] = 0
    save_nifti(data_img, img_comp.affine, img_name)
