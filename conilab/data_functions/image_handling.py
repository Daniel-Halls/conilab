import nibabel as nib
import numpy as np
from sklearn.preprocessing import StandardScaler


def save_nifti(data: np.array, affine: np.array, filename: str) -> None:
    """
    Function to save nifti file from a
    numpy array

    Parameters
    ---------
    data: np.array
        array of data to save as image
    affine: np.array
        affine of image
    filename: str
        filename of image to save

    Returns
    -------
    None
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


def thresholding(
    data_img: np.array,
    threshold: int,
    two_sided: bool = False,
    remove_pos_onlys: bool = False,
) -> np.array:
    """
    Function to threshold an array

    Parameters
    ----------
    data_img: np.array
        array
    threshold: int
        threshold value
    two_sided: bool=False
        remove both positive and negative values
    remove_pos_onlys: bool=False
        remove only positive values

    Returns
    -------
    None
    """

    if remove_pos_onlys:
        data_img[data_img > threshold] = 0
        return data_img

    if two_sided:
        data_img[np.abs(data_img) < threshold] = 0
        return data_img

    data_img[data_img < threshold] = 0
    return data_img


def threshold_img(
    img_path: str,
    img_name: str,
    threshold: int,
    two_sided: bool = False,
    remove_pos_onlys: bool = False,
) -> None:
    """
    Parameters
    ----------
    img_path: str
        str to path
    img_name: str
        str of img name
    threshold: int
        threshold value
    two_sided: bool=False
        remove both positive and negative values
    remove_pos_onlys: bool=False
        remove only positive values


    Returns
    -------
    None
    """
    img_comp = nib.load(img_path)
    data_img = img_comp.get_fdata()
    img_data = thresholding(data_img, threshold, two_sided, remove_pos_onlys)
    save_nifti(img_data, img_comp.affine, img_name)


def mean_image(img_path: str, img_name: str) -> None:
    """
    Function to calculate the mean image

    Parameters
    ----------
    img_path: str
        path to image
    img_name: str
        name of image to save

    Returns
    -------
    None
    """
    comp = nib.load(img_path)
    img_4d_data = comp.get_fdata()
    mean_img = np.mean(img_4d_data, axis=-1)
    save_nifti(mean_img, comp.affine, img_name)


def create_mask(img_path: str, img_name: str):
    """
    Function to create a binary mask

    Parameters
    ----------
    img_path: str
        path to image
    img_name: str
        name of image to save

    Returns
    -------
    None
    """
    image_nifti = nib.load(img_path)
    image_data = image_nifti.get_fdata()
    binary_mask = np.where(np.abs(image_data) > 0, 1, 0)
    save_nifti(binary_mask, image_nifti.affine, img_name)


def masked_values(img_path: str, mask_path: np.array, img_name: str):
    """
    Function to extract values from
    an image using a mask

    Parameters
    ----------
    img_path: str
        path to image
    mask_path: str
        path to mask
    img_name: str
        name of image to save

    Returns
    -------
    None
    """

    img_comp = nib.load(img_path)
    mask = nib.load(mask_path)
    img_data = img_comp.get_fdata()
    mask_data = mask.get_fdata()
    masked_image_data = np.zeros_like(img_data)
    masked_values = img_data[mask_data != 0]
    masked_image_data[mask_data != 0] = masked_values
    save_nifti(masked_image_data, img_comp.affine, img_name)


def threshold_map(img_path: str, img_name: str, percentile: int):
    """
    Function to z score a map and retain
    a percentile of

    Parameteres
    -----------
    img_path: str
        path to image
    img_name: str
        name of image to save
    percentile: int
        percentile to retain

    Returns
    -------
    None
    """
    img_map = nib.load(img_path)
    img_map_data = img_map.get_fdata()
    mean = np.mean(img_map_data)
    std = np.std(img_map_data)
    normalized_data = (img_map_data - mean) / std
    thresholded_data = np.where(
        np.abs(normalized_data) > np.percentile(normalized_data, percentile),
        normalized_data,
        0,
    )
    new_img = nib.Nifti1Image(thresholded_data, img_map.affine, img_map.header)
    nib.save(new_img, img_name)
