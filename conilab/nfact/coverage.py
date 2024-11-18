import nibabel as nib
from nilearn import image as img
import numpy as np
import os
from ..data_functions.image_handling import save_nifti, normalization


def areas_of_difference(mask_1_path: str, mask_2_path: str, save_path: str) -> None:
    """
    Function to calculate areas of difference between
    two masks

    Parameters
    ----------
    mask_1_path: str
        path to mask 1
    mask_2_path: str
        path to mask 2
    save_path: str
        Path to save images

    Returns
    -------
    None
    """
    mask1 = nib.load(mask_1_path)
    mask2 = nib.load(mask_2_path)
    mask1_name = os.path.basename(mask_1_path).lstrip(".nii.gz")
    mask2_name = os.path.basename(mask_2_path).lstrip(".nii.gz")
    sub_img = img.math_img("img1 - img2", img1=mask1, img2=mask2)
    sub_img.to_filename(f"{save_path}/{mask1_name}>{mask2_name}.nii.gz")
    sub_img = img.math_img("img2 - img1", img1=mask1, img2=mask2)
    sub_img.to_filename(f"{save_path}/{mask2_name}>{mask1_name}.nii.gz")


def calculate_coverage(img_path: str, bin_mask: str) -> float:
    """
    Function to calculate the coverage of
    of two image mask

    Parameters
    ----------
    img_path: str
        path to image
    bin_mask: str
        path to mask

    Returns
    --------
    float: float
       float of percentage coverage
    """
    cov_map = nib.load(img_path).get_fdata()
    brain_mask = nib.load(bin_mask).get_fdata()
    non_zero_voxels = np.sum((cov_map != 0) & (brain_mask != 0))
    total_brain_voxels = np.sum(brain_mask != 0)
    return (non_zero_voxels / total_brain_voxels) * 100


def coverage_map(img_path: str, img_name: str, threshold: int, no_threshold=False):
    """
    Function to create a binary coverage mask
    and a hitmap of voxels

    Parameters
    ----------
    img_path: str
        path to image
    img_name: str
        name of image
    threshold: int
        value to thres

    Returns
    --------
    float: float
       float of percentage coverage
    """
    img_comp = nib.load(img_path)
    img_data = img_comp.get_fdata()
    zscores = normalization(img_data)
    binary_masks = np.abs(zscores) > threshold
    coverage_map = np.sum(binary_masks, axis=-1)
    save_nifti(coverage_map, img_comp.affine, img_name)
    coverage_map_mask = np.any(binary_masks, axis=-1).astype(np.uint8)
    image_name_mask = os.path.join(
        os.path.dirname(img_name), f"mask_{os.path.basename(img_name)}"
    )
    save_nifti(coverage_map_mask, img_comp.affine, image_name_mask)


def gifti_grey_coverage(hitmap_path: str) -> float:
    """
    Calculate the percentage coverage
    from a hitmap

    Parameters
    ----------
    hitmap_path: str
        path to hitmap

    Returns
    -------
    float: float
        percentage as a float
        of coverage
    """
    hitmap = nib.load(hitmap_path)
    data_gm = hitmap.darrays[0].data
    return data_gm[data_gm > 0].shape[0] / data_gm.shape[0] * 100


def create_gifti_hitmap(seed_path: str, filename: str) -> None:
    """
    Function to create hitmap from
    seed.

    Parameters
    ----------
    seed_path: str
        str of path to seed
    filename: str
        name of file. Does not
        need .func.gii

    Returns
    -------
    None
    """
    seed = nib.load(seed_path)
    combinearray = np.array(
        [seed.darrays[idx].data for idx, _ in enumerate(seed.darrays)]
    )
    hitmap = np.sum(combinearray > 0, axis=0)
    darrays = [
        nib.gifti.GiftiDataArray(
            hitmap,
            datatype="NIFTI_TYPE_FLOAT32",
            intent=2001,
            meta=seed.darrays[0].meta,
        )
    ]
    nib.gifti.GiftiImage(darrays=darrays).to_filename(f"{filename}.func.gii")
