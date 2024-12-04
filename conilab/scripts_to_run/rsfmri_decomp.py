#!/usr/bin/env python
import nibabel as nib
import glob
import os
from sklearn.decomposition import PCA, FastICA
import numpy as np
from scipy.stats import scoreatpercentile
from conilab.data_functions.image_handling import decompose_cifti, save_gifti
import argparse
import sys


def run_ica(matrix: np.ndarray, n_dim: int):
    """
    Function to run ICA

    Parameters
    ----------
    matrix: np.ndarray
         matrix to decompose
    n_dim: int
        number of dimensions

    Returns
    -------
    np.ndarray: np.array
        array of decomposed matrix

    """
    ica = FastICA(n_dim, random_state=42, max_iter=1000)
    return ica.fit_transform(matrix)


def load_subject_rsfmi(files: list):
    """
    Function to load subjects
    rsfmri ciftis
    """
    data_list = [load_matrix(file) for file in files]
    return np.hstack(data_list)


def load_matrix(rs_img: str) -> np.ndarray:
    """
    Wrapper function to load
    a cifti file and get surface data

    Parameters
    ----------
    rs_img: str
        str of file to rs cifti
    """
    cifti = nib.load(rs_img)
    surf = decompose_cifti(cifti)
    return np.vstack([surf["L_surf"], surf["R_surf"]])


def pca_decomp(n_components, decomp_matrix) -> np.ndarray:
    """
    Function to perform PCA on

    Parameters
    ----------
    n_components: int
        number of components
    decomp_matrix: np.ndarray
        np.array of matrix to decompose

    Returns
    -------
    dict: dictionary
         dict of matrix and components
    """
    pca = PCA(n_components=n_components)
    pca_matrix = pca.fit_transform(decomp_matrix)
    return {"components": pca.components_, "matrix": pca_matrix}


def cmd_args() -> dict:
    """
    Function to define cmd arguments

    Parameters
    ----------
    None

    Returns
    -------
    dict: dictionary
        dictionary of cmd arguments
    """
    args = argparse.ArgumentParser()
    args.add_argument(
        "-i", "--input", dest="folder", help="Path to folder with participants ciftis"
    )
    args.add_argument(
        "-o",
        "--outdir",
        dest="outdir",
        help="Path to output folder",
    )
    args.add_argument(
        "-t",
        "--threshold",
        dest="threshold",
        default=95,
        help="Threshold value to retain values at",
    )
    args.add_argument(
        "-d",
        "--dim",
        dest="dim",
        type=int,
        help="Number of dimensions for the ICA",
    )
    args.add_argument(
        "-c",
        "--components",
        dest="components",
        default=2500,
        type=int,
        help="Number of components for the PCA",
    )
    if len(sys.argv) == 1:
        args.print_help(sys.stderr)
        sys.exit(1)
    return vars(args.parse_args())


def main():
    """
    main function of rsfmri_decomp

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    args = cmd_args()
    print("Loading subjects resting state")
    func_img = glob.glob(os.path.join(args["folder"], "*.nii"))
    decomp_matrix = load_subject_rsfmi(func_img)
    cifti = nib.load(func_img[0])
    img_dict = decompose_cifti(cifti)
    print("Running PCA")
    pca = pca_decomp(args["components"], decomp_matrix.T)
    print("Running ICA")
    group_ICs = run_ica(pca["matrix"].T, args["dim"])
    group_ICs_vertex_space = np.linalg.pinv(group_ICs) @ pca["components"]
    breakpoint()
    left_ica = group_ICs_vertex_space[:, : img_dict["L_surf"].shape[0]]
    right_ica = group_ICs_vertex_space[:, img_dict["L_surf"].shape[0] :]

    for hemisphere_ica in [left_ica, right_ica]:
        threshold = scoreatpercentile(hemisphere_ica, args["threshold"])
        hemisphere_ica[np.abs(hemisphere_ica) < threshold] = 0
    print(f'Saving files to {args["outdir"]}')
    save_gifti(
        left_ica, os.path.join(args["outdir"], "rsfmri_ica_decomp_left.func.gii")
    )
    save_gifti(
        right_ica, os.path.join(args["outdir"], "rsfmri_ica_decomp_right.func.gii")
    )


if __name__ == "__main__":
    main()
