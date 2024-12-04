#!/usr/bin/env python
# A quick script to clean HCP volume images

import glob
import nilearn.image as img
import numpy as np
import os
import sys

if __name__ == "__main__":
    files = glob.glob(f"{sys.argv[0]}/*/fMRI_CONCAT_ALL.nii.gz")
    motion = glob.glob(f"{sys.argv[0]}/*/Movement_Regressors_hp0_clean.txt")
    for idx, sub in enumerate(files):
        print(f"Cleaning img {sub}")
        confounds = np.loadtxt(motion[idx])
        img_to_clean = img.clean_img(
            files[idx],
            t_r=0.7,
            confounds=confounds,
            detrend=True,
            low_pass=0.08,
            high_pass=0.01,
            ensure_finite=True,
            standardize=False,
        )

    img_to_clean.to_filename(os.path.join(os.getcwd(), f"{idx}_detrended"))
