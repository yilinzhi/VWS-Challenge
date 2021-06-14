# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/03/14 09:32
# @Author  : yilinzhi
# @FileName: dcm_to_nii.py

"""Function:
    Changing dcm style files into nii style files
"""

import dicom2nifti
import os
from data_preprocess.first_version.data_parameter import parse_args


def main(args):
    dcm_path=args.datasets_path
    nii_path="/data/yilinzhi/Segmentation/VMS/datasets/nii_files/"
    for file in os.listdir(dcm_path):
        dcm_file_path = os.path.join(dcm_path, file)
        nii_file_path = nii_path + str(file) + '_.nii'
        dicom2nifti.dicom_series_to_nifti(dcm_file_path, nii_file_path, reorient_nifti=True)


if __name__=='__main__':
    args=parse_args()
    main(args)


