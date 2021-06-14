# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/05/17 16:06
# @Author  : Yi
# @FileName: data_parameter.py


import argparse


def parse_args():
    """
        1.Raw data
        2.Preprocessed image
        3.Preprocessed mask
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_path',
                        default="/data/yilinzhi/Segmentation/VMS/datasets/careIIChallenge",
                        help="The path of original datasets for images/masks.")

    parser.add_argument('--image_semi',
                        default="/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_semi_F",
                        help="To save separated images.")

    parser.add_argument('--mask_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/mask_semi_F',
                        help="To save separated masks.")

    args = parser.parse_args()
    return args