# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/03/22 22:25
# @Author  : yilinzhi
# @FileName: data_parameter.py

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_path',
                        default="/data/yilinzhi/Segmentation/VMS/datasets/careIIChallenge",
                        help="The path of original datasets for images/masks")

    parser.add_argument('--image_save_sep_position',
                        default="/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_sep_position",
                        help="To save images separated")

    parser.add_argument('--image_save_sep_position_new',
                        default="/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_sep_position_new",
                        help="To save images separated")

    parser.add_argument('--image_save_union_slices',
                        default="/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_union_slices",
                        help="To save union images")

    parser.add_argument('--full_mask_save_sep',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/full_mask_sep',
                        help="The save full mask separated")

    parser.add_argument('--circle_mask_save_sep',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/circle_mask_sep',
                        help="The save circle mask separated")

    args = parser.parse_args()
    return args
