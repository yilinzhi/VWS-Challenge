# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/05/08 20:59
# @Author  : Yi
# @FileName: data_Parameter.py

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_path',
                        default="/data/yilinzhi/Segmentation/VMS/datasets/careIIChallenge",
                        help="The path of original datasets for images/masks.")

    parser.add_argument('--image_save_sep_position',
                        default="/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_sep_position",
                        help="To save separated images.")

    parser.add_argument('--image_save_sep_position_new',
                        default="/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_sep_position_new",
                        help="To save separated images.")

    parser.add_argument('--image_save_sep_position_train',
                        default="/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_sep_position_train",
                        help="To save separated images for training.")

    parser.add_argument('--image_save_union_slices',
                        default="/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_union_slices",
                        help="To save union images.")

    parser.add_argument('--full_mask_save_sep',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/full_mask_sep',
                        help="To save separated full mask.")

    parser.add_argument('--circle_mask_save_sep',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/circle_mask_sep',
                        help="To save separated circle mask.")

    parser.add_argument('--circle_mask_save_sep_train',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/circle_mask_sep_train',
                        help="To save separate circle mask.")

    args = parser.parse_args()
    return args
