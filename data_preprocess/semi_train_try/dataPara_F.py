# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/05/28 14:36
# @Author  : Yi
# @FileName: dataPara_F.py

"""半监督学习：最终参数"""

import argparse


def parse_args():
    """
        1.原始数据
        2.image存储位置
        3.mask存储位置
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_path',
                        default="/data/yilinzhi/Segmentation/VMS/datasets/careIIChallenge",
                        help="The path of original datasets for images/masks.")

    parser.add_argument('--image_semi',
                        default="/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_semi_finally",
                        help="To save separated images.")

    parser.add_argument('--mask_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/mask_semi_finally',
                        help="To save separated masks.")

    args = parser.parse_args()
    return args