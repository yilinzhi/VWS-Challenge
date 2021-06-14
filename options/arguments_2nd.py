# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/05/09 15:42
# @Author  : Yi
# @FileName: arguments_2nd.py


import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_sep_ICAL',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_sep_position_train/ICAL',
                        help="The path of separated image")
    parser.add_argument('--train_mask_sep_ICAL',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/circle_mask_sep_train/ICAL',
                        help="The path of separated mask")
    parser.add_argument('--train_check_sep_ICAL',
                        default='/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train/ICAL',
                        help="save separated check points")

    parser.add_argument('--train_image_sep_ICAR',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_sep_position_train/ICAR',
                        help="The path of separated image")
    parser.add_argument('--train_mask_sep_ICAR',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/circle_mask_sep_train/ICAR',
                        help="The path of separated mask")
    parser.add_argument('--train_check_sep_ICAR',
                        default='/data/yilinzhi/Segmentation/VMS/checkpoints/ICAR',
                        help="save separated check points")

    parser.add_argument('--train_image_sep_ECAL',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_sep_position/ECAL',
                        help="The path of separated image")
    parser.add_argument('--train_mask_sep_ECAL',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/circle_mask_sep/ECAL',
                        help="The path of separated mask")
    parser.add_argument('--train_check_sep_ECAL',
                        default='/data/yilinzhi/Segmentation/VMS/checkpoints/ECAL',
                        help="save separated check points")

    parser.add_argument('--train_image_sep_ECAR',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_sep_position/ECAR',
                        help="The path of separated image")
    parser.add_argument('--train_mask_sep_ECAR',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/circle_mask_sep/ECAR',
                        help="The path of separated mask")
    parser.add_argument('--train_check_sep_ECAR',
                        default='/data/yilinzhi/Segmentation/VMS/checkpoints/ECAR',
                        help="save separated check points")

    args = parser.parse_args()
    return args