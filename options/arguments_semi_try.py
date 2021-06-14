# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/05/28 22:34
# @Author  : Yi
# @FileName: arguments_semi_try.py


import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_path',
                        default="/data/yilinzhi/Segmentation/VMS/datasets/careIIChallenge",
                        help="The path of original datasets for images/masks.")

    parser.add_argument('--image_ICAL_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_semi_finally/ICAL_train',
                        help="The path of separated image")
    parser.add_argument('--mask_ICAL_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/mask_semi_finally/ICAL_train',
                        help="The path of separated mask")
    parser.add_argument('--check_ICAL_semi',
                        default='/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train_finally/ICAL_train',
                        help="save separated check points")

    parser.add_argument('--image_ICAR_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_semi_finally/ICAR_train',
                        help="The path of separated image")
    parser.add_argument('--mask_ICAR_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/mask_semi_finally/ICAR_train',
                        help="The path of separated mask")
    parser.add_argument('--check_ICAR_semi',
                        default='/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train_finally/ICAR_train',
                        help="save separated check points")

    parser.add_argument('--image_ECAL_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_semi_finally/ECAL_train',
                        help="The path of separated image")
    parser.add_argument('--mask_ECAL_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/mask_semi_finally/ECAL_train',
                        help="The path of separated mask")
    parser.add_argument('--check_ECAL_semi',
                        default='/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train_finally/ECAL_train',
                        help="save separated check points")

    parser.add_argument('--image_ECAR_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_semi_finally/ECAR_train',
                        help="The path of separated image")
    parser.add_argument('--mask_ECAR_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/mask_semi_finally/ECAR_train',
                        help="The path of separated mask")
    parser.add_argument('--check_ECAR_semi',
                        default='/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train_finally/ECAR_train',
                        help="save separated check points")

    args = parser.parse_args()
    return args