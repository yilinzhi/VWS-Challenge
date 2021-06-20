# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/05/17 21:09
# @Author  : Yi
# @FileName: arguments_semi_submit.py


import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_path',
                        default="/data/yilinzhi/Segmentation/VMS/datasets/careIIChallenge",
                        help="The path of original datasets for images/masks.")

    parser.add_argument('--image_ICAL_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_semi_F/train/ICAL',
                        help="Training images.")
    parser.add_argument('--mask_ICAL_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/mask_semi_F/train/ICAL',
                        help="Training masks.")
    parser.add_argument('--check_ICAL_semi',
                        default='/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train_F/train/ICAL',
                        help="Training saving.")
    parser.add_argument('--test_ICAL_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_semi_F/test/ICAL',
                        help="Test images.")
    parser.add_argument('--true_ICAL_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/mask_semi_F/test/ICAL',
                        help="Test masks.")
    parser.add_argument('--pred_semi',
                        default='/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train_F/pred_',
                        help="Pred image png")

    parser.add_argument('--image_ICAR_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_semi_F/train/ICAR',
                        help="Training images.")
    parser.add_argument('--mask_ICAR_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/mask_semi_F/train/ICAR',
                        help="Training masks.")
    parser.add_argument('--check_ICAR_semi',
                        default='/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train_F/train/ICAR',
                        help="Training saving.")
    parser.add_argument('--test_ICAR_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_semi_F/test/ICAR',
                        help="Test images.")
    parser.add_argument('--true_ICAR_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/mask_semi_F/test/ICAR',
                        help="Test masks.")

    parser.add_argument('--image_ECAL_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_semi_F/train/ECAL',
                        help="Training images.")
    parser.add_argument('--mask_ECAL_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/mask_semi_F/train/ECAL',
                        help="Training masks.")
    parser.add_argument('--check_ECAL_semi',
                        default='/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train_F/train/ECAL',
                        help="Training saving.")
    parser.add_argument('--test_ECAL_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_semi_F/test/ECAL',
                        help="Test images.")
    parser.add_argument('--true_ECAL_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/mask_semi_F/test/ECAL',
                        help="Test masks.")
    
    parser.add_argument('--image_ECAR_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_semi_F/train/ECAR',
                        help="Training images.")
    parser.add_argument('--mask_ECAR_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/mask_semi_F/train/ECAR',
                        help="Training masks.")
    parser.add_argument('--check_ECAR_semi',
                        default='/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train_F/train/ECAR',
                        help="Training saving.")
    parser.add_argument('--test_ECAR_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_semi_F/test/ECAR',
                        help="Test images.")
    parser.add_argument('--true_ECAR_semi',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/mask_semi_F/test/ECAR',
                        help="Test masks.")

    args = parser.parse_args()
    return args