# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/03/25 22:21
# @Author  : yilinzhi
# @FileName: show_slices.py


import os
import pydicom
import glob
import shutil
import random
import numpy as np
import cv2
import skimage.io as io
from data_parameter import parse_args
import matplotlib.pyplot as plt


def dir_create(path):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)


def read_dicom(path):
    print(os.path.basename(path))

    pi = os.path.basename(path).split("_")[1]
    dcm_size = len(glob.glob(path + "/*.dcm"))
    dcms = [
        path + "/E" + pi + "S101I%d.dcm" % dicom_slicei
        for dicom_slicei in range(1, dcm_size + 1)
    ]

    length = int(len(dcms))
    print(length)

    dcm_f = pydicom.read_file(dcms[0]).pixel_array
    dcm_size = max(max(dcm_f.shape), 720)
    # print(dcm_f.shape)

    dcm_img = np.zeros((dcm_size, dcm_size, dcm_size), dtype=np.float32)

    for dcmi in range(len(dcms)):
        cdcm = pydicom.read_file(dcms[dcmi]).pixel_array.astype(np.float32)

        cdcm -= np.mean(cdcm)
        cdcm /= np.std(cdcm)

        dcm_img[
        dcm_size // 2 - cdcm.shape[0] // 2: dcm_size // 2 + cdcm.shape[0] // 2,
        dcm_size // 2 - cdcm.shape[1] // 2: dcm_size // 2 + cdcm.shape[1] // 2,
        dcmi,
        ] = cdcm

    return dcm_img


def show_image(input_dir):
    """Random show 10 slices in a patient case """

    # special cases: index=15/16，160*640*640
    for casei in os.listdir(input_dir)[14:15]:
        pi = casei.split("_")[1]
        dcm_img = read_dicom(input_dir + "/" + casei)
        print("Dcm shape: ", dcm_img.shape)

        choices = random.sample(list(np.arange(0, 720, 1)), 10)
        # choices.append(316)

        for i in choices:
            fig = plt.figure(num=i, figsize=(10, 10))
            ax = fig.add_subplot(111)
            img=ax.imshow(dcm_img[:, :, i], cmap='gray')
            ax.set_title(pi + '_' + str(i))
            plt.colorbar(img)
            plt.show()


def show_image_avail(input_dir):
    """Random show 10 available slices"""

    choices = random.sample(os.listdir(input_dir), 15)
    for file in choices:
        image_numpy = np.load(input_dir + '/' + file)

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(111)
        img1=ax1.imshow(image_numpy, cmap='gray')
        ax1.set_title(str(file))
        plt.colorbar(img1)
        plt.show()


def show_mask(input_dir):
    """Masks with 2 channels"""

    index = 0
    choices = random.sample(os.listdir(input_dir), 10)
    for file in choices:
        mask_numpy = np.load(input_dir + '/' + file)

        fig = plt.figure(num=index, figsize=(10, 5))
        ax1 = fig.add_subplot(211)
        ax1.imshow(mask_numpy[:, :, 0], cmap='gray')
        ax1.set_title(str(file) + '_outer')
        ax2 = fig.add_subplot(212)
        ax2.imshow(mask_numpy[:, :, 1], cmap='gray')
        ax2.set_title(str(file) + '_luman')
        plt.show()
        index += 1


def show_mask_circle(input_dir):

    choices = random.sample(os.listdir(input_dir), 10)
    for file in choices:
        mask_numpy = np.load(input_dir + '/' + file)

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(111)
        img1=ax1.imshow(mask_numpy[:, :], cmap='gray')
        ax1.set_title(str(file) + '_circle')
        plt.colorbar(img1)

        plt.show()


def show_image_mask(image_path,mask_path):
    """Checking whether the image / mask corresponds """

    files_choice=random.sample(os.listdir(image_path),10)

    for file_name in files_choice:
        image_numpy=np.load(image_path+'/'+file_name)
        mask_numpy =np.load(mask_path+'/'+file_name)

        fig =plt.figure(figsize=(10,5))
        ax1 =fig.add_subplot(211)
        img1=ax1.imshow(image_numpy,cmap='gray')
        ax1.set_title(str(file_name))
        plt.colorbar(img1)

        ax2=fig.add_subplot(212)
        img2=ax2.imshow(mask_numpy,cmap='gray')
        # ax2.set_title(str(file_name))
        plt.colorbar(img2)
        plt.show()


def main(args):
    image_input_dir = args.datasets_path

    # available slices of image
    image_avail_dir = args.image_save_sep_position + '/ICAR/positive'
    # image_avail_dir = args.image_save_sep_position + '/ICAR/negative'

    # Corresponding mask
    circle_mask_dir=args.circle_mask_save_sep+'/ICAR/positive'
    # circle_mask_dir = args.circle_mask_save_sep + '/ICAR/negative'

    # show_image(image_input_dir)
    # show_image_avail(image_avail_dir)
    # show_mask_circle(circle_mask_dir)

    show_image_mask(image_avail_dir,circle_mask_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
