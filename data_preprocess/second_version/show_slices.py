# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/05/08 20:06
# @Author  : Yi
# @FileName: show_slices.py

import os
import pydicom
import glob
import shutil
import random
import numpy as np
import cv2
import skimage.io as io

from data_Parameter import parse_args
import matplotlib.pyplot as plt


def dir_create(path):
    """创造新的文件夹。

    :param path: 文件夹路径
    :return:
    """
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)


def read_dicom(path):
    """读取一个病例所有的slices，并转成一个720*720*720的numpy.array.

    :param path: 一个病例dcm路径
    :return:
    """
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
    """随机展示一个病例一些病理图像。

    :param input_dir:
    :return:
    """

    # special cases: "P556", "P576", "P887"，160*640*640
    for casei in os.listdir(input_dir)[5:6]:
        pi = casei.split("_")[1]
        dcm_img = read_dicom(input_dir + "/" + casei)
        print("Dcm shape: ", dcm_img.shape)

        # choices = random.sample(list(np.arange(0, 720, 1)), 10)
        # choices.append(316)

        choices = range(330,350)

        for i in choices:
            fig = plt.figure(num=i, figsize=(10, 10))
            ax = fig.add_subplot(111)
            img=ax.imshow(dcm_img[:, :, i], cmap='gray')
            ax.set_title(pi + '_' + str(i))
            plt.colorbar(img)
            plt.show()


def show_image_avail(input_dir):
    """随机展示一个位置的一些有标注的病例图像。

    :param input_dir:
    :return:
    """

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
    """随机展示一个位置标注的mask，2个channels.

    :param input_dir:
    :return:
    """

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
    """随机展示一个位置标注的mask环。

    :param input_dir:
    :return:
    """

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
    """随机展示一个位置的病例图像及其标注。

    :param image_path:
    :param mask_path:
    :return:
    """

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

    # image_avail_dir = args.image_save_sep_position + '/ICAR/positive'
    # image_avail_dir = args.image_save_sep_position + '/ICAR/negative'

    # circle_mask_dir=args.circle_mask_save_sep+'/ICAR/positive'
    circle_mask_dir = args.circle_mask_save_sep + '/ICAR/positive'

    # show_image(image_input_dir)  # 随机展示一些病例图像。
    # show_image_avail(image_avail_dir)
    show_mask_circle(circle_mask_dir)

    # show_image_mask(image_avail_dir,circle_mask_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)