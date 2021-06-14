# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/05/02 19:54
# @Author  : Yi
# @FileName: image.py

import os
import pydicom
import glob
import shutil
import random
import numpy as np
from data_Parameter import parse_args
import xml.etree.ElementTree as ET


def create_dir(path):
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
    print(dcm_f.shape)

    dcm_img = np.zeros((dcm_size, dcm_size, dcm_size), dtype=np.float32)

    for dcmi in range(len(dcms)):  # len(dcms) = 640/720
        cdcm = pydicom.read_file(dcms[dcmi]).pixel_array.astype(np.float32)
        cdcm -= np.mean(cdcm)
        cdcm /= np.std(cdcm)

        dcm_img[
        dcm_size // 2 - cdcm.shape[0] // 2: dcm_size // 2 + cdcm.shape[0] // 2,
        dcm_size // 2 - cdcm.shape[1] // 2: dcm_size // 2 + cdcm.shape[1] // 2,
        dcmi,
        ] = cdcm

    return dcm_img


def list_contour_slices(qvsroot):
    avail_slices = []
    qvasimg = qvsroot.findall("QVAS_Image") # QVAS_Image is a list，length=dcm_img.shape[2]

    for dicom_slicei in range(720):  # index of avail_slices: 0-719
        conts = qvasimg[dicom_slicei - 1].findall("QVAS_Contour")

        if len(conts):
            avail_slices.append(dicom_slicei)
    return avail_slices


def list_contour_slices_min(qvsroot):
    avail_slices = []
    qvasimg = qvsroot.findall("QVAS_Image")

    for dicom_slicei in range(640):
        conts = qvasimg[dicom_slicei - 1].findall("QVAS_Contour")

        if len(conts):
            avail_slices.append(dicom_slicei)
    return avail_slices


def get_contour(qvsroot, dicomslicei, conttype, dcmsz=720):
    qvasimg = qvsroot.findall("QVAS_Image")

    if dicomslicei - 1 > len(qvasimg):
        print("no slice", dicomslicei)
        return

    assert int(qvasimg[dicomslicei - 1].get("ImageName").split("I")[-1]) == dicomslicei

    # TODO: how to understand dicomslicei-1 ?
    conts = qvasimg[dicomslicei - 1].findall("QVAS_Contour")

    tconti = -1
    for conti in range(len(conts)):
        if conts[conti].find("ContourType").text == conttype:
            tconti = conti
            break
    if tconti == -1:
        print("no such contour", conttype)
        return

    pts = conts[tconti].find("Contour_Point").findall("Point")
    contours = []
    for pti in pts:
        contx = float(pti.get("x")) / 512 * dcmsz
        conty = float(pti.get("y")) / 512 * dcmsz
        # if current pt is different from last pt, add to contours
        if len(contours) == 0 or contours[-1][0] != contx or contours[-1][1] != conty:
            contours.append([contx, conty])
    return np.array(contours)


def save_image(data_input_dir, image_save_path):
    """存储特定区间的正负样本：正负样本一起存储，先正后负。slices indices:101-600

    :param data_input_dir: 原始数据路径
    :param image_save_path: 图片存储位置
    """

    create_dir(image_save_path + '/' + 'ICAL' )
    create_dir(image_save_path + '/' + 'ICAR' )
    create_dir(image_save_path + '/' + 'ECAL' )
    create_dir(image_save_path + '/' + 'ECAR' )

    for casei in os.listdir(data_input_dir):
        pi = casei.split("_")[1]
        dcm_img = read_dicom(data_input_dir + "/" + casei)
        print("Dcm shape: ", dcm_img.shape)

        for arti in ["ICAL", "ICAR", "ECAL", "ECAR"]:
            cas_dir = data_input_dir + "/" + casei + "/CASCADE-" + arti
            qvs_path = cas_dir + "/E" + pi + "S101_L.QVS"
            qvsroot = ET.parse(qvs_path).getroot()

            all_slices = list(np.arange(101, 601, 1)) # 取index：100-600之间的slices
            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)  # 正样本slices
            else:
                avail_slices = list_contour_slices(qvsroot)
            not_avail_slices = list(set(all_slices).difference(set(avail_slices)))  # 负样本slices

            print("case", pi, "art", arti, "avail_slices", avail_slices)
            # print("case", pi, "art", arti, "avail_slices length", len(avail_slices))

            # TODO: 改进采样器Sampler.

            if len(avail_slices):  # 存储正样本：160*500
                for index in range(len(avail_slices)):
                    img_save = dcm_img[280:440, 110:610, avail_slices[index]]
                    # print(img_save.shape)

                    np.save(image_save_path + '/' + arti + '/' + 'positive_'  + pi + '_' + arti + '_' + str(
                        avail_slices[index]) + '_.npy',
                            img_save)

            if len(not_avail_slices):  # 存储负样本，同上
                print("case", pi, "art", arti, "no_avail_slices", not_avail_slices)
                # print("case", pi, "art", arti, "no_avail_slices length", len(choice_index))

                for index in not_avail_slices:
                    neg_img_save = dcm_img[280:440, 110:610, index]
                    # print(neg_img_save.shape)

                    np.save(image_save_path + '/' + arti + '/' + 'negative_' +  pi + '_' + arti + '_' + str(
                        index) + '_.npy',
                            neg_img_save)


def save_image_for_training(data_input_dir, image_save_path):
    """存储确定image：正负样本用于训练。"""

    create_dir(image_save_path + '/' + 'ICAL' )
    create_dir(image_save_path + '/' + 'ICAR' )
    create_dir(image_save_path + '/' + 'ECAL' )
    create_dir(image_save_path + '/' + 'ECAR' )

    for casei in os.listdir(data_input_dir):
        pi = casei.split("_")[1]
        dcm_img = read_dicom(data_input_dir + "/" + casei)
        print("Dcm shape: ", dcm_img.shape)

        for arti in ["ICAL", "ICAR", "ECAL", "ECAR"]:
            cas_dir = data_input_dir + "/" + casei + "/CASCADE-" + arti
            qvs_path = cas_dir + "/E" + pi + "S101_L.QVS"
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)  # 正样本slices
            else:
                avail_slices = list_contour_slices(qvsroot)

            not_avail_slices=list(range(100))
            list2=list(range(600,640))
            not_avail_slices.extend(list2)

            print("case", pi, "art", arti, "avail_slices", avail_slices)
            # print("case", pi, "art", arti, "avail_slices length", len(avail_slices))

            if len(avail_slices):  # 存储正样本：160*500
                for index in range(len(avail_slices)):
                    img_save = dcm_img[280:440, 110:610, avail_slices[index]]
                    # print(img_save.shape)

                    np.save(image_save_path + '/' + arti + '/' + 'positive_' + pi + '_' + arti + '_' + str(
                        avail_slices[index]) + '_.npy',
                            img_save)

            if len(not_avail_slices):  # 存储负样本，同上
                print("case", pi, "art", arti, "no_avail_slices", not_avail_slices)
                # print("case", pi, "art", arti, "no_avail_slices length", len(choice_index))

                for index in not_avail_slices:
                    neg_img_save = dcm_img[280:440, 110:610, index]
                    # print(neg_img_save.shape)

                    np.save(image_save_path + '/' + arti + '/' + 'negative_' +  pi + '_' + arti + '_' + str(
                        index) + '_.npy',
                            neg_img_save)


def main(args):
    data_input_dir = args.datasets_path
    image_save_path = args.image_save_sep_position_new
    image_save_path_training = args.image_save_sep_position_train

    # save_image(data_input_dir, image_save_path)
    save_image_for_training(data_input_dir,image_save_path_training)


if __name__ == '__main__':
    args = parse_args()
    main(args)

