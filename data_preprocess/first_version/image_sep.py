# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/03/15 10:53
# @Author  : yilinzhi
# @FileName: image_sep.py

"""Function: Save available slices of positions masks"""

import os
import pydicom
import glob
import shutil
import random
import numpy as np
from data_parameter import parse_args
import xml.etree.ElementTree as ET


def create_dir(path):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)

    if not os.path.exists(path):
        os.makedirs(path)


def read_dicom(path):
    """Read all slice.

        Args:
            path: the path of dicom.

        Returns:
            dcm_img: numpy, h*w*s: [720,720,720]

    """

    print(os.path.basename(path))  # Showing which patient case is processing
    pi = os.path.basename(path).split("_")[1]
    dcm_size = len(glob.glob(path + "/*.dcm"))

    dcms = [
        path + "/E" + pi + "S101I%d.dcm" % dicom_slicei
        for dicom_slicei in range(1, dcm_size + 1)
    ]

    length = int(len(dcms))
    print(length)

    # Reading dcm files and changing into numpy file
    # Choosing max(h, w, 720)
    dcm_f = pydicom.read_file(dcms[0]).pixel_array
    dcm_size = max(max(dcm_f.shape), 720)
    print(dcm_f.shape)

    dcm_img = np.zeros((dcm_size, dcm_size, dcm_size), dtype=np.float32)

    for dcmi in range(len(dcms)):  # len(dcms) = 640/720
        # TODO: whether choosing a threshold?
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
    """Find available slices of mask."""
    avail_slices = []

    # QVAS_Image is a list，length=dcm_img.shape[2]
    qvasimg = qvsroot.findall("QVAS_Image")

    for dicom_slicei in range(720):  # index of avail_slices: 0-719
        conts = qvasimg[dicom_slicei - 1].findall("QVAS_Contour")

        if len(conts):
            avail_slices.append(dicom_slicei)
    return avail_slices


def list_contour_slices_min(qvsroot):
    """Special examples: 640 slices"""
    avail_slices = []

    qvasimg = qvsroot.findall("QVAS_Image")

    for dicom_slicei in range(640):
        conts = qvasimg[dicom_slicei - 1].findall("QVAS_Contour")

        if len(conts):
            avail_slices.append(dicom_slicei)
    return avail_slices


def get_contour(qvsroot, dicomslicei, conttype, dcmsz=720):
    """Get points of lumen/outer wall in dicomslicei slice."""
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


def create_image(data_input_dir, image_save_path):
    """Save positive/negative images of four positions"""

    # Positive samples
    create_dir(image_save_path + '/' + 'ICAL' + '/' + 'positive')
    create_dir(image_save_path + '/' + 'ICAR' + '/' + 'positive')
    create_dir(image_save_path + '/' + 'ECAL' + '/' + 'positive')
    create_dir(image_save_path + '/' + 'ECAR' + '/' + 'positive')

    # Negative samples
    create_dir(image_save_path + '/' + 'ICAL' + '/' + 'negative')
    create_dir(image_save_path + '/' + 'ICAR' + '/' + 'negative')
    create_dir(image_save_path + '/' + 'ECAL' + '/' + 'negative')
    create_dir(image_save_path + '/' + 'ECAR' + '/' + 'negative')

    for casei in os.listdir(data_input_dir):
        pi = casei.split("_")[1]
        dcm_img = read_dicom(data_input_dir + "/" + casei)
        print("Dcm shape: ", dcm_img.shape)

        for arti in ["ICAL", "ICAR", "ECAL", "ECAR"]:
            cas_dir = data_input_dir + "/" + casei + "/CASCADE-" + arti
            qvs_path = cas_dir + "/E" + pi + "S101_L.QVS"
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                all_slices = list(np.arange(0, 640, 1))
                avail_slices = list_contour_slices_min(qvsroot)
                not_avail_slices = list(set(all_slices).difference(set(avail_slices)))

            else:
                all_slices = list(np.arange(0, 720, 1))
                avail_slices = list_contour_slices(qvsroot)
                not_avail_slices = list(set(all_slices).difference(set(avail_slices)))

            print("case", pi, "art", arti, "avail_slices", avail_slices)
            # print("case", pi, "art", arti, "avail_slices length", len(avail_slices))

            if len(avail_slices):  # positive samples
                for index in range(len(avail_slices)):
                    img_save = dcm_img[280:440, 110:610, avail_slices[index]]
                    # print(img_save.shape)

                    np.save(image_save_path + '/' + arti + '/' + 'positive' + '/' + pi + '_' + arti + '_' + str(
                        avail_slices[index]) + '_.npy',
                            img_save)

            if len(avail_slices):  # negative samples
                choice_index = random.sample(not_avail_slices, len(avail_slices))
                choice_index.sort()
                print("case", pi, "art", arti, "no_avail_slices", choice_index)
                # print("case", pi, "art", arti, "no_avail_slices length", len(choice_index))

                for index in choice_index:
                    neg_img_save = dcm_img[280:440, 110:610, index]
                    # print(neg_img_save.shape)

                    np.save(image_save_path + '/' + arti + '/' + 'negative' + '/' + pi + '_' + arti + '_' + str(
                        index) + '_.npy',
                            neg_img_save)


def main(args):
    data_input_dir = args.datasets_path
    image_save_path = args.image_save_sep_position

    create_image(data_input_dir, image_save_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
