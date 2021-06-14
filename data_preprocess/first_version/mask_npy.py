# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/03/13 19:46
# @Author  : yilinzhi
# @FileName: mask_npy.py


import os
import cv2
import shutil
import pydicom
import glob
import random
import numpy as np
import xml.etree.ElementTree as ET
from data_parameter import parse_args


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

    dcm_img = np.zeros((dcm_size, dcm_size, dcm_size), dtype=np.float32)

    for dcmi in range(len(dcms)):
        cdcm = pydicom.read_file(dcms[dcmi]).pixel_array
        dcm_img[
        dcm_size // 2 - cdcm.shape[0] // 2: dcm_size // 2 + cdcm.shape[0] // 2,
        dcm_size // 2 - cdcm.shape[1] // 2: dcm_size // 2 + cdcm.shape[1] // 2,
        dcmi,
        ] = cdcm

    return dcm_img


def list_contour_slices(qvsroot):
    avail_slices = []
    qvasimg = qvsroot.findall('QVAS_Image')
    for dicom_slicei in range(720):
        conts = qvasimg[dicom_slicei - 1].findall('QVAS_Contour')
        if len(conts):
            avail_slices.append(dicom_slicei)
    return avail_slices


def list_contour_slices_min(qvsroot):
    avail_slices = []
    qvasimg = qvsroot.findall("QVAS_Image")
    # print(qvasimg)
    for dicom_slicei in range(640):
        conts = qvasimg[dicom_slicei - 1].findall("QVAS_Contour")
        #         print(conts)
        if len(conts):
            avail_slices.append(dicom_slicei)
    return avail_slices


def get_contour(qvsroot, dicomslicei, conttype, dcmsz=720):
    qvasimg = qvsroot.findall("QVAS_Image")

    if dicomslicei - 1 > len(qvasimg):
        print("no slice", dicomslicei)
        return

    assert int(qvasimg[dicomslicei - 1].get("ImageName").split("I")[-1]) == dicomslicei

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


height = 720
width = 720


def create_mask_file(background, contours, channels):
    """Mask circle"""
    mask = np.full((height, width, channels), background, dtype=np.float32)
    mask = cv2.drawContours(mask, [contours.astype(int)], -1, color=(1), thickness=2)
    mask = cv2.fillPoly(mask, [contours.astype(int)], color=(1))
    return mask


def create_mask_circle_file(background, contours_inner,contours_outer, channels):
    """Two channels mask"""
    mask = np.full((height, width, channels), background, dtype=np.float32)
    mask = cv2.drawContours(mask, [contours_inner.astype(int)], -1, color=(1), thickness=2)
    mask = cv2.drawContours(mask, [contours_outer.astype(int)], -1, color=(1), thickness=2)
    mask = cv2.fillPoly(mask, [contours_outer.astype(int)], color=(1))
    mask = cv2.fillPoly(mask, [contours_inner.astype(int)], color=(0))
    return mask


def create_mask(input_dir, save_path):
    """Saving four positions of masks."""
    dir_create(save_path + '/' + 'ICAL' + '/positive')
    dir_create(save_path + '/' + 'ICAR' + '/positive')
    dir_create(save_path + '/' + 'ECAL' + '/positive')
    dir_create(save_path + '/' + 'ECAR' + '/positive')

    dir_create(save_path + '/' + 'ICAL' + '/negative')
    dir_create(save_path + '/' + 'ICAR' + '/negative')
    dir_create(save_path + '/' + 'ECAL' + '/negative')
    dir_create(save_path + '/' + 'ECAR' + '/negative')

    for casei in os.listdir(input_dir):
        pi = casei.split('_')[1]
        print(input_dir + '/' + casei)
        dcm_img = read_dicom(input_dir + '/' + casei)
        print('Dcm shape', dcm_img.shape)

        for arti in ['ICAL', 'ICAR', 'ECAL', 'ECAR']:
            cas_dir = input_dir + '/' + casei + '/CASCADE-' + arti
            qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
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

            if len(avail_slices):
                for index in range(len(avail_slices)):
                    dicom_slicei = avail_slices[index]

                    lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                    wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')

                    background_1 = np.zeros((height, width, 1), dtype=np.float32)
                    background_2 = np.zeros((height, width, 1), dtype=np.float32)

                    background = np.zeros((height, width, 1), dtype=np.float32)

                    # # two channels of masks
                    # mask_1 = create_mask_file(background_1, wall_cont, 1)
                    # mask_2 = create_mask_file(background_2, lumen_cont, 1)
                    # mask_save_1 = mask_1[280:440, 110:610, 0]
                    # mask_save_2 = mask_2[280:440, 110:610, 0]
                    #
                    # # outer/inner contour: 0/1 , h*w*c
                    # mask = np.stack((mask_save_1, mask_save_2), axis=2)
                    # print(mask.shape)
                    #
                    # np.save(save_path + '/' + arti + '/' + 'positive/' + pi + '_' + arti + '_' + str(
                    #     avail_slices[index]) + '_.npy',
                    #         mask)

                    # mask circle
                    mask_return=create_mask_circle_file(background,lumen_cont,wall_cont,1)
                    mask_save=mask_return[280:440, 110:610, 0]
                    # mask_save=mask_save[:,:,np.newaxis]
                    print(mask_save.shape)
                    np.save(save_path + '/' + arti + '/' + 'positive/' + pi + '_' + arti + '_' + str(
                        avail_slices[index]) + '_.npy',
                            mask_save)

            if len(avail_slices):
                choice_index = random.sample(not_avail_slices, len(avail_slices))
                choice_index.sort()
                print("case", pi, "art", arti, "not_avail_slices", choice_index)
                for index in choice_index:
                    # neg_mask_save = np.zeros((160, 500,2), dtype=np.float32)

                    neg_mask_save = np.zeros((160, 500), dtype=np.float32)
                    print(neg_mask_save.shape)

                    np.save(save_path + '/' + arti + '/negative/' + pi + '_' + arti + '_' + str(
                        index) + '_.npy',
                            neg_mask_save)


def main(args):
    input_dir = args.datasets_path
    # save_path = args.full_mask_save_sep
    save_path=args.circle_mask_save_sep

    create_mask(input_dir, save_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
