# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/05/28 14:37
# @Author  : Yi
# @FileName: mask_F.py


import os
import cv2
import shutil
import pydicom
import glob
import random
import numpy as np
import xml.etree.ElementTree as ET
from dataPara_F import parse_args


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


def create_mask_circle_file(background, contours_inner, contours_outer, channels):
    """血管环：mask circle"""
    mask = np.full((height, width, channels), background, dtype=np.float32)
    mask = cv2.drawContours(mask, [contours_inner.astype(int)], -1, color=(1), thickness=2)
    mask = cv2.drawContours(mask, [contours_outer.astype(int)], -1, color=(1), thickness=2)
    mask = cv2.fillPoly(mask, [contours_outer.astype(int)], color=(1))
    mask = cv2.fillPoly(mask, [contours_inner.astype(int)], color=(0))
    return mask


def save_internal_train(input_dir, save_path):
    dir_create(save_path + '/' + 'ICAL_train')
    dir_create(save_path + '/' + 'ICAR_train')

    test_patience = ["P206", "P432", "P576", "P891"]  # 随意选出四个作为测试病例

    for casei in os.listdir(input_dir):
        pi = casei.split('_')[1]

        if pi in test_patience:
            continue

        # print(input_dir + '/' + casei)
        dcm_img = read_dicom(input_dir + '/' + casei)
        # print('Dcm shape', dcm_img.shape)

        for arti in ['ICAL', 'ICAR']:
            cas_dir = input_dir + '/' + casei + '/CASCADE-' + arti
            qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)

            if len(avail_slices):
                lower_index = min(avail_slices)  # 正样本下界
                upper_index = max(avail_slices)  # 正样本上界

                pos_slice = list(range(lower_index, upper_index + 1))  # 所有正样本区间
                pos_not_avail = list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice = list(range(100, lower_index))
                neg_slice1 = list(range(upper_index + 1, 600))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in list(range(100, 600)):
                    if index in avail_slices:
                        dicom_slicei = index

                        lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                        wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                        background = np.zeros((height, width, 1), dtype=np.float32)

                        # mask circle
                        mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                        posLabel_save = mask_return[280:440, 110:610, 0]

                        np.save(save_path + '/' + arti + '_train/' + pi + '_' + arti + '_' + str(
                            index) + '_posLabel_.npy',
                                posLabel_save)
                    elif index in pos_not_avail:
                        posUn_save = np.zeros((160, 500), dtype=np.float32)
                        np.save(save_path + '/' + arti + '_train/' + pi + '_' + arti + '_' + str(
                            index) + '_posUn_.npy',
                                posUn_save)
                    else:
                        negLabel_save = np.zeros((160, 500), dtype=np.float32)
                        np.save(save_path + '/' + arti + '_train/' + pi + '_' + arti + '_' + str(
                            index) + '_negLabel_.npy',
                                negLabel_save)


def save_internal_test(input_dir, save_path):
    dir_create(save_path + '/' + 'ICAL_test')
    dir_create(save_path + '/' + 'ICAR_test')

    test_patience = ["P206", "P432", "P576", "P891"]  # 随意选出四个作为测试病例

    for casei in os.listdir(input_dir):
        pi = casei.split('_')[1]

        if pi not in test_patience:
            continue

        # print(input_dir + '/' + casei)
        dcm_img = read_dicom(input_dir + '/' + casei)
        # print('Dcm shape', dcm_img.shape)

        for arti in ['ICAL', 'ICAR']:
            cas_dir = input_dir + '/' + casei + '/CASCADE-' + arti
            qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)

            if len(avail_slices):
                lower_index = min(avail_slices)  # 正样本下界
                upper_index = max(avail_slices)  # 正样本上界

                pos_slice = list(range(lower_index, upper_index + 1))  # 所有正样本区间
                pos_not_avail = list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice = list(range(100, lower_index))
                neg_slice1 = list(range(upper_index + 1, 600))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in list(range(100, 600)):
                    if index in avail_slices:
                        dicom_slicei = index

                        lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                        wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                        background = np.zeros((height, width, 1), dtype=np.float32)

                        # mask circle
                        mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                        posLabel_save = mask_return[280:440, 110:610, 0]

                        np.save(save_path + '/' + arti + '_test/' + pi + '_' + arti + '_' + str(
                            index) + '_posLabel_.npy',
                                posLabel_save)
                    elif index in pos_not_avail:
                        posUn_save = np.zeros((160, 500), dtype=np.float32)
                        np.save(save_path + '/' + arti + '_test/' + pi + '_' + arti + '_' + str(
                            index) + '_posUn_.npy',
                                posUn_save)
                    else:
                        negLabel_save = np.zeros((160, 500), dtype=np.float32)
                        np.save(save_path + '/' + arti + '_test/' + pi + '_' + arti + '_' + str(
                            index) + '_negLabel_.npy',
                                negLabel_save)


def save_external_left_train(input_dir, save_path):
    dir_create(save_path + '/' + 'ECAL_train')

    test_patience = ["P429"]  # [P887,P429,P732,P556,P723,P530]

    for casei in os.listdir(input_dir):
        pi = casei.split('_')[1]

        if pi in test_patience:
            continue

        # print(input_dir + '/' + casei)
        dcm_img = read_dicom(input_dir + '/' + casei)
        # print('Dcm shape', dcm_img.shape)

        for arti in ['ECAL']:
            cas_dir = input_dir + '/' + casei + '/CASCADE-' + arti
            qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)

            if len(avail_slices):
                lower_index = min(avail_slices)  # 正样本下界
                upper_index = max(avail_slices)  # 正样本上界

                pos_slice = list(range(lower_index, upper_index + 1))  # 所有正样本区间
                pos_not_avail = list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice = list(range(200, lower_index))
                neg_slice1 = list(range(upper_index + 1, 400))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in list(range(200, 400)):
                    if index in avail_slices:
                        dicom_slicei = index

                        lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                        wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                        background = np.zeros((height, width, 1), dtype=np.float32)

                        # mask circle
                        mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                        posLabel_save = mask_return[280:440, 110:610, 0]

                        np.save(save_path + '/' + arti + '_train/' + pi + '_' + arti + '_' + str(
                            index) + '_posLabel_.npy',
                                posLabel_save)
                    elif index in pos_not_avail:
                        posUn_save = np.zeros((160, 500), dtype=np.float32)
                        np.save(save_path + '/' + arti + '_train/' + pi + '_' + arti + '_' + str(
                            index) + '_posUn_.npy',
                                posUn_save)
                    else:
                        negLabel_save = np.zeros((160, 500), dtype=np.float32)
                        np.save(save_path + '/' + arti + '_train/' + pi + '_' + arti + '_' + str(
                            index) + '_negLabel_.npy',
                                negLabel_save)


def save_external_left_test(input_dir, save_path):
    dir_create(save_path + '/' + 'ECAL_test')

    test_patience = ["P429"]  # [P887,P429,P732,P556,P723,P530]

    for casei in os.listdir(input_dir):
        pi = casei.split('_')[1]

        if pi not in test_patience:
            continue

        # print(input_dir + '/' + casei)
        dcm_img = read_dicom(input_dir + '/' + casei)
        # print('Dcm shape', dcm_img.shape)

        for arti in ['ECAL']:
            cas_dir = input_dir + '/' + casei + '/CASCADE-' + arti
            qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)

            if len(avail_slices):
                lower_index = min(avail_slices)  # 正样本下界
                upper_index = max(avail_slices)  # 正样本上界

                pos_slice = list(range(lower_index, upper_index + 1))  # 所有正样本区间
                pos_not_avail = list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice = list(range(200, lower_index))
                neg_slice1 = list(range(upper_index + 1, 400))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in list(range(200, 400)):
                    if index in avail_slices:
                        dicom_slicei = index

                        lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                        wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                        background = np.zeros((height, width, 1), dtype=np.float32)

                        # mask circle
                        mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                        posLabel_save = mask_return[280:440, 110:610, 0]

                        np.save(save_path + '/' + arti + '_test/' + pi + '_' + arti + '_' + str(
                            index) + '_posLabel_.npy',
                                posLabel_save)
                    elif index in pos_not_avail:
                        posUn_save = np.zeros((160, 500), dtype=np.float32)
                        np.save(save_path + '/' + arti + '_test/' + pi + '_' + arti + '_' + str(
                            index) + '_posUn_.npy',
                                posUn_save)
                    else:
                        negLabel_save = np.zeros((160, 500), dtype=np.float32)
                        np.save(save_path + '/' + arti + '_test/' + pi + '_' + arti + '_' + str(
                            index) + '_negLabel_.npy',
                                negLabel_save)


def save_external_right_train(input_dir, save_path):
    dir_create(save_path + '/' + 'ECAR_train')

    test_patience = ["P429"]  # [P887,P429,P732,P556,P723,P530]

    for casei in os.listdir(input_dir):
        pi = casei.split('_')[1]

        if pi in test_patience:
            continue

        # print(input_dir + '/' + casei)
        dcm_img = read_dicom(input_dir + '/' + casei)
        # print('Dcm shape', dcm_img.shape)

        for arti in ['ECAR']:
            cas_dir = input_dir + '/' + casei + '/CASCADE-' + arti
            qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)

            if len(avail_slices):
                lower_index = min(avail_slices)  # 正样本下界
                upper_index = max(avail_slices)  # 正样本上界

                pos_slice = list(range(lower_index, upper_index + 1))  # 所有正样本区间
                pos_not_avail = list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice = list(range(200, lower_index))
                neg_slice1 = list(range(upper_index + 1, 400))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in list(range(200, 400)):
                    if index in avail_slices:
                        dicom_slicei = index

                        lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                        wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                        background = np.zeros((height, width, 1), dtype=np.float32)

                        # mask circle
                        mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                        posLabel_save = mask_return[280:440, 110:610, 0]

                        np.save(save_path + '/' + arti + '_train/' + pi + '_' + arti + '_' + str(
                            index) + '_posLabel_.npy',
                                posLabel_save)
                    elif index in pos_not_avail:
                        posUn_save = np.zeros((160, 500), dtype=np.float32)
                        np.save(save_path + '/' + arti + '_train/' + pi + '_' + arti + '_' + str(
                            index) + '_posUn_.npy',
                                posUn_save)
                    else:
                        negLabel_save = np.zeros((160, 500), dtype=np.float32)
                        np.save(save_path + '/' + arti + '_train/' + pi + '_' + arti + '_' + str(
                            index) + '_negLabel_.npy',
                                negLabel_save)


def save_external_right_test(input_dir, save_path):
    dir_create(save_path + '/' + 'ECAR_test')

    test_patience = ["P429"]  # [P887,P429,P732,P556,P723,P530]

    for casei in os.listdir(input_dir):
        pi = casei.split('_')[1]

        if pi not in test_patience:
            continue

        # print(input_dir + '/' + casei)
        dcm_img = read_dicom(input_dir + '/' + casei)
        # print('Dcm shape', dcm_img.shape)

        for arti in ['ECAR']:
            cas_dir = input_dir + '/' + casei + '/CASCADE-' + arti
            qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)

            if len(avail_slices):
                lower_index = min(avail_slices)  # 正样本下界
                upper_index = max(avail_slices)  # 正样本上界

                pos_slice = list(range(lower_index, upper_index + 1))  # 所有正样本区间
                pos_not_avail = list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice = list(range(200, lower_index))
                neg_slice1 = list(range(upper_index + 1, 400))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in list(range(200, 400)):
                    if index in avail_slices:
                        dicom_slicei = index

                        lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                        wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                        background = np.zeros((height, width, 1), dtype=np.float32)

                        # mask circle
                        mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                        posLabel_save = mask_return[280:440, 110:610, 0]

                        np.save(save_path + '/' + arti + '_test/' + pi + '_' + arti + '_' + str(
                            index) + '_posLabel_.npy',
                                posLabel_save)
                    elif index in pos_not_avail:
                        posUn_save = np.zeros((160, 500), dtype=np.float32)
                        np.save(save_path + '/' + arti + '_test/' + pi + '_' + arti + '_' + str(
                            index) + '_posUn_.npy',
                                posUn_save)
                    else:
                        negLabel_save = np.zeros((160, 500), dtype=np.float32)
                        np.save(save_path + '/' + arti + '_test/' + pi + '_' + arti + '_' + str(
                            index) + '_negLabel_.npy',
                                negLabel_save)


def main(args):
    input_dir = args.datasets_path
    save_path = args.mask_semi

    save_internal_train(input_dir, save_path)
    save_internal_test(input_dir, save_path)

    save_external_left_train(input_dir, save_path)
    save_external_left_test(input_dir, save_path)

    save_external_right_train(input_dir, save_path)
    save_external_right_test(input_dir, save_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
