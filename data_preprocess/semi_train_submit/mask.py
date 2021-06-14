# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/05/17 17:16
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
    qvasimg = qvsroot.findall('QVAS_Image')   #
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


def save_internal(input_dir, save_path):

    dir_create(save_path + '/train/ICAL')
    dir_create(save_path + '/train/ICAR')
    dir_create(save_path + '/test/ICAL')
    dir_create(save_path + '/test/ICAR')

    test_patience = ["P206", "P432", "P576", "P891"]

    for casei in os.listdir(input_dir):
        pi = casei.split('_')[1]

        if pi in test_patience: # 跳过测试样例
            continue

        print(input_dir + '/' + casei)
        dcm_img = read_dicom(input_dir + '/' + casei)
        print('Dcm shape', dcm_img.shape)

        for arti in ['ICAL', 'ICAR']:
            cas_dir = input_dir + '/' + casei + '/CASCADE-' + arti
            qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
            qvsroot = ET.parse(qvs_path).getroot()  # 获取根元素

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)
            # 存储不同位置对应的mask
            if len(avail_slices):
                lower_index=min(avail_slices)  # 正样本下界
                upper_index=max(avail_slices)  # 正样本上界

                pos_slice=list(range(lower_index,upper_index+1))  # 所有正样本区间
                pos_not_avail=list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice=list(range(100,lower_index))
                neg_slice1=list(range(upper_index+1,600))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in range(len(avail_slices)):  # 存储正样本+标签：环
                    dicom_slicei = avail_slices[index]

                    lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                    wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                    background = np.zeros((height, width, 1), dtype=np.float32)

                    # mask circle
                    mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                    mask_save = mask_return[280:440, 110:610, 0]

                    np.save(save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        avail_slices[index]) + '_posLabel_.npy',
                            mask_save)

                fake_mask_index=[] # 寻找最近的伪标签
                for index in pos_not_avail: # 找出没有标签的正样本，对用的fake label的index
                    close_index=1000
                    best_index=-1
                    for idx in avail_slices:
                        if abs(index-idx)<close_index:
                            close_index=abs(index-idx)
                            best_index=idx
                    fake_mask_index.append(best_index)

                for index in range(len(pos_not_avail)):  # 存储正样本+无标签
                    dicom_slicei = fake_mask_index[index]
                    lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                    wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                    background = np.zeros((height, width, 1), dtype=np.float32)

                    mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                    mask_save = mask_return[280:440, 110:610, 0]

                    # 同时保存了真实的slice，和最近fake slice
                    np.save(save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        pos_not_avail[index])+ '_'+str(fake_mask_index[index])+ '_posUnLabel_.npy',
                            mask_save)

                for index in neg_slice: # 存储负样本
                    neg_mask_save = np.zeros((160, 500), dtype=np.float32)
                    np.save(save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_negLabel_.npy',
                            neg_mask_save)

    for casei in os.listdir(input_dir):
        pi = casei.split('_')[1]

        if pi not in test_patience:  # 跳过训练集
            continue

        print(input_dir + '/' + casei)
        dcm_img = read_dicom(input_dir + '/' + casei)
        print('Dcm shape', dcm_img.shape)

        for arti in ['ICAL', 'ICAR']:
            cas_dir = input_dir + '/' + casei + '/CASCADE-' + arti
            qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)
            # 存储不同位置对应的mask
            if len(avail_slices):
                lower_index = min(avail_slices)  # 正样本下界
                upper_index = max(avail_slices)  # 正样本上界

                pos_slice = list(range(lower_index, upper_index + 1))  # 所有正样本区间
                pos_not_avail = list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice = list(range(100, lower_index))
                neg_slice1 = list(range(upper_index + 1, 600))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in range(len(avail_slices)):  # 存储正样本+标签：环
                    dicom_slicei = avail_slices[index]

                    lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                    wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                    background = np.zeros((height, width, 1), dtype=np.float32)

                    # mask circle
                    mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                    mask_save = mask_return[280:440, 110:610, 0]

                    np.save(save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        avail_slices[index]) + '_posLabel_.npy',
                            mask_save)

                fake_mask_index = []  # 寻找最近的伪标签
                for index in pos_not_avail:  # 找出没有标签的正样本，对用的fake label的index
                    close_index = 1000
                    best_index = -1
                    for idx in avail_slices:
                        if abs(index - idx) < close_index:
                            close_index = abs(index - idx)
                            best_index = idx
                    fake_mask_index.append(best_index)

                for index in range(len(pos_not_avail)):  # 存储正样本+无标签
                    dicom_slicei = fake_mask_index[index]
                    lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                    wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                    background = np.zeros((height, width, 1), dtype=np.float32)

                    mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                    mask_save = mask_return[280:440, 110:610, 0]

                    # 同时保存了真实的slice，和最近fake slice
                    np.save(save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        pos_not_avail[index]) + '_' + str(fake_mask_index[index]) + '_posUnLabel_.npy',
                            mask_save)

                for index in neg_slice:  # 存储负样本
                    neg_mask_save = np.zeros((160, 500), dtype=np.float32)
                    np.save(save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_negLabel_.npy',
                            neg_mask_save)


def save_external_left(input_dir, save_path):

    dir_create(save_path + '/train/ECAL')
    dir_create(save_path + '/test/ECAL')

    test_patience = ["P429"]

    for casei in os.listdir(input_dir):
        pi = casei.split('_')[1]

        if pi in test_patience: # 跳过测试样例
            continue

        print(input_dir + '/' + casei)
        dcm_img = read_dicom(input_dir + '/' + casei)
        print('Dcm shape', dcm_img.shape)

        for arti in ['ECAL']:
            cas_dir = input_dir + '/' + casei + '/CASCADE-' + arti
            qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)
            # 存储不同位置对应的mask
            if len(avail_slices):
                lower_index=min(avail_slices)  # 正样本下界
                upper_index=max(avail_slices)  # 正样本上界

                pos_slice=list(range(lower_index,upper_index+1))  # 所有正样本区间
                pos_not_avail=list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice=list(range(200,lower_index))
                neg_slice1=list(range(upper_index+1,400))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in range(len(avail_slices)):  # 存储正样本+标签：环
                    dicom_slicei = avail_slices[index]

                    lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                    wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                    background = np.zeros((height, width, 1), dtype=np.float32)

                    # mask circle
                    mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                    mask_save = mask_return[280:440, 110:610, 0]

                    np.save(save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        avail_slices[index]) + '_posLabel_.npy',
                            mask_save)

                fake_mask_index=[] # 寻找最近的伪标签
                for index in pos_not_avail: # 找出没有标签的正样本，对用的fake label的index
                    close_index=1000
                    best_index=-1
                    for idx in avail_slices:
                        if abs(index-idx)<close_index:
                            close_index=abs(index-idx)
                            best_index=idx
                    fake_mask_index.append(best_index)

                for index in range(len(pos_not_avail)):  # 存储正样本+无标签
                    dicom_slicei = fake_mask_index[index]
                    lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                    wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                    background = np.zeros((height, width, 1), dtype=np.float32)

                    mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                    mask_save = mask_return[280:440, 110:610, 0]

                    # 同时保存了真实的slice，和最近fake slice
                    np.save(save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        pos_not_avail[index])+ '_'+str(fake_mask_index[index])+ '_posUnLabel_.npy',
                            mask_save)

                for index in neg_slice: # 存储负样本
                    neg_mask_save = np.zeros((160, 500), dtype=np.float32)
                    np.save(save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_negLabel_.npy',
                            neg_mask_save)

    for casei in os.listdir(input_dir):
        pi = casei.split('_')[1]

        if pi not in test_patience:  # 跳过训练样例
            continue

        print(input_dir + '/' + casei)
        dcm_img = read_dicom(input_dir + '/' + casei)
        print('Dcm shape', dcm_img.shape)

        for arti in ['ECAL']:
            cas_dir = input_dir + '/' + casei + '/CASCADE-' + arti
            qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)
            # 存储不同位置对应的mask
            if len(avail_slices):
                lower_index = min(avail_slices)  # 正样本下界
                upper_index = max(avail_slices)  # 正样本上界

                pos_slice = list(range(lower_index, upper_index + 1))  # 所有正样本区间
                pos_not_avail = list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice = list(range(200, lower_index))
                neg_slice1 = list(range(upper_index + 1, 400))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in range(len(avail_slices)):  # 存储正样本+标签：环
                    dicom_slicei = avail_slices[index]

                    lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                    wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                    background = np.zeros((height, width, 1), dtype=np.float32)

                    # mask circle
                    mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                    mask_save = mask_return[280:440, 110:610, 0]

                    np.save(save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        avail_slices[index]) + '_posLabel_.npy',
                            mask_save)

                fake_mask_index = []  # 寻找最近的伪标签
                for index in pos_not_avail:  # 找出没有标签的正样本，对用的fake label的index
                    close_index = 1000
                    best_index = -1
                    for idx in avail_slices:
                        if abs(index - idx) < close_index:
                            close_index = abs(index - idx)
                            best_index = idx
                    fake_mask_index.append(best_index)

                for index in range(len(pos_not_avail)):  # 存储正样本+无标签
                    dicom_slicei = fake_mask_index[index]
                    lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                    wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                    background = np.zeros((height, width, 1), dtype=np.float32)

                    mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                    mask_save = mask_return[280:440, 110:610, 0]

                    # 同时保存了真实的slice，和最近fake slice
                    np.save(save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        pos_not_avail[index]) + '_' + str(fake_mask_index[index]) + '_posUnLabel_.npy',
                            mask_save)

                for index in neg_slice:  # 存储负样本
                    neg_mask_save = np.zeros((160, 500), dtype=np.float32)
                    np.save(save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_negLabel_.npy',
                            neg_mask_save)

def save_external_right(input_dir, save_path):

    dir_create(save_path + '/train/ECAR')
    dir_create(save_path + '/test/ECAR')

    test_patience = ["P429"]

    for casei in os.listdir(input_dir):
        pi = casei.split('_')[1]

        if pi in test_patience: # 跳过测试样例
            continue

        print(input_dir + '/' + casei)
        dcm_img = read_dicom(input_dir + '/' + casei)
        print('Dcm shape', dcm_img.shape)

        for arti in ['ECAR']:
            cas_dir = input_dir + '/' + casei + '/CASCADE-' + arti
            qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)
            # 存储不同位置对应的mask
            if len(avail_slices):
                lower_index=min(avail_slices)  # 正样本下界
                upper_index=max(avail_slices)  # 正样本上界

                pos_slice=list(range(lower_index,upper_index+1))  # 所有正样本区间
                pos_not_avail=list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice=list(range(200,lower_index))
                neg_slice1=list(range(upper_index+1,400))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in range(len(avail_slices)):  # 存储正样本+标签：环
                    dicom_slicei = avail_slices[index]

                    lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                    wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                    background = np.zeros((height, width, 1), dtype=np.float32)

                    # mask circle
                    mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                    mask_save = mask_return[280:440, 110:610, 0]

                    np.save(save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        avail_slices[index]) + '_posLabel_.npy',
                            mask_save)

                fake_mask_index=[] # 寻找最近的伪标签
                for index in pos_not_avail: # 找出没有标签的正样本，对用的fake label的index
                    close_index=1000
                    best_index=-1
                    for idx in avail_slices:
                        if abs(index-idx)<close_index:
                            close_index=abs(index-idx)
                            best_index=idx
                    fake_mask_index.append(best_index)

                for index in range(len(pos_not_avail)):  # 存储正样本+无标签
                    dicom_slicei = fake_mask_index[index]
                    lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                    wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                    background = np.zeros((height, width, 1), dtype=np.float32)

                    mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                    mask_save = mask_return[280:440, 110:610, 0]

                    # 同时保存了真实的slice，和最近fake slice
                    np.save(save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        pos_not_avail[index])+ '_'+str(fake_mask_index[index])+ '_posUnLabel_.npy',
                            mask_save)

                for index in neg_slice: # 存储负样本
                    neg_mask_save = np.zeros((160, 500), dtype=np.float32)
                    np.save(save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_negLabel_.npy',
                            neg_mask_save)

    for casei in os.listdir(input_dir):
        pi = casei.split('_')[1]

        if pi not in test_patience:  # 跳过训练样例
            continue

        print(input_dir + '/' + casei)
        dcm_img = read_dicom(input_dir + '/' + casei)
        print('Dcm shape', dcm_img.shape)

        for arti in ['ECAR']:
            cas_dir = input_dir + '/' + casei + '/CASCADE-' + arti
            qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)
            # 存储不同位置对应的mask
            if len(avail_slices):
                lower_index = min(avail_slices)  # 正样本下界
                upper_index = max(avail_slices)  # 正样本上界

                pos_slice = list(range(lower_index, upper_index + 1))  # 所有正样本区间
                pos_not_avail = list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice = list(range(200, lower_index))
                neg_slice1 = list(range(upper_index + 1, 400))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in range(len(avail_slices)):  # 存储正样本+标签：环
                    dicom_slicei = avail_slices[index]

                    lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                    wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                    background = np.zeros((height, width, 1), dtype=np.float32)

                    # mask circle
                    mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                    mask_save = mask_return[280:440, 110:610, 0]

                    np.save(save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        avail_slices[index]) + '_posLabel_.npy',
                            mask_save)

                fake_mask_index = []  # 寻找最近的伪标签
                for index in pos_not_avail:  # 找出没有标签的正样本，对用的fake label的index
                    close_index = 1000
                    best_index = -1
                    for idx in avail_slices:
                        if abs(index - idx) < close_index:
                            close_index = abs(index - idx)
                            best_index = idx
                    fake_mask_index.append(best_index)

                for index in range(len(pos_not_avail)):  # 存储正样本+无标签
                    dicom_slicei = fake_mask_index[index]
                    lumen_cont = get_contour(qvsroot, dicom_slicei, 'Lumen')
                    wall_cont = get_contour(qvsroot, dicom_slicei, 'Outer Wall')
                    background = np.zeros((height, width, 1), dtype=np.float32)

                    mask_return = create_mask_circle_file(background, lumen_cont, wall_cont, 1)
                    mask_save = mask_return[280:440, 110:610, 0]

                    # 同时保存了真实的slice，和最近fake slice
                    np.save(save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        pos_not_avail[index]) + '_' + str(fake_mask_index[index]) + '_posUnLabel_.npy',
                            mask_save)

                for index in neg_slice:  # 存储负样本
                    neg_mask_save = np.zeros((160, 500), dtype=np.float32)
                    np.save(save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_negLabel_.npy',
                            neg_mask_save)


def main(args):
    input_dir = args.datasets_path
    save_path = args.mask_semi

    save_internal(input_dir, save_path)
    save_external_left(input_dir,save_path)
    save_external_right(input_dir,save_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)