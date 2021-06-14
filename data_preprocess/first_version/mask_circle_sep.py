# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/03/13 19:46
# @Author  : yilinzhi
# @FileName: mask_circle_sep.py


""""目标：25个不同病例，不同位置的label都进行mask,
    1.25个病例不同位置label的avail_slice
    2.用cv2工具
"""

import os
import cv2
import shutil
import pydicom
import glob
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


# A useful function for creating a new directory and
# recursively deleting the contents of an existing one:
def dir_create(path):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)


def readDicom(path):
    print(os.path.basename(path))
    pi = os.path.basename(path).split("_")[1]
    dcm_size = len(glob.glob(path + "/*.dcm"))
    dcms = [
        path + "/E" + pi + "S101I%d.dcm" % dicom_slicei
        for dicom_slicei in range(1, dcm_size + 1)
    ]

    length = int(len(dcms))
    print(length)

    # 读取dcm文档并且转换成numpy.ndarray格式
    dcm_f = pydicom.read_file(dcms[0]).pixel_array
    dcm_size = max(max(dcm_f.shape), 720)

    # 将一个案例中所有切片堆叠起来->h,w,s：(720, 720, 720)
    dcm_img = np.zeros((dcm_size, dcm_size, dcm_size), dtype=np.float32)

    for dcmi in range(len(dcms)):
        cdcm = pydicom.read_file(dcms[dcmi]).pixel_array
        dcm_img[
        dcm_size // 2 - cdcm.shape[0] // 2: dcm_size // 2 + cdcm.shape[0] // 2,
        dcm_size // 2 - cdcm.shape[1] // 2: dcm_size // 2 + cdcm.shape[1] // 2,
        dcmi,
        ] = cdcm

    return dcm_img


def listContourSlices(qvsroot):
    avail_slices = []
    qvasimg = qvsroot.findall('QVAS_Image')
    for dicom_slicei in range(dcm_img.shape[2]):
        conts = qvasimg[dicom_slicei - 1].findall('QVAS_Contour')
        if len(conts):
            avail_slices.append(dicom_slicei)
    return avail_slices


def listContourSlices_min(qvsroot):
    avail_slices = []

    # QVAS_Image是一个列表，长度=dcm_img.shape[2]
    qvasimg = qvsroot.findall("QVAS_Image")
    # print(qvasimg)
    for dicom_slicei in range(640):
        conts = qvasimg[dicom_slicei - 1].findall("QVAS_Contour")
        #         print(conts)
        if len(conts):
            avail_slices.append(dicom_slicei)
    return avail_slices


# luman/out waller contour of a slice, dtype=np.float32
# 读取一个病例中avail_slices中contour points
def getContour(qvsroot, dicomslicei, conttype, dcmsz=720):
    qvasimg = qvsroot.findall("QVAS_Image")

    if dicomslicei - 1 > len(qvasimg):
        print("no slice", dicomslicei)
        return

    assert int(qvasimg[dicomslicei - 1].get("ImageName").split("I")[-1]) == dicomslicei

    # 特定的一个avial slice
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
    return np.array(contours, dtype=np.int)


height = 720
width = 720

# 此时的mask是一个环，灰度值为255为了直接可视化
def create_mask_file(background, contours_i,contours_e,channels):

    mask = np.full((height, width, channels), background, dtype=np.float32)
    mask = cv2.drawContours(mask, [contours_i.astype(int)], -1, color=(1), thickness=2)
    mask = cv2.drawContours(mask, [contours_e.astype(int)], -1, color=(1), thickness=2)
    mask = cv2.fillPoly(mask, [contours_e.astype(int)], color=(1))
    mask = cv2.fillPoly(mask, [contours_i.astype(int)], color=(0))
    return mask


# the original path
cdir = '/data/yilinzhi/Segmentation/VMS/datasets/careIIChallenge'
odir = '/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_png'


dir_create(odir+'/'+'ICAL')
dir_create(odir+'/'+'ICAR')
dir_create(odir+'/'+'ECAL')
dir_create(odir+'/'+'ECAR')

for casei in os.listdir(cdir):
    pi = casei.split('_')[1]
    print(cdir + '/' + casei)
    dcm_img = readDicom(cdir + '/' + casei)
    print('Dcm shape', dcm_img.shape)

    # 每个案例不同位置的血管有不同的avail_slice，每个病人四个label，每个label对应两个contour
    avail_slices_all = []
    avail_slices_union = []
    lumen_cont_all = []
    wall_cont_all = []

    for arti in ['ICAL', 'ICAR', 'ECAL', 'ECAR']:
        cas_dir = cdir + '/' + casei + '/CASCADE-' + arti
        qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
        qvsroot = ET.parse(qvs_path).getroot()

        if pi in ["P556", "P576", "P887"]:
            avail_slices = listContourSlices_min(qvsroot)
        else:
            avail_slices = listContourSlices(qvsroot)
        print("case", pi, "art", arti, "avail_slices", avail_slices)

        # 开始对一个案例的每个有用层进行mask
        if len(avail_slices):
            for index in range(len(avail_slices)):
                dicom_slicei=avail_slices[index]

                lumen_cont = getContour(qvsroot, dicom_slicei, 'Lumen')
                wall_cont = getContour(qvsroot, dicom_slicei, 'Outer Wall')

                background = np.zeros((height, width, 1), dtype=np.float32)
                mask_save=np.zeros((160,500),dtype=np.float32)

                mask=create_mask_file(background,lumen_cont,wall_cont,1)
                mask_save=mask[280:440, 110:610,:]
                mask_save=np.squeeze(mask_save)
                print(mask_save.shape)

                cv2.imwrite(odir + '/' +arti+'/'+ pi + '_' + arti+'_' + str(avail_slices[index])+'_.png', mask_save)




