# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/03/15 10:53
# @Author  : yilinzhi
# @FileName: image_union.py

"""去除没有标记的slices，并对所有avail_slices进行高斯标准化,进行了相应的裁剪"""

import os
import pydicom
import glob
import numpy as np
import shutil
import xml.etree.ElementTree as ET
from data_parameter import parse_args


# 创建文件夹
def dir_create(path):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)


# 读取一个病例中所有的dcm文件，然后堆叠起来：height, width, slices
def readDicom(path):
    # print(os.path.basename(path))
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

    # len(dcms)=640/720
    for dcmi in range(len(dcms)):
        """是否要找到一个阈值，然后在对特定区域进行高斯标准化???????"""
        # 读取一个案例的每一层 并作高斯标准化
        cdcm = pydicom.read_file(dcms[dcmi]).pixel_array.astype(np.float32)
        cdcm -= np.mean(cdcm > 0)
        cdcm /= np.std(cdcm > 0)

        dcm_img[
        dcm_size // 2 - cdcm.shape[0] // 2: dcm_size // 2 + cdcm.shape[0] // 2,
        dcm_size // 2 - cdcm.shape[1] // 2: dcm_size // 2 + cdcm.shape[1] // 2,
        dcmi,
        ] = cdcm

    return dcm_img


# 读取一个病例中有label的avail_slices
def listContourSlices(qvsroot):
    avail_slices = []

    # QVAS_Image是一个列表，长度=dcm_img.shape[2]
    qvasimg = qvsroot.findall("QVAS_Image")

    # avail_slices的index是从0开始的
    """为什么dicom_slicei-1？？？？？？"""
    for dicom_slicei in range(720):
        conts = qvasimg[dicom_slicei - 1].findall("QVAS_Contour")

        if len(conts):
            avail_slices.append(dicom_slicei)
    return avail_slices


# 几个特殊病例：640
def listContourSlices_min(qvsroot):
    avail_slices = []

    # 查找QVAS_Image属性，长度=640，几个特殊的案例
    qvasimg = qvsroot.findall("QVAS_Image")

    for dicom_slicei in range(640):
        conts = qvasimg[dicom_slicei - 1].findall("QVAS_Contour")

        if len(conts):
            avail_slices.append(dicom_slicei)
    return avail_slices


# 读取一个病例中avail_slices中contour points
def getContour(qvsroot, dicomslicei, conttype, dcmsz=720):
    qvasimg = qvsroot.findall("QVAS_Image")

    if dicomslicei - 1 > len(qvasimg):
        print("no slice", dicomslicei)
        return

    assert int(qvasimg[dicomslicei - 1].get("ImageName").split("I")[-1]) == dicomslicei

    ###############################
    # 如何理解dicomslice-1???????
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


# 暂时没有保存负数样本
def createTrainData(input_dir, save_path):
    for casei in os.listdir(input_dir):
        pi = casei.split("_")[1]
        # 读取dcm_img数据：720*720*720
        dcm_img = readDicom(input_dir + "/" + casei)
        print("Dcm shape: ", dcm_img.shape)

        avail_slices_all = []
        # 四个label
        for arti in ["ICAL", "ICAR", "ECAL", "ECAR"]:
            cas_dir = input_dir + "/" + casei + "/CASCADE-" + arti
            qvs_path = cas_dir + "/E" + pi + "S101_L.QVS"
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = listContourSlices_min(qvsroot)
            else:
                avail_slices = listContourSlices(qvsroot)
            print("case", pi, "art", arti, "avail_slices", avail_slices)

            avail_slices_all.append(avail_slices)

        avail_slices_union = list(
            set(avail_slices_all[0]).union(avail_slices_all[1], avail_slices_all[2], avail_slices_all[3]))
        avail_slices_union.sort()
        print("avail_slices_union_sorted:", avail_slices_union)

        for index in range(len(avail_slices_union)):
            # 需要逐个对输入图片进行标准化
            img = dcm_img[:, :, avail_slices_union[index]]

            img_save = img[280:440, 110:610]
            print(avail_slices_union[index])
            print(img_save.shape)

            np.save(save_path + '/' + pi + '_' + str(avail_slices_union[index]) + '_.npy', img_save)


def main(args):
    input_dir = args.datasets_path
    save_path = args.image_save_union_slices

    dir_create(save_path)
    createTrainData(input_dir, save_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
