# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/05/17 16:07
# @Author  : Yi
# @FileName: image_F.py

"""
    按照病例划分"训练集+验证集"，"测试集"→分开存储
"""


import os
import pydicom
import glob
import shutil
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
    """读取一个案例，并返回整个案列的numpy文件。"""

    print(os.path.basename(path))
    pi = os.path.basename(path).split("_")[1]
    dcm_size = len(glob.glob(path + "/*.dcm"))  #找到病例下所有的.dcm文件

    dcms = [
        path + "/E" + pi + "S101I%d.dcm" % dicom_slicei
        for dicom_slicei in range(1, dcm_size + 1)
    ]  # dcm路径列表

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
    """找到有标注的slices序号。"""

    avail_slices = []
    qvasimg = qvsroot.findall("QVAS_Image")  # QVAS_Image is a list，length=dcm_img.shape[2]

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
    """返回lumen/contour标注点的坐标：[[x,y]]"""

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


def save_internal(data_input_dir, image_save_path):
    """Store internal samples."""

    create_dir(image_save_path + '/train/ICAL')
    create_dir(image_save_path + '/train/ICAR')
    create_dir(image_save_path + '/test/ICAL')
    create_dir(image_save_path + '/test/ICAR')

    test_patience=["P206","P432","P576","P891"]  # 随意选出四个病例作为internal测试病例

    for casei in os.listdir(data_input_dir):  # 存储训练集
        pi = casei.split("_")[1]

        if pi in test_patience: # 跳过测试样例
            continue

        print(data_input_dir + '/' + casei)
        dcm_img = read_dicom(data_input_dir + "/" + casei)
        print("Dcm shape: ", dcm_img.shape)

        for arti in ["ICAL", "ICAR"]:  # 处理internal处血管：ICAL/ICAR
            cas_dir = data_input_dir + "/" + casei + "/CASCADE-" + arti
            qvs_path = cas_dir + "/E" + pi + "S101_L.QVS"
            qvsroot = ET.parse(qvs_path).getroot()  #读取xml文件并获取根元素

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)  # 正样本+标签：slices
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)

            if len(avail_slices):  # 只有当该位置的样本存在时候，才保存对应的正负样本，internal必定存在，external有可能不存在
                lower_index = min(avail_slices)  # 正样本下界
                upper_index = max(avail_slices)  # 正样本上界

                pos_slice = list(range(lower_index, upper_index + 1))  # 所有正样本区间
                pos_not_avail = list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice = list(range(100, lower_index))
                neg_slice1 = list(range(upper_index + 1, 600))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in avail_slices:  # 存储：正样本+有标签：样本
                    img_save = dcm_img[280:440, 110:610, index]
                    # print(img_save.shape)

                    np.save(image_save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_posLabel_.npy',
                            img_save)

                fake_mask_index = []  # 寻找最近的伪标签
                for index in pos_not_avail:  # 找出没有标签的正样本，对用的fake label的index
                    close_index = 1000
                    best_index = -1
                    for idx in avail_slices:
                        if abs(index - idx) < close_index:
                            close_index = abs(index - idx)
                            best_index = idx
                    fake_mask_index.append(best_index)

                for index in range(len(pos_not_avail)):  # 存储正样本+没有标签
                    img_save = dcm_img[280:440, 110:610, pos_not_avail[index]]
                    np.save(image_save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        pos_not_avail[index]) + '_' + str(fake_mask_index[index]) + '_posUnLabel_.npy',
                            img_save)

                for index in neg_slice:  # 存储负样本
                    img_save = dcm_img[280:440, 110:610, index]
                    np.save(image_save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_negLabel_.npy',
                            img_save)

    for casei in os.listdir(data_input_dir):  # 存储测试集
        pi = casei.split("_")[1]

        if pi not in test_patience:
            continue

        print(data_input_dir + '/' + casei)
        dcm_img = read_dicom(data_input_dir + "/" + casei)
        print("Dcm shape: ", dcm_img.shape)

        for arti in ["ICAL", "ICAR"]:  # 分成内外，处理
            cas_dir = data_input_dir + "/" + casei + "/CASCADE-" + arti
            qvs_path = cas_dir + "/E" + pi + "S101_L.QVS"
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)  # 正样本+标签：slices
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)

            if len(avail_slices):  # 只有当该位置的样本存在时候，才保存对应的正负样本，internal必定存在，external有可能不存在
                lower_index = min(avail_slices)  # 正样本下界
                upper_index = max(avail_slices)  # 正样本上界

                pos_slice = list(range(lower_index, upper_index + 1))  # 所有正样本区间
                pos_not_avail = list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice = list(range(100, lower_index))
                neg_slice1 = list(range(upper_index + 1, 600))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in avail_slices:  # 存储：正样本+有标签：样本
                    img_save = dcm_img[280:440, 110:610, index]
                    # print(img_save.shape)

                    np.save(image_save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_posLabel_.npy',
                            img_save)

                fake_mask_index = []  # 寻找最近的伪标签
                for index in pos_not_avail:  # 找出没有标签的正样本，对用的fake label的index
                    close_index = 1000
                    best_index = -1
                    for idx in avail_slices:
                        if abs(index - idx) < close_index:
                            close_index = abs(index - idx)
                            best_index = idx
                    fake_mask_index.append(best_index)

                for index in range(len(pos_not_avail)):  # 存储正样本+没有标签
                    img_save = dcm_img[280:440, 110:610, pos_not_avail[index]]
                    np.save(image_save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        pos_not_avail[index]) + '_' + str(fake_mask_index[index]) + '_posUnLabel_.npy',
                            img_save)

                for index in neg_slice:  # 存储负样本
                    img_save = dcm_img[280:440, 110:610, index]
                    np.save(image_save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_negLabel_.npy',
                            img_save)


def save_external_left(data_input_dir, image_save_path):
    """Store external left samples"""

    create_dir(image_save_path + '/train/ECAL')
    create_dir(image_save_path + '/test/ECAL')

    test_patience=["P429"]  # ECAL: [P887,P429,P732,P556,P723,P530]

    for casei in os.listdir(data_input_dir):  # 处理每个样本
        pi = casei.split("_")[1]

        if pi in test_patience:
            continue

        print(data_input_dir + '/' + casei)
        dcm_img = read_dicom(data_input_dir + "/" + casei)
        print("Dcm shape: ", dcm_img.shape)

        for arti in ["ECAL"]:  # 处理外部左边
            cas_dir = data_input_dir + "/" + casei + "/CASCADE-" + arti
            qvs_path = cas_dir + "/E" + pi + "S101_L.QVS"
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)  # 正样本+标签：slices
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)

            if len(avail_slices):  # 只有当该位置的样本存在时候，才保存对应的正负样本，internal必定存在，external有可能不存在
                lower_index = min(avail_slices)  # 正样本下界
                upper_index = max(avail_slices)  # 正样本上界

                pos_slice = list(range(lower_index, upper_index + 1))  # 所有正样本区间
                pos_not_avail = list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice = list(range(200, lower_index))
                neg_slice1 = list(range(upper_index + 1, 400))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in avail_slices:  # 存储：正样本+有标签：样本
                    img_save = dcm_img[280:440, 110:610, index]
                    # print(img_save.shape)

                    np.save(image_save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_posLabel_.npy',
                            img_save)

                fake_mask_index = []  # 寻找最近的伪标签
                for index in pos_not_avail:  # 找出没有标签的正样本，对用的fake label的index
                    close_index = 1000
                    best_index = -1
                    for idx in avail_slices:
                        if abs(index - idx) < close_index:
                            close_index = abs(index - idx)
                            best_index = idx
                    fake_mask_index.append(best_index)

                for index in range(len(pos_not_avail)):  # 存储正样本+没有标签
                    img_save = dcm_img[280:440, 110:610, pos_not_avail[index]]
                    np.save(image_save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        pos_not_avail[index]) + '_' + str(fake_mask_index[index]) + '_posUnLabel_.npy',
                            img_save)

                for index in neg_slice:  # 存储负样本
                    img_save = dcm_img[280:440, 110:610, index]
                    np.save(image_save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_negLabel_.npy',
                            img_save)

    for casei in os.listdir(data_input_dir):  # 处理每个样本
        pi = casei.split("_")[1]

        if pi not in test_patience:
            continue

        print(data_input_dir + '/' + casei)
        dcm_img = read_dicom(data_input_dir + "/" + casei)
        # print("Dcm shape: ", dcm_img.shape)

        for arti in ["ECAL"]:  # 处理外部左边
            cas_dir = data_input_dir + "/" + casei + "/CASCADE-" + arti
            qvs_path = cas_dir + "/E" + pi + "S101_L.QVS"
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)  # 正样本+标签：slices
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)

            if len(avail_slices):  # 只有当该位置的样本存在时候，才保存对应的正负样本，internal必定存在，external有可能不存在
                lower_index = min(avail_slices)  # 正样本下界
                upper_index = max(avail_slices)  # 正样本上界

                pos_slice = list(range(lower_index, upper_index + 1))  # 所有正样本区间
                pos_not_avail = list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice = list(range(200, lower_index))
                neg_slice1 = list(range(upper_index + 1, 400))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in avail_slices:  # 存储：正样本+有标签：样本
                    img_save = dcm_img[280:440, 110:610, index]
                    # print(img_save.shape)

                    np.save(image_save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_posLabel_.npy',
                            img_save)

                fake_mask_index = []  # 寻找最近的伪标签
                for index in pos_not_avail:  # 找出没有标签的正样本，对用的fake label的index
                    close_index = 1000
                    best_index = -1
                    for idx in avail_slices:
                        if abs(index - idx) < close_index:
                            close_index = abs(index - idx)
                            best_index = idx
                    fake_mask_index.append(best_index)

                for index in range(len(pos_not_avail)):  # 存储正样本+没有标签
                    img_save = dcm_img[280:440, 110:610, pos_not_avail[index]]
                    np.save(image_save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        pos_not_avail[index]) + '_' + str(fake_mask_index[index]) + '_posUnLabel_.npy',
                            img_save)

                for index in neg_slice:  # 存储负样本
                    img_save = dcm_img[280:440, 110:610, index]
                    np.save(image_save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_negLabel_.npy',
                            img_save)

def save_external_right(data_input_dir, image_save_path):
    """Store external right samples"""

    create_dir(image_save_path + '/train/ECAR')
    create_dir(image_save_path + '/test/ECAR')

    test_patience=["P429"] # [P887,P891,P429,P438,P125,P530]

    for casei in os.listdir(data_input_dir):  # 处理每个样本
        pi = casei.split("_")[1]

        if pi in test_patience:
            continue

        print(data_input_dir + '/' + casei)
        dcm_img = read_dicom(data_input_dir + "/" + casei)
        # print("Dcm shape: ", dcm_img.shape)

        for arti in ["ECAR"]:  # 处理外部左边
            cas_dir = data_input_dir + "/" + casei + "/CASCADE-" + arti
            qvs_path = cas_dir + "/E" + pi + "S101_L.QVS"
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)  # 正样本+标签：slices
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)

            if len(avail_slices):  # 只有当该位置的样本存在时候，才保存对应的正负样本，internal必定存在，external有可能不存在
                lower_index = min(avail_slices)  # 正样本下界
                upper_index = max(avail_slices)  # 正样本上界

                pos_slice = list(range(lower_index, upper_index + 1))  # 所有正样本区间
                pos_not_avail = list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice = list(range(200, lower_index))
                neg_slice1 = list(range(upper_index + 1, 400))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in avail_slices:  # 存储：正样本+有标签：样本
                    img_save = dcm_img[280:440, 110:610, index]
                    # print(img_save.shape)

                    np.save(image_save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_posLabel_.npy',
                            img_save)

                fake_mask_index = []  # 寻找最近的伪标签
                for index in pos_not_avail:  # 找出没有标签的正样本，对用的fake label的index
                    close_index = 1000
                    best_index = -1
                    for idx in avail_slices:
                        if abs(index - idx) < close_index:
                            close_index = abs(index - idx)
                            best_index = idx
                    fake_mask_index.append(best_index)

                for index in range(len(pos_not_avail)):  # 存储正样本+没有标签
                    img_save = dcm_img[280:440, 110:610, pos_not_avail[index]]
                    np.save(image_save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        pos_not_avail[index]) + '_' + str(fake_mask_index[index]) + '_posUnLabel_.npy',
                            img_save)

                for index in neg_slice:  # 存储负样本
                    img_save = dcm_img[280:440, 110:610, index]
                    np.save(image_save_path + '/train/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_negLabel_.npy',
                            img_save)

    for casei in os.listdir(data_input_dir):  # 处理每个样本
        pi = casei.split("_")[1]

        if pi not in test_patience:
            continue

        print(data_input_dir + '/' + casei)
        dcm_img = read_dicom(data_input_dir + "/" + casei)
        # print("Dcm shape: ", dcm_img.shape)

        for arti in ["ECAR"]:  # 处理外部左边
            cas_dir = data_input_dir + "/" + casei + "/CASCADE-" + arti
            qvs_path = cas_dir + "/E" + pi + "S101_L.QVS"
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)  # 正样本+标签：slices
            else:
                avail_slices = list_contour_slices(qvsroot)

            print("case", pi, "art", arti, "avail_slices", avail_slices)

            if len(avail_slices):  # 只有当该位置的样本存在时候，才保存对应的正负样本，internal必定存在，external有可能不存在
                lower_index = min(avail_slices)  # 正样本下界
                upper_index = max(avail_slices)  # 正样本上界

                pos_slice = list(range(lower_index, upper_index + 1))  # 所有正样本区间
                pos_not_avail = list(set(pos_slice).difference(avail_slices))  # 正样本+没有标签：slices

                neg_slice = list(range(200, lower_index))
                neg_slice1 = list(range(upper_index + 1, 400))
                neg_slice.extend(neg_slice1)  # 负样本区间

                for index in avail_slices:  # 存储：正样本+有标签：样本
                    img_save = dcm_img[280:440, 110:610, index]
                    # print(img_save.shape)

                    np.save(image_save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_posLabel_.npy',
                            img_save)

                fake_mask_index = []  # 寻找最近的伪标签
                for index in pos_not_avail:  # 找出没有标签的正样本，对用的fake label的index
                    close_index = 1000
                    best_index = -1
                    for idx in avail_slices:
                        if abs(index - idx) < close_index:
                            close_index = abs(index - idx)
                            best_index = idx
                    fake_mask_index.append(best_index)

                for index in range(len(pos_not_avail)):  # 存储正样本+没有标签
                    img_save = dcm_img[280:440, 110:610, pos_not_avail[index]]
                    np.save(image_save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        pos_not_avail[index]) + '_' + str(fake_mask_index[index]) + '_posUnLabel_.npy',
                            img_save)

                for index in neg_slice:  # 存储负样本
                    img_save = dcm_img[280:440, 110:610, index]
                    np.save(image_save_path + '/test/' + arti + '/' + pi + '_' + arti + '_' + str(
                        index) + '_negLabel_.npy',
                            img_save)


def main(args):
    data_input_dir = args.datasets_path
    image_save_path = args.image_semi

    save_internal(data_input_dir,image_save_path)
    save_external_left(data_input_dir,image_save_path)
    save_external_right(data_input_dir,image_save_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)


















