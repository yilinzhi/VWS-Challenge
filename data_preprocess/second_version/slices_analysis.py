# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/04/30 15:45
# @Author  : Yi
# @FileName: slices_analysis.py


import os
import sys
import pydicom
import glob
import shutil
import random
import numpy as np
from data_Parameter import parse_args
import xml.etree.ElementTree as ET


class Logger(object):
    """定义Logger类，记录terminal的输出信息。"""

    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def create_dir(path):
    """创造一个文件夹。"""

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
    # print(length)

    dcm_f = pydicom.read_file(dcms[0]).pixel_array  # Reading dcm files and changing into numpy file.
    print(dcm_f.shape)  # Showing the shape of original patient case: 160*640*640 or 100*720*720
    dcm_size = max(max(dcm_f.shape), 720)  # Choosing max number in (h, w, 720).

    dcm_img = np.zeros((dcm_size, dcm_size, dcm_size), dtype=np.float32)
    for dcmi in range(len(dcms)):  # Iteration all slices.
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
    """找到一个病例中标注了的slices"""

    avail_slices = []
    qvasimg = qvsroot.findall("QVAS_Image")  # QVAS_Image is a list，length=dcm_img.shape[2]

    for dicom_slicei in range(720):  # index of avail_slices: 0-719
        conts = qvasimg[dicom_slicei - 1].findall("QVAS_Contour")

        if len(conts):
            avail_slices.append(dicom_slicei)
    return avail_slices


def list_contour_slices_min(qvsroot):
    """特殊病例: 640 slices"""

    avail_slices = []
    qvasimg = qvsroot.findall("QVAS_Image")

    for dicom_slicei in range(640):
        conts = qvasimg[dicom_slicei - 1].findall("QVAS_Contour")

        if len(conts):
            avail_slices.append(dicom_slicei)
    return avail_slices


def get_contour(qvsroot, dicomslicei, conttype, dcmsz=720):
    """得到有标注图像里血管的内外壁轮廓。"""

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


def analysis_image(data_input_dir, image_save_path):
    """逐个分析案例各个位置的avail_slices"""

    sys.stdout = Logger('avail_slices_log.txt')  # 记录terminal输出
    max_index = 0
    min_index = 720
    max_ext = 0
    min_ext = 720

    for casei in os.listdir(data_input_dir):  # 逐个案例分析
        pi = casei.split("_")[1]
        dcm_img = read_dicom(data_input_dir + "/" + casei)

        for arti in ["ICAL", "ICAR", "ECAL", "ECAR"]:  # 不同位置分析
            cas_dir = data_input_dir + "/" + casei + "/CASCADE-" + arti
            qvs_path = cas_dir + "/E" + pi + "S101_L.QVS"
            qvsroot = ET.parse(qvs_path).getroot()

            if pi in ["P556", "P576", "P887"]:
                avail_slices = list_contour_slices_min(qvsroot)
            else:
                avail_slices = list_contour_slices(qvsroot)

            if len(avail_slices) > 0:
                if min(avail_slices) < min_index:
                    min_index = min(avail_slices)
                if max(avail_slices) > max_index:
                    max_index = max(avail_slices)

                if arti in ["ECAL", "ECAR"]:
                    if min(avail_slices) < min_ext:
                        min_ext = min(avail_slices)
                    if max(avail_slices) > max_ext:
                        max_ext = max(avail_slices)

            print("case", pi, "art", arti, "avail_slices length", len(avail_slices))
            print("case", pi, "art", arti, "avail_slices：\n", avail_slices)
        print("=" * 50)

    print("\nmin-max: %d-%d" % (min_index, max_index))
    print("\nmin-max of external: %d-%d" % (min_ext, max_ext))


def main(args):
    data_input_dir = args.datasets_path
    image_save_path = args.image_save_sep_position
    analysis_image(data_input_dir, image_save_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
