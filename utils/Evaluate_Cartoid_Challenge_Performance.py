# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/06/05 22:44
# @Author  : Yi
# @FileName: Evaluate_Cartoid_Challenge_Performance.py

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import glob
from scipy.spatial.distance import directed_hausdorff


def DSC(labelimg, predict_img_thres):
    """计算分割的dice值

    :param labelimg:
    :param predict_img_thres:
    :return:
    """
    A = labelimg > 0.5 * np.max(labelimg)
    B = predict_img_thres > 0.5 * np.max(predict_img_thres)
    return 2 * np.sum(A[A == B]) / (np.sum(A) + np.sum(B))


def diffmap(A, B):
    """

    :param A:
    :param B:
    :return:
    """
    diffmap = np.zeros((A.shape[0], A.shape[1], 3))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i][j] == 1 and B[i][j] == 1:
                diffmap[i, j, 2] = 1
            elif A[i][j] == 1 and B[i][j] == 0:
                diffmap[i, j, 1] = 1
            elif A[i][j] == 0 and B[i][j] == 1:
                diffmap[i, j, 0] = 1
    return diffmap


class CASCADE:
    """自定义的xml Tree格式！
    """

    def __init__(self, QVJname, QVJdir):
        cas_dir = QVJdir  # 文件夹目录
        qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'  # QVS完整的路径
        self.qvsroots = [ET.parse(qvs_path).getroot()]  # QVS根元素

        # TODO：增加QVJ文件！
        qvj_path = cas_dir + '/E' + pi + '.QVJ'  # QVS完整的路径
        self.qvjroot = ET.parse(qvj_path).getroot()  # QVS根元素
        self.dcmsz = len(os.listdir(QVJdir[:-len('CASCADE-ICAR')])) - 4  # TODO：？？？？

    def getContour(self, dicomslicei, conttype):
        """获得xml文件中轮廓的坐标信息

        :param dicom_slicei:
        :param conttype:
        :return:
        """
        qvsroot = self.qvsroots[0]
        qvasimg = qvsroot.findall('QVAS_Image')
        if dicomslicei - 1 > len(qvasimg):
            print('no slice', dicomslicei)
            return
        assert int(qvasimg[dicomslicei - 1].get('ImageName').split('I')[-1]) == dicomslicei
        conts = qvasimg[dicomslicei - 1].findall('QVAS_Contour')
        tconti = -1
        for conti in range(len(conts)):
            if conts[conti].find('ContourType').text == conttype:
                tconti = conti
                break
        if tconti == -1:
            print('no such contour', conttype)
            return
        pts = conts[tconti].find('Contour_Point').findall('Point')
        contours = []
        for pti in pts:
            contx = int(round(float(pti.get('x')) / 512 * self.dcmsz))
            conty = int(round(float(pti.get('y')) / 512 * self.dcmsz))
            # if current pt is different from last pt, add to contours
            if len(contours) == 0 or contours[-1][0] != contx or contours[-1][1] != conty:
                contours.append([contx, conty])
        return np.array(contours)

    def getBirSlice(self):
        """找到分叉部的slice的索引"""
        if self.qvjroot.find('QVAS_System_Info').find('BifurcationLocation'):
            bif_slice = int(
                self.qvjroot.find('QVAS_System_Info').find('BifurcationLocation').find('BifurcationImageIndex').get(
                    'ImageIndex'))
            return bif_slice
        else:
            return -1


# TODO：modify this path：预测mask路径！
# target_result_dir = r'\\DESKTOP2\GiCafe\result\careIIChallenge'
target_result_dir = "/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train_F/pred/raw/QVS"



dsc_all = {}
wall_area_all = {}
lumen_area_all = {}
nwi_all = {}
hd_wall_all = {}
hd_lumen_all = {}
errs_all = {}

# pilist = ['P125', 'P176', 'P196', 'P206', 'P211', 'P379', 'P388', 'P403', 'P429', 'P432',
#           'P438', 'P470', 'P481', 'P530', 'P551', 'P556', 'P576', 'P673', 'P674', 'P723',
#           'P732', 'P789', 'P887', 'P891', 'P910']

# pilist = ["P206", "P432", "P576", "P891"]
pilist = ["P206"]

# TODO：计算预测mask和标记mask各种评价指标
for pi in pilist[:]:

    QVJname = 'E' + pi + '_L'

    # target is latte result
    # AICafe
    # TODO：预测mask的路径！
    target_icafe_dir = target_result_dir + '/0_' + pi + '_U/CASCADE'

    # source is original review
    # TODO：标记mask路径
    # src_icafe_dir = r'D:\LiChen\careIIChallenge\0_' + pi + '_U'
    src_icafe_dir = "/data/yilinzhi/Segmentation/VMS/datasets/careIIChallenge/0_" + pi + "_U"

    for arti in ['ICAR', 'ICAL', 'ECAR', 'ECAL']:  #
        if arti in ['ICAR', 'ECAR', 'ECAL']:
            continue

        if arti not in dsc_all:  # 文件字典，key：arti, value：list
            dsc_all[arti] = []
            wall_area_all[arti] = []
            lumen_area_all[arti] = []
            nwi_all[arti] = []
            hd_wall_all[arti] = []
            hd_lumen_all[arti] = []
            errs_all[arti] = []

        src_art_dir = src_icafe_dir + '/CASCADE-' + arti  # 标记mask路径
        src_cas = CASCADE(QVJname, src_art_dir)  # 实例化一个对象
        src_cas.dcmsz = 720
        src_qvasimg = src_cas.qvsroots[0].findall('QVAS_Image')

        target_art_dir = target_icafe_dir + '/CASCADE-' + arti  # 预测mask路径

        # no matching artery
        if not os.path.exists(target_art_dir):
            target_art_dirs = glob.glob(target_icafe_dir + '/CASCADE-*' + arti)  # 找到预测mask文件夹
            if len(target_art_dirs) == 0:  # 预测mask文件不存在
                print('no matching art, adding errs for all src slices with contours')
                for dicomslicei in range(1, len(src_qvasimg) + 1):  # 标记mask
                    if dicomslicei < src_cas.getBirSlice() - 99:
                        continue
                    if len(src_qvasimg[dicomslicei - 1].findall('QVAS_Contour')):
                        src_lumen_cont = src_cas.getContour(dicomslicei, 'Lumen')
                        if src_lumen_cont is None:
                            # skip no contour slices in src
                            continue
                        errs_all[arti].append([dicomslicei, pi, arti])  # 添加错误信息
                continue
            else:
                target_art_dir = target_art_dirs[0]
                print('auto sel', target_art_dir)
        target_cas = CASCADE(QVJname, target_art_dir)
        target_cas.dcmsz = 720
        target_qvasimg = target_cas.qvsroots[0].findall('QVAS_Image')

        for dicomslicei in range(1, len(src_qvasimg) + 1):
            if dicomslicei < src_cas.getBirSlice() - 99:
                continue

            assert int(src_qvasimg[dicomslicei - 1].get('ImageName').split('I')[-1]) == dicomslicei

            if len(src_qvasimg[dicomslicei - 1].findall('QVAS_Contour')):
                src_lumen_cont = src_cas.getContour(dicomslicei, 'Lumen')  # 标记mask处的contour
                if src_lumen_cont is None:
                    continue
                src_wall_cont = src_cas.getContour(dicomslicei, 'Outer Wall')
                if src_wall_cont is None:
                    print('no wall', dicomslicei)
                    continue
                src_wall_img = np.zeros((720, 720))
                cv2.fillPoly(src_wall_img, pts=[np.array(src_wall_cont)], color=(1, 1, 1))
                src_wall_area = np.sum(src_wall_img)
                src_lumen_img = np.zeros((720, 720))
                cv2.fillPoly(src_lumen_img, pts=[np.array(src_lumen_cont)], color=(1, 1, 1))
                src_lumen_area = np.sum(src_lumen_img)
                src_img = src_wall_img - src_lumen_img  # 标记mask图片

                target_lumen_cont = target_cas.getContour(dicomslicei, 'Lumen')  # 预测mask处的contour
                target_wall_cont = target_cas.getContour(dicomslicei, 'Outer Wall')
                if target_wall_cont is None:
                    errs_all[arti].append([dicomslicei, pi, arti])
                    continue
                target_wall_img = np.zeros((720, 720))
                cv2.fillPoly(target_wall_img, pts=[np.array(target_wall_cont)], color=(1, 1, 1))
                target_wall_area = np.sum(target_wall_img)
                target_lumen_img = np.zeros((720, 720))
                cv2.fillPoly(target_lumen_img, pts=[np.array(target_lumen_cont)], color=(1, 1, 1))
                target_lumen_area = np.sum(target_lumen_img)
                target_img = target_wall_img - target_lumen_img  # 预测mask图片

                cdsc = DSC(src_img, target_img)  # 计算dice值
                print("cdsc: ",cdsc)

                dsc_all[arti].append(cdsc)  # 评价指标1：dice值
                print(pi, arti, dicomslicei, cdsc)
                if cdsc < 0:
                    plt.imshow(diffmap(src_img, target_img))
                    plt.title(cdsc)
                    plt.show()

                area_diff_lumen = abs(target_lumen_area - src_lumen_area) / src_lumen_area  # 预测和标记的差异
                area_diff_wall = abs(target_wall_area - src_wall_area) / src_wall_area

                lumen_area_all[arti].append(area_diff_lumen)  # 评价指标2：lumen area
                wall_area_all[arti].append(area_diff_wall)  # 评价指标3：outer area

                target_nwi = (target_wall_area - target_lumen_area) / target_wall_area
                src_nwi = (src_wall_area - src_lumen_area) / src_wall_area
                nwi_diff = abs(target_nwi - src_nwi) / src_nwi
                nwi_all[arti].append(nwi_diff)  # 评价指标3

                hd_wall = max(directed_hausdorff(src_wall_cont, target_wall_cont)[0],
                              directed_hausdorff(target_wall_cont, src_wall_cont)[0])
                target_wall_radius = np.sqrt(target_wall_area / np.pi)
                hd_wall_all[arti].append(hd_wall / target_wall_radius)  # 评价指标4：距离
                hd_lumen = max(directed_hausdorff(src_lumen_cont, target_lumen_cont)[0],
                               directed_hausdorff(target_lumen_cont, src_lumen_cont)[0])
                target_lumen_radius = np.sqrt(target_lumen_area / np.pi)
                hd_lumen_all[arti].append(hd_lumen / target_lumen_radius)  # 评价指标5：距离

# TODO：计算最后得分！

dscs = []
lumen_area = []
wall_area = []
nwis = []
hd_lumens = []
hd_walls = []
quant_scores = []
errs = []

for arti in dsc_all:
    dscs.extend(dsc_all[arti])
    lumen_area.extend(lumen_area_all[arti])
    wall_area.extend(wall_area_all[arti])
    nwis.extend(nwi_all[arti])
    hd_walls.extend(hd_wall_all[arti])
    hd_lumens.extend(hd_lumen_all[arti])

    # TODO：计算得分！
    quant_scores.extend([0.5 * dsc_all[arti][i] +
                         0.1 * (max(0, 1 - lumen_area_all[arti][i])) +
                         0.1 * (max(0, 1 - wall_area_all[arti][i])) +
                         0.2 * (max(0, 1 - nwi_all[arti][i])) +
                         0.05 * (max(0, 1 - hd_wall_all[arti][i])) +
                         0.05 * (max(0, 1 - hd_lumen_all[arti][i]))
                         for i in range(len(dsc_all[arti]))])
    errs.extend(errs_all[arti])
    # penalize err slices
    quant_scores.extend([0] * len(errs_all[arti]))

print('DSC: %.3f±%.3f (N=%d)' % (np.mean(dscs), np.std(dscs), len(dscs)))
print('Lumen area difference: %.3f±%.3f (N=%d)' % (np.mean(lumen_area), np.std(lumen_area), len(lumen_area)))
print('Wall area difference: %.3f±%.3f (N=%d)' % (np.mean(wall_area), np.std(wall_area), len(wall_area)))
print('Normalized wall index difference: %.3f±%.3f (N=%d)' % (np.mean(nwis), np.std(nwis), len(nwis)))
print('Hausdorff distance on lumen normalized by radius: %.3f±%.3f (N=%d)' % (
    np.mean(hd_lumens), np.std(hd_lumens), len(hd_lumens)))
print('Hausdorff distance on wall normalized by radius: %.3f±%.3f (N=%d)' % (
    np.mean(hd_walls), np.std(hd_walls), len(hd_walls)))
print('Quantitative score: %.3f±%.3f (N=%d)' % (np.mean(quant_scores), np.std(quant_scores), len(quant_scores)))
print('No matching slices: %d' % (len(errs)))
