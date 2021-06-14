# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/06/04 14:34
# @Author  : Yi
# @FileName: post_process_semi_submit.py


# TODO：一个病例所有的读取预测图片，然后获取轮廓点坐标
# TODO: 根据点坐标生成自定义的xml文档


import cv2
import os
import numpy as np
from skimage import io
import glob
import xml.etree.ElementTree as ET


"""
     Read image;
     find contours with hierarchy;
"""


def png_contour_xml(pred_path):
    """Read the test prediction picture and generate the corresponding xml file.

    :param pred_img_path:
    :return:
    """
    prefix = 'E'
    med = 'S101I'
    test_patience = ["P206", "P432", "P576", "P891"]
    for case in test_patience:
        print("%" * 10, case, "&" * 10)
        pred_png_path = pred_path + case + '_png/'
        pred_png_list = os.listdir(pred_png_path)  # 找到所有预测图片文件名
        pred_png_list.sort()

        QVAS_Series = ET.Element('QVAS_Series')  # 创建根结点
        QVAS_Version = ET.SubElement(QVAS_Series, 'QVAS_Version')  # root子元素：版本
        QVAS_Series_Info = ET.SubElement(QVAS_Series, 'QVAS_Series_Info')  # root子元素：信息
        QVAS_name = prefix + case + med + '_L'
        for file in pred_png_list:  # 处理每个图片
            blob = io.imread(pred_png_path + file)  # 读取文件
            index = file.split('/')[-1].split('_')[2]  # 得到对应的索引号
            # print("index: ",index)
            image_name = prefix + case + med + index

            contours, hier = cv2.findContours(blob, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # 获取轮廓点坐标

            QVAS_Image = ET.SubElement(QVAS_Series, 'QVAS_Image')  # root子元素：逐个slice信息
            QVAS_Image.set('xmlns', '')  # QVAS_Image元素：属性
            QVAS_Image.set('ImageName', image_name)  # QVAS_Image元素：属性，图片名字

            Translation = ET.SubElement(QVAS_Image, 'Translation')  # QVAS_Image子元素1：Translation
            ImageFilePath = ET.SubElement(QVAS_Image, 'Rotation')  # QVAS_Image子元素2
            ImageMode = ET.SubElement(QVAS_Image, 'ImageMode')  # QVAS_Image子元素3
            ImageBifurcationLevel = ET.SubElement(QVAS_Image, 'ImageBifurcationLevel')  # QVAS_Image子元素4
            ImageType = ET.SubElement(QVAS_Image, 'ImageType')  # QVAS_Image子元素5

            Rotation = ET.SubElement(Translation, 'Rotation')  # Translation子元素：Rotation
            ShiftAfterRotation = ET.SubElement(Translation, 'ShiftAfterRotation')  # Translation子元素：ShiftAfterRotation
            ShiftAfterRotation.set('x', '  0.00')
            ShiftAfterRotation.set('y', '  0.00')

            Angle = ET.SubElement(Rotation, 'Rotation')  # Rotation子元素：Angle
            Angle.text = '0.00'
            Point = ET.SubElement(Rotation, 'Point')  # Rotation子元素：Point
            Point.set('x', '0.000000')
            Point.set('y', '0.000000')

            # TODO：保存数据最多的两个contour轮廓
            print("index:", index, "len: ", len(contours))
            if contours == []:  # 负样本：没有轮廓，直接跳过
                continue
            if len(contours) == 1:  # 一个轮廓，跳过
                continue
            contours.sort(key=lambda i: len(i), reverse=True)

            # TODO：保存outer轮廓数据点
            QVAS_Contour = ET.SubElement(QVAS_Image, 'QVAS_Contour')  # QVAS_Image子元素：QVAS_Contour
            Contour_Point = ET.SubElement(QVAS_Contour, 'Contour_Point')
            ContourType = ET.SubElement(QVAS_Contour, 'ContourType')
            ContourType.text = 'Outer Wall'

            # TODO：注意坐标数值转换，512*512下坐标！
            for cor in contours[0]:
                Point = ET.SubElement(Contour_Point, 'Point')
                Point.set('x', str((cor[0][0] + 110) * 512.0 / 720))
                Point.set('y', str((cor[0][1] + 280) * 512.0 / 720))

            # TODO：保存lumen
            QVAS_Contour = ET.SubElement(QVAS_Image, 'QVAS_Contour')  # QVAS_Image子元素：QVAS_Contour
            Contour_Point = ET.SubElement(QVAS_Contour, 'Contour_Point')
            ContourType = ET.SubElement(QVAS_Contour, 'ContourType')
            ContourType.text = 'Lumen'

            for cor in contours[1]:
                Point = ET.SubElement(Contour_Point, 'Point')
                Point.set('x', str((cor[0][0] + 110) * 512.0 / 720))
                Point.set('y', str((cor[0][1] + 280) * 512.0 / 720))

        # Data = ET.tostring(QVAS_Series, encoding='utf-8')  # 增加了xml声明
        # File = open("/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train/pred_ICAR/P206.QVS", "wb")
        # File.write(Data)
        tree = ET.ElementTree(QVAS_Series)
        tree.write("/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train/pred_ICAR/%s.QVS" % (QVAS_name),
                   encoding='UTF-8', xml_declaration=True)


if __name__=='__main__':
    pred_path="/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train/pred_ICAR/"
    png_contour_xml(pred_path)






