# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/05/25 19:02
# @Author  : Yi
# @FileName: predict.py

"""TODO:进行预测，python predict.py 0_P001_U
    预测四个位置：读取四个模型！
        1.读取一个病例所有的dcm，注意dcm文件的个数
        2.读取文件后，要进行预处理：高斯标准化+裁剪 → 预测
        3.计算评价指标dice，同时保存为.png格式
        4.进行后处理：读取.png格式，获得contour轮廓→生成QVS（类似xml文件）
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np
import scipy.spatial
import glob
import cv2
import argparse
import pydicom
from skimage import io
import xml.etree.ElementTree as ET
import torch.optim as optim

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import xlrd
import xlwt
from xlutils.copy import copy
import shutil

import sys
sys.path.append("/data/yilinzhi/Segmentation/VMS/")
from models.UNet import UNet
from models.udensedown import udensedown
from models.udensenet import udensenet


def create_dir(path):
    """Create a new directory.

        Args:
            path: Path of a new directory.

        Returns:
            A new directory
    """
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)


def write_excel_xls(path, sheet_name, value):
    """保存结果
    """
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
    workbook.save(path)  # 保存工作簿
    print("xls格式表格写入数据成功！")


def getDSC(true_mask, pred_mask):
    """测试集评价指标：计算有label图像的dice值"""

    t_mask = true_mask.flatten()
    p_mask = pred_mask.flatten()

    # similarity = 1.0 - dissimilarity
    return 1.0 - scipy.spatial.distance.dice(t_mask, p_mask)


def DSC(labelimg, predict_img_thres):
    """计算分割的dice值

    :param labelimg:
    :param predict_img_thres:
    :return:
    """
    A = labelimg > 0.5 * np.max(labelimg)
    B = predict_img_thres > 0.5 * np.max(predict_img_thres)
    return 2 * np.sum(A[A == B]) / (np.sum(A) + np.sum(B))


def get_args():
    """修改参数"""
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TODO：已经预处理的参数
    parser.add_argument('--test_case_path', '-p',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_semi_F/test/',
                        metavar='FILE',
                        help="The test cases.")

    parser.add_argument('--model_path_ICAL',
                        default='/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train/train_ICAL/Udensenet/0.001/4/',
                        metavar='FILE',
                        help="Training models of ICAL.")
    parser.add_argument('--model_path_ICAR',
                        default='/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train/train_ICAR/Udensenet/0.001/4/',
                        metavar='FILE',
                        help="Training models of ICAR.")

    parser.add_argument('--save_path', '-s',
                        default='/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train_F/pred',
                        metavar='FILE',
                        help="Predicted result saving.")

    parser.add_argument('--label_path_ICAL',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/mask_semi_F/test/ICAL',
                        metavar='FILE',
                        help="Label saving of ICAL.")
    parser.add_argument('--label_path_ICAR',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/train_label/mask_semi_F/test/ICAR',
                        metavar='FILE',
                        help="Label saving of ICAR.")

    # TODO：没有预处理的参数
    parser.add_argument('--test_raw_case_path',
                        default='/data/yilinzhi/Segmentation/VMS/datasets/careIIChallenge/',
                        metavar='FILE',
                        help="The test cases.")

    parser.add_argument('--case_name', '-c', metavar='INPUT',
                        help='The name of test case.', required=True)

    # parser.add_argument('--model_path_ICAL',
    #                     default='/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train/train_ICAL/Udensenet/0.001/4/',
    #                     metavar='FILE',
    #                     help="Training models of ICAL.")
    # parser.add_argument('--model_path_ICAR',
    #                     default='/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train/train_ICAR/Udensenet/0.001/4/',
    #                     metavar='FILE',
    #                     help="Training models of ICAR.")

    return parser.parse_args()


def pred_post(args, case_name, model_name):
    """预测一个测试集：已经进行预处理的图片

    :param args:
    :param case_name:
    :param model_name:
    :return:
    """

    # 预测一个病例四个不同位置！
    for CART in ['ICAL', 'ICAR', ]:
        test_case_path = args.test_case_path + CART  # 测试集未知
        save_path = args.save_path  # 预测结果保存未知

        img_files = glob.glob(test_case_path + '/' + case_name + '*_.npy')  # 找到测试集对应病例下所有文件名
        img_files.sort()  # 排序
        # print(img_files)

        if CART == 'ICAR':
            continue

        # 初始化不同模型
        # model = UNet(1, 1)
        # model = udensedown(1,1)
        model = udensenet(1, 1)
        device = torch.device('cuda:0')
        model = nn.DataParallel(model, device_ids=[0, 1])
        model.to(device=device)

        # 加载已经训练好的模型
        # TODO:不同位置，加载不同的model_path!
        if CART == 'ICAL':
            model_path = args.model_path_ICAL
            label_path = args.label_path_ICAL
        else:
            model_path = args.model_path_ICAR
            label_path = args.label_path_ICAR
        model.load_state_dict(torch.load(model_path + model_name))
        model.eval()

        create_dir(save_path + '/post_process/' + case_name + '_png_' + CART)  # 预测保存：病例+位置
        create_dir(save_path + '/post_process/' + case_name + '_npy_' + CART)

        label_cnt = 0  # 统计有标记的label
        dice = 0.0
        for img_file in img_files:  # 从下往上逐个slice预测
            # 预测已经处理好的图片
            test_data = np.load(img_file)  # 导入测试image对应的slices
            file_name = img_file.split('/')[-1]
            label = file_name.split('_')[-2]
            name = file_name.rsplit('_', 1)[0]

            true_mask = np.load(label_path + '/' + file_name)  # 导入标记的mask

            test_data_1 = torch.from_numpy(test_data)
            test_data_2 = test_data_1.unsqueeze(0)
            test_data_3 = test_data_2.unsqueeze(0)
            test_data_3.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred = model(test_data_3)
                result = pred.squeeze().cpu().numpy()
                # print(np.shape(result))
            np.save(save_path + '/post_process/' + case_name + '_npy_' + CART + '/' + name + '_.npy', result)
            cv2.imwrite(save_path + '/post_process/' + case_name + '_png_' + CART + '/' + name + '_.png', result)

            if label == 'posLabel':
                label_cnt += 1

                dice_1 = DSC(true_mask, result)
                dice += dice_1

        dice_avg = dice / label_cnt
        print(case_name, CART, label_cnt, dice_avg)
        print()

def png_contour_xml_post(case_name, pred_png_path, save_path):
    """Read the test prediction picture and generate the corresponding xml file.

    :param pred_img_path:
    :return:
    """

    prefix = 'E'
    suffix = 'S101_L.QVS'
    print("%" * 10, case_name, "&" * 10)

    for CART in ['ICAL', 'ICAR', ]:  # 一个病例保存四个不同未知的QVS(xml）文件
        if CART == 'ICAR':
            continue

        pred_png_path = pred_png_path + CART
        pred_png_list = os.listdir(pred_png_path)  # 找到所有预测图片文件名
        pred_png_list.sort()

        QVAS_name = prefix + case_name + suffix  # QVS文件名
        create_dir(save_path + case_name + '_QVS/CASCADE-' + CART + '/')
        save_path = save_path + case_name + '_QVS/CASCADE-' + CART + '/' + QVAS_name

        QVAS_Series = ET.Element('QVAS_Series')  # 创建根结点
        QVAS_Series.set("xmlns", "vil.rad.washington.edu")

        QVAS_Version = ET.SubElement(QVAS_Series, 'QVAS_Version')  # root子元素：版本
        QVAS_Version.set(" xmlns", "")
        QVAS_Series.text = "1.0"

        QVAS_Series_Info = ET.SubElement(QVAS_Series, 'QVAS_Series_Info')  # root子元素：信息
        QVAS_Series_Info.set(" xmlns", "")

        for file in pred_png_list:  # 处理每个图片，file：文件名
            blob = io.imread(pred_png_path + '/' + file)  # 读取文件
            index = file.split('/')[-1].split('_')[2]  # 得到对应的索引号

            image_name = prefix + case_name + 'S101I' + index

            contours, hier = cv2.findContours(blob, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # 获取轮廓点坐标

            QVAS_Image = ET.SubElement(QVAS_Series, 'QVAS_Image')  # root子元素：逐个slice信息
            QVAS_Image.set('xmlns', '')  # QVAS_Image元素：属性
            QVAS_Image.set('ImageName', image_name)  # QVAS_Image元素：属性，图片名字

            Translation = ET.SubElement(QVAS_Image, 'Translation')  # QVAS_Image子元素1：Translation

            Rotation = ET.SubElement(Translation, 'Rotation')  # Translation子元素：Rotation
            Angle = ET.SubElement(Rotation, 'Rotation')  # Rotation子元素：Angle
            Angle.text = '0.00'
            Point = ET.SubElement(Rotation, 'Point')  # Rotation子元素：Point
            Point.set('x', '0.000000')
            Point.set('y', '0.000000')

            ShiftAfterRotation = ET.SubElement(Translation, 'ShiftAfterRotation')  # Translation子元素：ShiftAfterRotation
            ShiftAfterRotation.set('x', '  0.00')
            ShiftAfterRotation.set('y', '  0.00')

            ImageFilePath = ET.SubElement(QVAS_Image, 'Rotation')  # QVAS_Image子元素2
            ImageMode = ET.SubElement(QVAS_Image, 'ImageMode')  # QVAS_Image子元素3
            ImageBifurcationLevel = ET.SubElement(QVAS_Image, 'ImageBifurcationLevel')  # QVAS_Image子元素4
            ImageBifurcationLevel.text = '-999'
            ImageType = ET.SubElement(QVAS_Image, 'ImageType')  # QVAS_Image子元素5

            # TODO：保存数据最多的两个contour轮廓,此处其实是有问题！
            # print("index:", index, "len: ", len(contours))
            if contours == []:  # 负样本：没有轮廓，直接跳过
                continue
            if len(contours) == 1:  # 一个轮廓，跳过
                continue
            contours.sort(key=lambda i: len(i), reverse=True)

            # TODO：保存outer轮廓数据点
            QVAS_Contour = ET.SubElement(QVAS_Image, 'QVAS_Contour')  # QVAS_Image子元素：QVAS_Contour

            # TODO：注意坐标数值转换，512*512下坐标！
            Contour_Point = ET.SubElement(QVAS_Contour, 'Contour_Point')
            for cor in contours[0]:
                Point = ET.SubElement(Contour_Point, 'Point')
                Point.set('x', str((cor[0][0] + 110) * 512.0 / 720))
                Point.set('y', str((cor[0][1] + 280) * 512.0 / 720))
            ContourType = ET.SubElement(QVAS_Contour, 'ContourType')
            ContourType.text = 'Outer Wall'

            # TODO：保存lumen
            QVAS_Contour = ET.SubElement(QVAS_Image, 'QVAS_Contour')  # QVAS_Image子元素：QVAS_Contour

            Contour_Point = ET.SubElement(QVAS_Contour, 'Contour_Point')
            for cor in contours[1]:
                Point = ET.SubElement(Contour_Point, 'Point')
                Point.set('x', str((cor[0][0] + 110) * 512.0 / 720))
                Point.set('y', str((cor[0][1] + 280) * 512.0 / 720))
            ContourType = ET.SubElement(QVAS_Contour, 'ContourType')
            ContourType.text = 'Lumen'

        tree = ET.ElementTree(QVAS_Series)
        tree.write(save_path, encoding='UTF-8', xml_declaration=True)



def read_dicom(path):
    """读取一个案例，并返回整个案列的numpy文件。"""

    print(os.path.basename(path))
    pi = os.path.basename(path).split("_")[1]
    dcm_size = len(glob.glob(path + "/*.dcm"))  # 找到病例下所有的.dcm文件

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


# /data/yilinzhi/Segmentation/VMS/datasets/careIIChallenge/
# python predict.py 0_P002_U

def pred_raw(args, model_name):
    """预测一个测试集：原始数据集"""

    case_name=args.case_name
    save_path=args.save_path

    test_case_path = args.test_raw_case_path + case_name
    img_files = glob.glob(test_case_path + '/' + '*.dcm')  # 原始dcm文档
    img_files.sort()

    dcm_img = read_dicom(test_case_path)  # 读取一个病例所有的dcm文件，并转成一个numpy文件
    for CART in ['ICAL', 'ICAR']:
        if CART == 'ICAR':
            continue

        # 初始化不同模型
        # model = UNet(1, 1)
        # model = udensedown(1,1)
        model = udensenet(1, 1)
        device = torch.device('cuda:0')
        model = nn.DataParallel(model, device_ids=[0, 1])
        model.to(device=device)

        # 加载已经训练好的模型
        # TODO:不同位置，加载不同的model_path!
        if CART == 'ICAL':
            model_path = args.model_path_ICAL
        else:
            model_path = args.model_path_ICAR

        model.load_state_dict(torch.load(model_path + model_name))
        model.eval()

        create_dir(save_path + '/raw/png/' + case_name + '_'+CART)  # 预测保存：病例+位置

        for index in range(len(img_files)):
            test_img = dcm_img[280:440, 110:610, index]  # 要预测的img：已经经过预处理

            test_img_1 = torch.from_numpy(test_img)
            test_img_2 = test_img_1.unsqueeze(0)
            test_img_3 = test_img_2.unsqueeze(0)
            test_img_3.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred = model(test_img_3)
                result = pred.squeeze().cpu().numpy()

            cv2.imwrite(save_path + '/raw/png/' + case_name +'_'+ CART + '/'  + str(index+1) + '_.png',
                        result)

def png_contour_qvs_raw(args):
    """Read the test prediction picture and generate the corresponding xml file.

    :param pred_img_path:
    :return:
    """

    # TODO：目前只生成QVS文件，人需要生成QJV文件
    case_name=args.case_name
    name=case_name.split('_')[1]

    prefix = 'E'
    suffix = 'S101_L.QVS'
    print("%" * 10, name, "&" * 10)

    for CART in ['ICAL', 'ICAR' ]:  # 一个病例保存四个不同未知的QVS(xml）文件
        if CART == 'ICAR':
            continue

        pred_png_path = args.save_path + '/raw/png/' + case_name+'_'  + CART + '/'
        pred_png_list = os.listdir(pred_png_path)  # 找到所有预测图片文件名
        pred_png_list.sort(key = lambda  name : int(name.split('_')[0]))

        QVAS_name = prefix + name + suffix  # QVS文件名
        create_dir(args.save_path+'/raw/QVS/' + case_name + '/CASCADE/CASCADE-' + CART + '/')  #QVS保存路径
        save_path = args.save_path+ '/raw/QVS/'+ case_name + '/CASCADE/CASCADE-' + CART + '/' + QVAS_name

        QVAS_Series = ET.Element('QVAS_Series')  # 创建根结点
        QVAS_Series.set("xmlns", "vil.rad.washington.edu")

        QVAS_Version = ET.SubElement(QVAS_Series, 'QVAS_Version')  # root子元素：版本
        QVAS_Version.set(" xmlns", "")
        QVAS_Version.text = "1.0"

        QVAS_Series_Info = ET.SubElement(QVAS_Series, 'QVAS_Series_Info')  # root子元素：信息
        QVAS_Series_Info.set(" xmlns", "")

        for file in pred_png_list:  # 处理每个图片，file：文件名
            blob = io.imread(pred_png_path + '/' + file)  # 读取文件
            index = file.split('/')[-1].split('_')[0]  # 得到对应的索引号
            print("index:", index)

            image_name = prefix + name + 'S101I' + index

            contours, hier = cv2.findContours(blob, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # 获取轮廓点坐标

            QVAS_Image = ET.SubElement(QVAS_Series, 'QVAS_Image')  # root子元素：逐个slice信息
            QVAS_Image.set('xmlns', '')  # QVAS_Image元素：属性
            QVAS_Image.set('ImageName', image_name)  # QVAS_Image元素：属性，图片名字

            Translation = ET.SubElement(QVAS_Image, 'Translation')  # QVAS_Image子元素1：Translation

            Rotation = ET.SubElement(Translation, 'Rotation')  # Translation子元素：Rotation
            Angle = ET.SubElement(Rotation, 'Rotation')  # Rotation子元素：Angle
            Angle.text = '0.00'
            Point = ET.SubElement(Rotation, 'Point')  # Rotation子元素：Point
            Point.set('x', '0.000000')
            Point.set('y', '0.000000')

            ShiftAfterRotation = ET.SubElement(Translation, 'ShiftAfterRotation')  # Translation子元素：ShiftAfterRotation
            ShiftAfterRotation.set('x', '  0.00')
            ShiftAfterRotation.set('y', '  0.00')

            ImageFilePath = ET.SubElement(QVAS_Image, 'Rotation')  # QVAS_Image子元素2
            ImageMode = ET.SubElement(QVAS_Image, 'ImageMode')  # QVAS_Image子元素3
            ImageBifurcationLevel = ET.SubElement(QVAS_Image, 'ImageBifurcationLevel')  # QVAS_Image子元素4
            ImageBifurcationLevel.text = '-999'
            ImageType = ET.SubElement(QVAS_Image, 'ImageType')  # QVAS_Image子元素5

            # TODO：保存数据最多的两个contour轮廓,此处其实是有问题！
            # print("index:", index, "len: ", len(contours))
            if contours == []:  # 负样本：没有轮廓，直接跳过
                continue
            if len(contours) == 1:  # 一个轮廓，跳过
                continue
            contours.sort(key=lambda i: len(i), reverse=True)

            # TODO：保存outer轮廓数据点
            QVAS_Contour = ET.SubElement(QVAS_Image, 'QVAS_Contour')  # QVAS_Image子元素：QVAS_Contour

            # TODO：注意坐标数值转换，512*512下坐标！
            Contour_Point = ET.SubElement(QVAS_Contour, 'Contour_Point')
            for cor in contours[0]:
                Point = ET.SubElement(Contour_Point, 'Point')
                Point.set('x', str((cor[0][0] + 110) * 512.0 / 720))
                Point.set('y', str((cor[0][1] + 280) * 512.0 / 720))
            ContourType = ET.SubElement(QVAS_Contour, 'ContourType')
            ContourType.text = 'Outer Wall'

            # TODO：保存lumen
            QVAS_Contour = ET.SubElement(QVAS_Image, 'QVAS_Contour')  # QVAS_Image子元素：QVAS_Contour

            Contour_Point = ET.SubElement(QVAS_Contour, 'Contour_Point')
            for cor in contours[1]:
                Point = ET.SubElement(Contour_Point, 'Point')
                Point.set('x', str((cor[0][0] + 110) * 512.0 / 720))
                Point.set('y', str((cor[0][1] + 280) * 512.0 / 720))
            ContourType = ET.SubElement(QVAS_Contour, 'ContourType')
            ContourType.text = 'Lumen'

        tree = ET.ElementTree(QVAS_Series)
        tree.write(save_path, encoding='UTF-8', xml_declaration=True)



def png_contour_qvj_raw(args):
    """Read the test prediction picture and generate the corresponding xml file.

    :param pred_img_path:
    :return:
    """

    # TODO：目前只生成QVS文件，人需要生成QJV文件
    case_name=args.case_name
    name=case_name.split('_')[1]

    prefix = 'E'
    suffix = '.QVJ'
    print("%" * 10, name, "%" * 10)


    # TODO：找到分叉图片的序号
    bi_left_index=1000
    for CART in ['ECAR', 'ECAL']:
        if CART == 'ECAL':
            continue

        pred_png_path = args.save_path + '/raw/png/' + case_name + '_' + CART + '/'
        pred_png_list = os.listdir(pred_png_path)  # 找到所有预测图片文件名
        pred_png_list.sort(key=lambda name: int(name.split('_')[0]))

        for file in pred_png_list:  # 处理每个图片，file：文件名
            blob = io.imread(pred_png_path + '/' + file)  # 读取文件
            index = file.split('/')[-1].split('_')[0]  # 得到对应的索引号
            print("index:", index)


            contours, hier = cv2.findContours(blob, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # 获取轮廓点坐标

            if contours == []:  # 负样本：没有轮廓，直接跳过
                continue
            if len(contours) == 1:  # 一个轮廓，跳过
                continue

            if index<bi_left_index:
                bi_left_index=index
            continue


    for CART in ['ECAR', 'ECAL' ]:  # 一个病例保存四个不同未知的QVS(xml）文件

        pred_png_path = args.save_path + '/raw/png/' + case_name+'_'  + CART + '/'
        pred_png_list = os.listdir(pred_png_path)  # 找到所有预测图片文件名
        pred_png_list.sort(key = lambda  name : int(name.split('_')[0]))


        QVAS_Project_name = prefix + name + suffix  # QVJ文件名
        create_dir(args.save_path+'/raw/QVS/' + case_name + '/CASCADE/CASCADE-' + CART + '/')  #QVS保存路径
        save_path = args.save_path+ '/raw/QVS/'+ case_name + '/CASCADE/CASCADE-' + CART + '/' + QVAS_Project_name

        QVAS_Project = ET.Element('ns0:QVAS_Project')  # 创建根结点
        QVAS_Project.set("xmlns:ns0", "vil.rad.washington.edu")

        QVAS_Version = ET.SubElement(QVAS_Project, 'QVAS_Version')  # root子元素：版本
        QVAS_Version.set(" xmlns", "")
        QVAS_Version.text = "1.0"

        QVAS_System_Info = ET.SubElement(QVAS_Project, 'QVAS_System_Info')  # root子元素：信息
        QVAS_System_Info.set(" xmlns", "")

        AnalysisMode=ET.SubElement(QVAS_System_Info, 'AnalysisMode')
        AnalysisMode.text='1'
        ImageLocationStatus=ET.SubElement(QVAS_System_Info, 'ImageLocationStatus')
        ImageLocationStatus.text='2'
        BifurcationLocation=ET.SubElement(QVAS_System_Info, 'BifurcationLocation')

        BifurcationImageIndex=ET.SubElement(BifurcationLocation, 'BifurcationImageIndex')
        BifurcationImageIndex.set('ImageIndex',str(bi_left_index))
        BifurcationImageIndex.set('SeriesName',QVAS_Project_name)






if __name__ == '__main__':
    test_patience = ["P206", "P432", "P576", "P891"]
    args = get_args()

    # for case_name in test_patience:
    #     model_name = 'Udensenet_80.pkl'
    #     pred_post(args, case_name, model_name)
    #
    # for case_name in test_patience:
    #     pred_png_path = args.save_path + '/post_process/' + case_name + '_png_'
    #     save_path = args.save_path + '/post_process/'
    #     png_contour_xml_post(case_name, pred_png_path, save_path)

    model_name = 'Udensenet_80.pkl'
    pred_raw(args,model_name)
    png_contour_qvs_raw(args)
    # png_contour_qvj_raw(args)
