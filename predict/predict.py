import sys
import os
import pydicom

import glob
import torch
import numpy as np
import shutil
import cv2
import torch.nn as nn
from skimage import io

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import sys

sys.path.append('/data/yilinzhi/Segmentation/VMS')
# sys.path.append('/VW/')
from models.udensenet import udensenet
import xml.etree.ElementTree as ET


def create_dir(path):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)


def read_dicom(path):
    # print(os.path.basename(path))
    pi = os.path.basename(path).split("_")[1]
    dcm_size = len(glob.glob(path + "/*.dcm"))  # 找到病例下所有的.dcm文件

    dcms = [
        path + "/E" + pi + "S101I%d.dcm" % dicom_slicei
        for dicom_slicei in range(1, dcm_size + 1)
    ]  # dcm路径列表

    length = int(len(dcms))
    # print(length)

    dcm_f = pydicom.read_file(dcms[0]).pixel_array
    dcm_size = max(max(dcm_f.shape), 720)
    # print(dcm_f.shape)

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


print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

# result_dir = '/results/' + sys.argv[1]
# data_dir = '/data/' + sys.argv[1]

# # todo: modification 1
result_dir = '/data/yilinzhi/Segmentation/VMS/results/' + sys.argv[1]
data_dir = '/data/yilinzhi/Segmentation/VMS/datasets/careIIChallenge/' + sys.argv[1]
pi = sys.argv[1].split('_')[1]  # 案例的名字

if not os.path.exists(result_dir):
    os.mkdir(result_dir)
# print(os.listdir('/results'))

### todo:读取data_dir：测试集文件夹下的所有测试图片dcm，转成numpy文件
dcm_img = read_dicom(data_dir)
model_name = 'Udensenet_80.pkl'

img_files = glob.glob(data_dir + '/' + '*.dcm')  # 读取测试集原始dcm文档
img_files.sort()

# todo：vms的模型+生成的png结果保存在/VWS/文件下
# png_save_path = '/VW/png/' + sys.argv[1]
# model_path_ECAL = '/VW/model/ECAL/'
# model_path_ECAR = '/VW/model/ECAR/'
# model_path_ICAL = '/VW/model/ICAL/'
# model_path_ICAR = '/VW/model/ICAR/'

# todo: modification 2
png_save_path = '/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train_F/pred/png/' + sys.argv[1]
npy_save_path = '/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train_F/pred/npy/' + sys.argv[1]

model_path_ICAL = '/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train/train_ICAL/Udensenet/0.001/4/'
model_path_ICAR = '/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train_F/train/ICAR/Udensenet_1/0.0001/0/'
model_path_ECAL = '/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train_F/train/ECAL/Udensenet_TF_1/0.0001/0/'
model_path_ECAR = '/data/yilinzhi/Segmentation/VMS/checkpoints/semi_train_F/train/ECAR/Udensenet_TF_1/0.001/1/'


for arti in ['CASCADE-ECAL', 'CASCADE-ECAR', 'CASCADE-ICAL', 'CASCADE-ICAR']:
    # art_dir = result_dir + '/' + arti  # 结果目录位置
    # if not os.path.exists(art_dir):
    #     os.mkdir(art_dir)

    print('dcm', arti, len(os.listdir(data_dir)))

    # process
    # todo：阶段一，预测图片
    model = udensenet(1, 1)
    device = torch.device('cuda:0')

    # todo: problem，单GPU还是多GPU?
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device=device)

    ### 加载已经训练好的模型：不同位置，不同模型（保存位置）
    if arti == 'CASCADE-ECAL':
        model_path = model_path_ECAL
    elif arti == 'CASCADE-ECAR':
        model_path = model_path_ECAR
    elif arti == 'CASCADE-ICAL':
        model_path = model_path_ICAL
    else:
        model_path = model_path_ICAR

    model.load_state_dict(torch.load(model_path + model_name))
    model.eval()

    create_dir(png_save_path + '/' + pi + '_' + arti)  # 预测图片保存位置：病例+位置
    # create_dir(npy_save_path + '/' + pi + '_' + arti)

    for index in range(len(img_files)):
        test_img = dcm_img[280:440, 110:610, index]  # 要预测的img：已经经过预处理

        test_img_1 = torch.from_numpy(test_img)
        test_img_2 = test_img_1.unsqueeze(0)
        test_img_3 = test_img_2.unsqueeze(0)
        test_img_3.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            pred = model(test_img_3)
            result = pred.squeeze().cpu().numpy()

        cv2.imwrite(png_save_path + '/' + pi + '_' + arti + '/' + str(index + 1) + '_.png',
                    result)
        # np.save(npy_save_path+'/' + pi + '_' + arti + '/' + str(index + 1) + '_.npy',result)

    # # save results in cascade format
    # qvjname = art_dir + '/E' + pi + '.QVJ'  # 保存QVJ文件到相应文件夹
    # with open(qvjname, 'w') as fp:
    #     fp.write('<?xml version="1.0" encoding="UTF-8"?>')
    #
    #
    # qvsname = art_dir + '/E' + pi + 'S101_L.QVS'  # 保存QVS到相应文件夹
    # with open(qvsname, 'w') as fp:
    #     fp.write('<?xml version="1.0" encoding="UTF-8"?>')




prefix = 'E'
suffix = 'S101_L.QVS'
print("%" * 10, pi, "&" * 10)

for arti in ['CASCADE-ECAL', 'CASCADE-ECAR', 'CASCADE-ICAL', 'CASCADE-ICAR']:  # 一个病例保存四个不同未知的QVS(xml）文件
    pred_png_path = png_save_path + '/' + pi + '_' + arti + '/'  # 预测图片保存位置
    pred_png_list = os.listdir(pred_png_path)  # 找到所有预测图片文件名
    pred_png_list.sort(key=lambda name: int(name.split('_')[0]))  # 按照slice的index排序

    QVAS_name = prefix + pi + suffix  # QVS文件名
    create_dir(result_dir + '/CASCADE/' + arti)  # QVS保存路径：一个病例四个不同位置
    save_path = result_dir + '/CASCADE/' + arti + '/' + QVAS_name

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
        # print("index:", index)

        image_name = prefix + pi + 'S101I' + index

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
        if len(contours) == 1:  # todo: 一个轮廓暂时先跳过
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
    print('%s %s QVS file finished.' % (pi, arti))

# TODO：生成QVJ文件！
print('%' * 10)
suffix_qvj = '.QVJ'
suffix_qvj_1 = 'S101_L'

# TODO：找到分叉图片的序号
bi_left_index = 200

for arti in ['CASCADE-ECAL', 'CASCADE-ECAR', 'CASCADE-ICAL', 'CASCADE-ICAR']:  # 一个病例保存四个不同未知的QVS(xml）文件
    QVAS_Project_name = prefix + pi + suffix_qvj  # QVJ文件名
    save_path = result_dir + '/CASCADE/' + arti + '/' + QVAS_Project_name

    QVAS_Project = ET.Element('ns0:QVAS_Project')  # 创建根结点
    QVAS_Project.set("xmlns:ns0", "vil.rad.washington.edu")

    QVAS_Version = ET.SubElement(QVAS_Project, 'QVAS_Version')  # root子元素：版本
    QVAS_Version.set(" xmlns", "")
    QVAS_Version.text = "1.0"

    QVAS_System_Info = ET.SubElement(QVAS_Project, 'QVAS_System_Info')  # root子元素：信息
    QVAS_System_Info.set(" xmlns", "")

    AnalysisMode = ET.SubElement(QVAS_System_Info, 'AnalysisMode')
    AnalysisMode.text = '1'
    ImageLocationStatus = ET.SubElement(QVAS_System_Info, 'ImageLocationStatus')
    ImageLocationStatus.text = '2'
    BifurcationLocation = ET.SubElement(QVAS_System_Info, 'BifurcationLocation')

    BifurcationImageIndex = ET.SubElement(BifurcationLocation, 'BifurcationImageIndex')
    BifurcationImageIndex.set('ImageIndex', str(bi_left_index))
    SeriesName = prefix + pi + suffix_qvj_1
    BifurcationImageIndex.set('SeriesName', SeriesName)

    tree = ET.ElementTree(QVAS_Project)
    tree.write(save_path, encoding='UTF-8', xml_declaration=True)
    print('%s %s QVJ file finished.' % (pi, arti))
