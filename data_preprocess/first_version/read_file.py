# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/03/17 15:38
# @Author  : yilinzhi
# @FileName: read_file.py

import SimpleITK as sitk
import numpy as np
import skimage.io as io
import os
import matplotlib.pyplot as plt


# # wmh：image and mask
# img_flair = sitk.ReadImage(
#     os.path.join(image_path, 'FLAIR.nii.gz'))
# shape_img = img_flair.GetSize()
# print("shape of image:", shape_img)
#
# img_manual = sitk.ReadImage(os.path.join(image_path, 'wmh.nii.gz'))
#
# array_flair = sitk.GetArrayFromImage(img_flair)
# array_manual = sitk.GetArrayFromImage(img_manual)
# print("array_flair.shape: ", array_flair.shape)
#
# img_process_1=np.load(os.path.join(image_path,"30_train_data.npy"))
# img_process_1=np.swapaxes(img_process_1,1,3)
# print("img_process_1.shape: ",img_process_1.shape)
# img_process_2=img_process_1[20][0]
# img_process_2=img_process_2.transpose((1,0))
# print("img_process_2.shape: ",img_process_2.shape)


# if np.max(array_flair)>0:
#     print(np.max(array_flair))
#     print(np.min(array_flair))
#
#
def show_img(data):
    for i in range(data.shape[0]):
        io.imshow(data[i,:,:],cmap='gray')
        print(i)
        io.show()


def show_img_1(ori_img):
    io.imshow(ori_img[25], cmap = 'gray')
    io.show()

# show_img_1(array_flair)
# io.imshow(img_process_2,cmap = 'gray')
# io.show()
# show_img_1(array_manual)

# # 读取dcm文件和png
# dcm_img=sitk.ReadImage(os.path.join(image_path,"EP429S101I316.dcm"))
# dcm_array=sitk.GetArrayFromImage(dcm_img)
# show_img(dcm_array)
# print("dcm.shape: ",dcm_array.shape)

print("++++++++++++++++++++++++++++++")
train_pos_data_path=r'E:\Win10_data\Segmentation\VMS_Unet\datasets\train_data\sep\ICAR\positive'
img_process=np.load(os.path.join(train_pos_data_path,"P125_ICAR_308_.npy"))
print("process_shape:",img_process.shape)
io.imshow(img_process,cmap='gray')
io.show()

train_neg_data_path=r'E:\Win10_data\Segmentation\VMS_Unet\datasets\train_data\sep\ICAL\negative'
img_process=np.load(os.path.join(train_neg_data_path,"P125_ICAL_215_.npy"))
print("process_shape:",img_process.shape)
io.imshow(img_process,cmap='gray')
io.show()


train_pos_label_path=r'E:\Win10_data\Segmentation\VMS_Unet\datasets\train_label\mask_sep\ICAR\positive'
img_process=np.load(os.path.join(train_pos_label_path,"P125_ICAR_461_.npy"))
print("process_shape:",img_process.shape)
io.imshow(img_process[:,:,0],cmap='gray')
io.show()

train_neg_label_path=r'E:\Win10_data\Segmentation\VMS_Unet\datasets\train_label\mask_sep\ICAR\negative'
img_process=np.load(os.path.join(train_neg_label_path,"P125_ICAR_201_.npy"))
print("process_shape:",img_process.shape)
io.imshow(img_process[:,:,0],cmap='gray')
io.show()



