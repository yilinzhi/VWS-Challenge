# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/03/14 23:11
# @Author  : yilinzhi
# @FileName: mask_to_npy.py

import os
import numpy as np
from PIL import Image

input_dir = 'E:/Win10_data/Medical_Research/VMS/datasets/train_label/union_image_1'
output_dir = 'E:/Win10_data/Medical_Research/VMS/datasets/train_label/npy_avail_1'

for case in os.listdir(input_dir):
    path = input_dir+'/'+case
    print(os.listdir(path))

    if case =="P388":
        continue

    mask_pos=[]
    for name in os.listdir(path):
        img_crop=np.zeros((100,720),dtype=np.float32)
        pi =int(name.split('_')[2])
        mask_pos.append(int(pi))
        print(pi)

        img_path =path+'/'+name
        img = np.array(Image.open(img_path))
        img_crop=img[310:410, 110:610]
        print(img_crop.shape)

        np.save(output_dir+'/'+case+'_'+str(pi)+'_.npy',img_crop)
