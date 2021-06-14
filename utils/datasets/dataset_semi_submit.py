# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/05/17 21:18
# @Author  : Yi
# @FileName: dataset_semi_submit.py


import torch
import random
import os
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from options.arguments_semi_submit import parse_args
args = parse_args()


def data_augmentation(is_traning, image, mask):
    #  image/mask：PIL or tensor, here is PIL
    if is_traning == True:
        # p1 = random.randint(0, 1)
        # p2 = random.randint(0, 1)

        # horizontal flip
        # if p1>0:
        #     image=F.hflip(image)
        #     mask=F.hflip(mask)

        # vertical flip
        # if p2 > 0:
        #     image = F.vflip(image)
        #     mask = F.vflip(mask)

        angle = float(random.randint(-5, 5))  # 旋转角度
        p3 = random.random()
        if p3 > 0.5:
            image = F.rotate(image, angle=angle)
            mask = F.rotate(mask, angle=angle)

        p4 = random.random()  # 缩放：放大或缩小
        if p4 > 0.5:
            p5 = random.random()
            if p5 > 0.5:  # enlarge
                image = F.resize(image, size=(176, 550))
                mask = F.resize(mask, size=(176, 550))

                image = F.crop(image, 8, 25, 160, 500)
                mask = F.crop(mask, 8, 25, 160, 500)
            else:  # shrink
                image = F.resize(image, size=(144, 450))
                mask = F.resize(mask, size=(144, 450))

                image = F.pad(image, padding=(25, 8), fill=0)
                mask = F.pad(mask, padding=(25, 8), fill=0)

        p6 = random.random()  # 仿射变换
        if p6 > 0.5:
            shear = random.uniform(-5.0, 5.0)
            image = F.affine(image, 0, translate=(0, 0), scale=(1.0), shear=shear)
            mask = F.affine(mask, 0, translate=(0, 0), scale=(1.0), shear=shear)

        # PIL=>numpy => tensor
        image = F.to_tensor(np.array(image))
        mask = F.to_tensor(np.array(mask))

        return image, mask

    elif is_traning == False:
        image = F.to_tensor(np.array(image))
        mask = F.to_tensor(np.array(mask))
        return image, mask
    else:
        print("No such state!")


class MyDataset_train(Dataset):
    """定义训练集：有标签数据集，将image和mask组合数据集
    """

    def __init__(self, img_train_list, mask_train_list):
        super(MyDataset_train, self).__init__()

        self.img_train_list = img_train_list  # image训练集列表：正负样本
        self.mask_train_list = mask_train_list  # mask训练集列表：正负样本

    def __getitem__(self, idx):
        # image/mask shape: 160*500
        # image_positive_train = np.load(
        #     os.path.join(self.args.image_ICAL_semi, self.img_train_list[idx]))
        # mask_positive_train = np.load(
        #     os.path.join(self.args.mask_ICAL_semi, self.mask_train_list[idx]))

        image_train = np.load(self.img_train_list[idx])
        mask_train = np.load(self.mask_train_list[idx])

        # change numpy to PIL
        image_train = Image.fromarray(image_train)
        mask_train = Image.fromarray(mask_train)

        is_training = True  # 是否进行数据增强
        image_aug, mask_aug = data_augmentation(is_training, image_train, mask_train)

        return {
            "image": image_aug,
            "mask": mask_aug
        }


    def __len__(self):
        return len(self.img_train_list)


class MyDataset_train_un(Dataset):
    """定义训练集：无标签，将image和mask组合数据集
        返回image,mask,index,next_index
    """

    def __init__(self, img_train_list, mask_train_list):
        super(MyDataset_train_un, self).__init__()

        self.img_train_list = img_train_list  # image训练集列表：正负样本
        self.mask_train_list = mask_train_list  # mask训练集列表：正负样本

    def __getitem__(self, idx):
        # image/mask shape: 160*500
        image_positive_train = np.load(self.img_train_list[idx])
        mask_positive_train = np.load(self.mask_train_list[idx])

        split_name = self.img_train_list[idx].split('_')
        true_index = split_name[-4]
        fake_index = split_name[-3]

        # change numpy to PIL
        image_positive_train = Image.fromarray(image_positive_train)
        mask_positive_train = Image.fromarray(mask_positive_train)

        is_training = True  # 是否进行数据增强
        image_aug, mask_aug = data_augmentation(is_training, image_positive_train, mask_positive_train)

        return {
            "image": image_aug,
            "mask": mask_aug,
            "true_index": true_index,
            "fake_index": fake_index
        }

    def __len__(self):
        return len(self.img_train_list)


class MyDataset_val(Dataset):
    """Custom validation datasets.

       Attributes:
           Same meaning in Mydataset_training
       """

    def __init__(self, img_val_list, mask_val_list):
        super(MyDataset_val, self).__init__()

        self.img_val_list = img_val_list
        self.mask_val_list = mask_val_list

    def __getitem__(self, idx):

        image_val = np.load(self.img_val_list[idx])
        mask_val = np.load(self.mask_val_list[idx])

        image_val = Image.fromarray(image_val)
        mask_val = Image.fromarray(mask_val)

        is_training = False
        image_aug, mask_aug = data_augmentation(is_training, image_val,
                                                mask_val)

        return {
            "image": image_aug,
            "mask": mask_aug
        }

    def __len__(self):
        return len(self.img_val_list)


class MyDataset_val_un(Dataset):
    """定义训练集：无标签，将image和mask组合数据集
        返回image,mask,index,next_index
    """

    def __init__(self, img_val_list, mask_val_list):
        super(MyDataset_val_un, self).__init__()

        self.img_val_list = img_val_list  # image训练集列表：正负样本
        self.mask_val_list = mask_val_list  # mask训练集列表：正负样本

    def __getitem__(self, idx):
        # image/mask shape: 160*500
        image_positive_val = np.load(self.img_val_list[idx])
        mask_positive_val = np.load(self.mask_val_list[idx])

        split_name = self.img_val_list[idx].split('_')
        true_index = split_name[-4]
        fake_index = split_name[-3]

        # change numpy to PIL
        image_positive_val = Image.fromarray(image_positive_val)
        mask_positive_val = Image.fromarray(mask_positive_val)

        is_valing = False  # 是否进行数据增强
        image_aug, mask_aug = data_augmentation(is_valing, image_positive_val, mask_positive_val)

        return {
            "image": image_aug,
            "mask": mask_aug,
            "true_index": true_index,
            "fake_index": fake_index
        }

    def __len__(self):
        return len(self.img_val_list)


if __name__ == "__main__":
    # image_path = "/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_semi_F/train/ECAL/P887_ECAL_322_posLabel_.npy"
    # mask_path = "/data/yilinzhi/Segmentation/VMS/datasets/train_label/mask_semi_F/train/ECAL/P887_ECAL_322_posLabel_.npy"

    image_path=args.image_ECAL_semi+'/P887_ECAL_322_posLabel_.npy'
    mask_path=args.mask_ECAL_semi+'/P887_ECAL_322_posLabel_.npy'


    image = np.load(image_path)
    mask = np.load(mask_path)

    # Changing numpy to PIL
    # image_PIL=Image.fromarray(image)
    # mask_PIL=Image.fromarray(mask)

    image_1 = F.to_pil_image(image)
    mask_1 = F.to_pil_image(mask)

    is_training = True
    image_aug, mask_aug = data_augmentation(is_training, image_1, mask_1)

    # image_aug/mask_aug: tensor ,1*160*500
    # c*h*w => h*w
    image_aug = np.squeeze(image_aug)
    mask_aug = np.squeeze(mask_aug)

    print(image_aug.shape)
    print(mask_aug.shape)

    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(2, 2, 1)
    img1 = ax1.imshow(image, cmap='gray')
    plt.colorbar(img1)

    ax2 = fig.add_subplot(2, 2, 2)
    img2 = ax2.imshow(image_aug, cmap='gray')
    plt.colorbar(img2)

    ax3 = fig.add_subplot(2, 2, 3)
    img3 = ax3.imshow(mask, cmap='gray')
    plt.colorbar(img3)

    ax4 = fig.add_subplot(2, 2, 4)
    img4 = ax4.imshow(mask_aug, cmap='gray')
    plt.colorbar(img4)

    plt.show()
