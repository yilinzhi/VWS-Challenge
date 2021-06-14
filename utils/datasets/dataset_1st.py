# step1: 定义MyDataset类， 继承Dataset, 重写抽象方法：__len()__, __getitem()__
# 多分类：image_data, label1, label2, label3...
# 此处dataset：所有数据作为一个整体导入
# class MyDataset(Dataset):
#     # *args表示可能会有不固定数量的label
#     def __init__(self, train_data, train_label, *args):
#         self.train_data = train_data
#         self.train_label = train_label
#         self.para = 2
#         if(len(args)>=1):
#             self.train_label_100 = args[0]
#             self.para += 1
#         if(len(args)>=2):
#             self.train_label_50 = args[1]
#             self.para += 1
#         if(len(args)>=3):
#             self.train_label_25 = args[2]
#             self.para += 1
#
#     def __getitem__(self, index):
#         if self.para == 2:
#             return self.train_data[index], self.train_label[index]
#         if self.para == 3:
#             return self.train_data[index], self.train_label[index],self.train_label_100[index]
#         if self.para == 4:
#             return self.train_data[index], self.train_label[index],self.train_label_100[index],/
#             self.train_label_50[index]
#         if self.para == 5:
#             return self.train_data[index], self.train_label[index],self.train_label_100[index],/
#             self.train_label_50[index],self.train_label_25[index]
#
#
#     def __len__(self):
#         return self.train_data.size()[0]

import torch
import random
import os
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image

import sys

sys.path.append("/data/yilinzhi/Segmentation/VMS")
from options.arguments_1st import parse_args

args = parse_args()


def data_augmentation(is_traning, image, mask):
    #  image/mask：PIL or tensor, here is PIL
    if is_traning == True:
        p1 = random.randint(0, 1)
        p2 = random.randint(0, 1)

        # horizontal flip
        # if p1>0:
        #     image=F.hflip(image)
        #     mask=F.hflip(mask)

        # vertical flip
        if p2 > 0:
            image = F.vflip(image)
            mask = F.vflip(mask)

        # rotate
        angle = float(random.randint(-10, 10))
        p3 = random.random()
        if p3 > 0.5:
            image = F.rotate(image, angle=angle)
            mask = F.rotate(mask, angle=angle)

        # resize in different scale
        p4 = random.random()
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

        # shear mapping
        p6 = random.random()
        if p6 > 0.5:
            shear = random.uniform(-10.0, 10.0)
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
    """Custom Training Datasets.

       Attributes:
           split_datasets_files: Lists of training/validation files
           state: Training or validation
           args: Argument
    """

    def __init__(self, split_datasets_files, state, args):
        super(MyDataset_train, self).__init__()

        self.split_datasets_files = split_datasets_files
        self.state = state
        self.args = args

    def __getitem__(self, idx):
        if idx % 2 == 0:  # even: positive training sample(image/mask)
            if self.state == "train":
                # image/mask shape: 160*500
                image_positive_train = np.load(
                    os.path.join(self.args.train_image_sep_ICAR + '/positive', self.split_datasets_files[0][idx]))
                mask_positive_train = np.load(
                    os.path.join(self.args.train_mask_sep_ICAR + '/positive', self.split_datasets_files[2][idx]))

                # change numpy to PIL
                image_positive_train = Image.fromarray(image_positive_train)
                mask_positive_train = Image.fromarray(mask_positive_train)

                is_training = True
                image_aug, mask_aug = data_augmentation(is_training, image_positive_train, mask_positive_train)

                return {
                    "image": image_aug,
                    "mask": mask_aug
                }
            else:
                raise NotImplementedError("No this state")
        else:  # odd : negative training sample(image/mask)
            if self.state == "train":
                image_negative_train = np.load(
                    os.path.join(self.args.train_image_sep_ICAR + '/negative', self.split_datasets_files[1][idx]))
                mask_negative_train = np.load(
                    os.path.join(self.args.train_mask_sep_ICAR + '/negative', self.split_datasets_files[3][idx]))

                image_negative_train = Image.fromarray(image_negative_train)
                mask_negative_train = Image.fromarray(mask_negative_train)

                is_training = True
                image_aug, mask_aug = data_augmentation(is_training, image_negative_train, mask_negative_train)

                return {
                    "image": image_aug,
                    "mask": mask_aug
                }
            else:
                raise NotImplementedError("No this state")

    def __len__(self):
        return len(self.split_datasets_files[0])


class MyDataset_validation(Dataset):
    """Custom validation datasets.

       Attributes:
           Same meaning in Mydataset_training
       """

    def __init__(self, split_datasets_files, state, args):
        super(MyDataset_validation, self).__init__()

        self.split_datasets_files = split_datasets_files
        self.state = state
        self.args = args

    def __getitem__(self, idx):
        if idx % 2 == 0:
            if self.state == "validation":
                image_positive_validation = np.load(
                    os.path.join(self.args.train_image_sep_ICAR + '/positive', self.split_datasets_files[4][idx]))
                mask_positive_validation = np.load(
                    os.path.join(self.args.train_mask_sep_ICAR + '/positive', self.split_datasets_files[6][idx]))

                is_training = False

                image_positive_validation = Image.fromarray(image_positive_validation)
                mask_positive_validation = Image.fromarray(mask_positive_validation)

                image_aug, mask_aug = data_augmentation(is_training, image_positive_validation,
                                                        mask_positive_validation)

                return {
                    "image": image_aug,
                    "mask": mask_aug
                }
            else:
                raise NotImplementedError("No this state")
        else:
            if self.state == "validation":
                image_negative_validation = np.load(
                    os.path.join(self.args.train_image_sep_ICAR + '/negative', self.split_datasets_files[5][idx]))
                mask_negative_validation = np.load(
                    os.path.join(self.args.train_mask_sep_ICAR + '/negative', self.split_datasets_files[7][idx]))

                image_negative_validation = Image.fromarray(image_negative_validation)
                mask_negative_validation = Image.fromarray(mask_negative_validation)

                is_training = False
                image_aug, mask_aug = data_augmentation(is_training, image_negative_validation,
                                                        mask_negative_validation)

                return {
                    "image": image_aug,
                    "mask": mask_aug
                }
            else:
                raise NotImplementedError("No this state")

    def __len__(self):
        return len(self.split_datasets_files[4])


if __name__ == "__main__":
    image_path = "/data/yilinzhi/Segmentation/VMS/datasets/train_data/image_sep_position/ECAL/positive/P887_ECAL_322_.npy"
    mask_path = "/data/yilinzhi/Segmentation/VMS/datasets/train_label/circle_mask_sep/ECAL/positive/P887_ECAL_322_.npy"

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
