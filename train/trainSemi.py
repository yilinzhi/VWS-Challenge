# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/06/03 19:27
# @Author  : Yi
# @FileName: trainSemi.py

import os
import shutil
import sys
import time
import random
import copy
import glob
import scipy.spatial
import torchvision
import pandas as pd
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
import torch.nn.functional as F
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

sys.path.append('E:/Win10_data/Segmentation/VMS_Unet/models')
from models.UNet import UNet
from models.udensenet import udensenet

from utils.datasets.dataset_semi_submit import MyDataset_train, MyDataset_val, MyDataset_train_un, MyDataset_val_un
from options.arguments_semi_submit import parse_args


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 计算精度和效率的平衡
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class dice_ratio_loss(nn.Module):
    """Compute dice loss for pred_masks and truth_masks."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_pred_f = y_pred.view(-1)
        y_true_f = y_true.view(-1)
        intersection = torch.sum(y_true_f * y_pred_f)
        return 1 - (2.0 * intersection + 1) / (torch.sum(y_true_f) +
                                               torch.sum(y_pred_f) + 1)


class FocalLoss(nn.Module):
    """Compute focal loss for unbalanced examples and hard learned examples.

    Attributes:
        alpha: A weighted factor for addressing class imbalance.
            Alpha for class 1, 1-alpha for class -1.
        gamma: An factor to differentiate easy/hard examples.
        logits：Whether computed by a Sigmoid layer.
        reduce: Should by changed by reduction
    """

    def __init__(self, alpha=0.5, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        # ‘none’ 为求 minibatch 中每个 sample 的 loss 值.
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # 返回平均值
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class Logger(object):
    """Logging information in the terminal."""

    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def getDSC(true_mask, pred_mask):
    """评价指标一：计算dice值"""

    t_mask = true_mask.flatten()
    p_mask = pred_mask.flatten()

    # similarity = 1.0 - dissimilarity
    return 1.0 - scipy.spatial.distance.dice(t_mask, p_mask)


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


# TODO：检查划分是否正确：划分出训练+验证，测试集：有标记的正样本，负样本，无标记的正样本
# TODO：进行划分 + 检查是否划分正确！
def split_Label_test(img_path, mask_path):
    """对有标签的数据集进行划分：训练+验证，测试集

        样本集都是先负后正：找出正负样本，按比例划分。

    :param img_path: 图像路径
    :param mask_path: mask路径
    :return:
    """

    neg_img_list = glob.glob(img_path + "/*_negLabel_.npy")  # 负样本：图片，文件名列表
    pos_img_list = glob.glob(img_path + "/*_posLabel_.npy")  # 正样本：图片

    neg_mask_list = glob.glob(mask_path + "/*_negLabel_.npy")  # 负样本：mask
    pos_mask_list = glob.glob(mask_path + "/*_posLabel_.npy")  # 正样本：mask

    total_nums = len(os.listdir(img_path))  # 图像/mask：正负样本总的数目
    neg_nums = len(neg_mask_list)  # 负样本数目
    pos_nums = len(pos_mask_list)  # 正样本数目

    k = 10
    neg_batch = neg_nums // k  # 9份：训练集+验证集，1份：测试集
    pos_batch = pos_nums // k

    neg_start = 0
    neg_end = neg_batch
    pos_start = 0
    pos_end = pos_batch

    neg_shuffle_indices = np.random.permutation(neg_nums)  # 负样本随机排列
    pos_shuffle_indices = np.random.permutation(pos_nums)  # 正样本随机排列

    neg_img_shuffle = []  # 图像：负
    pos_img_shuffle = []  # 图像：正
    neg_mask_shuffle = []  # mask：负
    pos_mask_shuffle = []  # mask：正

    for index in neg_shuffle_indices:  # 负样本
        neg_img_shuffle.append(neg_img_list[index])
        neg_mask_shuffle.append(neg_mask_list[index])

    for index in pos_shuffle_indices:  # 正样本
        pos_img_shuffle.append(pos_img_list[index])
        pos_mask_shuffle.append(pos_mask_list[index])

    split_label_data = []

    neg_img_test = neg_img_shuffle[neg_start:neg_end]  # 负样本，图片，测试集
    neg_mask_test = neg_mask_shuffle[neg_start:neg_end]  # 负样本，mask，测试集

    del neg_img_shuffle[neg_start:neg_end]  # 负样本，图片，训练+验证集
    del neg_mask_shuffle[neg_start:neg_end]  # 负样本，mask，训练+验证集

    pos_img_test = pos_img_shuffle[pos_start:pos_end]  # 正样本，图片，测试集
    pos_mask_test = pos_mask_shuffle[pos_start:pos_end]  # 正样本，mask，测试集

    del pos_img_shuffle[pos_start:pos_end]  # 正样本，图片，训练+验证集
    del pos_mask_shuffle[pos_start:pos_end]  # 正样本，mask，训练+验证集

    split_label_data.append(neg_img_shuffle)  # 图片：负样本，训练+验证
    split_label_data.append(pos_img_shuffle)  # 图片：正样本，训练+验证
    split_label_data.append(neg_mask_shuffle)
    split_label_data.append(pos_mask_shuffle)
    split_label_data.append(neg_img_test)
    split_label_data.append(pos_img_test)
    split_label_data.append(neg_mask_test)
    split_label_data.append(pos_mask_test)

    return split_label_data


def train_val_label(split_data, kfolds, fold, batch_size):
    """对有标签：正负样本（image/mask）进行划分

    :param split_data:
    :return:
    """

    # step1：划分训练集和测试集，image/mask
    neg_img = split_data[0]
    pos_img = split_data[1]
    neg_mask = split_data[2]
    pos_mask = split_data[3]

    neg_nums = len(neg_img)  # 负样本数目
    pos_nums = len(pos_img)  # 正样本数目

    neg_batch = neg_nums // kfolds
    pos_batch = pos_nums // kfolds

    neg_start = neg_batch * fold
    neg_end = neg_batch * (fold + 1)
    pos_start = pos_batch * fold
    pos_end = pos_batch * (fold + 1)

    neg_img_val = neg_img[neg_start:neg_end]  # 第fold+1折作为验证集
    pos_img_val = pos_img[pos_start:pos_end]
    neg_mask_val = neg_mask[neg_start:neg_end]
    pos_mask_val = pos_mask[pos_start:pos_end]

    del neg_img[neg_start:neg_end]  # 训练集
    del pos_img[pos_start:pos_end]
    del neg_mask[neg_start:neg_end]
    del pos_mask[pos_start:pos_end]

    neg_train_nums = len(neg_img)  # 训练集：负样本
    pos_train_nums = len(pos_img)  # 训练集：正样本
    neg_val_nums = len(neg_img_val)  # 验证集：负样本
    pos_val_nums = len(pos_img_val)  # 验证集：正样本

    neg_img.extend(pos_img)  # image：训练集，正样本+负样本
    neg_mask.extend(pos_mask)  # mask：训练集
    neg_img_val.extend(pos_img_val)  # image：验证集
    neg_mask_val.extend(pos_mask_val)  # mask：验证集

    # step2：组合成训练集：(image,mask)

    train_datasets = MyDataset_train(neg_img, neg_mask)

    # step3：定义训练集采样器sampler
    class_count = torch.tensor([neg_train_nums, pos_train_nums])
    target = torch.cat(
        (torch.zeros(class_count[0], dtype=torch.long), torch.ones(class_count[1], dtype=torch.long)))
    class_sample_count = torch.tensor(
        [(target == t).sum() for t in torch.unique(target, sorted=True)])
    weight = 1.0 / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in target])

    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    train_loader_label = DataLoader(train_datasets, batch_size=batch_size, sampler=sampler, num_workers=8,
                                    pin_memory=True)

    # step4：组合成验证集：(image,mask)
    val_datasets = MyDataset_val(neg_img_val, neg_mask_val)

    # step5：定义验证集采样器sampler
    class_count_val = torch.tensor([neg_val_nums, pos_val_nums])
    target_val = torch.cat(
        (torch.zeros(class_count_val[0], dtype=torch.long), torch.ones(class_count_val[1], dtype=torch.long)))
    class_sample_count_val = torch.tensor(
        [(target_val == t).sum() for t in torch.unique(target_val, sorted=True)])
    weight_val = 1.0 / class_sample_count_val.float()
    samples_weight_val = torch.tensor([weight_val[t] for t in target_val])

    sampler_val = WeightedRandomSampler(samples_weight_val, len(samples_weight_val))
    val_loader_label = DataLoader(val_datasets, batch_size=batch_size, sampler=sampler_val, num_workers=8,
                                  pin_memory=True)

    return train_loader_label, val_loader_label


def split_Unlabel_test(image_path, mask_path):
    """对伪标签数据集（全是正样本）进行划分：训练+验证，测试"""

    image_list = glob.glob(image_path + "/*_posUnLabel_.npy")  # 图片文件名列表
    mask_list = glob.glob(mask_path + "/*_posUnLabel_.npy")  # mask文件名列表

    nums = len(image_list)  # 伪标签：图片/mask个数
    size = nums // 10
    start = 0
    end = size
    # print("nums ", nums)

    image_files_shuffle = []
    mask_files_shuffle = []

    shuffle_indices = np.random.permutation(nums)  # 随机排列
    for index in shuffle_indices:
        image_files_shuffle.append(image_list[index])
        mask_files_shuffle.append(mask_list[index])

    image_test = image_files_shuffle[start:end]
    mask_test = mask_files_shuffle[start:end]

    del image_files_shuffle[start: end]
    del mask_files_shuffle[start: end]

    split_unlabel_data = []
    split_unlabel_data.append(image_files_shuffle)
    split_unlabel_data.append(mask_files_shuffle)
    split_unlabel_data.append(image_test)
    split_unlabel_data.append(mask_test)

    return split_unlabel_data


def train_val_unLabel(split_data, kfolds, fold, batch_size):
    """划分未标注的数据集

    :param split_data:
    :return:
    """

    # step1：划分训练集和测试集，image/mask
    img = split_data[0]
    mask = split_data[1]

    nums = len(img)  # image/mask:数目
    batch = nums // kfolds
    start = batch * fold
    end = batch * (fold + 1)

    img_val = img[start:end]  # 第fold+1折作为验证集
    mask_val = mask[start:end]

    del img[start:end]
    del mask[start:end]

    # step2：组合成训练集：(image,mask)
    train_datasets = MyDataset_train_un(img, mask)
    train_loader_unLabel = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=8,
                                      pin_memory=True)

    # step3：组合成验证集：(image,mask)
    val_datasets = MyDataset_val_un(img_val, mask_val)
    val_loader_unLabel = DataLoader(val_datasets, batch_size=batch_size, shuffle=False, num_workers=8,
                                    pin_memory=True)

    return train_loader_unLabel, val_loader_unLabel


def train_net(
        device,
        check_path,
        split_data,
        split_unLable_data,
        lr,
        kfolds,
        epochs,
        batch_size1,
        batch_size2):
    """Training Unet for VMS.

        Args:
            device: CPU or CUDA
            image_path/mask_path：Datasets path
            check_path: Saved path for checking
            split_data: image/mask 划分的训验集，测试集
            args: Arguments
            lr: Learning rate
            kfolds: 5 folds for training
            epochs: Training epochs of each fold
            batch_size: Input size of image/mask

        Saved:
            model
            training/validation loss
            validation dice value
        """
    # TODO：修改
    create_dir(check_path)  # 创造保存文件夹
    shutil.copyfile("trainSemi.py", check_path + '/trainSemi.py')  # 保存代码，即保存相关参数。
    shutil.copyfile("../utils/datasets/dataset_semi_submit.py", check_path + '/dataset_semi_submit.py')

    for fold in range(kfolds):
        print("=" * 20, "kfold: ", fold + 1, "=" * 20)

        # each fold should initialize Unet/Udensenet model
        # binary segmentation：sigmoid()->prob
        # multi segmentation: softmax()->prob
        # multi_GPU training

        # TODO: Different model!
        # net = UNet(1, 1)
        net = udensenet(1, 1)
        net = nn.DataParallel(net, device_ids=[0, 1,2])
        net.to(device=device)

        # TODO: Modification!
        if fold > 1:  # hyper parameters adjusting
            break

        # dice loss for binary segmentation
        # BCE loss -> focal loss
        # TODO: 参考文章修改BCE loss和dice loss的比例
        # TODO: 论文代码：BCE loss如何调整？
        criterion_dice = dice_ratio_loss()
        dice_rate = 1

        criterion_bce = nn.BCELoss()
        bce_rate = 1

        optimizer = optim.Adam(net.parameters(), lr=lr)

        # 采用不同的学习率调整策略：规律，不规律，按照指标
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 60], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1,
                                                               patience=5,threshold=0.1)

        fold_check_path = check_path + '/' + str(fold)
        create_dir(fold_check_path)

        # Information saving: log.txt and tensorboardX
        # TODO: 模型_位置_数据增强_损失函数_log.txt
        sys.stdout = Logger(fold_check_path + '/Udensenet_il_da1_1dice_1bce_log.txt')
        writer = SummaryWriter(logdir=fold_check_path + '/run', comment=f'LR_{lr}')

        train_loss_all_epochs = []  # 记录所有轮次，训练集的loss
        val_loss_all_epochs = []  # 记录所有轮次，验证集的loss
        dice_similarity_all_epochs = []  # 记录所有轮次，验证集的loss

        batch_num_all_epochs = 0  # 训练集，所有epochs里累计batch数目
        val_batch_num_all_epochs = 0  # 验证集，所有epochs里累计batch数目

        # TODO：训练+验证集 => 按照第fold+1折划分，训练集+验证集：负样本+正样本，需要定义权重采样器。

        # step1：得到第fold+1折对应的训练集和验证集：image和mask
        # neg_img = split_data[0]
        # pos_img = split_data[1]
        # neg_mask = split_data[2]
        # pos_mask = split_data[3]
        #
        # neg_nums = len(neg_img)  # 负样本数目
        # pos_nums = len(pos_img)  # 正样本数目
        #
        # neg_batch = neg_nums // kfolds
        # pos_batch = pos_nums // kfolds
        #
        # neg_start = neg_batch * fold
        # neg_end = neg_batch * (fold + 1)
        # pos_start = pos_batch * fold
        # pos_end = pos_batch * (fold + 1)
        #
        # neg_img_val = neg_img[neg_start:neg_end]  # 第fold+1折作为验证集
        # pos_img_val = pos_img[pos_start:pos_end]
        # neg_mask_val = neg_mask[neg_start:neg_end]
        # pos_mask_val = pos_mask[pos_start:pos_end]
        #
        # del neg_img[neg_start:neg_end]
        # del pos_img[pos_start:pos_end]
        # del neg_mask[neg_start:neg_end]
        # del pos_mask[pos_start:pos_end]
        #
        # neg_train_nums = len(neg_img)
        # pos_train_nums = len(pos_img)
        # neg_val_nums = len(neg_img_val)
        # pos_val_nums = len(pos_img_val)
        #
        # neg_img.extend(pos_img)  # image：训练集
        # neg_mask.extend(pos_mask)  # mask：训练集
        # neg_img_val.extend(pos_img_val)  # image：验证集
        # neg_mask_val.extend(pos_mask_val)  # mask：验证集
        #
        # # step2: img/mask得到训练集(x,y)
        # train_datasets = MyDataset_train(neg_img, neg_mask, state="train", args=args)
        #
        # # step3：定义sampler：需要查看是否正确
        # class_count = torch.tensor([neg_train_nums, pos_train_nums])
        # target = torch.cat(
        #     (torch.zeros(class_count[0], dtype=torch.long), torch.ones(class_count[1], dtype=torch.long)))
        # class_sample_count = torch.tensor(
        #     [(target == t).sum() for t in torch.unique(target, sorted=True)])
        # weight = 1.0 / class_sample_count.float()
        # samples_weight = torch.tensor([weight[t] for t in target])
        #
        # sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        # train_loader = DataLoader(train_datasets, batch_size=batch_size, sampler=sampler, num_workers=8,
        #                           pin_memory=True)

        # 有标签的训练集+验证集：dataloader(小），无标签的训练集+验证集：dataloader（大）
        # 有标签：训练+验证：batch_size1，无标签：训练+验证：batch_size2
        train_loader_label, val_loader_label = train_val_label(split_data, kfolds, fold, batch_size1)
        train_loader_unLabel, val_loader_unLabel = train_val_unLabel(split_unLable_data, kfolds, fold, batch_size2)

        for epoch in range(epochs):  # 每折，训练总的轮次，每个轮次都会训练所有的样本

            train_epoch_loss = 0
            train_epoch_bce = 0
            train_epoch_dice = 0

            val_epoch_loss = 0
            val_epoch_dice = 0

            # training part:
            net.train()
            batch_num_each_epoch = 0  # 每折中batch数目
            for i_batch, (data, un_data) in enumerate(zip(cycle(train_loader_label), train_loader_unLabel)):
                # 少量有标签的训练集，大量无标签的训练集
                images = data['image']  # 有标签数据batch, [batch_size1,1,160,500]
                true_masks = data['mask']

                un_images = un_data['image']
                un_true_masks = un_data['mask']
                true_idx = un_data['true_index']  # batch_size2
                fake_idx = un_data['fake_index']

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                un_images = un_images.to(device=device, dtype=torch.float32)
                un_true_masks = un_true_masks.to(device=device, dtype=torch.float32)

                pred_masks = net(images)  # 有标签数据集预测
                un_pred_masks = net(un_images)  # 无标签数据预测

                loss_dice = criterion_dice(pred_masks, true_masks)  # 计算有标签训练集:loss
                loss_bce = criterion_bce(pred_masks, true_masks)
                loss_label = loss_dice * dice_rate + loss_bce * bce_rate

                loss_un = 0.0
                for idx in range(len(true_idx)):  # 迭代每个无标签样本，大小是batch_size，但是有时不能整除！
                    j = int(true_idx[idx])
                    i = int(fake_idx[idx])

                    un_loss_dice = criterion_dice(un_pred_masks[idx], un_true_masks[idx])
                    un_loss_bce = criterion_bce(un_pred_masks[idx], un_true_masks[idx])
                    loss_un += (1.0 / ((i - j) ** 2 + 0.01)) * (dice_rate * un_loss_dice + bce_rate * un_loss_bce)

                loss_un = float(loss_un) / len(true_idx)  # 计算无标签样本的平均值

                # TODO：有无标签比例
                loss = (loss_label + loss_un) / 2.0  # 总loss

                optimizer.zero_grad()  # 优化器更新
                loss.backward()  # loss反向传播
                nn.utils.clip_grad_value_(net.parameters(), 0.1)  # 防止梯度爆炸，进行梯度裁剪
                optimizer.step()

                train_epoch_loss += loss.item()  # 每个epoch，累计计算一个batch的loss
                train_epoch_bce += loss_bce.item()  # 训练集：有标签的bce loss
                train_epoch_dice += loss_dice.item()

                batch_num_each_epoch += 1  # 每个epoch：batch_num
                batch_num_all_epochs += 1  # 所有epochs

                if batch_num_all_epochs % 5 == 0:
                    writer.add_scalar("training loss/batch", loss.item(), global_step=batch_num_all_epochs)
                    writer.add_scalar("training label loss_dice/batch", loss_dice.item(),
                                      global_step=batch_num_all_epochs)
                    writer.add_scalar("training label loss_bce/batch", loss_bce.item(),
                                      global_step=batch_num_all_epochs)

                    for name, para in net.named_parameters():
                        writer.add_histogram(name, para.data.cpu().numpy(), global_step=batch_num_all_epochs)

                    for i in range(pred_masks.shape[0]):  # 有标签
                        image_1 = images[i, :, :, :]
                        image_1 = image_1[np.newaxis, :, :, :]
                        true_masks_1 = true_masks[i, :, :, :]
                        true_masks_1 = true_masks_1[np.newaxis, :, :, :]
                        pred_masks_1 = pred_masks[i, :, :, :]
                        pred_masks_1 = pred_masks_1[np.newaxis, :, :, :]

                        # original image, truth mask and pred mask
                        images_list = torch.cat(
                            (image_1, true_masks_1, pred_masks_1), dim=0)
                        writer.add_image("image : mask : pred:" + f"{i + 1}",
                                         torchvision.utils.make_grid(images_list, nrow=3, padding=10, normalize=True),
                                         global_step=batch_num_all_epochs)

                    for i in range(un_pred_masks.shape[0]):  # 无标签
                        image_2 = un_images[i, :, :, :]
                        image_2 = image_2[np.newaxis, :, :, :]
                        true_masks_2 = un_true_masks[i, :, :, :]
                        true_masks_2 = true_masks_2[np.newaxis, :, :, :]
                        pred_masks_2 = un_pred_masks[i, :, :, :]
                        pred_masks_2 = pred_masks_2[np.newaxis, :, :, :]

                        # original image, truth mask and pred mask
                        images_list = torch.cat(
                            (image_2, true_masks_2, pred_masks_2), dim=0)
                        writer.add_image("unLabel image:fake mask:pred:" + f"{i + 1}",
                                         torchvision.utils.make_grid(images_list, nrow=3, padding=10, normalize=True),
                                         global_step=batch_num_all_epochs)

            # averaged training loss for each epoch
            train_epoch_avg_loss = train_epoch_loss / batch_num_each_epoch
            train_loss_all_epochs.append(train_epoch_avg_loss)
            print(
                'Epoch:{} finished ! \ttrain_Loss: {:.4f},\tlabel_dice_loss:{:.4f},\tlabel_bce_loss:{:.4f}'.format(
                    epoch + 1, train_loss_all_epochs[-1], float(train_epoch_dice) / batch_num_each_epoch,
                    float(train_epoch_bce) / batch_num_each_epoch))

            # TODO: modification
            if (epoch + 1) % 15 == 0:
                torch.save(net.state_dict(), fold_check_path + '/Udensenet_' + str(epoch + 1) + '.pkl')

            # 验证阶段:
            # val_datasets = MyDataset_validation(neg_img_val, neg_mask_val, state="validation", args=args)
            #
            # class_count_val = torch.tensor([neg_val_nums, pos_val_nums])
            # target_val = torch.cat(
            #     (torch.zeros(class_count_val[0], dtype=torch.long), torch.ones(class_count_val[1], dtype=torch.long)))
            # class_sample_count_val = torch.tensor(
            #     [(target_val == t).sum() for t in torch.unique(target_val, sorted=True)])
            # weight_val = 1.0 / class_sample_count_val.float()
            # samples_weight_val = torch.tensor([weight_val[t] for t in target_val])
            #
            # sampler_val = WeightedRandomSampler(samples_weight_val, len(samples_weight_val))
            # validation_loader = DataLoader(val_datasets, batch_size=batch_size, sampler=sampler_val, num_workers=8,
            #                                pin_memory=True)

            net.eval()
            val_batch_num_each_epoch = 0
            for val_i_batch, (val_data, un_val_data) in enumerate(zip(cycle(val_loader_label), val_loader_unLabel)):
                with torch.no_grad():
                    val_images = val_data['image']  # 验证集：有标签
                    val_true_masks = val_data['mask']

                    un_val_images = un_val_data['image']  # 验证集：无标签
                    un_val_true_masks = un_val_data['mask']
                    un_true_idx = un_val_data['true_index']
                    un_fake_idx = un_val_data['fake_index']

                    val_images = val_images.to(device=device, dtype=torch.float32)
                    val_true_masks = val_true_masks.to(device=device, dtype=torch.float32)
                    un_val_images = un_val_images.to(device=device, dtype=torch.float32)
                    un_val_true_masks = un_val_true_masks.to(device=device, dtype=torch.float32)

                    val_pred_masks = net(val_images)
                    un_val_pred_masks = net(un_val_images)

                    val_loss_dice = criterion_dice(val_pred_masks, val_true_masks)
                    val_loss_bce = criterion_bce(val_pred_masks, val_true_masks)

                    val_loss_label = val_loss_dice * dice_rate + val_loss_bce * bce_rate  # 验证集：有标签

                    val_loss_un = 0.0
                    for idx in range(len(un_true_idx)):
                        i = int(un_true_idx[idx])
                        j = int(un_fake_idx[idx])

                        un_val_loss_dice = criterion_dice(un_val_pred_masks[idx], un_val_true_masks[idx])
                        un_val_loss_bce = criterion_bce(un_val_pred_masks[idx], un_val_true_masks[idx])

                        val_loss_un += (1.0 / ((i - j) ** 2 + 0.01)) * (
                                dice_rate * un_val_loss_dice + bce_rate * un_val_loss_bce)
                    val_loss_un = float(val_loss_un) / len(un_true_idx)

                    # TODO：有无标签比例
                    val_loss = (val_loss_un + val_loss_label) / 2.0

                    val_epoch_loss += val_loss.item()
                    val_batch_num_all_epochs += 1
                    val_batch_num_each_epoch += 1

                    # 验证集的评价指标：有标签数据的dice值
                    dice_similarity = getDSC(val_true_masks.cpu().numpy(), val_pred_masks.cpu().numpy())
                    val_epoch_dice += dice_similarity

                    if val_batch_num_all_epochs % 5 == 0:
                        writer.add_scalar("validation loss/batch", val_loss.item(),
                                          global_step=val_batch_num_all_epochs)
                        writer.add_scalar("dice_similarity/batch", dice_similarity,
                                          global_step=val_batch_num_all_epochs)

                        for i in range(val_pred_masks.shape[0]):  # 有标签验证集
                            image_3 = val_images[i, :, :, :]
                            image_3 = image_3[np.newaxis, :, :, :]
                            true_masks_3 = val_true_masks[i, :, :, :]
                            true_masks_3 = true_masks_3[np.newaxis, :, :, :]
                            pred_masks_3 = val_pred_masks[i, :, :, :]
                            pred_masks_3 = pred_masks_3[np.newaxis, :, :, :]

                            images_list = torch.cat(
                                (image_3, true_masks_3, pred_masks_3), dim=0)
                            writer.add_image("val image:mask:pred:" + f"{i + 1}",
                                             torchvision.utils.make_grid(images_list, nrow=3, padding=10,
                                                                         normalize=True),
                                             global_step=val_batch_num_all_epochs)

                        for i in range(un_val_pred_masks.shape[0]):  # 无标签验证集
                            image_4 = un_val_images[i, :, :, :]
                            image_4 = image_4[np.newaxis, :, :, :]
                            true_masks_4 = un_val_true_masks[i, :, :, :]
                            true_masks_4 = true_masks_4[np.newaxis, :, :, :]
                            pred_masks_4 = un_val_pred_masks[i, :, :, :]
                            pred_masks_4 = pred_masks_4[np.newaxis, :, :, :]

                            images_list = torch.cat(
                                (image_4, true_masks_4, pred_masks_4), dim=0)
                            writer.add_image("val image:fake mask:pred:" + f"{i + 1}",
                                             torchvision.utils.make_grid(images_list, nrow=3, padding=10,
                                                                         normalize=True),
                                             global_step=val_batch_num_all_epochs)

            val_epoch_avg_loss = val_epoch_loss / val_batch_num_each_epoch
            scheduler.step(val_epoch_avg_loss)
            val_loss_all_epochs.append(val_epoch_avg_loss)

            val_epoch_avg_dice = val_epoch_dice / val_batch_num_each_epoch
            dice_similarity_all_epochs.append(val_epoch_avg_dice)

            # scheduler.step()

            print('Epoch:{} finished ! \tval_Loss: {:.4f}, \tlabel_dice_val:{:.4f}'.format(
                epoch + 1, val_loss_all_epochs[-1], dice_similarity_all_epochs[-1]))

        train_process = pd.DataFrame(
            data={"epoch": range(1, epochs + 1), "train_loss_all_epochs": train_loss_all_epochs,
                  "val_loss_all_epochs": val_loss_all_epochs,
                  "dice_similarity_all_epochs": dice_similarity_all_epochs})
        train_process.to_csv(fold_check_path + '/data.csv')

        # TODO: how to add_graph? Here is something wrong with it.
        # torch.jit._trace.TracingCheckError: Tracing failed sanity checks!
        # ERROR: Graphs differed across invocations!

        # fake_image = torch.randn(1, 1, 160, 500).cuda()
        # writer.add_graph(net, fake_image)
        writer.close()


if __name__ == '__main__':
    seed = 10
    set_random_seed(seed)
    args = parse_args()
    device = torch.device('cuda:0')
    kfolds = 5
    epochs = 60
    batch_size1 = 8
    batch_size2 = 4

    # # TODO：核验分割函数
    # neg_img_list = glob.glob(args.image_ICAL_semi + "/neg_*")  # 负样本：图片，文件名列表
    # pos_img_list = glob.glob(args.image_ICAL_semi + "/posLabel_*")  # 正样本：图片
    # un_list = glob.glob(args.image_ICAL_semi + "/posUnLabel_*")  # 正样本：图片
    # print(len(neg_img_list))
    # print(len(pos_img_list))
    # print(len(un_list))
    #
    # neg_mask_list = glob.glob(args.mask_ICAL_semi + "/neg_*")  # 负样本：图片，文件名列表
    # pos_mask_list = glob.glob(args.mask_ICAL_semi + "/posLabel_*")  # 正样本：图片
    # un_list = glob.glob(args.mask_ICAL_semi + "/posUnLabel_*")  # 正样本：图片
    # print(len(neg_mask_list))
    # print(len(pos_mask_list))
    # print(len(un_list))

    split_label_data = split_Label_test(args.image_ICAL_semi, args.mask_ICAL_semi)
    split_UnLabel_data = split_Unlabel_test(args.image_ICAL_semi, args.mask_ICAL_semi)

    # TODO；创造出test文件夹

    # print(len(split_label_data[0]))
    # print(len(split_label_data[1]))
    # print(len(split_label_data[2]))
    # print(len(split_label_data[3]))
    # print(len(split_label_data[4]))
    # print(len(split_label_data[5]))
    # print(len(split_label_data[6]))
    # print(len(split_label_data[7]))
    #
    # print(split_label_data[0][10])
    # print(split_label_data[1][10])
    # print(split_label_data[2][10])
    # print(split_label_data[3][10])
    #
    # print('-----')
    #
    # print(len(split_UnLabel_data[0]))
    # print(len(split_UnLabel_data[1]))
    # print(len(split_UnLabel_data[2]))
    # print(len(split_UnLabel_data[3]))
    #
    #
    # print(split_UnLabel_data[0][20])
    # print(split_UnLabel_data[1][20])
    # print(split_UnLabel_data[2][20])
    # print(split_UnLabel_data[3][20])

    # TODO: Should Synchronous change custom_dataset.py !
    # ICAL: data_aug(angle/shear:10) + Udensenet : testing
    for lr in [0.001, 0.0001]:
        print("##" * 20, "lr=", str(lr), "##" * 20)

        check_path = args.check_ICAL_semi + '/Udensenet_1/' + str(lr)
        train_net(device=device, check_path=check_path,
                  split_data=split_label_data,
                  split_unLable_data=split_UnLabel_data,
                  lr=lr, kfolds=kfolds, epochs=epochs, batch_size1=batch_size1, batch_size2=batch_size2)