# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/06/13 16:50
# @Author  : Yi
# @FileName: train_supervised_internal.py

# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/05/17 21:38
# @Author  : Yi
# @FileName: train_internal_semi_submit.py


import shutil
import sys
import random
import glob
import scipy.spatial
import torchvision
import pandas as pd
from tensorboardX import SummaryWriter
import numpy as np
import math
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import sys
sys.path.append('/data/yilinzhi/Segmentation/VMS/')
from models.udensenet import udensenet
from models.UNet import UNet
from models.udensedown import udensedown
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
        # reduction：mean, sum, none
        # CE = -log(pt), FL= alpha*（1-pt)^(r)*(-log(pt))
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

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
    """验证集评价指标：计算dice值"""

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


def split_val(data_path, img_path, mask_path, folds, k, batch_size1, batch_size2):
    """按照病例划分：训练集+验证集，20个病例。16个：训练，4个验证。
        k: 0-folds-1

        训练集：有标签的dataLoader,没有标签的dataLoader
        验证集：有标签的dataLoader,没有标签的dataLoader
    """
    names = os.listdir(data_path) # 获取所有训练案例名字

    file_names = []
    for name in names:
        split_name = name.split('_')[1]
        if split_name in ["P206", "P432", "P576", "P891"]:  # 排除测试集：internal 部分
            continue
        file_names.append(split_name)

    # file_names.sort()
    nums = len(file_names)
    fold_size = nums // folds
    # print("file_names:", file_names)

    val_k = []  # 第k折的验证集病例，随机打乱
    for i in range(fold_size):
        val_k.append(file_names[k + folds * i])
    # val_k.sort()
    # print("val_k",val_k)

    train_k = list(set(file_names).difference(val_k))  # 第k折的训练集病例
    # train_k.sort()
    # print("train_k",train_k)

    train_img_pos_label = []  # 训练集：正样本，有标签
    train_mask_pos_label = []
    for case in train_k:
        case_name = str(case)
        img = glob.glob(img_path + "/" + case_name + "*_posLabel_.npy")
        train_img_pos_label.extend(img)
        mask = glob.glob(mask_path + "/" + case_name + "*_posLabel_.npy")
        train_mask_pos_label.extend(mask)

    train_img_neg_label = []  # 训练集：负样本，有标签
    train_mask_neg_label = []
    for case in train_k:
        case_name = str(case)
        img = glob.glob(img_path + "/" + case_name + "*_negLabel_.npy")
        train_img_neg_label.extend(img)
        mask = glob.glob(mask_path + "/" + case_name + "*_negLabel_.npy")
        train_mask_neg_label.extend(mask)

    train_negLabel_nums = len(train_img_neg_label)
    train_posLabel_nums = len(train_img_pos_label)

    train_img_neg_label.extend(train_img_pos_label)
    train_mask_neg_label.extend(train_mask_pos_label)

    # 有标签的训练集
    train_datasets = MyDataset_train(train_img_neg_label, train_mask_neg_label)

    class_count = torch.tensor([train_negLabel_nums, train_posLabel_nums])
    target = torch.cat(
        (torch.zeros(class_count[0], dtype=torch.long), torch.ones(class_count[1], dtype=torch.long)))
    class_sample_count = torch.tensor(
        [(target == t).sum() for t in torch.unique(target, sorted=True)])
    weight = 1.0 / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in target])

    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    train_loader_label = DataLoader(train_datasets, batch_size=batch_size1, sampler=sampler, num_workers=8,
                                    pin_memory=True)

    train_img_pos_unLabel = []  # 训练集：正样本，无标签
    train_mask_pos_unLabel = []
    for case in train_k:
        case_name = str(case)
        img = glob.glob(img_path + "/" + case_name + "*_posUnLabel_.npy")
        train_img_pos_unLabel.extend(img)
        mask = glob.glob(mask_path + "/" + case_name + "*_posUnLabel_.npy")
        train_mask_pos_unLabel.extend(mask)

    # 无标签的训练集
    train_datasets = MyDataset_train_un(train_img_pos_unLabel, train_mask_pos_unLabel)
    train_loader_unLabel = DataLoader(train_datasets, batch_size=batch_size2, shuffle=True, num_workers=8,
                                      pin_memory=True)

    val_img_pos_label = []  # 验证集：正样本，有标签
    val_mask_pos_label = []
    for case in val_k:
        case_name = str(case)
        img = glob.glob(img_path + "/" + case_name + "*_posLabel_.npy")
        val_img_pos_label.extend(img)
        mask = glob.glob(mask_path + "/" + case_name + "*_posLabel_.npy")
        val_mask_pos_label.extend(mask)

    val_img_neg_label = []  # 验证集：负样本，有标签
    val_mask_neg_label = []
    for case in val_k:
        case_name = str(case)
        img = glob.glob(img_path + "/" + case_name + "*_negLabel_.npy")
        val_img_neg_label.extend(img)
        mask = glob.glob(mask_path + "/" + case_name + "*_negLabel_.npy")
        val_mask_neg_label.extend(mask)

    val_negLabel_nums = len(val_img_neg_label)
    val_posLabel_nums = len(val_img_pos_label)

    val_img_neg_label.extend(val_img_pos_label)
    val_mask_neg_label.extend(val_mask_pos_label)

    # 有标签的验证集
    val_datasets = MyDataset_val(val_img_neg_label, val_mask_neg_label)

    val_class_count = torch.tensor([val_negLabel_nums, val_posLabel_nums])
    val_target = torch.cat(
        (torch.zeros(val_class_count[0], dtype=torch.long), torch.ones(val_class_count[1], dtype=torch.long)))
    val_class_sample_count = torch.tensor(
        [(val_target == t).sum() for t in torch.unique(val_target, sorted=True)])
    val_weight = 1.0 / val_class_sample_count.float()
    val_samples_weight = torch.tensor([val_weight[t] for t in val_target])

    val_sampler = WeightedRandomSampler(val_samples_weight, len(val_samples_weight))
    val_loader_label = DataLoader(val_datasets, batch_size=batch_size1, sampler=val_sampler, num_workers=8,
                                  pin_memory=True)

    val_img_pos_unLabel = []  # 验证集：正样本，无标签
    val_mask_pos_unLabel = []
    for case in val_k:
        case_name = str(case)
        img = glob.glob(img_path + "/" + case_name + "*_posUnLabel_.npy")
        val_img_pos_unLabel.extend(img)
        mask = glob.glob(mask_path + "/" + case_name + "*_posUnLabel_.npy")
        val_mask_pos_unLabel.extend(mask)

    val_datasets = MyDataset_val_un(val_img_pos_unLabel, val_mask_pos_unLabel)
    val_loader_unLabel = DataLoader(val_datasets, batch_size=batch_size2, shuffle=False, num_workers=8,
                                    pin_memory=True)

    return train_loader_label, train_loader_unLabel, val_loader_label, val_loader_unLabel


def train_net(
        device,
        check_path,
        data_path,
        img_path,
        mask_path,
        lr,
        folds,
        epochs,
        batch_size1,
        batch_size2):
    """训练函数：交叉训练"""

    # TODO：修改保存文件
    create_dir(check_path)  # 创造保存文件夹
    shutil.copyfile("train_supervised_internal.py", check_path + '/train_supervised_internal.py')  # 保存代码，即保存相关参数。
    shutil.copyfile("../../utils/datasets/dataset_semi_submit.py", check_path + '/dataset_semi_submit.py')

    for fold in range(folds):
        print("=" * 20, "kfold: ", fold + 1, "=" * 20)

        # each fold should initialize Unet/Udensenet model
        # binary segmentation：sigmoid()->prob
        # multi segmentation: softmax()->prob
        # multi_GPU training

        # TODO: 3个不同的模型：如何集成？
        # net = UNet(1, 1)
        # net = udensedown(1,1)
        net = udensenet(1, 1)

        net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device=device)

        # TODO: 超参数调节！
        if fold > 1:  # hyper parameters adjusting
            break

        # dice loss for binary segmentation
        # BCE loss -> focal loss
        criterion_dice = dice_ratio_loss()
        dice_rate = 1
        criterion_bce = nn.BCELoss()
        bce_rate = 1

        optimizer = optim.Adam(net.parameters(), lr=lr)

        # 采用不同的学习率调整策略：规律，不规律，按照指标，采用第三个学习率衰减方法。
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 60], gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1,threshold=0.02,
        #                                                        patience=5, verbose=True)

        fold_check_path = check_path + '/' + str(fold)  #每折保存训练模型+结果
        create_dir(fold_check_path)

        # TODO: 记录terminal输出文件：模型_位置_log.txt
        sys.stdout = Logger(fold_check_path + '/Udensenet_IL_log.txt')
        writer = SummaryWriter(logdir=fold_check_path + '/run', comment=f'LR_{lr}')

        train_loss_all_epochs = []  # 记录所有轮次，训练集的loss
        val_loss_all_epochs = []  # 记录所有轮次，验证集的loss
        dice_similarity_all_epochs = []  # 记录所有轮次，验证集的loss

        batch_num_all_epochs = 0  # 训练集，所有epochs里累计batch数目
        val_batch_num_all_epochs = 0  # 验证集，所有epochs里累计batch数目

        # 训练集+验证集：有标签+无标签
        train_loader_label, train_loader_un, val_loader_label, val_loader_un = split_val(data_path, img_path, mask_path,
                                                                                         folds, fold, batch_size1,
                                                                                         batch_size2)

        for epoch in range(epochs):  # 每折，训练总的轮次，每个轮次都会训练所有的样本

            train_epoch_loss = 0
            train_epoch_bce = 0
            train_epoch_dice = 0

            val_epoch_loss = 0
            val_epoch_dice = 0

            # training part:
            net.train()
            batch_num_each_epoch = 0  # 每折中batch数目
            for i_batch, (data, un_data) in enumerate(zip(cycle(train_loader_label), train_loader_un)):
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

                loss_dice_un = criterion_dice(un_pred_masks, un_true_masks)  # 计算有标签训练集:loss
                loss_bce_un = criterion_bce(un_pred_masks, un_true_masks)
                loss_un = loss_dice_un * dice_rate + loss_bce_un * bce_rate

                # TODO：转成数组相乘方式！
                # true_index = list(map(int, true_idx))
                # fake_index=list(map(int,fake_idx))
                # coefficient = list(map(lambda x: math.exp(-(x[0] - x[1])**2), zip(true_index, fake_index)))

                # TODO：尝试两种for方式！
                # loss_un = 0.0
                # # loss_un_list=[]
                # for idx in range(len(true_idx)):  # 迭代每个无标签样本，大小是batch_size，但是有时不能整除！
                #     j = int(true_idx[idx])
                #     i = int(fake_idx[idx])
                #
                #     un_loss_dice = criterion_dice(un_pred_masks[idx], un_true_masks[idx])
                #     un_loss_bce = criterion_bce(un_pred_masks[idx], un_true_masks[idx])
                #     # TODO：修改系数，exp(-i)
                #     loss_un = loss_un + (1.0 / ((i - j) ** 2 + 0.01)) * (
                #                 dice_rate * un_loss_dice + bce_rate * un_loss_bce)

                    # loss_un =  (1.0 / ((i - j) ** 2 + 0.01)) * (
                    #         dice_rate * un_loss_dice + bce_rate * un_loss_bce)
                    # loss_un_list.append(loss_un)

                # loss_un = loss_un/len(true_idx)

                loss =  loss_un+loss_label  # 总loss

                optimizer.zero_grad()  # 优化器更新
                loss.backward()  # loss反向传播
                nn.utils.clip_grad_value_(net.parameters(), 0.1)  # 防止梯度爆炸，进行梯度裁剪
                optimizer.step()

                train_epoch_loss += loss.item()  # 每个epoch，累计计算一个batch的loss
                train_epoch_bce += loss_bce.item()  # 训练集：有标签的bce loss
                train_epoch_dice += loss_dice.item()  # 训练集：有标签的dice loss

                batch_num_each_epoch += 1  # 每个epoch：batch_num
                batch_num_all_epochs += 1  # 所有epochs

                if batch_num_all_epochs % 100 == 0:
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
                        diff_1 = torch.abs(pred_masks_1 - true_masks_1)

                        # original image, truth mask and pred mask
                        images_list = torch.cat(
                            (image_1, true_masks_1, pred_masks_1, diff_1), dim=0)
                        writer.add_image("train image : true mask : pred mask : difference" + f"{i + 1}",
                                         torchvision.utils.make_grid(images_list, nrow=4, padding=10, normalize=True),
                                         global_step=batch_num_all_epochs)

                    for i in range(un_pred_masks.shape[0]):  # 无标签
                        image_2 = un_images[i, :, :, :]
                        image_2 = image_2[np.newaxis, :, :, :]
                        true_masks_2 = un_true_masks[i, :, :, :]
                        true_masks_2 = true_masks_2[np.newaxis, :, :, :]
                        pred_masks_2 = un_pred_masks[i, :, :, :]
                        pred_masks_2 = pred_masks_2[np.newaxis, :, :, :]
                        diff_2 = torch.abs(pred_masks_2 - true_masks_2)

                        # original image, truth mask and pred mask
                        images_list = torch.cat(
                            (image_2, true_masks_2, pred_masks_2, diff_2), dim=0)
                        writer.add_image("unLabel train image : fake mask : pred mask : difference" + f"{i + 1}",
                                         torchvision.utils.make_grid(images_list, nrow=4, padding=10, normalize=True),
                                         global_step=batch_num_all_epochs)

            # averaged training loss for each epoch
            train_epoch_avg_loss = train_epoch_loss / batch_num_each_epoch
            train_loss_all_epochs.append(train_epoch_avg_loss)
            print(
                'Epoch:{} finished ! \ttrain_Loss: {:.4f},\tlabel_dice_loss:{:.4f},\tlabel_bce_loss:{:.4f}'.format(
                    epoch + 1, train_loss_all_epochs[-1], float(train_epoch_dice) / batch_num_each_epoch,
                    float(train_epoch_bce) / batch_num_each_epoch))

            # TODO: 模型保存！
            if (epoch + 1) % 20 == 0:
                torch.save(net.state_dict(), fold_check_path + '/Udensenet_' + str(epoch + 1) + '.pkl')

            net.eval()  # 进入验证阶段
            val_batch_num_each_epoch = 0
            for val_i_batch, (val_data, un_val_data) in enumerate(zip(cycle(val_loader_label), val_loader_un)):
                with torch.no_grad():
                    val_images = val_data['image']  # 验证集：有标签
                    val_true_masks = val_data['mask']

                    un_val_images = un_val_data['image']  # 验证集：无标签
                    un_val_true_masks = un_val_data['mask']
                    val_true_idx = un_val_data['true_index']
                    val_fake_idx = un_val_data['fake_index']

                    val_images = val_images.to(device=device, dtype=torch.float32)
                    val_true_masks = val_true_masks.to(device=device, dtype=torch.float32)
                    un_val_images = un_val_images.to(device=device, dtype=torch.float32)
                    un_val_true_masks = un_val_true_masks.to(device=device, dtype=torch.float32)

                    val_pred_masks = net(val_images)
                    un_val_pred_masks = net(un_val_images)

                    val_loss_dice = criterion_dice(val_pred_masks, val_true_masks)
                    val_loss_bce = criterion_bce(val_pred_masks, val_true_masks)

                    val_loss_label = val_loss_dice * dice_rate + val_loss_bce * bce_rate  # 验证集：有标签

                    val_loss_dice_un = criterion_dice(un_val_pred_masks, un_val_true_masks)
                    val_loss_bce_un = criterion_bce(un_val_pred_masks, un_val_true_masks)

                    val_loss_un = val_loss_dice_un * dice_rate + val_loss_bce_un * bce_rate  # 验证集：有标签

                    # val_loss_un = 0.0
                    # for idx in range(len(val_true_idx)):
                    #     i = int(val_true_idx[idx])
                    #     j = int(val_fake_idx[idx])
                    #
                    #     un_val_loss_dice = criterion_dice(un_val_pred_masks[idx], un_val_true_masks[idx])
                    #     un_val_loss_bce = criterion_bce(un_val_pred_masks[idx], un_val_true_masks[idx])
                    #     val_loss_un = val_loss_un + (1.0 / ((i - j) ** 2 + 0.01)) * (
                    #             dice_rate * un_val_loss_dice + bce_rate * un_val_loss_bce)
                    #
                    # val_loss_un = val_loss_un / len(val_true_idx)

                    val_loss = val_loss_un + val_loss_label

                    val_epoch_loss += val_loss.item()
                    val_batch_num_all_epochs += 1
                    val_batch_num_each_epoch += 1

                    # 验证集的评价指标：有标签数据的dice值
                    dice_similarity = getDSC(val_true_masks.cpu().numpy(), val_pred_masks.cpu().numpy())
                    val_epoch_dice += dice_similarity

                    if val_batch_num_all_epochs % 100 == 0:
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
                            diff_3 = torch.abs(pred_masks_3 - true_masks_3)

                            images_list = torch.cat(
                                (image_3, true_masks_3, pred_masks_3, diff_3), dim=0)
                            writer.add_image("val image : true mask : pred mask: difference" + f"{i + 1}",
                                             torchvision.utils.make_grid(images_list, nrow=4, padding=10,
                                                                         normalize=True),
                                             global_step=val_batch_num_all_epochs)

                        for i in range(un_val_pred_masks.shape[0]):  # 无标签验证集
                            image_4 = un_val_images[i, :, :, :]
                            image_4 = image_4[np.newaxis, :, :, :]
                            true_masks_4 = un_val_true_masks[i, :, :, :]
                            true_masks_4 = true_masks_4[np.newaxis, :, :, :]
                            pred_masks_4 = un_val_pred_masks[i, :, :, :]
                            pred_masks_4 = pred_masks_4[np.newaxis, :, :, :]
                            diff_4 = torch.abs(pred_masks_4 - true_masks_4)

                            images_list = torch.cat(
                                (image_4, true_masks_4, pred_masks_4, diff_4), dim=0)
                            writer.add_image("unLabel val image : fake mask: pred mask: difference" + f"{i + 1}",
                                             torchvision.utils.make_grid(images_list, nrow=4, padding=10,
                                                                         normalize=True),
                                             global_step=val_batch_num_all_epochs)

            val_epoch_avg_loss = val_epoch_loss / val_batch_num_each_epoch
            # scheduler.step(val_epoch_avg_loss)
            val_loss_all_epochs.append(val_epoch_avg_loss)

            val_epoch_avg_dice = val_epoch_dice / val_batch_num_each_epoch
            dice_similarity_all_epochs.append(val_epoch_avg_dice)

            scheduler.step()

            print('Epoch:{} finished ! \tval_Loss: {:.4f}, \tlabel_dice_val:{:.4f}'.format(
                epoch + 1, val_loss_all_epochs[-1], dice_similarity_all_epochs[-1]))

        train_process = pd.DataFrame(
            data={"epoch": range(1, epochs + 1), "train_loss_all_epochs": train_loss_all_epochs,
                  "val_loss_all_epochs": val_loss_all_epochs,
                  "dice_similarity_all_epochs": dice_similarity_all_epochs})
        train_process.to_csv(fold_check_path + '/data.csv')

        writer.close()


if __name__ == '__main__':
    seed = 10
    set_random_seed(seed)
    args = parse_args()
    device = torch.device('cuda:0')
    folds = 5
    epochs = 80
    batch_size1 = 8
    batch_size2 = 4

    # data_path = args.datasets_path
    # img_path = args.image_ICAL_semi
    # mask_path = args.mask_ICAL_semi
    # split_val(data_path, img_path, mask_path, folds, 0, batch_size1, batch_size2)
    # print()
    # split_val(data_path, img_path, mask_path, folds, 1, batch_size1, batch_size2)
    # print()
    # split_val(data_path, img_path, mask_path, folds, 2, batch_size1, batch_size2)


    # ICAL: work station 2
    for lr in [0.001,0.0001]:
        print("##" * 20, "lr=", str(lr), "##" * 20)

        check_path = args.check_ICAL_semi + '/Super_Ud_1/' + str(lr)
        data_path = args.datasets_path
        img_path = args.image_ICAL_semi
        mask_path = args.mask_ICAL_semi
        print(check_path)
        train_net(device=device, check_path=check_path, data_path=data_path, img_path=img_path, mask_path=mask_path,
                  lr=lr, folds=folds, epochs=epochs, batch_size1=batch_size1, batch_size2=batch_size2)

    # ICAR: work station 2
    for lr in [0.001,0.0001]:
        print("##" * 20, "lr=", str(lr), "##" * 20)

        check_path = args.check_ICAR_semi + '/Super_Ud_1/' + str(lr)
        data_path = args.datasets_path
        img_path = args.image_ICAR_semi
        mask_path = args.mask_ICAR_semi
        print(check_path)
        train_net(device=device, check_path=check_path, data_path=data_path, img_path=img_path, mask_path=mask_path,
                  lr=lr, folds=folds, epochs=epochs, batch_size1=batch_size1, batch_size2=batch_size2)
