# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/05/29 09:26
# @Author  : Yi
# @FileName: train_semi_internal.py

import shutil
import sys
import math
import random
import glob
import scipy.spatial
import torchvision
import pandas as pd
from tensorboardX import SummaryWriter
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

sys.path.append('E:/Win10_data/Segmentation/VMS_Unet/models')
from models.UNet import UNet

from utils.datasets.dataset_semi_try import MyDataset_train, MyDataset_val
from options.arguments_semi_try import parse_args


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
    """计算dice loss"""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_pred_f = y_pred.view(-1)  # requires_grad=True
        y_true_f = y_true.view(-1)  # requires_grad=False
        intersection = torch.sum(y_true_f * y_pred_f)
        return 1 - (2.0 * intersection + 1) / (torch.sum(y_true_f) +
                                               torch.sum(y_pred_f) + 1)


class FocalLoss(nn.Module):
    """计算focal loss"""

    def __init__(self, alpha=0.75, gamma=2, logits=False, reduce=True):
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
    """验证集评价指标：计算dice值=1-dice loss"""

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


def split_val(data_path, img_path, mask_path, folds, k):
    """按照病例划分：训练集+验证集，20个病例。16个：训练，4个验证。
        k: 0-folds-1

        return：训练集，验证集：文件名列表
    """
    names = os.listdir(data_path)

    file_names = []
    for name in names:
        split_name = name.split('_')[1]
        if split_name in ["P206", "P432", "P576", "P891"]:  # 排除测试集
            continue
        file_names.append(split_name)

    file_names.sort()
    nums = len(file_names)
    fold_size = nums // folds
    # print(file_names)

    val_k = []  # 第k折的验证集病例
    for i in range(fold_size):
        val_k.append(file_names[k + folds * i])
    val_k.sort()
    # print(val_k)

    train_k = list(set(file_names).difference(val_k))  # 第k折的训练集病例
    train_k.sort()
    # print(train_k)

    # 依据病例找出训练集+验证集：所有的病例image和mask
    train_img = []
    train_mask = []
    for case in train_k:
        case_name = str(case)
        img = glob.glob(img_path + "/" + case_name + "_*")
        train_img.extend(img)
        mask = glob.glob(mask_path + "/" + case_name + "_*")
        train_mask.extend(mask)

    train_img.sort()
    train_mask.sort()

    val_img = []
    val_mask = []
    for case in val_k:
        case_name = str(case)
        img = glob.glob(img_path + "/" + case_name + "_*")
        val_img.extend(img)
        mask = glob.glob(mask_path + "/" + case_name + "_*")
        val_mask.extend(mask)

    val_img.sort()
    val_mask.sort()

    return train_img, train_mask, val_img, val_mask


def train_net(
        device,
        check_path,
        data_path,
        img_path,
        mask_path,
        lr,
        folds,
        epochs,
        batch_size):
    """k折训练函数

    :param device:
    :param check_path:
    :param data_path:
    :param img_path:
    :param mask_path:
    :param lr:
    :param folds:
    :param epochs:
    :param batch_size1

    :return:
    """

    # TODO：修改这里文件！
    create_dir(check_path)  # 创造保存文件夹
    shutil.copyfile("../semi_train_submit/train_internal_semi_submit.py", check_path + '/train_internal_semi_submit.py')  # 保存代码，即保存相关参数。
    shutil.copyfile("../../utils/datasets/dataset_semi_submit.py", check_path + '/dataset_semi_submit.py')

    for fold in range(folds):
        print("=" * 20, "k fold: ", fold, "=" * 20)

        # each fold should initialize Unet/Udensenet/Udensedown model
        # binary segmentation：sigmoid()->prob
        # multi segmentation: softmax()->prob
        # multi_GPU training

        # TODO: 使用不同的网络模型!
        net = UNet(1, 1)
        # net = udensenet(1, 1)
        # net = udensedown(1,1)
        net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device=device)

        # TODO: 测试选择超参数。
        if fold > 1:  # hyper parameters adjusting
            break

        # dice loss for binary segmentation
        # BCE loss -> focal loss
        # TODO：调整不同损失函数比例
        criterion_dice = dice_ratio_loss()
        dice_rate = 0.75

        criterion_bce = nn.BCELoss()
        bce_rate = 0.25

        optimizer = optim.Adam(net.parameters(), lr=lr)

        # 采用不同的学习率调整策略：规律，不规律，按照指标
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 60], gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2,
        #                                                        patience=10, verbose=True)

        fold_check_path = check_path + '/' + str(fold)  # 保存每折训练结果
        create_dir(fold_check_path)

        # TODO: 记录文件：模型_位置_数据增强_损失函数_log.txt
        sys.stdout = Logger(fold_check_path + '/Unet_il_da1_1dice_1bce_log.txt')
        writer = SummaryWriter(logdir=fold_check_path + '/run', comment=f'LR_{lr}')

        train_loss_all_epochs = []  # 记录所有轮次，训练集的loss
        val_loss_all_epochs = []  # 记录所有轮次，验证集的loss
        dice_similarity_all_epochs = []  # 记录所有轮次，验证集的loss

        batch_num_all_epochs = 0  # 训练集，所有epochs里累计batch数目
        val_batch_num_all_epochs = 0  # 验证集，所有epochs里累计batch数目

        # 划分训练集和验证集：第k折划分
        train_img, train_mask, val_img, val_mask = split_val(data_path, img_path, mask_path, folds, fold)

        # img/mask得到训练，验证集(x,y)=>进行顺序采样，关注dataset
        train_datasets = MyDataset_train(train_img, train_mask)
        train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        val_datasets = MyDataset_val(val_img, val_mask)
        val_loader = DataLoader(val_datasets, batch_size=batch_size, shuffle=False, num_workers=8,pin_memory=True)

        for epoch in range(epochs):  # 每折，训练总的轮次，每个轮次都会训练所有的样本
            train_epoch_loss = 0
            train_epoch_bce = 0
            train_epoch_dice = 0

            val_epoch_loss = 0
            val_epoch_dice = 0

            # training part:
            net.train()
            batch_num_each_epoch = 0  # 每轮次epoch中batch_size数目
            for i_batch, data in enumerate(train_loader):

                images = data['image']
                true_masks = data['mask']

                labels = data['label']
                indices = data['index']
                positions = data['pos']
                cases = data['case']

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                pred_masks = net(images)  # batch_size大小的样本预测结果：包含有标签，没有标签

                posLable_idx = []
                posLable_idx_sample = []
                for idx in range(len(images)):
                    if labels[idx] == 'posLabel':  # 找到有标签的index
                        posLable_idx.append(indices[idx])  # 有标签样本：真实索引
                        posLable_idx_sample.append(idx)  # 有标签样本：样本集中索引

                label_nums = 0
                un_nums = 0
                loss_label = torch.tensor(0.0, requires_grad=True)
                loss_un = torch.tensor(0.0, requires_grad=True)

                # 修改debug
                for idx in range(len(images)):  # 对每个样本开始区分
                    if labels[idx] == 'posLabel' or labels[idx] == 'negLabel':  # 计算有标签样本的loss值
                        loss_dice = criterion_dice(pred_masks[idx], true_masks[idx])
                        # loss_bce = criterion_bce(pred_masks[idx], true_masks[idx])
                        loss_label_sample = loss_dice

                        loss_label = loss_label + loss_label_sample
                        label_nums += 1

                    if labels[idx] == 'posUn':  # 计算无标签样本的loss值
                        posUn_index = int(indices[idx])  # 无标签样本的真实索引

                        # TODO：有个问题：每个batch_size里的有标签正样本，无标签正样本都不一样，后面一项的loss计算个数不确定！
                        for idx_1 in range(len(posLable_idx)):
                            pos_index = int(posLable_idx[idx_1])  # 有标签样本真实索引
                            alpha = math.exp(-(pos_index - posUn_index) ** 2)  # 计算系数
                            # alpha = 1.0 / ((pos_index - posUn_index + 0.1) ** 2)  # 计算系数

                            loss_dice_un = criterion_dice(pred_masks[idx_1],
                                                          pred_masks[int(posLable_idx_sample[idx_1])].detach())
                            # loss_bce_un = criterion_bce(pred_masks[idx_1], pred_masks[int(posLable_idx_sample[idx_1])].detach())
                            loss_un_sample = loss_dice_un

                            loss_un = loss_un + loss_un_sample * alpha
                            un_nums += 1

                # print("loss_label,\tloss_un,\tlabel_nums,\tun_nums,\tsize",loss_label.item(),loss_un.item(),label_nums,un_nums,len(images))

                loss = (loss_label + loss_un)
                loss_label.backward(retain_graph=True)

                optimizer.zero_grad()  # 优化器更新
                loss.backward()  # loss反向传播
                nn.utils.clip_grad_value_(net.parameters(), 0.1)  # 防止梯度爆炸，进行梯度裁剪
                optimizer.step()

                train_epoch_loss += loss.item()  # 每个epoch，累计计算一个batch的loss

                batch_num_each_epoch += 1  # 每个epoch：batch_num
                batch_num_all_epochs += 1  # 所有epochs

                if batch_num_all_epochs % 50 == 0:
                    writer.add_scalar("training loss/batch", loss.item(), global_step=batch_num_all_epochs)

                    for name, para in net.named_parameters():
                        writer.add_histogram(name, para.data.cpu().numpy(), global_step=batch_num_all_epochs)

                    # 增加一个插值图像
                    for i in range(pred_masks.shape[0]):  #
                        image_1 = images[i, :, :, :]
                        image_1 = image_1[np.newaxis, :, :, :]
                        true_masks_1 = true_masks[i, :, :, :]
                        true_masks_1 = true_masks_1[np.newaxis, :, :, :]
                        pred_masks_1 = pred_masks[i, :, :, :]
                        pred_masks_1 = pred_masks_1[np.newaxis, :, :, :]

                        different_mask = torch.abs(pred_masks_1 - true_masks_1)

                        # original image, truth mask and pred mask
                        images_list = torch.cat(
                            (image_1, true_masks_1, pred_masks_1,different_mask), dim=0)
                        writer.add_image("train image : mask : pred mask: different" + f"{i + 1}",
                                         torchvision.utils.make_grid(images_list, nrow=4, padding=10, normalize=True),
                                         global_step=batch_num_all_epochs)

            # averaged training loss for each epoch
            train_epoch_avg_loss = train_epoch_loss / batch_num_each_epoch
            train_loss_all_epochs.append(train_epoch_avg_loss)
            print('Epoch:{} finished ! \ttrain_Loss: {:.4f}'.format(epoch + 1, train_loss_all_epochs[-1]))

            # TODO: 保存模型
            if (epoch + 1) % 20 == 0:
                torch.save(net.state_dict(), fold_check_path + '/Unet_' + str(epoch + 1) + '.pkl')

            # 验证阶段:
            net.eval()
            val_batch_num_each_epoch = 0
            for val_i_batch, val_data in enumerate(val_loader):
                with torch.no_grad():
                    val_images = val_data['image']
                    val_true_masks = val_data['mask']

                    val_labels = val_data['label']
                    val_indices = val_data['index']
                    val_positions = val_data['pos']
                    val_cases = val_data['case']

                    val_images = val_images.to(device=device, dtype=torch.float32)
                    val_true_masks = val_true_masks.to(device=device, dtype=torch.float32)
                    val_pred_masks = net(val_images)

                    val_posLabel_idx = []
                    val_posLabel_idx_sample = []
                    for idx in range(len(val_images)):
                        if val_labels[idx] == 'posLabel':  # 找到有标签的index
                            val_posLabel_idx.append(val_indices[idx])  # 有标签：真实索引
                            val_posLabel_idx_sample.append(idx)  # 有标签样本：样本集中索引

                    val_loss_label = torch.tensor(0.0, requires_grad=True)
                    val_loss_un = torch.tensor(0.0, requires_grad=True)
                    for idx in range(len(val_images)):  # 对每个样本开始区分
                        if val_labels[idx] == 'posLabel' or val_labels[idx] == 'negLabel':  # 计算有标签样本的loss值
                            val_loss_dice = criterion_dice(val_pred_masks[idx], val_true_masks[idx])
                            # val_loss_bce = criterion_bce(val_pred_masks[idx], val_true_masks[idx])
                            val_loss_label_sample = val_loss_dice
                            val_loss_label = val_loss_label + val_loss_label_sample

                        if val_labels[idx] == 'posUn':
                            val_posUn_index = int(val_indices[idx])  # 无标签样本的真实索引
                            for idx_1 in range(len(val_posLabel_idx)):
                                val_pos_index = int(val_posLabel_idx[idx_1])  # 有标签样本真实索引
                                alpha = math.exp(-(val_pos_index - val_posUn_index) ** 2)  # 计算系数
                                # alpha = 1.0 / ((val_pos_index - val_posUn_index + 0.1) ** 2)  # 计算系数

                                val_loss_dice_un = criterion_dice(val_pred_masks[idx_1],
                                                                  val_pred_masks[
                                                                      int(val_posLabel_idx_sample[idx_1])].detach())
                                # val_loss_bce_un = criterion_bce(val_pred_masks[idx_1],
                                #                             val_pred_masks[int(val_posLabel_idx_sample[idx_1])].detach())
                                val_loss_un_sample = val_loss_dice_un

                                val_loss_un = val_loss_un + val_loss_un_sample * alpha

                    val_loss = val_loss_label + val_loss_un

                    # 计算损失函数
                    val_epoch_loss += val_loss.item()

                    # TODO:验证集的评价指标：有标签数据的dice值
                    label_cnt = 0
                    dice_similarity = 0.0
                    for i in range(len(val_images)):
                        if val_labels[i] == 'negLabel' or val_labels[i] == 'posLabel':
                            dice_similarity += getDSC(val_true_masks[i].cpu().numpy(), val_pred_masks[i].cpu().numpy())
                            label_cnt += 1

                    if label_cnt == 0:  # 有可能找不到有标签的样本
                        continue
                    dice_similarity = dice_similarity / float(label_cnt)
                    val_epoch_dice += dice_similarity

                    val_batch_num_all_epochs += 1
                    val_batch_num_each_epoch += 1

                    if val_batch_num_all_epochs % 50 == 0:
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
                            writer.add_image("val image : mask : pred mask:" + f"{i + 1}",
                                             torchvision.utils.make_grid(images_list, nrow=3, padding=10,
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
    batch_size = 8

    # data_path='/data/yilinzhi/Segmentation/VMS/datasets/careIIChallenge'
    #
    # # case_name='P125'
    # # print(args.image_ECAL_semi)
    # # # print(len(os.listdir(args.image_ECAL_semi)))
    # # img = glob.glob(args.image_ICAL_semi + "/" + case_name + "_*")
    # # print(img)
    #
    # train_img,train_mask,test_img,test_mask = split_val(data_path,args.image_ICAL_semi, args.mask_ICAL_semi,5,0)
    # print(test_img[:10])
    # print(test_mask[:10])
    # train_img1, train_mask1, test_img1, test_mask1 = split_val(data_path, args.image_ICAL_semi, args.mask_ICAL_semi, 5, 1)
    # train_img2, train_mask2, test_img2, test_mask2 = split_val(data_path, args.image_ICAL_semi, args.mask_ICAL_semi, 5, 2)
    # train_img3, train_mask3, test_img3, test_mask3 = split_val(data_path, args.image_ICAL_semi, args.mask_ICAL_semi, 5, 3)
    # train_img4, train_mask4, test_img4, test_mask4 = split_val(data_path, args.image_ICAL_semi, args.mask_ICAL_semi, 5, 4)

    #  TODO: 同时修改 dataset_1st.py !
    # ICAL: data_aug + Udensenet : testing
    for lr in [0.01, 0.001, 0.0001]:
        print("##" * 20, "lr=", str(lr), "##" * 20)
        check_path = args.check_ICAL_semi + '/semi_F_Unet/' + str(lr)
        train_net(device=device, check_path=check_path, data_path=args.datasets_path, img_path=args.image_ICAL_semi,
                  mask_path=args.mask_ICAL_semi, lr=lr, folds=folds, epochs=epochs, batch_size=batch_size)
