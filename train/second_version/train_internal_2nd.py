# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/05/09 15:45
# @Author  : Yi
# @FileName: train_internal_2nd.py


"""训练不定比例，确定标签的正负样本：

    TODO:



"""

import shutil
import sys
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
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

sys.path.append('E:/Win10_data/Segmentation/VMS_Unet/models')
from models.udensenet import udensenet

from utils.datasets.dataset_2nd import MyDataset_train, MyDataset_validation
from options.arguments_2nd import parse_args


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
        self.alpha = alpha  # TODO：正负样本比例
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

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
    """Compute the Dice Similarity Coefficient."""

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


# TODO：检查划分是否正确
def split_test(img_path, mask_path):
    """划分出：训练集，验证集，测试集：6：2：2

        样本集都是先负后正：找出正负样本，按比例划分。

    :param img_path: 图像路径
    :param mask_path: mask路径
    :return:
    """

    neg_img_list = glob.glob(img_path + "/negative_*")  # 负样本图片，文件名list
    pos_img_list = glob.glob(img_path + "/positive_*")  # 正样本图片，文件名list

    neg_mask_list = glob.glob(mask_path + "/negative_*")
    pos_mask_list = glob.glob(mask_path + "/positive_*")

    total_nums = len(os.listdir(img_path))  # 图像/mask：正负样本总的数目
    neg_nums = len(neg_mask_list)  # 正样本数目
    pos_nums = len(pos_mask_list)  # 负样本数目

    k = 5
    neg_batch = neg_nums // k  # 4份：训练集+验证集，1份：测试集
    pos_batch = pos_nums // k

    # neg_left_nums = neg_nums - neg_batch  # 训练+验证：负样本数目
    # pos_left_nums = pos_nums - pos_batch  # 训练+验证：正样本数目

    neg_start = 0
    neg_end = neg_batch
    pos_start = 0
    pos_end = pos_batch

    neg_shuffle_indices = np.random.permutation(neg_nums)  # 负样本排列
    pos_shuffle_indices = np.random.permutation(pos_nums)  # 正样本排列

    pos_img_shuffle = []  # 正负样本的顺序打乱
    neg_img_shuffle = []
    pos_mask_shuffle = []
    neg_mask_shuffle = []

    for index in neg_shuffle_indices:
        neg_img_shuffle.append(neg_img_list[index])
        neg_mask_shuffle.append(neg_mask_list[index])

    for index in pos_shuffle_indices:
        pos_img_shuffle.append(pos_img_list[index])
        pos_mask_shuffle.append(pos_mask_list[index])

    split_data = []

    neg_img_test = neg_img_shuffle[neg_start:neg_end]  # 负样本，图片，测试集
    neg_mask_test = neg_mask_shuffle[neg_start:neg_end]  # 负样本，mask，测试集

    del neg_img_shuffle[neg_start:neg_end]  # 负样本，图片，训练+验证集
    del neg_mask_shuffle[neg_start:neg_end]  # 负样本，mask，训练+验证集

    pos_img_test = pos_img_shuffle[pos_start:pos_end]  # 正样本，图片，测试集
    pos_mask_test = pos_mask_shuffle[pos_start:pos_end]  # 正样本，mask，测试集

    del pos_img_shuffle[pos_start:pos_end]  # 正样本，图片，训练+验证集
    del pos_mask_shuffle[pos_start:pos_end]  # 正样本，mask，训练+验证集

    # neg_img_test.extend(pos_img_test)  # 图片测试集：负+正样本
    # neg_mask_test.extend(pos_mask_test)  # mask测试集：负+正样本
    # neg_img_shuffle.extend(pos_img_shuffle)  # 图片：训练+验证集
    # neg_mask_shuffle.extend(pos_mask_shuffle)  # mask：训练+验证集

    split_data.append(neg_img_shuffle)  # 主要此处neg，忽略：图片训验，mask训验，图片测试，mask测试
    split_data.append(pos_img_shuffle)
    split_data.append(neg_mask_shuffle)
    split_data.append(pos_mask_shuffle)
    split_data.append(neg_img_test)
    split_data.append(pos_img_test)
    split_data.append(neg_mask_test)
    split_data.append(pos_mask_test)

    return split_data


def train_net(
        device,
        check_path,
        split_data,
        args,
        lr,
        kfolds,
        epochs,
        batch_size):
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
    create_dir(check_path)
    shutil.copyfile("../first_version/train_internal_1st.py", check_path + '/train_internal_1st.py')  # 保存代码，即保存相关参数。
    shutil.copyfile("../../utils/datasets/dataset_1st.py", check_path + '/dataset_1st.py')

    for fold in range(kfolds):
        print("=" * 25, "kfold: ", fold + 1, "=" * 25)

        # each fold should initialize Unet/Udensenet model
        # binary segmentation：sigmoid()->prob
        # multi segmentation: softmax()->prob
        # multi_GPU training

        # TODO: Different model!
        # net = UNet(1, 1)
        net = udensenet(1, 1)
        net = nn.DataParallel(net, device_ids=[0, 1])
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

        criterion_focal = FocalLoss()
        focal_rate = 0

        optimizer = optim.Adam(net.parameters(), lr=lr)

        # 采用不同的学习率调整策略
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 60], gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2,
        #                                                        patience=10, verbose=True)

        fold_check_path = check_path + '/' + str(fold)
        create_dir(fold_check_path)

        # Information saving: log.txt and tensorboardX
        # TODO: 模型_位置_数据增强_损失函数_log.txt
        sys.stdout = Logger(fold_check_path + '/Udensenet_il_da1_1dice_1bce_log.txt')
        writer = SummaryWriter(logdir=fold_check_path + '/run', comment=f'LR_{lr}_BS_{batch_size}')

        train_loss_all_epochs = []
        val_loss_all_epochs = []
        dice_similarity_all_epochs = []

        batch_num_all_epochs = 0
        val_batch_num_all_epochs = 0

        # TODO：训练+验证集 => 按照第fold+1折划分，训练集+验证集：负样本+正样本，需要定义权重采样器。
        # step3：定义一个权重采样器：weightedSampler
        # step4：得到dataLoader

        # step1：得到第fold+1折对应的训练集和验证集：image和mask
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

        del neg_img[neg_start:neg_end]
        del pos_img[pos_start:pos_end]
        del neg_mask[neg_start:neg_end]
        del pos_mask[pos_start:pos_end]

        neg_train_nums = len(neg_img)
        pos_train_nums = len(pos_img)
        neg_val_nums = len(neg_img_val)
        pos_val_nums = len(pos_img_val)

        neg_img.extend(pos_img)  # image：训练集
        neg_mask.extend(pos_mask)  # mask：训练集
        neg_img_val.extend(pos_img_val)  # image：验证集
        neg_mask_val.extend(pos_mask_val)  # mask：验证集

        # step2: img/mask得到训练集(x,y)
        train_datasets = MyDataset_train(neg_img, neg_mask, state="train", args=args)

        # step3：定义sampler：需要查看是否正确
        class_count = torch.tensor([neg_train_nums, pos_train_nums])
        target = torch.cat(
            (torch.zeros(class_count[0], dtype=torch.long), torch.ones(class_count[1], dtype=torch.long)))
        class_sample_count = torch.tensor(
            [(target == t).sum() for t in torch.unique(target, sorted=True)])
        weight = 1.0 / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in target])

        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_loader = DataLoader(train_datasets, batch_size=batch_size, sampler=sampler, num_workers=8,
                                  pin_memory=True)

        for epoch in range(epochs):

            train_epoch_loss = 0
            train_epoch_bce = 0
            train_epoch_dice = 0
            train_epoch_focal = 0

            val_epoch_loss = 0
            val_epoch_dice = 0

            # training part:
            net.train()
            batch_num_each_epoch = 0
            for i_batch, data in enumerate(train_loader):

                images = data['image']  # images.shape: [batch_size,1,160,500]
                true_masks = data['mask']

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                pred_masks = net(images)

                loss_dice = criterion_dice(pred_masks, true_masks)
                loss_focal = criterion_focal(pred_masks, true_masks)
                loss_bce = criterion_bce(pred_masks, true_masks)
                loss = loss_dice * dice_rate + loss_focal * focal_rate + loss_bce * bce_rate

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_epoch_loss += loss.item()
                train_epoch_bce += loss_bce.item()
                train_epoch_dice += loss_dice.item()
                train_epoch_focal += loss_focal.item()

                batch_num_each_epoch += 1
                batch_num_all_epochs += 1

                if batch_num_all_epochs % 5 == 0:
                    writer.add_scalar("training loss/batch", loss.item(), global_step=batch_num_all_epochs)
                    writer.add_scalar("training loss_dice/batch", loss_dice.item(), global_step=batch_num_all_epochs)
                    writer.add_scalar("training loss_focal/batch", loss_focal.item(), global_step=batch_num_all_epochs)
                    writer.add_scalar("training loss_bce/batch", loss_bce.item(), global_step=batch_num_all_epochs)

                    for name, para in net.named_parameters():
                        writer.add_histogram(name, para.data.cpu().numpy(), global_step=batch_num_all_epochs)

                    for i in range(pred_masks.shape[0]):
                        image_1 = images[i, :, :, :]
                        image_1 = image_1[np.newaxis, :, :, :]
                        true_masks_1 = true_masks[i, :, :, :]
                        true_masks_1 = true_masks_1[np.newaxis, :, :, :]
                        pred_masks_1 = pred_masks[i, :, :, :]
                        pred_masks_1 = pred_masks_1[np.newaxis, :, :, :]

                        # original image, truth mask and pred mask
                        images_list = torch.cat(
                            (image_1, true_masks_1, pred_masks_1), dim=0)
                        writer.add_image("image_mask_pred:" + f"{i + 1}",
                                         torchvision.utils.make_grid(images_list, nrow=3, padding=10, normalize=True),
                                         global_step=batch_num_all_epochs)

            # averaged training loss for each epoch
            train_epoch_avg_loss = train_epoch_loss / batch_num_each_epoch
            train_loss_all_epochs.append(train_epoch_avg_loss)
            print(
                'Epoch:{} finished ! \ttrain_Loss: {:.4f},\tdice_loss:{:.4f},\tbce_loss:{:.4f},\tfocal_loss:{:.4f}'.format(
                    epoch + 1, train_loss_all_epochs[-1], float(train_epoch_dice) / batch_num_each_epoch,
                    float(train_epoch_bce) / batch_num_each_epoch, float(train_epoch_focal) / batch_num_each_epoch))

            # TODO: modification
            if (epoch + 1) % 15 == 0:
                torch.save(net.state_dict(), fold_check_path + '/Udensenet_' + str(epoch + 1) + '.pkl')

            # 验证阶段:
            val_datasets = MyDataset_validation(neg_img_val, neg_mask_val, state="validation", args=args)

            class_count_val = torch.tensor([neg_val_nums, pos_val_nums])
            target_val = torch.cat(
                (torch.zeros(class_count_val[0], dtype=torch.long), torch.ones(class_count_val[1], dtype=torch.long)))
            class_sample_count_val = torch.tensor(
                [(target_val == t).sum() for t in torch.unique(target_val, sorted=True)])
            weight_val = 1.0 / class_sample_count_val.float()
            samples_weight_val = torch.tensor([weight_val[t] for t in target_val])

            sampler_val = WeightedRandomSampler(samples_weight_val, len(samples_weight_val))
            validation_loader = DataLoader(val_datasets, batch_size=batch_size, sampler=sampler_val, num_workers=8,
                                           pin_memory=True)

            net.eval()
            val_batch_num_each_epoch = 0
            for val_i_batch, val_data in enumerate(validation_loader):
                with torch.no_grad():
                    val_images = val_data['image']
                    val_true_masks = val_data['mask']

                    val_images = val_images.to(device=device, dtype=torch.float32)
                    val_true_masks = val_true_masks.to(device=device, dtype=torch.float32)

                    val_pred_masks = net(val_images)

                    loss_dice = criterion_dice(val_pred_masks, val_true_masks)
                    loss_focal = criterion_focal(val_pred_masks, val_true_masks)
                    loss_bce = criterion_bce(val_pred_masks, val_true_masks)
                    loss = loss_dice * dice_rate + loss_focal * focal_rate + loss_bce * bce_rate

                    val_epoch_loss += loss.item()
                    val_batch_num_all_epochs += 1
                    val_batch_num_each_epoch += 1

                    dice_similarity = getDSC(val_true_masks.cpu().numpy(), val_pred_masks.cpu().numpy())
                    val_epoch_dice += dice_similarity

                    if val_batch_num_all_epochs % 5 == 0:
                        writer.add_scalar("validation loss/batch", loss.item(), global_step=val_batch_num_all_epochs)
                        writer.add_scalar("validation loss_dice/batch", loss_dice.item(),
                                          global_step=val_batch_num_all_epochs)
                        writer.add_scalar("validation loss_focal/batch", loss_focal.item(),
                                          global_step=val_batch_num_all_epochs)
                        writer.add_scalar("validation loss_bce/batch", loss_bce.item(),
                                          global_step=val_batch_num_all_epochs)
                        writer.add_scalar("dice_similarity/batch", dice_similarity,
                                          global_step=val_batch_num_all_epochs)

                        for i in range(val_pred_masks.shape[0]):
                            image_2 = val_images[i, :, :, :]
                            image_2 = image_2[np.newaxis, :, :, :]
                            true_masks_2 = val_true_masks[i, :, :, :]
                            true_masks_2 = true_masks_2[np.newaxis, :, :, :]
                            pred_masks_2 = val_pred_masks[i, :, :, :]
                            pred_masks_2 = pred_masks_2[np.newaxis, :, :, :]

                            images_list = torch.cat(
                                (image_2, true_masks_2, pred_masks_2), dim=0)
                            writer.add_image("val_image_mask_pred:" + f"{i + 1}",
                                             torchvision.utils.make_grid(images_list, nrow=3, padding=10,
                                                                         normalize=True),
                                             global_step=val_batch_num_all_epochs)

            val_epoch_avg_loss = val_epoch_loss / val_batch_num_each_epoch
            # scheduler.step(val_epoch_avg_loss)
            val_loss_all_epochs.append(val_epoch_avg_loss)

            val_epoch_avg_dice = val_epoch_dice / val_batch_num_each_epoch
            dice_similarity_all_epochs.append(val_epoch_avg_dice)

            scheduler.step()

            print('Epoch:{} finished ! \tval_Loss: {:.4f}, \tdice_val:{:.4f}'.format(
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
    """Hyper parameters:

        In the work station, should use multiply GPUs and change batch_size!
        Checkpoints：/position/model+data_augmentation+other_modification/lr/fold
        Saving train_internal_1st.py in each fold.
        Different Model: different batch_size
    """
    seed = 10
    set_random_seed(seed)
    args = parse_args()
    device = torch.device('cuda:0')
    kfolds = 5
    epochs = 80
    batch_size = 16

    split_data = split_test(args.train_image_sep_ICAL, args.train_mask_sep_ICAL)
    # print(len(split_data[0]))
    # print(len(split_data[1]))
    # print(len(split_data[2]))
    # print(len(split_data[3]))
    # print(len(split_data[4]))
    # print(len(split_data[5]))
    # print(len(split_data[6]))
    # print(len(split_data[7]))

    # print(split_data[0][10])
    # print(split_data[1][10])
    # print(split_data[2][10])
    # print(split_data[3][10])

    # TODO: how to find best hyper parameters?
    # TODO: Should Synchronous change dataset_1st.py !

    # ICAL: data_aug(angle/shear:10) + Udensenet : testing
    for lr in [0.1, 0.01, 0.001, 0.0001]:
        print("#" * 30, "lr=", str(lr), "#" * 30)
        image_path = args.train_image_sep_ICAL
        mask_path = args.train_mask_sep_ICAL
        check_path = args.train_check_sep_ICAL + '/Udensenet_DA1/' + str(lr)

        train_net(device=device, check_path=check_path,
                  split_data=split_data,
                  args=args, lr=lr,
                  kfolds=kfolds, epochs=epochs, batch_size=batch_size)
