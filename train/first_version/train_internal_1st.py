# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/03/18 19:51
# @Author  : yilinzhi
# @FileName: train_internal_1st.py


"""
TODO(Yi):
    1.需要再次检查，打乱后image/mask是否对应？
    2.数据增强：目前采用，翻转+旋转+缩放+仿射，对多样本/少样本是否有效？
    3.损失函数：focal loss过于小，如何起作用？
    4.处理对象：小尺度对象=>合适的loss?，正负样本是否按比例采样？
    5.分割预测错误问题：有些没有学出来，同时有些学出来，但是学错了
    6.环状mask处理：少样本：有/无数据增强，同理大样本
    7.mask：同上
    8.使用新的网络：Unet -> Undensenet, 如何适当改进加深网络？
    9. 如何调参？
    10.验证集loss < 训练集loss ?
    11.GPU利用率忽高忽低
    12.Tensorlayer的数据增强
"""

import shutil
import sys
import random
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

sys.path.append('E:/Win10_data/Segmentation/VMS_Unet/models')
from models.udensenet import udensenet

from utils.datasets.dataset_1st import MyDataset_train, MyDataset_validation
from options.arguments_1st import parse_args


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


def split_k_train_valdation(k, i, image_path, mask_path, shuffle_indices):
    """Split datasets(image/mask) to training datasets and validation datasets.

        Args:
            k：Total folds.
            i: ith fold.
            image_path/mask_path: data path.
            shuffle_indices: Random sampling.

        Returns:
            split_dataset_file: lists of image/mask(positive,negative) for training/validation
        """

    image_positive_files = os.listdir(image_path + '/positive')
    image_negative_files = os.listdir(image_path + "/negative")

    mask_positive_files = os.listdir(mask_path + "/positive")
    mask_negative_files = os.listdir(mask_path + "/negative")

    # shuffle indices for image/mask files
    image_negative_files_shuffle = []
    image_positive_files_shuffle = []
    mask_negative_files_shuffle = []
    mask_positive_files_shuffle = []

    for index in shuffle_indices:
        image_positive_files_shuffle.append(image_positive_files[index])
        image_negative_files_shuffle.append(image_negative_files[index])
        mask_positive_files_shuffle.append(mask_positive_files[index])
        mask_negative_files_shuffle.append(mask_negative_files[index])

    files_num = len(image_positive_files)
    fold_size = files_num // k

    start = int(i * fold_size)
    end = int((i + 1) * fold_size)

    split_dataset_file = []  # return split files list

    # image files: positive/negative
    image_positive_files_validation = image_positive_files_shuffle[start:end]
    image_negative_files_validation = image_negative_files_shuffle[start:end]

    del image_positive_files_shuffle[start: end]
    del image_negative_files_shuffle[start: end]

    # mask files：positive/negative
    mask_positive_files_validation = mask_positive_files_shuffle[start:end]
    mask_negative_files_validation = mask_negative_files_shuffle[start:end]

    del mask_positive_files_shuffle[start:end]
    del mask_negative_files_shuffle[start:end]

    # save training files
    split_dataset_file.append(image_positive_files_shuffle)
    split_dataset_file.append(image_negative_files_shuffle)
    split_dataset_file.append(mask_positive_files_shuffle)
    split_dataset_file.append(mask_negative_files_shuffle)

    # save validation files
    split_dataset_file.append(image_positive_files_validation)
    split_dataset_file.append(image_negative_files_validation)
    split_dataset_file.append(mask_positive_files_validation)
    split_dataset_file.append(mask_negative_files_validation)

    return split_dataset_file


def train_net(
        device,
        image_path,
        mask_path,
        check_path,
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

    file_nums = len(os.listdir(image_path + '/positive'))
    shuffle_indices = np.random.permutation(file_nums)
    print(len(shuffle_indices))

    create_dir(check_path)
    shutil.copyfile("train_internal_1st.py", check_path + '/train_internal_1st.py')
    shutil.copyfile("../../utils/datasets/dataset_1st.py", check_path + '/dataset_1st.py')

    for fold in range(kfolds):
        print("=" * 15, "kfold: ", fold + 1, "=" * 15)

        # each fold should initialize Unet model
        # binary segmentation：sigmoid()->prob
        # multi segmentation: softmax()->prob
        # multi_GPU training

        # TODO: Different model!
        # net = UNet(1, 1)
        net = udensenet(1, 1)
        net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device=device)

        # TODO: Modification!
        # if fold > 1:  # hyper parameters adjusting
        #     break

        # dice loss for binary segmentation
        # BCE loss -> focal loss
        # TODO: how to use focal loss? Focal loss is too small?
        criterion_dice = dice_ratio_loss()
        dice_rate = 1

        criterion_bce = nn.BCELoss()
        bce_rate = 1
        criterion_focal = FocalLoss()
        focal_rate = 1

        optimizer = optim.Adam(net.parameters(), lr=lr)

        # learning rate adjusting
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 60], gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2,
        #                                                        patience=10, verbose=True)

        fold_check_path = check_path + '/' + str(fold)
        create_dir(fold_check_path)

        # Information saving: log.txt and tensorboardX
        # TODO: for each training, rename log.txt: MODEL_POS_DA_VERSION_log.txt!
        sys.stdout = Logger(fold_check_path + '/Udensenet_il_nda_log.txt')
        writer = SummaryWriter(logdir=fold_check_path + '/run', comment=f'LR_{lr}_BS_{batch_size}')

        train_loss_all_epochs = []
        val_loss_all_epochs = []
        dice_similarity_all_epochs = []

        batch_num_all_epochs = 0
        val_batch_num_all_epochs = 0

        # lists of training/validation files
        split_k_datasets_files = split_k_train_valdation(kfolds, fold, image_path, mask_path, shuffle_indices)

        train_datasets = MyDataset_train(split_k_datasets_files, state="train", args=args)
        train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

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

                    # two channels masks
                    # for i in range(batch_size):
                    #     image_1 = images[i, :, :, :]
                    #     image_1 = image_1[np.newaxis, :, :, :]
                    #     true_masks_1_o = true_masks[i, 0, :, :]
                    #     true_masks_1_o = true_masks_1_o[np.newaxis, np.newaxis, :, :]
                    #     true_mask_1_i = true_masks[i, 1, :, :]
                    #     true_mask_1_i = true_mask_1_i[np.newaxis, np.newaxis, :, :]
                    #     pred_masks_1_o = pred_masks[i, 0, :, :]
                    #     pred_masks_1_o = pred_masks_1_o[np.newaxis, np.newaxis, :, :]
                    #     pred_masks_1_i = pred_masks[i, 1, :, :]
                    #     pred_masks_1_i = pred_masks_1_i[np.newaxis, np.newaxis, :, :]
                    #
                    #     # 原始图像，mask，预测图像拼接起来
                    #     images_list = torch.cat(
                    #         (image_1, true_masks_1_o, pred_masks_1_o, image_1, true_mask_1_i, pred_masks_1_i), dim=0)
                    #     writer.add_image("image_mask_pred:" + f"{i}",
                    #                      torchvision.utils.make_grid(images_list, nrow=3, padding=50, normalize=True),
                    #                      global_step=batch_num_all_epochs)

                    #  1 channel masks
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

            # validation part:
            validation_datasets = MyDataset_validation(split_k_datasets_files, state="validation", args=args)
            validation_loader = DataLoader(validation_datasets, batch_size=batch_size, shuffle=False, num_workers=8,
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
    epochs = 100
    batch_size = 16

    # TODO: how to find best hyper parameters?
    # TODO: Should Synchronous change dataset_1st.py !

    # ICAL: data_aug(angle/shear:10) + Unet :lr=0.001, dice=0.58-0.62
    # ICAL: no_data_aug + Unet：0.001: lr=0.001, dice=0.58-0.63
    # ICAL: data_aug(angle/shear:10) + Unet :lr=0.001, dice=0.63, lr=0.0001, dice=0.66
    # Conclusion: There is no useful of data augmentation. How to modify?
    for lr in [0.01,0.001,0.0001]:
        print("#" * 20, "lr=", str(lr), "#" * 20)
        image_path = args.train_image_sep_ICAL
        mask_path = args.train_mask_sep_ICAL
        # check_path = args.train_check_sep_ICAL + '/Udensenet_DA_V1/' + str(lr)
        check_path = args.train_check_sep_ICAL + '/Udensenet_NDA/' + str(lr)

        train_net(device=device, image_path=image_path, mask_path=mask_path, check_path=check_path, args=args, lr=lr,
                  kfolds=kfolds, epochs=epochs, batch_size=batch_size)

    # ICAR: data_aug(angle/shear:10) + Unet: lr=0.001, dice = 0.60-0.69
    # ICAR：no_data_aug + Unet: lr=0.001, dice = 0.60-0.67
    # ICAR: data_aug(angle/shear:10) +Udensenet: lr=0.0001,dice=0.64-0.70
    # ICAR：no_data_aug + Udensenet: lr=0.01/0.001, dice=0.62-0.69
    # for lr in [0.01,0.001,0.0001]:
    #     print("#" * 20, "lr=", str(lr), "#" * 20)
    #     image_path = args.train_image_sep_ICAR
    #     mask_path = args.train_mask_sep_ICAR
    #     check_path = args.train_check_sep_ICAR + '/Udensenet_DA_V1/' + str(lr)
    #     # check_path = args.train_check_sep_ICAR + '/Udensenet_NDA/' + str(lr)
    #
    #     train_net(device=device, image_path=image_path, mask_path=mask_path, check_path=check_path, args=args,
    #               lr=lr,kfolds=kfolds,epochs=epochs,batch_size=batch_size)


    # ECAL：ICAR: data_aug(angle/shear:10) + Unet: terrible
    # ECAL：ICAR: data_aug(angle/shear:10) + Udensenet: testing
    # for lr in [0.1, 0.01, 0.001, 0.0001]:
    #     print("#" * 20, "lr=", str(lr), "#" * 20)
    #     image_path = args.train_image_sep_ECAL
    #     mask_path = args.train_mask_sep_ECAL
    #     check_path = args.train_check_sep_ECAL + '/Udensenet_DA_V1/' + str(lr)
    #     # check_path = args.train_check_sep_ECAL + '/Unet_NDA/' + str(lr)
    #     train_net(device=device, image_path=image_path, mask_path=mask_path, check_path=check_path, args=args, lr=lr,
    #               kfolds=kfolds, epochs=epochs, batch_size=batch_size)

    # for lr in [0.01, 0.003, 0.001, 0.0003]:
    #     print("#" * 20, "lr=", str(lr), "#" * 20)
    #     image_path = args.train_image_sep_ECAR
    #     mask_path = args.train_mask_sep_ECAR
    #     check_path = args.train_check_sep_ECAR + '/Unet/data_aug/' + str(lr)
    #     # check_path = args.train_check_sep_ECAR + '/Unet/no_data_aug/' + str(lr)
    #     train_net(device=device, image_path=image_path, mask_path=mask_path, check_path=check_path, args=args, lr=lr)

    # To check spilt data files are right
    # image_path = r"E:\Win10_data\Segmentation\VMS_Unet\datasets\train_data\image_sep_position\ECAL"
    # mask_path = r"E:\Win10_data\Segmentation\VMS_Unet\datasets\train_label\circle_mask_sep\ECAL"
    # shuffle_indices=np.random.permutation(len(os.listdir(image_path+'/positive')))
    # print(shuffle_indices)
    # result = split_k_train_valdation(5, 1, image_path, mask_path,shuffle_indices)
    # for i in range(8):
    #     if i % 2==1:
    #         continue
    #     print("-" * (i + 1))
    #     print(len(result[i]))
    #     print(result[i])
