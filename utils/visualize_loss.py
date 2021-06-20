# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2021/03/30 21:54
# @Author  : yilinzhi
# @FileName: visualize_loss.py

import pandas as pd
from options.arguments_semi_submit import parse_args
import matplotlib.pyplot as plt
import numpy as np

args = parse_args()


def plot_loss(args):
    for flod in range(5):
        # if flod > 0:
        #     break

        # todo：修改位置！
        loss_path = args.check_ECAR_semi + "/Udensenet_TF_1/0.0001/" + str(flod) + '/data.csv'
        loss_data = pd.read_csv(loss_path)

        loss_data_numpy = np.zeros((len(loss_data), 4), dtype=np.float32)

        for i in range(len(loss_data)):
            loss_data_numpy[i][0] = int(loss_data.epoch.iloc[i])
            loss_data_numpy[i][1] = float(loss_data.train_loss_all_epochs[i])
            loss_data_numpy[i][2] = float(loss_data.val_loss_all_epochs[i])
            loss_data_numpy[i][3] = float(loss_data.dice_similarity_all_epochs[i])

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        ax.plot(loss_data_numpy[:, 0], loss_data_numpy[:, 1], "ro-", linewidth=1.0, label='train_loss')
        ax.plot(loss_data_numpy[:, 0], loss_data_numpy[:, 2], 'bs-', linewidth=1.0, label="val_loss")
        ax.plot(loss_data_numpy[:, 0], loss_data_numpy[:, 3], "g*-", linewidth=1.0, label='dice_value')
        ax.set_xlabel("epochs")
        ax.set_title("VMS_" + 'lr:0.0001_'+str(flod+1)+'_fold')
        ax.legend()

        fig.show()


if __name__ == '__main__':
    plot_loss(args)
