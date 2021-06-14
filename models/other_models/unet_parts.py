# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def Cropping2D(target, refer):
    diffY = target.size()[2] - refer.size()[2]
    diffX = target.size()[3] - refer.size()[3]
    croped_target = target[:,:,(diffY // 2):(target.size()[2]-(diffY - diffY//2)),
                    (diffX // 2):(target.size()[3]-(diffX - diffX//2))]
    return croped_target


def Padding2D(outputs, inputs):
    diffY = inputs.size()[2] - outputs.size()[2]
    diffX = inputs.size()[3] - outputs.size()[3]
    paded_outputs = F.pad(outputs, (diffX // 2, diffX - diffX//2,
                                    diffY // 2, diffY - diffY//2))
    return paded_outputs


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      padding=(kernel_size//2, kernel_size-1-kernel_size//2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size,
                      padding=(kernel_size//2, kernel_size-1-kernel_size//2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class quadr_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super(quadr_conv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      padding=(kernel_size//2, kernel_size-1-kernel_size//2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size,
                      padding=(kernel_size//2, kernel_size-1-kernel_size//2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        return x

class inconv_bran(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1):
        super(inconv_bran, self).__init__()
        self.conv = double_conv(in_ch, out_ch, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return x
        

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return x

        
class in4conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super(in4conv, self).__init__()
        self.conv = quadr_conv(in_ch, out_ch, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return x
        

class down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, kernel_size)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
        
class down_drop(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(down_drop, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(0.5), 
            double_conv(in_ch, out_ch, kernel_size)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class down_and_concat_input(nn.Module):
    def __init__(self, in_ch, out_ch, pool_size, kernel_size=3):
        super(down_and_concat_input, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, kernel_size)
        )
        self.pool_input = nn.MaxPool2d(pool_size)

    def forward(self, x1, input):
        input_pool = self.pool_input(input)
        x = torch.cat([input_pool, x1], dim=1)
        x = self.mpconv(x)
        return x
        
class se_up(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, nearest=True):
        super(se_up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if nearest:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.se = SELayer(in_ch)
        self.conv = double_conv(in_ch, out_ch, kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = Padding2D(x1, x2)
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.se(x)
        x = self.conv(x)
        return x        
        
class up(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, nearest=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if nearest:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = Padding2D(x1, x2)
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class outconv_padding(nn.Module):
    def __init__(self):
        super(outconv_padding, self).__init__()

    def forward(self, outputs, inputs):
        x = Padding2D(outputs, inputs)
        return x