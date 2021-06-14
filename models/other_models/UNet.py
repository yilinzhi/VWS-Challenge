# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch
from model.unet_parts import *
from torchsummary import summary 
from torchviz import make_dot
from model.pointrend import *
class UNet_add_branch_2(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_add_branch_2, self).__init__()
        self.branch = inconv_bran(n_channels, 64)
        self.inc = inconv(64, 64)
        self.down1 = down(64, 96)
        self.down2 = down(96, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.up1 = up(768, 256)
        self.up2 = up(384, 128)
        self.up3 = up(224, 96)
        self.up4 = up(160, 64)
        self.outc = outconv(64+64, n_classes)
        self.outc_padding = outconv_padding()

    def forward(self, input):
        bran_x = self.branch(input)
        x1 = self.inc(bran_x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x10 = self.outc_padding(torch.cat([bran_x, x9], dim=1), input)
        output = self.outc(x10)
        return torch.sigmoid(output)




class UNet_add_branch(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_add_branch, self).__init__()
        self.branch = inconv_bran(n_channels, 64)
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 96)
        self.down2 = down(96, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.up1 = up(768, 256)
        self.up2 = up(384, 128)
        self.up3 = up(224, 96)
        self.up4 = up(160, 64)
        self.outc = outconv(64+64, n_classes)
        self.outc_padding = outconv_padding()

    def forward(self, input):
        bran_x = self.branch(input)
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x10 = self.outc_padding(torch.cat([bran_x, x9], dim=1), input)
        output = self.outc(x10)
        return torch.sigmoid(output)

class UNet_rend(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_rend, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 96)
        self.down2 = down(96, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.up1 = up(768, 256)
        self.up2 = up(384, 128)
        self.up3 = up(224, 96)
        self.up4 = up(160, 64)
        self.outc = outconv(64, n_classes)
        self.outc2 = outconv(96, n_classes)
        self.outc_padding = outconv_padding()
        self.pointrend = PointHead()
        
    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        ### 输出100*100的输出
        output_2 = self.outc_padding(x8, x2)
        output_2 = self.outc2(output_2)
        ###
        x9 = self.up4(x8, x1)
        x10 = self.outc_padding(x9, input)
        ### 利用高一级的特征和低分辨率的输出
        rend = self.pointrend(input, x10, output_2)
        # output = self.outc(x10)
        return rend, output_2

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 96)
        self.down2 = down(96, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.up1 = up(768, 256)
        self.up2 = up(384, 128)
        self.up3 = up(224, 96)
        self.up4 = up(160, 64)
        self.outc = outconv(64, n_classes)
        self.outc_padding = outconv_padding()

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x10 = self.outc_padding(x9, input)
        output = self.outc(x10)
        #return F.softmax(output, dim=1)
        return torch.sigmoid(output)
        
class UNet_se(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_se, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 96)
        self.down2 = down(96, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.up1 = se_up(768, 256)
        self.up2 = se_up(384, 128)
        self.up3 = se_up(224, 96)
        self.up4 = se_up(160, 64)
        self.outc = outconv(64, n_classes)
        self.outc_padding = outconv_padding()
        
    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x10 = self.outc_padding(x9, input)
        output = self.outc(x10)
        return torch.sigmoid(output)

class UNet_cas(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 96)
        self.down2 = down(96, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.up1 = up(768, 256)
        self.up2 = up(384, 128)
        self.up3 = up(224, 96)
        self.up4 = up(160, 64)
        self.outc = outconv(64, n_classes)
        self.outc_padding = outconv_padding()

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x10 = self.outc_padding(x9, input)
        output = self.outc(x10)
        output = torch.sigmoid(output)
        ###
        flair = input[:,0:1,...]
        
        return torch.sigmoid(output)


class UNet_multiscale(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_multiscale, self).__init__()
        ### 3*3 kernel size
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 96)
        self.down2 = down(96, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.up1 = up(768, 256)
        self.up2 = up(384, 128)
        self.up3 = up(224, 96)
        self.up4 = up(160, 64)
        self.outc = outconv(64+64, n_classes)
        self.outc_padding = outconv_padding()
        ### 5*5
        self.inc_5 = inconv(n_channels, 64, 5)
        self.down1_5 = down(64, 96, 5)
        self.down2_5 = down(96, 128, 5)
        self.down3_5 = down(128, 256, 5)
        self.down4_5 = down(256, 512, 5)
        self.up1_5 = up(768, 256, 5)
        self.up2_5 = up(384, 128, 5)
        self.up3_5 = up(224, 96, 5)
        self.up4_5 = up(160, 64, 5)
    def forward(self, input):
        ### 3*3 kernel size
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x10 = self.outc_padding(x9, input)
        ### 5*5 kernel size
        x1_5 = self.inc_5(input)
        x2_5 = self.down1_5(x1_5)
        x3_5 = self.down2_5(x2_5)
        x4_5 = self.down3_5(x3_5)
        x5_5 = self.down4_5(x4_5)
        x6_5 = self.up1_5(x5_5, x4_5)
        x7_5 = self.up2_5(x6_5, x3_5)
        x8_5 = self.up3_5(x7_5, x2_5)
        x9_5 = self.up4_5(x8_5, x1_5)
        x10_5 = self.outc_padding(x9_5, input)
        ###
        x11 = torch.cat([x10, x10_5], dim=1)
        output = self.outc(x11)
        #return F.softmax(output, dim=1)
        return torch.sigmoid(output)    
if __name__ == '__main__':
    model = UNet(2,1)
    x = torch.zeros(1, 2, 192, 192, dtype=torch.float, requires_grad=False)
    out = model(x)
    g = make_dot(out)
    g.view()
