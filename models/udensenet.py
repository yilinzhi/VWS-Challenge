# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch
from models.unet_parts import *
from torchsummary import summary 


class udensenet(nn.Module):
    """Attention:
        Binary segmentation: sigmoid+BCE
        Multi segmentation: log_softmax+NULLoss == Cross Entropy
    """

    def __init__(self, n_channels, n_classes):
        super(udensenet, self).__init__()
        self.n_channels=n_channels
        self.n_classes=n_classes

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 96)
        self.down2 = down(96, 128)
        self.down3 = down(128, 256)
        self.down4 = down_drop(256, 512)
        self.tup = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1 = up(768, 256)
        self.up2 = up(384, 128)
        self.up3 = up(224, 96)
        self.up4 = up(160, 64)
        self.outc = outconv(544, n_classes)
        self.se = SELayer(544)
        self.outc_padding = outconv_padding()


    def forward(self, input):
        x1 = self.inc(input)# input->2 x1->64 200
        x2 = self.down1(x1)# x2->96 100
        x3 = self.down2(x2)#
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        tmp_up = x6
        x7 = self.up2(x6, x3)
        tmp_up = torch.cat([Padding2D(self.tup(tmp_up), x7),x7], dim=1)
        x8 = self.up3(x7, x2)
        tmp_up = torch.cat([Padding2D(self.tup(tmp_up), x8),x8], dim=1)
        x9 = self.up4(x8, x1)
        tmp_up = torch.cat([Padding2D(self.tup(tmp_up), x9),x9], dim=1)
        x10 = self.outc_padding(tmp_up, input)
        x11 = self.se(x10)
        output = self.outc(x11)
        return torch.sigmoid(output)

        
if __name__ == '__main__':
    model = udensenet(2,1)
    model.to('cuda')
    summary(model,(2,200,200))