# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch
from unet_parts import *
from torchsummary import summary 

class UNet_deep_sup(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_deep_sup, self).__init__()
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

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        output_2 = self.outc_padding(x8, x2)
        output_2 = self.outc2(output_2)
        ### last
        x9 = self.up4(x8, x1)
        x10 = self.outc_padding(x9, input)
        output = self.outc(x10)
        return torch.sigmoid(output_2),torch.sigmoid(output)

        
if __name__ == '__main__':
    model = UNet(2,1)
    model.to('cuda')
    summary(model,(2,200,200))