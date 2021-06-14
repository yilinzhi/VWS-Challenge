# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch
from unet_parts import *
from torchsummary import summary 


class udensedown(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(udensedown, self).__init__()
        self.inc = in4conv(n_channels, 64)
        self.down1 = down_and_concat_input(64+2, 96, 1)
        self.down2 = down_and_concat_input(96+2, 128, 2)
        self.down3 = down_and_concat_input(128+2, 256, 4)
        self.down4 = down_and_concat_input(256+2, 512, 8)
        self.up1 = up(768, 256)
        self.up2 = up(384, 128)
        self.up3 = up(224, 96)
        self.up4 = up(160, 64)
        self.outc = outconv(64, n_classes)
        self.outc_padding = outconv_padding()

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1, input)
        x3 = self.down2(x2, input)
        x4 = self.down3(x3, input)
        x5 = self.down4(x4, input)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x10 = self.outc_padding(x9, input)
        output = self.outc(x10)
        return torch.sigmoid(output)

        
if __name__ == '__main__':
    model = udensedown(2,1)
    model.to('cuda')
    summary(model,(2,200,200))