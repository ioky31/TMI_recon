import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class module(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(module, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=(1, 1), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=(1, 1), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=(1, 1), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=(1, 1), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.Module1 = module(1, 64)
        self.Module2 = module(64, 64)
        self.Module3 = module(64, 64)
        self.Module4 = module(64, 64)

        self.Module5 = module(64, 64)

        self.Module6 = module(128, 64)
        self.Module7 = module(128, 64)
        self.Module8 = module(128, 64)
        self.Module9 = module(128, 64)

        self.End_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        e1 = self.Module1(x)
        e2 = self.Module2(e1)
        e3 = self.Module3(e2)
        e4 = self.Module4(e3)

        e5 = self.Module5(e4)
        e5 = torch.cat((e4, e5), dim=1)
        e6 = self.Module6(e5)
        e6 = torch.cat((e3, e6), dim=1)
        e7 = self.Module7(e6)
        e7 = torch.cat((e2, e7), dim=1)
        e8 = self.Module8(e7)
        e8 = torch.cat((e1, e8), dim=1)
        e9 = self.Module9(e8)

        out = self.End_conv(e9)

        return out
