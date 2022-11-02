import torch
import torch.nn as nn

# 设置随机数种子,使结果可复现

class Euclidean_loss(nn.Module):
    def __init__(self):
        super(Euclidean_loss, self).__init__()

    def forward(self, inputs, targets):
        return torch.sqrt(torch.sum((inputs - targets) ** 2))

class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, inputs, targets):
        MSE = torch.sqrt(torch.sum((inputs - targets) ** 2, (1, 2, 3))/(inputs.shape[1]*inputs.shape[2]*inputs.shape[3]))
        return torch.mean(10*torch.log10(1 / MSE))

class CNR(nn.Module):
    def __init__(self):
        super(CNR, self).__init__()

    def forward(self, inputs, targets):
        return torch.abs(torch.mean(inputs, (1, 2, 3)) - torch.mean(targets, (1, 2, 3))) / \
               torch.sqrt(torch.var(inputs, (1, 2, 3)) + torch.var(targets, (1, 2, 3)))
