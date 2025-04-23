import torch.nn as nn
import torch
from torch import einsum
import torch.nn.functional as F
from einops import rearrange,repeat
from timm.models.vision_transformer import Block

class DSF(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=2,stride=2,groups=dim)
        self.conv1 = nn.Conv2d(2 * dim,out_channels=1, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(2 * dim,out_channels=1, kernel_size=1, bias=True)
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.sig = nn.Sigmoid()
        self.dim = dim
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, x1, x2):
        
        x1 = self.upconv(x1)

        s1 = torch.cat((x1,x),dim=1)
        s1 = self.conv1(s1)
        s1 = self.sig(s1)
        x1 = x1 * s1 

        s2 = torch.cat((x2,x),dim=1)
        s2 = self.conv2(s2)
        s2 = self.sig(s2)
        x2 = x2 * s2 

        c = torch.cat((x1,x2),dim=1)
        c = self.pooling(c)
        c = self.softmax(c)
        c1 = c[:, :self.dim, :, :]  # assuming self.dim represents the number of channels for c1
        c2 = c[:, self.dim:, :, :]  # the remaining channels go to c2

        x = c1 * x1 + c2 * x2 
        return x