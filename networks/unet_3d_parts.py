# -*- coding = utf-8 -*-
# @File Name : unet_3d_parts
# @Date : 2022/10/6 15:50
# @Author : dengzhiwei
# @E-mail : zhiweide@usc.edu


import torch
import torch.nn as nn
from loss.loss_func import calc_local_contrast


# following parts are components of the 3D-UNet
class SingleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, size=3, pad=1, act=nn.ELU()):
        super(SingleConv3d, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=size, padding=pad, bias=False)
        self.norm = nn.BatchNorm3d(out_ch)
        self.relu = act

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, size=3, pad=1, act=nn.ELU()):
        super(DoubleConv3d, self).__init__()
        self.conv1 = SingleConv3d(in_ch, out_ch, size, pad, act)
        self.conv2 = SingleConv3d(out_ch, out_ch, size, pad, act)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNetConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, size=3, pad=1):
        super(ResNetConv3d, self).__init__()
        self.residual = nn.Conv3d(in_ch, out_ch, kernel_size=size, padding=pad, bias=False)
        self.norm = nn.BatchNorm3d(out_ch)
        self.conv1 = SingleConv3d(in_ch, out_ch, size, pad)
        self.conv2 = SingleConv3d(in_ch, out_ch, size, pad)

    def forward(self, x):
        x = self.norm(self.residual(x))
        res = x
        x = self.conv1(x)
        x = self.conv2(x + res)
        return x


class SingleUpSample3d(nn.Module):
    def __init__(self, in_ch, out_ch, size=3, stride=2, pad=1, out_pad=1):
        super(SingleUpSample3d, self).__init__()
        size, stride, pad = (size, size, size), (stride, stride, stride), (pad, pad, pad)
        self.up_sample = nn.ConvTranspose3d(in_ch, out_ch,
                                            kernel_size=size,
                                            stride=stride,
                                            padding=pad,
                                            output_padding=out_pad,
                                            bias=False)

    def forward(self, size, x):
        # output_size = encoder_features.size()[2:]
        x = self.up_sample(x, size)
        return x


class SingleEncoder3d(nn.Module):
    def __init__(self, in_ch, out_ch, size=3, pool_size=2, pad=1, apply_pool=False):
        super(SingleEncoder3d, self).__init__()
        self.apply_pool = apply_pool
        if self.apply_pool:
            self.pool = nn.MaxPool3d(kernel_size=pool_size)
        self.conv = DoubleConv3d(in_ch, out_ch, size, pad)

    def forward(self, x):
        if self.apply_pool:
            x = self.pool(x)
        x = self.conv(x)
        return x


class SingleDecoder3d(nn.Module):
    def __init__(self, in_ch, out_ch, size=3, stride=2, pad=0, out_pad=0):
        super(SingleDecoder3d, self).__init__()
        self.up_sample = SingleUpSample3d(in_ch-out_ch, in_ch-out_ch, size, stride, pad, out_pad)
        self.conv = DoubleConv3d(in_ch, out_ch, size)

    def forward(self, encoder_features, x):
        x = self.up_sample(encoder_features.size()[2:], x)
        x = torch.cat((encoder_features, x), dim=1)
        x = self.conv(x)
        return x


class SpatialAttention3d(nn.Module):
    def __init__(self, out_ch, min_scale, max_scale, size=3, pad=1, sample_num=16, sample_layers=3):
        super(SpatialAttention3d, self).__init__()
        self.min_scale, self.max_scale = min_scale, max_scale
        self.sample_num = sample_num
        self.sample_layer = sample_layers
        self.radius_conv = SingleConv3d(out_ch, 1, size, pad, act=nn.Sigmoid())
        self.attention_conv = SingleConv3d(3, 1, size, pad, act=nn.Sigmoid())

    def forward(self, image, x, radius=None):
        if radius is None:
            radius = self.radius_conv(x) * self.max_scale + self.min_scale
        contrast = calc_local_contrast(image, radius, self.sample_num, self.sample_layer)
        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_x, max_x, contrast], dim=1)
        attention = self.attention_conv(attention)
        return attention
