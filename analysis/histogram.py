# -*- coding = utf-8 -*-
# @File Name : histogram
# @Date : 1/14/23 9:09 PM
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import torch
import random
import numpy as np
import SimpleITK as sitk

from matplotlib import pyplot as plt
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode


def rotate_3d(image, angles, expand=False, fill=0.0):
    """
    rotate the 3d images with specified angles
    :param image: 3d volume images, [H, W, D]
    :param angles: angles of rotation in 3 different axis, (x, y, z), [,3]
    :param expand: ignored
    :param fill: if given a number, the value is used for all bands respectively.
    :return: image, rotated image or recovered image
    """
    image = rotate(image, interpolation=InterpolationMode.BILINEAR, angle=angles[0].item(), expand=expand, fill=fill)
    image = image.permute((1, 0, 2))
    image = rotate(image, interpolation=InterpolationMode.BILINEAR, angle=-angles[1].item(), expand=expand, fill=fill)
    image = image.permute((1, 0, 2))
    image = image.permute((2, 1, 0))
    image = rotate(image, interpolation=InterpolationMode.BILINEAR, angle=-angles[2].item(), expand=expand, fill=fill)
    image = image.permute((2, 1, 0))
    return image.squeeze(0)


def rotate_3d_batch(batch_images, batch_angles):
    """
    rotate the 3d images with specified angles
    :param batch_images: 3d volume images, [B, C, H, W, D]
    :param batch_angles: angles of rotation in 3 different axis, (x, y, z), [B, 3]
    :return: batch_images_rotated, rotated image or recovered image
    """
    device = batch_images.device
    batch_num = batch_images.size(0)
    channel_num = batch_images.size(1)
    batch_images_rotated = torch.zeros(batch_images.size())
    for i in range(batch_num):
        for j in range(channel_num):
            batch_images_rotated[i][j] = rotate_3d(batch_images[i][j], batch_angles[i])
    return batch_images_rotated.to(device)


def rotation_matrix(angles):
    alpha, beta, gamma = angles / 180 * torch.pi
    yaw = torch.tensor([[1, 0, 0],
                       [0, torch.cos(alpha), -torch.sin(alpha)],
                       [0, torch.sin(alpha), torch.cos(alpha)]])
    pitch = torch.tensor([[torch.cos(beta), 0, torch.sin(beta)],
                         [0, 1, 0],
                         [-torch.sin(beta), 0, torch.cos(beta)]])
    roll = torch.tensor([[torch.cos(gamma), -torch.sin(gamma), 0],
                        [torch.sin(gamma), torch.cos(gamma), 0],
                        [0, 0, 1]])
    r_matrix = torch.matmul(roll, torch.matmul(pitch, yaw))
    return r_matrix


original_dirs = torch.rand(4, 3, 32, 32, 32)
rotation_angles = torch.rand(4, 3)

original_dirs[0, :, 16, 0, 0] = 100.0
rotation_angles[0, :] = torch.tensor([0.0, 90.0, 0.0])
rotated_dirs = rotate_3d_batch(original_dirs, rotation_angles)

r_matrix_0 = rotation_matrix(rotation_angles[0])
a = torch.tensor([[1.0, 1.0, 1.0]])
b = torch.matmul(r_matrix_0, a.T)
print(b)

c = torch.tensor([[1.0, 0, 0]])
d = torch.matmul(r_matrix_0, c.T)
print(d)

# xs = [16, 16, 0, 31, 31, 0, 16]
# ys = [31, 0, 0, 0, 16, 15, 0]
# zs = [0, 31, 15, 16, 0, 0, 0]
#
# for i in range(7):
#     print('{}, {}, {}'.format(xs[i], ys[i], zs[i]), rotated_dirs[0, :, xs[i], ys[i], zs[i]])

