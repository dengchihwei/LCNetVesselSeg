# -*- coding = utf-8 -*-
# @File Name : visualization
# @Date : 1/18/23 10:21 AM
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import torch
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt


x_loc = 40
mra_image = sitk.ReadImage('/Users/zhiweideng/Desktop/NICR/DarkVessels/UnilateralData/T1SPC_NLM03_h1_subj_17.nii')
mra_image = sitk.GetArrayFromImage(mra_image)
learnt_dirs = np.load('/Users/zhiweideng/Desktop/NICR/DarkVessels/analysis/LSA_h1_subj_17_dir.npy')
learnt_rads = np.load('/Users/zhiweideng/Desktop/NICR/DarkVessels/analysis/LSA_h1_subj_17_rad.npy')


# plt.figure(figsize=(15, 15), num=1234)
# plt.imshow(mra_image[x_loc, :, :], cmap='gray')
# plt.show()

direction_slice = learnt_dirs[:, x_loc, :, :]
print(direction_slice.shape)


def overlay_quiver(image, flow):
    """
    :param image: [H, W]
    :param flow: [2, H, W]
    :return:
    """
    scale = 1
    vx, vy = flow[0], flow[1]
    # norm = np.sqrt(vx ** 2 + vy ** 2 + 1e-20)
    # vx, vy = vx / norm, vy / norm
    x = np.arange(124)
    y = np.arange(124)
    xx, yy = np.meshgrid(x, y)
    xx = xx[::scale, ::scale]
    yy = yy[::scale, ::scale]
    vx = vx[::scale, ::scale]
    vy = vy[::scale, ::scale]
    plt.figure(figsize=(10, 10), num=1234)
    plt.imshow(image, 'gray')
    plt.quiver(xx, yy, vx, vy, color='cyan')
    plt.axis('off')
    plt.show()


def get_sampling_vec(num_pts):
    indices = np.arange(0, num_pts, dtype=np.float32)
    phi = np.arccos(1 - 2 * indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    # flip coordinates according to the sample grid
    vectors = np.vstack((z, y, x)).T  # This is a sphere sampling
    return vectors


overlay_quiver(mra_image[x_loc, :, :], direction_slice[:2])

# mean_size = np.mean(learnt_rads, axis=0)
# plt.figure(figsize=(15, 15), num=1234)
# plt.imshow(mean_size[x_loc, :, :], cmap='hot')
# plt.colorbar()
# plt.show()


def overlay_sizes(image, flow):
    """
    :param image: [H, W]
    :param flow: [128, 2, H, W]
    :return:
    """
    scale = 1
    vx, vy = flow[:, 0], flow[:, 1]
    # norm = np.sqrt(vx ** 2 + vy ** 2 + 1e-20)
    # vx, vy = vx / norm, vy / norm
    x = np.arange(124)
    y = np.arange(124)
    xx, yy = np.meshgrid(x, y)
    xx = xx[::scale, ::scale]
    yy = yy[::scale, ::scale]
    vx = vx[:, ::scale, ::scale]
    vy = vy[:, ::scale, ::scale]
    plt.figure(figsize=(10, 10), num=1234)
    plt.imshow(image, 'gray')
    for i in range(128):
        # if vx[i, 81, 46] > 1.0 or vy[i, 81, 46] > 1.0:
        #     continue
        _vx, _vy = vx[i], vy[i]
        plt.quiver(xx, yy, _vx, _vy, color='cyan', scale_units='xy', scale=1)
    plt.axis('off')
    plt.show()


# learnt_rads = torch.tensor(learnt_rads).unsqueeze(1).permute(2, 3, 4, 0, 1)
# sphere_vectors = torch.tensor(get_sampling_vec(128)).repeat(60, 124, 124, 1, 1)
# scaled_rads = torch.multiply(sphere_vectors, learnt_rads).permute(3, 4, 0, 1, 2)
# print(scaled_rads.size())
#
#
# y, z = 45, 36
# direction_slice = scaled_rads[:, :, x_loc, :, :]
# zero_slices = torch.zeros(direction_slice.size())
# zero_slices[:, :, y, z] = direction_slice[:, :, y, z]
# overlay_sizes(mra_image[x_loc, :, :], zero_slices[:, :2])

# learnt_attentions = np.load('/Users/zhiweideng/Desktop/NICR/DarkVessels/analysis/LSA_h1_subj_13_att.npy')
# mask_image = sitk.ReadImage('/Users/zhiweideng/Desktop/NICR/DarkVessels/UnilateralData/mask_h1_subj_13.nii')
# mask_image = sitk.GetArrayFromImage(mask_image)
# # learnt_attentions = np.multiply(learnt_attentions, mask_image)
#
# plt.figure(figsize=(15, 15), num=1234)
# attention_slice = learnt_attentions[10, :, :]
# # attention_slice = (attention_slice - np.min(attention_slice)) / (np.max(attention_slice) - np.min(attention_slice))
# plt.imshow(attention_slice, cmap='Reds')
# plt.colorbar()
# plt.show()
