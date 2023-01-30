# -*- coding = utf-8 -*-
# @File Name : max_intensity_map
# @Date : 11/23/22 5:37 PM
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import numpy as np
import nibabel as nib
from PIL import Image


subject_id = 'M027'
tof_path = '/Users/zhiweideng/Desktop/NICR/7T_MRA/organized/test/{}/{}_TOF_MASKED.nii.gz'.format(subject_id, subject_id)
axi_path = '/Users/zhiweideng/Desktop/NICR/7T_MRA/organized/test/{}/{}_TOF_MIP_AXI.png'.format(subject_id, subject_id)
sag_path = '/Users/zhiweideng/Desktop/NICR/7T_MRA/organized/test/{}/{}_TOF_MIP_SAG.png'.format(subject_id, subject_id)
cor_path = '/Users/zhiweideng/Desktop/NICR/7T_MRA/organized/test/{}/{}_TOF_MIP_COR.png'.format(subject_id, subject_id)


def normalize(image):
    max_val, min_val = image.max(), image.min()
    return (image - min_val) / (max_val - min_val)


tof_image = nib.load(tof_path)
axi_mip_image_data = np.flip(np.max(tof_image.get_fdata(), axis=2).T, axis=[0, 1])
sag_mip_image_data = np.flip(np.max(tof_image.get_fdata(), axis=0).T, axis=[0, 1])
cor_mip_image_data = np.flip(np.max(tof_image.get_fdata(), axis=1).T, axis=[0, 1])
axi_mip_image_data = normalize(axi_mip_image_data) * 255.0
sag_mip_image_data = normalize(sag_mip_image_data) * 255.0
cor_mip_image_data = normalize(cor_mip_image_data) * 255.0

axi_mip_image = Image.fromarray(axi_mip_image_data.astype(np.uint8), mode='L')
axi_mip_image.save(axi_path)

sag_mip_image = Image.fromarray(sag_mip_image_data.astype(np.uint8), mode='L')
sag_mip_image = sag_mip_image.resize((sag_mip_image.size[0], sag_mip_image.size[1] * 2))
sag_mip_image.save(sag_path)

cor_mip_image = Image.fromarray(cor_mip_image_data.astype(np.uint8), mode='L')
cor_mip_image = cor_mip_image.resize((cor_mip_image.size[0], cor_mip_image.size[1] * 2))
cor_mip_image.save(cor_path)
