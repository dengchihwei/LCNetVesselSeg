# -*- coding = utf-8 -*-
# @File Name : vessel12
# @Date : 2022/7/25 00:39
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from PIL import Image
from PIL import ImageFilter


# parent path
train_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/vessel12/train'
valid_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/vessel12/test'


# read tr file func
def convert_mhd_to_nifti(parent_path, subject_id, image_type=''):
    """
    change .mha file to nifti file
    :param parent_path: dataset folder
    :param subject_id: subject folder
    :param image_type: ct image or mask
    :return: None
    """
    mra_path = os.path.join(parent_path, subject_id, image_type + subject_id + '.mhd')
    mra_img = sitk.GetArrayFromImage(sitk.ReadImage(mra_path))
    mra_output_path = os.path.join(parent_path, subject_id, '{}{}.nii.gz'.format(image_type, subject_id))
    sitk.WriteImage(sitk.GetImageFromArray(mra_img), mra_output_path)
    print('Saved MRA Image at', mra_output_path)


def gen_mip_images(parent_path, subject_id):
    """
    generate maximum intensity projections in three axis
    :param parent_path: dataset folder
    :param subject_id: subject folder
    :return: None
    """
    ct_image_path = os.path.join(parent_path, subject_id, subject_id + '.mhd')
    mask_image_path = os.path.join(parent_path, subject_id, 'mask_{}.mhd'.format(subject_id))

    # read image
    ct_img = sitk.GetArrayFromImage(sitk.ReadImage(ct_image_path))
    mask_img = sitk.GetArrayFromImage(sitk.ReadImage(mask_image_path))
    masked_ct_img = ct_img * mask_img
    masked_ct_img = np.clip(masked_ct_img, a_min=-900, a_max=100)

    # get maximum intensity projection
    axi_mip_image_data = np.flip(np.max(masked_ct_img, axis=2).T, axis=[0, 1])
    sag_mip_image_data = np.flip(np.max(masked_ct_img, axis=0).T, axis=[0, 1])
    cor_mip_image_data = np.flip(np.max(masked_ct_img, axis=1).T, axis=[0, 1])
    axi_mip_image_data = normalize(axi_mip_image_data) * 255.0
    sag_mip_image_data = normalize(sag_mip_image_data) * 255.0
    cor_mip_image_data = normalize(cor_mip_image_data) * 255.0

    # get maximum intensity projection path
    os.makedirs(os.path.join(parent_path, subject_id, 'mips'), exist_ok=True)
    axi_path = os.path.join(parent_path, subject_id, 'mips', 'AXI.png')
    sag_path = os.path.join(parent_path, subject_id, 'mips', 'SAG.png')
    cor_path = os.path.join(parent_path, subject_id, 'mips', 'COR.png')

    axi_mip_image = Image.fromarray(axi_mip_image_data.astype(np.uint8), mode='L')
    axi_mip_image = axi_mip_image.filter(ImageFilter.SMOOTH)
    axi_mip_image.save(axi_path)
    sag_mip_image = Image.fromarray(sag_mip_image_data.astype(np.uint8), mode='L')
    sag_mip_image = sag_mip_image.filter(ImageFilter.SMOOTH)
    sag_mip_image.save(sag_path)
    cor_mip_image = Image.fromarray(cor_mip_image_data.astype(np.uint8), mode='L')
    cor_mip_image = cor_mip_image.filter(ImageFilter.SMOOTH)
    cor_mip_image.save(cor_path)


def normalize(image):
    max_val, min_val = image.max(), image.min()
    return (image - min_val) / (max_val - min_val)


if __name__ == '__main__':
    train_subjects = sorted(os.listdir(train_path))
    valid_subjects = sorted(os.listdir(valid_path))

    for folder in tqdm(train_subjects):
        # convert_mhd_to_nifti(train_path, folder)
        # convert_mhd_to_nifti(train_path, folder, image_type='mask_')
        gen_mip_images(train_path, folder)

    for folder in tqdm(valid_subjects):
        # convert_mhd_to_nifti(valid_path, folder)
        # convert_mhd_to_nifti(valid_path, folder, image_type='mask_')
        gen_mip_images(valid_path, folder)
