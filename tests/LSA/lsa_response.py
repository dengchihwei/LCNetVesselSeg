# -*- coding = utf-8 -*-
# @File Name : lsa_test
# @Date : 1/9/23 12:28 AM
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import sys
import torch
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
sys.path.append('/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/UnsupervisedVesselSeg/codes')
from trainer import Trainer
from loss.loss_func import flux_loss_asymmetry
from datasets.datasets_3d import get_data_loader_3d


IMG_HEIGHT = 60
IMG_WIDTH = 124
IMG_DEPTH = 124
patch_size = 48

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='ours')
parser.add_argument('--model_path', type=str,
                    default='../../../trained_models/DARK_LSA/2023-02-04/80-epoch-2023-02-04.pth')
parser.add_argument('--device', type=str, default='cuda:2')
parser.add_argument('--type', type=str, default='adaptive_lc')
parser.add_argument('--split', type=str, default='test')


def load_model(arguments):
    device = arguments.device
    model_path = arguments.model_path
    checkpoint = torch.load(model_path, map_location=device)
    model = Trainer(configer=checkpoint['configer']).get_model()
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    print('load model done')
    return model


def model_response(arguments):
    data_loader = get_data_loader_3d(data_name='LSA', split=arguments.split, batch_size=1, shuffle=False)
    model, device = load_model(arguments), arguments.device

    num_files = len(data_loader.dataset.mask_files)
    responses = np.zeros((num_files, 60, 124, 124))
    radius = np.zeros((num_files, 128, 60, 124, 124))
    directions = np.zeros((num_files, 3, 60, 124, 124))
    attentions = np.zeros((num_files, 60, 124, 124))
    for batch in tqdm(data_loader, desc=str(0), unit='b'):
        images = batch['image'].to(device)
        with torch.no_grad():
            output = model(images)
            curr_radius = output['radius']
            curr_direction = output['vessel']
            curr_attention = output['attentions'][0]
            curr_response, _ = flux_loss_asymmetry(images, output, 128, (2, 3, 4))
            for i in range(curr_response.size(0)):
                image_id, c = batch['image_id'][i], batch['start_coord'][i]
                patch_response = curr_response[i]
                patch_response = patch_response.cpu().detach().numpy()

                patch_radii = curr_radius[i]
                patch_radii = patch_radii.cpu().detach().numpy()
                patch_direction = curr_direction[i]
                patch_direction = patch_direction.cpu().detach().numpy()
                patch_attention = curr_attention[i][0]
                patch_attention = patch_attention.cpu().detach().numpy()

                start_x, start_y, start_z = c[0], c[1], c[2]
                end_x, end_y, end_z = c[0] + patch_size, c[1] + patch_size, c[2] + patch_size
                responses[image_id, start_x:end_x, start_y:end_y, start_z:end_z] = \
                    np.maximum(patch_response, responses[image_id, start_x:end_x, start_y:end_y, start_z:end_z])
                radius[image_id, :, start_x:end_x, start_y:end_y, start_z:end_z] = patch_radii
                directions[image_id, :, start_x:end_x, start_y:end_y, start_z:end_z] = patch_direction
                attentions[image_id, start_x:end_x, start_y:end_y, start_z:end_z] = \
                    np.maximum(patch_attention, attentions[image_id, start_x:end_x, start_y:end_y, start_z:end_z])

    os.makedirs("LSA_{}_{}".format(arguments.method, arguments.split), exist_ok=True)
    for i in range(num_files):
        mask_path = data_loader.dataset.mask_files[i]
        subject_id = mask_path.split('/')[-2]
        mask_image = sitk.ReadImage(mask_path)
        res_path = 'LSA_{}_{}/LSA_{}.nii.gz'.format(arguments.method, arguments.split, subject_id)
        response = np.multiply(responses[i], sitk.GetArrayFromImage(mask_image))
        res_image = sitk.GetImageFromArray(response)
        res_image.CopyInformation(mask_image)
        sitk.WriteImage(res_image, res_path)
        # save radius
        # np.save('LSA_{}_{}/LSA_{}_rad.npy'.format(arguments.method, arguments.split, subject_id), radius[i])
        # np.save('LSA_{}_{}/LSA_{}_dir.npy'.format(arguments.method, arguments.split, subject_id), directions[i])
        # np.save('LSA_{}_{}/LSA_{}_att.npy'.format(arguments.method, arguments.split, subject_id), attentions[i])
        print('Model Responses Saved at LSA_{}_{}/{}'.format(arguments.method, arguments.split, res_path))


if __name__ == '__main__':
    args = parser.parse_args()
    model_response(args)
