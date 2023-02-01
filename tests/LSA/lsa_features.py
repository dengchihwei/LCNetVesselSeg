# -*- coding = utf-8 -*-
# @File Name : lsa_features
# @Date : 1/30/23 1:58 PM
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import sys
import torch
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from torch.nn import functional as F
sys.path.append('/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/UnsupervisedVesselSeg/codes')
from networks.unet_3d import LocalContrastNet3D
from datasets.datasets_3d import get_data_loader_3d


IMG_HEIGHT = 60
IMG_WIDTH = 124
IMG_DEPTH = 124
patch_size = 48


parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='ours')
parser.add_argument('--model_path', type=str,
                    default='../../../trained_models/DARK_LSA/2023-01-29/100-epoch-2023-01-29.pth')
parser.add_argument('--device', type=str, default='cuda:2')
parser.add_argument('--type', type=str, default='adaptive_lc')
parser.add_argument('--split', type=str, default='train')


class LCNetFeature3D(LocalContrastNet3D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, im):
        # encoding
        x1 = self.encoder1(im)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        # apply attention
        attention1 = self.level_attention1(im, x1)
        x1 = torch.mul(attention1, x1)
        x = F.interpolate(im, scale_factor=0.5, mode='trilinear')
        attention2 = self.level_attention2(x, x2)
        x2 = torch.mul(attention2, x2)
        x = F.interpolate(x, scale_factor=0.5, mode='trilinear')
        attention3 = self.level_attention3(x, x3)
        x3 = torch.mul(attention3, x3)

        # decoding
        x = self.decoder1(x3, x4)
        x = self.decoder2(x2, x)
        x = self.decoder3(x1, x)
        return x


def model_features(arguments):
    device = arguments.device
    data_loader = get_data_loader_3d(data_name='LSA', split=arguments.split, batch_size=1, shuffle=False)
    num_files = len(data_loader.dataset.mask_files)
    features = np.zeros((num_files, 64, 60, 124, 124))

    # define model
    checkpoint = torch.load(arguments.model_path, map_location=device)
    configer = checkpoint['configer']
    input_ch = configer['arch']['in_channels']
    output_ch = configer['arch']['out_channels']
    min_r_scale = configer['arch']['min_scale']
    max_r_scale = configer['arch']['max_scale']
    radius_sample_num = configer['arch']['radius_num']
    feature_dims = configer['arch']['feature_dims']
    model = LCNetFeature3D(input_ch, output_ch, min_r_scale, max_r_scale, radius_sample_num, feature_dims)
    # load trained model
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    print('load model done')

    for batch in tqdm(data_loader, desc=str(0), unit='b'):
        images = batch['image'].to(device)
        with torch.no_grad():
            curr_features = model(images)
            for i in range(curr_features.size(0)):
                image_id, c = batch['image_id'][i], batch['start_coord'][i]
                patch_features = curr_features[i]
                patch_features = patch_features.cpu().detach().numpy()
                start_x, start_y, start_z = c[0], c[1], c[2]
                end_x, end_y, end_z = c[0] + patch_size, c[1] + patch_size, c[2] + patch_size
                features[image_id, :, start_x:end_x, start_y:end_y, start_z:end_z] = patch_features
    os.makedirs("LSA_{}_{}".format(arguments.method, arguments.split), exist_ok=True)
    for i in range(num_files):
        mask_path = data_loader.dataset.mask_files[i]
        subject_id = mask_path.split('/')[-2]
        # save feature
        np.save('LSA_{}_{}/LSA_{}_feat.npy'.format(arguments.method, arguments.split, subject_id), features[i])
        print('Model Responses Saved at LSA_{}_{}/{}'.format(arguments.method, arguments.split, subject_id))


if __name__ == '__main__':
    args = parser.parse_args()
    model_features(args)
