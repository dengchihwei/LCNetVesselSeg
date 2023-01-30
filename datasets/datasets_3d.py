# -*- coding = utf-8 -*-
# @File Name : datasets_3d
# @Date : 12/31/22 3:33 AM
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import torch
import numpy as np
from tqdm import tqdm
import SimpleITK as SiTk
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


patch_size, spacing = 64, 48
dataset_paths = {
    'TUBETK': '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/TubeTK',
    'VESSEL12': '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/vessel12',
    '7T': '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/7T_Lirong/organized',
    'LSA': '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/DarkVessels/UnilateralData'
}


class Dataset3D(Dataset):
    def __init__(self, data_dir, data_name='TUBETK', split='train', supervised=False):
        self.split = split
        self.data_name = data_name
        self.total_patch_num = 0
        self.supervised = supervised
        self.data_dir = self.get_data_dir(data_dir)
        # define patch size and spacing
        self.patch_size = int(patch_size * 3 / 4) if self.data_name == 'LSA' else patch_size
        self.spacing = int(spacing * 3 / 4) if self.data_name == 'LSA' else spacing

        # get image, mask and label files
        self.image_files, self.mask_files, self.label_files = self.get_file_lists()

        # load images, use file path as dict
        self.images_dict = self.load_images()

    def __len__(self):
        return self.total_patch_num

    def __getitem__(self, index):
        # calculate the patch index is landed in which image and patch exactly
        image_idx, patch_idx = self._calc_patch_index(index)

        # load images
        image_path = self.image_files[image_idx]
        image = self.images_dict[image_path]['image']
        image = -image if self.data_name == 'LSA' else image

        # get image, label and mask patch
        start_coord = self.get_start_coord(image_path, image.shape, patch_idx)
        image_patch = self.crop_image_patch(image, start_coord)
        '''
        if 'mask' in self.images_dict[image_path].keys():
            mask = self.images_dict[image_path]['mask']
            mask_patch = self.crop_image_patch(mask, start_coord)
            # apply the mask
            image_patch = np.multiply(image_patch, mask_patch)
        '''

        # get random rotation angles
        rotation_angles = self.get_random_angle(180, 1.0)

        # convert to torch types
        image_patch = torch.from_numpy(image_patch.copy()).unsqueeze(0)
        item = {
            'image_id': image_idx,
            'image': image_patch.float(),
            'start_coord': torch.LongTensor(start_coord),
            'rotation_angles': torch.tensor(rotation_angles)
        }

        # only for dataset with dense labels and supervised learning
        if self.supervised:
            label = self.images_dict[image_path]['label']
            label_patch = self.crop_image_patch(label, start_coord)
            # convert to torch
            label_patch = torch.from_numpy(label_patch.copy()).unsqueeze(0)
            item['label'] = label_patch
        return item

    def get_start_coord(self, image_path, image_shape, patch_idx):
        """
        get the image patch's start pixel position
        :param image_path: path to the image
        :param image_shape: the shape of the image, no need to pass all the images
        :param patch_idx: index of the image patch
        :return: start_coord, location of the start pixel of patch
        """
        start_coord = np.zeros(3)
        patch_num_dim = self.images_dict[image_path]['num_dim']
        for i in range(3):
            start_coord[i] = patch_idx % patch_num_dim[i]
            patch_idx = patch_idx // patch_num_dim[i]
        start_coord = (self.spacing * start_coord).astype(np.int16)
        end_coord = start_coord + self.patch_size
        # in case of exceed the boundaries
        for i in range(3):
            if end_coord[i] > image_shape[i]:
                end_coord[i] = image_shape[i]
                start_coord[i] = end_coord[i] - self.patch_size
        return start_coord

    def _calc_patch_index(self, index):
        image_idx = 0
        curr_image_path = self.image_files[image_idx]
        while index >= self.images_dict[curr_image_path]['total_num']:
            index -= self.images_dict[curr_image_path]['total_num']
            image_idx += 1
            curr_image_path = self.image_files[image_idx]
        patch_index = index
        return image_idx, patch_index

    def get_data_dir(self, data_dir):
        """
        get dataset directory from the dataset name and split
        :param data_dir: data directory
        :return: data_dir transformed data directory
        """
        if self.data_name == 'TUBETK':
            data_dir = data_dir
        elif self.data_name == 'VESSEL12':
            data_dir = os.path.join(data_dir, self.split)
        elif self.data_name == '7T':
            data_dir = os.path.join(data_dir, self.split)
        elif self.data_name == 'LSA':
            data_dir = data_dir
        else:
            data_dir = None
        assert data_dir is not None
        return data_dir

    def get_file_lists(self):
        """
        get the image, mask and label files' path for later read the files
        :return: image_files, image file path list
                 mask_files, mask file path list
                 label_files, label file path list
        """
        # get all the subject folder path
        subject_folders = sorted(os.listdir(self.data_dir))
        image_files, mask_files, label_files = [], [], []

        # iterate all the folders to get the files' paths
        for folder in subject_folders:
            subject_path = os.path.join(self.data_dir, folder)
            if not os.path.isdir(subject_path) or folder in ['M007', 'M008', 'mask_files']:
                continue
            # TubeTk Dataset
            if self.data_name == 'TUBETK':
                modalities = os.listdir(subject_path)
                mra_path = os.path.join(subject_path, 'MRA')
                mra_path = os.path.join(mra_path, os.listdir(mra_path)[0])
                if 'AuxillaryData' in modalities:
                    if self.split == 'train':
                        continue
                    else:
                        label_path = os.path.join(subject_path, 'gt_{}.npy'.format(folder))
                        mask_path = os.path.join(subject_path, 'mra_mask.mha')
                        image_files.append(mra_path)
                        label_files.append(label_path)
                        mask_files.append(mask_path)
                if self.split == 'train':
                    image_files.append(mra_path)
            # Vessel12 Dataset
            elif self.data_name == 'VESSEL12':
                ct_path = os.path.join(subject_path, '{}.mhd'.format(folder))
                mask_path = os.path.join(subject_path, 'mask_{}.mhd'.format(folder))
                label_path = os.path.join(subject_path, '{}_Annotations.csv'.format(folder))
                if label_path in os.listdir(subject_path):
                    label_files.append(label_path)
                image_files.append(ct_path)
                mask_files.append(mask_path)
            # 7-tesla Dataset
            elif self.data_name == '7T':
                mra_path = os.path.join(subject_path, '{}_TOF.nii.gz'.format(folder))
                mask_path = os.path.join(subject_path, '{}_TOF_MASKED.nii.gz'.format(folder))
                image_files.append(mra_path)
                mask_files.append(mask_path)
            elif self.data_name == 'LSA':
                subject_num = int(folder.split('_')[-1])
                image_path = os.path.join(subject_path, 'T1SPC_NLM03_{}.nii'.format(folder))
                mask_path = os.path.join(subject_path, 'mask_{}.nii'.format(folder))
                label_path = os.path.join(subject_path, 'label_{}.nii'.format(folder))
                if self.split == 'train' and subject_num % 4 != 1:
                    image_files.append(image_path)
                    mask_files.append(mask_path)
                    label_files.append(label_path)
                elif self.split != 'train' and subject_num % 4 == 1:
                    image_files.append(image_path)
                    mask_files.append(mask_path)
                    label_files.append(label_path)
            else:
                raise ValueError('Dataset Name Not Found!')
        return image_files, mask_files, label_files

    def load_images(self):
        """
        read all the image files, mask files and label files
        :return: image_dict, image dict list
        """
        image_dict = {}
        for i in tqdm(range(len(self.image_files))):
            image_path = self.image_files[i]
            image_file = SiTk.ReadImage(image_path)
            image = SiTk.GetArrayFromImage(image_file)
            patch_num_dim = np.ceil((np.array(image.shape) - self.patch_size) / self.spacing + 1).astype(np.int16)
            patch_total_num = np.prod(patch_num_dim)
            self.total_patch_num += patch_total_num
            curr_image = {
                'image': self.normalize(image),
                'num_dim': patch_num_dim,
                'total_num': patch_total_num
            }
            if len(self.mask_files) != 0:
                mask_path = self.mask_files[i]
                mask_file = SiTk.ReadImage(mask_path)
                mask = SiTk.GetArrayFromImage(mask_file)
                # valid_poses = np.argwhere(mask > 0)
                # min_x, min_y, min_z = np.clip(np.min(valid_poses, axis=0) - 8, a_min=0, a_max=None)
                # max_x, max_y, max_z = np.max(valid_poses, axis=0) + 8
                # mask = mask[min_x:max_x, min_y:max_y, min_z:max_z]
                # image = image[min_x:max_x, min_y:max_y, min_z:max_z]
                assert image.shape == mask.shape
                curr_image['mask'] = mask
            if len(self.label_files) != 0:
                label_path = self.label_files[i]
                if self.data_name == 'TUBETK':
                    label = np.load(label_path)
                elif self.data_name == 'VESSEL12':
                    label = self.read_csv(label_path)
                elif self.data_name == 'LSA':
                    label_file = SiTk.ReadImage(label_path)
                    label = SiTk.GetArrayFromImage(label_file)
                else:
                    label = None
                assert label is not None
                curr_image['label'] = label
            image_dict[image_path] = curr_image
        return image_dict

    def crop_image_patch(self, image, start_coord):
        """
        get the image patch based on the patch index
        :param image: image numpy array
        :param start_coord: location of the start pixel of patch
        :return: img_patch, image patch array
        """
        h, w, d = start_coord
        img_patch = image[h:h + self.patch_size, w:w + self.patch_size, d:d + self.patch_size]
        return img_patch

    def normalize(self, image):
        if self.data_name == 'VESSEL12':
            max_val, min_val = min(255, image.max()), max(-900, image.min())
            image = np.clip(image, a_max=max_val, a_min=min_val)
        elif self.data_name == 'LSA':
            max_val, min_val = min(255, image.max()), max(0, image.min())
            image = np.clip(image, a_max=max_val, a_min=min_val)
        else:
            max_val, min_val = image.max(), image.min()
        return 2.0 * (image - min_val) / (max_val - min_val) - 1.0

    @staticmethod
    def read_csv(csv_file):
        """
        read cvs label files of VESSEL12 as a dictionary
        :param csv_file: .csv file label
        :return:label, label location dictionary
        """
        labels = {}
        lines = open(csv_file, "r").readlines()
        for line in lines:
            strs = line.split(',')
            x, y, z = int(strs[0]), int(strs[1]), int(strs[2])
            label = int(strs[3][0])
            labels[(z, y, x)] = label
        return labels

    @staticmethod
    def get_random_angle(max_angle, p):
        if np.random.uniform() < p:
            angles = np.random.uniform(-max_angle, max_angle, size=3)
        else:
            angles = np.zeros(3)
        return angles


def get_data_loader_3d(data_name='TUBETK', split='train', batch_size=1, shuffle=True):
    """
    get 3D data loaders for different datasets
    :param data_name: Dataset name
    :param split: Dataset name
    :param batch_size: Dataloader batch size
    :param shuffle: Shuffle or not
    :return: dataloader, 2D specified dataloader
    """
    data_dir = dataset_paths[data_name]
    dataset = Dataset3D(data_dir, data_name=data_name, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader


if __name__ == '__main__':
    # define the datasets for unit test
    # tubetk_train = Dataset3D(dataset_paths['TUBETK'], data_name='TUBETK', split='train')
    # tubetk_test = Dataset3D(dataset_paths['TUBETK'], data_name='TUBETK', split='test')
    # print('TubeTK: Train set size: {}; Test set size: {} '.format(len(tubetk_train), len(tubetk_test)))
    # print('TubeTK Patch Size is {}'.format(tubetk_train[0]['image'].shape))
    #
    # vessel12_train = Dataset3D(dataset_paths['VESSEL12'], data_name='VESSEL12', split='train')
    # vessel12_test = Dataset3D(dataset_paths['VESSEL12'], data_name='VESSEL12', split='test')
    # print('VESSEL12: Train set size: {}; Test set size: {} '.format(len(vessel12_train), len(vessel12_test)))
    # print('VESSEL12 Patch Size is {}'.format(vessel12_train[0]['image'].shape))
    #
    # tof_7t_train = Dataset3D(dataset_paths['7T'], data_name='7T', split='train')
    # tof_7t_test = Dataset3D(dataset_paths['7T'], data_name='7T', split='test')
    # print('7T: Train set size: {}; Test set size: {} '.format(len(tof_7t_train), len(tof_7t_test)))
    # print('7T Patch Size is {}'.format(tof_7t_train[0]['image'].shape))

    lsa_train = Dataset3D(dataset_paths['LSA'], data_name='LSA', split='train')
    lsa_test = Dataset3D(dataset_paths['LSA'], data_name='LSA', split='test')
    print('LSA: Train set size: {}; Test set size: {} '.format(len(lsa_train), len(lsa_test)))
    print('LSA Patch Size is {}'.format(lsa_train[0]['image'].shape))
