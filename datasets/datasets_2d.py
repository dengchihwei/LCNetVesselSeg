# -*- coding = utf-8 -*-
# @File Name : datasets_2d
# @Date : 12/30/22 7:32 PM
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.ndimage.morphology import binary_erosion


DRIVE_SIZE = [584, 565]
STARE_SIZE = [605, 700]
HRF_SIZE = [2336, 3504]
patch_size, spacing = 256, 192
dataset_paths = {
    'DRIVE': '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/DRIVE',
    'STARE': '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/STARE',
    'HRF': '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/HRF'
}


class Dataset2D(Dataset):
    def __init__(self, data_dir, data_name='DRIVE', split='train', augment=True):
        self.split = split
        self.augment = augment
        self.data_name = data_name

        # get image size for this dataset
        self.dims = self.get_image_dims()

        # get patch num of each image
        self.patch_num_dim = np.ceil((self.dims - patch_size) / spacing + 1)
        self.patch_nums = np.prod(self.patch_num_dim.astype(np.int16))
        print("{} patches per image.".format(self.patch_nums))

        # get data directory based on different datasets
        self.data_dir = self.get_data_dir(data_dir)
        assert self.data_dir is not None

        # get paths of images, labels and masks
        self.image_path, self.label_path, self.mask_path = self.get_all_paths()
        self.image_files = [os.path.join(self.image_path, file) for file in sorted(os.listdir(self.image_path))]
        self.label_files = [os.path.join(self.label_path, file) for file in sorted(os.listdir(self.label_path))]
        self.mask_files = [os.path.join(self.mask_path, file) for file in sorted(os.listdir(self.mask_path))]

        # split the dataset
        self.split_data()

        # load images, labels and masks
        self.images, self.labels, self.masks = [], [], []
        for i in tqdm(range(len(self.image_files))):
            image = np.asarray(Image.open(self.image_files[i]))[..., 1]
            label = np.asarray(Image.open(self.label_files[i]))
            mask = np.asarray(Image.open(self.mask_files[i]).convert('L'))
            # normalization
            image = (image - image.min()) / (image.max() - image.min())
            mask = np.array(mask > 0.5).astype(float)
            mask = binary_erosion(mask, np.ones((7, 7)))
            self.images.append(image)
            self.labels.append(label)
            self.masks.append(mask)

    def __len__(self):
        return len(self.images) * self.patch_nums

    def __getitem__(self, index):
        # get image and label
        image_idx = index // self.patch_nums
        image = 1.0 - self.images[image_idx]
        label = self.labels[image_idx]
        mask = self.masks[image_idx]

        # get image, label and mask patch
        patch_idx = index % self.patch_nums
        start_coord = self.get_start_coord(patch_idx)
        image_patch = self.crop_image_patch(image, start_coord)
        label_patch = self.crop_image_patch(label, start_coord)
        mask_patch = self.crop_image_patch(mask, start_coord)
        # apply the image mask
        image_patch = np.multiply(image_patch, mask_patch)

        # image augmentation
        if self.augment:
            image_patch, label_patch = self.flip(image_patch, label_patch)
            image_patch, label_patch = self.rotate(image_patch, label_patch)
            # image_patch = self.add_gaussian_noise(image_patch)

        # convert to torch types
        image_patch = torch.from_numpy(image_patch.copy()).unsqueeze(0)
        label_patch = torch.from_numpy(label_patch.copy()).unsqueeze(0)
        start_coord = torch.LongTensor(start_coord)
        item = {
            'image_id': image_idx,
            'image': image_patch.float(),
            'start_coord': start_coord,
            'label': label_patch.float()
        }
        return item

    def get_data_dir(self, data_dir):
        """
        get dataset directory from the dataset name and split
        :param data_dir: data directory
        :return: data_dir transformed data directory
        """
        if self.data_name == 'DRIVE':
            data_dir = os.path.join(data_dir, self.split)
        elif self.data_name == 'STARE':
            data_dir = data_dir
        elif self.data_name == 'HRF':
            data_dir = data_dir
        else:
            data_dir = None
        return data_dir

    def get_all_paths(self):
        """
        get directories of image, label and masks
        :param: None
        :return: image_path, label_path, mask_path.
                 Paths to images, labels and masks
        """
        if self.data_name == 'DRIVE':
            label_folder = '1st_manual'
        elif self.data_name == 'STARE':
            label_folder = 'labels-ah'
        elif self.data_name == 'HRF':
            label_folder = 'manual1'
        else:
            label_folder = None
        assert label_folder is not None

        # images, labels and masks
        image_path = os.path.join(self.data_dir, 'images')
        label_path = os.path.join(self.data_dir, label_folder)
        mask_path = os.path.join(self.data_dir, 'mask')
        return image_path, label_path, mask_path

    def get_image_dims(self):
        """
        get image sizes based on datasets
        :param: None
        :return: dims, image size
        """
        if self.data_name == 'DRIVE':
            dims = np.array(DRIVE_SIZE)
        elif self.data_name == 'STARE':
            dims = np.array(STARE_SIZE)
        elif self.data_name == 'HRF':
            dims = np.array(HRF_SIZE)
        else:
            dims = None
        assert dims is not None
        return dims

    def split_data(self):
        """
        split the data according to the split string
        :param: None
        :return: None
        """
        if self.data_name == 'DRIVE':
            return
        elif self.data_name == 'STARE':
            self.image_files = self.image_files[:10] if self.split == 'train' else self.image_files[10:]
            self.label_files = self.label_files[:10] if self.split == 'train' else self.label_files[10:]
            self.mask_files = self.mask_files[:10] if self.split == 'train' else self.mask_files[10:]
        elif self.data_name == 'HRF':
            self.image_files = self.image_files[:15] if self.split == 'train' else self.image_files[15:]
            self.label_files = self.label_files[:15] if self.split == 'train' else self.label_files[15:]
            self.mask_files = self.mask_files[:15] if self.split == 'train' else self.mask_files[15:]
        else:
            raise ValueError('Dataset Name Not Found.')

    def get_start_coord(self, patch_idx):
        """
        get the image patch's start pixel position
        :param patch_idx: index of the image patch
        :return: start_coord, location of the start pixel of patch
        """
        start_coord = np.zeros(2)
        for i in range(2):
            start_coord[i] = patch_idx % self.patch_num_dim[i]
            patch_idx = patch_idx // self.patch_num_dim[i]

        # final start coordinate
        start_coord = (spacing * start_coord).astype(np.int16)
        end_coord = start_coord + patch_size

        # in case of exceed the boundaries
        image_shape = self.dims
        for i in range(2):
            if end_coord[i] > image_shape[i]:
                end_coord[i] = image_shape[i]
                start_coord[i] = end_coord[i] - patch_size
        return start_coord

    @staticmethod
    def crop_image_patch(image, start_coord):
        """
        get the image patch based on the patch index
        :param image: image numpy array
        :param start_coord: location of the start pixel of patch
        :return: img_patch, image patch array
        """
        h, w = start_coord
        img_patch = image[h:h+patch_size, w:w+patch_size]
        return img_patch

    @staticmethod
    def flip(img_patch, gt_patch, p=0.3):
        """
        randomly flip the image patch along x and y-axis
        :param img_patch: image patch array
        :param gt_patch: ground truth array
        :param p: probability of applying the flip
        :return: img_patch, flipped image patch array
                 gt_patch, flipped ground truth array
        """
        # flip the image horizontally with prob p
        if np.random.uniform() < p:
            img_patch = np.flip(img_patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)
        # flip the image vertically with prob p
        if np.random.uniform() < p:
            img_patch = np.flip(img_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        return img_patch, gt_patch

    @staticmethod
    def rotate(img_patch, gt_patch, p=0.3):
        """
        randomly rotate the image patch among {0, 90, 180, 270} degrees
        :param img_patch: image patch array
        :param gt_patch: ground truth array
        :param p: probability of applying the rotation
        :return: img_patch, flipped image patch array
                 gt_patch, flipped ground truth array
        """
        if np.random.uniform() < p:
            k = np.random.randint(0, 4)
            img_patch = np.rot90(img_patch, k, axes=(0, 1))
            gt_patch = np.rot90(gt_patch, k, axes=(0, 1))
        return img_patch, gt_patch

    @staticmethod
    def add_gaussian_noise(img_patch, p=0.5):
        """
        add gaussian noises to the image patch
        :param img_patch: image patch array
        :param p: probability of adding the noises
        :return: img_patch, noised added patch
        """
        if np.random.uniform() < p:
            gaussian_noise = np.random.normal(0.01, 0.02, img_patch.shape)
            img_patch = gaussian_noise + img_patch
        return img_patch


def get_data_loader_2d(data_name='DRIVE', split='train', augment=True, batch_size=2, shuffle=True):
    """
    get 2D data loaders for different datasets
    :param data_name: Dataset name
    :param split: Dataset name
    :param augment: Augment or not
    :param batch_size: Dataloader batch size
    :param shuffle: Shuffle or not
    :return: dataloader, 2D specified dataloader
    """
    data_dir = dataset_paths[data_name]
    dataset = Dataset2D(data_dir, data_name=data_name, split=split, augment=augment)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=batch_size)
    return dataloader


if __name__ == '__main__':
    # define the datasets for unit test
    drive_train = Dataset2D(dataset_paths['DRIVE'], data_name='DRIVE', split='train', augment=True)
    drive_test = Dataset2D(dataset_paths['DRIVE'], data_name='DRIVE', split='test', augment=False)
    print('DRIVE: Train set size: {}; Test set size: {} '.format(len(drive_train), len(drive_test)))
    print('DRIVE Patch Size is {}'.format(drive_train[0]['image'].shape))

    stare_train = Dataset2D(dataset_paths['STARE'], data_name='STARE', split='train', augment=True)
    stare_test = Dataset2D(dataset_paths['STARE'], data_name='STARE', split='test', augment=False)
    print('STARE: Train set size: {}; Test set size: {} '.format(len(stare_train), len(stare_test)))
    print('STARE Patch Size is {}'.format(stare_train[0]['image'].shape))

    hrf_train = Dataset2D(dataset_paths['HRF'], data_name='HRF', split='train', augment=True)
    hrf_test = Dataset2D(dataset_paths['HRF'], data_name='HRF', split='test', augment=False)
    print('HRF: Train set size: {}; Test set size: {} '.format(len(hrf_train), len(hrf_test)))
    print('HRF Patch Size is {}'.format(hrf_train[0]['image'].shape))
