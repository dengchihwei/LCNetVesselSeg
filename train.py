# -*- coding = utf-8 -*-
# @File Name : train
# @Date : 1/6/23 12:26 AM
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import argparse
from trainer import Trainer
from datasets.datasets_2d import get_data_loader_2d
from datasets.datasets_3d import get_data_loader_3d

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='2D')
parser.add_argument('--dataset', type=str, default='DRIVE')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--config_file', type=str, default='./config/drive/drive_adaptive_with_lc.json')


if __name__ == '__main__':
    args = parser.parse_args()
    if args.type == '2D':
        data_loader = get_data_loader_2d(data_name=args.dataset,
                                         split='train',
                                         augment=True,
                                         batch_size=args.batch_size,
                                         shuffle=True)
    elif args.type == '3D':
        data_loader = get_data_loader_3d(data_name=args.dataset,
                                         split='train',
                                         batch_size=args.batch_size,
                                         shuffle=True)
    else:
        raise ValueError('Dataset Type {} Not Implemented'.format(args.type))

    # define trainer
    trainer = Trainer(args.config_file, data_loader)
    trainer.train()
