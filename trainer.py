# -*- coding = utf-8 -*-
# @File Name : trainer
# @Date : 1/4/23 12:38 AM
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from datetime import date
from collections import OrderedDict
from torch.nn.parallel import DataParallel
from logger import Logger
from networks.unet_2d import UNet2D, LocalContrastNet2D
from networks.unet_3d import UNet3D, LocalContrastNet3D
from loss.loss_func import vessel_loss, supervised_loss


class Trainer(object):
    def __init__(self, config_file=None, train_loader=None, valid_loader=None, configer=None):
        self.date = date.today()
        # set up configer of the trainer
        if config_file is not None:
            self.configer = self.read_json(config_file)
        elif configer is not None:
            self.configer = configer
        else:
            raise ValueError('No Config File or Dict Specified!')
        self.loss_conf = self.configer['loss']
        self.trainer_conf = self.configer['trainer']
        self.logger = self.get_logger()
        self.loss_func = self.get_loss_func()

        # set the dataloaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # get device attributes
        self.device_num = self.trainer_conf['gpu_device_num']

        # get model related attributes
        self.model = self.get_model().cuda()
        if self.device_num > 1:
            self.model = DataParallel(self.model, device_ids=list(range(self.device_num)))
        self.opt = self.get_optimizer()
        self.lr_scheduler = self.get_scheduler()

        # resume training
        if self.trainer_conf['resume']:
            self.resume_checkpoint()

    def train_epoch(self, epoch_idx):
        """
        train one epoch for the dataset
        :param epoch_idx: the index of the epoch
        :return: average_losses, average losses
        """
        n_samples = 0
        total_batch_loss = 0.0
        flux_batch_loss = 0.0
        dirs_batch_loss = 0.0
        ints_batch_loss = 0.0
        rcon_batch_loss = 0.0
        attn_batch_loss = 0.0
        augment_batch_loss = 0.0

        # start to training
        self.model.train()
        for idx, batch in enumerate(tqdm(self.train_loader, desc=str(epoch_idx), unit='b')):
            images = batch['image'].cuda()
            output = self.model(images)

            # losses from loss function
            if self.trainer_conf['supervision']:
                labels = batch['label'].cuda()
                total_loss = self.loss_func(labels, output)
                batch_losses = {'total_loss': total_loss}
            else:
                batch_losses = self.loss_func(images, output, self.loss_conf)

            # step optimizer
            if not torch.isnan(batch_losses['total_loss']):
                self.opt.zero_grad()
                batch_losses['total_loss'].backward()
                self.opt.step()
            else:
                raise ValueError('Loss Value Explosion!!!')

            # accumulate losses through batches
            curr_batch_len = images.size(0)
            n_samples += curr_batch_len
            total_batch_loss += batch_losses['total_loss'].detach().item() * curr_batch_len
            if 'flux_loss' in batch_losses.keys():
                flux_batch_loss += batch_losses['flux_loss'].detach().item() * curr_batch_len
            if 'dirs_loss' in batch_losses.keys():
                dirs_batch_loss += batch_losses['dirs_loss'].detach().item() * curr_batch_len
            if 'ints_loss' in batch_losses.keys():
                ints_batch_loss += batch_losses['ints_loss'].detach().item() * curr_batch_len
            if 'rcon_loss' in batch_losses.keys():
                rcon_batch_loss += batch_losses['rcon_loss'].detach().item() * curr_batch_len
            if 'attn_loss' in batch_losses.keys():
                attn_batch_loss += batch_losses['attn_loss'].detach().item() * curr_batch_len
            if 'augment_loss' in batch_losses.keys():
                augment_batch_loss += batch_losses['augment_loss'].detach().item() * curr_batch_len

        # learning rate decrease
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # get the loss dicts
        average_losses = {'avg_total_loss': total_batch_loss / n_samples}
        if flux_batch_loss != 0:
            average_losses['avg_flux_loss'] = flux_batch_loss / n_samples
        if dirs_batch_loss != 0:
            average_losses['avg_dirs_loss'] = dirs_batch_loss / n_samples
        if ints_batch_loss != 0:
            average_losses['avg_ints_loss'] = ints_batch_loss / n_samples
        if rcon_batch_loss != 0:
            average_losses['avg_rcon_loss'] = rcon_batch_loss / n_samples
        if attn_batch_loss != 0:
            average_losses['avg_attn_loss'] = attn_batch_loss / n_samples
        if augment_batch_loss != 0:
            average_losses['avg_augment_loss'] = augment_batch_loss / n_samples
        return average_losses

    def valid_epoch(self, epoch_idx):
        """
        evaluate one epoch for the valid dataloader
        :param epoch_idx: the index of the epoch
        :return: average_total_loss, average total loss to monitor
        """
        pass

    def train(self):
        """
        train process
        :return: None
        """
        if self.train_loader is None:
            raise ValueError('Training Dataloader is Not Specified.')
        # clear all the content of the logger file
        self.logger.flush()
        # log the config file of current running
        self.logger.write_dict(self.configer)

        # start training
        epoch_num = self.trainer_conf['epoch_num']
        for epoch in range(1, epoch_num + 1):
            # train current epoch
            losses = self.train_epoch(epoch)
            # separate the epoch using one line '*'
            self.logger.write_block(1)
            self.logger.write('EPOCH: {}'.format(str(epoch)))
            # log the losses to the file
            for key, value in losses.items():
                message = '{}: {}'.format(str(key), value)
                self.logger.write(message)

            # save as period
            if epoch % self.trainer_conf['save_period'] == 0:
                self.save_checkpoint(epoch, losses)
            self.logger.write_block(2)

    def save_checkpoint(self, epoch_idx, losses):
        """
        save trained model check point
        :param epoch_idx: the index of the epoch
        :param losses: losses log
        :return: None
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch_idx,
            'configer': self.configer,
            'model': self.model.module.state_dict(),
            'optimizer': self.opt.state_dict(),
            'current_loss': losses['avg_total_loss']
        }
        chkpt_dir = os.path.join(self.trainer_conf['checkpoint_dir'], self.configer['name'], str(self.date))
        os.makedirs(chkpt_dir, exist_ok=True)
        filename = os.path.join(chkpt_dir, '{}-epoch-{}.pth'.format(epoch_idx, str(self.date)))
        self.logger.write("Saving checkpoint at: {} ...".format(filename))
        torch.save(state, filename)

    def resume_checkpoint(self):
        """
        resume the trained model check point
        :return:
        """
        self.logger.write("Loading checkpoint: {} ...".format(self.trainer_conf['resume_path']))
        checkpoint = torch.load(self.trainer_conf['resume_path'])

        # load state dicts
        if checkpoint['configer']['arch'] != self.configer['arch']:
            raise ValueError('Checkpoint Architecture Does Not Match to The Config File.')
        self.model.module.load_state_dict(checkpoint['model'])

        # load optimizer dicts
        if not self.configer['trainer']['supervision']:
            if checkpoint['configer']['optimizer']['type'] != self.configer['optimizer']['type']:
                raise ValueError('Checkpoint Optimizer Does Not Match to The Config File.')
            self.opt.load_state_dict(checkpoint['optimizer'])
            print('Optimizer resumed from before.')

        # if initial supervised learning
        if self.configer['trainer']['supervision'] and not self.configer['trainer']['fine-tune']:
            try:
                self.model.reinit_last_layers()
                self.logger.write('Last Layers Re-initialized!')
                self.model.freeze_u_net()
                self.logger.write('U-Net Layers are Frozen!')
            except AttributeError:
                print('Current Model Does Not Support Supervision Training! Please check codes.')
        self.logger.write("Resume training from epoch {}".format(checkpoint['epoch']))

    def get_model(self):
        """
        get the model from configer
        :return: model, torch model architecture
        """
        model_type = self.configer['arch']['type']
        input_ch = self.configer['arch']['in_channels']
        output_ch = self.configer['arch']['out_channels']
        min_r_scale = self.configer['arch']['min_scale']
        max_r_scale = self.configer['arch']['max_scale']
        radius_sample_num = self.configer['arch']['radius_num']
        feature_dims = self.configer['arch']['feature_dims']

        # define architectures based on model's type
        if model_type == 'UNet2D':
            model = UNet2D(input_ch, output_ch, min_r_scale, max_r_scale, radius_sample_num, feature_dims)
        elif model_type == 'LocalContrastNet2D':
            model = LocalContrastNet2D(input_ch, output_ch, min_r_scale, max_r_scale, radius_sample_num, feature_dims)
        elif model_type == 'UNet3D':
            model = UNet3D(input_ch, output_ch, min_r_scale, max_r_scale, radius_sample_num, feature_dims)
        elif model_type == 'LocalContrastNet3D':
            model = LocalContrastNet3D(input_ch, output_ch, min_r_scale, max_r_scale, radius_sample_num, feature_dims)
        else:
            raise ValueError('Model Type Not Found.')
        return model

    def get_optimizer(self):
        """
        get the optimizer from configer
        :return: optimizer, torch optimizer based on model architecture
        """
        opt_type = self.configer['optimizer']['type']
        lr = self.configer['optimizer']['learning_rate']
        decay = self.configer['optimizer']['weight_decay']
        if opt_type == 'Adam':
            amsgrad = self.configer['optimizer']['amsgrad']
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay, amsgrad=amsgrad)
        elif opt_type == 'SGD':
            momentum = self.configer['optimizer']['momentum']
            optimizer = optim.SGD(self.model.parameters(), momentum=momentum, lr=lr, weight_decay=decay)
        else:
            raise ValueError('Optimizer Type Not Found.')
        return optimizer

    def get_loss_func(self):
        """
        determine which loss function that used in training process
        :return: loss_func, loss function of vessel function
        """
        if self.configer['trainer']['supervision']:
            loss_func = supervised_loss
        else:
            loss_func = vessel_loss
        return loss_func

    def get_logger(self):
        """
        get the customized logger
        :return: logger, logger object
        """
        log_dir = os.path.join(self.trainer_conf['checkpoint_dir'], 'loggers', self.configer['name'])
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, self.trainer_conf['train_type'] + '-' + str(self.date) + '.txt')
        logger = Logger(log_filename)
        return logger

    def get_scheduler(self):
        """
        get the scheduler of the optimizer
        :return: scheduler, optimizer scheduler
        """
        sche_type = self.configer['lr_scheduler']['type']
        if sche_type == 'StepLR':
            step_size = self.configer['lr_scheduler']['step_size']
            gamma = self.configer['lr_scheduler']['gamma']
            scheduler = optim.lr_scheduler.StepLR(self.opt, step_size=step_size, gamma=gamma)
        elif sche_type == 'MultiStepLR':
            milestones = self.configer['lr_scheduler']['milestones']
            gamma = self.configer['lr_scheduler']['gamma']
            scheduler = optim.lr_scheduler.MultiStepLR(self.opt, milestones=milestones, gamma=gamma)
        else:
            raise ValueError('Scheduler Type Not Found.')
        return scheduler

    @staticmethod
    def read_json(config_file):
        """
        read the json file to config the training and dataset
        :param config_file: config file path
        :return: dictionary of config keys
        """
        config_file = Path(config_file)
        with config_file.open('rt') as handle:
            return json.load(handle, object_hook=OrderedDict)


if __name__ == '__main__':
    # only for unit test
    from datasets.datasets_2d import get_data_loader_2d
    from datasets.datasets_3d import get_data_loader_3d

    # 2D Dataloader
    train_loader_2d = get_data_loader_2d(data_name='HRF', split='train')
    trainer1_2d = Trainer('./config/hrf/hrf_adaptive_with_lc.json', train_loader_2d)
    trainer2_2d = Trainer('./config/hrf/hrf_adaptive_with_lc_supervised.json', train_loader_2d)
    trainer3_2d = Trainer('./config/hrf/hrf_adaptive_without_lc.json', train_loader_2d)
    trainer4_2d = Trainer('./config/hrf/hrf_naive_with_lc.json', train_loader_2d)
    trainer5_2d = Trainer('./config/hrf/hrf_naive_without_lc.json', train_loader_2d)
    trainer6_2d = Trainer('./config/hrf/hrf_naive_without_lc_sym.json', train_loader_2d)

    # 3D Dataloader
    train_loader_3d = get_data_loader_3d(data_name='7T', split='train')
    trainer1_3d = Trainer('./config/7T/seven_tesla_adaptive_with_lc.json', train_loader_3d)
    trainer2_3d = Trainer('./config/7T/seven_tesla_adaptive_with_lc_supervised.json', train_loader_3d)
    trainer3_3d = Trainer('./config/7T/seven_tesla_adaptive_without_lc.json', train_loader_3d)
    trainer4_3d = Trainer('./config/7T/seven_tesla_naive_with_lc.json', train_loader_3d)
    trainer5_3d = Trainer('./config/7T/seven_tesla_naive_without_lc.json', train_loader_3d)
    trainer6_3d = Trainer('./config/7T/seven_tesla_naive_without_lc_sym.json', train_loader_3d)
