# -*- coding = utf-8 -*-
# @File Name : unet_3d
# @Date : 2022/10/6 15:57
# @Author : dengzhiwei
# @E-mail : zhiweide@usc.edu


from torch.nn import functional as F
from networks.unet_3d_parts import *
from networks.unet_utils import Initializer


class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, min_scale=0.1, max_scale=10.5, radius_num=128,
                 feat_dims=None, size=3, pool_size=2, stride=2, pad=1, out_pad=1):
        super(UNet3D, self).__init__()
        self.min_scale, self.max_scale, self.out_ch = min_scale, max_scale, out_ch
        # encoder part
        self.encoder1 = SingleEncoder3d(in_ch, feat_dims[0], size, pool_size, pad, apply_pool=False)
        self.encoder2 = SingleEncoder3d(feat_dims[0], feat_dims[1], size, pool_size, pad, apply_pool=True)
        self.encoder3 = SingleEncoder3d(feat_dims[1], feat_dims[2], size, pool_size, pad, apply_pool=True)
        self.encoder4 = SingleEncoder3d(feat_dims[2], feat_dims[3], size, pool_size, pad, apply_pool=True)
        # decoder part
        self.decoder1 = SingleDecoder3d(feat_dims[3] + feat_dims[2], feat_dims[2], size, stride, pad, out_pad)
        self.decoder2 = SingleDecoder3d(feat_dims[2] + feat_dims[1], feat_dims[1], size, stride, pad, out_pad)
        self.decoder3 = SingleDecoder3d(feat_dims[1] + feat_dims[0], feat_dims[0], size, stride, pad, out_pad)
        # final conv
        self.recon_conv = nn.Conv3d(feat_dims[0], in_ch, 1)
        self.direction_conv = nn.Conv3d(feat_dims[0], out_ch, 1)
        self.radius_conv = nn.Conv3d(feat_dims[0], radius_num, 1)
        Initializer.weights_init(self)

    def forward(self, im):
        x1 = self.encoder1(im)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        x = self.decoder1(x3, x4)
        x = self.decoder2(x2, x)
        x = self.decoder3(x1, x)

        recon = self.recon_conv(x)
        direction = self.direction_conv(x)
        radius = torch.sigmoid(self.radius_conv(x)) * self.max_scale + self.min_scale
        output = {
            'vessel': direction,
            'radius': radius,
            'recon': recon
        }
        return output


class LocalContrastNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, min_scale=0.1, max_scale=10.5, radius_num=128,
                 feat_dims=None, size=3, pool_size=2, stride=2, pad=1, out_pad=1):
        super(LocalContrastNet3D, self).__init__()
        if feat_dims is None:
            feat_dims = [64, 128, 256, 512]
        self.min_scale, self.max_scale = min_scale, max_scale
        # encoder part
        self.encoder1 = SingleEncoder3d(in_ch, feat_dims[0], size, pool_size, pad, apply_pool=False)
        self.encoder2 = SingleEncoder3d(feat_dims[0], feat_dims[1], size, pool_size, pad, apply_pool=True)
        self.encoder3 = SingleEncoder3d(feat_dims[1], feat_dims[2], size, pool_size, pad, apply_pool=True)
        self.encoder4 = SingleEncoder3d(feat_dims[2], feat_dims[3], size, pool_size, pad, apply_pool=True)
        # decoder part
        self.decoder1 = SingleDecoder3d(feat_dims[3] + feat_dims[2], feat_dims[2], size, stride, pad, out_pad)
        self.decoder2 = SingleDecoder3d(feat_dims[2] + feat_dims[1], feat_dims[1], size, stride, pad, out_pad)
        self.decoder3 = SingleDecoder3d(feat_dims[1] + feat_dims[0], feat_dims[0], size, stride, pad, out_pad)
        # spatial attention part
        self.level_attention1 = SpatialAttention3d(feat_dims[0], min_scale, max_scale, size, pad)
        self.level_attention2 = SpatialAttention3d(feat_dims[1], min_scale, max_scale, size, pad)
        self.level_attention3 = SpatialAttention3d(feat_dims[2], min_scale, max_scale, size, pad)
        self.global_attention = SpatialAttention3d(feat_dims[0], min_scale, max_scale, size, pad, radius_num, 3)
        # final conv
        self.recon_conv = nn.Conv3d(feat_dims[0], in_ch, 1)
        self.direction_conv = nn.Conv3d(feat_dims[0], out_ch, 1)
        self.radius_conv = nn.Conv3d(feat_dims[0], radius_num, 1)
        Initializer.weights_init(self)

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

        # final conv
        recon = self.recon_conv(x)
        direction = self.direction_conv(x)
        radius = torch.sigmoid(self.radius_conv(x)) * self.max_scale + self.min_scale
        attention = self.global_attention(im, x, radius).squeeze(1)
        attentions = [attention1, attention2, attention3, attention]
        output = {
            'recon': recon,
            'radius': radius,
            'vessel': direction,
            'attentions': attentions
        }
        return output


if __name__ == '__main__':
    from torchsummary import summary
    feature_dims = [64, 128, 256, 512]
    """ 3D UNet Test """
    contrast_model = LocalContrastNet3D(1, 3, 0.1, 10.5, 128, feature_dims).cuda(3)
    print(type(contrast_model).__name__)
    summary(contrast_model, input_size=(8, 1, 64, 64, 64))
