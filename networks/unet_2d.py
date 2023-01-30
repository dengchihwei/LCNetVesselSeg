# -*- coding = utf-8 -*-
# @File Name : unet_2d
# @Date : 2022/10/6 15:53
# @Author : dengzhiwei
# @E-mail : zhiweide@usc.edu


from torch.nn import functional as F
from networks.unet_2d_parts import *
from networks.unet_utils import Initializer


class UNet2D(nn.Module):
    def __init__(self, in_ch, out_ch, min_scale, max_scale, radius_num, feat_dims,
                 size=3, pool_size=2, stride=2, pad=1, out_pad=1):
        super(UNet2D, self).__init__()
        self.min_scale, self.max_scale, self.out_ch = min_scale, max_scale, out_ch
        # encoder part
        self.encoder1 = SingleEncoder2d(in_ch, feat_dims[0], size, pool_size, pad, apply_pool=False)
        self.encoder2 = SingleEncoder2d(feat_dims[0], feat_dims[1], size, pool_size, pad, apply_pool=True)
        self.encoder3 = SingleEncoder2d(feat_dims[1], feat_dims[2], size, pool_size, pad, apply_pool=True)
        self.encoder4 = SingleEncoder2d(feat_dims[2], feat_dims[3], size, pool_size, pad, apply_pool=True)
        # decoder part
        self.decoder1 = SingleDecoder2d(feat_dims[3] + feat_dims[2], feat_dims[2], size, stride, pad, out_pad)
        self.decoder2 = SingleDecoder2d(feat_dims[2] + feat_dims[1], feat_dims[1], size, stride, pad, out_pad)
        self.decoder3 = SingleDecoder2d(feat_dims[1] + feat_dims[0], feat_dims[0], size, stride, pad, out_pad)
        # final conv
        self.recon_conv = nn.Conv2d(feat_dims[0], in_ch, 1)
        self.direction_conv = nn.Conv2d(feat_dims[0], out_ch, 1)
        self.radius_conv = nn.Conv2d(feat_dims[0], radius_num, 1)
        Initializer.weights_init(self)

    def forward(self, x):
        x1 = self.encoder1(x)
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


class LocalContrastNet2D(nn.Module):
    def __init__(self, in_ch, out_ch, min_scale, max_scale, radius_num,
                 feat_dims, size=3, pool_size=2, stride=2, pad=1, out_pad=1):
        super(LocalContrastNet2D, self).__init__()
        self.min_scale, self.max_scale = min_scale, max_scale
        # encoder part
        self.encoder1 = SingleEncoder2d(in_ch, feat_dims[0], size, pool_size, pad, apply_pool=False)
        self.encoder2 = SingleEncoder2d(feat_dims[0], feat_dims[1], size, pool_size, pad, apply_pool=True)
        self.encoder3 = SingleEncoder2d(feat_dims[1], feat_dims[2], size, pool_size, pad, apply_pool=True)
        self.encoder4 = SingleEncoder2d(feat_dims[2], feat_dims[3], size, pool_size, pad, apply_pool=True)
        # decoder part
        self.decoder1 = SingleDecoder2d(feat_dims[3] + feat_dims[2], feat_dims[2], size, stride, pad, out_pad)
        self.decoder2 = SingleDecoder2d(feat_dims[2] + feat_dims[1], feat_dims[1], size, stride, pad, out_pad)
        self.decoder3 = SingleDecoder2d(feat_dims[1] + feat_dims[0], feat_dims[0], size, stride, pad, out_pad)
        # spatial attention part
        self.level_attention1 = SpatialAttention2d(feat_dims[0], min_scale, max_scale, size, pad)
        self.level_attention2 = SpatialAttention2d(feat_dims[1], min_scale, max_scale, size, pad)
        self.level_attention3 = SpatialAttention2d(feat_dims[2], min_scale, max_scale, size, pad)
        self.global_attention = SpatialAttention2d(feat_dims[0], min_scale, max_scale, size, pad, radius_num, 5)
        # final conv
        self.recon_conv = nn.Conv2d(feat_dims[0], in_ch, 1)
        self.direction_conv = nn.Conv2d(feat_dims[0], out_ch, 1)
        self.radius_conv = nn.Conv2d(feat_dims[0], radius_num, 1)
        Initializer.weights_init(self)

    def reinit_last_layers(self):
        # Reinit last layers
        nn.init.normal_(self.recon_conv.weight.data)
        # nn.init.normal_(self.direction_conv.weight.data)
        # nn.init.normal_(self.radius_conv.weight.data)

    def freeze_u_net(self):
        # self.freeze_layer(self.encoder1)
        # self.freeze_layer(self.encoder2)
        # self.freeze_layer(self.encoder3)
        # self.freeze_layer(self.encoder4)
        # self.freeze_layer(self.decoder1)
        # self.freeze_layer(self.decoder2)
        # self.freeze_layer(self.decoder3)
        # self.freeze_layer(self.level_attention1)
        # self.freeze_layer(self.level_attention2)
        # self.freeze_layer(self.level_attention3)
        self.freeze_layer(self.global_attention)
        self.freeze_layer(self.direction_conv)
        self.freeze_layer(self.radius_conv)

    @ staticmethod
    def freeze_layer(layer):
        for param in layer.parameters():
            param.requires_grad = False

    def forward(self, im):
        # encoding
        x1 = self.encoder1(im)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        # apply attention
        attention1 = self.level_attention1(im, x1)
        x1 = torch.mul(attention1, x1)
        x = F.interpolate(im, scale_factor=0.5, mode='bilinear')
        attention2 = self.level_attention2(x, x2)
        x2 = torch.mul(attention2, x2)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
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
            'attentions': attentions,
        }
        return output


if __name__ == '__main__':
    from torchsummary import summary
    feature_dims = [64, 128, 256, 512]
    """ 2D UNet Test """
    vessel_seg_model_2d = UNet2D(1, 2, 0.1, 10.5, 128, feature_dims).cuda(0)
    print(type(vessel_seg_model_2d).__name__)
    summary(vessel_seg_model_2d, input_size=(8, 3, 256, 256))
