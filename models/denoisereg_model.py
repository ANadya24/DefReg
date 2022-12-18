import torch
import torch.nn as nn
import torch.nn.functional as F

from basic_nets.unet import UNet
from models.defregnet_model import DefRegNet


class DenoiseRegNet(nn.Module):
    """
    DenoiseReg: UNSUPERVISED JOINT DENOISING AND REGISTRATION OF TIME-LAPSE
    LIVE CELL MICROSCOPY IMAGES USING DEEP LEARNING
    """

    def __init__(self, in_channels=1, image_size=256, device='cpu'):
        super(DenoiseRegNet, self).__init__()

        ####Denoising part####
        self.denoise_unet = UNet(in_channels, 1, inter_channel=(16, 32, 64))
        self.denoise_global = nn.AdaptiveAvgPool2d((1, 1))

        ####Registration part####
        
        self.reg_fc = nn.Sequential(
            nn.Linear(64 * 2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2),
            nn.BatchNorm1d(6),
            nn.ReLU(True),
        )

        self.reg_fc[3].weight.data.zero_()
        self.reg_fc[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.denoise_unet.to(device)
        self.denoise_global.to(device)
        self.reg_fc.to(device)

    def add_noise(self, input_image):
        shape = input_image.shape
        size = 1
        for i in shape:
            size *= i
        mask = torch.zeros(size, requires_grad=False)
        idxs = torch.randint(0, size - 1, size=(int(0.5 * size),))
        mask[idxs] = 1
        mask = mask.reshape(*shape).to(input_image.device)
        noise_image = input_image * mask
        return noise_image, 1 - mask

    def run_denoising(self, input_image):
        encode_block1 = self.denoise_unet.conv_encode1(input_image)
        encode_pool1 = self.denoise_unet.conv_maxpool1(encode_block1)
        encode_block2 = self.denoise_unet.conv_encode2(encode_pool1)
        encode_pool2 = self.denoise_unet.conv_maxpool2(encode_block2)
        encode_block3 = self.denoise_unet.conv_encode3(encode_pool2)
        encode_pool3 = self.denoise_unet.conv_maxpool3(encode_block3)
        # Features
        features = self.denoise_global(encode_pool3)
        # Bottleneck
        bottleneck1 = self.denoise_unet.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.denoise_unet.crop_and_concat(bottleneck1, encode_block3)
        cat_layer2 = self.denoise_unet.conv_decode3(decode_block3)
        decode_block2 = self.denoise_unet.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.denoise_unet.conv_decode2(decode_block2)
        decode_block1 = self.denoise_unet.crop_and_concat(cat_layer1, encode_block1)
        final_layer = self.denoise_unet.final_layer(decode_block1)
        return final_layer, features

    def forward(self, fixed_image, moving_image):

        noise_fixed_image, fixed_mask = self.add_noise(fixed_image)
        noise_moving_image, moving_mask = self.add_noise(moving_image)

        denoised_moving, moving_features = self.run_denoising(noise_moving_image)
        denoised_fixed, fixed_features = self.run_denoising(noise_fixed_image)

        pair_fm = torch.cat([fixed_features, moving_features], dim=1)
        pair_mf = torch.cat([moving_features, fixed_features], dim=1)
        
        batch_size = fixed_image.size()[0]
        theta_fm = self.reg_fc(pair_fm.reshape(batch_size, -1)).reshape(batch_size, 2, 3)
        grid = F.affine_grid(theta_fm, denoised_moving.size())
        trf_moving_image = F.grid_sample(denoised_moving, grid)

        theta_mf = self.reg_fc(pair_mf.reshape(batch_size, -1)).reshape(batch_size, 2, 3)
        grid = F.affine_grid(theta_mf, denoised_fixed.size())
        trf_fixed_image = F.grid_sample(denoised_fixed, grid)

        output = {'denoised_fixed': denoised_fixed,
                  'denoised_moving': denoised_moving,
                  'theta_moving2fixed': theta_fm,
                  'theta_fixed2moving': theta_mf,
                  'affine_fixed_image': trf_fixed_image,
                  'affine_moving_image': trf_moving_image,
                  'pixel_mask_fixed': fixed_mask,
                  'pixel_mask_moving': moving_mask}
        return output


class DenoiseDefRegNet(nn.Module):
    """
    DenoiseReg + DefRegNet
    #TODO add symmetric deformation application
    """

    def __init__(self, in_channels, image_size=128, device='cpu',
                 use_theta: bool = True):
        super(DenoiseDefRegNet, self).__init__()

        self.denoise_reg_net = DenoiseRegNet(in_channels=in_channels)
        self.def_reg_net = DefRegNet(in_channels=in_channels, 
                                     image_size=image_size,
                                     device=device,
                                     use_theta=use_theta)

    def forward(self, fixed_image, moving_image):
        output = self.denoise_reg_net(fixed_image, moving_image)
        output.update(self.def_reg_net(output['denoised_fixed'],
                                       output['trf_moving_image']))

        return output
