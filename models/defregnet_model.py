import torch
import torch.nn as nn
import torch.nn.functional as F

from basic_nets.spatial_transform import SpatialTransformation
from basic_nets.unet import UNet
from basic_nets.utils import init_weights
from basic_nets.diffeomorphic import Diffeomorphic


class DefRegNet(nn.Module):
    """
    Network for image registration: outputs the deformation
    field and the affine matrix.
    """

    def __init__(self, in_channels, image_size=128, device='cpu',
                 use_theta: bool = True, use_diffeomorphic: bool = False):
        super(DefRegNet, self).__init__()

        #################################
        self.use_theta = use_theta
        self.use_diffeomorphic = use_diffeomorphic
        if self.use_diffeomorphic:
            self.diffeomorphic = Diffeomorphic(image_size=(image_size, image_size),
                                               scaling=2, dtype=torch.float32, device=device)
        if self.use_theta:
            self.localization = nn.Sequential(
                nn.Conv2d(in_channels, 6, kernel_size=(7, 7)),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(6, 10, kernel_size=(5, 5)),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
            )
            out_image_size = int(image_size / 4) - 4
            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = nn.Sequential(
                nn.Linear(out_image_size * out_image_size * 10, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
            )

            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            self.localization = self.localization.to(device)
            self.fc_loc = self.fc_loc.to(device)
        else:
            self.localization = None
            self.fc_loc = None
        ####################################################################

        self.unet = UNet(in_channels, 2)
        self.spatial_transform = SpatialTransformation(device=device)
        self.unet.apply(init_weights)

        self.unet = self.unet.to(device)
        # self.spatial_transform = self.spatial_transform.to(device)
        self.spatial_transform.device = device

        
    @property
    def device(self):
        return next(self.parameters()).device
    
    def to(self, device):
        super().to(device)
        if self.use_theta:
            self.localization = self.localization.to(device)
            self.fc_loc = self.fc_loc.to(device)
        self.unet = self.unet.to(device)
        self.spatial_transform.device = device
        return self

    def stn(self, x, y):
        """Spatial trasformer network."""
        z = torch.cat([y, x], dim=1)
        xs = self.localization(z)
        h, w = xs.shape[-2:]
        xs = xs.view(-1, h * w * 10)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, y.size())
        y = F.grid_sample(y, grid)

        return y, theta

    def forward(self, batch_moving, batch_fixed):
        if self.use_theta:
            batch_affine_moving, theta = self.stn(batch_fixed, batch_moving)
        else:
            batch_affine_moving = batch_moving
        # diff = batch_fixed - batch_moving
        x = torch.cat([batch_affine_moving, batch_fixed], dim=1)
        batch_deformation = self.unet(x)

        if self.use_diffeomorphic:
            batch_deformation = self.diffeomorphic.calculate(batch_deformation)

        batch_registered = self.spatial_transform(batch_affine_moving,
                                                  batch_deformation)
        # print('batch_deformation', batch_deformation.min(), batch_deformation.max())
        # print('batch_registered', batch_registered.min(), batch_registered.max())

        output = {'batch_registered': batch_registered,
                  'batch_deformation': batch_deformation,
                 }

        if self.use_theta:
            output.update({'theta': theta, 'affine_trf_registered': batch_affine_moving})

        nan_flag = False
        items = list(output.keys())
        for item in items:
            if torch.isnan(output[item]).any():
                print('batch_moving', batch_moving.min(), batch_moving.max())
                print('batch_fixed', batch_fixed.min(), batch_fixed.max())
                print()
                nans = False
                for param in self.unet.parameters():
                    nans += torch.isnan(param).any()
                print('unet', nans)
                
                nans = False
                for param in self.spatial_transform.parameters():
                    nans += torch.isnan(param).any()
                print('spatial_transform', nans)
                
                nans = False
                for param in self.localization.parameters():
                    nans += torch.isnan(param).any()
                print('localization', nans)
                
                nans = False
                for param in self.fc_loc.parameters():
                    nans += torch.isnan(param).any()
                print('fc_loc', nans)
                print()
                nan_flag = True
        output['nans'] = nan_flag

        return output

    def forward_seq(self, batch_sequence):
        final_batch_deformation = None
        batch_moving = batch_sequence[:, -1]
        for i in range(1, len(batch_sequence[0])):
            batch_fixed = batch_sequence[:, i-1]
            batch_moving = batch_sequence[:, i]
            # if self.use_theta:
            #     batch_affine_moving, theta = self.stn(batch_fixed, batch_moving)
            # else:
            #     batch_affine_moving = batch_moving

            x = torch.cat([batch_moving, batch_fixed], dim=1)
            batch_deformation = self.unet(x)
            if final_batch_deformation is None:
                final_batch_deformation = batch_deformation
            else:
                final_batch_deformation += self.spatial_transform(
                    batch_deformation, final_batch_deformation)
        batch_registered = self.spatial_transform(batch_moving, final_batch_deformation)

        output = {'batch_registered': batch_registered,
                  'batch_deformation': final_batch_deformation,
                  }

        # if self.use_theta:
        #     output.update({'theta': theta, 'affine_trf_registered': batch_affine_moving})

        return output
