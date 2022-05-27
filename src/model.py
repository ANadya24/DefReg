import torch
import torch.nn as nn
import torch.nn.functional as F


from DefReg.basic_nets.spatial_transform import SpatialTransformation
from DefReg.basic_nets.unet import UNet


class DefRegNet(nn.Module):
    """
    Network for image registration: outputs the deformation
    field and the affine matrix.
    """
    def __init__(self, in_channels, image_size=128, device='cpu'):
        super(DefRegNet, self).__init__()

        #################################
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(6, 10, kernel_size=5),
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
        ####################################################################

        self.unet = UNet(in_channels, 2)
        self.spatial_transform = SpatialTransformation(device=device)
        self.unet.apply(init_weights)

        self.localization = self.localization.to(device)
        self.fc_loc = self.fc_loc.to(device)
        self.unet = self.unet.to(device)
        self.spatial_transform = self.spatial_transform.to(device)

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

    def forward(self, moving_image, fixed_image):
        moving_image, theta = self.stn(fixed_image, moving_image)
        diff = fixed_image - moving_image
        x = torch.cat([moving_image, fixed_image], dim=1)
        deformation_matrix = self.unet(x)
        registered_image = self.spatial_transform(moving_image,
                                                  deformation_matrix)

        return registered_image, deformation_matrix, theta, diff
