import torch
from torch import nn
import torchx
from torch import functional as F
from torchvision.models import vgg13
from voxelmorph import SpatialTransformer


def conv3x3(in_channels, out_channels, dilation=1):
    return nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()

        self.batch_norm = batch_norm

        self.conv1 = conv3x3(in_channels, out_channels)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        if self.batch_norm:
            self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.relu2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, num_filters, num_blocks):
        super().__init__()

        self.num_blocks = num_blocks
        for i in range(num_blocks):
            in_channels = in_channels if not i else num_filters * 2 ** (i - 1)
            out_channels = num_filters * 2 ** i
            self.add_module(f'block{i + 1}', EncoderBlock(in_channels, out_channels))
            if i != num_blocks - 1:
                self.add_module(f'pool{i + 1}', nn.MaxPool2d(2, 2))

    def forward(self, x):
        acts = []
        for i in range(self.num_blocks):
            x = self.__getattr__(f'block{i + 1}')(x)
            acts.append(x)
            if i != self.num_blocks - 1:
                x = self.__getattr__(f'pool{i + 1}')(x)
        return acts


class VGG13Encoder(nn.Module):
    def __init__(self, num_blocks, pretrained=True):
        super().__init__()

        backbone = vgg13(pretrained=pretrained).features

        self.num_blocks = num_blocks
        for i in range(self.num_blocks):
            block = nn.Sequential(*[backbone[j] for j in range(i * 5, i * 5 + 4)])
            self.add_module(f'block{i + 1}', block)
            if i != num_blocks - 1:
                self.add_module(f'pool{i + 1}', nn.MaxPool2d(2, 2))

    def forward(self, x):
        acts = []
        for i in range(self.num_blocks):
            x = self.__getattr__(f'block{i + 1}')(x)
            acts.append(x)
            if i != self.num_blocks - 1:
                x = self.__getattr__(f'pool{i + 1}')(x)
        return acts


class DecoderBlock(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.uppool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv = conv3x3(out_channels * 2, out_channels)
        self.conv1 = conv3x3(out_channels * 2, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, down, left):
        x = self.uppool(down)
        x = self.upconv(x)
        x = torch.cat([left, x], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_filters, num_blocks):
        super().__init__()

        for i in range(num_blocks):
            self.add_module(f'block{num_blocks - i}', DecoderBlock(num_filters * 2 ** i))

    def forward(self, acts):
        up = acts[-1]
        for i, left in enumerate(acts[-2::-1]):
            up = self.__getattr__(f'block{i + 1}')(up, left)
        return up


class DefNet(nn.Module):
    def __init__(self, vol_size, in_channels=3, num_filters=64, num_blocks=4):
        super().__init__()

        print(f'=> Building {num_blocks}-blocks {num_filters}-filter U-Net')

        # self.encoder = VGG13Encoder(num_blocks)
        self.encoder = Encoder(2, num_filters, num_blocks)
        self.decoder = Decoder(num_filters, num_blocks - 1)
        self.deformation = nn.Conv2d(num_filters, 2, 1)
        self.spatial_transform = SpatialTransformer(vol_size)

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        acts = self.encoder(x)
        x = self.decoder(acts)
        deform = self.deformation(x)
        y = self.spatial_transform(src, deform*10.)
        return y, deform


class Bottleneck(torch.nn.Module):
    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(Bottleneck, self).__init__()

        self.shortcut = stride == 1 and in_channels == out_channels
        self.block = torch.nn.Sequential(
            torchx.nn.Conv2dGroup(
                in_channels=in_channels,
                out_channels=in_channels * t,
                kernel_size=1,
                stride=1,
            ),
            torchx.nn.DWConv(
                in_channels=in_channels * t, out_channels=in_channels * t, stride=stride
            ),
            torch.nn.Conv2d(
                in_channels=in_channels * t,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.shortcut:
            return x + self.block(x)

        else:
            return self.block(x)


class PyramidPooling(torch.nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = torchx.nn.Conv2dGroup(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = torchx.nn.Conv2dGroup(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = torchx.nn.Conv2dGroup(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = torchx.nn.Conv2dGroup(in_channels, inter_channels, 1, **kwargs)
        self.out = torchx.nn.Conv2dGroup(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = torch.nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(
            x, list(map(int, size)), mode="bilinear", align_corners=True
        )

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x


class FeatureFusionModule(torch.nn.Module):
    def __init__(
            self,
            highter_in_channels,
            lower_in_channels,
            out_channels,
            scale_factor=4,
            **kwargs
    ):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = torchx.nn.DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, 1),
            torch.nn.BatchNorm2d(out_channels),
        )
        self.conv_higher_res = torch.nn.Sequential(
            torch.nn.Conv2d(highter_in_channels, out_channels, 1),
            torch.nn.BatchNorm2d(out_channels),
        )
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(
            lower_res_feature, scale_factor=4, mode="bilinear", align_corners=True
        )
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class DefSCNN(torchx.nn.Module):
    def __init__(
            self,
            vol_size,
            **kwargs
    ):
        super().__init__()

        self.learning_to_downsample = torch.nn.Sequential(
            torchx.nn.Conv2dGroup(
                in_channels=2, out_channels=32, kernel_size=3, stride=2
            ),
            torchx.nn.DSConv(in_channels=32, out_channels=48, stride=2),
            torchx.nn.DSConv(in_channels=48, out_channels=64, stride=2),
        )

        self.global_feature_extractor = torch.nn.Sequential(
            *[
                Bottleneck(
                    in_channels=in_channel, out_channels=out_channel, stride=stride
                )
                for in_channel, out_channel, stride in zip(
                    (64, 64, 64, 64, 96, 96, 96, 128, 128),
                    (64, 64, 64, 96, 96, 96, 128, 128, 128),
                    (2, 1, 1, 2, 1, 1, 1, 1, 1),
                )
            ],
            PyramidPooling(128, 128)
        )

        self.feature_fusion_module = FeatureFusionModule(64, 128, 128)

        self.deformation = torch.nn.Sequential(
            *[
                torchx.nn.DSConv(in_channels=128, out_channels=128, kernel_size=1)
                for n in range(2)
            ],
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)
        )

        self.spatial_transform = SpatialTransformer(vol_size)

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion_module(higher_res_features, x)
        deform = self.deformation(x)
        deform = F.interpolate(deform, list(map(int, size)), mode="bilinear", align_corners=True) * 10.
        y = self.spatial_transform(src, deform)

        return y, deform
