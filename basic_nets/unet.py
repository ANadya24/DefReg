import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """UNet architecture."""
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,
                            out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels,
                            out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,
                            out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel,
                            out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels,
                                     kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,
                            out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel,
                            out_channels=out_channels, padding=1),  # stride=2),#added stride
            torch.nn.BatchNorm2d(out_channels),
            #                    torch.nn.Upsample(scale_factor=2, mode='linear')#addedUpsample
            # torch.nn.Sigmoid()
        )
        return block

    def __init__(self, in_channel, out_channel, inter_channel=(32, 64, 128)):
        super(UNet, self).__init__()
        # Encode
        assert len(inter_channel) == 3, 'Depth of the net must be equal to 3!'
        self.conv_encode1 = self.contracting_block(in_channels=in_channel,
                                                   out_channels=inter_channel[0])
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(inter_channel[0], inter_channel[1])
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(inter_channel[1], inter_channel[2])
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        mid_channel = inter_channel[2]
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel,
                            out_channels=mid_channel * 2, padding=1),
            torch.nn.BatchNorm2d(mid_channel * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel * 2,
                            out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=mid_channel,
                                     out_channels=mid_channel, kernel_size=3,
                                     stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
        )
        # Decode
        self.conv_decode3 = self.expansive_block(inter_channel[2]*2,
                                                 inter_channel[2], inter_channel[1])
        self.conv_decode2 = self.expansive_block(inter_channel[2], inter_channel[1],
                                                 inter_channel[0])
        self.final_layer = self.final_block(inter_channel[1], inter_channel[0], out_channel)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, [-c, -c, -c, -c])
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        final_layer = self.final_layer(decode_block1)
        return final_layer