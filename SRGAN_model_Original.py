import torch
import numpy as np
from torch import nn

ACTIVATIONS = {'relu': nn.ReLU(),
               'tanh': nn.Tanh(),
               'sigmoid': nn.Sigmoid(),
               'prelu': nn.PReLU(),
               'leakyrelu': nn.LeakyReLU(0.2),
               'none': None}


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 activation: str = 'relu', batch_norm: bool = True):
        super().__init__()
        act_func = activation.lower()
        if act_func not in list(ACTIVATIONS.keys()):
            print(f'Specified wrong activation function (available values: {list(ACTIVATIONS.keys())})')
            raise ValueError
        else:
            layers = []
            conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=kernel_size//2,
                                  bias=~batch_norm)
            bn = nn.BatchNorm2d(out_channels)
            act = ACTIVATIONS[act_func]

            layers.append(conv)
            if batch_norm: layers.append(bn)
            if ACTIVATIONS[act_func] is not None: layers.append(act)
            self.convBlock = nn.Sequential(*layers)

    def forward(self, x):
        out = self.convBlock(x)
        return out


class UpBlock(nn.Module):

    def __init__(self, in_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(UpBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.leaky_relu(self.conv(nn.functional.interpolate(x, scale_factor=2,
                                                                            mode="bilinear",
                                                                            align_corners=True)), 0.2, True)


class ResBlock(nn.Module):
    def __init__(self, n_c: int, ks: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=n_c, out_channels=n_c, kernel_size=ks,
                               activation='PReLU', batch_norm=True)

        self.conv2 = ConvBlock(in_channels=n_c, out_channels=n_c, kernel_size=ks,
                               activation='none', batch_norm=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        return out


class ScalePixelConvBlock(nn.Module):
    def __init__(self, in_channels: int, scale_factor: int, kernel_size: int):
        super().__init__()
        conv = nn.Conv2d(in_channels=in_channels,
                         out_channels=in_channels*(scale_factor**2),
                         kernel_size=kernel_size,
                         padding=kernel_size//2)

        pixShuffle = nn.PixelShuffle(upscale_factor=scale_factor)
        act = nn.PReLU()
        self.layers = nn.Sequential(conv, pixShuffle, act)

    def forward(self, x):
        out = self.layers(x)
        return out


class Generator(nn.Module):
    def __init__(self,
                 large_kernel_size: int = 9, small_kernel_size: int = 3,
                 channels_num: int = 64, n_blocks: int = 16, scaling_factor: int = 2):
        super(Generator, self).__init__()

        assert type(scaling_factor) is int and scaling_factor in [2, 4, 8]

        self.conv_block1 = ConvBlock(in_channels=3, out_channels=channels_num,
                                     kernel_size=large_kernel_size,
                                     batch_norm=False,
                                     activation='PReLU')

        self.residual_blocks = nn.Sequential(
            *[ResBlock(ks=small_kernel_size, n_c=channels_num) for i in range(n_blocks)])

        self.conv_block2 = ConvBlock(in_channels=channels_num,
                                     out_channels=channels_num,
                                     kernel_size=small_kernel_size,
                                     batch_norm=True, activation='none')

        n_upsample_blocks = int(np.log2(scaling_factor))
        self.upsample_blocks = nn.Sequential(
            *[UpBlock(kernel_size=small_kernel_size, in_channels=channels_num) for i
              in range(n_upsample_blocks)])

        self.conv_block3 = ConvBlock(in_channels=channels_num,
                                     out_channels=3,
                                     kernel_size=large_kernel_size,
                                     batch_norm=False, activation='Tanh')

    def forward(self, x):
        x_conv1 = self.conv_block1(x)
        x_res = self.residual_blocks(x_conv1)
        output = self.conv_block2(x_res)
        output = output + x_conv1
        output = self.upsample_blocks(output)
        output = self.conv_block3(output)
        return output


class Discriminator(nn.Module):
    def __init__(self,
                 in_channels: int = 3, kernel_size: int = 3, channels_num: int = 64,
                 n_blocks: int = 8, fc_size: int=1024):
        super().__init__()

        convBlocks = list()
        for i in range(n_blocks):
            out_channels = (channels_num if i is 0 else in_channels * 2) if i % 2 is 0 else in_channels
            convBlocks.append(
                ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                          stride=1 if i % 2 is 0 else 2, batch_norm=i is not 0, activation='LeakyReLU'))
            in_channels = out_channels
        self.convBlocks = nn.Sequential(*convBlocks)
        self.adaPool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)

        self.leakyRelu = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, imgs):
        batch_size = imgs.size(0)
        output = self.convBlocks(imgs)
        output = self.adaPool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leakyRelu(output)
        logit = self.fc2(output)

        return logit


