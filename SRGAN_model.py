import torch
import numpy as np
from torch import nn
from torchvision.models import vgg19

ACTIVATIONS = {'relu': nn.ReLU(),
               'tanh': nn.Tanh(),
               'sigmoid': nn.Sigmoid(),
               'prelu': nn.PReLU(),
               'leakyrelu': nn.LeakyReLU(0.2),
               'none': None}

#feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])
    
    def forward(self, img):
        return self.feature_extractor(img)
    
#Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, eps=1e-5, momentum=0.01, affine=True),
        )

    def forward(self, x):
        return x + self.conv_block(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, batch_norm=True, activation='relu'):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        ]
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'PReLU':
            layers.append(nn.PReLU())
        elif activation == 'Tanh':
            layers.append(nn.Tanh())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'none':
            pass
        else:
            raise NotImplementedError(f"Activation '{activation}' is not implemented.")
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, kernel_size, in_channels):
        super(UpBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, 32, kernel_size=kernel_size, stride=2, padding=1,
                                       output_padding=1)

    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, ks, n_c):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(n_c, n_c, kernel_size=ks)
        self.conv2 = ConvBlock(n_c, n_c, kernel_size=ks)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
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
                 large_kernel_size: int = 9, small_kernel_size: int = 4,
                 channels_num: int = 32, n_blocks: int = 16, scaling_factor: int = 2):
        super(Generator, self).__init__()

        assert type(scaling_factor) is int and scaling_factor in [2, 4, 8]

        self.conv_block1 = ConvBlock(in_channels=3, out_channels=32,
                                     kernel_size=small_kernel_size,
                                     batch_norm=False,
                                     activation='PReLU')

        self.residual_blocks = nn.Sequential(
            *[ResBlock(ks=3, n_c=channels_num) for i in range(n_blocks)])

        self.conv_block2 = ConvBlock(in_channels=channels_num,
                                     out_channels=32,
                                     kernel_size=3,
                                     batch_norm=True, activation='none')

        n_upsample_blocks = int(np.log2(scaling_factor))  # Determine number of upsampling blocks
        self.upsample_blocks = nn.Sequential(
            *[UpBlock(kernel_size=3, in_channels=32) for i
              in range(n_upsample_blocks)])

        self.conv_block3 = ConvBlock(
                                    in_channels=32,
                                    out_channels=3,
                                    kernel_size=9,
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
    """ A 4-layer Markovian discriminator
    """
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            #Returns downsampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # print("Size of img_A: ", img_A.size())
        # print("Size of img_B: ", img_B.size())
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

