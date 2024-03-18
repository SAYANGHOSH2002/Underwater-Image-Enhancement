import torch
import torch.nn as nn 
from torch import tensor
import torch.nn.functional as F 

# class SelfAttention(nn.Module):
#     """ Self attention Layer"""
#     def __init__(self, in_dim):
#         super(SelfAttention, self).__init__()
#         self.chanel_in = in_dim
#         activation = nn.ReLU()
#         self.activation = activation

#         # Define convolutional layers
#         self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         """
#             inputs:
#                 x: input feature maps (B x C x W x H)
#             returns:
#                 out: self attention value + input feature
#                 attention: B x N x N (N is Width * Height)
#         """
#         m_batchsize, C, width, height = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x CX(N)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B x C x (*W*H)
#         energy = torch.bmm(proj_query, proj_key)  # transpose check
#         attention = self.softmax(energy)  # B X (N) X (N)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B x C X N

#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, width, height)

#         out = self.gamma * out + x
#         return out, attention


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn:
            layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(ResidualBlock(out_size, out_size))  # Add residual block
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x
       
class MuLA_GAN_Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(MuLA_GAN_Generator, self).__init__()
        # Encoding layers
        self.down1 = UNetDown(in_channels, 32, bn=False)
        self.down2 = UNetDown(32, 64)
        self.down3 = UNetDown(64, 128)
        
        # # Attention layers
        # self.attention1 = SelfAttention(32)
        # self.attention2 = SelfAttention(64)

        # Decoding layers
        self.up1 = UNetUp(128, 64)
        self.up2 = UNetUp(128, 32)

        # Final layer
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, kernel_size=4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):

        # Encoding
        x1 = self.down1(x)
        #x1_att = self.attention1(x1)
        x2 = self.down2(x1)
        #x2_att = self.attention2(x2)
        x3 = self.down3(x2)
        
        # Decoding
        x_up1 = self.up1(x3, x2)
        x_up2 = self.up2(x_up1, x1)
        
        # Final layer
        out = self.final(x_up2)

        return out


    

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
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

