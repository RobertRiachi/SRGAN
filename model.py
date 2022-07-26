import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import LOW_RESOLUTION, HIGH_RESOLUTION

'''
Model architecture from SRGan paper https://arxiv.org/pdf/1609.04802.pdf
'''

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, normalization=None, activation=None) -> None:
        super().__init__()

        self.bias = False if normalization else True
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=self.bias)
        self.bn = nn.BatchNorm2d(out_channels) if normalization else None
        self.activation = activation
    
    def forward(self, x):
        x = self.conv(x)

        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        
        return x

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, use_normalization=False, use_activation=False) -> None:
        super().__init__()

        self.conv_1 = ConvBlock(in_channels,
                                in_channels, 
                                normalization=nn.BatchNorm2d(in_channels) if use_normalization else None, 
                                activation=nn.PReLU(in_channels) if use_activation else None)
        self.conv_2 = ConvBlock(in_channels,
                                in_channels, 
                                normalization=nn.BatchNorm2d(in_channels) if use_normalization else None)

    def forward(self, x):
        
        res = self.conv_1(x)
        res = self.conv_2(res)

        # skip connection
        return res + x


class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, scale_factor, kernel_size=3, stride=1, padding=1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor** 2, kernel_size, stride, padding)
        self.pixel_suffle = nn.PixelShuffle(scale_factor)
        self.activation = nn.PReLU(in_channels)
    
    def forward(self, x):

        return self.activation(self.pixel_suffle(self.conv(x)))


class GeneratorNetwork(nn.Module):
    
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16) -> None:
        super().__init__()

        self.conv_1 = ConvBlock(in_channels, num_channels, kernel_size=9, padding=4, activation=nn.PReLU(num_parameters=num_channels))
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(num_channels, use_normalization=True, use_activation=True))

        self.residual_blocks = nn.Sequential(*blocks)
        self.conv_2 = ConvBlock(num_channels, num_channels, normalization=nn.BatchNorm2d(num_features=num_channels))
        self.up_sample_1 = UpsampleBlock(num_channels, scale_factor=2)
        self.up_sample_2 = UpsampleBlock(num_channels, scale_factor=2)
        self.conv_3 = ConvBlock(num_channels, in_channels, kernel_size=9, padding=4)
    
    def forward(self, x):

        x = self.conv_1(x)
        res = self.residual_blocks(x)
        res = self.conv_2(res)

        # skip connection
        res = res + x
        res = self.up_sample_1(res)
        res = self.up_sample_2(res)
        res = self.conv_3(res)

        return torch.tanh_(res)

class Classifier(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()

        # Garauntee size for larger resolution image training sets
        self.adaptive = nn.AdaptiveAvgPool2d(input_dim)
        self.flat = nn.Flatten()
        self.fcn_1 = nn.Linear(512*6*6, 1024)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.fcn_2 = nn.Linear(1024, 1)
    
    def forward(self, x):
        x = self.adaptive(x)
        x = self.flat(x)
        x = self.fcn_1(x)
        x = self.leaky_relu(x)

        return torch.sigmoid_(self.fcn_2(x))


class DiscriminatorNetwork(nn.Module):
    def __init__(self, in_channels=3, num_blocks=8) -> None:
        super().__init__()

        blocks = []
        out_channels = 64
        for num_block in range(num_blocks):
            blocks.append(ConvBlock(in_channels=in_channels,
                                    out_channels=out_channels,
                                    stride=1 + num_block % 2,
                                    normalization=nn.BatchNorm2d(in_channels) if num_block != 0 else None,
                                    activation=nn.LeakyReLU(0.2, inplace=True)))

            in_channels = out_channels

            if num_block % 2 == 1:
                out_channels *= 2
            
        
        self.conv_blocks = nn.Sequential(*blocks)
        self.classifier = Classifier((6,6))
    
    def forward(self, x):
        x = self.conv_blocks(x)

        return self.classifier(x)



if __name__ == "__main__":
    
    # Output will be low_res * 2 * 2 by low_res * 2 * 2
    with torch.cuda.amp.autocast():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"The model will be running on {device} device")

        x = torch.randn((5, 3, LOW_RESOLUTION, LOW_RESOLUTION), device=device)
        gen = GeneratorNetwork().to(device)
        gen_out = gen(x)
        disc = DiscriminatorNetwork().to(device)
        disc_out = disc(gen_out)

        print(gen_out.shape)
        print(disc_out.shape)