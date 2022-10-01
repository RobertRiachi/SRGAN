import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from dataset import LOW_RESOLUTION, HIGH_RESOLUTION

'''
Model architecture from SRGan paper https://arxiv.org/pdf/1609.04802.pdf
'''

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, discriminator=False, kernel_size=3, stride=1, padding=1, use_activation=True, use_batch_norm=True) -> None:
        super().__init__()

        self.use_activation = use_activation
        self.bias = not use_batch_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=self.bias)
        self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(num_parameters=out_channels)
    
    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.activation(x) if self.use_activation else x

class ResidualBlock(nn.Module):

    def __init__(self, in_channels) -> None:
        super().__init__()

        self.conv_1 = ConvBlock(in_channels, in_channels)
        self.conv_2 = ConvBlock(in_channels, in_channels, use_activation=False) 

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
        self.activation = nn.PReLU(num_parameters=in_channels)
    
    def forward(self, x):

        return self.activation(self.pixel_suffle(self.conv(x)))


class GeneratorNetwork(nn.Module):
    
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16) -> None:
        super().__init__()

        self.conv_1 = ConvBlock(in_channels, num_channels, kernel_size=9, padding=4, use_batch_norm=False)
        
        self.residual_blocks = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.conv_2 = ConvBlock(num_channels, num_channels, use_activation=False)
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, scale_factor=2), UpsampleBlock(num_channels, scale_factor=2))
        self.conv_3 = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)
    
    def forward(self, x):

        output = self.conv_1(x)
        x = self.residual_blocks(output)
        x = self.conv_2(x) + output
        x = self.upsamples(x)
        return torch.tanh(self.conv_3(x))


class DiscriminatorNetwork(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]) -> None:
        super().__init__()

        blocks = []

        for idx, feature in enumerate(features):
            blocks.append(ConvBlock(in_channels, 
                                    feature, 
                                    stride=1 + idx % 2, 
                                    discriminator=True, 
                                    use_activation=True, 
                                    use_batch_norm=False if idx == 0 else True))
            in_channels = feature           
        
        self.conv_blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )
    
    def forward(self, x):
        return self.classifier(self.conv_blocks(x))

class VGGLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vgg_model = vgg19(pretrained=True).features[:36].eval().to(device)
        self.loss = nn.MSELoss()

        for param in self.vgg_model.parameters():
            param.requires_grad = False
    
    def forward(self, input, expected):
        return self.loss(self.vgg_model(input), self.vgg_model(expected))


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