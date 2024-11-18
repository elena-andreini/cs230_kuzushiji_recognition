import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights 

class ResNet18Encoder(nn.Module):
    def __init__(self):
        super(ResNet18Encoder, self).__init__()
        self.resnet18 =  models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])  # Remove the last two layers
        self.conv_reduce = nn.Conv2d(512, 256, kernel_size=1)
    def forward(self, x):
        x = self.resnet18(x)
        x = self.conv_reduce(x)
        return x


class ResNet18Decoder(nn.Module):
    def __init__(self):
        super(ResNet18Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class ResNet18Autoencoder(nn.Module):
    def __init__(self):
        super(ResNet18Autoencoder, self).__init__()
        self.encoder = ResNet18Encoder()
        self.decoder = ResNet18Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


