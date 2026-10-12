"""Original U-Net architecture from UNET/U-Net-Dataloader.ipynb, cell 3."""

import torch
from torch import nn
from torch.nn import functional as F


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.bottleneck = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = nn.Conv2d(1024, 256, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Conv2d(512, 128, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Conv2d(256, 64, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(128, 32, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(64, 16, kernel_size=3, padding=1)

        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(self.pool(e1)))
        e3 = F.relu(self.enc3(self.pool(e2)))
        e4 = F.relu(self.enc4(self.pool(e3)))
        e5 = F.relu(self.enc5(self.pool(e4)))

        b = F.relu(self.bottleneck(self.pool(e5)))
        
        d5 = F.relu(self.dec5(torch.cat([self.upconv5(b), e5], dim=1)))
        
        d4 = F.relu(self.dec4(torch.cat([self.upconv4(d5), e4], dim=1)))
        
        d3 = F.relu(self.dec3(torch.cat([self.upconv3(d4), e3], dim=1)))
        
        d2 = F.relu(self.dec2(torch.cat([self.upconv2(d3), e2], dim=1)))
        
        d1 = F.relu(self.dec1(torch.cat([self.upconv1(d2), e1], dim=1)))
        
        return self.final(d1)
