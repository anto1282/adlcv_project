import torch
import torch.nn as nn
import torch.nn.functional as F

class ControlNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=320):
        """
        A simple ControlNet that converts box maps into feature maps
        compatible with the first stage of the VPD UNet.
        
        - in_channels: e.g., 1 for box maps or 3 for RGB-style hints
        - out_channels: must match the early feature map channels of UNet
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)
