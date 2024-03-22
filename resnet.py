import os
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50(nn.Module):
    def __init__(self, num_input_channels=1, num_outputs=1):
        super(ResNet50, self).__init__()

        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_outputs)

    def forward(self, x):
        x = self.resnet(x)
        return x