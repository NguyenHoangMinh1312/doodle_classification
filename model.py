import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class HandDrawingCNN(nn.Module):
    def __init__(self, num_classes=10):
        """
        CNN architecture for classifying 28x28 grayscale images.
        
        Args:
            num_classes: Number of output classes (default: 10 for MNIST/Fashion-MNIST)
        """
        super(HandDrawingCNN, self).__init__()
        
        self.block1 = self.block(1, 256)
        self.block2 = self.block(256, 512)

        self.flatten_size = 512 * 7 * 7
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        x = x.view(-1, self.flatten_size)
        
        # Fully connected
        x = self.fc(x)
        
        return x
    
if __name__ == "__main__":
    x = torch.rand((2, 1, 28, 28))
    model = HandDrawingCNN(num_classes=20)
    y = model(x)
    print(y.shape)