"""
Deep Learning Models for Music Genre Classification.

This module defines the neural network architectures used in the project:
1. MusicMLP: A Multi-Layer Perceptron for feature-based classification (MFCCs).
2. MusicCNN: A custom Convolutional Neural Network for spectrogram-based classification.
3. MusicResNet18: A transfer learning model based on ResNet-18 adapted for single-channel audio spectrograms.
"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MusicMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for music genre classification using extracted audio features.
    
    Architecture:
        - Input Layer: Accepts a vector of features (default dim=44, e.g., MFCC statistics).
        - Hidden Layer 1: Linear -> BatchNorm -> ReLU -> Dropout.
        - Hidden Layer 2: Linear -> BatchNorm -> ReLU -> Dropout.
        - Output Layer: Linear layer mapping to class logits.
    """
    def __init__(self, input_dim=44, num_classes=10, hidden_dim=256):
        """
        Args:
            input_dim (int): Dimensionality of the input feature vector.
            num_classes (int): Number of target classes (genres).
            hidden_dim (int): Number of neurons in the first hidden layer.
        """
        super(MusicMLP, self).__init__()
        # Layer 1
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        # Layer 2
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.3)
        
        # Output Layer
        self.layer3 = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        """
        Forward pass of the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.layer3(x)
        return x

class MusicCNN(nn.Module):
    """
    Custom Convolutional Neural Network (CNN) for audio spectrogram classification.
    
    Architecture:
        - 4 Convolutional Blocks: (Conv2d -> BatchNorm -> ReLU -> MaxPool).
        - Flattening.
        - Fully Connected Layers with Dropout.
        
    Designed to process Mel-spectrograms as 1-channel images (128x128).
    """
    def __init__(self, num_classes=10):
        """
        Args:
            num_classes (int): Number of target classes (genres).
        """
        super(MusicCNN, self).__init__()
        # Input shape: (Batch, 1, 128, 128)
        # 1 channel (grayscale spectrogram), 128 mel bands, 128 time steps
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2) # 128 -> 64
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # pool: 64 -> 32
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # pool: 32 -> 16
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # pool: 16 -> 8
        
        self.flatten_size = 256 * 8 * 8
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass of the CNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 128, 128).
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # x: (batch, 1, 128, 128)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MusicResNet18(nn.Module):
    """
    ResNet-18 model adapted for single-channel audio spectrograms.
    
    Uses a pre-trained ResNet-18 backbone (ImageNet weights) with:
    1. Modified first convolutional layer to accept 1 input channel (instead of 3 RGB).
    2. Modified final fully connected layer to output `num_classes` logits.
    """
    def __init__(self, num_classes=10):
        """
        Args:
            num_classes (int): Number of target classes (genres).
        """
        super(MusicResNet18, self).__init__()
        # Load pre-trained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify the first convolutional layer to accept 1 channel instead of 3
        # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the final fully connected layer to output num_classes
        # Original: nn.Linear(512, 1000)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        """
        Forward pass of the ResNet-18.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 128, 128).
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        return self.resnet(x)
