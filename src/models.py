"""
Model definitions.
Includes MLP for feature-based classification and CNN for spectrograms.
"""
import torch.nn as nn
import torch.nn.functional as F

class MusicMLP(nn.Module):
    def __init__(self, input_dim=44, num_classes=10, hidden_dim=256):
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
