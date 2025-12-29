"""
Convolutional Neural Network (CNN) Model
=========================================
1D CNN for sequence-like feature data.

Architecture:
- Reshape input to sequence format
- 1D Convolutional layers with increasing filters
- Global Average Pooling
- Fully connected layers for classification

Justification:
- CNNs can capture local patterns in features
- 1D convolutions work well for sequential/tabular data
- Global pooling reduces parameters
- Effective for feature extraction
"""

import torch.nn as nn
import torch


class CNNModel(nn.Module):
    def __init__(self, input_features, num_classes, num_filters=[64, 128, 256], 
                 kernel_sizes=[3, 3, 3], dropout_rate=0.3):
        """
        Initialize CNN model.
        
        Args:
            input_features: Number of input features
            num_classes: Number of output classes
            num_filters: List of filter sizes for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            dropout_rate: Dropout probability
        """
        super(CNNModel, self).__init__()
        self.input_features = input_features
        
        # Reshape input: (batch, features) -> (batch, 1, features) for 1D conv
        # We'll treat features as a sequence
        
        conv_layers = []
        in_channels = 1
        
        for i, (out_channels, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_filters[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Reshape: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Global pooling: (batch, channels, length) -> (batch, channels, 1)
        x = self.global_pool(x)
        
        # Flatten: (batch, channels, 1) -> (batch, channels)
        x = x.squeeze(-1)
        
        # Classification
        x = self.classifier(x)
        return x

