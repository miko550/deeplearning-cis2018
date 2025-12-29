"""
Multi-Layer Perceptron (MLP) Model
===================================
A simple feedforward neural network with multiple fully connected layers.

Architecture:
- Input Layer: input_features -> 256
- Hidden Layer 1: 256 -> 128 (with BatchNorm, GELU, Dropout)
- Hidden Layer 2: 128 -> 64 (with BatchNorm, GELU, Dropout)
- Output Layer: 64 -> num_classes

Justification:
- Simple and effective for tabular data
- BatchNorm stabilizes training
- GELU activation provides smooth gradients
- Dropout prevents overfitting
"""

import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, input_features, num_classes, hidden_dims=[256, 128, 64], 
                 dropout_rate=0.2, activation='gelu'):
        """
        Initialize MLP model.
        
        Args:
            input_features: Number of input features
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            activation: Activation function ('gelu', 'relu', 'tanh')
        """
        super(MLPModel, self).__init__()
        
        # Activation function selection
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()
        }
        act_fn = activations.get(activation.lower(), nn.GELU())
        
        layers = []
        prev_dim = input_features
        
        # Build hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

