"""
Recurrent Neural Network (RNN) Model
====================================
RNN for sequential feature processing.

Architecture:
- Reshape input to sequence format
- Multi-layer RNN
- Last hidden state extraction
- Fully connected classifier

Justification:
- RNNs capture temporal/sequential dependencies
- Useful if features have sequential relationships
- Can model long-range dependencies
"""

import torch.nn as nn
import torch


class RNNModel(nn.Module):
    def __init__(self, input_features, num_classes, hidden_size=128, 
                 num_layers=2, dropout_rate=0.3, bidirectional=False):
        """
        Initialize RNN model.
        
        Args:
            input_features: Number of input features
            num_classes: Number of output classes
            hidden_size: RNN hidden state size
            num_layers: Number of RNN layers
            dropout_rate: Dropout probability
            bidirectional: Use bidirectional RNN
        """
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Reshape: treat each feature as a time step
        # Input: (batch, features) -> (batch, features, 1)
        # Or we can reshape to (batch, 1, features) and use features as sequence length
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=1,  # Each feature is a scalar
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output size
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Reshape: (batch, features) -> (batch, features, 1)
        # Each feature becomes a time step
        x = x.unsqueeze(-1)
        
        # RNN forward pass
        output, hidden = self.rnn(x)
        
        # Use last output (or last hidden state)
        # output shape: (batch, seq_len, hidden_size)
        x = output[:, -1, :]  # Take last time step
        
        # Classification
        x = self.classifier(x)
        return x

