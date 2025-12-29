"""
Long Short-Term Memory (LSTM) Model
====================================
LSTM for capturing long-term dependencies in sequential data.

Architecture:
- Reshape input to sequence format
- Multi-layer LSTM
- Last hidden state extraction
- Fully connected classifier

Justification:
- LSTM handles vanishing gradient problem better than RNN
- Can capture long-term dependencies
- Memory cells store important information
- Effective for sequential pattern recognition
"""

import torch.nn as nn
import torch


class LSTMModel(nn.Module):
    def __init__(self, input_features, num_classes, hidden_size=128, 
                 num_layers=2, dropout_rate=0.3, bidirectional=False):
        """
        Initialize LSTM model.
        
        Args:
            input_features: Number of input features
            num_classes: Number of output classes
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout_rate: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=1,  # Each feature is a scalar
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Reshape: (batch, features) -> (batch, features, 1)
        x = x.unsqueeze(-1)
        
        # LSTM forward pass
        output, (hidden, cell) = self.lstm(x)
        
        # Use last output
        x = output[:, -1, :]  # Take last time step
        
        # Classification
        x = self.classifier(x)
        return x

