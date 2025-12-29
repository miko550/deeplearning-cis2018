"""
Gated Recurrent Unit (GRU) Model
=================================
GRU is a simpler alternative to LSTM with similar performance.

Architecture:
- Reshape input to sequence format
- Multi-layer GRU
- Last hidden state extraction
- Fully connected classifier

Justification:
- GRU is computationally more efficient than LSTM
- Still captures long-term dependencies
- Fewer parameters than LSTM
- Good balance between performance and efficiency
"""

import torch.nn as nn
import torch


class GRUModel(nn.Module):
    def __init__(self, input_features, num_classes, hidden_size=128, 
                 num_layers=2, dropout_rate=0.3, bidirectional=False):
        """
        Initialize GRU model.
        
        Args:
            input_features: Number of input features
            num_classes: Number of output classes
            hidden_size: GRU hidden state size
            num_layers: Number of GRU layers
            dropout_rate: Dropout probability
            bidirectional: Use bidirectional GRU
        """
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=1,  # Each feature is a scalar
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output size
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Reshape: (batch, features) -> (batch, features, 1)
        x = x.unsqueeze(-1)
        
        # GRU forward pass
        output, hidden = self.gru(x)
        
        # Use last output
        x = output[:, -1, :]  # Take last time step
        
        # Classification
        x = self.classifier(x)
        return x

