"""
Transformer Model
=================
Transformer architecture for sequence modeling with attention mechanism.

Architecture:
- Input embedding
- Positional encoding
- Multi-head self-attention layers
- Feed-forward networks
- Classification head

Justification:
- Attention mechanism captures relationships between features
- Parallel processing (faster than RNN/LSTM)
- State-of-the-art for many sequence tasks
- Can model complex feature interactions
"""

import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    def __init__(self, input_features, num_classes, d_model=128, nhead=8, 
                 num_layers=3, dim_feedforward=512, dropout_rate=0.1):
        """
        Initialize Transformer model.
        
        Args:
            input_features: Number of input features
            num_classes: Number of output classes
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension
            dropout_rate: Dropout probability
        """
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        # Input embedding: project features to d_model
        self.input_embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_features)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Reshape: (batch, features) -> (batch, features, 1)
        x = x.unsqueeze(-1)
        
        # Embedding
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global pooling: (batch, seq_len, d_model) -> (batch, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
        # Classification
        x = self.classifier(x)
        return x

