"""
Hybrid Models
=============
Combining CNN feature extraction with RNN/LSTM/GRU for sequential modeling.

Architecture:
- CNN layers for local pattern extraction
- RNN/LSTM/GRU for sequential modeling
- Fully connected classifier

Justification:
- CNN extracts local features/patterns
- RNN/LSTM/GRU models temporal dependencies
- Combines benefits of both architectures
- Effective for complex pattern recognition
"""

import torch.nn as nn
import torch


class HybridCNNRNN(nn.Module):
    """CNN + RNN hybrid model."""
    
    def __init__(self, input_features, num_classes, cnn_filters=[64, 128], 
                 rnn_hidden=128, num_rnn_layers=2, dropout_rate=0.3):
        super(HybridCNNRNN, self).__init__()
        
        # CNN feature extraction
        cnn_layers = []
        in_channels = 1
        for out_channels in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # RNN for sequential modeling
        self.rnn = nn.RNN(
            input_size=cnn_filters[-1],
            hidden_size=rnn_hidden,
            num_layers=num_rnn_layers,
            dropout=dropout_rate if num_rnn_layers > 1 else 0,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Reshape for CNN
        x = x.unsqueeze(1)  # (batch, 1, features)
        
        # CNN feature extraction
        x = self.cnn(x)  # (batch, filters, length)
        x = x.transpose(1, 2)  # (batch, length, filters)
        
        # RNN
        output, hidden = self.rnn(x)
        x = output[:, -1, :]  # Last time step
        
        # Classification
        x = self.classifier(x)
        return x


class HybridCNNLSTM(nn.Module):
    """CNN + LSTM hybrid model."""
    
    def __init__(self, input_features, num_classes, cnn_filters=[64, 128], 
                 lstm_hidden=128, num_lstm_layers=2, dropout_rate=0.3):
        super(HybridCNNLSTM, self).__init__()
        
        # CNN feature extraction
        cnn_layers = []
        in_channels = 1
        for out_channels in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # LSTM for sequential modeling
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        output, (hidden, cell) = self.lstm(x)
        x = output[:, -1, :]
        x = self.classifier(x)
        return x


class HybridCNNGRU(nn.Module):
    """CNN + GRU hybrid model."""
    
    def __init__(self, input_features, num_classes, cnn_filters=[64, 128], 
                 gru_hidden=128, num_gru_layers=2, dropout_rate=0.3):
        super(HybridCNNGRU, self).__init__()
        
        # CNN feature extraction
        cnn_layers = []
        in_channels = 1
        for out_channels in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # GRU for sequential modeling
        self.gru = nn.GRU(
            input_size=cnn_filters[-1],
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            dropout=dropout_rate if num_gru_layers > 1 else 0,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        output, hidden = self.gru(x)
        x = output[:, -1, :]
        x = self.classifier(x)
        return x

