# Model Architectures

This directory contains modular neural network architectures for Intrusion Detection System (IDS).

## Available Models

### 1. **MLP (Multi-Layer Perceptron)**
- **File**: `mlp.py`
- **Use Case**: Simple feedforward network, good for tabular data
- **Architecture**: Fully connected layers with BatchNorm, GELU, Dropout
- **Justification**: Simple, effective, fast training

### 2. **CNN (Convolutional Neural Network)**
- **File**: `cnn.py`
- **Use Case**: Captures local patterns in features
- **Architecture**: 1D convolutions + Global Average Pooling
- **Justification**: Good for feature extraction, captures local patterns

### 3. **RNN (Recurrent Neural Network)**
- **File**: `rnn.py`
- **Use Case**: Sequential/temporal dependencies
- **Architecture**: Multi-layer RNN
- **Justification**: Models sequential relationships in features

### 4. **LSTM (Long Short-Term Memory)**
- **File**: `lstm.py`
- **Use Case**: Long-term dependencies
- **Architecture**: Multi-layer LSTM
- **Justification**: Handles vanishing gradients, captures long-term patterns

### 5. **GRU (Gated Recurrent Unit)**
- **File**: `gru.py`
- **Use Case**: Efficient sequential modeling
- **Architecture**: Multi-layer GRU
- **Justification**: Simpler than LSTM, similar performance, faster

### 6. **Transformer**
- **File**: `transformer.py`
- **Use Case**: Attention-based feature relationships
- **Architecture**: Multi-head self-attention + Feed-forward
- **Justification**: Captures complex feature interactions, parallel processing

### 7. **Hybrid Models**
- **Files**: `hybrid.py`
- **Available**: CNN-RNN, CNN-LSTM, CNN-GRU
- **Use Case**: Combine CNN feature extraction with RNN sequential modeling
- **Justification**: Best of both worlds - local patterns + temporal dependencies

## Usage

### Single Model Training

Edit `config.yaml`:
```yaml
model_name: lstm  # Choose: mlp, cnn, rnn, lstm, gru, transformer, cnn_rnn, cnn_lstm, cnn_gru
```

Run:
```bash
python main.py
```

### Compare Multiple Models

```bash
python compare_models.py
```

This will train all models and compare their performance.

### Custom Model Parameters

In `config.yaml`:
```yaml
model_name: lstm
model_params:
  hidden_size: 256
  num_layers: 3
  dropout_rate: 0.3
  bidirectional: true
```

## Model Interface

All models follow the same interface:
```python
model = ModelClass(input_features=num_features, num_classes=num_classes, **kwargs)
output = model(input_tensor)  # Shape: (batch_size, num_classes)
```

## Architecture Justifications

### Why MLP?
- **Simple**: Easy to understand and debug
- **Fast**: Quick training and inference
- **Effective**: Works well for tabular data
- **Baseline**: Good starting point for comparison

### Why CNN?
- **Pattern Recognition**: Detects local patterns in features
- **Translation Invariant**: Robust to feature ordering
- **Efficient**: Shared weights reduce parameters

### Why RNN/LSTM/GRU?
- **Sequential Modeling**: If features have temporal relationships
- **Memory**: Can remember important information
- **Dependencies**: Captures dependencies between features

### Why Transformer?
- **Attention**: Focuses on important features
- **Parallel**: Faster training than RNNs
- **Relationships**: Models complex feature interactions
- **State-of-the-art**: Best performance for many tasks

### Why Hybrid?
- **Combined Benefits**: CNN extracts features, RNN models sequences
- **Flexibility**: Can adapt to different data characteristics
- **Performance**: Often outperforms single architectures

## Recommendations

1. **Start with MLP**: Simple baseline
2. **Try CNN**: If features have spatial relationships
3. **Try LSTM/GRU**: If features are sequential
4. **Try Transformer**: For best performance (if computational resources allow)
5. **Try Hybrid**: If you need both feature extraction and sequential modeling

## Performance Tips

- **Batch Size**: Larger for CNN/Transformer, smaller for RNN/LSTM
- **Learning Rate**: May need adjustment for different architectures
- **Dropout**: Higher for complex models (Transformer, Hybrid)
- **Hidden Size**: Start with 128, increase if underfitting

