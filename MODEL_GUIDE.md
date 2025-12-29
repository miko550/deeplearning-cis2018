# Model Architecture Guide

## Quick Start

### 1. Train a Single Model

Edit `config.yaml`:
```yaml
model_name: mlp  # Change this to test different models
```

Run:
```bash
un run main.py
```

### 2. Compare All Models

```bash
uv run compare_models.py
```

This will train all models and show you which performs best.

## Available Models

| Model | Config Name | Best For | Speed |
|-------|------------|----------|-------|
| MLP | `mlp` | Tabular data, baseline | ⚡⚡⚡ Fast |
| CNN | `cnn` | Local patterns | ⚡⚡ Medium |
| RNN | `rnn` | Sequential data | ⚡ Slow |
| LSTM | `lstm` | Long dependencies | ⚡ Slow |
| GRU | `gru` | Sequential (efficient) | ⚡⚡ Medium |
| Transformer | `transformer` | Complex relationships | ⚡⚡ Medium |
| CNN-RNN | `cnn_rnn` | Patterns + sequences | ⚡ Slow |
| CNN-LSTM | `cnn_lstm` | Patterns + long deps | ⚡ Slow |
| CNN-GRU | `cnn_gru` | Patterns + sequences | ⚡⚡ Medium |

## Configuration Examples

### Basic MLP
```yaml
model_name: mlp
```

### LSTM with Custom Parameters
```yaml
model_name: lstm
model_params:
  hidden_size: 256
  num_layers: 3
  dropout_rate: 0.3
  bidirectional: true
```

### Transformer
```yaml
model_name: transformer
model_params:
  d_model: 128
  nhead: 8
  num_layers: 4
  dropout_rate: 0.1
```

## Architecture Justifications

### MLP (Multi-Layer Perceptron)
**Why**: Simple, fast, effective baseline
- Fully connected layers
- BatchNorm for stability
- GELU activation
- Dropout for regularization

### CNN (Convolutional Neural Network)
**Why**: Captures local patterns in features
- 1D convolutions detect patterns
- Global pooling reduces parameters
- Good for feature extraction

### LSTM (Long Short-Term Memory)
**Why**: Handles long-term dependencies
- Memory cells store information
- Solves vanishing gradient problem
- Good for sequential relationships

### GRU (Gated Recurrent Unit)
**Why**: Efficient alternative to LSTM
- Simpler than LSTM
- Similar performance
- Faster training

### Transformer
**Why**: State-of-the-art attention mechanism
- Self-attention captures relationships
- Parallel processing
- Best for complex patterns

### Hybrid Models
**Why**: Combine benefits of multiple architectures
- CNN extracts local features
- RNN/LSTM/GRU models sequences
- Best of both worlds

## Recommendations

1. **Start**: MLP (baseline)
2. **If features have patterns**: Try CNN
3. **If features are sequential**: Try LSTM or GRU
4. **For best performance**: Try Transformer
5. **For complex data**: Try Hybrid models

## Performance Tips

- **Batch Size**: 
  - MLP/CNN: 128-256
  - RNN/LSTM: 64-128
  - Transformer: 32-64

- **Learning Rate**:
  - Start with 0.0001
  - Increase if training is slow
  - Decrease if loss is unstable

- **Dropout**:
  - Simple models (MLP): 0.2
  - Complex models (Transformer): 0.3-0.5

## Example Workflow

1. **Baseline**: Train MLP
   ```bash
   # config.yaml: model_name: mlp
   uv run main.py
   ```

2. **Compare**: Test multiple architectures
   ```bash
  uv run compare_models.py
   ```

3. **Optimize**: Tune best model's hyperparameters
   ```yaml
   model_name: lstm  # Best from comparison
   model_params:
     hidden_size: 256
     num_layers: 3
   ```

4. **Final Training**: Train with best configuration
   ```bash
    uv run main.py
   ```

