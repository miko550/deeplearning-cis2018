"""
Modular Model Architecture System
==================================
This module provides a factory pattern for creating different neural network architectures.
All models follow the same interface: (input_features, num_classes) -> model
"""

from .mlp import MLPModel
from .cnn import CNNModel
from .rnn import RNNModel
from .lstm import LSTMModel
from .gru import GRUModel
from .transformer import TransformerModel
from .hybrid import HybridCNNRNN, HybridCNNLSTM, HybridCNNGRU

# Model registry - maps model names to model classes
MODEL_REGISTRY = {
    'mlp': MLPModel,
    'cnn': CNNModel,
    'rnn': RNNModel,
    'lstm': LSTMModel,
    'gru': GRUModel,
    'transformer': TransformerModel,
    'cnn_rnn': HybridCNNRNN,
    'cnn_lstm': HybridCNNLSTM,
    'cnn_gru': HybridCNNGRU,
}


def create_model(model_name: str, input_features: int, num_classes: int, **kwargs):
    """
    Factory function to create a model instance.
    
    Args:
        model_name: Name of the model architecture (e.g., 'mlp', 'cnn', 'lstm')
        input_features: Number of input features
        num_classes: Number of output classes
        **kwargs: Additional model-specific parameters
    
    Returns:
        Model instance
    
    Raises:
        ValueError: If model_name is not in registry
    """
    model_name = model_name.lower()
    
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: '{model_name}'. Available models: {available}"
        )
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(input_features=input_features, num_classes=num_classes, **kwargs)


def list_available_models():
    """List all available model architectures."""
    return list(MODEL_REGISTRY.keys())

