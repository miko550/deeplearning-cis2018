"""
Utility Functions
================
Helper functions for handling class imbalance and other utilities.
"""

import numpy as np
import torch


def calculate_class_weights(train_labels, method='balanced'):
    """
    Calculate class weights for handling imbalanced datasets.
    
    Args:
        train_labels: Array of training labels
        method: 'balanced' (sklearn style) or 'inverse' (simple inverse frequency)
    
    Returns:
        torch.Tensor of class weights
    """
    unique, counts = np.unique(train_labels, return_counts=True)
    n_samples = len(train_labels)
    n_classes = len(unique)
    
    if method == 'balanced':
        # sklearn style: n_samples / (n_classes * count_per_class)
        class_weights = n_samples / (n_classes * counts)
    elif method == 'inverse':
        # Simple inverse frequency
        class_weights = 1.0 / counts
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * n_classes
    
    # Create a weight array for all classes (in case some classes are missing)
    max_class = int(train_labels.max()) + 1
    weights_array = np.ones(max_class, dtype=np.float32)
    for class_idx, weight in zip(unique, class_weights):
        weights_array[int(class_idx)] = weight
    
    print(f"\nClass Weights (method={method}):")
    for class_idx, weight in zip(unique, class_weights):
        print(f"  Class {class_idx}: {weight:.4f} (count: {counts[list(unique).index(class_idx)]:,})")
    
    return torch.FloatTensor(weights_array)


def analyze_class_distribution(labels, class_names=None):
    """
    Analyze and print class distribution.
    
    Args:
        labels: Array of labels
        class_names: Optional list of class names
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    print(f"\nClass Distribution (Total: {total:,} samples):")
    print("="*60)
    for class_idx, count in zip(unique, counts):
        percentage = (count / total) * 100
        class_name = class_names[int(class_idx)] if class_names else f"Class {class_idx}"
        print(f"  {class_name:30s}: {count:8,} samples ({percentage:6.2f}%)")
    print("="*60)
    
    # Calculate imbalance ratio
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"Imbalance Ratio: {imbalance_ratio:.2f}x (max/min)")

