"""
Data Preprocessing Module for PyTorch
======================================
This module preprocesses numpy arrays containing features and labels,
applies standardization, and creates PyTorch DataLoaders for training.
"""

import numpy as np
import pickle
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader


def preprocess(train, test, val, batch_size, scaler_save_path=None):
    """
    Preprocess data arrays and create PyTorch DataLoaders.
    
    This function:
    1. Fits a StandardScaler on training features (normalizes to mean=0, std=1)
    2. Optionally saves the scaler for later use
    3. Transforms all datasets (train/test/val) using the fitted scaler
    4. Separates features (all columns except last) from labels (last column)
    5. Converts numpy arrays to PyTorch tensors
    6. Creates PyTorch DataLoaders for efficient batch processing
    
    Args:
        train: Training data numpy array (shape: [n_samples, n_features+1])
               Last column contains labels, all other columns are features
        test: Test data numpy array (same format as train)
        val: Validation data numpy array (same format as train)
        batch_size: Number of samples per batch for DataLoaders
        scaler_save_path: Optional path to save the fitted scaler (for reproducibility)
    
    Returns:
        Tuple of (train_loader, test_loader, val_loader): PyTorch DataLoaders
        - train_loader: Shuffled batches for training
        - test_loader: Non-shuffled batches for testing
        - val_loader: Non-shuffled batches for validation
    """
    
    # Step 1: Fit StandardScaler on training features only
    # StandardScaler normalizes features: (x - mean) / std
    # This ensures features have mean=0 and std=1, which helps neural network training
    # train[:, :-1] extracts all columns except the last one (features only)
    scaler = StandardScaler().fit(train[:, :-1])
    
    # Step 2: Optionally save the scaler to disk
    # This allows us to use the same scaling parameters later (e.g., for inference)
    if scaler_save_path:
        with open(scaler_save_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_save_path}")
    
    # Step 3: Transform all datasets using the fitted scaler
    # Apply the same scaling (mean/std from training) to train, test, and val
    # This ensures consistent scaling across all datasets
    # train[:, :-1] = all features (all columns except last)
    train_x = scaler.transform(train[:, :-1])  # Scaled training features
    test_x = scaler.transform(test[:, :-1])    # Scaled test features
    val_x = scaler.transform(val[:, :-1])      # Scaled validation features
    
    # Step 4: Extract labels from the last column of each dataset
    # train[:, -1] extracts the last column (labels)
    # Convert to int64 (long) for classification tasks (PyTorch expects integer labels)
    train_y = train[:, -1].astype(np.int64)   # Training labels
    test_y = test[:, -1].astype(np.int64)      # Test labels
    val_y = val[:, -1].astype(np.int64)        # Validation labels
    
    # Step 5: Create PyTorch TensorDatasets
    # Convert numpy arrays to PyTorch tensors and pair features with labels
    # .float() converts features to float32 (required for neural networks)
    # TensorDataset creates (feature, label) pairs
    train_dataset = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y))
    test_dataset = TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y))
    val_dataset = TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y))
    
    # Step 6: Create PyTorch DataLoaders
    # DataLoaders handle batching, shuffling, and efficient data loading
    # shuffle=True for training (randomizes order each epoch)
    # shuffle=False for test/val (deterministic order for evaluation)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    
    return train_loader, test_loader, val_loader
