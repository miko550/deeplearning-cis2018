"""
Main Entry Point
================
This is the main script that orchestrates the entire machine learning pipeline:
1. Load configuration
2. Load and preprocess data
3. Create model
4. Train model
5. Test model

Usage:
    python main.py
"""

import torch
import numpy as np
import yaml
from torchsummary import summary
import wandb
import torch.optim as optim
from datetime import datetime

from preprocess import preprocess
from models import create_model, list_available_models
from train import train_model
from test import test_and_report
from utils import calculate_class_weights, analyze_class_distribution


def main():
    """
    Main function that orchestrates the entire ML pipeline.
    """
    print("="*60)
    print("Deep Learning for Intrusion Detection System (IDS)")
    print("="*60)
    
    # Step 1: Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/6] Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    
    # Step 2: Load configuration
    print(f"\n[2/7] Loading configuration...")
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print(f"  ✓ Configuration loaded from config.yaml")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Learning rate: {config['learning_rate']}")
        print(f"  Number of epochs: {config['num_epochs']}")
        # Note: class_names will be loaded from generated file, not config
    except FileNotFoundError:
        print("  ❌ ERROR: config.yaml not found!")
        return
    except Exception as e:
        print(f"  ❌ ERROR loading config: {e}")
        return
    
    # Step 3: Load data and class names
    print(f"\n[3/7] Loading data and class names...")
    try:
        train = np.load('data/train.npy')
        test = np.load('data/test.npy')
        val = np.load('data/val.npy')
        
        print(f"  ✓ Data loaded successfully")
        print(f"  Train shape: {train.shape}")
        print(f"  Test shape: {test.shape}")
        print(f"  Val shape: {val.shape}")
    except FileNotFoundError as e:
        print(f"  ❌ ERROR: Data files not found: {e}")
        print("  Make sure to run preprocess_csv.py first to generate .npy files")
        return
    except Exception as e:
        print(f"  ❌ ERROR loading data: {e}")
        return
    
    # Load class names from generated file
    print(f"\n[4/7] Loading class names...")
    try:
        class_names = np.load('data/class_names.npy', allow_pickle=True)
        class_names = [str(name) for name in class_names]
        print(f"  ✓ Class names loaded from data/class_names.npy")
        print(f"  ✓ Found {len(class_names)} classes:")
        for idx, name in enumerate(class_names):
            print(f"    Index {idx}: {name}")
    except FileNotFoundError:
        print(f"  ❌ ERROR: data/class_names.npy not found!")
        print("  Make sure to run preprocess_csv.py first to generate class_names.npy")
        return
    except Exception as e:
        print(f"  ❌ ERROR loading class names: {e}")
        return
    
    # Step 5: Preprocess data
    print(f"\n[5/7] Preprocessing data...")
    try:
        train_loader, test_loader, val_loader = preprocess(
            train, 
            test, 
            val, 
            config['batch_size'], 
            scaler_save_path='scaler.pkl'
        )
        print(f"  ✓ Data preprocessing complete")
    except Exception as e:
        print(f"  ❌ ERROR during preprocessing: {e}")
        return
    
    # Step 6: Create model
    print(f"\n[6/7] Creating model...")
    try:
        num_classes = len(class_names)
        num_features = train.shape[1] - 1
        
        # Get model name from config
        model_name = config.get('model_name', 'mlp')
        model_params = config.get('model_params', {})
        
        print(f"  Available models: {', '.join(list_available_models())}")
        print(f"  Selected model: {model_name}")
        
        # Create model using factory function
        model = create_model(
            model_name=model_name,
            input_features=num_features,
            num_classes=num_classes,
            **model_params
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
        num_epochs = config['num_epochs']
        
        # Calculate class weights for imbalanced dataset
        train_labels = train[:, -1].astype(int)
        class_weights = calculate_class_weights(train_labels, method='balanced')
        class_weights = class_weights.to(device)
        
        # Use weighted loss to handle class imbalance
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        print(f"  Using weighted CrossEntropyLoss to handle class imbalance")
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        
        print(summary(model, input_size=(num_features,), device=device))
        print(f"  ✓ Model created successfully")
    except Exception as e:
        print(f"  ❌ ERROR creating model: {e}")
        return
    
    # Step 7: Initialize wandb
    print(f"\n[7/7] Initializing experiment tracking...")
    try:
        # Create unique run name with timestamp and key hyperparameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}_bs{config['batch_size']}_lr{config['learning_rate']}_ep{config['num_epochs']}"
        
        wandb.init(
            project="DL-CIS2018",
            name=run_name,
            config={
                "model_name": model_name,
                "batch_size": config['batch_size'],
                "num_epochs": config['num_epochs'],
                "learning_rate": config['learning_rate'],
                "optimizer": "AdamW",
                "scheduler": "ReduceLROnPlateau",
                **model_params
            }
        )
        print(f"  ✓ Wandb initialized with run name: {run_name}")
        wandb.watch(model, log="all")
        print(f"  ✓ Wandb initialized")
    except Exception as e:
        print(f"  ⚠ WARNING: Could not initialize wandb: {e}")
        print(f"  Continuing without wandb logging...")
    
    # Train model
    print(f"\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)
    try:
        model = train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epochs)
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test model
    print(f"\n" + "="*60)
    print("TESTING PHASE")
    print("="*60)
    try:
        test_and_report(model, test_loader, device, class_names)
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final summary
    print(f"\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"✓ Model saved to: DL-CIS2018.pth")
    print(f"✓ Scaler saved to: scaler.pkl")
    print("="*60)


if __name__ == "__main__":
    main()
