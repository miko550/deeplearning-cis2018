"""
Model Comparison Script
=======================
This script trains and compares multiple model architectures to find the best one.

Usage:
    python compare_models.py
"""

import torch
import numpy as np
import yaml
from torchsummary import summary
import wandb
import torch.optim as optim
from datetime import datetime

from preprocess import preprocess
from models import list_available_models, create_model
from train import train_model
from test import test_and_report
from utils import calculate_class_weights


def compare_models(model_names=None, config_path='config.yaml'):
    """
    Compare multiple model architectures.
    
    Args:
        model_names: List of model names to compare (None = all models)
        config_path: Path to config file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    train = np.load('data/train.npy')
    test = np.load('data/test.npy')
    val = np.load('data/val.npy')
    class_names = np.load('data/class_names.npy', allow_pickle=True)
    class_names = [str(name) for name in class_names]
    
    # Preprocess data
    train_loader, test_loader, val_loader = preprocess(
        train, test, val, 
        config['batch_size'], 
        scaler_save_path='scaler.pkl'
    )
    
    # Get models to compare
    if model_names is None:
        model_names = list_available_models()
    
    print(f"\n{'='*70}")
    print(f"COMPARING {len(model_names)} MODELS")
    print(f"{'='*70}\n")
    
    results = {}
    
    for model_name in model_names:
        print(f"\n{'='*70}")
        print(f"Training Model: {model_name.upper()}")
        print(f"{'='*70}\n")
        
        try:
            # Create model
            num_features = train.shape[1] - 1
            num_classes = len(class_names)
            model_params = config.get('model_params', {})
            
            model = create_model(
                model_name=model_name,
                input_features=num_features,
                num_classes=num_classes,
                **model_params
            ).to(device)
            
            # Setup training
            optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
            train_labels = train[:, -1].astype(int)
            class_weights = calculate_class_weights(train_labels, method='balanced').to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
            
            # Initialize wandb for this model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"compare_{model_name}_{timestamp}"
            
            wandb.init(
                project="DL-CIS2018-Comparison",
                name=run_name,
                config={
                    "model_name": model_name,
                    "batch_size": config['batch_size'],
                    "learning_rate": config['learning_rate'],
                    "num_epochs": config['num_epochs'],
                    **model_params
                },
                reinit=True
            )
            wandb.watch(model, log="all")
            
            # Train
            trained_model = train_model(
                model, train_loader, val_loader, device,
                criterion, optimizer, scheduler, config['num_epochs']
            )
            
            # Test
            test_accuracy = test_and_report(
                trained_model, test_loader, device, class_names
            )
            
            # Save results
            results[model_name] = {
                'test_accuracy': test_accuracy,
                'model': trained_model
            }
            
            print(f"\n✓ {model_name.upper()} - Test Accuracy: {test_accuracy*100:.2f}%")
            
            wandb.finish()
            
        except Exception as e:
            print(f"\n❌ Error training {model_name}: {e}")
            results[model_name] = {'error': str(e)}
            wandb.finish()
            continue
    
    # Print comparison summary
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}\n")
    
    sorted_results = sorted(
        [(name, res.get('test_accuracy', 0)) for name, res in results.items() 
         if 'test_accuracy' in res],
        key=lambda x: x[1],
        reverse=True
    )
    
    print("Ranking by Test Accuracy:")
    for rank, (model_name, accuracy) in enumerate(sorted_results, 1):
        print(f"  {rank}. {model_name.upper()}: {accuracy*100:.2f}%")
    
    if sorted_results:
        best_model = sorted_results[0][0]
        print(f"\n🏆 Best Model: {best_model.upper()} ({sorted_results[0][1]*100:.2f}%)")
    
    return results


if __name__ == "__main__":
    # Compare specific models or all models
    # compare_models(['mlp', 'cnn', 'lstm'])  # Compare specific models
    compare_models()  # Compare all models

