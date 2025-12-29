"""
Test script to validate notebook code
"""
import sys
import traceback

def test_imports():
    """Test all imports from the notebook"""
    print("Testing imports...")
    try:
        import torch
        import numpy as np
        import yaml
        from datetime import datetime
        import torch.optim as optim
        from torchsummary import summary
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import wandb
        from tqdm import tqdm
        from preprocess import preprocess
        from model import IDSModel
        from train import train_model
        from test import test_and_report, evaluate_model
        from utils import calculate_class_weights
        print("✓ All imports successful!")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False

def test_config():
    """Test config loading"""
    print("\nTesting config loading...")
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        assert 'batch_size' in config
        assert 'learning_rate' in config
        assert 'num_epochs' in config
        print("✓ Config loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data file loading"""
    print("\nTesting data loading...")
    try:
        import numpy as np
        import os
        
        files = ['data/train.npy', 'data/test.npy', 'data/val.npy', 'data/class_names.npy']
        missing = []
        for f in files:
            if not os.path.exists(f):
                missing.append(f)
        
        if missing:
            print(f"⚠ Missing files: {missing}")
            print("  Run: uv run data/preprocess_csv_v2.py")
            return False
        
        train = np.load('data/train.npy')
        test = np.load('data/test.npy')
        val = np.load('data/val.npy')
        class_names = np.load('data/class_names.npy', allow_pickle=True)
        
        assert train.shape[1] > 1, "Train data should have features + label"
        assert test.shape[1] > 1, "Test data should have features + label"
        assert val.shape[1] > 1, "Val data should have features + label"
        assert len(class_names) > 0, "Should have at least one class"
        
        print(f"✓ Data loaded successfully!")
        print(f"  Train shape: {train.shape}")
        print(f"  Test shape: {test.shape}")
        print(f"  Val shape: {val.shape}")
        print(f"  Classes: {len(class_names)}")
        return True
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    try:
        import torch
        import numpy as np
        from model import IDSModel
        
        # Load a small sample to get dimensions
        try:
            train = np.load('data/train.npy')
            class_names = np.load('data/class_names.npy', allow_pickle=True)
            num_features = train.shape[1] - 1
            num_classes = len(class_names)
        except:
            # Use dummy values if data doesn't exist
            num_features = 78
            num_classes = 7
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = IDSModel(input_features=num_features, num_classes=num_classes).to(device)
        
        # Test forward pass
        dummy_input = torch.randn(2, num_features).to(device)
        output = model(dummy_input)
        
        assert output.shape == (2, num_classes), f"Expected (2, {num_classes}), got {output.shape}"
        print(f"✓ Model created and tested successfully!")
        print(f"  Input features: {num_features}")
        print(f"  Output classes: {num_classes}")
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        traceback.print_exc()
        return False

def test_class_weights():
    """Test class weights calculation"""
    print("\nTesting class weights calculation...")
    try:
        import numpy as np
        from utils import calculate_class_weights
        
        # Create dummy labels
        dummy_labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        weights = calculate_class_weights(dummy_labels, method='balanced')
        
        assert len(weights) > 0, "Should return weights"
        print(f"✓ Class weights calculated successfully!")
        return True
    except Exception as e:
        print(f"✗ Class weights error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("Notebook Code Validation")
    print("="*60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Data Loading", test_data_loading()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Class Weights", test_class_weights()))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(r for _, r in results)
    if all_passed:
        print("\n✓ All tests passed! Notebook should run correctly.")
    else:
        print("\n⚠ Some tests failed. Fix issues before running notebook.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

