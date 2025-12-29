"""
Quick Test Script for Models
============================
Test if all models can be instantiated correctly.

Usage:
    python test_models.py
"""

import torch
from models import create_model, list_available_models

def test_all_models():
    """Test that all models can be created."""
    print("Testing Model Creation...")
    print("="*60)
    
    input_features = 78  # Example: adjust to your feature count
    num_classes = 7      # Example: adjust to your class count
    
    available_models = list_available_models()
    print(f"Available models: {available_models}\n")
    
    results = {}
    
    for model_name in available_models:
        try:
            print(f"Testing {model_name.upper()}...", end=" ")
            model = create_model(
                model_name=model_name,
                input_features=input_features,
                num_classes=num_classes
            )
            
            # Test forward pass
            dummy_input = torch.randn(2, input_features)  # batch_size=2
            output = model(dummy_input)
            
            assert output.shape == (2, num_classes), f"Wrong output shape: {output.shape}"
            
            print("✓ PASSED")
            results[model_name] = "PASSED"
            
            # Print model info
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {num_params:,}")
            print(f"  Output shape: {output.shape}\n")
            
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            results[model_name] = f"FAILED: {e}"
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for r in results.values() if r == "PASSED")
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for model_name, result in results.items():
        status = "✓" if result == "PASSED" else "✗"
        print(f"  {status} {model_name}: {result}")
    
    return results


if __name__ == "__main__":
    test_all_models()

