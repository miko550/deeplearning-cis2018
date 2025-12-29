"""
Testing and Evaluation Module
==============================
This module contains functions for evaluating and testing neural network models.
"""

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import wandb


def evaluate_model(model, dataloader, device, criterion=None):
    """
    Evaluates the model on a given dataloader.
    Can compute loss and/or return predictions and labels.

    Args:
        model: The model to evaluate.
        dataloader: The data loader to evaluate on.
        device: The device to evaluate on.
        criterion: The loss function.

    Returns:
        The average loss, all predictions, and all labels.
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for samples, labels in dataloader:
            samples = samples.to(device).float()
            labels = labels.to(device)

            predictions = model(samples)

            if criterion:
                loss = criterion(predictions, labels.long())
                total_loss += loss.item()

            preds = torch.argmax(predictions, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if criterion and len(dataloader) > 0 else 0
    return avg_loss, all_preds, all_labels


def test_and_report(model, test_loader, device, class_names):
    """
    Evaluates a model on the test set and prints a classification report
    and confusion matrix.

    Args:
        model: The model to evaluate.
        test_loader: The test data loader.
        device: The device to evaluate on.
        class_names: The class names.

    Returns:
        The accuracy of the model on the test set.
    """
    print("\n--- Starting Final Test ---")
    model.eval()

    all_preds, all_labels = [], []
    with torch.inference_mode():
        for samples, labels in test_loader:
            samples = samples.to(device)
            labels = labels.to(device).long()
            
            predictions = model(samples)
            
            preds = torch.argmax(predictions, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Final Test Accuracy: {acc*100:.2f}%")

    wandb.log({"Accuracy": acc*100})

    print('--- Classification Report ---')
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))
    
    print('--- Confusion Matrix ---')
    print(confusion_matrix(all_labels, all_preds))
    return acc

