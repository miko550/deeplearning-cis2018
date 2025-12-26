"""
Training Module
===============
This module contains functions for training neural network models.
"""

import torch
from tqdm import tqdm
import wandb
from test import evaluate_model


def train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epochs):
    """
    Trains the model for a given number of epochs.
    
    Args:
        model: The model to train.
        train_loader: The training data loader.
        val_loader: The validation data loader.
        device: The device to train on.
        criterion: The loss function.
        optimizer: The optimizer.
        scheduler: The scheduler.
        num_epochs: The number of epochs to train for.

    Returns:
        The trained model.
    """
    best_loss = float('inf')
    patience = 5
    no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

            data, target = data.to(device), target.to(device).long()
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, _, _ = evaluate_model(model, val_loader, device, criterion)

        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'DL-CIS2018.pth')
            print(f"Model saved to DL-CIS2018.pth")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Epoch {epoch+1}/{num_epochs}, best_val={best_loss:.6f}")
                break

        scheduler.step(val_loss)
        wandb.log({
            "Train Loss": train_loss, 
            "Val Loss": val_loss,
            "Learning Rate": optimizer.param_groups[0]['lr']
        })
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    model.load_state_dict(torch.load('DL-CIS2018.pth', map_location=device))
    return model
