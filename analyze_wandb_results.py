"""
Wandb Results Analyzer
======================
This script helps analyze wandb training results and provides suggestions
for improving model performance.

Usage:
    python analyze_wandb_results.py
    OR
    python analyze_wandb_results.py --project DL-CIS2018 --run <run_id>
"""

import wandb
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def analyze_training_metrics(run: wandb.run) -> Dict:
    """
    Analyze training metrics from a wandb run.
    
    Args:
        run: Wandb run object
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'run_name': run.name,
        'config': run.config,
        'metrics': {},
        'suggestions': []
    }
    
    # Get history (all logged metrics)
    history = run.history()
    
    if history.empty:
        analysis['suggestions'].append("⚠ No metrics found in this run")
        return analysis
    
    # Extract key metrics
    metrics_to_check = ['Train Loss', 'Val Loss', 'Learning Rate', 'Accuracy']
    available_metrics = [m for m in metrics_to_check if m in history.columns]
    
    if not available_metrics:
        analysis['suggestions'].append("⚠ No standard metrics found. Check metric names.")
        return analysis
    
    # Analyze each metric
    for metric in available_metrics:
        if metric in history.columns:
            values = history[metric].dropna()
            if len(values) > 0:
                analysis['metrics'][metric] = {
                    'initial': float(values.iloc[0]),
                    'final': float(values.iloc[-1]),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'trend': 'improving' if (metric.endswith('Loss') and values.iloc[-1] < values.iloc[0]) 
                             or (metric == 'Accuracy' and values.iloc[-1] > values.iloc[0])
                             else 'degrading'
                }
    
    # Generate suggestions based on metrics
    generate_suggestions(analysis, history)
    
    return analysis


def generate_suggestions(analysis: Dict, history) -> None:
    """
    Generate suggestions based on training metrics.
    
    Args:
        analysis: Analysis dictionary to update
        history: Wandb history dataframe
    """
    suggestions = analysis['suggestions']
    metrics = analysis['metrics']
    config = analysis['config']
    
    # Check training loss
    if 'Train Loss' in metrics:
        train_metrics = metrics['Train Loss']
        if train_metrics['trend'] == 'degrading':
            suggestions.append("❌ Train Loss is increasing - Model may be overfitting or learning rate too high")
        elif train_metrics['final'] > train_metrics['initial'] * 0.9:
            suggestions.append("⚠ Train Loss not decreasing much - Consider increasing learning rate or training longer")
        elif train_metrics['final'] < 0.1:
            suggestions.append("✓ Train Loss is very low - Model is learning well")
    
    # Check validation loss
    if 'Val Loss' in metrics:
        val_metrics = metrics['Val Loss']
        if val_metrics['trend'] == 'degrading':
            suggestions.append("❌ Val Loss is increasing - Model is overfitting! Add dropout or reduce model complexity")
        elif val_metrics['final'] > val_metrics['min'] * 1.5:
            suggestions.append("⚠ Val Loss increased significantly from minimum - Early stopping may have triggered too late")
    
    # Check train vs val loss gap
    if 'Train Loss' in metrics and 'Val Loss' in metrics:
        train_final = metrics['Train Loss']['final']
        val_final = metrics['Val Loss']['final']
        gap = val_final - train_final
        
        if gap > 0.5:
            suggestions.append("⚠ Large gap between Train and Val Loss - Model is overfitting. Suggestions:")
            suggestions.append("  - Increase dropout rate")
            suggestions.append("  - Add more regularization")
            suggestions.append("  - Reduce model complexity")
            suggestions.append("  - Use data augmentation")
        elif gap < -0.1:
            suggestions.append("⚠ Val Loss lower than Train Loss - This is unusual, check data leakage or validation set size")
        else:
            suggestions.append("✓ Train and Val Loss are well balanced")
    
    # Check learning rate
    if 'Learning Rate' in metrics:
        lr_metrics = metrics['Learning Rate']
        if lr_metrics['final'] < lr_metrics['initial'] * 0.01:
            suggestions.append("✓ Learning rate reduced significantly - Scheduler is working")
        elif lr_metrics['final'] == lr_metrics['initial']:
            suggestions.append("⚠ Learning rate never changed - Scheduler may not be working or loss not improving")
    
    # Check accuracy
    if 'Accuracy' in metrics:
        acc_metrics = metrics['Accuracy']
        if acc_metrics['final'] > 0.95:
            suggestions.append("✓ Excellent accuracy (>95%)!")
        elif acc_metrics['final'] > 0.90:
            suggestions.append("✓ Good accuracy (90-95%)")
        elif acc_metrics['final'] < 0.80:
            suggestions.append("⚠ Accuracy below 80% - Consider:")
            suggestions.append("  - Train for more epochs")
            suggestions.append("  - Adjust learning rate")
            suggestions.append("  - Check class imbalance")
            suggestions.append("  - Review model architecture")
    
    # Check hyperparameters
    if 'learning_rate' in config:
        lr = config['learning_rate']
        if lr > 0.01:
            suggestions.append("⚠ Learning rate seems high (>0.01) - Consider reducing to 0.001 or lower")
        elif lr < 0.00001:
            suggestions.append("⚠ Learning rate seems very low (<0.00001) - Training may be too slow")
    
    if 'batch_size' in config:
        batch_size = config['batch_size']
        if batch_size < 32:
            suggestions.append("⚠ Small batch size (<32) - Consider increasing for more stable training")
        elif batch_size > 512:
            suggestions.append("⚠ Very large batch size (>512) - May need to adjust learning rate accordingly")
    
    if 'num_epochs' in config:
        epochs = config['num_epochs']
        if epochs < 10:
            suggestions.append("⚠ Few epochs (<10) - Model may need more training")
        elif epochs > 100:
            suggestions.append("⚠ Many epochs (>100) - Check for overfitting")


def print_analysis(analysis: Dict) -> None:
    """
    Print analysis results in a readable format.
    
    Args:
        analysis: Analysis dictionary
    """
    print("="*70)
    print("WANDB TRAINING RESULTS ANALYSIS")
    print("="*70)
    print(f"\nRun Name: {analysis['run_name']}")
    print(f"\nConfiguration:")
    for key, value in analysis['config'].items():
        print(f"  {key}: {value}")
    
    print(f"\n{'='*70}")
    print("METRICS SUMMARY")
    print("="*70)
    
    for metric_name, metric_data in analysis['metrics'].items():
        print(f"\n{metric_name}:")
        print(f"  Initial: {metric_data['initial']:.6f}")
        print(f"  Final:   {metric_data['final']:.6f}")
        print(f"  Min:     {metric_data['min']:.6f}")
        print(f"  Max:     {metric_data['max']:.6f}")
        print(f"  Trend:   {metric_data['trend']}")
    
    print(f"\n{'='*70}")
    print("SUGGESTIONS & RECOMMENDATIONS")
    print("="*70)
    
    for i, suggestion in enumerate(analysis['suggestions'], 1):
        print(f"{i}. {suggestion}")
    
    print("\n" + "="*70)


def main():
    """Main function to analyze wandb results."""
    parser = argparse.ArgumentParser(description='Analyze wandb training results')
    parser.add_argument('--project', type=str, default='DL-CIS2018', 
                       help='Wandb project name')
    parser.add_argument('--run', type=str, default=None,
                       help='Specific run ID to analyze (default: latest run)')
    parser.add_argument('--all', action='store_true',
                       help='Analyze all runs in the project')
    
    args = parser.parse_args()
    
    # Initialize wandb API
    api = wandb.Api()
    
    try:
        if args.all:
            # Analyze all runs
            runs = api.runs(args.project)
            print(f"Found {len(runs)} runs in project '{args.project}'\n")
            
            for run in runs:
                print(f"\n{'='*70}")
                analysis = analyze_training_metrics(run)
                print_analysis(analysis)
        else:
            # Analyze specific or latest run
            if args.run:
                run = api.run(f"{args.project}/{args.run}")
            else:
                # Get latest run
                runs = api.runs(args.project, order="-created_at", per_page=1)
                if not runs:
                    print(f"❌ No runs found in project '{args.project}'")
                    return
                run = runs[0]
            
            analysis = analyze_training_metrics(run)
            print_analysis(analysis)
            
    except Exception as e:
        print(f"❌ Error analyzing wandb results: {e}")
        print("\nMake sure you're logged in to wandb:")
        print("  wandb login")
        print("\nOr provide the project name:")
        print(f"  python analyze_wandb_results.py --project DL-CIS2018")


if __name__ == "__main__":
    main()

