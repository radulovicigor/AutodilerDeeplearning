"""
Visualization utilities for generating training plots
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import json


def generate_training_plots(
    metrics: Dict[str, Any],
    task: str,
    model_type: str,
    output_dir: Path,
):
    """
    Generate all training visualization plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('dark_background')
    sns.set_palette("husl")
    
    if task == 'regression':
        # Predicted vs Actual scatter
        if 'predictions' in metrics:
            _plot_pred_vs_actual(metrics['predictions'], output_dir / 'pred_vs_actual.png')
            _plot_residuals(metrics['predictions'], output_dir / 'residuals.png')
    
    else:  # classification
        # Confusion matrix
        if 'test' in metrics and 'confusion_matrix' in metrics['test']:
            _plot_confusion_matrix(
                metrics['test']['confusion_matrix'],
                metrics['test']['labels'],
                output_dir / 'confusion_matrix.png'
            )
    
    # Feature importance (for all models)
    if 'feature_importance' in metrics:
        _plot_feature_importance(metrics['feature_importance'], output_dir / 'feature_importance.png')
    
    # Learning curves (for MLP)
    if 'history' in metrics:
        _plot_learning_curves(metrics['history'], task, output_dir / 'learning_curves.png')
    
    plt.close('all')


def _plot_pred_vs_actual(predictions: Dict, output_path: Path):
    """Plot predicted vs actual values"""
    y_true = np.array(predictions['y_test_true'])
    y_pred = np.array(predictions['y_test_pred'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, c='#00ff88', edgecolors='white', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Price (€)', fontsize=12)
    ax.set_ylabel('Predicted Price (€)', fontsize=12)
    ax.set_title('Predicted vs Actual Prices', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e')
    plt.close()


def _plot_residuals(predictions: Dict, output_path: Path):
    """Plot residuals histogram"""
    residuals = np.array(predictions['residuals'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(residuals, bins=50, color='#00ff88', edgecolor='white', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Residual (Actual - Predicted)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Residuals Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals vs Predicted
    y_pred = np.array(predictions['y_test_pred'])
    axes[1].scatter(y_pred, residuals, alpha=0.5, c='#ff6b6b', edgecolors='white', linewidth=0.5)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Price (€)', fontsize=12)
    axes[1].set_ylabel('Residual', fontsize=12)
    axes[1].set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e')
    plt.close()


def _plot_confusion_matrix(cm: List[List[int]], labels: List[str], output_path: Path):
    """Plot confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm_array = np.array(cm)
    
    sns.heatmap(
        cm_array,
        annot=True,
        fmt='d',
        cmap='YlGnBu',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e')
    plt.close()


def _plot_feature_importance(importance: Dict[str, float], output_path: Path, top_n: int = 20):
    """Plot feature importance bar chart"""
    # Sort and take top N
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    bars = ax.barh(range(len(features)), values, color=colors)
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e')
    plt.close()


def _plot_learning_curves(history: Dict[str, List], task: str, output_path: Path):
    """Plot learning curves (loss and metrics over epochs)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curves', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metric curves
    if task == 'regression' and 'train_mae' in history:
        axes[1].plot(epochs, history['train_mae'], 'b-', label='Train MAE', linewidth=2)
        axes[1].plot(epochs, history['val_mae'], 'r-', label='Val MAE', linewidth=2)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('MAE Curves', fontsize=14, fontweight='bold')
    elif 'train_acc' in history:
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate subplot (if available)
    if 'learning_rate' in history:
        # Add as secondary axis
        ax2 = axes[0].twinx()
        ax2.plot(epochs, history['learning_rate'], 'g--', alpha=0.5, label='LR')
        ax2.set_ylabel('Learning Rate', fontsize=10, color='green')
        ax2.tick_params(axis='y', labelcolor='green')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e')
    plt.close()


def get_plot_paths(plots_dir: str) -> Dict[str, str]:
    """Get paths to all generated plots"""
    plots_dir = Path(plots_dir)
    plots = {}
    
    for plot_file in plots_dir.glob('*.png'):
        plots[plot_file.stem] = str(plot_file)
    
    return plots
