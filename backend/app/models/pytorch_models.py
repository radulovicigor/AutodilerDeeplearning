"""
PyTorch MLP models for regression and classification
Configurable architecture with visualization support
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Callable
from pathlib import Path
import json
import time

from app.models.sklearn_models import evaluate_regression, evaluate_classification


# Check for GPU
def get_device(use_gpu: bool = True) -> torch.device:
    """Get available device (CUDA if available and requested)"""
    if use_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# Activation function mapping
ACTIVATIONS = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'elu': nn.ELU,
    'selu': nn.SELU,
}


class MLPRegressor(nn.Module):
    """
    Configurable MLP for regression
    """
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = [128, 64, 32],
        activation: str = 'relu',
        dropout: float = 0.2,
        batch_norm: bool = True,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.dropout_rate = dropout
        self.use_batch_norm = batch_norm
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(ACTIVATIONS[activation]())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # For storing activations during forward pass
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activations"""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach().cpu().numpy()
            return hook
        
        layer_idx = 0
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                layer.register_forward_hook(get_activation(f'layer_{layer_idx}'))
                layer_idx += 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get weight matrices for visualization"""
        weights = {}
        layer_idx = 0
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                weights[f'layer_{layer_idx}'] = {
                    'weight': layer.weight.detach().cpu().numpy(),
                    'bias': layer.bias.detach().cpu().numpy(),
                }
                layer_idx += 1
        return weights
    
    def get_architecture(self) -> Dict[str, Any]:
        """Get architecture description for visualization"""
        arch = {
            'input_size': self.input_size,
            'layers': [],
            'activation': self.activation_name,
            'dropout': self.dropout_rate,
        }
        
        # Add input layer
        arch['layers'].append({
            'name': 'input',
            'neurons': self.input_size,
            'type': 'input'
        })
        
        # Add hidden layers
        for i, size in enumerate(self.hidden_layers):
            arch['layers'].append({
                'name': f'hidden_{i}',
                'neurons': size,
                'type': 'hidden',
                'activation': self.activation_name,
            })
        
        # Add output layer
        arch['layers'].append({
            'name': 'output',
            'neurons': 1,
            'type': 'output'
        })
        
        return arch


class MLPClassifier(nn.Module):
    """
    Configurable MLP for classification
    """
    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        hidden_layers: List[int] = [128, 64, 32],
        activation: str = 'relu',
        dropout: float = 0.2,
        batch_norm: bool = True,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.dropout_rate = dropout
        self.use_batch_norm = batch_norm
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(ACTIVATIONS[activation]())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # For storing activations
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activations"""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach().cpu().numpy()
            return hook
        
        layer_idx = 0
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                layer.register_forward_hook(get_activation(f'layer_{layer_idx}'))
                layer_idx += 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get weight matrices for visualization"""
        weights = {}
        layer_idx = 0
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                weights[f'layer_{layer_idx}'] = {
                    'weight': layer.weight.detach().cpu().numpy(),
                    'bias': layer.bias.detach().cpu().numpy(),
                }
                layer_idx += 1
        return weights
    
    def get_architecture(self) -> Dict[str, Any]:
        """Get architecture description for visualization"""
        arch = {
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'layers': [],
            'activation': self.activation_name,
            'dropout': self.dropout_rate,
        }
        
        arch['layers'].append({
            'name': 'input',
            'neurons': self.input_size,
            'type': 'input'
        })
        
        for i, size in enumerate(self.hidden_layers):
            arch['layers'].append({
                'name': f'hidden_{i}',
                'neurons': size,
                'type': 'hidden',
                'activation': self.activation_name,
            })
        
        arch['layers'].append({
            'name': 'output',
            'neurons': self.num_classes,
            'type': 'output'
        })
        
        return arch


def train_mlp_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    log_transform: bool = False,
    hidden_layers: List[int] = [128, 64, 32],
    activation: str = 'relu',
    dropout: float = 0.2,
    batch_norm: bool = True,
    learning_rate: float = 0.001,
    optimizer_type: str = 'adam',
    epochs: int = 100,
    batch_size: int = 32,
    early_stopping_patience: int = 30,
    use_gpu: bool = True,
    progress_callback: Optional[Callable] = None,
    **kwargs
) -> Tuple[MLPRegressor, Dict[str, Any]]:
    """
    Train MLP Regressor with full tracking for visualization
    """
    device = get_device(use_gpu)
    print(f"Training on: {device}")
    
    # Ensure arrays and handle NaN values
    X_train = np.array(X_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32).flatten()
    y_val = np.array(y_val, dtype=np.float32).flatten()
    y_test = np.array(y_test, dtype=np.float32).flatten()
    
    # Handle NaN values in input data
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).reshape(-1, 1).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = MLPRegressor(
        input_size=X_train.shape[1],
        hidden_layers=hidden_layers,
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm,
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler - increased patience
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'learning_rate': [],
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss at epoch {epoch}, skipping batch")
                continue
                
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            
            train_outputs = model(X_train_t)
            train_loss = criterion(train_outputs, y_train_t).item()
        
        # Check for NaN
        if np.isnan(val_loss) or np.isnan(train_loss):
            print(f"Warning: NaN loss detected at epoch {epoch}")
            continue
        
        # Calculate MAE
        train_mae = float(torch.mean(torch.abs(train_outputs - y_train_t)).cpu())
        val_mae = float(torch.mean(torch.abs(val_outputs - y_val_t)).cpu())
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Progress callback
        if progress_callback:
            try:
                progress_callback(epoch + 1, epochs, train_loss, val_loss, best_val_loss)
            except InterruptedError:
                print("Training cancelled by user")
                break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    training_time = time.time() - start_time
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train_t).cpu().numpy().flatten()
        y_val_pred = model(X_val_t).cpu().numpy().flatten()
        y_test_pred = model(X_test_t).cpu().numpy().flatten()
    
    # Handle any NaN predictions
    train_mean = float(np.nanmean(y_train)) if len(y_train) > 0 else 0.0
    val_mean = float(np.nanmean(y_val)) if len(y_val) > 0 else train_mean
    test_mean = float(np.nanmean(y_test)) if len(y_test) > 0 else train_mean
    
    y_train_pred = np.nan_to_num(y_train_pred, nan=train_mean)
    y_val_pred = np.nan_to_num(y_val_pred, nan=val_mean)
    y_test_pred = np.nan_to_num(y_test_pred, nan=test_mean)
    
    # Metrics
    metrics = {
        'train': evaluate_regression(y_train, y_train_pred, log_transform),
        'val': evaluate_regression(y_val, y_val_pred, log_transform),
        'test': evaluate_regression(y_test, y_test_pred, log_transform),
        'training_time': training_time,
        'epochs_trained': len(history['train_loss']),
        'best_val_loss': best_val_loss,
    }
    
    # Predictions for visualization - ensure 1D arrays
    y_test_flat = np.array(y_test).flatten()
    y_test_pred_flat = np.array(y_test_pred).flatten()
    
    if log_transform:
        y_test_vis = np.expm1(y_test_flat)
        y_pred_vis = np.expm1(y_test_pred_flat)
    else:
        y_test_vis = y_test_flat
        y_pred_vis = y_test_pred_flat
    
    # Handle any infinities
    y_test_vis = np.nan_to_num(y_test_vis, nan=0.0, posinf=1e10, neginf=0.0)
    y_pred_vis = np.nan_to_num(y_pred_vis, nan=0.0, posinf=1e10, neginf=0.0)
    
    metrics['predictions'] = {
        'y_test_true': y_test_vis.tolist(),
        'y_test_pred': y_pred_vis.tolist(),
        'residuals': (y_test_vis - y_pred_vis).tolist(),
    }
    
    # Learning history
    metrics['history'] = history
    
    # Architecture
    metrics['architecture'] = model.get_architecture()
    
    # Get sample activations (for visualization)
    model.eval()
    with torch.no_grad():
        n_samples = min(64, X_val.shape[0])
        sample_indices = np.random.choice(X_val.shape[0], n_samples, replace=False)
        sample_X = torch.FloatTensor(X_val[sample_indices]).to(device)
        _ = model(sample_X)  # This populates model.activations
        
    metrics['sample_activations'] = {k: v.tolist() for k, v in model.activations.items()}
    
    # Feature importance via gradient sensitivity
    feature_importance = compute_gradient_importance(model, X_val_t, feature_names, device)
    metrics['feature_importance'] = feature_importance
    
    # Move model to CPU for saving
    model = model.cpu()
    
    return model, metrics


def train_mlp_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    hidden_layers: List[int] = [128, 64, 32],
    activation: str = 'relu',
    dropout: float = 0.2,
    batch_norm: bool = True,
    learning_rate: float = 0.001,
    optimizer_type: str = 'adam',
    epochs: int = 100,
    batch_size: int = 32,
    early_stopping_patience: int = 30,
    use_gpu: bool = True,
    progress_callback: Optional[Callable] = None,
    **kwargs
) -> Tuple[MLPClassifier, Dict[str, Any]]:
    """
    Train MLP Classifier with full tracking for visualization
    """
    device = get_device(use_gpu)
    print(f"Training on: {device}")
    
    # Ensure arrays and handle NaN values
    X_train = np.array(X_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Encode labels
    labels = ['budget', 'mid', 'premium']
    label_to_idx = {l: i for i, l in enumerate(labels)}
    
    y_train_encoded = np.array([label_to_idx.get(str(l), 0) for l in y_train])
    y_val_encoded = np.array([label_to_idx.get(str(l), 0) for l in y_val])
    y_test_encoded = np.array([label_to_idx.get(str(l), 0) for l in y_test])
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train_encoded).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val_encoded).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test_encoded).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = MLPClassifier(
        input_size=X_train.shape[1],
        num_classes=len(labels),
        hidden_layers=hidden_layers,
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm,
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rate': [],
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            
            train_outputs = model(X_train_t)
            train_loss = criterion(train_outputs, y_train_t).item()
            
            # Accuracy
            train_acc = (train_outputs.argmax(1) == y_train_t).float().mean().item()
            val_acc = (val_outputs.argmax(1) == y_val_t).float().mean().item()
        
        if np.isnan(val_loss) or np.isnan(train_loss):
            continue
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if progress_callback:
            try:
                progress_callback(epoch + 1, epochs, train_loss, val_loss, best_val_loss)
            except InterruptedError:
                print("Training cancelled by user")
                break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
    
    training_time = time.time() - start_time
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_indices = model(X_train_t).argmax(1).cpu().numpy()
        val_indices = model(X_val_t).argmax(1).cpu().numpy()
        test_indices = model(X_test_t).argmax(1).cpu().numpy()
        
        y_train_pred = [labels[int(i)] for i in train_indices]
        y_val_pred = [labels[int(i)] for i in val_indices]
        y_test_pred = [labels[int(i)] for i in test_indices]
    
    metrics = {
        'train': evaluate_classification(list(y_train), y_train_pred, labels),
        'val': evaluate_classification(list(y_val), y_val_pred, labels),
        'test': evaluate_classification(list(y_test), y_test_pred, labels),
        'training_time': training_time,
        'epochs_trained': len(history['train_loss']),
        'best_val_loss': best_val_loss,
    }
    
    metrics['history'] = history
    metrics['architecture'] = model.get_architecture()
    
    # Sample activations
    model.eval()
    with torch.no_grad():
        n_samples = min(64, X_val.shape[0])
        sample_indices = np.random.choice(X_val.shape[0], n_samples, replace=False)
        sample_X = torch.FloatTensor(X_val[sample_indices]).to(device)
        _ = model(sample_X)
        
    metrics['sample_activations'] = {k: v.tolist() for k, v in model.activations.items()}
    
    # Feature importance
    feature_importance = compute_gradient_importance_classifier(model, X_val_t, feature_names, device)
    metrics['feature_importance'] = feature_importance
    
    model = model.cpu()
    
    return model, metrics


def compute_gradient_importance(
    model: nn.Module,
    X: torch.Tensor,
    feature_names: list,
    device: torch.device
) -> Dict[str, float]:
    """
    Compute feature importance via gradient sensitivity
    """
    model.eval()
    X = X.clone().requires_grad_(True)
    
    outputs = model(X)
    outputs.sum().backward()
    
    # Average absolute gradients across samples
    gradients = X.grad.abs().mean(dim=0).cpu().numpy()
    
    importance = {}
    for name, grad in zip(feature_names, gradients):
        importance[name] = float(grad)
    
    # Normalize
    total = sum(importance.values())
    if total > 0:
        importance = {k: v / total for k, v in importance.items()}
    
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def compute_gradient_importance_classifier(
    model: nn.Module,
    X: torch.Tensor,
    feature_names: list,
    device: torch.device
) -> Dict[str, float]:
    """
    Compute feature importance via gradient sensitivity for classifier
    """
    model.eval()
    X = X.clone().requires_grad_(True)
    
    outputs = model(X)
    # Use sum of all class outputs
    outputs.sum().backward()
    
    gradients = X.grad.abs().mean(dim=0).cpu().numpy()
    
    importance = {}
    for name, grad in zip(feature_names, gradients):
        importance[name] = float(grad)
    
    total = sum(importance.values())
    if total > 0:
        importance = {k: v / total for k, v in importance.items()}
    
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def save_pytorch_model(model: nn.Module, path: Path) -> str:
    """Save PyTorch model to disk"""
    torch.save({
        'state_dict': model.state_dict(),
        'architecture': model.get_architecture(),
    }, path)
    return str(path)


def load_pytorch_model(path: Path, model_class: type) -> nn.Module:
    """Load PyTorch model from disk"""
    checkpoint = torch.load(path, map_location='cpu')
    arch = checkpoint['architecture']
    
    if model_class == MLPRegressor:
        model = MLPRegressor(
            input_size=arch['input_size'],
            hidden_layers=[l['neurons'] for l in arch['layers'][1:-1]],
            activation=arch['activation'],
            dropout=arch['dropout'],
        )
    else:
        model = MLPClassifier(
            input_size=arch['input_size'],
            num_classes=arch['num_classes'],
            hidden_layers=[l['neurons'] for l in arch['layers'][1:-1]],
            activation=arch['activation'],
            dropout=arch['dropout'],
        )
    
    model.load_state_dict(checkpoint['state_dict'])
    return model
