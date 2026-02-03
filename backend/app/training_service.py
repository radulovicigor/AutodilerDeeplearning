"""
Training service - orchestrates model training and experiment management
"""
import uuid
import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from threading import Thread, Lock
import traceback

from sqlmodel import Session, select

from app.config import settings
from app.database import engine, Experiment, TrainingJob
from app.data_processing import (
    load_dataset, prepare_data, save_preprocessor,
    augment_data, augment_classification_data,
    NUMERIC_COLS, CATEGORICAL_COLS, TARGET_COL
)
from app.models.sklearn_models import (
    train_linear_regression,
    train_random_forest_regressor,
    train_xgboost_regressor,
    train_logistic_regression,
    train_random_forest_classifier,
    save_sklearn_model,
)
from app.models.pytorch_models import (
    train_mlp_regressor,
    train_mlp_classifier,
    save_pytorch_model,
)
from app.visualization import generate_training_plots


# ============== REPRODUCIBILITY ==============
def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.
    Called at the start of each training job.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ============== RATE LIMITING ==============
MAX_CONCURRENT_TRAINING = 2
_training_lock = Lock()
_active_training_count = 0


def acquire_training_slot() -> bool:
    """Try to acquire a training slot. Returns False if limit reached."""
    global _active_training_count
    with _training_lock:
        if _active_training_count >= MAX_CONCURRENT_TRAINING:
            return False
        _active_training_count += 1
        return True


def release_training_slot():
    """Release a training slot when job completes."""
    global _active_training_count
    with _training_lock:
        _active_training_count = max(0, _active_training_count - 1)


def get_active_training_count() -> int:
    """Get current number of active training jobs."""
    with _training_lock:
        return _active_training_count


# In-memory job store for simplicity
JOBS: Dict[str, TrainingJob] = {}

# Store for cancellation flags
CANCEL_FLAGS: Dict[str, bool] = {}


class TrainingLimitExceeded(Exception):
    """Raised when max concurrent training limit is reached"""
    pass


def create_training_job(
    task: str,
    model_type: str,
    config: Dict[str, Any],
    use_gpu: bool = True,
) -> str:
    """
    Create a new training job and start it in background
    
    Returns job_id
    Raises TrainingLimitExceeded if max concurrent training limit reached
    """
    # Check rate limit
    if not acquire_training_slot():
        raise TrainingLimitExceeded(
            f"Maximum concurrent training limit ({MAX_CONCURRENT_TRAINING}) reached. "
            "Please wait for a training job to complete."
        )
    
    job_id = str(uuid.uuid4())[:8]
    
    # Create job record
    with Session(engine) as session:
        job = TrainingJob(
            job_id=job_id,
            state="pending",
            progress=0,
            current_step="Initializing...",
        )
        session.add(job)
        session.commit()
        session.refresh(job)
    
    # Store in memory for quick access
    JOBS[job_id] = {
        'state': 'pending',
        'progress': 0,
        'current_step': 'Initializing...',
        'logs': [],
        'experiment_id': None,
        'epoch': 0,
        'total_epochs': config.get('epochs', 100),
        'train_loss': None,
        'val_loss': None,
        'best_metric': None,
    }
    
    # Initialize cancel flag
    CANCEL_FLAGS[job_id] = False
    
    # Start training in background thread
    thread = Thread(
        target=_run_training,
        args=(job_id, task, model_type, config, use_gpu),
        daemon=True
    )
    thread.start()
    
    return job_id


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get current job status"""
    if job_id in JOBS:
        return JOBS[job_id]
    
    # Check database
    with Session(engine) as session:
        job = session.exec(
            select(TrainingJob).where(TrainingJob.job_id == job_id)
        ).first()
        
        if job:
            return {
                'state': job.state,
                'progress': job.progress,
                'current_step': job.current_step,
                'logs': job.logs.split('\n') if job.logs else [],
                'experiment_id': job.experiment_id,
            }
    
    return None


def cancel_job(job_id: str) -> bool:
    """Cancel a running training job"""
    if job_id in JOBS:
        CANCEL_FLAGS[job_id] = True
        JOBS[job_id]['state'] = 'cancelling'
        JOBS[job_id]['current_step'] = 'Cancelling...'
        return True
    return False


def is_cancelled(job_id: str) -> bool:
    """Check if job is cancelled"""
    return CANCEL_FLAGS.get(job_id, False)


def _update_job(job_id: str, **kwargs):
    """Update job status"""
    if job_id in JOBS:
        JOBS[job_id].update(kwargs)
    
    # Also update database
    with Session(engine) as session:
        job = session.exec(
            select(TrainingJob).where(TrainingJob.job_id == job_id)
        ).first()
        
        if job:
            for key, value in kwargs.items():
                if key == 'logs' and isinstance(value, list):
                    value = '\n'.join(value)
                if hasattr(job, key):
                    setattr(job, key, value)
            job.updated_at = datetime.utcnow()
            session.add(job)
            session.commit()


def _add_log(job_id: str, message: str):
    """Add log message to job"""
    if job_id in JOBS:
        JOBS[job_id]['logs'].append(message)
    print(f"[{job_id}] {message}")


def _run_training(
    job_id: str,
    task: str,
    model_type: str,
    config: Dict[str, Any],
    use_gpu: bool
):
    """
    Main training function (runs in background thread)
    """
    try:
        # Set random seeds for reproducibility
        seed = config.get('random_seed', 42)
        set_seed(seed)
        _add_log(job_id, f"Random seed set to {seed} for reproducibility")
        
        _update_job(job_id, state='running', progress=5, current_step='Loading data...')
        _add_log(job_id, "Loading dataset...")
        
        # Load data
        df = load_dataset()
        _add_log(job_id, f"Loaded {len(df)} rows")
        
        # Prepare data
        _update_job(job_id, progress=10, current_step='Preprocessing data...')
        _add_log(job_id, f"Preprocessing with outlier_mode={config.get('outlier_mode', 'none')}")
        
        data = prepare_data(
            df,
            task=task,
            outlier_mode=config.get('outlier_mode', 'none'),
            test_size=config.get('test_size', 0.15),
            val_size=config.get('val_size', 0.15),
            random_seed=config.get('random_seed', settings.default_random_seed),
        )
        
        _add_log(job_id, f"Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")
        _add_log(job_id, f"Features: {data['n_features']}")
        
        # Apply augmentation to training data only
        aug_mode = config.get('augmentation_mode', 'none')
        if aug_mode != 'none':
            _add_log(job_id, f"Applying augmentation: {aug_mode}")
            noise_level = config.get('noise_level', 0.01)
            
            if task == 'regression':
                data['X_train'], data['y_train'] = augment_data(
                    data['X_train'], data['y_train'],
                    mode=aug_mode, noise_level=noise_level
                )
            else:
                # For classification, use SMOTE-like augmentation
                aug_mode_cls = 'smote' if aug_mode == 'oversample' else aug_mode
                data['X_train'], data['y_train'] = augment_classification_data(
                    data['X_train'], data['y_train'],
                    mode=aug_mode_cls, noise_level=noise_level
                )
            
            _add_log(job_id, f"After augmentation: {len(data['X_train'])} training samples")
        
        # Create experiment record
        _update_job(job_id, progress=15, current_step='Creating experiment...')
        
        model_name = _get_model_name(model_type, task, config, data['n_features'])
        
        with Session(engine) as session:
            experiment = Experiment(
                task=task,
                model_type=model_type,
                model_name=model_name,
                config_json=json.dumps(config),
                status='training',
            )
            session.add(experiment)
            session.commit()
            session.refresh(experiment)
            experiment_id = experiment.id
        
        _update_job(job_id, experiment_id=experiment_id)
        JOBS[job_id]['experiment_id'] = experiment_id
        
        # Train model
        _update_job(job_id, progress=20, current_step=f'Training {model_name}...')
        _add_log(job_id, f"Training {model_name}...")
        
        # Progress callback for MLP
        def progress_callback(epoch, total_epochs, train_loss, val_loss, best_metric=None):
            progress = 20 + int((epoch / total_epochs) * 60)
            
            # Update job with detailed metrics
            if job_id in JOBS:
                JOBS[job_id]['epoch'] = epoch
                JOBS[job_id]['total_epochs'] = total_epochs
                JOBS[job_id]['train_loss'] = train_loss
                JOBS[job_id]['val_loss'] = val_loss
                if best_metric is not None:
                    JOBS[job_id]['best_metric'] = best_metric
            
            _update_job(
                job_id,
                progress=progress,
                current_step=f'Epoch {epoch}/{total_epochs}'
            )
            
            # Check if cancelled
            if is_cancelled(job_id):
                raise InterruptedError("Training cancelled by user")
        
        model, metrics = _train_model(
            task=task,
            model_type=model_type,
            data=data,
            config=config,
            use_gpu=use_gpu,
            progress_callback=progress_callback if model_type == 'mlp' else None,
        )
        
        _add_log(job_id, f"Training complete!")
        
        # Save model and preprocessor
        _update_job(job_id, progress=85, current_step='Saving model...')
        
        models_dir = settings.models_dir_resolved
        experiment_dir = models_dir / f"exp_{experiment_id}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessor
        preprocessor_path = experiment_dir / "preprocessor.joblib"
        save_preprocessor(data['preprocessor'], preprocessor_path)
        
        # Compute and save baseline from training data (for feature importance)
        # Numeric: median, Categorical: mode
        baseline = {}
        X_train_raw = data.get('X_train_raw')
        if X_train_raw is not None and hasattr(X_train_raw, 'median'):
            for col in NUMERIC_COLS:
                if col in X_train_raw.columns:
                    baseline[col] = float(X_train_raw[col].median())
            for col in CATEGORICAL_COLS:
                if col in X_train_raw.columns:
                    mode_val = X_train_raw[col].mode()
                    baseline[col] = str(mode_val.iloc[0]) if len(mode_val) > 0 else ''
            
            baseline_path = experiment_dir / "baseline.json"
            with open(baseline_path, 'w') as f:
                json.dump(baseline, f, indent=2)
            _add_log(job_id, f"Saved dataset baseline for feature importance")
        
        # Save model
        if model_type == 'mlp':
            model_path = experiment_dir / "model.pt"
            save_pytorch_model(model, model_path)
        else:
            model_path = experiment_dir / "model.joblib"
            save_sklearn_model(model, model_path)
        
        # Save metrics and visualization data
        metrics_path = experiment_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate plots
        _update_job(job_id, progress=90, current_step='Generating visualizations...')
        plots_dir = experiment_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        generate_training_plots(
            metrics=metrics,
            task=task,
            model_type=model_type,
            output_dir=plots_dir,
        )
        
        # Save weights and activations for MLP
        if model_type == 'mlp':
            weights_path = experiment_dir / "weights.json"
            weights_data = model.get_weights()
            # Convert numpy to list for JSON serialization
            weights_json = {}
            for layer_name, layer_data in weights_data.items():
                weights_json[layer_name] = {
                    'weight': layer_data['weight'].tolist(),
                    'bias': layer_data['bias'].tolist(),
                }
            with open(weights_path, 'w') as f:
                json.dump(weights_json, f)
            
            # Save loss history
            if 'history' in metrics:
                history_path = experiment_dir / "loss_history.json"
                with open(history_path, 'w') as f:
                    json.dump(metrics['history'], f)
            
            # Save activations
            if 'sample_activations' in metrics:
                activations_path = experiment_dir / "activations.json"
                with open(activations_path, 'w') as f:
                    json.dump(metrics['sample_activations'], f)
        
        # Update experiment with results
        _update_job(job_id, progress=95, current_step='Finalizing...')
        
        with Session(engine) as session:
            experiment = session.get(Experiment, experiment_id)
            experiment.status = 'completed'
            experiment.model_path = str(model_path)
            experiment.preprocessor_path = str(preprocessor_path)
            experiment.plots_dir = str(plots_dir)
            experiment.metrics_json = json.dumps({
                'train': metrics.get('train', {}),
                'val': metrics.get('val', {}),
                'test': metrics.get('test', {}),
                'feature_importance': metrics.get('feature_importance', {}),
            })
            
            if model_type == 'mlp':
                experiment.weights_path = str(weights_path) if 'weights_path' in dir() else None
                experiment.loss_history_path = str(history_path) if 'history_path' in dir() else None
                experiment.activations_path = str(activations_path) if 'activations_path' in dir() else None
            
            session.add(experiment)
            session.commit()
        
        _update_job(
            job_id,
            state='completed',
            progress=100,
            current_step='Done!',
            experiment_id=experiment_id,
        )
        
        _add_log(job_id, f"Experiment {experiment_id} completed successfully!")
        
        # Log metrics summary
        if task == 'regression':
            test_metrics = metrics.get('test', {})
            _add_log(job_id, f"Test MAE: {test_metrics.get('mae', 'N/A'):.2f}")
            _add_log(job_id, f"Test RMSE: {test_metrics.get('rmse', 'N/A'):.2f}")
            _add_log(job_id, f"Test R2: {test_metrics.get('r2', 'N/A'):.4f}")
        else:
            test_metrics = metrics.get('test', {})
            _add_log(job_id, f"Test Accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}")
            _add_log(job_id, f"Test F1 (macro): {test_metrics.get('f1_macro', 'N/A'):.4f}")
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        _add_log(job_id, error_msg)
        _update_job(
            job_id,
            state='failed',
            current_step=f'Failed: {str(e)}',
        )
        
        # Update experiment if created
        if JOBS.get(job_id, {}).get('experiment_id'):
            with Session(engine) as session:
                experiment = session.get(Experiment, JOBS[job_id]['experiment_id'])
                if experiment:
                    experiment.status = 'failed'
                    experiment.error_message = str(e)
                    session.add(experiment)
                    session.commit()
    
    finally:
        # Always release training slot when job finishes (success, fail, or cancel)
        release_training_slot()
        _add_log(job_id, "Training slot released")


def _get_model_name(model_type: str, task: str, config: Dict[str, Any] = None, n_features: int = 0) -> str:
    """Get human-readable model name with parameter count for MLP"""
    names = {
        'linear': 'Linear Regression' if task == 'regression' else 'Logistic Regression',
        'rf': 'Random Forest',
        'xgboost': 'XGBoost / Gradient Boosting',
        'mlp': 'Neural Network (MLP)',
    }
    
    base_name = names.get(model_type, model_type.upper())
    
    # Add parameter count for MLP
    if model_type == 'mlp' and config and n_features > 0:
        params = _calculate_mlp_params(config, n_features, task)
        param_str = _format_params(params)
        return f"{base_name} [{param_str}]"
    
    return base_name


def _calculate_mlp_params(config: Dict[str, Any], n_features: int, task: str) -> int:
    """Calculate total trainable parameters in MLP"""
    hidden_layers = config.get('hidden_layers', [128, 64, 32])
    batch_norm = config.get('batch_norm', True)
    
    total_params = 0
    prev_size = n_features
    
    # Hidden layers
    for neurons in hidden_layers:
        # Weights + biases
        total_params += prev_size * neurons + neurons
        # Batch norm (gamma + beta)
        if batch_norm:
            total_params += neurons * 2
        prev_size = neurons
    
    # Output layer
    output_size = 1 if task == 'regression' else 3
    total_params += prev_size * output_size + output_size
    
    return total_params


def _format_params(params: int) -> str:
    """Format parameter count for display"""
    if params >= 1_000_000:
        return f"{params / 1_000_000:.2f}M params"
    elif params >= 1000:
        return f"{params / 1000:.1f}K params"
    return f"{params} params"


def _train_model(
    task: str,
    model_type: str,
    data: Dict[str, Any],
    config: Dict[str, Any],
    use_gpu: bool = True,
    progress_callback=None,
):
    """
    Train the specified model
    """
    log_transform = config.get('outlier_mode') == 'log'
    
    common_args = {
        'X_train': data['X_train'],
        'y_train': data['y_train'],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
        'X_test': data['X_test'],
        'y_test': data['y_test'],
        'feature_names': data['feature_names'],
    }
    
    if task == 'regression':
        if model_type == 'linear':
            return train_linear_regression(**common_args, log_transform=log_transform)
        elif model_type == 'rf':
            return train_random_forest_regressor(
                **common_args,
                log_transform=log_transform,
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth'),
                random_state=config.get('random_seed', 42),
            )
        elif model_type == 'xgboost':
            return train_xgboost_regressor(
                **common_args,
                log_transform=log_transform,
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', 6),
                learning_rate=config.get('tree_learning_rate', 0.1),
                random_state=config.get('random_seed', 42),
            )
        elif model_type == 'mlp':
            return train_mlp_regressor(
                **common_args,
                log_transform=log_transform,
                hidden_layers=config.get('hidden_layers', [128, 64, 32]),
                activation=config.get('activation', 'relu'),
                dropout=config.get('dropout', 0.2),
                batch_norm=config.get('batch_norm', True),
                learning_rate=config.get('learning_rate', 0.001),
                optimizer_type=config.get('optimizer', 'adam'),
                epochs=config.get('epochs', 100),
                batch_size=config.get('batch_size', 32),
                early_stopping_patience=config.get('early_stopping_patience', 10),
                use_gpu=use_gpu,
                progress_callback=progress_callback,
            )
    
    else:  # classification
        if model_type == 'linear':
            return train_logistic_regression(
                **common_args,
                random_state=config.get('random_seed', 42),
            )
        elif model_type == 'rf':
            return train_random_forest_classifier(
                **common_args,
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth'),
                random_state=config.get('random_seed', 42),
            )
        elif model_type == 'mlp':
            return train_mlp_classifier(
                **common_args,
                hidden_layers=config.get('hidden_layers', [128, 64, 32]),
                activation=config.get('activation', 'relu'),
                dropout=config.get('dropout', 0.2),
                batch_norm=config.get('batch_norm', True),
                learning_rate=config.get('learning_rate', 0.001),
                optimizer_type=config.get('optimizer', 'adam'),
                epochs=config.get('epochs', 100),
                batch_size=config.get('batch_size', 32),
                early_stopping_patience=config.get('early_stopping_patience', 10),
                use_gpu=use_gpu,
                progress_callback=progress_callback,
            )
    
    raise ValueError(f"Unknown model type: {model_type} for task: {task}")
