"""
Prediction service - handles making predictions with trained models

NOTE ON FEATURE IMPORTANCE:
- Feature importance is computed using PERTURBATION-BASED method (not SHAP)
- For each feature, we swap it with the baseline value and measure prediction change
- Baseline values are computed from TRAINING SET (median for numeric, mode for categorical)
- All percentages are normalized to sum to 100%
"""
import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from sqlmodel import Session, select

from app.database import engine, Experiment, Prediction
from app.data_processing import (
    load_preprocessor, NUMERIC_COLS, CATEGORICAL_COLS
)
from app.models.sklearn_models import load_sklearn_model
from app.models.pytorch_models import (
    load_pytorch_model, MLPRegressor, MLPClassifier, get_device
)


# Default baseline (used if dataset baseline not available)
DEFAULT_BASELINE = {
    'god': 2015,
    'snaga': 120,
    'kilometraza': 150000,
    'kubikaza': 1600,
    'marka': 'Volkswagen',
    'model': 'Golf',
    'ostecenje': 'Bez ostecenja',
    'registrovan': 'Da',
    'gorivo': 'Dizel',
    'mjenjac': 'Manuelni',
}


def load_or_compute_baseline(experiment_dir: Path) -> Dict[str, Any]:
    """
    Load baseline from experiment folder if exists, otherwise return default.
    Baseline is saved during training as baseline.json
    """
    baseline_path = experiment_dir / "baseline.json"
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            return json.load(f)
    return DEFAULT_BASELINE.copy()


def compute_local_importance(
    features: Dict[str, Any],
    base_prediction: float,
    preprocessor,
    model,
    model_type: str,
    config: Dict[str, Any],
    baseline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute local feature importance for THIS specific prediction.
    Uses PERTURBATION method: change each feature to baseline and measure prediction change.
    All percentages sum to 100%.
    
    Args:
        features: User's input features
        base_prediction: Prediction for user's input
        preprocessor: Fitted preprocessor
        model: Trained model
        model_type: 'mlp', 'rf', etc.
        config: Model config
        baseline: Baseline values (median/mode from training set). If None, uses default.
    """
    feature_display = {
        'god': 'Godina proizvodnje',
        'snaga': 'Snaga (HP)',
        'kilometraza': 'Kilometraza (km)',
        'kubikaza': 'Kubikaza (ccm)',
        'marka': 'Marka',
        'model': 'Model',
        'ostecenje': 'Ostecenje',
        'registrovan': 'Registrovan',
        'gorivo': 'Gorivo',
        'mjenjac': 'Mjenjac',
    }
    
    # Use provided baseline or default
    if baseline is None:
        baseline = DEFAULT_BASELINE.copy()
    
    log_transform = config.get('outlier_mode') == 'log'
    
    def get_pred(feat_dict):
        """Get prediction for given features"""
        df = pd.DataFrame([feat_dict])
        df = df[NUMERIC_COLS + CATEGORICAL_COLS]
        X = preprocessor.transform(df)
        
        if model_type == 'mlp':
            X_tensor = torch.FloatTensor(X)
            with torch.no_grad():
                pred = model(X_tensor).item()
        else:
            pred = model.predict(X)[0]
        
        if log_transform:
            pred = np.expm1(pred)
        return pred
    
    importance_values = {}
    
    # For each feature, measure impact by changing to baseline
    for feature in feature_display.keys():
        user_value = features.get(feature)
        base_value = baseline.get(feature)
        
        if user_value == base_value:
            # Same as baseline - small impact
            importance_values[feature] = 0.1
        else:
            try:
                # Prediction with baseline value for this feature
                modified = features.copy()
                modified[feature] = base_value
                modified_pred = get_pred(modified)
                
                # Absolute difference = importance
                diff = abs(base_prediction - modified_pred)
                importance_values[feature] = diff if diff > 0 else 0.1
            except:
                importance_values[feature] = 0.1
    
    # Normalize to 100%
    total = sum(importance_values.values())
    
    result = {}
    for feature, display_name in feature_display.items():
        user_value = features.get(feature, '')
        raw_imp = importance_values.get(feature, 0)
        pct = (raw_imp / total * 100) if total > 0 else 0
        
        result[feature] = {
            'display_name': display_name,
            'importance': raw_imp,
            'user_value': user_value,
            'percentage': pct,
            'hint': None,
            'is_numeric': feature in ['god', 'snaga', 'kilometraza', 'kubikaza'],
        }
    
    # Sort by percentage
    return dict(sorted(result.items(), key=lambda x: x[1]['percentage'], reverse=True))


def make_prediction(
    features: Dict[str, Any],
    task: str = 'regression',
    experiment_id: Optional[int] = None,
) -> Tuple[Dict[str, Any], int]:
    """
    Make a prediction using the specified or best model
    
    Args:
        features: Input features as dictionary
        task: 'regression' or 'classification'
        experiment_id: Specific experiment to use, or None for best
    
    Returns:
        Tuple of (prediction result, experiment_id used)
    """
    # Get experiment
    with Session(engine) as session:
        if experiment_id:
            experiment = session.get(Experiment, experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
        else:
            # Get best model for task
            experiment = session.exec(
                select(Experiment)
                .where(Experiment.task == task)
                .where(Experiment.status == 'completed')
                .order_by(Experiment.created_at.desc())
            ).first()
            
            if not experiment:
                raise ValueError(f"No trained models found for task: {task}")
        
        experiment_id = experiment.id
        model_type = experiment.model_type
        model_path = experiment.model_path
        preprocessor_path = experiment.preprocessor_path
        model_name = experiment.model_name
        config = experiment.config
        metrics = experiment.metrics
    
    # Load preprocessor
    preprocessor = load_preprocessor(Path(preprocessor_path))
    
    # Create DataFrame from features
    feature_df = pd.DataFrame([features])
    
    # Ensure correct column order
    feature_df = feature_df[NUMERIC_COLS + CATEGORICAL_COLS]
    
    # Transform features
    X = preprocessor.transform(feature_df)
    
    # Load and use model
    if model_type == 'mlp':
        model_class = MLPClassifier if task == 'classification' else MLPRegressor
        model = load_pytorch_model(Path(model_path), model_class)
        model.eval()
        
        device = get_device(use_gpu=False)  # Use CPU for inference
        X_tensor = torch.FloatTensor(X).to(device)
        
        with torch.no_grad():
            output = model(X_tensor)
            
            if task == 'classification':
                probs = torch.softmax(output, dim=1)
                pred_idx = output.argmax(1).item()
                labels = ['budget', 'mid', 'premium']
                prediction = labels[pred_idx]
                confidence = probs[0, pred_idx].item()
                prediction_value = pred_idx
            else:
                prediction = output.item()
                # Inverse log transform if needed
                if config.get('outlier_mode') == 'log':
                    prediction = np.expm1(prediction)
                prediction_value = prediction
                confidence = None
        
        # Get activations for this input
        activations = {k: v.tolist() for k, v in model.activations.items()}
    
    else:
        model = load_sklearn_model(Path(model_path))
        
        if task == 'classification':
            prediction = model.predict(X)[0]
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[0]
                labels = model.classes_.tolist()
                pred_idx = labels.index(prediction)
                confidence = probs[pred_idx]
            else:
                confidence = None
            prediction_value = prediction
        else:
            prediction = model.predict(X)[0]
            # Inverse log transform if needed
            if config.get('outlier_mode') == 'log':
                prediction = np.expm1(prediction)
            prediction_value = float(prediction)
            confidence = None
        
        activations = None
    
    # Load baseline from experiment folder (dataset median/mode, saved during training)
    experiment_dir = Path(model_path).parent
    baseline = load_or_compute_baseline(experiment_dir)
    
    # Compute LOCAL feature importance for this specific prediction
    local_importance = compute_local_importance(
        features=features,
        base_prediction=prediction_value if task == 'regression' else 0,
        preprocessor=preprocessor,
        model=model,
        model_type=model_type,
        config=config,
        baseline=baseline,
    )
    
    # Save prediction to history
    with Session(engine) as session:
        pred_record = Prediction(
            experiment_id=experiment_id,
            input_features_json=json.dumps(features),
            prediction=float(prediction_value) if isinstance(prediction_value, (int, float, np.number)) else 0,
            prediction_label=prediction if task == 'classification' else None,
            confidence=confidence,
        )
        session.add(pred_record)
        session.commit()
    
    result = {
        'prediction': prediction_value,
        'prediction_label': prediction if task == 'classification' else None,
        'confidence': confidence,
        'experiment_id': experiment_id,
        'model_name': model_name,
        'top_features': local_importance,
    }
    
    if activations:
        result['activations'] = activations
    
    return result, experiment_id


def get_forward_pass_visualization(
    experiment_id: int,
    features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get data for forward pass visualization
    
    Args:
        experiment_id: Experiment to visualize
        features: Input features, or None for random sample
    
    Returns:
        Visualization data including architecture, weights, activations
    """
    with Session(engine) as session:
        experiment = session.get(Experiment, experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if experiment.model_type != 'mlp':
            raise ValueError("Forward pass visualization only available for MLP models")
        
        model_path = experiment.model_path
        preprocessor_path = experiment.preprocessor_path
        weights_path = experiment.weights_path
        task = experiment.task
    
    # Load model
    model_class = MLPClassifier if task == 'classification' else MLPRegressor
    model = load_pytorch_model(Path(model_path), model_class)
    model.eval()
    
    # Load preprocessor
    preprocessor = load_preprocessor(Path(preprocessor_path))
    
    # Prepare input
    if features:
        feature_df = pd.DataFrame([features])
        feature_df = feature_df[NUMERIC_COLS + CATEGORICAL_COLS]
        X = preprocessor.transform(feature_df)
        input_info = features
    else:
        # Load a random sample from dataset
        from app.data_processing import load_dataset
        df = load_dataset()
        sample = df.sample(1)
        feature_df = sample[NUMERIC_COLS + CATEGORICAL_COLS]
        X = preprocessor.transform(feature_df)
        input_info = sample.iloc[0].to_dict()
    
    # Get activations
    X_tensor = torch.FloatTensor(X)
    with torch.no_grad():
        output = model(X_tensor)
    
    activations = {k: v.tolist() for k, v in model.activations.items()}
    
    # Load weights
    if weights_path and Path(weights_path).exists():
        with open(weights_path, 'r') as f:
            weights = json.load(f)
    else:
        weights_data = model.get_weights()
        weights = {}
        for layer_name, layer_data in weights_data.items():
            weights[layer_name] = {
                'weight': layer_data['weight'].tolist(),
                'bias': layer_data['bias'].tolist(),
            }
    
    return {
        'architecture': model.get_architecture(),
        'weights': weights,
        'activations': activations,
        'input_features': input_info,
        'output': output.tolist() if task == 'regression' else torch.softmax(output, dim=1).tolist(),
    }
