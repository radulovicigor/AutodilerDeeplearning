"""
Sklearn baseline models for regression and classification
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix
)
from sklearn.inspection import permutation_importance
import joblib
from pathlib import Path

# Try to import XGBoost, fall back to GradientBoosting
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, 
                       log_transform: bool = False) -> Dict[str, float]:
    """Calculate regression metrics"""
    # Handle NaN values
    y_true = np.nan_to_num(y_true, nan=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0)
    
    if log_transform:
        # Inverse log transform for metrics
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
        # Handle any inf values after expm1
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=1e10, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=0.0)
    
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred)),
    }


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray,
                           labels: Optional[list] = None) -> Dict[str, Any]:
    """Calculate classification metrics"""
    if labels is None:
        labels = ['budget', 'mid', 'premium']
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'f1_per_class': {
            label: float(f1_score(y_true, y_pred, labels=[label], average='micro'))
            for label in labels
        },
        'confusion_matrix': cm.tolist(),
        'labels': labels,
    }


def get_feature_importance(model: Any, feature_names: list,
                          X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
    """Get feature importance from model"""
    importance_dict = {}
    
    # Tree-based models have built-in feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for name, imp in zip(feature_names, importances):
            importance_dict[name] = float(imp)
    
    # Linear models have coefficients
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        if len(coef.shape) > 1:
            coef = np.abs(coef).mean(axis=0)
        else:
            coef = np.abs(coef)
        for name, imp in zip(feature_names, coef):
            importance_dict[name] = float(imp)
    
    # Fallback: permutation importance
    elif X_val is not None and y_val is not None:
        result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42)
        for name, imp in zip(feature_names, result.importances_mean):
            importance_dict[name] = float(max(0, imp))
    
    # Sort by importance
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    return importance_dict


# ============== REGRESSION MODELS ==============

def train_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    log_transform: bool = False,
    **kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """Train Linear Regression baseline"""
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'train': evaluate_regression(y_train, y_train_pred, log_transform),
        'val': evaluate_regression(y_val, y_val_pred, log_transform),
        'test': evaluate_regression(y_test, y_test_pred, log_transform),
        'feature_importance': get_feature_importance(model, feature_names),
    }
    
    # Predictions for visualization
    metrics['predictions'] = {
        'y_test_true': y_test.tolist() if not log_transform else np.expm1(y_test).tolist(),
        'y_test_pred': y_test_pred.tolist() if not log_transform else np.expm1(y_test_pred).tolist(),
        'residuals': (y_test - y_test_pred).tolist() if not log_transform else (np.expm1(y_test) - np.expm1(y_test_pred)).tolist(),
    }
    
    return model, metrics


def train_random_forest_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    log_transform: bool = False,
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
    random_state: int = 42,
    **kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """Train Random Forest Regressor"""
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'train': evaluate_regression(y_train, y_train_pred, log_transform),
        'val': evaluate_regression(y_val, y_val_pred, log_transform),
        'test': evaluate_regression(y_test, y_test_pred, log_transform),
        'feature_importance': get_feature_importance(model, feature_names),
    }
    
    metrics['predictions'] = {
        'y_test_true': y_test.tolist() if not log_transform else np.expm1(y_test).tolist(),
        'y_test_pred': y_test_pred.tolist() if not log_transform else np.expm1(y_test_pred).tolist(),
        'residuals': (y_test - y_test_pred).tolist() if not log_transform else (np.expm1(y_test) - np.expm1(y_test_pred)).tolist(),
    }
    
    return model, metrics


def train_xgboost_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    log_transform: bool = False,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    **kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """Train XGBoost or GradientBoosting Regressor"""
    
    if HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'train': evaluate_regression(y_train, y_train_pred, log_transform),
        'val': evaluate_regression(y_val, y_val_pred, log_transform),
        'test': evaluate_regression(y_test, y_test_pred, log_transform),
        'feature_importance': get_feature_importance(model, feature_names),
    }
    
    metrics['predictions'] = {
        'y_test_true': y_test.tolist() if not log_transform else np.expm1(y_test).tolist(),
        'y_test_pred': y_test_pred.tolist() if not log_transform else np.expm1(y_test_pred).tolist(),
        'residuals': (y_test - y_test_pred).tolist() if not log_transform else (np.expm1(y_test) - np.expm1(y_test_pred)).tolist(),
    }
    
    return model, metrics


# ============== CLASSIFICATION MODELS ==============

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    max_iter: int = 1000,
    random_state: int = 42,
    **kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """Train Logistic Regression classifier"""
    
    model = LogisticRegression(max_iter=max_iter, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    labels = ['budget', 'mid', 'premium']
    
    # Metrics
    metrics = {
        'train': evaluate_classification(y_train, y_train_pred, labels),
        'val': evaluate_classification(y_val, y_val_pred, labels),
        'test': evaluate_classification(y_test, y_test_pred, labels),
        'feature_importance': get_feature_importance(model, feature_names),
    }
    
    return model, metrics


def train_random_forest_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    n_estimators: int = 100,
    max_depth: int = None,
    random_state: int = 42,
    **kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """Train Random Forest Classifier"""
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    labels = ['budget', 'mid', 'premium']
    
    # Metrics
    metrics = {
        'train': evaluate_classification(y_train, y_train_pred, labels),
        'val': evaluate_classification(y_val, y_val_pred, labels),
        'test': evaluate_classification(y_test, y_test_pred, labels),
        'feature_importance': get_feature_importance(model, feature_names),
    }
    
    return model, metrics


def save_sklearn_model(model: Any, path: Path) -> str:
    """Save sklearn model to disk"""
    joblib.dump(model, path)
    return str(path)


def load_sklearn_model(path: Path) -> Any:
    """Load sklearn model from disk"""
    return joblib.load(path)
