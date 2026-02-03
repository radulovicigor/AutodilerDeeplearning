"""
Data loading, preprocessing, and transformation utilities
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from app.config import settings


# Column definitions
NUMERIC_COLS = ['snaga', 'kilometraza', 'kubikaza', 'god']
CATEGORICAL_COLS = ['marka', 'model', 'ostecenje', 'registrovan', 'gorivo', 'mjenjac']
TARGET_COL = 'cena'

# Reasonable limits for car data (for outlier removal)
COLUMN_LIMITS = {
    'cena': {'min': 500, 'max': 200000},           # Price: 500 - 200,000 EUR
    'snaga': {'min': 30, 'max': 800},               # Power: 30 - 800 HP
    'kilometraza': {'min': 0, 'max': 500000},       # Mileage: 0 - 500,000 km
    'kubikaza': {'min': 500, 'max': 8000},          # Engine: 500 - 8000 cc
    'god': {'min': 1980, 'max': 2026},              # Year: 1980 - 2026
}


def clean_outliers(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove extreme outliers from all numeric columns based on predefined limits.
    This ensures data quality and prevents model training issues.
    
    Args:
        df: Input DataFrame
        verbose: Print cleaning statistics
        
    Returns:
        Cleaned DataFrame with outliers removed
    """
    df = df.copy()
    original_len = len(df)
    removed_stats = {}
    
    for col, limits in COLUMN_LIMITS.items():
        if col not in df.columns:
            continue
            
        before = len(df)
        
        # Apply min/max filters
        mask = (df[col] >= limits['min']) & (df[col] <= limits['max'])
        df = df[mask]
        
        removed = before - len(df)
        if removed > 0:
            removed_stats[col] = {
                'removed': removed,
                'min': limits['min'],
                'max': limits['max']
            }
    
    total_removed = original_len - len(df)
    
    if verbose and total_removed > 0:
        print(f"\n{'='*50}")
        print(f"DATA CLEANING REPORT")
        print(f"{'='*50}")
        print(f"Original rows: {original_len}")
        print(f"Rows removed: {total_removed} ({100*total_removed/original_len:.1f}%)")
        print(f"Clean rows: {len(df)}")
        print(f"\nPer-column breakdown:")
        for col, stats in removed_stats.items():
            print(f"  - {col}: removed {stats['removed']} rows (outside {stats['min']}-{stats['max']})")
        print(f"{'='*50}\n")
    
    return df


def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a data quality report showing outlier statistics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with quality statistics per column
    """
    report = {
        'total_rows': len(df),
        'columns': {}
    }
    
    for col, limits in COLUMN_LIMITS.items():
        if col not in df.columns:
            continue
            
        col_data = df[col]
        outliers_below = (col_data < limits['min']).sum()
        outliers_above = (col_data > limits['max']).sum()
        
        report['columns'][col] = {
            'min': float(col_data.min()),
            'max': float(col_data.max()),
            'mean': float(col_data.mean()),
            'median': float(col_data.median()),
            'std': float(col_data.std()),
            'outliers_below_limit': int(outliers_below),
            'outliers_above_limit': int(outliers_above),
            'total_outliers': int(outliers_below + outliers_above),
            'limit_min': limits['min'],
            'limit_max': limits['max'],
        }
    
    return report


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load CSV with robust encoding handling
    Tries utf-8, then latin1, then cp1252
    """
    if path is None:
        path = settings.data_path_resolved
    
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            print(f"Successfully loaded dataset with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError(f"Could not load CSV with any encoding: {encodings}")
    
    # Clean text columns - fix encoding issues in text
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(clean_text)
    
    # Validate required columns
    required_cols = NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Handle missing values in numeric columns
    for col in NUMERIC_COLS + [TARGET_COL]:
        if col in df.columns:
            # Fill NaN with median
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            # Convert to numeric, coercing errors
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(median_val)
    
    # Handle missing values in categorical columns
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    # Remove rows with any remaining NaN in target
    df = df.dropna(subset=[TARGET_COL])
    
    # Remove rows where price is 0 or negative
    df = df[df[TARGET_COL] > 0]
    
    # ===== COMPREHENSIVE OUTLIER CLEANING =====
    # Remove extreme outliers from ALL numeric columns
    df = clean_outliers(df, verbose=True)
    
    # Print final data summary
    print(f"\nFinal dataset summary:")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Price range: {df[TARGET_COL].min():.0f} - {df[TARGET_COL].max():.0f} EUR")
    print(f"  - Power range: {df['snaga'].min():.0f} - {df['snaga'].max():.0f} HP")
    print(f"  - Mileage range: {df['kilometraza'].min():.0f} - {df['kilometraza'].max():.0f} km")
    print(f"  - Year range: {df['god'].min():.0f} - {df['god'].max():.0f}")
    
    return df


def clean_text(text: str) -> str:
    """Clean text with potential encoding issues"""
    if not isinstance(text, str):
        return str(text)
    return text.strip()


def handle_outliers(
    df: pd.DataFrame,
    mode: str = 'none',
    columns: List[str] = ['cena', 'kilometraza']
) -> pd.DataFrame:
    """
    Handle outliers in specified columns
    
    Args:
        df: DataFrame
        mode: 'none', 'clip', or 'log'
        columns: columns to process
    
    Returns:
        Processed DataFrame
    """
    df = df.copy()
    
    if mode == 'none':
        return df
    
    elif mode == 'clip':
        # Winsorize at 1% and 99%
        for col in columns:
            if col in df.columns:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=lower, upper=upper)
    
    elif mode == 'log':
        # Log transform target
        if TARGET_COL in columns and TARGET_COL in df.columns:
            df[f'{TARGET_COL}_original'] = df[TARGET_COL]
            df[TARGET_COL] = np.log1p(df[TARGET_COL])
    
    return df


def augment_data(
    X: np.ndarray,
    y: np.ndarray,
    mode: str = 'none',
    noise_level: float = 0.01,
    oversample_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment training data
    
    Args:
        X: Feature matrix
        y: Target values
        mode: 'none', 'noise', 'oversample', 'both'
        noise_level: Standard deviation of Gaussian noise (fraction of feature std)
        oversample_factor: Factor to increase dataset size
    
    Returns:
        Augmented X, y
    """
    if mode == 'none':
        return X, y
    
    X_aug = X.copy()
    y_aug = y.copy()
    
    if mode in ['noise', 'both']:
        # Add Gaussian noise to numeric features
        n_samples = X.shape[0]
        noise = np.random.normal(0, noise_level, X.shape) * np.std(X, axis=0)
        X_noisy = X + noise
        X_aug = np.vstack([X_aug, X_noisy])
        y_aug = np.concatenate([y_aug, y])
        print(f"Added noise augmentation: {n_samples} -> {len(y_aug)} samples")
    
    if mode in ['oversample', 'both']:
        # Simple oversampling with slight perturbation
        n_new = int(len(X) * oversample_factor)
        if n_new > 0:
            indices = np.random.choice(len(X), n_new, replace=True)
            X_over = X[indices] + np.random.normal(0, noise_level * 0.5, (n_new, X.shape[1])) * np.std(X, axis=0)
            y_over = y[indices]
            X_aug = np.vstack([X_aug, X_over])
            y_aug = np.concatenate([y_aug, y_over])
            print(f"Added oversampling: -> {len(y_aug)} samples")
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(y_aug))
    return X_aug[shuffle_idx], y_aug[shuffle_idx]


def augment_classification_data(
    X: np.ndarray,
    y: np.ndarray,
    mode: str = 'none',
    noise_level: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment classification data with class balancing
    
    Args:
        X: Feature matrix
        y: Target labels
        mode: 'none', 'noise', 'smote', 'both'
        noise_level: Standard deviation of Gaussian noise
    
    Returns:
        Augmented X, y
    """
    if mode == 'none':
        return X, y
    
    X_aug = X.copy()
    y_aug = y.copy() if isinstance(y, np.ndarray) else np.array(y)
    
    # Get unique classes and their counts
    unique_classes, counts = np.unique(y_aug, return_counts=True)
    max_count = max(counts)
    
    if mode in ['smote', 'both']:
        # Simple SMOTE-like oversampling for minority classes
        for cls, count in zip(unique_classes, counts):
            if count < max_count:
                # Find samples of this class
                cls_indices = np.where(y_aug == cls)[0]
                n_to_add = max_count - count
                
                # Generate new samples by interpolating between existing ones
                for _ in range(n_to_add):
                    idx1, idx2 = np.random.choice(cls_indices, 2, replace=True)
                    alpha = np.random.random()
                    new_sample = alpha * X_aug[idx1] + (1 - alpha) * X_aug[idx2]
                    X_aug = np.vstack([X_aug, new_sample.reshape(1, -1)])
                    y_aug = np.append(y_aug, cls)
        
        print(f"SMOTE balancing: {len(y)} -> {len(y_aug)} samples")
    
    if mode in ['noise', 'both']:
        # Add noise to all samples
        n_samples = len(X_aug)
        noise = np.random.normal(0, noise_level, X_aug.shape) * np.std(X_aug, axis=0)
        X_noisy = X_aug + noise
        X_aug = np.vstack([X_aug, X_noisy])
        y_aug = np.concatenate([y_aug, y_aug])
        print(f"Added noise augmentation: -> {len(y_aug)} samples")
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(y_aug))
    return X_aug[shuffle_idx], y_aug[shuffle_idx]


def create_price_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create price segments for classification task
    Based on percentiles: bottom 33% = budget, 34-66% = mid, top 33% = premium
    """
    df = df.copy()
    
    # Use original price if log-transformed
    price_col = f'{TARGET_COL}_original' if f'{TARGET_COL}_original' in df.columns else TARGET_COL
    
    # Calculate percentiles
    p33 = df[price_col].quantile(0.33)
    p66 = df[price_col].quantile(0.66)
    
    def segment(price):
        if price <= p33:
            return 'budget'
        elif price <= p66:
            return 'mid'
        else:
            return 'premium'
    
    df['price_segment'] = df[price_col].apply(segment)
    
    return df


def create_preprocessor(numeric_cols: List[str] = NUMERIC_COLS,
                       categorical_cols: List[str] = CATEGORICAL_COLS) -> ColumnTransformer:
    """
    Create sklearn preprocessing pipeline
    - StandardScaler for numeric features
    - OneHotEncoder for categorical features
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    task: str = 'regression',
    outlier_mode: str = 'none',
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Prepare data for training
    
    Args:
        df: Raw DataFrame
        task: 'regression' or 'classification'
        outlier_mode: 'none', 'clip', or 'log'
        test_size: fraction for test set
        val_size: fraction for validation set
        random_seed: for reproducibility
    
    Returns:
        Dictionary with train/val/test splits and preprocessor
    """
    # Handle outliers
    df = handle_outliers(df, mode=outlier_mode)
    
    # Create segments for classification
    if task == 'classification':
        df = create_price_segments(df)
        target_col = 'price_segment'
    else:
        target_col = TARGET_COL
    
    # Features and target
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[target_col]
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y if task == 'classification' else None
    )
    
    # Second split: train vs val
    val_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_adjusted,
        random_state=random_seed,
        stratify=y_temp if task == 'classification' else None
    )
    
    # Create and fit preprocessor
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    feature_names = get_feature_names(preprocessor)
    
    return {
        'X_train': X_train_processed,
        'X_val': X_val_processed,
        'X_test': X_test_processed,
        'y_train': y_train.values,
        'y_val': y_val.values,
        'y_test': y_test.values,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'n_features': X_train_processed.shape[1],
        'outlier_mode': outlier_mode,
        'task': task,
        'target_col': target_col,
        # Original data for visualization
        'X_train_raw': X_train,
        'X_val_raw': X_val,
        'X_test_raw': X_test,
    }


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract feature names from fitted preprocessor"""
    feature_names = []
    
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(columns)
        elif name == 'cat':
            # Get one-hot encoded feature names
            if hasattr(transformer, 'get_feature_names_out'):
                cat_features = transformer.get_feature_names_out(columns)
                feature_names.extend(cat_features)
            else:
                # Fallback for older sklearn
                for col in columns:
                    categories = transformer.categories_[columns.index(col)]
                    for cat in categories:
                        feature_names.append(f"{col}_{cat}")
    
    return feature_names


def get_dataset_schema() -> Dict[str, Any]:
    """Get schema for frontend form generation"""
    # Load sample to get unique values
    try:
        df = load_dataset()
    except:
        df = None
    
    schema = {
        'numeric_features': [],
        'categorical_features': [],
        'target': TARGET_COL,
        'brand_models': {}  # Mapping of brand -> list of models
    }
    
    # Numeric features
    for col in NUMERIC_COLS:
        feature = {
            'name': col,
            'type': 'numeric',
            'label': col.replace('_', ' ').title()
        }
        if df is not None:
            feature['min'] = float(df[col].min())
            feature['max'] = float(df[col].max())
            feature['mean'] = float(df[col].mean())
        schema['numeric_features'].append(feature)
    
    # Categorical features
    for col in CATEGORICAL_COLS:
        feature = {
            'name': col,
            'type': 'categorical',
            'label': col.replace('_', ' ').title()
        }
        if df is not None:
            feature['options'] = sorted(df[col].unique().tolist())
        else:
            feature['options'] = []
        schema['categorical_features'].append(feature)
    
    # Create brand -> models mapping
    if df is not None:
        brand_models = df.groupby('marka')['model'].apply(lambda x: sorted(x.unique().tolist())).to_dict()
        schema['brand_models'] = brand_models
    
    return schema


def get_price_distribution(outlier_mode: str = 'none') -> Dict[str, Any]:
    """Get price distribution for histogram visualization"""
    df = load_dataset()
    
    original_prices = df[TARGET_COL].values.tolist()
    
    df_processed = handle_outliers(df, mode=outlier_mode)
    processed_prices = df_processed[TARGET_COL].values.tolist()
    
    return {
        'original': {
            'values': original_prices,
            'min': float(min(original_prices)),
            'max': float(max(original_prices)),
            'mean': float(np.mean(original_prices)),
            'median': float(np.median(original_prices)),
        },
        'processed': {
            'values': processed_prices,
            'min': float(min(processed_prices)),
            'max': float(max(processed_prices)),
            'mean': float(np.mean(processed_prices)),
            'median': float(np.median(processed_prices)),
        },
        'outlier_mode': outlier_mode
    }


def save_preprocessor(preprocessor: ColumnTransformer, path: Path) -> str:
    """Save preprocessor to disk"""
    joblib.dump(preprocessor, path)
    return str(path)


def load_preprocessor(path: Path) -> ColumnTransformer:
    """Load preprocessor from disk"""
    return joblib.load(path)
