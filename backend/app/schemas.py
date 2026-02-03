"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============== TRAINING ==============

class TrainRequest(BaseModel):
    """Request body for training endpoint"""
    model_config = ConfigDict(protected_namespaces=())
    
    task: str = Field(..., description="'regression' or 'classification'")
    model_type: str = Field(..., description="'linear', 'rf', 'xgboost', or 'mlp'")
    
    # Data processing
    outlier_mode: str = Field(default='none', description="'none', 'clip', or 'log'")
    augmentation_mode: str = Field(default='none', description="'none', 'noise', 'oversample/smote', 'both'")
    noise_level: float = Field(default=0.01, ge=0.001, le=0.1, description="Noise level for augmentation")
    test_size: float = Field(default=0.15, ge=0.05, le=0.4)
    val_size: float = Field(default=0.15, ge=0.05, le=0.4)
    random_seed: int = Field(default=42)
    
    # Tree model config
    n_estimators: int = Field(default=100, ge=10, le=1000)
    max_depth: Optional[int] = Field(default=None, ge=1, le=50)
    tree_learning_rate: float = Field(default=0.1, ge=0.001, le=1.0)
    
    # MLP config
    hidden_layers: List[int] = Field(default=[128, 64, 32])
    activation: str = Field(default='relu', description="'relu', 'leaky_relu', 'tanh', 'elu'")
    dropout: float = Field(default=0.2, ge=0.0, le=0.8)
    batch_norm: bool = Field(default=True)
    learning_rate: float = Field(default=0.001, ge=0.00001, le=0.1)
    optimizer: str = Field(default='adam', description="'adam', 'sgd', 'adamw'")
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=32, ge=8, le=512)
    early_stopping_patience: int = Field(default=10, ge=1, le=50)
    
    # Hardware
    use_gpu: bool = Field(default=True)


class TrainResponse(BaseModel):
    """Response from training endpoint"""
    job_id: str
    message: str


class JobStatusResponse(BaseModel):
    """Training job status"""
    job_id: str
    state: str  # pending, running, completed, failed
    progress: float
    current_step: str
    logs: List[str]
    experiment_id: Optional[int] = None


# ============== EXPERIMENTS ==============

class ExperimentSummary(BaseModel):
    """Summary of an experiment for listing"""
    model_config = ConfigDict(protected_namespaces=())
    
    id: int
    task: str
    model_type: str
    model_name: str
    status: str
    created_at: datetime
    metrics: Dict[str, Any]
    config: Dict[str, Any]


class ExperimentDetail(BaseModel):
    """Full experiment details"""
    model_config = ConfigDict(protected_namespaces=())
    
    id: int
    task: str
    model_type: str
    model_name: str
    status: str
    created_at: datetime
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    
    # Paths
    model_path: Optional[str]
    preprocessor_path: Optional[str]
    
    # Visualization data
    plots: Dict[str, str]  # Plot name -> base64 encoded image or URL
    
    # MLP specific
    architecture: Optional[Dict[str, Any]] = None
    weights: Optional[Dict[str, Any]] = None
    loss_history: Optional[Dict[str, List[float]]] = None
    sample_activations: Optional[Dict[str, Any]] = None


# ============== PREDICTION ==============

class PredictRequest(BaseModel):
    """Prediction request with validated feature ranges"""
    # Features with range validation
    marka: str
    model: str
    ostecenje: str
    registrovan: str
    gorivo: str
    mjenjac: str
    snaga: float = Field(..., ge=30, le=800, description="Engine power in HP (30-800)")
    kilometraza: float = Field(..., ge=0, le=500000, description="Mileage in km (0-500000)")
    kubikaza: float = Field(..., ge=500, le=8000, description="Engine displacement in cc (500-8000)")
    god: int = Field(..., ge=1980, le=2026, description="Year of manufacture (1980-2026)")
    
    # Model selection
    task: str = Field(default='regression')
    experiment_id: Optional[int] = Field(default=None, description="Use specific model, or latest best if None")


class FeatureInfluence(BaseModel):
    """Influence of a single feature on prediction"""
    display_name: str
    importance: float
    user_value: Any
    percentage: float = 0  # Normalized percentage (all features sum to 100%)
    hint: Optional[str] = None  # Direction hint for numeric features
    is_numeric: Optional[bool] = False


class PredictResponse(BaseModel):
    """Prediction response"""
    model_config = ConfigDict(protected_namespaces=())
    
    prediction: float
    prediction_label: Optional[str] = None  # For classification
    confidence: Optional[float] = None
    experiment_id: int
    model_name: str
    
    # Explanation - each feature with its display name, importance, and user's input value
    top_features: Dict[str, FeatureInfluence]


# ============== SCHEMA ==============

class FeatureSchema(BaseModel):
    """Schema for a single feature"""
    name: str
    type: str  # 'numeric' or 'categorical'
    label: str
    options: Optional[List[str]] = None  # For categorical
    min: Optional[float] = None  # For numeric
    max: Optional[float] = None
    mean: Optional[float] = None


class DatasetSchema(BaseModel):
    """Full dataset schema for frontend form generation"""
    numeric_features: List[FeatureSchema]
    categorical_features: List[FeatureSchema]
    target: str
    brand_models: Dict[str, List[str]] = {}  # Mapping of brand -> list of models


# ============== DATA DISTRIBUTION ==============

class PriceDistribution(BaseModel):
    """Price distribution for visualization"""
    original: Dict[str, Any]
    processed: Dict[str, Any]
    outlier_mode: str


# ============== NETWORK VISUALIZATION ==============

class NetworkVisualizationRequest(BaseModel):
    """Request for forward pass visualization"""
    experiment_id: int
    input_features: Optional[Dict[str, Any]] = None  # If None, use random sample


class NetworkVisualizationResponse(BaseModel):
    """Network visualization data"""
    architecture: Dict[str, Any]
    weights: Dict[str, Any]
    activations: Dict[str, List[List[float]]]  # Layer -> activations for each sample
    input_features: Dict[str, Any]
