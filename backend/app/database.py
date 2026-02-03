"""
Database setup and models using SQLModel
"""
from sqlmodel import SQLModel, Field, create_engine, Session
from typing import Optional
from datetime import datetime
import json
from pydantic import ConfigDict

from app.config import settings


# ============== MODELS ==============

class Experiment(SQLModel, table=True):
    """Stores experiment/training run information"""
    model_config = ConfigDict(protected_namespaces=())
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Task info
    task: str  # 'regression' or 'classification'
    model_type: str  # 'linear', 'rf', 'xgboost', 'mlp'
    model_name: str  # Human readable name
    
    # Configuration (stored as JSON)
    config_json: str = "{}"
    
    # Metrics (stored as JSON)
    metrics_json: str = "{}"
    
    # File paths
    model_path: Optional[str] = None
    preprocessor_path: Optional[str] = None
    plots_dir: Optional[str] = None
    
    # MLP specific - activations and weights for visualization
    activations_path: Optional[str] = None
    weights_path: Optional[str] = None
    loss_history_path: Optional[str] = None
    
    # Metadata
    training_time_seconds: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending, training, completed, failed
    error_message: Optional[str] = None
    
    @property
    def config(self) -> dict:
        return json.loads(self.config_json)
    
    @config.setter
    def config(self, value: dict):
        self.config_json = json.dumps(value)
    
    @property
    def metrics(self) -> dict:
        return json.loads(self.metrics_json)
    
    @metrics.setter
    def metrics(self, value: dict):
        self.metrics_json = json.dumps(value)


class Prediction(SQLModel, table=True):
    """Stores prediction history"""
    id: Optional[int] = Field(default=None, primary_key=True)
    
    experiment_id: int = Field(foreign_key="experiment.id")
    
    # Input features (stored as JSON)
    input_features_json: str = "{}"
    
    # Prediction result
    prediction: float
    prediction_label: Optional[str] = None  # For classification
    confidence: Optional[float] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def input_features(self) -> dict:
        return json.loads(self.input_features_json)
    
    @input_features.setter
    def input_features(self, value: dict):
        self.input_features_json = json.dumps(value)


class TrainingJob(SQLModel, table=True):
    """Tracks async training jobs"""
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(index=True, unique=True)
    
    experiment_id: Optional[int] = Field(default=None, foreign_key="experiment.id")
    
    state: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0  # 0-100
    current_step: str = ""
    logs: str = ""
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ============== DATABASE ENGINE ==============

engine = create_engine(
    settings.database_url,
    echo=False,
    connect_args={"check_same_thread": False}  # SQLite specific
)


def create_db_and_tables():
    """Create all database tables"""
    SQLModel.metadata.create_all(engine)


def get_session():
    """Get database session"""
    with Session(engine) as session:
        yield session
