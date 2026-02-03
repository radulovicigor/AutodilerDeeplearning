"""
Application configuration using pydantic-settings
"""
from pydantic_settings import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///./autovalue.db"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # ML Settings
    default_random_seed: int = 42
    use_gpu: bool = True
    
    # Paths (relative to backend folder)
    data_path: str = "../data/autici_7k.csv"
    models_dir: str = "../models"
    experiments_dir: str = "../experiments"
    
    # Base directory (project root)
    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent.parent.parent
    
    class Config:
        env_file = ".env"
        extra = "ignore"
    
    @property
    def data_path_resolved(self) -> Path:
        """Get resolved data path"""
        # Path relative to project root (not backend)
        return Path(__file__).parent.parent.parent / "data" / "autici_7k.csv"
    
    @property
    def models_dir_resolved(self) -> Path:
        """Get resolved models directory"""
        path = Path(__file__).parent.parent.parent / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def experiments_dir_resolved(self) -> Path:
        """Get resolved experiments directory"""
        path = Path(__file__).parent.parent.parent / "experiments"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
