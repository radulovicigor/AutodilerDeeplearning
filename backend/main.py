"""
AutoValue AI - FastAPI Backend
Main application entry point
"""
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlmodel import Session, select
from pathlib import Path
import json
import base64
from typing import List, Optional

from app.config import settings
from app.database import create_db_and_tables, get_session, Experiment, Prediction
from app.schemas import (
    TrainRequest, TrainResponse, JobStatusResponse,
    ExperimentSummary, ExperimentDetail,
    PredictRequest, PredictResponse,
    DatasetSchema, PriceDistribution,
    NetworkVisualizationRequest, NetworkVisualizationResponse,
)
from app.training_service import create_training_job, get_job_status, cancel_job, TrainingLimitExceeded
from app.prediction_service import make_prediction, get_forward_pass_visualization
from app.data_processing import get_dataset_schema, get_price_distribution, load_dataset, get_data_quality_report


# Create FastAPI app
app = FastAPI(
    title="AutoValue AI",
    description="Deep Learning for Car Price Prediction",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for plots
plots_dir = settings.experiments_dir_resolved
if plots_dir.exists():
    app.mount("/static", StaticFiles(directory=str(plots_dir.parent)), name="static")


# ============== STARTUP ==============

@app.on_event("startup")
def on_startup():
    """Initialize database on startup"""
    create_db_and_tables()
    print("Database initialized")
    
    # Ensure directories exist
    settings.models_dir_resolved.mkdir(parents=True, exist_ok=True)
    settings.experiments_dir_resolved.mkdir(parents=True, exist_ok=True)
    print("Directories ready")


# ============== HEALTH ==============

@app.get("/health")
def health_check():
    """Health check endpoint"""
    import torch
    
    cuda_available = torch.cuda.is_available()
    cuda_device = None
    cuda_version = None
    
    if cuda_available:
        try:
            cuda_device = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
        except Exception as e:
            cuda_device = f"Error: {str(e)}"
    
    return {
        "status": "healthy",
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "cuda_version": cuda_version,
        "pytorch_version": torch.__version__,
    }


# ============== SCHEMA ==============

@app.get("/schema", response_model=DatasetSchema)
def get_schema():
    """Get dataset schema for frontend form generation"""
    try:
        schema = get_dataset_schema()
        return schema
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/price-distribution")
def get_distribution(outlier_mode: str = Query(default='none')):
    """Get price distribution for visualization"""
    try:
        return get_price_distribution(outlier_mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data-quality")
def get_quality_report():
    """
    Get data quality report showing outlier statistics for all columns.
    This helps understand data cleaning that was applied.
    """
    try:
        df = load_dataset()
        report = get_data_quality_report(df)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== TRAINING ==============

@app.post("/train", response_model=TrainResponse)
def start_training(request: TrainRequest):
    """
    Start model training job
    
    Returns job_id for status polling
    Returns 429 if max concurrent training limit is reached
    """
    try:
        config = request.model_dump()
        
        job_id = create_training_job(
            task=request.task,
            model_type=request.model_type,
            config=config,
            use_gpu=request.use_gpu,
        )
        
        return TrainResponse(
            job_id=job_id,
            message=f"Training started. Poll /train/{job_id}/status for updates."
        )
    except TrainingLimitExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/train/{job_id}/status")
def get_training_status(job_id: str):
    """Get training job status with detailed metrics"""
    status = get_job_status(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return {
        'job_id': job_id,
        'state': status.get('state', 'unknown'),
        'progress': status.get('progress', 0),
        'current_step': status.get('current_step', ''),
        'logs': status.get('logs', []),
        'experiment_id': status.get('experiment_id'),
        'epoch': status.get('epoch', 0),
        'total_epochs': status.get('total_epochs', 100),
        'train_loss': status.get('train_loss'),
        'val_loss': status.get('val_loss'),
        'best_metric': status.get('best_metric'),
    }


@app.post("/train/{job_id}/cancel")
def cancel_training(job_id: str):
    """Cancel a running training job"""
    success = cancel_job(job_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found or already completed")
    
    return {"message": f"Cancellation requested for job {job_id}"}


# ============== EXPERIMENTS ==============

@app.get("/experiments", response_model=List[ExperimentSummary])
def list_experiments(
    task: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    session: Session = Depends(get_session)
):
    """List all experiments with optional filtering"""
    query = select(Experiment)
    
    if task:
        query = query.where(Experiment.task == task)
    if status:
        query = query.where(Experiment.status == status)
    
    query = query.order_by(Experiment.created_at.desc())
    
    experiments = session.exec(query).all()
    
    return [
        ExperimentSummary(
            id=exp.id,
            task=exp.task,
            model_type=exp.model_type,
            model_name=exp.model_name,
            status=exp.status,
            created_at=exp.created_at,
            metrics=exp.metrics,
            config=exp.config,
        )
        for exp in experiments
    ]


@app.delete("/experiments/{experiment_id}")
def delete_experiment(experiment_id: int, session: Session = Depends(get_session)):
    """Delete an experiment"""
    experiment = session.get(Experiment, experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    # Delete associated files
    if experiment.model_path and Path(experiment.model_path).exists():
        Path(experiment.model_path).unlink()
    if experiment.preprocessor_path and Path(experiment.preprocessor_path).exists():
        Path(experiment.preprocessor_path).unlink()
    if experiment.plots_dir and Path(experiment.plots_dir).exists():
        import shutil
        shutil.rmtree(experiment.plots_dir, ignore_errors=True)
    if experiment.weights_path and Path(experiment.weights_path).exists():
        Path(experiment.weights_path).unlink()
    if experiment.loss_history_path and Path(experiment.loss_history_path).exists():
        Path(experiment.loss_history_path).unlink()
    if experiment.activations_path and Path(experiment.activations_path).exists():
        Path(experiment.activations_path).unlink()
    
    session.delete(experiment)
    session.commit()
    
    return {"message": f"Experiment {experiment_id} deleted successfully"}


@app.patch("/experiments/{experiment_id}")
def update_experiment(
    experiment_id: int, 
    name: str = Query(..., description="New name for the experiment"),
    session: Session = Depends(get_session)
):
    """Rename an experiment"""
    experiment = session.get(Experiment, experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    experiment.model_name = name
    session.add(experiment)
    session.commit()
    session.refresh(experiment)
    
    return {"message": f"Experiment renamed to '{name}'", "model_name": name}


@app.get("/experiments/{experiment_id}", response_model=ExperimentDetail)
def get_experiment(experiment_id: int, session: Session = Depends(get_session)):
    """Get detailed experiment information"""
    experiment = session.get(Experiment, experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    # Load plots as base64 or paths
    plots = {}
    if experiment.plots_dir and Path(experiment.plots_dir).exists():
        plots_dir = Path(experiment.plots_dir)
        for plot_file in plots_dir.glob("*.png"):
            # Return relative URL path
            relative_path = plot_file.relative_to(settings.experiments_dir_resolved.parent)
            plots[plot_file.stem] = f"/static/{relative_path.as_posix()}"
    
    # Load MLP-specific data
    architecture = None
    weights = None
    loss_history = None
    sample_activations = None
    
    if experiment.model_type == 'mlp':
        metrics = experiment.metrics
        
        # Architecture from metrics
        if 'architecture' in metrics:
            architecture = metrics.get('architecture')
        
        # Load weights if available
        if experiment.weights_path and Path(experiment.weights_path).exists():
            with open(experiment.weights_path, 'r') as f:
                weights = json.load(f)
        
        # Load loss history
        if experiment.loss_history_path and Path(experiment.loss_history_path).exists():
            with open(experiment.loss_history_path, 'r') as f:
                loss_history = json.load(f)
        
        # Load activations
        if experiment.activations_path and Path(experiment.activations_path).exists():
            with open(experiment.activations_path, 'r') as f:
                sample_activations = json.load(f)
    
    return ExperimentDetail(
        id=experiment.id,
        task=experiment.task,
        model_type=experiment.model_type,
        model_name=experiment.model_name,
        status=experiment.status,
        created_at=experiment.created_at,
        config=experiment.config,
        metrics=experiment.metrics,
        model_path=experiment.model_path,
        preprocessor_path=experiment.preprocessor_path,
        plots=plots,
        architecture=architecture,
        weights=weights,
        loss_history=loss_history,
        sample_activations=sample_activations,
    )


@app.delete("/experiments/{experiment_id}")
def delete_experiment(experiment_id: int, session: Session = Depends(get_session)):
    """Delete an experiment and its artifacts"""
    experiment = session.get(Experiment, experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    # Delete associated predictions
    predictions = session.exec(
        select(Prediction).where(Prediction.experiment_id == experiment_id)
    ).all()
    for pred in predictions:
        session.delete(pred)
    
    # Delete experiment
    session.delete(experiment)
    session.commit()
    
    # Optionally delete files (model, plots, etc.)
    # For now, keep them for debugging
    
    return {"message": f"Experiment {experiment_id} deleted"}


# ============== PREDICTION ==============

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Make a prediction using trained model"""
    try:
        features = {
            'marka': request.marka,
            'model': request.model,
            'ostecenje': request.ostecenje,
            'registrovan': request.registrovan,
            'gorivo': request.gorivo,
            'mjenjac': request.mjenjac,
            'snaga': request.snaga,
            'kilometraza': request.kilometraza,
            'kubikaza': request.kubikaza,
            'god': request.god,
        }
        
        result, experiment_id = make_prediction(
            features=features,
            task=request.task,
            experiment_id=request.experiment_id,
        )
        
        return PredictResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== NETWORK VISUALIZATION ==============

@app.post("/network/forward-pass")
def forward_pass_visualization(request: NetworkVisualizationRequest):
    """Get data for forward pass visualization"""
    try:
        result = get_forward_pass_visualization(
            experiment_id=request.experiment_id,
            features=request.input_features,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== DATASET INFO ==============

@app.get("/dataset/stats")
def get_dataset_stats():
    """Get basic dataset statistics"""
    try:
        df = load_dataset()
        
        return {
            'total_rows': len(df),
            'columns': list(df.columns),
            'price_stats': {
                'min': float(df['cena'].min()),
                'max': float(df['cena'].max()),
                'mean': float(df['cena'].mean()),
                'median': float(df['cena'].median()),
                'std': float(df['cena'].std()),
            },
            'categorical_counts': {
                col: df[col].value_counts().to_dict()
                for col in ['marka', 'gorivo', 'mjenjac']
            },
            'year_range': {
                'min': int(df['god'].min()),
                'max': int(df['god'].max()),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== MODEL COMPARISON ==============

@app.get("/compare")
def compare_models(
    task: str = Query(default='regression'),
    session: Session = Depends(get_session)
):
    """Compare all trained models for a given task"""
    experiments = session.exec(
        select(Experiment)
        .where(Experiment.task == task)
        .where(Experiment.status == 'completed')
        .order_by(Experiment.created_at.desc())
    ).all()
    
    comparison = []
    for exp in experiments:
        metrics = exp.metrics
        test_metrics = metrics.get('test', {})
        
        entry = {
            'id': exp.id,
            'model_name': exp.model_name,
            'model_type': exp.model_type,
            'created_at': exp.created_at.isoformat(),
        }
        
        if task == 'regression':
            entry.update({
                'mae': test_metrics.get('mae'),
                'rmse': test_metrics.get('rmse'),
                'r2': test_metrics.get('r2'),
            })
        else:
            entry.update({
                'accuracy': test_metrics.get('accuracy'),
                'f1_macro': test_metrics.get('f1_macro'),
            })
        
        # Add config summary
        config = exp.config
        if exp.model_type == 'mlp':
            entry['config_summary'] = f"layers={config.get('hidden_layers')}, lr={config.get('learning_rate')}, opt={config.get('optimizer')}"
        else:
            entry['config_summary'] = f"n_est={config.get('n_estimators', 'N/A')}"
        
        comparison.append(entry)
    
    return comparison


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
