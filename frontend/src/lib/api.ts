import axios from 'axios'

const API_BASE = 'http://localhost:8000'

export const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Types
export interface FeatureSchema {
  name: string
  type: 'numeric' | 'categorical'
  label: string
  options?: string[]
  min?: number
  max?: number
  mean?: number
}

export interface DatasetSchema {
  numeric_features: FeatureSchema[]
  categorical_features: FeatureSchema[]
  target: string
  brand_models: Record<string, string[]>  // Mapping of brand -> list of models
}

export interface TrainRequest {
  task: 'regression' | 'classification'
  model_type: 'linear' | 'rf' | 'xgboost' | 'mlp'
  outlier_mode?: 'none' | 'clip' | 'log'
  augmentation_mode?: 'none' | 'noise' | 'oversample' | 'both'
  noise_level?: number
  test_size?: number
  val_size?: number
  random_seed?: number
  // Tree model config
  n_estimators?: number
  max_depth?: number | null
  tree_learning_rate?: number
  // MLP config
  hidden_layers?: number[]
  activation?: string
  dropout?: number
  batch_norm?: boolean
  learning_rate?: number
  optimizer?: string
  epochs?: number
  batch_size?: number
  early_stopping_patience?: number
  use_gpu?: boolean
}

export interface JobStatus {
  job_id: string
  state: 'pending' | 'running' | 'completed' | 'failed' | 'cancelling' | 'cancelled'
  progress: number
  current_step: string
  logs: string[]
  experiment_id?: number
  epoch?: number
  total_epochs?: number
  train_loss?: number | null
  val_loss?: number | null
  best_metric?: number | null
}

export interface ExperimentSummary {
  id: number
  task: string
  model_type: string
  model_name: string
  status: string
  created_at: string
  metrics: {
    train?: any
    val?: any
    test?: any
    feature_importance?: Record<string, number>
  }
  config: any
}

export interface ExperimentDetail extends ExperimentSummary {
  model_path?: string
  preprocessor_path?: string
  plots: Record<string, string>
  architecture?: any
  weights?: any
  loss_history?: {
    train_loss: number[]
    val_loss: number[]
    train_mae?: number[]
    val_mae?: number[]
    train_acc?: number[]
    val_acc?: number[]
    learning_rate?: number[]
  }
  sample_activations?: any
}

export interface PredictRequest {
  marka: string
  model: string
  ostecenje: string
  registrovan: string
  gorivo: string
  mjenjac: string
  snaga: number
  kilometraza: number
  kubikaza: number
  god: number
  task?: string
  experiment_id?: number
}

export interface PredictResponse {
  prediction: number
  prediction_label?: string
  confidence?: number
  experiment_id: number
  model_name: string
  top_features: Record<string, number>
}

// API Functions
export const getHealth = () => api.get('/health')

export const getSchema = () => api.get<DatasetSchema>('/schema')

export const getDatasetStats = () => api.get('/dataset/stats')

export const getPriceDistribution = (outlierMode: string = 'none') => 
  api.get('/price-distribution', { params: { outlier_mode: outlierMode } })

export const startTraining = (config: TrainRequest) => 
  api.post<{ job_id: string; message: string }>('/train', config)

export const getJobStatus = (jobId: string) => 
  api.get<JobStatus>(`/train/${jobId}/status`)

export const cancelTraining = (jobId: string) => 
  api.post(`/train/${jobId}/cancel`)

export const getExperiments = (task?: string, status?: string) => 
  api.get<ExperimentSummary[]>('/experiments', { params: { task, status } })

export const getExperiment = (id: number) => 
  api.get<ExperimentDetail>(`/experiments/${id}`)

export const deleteExperiment = (id: number) => 
  api.delete(`/experiments/${id}`)

export const renameExperiment = (id: number, name: string) => 
  api.patch(`/experiments/${id}`, null, { params: { name } })

export const predict = (request: PredictRequest) => 
  api.post<PredictResponse>('/predict', request)

export const compareModels = (task: string) => 
  api.get('/compare', { params: { task } })

export const getForwardPass = (experimentId: number, inputFeatures?: any) => 
  api.post('/network/forward-pass', { 
    experiment_id: experimentId, 
    input_features: inputFeatures 
  })
