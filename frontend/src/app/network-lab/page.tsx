'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Select } from '@/components/ui/Select'
import { Badge } from '@/components/ui/Badge'
import { Progress } from '@/components/ui/Progress'
import { Spinner } from '@/components/ui/Spinner'
import { startTraining, getJobStatus, getExperiment, getForwardPass, cancelTraining, TrainRequest, JobStatus, ExperimentDetail } from '@/lib/api'
import { cn, formatNumber } from '@/lib/utils'
import { 
  Brain, 
  Play, 
  Square,
  Settings,
  Layers,
  Plus,
  Minus,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Activity,
  Cpu,
  Target,
  TrendingDown,
  Sparkles,
  HelpCircle,
} from 'lucide-react'
import { Tooltip } from '@/components/ui/Tooltip'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
} from 'recharts'
import NetworkVisualization from '@/components/NetworkVisualization'

type LayerConfig = {
  neurons: number
  activation: string
}

export default function NetworkLabPage() {
  // Network architecture state - larger default for tabular data
  const [layers, setLayers] = useState<LayerConfig[]>([
    { neurons: 256, activation: 'relu' },
    { neurons: 128, activation: 'relu' },
    { neurons: 64, activation: 'relu' },
  ])
  
  // Training config state - tuned defaults for car price prediction
  const [config, setConfig] = useState({
    task: 'regression' as 'regression' | 'classification',
    dropout: 0.3,
    batchNorm: true,
    learningRate: 0.0005,
    optimizer: 'adam',
    epochs: 200,
    batchSize: 64,
    outlierMode: 'log' as 'none' | 'clip' | 'log',  // Log transform - essential for price prediction
    augmentationMode: 'none' as 'none' | 'noise' | 'oversample' | 'both',
    noiseLevel: 0.01,
    useGpu: true,
  })

  // Training state
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [isTraining, setIsTraining] = useState(false)
  const [trainedExperiment, setTrainedExperiment] = useState<ExperimentDetail | null>(null)
  const [visualizationData, setVisualizationData] = useState<any>(null)
  
  // UI state
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [selectedNeuron, setSelectedNeuron] = useState<{ layer: number; neuron: number } | null>(null)

  // Calculate total parameters in the network
  const calculateParameters = () => {
    // Input features (approximately 589 after one-hot encoding based on dataset)
    const inputSize = 589
    let totalParams = 0
    let prevSize = inputSize
    
    // Hidden layers
    for (const layer of layers) {
      // Weights + biases
      totalParams += prevSize * layer.neurons + layer.neurons
      
      // Batch norm params (gamma + beta)
      if (config.batchNorm) {
        totalParams += layer.neurons * 2
      }
      
      prevSize = layer.neurons
    }
    
    // Output layer (1 for regression, 3 for classification)
    const outputSize = config.task === 'regression' ? 1 : 3
    totalParams += prevSize * outputSize + outputSize
    
    return totalParams
  }

  const totalParameters = calculateParameters()

  // Format parameter count for display
  const formatParams = (params: number) => {
    if (params >= 1000000) {
      return `${(params / 1000000).toFixed(2)}M`
    } else if (params >= 1000) {
      return `${(params / 1000).toFixed(1)}K`
    }
    return params.toString()
  }

  // Poll job status
  useEffect(() => {
    if (!jobId || !isTraining) return

    const interval = setInterval(async () => {
      try {
        const res = await getJobStatus(jobId)
        setJobStatus(res.data)

          if (res.data.state === 'completed' || res.data.state === 'failed' || res.data.state === 'cancelled') {
          setIsTraining(false)
          
          if (res.data.state === 'completed' && res.data.experiment_id) {
            const expRes = await getExperiment(res.data.experiment_id)
            setTrainedExperiment(expRes.data)
            
            // Load visualization data
            try {
              const vizRes = await getForwardPass(res.data.experiment_id)
              setVisualizationData(vizRes.data)
            } catch (e) {
              console.log('No visualization data available')
            }
          }
        }
      } catch (error) {
        console.error('Failed to get job status:', error)
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [jobId, isTraining])

  const addLayer = () => {
    if (layers.length >= 8) return
    setLayers([...layers, { neurons: 32, activation: 'relu' }])
  }

  const removeLayer = (index: number) => {
    if (layers.length <= 1) return
    setLayers(layers.filter((_, i) => i !== index))
  }

  const updateLayer = (index: number, field: keyof LayerConfig, value: any) => {
    const newLayers = [...layers]
    newLayers[index] = { ...newLayers[index], [field]: value }
    setLayers(newLayers)
  }

  const startTrain = async () => {
    setIsTraining(true)
    setJobStatus(null)
    setTrainedExperiment(null)
    setVisualizationData(null)

    try {
      const request: TrainRequest = {
        task: config.task,
        model_type: 'mlp',
        hidden_layers: layers.map(l => l.neurons),
        activation: layers[0]?.activation || 'relu',
        dropout: config.dropout,
        batch_norm: config.batchNorm,
        learning_rate: config.learningRate,
        optimizer: config.optimizer,
        epochs: config.epochs,
        batch_size: config.batchSize,
        outlier_mode: config.outlierMode,
        augmentation_mode: config.augmentationMode,
        noise_level: config.noiseLevel,
        use_gpu: config.useGpu,
      }

      const res = await startTraining(request)
      setJobId(res.data.job_id)
    } catch (error) {
      console.error('Failed to start training:', error)
      setIsTraining(false)
    }
  }

  const stopTraining = async () => {
    if (jobId) {
      try {
        await cancelTraining(jobId)
      } catch (error) {
        console.error('Failed to cancel training:', error)
      }
    }
  }

  const resetConfig = () => {
    setLayers([
      { neurons: 256, activation: 'relu' },
      { neurons: 128, activation: 'relu' },
      { neurons: 64, activation: 'relu' },
    ])
    setConfig({
      task: 'regression',
      dropout: 0.3,
      batchNorm: true,
      learningRate: 0.0005,
      optimizer: 'adam',
      epochs: 200,
      batchSize: 64,
      outlierMode: 'log',  // Essential for price prediction
      augmentationMode: 'none',
      noiseLevel: 0.01,
      useGpu: true,
    })
  }

  // Generate random recommended parameters
  const generateRecommended = () => {
    // Carefully tuned configurations for car price prediction
    // Based on: ~7000 samples, ~600 features (one-hot encoded)
    // ALL use 'log' outlier mode - essential for price prediction
    const recommendations = [
      {
        name: 'Stable Learner (Preporuceno)',
        layers: [{ neurons: 256, activation: 'relu' }, { neurons: 128, activation: 'relu' }, { neurons: 64, activation: 'relu' }],
        config: { learningRate: 0.0005, dropout: 0.3, batchSize: 64, epochs: 200, optimizer: 'adam', batchNorm: true, outlierMode: 'log' }
      },
      {
        name: 'Deep Regularized',
        layers: [{ neurons: 512, activation: 'leaky_relu' }, { neurons: 256, activation: 'leaky_relu' }, { neurons: 128, activation: 'leaky_relu' }, { neurons: 64, activation: 'leaky_relu' }],
        config: { learningRate: 0.0003, dropout: 0.4, batchSize: 64, epochs: 300, optimizer: 'adamw', batchNorm: true, outlierMode: 'log' }
      },
      {
        name: 'Wide Shallow',
        layers: [{ neurons: 512, activation: 'relu' }, { neurons: 256, activation: 'relu' }],
        config: { learningRate: 0.0005, dropout: 0.35, batchSize: 128, epochs: 250, optimizer: 'adam', batchNorm: true, outlierMode: 'log' }
      },
      {
        name: 'Conservative',
        layers: [{ neurons: 128, activation: 'elu' }, { neurons: 64, activation: 'elu' }, { neurons: 32, activation: 'elu' }],
        config: { learningRate: 0.0001, dropout: 0.2, batchSize: 32, epochs: 400, optimizer: 'adamw', batchNorm: true, outlierMode: 'log' }
      },
      {
        name: 'High Capacity',
        layers: [{ neurons: 1024, activation: 'leaky_relu' }, { neurons: 512, activation: 'leaky_relu' }, { neurons: 256, activation: 'leaky_relu' }],
        config: { learningRate: 0.0002, dropout: 0.5, batchSize: 64, epochs: 300, optimizer: 'adamw', batchNorm: true, outlierMode: 'log' }
      },
      {
        name: 'Simple Effective',
        layers: [{ neurons: 256, activation: 'relu' }, { neurons: 128, activation: 'relu' }],
        config: { learningRate: 0.001, dropout: 0.25, batchSize: 64, epochs: 200, optimizer: 'adam', batchNorm: true, outlierMode: 'log' }
      },
    ]
    
    const picked = recommendations[Math.floor(Math.random() * recommendations.length)]
    setLayers(picked.layers)
    setConfig(prev => ({
      ...prev,
      ...picked.config,
    }))
    
    // Show which config was picked
    console.log(`Applied: ${picked.name}`)
  }

  // Prepare loss chart data
  const lossChartData = trainedExperiment?.loss_history?.train_loss?.map((loss, i) => ({
    epoch: i + 1,
    train: loss,
    val: trainedExperiment.loss_history!.val_loss[i],
  })) || []

  return (
    <div className="min-h-screen py-8 px-4" suppressHydrationWarning>
      <div className="max-w-7xl mx-auto" suppressHydrationWarning>
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <Badge variant="info" className="mb-2">
            <Brain className="w-3 h-3 mr-1" />
            Network Lab
          </Badge>
          <h1 className="text-3xl font-bold text-white mb-2">
            Build Your Neural Network
          </h1>
          <p className="text-gray-400">
            Design custom MLP architectures, train them, and visualize the learning process
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Configuration Panel */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            {/* Task Selection */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5 text-neon-green" />
                  Task
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Select
                  value={config.task}
                  onChange={(e) => setConfig({ ...config, task: e.target.value as any })}
                  options={[
                    { value: 'regression', label: 'Price Regression' },
                    { value: 'classification', label: 'Price Segment Classification' },
                  ]}
                />
              </CardContent>
            </Card>

            {/* Layer Configuration */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <Layers className="w-5 h-5 text-neon-blue" />
                    Hidden Layers
                    <Tooltip content="Skriveni slojevi mreze. Vise slojeva = dublja mreza, moze nauciti kompleksnije uzorke. Broj neurona po sloju odredjuje sirinu - vise neurona = vise kapaciteta za ucenje.">
                      <HelpCircle className="w-3 h-3 text-gray-500 cursor-help" />
                    </Tooltip>
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    icon={<Plus className="w-4 h-4" />}
                    onClick={addLayer}
                    disabled={layers.length >= 8}
                  >
                    Add
                  </Button>
                </CardTitle>
                <CardDescription className="flex items-center gap-2">
                  Neuroni: broj jedinica u sloju | Aktivacija: 
                  <Tooltip content="ReLU - najcesca, brza, moze umrijeti. LeakyReLU - rjesava dying ReLU problem. Tanh - za podatke centrirane oko 0. ELU - glatka alternativa LeakyReLU.">
                    <span className="text-neon-blue cursor-help underline decoration-dotted">?</span>
                  </Tooltip>
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {layers.map((layer, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex items-center gap-2 p-3 bg-dark-700 rounded-lg"
                  >
                    <span className="text-xs text-gray-500 w-8">L{index + 1}</span>
                    <Input
                      type="number"
                      value={layer.neurons}
                      onChange={(e) => updateLayer(index, 'neurons', parseInt(e.target.value) || 1)}
                      className="w-20 py-2 text-center"
                      min={1}
                      max={512}
                    />
                    <Select
                      value={layer.activation}
                      onChange={(e) => updateLayer(index, 'activation', e.target.value)}
                      options={[
                        { value: 'relu', label: 'ReLU' },
                        { value: 'leaky_relu', label: 'LeakyReLU' },
                        { value: 'tanh', label: 'Tanh' },
                        { value: 'elu', label: 'ELU' },
                      ]}
                      className="flex-1"
                    />
                    <Button
                      variant="ghost"
                      size="sm"
                      icon={<Minus className="w-4 h-4" />}
                      onClick={() => removeLayer(index)}
                      disabled={layers.length <= 1}
                      className="text-red-400 hover:text-red-300"
                    />
                  </motion.div>
                ))}
                
                {/* Total Parameters Display */}
                <div className="mt-4 p-3 bg-gradient-to-r from-neon-green/10 to-neon-blue/10 rounded-lg border border-neon-green/20">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Total Parameters</span>
                    <span className="text-xl font-bold text-neon-green font-mono">
                      {formatParams(totalParameters)}
                    </span>
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {totalParameters.toLocaleString()} trainable parameters
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Advanced Settings */}
            <Card>
              <CardHeader>
                <button 
                  className="w-full flex items-center justify-between cursor-pointer"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  type="button"
                >
                  <span className="flex items-center gap-2">
                    <Settings className="w-5 h-5 text-neon-purple" />
                    <span className="text-lg font-semibold text-white">Advanced Settings</span>
                  </span>
                  {showAdvanced ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
                </button>
              </CardHeader>
              
              {showAdvanced && (
                <CardContent className="space-y-4 pt-0">
                  {/* Random Recommended Button */}
                  <Button
                    variant="secondary"
                    size="sm"
                    className="w-full border-dashed border-neon-purple/50 hover:border-neon-purple"
                    icon={<Sparkles className="w-4 h-4 text-neon-purple" />}
                    onClick={generateRecommended}
                  >
                    Generisi Preporucene Parametre
                  </Button>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label className="text-sm text-gray-300">Learning Rate</label>
                        <Tooltip content="Brzina ucenja - koliko brzo model azurira tezine. Manja vrijednost = sporije ali stabilnije ucenje. Preporuceno: 0.001 za Adam, 0.01 za SGD.">
                          <HelpCircle className="w-3 h-3 text-gray-500 cursor-help" />
                        </Tooltip>
                      </div>
                      <Input
                        type="number"
                        value={config.learningRate}
                        onChange={(e) => setConfig({ ...config, learningRate: parseFloat(e.target.value) || 0.001 })}
                        step={0.0001}
                        min={0.00001}
                        max={0.1}
                      />
                    </div>
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label className="text-sm text-gray-300">Optimizer</label>
                        <Tooltip content="Algoritam optimizacije. Adam - adaptivni, dobar za vecinu slucajeva. SGD - klasican, zahtijeva fino podesavanje. AdamW - Adam sa weight decay regularizacijom.">
                          <HelpCircle className="w-3 h-3 text-gray-500 cursor-help" />
                        </Tooltip>
                      </div>
                      <Select
                        value={config.optimizer}
                        onChange={(e) => setConfig({ ...config, optimizer: e.target.value })}
                        options={[
                          { value: 'adam', label: 'Adam' },
                          { value: 'sgd', label: 'SGD' },
                          { value: 'adamw', label: 'AdamW' },
                        ]}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label className="text-sm text-gray-300">Dropout</label>
                        <Tooltip content="Procenat neurona koji se nasumicno iskljucuje tokom treninga. Sprjecava overfitting. 0.2-0.5 je uobicajeno. Veci = jaca regularizacija.">
                          <HelpCircle className="w-3 h-3 text-gray-500 cursor-help" />
                        </Tooltip>
                      </div>
                      <Input
                        type="number"
                        value={config.dropout}
                        onChange={(e) => setConfig({ ...config, dropout: parseFloat(e.target.value) || 0 })}
                        step={0.1}
                        min={0}
                        max={0.8}
                      />
                    </div>
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label className="text-sm text-gray-300">Epochs</label>
                        <Tooltip content="Broj prolaza kroz cijeli dataset. Vise epoha = duze ucenje. Early stopping automatski zaustavlja ako se model prestane poboljsavati.">
                          <HelpCircle className="w-3 h-3 text-gray-500 cursor-help" />
                        </Tooltip>
                      </div>
                      <Input
                        type="number"
                        value={config.epochs}
                        onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) || 100 })}
                        min={1}
                        max={1000}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label className="text-sm text-gray-300">Batch Size</label>
                        <Tooltip content="Broj uzoraka u jednom koraku ucenja. Manji = manje memorije, vise suma u gradijentima. Veci = stabilnije, ali sporije. 32-64 je standard.">
                          <HelpCircle className="w-3 h-3 text-gray-500 cursor-help" />
                        </Tooltip>
                      </div>
                      <Input
                        type="number"
                        value={config.batchSize}
                        onChange={(e) => setConfig({ ...config, batchSize: parseInt(e.target.value) || 32 })}
                        min={8}
                        max={512}
                      />
                    </div>
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label className="text-sm text-gray-300">Outlier Mode</label>
                        <Tooltip content="PREPORUCENO: Log Transform! Cijene (1k-50k EUR) su prevelike za NN. Log kompresuje range i drasticno poboljsava ucenje. None cesto daje R2=0.">
                          <HelpCircle className="w-3 h-3 text-yellow-500 cursor-help" />
                        </Tooltip>
                      </div>
                      <Select
                        value={config.outlierMode}
                        onChange={(e) => setConfig({ ...config, outlierMode: e.target.value as any })}
                        options={[
                          { value: 'none', label: 'None (nije preporuceno)' },
                          { value: 'clip', label: 'Clip (Winsorize)' },
                          { value: 'log', label: 'Log Transform (preporuceno)' },
                        ]}
                      />
                    </div>
                  </div>

                  {/* Data Augmentation */}
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label className="text-sm text-gray-300">Data Augmentation</label>
                        <Tooltip content="Povecanje dataseta umjetnim uzorcima. Noise = dodaje Gaussian sum. SMOTE = balansira klase stvaranjem sintetickih uzoraka. Pomaze kod malih dataseta i overfittinga.">
                          <HelpCircle className="w-3 h-3 text-gray-500 cursor-help" />
                        </Tooltip>
                      </div>
                      <Select
                        value={config.augmentationMode}
                        onChange={(e) => setConfig({ ...config, augmentationMode: e.target.value as any })}
                        options={[
                          { value: 'none', label: 'None' },
                          { value: 'noise', label: 'Gaussian Noise' },
                          { value: 'oversample', label: config.task === 'classification' ? 'SMOTE Balance' : 'Oversample' },
                          { value: 'both', label: 'Noise + Oversample' },
                        ]}
                      />
                    </div>
                    {config.augmentationMode !== 'none' && (
                      <div>
                        <div className="flex items-center gap-1 mb-1">
                          <label className="text-sm text-gray-300">Noise Level</label>
                          <Tooltip content="Intenzitet suma koji se dodaje podacima. 0.01 = 1% standardne devijacije. Vece vrijednosti = vise varijacije u podacima.">
                            <HelpCircle className="w-3 h-3 text-gray-500 cursor-help" />
                          </Tooltip>
                        </div>
                        <Input
                          type="number"
                          value={config.noiseLevel}
                          onChange={(e) => setConfig({ ...config, noiseLevel: parseFloat(e.target.value) || 0.01 })}
                          step={0.005}
                          min={0.001}
                          max={0.1}
                        />
                      </div>
                    )}
                  </div>

                  <div className="flex items-center gap-4 pt-2">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={config.batchNorm}
                        onChange={(e) => setConfig({ ...config, batchNorm: e.target.checked })}
                        className="w-4 h-4 rounded border-dark-500 bg-dark-700 text-neon-green focus:ring-neon-green"
                      />
                      <span className="text-sm text-gray-300">Batch Norm</span>
                      <Tooltip content="Batch Normalization - normalizuje izlaze svakog sloja. Ubrzava trening, omogucava veci learning rate i djeluje kao regularizacija.">
                        <HelpCircle className="w-3 h-3 text-gray-500 cursor-help" />
                      </Tooltip>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={config.useGpu}
                        onChange={(e) => setConfig({ ...config, useGpu: e.target.checked })}
                        className="w-4 h-4 rounded border-dark-500 bg-dark-700 text-neon-green focus:ring-neon-green"
                      />
                      <span className="text-sm text-gray-300">Use GPU</span>
                      <Tooltip content="Koristi CUDA GPU za brzi trening (ako je dostupan). GPU moze ubrzati trening 10-100x u odnosu na CPU.">
                        <HelpCircle className="w-3 h-3 text-gray-500 cursor-help" />
                      </Tooltip>
                    </label>
                  </div>
                </CardContent>
              )}
            </Card>

            {/* Warning for no log transform */}
            {config.outlierMode !== 'log' && config.task === 'regression' && (
              <div className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg" suppressHydrationWarning>
                <p className="text-yellow-400 text-sm flex items-center gap-2" suppressHydrationWarning>
                  <span className="text-lg font-bold">!</span>
                  <span suppressHydrationWarning>
                    <strong>Preporuka:</strong> Za predikciju cijena, koristi Log Transform outlier mode. 
                    Bez toga, mreza ima poteskoca sa velikim vrijednostima.
                  </span>
                </p>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-3">
              <Button
                className="flex-1"
                size="lg"
                onClick={startTrain}
                loading={isTraining}
                icon={<Play className="w-5 h-5" />}
                disabled={isTraining}
              >
                {isTraining ? 'Training...' : 'Train Network'}
              </Button>
              <Button
                variant="secondary"
                size="lg"
                icon={<RotateCcw className="w-5 h-5" />}
                onClick={resetConfig}
                disabled={isTraining}
              />
            </div>
          </motion.div>

          {/* Visualization Panel */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-2 space-y-6"
          >
            {/* Network Visualization */}
            <Card className="min-h-[550px]" style={{ resize: 'vertical', overflow: 'hidden' }}>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <Activity className="w-5 h-5 text-neon-green" />
                    Network Architecture
                  </span>
                  <div className="flex items-center gap-2">
                    {trainedExperiment && (
                      <Badge variant="success">Trained</Badge>
                    )}
                    <span className="text-xs text-gray-500">Drag corner to resize</span>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent className="h-[calc(100%-4rem)]">
                <NetworkVisualization
                  layers={[
                    { name: 'input', neurons: 50, type: 'input' },
                    ...layers.map((l, i) => ({
                      name: `hidden_${i}`,
                      neurons: l.neurons,
                      type: 'hidden' as const,
                      activation: l.activation,
                    })),
                    { name: 'output', neurons: config.task === 'regression' ? 1 : 3, type: 'output' },
                  ]}
                  weights={visualizationData?.weights}
                  activations={visualizationData?.activations}
                  onNeuronClick={setSelectedNeuron}
                />
              </CardContent>
            </Card>

            {/* Training Progress / Results */}
            {(isTraining || jobStatus) && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span className="flex items-center gap-2">
                      <Cpu className="w-5 h-5 text-neon-blue" />
                      Training Progress
                    </span>
                    {isTraining && (
                      <Button
                        variant="danger"
                        size="sm"
                        icon={<Square className="w-4 h-4" />}
                        onClick={stopTraining}
                      >
                        Stop
                      </Button>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {jobStatus && (
                    <div className="space-y-4">
                      {/* Status and Epoch */}
                      <div className="flex items-center justify-between">
                        <Badge 
                          variant={
                            jobStatus.state === 'completed' ? 'success' :
                            jobStatus.state === 'failed' ? 'error' :
                            jobStatus.state === 'cancelled' ? 'warning' :
                            jobStatus.state === 'running' ? 'info' : 'warning'
                          }
                        >
                          {jobStatus.state.toUpperCase()}
                        </Badge>
                        <span className="text-sm font-mono text-white">
                          Epoch {jobStatus.epoch || 0} / {jobStatus.total_epochs || config.epochs}
                        </span>
                      </div>
                      
                      {/* Progress bar with percentage */}
                      <div>
                        <div className="flex justify-between text-xs text-gray-400 mb-1">
                          <span>Progress</span>
                          <span className="font-mono">{jobStatus.progress.toFixed(1)}%</span>
                        </div>
                        <Progress value={jobStatus.progress} />
                      </div>

                      {/* Real-time metrics */}
                      {(jobStatus.train_loss !== null || jobStatus.val_loss !== null) && (
                        <div className="grid grid-cols-3 gap-3">
                          <div className="bg-dark-700 rounded-lg p-3 text-center">
                            <p className="text-xs text-gray-500 mb-1">Train Loss</p>
                            <p className="text-lg font-mono text-neon-green">
                              {jobStatus.train_loss?.toFixed(4) || '-'}
                            </p>
                          </div>
                          <div className="bg-dark-700 rounded-lg p-3 text-center">
                            <p className="text-xs text-gray-500 mb-1">Val Loss</p>
                            <p className="text-lg font-mono text-neon-blue">
                              {jobStatus.val_loss?.toFixed(4) || '-'}
                            </p>
                          </div>
                          <div className="bg-dark-700 rounded-lg p-3 text-center">
                            <p className="text-xs text-gray-500 mb-1">Best Loss</p>
                            <p className="text-lg font-mono text-yellow-400">
                              {jobStatus.best_metric?.toFixed(4) || '-'}
                            </p>
                          </div>
                        </div>
                      )}

                      {/* Logs */}
                      <div>
                        <p className="text-xs text-gray-500 mb-2">Training Log</p>
                        <div className="max-h-24 overflow-auto bg-dark-700 rounded-lg p-3 font-mono text-xs text-gray-400">
                          {jobStatus.logs.slice(-8).map((log, i) => (
                            <div key={i} className="truncate">{log}</div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Results */}
            {trainedExperiment && (
              <div className="grid md:grid-cols-2 gap-6">
                {/* Metrics */}
                <Card>
                  <CardHeader>
                    <CardTitle>Results</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {config.task === 'regression' ? (
                        <>
                          <div className="flex justify-between items-center p-3 bg-dark-700 rounded-lg">
                            <span className="text-gray-400">Test R2</span>
                            <span className="text-xl font-bold text-neon-green" suppressHydrationWarning>
                              {trainedExperiment.metrics?.test?.r2?.toFixed(4) || '-'}
                            </span>
                          </div>
                          <div className="flex justify-between items-center p-3 bg-dark-700 rounded-lg">
                            <span className="text-gray-400">Test MAE</span>
                            <span className="text-xl font-bold text-neon-blue" suppressHydrationWarning>
                              EUR {trainedExperiment.metrics?.test?.mae?.toFixed(0) || '-'}
                            </span>
                          </div>
                          <div className="flex justify-between items-center p-3 bg-dark-700 rounded-lg">
                            <span className="text-gray-400">Test RMSE</span>
                            <span className="text-xl font-bold text-white" suppressHydrationWarning>
                              EUR {trainedExperiment.metrics?.test?.rmse?.toFixed(0) || '-'}
                            </span>
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="flex justify-between items-center p-3 bg-dark-700 rounded-lg">
                            <span className="text-gray-400">Test Accuracy</span>
                            <span className="text-xl font-bold text-neon-green">
                              {((trainedExperiment.metrics?.test?.accuracy || 0) * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="flex justify-between items-center p-3 bg-dark-700 rounded-lg">
                            <span className="text-gray-400">Test F1 (Macro)</span>
                            <span className="text-xl font-bold text-neon-blue">
                              {((trainedExperiment.metrics?.test?.f1_macro || 0) * 100).toFixed(1)}%
                            </span>
                          </div>
                        </>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Learning Curves */}
                <Card>
                  <CardHeader>
                    <CardTitle>Learning Curves</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={lossChartData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#2d2d4a" />
                          <XAxis dataKey="epoch" stroke="#9ca3af" fontSize={10} />
                          <YAxis stroke="#9ca3af" fontSize={10} />
                          <RechartsTooltip
                            contentStyle={{
                              backgroundColor: '#1a1a2e',
                              border: '1px solid #2d2d4a',
                              borderRadius: '8px',
                            }}
                          />
                          <Line
                            type="monotone"
                            dataKey="train"
                            stroke="#00ff88"
                            strokeWidth={2}
                            dot={false}
                            name="Train"
                          />
                          <Line
                            type="monotone"
                            dataKey="val"
                            stroke="#00d4ff"
                            strokeWidth={2}
                            dot={false}
                            name="Val"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

          </motion.div>
        </div>
      </div>
    </div>
  )
}
