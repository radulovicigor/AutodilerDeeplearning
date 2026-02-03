'use client'

import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Select } from '@/components/ui/Select'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { Tooltip } from '@/components/ui/Tooltip'
import { getSchema, getExperiments, predict, DatasetSchema, ExperimentSummary, PredictResponse } from '@/lib/api'
import { formatCurrency, cn } from '@/lib/utils'
import { 
  Calculator, 
  Car, 
  Gauge, 
  Calendar, 
  Fuel, 
  Settings2,
  Info,
  TrendingUp,
  CheckCircle2,
  AlertCircle
} from 'lucide-react'

export default function PredictPage() {
  const [schema, setSchema] = useState<DatasetSchema | null>(null)
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [predicting, setPredicting] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Form state
  const [formData, setFormData] = useState({
    marka: '',
    model: '',
    ostecenje: '',
    registrovan: '',
    gorivo: '',
    mjenjac: '',
    snaga: '',
    kilometraza: '',
    kubikaza: '',
    god: '',
    task: 'regression',
    experiment_id: '',
  })

  useEffect(() => {
    async function fetchData() {
      try {
        const [schemaRes, expRes] = await Promise.all([
          getSchema(),
          getExperiments(undefined, 'completed'),
        ])
        setSchema(schemaRes.data)
        setExperiments(expRes.data)
        
        // Set default values from schema
        const data = schemaRes.data
        const defaultBrand = data.categorical_features.find(f => f.name === 'marka')?.options?.[0] || ''
        const defaultModel = data.brand_models?.[defaultBrand]?.[0] || ''
        
        setFormData(prev => ({
          ...prev,
          marka: defaultBrand,
          model: defaultModel,
          ostecenje: data.categorical_features.find(f => f.name === 'ostecenje')?.options?.[0] || '',
          registrovan: data.categorical_features.find(f => f.name === 'registrovan')?.options?.[0] || '',
          gorivo: data.categorical_features.find(f => f.name === 'gorivo')?.options?.[0] || '',
          mjenjac: data.categorical_features.find(f => f.name === 'mjenjac')?.options?.[0] || '',
        }))
      } catch (err) {
        setError('Failed to load data. Make sure backend is running.')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setPredicting(true)
    setError(null)
    setResult(null)

    try {
      const response = await predict({
        marka: formData.marka,
        model: formData.model,
        ostecenje: formData.ostecenje,
        registrovan: formData.registrovan,
        gorivo: formData.gorivo,
        mjenjac: formData.mjenjac,
        snaga: parseFloat(formData.snaga),
        kilometraza: parseFloat(formData.kilometraza),
        kubikaza: parseFloat(formData.kubikaza),
        god: parseInt(formData.god),
        task: formData.task,
        experiment_id: formData.experiment_id ? parseInt(formData.experiment_id) : undefined,
      })
      setResult(response.data)
    } catch (err: any) {
      // Handle different error formats
      const errorData = err.response?.data
      let errorMessage = 'Prediction failed'
      
      if (typeof errorData === 'string') {
        errorMessage = errorData
      } else if (errorData?.detail) {
        if (typeof errorData.detail === 'string') {
          errorMessage = errorData.detail
        } else if (Array.isArray(errorData.detail)) {
          // Pydantic validation errors
          errorMessage = errorData.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
        } else {
          errorMessage = JSON.stringify(errorData.detail)
        }
      }
      
      setError(errorMessage)
    } finally {
      setPredicting(false)
    }
  }

  const updateFormData = (field: string, value: string) => {
    setFormData(prev => {
      const newData = { ...prev, [field]: value }
      
      // When brand changes, reset model to first available model for that brand
      if (field === 'marka' && schema?.brand_models) {
        const models = schema.brand_models[value] || []
        newData.model = models[0] || ''
      }
      
      return newData
    })
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Spinner size="lg" />
      </div>
    )
  }

  if (!schema) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Card className="max-w-md text-center">
          <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
          <CardTitle>Connection Error</CardTitle>
          <CardDescription>
            Could not connect to backend. Make sure the server is running on port 8000.
          </CardDescription>
        </Card>
      </div>
    )
  }

  const getOptions = (name: string) => {
    const feature = schema.categorical_features.find(f => f.name === name)
    return feature?.options?.map(opt => ({ value: opt, label: opt })) || []
  }

  // Get models filtered by selected brand
  const getModelOptions = () => {
    if (!schema?.brand_models || !formData.marka) {
      return getOptions('model')
    }
    const models = schema.brand_models[formData.marka] || []
    return models.map(m => ({ value: m, label: m }))
  }

  const filteredExperiments = experiments.filter(e => e.task === formData.task)

  return (
    <div className="min-h-screen py-8 px-4" suppressHydrationWarning>
      <div className="max-w-6xl mx-auto" suppressHydrationWarning>
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <Badge variant="success" className="mb-4">
            <Calculator className="w-3 h-3 mr-1" />
            Price Prediction
          </Badge>
          <h1 className="text-3xl font-bold text-white mb-2">
            Predict Car Price
          </h1>
          <p className="text-gray-400">
            Enter vehicle details to get an estimated price using trained ML models
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Form */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-2"
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Car className="w-5 h-5 text-neon-green" />
                  Vehicle Details
                </CardTitle>
                <CardDescription>
                  Fill in all fields to get an accurate prediction
                </CardDescription>
              </CardHeader>
              
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-6">
                  {/* Basic Info */}
                  <div className="grid md:grid-cols-2 gap-4">
                    <Select
                      label="Marka (Brand)"
                      value={formData.marka}
                      onChange={(e) => updateFormData('marka', e.target.value)}
                      options={getOptions('marka')}
                    />
                    <Select
                      label="Model"
                      value={formData.model}
                      onChange={(e) => updateFormData('model', e.target.value)}
                      options={getModelOptions()}
                    />
                  </div>

                  {/* Condition & Registration */}
                  <div className="grid md:grid-cols-2 gap-4">
                    <Select
                      label="Oštećenje (Damage)"
                      value={formData.ostecenje}
                      onChange={(e) => updateFormData('ostecenje', e.target.value)}
                      options={getOptions('ostecenje')}
                    />
                    <Select
                      label="Registrovan (Registered)"
                      value={formData.registrovan}
                      onChange={(e) => updateFormData('registrovan', e.target.value)}
                      options={getOptions('registrovan')}
                    />
                  </div>

                  {/* Engine & Transmission */}
                  <div className="grid md:grid-cols-2 gap-4">
                    <Select
                      label="Gorivo (Fuel Type)"
                      value={formData.gorivo}
                      onChange={(e) => updateFormData('gorivo', e.target.value)}
                      options={getOptions('gorivo')}
                    />
                    <Select
                      label="Mjenjač (Transmission)"
                      value={formData.mjenjac}
                      onChange={(e) => updateFormData('mjenjac', e.target.value)}
                      options={getOptions('mjenjac')}
                    />
                  </div>

                  {/* Numeric Fields */}
                  <div className="grid md:grid-cols-2 gap-4">
                    <Input
                      label="Snaga (Power - HP)"
                      type="number"
                      value={formData.snaga}
                      onChange={(e) => updateFormData('snaga', e.target.value)}
                      placeholder="e.g., 150"
                      hint={schema.numeric_features.find(f => f.name === 'snaga')?.mean 
                        ? `Average: ${schema.numeric_features.find(f => f.name === 'snaga')?.mean?.toFixed(0)} HP`
                        : undefined}
                    />
                    <Input
                      label="Kilometraža (Mileage)"
                      type="number"
                      value={formData.kilometraza}
                      onChange={(e) => updateFormData('kilometraza', e.target.value)}
                      placeholder="e.g., 150000"
                      hint="In kilometers"
                    />
                  </div>

                  <div className="grid md:grid-cols-2 gap-4">
                    <Input
                      label="Kubikaža (Engine Size cc)"
                      type="number"
                      value={formData.kubikaza}
                      onChange={(e) => updateFormData('kubikaza', e.target.value)}
                      placeholder="e.g., 1998"
                    />
                    <Input
                      label="Godina (Year)"
                      type="number"
                      value={formData.god}
                      onChange={(e) => updateFormData('god', e.target.value)}
                      placeholder="e.g., 2020"
                      hint={`Range: ${schema.numeric_features.find(f => f.name === 'god')?.min} - ${schema.numeric_features.find(f => f.name === 'god')?.max}`}
                    />
                  </div>

                  {/* Model Selection */}
                  <div className="border-t border-dark-500 pt-6">
                    <div className="grid md:grid-cols-2 gap-4">
                      <Select
                        label="Task Type"
                        value={formData.task}
                        onChange={(e) => updateFormData('task', e.target.value)}
                        options={[
                          { value: 'regression', label: 'Price Regression (exact price)' },
                          { value: 'classification', label: 'Price Segment (budget/mid/premium)' },
                        ]}
                      />
                      <Select
                        label="Model to Use"
                        value={formData.experiment_id}
                        onChange={(e) => updateFormData('experiment_id', e.target.value)}
                        options={[
                          { value: '', label: 'Latest Best Model' },
                          ...filteredExperiments.map(exp => ({
                            value: exp.id.toString(),
                            label: `${exp.model_name} (${exp.task === 'regression' 
                              ? `R²: ${exp.metrics?.test?.r2?.toFixed(3) || '-'}` 
                              : `Acc: ${((exp.metrics?.test?.accuracy || 0) * 100).toFixed(1)}%`})`,
                          }))
                        ]}
                      />
                    </div>
                  </div>

                  {/* Submit */}
                  <Button 
                    type="submit" 
                    size="lg" 
                    className="w-full"
                    loading={predicting}
                    icon={<TrendingUp className="w-5 h-5" />}
                  >
                    {predicting ? 'Predicting...' : 'Predict Price'}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </motion.div>

          {/* Result Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card className="sticky top-24" glow={!!result}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-neon-blue" />
                  Prediction Result
                </CardTitle>
              </CardHeader>
              
              <CardContent>
                <AnimatePresence mode="wait">
                  {error && (
                    <motion.div
                      key="error"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg"
                    >
                      <div className="flex items-center gap-2 text-red-400">
                        <AlertCircle className="w-5 h-5" />
                        <p>{error}</p>
                      </div>
                    </motion.div>
                  )}

                  {!result && !error && (
                    <motion.div
                      key="empty"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="text-center py-8 text-gray-500"
                    >
                      <Calculator className="w-12 h-12 mx-auto mb-3 opacity-50" />
                      <p>Fill in the form and click predict to see results</p>
                    </motion.div>
                  )}

                  {result && (
                    <motion.div
                      key="result"
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="space-y-6"
                    >
                      {/* Main Prediction */}
                      <div className="text-center p-6 bg-gradient-to-br from-neon-green/10 to-neon-blue/10 rounded-xl border border-neon-green/30">
                        <p className="text-sm text-gray-400 mb-2">Estimated Price</p>
                        {formData.task === 'regression' ? (
                          <p className="text-4xl font-bold text-neon-green neon-text">
                            {formatCurrency(result.prediction)}
                          </p>
                        ) : (
                          <div>
                            <Badge 
                              variant={result.prediction_label === 'premium' ? 'success' : result.prediction_label === 'mid' ? 'info' : 'warning'}
                              className="text-lg px-4 py-2"
                            >
                              {result.prediction_label?.toUpperCase()}
                            </Badge>
                            {result.confidence && (
                              <p className="text-sm text-gray-400 mt-2">
                                Confidence: {(result.confidence * 100).toFixed(1)}%
                              </p>
                            )}
                          </div>
                        )}
                      </div>

                      {/* Model Info */}
                      <div className="p-4 bg-dark-700 rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <CheckCircle2 className="w-4 h-4 text-neon-green" />
                          <p className="text-sm font-medium text-white">Model Used</p>
                        </div>
                        <p className="text-gray-400">{result.model_name}</p>
                      </div>

                      {/* Feature Influence - sums to 100% */}
                      {result.top_features && Object.keys(result.top_features).length > 0 && (
                        <div>
                          <p className="text-sm font-medium text-white mb-3 flex items-center gap-2">
                            <Info className="w-4 h-4 text-neon-blue" />
                            Uticaj Parametara na Cijenu
                          </p>
                          <div className="space-y-3">
                            {Object.entries(result.top_features).slice(0, 8).map(([feature, data]: [string, any]) => {
                              const pct = data.percentage || 0
                              
                              return (
                                <div key={feature} className="p-3 bg-dark-700 rounded-lg">
                                  <div className="flex justify-between items-center mb-2">
                                    <span className="text-sm font-medium text-white">
                                      {data.display_name || feature}
                                    </span>
                                    <span className="text-sm text-neon-green font-mono">
                                      {data.user_value}
                                    </span>
                                  </div>
                                  <div className="h-2 bg-dark-600 rounded-full overflow-hidden">
                                    <div 
                                      className="h-full bg-gradient-to-r from-neon-green to-neon-blue transition-all duration-500"
                                      style={{ width: `${pct}%` }}
                                    />
                                  </div>
                                  <span className="text-xs text-gray-500">
                                    {pct.toFixed(1)}%
                                  </span>
                                </div>
                              )
                            })}
                          </div>
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
