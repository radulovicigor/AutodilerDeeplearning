'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import Image from 'next/image'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Select } from '@/components/ui/Select'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { getExperiments, getExperiment, compareModels, deleteExperiment, renameExperiment, ExperimentSummary, ExperimentDetail } from '@/lib/api'
import { formatCurrency, formatNumber, getStatusColor, cn } from '@/lib/utils'
import { 
  BarChart3, 
  TrendingUp, 
  Brain,
  Calendar,
  Eye,
  Trash2,
  RefreshCw,
  ArrowUpDown,
  CheckCircle2,
  XCircle,
  Pencil,
  Clock,
  X,
  Check
} from 'lucide-react'
import { Input } from '@/components/ui/Input'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  Cell,
} from 'recharts'

const COLORS = ['#00ff88', '#00d4ff', '#a855f7', '#ff6b9d', '#ff8c00']

export default function ComparePage() {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([])
  const [selectedExperiment, setSelectedExperiment] = useState<ExperimentDetail | null>(null)
  const [comparison, setComparison] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [detailLoading, setDetailLoading] = useState(false)
  const [task, setTask] = useState<'regression' | 'classification'>('regression')
  const [sortBy, setSortBy] = useState<string>('created_at')
  const [editingId, setEditingId] = useState<number | null>(null)
  const [editName, setEditName] = useState('')
  const [deleteConfirmId, setDeleteConfirmId] = useState<number | null>(null)
  
  // Two-model comparison mode
  const [compareMode, setCompareMode] = useState(false)
  const [model1, setModel1] = useState<ExperimentDetail | null>(null)
  const [model2, setModel2] = useState<ExperimentDetail | null>(null)
  const [loadingCompare, setLoadingCompare] = useState(false)

  const fetchData = async () => {
    setLoading(true)
    try {
      const [expRes, compRes] = await Promise.all([
        getExperiments(task, 'completed'),
        compareModels(task),
      ])
      setExperiments(expRes.data)
      setComparison(compRes.data)
    } catch (error) {
      console.error('Failed to fetch experiments:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [task])

  const loadExperimentDetail = async (id: number) => {
    setDetailLoading(true)
    try {
      const res = await getExperiment(id)
      setSelectedExperiment(res.data)
    } catch (error) {
      console.error('Failed to load experiment:', error)
    } finally {
      setDetailLoading(false)
    }
  }

  const handleDelete = async (id: number) => {
    try {
      await deleteExperiment(id)
      setDeleteConfirmId(null)
      if (selectedExperiment?.id === id) {
        setSelectedExperiment(null)
      }
      fetchData()
    } catch (error) {
      console.error('Failed to delete experiment:', error)
    }
  }

  const handleRename = async (id: number) => {
    if (!editName.trim()) return
    try {
      await renameExperiment(id, editName.trim())
      setEditingId(null)
      setEditName('')
      fetchData()
    } catch (error) {
      console.error('Failed to rename experiment:', error)
    }
  }

  const startEditing = (id: number, currentName: string) => {
    setEditingId(id)
    setEditName(currentName)
  }

  const formatDateTime = (dateStr: string) => {
    try {
      const date = new Date(dateStr)
      // Use fixed format to avoid hydration mismatch
      const day = String(date.getDate()).padStart(2, '0')
      const month = String(date.getMonth() + 1).padStart(2, '0')
      const year = String(date.getFullYear()).slice(-2)
      const hours = String(date.getHours()).padStart(2, '0')
      const minutes = String(date.getMinutes()).padStart(2, '0')
      return {
        date: `${day}.${month}.${year}`,
        time: `${hours}:${minutes}`
      }
    } catch {
      return { date: '-', time: '-' }
    }
  }

  const loadModelForComparison = async (id: number, slot: 1 | 2) => {
    setLoadingCompare(true)
    try {
      const res = await getExperiment(id)
      if (slot === 1) {
        setModel1(res.data)
      } else {
        setModel2(res.data)
      }
    } catch (error) {
      console.error('Failed to load model for comparison:', error)
    } finally {
      setLoadingCompare(false)
    }
  }

  const clearComparison = () => {
    setModel1(null)
    setModel2(null)
    setCompareMode(false)
  }

  const sortedComparison = [...comparison].sort((a, b) => {
    if (sortBy === 'created_at') {
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    }
    if (task === 'regression') {
      if (sortBy === 'r2') return (b.r2 || 0) - (a.r2 || 0)
      if (sortBy === 'mae') return (a.mae || Infinity) - (b.mae || Infinity)
      if (sortBy === 'rmse') return (a.rmse || Infinity) - (b.rmse || Infinity)
    } else {
      if (sortBy === 'accuracy') return (b.accuracy || 0) - (a.accuracy || 0)
      if (sortBy === 'f1_macro') return (b.f1_macro || 0) - (a.f1_macro || 0)
    }
    return 0
  })

  // Prepare chart data
  const chartData = sortedComparison.map((exp, index) => ({
    name: exp.model_name.split(' ')[0],
    fullName: exp.model_name,
    ...(task === 'regression' 
      ? { r2: exp.r2, mae: exp.mae, rmse: exp.rmse }
      : { accuracy: exp.accuracy, f1: exp.f1_macro }
    ),
    color: COLORS[index % COLORS.length],
  }))

  return (
    <div className="min-h-screen py-8 px-4" suppressHydrationWarning>
      <div className="max-w-7xl mx-auto" suppressHydrationWarning>
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8"
        >
          <div>
            <Badge variant="info" className="mb-2">
              <BarChart3 className="w-3 h-3 mr-1" />
              Model Comparison
            </Badge>
            <h1 className="text-3xl font-bold text-white">Compare Models</h1>
            <p className="text-gray-400">Analyze and compare performance across trained models</p>
          </div>

          <div className="flex items-center gap-3">
            <Select
              value={task}
              onChange={(e) => setTask(e.target.value as 'regression' | 'classification')}
              options={[
                { value: 'regression', label: 'Regression' },
                { value: 'classification', label: 'Classification' },
              ]}
            />
            <Button
              variant={compareMode ? 'primary' : 'secondary'}
              onClick={() => {
                setCompareMode(!compareMode)
                if (compareMode) clearComparison()
              }}
            >
              {compareMode ? 'Exit Compare' : '⚔️ Compare 2'}
            </Button>
            <Button
              variant="secondary"
              icon={<RefreshCw className="w-4 h-4" />}
              onClick={fetchData}
              loading={loading}
            >
              Refresh
            </Button>
          </div>
        </motion.div>

        {/* Two-Model Comparison Panel */}
        {compareMode && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6"
          >
            <Card className="border-2 border-neon-green/30">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  ⚔️ Side-by-Side Comparison
                </CardTitle>
                <CardDescription>
                  Select two models from the table below to compare them directly
                </CardDescription>
              </CardHeader>
              <CardContent>
                {/* Model Selection */}
                <div className="grid md:grid-cols-2 gap-6 mb-6">
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Model 1</label>
                    <Select
                      value={model1?.id?.toString() || ''}
                      onChange={(e) => e.target.value && loadModelForComparison(parseInt(e.target.value), 1)}
                      options={[
                        { value: '', label: 'Select model...' },
                        ...comparison.map(exp => ({
                          value: exp.id.toString(),
                          label: `${exp.model_name} (${task === 'regression' ? `R²: ${exp.r2?.toFixed(3)}` : `Acc: ${(exp.accuracy * 100).toFixed(1)}%`})`
                        }))
                      ]}
                    />
                  </div>
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Model 2</label>
                    <Select
                      value={model2?.id?.toString() || ''}
                      onChange={(e) => e.target.value && loadModelForComparison(parseInt(e.target.value), 2)}
                      options={[
                        { value: '', label: 'Select model...' },
                        ...comparison.map(exp => ({
                          value: exp.id.toString(),
                          label: `${exp.model_name} (${task === 'regression' ? `R²: ${exp.r2?.toFixed(3)}` : `Acc: ${(exp.accuracy * 100).toFixed(1)}%`})`
                        }))
                      ]}
                    />
                  </div>
                </div>

                {/* Comparison Results */}
                {model1 && model2 && (
                  <div className="space-y-6">
                    {/* Metrics Comparison */}
                    <div className="grid md:grid-cols-2 gap-4">
                      {/* Model 1 Card */}
                      <div className="p-4 bg-gradient-to-br from-neon-green/10 to-transparent rounded-lg border border-neon-green/20">
                        <h4 className="font-semibold text-neon-green mb-3">{model1.model_name}</h4>
                        <div className="space-y-2 text-sm">
                          {task === 'regression' ? (
                            <>
                              <div className="flex justify-between">
                                <span className="text-gray-400">R²</span>
                                <span className="font-mono text-white">{model1.metrics?.test?.r2?.toFixed(4)}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">MAE</span>
                                <span className="font-mono text-white">€{model1.metrics?.test?.mae?.toFixed(0)}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">RMSE</span>
                                <span className="font-mono text-white">€{model1.metrics?.test?.rmse?.toFixed(0)}</span>
                              </div>
                            </>
                          ) : (
                            <>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Accuracy</span>
                                <span className="font-mono text-white">{((model1.metrics?.test?.accuracy || 0) * 100).toFixed(2)}%</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">F1 Score</span>
                                <span className="font-mono text-white">{((model1.metrics?.test?.f1_macro || 0) * 100).toFixed(2)}%</span>
                              </div>
                            </>
                          )}
                          <div className="flex justify-between">
                            <span className="text-gray-400">Training Time</span>
                            <span className="font-mono text-white">{model1.metrics?.training_time?.toFixed(1)}s</span>
                          </div>
                        </div>
                      </div>

                      {/* Model 2 Card */}
                      <div className="p-4 bg-gradient-to-br from-neon-blue/10 to-transparent rounded-lg border border-neon-blue/20">
                        <h4 className="font-semibold text-neon-blue mb-3">{model2.model_name}</h4>
                        <div className="space-y-2 text-sm">
                          {task === 'regression' ? (
                            <>
                              <div className="flex justify-between">
                                <span className="text-gray-400">R²</span>
                                <span className="font-mono text-white">{model2.metrics?.test?.r2?.toFixed(4)}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">MAE</span>
                                <span className="font-mono text-white">€{model2.metrics?.test?.mae?.toFixed(0)}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">RMSE</span>
                                <span className="font-mono text-white">€{model2.metrics?.test?.rmse?.toFixed(0)}</span>
                              </div>
                            </>
                          ) : (
                            <>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Accuracy</span>
                                <span className="font-mono text-white">{((model2.metrics?.test?.accuracy || 0) * 100).toFixed(2)}%</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">F1 Score</span>
                                <span className="font-mono text-white">{((model2.metrics?.test?.f1_macro || 0) * 100).toFixed(2)}%</span>
                              </div>
                            </>
                          )}
                          <div className="flex justify-between">
                            <span className="text-gray-400">Training Time</span>
                            <span className="font-mono text-white">{model2.metrics?.training_time?.toFixed(1)}s</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Difference Summary */}
                    <div className="p-4 bg-dark-700 rounded-lg">
                      <h4 className="text-sm font-semibold text-white mb-3">📊 Difference (Model 1 vs Model 2)</h4>
                      <div className="grid grid-cols-3 gap-4 text-center">
                        {task === 'regression' ? (
                          <>
                            <div>
                              <p className="text-xs text-gray-500">R² Diff</p>
                              <p className={cn(
                                "text-lg font-mono",
                                (model1.metrics?.test?.r2 || 0) > (model2.metrics?.test?.r2 || 0) 
                                  ? "text-neon-green" : "text-red-400"
                              )}>
                                {((model1.metrics?.test?.r2 || 0) - (model2.metrics?.test?.r2 || 0) > 0 ? '+' : '')}
                                {((model1.metrics?.test?.r2 || 0) - (model2.metrics?.test?.r2 || 0)).toFixed(4)}
                              </p>
                            </div>
                            <div>
                              <p className="text-xs text-gray-500">MAE Diff</p>
                              <p className={cn(
                                "text-lg font-mono",
                                (model1.metrics?.test?.mae || 0) < (model2.metrics?.test?.mae || 0) 
                                  ? "text-neon-green" : "text-red-400"
                              )}>
                                €{((model1.metrics?.test?.mae || 0) - (model2.metrics?.test?.mae || 0)).toFixed(0)}
                              </p>
                            </div>
                            <div>
                              <p className="text-xs text-gray-500">RMSE Diff</p>
                              <p className={cn(
                                "text-lg font-mono",
                                (model1.metrics?.test?.rmse || 0) < (model2.metrics?.test?.rmse || 0) 
                                  ? "text-neon-green" : "text-red-400"
                              )}>
                                €{((model1.metrics?.test?.rmse || 0) - (model2.metrics?.test?.rmse || 0)).toFixed(0)}
                              </p>
                            </div>
                          </>
                        ) : (
                          <>
                            <div>
                              <p className="text-xs text-gray-500">Accuracy Diff</p>
                              <p className={cn(
                                "text-lg font-mono",
                                (model1.metrics?.test?.accuracy || 0) > (model2.metrics?.test?.accuracy || 0) 
                                  ? "text-neon-green" : "text-red-400"
                              )}>
                                {(((model1.metrics?.test?.accuracy || 0) - (model2.metrics?.test?.accuracy || 0)) * 100).toFixed(2)}%
                              </p>
                            </div>
                            <div>
                              <p className="text-xs text-gray-500">F1 Diff</p>
                              <p className={cn(
                                "text-lg font-mono",
                                (model1.metrics?.test?.f1_macro || 0) > (model2.metrics?.test?.f1_macro || 0) 
                                  ? "text-neon-green" : "text-red-400"
                              )}>
                                {(((model1.metrics?.test?.f1_macro || 0) - (model2.metrics?.test?.f1_macro || 0)) * 100).toFixed(2)}%
                              </p>
                            </div>
                            <div>
                              <p className="text-xs text-gray-500">Winner</p>
                              <p className="text-lg">
                                {(model1.metrics?.test?.accuracy || 0) > (model2.metrics?.test?.accuracy || 0) 
                                  ? '🏆 Model 1' : '🏆 Model 2'}
                              </p>
                            </div>
                          </>
                        )}
                      </div>
                    </div>

                    {/* Config Comparison */}
                    <div className="grid md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="text-sm font-semibold text-white mb-2">Model 1 Config</h4>
                        <pre className="p-3 bg-dark-700 rounded-lg font-mono text-xs text-gray-300 max-h-32 overflow-auto">
                          {JSON.stringify(model1.config, null, 2)}
                        </pre>
                      </div>
                      <div>
                        <h4 className="text-sm font-semibold text-white mb-2">Model 2 Config</h4>
                        <pre className="p-3 bg-dark-700 rounded-lg font-mono text-xs text-gray-300 max-h-32 overflow-auto">
                          {JSON.stringify(model2.config, null, 2)}
                        </pre>
                      </div>
                    </div>

                    {/* Visualizations Side by Side */}
                    {(model1.plots || model2.plots) && (
                      <div>
                        <h4 className="text-sm font-semibold text-white mb-3">📈 Visual Comparison</h4>
                        <div className="grid md:grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <p className="text-xs text-neon-green text-center">Model 1</p>
                            {model1.plots && Object.entries(model1.plots).slice(0, 3).map(([name, url]) => (
                              <img
                                key={name}
                                src={`http://localhost:8000${url}`}
                                alt={name}
                                className="w-full rounded border border-dark-600"
                              />
                            ))}
                          </div>
                          <div className="space-y-2">
                            <p className="text-xs text-neon-blue text-center">Model 2</p>
                            {model2.plots && Object.entries(model2.plots).slice(0, 3).map(([name, url]) => (
                              <img
                                key={name}
                                src={`http://localhost:8000${url}`}
                                alt={name}
                                className="w-full rounded border border-dark-600"
                              />
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {loadingCompare && (
                  <div className="flex justify-center py-8">
                    <Spinner />
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}

        {loading ? (
          <div className="flex justify-center py-20">
            <Spinner size="lg" />
          </div>
        ) : experiments.length === 0 ? (
          <Card className="text-center py-12">
            <Brain className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">No Models Trained Yet</h3>
            <p className="text-gray-400 mb-4">
              Go to Network Lab to train your first model
            </p>
            <Button href="/network-lab" icon={<Brain className="w-4 h-4" />}>
              Open Network Lab
            </Button>
          </Card>
        ) : (
          <div className="grid lg:grid-cols-3 gap-6">
            {/* Comparison Table & Charts */}
            <div className="lg:col-span-2 space-y-6">
              {/* Summary Cards */}
              <div className="grid md:grid-cols-3 gap-4">
                <Card className="p-4">
                  <p className="text-sm text-gray-400">Total Models</p>
                  <p className="text-2xl font-bold text-white">{experiments.length}</p>
                </Card>
                <Card className="p-4">
                  <p className="text-sm text-gray-400">
                    Best {task === 'regression' ? 'R²' : 'Accuracy'}
                  </p>
                  <p className="text-2xl font-bold text-neon-green">
                    {task === 'regression'
                      ? Math.max(...comparison.map(c => c.r2 || 0)).toFixed(3)
                      : `${(Math.max(...comparison.map(c => c.accuracy || 0)) * 100).toFixed(1)}%`
                    }
                  </p>
                </Card>
                <Card className="p-4">
                  <p className="text-sm text-gray-400">
                    Best {task === 'regression' ? 'MAE' : 'F1 Score'}
                  </p>
                  <p className="text-2xl font-bold text-neon-blue">
                    {task === 'regression'
                      ? `€${Math.min(...comparison.filter(c => c.mae).map(c => c.mae)).toFixed(0)}`
                      : `${(Math.max(...comparison.map(c => c.f1_macro || 0)) * 100).toFixed(1)}%`
                    }
                  </p>
                </Card>
              </div>

              {/* Metrics Chart */}
              <Card>
                <CardHeader>
                  <CardTitle>Performance Comparison</CardTitle>
                  <CardDescription>
                    {task === 'regression' ? 'R² score by model (higher is better)' : 'Accuracy by model'}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d2d4a" />
                        <XAxis dataKey="name" stroke="#9ca3af" fontSize={12} />
                        <YAxis stroke="#9ca3af" fontSize={12} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#1a1a2e',
                            border: '1px solid #2d2d4a',
                            borderRadius: '8px',
                          }}
                          labelFormatter={(value, payload) => payload?.[0]?.payload?.fullName || value}
                        />
                        {task === 'regression' ? (
                          <Bar dataKey="r2" name="R² Score">
                            {chartData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                          </Bar>
                        ) : (
                          <Bar dataKey="accuracy" name="Accuracy">
                            {chartData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                          </Bar>
                        )}
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* Comparison Table */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between">
                  <div>
                    <CardTitle>Model Leaderboard</CardTitle>
                    <CardDescription>Click on a model to see details</CardDescription>
                  </div>
                  <Select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value)}
                    options={task === 'regression' ? [
                      { value: 'r2', label: 'Sort by R²' },
                      { value: 'mae', label: 'Sort by MAE' },
                      { value: 'rmse', label: 'Sort by RMSE' },
                      { value: 'created_at', label: 'Sort by Date' },
                    ] : [
                      { value: 'accuracy', label: 'Sort by Accuracy' },
                      { value: 'f1_macro', label: 'Sort by F1' },
                      { value: 'created_at', label: 'Sort by Date' },
                    ]}
                  />
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Model</th>
                          <th>Trained</th>
                          {task === 'regression' ? (
                            <>
                              <th>R²</th>
                              <th>MAE</th>
                              <th>RMSE</th>
                            </>
                          ) : (
                            <>
                              <th>Accuracy</th>
                              <th>F1 (Macro)</th>
                            </>
                          )}
                          <th>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {sortedComparison.map((exp, index) => {
                          const dt = formatDateTime(exp.created_at)
                          return (
                          <tr 
                            key={exp.id}
                            className={cn(
                              'cursor-pointer transition-colors',
                              selectedExperiment?.id === exp.id && 'bg-neon-green/10'
                            )}
                            onClick={() => loadExperimentDetail(exp.id)}
                          >
                            <td>
                              {index === 0 && sortBy !== 'created_at' ? (
                                <span className="text-neon-green">🏆</span>
                              ) : (
                                index + 1
                              )}
                            </td>
                            <td>
                              {editingId === exp.id ? (
                                <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                                  <Input
                                    value={editName}
                                    onChange={(e) => setEditName(e.target.value)}
                                    className="py-1 text-sm w-40"
                                    autoFocus
                                    onKeyDown={(e) => {
                                      if (e.key === 'Enter') handleRename(exp.id)
                                      if (e.key === 'Escape') setEditingId(null)
                                    }}
                                  />
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    icon={<Check className="w-4 h-4 text-neon-green" />}
                                    onClick={() => handleRename(exp.id)}
                                  />
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    icon={<X className="w-4 h-4 text-red-400" />}
                                    onClick={() => setEditingId(null)}
                                  />
                                </div>
                              ) : (
                                <div>
                                  <p className="font-medium text-white">{exp.model_name}</p>
                                  <p className="text-xs text-gray-500">{exp.model_type}</p>
                                </div>
                              )}
                            </td>
                            <td suppressHydrationWarning>
                              <div className="text-xs">
                                <p className="text-gray-400">{dt.date}</p>
                                <p className="text-gray-500 flex items-center gap-1">
                                  <Clock className="w-3 h-3" />
                                  {dt.time}
                                </p>
                              </div>
                            </td>
                            {task === 'regression' ? (
                              <>
                                <td className="font-mono text-neon-green">
                                  {exp.r2?.toFixed(3) || '-'}
                                </td>
                                <td className="font-mono">€{exp.mae?.toFixed(0) || '-'}</td>
                                <td className="font-mono">€{exp.rmse?.toFixed(0) || '-'}</td>
                              </>
                            ) : (
                              <>
                                <td className="font-mono text-neon-green">
                                  {((exp.accuracy || 0) * 100).toFixed(1)}%
                                </td>
                                <td className="font-mono text-neon-blue">
                                  {((exp.f1_macro || 0) * 100).toFixed(1)}%
                                </td>
                              </>
                            )}
                            <td>
                              <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  icon={<Eye className="w-4 h-4" />}
                                  onClick={() => loadExperimentDetail(exp.id)}
                                />
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  icon={<Pencil className="w-4 h-4 text-neon-blue" />}
                                  onClick={() => startEditing(exp.id, exp.model_name)}
                                />
                                {deleteConfirmId === exp.id ? (
                                  <>
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      icon={<Check className="w-4 h-4 text-red-400" />}
                                      onClick={() => handleDelete(exp.id)}
                                    />
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      icon={<X className="w-4 h-4" />}
                                      onClick={() => setDeleteConfirmId(null)}
                                    />
                                  </>
                                ) : (
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    icon={<Trash2 className="w-4 h-4 text-red-400" />}
                                    onClick={() => setDeleteConfirmId(exp.id)}
                                  />
                                )}
                              </div>
                            </td>
                          </tr>
                        )})}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Detail Panel */}
            <div className="space-y-6">
              <Card className="sticky top-24">
                <CardHeader>
                  <CardTitle>Model Details</CardTitle>
                </CardHeader>
                <CardContent>
                  {detailLoading ? (
                    <div className="flex justify-center py-8">
                      <Spinner />
                    </div>
                  ) : selectedExperiment ? (
                    <div className="space-y-4">
                      {/* Model Info */}
                      <div className="p-4 bg-dark-700 rounded-lg">
                        <h4 className="font-semibold text-white mb-2">
                          {selectedExperiment.model_name}
                        </h4>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <p className="text-gray-500">Type</p>
                            <p className="text-white">{selectedExperiment.model_type}</p>
                          </div>
                          <div>
                            <p className="text-gray-500">Task</p>
                            <p className="text-white capitalize">{selectedExperiment.task}</p>
                          </div>
                        </div>
                        {selectedExperiment.created_at && (
                          <div className="mt-3 pt-3 border-t border-dark-600 flex items-center gap-2 text-xs text-gray-400">
                            <Clock className="w-3 h-3" />
                            <span suppressHydrationWarning>
                              {formatDateTime(selectedExperiment.created_at).date} {formatDateTime(selectedExperiment.created_at).time}
                            </span>
                          </div>
                        )}
                      </div>

                      {/* Loss Curves (MLP only) */}
                      {selectedExperiment.loss_history && (
                        <div>
                          <h4 className="text-sm font-medium text-white mb-2">Learning Curves</h4>
                          <div className="h-48">
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart
                                data={selectedExperiment.loss_history.train_loss.map((loss, i) => ({
                                  epoch: i + 1,
                                  train: loss,
                                  val: selectedExperiment.loss_history!.val_loss[i],
                                }))}
                              >
                                <CartesianGrid strokeDasharray="3 3" stroke="#2d2d4a" />
                                <XAxis dataKey="epoch" stroke="#9ca3af" fontSize={10} />
                                <YAxis stroke="#9ca3af" fontSize={10} />
                                <Tooltip
                                  contentStyle={{
                                    backgroundColor: '#1a1a2e',
                                    border: '1px solid #2d2d4a',
                                    borderRadius: '8px',
                                    fontSize: '12px',
                                  }}
                                />
                                <Line
                                  type="monotone"
                                  dataKey="train"
                                  stroke="#00ff88"
                                  strokeWidth={2}
                                  dot={false}
                                  name="Train Loss"
                                />
                                <Line
                                  type="monotone"
                                  dataKey="val"
                                  stroke="#00d4ff"
                                  strokeWidth={2}
                                  dot={false}
                                  name="Val Loss"
                                />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                      )}

                      {/* Plots */}
                      {selectedExperiment.plots && Object.keys(selectedExperiment.plots).length > 0 && (
                        <div>
                          <h4 className="text-sm font-medium text-white mb-2">Visualizations</h4>
                          <div className="grid grid-cols-2 gap-2">
                            {Object.entries(selectedExperiment.plots).map(([name, url]) => (
                              <a
                                key={name}
                                href={`http://localhost:8000${url}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="block p-2 bg-dark-700 rounded hover:bg-dark-600 transition-colors"
                              >
                                <img
                                  src={`http://localhost:8000${url}`}
                                  alt={name}
                                  className="w-full h-auto rounded"
                                />
                                <p className="text-xs text-gray-400 mt-1 truncate">
                                  {name.replace(/_/g, ' ')}
                                </p>
                              </a>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Config */}
                      <div>
                        <h4 className="text-sm font-medium text-white mb-2">Configuration</h4>
                        <div className="p-3 bg-dark-700 rounded-lg font-mono text-xs text-gray-300 max-h-48 overflow-auto">
                          <pre>{JSON.stringify(selectedExperiment.config, null, 2)}</pre>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      <Eye className="w-12 h-12 mx-auto mb-3 opacity-50" />
                      <p>Select a model to view details</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
