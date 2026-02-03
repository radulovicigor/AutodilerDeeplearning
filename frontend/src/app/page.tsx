'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { Card, CardTitle, CardDescription } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { getHealth, getDatasetStats, getExperiments } from '@/lib/api'
import { formatCurrency, formatNumber } from '@/lib/utils'
import { 
  Car, 
  Brain, 
  BarChart3, 
  Calculator, 
  Cpu, 
  Database,
  TrendingUp,
  Zap,
  ArrowRight
} from 'lucide-react'

export default function HomePage() {
  const [health, setHealth] = useState<any>(null)
  const [stats, setStats] = useState<any>(null)
  const [experiments, setExperiments] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchData() {
      try {
        const [healthRes, statsRes, expRes] = await Promise.all([
          getHealth(),
          getDatasetStats(),
          getExperiments(),
        ])
        setHealth(healthRes.data)
        setStats(statsRes.data)
        setExperiments(expRes.data)
      } catch (error) {
        console.error('Failed to fetch data:', error)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  const features = [
    {
      icon: Calculator,
      title: 'Price Prediction',
      description: 'Predict car prices using trained ML models',
      href: '/predict',
      color: 'from-neon-green to-emerald-500',
    },
    {
      icon: BarChart3,
      title: 'Model Comparison',
      description: 'Compare performance across different models',
      href: '/compare',
      color: 'from-neon-blue to-blue-500',
    },
    {
      icon: Brain,
      title: 'Network Lab',
      description: 'Build and visualize neural networks',
      href: '/network-lab',
      color: 'from-neon-purple to-purple-500',
    },
  ]

  return (
    <div className="min-h-screen" suppressHydrationWarning>
      {/* Hero Section */}
      <section className="relative py-20 px-4 overflow-hidden" suppressHydrationWarning>
        {/* Background grid */}
        <div className="absolute inset-0 grid-bg opacity-50" />
        
        {/* Gradient orbs */}
        <div className="absolute top-20 left-1/4 w-96 h-96 bg-neon-green/20 rounded-full blur-[128px]" />
        <div className="absolute bottom-20 right-1/4 w-96 h-96 bg-neon-blue/20 rounded-full blur-[128px]" />

        <div className="max-w-7xl mx-auto relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center"
          >
            <Badge variant="info" className="mb-4">
              <Zap className="w-3 h-3 mr-1" />
              Pokrenuto Dubokim Učenjem
            </Badge>
            
            <h1 className="text-5xl md:text-6xl font-bold text-white mb-6">
              Auto Diler <span className="text-transparent bg-clip-text bg-gradient-to-r from-neon-green to-neon-blue">AI</span>
            </h1>
            
            <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-8">
              Napredna predikcija cijena automobila korištenjem mašinskog učenja. 
              Trenirajte modele, vizualizujte neuronske mreže i razumijete duboko učenje.
            </p>

            <div className="flex items-center justify-center gap-4">
              <Link href="/predict">
                <Button size="lg" icon={<Calculator className="w-5 h-5" />}>
                  Pokreni Predikciju
                </Button>
              </Link>
              <Link href="/network-lab">
                <Button variant="secondary" size="lg" icon={<Brain className="w-5 h-5" />}>
                  Network Lab
                </Button>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-12 px-4 border-y border-dark-500 bg-dark-800/50">
        <div className="max-w-7xl mx-auto">
          {loading ? (
            <div className="flex justify-center py-8">
              <Spinner size="lg" />
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="text-center"
              >
                <Database className="w-8 h-8 text-neon-green mx-auto mb-2" />
                <p className="text-3xl font-bold text-white">
                  {stats?.total_rows ? formatNumber(stats.total_rows, 0) : '-'}
                </p>
                <p className="text-sm text-gray-400">Dataset Records</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="text-center"
              >
                <TrendingUp className="w-8 h-8 text-neon-blue mx-auto mb-2" />
                <p className="text-3xl font-bold text-white">
                  {stats?.price_stats ? formatCurrency(stats.price_stats.mean) : '-'}
                </p>
                <p className="text-sm text-gray-400">Average Price</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="text-center"
              >
                <Brain className="w-8 h-8 text-neon-purple mx-auto mb-2" />
                <p className="text-3xl font-bold text-white">
                  {experiments.filter(e => e.status === 'completed').length}
                </p>
                <p className="text-sm text-gray-400">Trained Models</p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="text-center"
              >
                <Cpu className={`w-8 h-8 mx-auto mb-2 ${health?.cuda_available ? 'text-neon-green' : 'text-neon-orange'}`} />
                <p className={`text-3xl font-bold ${health?.cuda_available ? 'text-neon-green' : 'text-white'}`}>
                  {health?.cuda_available ? 'GPU' : 'CPU'}
                </p>
                <p className="text-sm text-gray-400">
                  {health?.cuda_available ? health?.cuda_device : 'CPU Mode'}
                </p>
              </motion.div>
            </div>
          )}
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white mb-4">
              Explore the Platform
            </h2>
            <p className="text-gray-400">
              Everything you need to understand and apply machine learning for price prediction
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            {features.map((feature, index) => {
              const Icon = feature.icon
              return (
                <motion.div
                  key={feature.href}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 * index }}
                >
                  <Link href={feature.href}>
                    <Card className="h-full group cursor-pointer">
                      <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                        <Icon className="w-6 h-6 text-dark-900" />
                      </div>
                      <CardTitle className="flex items-center gap-2">
                        {feature.title}
                        <ArrowRight className="w-4 h-4 opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all" />
                      </CardTitle>
                      <CardDescription>{feature.description}</CardDescription>
                    </Card>
                  </Link>
                </motion.div>
              )
            })}
          </div>
        </div>
      </section>

      {/* Recent Experiments */}
      {experiments.length > 0 && (
        <section className="py-12 px-4 border-t border-dark-500">
          <div className="max-w-7xl mx-auto">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-white">Recent Experiments</h2>
              <Link href="/compare">
                <Button variant="ghost" size="sm">
                  View All <ArrowRight className="w-4 h-4 ml-1" />
                </Button>
              </Link>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {experiments.slice(0, 3).map((exp) => (
                <Card key={exp.id} className="p-4">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <p className="font-medium text-white">{exp.model_name}</p>
                      <p className="text-sm text-gray-400">{exp.task}</p>
                    </div>
                    <Badge variant={exp.status === 'completed' ? 'success' : 'warning'}>
                      {exp.status}
                    </Badge>
                  </div>
                  
                  {exp.metrics?.test && (
                    <div className="mt-3 pt-3 border-t border-dark-600">
                      {exp.task === 'regression' ? (
                        <div className="grid grid-cols-3 gap-2 text-sm">
                          <div>
                            <p className="text-gray-500">MAE</p>
                            <p className="text-white font-mono">
                              {exp.metrics.test.mae?.toFixed(0) || '-'}
                            </p>
                          </div>
                          <div>
                            <p className="text-gray-500">RMSE</p>
                            <p className="text-white font-mono">
                              {exp.metrics.test.rmse?.toFixed(0) || '-'}
                            </p>
                          </div>
                          <div>
                            <p className="text-gray-500">R²</p>
                            <p className="text-neon-green font-mono">
                              {exp.metrics.test.r2?.toFixed(3) || '-'}
                            </p>
                          </div>
                        </div>
                      ) : (
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <p className="text-gray-500">Accuracy</p>
                            <p className="text-neon-green font-mono">
                              {((exp.metrics.test.accuracy || 0) * 100).toFixed(1)}%
                            </p>
                          </div>
                          <div>
                            <p className="text-gray-500">F1 Score</p>
                            <p className="text-neon-blue font-mono">
                              {((exp.metrics.test.f1_macro || 0) * 100).toFixed(1)}%
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </Card>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* Educational Section */}
      <section className="py-20 px-4 border-t border-dark-500 bg-dark-800/30">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            Learn Deep Learning
          </h2>
          <p className="text-gray-400 mb-8">
            Use the Network Lab to understand how neural networks work. 
            Build custom architectures, visualize the forward pass, 
            and see how neurons activate in real-time.
          </p>
          
          <div className="grid md:grid-cols-3 gap-6 text-left">
            <div className="p-4 bg-dark-700 rounded-lg border border-dark-500">
              <div className="w-8 h-8 rounded bg-neon-green/20 flex items-center justify-center mb-3">
                <span className="text-neon-green font-bold">1</span>
              </div>
              <h3 className="font-semibold text-white mb-2">Configure Architecture</h3>
              <p className="text-sm text-gray-400">
                Choose layers, neurons, activation functions, and optimizer settings
              </p>
            </div>
            
            <div className="p-4 bg-dark-700 rounded-lg border border-dark-500">
              <div className="w-8 h-8 rounded bg-neon-blue/20 flex items-center justify-center mb-3">
                <span className="text-neon-blue font-bold">2</span>
              </div>
              <h3 className="font-semibold text-white mb-2">Train & Visualize</h3>
              <p className="text-sm text-gray-400">
                Watch the network learn in real-time with live loss curves
              </p>
            </div>
            
            <div className="p-4 bg-dark-700 rounded-lg border border-dark-500">
              <div className="w-8 h-8 rounded bg-neon-purple/20 flex items-center justify-center mb-3">
                <span className="text-neon-purple font-bold">3</span>
              </div>
              <h3 className="font-semibold text-white mb-2">Explore Internals</h3>
              <p className="text-sm text-gray-400">
                Click neurons to see weights, activations, and feature importance
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
