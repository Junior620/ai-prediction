'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api';
import { formatPrice, formatPercentage, getSentimentLabel, getSentimentColor } from '@/lib/utils';
import type { PredictionResponse } from '@/types/api';
import { 
  TrendingUp, TrendingDown, DollarSign, Brain, RefreshCw, AlertCircle, 
  Activity, Target, Shield, Zap, TrendingUpIcon, MinusCircle, ArrowUpCircle, ArrowDownCircle
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, Area, AreaChart, BarChart, Bar, RadarChart, 
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar 
} from 'recharts';

// Types pour les nouvelles features
interface TradingSignal {
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  trend: 'bullish' | 'bearish' | 'neutral';
  recommendation: string;
}

interface InfluencingFactor {
  name: string;
  impact: number;
  icon: string;
}

interface Scenario {
  name: string;
  price: number;
  probability: number;
}

export default function PremiumDashboard() {
  const [data, setData] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [horizons, setHorizons] = useState<number[]>([1, 7, 30]);
  const [includeSentiment, setIncludeSentiment] = useState(true);

  // Données historiques simulées (à remplacer par vraies données)
  const historicalData = [
    { date: '01 Mai', price: 4200 },
    { date: '02 Mai', price: 4250 },
    { date: '03 Mai', price: 4180 },
    { date: '04 Mai', price: 4300 },
    { date: '05 Mai', price: 4350 },
    { date: '06 Mai', price: 4400 },
    { date: '07 Mai', price: 4450 },
    { date: '08 Mai', price: 4500 },
    { date: '09 Mai', price: 4480 },
    { date: '10 Mai', price: 4520 },
    { date: '11 Mai', price: 4532 },
  ];

  const fetchPredictions = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.getPredictions({
        market: 'ICE_NY',
        horizons,
        include_sentiment: includeSentiment,
      });
      setData(response);
    } catch (err: any) {
      setError(err.response?.data?.message || 'Erreur lors du chargement des données');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPredictions();
  }, []);

  const currentPrice = 4532.00;

  // 🔥 FEATURE 1: Calcul du signal trading
  const calculateTradingSignal = (): TradingSignal => {
    if (!data || data.predictions.length === 0) {
      return { signal: 'HOLD', confidence: 0, trend: 'neutral', recommendation: 'En attente de données' };
    }

    const pred1d = data.predictions.find(p => p.horizon === 1);
    const pred7d = data.predictions.find(p => p.horizon === 7);
    
    if (!pred1d) {
      return { signal: 'HOLD', confidence: 0, trend: 'neutral', recommendation: 'Données insuffisantes' };
    }

    const change1d = ((pred1d.price - currentPrice) / currentPrice) * 100;
    const change7d = pred7d ? ((pred7d.price - currentPrice) / currentPrice) * 100 : 0;
    
    // Calcul de la confiance basé sur l'intervalle
    const intervalWidth = pred1d.confidence_interval[1] - pred1d.confidence_interval[0];
    const confidence = Math.max(0, Math.min(100, 100 - (intervalWidth / currentPrice) * 100));

    if (change1d > 1.5 && change7d > 2) {
      return {
        signal: 'BUY',
        confidence: Math.round(confidence),
        trend: 'bullish',
        recommendation: 'Tendance haussière confirmée. Position longue recommandée.'
      };
    } else if (change1d < -1.5 && change7d < -2) {
      return {
        signal: 'SELL',
        confidence: Math.round(confidence),
        trend: 'bearish',
        recommendation: 'Tendance baissière détectée. Réduction de position conseillée.'
      };
    } else {
      return {
        signal: 'HOLD',
        confidence: Math.round(confidence),
        trend: 'neutral',
        recommendation: 'Marché incertain. Maintenir les positions actuelles.'
      };
    }
  };

  // 🔥 FEATURE 2: Facteurs influents
  const getInfluencingFactors = (): InfluencingFactor[] => {
    return [
      { name: 'Stocks ICE faibles', impact: 12, icon: '📦' },
      { name: 'Baisse pluie Ghana', impact: 8, icon: '🌧️' },
      { name: 'USD fort', impact: -4, icon: '💵' },
      { name: 'Fret maritime élevé', impact: 3, icon: '🚢' },
      { name: 'Demande chocolat', impact: 6, icon: '🍫' },
    ];
  };

  // 🔥 FEATURE 3: Scénarios futurs
  const getScenarios = (): Scenario[] => {
    if (!data || data.predictions.length === 0) return [];
    
    const pred30d = data.predictions.find(p => p.horizon === 30);
    if (!pred30d) return [];

    return [
      { name: 'Optimiste', price: pred30d.confidence_interval[1], probability: 20 },
      { name: 'Neutre', price: pred30d.price, probability: 60 },
      { name: 'Pessimiste', price: pred30d.confidence_interval[0], probability: 20 },
    ];
  };

  const tradingSignal = calculateTradingSignal();
  const influencingFactors = getInfluencingFactors();
  const scenarios = getScenarios();

  // Préparer données pour graphique historique + forecast
  const combinedChartData = [
    ...historicalData.map(d => ({ ...d, type: 'historical' })),
    ...data?.predictions.map((pred, idx) => ({
      date: `+${pred.horizon}j`,
      price: pred.price,
      lower: pred.confidence_interval[0],
      upper: pred.confidence_interval[1],
      type: 'forecast'
    })) || []
  ];

  const signalColors = {
    BUY: { bg: 'bg-green-500', text: 'text-green-600', border: 'border-green-500', glow: 'shadow-green-500/50' },
    SELL: { bg: 'bg-red-500', text: 'text-red-600', border: 'border-red-500', glow: 'shadow-red-500/50' },
    HOLD: { bg: 'bg-yellow-500', text: 'text-yellow-600', border: 'border-yellow-500', glow: 'shadow-yellow-500/50' },
  };

  const signalColor = signalColors[tradingSignal.signal];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Premium Header with Gradient */}
      <header className="bg-gradient-to-r from-purple-900/90 to-indigo-900/90 backdrop-blur-lg shadow-2xl border-b border-purple-500/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="bg-gradient-to-br from-amber-500 to-orange-600 p-3 rounded-xl shadow-lg">
                <Activity className="h-8 w-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
                  Cocoa Intelligence Platform
                </h1>
                <p className="text-sm text-purple-200">
                  ICE New York • Powered by AI • Real-time Analytics
                </p>
              </div>
            </div>
            <button
              onClick={fetchPredictions}
              disabled={loading}
              className="flex items-center space-x-2 bg-gradient-to-r from-amber-500 to-orange-600 hover:from-amber-600 hover:to-orange-700 text-white px-6 py-3 rounded-xl transition-all shadow-lg hover:shadow-xl disabled:opacity-50"
            >
              <RefreshCw className={`h-5 w-5 ${loading ? 'animate-spin' : ''}`} />
              <span className="font-semibold">Actualiser</span>
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="bg-red-900/50 border-l-4 border-red-500 p-4 mb-6 rounded-lg backdrop-blur">
            <div className="flex items-center">
              <AlertCircle className="h-5 w-5 text-red-400 mr-2" />
              <p className="text-red-200">{error}</p>
            </div>
          </div>
        )}

        {loading && !data && (
          <div className="flex items-center justify-center h-64">
            <div className="relative">
              <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-amber-500"></div>
              <Activity className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 h-8 w-8 text-amber-500" />
            </div>
          </div>
        )}

        {data && (
          <div className="space-y-6">
            {/* 🔥 SIGNAL TRADING IA - FEATURE PREMIUM #1 */}
            <div className={`bg-gradient-to-r from-slate-800/90 to-slate-900/90 backdrop-blur-lg rounded-2xl shadow-2xl p-8 border-l-8 ${signalColor.border} ${signalColor.glow} shadow-xl`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-6">
                  <div className={`${signalColor.bg} p-6 rounded-2xl shadow-lg`}>
                    {tradingSignal.signal === 'BUY' && <ArrowUpCircle className="h-12 w-12 text-white" />}
                    {tradingSignal.signal === 'SELL' && <ArrowDownCircle className="h-12 w-12 text-white" />}
                    {tradingSignal.signal === 'HOLD' && <MinusCircle className="h-12 w-12 text-white" />}
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-400 uppercase tracking-wider">Signal IA Trading</p>
                    <h2 className={`text-5xl font-bold ${signalColor.text} mt-1`}>
                      {tradingSignal.signal}
                    </h2>
                    <p className="text-gray-300 mt-2">{tradingSignal.recommendation}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-400 mb-2">Confiance du modèle</p>
                  <div className="flex items-center space-x-3">
                    <div className="w-32 h-3 bg-gray-700 rounded-full overflow-hidden">
                      <div 
                        className={`h-full ${signalColor.bg} transition-all duration-1000`}
                        style={{ width: `${tradingSignal.confidence}%` }}
                      ></div>
                    </div>
                    <span className="text-3xl font-bold text-white">{tradingSignal.confidence}%</span>
                  </div>
                  <p className="text-xs text-gray-400 mt-2">
                    Tendance: <span className="font-semibold text-white capitalize">{tradingSignal.trend}</span>
                  </p>
                </div>
              </div>
            </div>

            {/* Prix actuel + Métriques clés */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="bg-gradient-to-br from-slate-800/90 to-slate-900/90 backdrop-blur-lg rounded-xl shadow-xl p-6 border border-purple-500/20">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-medium text-gray-400">💰 Prix Actuel</p>
                  <DollarSign className="h-5 w-5 text-amber-500" />
                </div>
                <p className="text-3xl font-bold text-white">{formatPrice(currentPrice)}</p>
                <p className="text-xs text-gray-500 mt-1">ICE NY • 11 Mai 2026</p>
              </div>

              <div className="bg-gradient-to-br from-slate-800/90 to-slate-900/90 backdrop-blur-lg rounded-xl shadow-xl p-6 border border-purple-500/20">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-medium text-gray-400">💭 Sentiment</p>
                  <Brain className="h-5 w-5 text-purple-500" />
                </div>
                <p className={`text-2xl font-bold ${getSentimentColor(data.sentiment_score)}`}>
                  {getSentimentLabel(data.sentiment_score)}
                </p>
                {data.sentiment_score !== null && (
                  <p className="text-sm text-gray-400 mt-1">Score: {data.sentiment_score.toFixed(3)}</p>
                )}
              </div>

              <div className="bg-gradient-to-br from-slate-800/90 to-slate-900/90 backdrop-blur-lg rounded-xl shadow-xl p-6 border border-purple-500/20">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-medium text-gray-400">🎯 Volatilité</p>
                  <Activity className="h-5 w-5 text-red-500" />
                </div>
                <p className="text-2xl font-bold text-red-400">ÉLEVÉE</p>
                <p className="text-sm text-gray-400 mt-1">Indice: 78/100</p>
              </div>

              <div className="bg-gradient-to-br from-slate-800/90 to-slate-900/90 backdrop-blur-lg rounded-xl shadow-xl p-6 border border-purple-500/20">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-medium text-gray-400">🤖 Modèle</p>
                  <Shield className="h-5 w-5 text-blue-500" />
                </div>
                <p className="text-xl font-bold text-white">
                  {data.model_version.split('_')[1] || 'v1'}
                </p>
                <p className="text-xs text-gray-500 mt-1">Hybride Amélioré</p>
              </div>
            </div>

            {/* 🔥 FACTEURS INFLUENTS - FEATURE PREMIUM #2 */}
            <div className="bg-gradient-to-br from-slate-800/90 to-slate-900/90 backdrop-blur-lg rounded-2xl shadow-2xl p-6 border border-purple-500/20">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                <Zap className="h-6 w-6 text-yellow-500 mr-2" />
                Facteurs Influents Aujourd'hui
              </h2>
              <div className="space-y-4">
                {influencingFactors.map((factor, idx) => (
                  <div key={idx} className="flex items-center justify-between p-4 bg-slate-700/50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <span className="text-2xl">{factor.icon}</span>
                      <span className="text-white font-medium">{factor.name}</span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className="w-32 h-2 bg-gray-600 rounded-full overflow-hidden">
                        <div 
                          className={`h-full ${factor.impact > 0 ? 'bg-green-500' : 'bg-red-500'}`}
                          style={{ width: `${Math.abs(factor.impact) * 5}%` }}
                        ></div>
                      </div>
                      <span className={`text-lg font-bold ${factor.impact > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {factor.impact > 0 ? '+' : ''}{factor.impact}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 🔥 GRAPHIQUE HISTORIQUE + FORECAST - FEATURE PREMIUM #3 */}
            <div className="bg-gradient-to-br from-slate-800/90 to-slate-900/90 backdrop-blur-lg rounded-2xl shadow-2xl p-6 border border-purple-500/20">
              <h2 className="text-2xl font-bold text-white mb-6">
                📊 Historique Réel + Prévisions IA
              </h2>
              <ResponsiveContainer width="100%" height={450}>
                <AreaChart data={combinedChartData}>
                  <defs>
                    <linearGradient id="colorHistorical" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="date" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" tickFormatter={(value) => `$${value}`} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1e293b', 
                      border: '1px solid #8b5cf6',
                      borderRadius: '8px'
                    }}
                    formatter={(value: number) => formatPrice(value)}
                  />
                  <Legend />
                  <Area 
                    type="monotone" 
                    dataKey="upper" 
                    stroke="#fbbf24" 
                    fill="url(#colorForecast)" 
                    fillOpacity={0.3}
                    name="IC Supérieur"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="lower" 
                    stroke="#fbbf24" 
                    fill="url(#colorForecast)" 
                    fillOpacity={0.3}
                    name="IC Inférieur"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="price" 
                    stroke="#f59e0b" 
                    strokeWidth={3}
                    dot={{ fill: '#f59e0b', r: 5 }}
                    name="Prix"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* 🔥 SCÉNARIOS FUTURS - FEATURE PREMIUM #4 */}
            <div className="bg-gradient-to-br from-slate-800/90 to-slate-900/90 backdrop-blur-lg rounded-2xl shadow-2xl p-6 border border-purple-500/20">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                <Target className="h-6 w-6 text-blue-500 mr-2" />
                Scénarios 30 Jours
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {scenarios.map((scenario, idx) => (
                  <div key={idx} className="bg-slate-700/50 rounded-xl p-6 text-center">
                    <p className="text-gray-400 text-sm mb-2">{scenario.name}</p>
                    <p className="text-3xl font-bold text-white mb-2">{formatPrice(scenario.price)}</p>
                    <div className="flex items-center justify-center space-x-2">
                      <div className="w-full h-2 bg-gray-600 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-blue-500"
                          style={{ width: `${scenario.probability}%` }}
                        ></div>
                      </div>
                      <span className="text-sm text-gray-400">{scenario.probability}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Prédictions détaillées */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {data.predictions.map((pred) => {
                const change = pred.price - currentPrice;
                const changePercent = (change / currentPrice) * 100;
                const isPositive = change > 0;

                return (
                  <div
                    key={pred.horizon}
                    className="bg-gradient-to-br from-slate-800/90 to-slate-900/90 backdrop-blur-lg rounded-xl shadow-xl p-6 border-l-4 border-purple-500"
                  >
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold text-white">
                        Dans {pred.horizon} jour{pred.horizon > 1 ? 's' : ''}
                      </h3>
                      {isPositive ? (
                        <TrendingUp className="h-6 w-6 text-green-400" />
                      ) : (
                        <TrendingDown className="h-6 w-6 text-red-400" />
                      )}
                    </div>
                    <p className="text-3xl font-bold text-white mb-2">
                      {formatPrice(pred.price)}
                    </p>
                    <p className={`text-lg font-semibold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                      {formatPercentage(changePercent)}
                    </p>
                    <div className="mt-4 pt-4 border-t border-gray-700">
                      <p className="text-xs text-gray-400">IC 95%</p>
                      <p className="text-sm font-medium text-gray-300">
                        {formatPrice(pred.confidence_interval[0])} - {formatPrice(pred.confidence_interval[1])}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </main>

      <footer className="bg-gradient-to-r from-purple-900/90 to-indigo-900/90 backdrop-blur-lg shadow-2xl mt-12 border-t border-purple-500/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-purple-200 text-sm">
            🍫 Cocoa Intelligence Platform | Powered by AI | © 2026 STE-SCPB / Afrexia
          </p>
        </div>
      </footer>
    </div>
  );
}
