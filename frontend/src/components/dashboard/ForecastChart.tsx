'use client';

import {
  Area, ComposedChart, CartesianGrid, Line, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { formatPrice } from '@/lib/utils';
import type { MarketBriefContent, PredictionItem } from '@/types/api';
import { BarChart3 } from 'lucide-react';

interface ForecastChartProps {
  historical?: { date: string; price: number }[];
  predictions: PredictionItem[];
  currentPrice: number;
  chartStroke: string;
  chartGradient: string;
  accentClass: string;
  brief?: MarketBriefContent | null;
}

function ChartTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: { name: string; value: number; dataKey: string }[];
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass-card px-4 py-3 !border-white/10">
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      {payload.filter(p => p.value != null).map((p, i) => (
        <p key={i} className="text-sm font-semibold text-white">
          {p.name}: <span className="text-amber-400">{formatPrice(p.value)}</span>
        </p>
      ))}
    </div>
  );
}

export function ForecastChart({
  historical = [],
  predictions,
  currentPrice,
  chartStroke,
  chartGradient,
  accentClass,
  brief,
}: ForecastChartProps) {
  const histPoints = [...historical]
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
    .map(p => ({
      date: new Date(p.date).toLocaleDateString('fr-FR', { day: '2-digit', month: 'short' }),
      actual: p.price,
      forecast: null as number | null,
      bandLow: null as number | null,
      bandHigh: null as number | null,
    }));

  const lastHist = histPoints[histPoints.length - 1];
  const forecastPoints = predictions.map(p => ({
    date: `+${p.horizon}j`,
    actual: null as number | null,
    forecast: p.price,
    bandLow: p.confidence_interval[0],
    bandHigh: p.confidence_interval[1],
  }));

  const bridge = lastHist
    ? [{
        ...lastHist,
        forecast: lastHist.actual,
        bandLow: lastHist.actual,
        bandHigh: lastHist.actual,
      }]
    : [];

  const chartData = [...histPoints.slice(0, -1), ...bridge, ...forecastPoints];

  const support = brief?.key_levels?.support;
  const resistance = brief?.key_levels?.resistance;

  return (
    <div className="glass-card p-5">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <h2 className="text-sm font-bold text-white flex items-center gap-2">
          <BarChart3 className={`w-4 h-4 ${accentClass}`} />
          Historique & prévisions
        </h2>
        <div className="flex flex-wrap gap-3 text-[10px] text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-slate-300 rounded" /> Prix réel
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 rounded" style={{ background: chartStroke }} /> Prévision
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-2 rounded opacity-40" style={{ background: chartStroke }} /> IC 90%
          </span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={chartData} margin={{ top: 8, right: 8, bottom: 5, left: 5 }}>
          <defs>
            <linearGradient id={chartGradient} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={chartStroke} stopOpacity={0.35} />
              <stop offset="100%" stopColor={chartStroke} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(148,163,184,0.06)" strokeDasharray="4 4" />
          <XAxis dataKey="date" stroke="#475569" fontSize={11} tickLine={false} axisLine={false} />
          <YAxis stroke="#475569" fontSize={11} tickLine={false} axisLine={false} tickFormatter={v => `$${v}`} width={55} />
          <Tooltip content={<ChartTooltip />} />

          <Area
            type="monotone"
            dataKey="bandHigh"
            stroke="none"
            fill={`url(#${chartGradient})`}
            connectNulls
            name="IC haut"
            legendType="none"
          />
          <Area
            type="monotone"
            dataKey="bandLow"
            stroke="none"
            fill="#06091a"
            connectNulls
            name="IC bas"
            legendType="none"
          />

          <Line
            type="monotone"
            dataKey="actual"
            stroke="#cbd5e1"
            strokeWidth={2}
            dot={{ r: 2, fill: '#cbd5e1', strokeWidth: 0 }}
            connectNulls
            name="Prix réel"
          />
          <Line
            type="monotone"
            dataKey="forecast"
            stroke={chartStroke}
            strokeWidth={2.5}
            strokeDasharray="6 3"
            dot={{ r: 3, fill: chartStroke, strokeWidth: 0 }}
            connectNulls
            name="Prévision"
          />

          {currentPrice > 0 && (
            <ReferenceLine y={currentPrice} stroke="#64748b" strokeDasharray="4 4" label={{ value: 'Actuel', position: 'insideTopRight', fill: '#64748b', fontSize: 10 }} />
          )}
          {support != null && (
            <ReferenceLine y={support} stroke="#10b981" strokeDasharray="5 5" label={{ value: 'Support', position: 'insideBottomLeft', fill: '#10b981', fontSize: 10 }} />
          )}
          {resistance != null && (
            <ReferenceLine y={resistance} stroke="#f43f5e" strokeDasharray="5 5" label={{ value: 'Résistance', position: 'insideTopLeft', fill: '#f43f5e', fontSize: 10 }} />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
