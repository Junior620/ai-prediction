'use client';

import { useMemo } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { FlaskConical } from 'lucide-react';
import type { ModelComparisonResponse } from '@/types/api';

interface Props {
  data: ModelComparisonResponse | null;
  loading?: boolean;
  accentClass?: string;
}

const SHORT: Record<string, string> = {
  M1_Prophet_Close: 'M1',
  M2_XGB_OHLCV: 'M2',
  M3_XGB_OHLCV_OI: 'M3',
  M4_XGB_Full: 'M4',
};

export function ModelComparisonPanel({
  data,
  loading = false,
  accentClass = 'text-amber-400',
}: Props) {
  const chartData = useMemo(() => {
    if (!data?.metrics?.length) return [];
    const byModel: Record<string, { name: string; label: string; j1?: number; j7?: number; j30?: number }> = {};
    for (const m of data.metrics) {
      if (!byModel[m.model]) {
        byModel[m.model] = {
          name: SHORT[m.model] || m.model,
          label: m.label,
        };
      }
      if (m.horizon === 1) byModel[m.model].j1 = Number(m.mape.toFixed(2));
      if (m.horizon === 7) byModel[m.model].j7 = Number(m.mape.toFixed(2));
      if (m.horizon === 30) byModel[m.model].j30 = Number(m.mape.toFixed(2));
    }
    // Exclure M1 du graphique (MAPE abérant ~155%) pour garder l'échelle lisible
    return Object.values(byModel).filter((r) => r.name !== 'M1');
  }, [data]);

  if (loading && !data) {
    return <div className="glass-card h-64 shimmer" />;
  }

  if (!data?.metrics?.length) {
    return null;
  }

  const bestJ1 = data.metrics
    .filter((m) => m.horizon === 1)
    .reduce((a, b) => (a.mape <= b.mape ? a : b));

  return (
    <div className="glass-card p-5 space-y-4">
      <div>
        <h3 className={`text-sm font-semibold flex items-center gap-2 ${accentClass}`}>
          <FlaskConical className="w-4 h-4" />
          Étude comparative M1–M4
        </h3>
        <p className="text-xs text-slate-500 mt-1">
          Split {data.split_date || '—'} · train {data.n_train ?? '—'} / test {data.n_test ?? '—'}
          {data.generated_at
            ? ` · ${new Date(data.generated_at).toLocaleDateString('fr-FR')}`
            : ''}
        </p>
      </div>

      <div className="rounded-xl bg-emerald-500/10 border border-emerald-500/20 px-3 py-2 text-xs text-emerald-200">
        Meilleur J+1 : <strong>{bestJ1.label}</strong> — MAPE {bestJ1.mape.toFixed(2)}%
        {data.note ? ` · ${data.note}` : ''}
      </div>

      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis
              dataKey="name"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              width={40}
              tickFormatter={(v) => `${v}%`}
            />
            <Tooltip
              contentStyle={{
                background: 'rgba(15,23,42,0.95)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 12,
              }}
              formatter={(value: number, name: string) => [
                `${value.toFixed(2)}%`,
                name === 'j1' ? 'J+1' : name === 'j7' ? 'J+7' : 'J+30',
              ]}
              labelFormatter={(label) => {
                const row = chartData.find((c) => c.name === label);
                return row?.label || String(label);
              }}
            />
            <Legend
              wrapperStyle={{ fontSize: 12, color: '#94a3b8' }}
              formatter={(v) => (v === 'j1' ? 'J+1' : v === 'j7' ? 'J+7' : 'J+30')}
            />
            <Bar dataKey="j1" fill="#34d399" radius={[4, 4, 0, 0]} />
            <Bar dataKey="j7" fill="#60a5fa" radius={[4, 4, 0, 0]} />
            <Bar dataKey="j30" fill="#c084fc" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-slate-500 border-b border-white/[0.06]">
              <th className="text-left py-2 font-medium">Modèle</th>
              <th className="text-right py-2 font-medium">J+1 MAPE</th>
              <th className="text-right py-2 font-medium">J+7 MAPE</th>
              <th className="text-right py-2 font-medium">J+30 MAPE</th>
            </tr>
          </thead>
          <tbody>
            {['M1_Prophet_Close', 'M2_XGB_OHLCV', 'M3_XGB_OHLCV_OI', 'M4_XGB_Full'].map((key) => {
              const rows = data.metrics.filter((m) => m.model === key);
              if (!rows.length) return null;
              const label = rows[0].label;
              const get = (h: number) => rows.find((r) => r.horizon === h)?.mape;
              const j1 = get(1);
              const isBest = bestJ1.model === key;
              return (
                <tr
                  key={key}
                  className={`border-b border-white/[0.04] ${isBest ? 'bg-emerald-500/5' : ''}`}
                >
                  <td className="py-2 text-slate-200 font-medium">{label}</td>
                  <td className="py-2 text-right tabular-nums text-emerald-300/90">
                    {j1 != null ? `${j1.toFixed(2)}%` : '—'}
                  </td>
                  <td className="py-2 text-right tabular-nums text-sky-300/90">
                    {get(7) != null ? `${get(7)!.toFixed(2)}%` : '—'}
                  </td>
                  <td className="py-2 text-right tabular-nums text-violet-300/90">
                    {get(30) != null ? `${get(30)!.toFixed(2)}%` : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
