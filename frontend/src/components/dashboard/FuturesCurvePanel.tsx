'use client';

import { useMemo } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { TrendingUp } from 'lucide-react';
import { formatPrice } from '@/lib/utils';
import type { FuturesCurveResponse } from '@/types/api';

interface FuturesCurvePanelProps {
  data: FuturesCurveResponse | null;
  loading?: boolean;
  accentClass?: string;
}

function pickPred(contract: FuturesCurveResponse['contracts'][0], horizon: number) {
  return contract.predictions.find((p) => p.horizon === horizon)?.price ?? null;
}

export function FuturesCurvePanel({
  data,
  loading = false,
  accentClass = 'text-amber-400',
}: FuturesCurvePanelProps) {
  const chartData = useMemo(() => {
    if (!data?.contracts?.length) return [];
    return data.contracts.map((c) => ({
      name: c.contract,
      actuel: c.price_usd,
      j1: pickPred(c, 1),
      j7: pickPred(c, 7),
      j30: pickPred(c, 30),
    }));
  }, [data]);

  if (loading && !data) {
    return <div className="glass-card h-80 shimmer" />;
  }

  if (!data?.contracts?.length) {
    return null;
  }

  const collected = data.collected_at
    ? new Date(data.collected_at).toLocaleString('fr-FR', {
        day: '2-digit',
        month: 'short',
        hour: '2-digit',
        minute: '2-digit',
      })
    : null;

  return (
    <div className="glass-card p-5 space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className={`text-sm font-semibold flex items-center gap-2 ${accentClass}`}>
            <TrendingUp className="w-4 h-4" />
            Courbe à terme cacao
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            Contrats ICE (Investing) · prévisions J+1 / J+7 / J+30
            {collected ? ` · MAJ ${collected}` : ''}
            {data.source ? ` · ${data.source}` : ''}
          </p>
        </div>
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis
              dataKey="name"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={['auto', 'auto']}
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              width={56}
              tickFormatter={(v) => `${Math.round(v)}`}
            />
            <Tooltip
              contentStyle={{
                background: 'rgba(15,23,42,0.95)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 12,
              }}
              labelStyle={{ color: '#94a3b8' }}
              formatter={(value: number, name: string) => [
                formatPrice(value),
                name === 'actuel' ? 'Actuel' : name === 'j1' ? 'J+1' : name === 'j7' ? 'J+7' : 'J+30',
              ]}
            />
            <Legend
              wrapperStyle={{ fontSize: 12, color: '#94a3b8' }}
              formatter={(value) =>
                value === 'actuel' ? 'Actuel' : value === 'j1' ? 'J+1' : value === 'j7' ? 'J+7' : 'J+30'
              }
            />
            <Line type="monotone" dataKey="actuel" stroke="#fbbf24" strokeWidth={2.5} dot={{ r: 3 }} />
            <Line type="monotone" dataKey="j1" stroke="#34d399" strokeWidth={1.75} strokeDasharray="4 3" dot={false} />
            <Line type="monotone" dataKey="j7" stroke="#60a5fa" strokeWidth={1.75} strokeDasharray="4 3" dot={false} />
            <Line type="monotone" dataKey="j30" stroke="#c084fc" strokeWidth={1.75} strokeDasharray="4 3" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-slate-500 border-b border-white/[0.06]">
              <th className="text-left py-2 font-medium">Contrat</th>
              <th className="text-right py-2 font-medium">Actuel</th>
              <th className="text-right py-2 font-medium">J+1</th>
              <th className="text-right py-2 font-medium">J+7</th>
              <th className="text-right py-2 font-medium">J+30</th>
            </tr>
          </thead>
          <tbody>
            {data.contracts.map((c) => {
              const j1 = pickPred(c, 1);
              const j7 = pickPred(c, 7);
              const j30 = pickPred(c, 30);
              return (
                <tr key={`${c.symbol}-${c.contract}`} className="border-b border-white/[0.04]">
                  <td className="py-2 text-slate-300">
                    <span className="font-medium text-white">{c.contract}</span>
                    <span className="text-slate-600 ml-1">{c.symbol}</span>
                  </td>
                  <td className="py-2 text-right text-amber-300 tabular-nums">{formatPrice(c.price_usd)}</td>
                  <td className="py-2 text-right text-emerald-300/90 tabular-nums">
                    {j1 != null ? formatPrice(j1) : '—'}
                  </td>
                  <td className="py-2 text-right text-sky-300/90 tabular-nums">
                    {j7 != null ? formatPrice(j7) : '—'}
                  </td>
                  <td className="py-2 text-right text-violet-300/90 tabular-nums">
                    {j30 != null ? formatPrice(j30) : '—'}
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
