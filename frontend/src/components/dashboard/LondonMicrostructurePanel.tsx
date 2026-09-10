'use client';

import { useMemo } from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  BarChart,
  Bar,
} from 'recharts';
import { Activity, Layers } from 'lucide-react';
import { formatPrice } from '@/lib/utils';
import type { LondonMarketResponse } from '@/types/api';

interface Props {
  data: LondonMarketResponse | null;
  loading?: boolean;
  accentClass?: string;
}

type MicroSignal = {
  label: string;
  detail: string;
  className: string;
};

function resolveMicroSignal(history: { price: number; volume: number | null; oi: number | null }[]): MicroSignal {
  if (history.length < 2) {
    return {
      label: 'Neutre',
      detail: 'Historique insuffisant',
      className: 'bg-slate-500/15 text-slate-300 border-slate-500/30',
    };
  }
  const a = history[history.length - 1];
  const b = history[history.length - 2];
  const priceUp = a.price > b.price;
  const priceDown = a.price < b.price;
  const volUp = a.volume != null && b.volume != null && a.volume > b.volume;
  const oiUp = a.oi != null && b.oi != null && a.oi > b.oi;
  const oiDown = a.oi != null && b.oi != null && a.oi < b.oi;

  if (priceUp && volUp && oiUp) {
    return {
      label: 'Accumulation',
      detail: 'Prix↑ Vol↑ OI↑ — nouvelles positions longues',
      className: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
    };
  }
  if (priceUp && volUp && oiDown) {
    return {
      label: 'Couverture / liquidation',
      detail: 'Prix↑ Vol↑ OI↓ — shorts couverts / liquidations',
      className: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
    };
  }
  if (priceDown && volUp && oiUp) {
    return {
      label: 'Distribution',
      detail: 'Prix↓ Vol↑ OI↑ — nouvelles positions courtes',
      className: 'bg-rose-500/15 text-rose-300 border-rose-500/30',
    };
  }
  if (priceDown && volUp && oiDown) {
    return {
      label: 'Liquidation longs',
      detail: 'Prix↓ Vol↑ OI↓ — longs sortis',
      className: 'bg-orange-500/15 text-orange-300 border-orange-500/30',
    };
  }
  return {
    label: 'Neutre',
    detail: 'Pas de signal Prix/Vol/OI convergent',
    className: 'bg-slate-500/15 text-slate-300 border-slate-500/30',
  };
}

export function LondonMicrostructurePanel({
  data,
  loading = false,
  accentClass = 'text-amber-400',
}: Props) {
  const history = useMemo(() => {
    if (!data?.history?.length) return [];
    return data.history.map((h) => ({
      date: h.date.slice(5),
      fullDate: h.date,
      price: h.price,
      volume: h.volume ?? null,
      oi: h.open_interest ?? null,
    }));
  }, [data]);

  const termChart = useMemo(() => {
    if (!data?.term_structure?.length) return [];
    return data.term_structure.map((t) => ({
      name: t.label.replace(' echeance', ''),
      close: t.close,
      oi: t.open_interest ?? null,
      volume: t.volume ?? null,
    }));
  }, [data]);

  const signal = useMemo(() => resolveMicroSignal(history), [history]);

  if (loading && !data) {
    return <div className="glass-card h-80 shimmer" />;
  }

  if (!data?.latest && !data?.term_structure?.length) {
    return null;
  }

  const latest = data.latest;
  const oiDelta =
    history.length >= 2 && history[history.length - 1].oi != null && history[history.length - 2].oi != null
      ? (history[history.length - 1].oi as number) - (history[history.length - 2].oi as number)
      : null;

  return (
    <div className="glass-card p-5 space-y-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className={`text-sm font-semibold flex items-center gap-2 ${accentClass}`}>
            <Activity className="w-4 h-4" />
            Microstructure Londres (Databento)
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            Volume + Open Interest + courbe d&apos;échéances ICE London · GBP/T
            {latest?.date ? ` · dernier ${latest.date}` : ''}
            {data.source ? ` · ${data.source}` : ''}
          </p>
        </div>
        <div
          className={`shrink-0 rounded-lg border px-2.5 py-1.5 text-right ${signal.className}`}
          title={signal.detail}
        >
          <div className="text-[10px] uppercase tracking-wide opacity-70">Signal</div>
          <div className="text-xs font-semibold">{signal.label}</div>
        </div>
      </div>

      <p className="text-[11px] text-slate-500 -mt-2">{signal.detail}</p>

      {latest && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] px-3 py-2.5">
            <div className="text-[10px] uppercase tracking-wide text-slate-500">Clôture</div>
            <div className="text-lg font-semibold text-white tabular-nums">
              {formatPrice(latest.price, 'GBP')}
            </div>
          </div>
          <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] px-3 py-2.5">
            <div className="text-[10px] uppercase tracking-wide text-slate-500">Volume</div>
            <div className="text-lg font-semibold text-slate-100 tabular-nums">
              {latest.volume != null ? Math.round(latest.volume).toLocaleString('fr-FR') : '—'}
            </div>
          </div>
          <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] px-3 py-2.5">
            <div className="text-[10px] uppercase tracking-wide text-slate-500">Open Interest</div>
            <div className="text-lg font-semibold text-slate-100 tabular-nums">
              {latest.open_interest != null
                ? Math.round(latest.open_interest).toLocaleString('fr-FR')
                : '—'}
            </div>
          </div>
          <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] px-3 py-2.5">
            <div className="text-[10px] uppercase tracking-wide text-slate-500">Δ OI (1j)</div>
            <div
              className={`text-lg font-semibold tabular-nums ${
                oiDelta == null
                  ? 'text-slate-500'
                  : oiDelta >= 0
                    ? 'text-emerald-300'
                    : 'text-rose-300'
              }`}
            >
              {oiDelta == null
                ? '—'
                : `${oiDelta >= 0 ? '+' : ''}${Math.round(oiDelta).toLocaleString('fr-FR')}`}
            </div>
          </div>
        </div>
      )}

      {history.some((h) => h.volume != null || h.oi != null) && (
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={history} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis
                dataKey="date"
                tick={{ fill: '#94a3b8', fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                minTickGap={24}
              />
              <YAxis
                yAxisId="left"
                tick={{ fill: '#94a3b8', fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                width={48}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                tick={{ fill: '#94a3b8', fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                width={48}
              />
              <Tooltip
                contentStyle={{
                  background: 'rgba(15,23,42,0.95)',
                  border: '1px solid rgba(255,255,255,0.08)',
                  borderRadius: 12,
                }}
                labelFormatter={(_, payload) => payload?.[0]?.payload?.fullDate ?? ''}
              />
              <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
              <Area
                yAxisId="left"
                type="monotone"
                dataKey="volume"
                name="Volume"
                fill="rgba(96,165,250,0.15)"
                stroke="#60a5fa"
                strokeWidth={1.5}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="oi"
                name="Open Interest"
                stroke="#fbbf24"
                strokeWidth={2}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {termChart.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-xs font-medium text-slate-400 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5" />
            Structure des échéances
            {data.term_date ? ` · ${data.term_date}` : ''}
          </h4>
          <div className="h-44 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={termChart} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
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
                  width={52}
                  tickFormatter={(v) => `${Math.round(v)}`}
                />
                <Tooltip
                  contentStyle={{
                    background: 'rgba(15,23,42,0.95)',
                    border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: 12,
                  }}
                  formatter={(value: number) => [formatPrice(value, 'GBP'), 'Clôture']}
                />
                <Bar dataKey="close" name="Clôture £" fill="#f59e0b" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-500 border-b border-white/[0.06]">
                  <th className="text-left py-2 font-medium">Échéance</th>
                  <th className="text-right py-2 font-medium">Clôture</th>
                  <th className="text-right py-2 font-medium">Volume</th>
                  <th className="text-right py-2 font-medium">OI</th>
                  <th className="text-right py-2 font-medium">Spread vs front</th>
                </tr>
              </thead>
              <tbody>
                {data.term_structure.map((t) => {
                  const front = data.term_structure[0]?.close;
                  const spread = front != null ? t.close - front : null;
                  return (
                    <tr key={t.contract_rank} className="border-b border-white/[0.04]">
                      <td className="py-2 text-slate-300">
                        <span className="font-medium text-white">{t.label}</span>
                        <span className="text-slate-600 ml-1">{t.symbol}</span>
                      </td>
                      <td className="py-2 text-right text-amber-300 tabular-nums">
                        {formatPrice(t.close, 'GBP')}
                      </td>
                      <td className="py-2 text-right text-slate-400 tabular-nums">
                        {t.volume != null ? Math.round(t.volume).toLocaleString('fr-FR') : '—'}
                      </td>
                      <td className="py-2 text-right text-slate-400 tabular-nums">
                        {t.open_interest != null
                          ? Math.round(t.open_interest).toLocaleString('fr-FR')
                          : '—'}
                      </td>
                      <td
                        className={`py-2 text-right tabular-nums ${
                          spread == null || spread === 0
                            ? 'text-slate-500'
                            : spread > 0
                              ? 'text-emerald-300'
                              : 'text-rose-300'
                        }`}
                      >
                        {spread == null
                          ? '—'
                          : `${spread >= 0 ? '+' : ''}${spread.toFixed(0)} £`}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
