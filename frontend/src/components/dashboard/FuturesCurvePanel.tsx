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
import { formatPrice, formatPriceGbp } from '@/lib/utils';
import { useUsdGbpRate } from '@/hooks/useUsdGbpRate';
import type { FuturesContractItem, FuturesCurveResponse } from '@/types/api';

interface FuturesCurvePanelProps {
  data: FuturesCurveResponse | null;
  loading?: boolean;
  accentClass?: string;
}

function pickPred(contract: FuturesContractItem, horizon: number) {
  return contract.predictions.find((p) => p.horizon === horizon)?.price ?? null;
}

function PriceCell({
  value,
  currency,
  usdGbp,
  className,
}: {
  value: number | null;
  currency: 'USD' | 'GBP';
  usdGbp: number | null;
  className?: string;
}) {
  if (value == null) {
    return <td className={`py-2 text-right tabular-nums ${className || ''}`}>—</td>;
  }
  if (currency === 'GBP') {
    return (
      <td className={`py-2 text-right tabular-nums ${className || ''}`}>
        {formatPrice(value, 'GBP')}
      </td>
    );
  }
  return (
    <td className={`py-2 text-right tabular-nums ${className || ''}`}>
      <div>{formatPrice(value)}</div>
      <div className="text-[10px] text-slate-500 font-normal mt-0.5">
        ≈ {formatPriceGbp(value, usdGbp)}
      </div>
    </td>
  );
}

function CurveBlock({
  title,
  subtitle,
  contracts,
  currency,
  usdGbp,
  accentClass,
}: {
  title: string;
  subtitle: string;
  contracts: FuturesContractItem[];
  currency: 'USD' | 'GBP';
  usdGbp: number | null;
  accentClass: string;
}) {
  const chartData = useMemo(
    () =>
      contracts.map((c) => ({
        name: c.contract,
        actuel: c.price_usd,
        j1: pickPred(c, 1),
        j7: pickPred(c, 7),
        j14: pickPred(c, 14),
        j30: pickPred(c, 30),
      })),
    [contracts],
  );

  const formatTip = (value: number) =>
    currency === 'GBP'
      ? formatPrice(value, 'GBP')
      : `${formatPrice(value)} · ≈ ${formatPriceGbp(value, usdGbp)}`;

  const horizonName = (name: string) =>
    ({ actuel: 'Actuel', j1: 'J+1', j7: 'J+7', j14: 'J+14', j30: 'J+30' } as Record<string, string>)[
      name
    ] || name;

  return (
    <div className="space-y-4">
      <div>
        <h3 className={`text-sm font-semibold flex items-center gap-2 ${accentClass}`}>
          <TrendingUp className="w-4 h-4" />
          {title}
        </h3>
        <p className="text-xs text-slate-500 mt-1">{subtitle}</p>
      </div>

      <div className="h-56 w-full">
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
              formatter={(value: number, name: string) => [formatTip(value), horizonName(name)]}
            />
            <Legend
              wrapperStyle={{ fontSize: 12, color: '#94a3b8' }}
              formatter={(value) => horizonName(String(value))}
            />
            <Line type="monotone" dataKey="actuel" stroke="#fbbf24" strokeWidth={2.5} dot={{ r: 3 }} />
            <Line type="monotone" dataKey="j1" stroke="#34d399" strokeWidth={1.75} strokeDasharray="4 3" dot={false} />
            <Line type="monotone" dataKey="j7" stroke="#60a5fa" strokeWidth={1.75} strokeDasharray="4 3" dot={false} />
            <Line type="monotone" dataKey="j14" stroke="#38bdf8" strokeWidth={1.75} strokeDasharray="4 3" dot={false} />
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
              <th className="text-right py-2 font-medium">J+14</th>
              <th className="text-right py-2 font-medium">J+30</th>
            </tr>
          </thead>
          <tbody>
            {contracts.map((c) => (
              <tr key={`${c.symbol}-${c.contract}`} className="border-b border-white/[0.04]">
                <td className="py-2 text-slate-300">
                  <span className="font-medium text-white">{c.contract}</span>
                  <span className="text-slate-600 ml-1">{c.symbol}</span>
                </td>
                <PriceCell value={c.price_usd} currency={currency} usdGbp={usdGbp} className="text-amber-300" />
                <PriceCell value={pickPred(c, 1)} currency={currency} usdGbp={usdGbp} className="text-emerald-300/90" />
                <PriceCell value={pickPred(c, 7)} currency={currency} usdGbp={usdGbp} className="text-sky-300/90" />
                <PriceCell value={pickPred(c, 14)} currency={currency} usdGbp={usdGbp} className="text-cyan-300/90" />
                <PriceCell value={pickPred(c, 30)} currency={currency} usdGbp={usdGbp} className="text-violet-300/90" />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function FuturesCurvePanel({
  data,
  loading = false,
  accentClass = 'text-amber-400',
}: FuturesCurvePanelProps) {
  const usdGbp = useUsdGbpRate();

  if (loading && !data) {
    return <div className="glass-card h-80 shimmer" />;
  }

  const named = data?.contracts?.length ? data.contracts : [];
  const london = data?.london_contracts?.length ? data.london_contracts : [];
  const namedIsGbp =
    (data?.currency || '').toUpperCase() === 'GBP' ||
    (data?.unit || '').includes('GBP') ||
    String(data?.source || '').includes('london_named') ||
    data?.source === 'databento_london';

  if (!named.length && !london.length) {
    return null;
  }

  const namedCollected = data?.collected_at
    ? new Date(data.collected_at).toLocaleString('fr-FR', {
        day: '2-digit',
        month: 'short',
        hour: '2-digit',
        minute: '2-digit',
      })
    : null;
  const londonCollected = data?.london_collected_at
    ? String(data.london_collected_at).slice(0, 10)
    : null;

  // Si seule courbe Londres (pas de snapshot Investing), un seul bloc
  if (!named.length || (namedIsGbp && !london.length)) {
    return (
      <div className="glass-card p-5">
        <CurveBlock
          title="Contrats à terme Londres"
          subtitle={`ICE London £/T · DEC26 / MAR27… · J+1 / J+7 / J+14 / J+30${namedCollected ? ` · MAJ ${namedCollected}` : ''}${data?.source ? ` · ${data.source}` : ''}`}
          contracts={named}
          currency="GBP"
          usdGbp={usdGbp}
          accentClass={accentClass}
        />
      </div>
    );
  }

  return (
    <div className="glass-card p-5 space-y-8">
      <CurveBlock
        title={namedIsGbp ? 'Contrats à terme Londres (mois nommés)' : 'Contrats à terme (mois nommés)'}
        subtitle={
          namedIsGbp
            ? `ICE London £/T · DEC26 / MAR27…${namedCollected ? ` · MAJ ${namedCollected}` : ''}${data?.source ? ` · ${data.source}` : ''}`
            : `ICE NY · Dec26 / Mar27… · USD${namedCollected ? ` · MAJ ${namedCollected}` : ''}${data?.source ? ` · ${data.source}` : ''}${usdGbp != null ? ` · FX ${usdGbp.toFixed(4)}` : ''}`
        }
        contracts={named}
        currency={namedIsGbp ? 'GBP' : 'USD'}
        usdGbp={usdGbp}
        accentClass={accentClass}
      />
      {london.length > 0 && (
        <div className="pt-4 border-t border-white/[0.06]">
          <CurveBlock
            title="Courbe Londres continue"
            subtitle={`ICE London £/T · C.v.0–3${londonCollected ? ` · ${londonCollected}` : ''}${data?.london_source ? ` · ${data.london_source}` : ''}`}
            contracts={london}
            currency="GBP"
            usdGbp={usdGbp}
            accentClass={accentClass}
          />
        </div>
      )}
    </div>
  );
}
