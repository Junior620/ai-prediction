'use client';

import { formatPercentage, formatPrice, formatPriceGbp } from '@/lib/utils';
import {
  deriveHorizonSignal,
  uncertaintyLevel,
  uncertaintyLabels,
} from '@/lib/marketAnalytics';
import { useUsdGbpRate } from '@/hooks/useUsdGbpRate';
import type { HorizonValidationMetrics, PredictionItem, ValidationMetricsResponse } from '@/types/api';
import { AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react';

const signalStyles = {
  BUY: { label: 'BUY', class: 'text-emerald-400 bg-emerald-500/15 border-emerald-500/30' },
  SELL: { label: 'SELL', class: 'text-rose-400 bg-rose-500/15 border-rose-500/30' },
  HOLD: { label: 'HOLD', class: 'text-amber-400 bg-amber-500/15 border-amber-500/30' },
};

interface PredictionHorizonCardProps {
  pred: PredictionItem;
  currentPrice: number;
  priceCurrency?: 'USD' | 'GBP';
  briefSignal?: 'BUY' | 'SELL' | 'HOLD';
  validation?: HorizonValidationMetrics | null;
  evaluatedPeriod?: Pick<
    ValidationMetricsResponse,
    'origin_start' | 'origin_end' | 'target_start' | 'target_end'
  > | null;
}

export function PredictionHorizonCard({
  pred,
  currentPrice,
  priceCurrency = 'USD',
  briefSignal,
  validation,
  evaluatedPeriod,
}: PredictionHorizonCardProps) {
  const price = pred.price;
  const change = price == null ? 0 : price - currentPrice;
  const pct = currentPrice && price != null ? (change / currentPrice) * 100 : 0;
  const up = change >= 0;
  const derived = deriveHorizonSignal(pct);
  const signal = pred.horizon === 7 && briefSignal ? briefSignal : derived;
  const sigStyle = signalStyles[signal];
  const interval = pred.confidence_interval;
  const [lo, hi] = interval ?? [pred.price ?? currentPrice, pred.price ?? currentPrice];
  const intervalWidth = hi - lo;
  const uncertainty = interval && price != null
    ? uncertaintyLevel(interval, currentPrice || price)
    : 'low';
  const periodLabel = evaluatedPeriod?.origin_start
    ? `${evaluatedPeriod.origin_start} → ${evaluatedPeriod.target_end || evaluatedPeriod.origin_end}`
    : null;
  const usdGbp = useUsdGbpRate();
  const isGbp = priceCurrency === 'GBP';
  const fmt = (n: number) => formatPrice(n, priceCurrency);
  const status = pred.status ?? pred.components?.status ?? 'experimental';
  const unavailable = status === 'unavailable' || price == null;
  const experimental = status !== 'validated';

  if (unavailable) {
    return (
      <div className="glass-card-hover p-4 flex flex-col h-full border-amber-500/30">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-bold text-slate-500 uppercase">
            {pred.horizon === 1 ? 'J+1' : `J+${pred.horizon}`}
          </span>
          <span className="text-xs font-bold text-slate-500">—</span>
        </div>
        <p className="text-xl font-black text-slate-400">Indisponible</p>
        <p className="text-[11px] text-slate-500 mt-3">Pas de prix ni d'intervalle pour cet horizon.</p>
      </div>
    );
  }

  return (
    <div className="glass-card-hover p-4 flex flex-col h-full">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-bold text-slate-500 uppercase">
          {pred.horizon === 1 ? 'J+1' : `J+${pred.horizon}`}
        </span>
        <span className={`text-xs font-bold px-1.5 py-0.5 rounded flex items-center gap-0.5 ${up ? 'text-emerald-400' : 'text-rose-400'}`}>
          {up ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          {formatPercentage(pct)}
        </span>
      </div>

      <p className="text-xl font-black text-white font-mono-price">{fmt(price as number)}</p>
      {!isGbp && (
        <p className="text-[11px] text-slate-400 font-mono-price mb-3">
          ≈ {formatPriceGbp(price as number, usdGbp)} / t
        </p>
      )}
      {isGbp && <div className="mb-3" />}

      <div className="flex flex-wrap gap-1.5 mb-3">
        {experimental && (
          <span className="text-[10px] font-bold px-2 py-0.5 rounded border text-sky-200 bg-sky-500/15 border-sky-500/30">
            Prévision expérimentale — non validée
          </span>
        )}
        <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${sigStyle.class}`}>
          Signal : {sigStyle.label}
        </span>
      </div>

      <p className="text-[10px] text-slate-400 mb-1">
        {validation?.mape != null ? (
          <>
            Erreur walk-forward MAPE{' '}
            <span className="text-slate-200 font-semibold">{validation.mape.toFixed(1)}%</span>
          </>
        ) : (
          'Erreur walk-forward non disponible'
        )}
        {validation?.mae != null && (
          <>
            {' · '}MAE{' '}
            <span className="text-slate-200 font-semibold">{fmt(validation.mae)}</span>
          </>
        )}
      </p>
      {periodLabel && (
        <p className="text-[10px] text-slate-500 mb-1">Période évaluée : {periodLabel}</p>
      )}

      {interval && (
        <p className="text-[10px] text-slate-500 mt-auto">
          {experimental
            ? 'Bande expérimentale — couverture non démontrée'
            : `IC ${Math.round((pred.confidence_level ?? 0.9) * 100)}%`}
          {' : '}
          {fmt(lo)} – {fmt(hi)}
          {' · '}largeur {fmt(intervalWidth)}
        </p>
      )}

      {uncertainty === 'high' && (
        <p className="text-[10px] text-amber-400/90 flex items-center gap-1 mt-2">
          <AlertTriangle className="w-3 h-3 shrink-0" />
          {uncertaintyLabels[uncertainty]} · intervalle très large
        </p>
      )}
      {uncertainty === 'medium' && (
        <p className="text-[10px] text-slate-500 mt-2">{uncertaintyLabels[uncertainty]}</p>
      )}
    </div>
  );
}
