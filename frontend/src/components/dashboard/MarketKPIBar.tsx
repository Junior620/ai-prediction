'use client';

import { formatPercentage, formatPrice, formatPriceGbp, getSentimentLabel } from '@/lib/utils';
import { computeChange24h } from '@/lib/marketAnalytics';
import { useUsdGbpRate } from '@/hooks/useUsdGbpRate';
import type { PredictionResponse } from '@/types/api';
import {
  Activity, Brain, DollarSign, Info, TrendingDown, TrendingUp,
} from 'lucide-react';

interface MarketKPIBarProps {
  data: PredictionResponse;
  unitLabel: string;
  includeSentiment: boolean;
  priceSource: string;
  lastUpdate: Date | null;
  accentClass: string;
  /** Affiche l’équivalent GBP (défaut: true pour cacao ICE). */
  showGbp?: boolean;
}

export function MarketKPIBar({
  data,
  unitLabel,
  includeSentiment,
  priceSource,
  lastUpdate,
  accentClass,
  showGbp = true,
}: MarketKPIBarProps) {
  const usdGbp = useUsdGbpRate();
  const currentPrice = data.current_price ?? 0;
  const change24h = computeChange24h(data.historical_prices);
  const garchVol = data.predictions?.[0]?.components?.garch_annualized_volatility;
  const highVol = data.predictions?.some(p => p.components?.high_volatility_regime);

  const currentDate = data.current_date
    ? new Date(data.current_date).toLocaleDateString('fr-FR', {
        day: 'numeric', month: 'long', year: 'numeric',
      })
    : '—';

  const gbpLine = showGbp
    ? `≈ ${formatPriceGbp(currentPrice, usdGbp)} / t`
    : null;

  const kpis = [
    {
      label: 'Prix actuel',
      value: formatPrice(currentPrice),
      sub: gbpLine
        ? `${unitLabel} · ${gbpLine} · ${currentDate}`
        : `${unitLabel} · ${currentDate}`,
      icon: DollarSign,
      highlight: true,
    },
    {
      label: 'Variation 24h',
      value: change24h != null ? formatPercentage(change24h) : '—',
      sub: change24h != null ? 'vs veille (source ML)' : 'Historique insuffisant',
      icon: change24h != null && change24h >= 0 ? TrendingUp : TrendingDown,
      valueClass:
        change24h == null ? 'text-slate-400' :
        change24h >= 0 ? 'text-emerald-400' : 'text-rose-400',
    },
    {
      label: 'Volatilité',
      value: garchVol != null ? `${Number(garchVol).toFixed(1)}%` : '—',
      sub: highVol ? 'Régime volatil' : 'GARCH annualisée',
      icon: Activity,
      valueClass: highVol ? 'text-rose-400' : 'text-emerald-400',
    },
    ...(includeSentiment
      ? [{
          label: 'Sentiment',
          value: getSentimentLabel(data.sentiment_score),
          sub: data.sentiment_score != null
            ? `Score ${data.sentiment_score.toFixed(2)}`
            : 'News cacao',
          icon: Brain,
          valueClass:
            data.sentiment_score == null ? 'text-slate-400' :
            data.sentiment_score > 0.2 ? 'text-emerald-400' :
            data.sentiment_score < -0.2 ? 'text-rose-400' : 'text-amber-400',
        }]
      : []),
  ];

  return (
    <div className="glass-card p-5 space-y-4">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {kpis.map(kpi => (
          <div key={kpi.label} className="min-w-0">
            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1 flex items-center gap-1">
              <kpi.icon className="w-3 h-3" /> {kpi.label}
            </p>
            <p className={`${kpi.highlight ? 'text-2xl font-black text-white font-mono-price' : `text-lg font-bold font-mono-price ${kpi.valueClass ?? 'text-white'}`}`}>
              {kpi.value}
            </p>
            <p className="text-[10px] text-slate-500 mt-0.5 truncate">{kpi.sub}</p>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 pt-3 border-t border-white/[0.06] text-[10px] text-slate-500">
        <span className="flex items-center gap-1">
          <Info className={`w-3 h-3 ${accentClass}`} />
          Source prix ML : {priceSource}
        </span>
        <span>·</span>
        <span>TradingView = CFD temps réel (peut différer de quelques $)</span>
        {showGbp && usdGbp != null && (
          <>
            <span>·</span>
            <span>FX USD/GBP : {usdGbp.toFixed(4)} (Frankfurter / BCE)</span>
          </>
        )}
        {lastUpdate && (
          <>
            <span>·</span>
            <span>
              Dernière MAJ :{' '}
              {lastUpdate.toLocaleTimeString('fr-FR', {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12: false,
              })}
            </span>
          </>
        )}
        <span>·</span>
        <span className="text-slate-600">
          Modèle v{data.model_version?.split('_')[1]?.slice(0, 8) || '—'}
        </span>
      </div>
    </div>
  );
}
