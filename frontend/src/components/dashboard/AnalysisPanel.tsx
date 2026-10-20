'use client';

import { formatPrice } from '@/lib/utils';
import {
  buildInfluentialFactors,
  buildScenarios,
  isForecastUnavailable,
} from '@/lib/marketAnalytics';
import type {
  MarketBriefContent,
  PredictionItem,
  ValidationMetricsResponse,
} from '@/types/api';
import {
  AlertTriangle, Layers, Target,
} from 'lucide-react';

interface AnalysisPanelProps {
  predictions: PredictionItem[];
  currentPrice: number;
  priceCurrency?: 'USD' | 'GBP';
  sentiment?: number | null;
  garchVol?: number | null;
  highVolRegime?: boolean;
  brief?: MarketBriefContent | null;
  validation?: ValidationMetricsResponse | null;
  modelVersion?: string;
}

const impactColors = {
  bearish: 'text-rose-400',
  bullish: 'text-emerald-400',
  neutral: 'text-slate-300',
  risk: 'text-amber-400',
};

const scenarioColors = {
  bearish: 'border-rose-500/20 bg-rose-500/5',
  neutral: 'border-white/[0.08] bg-white/[0.02]',
  bullish: 'border-emerald-500/20 bg-emerald-500/5',
};

export function AnalysisPanel({
  predictions,
  currentPrice,
  priceCurrency = 'USD',
  sentiment,
  garchVol,
  highVolRegime,
  brief,
  validation,
  modelVersion,
}: AnalysisPanelProps) {
  const pred7 = predictions.find(p => p.horizon === 7);
  const pred30 = predictions.find(p => p.horizon === 30) ?? predictions[predictions.length - 1];
  const pred7Validated =
    (pred7?.status ?? pred7?.components?.status) === 'validated' && !pred7?.components?.served_as_close;
  const change7 =
    pred7Validated && pred7 && pred7.price != null && currentPrice
      ? ((pred7.price - currentPrice) / currentPrice) * 100
      : null;

  const factors = buildInfluentialFactors({
    changePct7d: change7,
    garchVol,
    highVolRegime,
    sentiment,
    brief,
    pred: pred7,
  });

  const h30 = validation?.xgb_metrics?.find(row => row.horizon === (pred30?.horizon ?? 30));
  const pred30Status = pred30?.status ?? pred30?.components?.status;
  const pred30Usable = pred30 && pred30.price != null && pred30Status !== 'unavailable' ? pred30 : undefined;
  const scenarios = buildScenarios(pred30Usable, currentPrice);

  return (
    <div className="space-y-5">
      {/* Facteurs influents */}
      <section className="glass-card p-5">
        <h3 className="text-sm font-bold text-white flex items-center gap-2 mb-4">
          <Layers className="w-4 h-4 text-violet-400" />
          Facteurs influents
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 border-b border-white/[0.06]">
                <th className="text-left py-2 font-semibold">Facteur</th>
                <th className="text-right py-2 font-semibold">Impact</th>
              </tr>
            </thead>
            <tbody>
              {factors.map((f, i) => (
                <tr key={i} className="border-b border-white/[0.03] last:border-0">
                  <td className="py-2.5 text-slate-300">{f.factor}</td>
                  <td className={`py-2.5 text-right font-semibold ${impactColors[f.tone]}`}>
                    {f.impact}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Scénarios J+30 */}
      <section className="glass-card p-5">
        <h3 className="text-sm font-bold text-white flex items-center gap-2 mb-4">
          <Target className="w-4 h-4 text-amber-400" />
          Scénarios J+{pred30?.horizon ?? 30}
        </h3>
        {scenarios.length === 0 ? (
          <p className="text-xs text-amber-200/90 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            Prévision J+30 indisponible. Le modèle s’écarte trop du cours, le dernier cours n’est pas un scénario.
          </p>
        ) : (
          <>
            <div className="grid sm:grid-cols-3 gap-3">
              {scenarios.map(s => (
                <div key={s.label} className={`rounded-xl border p-4 ${scenarioColors[s.tone]}`}>
                  <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">
                    {s.label}
                  </p>
                  <p className="text-lg font-black text-white font-mono-price">{formatPrice(s.price, priceCurrency)}</p>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-slate-600 mt-3">
              Borne basse, centre et borne haute. Pas de pourcentage.
            </p>
          </>
        )}
      </section>

      {/* Disclaimer */}
      <div className="rounded-xl border border-amber-500/25 bg-amber-500/5 px-4 py-3 flex items-start gap-3">
        <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <p className="text-xs text-amber-200/90 leading-relaxed">
          <strong className="font-semibold">Analyse informative uniquement.</strong>{' '}
          Ne constitue pas un conseil financier. Les prédictions ML comportent une incertitude
          significative — vérifiez toujours avec vos propres critères de risque avant toute décision.
        </p>
      </div>
    </div>
  );
}
