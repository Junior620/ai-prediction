import type { MarketBriefContent, PredictionItem } from '@/types/api';

export type UncertaintyLevel = 'low' | 'medium' | 'high';
export type HorizonSignal = 'BUY' | 'SELL' | 'HOLD';

export function computeChange24h(
  historical?: { date: string; price: number }[],
): number | null {
  if (!historical || historical.length < 2) return null;
  const sorted = [...historical].sort(
    (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime(),
  );
  const last = sorted[sorted.length - 1];
  const prev = sorted[sorted.length - 2];
  if (!prev.price) return null;
  return ((last.price - prev.price) / prev.price) * 100;
}

export function ciRelativeWidth(
  ci: [number, number],
  anchor: number,
): number {
  if (!anchor || anchor <= 0) return 1;
  return (ci[1] - ci[0]) / anchor;
}

export function uncertaintyLevel(
  ci: [number, number],
  anchor: number,
): UncertaintyLevel {
  const w = ciRelativeWidth(ci, anchor);
  if (w < 0.35) return 'low';
  if (w < 0.7) return 'medium';
  return 'high';
}

/** True when the guard replaced the model price with the latest close. */
export function isForecastUnavailable(pred?: PredictionItem | null): boolean {
  return Boolean(pred?.components?.feature_failure);
}

export function deriveHorizonSignal(changePct: number): HorizonSignal {
  if (changePct > 1.5) return 'BUY';
  if (changePct < -1.5) return 'SELL';
  return 'HOLD';
}

export interface ScenarioOutlook {
  label: string;
  price: number;
  tone: 'bearish' | 'neutral' | 'bullish';
}

export function buildScenarios(
  pred: PredictionItem | undefined,
  _currentPrice: number,
): ScenarioOutlook[] {
  if (!pred) return [];
  const [lo, hi] = pred.confidence_interval;
  return [
    { label: 'Borne basse', price: lo, tone: 'bearish' },
    { label: 'Centre', price: pred.price, tone: 'neutral' },
    { label: 'Borne haute', price: hi, tone: 'bullish' },
  ];
}

export interface InfluentialFactor {
  factor: string;
  impact: string;
  tone: 'bearish' | 'bullish' | 'neutral' | 'risk';
}

export function buildInfluentialFactors(opts: {
  changePct7d: number;
  garchVol?: number | null;
  highVolRegime?: boolean;
  sentiment?: number | null;
  brief?: MarketBriefContent | null;
  pred?: PredictionItem;
}): InfluentialFactor[] {
  const factors: InfluentialFactor[] = [];

  if (opts.changePct7d < -5) {
    factors.push({ factor: 'Momentum court terme', impact: 'Baissier fort', tone: 'bearish' });
  } else if (opts.changePct7d > 5) {
    factors.push({ factor: 'Momentum court terme', impact: 'Haussier', tone: 'bullish' });
  } else {
    factors.push({ factor: 'Momentum court terme', impact: 'Neutre', tone: 'neutral' });
  }

  if (opts.highVolRegime || (opts.garchVol != null && opts.garchVol > 25)) {
    factors.push({ factor: 'Volatilité GARCH', impact: 'Risque élevé', tone: 'risk' });
  } else if (opts.garchVol != null) {
    factors.push({
      factor: 'Volatilité GARCH',
      impact: `${opts.garchVol.toFixed(1)}% ann.`,
      tone: 'neutral',
    });
  }

  if (opts.sentiment != null) {
    const tone = opts.sentiment > 0.2 ? 'bullish' : opts.sentiment < -0.2 ? 'bearish' : 'neutral';
    factors.push({
      factor: 'Sentiment news',
      impact: opts.sentiment > 0.2 ? 'Positif' : opts.sentiment < -0.2 ? 'Négatif' : 'Neutre',
      tone,
    });
  }

  const support = opts.brief?.key_levels?.support;
  const resistance = opts.brief?.key_levels?.resistance;
  if (support != null) {
    factors.push({ factor: `Support ${Math.round(support).toLocaleString('fr-FR')} $`, impact: 'Zone clé', tone: 'bullish' });
  }
  if (resistance != null) {
    factors.push({ factor: `Résistance ${Math.round(resistance).toLocaleString('fr-FR')} $`, impact: 'Seuil critique', tone: 'bearish' });
  }

  const comps = opts.pred?.components;
  if (comps?.prophet != null && comps?.baseline != null) {
    const spread = Math.abs(comps.prophet - comps.baseline) / (comps.baseline || 1);
    if (spread > 0.08) {
      factors.push({
        factor: 'Divergence Prophet / XGB',
        impact: spread > 0.15 ? 'Signal mixte' : 'Légère divergence',
        tone: 'risk',
      });
    }
  }

  if (opts.brief?.signal === 'SELL' && opts.brief.trend === 'bearish') {
    factors.push({ factor: 'Brief IA (Sonnet)', impact: 'Biais baissier confirmé', tone: 'bearish' });
  } else if (opts.brief?.signal === 'BUY') {
    factors.push({ factor: 'Brief IA (Sonnet)', impact: 'Biais haussier', tone: 'bullish' });
  }

  return factors.slice(0, 6);
}

export const uncertaintyLabels: Record<UncertaintyLevel, string> = {
  low: 'Incertitude faible',
  medium: 'Incertitude modérée',
  high: 'Incertitude élevée',
};
