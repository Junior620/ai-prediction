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

export function deriveHorizonSignal(changePct: number): HorizonSignal {
  if (changePct > 1.5) return 'BUY';
  if (changePct < -1.5) return 'SELL';
  return 'HOLD';
}

/** Confiance dérivée de la largeur de l'IC (inversement proportionnelle). */
export function deriveConfidenceScore(
  ci: [number, number],
  anchor: number,
): number {
  const w = ciRelativeWidth(ci, anchor);
  const score = Math.round(Math.max(35, Math.min(92, 100 - w * 85)));
  return score;
}

/** Probabilité directionnelle heuristique (MAPE walk-forward si dispo). */
export function deriveDirectionalProbability(
  changePct: number,
  directionalAccuracy?: number | null,
): number {
  const base = directionalAccuracy != null ? directionalAccuracy * 100 : 62;
  const magnitude = Math.min(Math.abs(changePct) / 30, 1);
  const adjusted = base + magnitude * 18;
  return Math.round(Math.min(88, Math.max(45, adjusted)));
}

export interface ScenarioOutlook {
  label: string;
  price: number;
  probability: number;
  tone: 'bearish' | 'neutral' | 'bullish';
}

export function buildScenarios(
  pred: PredictionItem | undefined,
  currentPrice: number,
): ScenarioOutlook[] {
  if (!pred) return [];
  const [lo, hi] = pred.confidence_interval;
  const mid = pred.price;
  const range = hi - lo || 1;
  const bearishProb = Math.round(Math.max(10, Math.min(70, ((mid - lo) / range) * 55)));
  const bullishProb = Math.round(Math.max(10, Math.min(70, ((hi - mid) / range) * 55)));
  const neutralProb = Math.max(5, 100 - bearishProb - bullishProb);

  return [
    { label: 'Baissier', price: lo, probability: bearishProb, tone: 'bearish' },
    { label: 'Neutre', price: mid, probability: neutralProb, tone: 'neutral' },
    { label: 'Haussier', price: hi, probability: bullishProb, tone: 'bullish' },
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
