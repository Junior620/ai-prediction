'use client';

import { formatPercentage, formatPrice, formatPriceGbp } from '@/lib/utils';
import {
  deriveHorizonSignal,
  uncertaintyLevel,
  uncertaintyLabels,
} from '@/lib/marketAnalytics';
import { useUsdGbpRate } from '@/hooks/useUsdGbpRate';
import type { HorizonValidationMetrics, PredictionItem, RevisionValueHorizon, ValidationMetricsResponse } from '@/types/api';
import { AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react';

const signalStyles = {
  BUY: { label: 'BUY', class: 'text-emerald-400 bg-emerald-500/15 border-emerald-500/30' },
  SELL: { label: 'SELL', class: 'text-rose-400 bg-rose-500/15 border-rose-500/30' },
  HOLD: { label: 'HOLD', class: 'text-amber-400 bg-amber-500/15 border-amber-500/30' },
};

const MONTHS = [
  'janvier', 'février', 'mars', 'avril', 'mai', 'juin',
  'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre',
];

function frenchDay(iso: string | null | undefined) {
  if (!iso) return '';
  const [year, month, day] = iso.slice(0, 10).split('-').map(Number);
  if (!year || !month || !day) return iso.slice(0, 10);
  const label = `${day} ${MONTHS[month - 1]}`;
  return year === new Date().getFullYear() ? label : `${label} ${year}`;
}

function sessionsLabel(count: number) {
  if (count <= 0) return 'échéance atteinte';
  return count === 1 ? '1 séance restante' : `${count} séances restantes`;
}

const VERDICT = {
  etabli: 'établi',
  defavorable: 'défavorable',
  incertain: 'encore incertain',
} as const;

const verdictClass = {
  etabli: 'text-emerald-300',
  defavorable: 'text-rose-300',
  incertain: 'text-amber-200',
} as const;

function signed(value: number, fmt: (n: number) => string) {
  return `${value > 0 ? '+' : ''}${fmt(value)}`;
}

function DelayMeasure({
  revisionValue,
  revisionAsOf,
  sessionsRemaining,
  openingHorizon,
  fmt,
}: {
  revisionValue: RevisionValueHorizon;
  revisionAsOf?: string | null;
  sessionsRemaining?: number | null;
  openingHorizon: number;
  fmt: (n: number) => string;
}) {
  const delay = revisionValue.steps?.find(
    step => step.comparison === 'initial_vs_remaining' && step.remaining === sessionsRemaining,
  );
  const level = Math.round(((delay?.interval_level ?? revisionValue.interval_level ?? 0.9) * 100));
  return (
    <div className="pt-1 space-y-1 text-slate-400">
      {delay ? (
        <>
          <p className={`font-semibold ${verdictClass[delay.verdict]}`}>
            Initiale J+{openingHorizon} contre révision à {sessionsLabel(delay.remaining)} : {VERDICT[delay.verdict]}.
          </p>
          <p>
            {delay.available} paires sur {delay.expected}
            {delay.availability_rate != null
              ? ` · disponibilité ${(delay.availability_rate * 100).toFixed(0)} %`
              : ''}
            {delay.abstentions > 0 ? ` · ${delay.abstentions} prix manquants` : ''}.
            {delay.mean_gain != null && (
              <>
                {' '}Erreur moyenne {fmt(delay.mean_initial_error ?? 0)} pour l&apos;initiale,{' '}
                {fmt(delay.mean_revised_error ?? 0)} pour la révision
                {delay.mean_spot_error != null
                  ? `, ${fmt(delay.mean_spot_error)} pour le cours connu ce jour-là`
                  : ''}
                . Gain moyen {signed(delay.mean_gain, fmt)}
                {delay.gain_low != null && delay.gain_high != null
                  ? ` (intervalle à ${level} % : ${signed(delay.gain_low, fmt)} à ${signed(delay.gain_high, fmt)})`
                  : ` (intervalle à ${level} % non calculé)`}
                .
              </>
            )}
          </p>
        </>
      ) : (
        <p>
          Pas de mesure historique pour ce délai
          {sessionsRemaining == null ? '' : ` (${sessionsLabel(sessionsRemaining)})`}.
        </p>
      )}
      <p>Le lot d&apos;aujourd&apos;hui n&apos;est pas jugé par ce rejeu.</p>
      <p className="text-slate-500">
        Initiale contre dernière estimation avant échéance : {VERDICT[revisionValue.verdict]}
        {revisionValue.expiries > 0 && revisionValue.mean_gain != null
          ? `. ${revisionValue.expiries} échéances${revisionAsOf ? ` notées au ${frenchDay(revisionAsOf)}` : ''}, gain moyen ${signed(revisionValue.mean_gain, fmt)}`
          : ''}
        .
      </p>
    </div>
  );
}

interface PredictionHorizonCardProps {
  pred: PredictionItem;
  currentPrice: number;
  priceCurrency?: 'USD' | 'GBP';
  validation?: HorizonValidationMetrics | null;
  evaluatedPeriod?: Pick<
    ValidationMetricsResponse,
    'origin_start' | 'origin_end' | 'target_start' | 'target_end'
  > | null;
  revisionValue?: RevisionValueHorizon | null;
  revisionAsOf?: string | null;
}

export function PredictionHorizonCard({
  pred,
  currentPrice,
  priceCurrency = 'USD',
  validation,
  evaluatedPeriod,
  revisionValue,
  revisionAsOf,
}: PredictionHorizonCardProps) {
  const price = pred.price;
  const frozen = pred.horizon === 7 || pred.horizon === 14 || pred.horizon === 30;
  const basis = frozen && pred.reference_price ? pred.reference_price : currentPrice;
  const change = price == null ? 0 : price - basis;
  const pct = basis && price != null ? (change / basis) * 100 : 0;
  const up = change >= 0;
  const signal = deriveHorizonSignal(pct);
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
  const openingHorizon = pred.initial_horizon ?? pred.horizon;
  const commitmentLabel = frozen && pred.target_date
    ? [
        `Horizon initial : J+${openingHorizon}`,
        `échéance : ${frenchDay(pred.target_date)}`,
        pred.sessions_remaining == null ? null : sessionsLabel(pred.sessions_remaining),
      ].filter((part): part is string => Boolean(part)).join(' — ')
    : null;

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
        <div>
          {commitmentLabel ? (
            <p className="text-[11px] font-semibold text-slate-300 leading-snug max-w-[18rem]">
              {commitmentLabel}
            </p>
          ) : (
            <span className="text-xs font-bold text-slate-500 uppercase">
              {pred.horizon === 1 ? 'J+1' : `J+${pred.horizon}`}
            </span>
          )}
        </div>
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

      {frozen && pred.target_date && (
        <div className="text-[11px] text-slate-300 mb-3 space-y-1">
          {(pred.initial?.model_version || pred.latest?.model_version) && (
            <p className="text-slate-500">
              {pred.initial?.model_version ? `Initiale ${pred.initial.model_version}` : 'Initiale'}
              {pred.latest?.model_version ? ` · révision ${pred.latest.model_version}` : ''}
            </p>
          )}
          {pred.latest_unavailable ? (
            <p>Dernière estimation indisponible.</p>
          ) : !pred.has_revision || pred.latest_price == null ? (
            <p>Pas encore de révision pour cette échéance.</p>
          ) : (
            <>
              <p>
                Dernière estimation du {frenchDay(pred.latest_date)} : {fmt(pred.latest_price)}
              </p>
              <p>
                Écart avec l&apos;initiale :{' '}
                {pred.latest_gap == null ? 'Indisponible' : fmt(pred.latest_gap)}
                {pred.latest_gap_pct == null ? '' : ` (${formatPercentage(pred.latest_gap_pct)})`}
              </p>
            </>
          )}
          {pred.revision_notice === 'inversion' && (
            <p className="text-rose-300 font-semibold">
              Inversion : la dernière estimation change le sens de l&apos;initiale.
            </p>
          )}
          {pred.revision_notice === 'revision' && (
            <p className="text-amber-200 font-semibold">
              Révision importante : l&apos;écart avec l&apos;initiale dépasse 1,5 %.
            </p>
          )}
          {revisionValue && (
            <DelayMeasure
              revisionValue={revisionValue}
              revisionAsOf={revisionAsOf}
              sessionsRemaining={pred.sessions_remaining}
              openingHorizon={openingHorizon}
              fmt={fmt}
            />
          )}
        </div>
      )}

      <div className="flex flex-wrap gap-1.5 mb-3">
        {experimental && (
          <span className="text-[10px] font-bold px-2 py-0.5 rounded border text-sky-200 bg-sky-500/15 border-sky-500/30">
            Prévision expérimentale — non validée
          </span>
        )}
        {pred.components?.warning && (
          <span className="text-[10px] font-bold px-2 py-0.5 rounded border text-amber-200 bg-amber-500/15 border-amber-500/30">
            {String(pred.components.warning)}
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
