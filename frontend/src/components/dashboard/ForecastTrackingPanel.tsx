'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '@/lib/api';
import type { ForecastTrackingPoint, ForecastTrackingResponse } from '@/types/api';
import { formatPrice } from '@/lib/utils';

/** Horizons suivis sur le site. J+60 et J+90 restent dans le journal, jamais ici. */
const SHOWN_HORIZONS = new Set([1, 7, 14, 30]);
/** Même bande que le journal (NEUTRAL_BAND_PCT) : en dessous, on parle de « stable ». */
const STABLE_BAND_PCT = 1.5;

function isNumber(value: number | null | undefined): value is number {
  return value != null && Number.isFinite(value);
}

function money(value: number | null | undefined, currency: 'USD' | 'GBP') {
  return isNumber(value) ? formatPrice(value, currency) : 'indisponible';
}

/** « +718,73 £GB » ou « −120,00 £GB » : le signe est toujours écrit. */
function signedMoney(value: number, currency: 'USD' | 'GBP') {
  const sign = value > 0 ? '+' : value < 0 ? '−' : '';
  return `${sign}${formatPrice(Math.abs(value), currency)}`;
}

function signedPct(value: number) {
  const sign = value > 0 ? '+' : value < 0 ? '−' : '';
  return `${sign}${Math.abs(value).toLocaleString('fr-FR', {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  })} %`;
}

function day(value: string | null | undefined, withYear = true) {
  if (!value) return '—';
  const date = new Date(`${value.slice(0, 10)}T12:00:00`);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString('fr-FR', {
    day: 'numeric',
    month: 'long',
    ...(withYear ? { year: 'numeric' } : {}),
  });
}

function shortDay(value: string | null | undefined) {
  if (!value) return '—';
  const date = new Date(`${value.slice(0, 10)}T12:00:00`);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString('fr-FR', { day: 'numeric', month: 'short' });
}

/** Variation contre le cours de départ. Le mot vient du même nombre que le pourcentage affiché. */
function move(price: number | null | undefined, reference: number | null | undefined) {
  if (!isNumber(price) || !isNumber(reference) || reference <= 0) return null;
  const pct = (price / reference - 1) * 100;
  const word =
    pct > STABLE_BAND_PCT ? 'une hausse' : pct < -STABLE_BAND_PCT ? 'une baisse' : 'un cours stable';
  return { pct, word };
}

function optionLabel(item: ForecastTrackingPoint) {
  const horizon = item.horizon != null ? ` (${item.horizon} j)` : '';
  const missing = isNumber(item.price) ? '' : ' · sans prix';
  return `Échéance du ${shortDay(item.target_date)} · prévue le ${shortDay(item.origin_date)}${horizon}${missing}`;
}

export function ForecastTrackingPanel({
  market,
  currency,
}: {
  market: string;
  currency: 'USD' | 'GBP';
}) {
  const [selected, setSelected] = useState<string | undefined>();
  const [data, setData] = useState<ForecastTrackingResponse | null>(null);

  useEffect(() => {
    setSelected(undefined);
    setData(null);
  }, [market]);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      api.getForecastTracking(market, selected, { limit: 100 }).then(payload => {
        if (cancelled) return;
        setData(payload);
        // Par défaut : l'échéance la plus récente.
        const first = payload?.initials?.find(
          item => item.forecast_id && item.target_date && SHOWN_HORIZONS.has(Number(item.horizon)),
        );
        if (!selected && first?.forecast_id) setSelected(first.forecast_id);
      });
    };
    load();
    const timer = window.setInterval(load, 30000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [market, selected]);

  const choices = useMemo(
    () =>
      (data?.initials ?? []).filter(
        item => item.forecast_id && item.target_date && SHOWN_HORIZONS.has(Number(item.horizon)),
      ),
    [data?.initials],
  );

  const initial = data?.initial;
  const shown = initial && SHOWN_HORIZONS.has(Number(initial.horizon)) ? initial : null;
  // Dernière révision qui a un prix (une révision « indisponible » ne remplace pas la précédente).
  const latest = [...(data?.revisions ?? [])]
    .reverse()
    .find(revision => isNumber(revision.price) && revision.price > 0);
  const hasRevision = latest != null;
  const reference = data?.reference_price ?? shown?.reference_price ?? shown?.origin_price;
  const planned = shown?.price;
  const realized = data?.mature ? data.realized_price : null;
  const plannedMove = move(planned, reference);
  const latestMove = move(latest?.price, reference);
  const gap = isNumber(realized) && isNumber(planned) ? realized - planned : null;
  const gapPct = gap != null && isNumber(planned) && planned !== 0 ? (gap / planned) * 100 : null;
  const latestGap =
    isNumber(realized) && isNumber(latest?.price) ? realized - (latest?.price as number) : null;
  // Courbe : la prévision initiale, puis la dernière estimation de chaque jour suivant.
  const chartPoints = (() => {
    if (!shown || !isNumber(planned)) return [];
    const byDay = new Map<string, number>();
    for (const revision of data?.revisions ?? []) {
      if (revision.origin_date && isNumber(revision.price) && revision.price > 0) {
        byDay.set(revision.origin_date.slice(0, 10), revision.price);
      }
    }
    return [
      { date: shown.origin_date ?? '', price: planned, label: 'Prévision initiale' },
      ...[...byDay.entries()]
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([date, price]) => ({ date, price, label: 'Estimation' })),
    ].map((point, index) => ({ ...point, index }));
  })();

  return (
    <div className="glass-card px-4 py-3 text-sm text-slate-300">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <p className="text-slate-500 uppercase tracking-wide text-xs">
          Suivi d&apos;une échéance
        </p>
        {choices.length > 1 && (
          <select
            aria-label="Choisir une échéance"
            value={selected ?? ''}
            onChange={event => setSelected(event.target.value || undefined)}
            className="max-w-full bg-slate-900 border border-white/10 rounded px-2 py-1 text-xs text-slate-300"
          >
            {choices.map(item => (
              <option key={item.forecast_id as string} value={item.forecast_id as string}>
                {optionLabel(item)}
              </option>
            ))}
          </select>
        )}
      </div>

      {!shown ? (
        <p className="text-slate-400">
          {choices.length ? 'Chargement de l’échéance…' : 'Aucune échéance suivie pour le moment.'}
        </p>
      ) : (
        <div className="space-y-2">
          <p className="text-white font-medium">
            Échéance du {day(shown.target_date)}
            <span className="text-slate-400 font-normal">
              {' '}
              — prévision faite le {day(shown.origin_date, false)}
              {shown.horizon != null ? `, à ${shown.horizon} jour${shown.horizon > 1 ? 's' : ''}` : ''}
            </span>
          </p>

          {isNumber(planned) ? (
            <p className="text-base leading-snug">
              Prévu par le modèle <strong className="text-white">{money(planned, currency)}</strong>
              {isNumber(realized) ? (
                <>
                  , réalisé <strong className="text-white">{money(realized, currency)}</strong>
                  {gap != null && (
                    <>
                      {' '}: écart{' '}
                      <strong className="text-white">
                        {signedMoney(gap, currency)}
                        {gapPct != null ? ` (${signedPct(gapPct)})` : ''}
                      </strong>
                      {gap > 0
                        ? ' — le cours a fini au-dessus de la prévision.'
                        : gap < 0
                          ? ' — le cours a fini en dessous de la prévision.'
                          : ' — exactement la prévision.'}
                    </>
                  )}
                </>
              ) : data?.mature ? (
                <>, cours réalisé indisponible.</>
              ) : (
                <>. Le cours réalisé sera connu le {day(shown.target_date, false)}.</>
              )}
            </p>
          ) : (
            <p>Le modèle n&apos;a pas donné de prix pour cette échéance.</p>
          )}

          {isNumber(reference) && (
            <p className="text-slate-400">
              Cours de départ {money(reference, currency)}
              {plannedMove
                ? ` : le modèle annonçait ${plannedMove.word} (${signedPct(plannedMove.pct)}).`
                : '.'}
            </p>
          )}

          {hasRevision && latest && isNumber(latest.price) && (
            <p className="text-slate-400">
              Dernière estimation {money(latest.price, currency)} le {day(latest.origin_date, false)}
              {latestMove ? `, soit ${latestMove.word} (${signedPct(latestMove.pct)}) par rapport au départ` : ''}
              {latestGap != null ? ` ; écart au réalisé ${signedMoney(latestGap, currency)}` : ''}.
              {(data?.inversions ?? 0) > 0 &&
                ` Le modèle a changé de sens ${data?.inversions} fois en cours de route.`}
            </p>
          )}

          {/* Un seul point ne dit rien : pas de graphique. */}
          {chartPoints.length >= 2 && (
            <div className="h-36 mt-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartPoints}>
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                  <XAxis
                    dataKey="index"
                    tickFormatter={index => shortDay(chartPoints[Number(index)]?.date)}
                    tick={{ fill: '#94a3b8', fontSize: 10 }}
                  />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} domain={['auto', 'auto']} width={48} />
                  <Tooltip
                    labelFormatter={index => {
                      const point = chartPoints[Number(index)];
                      return point ? `${point.label} du ${day(point.date, false)}` : '';
                    }}
                    formatter={value => [money(Number(value), currency), 'Modèle expérimental']}
                  />
                  {isNumber(reference) && (
                    <ReferenceLine
                      y={reference}
                      ifOverflow="extendDomain"
                      stroke="#64748b"
                      strokeDasharray="4 4"
                      label={{ value: 'cours de départ', fill: '#64748b', fontSize: 10, position: 'insideTopLeft' }}
                    />
                  )}
                  {isNumber(realized) && (
                    <ReferenceLine
                      y={realized}
                      ifOverflow="extendDomain"
                      stroke="#e2e8f0"
                      label={{ value: 'réalisé', fill: '#e2e8f0', fontSize: 10, position: 'insideBottomLeft' }}
                    />
                  )}
                  <Line type="linear" dataKey="price" stroke="#f5b942" dot={{ r: 2 }} />

                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          <p className="text-[11px] text-slate-500 pt-1">
            Prévision expérimentale — non validée. Le prix publié reste le cours inchangé.
          </p>
        </div>
      )}
    </div>
  );
}
