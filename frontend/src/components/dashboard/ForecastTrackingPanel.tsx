'use client';

import { useEffect, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '@/lib/api';
import type { ForecastTrackingResponse } from '@/types/api';
import { formatPrice } from '@/lib/utils';

const DIRECTION: Record<string, string> = {
  hausse: 'Hausse',
  baisse: 'Baisse',
  neutre: 'Neutre',
};

function money(value: number | null | undefined, currency: 'USD' | 'GBP') {
  if (value == null || !Number.isFinite(value)) return 'Indisponible';
  return formatPrice(value, currency);
}

export function ForecastTrackingPanel({
  market,
  currency,
}: {
  market: string;
  currency: 'USD' | 'GBP';
}) {
  const storageKey = `forecast-tracking:${market}`;
  const [selected, setSelected] = useState<string | undefined>();
  const [targetDate, setTargetDate] = useState('');
  const [horizon, setHorizon] = useState('');
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState<ForecastTrackingResponse | null>(null);

  useEffect(() => {
    const stored = window.localStorage.getItem(storageKey);
    if (stored) setSelected(stored);
  }, [storageKey]);

  useEffect(() => {
    if (selected) window.localStorage.setItem(storageKey, selected);
  }, [selected, storageKey]);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      api
        .getForecastTracking(market, selected, {
          targetDate: targetDate || undefined,
          horizon: horizon ? Number(horizon) : undefined,
          offset,
          limit: 12,
        })
        .then(payload => {
          if (cancelled) return;
          setData(payload);
          if (!selected && payload?.initials?.[0]?.forecast_id) {
            setSelected(payload.initials[0].forecast_id);
          }
        });
    };
    load();
    const timer = window.setInterval(load, 30000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [market, selected, targetDate, horizon, offset]);

  const initial = data?.initial;
  const latest = data?.latest;
  const sameDirection =
    !data?.direction || data.direction === data.direction_vs_emission_spot;

  return (
    <div className="glass-card px-4 py-3 text-sm text-slate-300">
      <p className="text-slate-500 uppercase tracking-wide text-xs mb-2">
        Suivi d&apos;une échéance
      </p>
      <p className="text-slate-400 mb-3">
        Prévision expérimentale — non validée. Le sens est mesuré contre le cours de départ
        de cette échéance.
      </p>
      {!data?.initials?.length && !initial ? (
        <p className="text-slate-400">Aucune échéance suivie pour le moment.</p>
      ) : (
        <div className="space-y-3">
          <form
            className="flex flex-wrap gap-2 items-end"
            onSubmit={event => {
              event.preventDefault();
              setOffset(0);
            }}
          >
            <label className="text-xs text-slate-400">
              Date cible
              <input
                type="date"
                value={targetDate}
                onChange={event => {
                  setTargetDate(event.target.value);
                  setOffset(0);
                }}
                className="ml-2 bg-transparent border border-white/10 rounded px-2 py-1"
              />
            </label>
            <label className="text-xs text-slate-400">
              Horizon
              <select
                value={horizon}
                onChange={event => {
                  setHorizon(event.target.value);
                  setOffset(0);
                }}
                className="ml-2 bg-transparent border border-white/10 rounded px-2 py-1"
              >
                <option value="">Tous</option>
                <option value="1">J+1</option>
                <option value="7">J+7</option>
                <option value="14">J+14</option>
                <option value="30">J+30</option>
              </select>
            </label>
            <button
              type="button"
              className="text-xs text-slate-400"
              disabled={offset <= 0}
              onClick={() => setOffset(value => Math.max(0, value - 12))}
            >
              Précédentes
            </button>
            <button
              type="button"
              className="text-xs text-slate-400"
              disabled={offset + 12 >= (data?.total_initials ?? 0)}
              onClick={() => setOffset(value => value + 12)}
            >
              Suivantes
            </button>
          </form>
          <div className="flex flex-wrap gap-2">
            {(data?.initials ?? []).map(item => (
              <button
                key={item.forecast_id || `${item.origin_date}-${item.target_date}`}
                type="button"
                onClick={() => setSelected(item.forecast_id || undefined)}
                className={`px-2 py-1 rounded-lg text-xs border ${
                  selected === item.forecast_id
                    ? 'border-amber-400/60 text-white'
                    : 'border-white/10 text-slate-400'
                }`}
              >
                {item.origin_date} → {item.target_date}
                {item.horizon != null ? ` · J+${item.horizon}` : ''}
              </button>
            ))}
          </div>
          {initial && (
            <div className="space-y-1">
              <p>
                Prix initial {money(initial.price, currency)} émis le {initial.origin_date}
                {initial.status === 'unavailable' ? ' · Indisponible' : ''}
              </p>
              <p>
                Dernière révision {money(latest?.price, currency)}
                {latest?.origin_date ? ` le ${latest.origin_date}` : ''}
                {data?.mature
                  ? ' · échéance atteinte'
                  : data?.remaining_horizon != null
                    ? ` · horizon restant J+${data.remaining_horizon}`
                    : ''}
              </p>
              <p>
                Écart{' '}
                {data?.gap == null
                  ? 'Indisponible'
                  : `${data.gap >= 0 ? '+' : ''}${data.gap.toFixed(2)} (${data.gap_pct?.toFixed(2) ?? '—'} %)`}
                {' · '}
                {DIRECTION[data?.direction || ''] || 'Indisponible'}
              </p>
              {(data?.inversions ?? 0) > 0 && (
                <p>
                  L&apos;objectif s&apos;est retourné {data?.inversions} fois contre le cours de départ.
                </p>
              )}
              {!sameDirection && (data?.inversions ?? 0) === 0 && (
                <p className="text-amber-200/90">
                  Le sens contre le cours du jour de l&apos;émission ({DIRECTION[data?.direction_vs_emission_spot || ''] || 'Indisponible'})
                  diffère du sens contre le cours de départ. La succession des objectifs ne s&apos;est pas retournée.
                </p>
              )}
              <p>
                Version initiale {initial.model_version || '—'} · version actuelle{' '}
                {latest?.model_version || '—'}
              </p>
              {data?.model_comparison && (
                <p>
                  Changement de lot :{' '}
                  {data.model_comparison.available
                    ? `${data.model_comparison.previous_model_version} ${money(data.model_comparison.previous_price, currency)} → ${money(data.model_comparison.new_price, currency)}`
                    : 'comparaison indisponible'}
                  . {data.model_comparison.note}
                </p>
              )}
              {data?.mature && (
                <p>
                  Prix réalisé {money(data.realized_price, currency)}
                  {' · erreur initiale '}
                  {data.initial_error == null ? 'Indisponible' : data.initial_error.toFixed(2)}
                  {' · erreur dernière '}
                  {data.latest_error == null ? 'Indisponible' : data.latest_error.toFixed(2)}
                </p>
              )}
              {(data?.revisions?.length ?? 0) === 0 && !data?.mature && (
                <p className="text-slate-500">Aucune révision avant l&apos;échéance.</p>
              )}
              {data?.mature && (data.revisions ?? []).map(revision => (
                <p key={revision.forecast_id || `${revision.origin_date}-${revision.price}`}>
                  Révision du {revision.origin_date || '—'} : erreur{' '}
                  {revision.error == null ? 'Indisponible' : revision.error.toFixed(2)}
                </p>
              ))}
              {!!data?.points?.length && (
                <div className="h-40 mt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data.points}>
                      <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                      <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                      <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} domain={['auto', 'auto']} />
                      <Tooltip />
                      <Line
                        type="monotone"
                        dataKey="price"
                        stroke="#f5b942"
                        dot={{ r: 2 }}
                        connectNulls={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
