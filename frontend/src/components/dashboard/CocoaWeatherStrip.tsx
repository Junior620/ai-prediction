'use client';

import type { CocoaWeatherResponse } from '@/types/api';
import { CloudSun } from 'lucide-react';

function formatWhen(value?: string | null): string {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value.slice(0, 16);
  return date.toLocaleString('fr-FR', {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function CocoaWeatherStrip({
  data,
}: {
  data: CocoaWeatherResponse | null;
}) {
  const locations = data?.locations ?? [];
  return (
    <div className="glass-card px-4 py-3">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-slate-300">
        <span className="text-slate-500 uppercase tracking-wide text-xs flex items-center gap-1.5">
          <CloudSun className="w-3.5 h-3.5" />
          Météo
        </span>
        {locations.length === 0 && (
          <span className="text-slate-400">
            Aucun relevé pour la Côte d&apos;Ivoire et le Ghana.
          </span>
        )}
        {locations.map(place => (
          <span key={place.location}>
            <span className="text-slate-100">{place.location}</span>
            {place.temperature_c != null && (
              <strong className="text-slate-100"> {place.temperature_c.toFixed(1)}°C</strong>
            )}
            {place.precipitation_mm != null && (
              <span> · {place.precipitation_mm.toFixed(1)} mm</span>
            )}
            {place.collected_at && (
              <span className="text-slate-500"> · {formatWhen(place.collected_at)}</span>
            )}
          </span>
        ))}
      </div>
      <p className="mt-1 text-xs text-slate-500">
        {data?.note ?? 'Contexte météo, non utilisé par la prévision.'}
      </p>
    </div>
  );
}
