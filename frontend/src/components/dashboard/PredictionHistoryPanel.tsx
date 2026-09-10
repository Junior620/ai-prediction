'use client';

import { History } from 'lucide-react';
import { formatPrice } from '@/lib/utils';
import type { PredictionHistoryItem } from '@/types/api';

interface PredictionHistoryPanelProps {
  items: PredictionHistoryItem[];
  loading?: boolean;
  priceCurrency?: 'USD' | 'GBP';
  accentClass?: string;
}

export function PredictionHistoryPanel({
  items,
  loading = false,
  priceCurrency = 'GBP',
  accentClass = 'text-amber-400',
}: PredictionHistoryPanelProps) {
  if (loading && !items.length) {
    return <div className="glass-card h-48 shimmer" />;
  }
  if (!items.length) {
    return null;
  }

  return (
    <div className="glass-card p-5 space-y-4">
      <div>
        <h3 className={`text-sm font-semibold flex items-center gap-2 ${accentClass}`}>
          <History className="w-4 h-4" />
          Historique des prédictions ML
        </h3>
        <p className="text-xs text-slate-500 mt-1">
          Dernières prédictions enregistrées (J+1 / J+7 / J+14 / J+30)
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-slate-500 border-b border-white/[0.06]">
              <th className="text-left py-2 font-medium">Date</th>
              <th className="text-right py-2 font-medium">Horizon</th>
              <th className="text-right py-2 font-medium">Prédiction</th>
              <th className="text-right py-2 font-medium">Intervalle</th>
              <th className="text-left py-2 font-medium pl-3">Modèle</th>
            </tr>
          </thead>
          <tbody>
            {items.map((row, idx) => {
              const when = new Date(row.created_at).toLocaleString('fr-FR', {
                day: '2-digit',
                month: 'short',
                hour: '2-digit',
                minute: '2-digit',
              });
              const lo = row.lower_bound;
              const hi = row.upper_bound;
              return (
                <tr key={`${row.created_at}-${row.horizon}-${idx}`} className="border-b border-white/[0.04]">
                  <td className="py-2 text-slate-300 whitespace-nowrap">{when}</td>
                  <td className="py-2 text-right text-slate-400 tabular-nums">J+{row.horizon}</td>
                  <td className="py-2 text-right text-amber-300 tabular-nums font-medium">
                    {formatPrice(row.predicted_price, priceCurrency)}
                  </td>
                  <td className="py-2 text-right text-slate-500 tabular-nums">
                    {lo != null && hi != null
                      ? `${formatPrice(lo, priceCurrency)} – ${formatPrice(hi, priceCurrency)}`
                      : '—'}
                  </td>
                  <td className="py-2 text-slate-600 pl-3 truncate max-w-[140px]">
                    {row.model_version || '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
