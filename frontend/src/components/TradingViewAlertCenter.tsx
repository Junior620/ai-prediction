'use client';

import type { LatestTradingViewAlert } from '@/types/api';
import {
  TV_SIGNAL_OPTIONS,
  type TvSignalFilter,
} from '@/hooks/useTradingViewAlertPoll';
import { Bell, X, Check } from 'lucide-react';

const signalLabels: Record<string, string> = {
  buy: 'Achat',
  sell: 'Vente',
  support_break: 'Cassure support',
  resistance_break: 'Cassure résistance',
  trend_change: 'Tendance',
  custom: 'Autre',
};

interface TradingViewAlertCenterProps {
  open: boolean;
  onClose: () => void;
  history: LatestTradingViewAlert[];
  filters: TvSignalFilter[];
  onToggleFilter: (id: TvSignalFilter) => void;
  muted: boolean;
  onToggleMute: () => void;
  notifEnabled: boolean;
  onToggleNotif: (value: boolean) => void;
  onSelectAlert?: (alert: LatestTradingViewAlert) => void;
}

export function TradingViewAlertCenter({
  open,
  onClose,
  history,
  filters,
  onToggleFilter,
  muted,
  onToggleMute,
  notifEnabled,
  onToggleNotif,
  onSelectAlert,
}: TradingViewAlertCenterProps) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[70] flex justify-end">
      <button
        type="button"
        className="absolute inset-0 bg-black/50 backdrop-blur-[1px]"
        aria-label="Fermer le panneau alertes"
        onClick={onClose}
      />
      <aside className="relative w-full max-w-md h-full bg-[#0a1024] border-l border-white/[0.08] shadow-2xl flex flex-col animate-fade-in">
        <div className="px-4 py-4 border-b border-white/[0.06] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bell className="w-4 h-4 text-amber-300" />
            <div>
              <p className="text-sm font-bold text-white">Alertes TradingView</p>
              <p className="text-[10px] text-slate-500">5 dernières · filtres · notifications</p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/[0.06]"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-4 space-y-4 overflow-y-auto flex-1">
          <section className="space-y-2">
            <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
              Notifications
            </p>
            <label className="flex items-center justify-between gap-3 text-xs text-slate-300 bg-white/[0.03] border border-white/[0.06] rounded-xl px-3 py-2.5">
              <span>Son des alertes</span>
              <button
                type="button"
                onClick={onToggleMute}
                className={`text-[11px] font-semibold px-2 py-1 rounded-lg border ${
                  muted
                    ? 'border-slate-600 text-slate-400'
                    : 'border-emerald-500/30 text-emerald-300 bg-emerald-500/10'
                }`}
              >
                {muted ? 'Coupé' : 'Activé'}
              </button>
            </label>
            <label className="flex items-center justify-between gap-3 text-xs text-slate-300 bg-white/[0.03] border border-white/[0.06] rounded-xl px-3 py-2.5">
              <span>Notification navigateur (onglet en fond)</span>
              <button
                type="button"
                onClick={() => onToggleNotif(!notifEnabled)}
                className={`text-[11px] font-semibold px-2 py-1 rounded-lg border ${
                  notifEnabled
                    ? 'border-emerald-500/30 text-emerald-300 bg-emerald-500/10'
                    : 'border-slate-600 text-slate-400'
                }`}
              >
                {notifEnabled ? 'Activé' : 'Désactivé'}
              </button>
            </label>
          </section>

          <section className="space-y-2">
            <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
              Filtrer les signaux
            </p>
            <div className="flex flex-wrap gap-1.5">
              {TV_SIGNAL_OPTIONS.map((opt) => {
                const on = filters.includes(opt.id);
                return (
                  <button
                    key={opt.id}
                    type="button"
                    onClick={() => onToggleFilter(opt.id)}
                    className={`text-[10px] font-semibold px-2.5 py-1 rounded-lg border transition-colors ${
                      on
                        ? 'border-amber-500/40 bg-amber-500/15 text-amber-200'
                        : 'border-white/[0.06] bg-white/[0.02] text-slate-500'
                    }`}
                  >
                    {on && <Check className="w-3 h-3 inline mr-1" />}
                    {opt.label}
                  </button>
                );
              })}
            </div>
          </section>

          <section className="space-y-2">
            <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
              Historique
            </p>
            {history.length === 0 ? (
              <p className="text-xs text-slate-500 py-6 text-center">
                Aucune alerte récente pour ce marché.
              </p>
            ) : (
              <ul className="space-y-2">
                {history.map((a) => (
                  <li key={`${a.id}-${a.received_at}`}>
                    <button
                      type="button"
                      onClick={() => onSelectAlert?.(a)}
                      className="w-full text-left rounded-xl border border-white/[0.06] bg-white/[0.03] hover:bg-white/[0.05] p-3 transition-colors"
                    >
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <span className="text-xs font-bold text-white">
                          {signalLabels[a.signal_type] || a.signal_type}
                        </span>
                        <span className="text-[10px] text-slate-500">
                          {new Date(a.received_at).toLocaleString('fr-FR', {
                            day: 'numeric',
                            month: 'short',
                            hour: '2-digit',
                            minute: '2-digit',
                          })}
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400">
                        {a.ticker || a.market}
                        {a.price != null ? ` · ${a.price.toLocaleString('fr-FR')}` : ''}
                        {a.brief_signal ? ` · Brief ${a.brief_signal}` : ''}
                      </p>
                      {a.brief_summary && (
                        <p className="text-[11px] text-slate-500 mt-1 line-clamp-2">
                          {a.brief_summary}
                        </p>
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      </aside>
    </div>
  );
}
