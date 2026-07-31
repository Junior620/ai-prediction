'use client';

import Link from 'next/link';
import type { DashboardNotification } from '@/types/api';
import { Bell, CheckCheck, Wifi, WifiOff, X, ExternalLink } from 'lucide-react';

interface NotificationsPanelProps {
  open: boolean;
  onClose: () => void;
  items: DashboardNotification[];
  unreadCount: number;
  wsStatus: 'connecting' | 'live' | 'offline';
  muted: boolean;
  onToggleMute: () => void;
  notifEnabled: boolean;
  onToggleNotif: (v: boolean) => void;
  onMarkRead: (id: string) => void;
  onMarkAllRead: () => void;
  onSelect?: (n: DashboardNotification) => void;
  historyHref?: string;
}

export function NotificationsPanel({
  open,
  onClose,
  items,
  unreadCount,
  wsStatus,
  muted,
  onToggleMute,
  notifEnabled,
  onToggleNotif,
  onMarkRead,
  onMarkAllRead,
  onSelect,
  historyHref,
}: NotificationsPanelProps) {
  if (!open) return null;

  const statusLabel =
    wsStatus === 'live' ? 'Temps réel' : wsStatus === 'connecting' ? 'Connexion…' : 'Hors ligne';

  return (
    <div className="fixed inset-0 z-[70] flex justify-end">
      <button type="button" className="absolute inset-0 bg-black/50" aria-label="Fermer" onClick={onClose} />
      <aside className="relative w-full max-w-md h-full bg-[#0a1024] border-l border-white/[0.08] shadow-2xl flex flex-col">
        <div className="px-4 py-4 border-b border-white/[0.06] flex items-start justify-between gap-3">
          <div>
            <p className="text-sm font-bold text-white flex items-center gap-2">
              <Bell className="w-4 h-4 text-amber-300" />
              Notifications
              {unreadCount > 0 && (
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-rose-500/90 text-white font-bold">
                  {unreadCount}
                </span>
              )}
            </p>
            <p className="text-[10px] text-slate-500 mt-1 flex items-center gap-1.5">
              {wsStatus === 'live' ? (
                <Wifi className="w-3 h-3 text-emerald-400" />
              ) : (
                <WifiOff className="w-3 h-3 text-slate-500" />
              )}
              {statusLabel} · WebSocket
            </p>
          </div>
          <button type="button" onClick={onClose} className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/[0.06]">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-4 space-y-3 border-b border-white/[0.06]">
          <div className="flex gap-2">
            <button
              type="button"
              onClick={onToggleMute}
              className={`flex-1 text-[11px] font-semibold py-2 rounded-lg border ${
                muted ? 'border-slate-600 text-slate-400' : 'border-emerald-500/30 text-emerald-300 bg-emerald-500/10'
              }`}
            >
              Son {muted ? 'coupé' : 'activé'}
            </button>
            <button
              type="button"
              onClick={() => onToggleNotif(!notifEnabled)}
              className={`flex-1 text-[11px] font-semibold py-2 rounded-lg border ${
                notifEnabled ? 'border-emerald-500/30 text-emerald-300 bg-emerald-500/10' : 'border-slate-600 text-slate-400'
              }`}
            >
              Navigateur {notifEnabled ? 'ON' : 'OFF'}
            </button>
          </div>
          <button
            type="button"
            onClick={onMarkAllRead}
            className="w-full text-[11px] font-semibold py-2 rounded-lg border border-white/[0.08] text-slate-300 hover:bg-white/[0.04] flex items-center justify-center gap-1.5"
          >
            <CheckCheck className="w-3.5 h-3.5" />
            Tout marquer comme lu
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {items.length === 0 ? (
            <p className="text-xs text-slate-500 text-center py-10">
              Aucune notification pour l’instant.
              <br />
              Elles apparaîtront ici dès une alerte TradingView.
            </p>
          ) : (
            items.map((n) => (
              <button
                key={n.id}
                type="button"
                onClick={() => {
                  if (!n.is_read) onMarkRead(n.id);
                  onSelect?.(n);
                }}
                className={`w-full text-left rounded-xl border p-3 transition-colors ${
                  n.is_read
                    ? 'border-white/[0.05] bg-white/[0.02]'
                    : 'border-amber-500/25 bg-amber-500/10'
                }`}
              >
                <div className="flex items-start justify-between gap-2 mb-1">
                  <p className="text-xs font-bold text-white">{n.title}</p>
                  {!n.is_read && <span className="w-2 h-2 rounded-full bg-amber-400 shrink-0 mt-1" />}
                </div>
                {n.body && (
                  <p className="text-[11px] text-slate-400 line-clamp-3 mb-1.5">{n.body}</p>
                )}
                <p className="text-[10px] text-slate-600">
                  {n.source} · {n.kind.replace(/_/g, ' ')} ·{' '}
                  {new Date(n.created_at).toLocaleString('fr-FR', {
                    day: 'numeric',
                    month: 'short',
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </p>
              </button>
            ))
          )}
        </div>

        {historyHref && (
          <div className="p-4 border-t border-white/[0.06]">
            <Link
              href={historyHref}
              onClick={onClose}
              className="w-full flex items-center justify-center gap-1.5 text-xs font-semibold py-2.5 rounded-xl border border-amber-500/30 bg-amber-500/10 text-amber-200 hover:bg-amber-500/20"
            >
              Voir tout l’historique
              <ExternalLink className="w-3.5 h-3.5" />
            </Link>
          </div>
        )}
      </aside>
    </div>
  );
}
