'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { api } from '@/lib/api';
import type { DashboardNotification } from '@/types/api';
import { useDashboardNotifications } from '@/hooks/useDashboardNotifications';
import { TradingViewAlertPopup } from '@/components/TradingViewAlertPopup';
import {
  ArrowLeft, Bell, CheckCheck, Filter, RefreshCw, Wifi, WifiOff,
} from 'lucide-react';

const MARKETS = [
  { id: 'ICE_NY', label: 'Cacao', hrefDash: '/' },
  { id: 'COFFEE_ROBUSTA', label: 'Café Robusta', hrefDash: '/coffee' },
] as const;

type MarketId = (typeof MARKETS)[number]['id'];

function kindLabel(kind: string) {
  const map: Record<string, string> = {
    buy: 'Achat',
    sell: 'Vente',
    support_break: 'Cassure support',
    resistance_break: 'Cassure résistance',
    trend_change: 'Tendance',
    custom: 'Autre',
  };
  return map[kind] || kind.replace(/_/g, ' ');
}

export default function NotificationsPageClient() {
  const search = useSearchParams();
  const initial = (search.get('market') || 'ICE_NY').toUpperCase();
  const [market, setMarket] = useState<MarketId>(
    initial === 'COFFEE_ROBUSTA' ? 'COFFEE_ROBUSTA' : 'ICE_NY',
  );
  const [unreadOnly, setUnreadOnly] = useState(false);
  const [kindFilter, setKindFilter] = useState<string>('all');
  const [extra, setExtra] = useState<DashboardNotification[]>([]);
  const [loadingMore, setLoadingMore] = useState(false);

  const onLive = useCallback(() => {}, []);

  const {
    items,
    unreadCount,
    popupAlert,
    dismissPopup,
    showAlert,
    wsStatus,
    muted,
    setMuted,
    markRead,
    markAllRead,
    refresh,
  } = useDashboardNotifications(market, onLive);

  useEffect(() => {
    setExtra([]);
  }, [market]);

  const loadFullHistory = useCallback(async () => {
    setLoadingMore(true);
    try {
      const res = await api.getNotifications(market, { limit: 100, unread_only: unreadOnly });
      setExtra(res.notifications);
    } catch {
      /* ignore */
    } finally {
      setLoadingMore(false);
    }
  }, [market, unreadOnly]);

  useEffect(() => {
    void loadFullHistory();
  }, [loadFullHistory]);

  const merged = useMemo(() => {
    const map = new Map<string, DashboardNotification>();
    [...items, ...extra].forEach((n) => map.set(n.id, n));
    let list = Array.from(map.values()).sort(
      (a, b) => +new Date(b.created_at) - +new Date(a.created_at),
    );
    if (unreadOnly) list = list.filter((n) => !n.is_read);
    if (kindFilter !== 'all') list = list.filter((n) => n.kind === kindFilter);
    return list;
  }, [items, extra, unreadOnly, kindFilter]);

  const kinds = useMemo(() => {
    const s = new Set(merged.map((n) => n.kind));
    return Array.from(s);
  }, [merged]);

  const dashHref = MARKETS.find((m) => m.id === market)?.hrefDash || '/';

  return (
    <div className="min-h-screen bg-[#06091a] bg-grid">
      <header className="sticky top-0 z-50 glass-card !rounded-none border-x-0 border-t-0">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <Link href={dashHref} className="shrink-0 rounded-xl overflow-hidden ring-1 ring-white/10 bg-black/40">
              <Image src="/logo.png" alt="SCPB" width={40} height={40} className="w-10 h-10 object-contain" />
            </Link>
            <div className="min-w-0">
              <h1 className="text-lg font-bold text-white flex items-center gap-2">
                <Bell className="w-4 h-4 text-amber-300" />
                Notifications
                {unreadCount > 0 && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-rose-500 text-white font-bold">
                    {unreadCount}
                  </span>
                )}
              </h1>
              <p className="text-[11px] text-slate-500 flex items-center gap-1.5">
                {wsStatus === 'live' ? (
                  <Wifi className="w-3 h-3 text-emerald-400" />
                ) : (
                  <WifiOff className="w-3 h-3 text-slate-500" />
                )}
                Historique · {wsStatus === 'live' ? 'Temps réel' : wsStatus === 'connecting' ? 'Connexion…' : 'Hors ligne'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Link
              href={dashHref}
              className="hidden sm:inline-flex items-center gap-1.5 text-xs text-slate-300 px-3 py-2 rounded-xl border border-white/[0.08] hover:bg-white/[0.05]"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              Dashboard
            </Link>
            <button
              type="button"
              onClick={() => {
                void refresh();
                void loadFullHistory();
              }}
              className="p-2 rounded-xl border border-white/[0.08] text-slate-300 hover:bg-white/[0.05]"
              title="Actualiser"
            >
              <RefreshCw className={`w-4 h-4 ${loadingMore ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8 space-y-6">
        <div className="flex flex-wrap gap-2 items-center">
          {MARKETS.map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => setMarket(m.id)}
              className={`text-xs font-semibold px-3 py-1.5 rounded-lg border transition-colors ${
                market === m.id
                  ? 'border-amber-500/40 bg-amber-500/15 text-amber-200'
                  : 'border-white/[0.08] text-slate-400 hover:text-white'
              }`}
            >
              {m.label}
            </button>
          ))}
          <span className="text-slate-700">|</span>
          <button
            type="button"
            onClick={() => setUnreadOnly((v) => !v)}
            className={`text-xs font-semibold px-3 py-1.5 rounded-lg border inline-flex items-center gap-1.5 ${
              unreadOnly
                ? 'border-rose-500/40 bg-rose-500/15 text-rose-200'
                : 'border-white/[0.08] text-slate-400'
            }`}
          >
            <Filter className="w-3 h-3" />
            Non lues
          </button>
          <button
            type="button"
            onClick={() => void markAllRead()}
            className="text-xs font-semibold px-3 py-1.5 rounded-lg border border-white/[0.08] text-slate-300 hover:bg-white/[0.05] inline-flex items-center gap-1.5 ml-auto"
          >
            <CheckCheck className="w-3.5 h-3.5" />
            Tout marquer lu
          </button>
        </div>

        {kinds.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            <button
              type="button"
              onClick={() => setKindFilter('all')}
              className={`text-[10px] font-semibold px-2.5 py-1 rounded-lg border ${
                kindFilter === 'all'
                  ? 'border-amber-500/40 bg-amber-500/15 text-amber-200'
                  : 'border-white/[0.06] text-slate-500'
              }`}
            >
              Tous
            </button>
            {kinds.map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => setKindFilter(k)}
                className={`text-[10px] font-semibold px-2.5 py-1 rounded-lg border ${
                  kindFilter === k
                    ? 'border-amber-500/40 bg-amber-500/15 text-amber-200'
                    : 'border-white/[0.06] text-slate-500'
                }`}
              >
                {kindLabel(k)}
              </button>
            ))}
          </div>
        )}

        <div className="space-y-3">
          {merged.length === 0 ? (
            <div className="glass-card p-10 text-center">
              <Bell className="w-8 h-8 text-slate-600 mx-auto mb-3" />
              <p className="text-sm text-slate-400">Aucune notification pour ce filtre.</p>
              <p className="text-xs text-slate-600 mt-1">
                Les alertes TradingView apparaîtront ici automatiquement.
              </p>
            </div>
          ) : (
            merged.map((n) => (
              <button
                key={n.id}
                type="button"
                onClick={() => {
                  if (!n.is_read) void markRead(n.id);
                  const p = (n.payload || {}) as Record<string, unknown>;
                  showAlert({
                    id: String(p.id ?? n.id),
                    market: n.market,
                    signal_type: String(p.signal_type ?? n.kind),
                    price: typeof p.price === 'number' ? p.price : null,
                    tf: typeof p.tf === 'string' ? p.tf : null,
                    ticker: typeof p.ticker === 'string' ? p.ticker : null,
                    message: n.body,
                    trend: typeof p.trend === 'string' ? p.trend : null,
                    momentum: typeof p.momentum === 'string' ? p.momentum : null,
                    support: typeof p.support === 'number' ? p.support : null,
                    resistance: typeof p.resistance === 'number' ? p.resistance : null,
                    change_pct: typeof p.change_pct === 'number' ? p.change_pct : null,
                    received_at: n.created_at,
                    brief_signal: typeof p.brief_signal === 'string' ? p.brief_signal : null,
                    brief_summary: typeof p.brief_summary === 'string' ? p.brief_summary : n.body,
                  });
                }}
                className={`w-full text-left glass-card p-4 transition-all hover:bg-white/[0.03] ${
                  n.is_read ? '' : 'ring-1 ring-amber-500/30'
                }`}
              >
                <div className="flex items-start justify-between gap-3 mb-2">
                  <div className="min-w-0">
                    <p className="text-sm font-bold text-white flex items-center gap-2">
                      {!n.is_read && <span className="w-2 h-2 rounded-full bg-amber-400 shrink-0" />}
                      {n.title}
                    </p>
                    <p className="text-[11px] text-slate-500 mt-0.5">
                      {kindLabel(n.kind)} · {n.source} ·{' '}
                      {new Date(n.created_at).toLocaleString('fr-FR', {
                        weekday: 'short',
                        day: 'numeric',
                        month: 'short',
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </p>
                  </div>
                  {typeof (n.payload as { brief_signal?: string } | undefined)?.brief_signal === 'string' && (
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded border shrink-0 ${
                        (n.payload as { brief_signal: string }).brief_signal === 'BUY'
                          ? 'text-emerald-300 border-emerald-500/30 bg-emerald-500/10'
                          : (n.payload as { brief_signal: string }).brief_signal === 'SELL'
                            ? 'text-rose-300 border-rose-500/30 bg-rose-500/10'
                            : 'text-amber-300 border-amber-500/30 bg-amber-500/10'
                      }`}
                    >
                      {(n.payload as { brief_signal: string }).brief_signal}
                    </span>
                  )}
                </div>
                {n.body && (
                  <p className="text-xs text-slate-400 leading-relaxed line-clamp-3">{n.body}</p>
                )}
              </button>
            ))
          )}
        </div>
      </main>

      <TradingViewAlertPopup
        alert={popupAlert}
        onDismiss={dismissPopup}
        muted={muted}
        onToggleMute={() => setMuted(!muted)}
      />
    </div>
  );
}
