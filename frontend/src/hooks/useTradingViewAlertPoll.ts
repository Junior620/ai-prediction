'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '@/lib/api';
import type { LatestTradingViewAlert } from '@/types/api';

const SEEN_KEY = 'scpb_tv_alert_seen';
const MUTE_KEY = 'scpb_tv_alert_mute';
const FILTER_KEY = 'scpb_tv_alert_filters';
const UNREAD_KEY = 'scpb_tv_alert_unread';
const NOTIF_KEY = 'scpb_tv_browser_notif';
const POLL_MS = 20_000;

export const TV_SIGNAL_OPTIONS = [
  { id: 'buy', label: 'Achat (RSI)' },
  { id: 'sell', label: 'Vente (RSI)' },
  { id: 'support_break', label: 'Cassure support' },
  { id: 'resistance_break', label: 'Cassure résistance' },
  { id: 'trend_change', label: 'Changement tendance' },
  { id: 'custom', label: 'Autre' },
] as const;

export type TvSignalFilter = (typeof TV_SIGNAL_OPTIONS)[number]['id'];

const DEFAULT_FILTERS: TvSignalFilter[] = TV_SIGNAL_OPTIONS.map((o) => o.id);

let sharedAudioCtx: AudioContext | null = null;

function getAudioContext(): AudioContext | null {
  if (typeof window === 'undefined') return null;
  const Ctx =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctx) return null;
  if (!sharedAudioCtx || sharedAudioCtx.state === 'closed') {
    sharedAudioCtx = new Ctx();
  }
  return sharedAudioCtx;
}

export async function unlockAlertAudio(): Promise<boolean> {
  const ctx = getAudioContext();
  if (!ctx) return false;
  try {
    if (ctx.state === 'suspended') await ctx.resume();
    return ctx.state === 'running';
  } catch {
    return false;
  }
}

export async function playAlertTone(signalType: string): Promise<boolean> {
  try {
    const ctx = getAudioContext();
    if (!ctx) return false;
    if (ctx.state === 'suspended') await ctx.resume();
    if (ctx.state !== 'running') return false;

    const now = ctx.currentTime;
    const bearish =
      signalType.includes('sell') ||
      signalType.includes('support') ||
      signalType.includes('bear');
    const bullish =
      signalType.includes('buy') ||
      signalType.includes('resistance') ||
      signalType.includes('bull');

    const freqs = bullish
      ? [587.33, 739.99, 880.0]
      : bearish
        ? [349.23, 293.66, 220.0]
        : [523.25, 659.25];

    freqs.forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'square';
      osc.frequency.value = freq;
      const t0 = now + i * 0.18;
      gain.gain.setValueAtTime(0.0001, t0);
      gain.gain.exponentialRampToValueAtTime(0.18, t0 + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.22);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(t0);
      osc.stop(t0 + 0.25);
    });

    if (typeof navigator !== 'undefined' && 'vibrate' in navigator) {
      try {
        navigator.vibrate(bullish ? [80, 40, 80] : [120, 60, 120, 60, 120]);
      } catch {
        /* ignore */
      }
    }
    return true;
  } catch {
    return false;
  }
}

function fingerprint(alert: LatestTradingViewAlert) {
  return `${alert.id}:${alert.received_at}`;
}

function loadFilters(market: string): TvSignalFilter[] {
  try {
    const raw = localStorage.getItem(`${FILTER_KEY}:${market}`);
    if (!raw) return [...DEFAULT_FILTERS];
    const parsed = JSON.parse(raw) as string[];
    const valid = parsed.filter((x): x is TvSignalFilter =>
      DEFAULT_FILTERS.includes(x as TvSignalFilter),
    );
    return valid.length ? valid : [...DEFAULT_FILTERS];
  } catch {
    return [...DEFAULT_FILTERS];
  }
}

function passesFilter(alert: LatestTradingViewAlert, filters: TvSignalFilter[]) {
  if (!filters.length) return false;
  const st = (alert.signal_type || 'custom').toLowerCase() as TvSignalFilter;
  if (filters.includes(st)) return true;
  // fallback soft match
  return filters.some((f) => st.includes(f) || f.includes(st));
}

async function showBrowserNotification(alert: LatestTradingViewAlert) {
  if (typeof window === 'undefined' || !('Notification' in window)) return;
  if (Notification.permission !== 'granted') return;
  if (document.visibilityState === 'visible') return; // popup suffit si onglet actif

  const title = `Alerte ${alert.signal_type.replace(/_/g, ' ')}`;
  const body = [
    alert.ticker || alert.market,
    alert.price != null ? `Prix ${alert.price}` : null,
    alert.brief_signal ? `Brief ${alert.brief_signal}` : null,
    alert.brief_summary?.slice(0, 120),
  ]
    .filter(Boolean)
    .join(' · ');

  try {
    const n = new Notification(title, {
      body,
      tag: fingerprint(alert),
      icon: '/logo.png',
    });
    n.onclick = () => {
      window.focus();
      n.close();
    };
  } catch {
    /* ignore */
  }
}

export function useTradingViewAlertPoll(
  market: string,
  onNewAlert?: (alert: LatestTradingViewAlert) => void,
) {
  const [history, setHistory] = useState<LatestTradingViewAlert[]>([]);
  const [popupAlert, setPopupAlert] = useState<LatestTradingViewAlert | null>(null);
  const [muted, setMutedState] = useState(false);
  const [filters, setFiltersState] = useState<TvSignalFilter[]>([...DEFAULT_FILTERS]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [panelOpen, setPanelOpen] = useState(false);
  const [notifEnabled, setNotifEnabledState] = useState(false);
  const [briefFlash, setBriefFlash] = useState(false);
  const [audioReady, setAudioReady] = useState(false);

  const seenRef = useRef<string | null>(null);
  const primedRef = useRef(false);
  const onNewAlertRef = useRef(onNewAlert);
  onNewAlertRef.current = onNewAlert;
  const filtersRef = useRef(filters);
  filtersRef.current = filters;

  useEffect(() => {
    if (typeof window === 'undefined') return;
    seenRef.current = sessionStorage.getItem(`${SEEN_KEY}:${market}`);
    setMutedState(sessionStorage.getItem(MUTE_KEY) === '1');
    setFiltersState(loadFilters(market));
    setUnreadCount(Number(sessionStorage.getItem(`${UNREAD_KEY}:${market}`) || '0') || 0);
    setNotifEnabledState(localStorage.getItem(NOTIF_KEY) === '1');

    const arm = () => {
      void unlockAlertAudio().then((ok) => {
        if (ok) setAudioReady(true);
      });
    };
    window.addEventListener('pointerdown', arm, { passive: true });
    window.addEventListener('keydown', arm, { passive: true });
    return () => {
      window.removeEventListener('pointerdown', arm);
      window.removeEventListener('keydown', arm);
    };
  }, [market]);

  const setMuted = useCallback((value: boolean) => {
    setMutedState(value);
    sessionStorage.setItem(MUTE_KEY, value ? '1' : '0');
    if (!value) {
      void unlockAlertAudio().then((ok) => {
        if (ok) setAudioReady(true);
      });
    }
  }, []);

  const setFilters = useCallback(
    (next: TvSignalFilter[]) => {
      const value = next.length ? next : [...DEFAULT_FILTERS];
      setFiltersState(value);
      localStorage.setItem(`${FILTER_KEY}:${market}`, JSON.stringify(value));
    },
    [market],
  );

  const toggleFilter = useCallback(
    (id: TvSignalFilter) => {
      setFilters(
        filtersRef.current.includes(id)
          ? filtersRef.current.filter((x) => x !== id)
          : [...filtersRef.current, id],
      );
    },
    [setFilters],
  );

  const clearUnread = useCallback(() => {
    setUnreadCount(0);
    sessionStorage.setItem(`${UNREAD_KEY}:${market}`, '0');
  }, [market]);

  const bumpUnread = useCallback(() => {
    setUnreadCount((c) => {
      const next = c + 1;
      sessionStorage.setItem(`${UNREAD_KEY}:${market}`, String(next));
      return next;
    });
  }, [market]);

  const dismissPopup = useCallback(() => {
    setPopupAlert(null);
  }, []);

  const showAlert = useCallback((alert: LatestTradingViewAlert) => {
    setPopupAlert(alert);
  }, []);

  const openPanel = useCallback(() => {
    setPanelOpen(true);
    clearUnread();
  }, [clearUnread]);

  const closePanel = useCallback(() => setPanelOpen(false), []);

  const enableBrowserNotifications = useCallback(async () => {
    if (!('Notification' in window)) return false;
    let perm = Notification.permission;
    if (perm === 'default') {
      perm = await Notification.requestPermission();
    }
    const ok = perm === 'granted';
    localStorage.setItem(NOTIF_KEY, ok ? '1' : '0');
    setNotifEnabledState(ok);
    return ok;
  }, []);

  const setNotifEnabled = useCallback(
    async (value: boolean) => {
      if (value) {
        await enableBrowserNotifications();
      } else {
        localStorage.setItem(NOTIF_KEY, '0');
        setNotifEnabledState(false);
      }
    },
    [enableBrowserNotifications],
  );

  const poll = useCallback(async () => {
    const recent = await api.getRecentTradingViewAlerts(market, 5);
    if (recent.length) setHistory(recent);

    const latest = recent[0] || (await api.getLatestTradingViewAlert(market));
    if (!latest?.id) return;

    const fp = fingerprint(latest);

    if (!primedRef.current) {
      primedRef.current = true;
      seenRef.current = fp;
      sessionStorage.setItem(`${SEEN_KEY}:${market}`, fp);
      return;
    }

    if (fp === seenRef.current) return;
    seenRef.current = fp;
    sessionStorage.setItem(`${SEEN_KEY}:${market}`, fp);

    if (!passesFilter(latest, filtersRef.current)) return;

    setPopupAlert(latest);
    bumpUnread();
    setBriefFlash(true);
    window.setTimeout(() => setBriefFlash(false), 4000);

    if (sessionStorage.getItem(MUTE_KEY) !== '1') {
      void playAlertTone(latest.signal_type || '');
    }
    if (localStorage.getItem(NOTIF_KEY) === '1') {
      void showBrowserNotification(latest);
    }
    onNewAlertRef.current?.(latest);
  }, [market, bumpUnread]);

  useEffect(() => {
    primedRef.current = false;
    void poll();
    const id = window.setInterval(() => {
      void poll();
    }, POLL_MS);
    return () => window.clearInterval(id);
  }, [poll]);

  const latestAlert = useMemo(() => history[0] ?? null, [history]);

  return {
    latestAlert,
    history,
    popupAlert,
    dismissPopup,
    showAlert,
    muted,
    setMuted,
    filters,
    setFilters,
    toggleFilter,
    unreadCount,
    clearUnread,
    panelOpen,
    openPanel,
    closePanel,
    notifEnabled,
    setNotifEnabled,
    briefFlash,
    audioReady,
    pollNow: poll,
  };
}
