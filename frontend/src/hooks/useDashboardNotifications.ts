'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/lib/api';
import type { DashboardNotification, LatestTradingViewAlert } from '@/types/api';
import {
  playAlertTone,
  unlockAlertAudio,
} from '@/hooks/useTradingViewAlertPoll';

const MUTE_KEY = 'scpb_tv_alert_mute';
const NOTIF_KEY = 'scpb_tv_browser_notif';

function toPopupAlert(n: DashboardNotification): LatestTradingViewAlert {
  const p = (n.payload || {}) as Record<string, unknown>;
  return {
    id: String(p.id ?? n.id),
    market: n.market,
    signal_type: String(p.signal_type ?? n.kind),
    price: typeof p.price === 'number' ? p.price : null,
    tf: typeof p.tf === 'string' ? p.tf : null,
    ticker: typeof p.ticker === 'string' ? p.ticker : null,
    message: typeof p.message === 'string' ? p.message : n.body,
    trend: typeof p.trend === 'string' ? p.trend : null,
    momentum: typeof p.momentum === 'string' ? p.momentum : null,
    support: typeof p.support === 'number' ? p.support : null,
    resistance: typeof p.resistance === 'number' ? p.resistance : null,
    change_pct: typeof p.change_pct === 'number' ? p.change_pct : null,
    received_at: n.created_at,
    brief_signal: typeof p.brief_signal === 'string' ? p.brief_signal : null,
    brief_summary: typeof p.brief_summary === 'string' ? p.brief_summary : n.body,
  };
}

async function browserNotify(n: DashboardNotification) {
  if (typeof window === 'undefined' || !('Notification' in window)) return;
  if (Notification.permission !== 'granted') return;
  if (document.visibilityState === 'visible') return;
  try {
    const note = new Notification(n.title, {
      body: n.body || '',
      tag: `notif-${n.id}`,
      icon: '/logo.png',
    });
    note.onclick = () => {
      window.focus();
      note.close();
    };
  } catch {
    /* ignore */
  }
}

export function useDashboardNotifications(
  market: string,
  onLive?: (alert: LatestTradingViewAlert) => void,
) {
  const [items, setItems] = useState<DashboardNotification[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [popupAlert, setPopupAlert] = useState<LatestTradingViewAlert | null>(null);
  const [wsStatus, setWsStatus] = useState<'connecting' | 'live' | 'offline'>('connecting');
  const [panelOpen, setPanelOpen] = useState(false);
  const [briefFlash, setBriefFlash] = useState(false);
  const [muted, setMutedState] = useState(false);
  const [notifEnabled, setNotifEnabledState] = useState(false);
  const seenIds = useRef<Set<string>>(new Set());
  const primed = useRef(false);
  const onLiveRef = useRef(onLive);
  onLiveRef.current = onLive;

  const refresh = useCallback(async () => {
    try {
      const res = await api.getNotifications(market, { limit: 40 });
      setItems(res.notifications);
      setUnreadCount(res.unread_count);
      res.notifications.forEach((n) => seenIds.current.add(n.id));
      primed.current = true;
    } catch {
      /* table maybe missing */
    }
  }, [market]);

  const handleIncoming = useCallback(
    async (n: DashboardNotification) => {
      if (seenIds.current.has(n.id)) return;
      seenIds.current.add(n.id);
      setItems((prev) => [n, ...prev.filter((x) => x.id !== n.id)].slice(0, 40));
      setUnreadCount((c) => c + (n.is_read ? 0 : 1));

      const alert = toPopupAlert(n);
      setPopupAlert(alert);
      setBriefFlash(true);
      window.setTimeout(() => setBriefFlash(false), 4000);

      if (sessionStorage.getItem(MUTE_KEY) !== '1') {
        void playAlertTone(alert.signal_type);
      }
      if (localStorage.getItem(NOTIF_KEY) === '1') {
        void browserNotify(n);
      }
      onLiveRef.current?.(alert);
    },
    [],
  );

  useEffect(() => {
    setMutedState(sessionStorage.getItem(MUTE_KEY) === '1');
    setNotifEnabledState(localStorage.getItem(NOTIF_KEY) === '1');
    const arm = () => {
      void unlockAlertAudio();
    };
    window.addEventListener('pointerdown', arm, { passive: true });
    return () => window.removeEventListener('pointerdown', arm);
  }, []);

  useEffect(() => {
    primed.current = false;
    seenIds.current.clear();
    void refresh();
  }, [refresh]);

  // WebSocket live
  useEffect(() => {
    let ws: WebSocket | null = null;
    let closed = false;
    let retry: number | undefined;
    let ping: number | undefined;
    let wsUrl: string | null = null;

    const connect = () => {
      if (closed || !wsUrl) return;
      setWsStatus('connecting');
      ws = new WebSocket(wsUrl);
      ws.onopen = () => {
        setWsStatus('live');
        ping = window.setInterval(() => {
          try {
            ws?.send('ping');
          } catch {
            /* ignore */
          }
        }, 25000);
      };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data as string);
          if (msg?.type === 'notification' && msg.data) {
            void handleIncoming(msg.data as DashboardNotification);
          }
        } catch {
          /* ignore */
        }
      };
      ws.onclose = () => {
        setWsStatus('offline');
        if (ping) window.clearInterval(ping);
        if (!closed) {
          retry = window.setTimeout(connect, 4000);
        }
      };
      ws.onerror = () => {
        try {
          ws?.close();
        } catch {
          /* ignore */
        }
      };
    };

    void (async () => {
      wsUrl = await api.notificationsWsUrl(market);
      if (!wsUrl) {
        setWsStatus('offline');
        return;
      }
      connect();
    })();

    // Fallback poll if WS down
    const poll = window.setInterval(() => {
      if (document.visibilityState === 'visible') void refresh();
    }, 60_000);

    return () => {
      closed = true;
      if (retry) window.clearTimeout(retry);
      if (ping) window.clearInterval(ping);
      window.clearInterval(poll);
      try {
        ws?.close();
      } catch {
        /* ignore */
      }
    };
  }, [market, handleIncoming, refresh]);

  const setMuted = useCallback((v: boolean) => {
    setMutedState(v);
    sessionStorage.setItem(MUTE_KEY, v ? '1' : '0');
    if (!v) void unlockAlertAudio();
  }, []);

  const setNotifEnabled = useCallback(async (v: boolean) => {
    if (v && 'Notification' in window) {
      let perm = Notification.permission;
      if (perm === 'default') perm = await Notification.requestPermission();
      const ok = perm === 'granted';
      localStorage.setItem(NOTIF_KEY, ok ? '1' : '0');
      setNotifEnabledState(ok);
      return;
    }
    localStorage.setItem(NOTIF_KEY, '0');
    setNotifEnabledState(false);
  }, []);

  const markRead = useCallback(async (id: string) => {
    await api.markNotificationRead(id);
    setItems((prev) =>
      prev.map((n) => (n.id === id ? { ...n, is_read: true } : n)),
    );
    setUnreadCount((c) => Math.max(0, c - 1));
  }, []);

  const markAllRead = useCallback(async () => {
    await api.markAllNotificationsRead(market);
    setItems((prev) => prev.map((n) => ({ ...n, is_read: true })));
    setUnreadCount(0);
  }, [market]);

  return {
    items,
    unreadCount,
    popupAlert,
    dismissPopup: () => setPopupAlert(null),
    showAlert: (a: LatestTradingViewAlert) => setPopupAlert(a),
    wsStatus,
    panelOpen,
    openPanel: () => {
      setPanelOpen(true);
    },
    closePanel: () => setPanelOpen(false),
    briefFlash,
    muted,
    setMuted,
    notifEnabled,
    setNotifEnabled,
    markRead,
    markAllRead,
    refresh,
  };
}
