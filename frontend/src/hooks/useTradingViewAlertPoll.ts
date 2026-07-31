'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/lib/api';
import type { LatestTradingViewAlert } from '@/types/api';

const SEEN_KEY = 'scpb_tv_alert_seen';
const MUTE_KEY = 'scpb_tv_alert_mute';
const POLL_MS = 20_000;

function playAlertTone(signalType: string) {
  try {
    const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (!Ctx) return;
    const ctx = new Ctx();
    const now = ctx.currentTime;
    const freqs =
      signalType.includes('buy') || signalType.includes('resistance')
        ? [523.25, 659.25, 783.99]
        : signalType.includes('sell') || signalType.includes('support')
          ? [392.0, 311.13, 261.63]
          : [440, 554.37];

    freqs.forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.exponentialRampToValueAtTime(0.12, now + 0.03 + i * 0.12);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.28 + i * 0.12);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now + i * 0.12);
      osc.stop(now + 0.35 + i * 0.12);
    });
    window.setTimeout(() => ctx.close().catch(() => undefined), 1200);
  } catch {
    // navigateurs bloquant l'audio sans geste utilisateur
  }
}

export function useTradingViewAlertPoll(
  market: string,
  onNewAlert?: (alert: LatestTradingViewAlert) => void,
) {
  const [alert, setAlert] = useState<LatestTradingViewAlert | null>(null);
  const [popupAlert, setPopupAlert] = useState<LatestTradingViewAlert | null>(null);
  const [muted, setMutedState] = useState(false);
  const seenRef = useRef<string | null>(null);
  const primedRef = useRef(false);
  const onNewAlertRef = useRef(onNewAlert);
  onNewAlertRef.current = onNewAlert;

  useEffect(() => {
    if (typeof window === 'undefined') return;
    seenRef.current = sessionStorage.getItem(`${SEEN_KEY}:${market}`);
    setMutedState(sessionStorage.getItem(MUTE_KEY) === '1');
  }, [market]);

  const setMuted = useCallback((value: boolean) => {
    setMutedState(value);
    if (typeof window !== 'undefined') {
      sessionStorage.setItem(MUTE_KEY, value ? '1' : '0');
    }
  }, []);

  const dismissPopup = useCallback(() => {
    setPopupAlert(null);
  }, []);

  const poll = useCallback(async () => {
    const latest = await api.getLatestTradingViewAlert(market);
    if (!latest?.id) return;

    setAlert(latest);
    const fingerprint = `${latest.id}:${latest.received_at}`;

    // Premier poll: mémorise sans popup (évite alerte historique au chargement)
    if (!primedRef.current) {
      primedRef.current = true;
      seenRef.current = fingerprint;
      if (typeof window !== 'undefined') {
        sessionStorage.setItem(`${SEEN_KEY}:${market}`, fingerprint);
      }
      return;
    }

    if (fingerprint === seenRef.current) return;

    seenRef.current = fingerprint;
    if (typeof window !== 'undefined') {
      sessionStorage.setItem(`${SEEN_KEY}:${market}`, fingerprint);
    }

    setPopupAlert(latest);
    if (sessionStorage.getItem(MUTE_KEY) !== '1') {
      playAlertTone(latest.signal_type || '');
    }
    onNewAlertRef.current?.(latest);
  }, [market]);

  useEffect(() => {
    primedRef.current = false;
    void poll();
    const id = window.setInterval(() => {
      void poll();
    }, POLL_MS);
    return () => window.clearInterval(id);
  }, [poll]);

  return {
    latestAlert: alert,
    popupAlert,
    dismissPopup,
    muted,
    setMuted,
    pollNow: poll,
  };
}
