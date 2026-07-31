'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/lib/api';
import type { LatestTradingViewAlert } from '@/types/api';

const SEEN_KEY = 'scpb_tv_alert_seen';
const MUTE_KEY = 'scpb_tv_alert_mute';
const POLL_MS = 20_000;

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

/** Débloque l’audio après un geste utilisateur (requis par Chrome/Safari). */
export async function unlockAlertAudio(): Promise<boolean> {
  const ctx = getAudioContext();
  if (!ctx) return false;
  try {
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
    return ctx.state === 'running';
  } catch {
    return false;
  }
}

export async function playAlertTone(signalType: string): Promise<boolean> {
  try {
    const ctx = getAudioContext();
    if (!ctx) return false;
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
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

    // Séquence plus audible (square + gain plus fort)
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

export function useTradingViewAlertPoll(
  market: string,
  onNewAlert?: (alert: LatestTradingViewAlert) => void,
) {
  const [alert, setAlert] = useState<LatestTradingViewAlert | null>(null);
  const [popupAlert, setPopupAlert] = useState<LatestTradingViewAlert | null>(null);
  const [muted, setMutedState] = useState(false);
  const [audioReady, setAudioReady] = useState(false);
  const seenRef = useRef<string | null>(null);
  const primedRef = useRef(false);
  const onNewAlertRef = useRef(onNewAlert);
  onNewAlertRef.current = onNewAlert;

  useEffect(() => {
    if (typeof window === 'undefined') return;
    seenRef.current = sessionStorage.getItem(`${SEEN_KEY}:${market}`);
    setMutedState(sessionStorage.getItem(MUTE_KEY) === '1');

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
    if (typeof window !== 'undefined') {
      sessionStorage.setItem(MUTE_KEY, value ? '1' : '0');
    }
    if (!value) {
      void unlockAlertAudio().then((ok) => {
        if (ok) setAudioReady(true);
      });
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
      void playAlertTone(latest.signal_type || '');
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
    audioReady,
    pollNow: poll,
  };
}
