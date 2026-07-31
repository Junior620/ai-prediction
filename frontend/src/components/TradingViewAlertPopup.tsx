'use client';

import { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import type { LatestTradingViewAlert } from '@/types/api';
import { playAlertTone, unlockAlertAudio } from '@/hooks/useTradingViewAlertPoll';
import {
  BellRing, TrendingDown, TrendingUp, Activity, X, Volume2, VolumeX,
} from 'lucide-react';

const signalLabels: Record<string, string> = {
  buy: 'Signal d’achat',
  sell: 'Signal de vente',
  support_break: 'Cassure de support',
  resistance_break: 'Cassure de résistance',
  trend_change: 'Changement de tendance',
  custom: 'Alerte marché',
};

function labelFor(signalType: string) {
  return signalLabels[signalType] || signalType.replace(/_/g, ' ');
}

function IconFor({ signalType }: { signalType: string }) {
  if (signalType.includes('buy') || signalType.includes('resistance')) {
    return <TrendingUp className="w-6 h-6 text-emerald-400" />;
  }
  if (signalType.includes('sell') || signalType.includes('support')) {
    return <TrendingDown className="w-6 h-6 text-rose-400" />;
  }
  return <Activity className="w-6 h-6 text-amber-400" />;
}

interface TradingViewAlertPopupProps {
  alert: LatestTradingViewAlert | null;
  onDismiss: () => void;
  muted: boolean;
  onToggleMute: () => void;
}

export function TradingViewAlertPopup({
  alert,
  onDismiss,
  muted,
  onToggleMute,
}: TradingViewAlertPopupProps) {
  const [soundHint, setSoundHint] = useState(false);

  useEffect(() => {
    if (!alert || muted) {
      setSoundHint(false);
      return;
    }
    let cancelled = false;
    void (async () => {
      const ok = await playAlertTone(alert.signal_type || '');
      if (!cancelled && !ok) setSoundHint(true);
    })();
    return () => {
      cancelled = true;
    };
  }, [alert, muted]);

  const enableSound = async () => {
    await unlockAlertAudio();
    if (alert) {
      const ok = await playAlertTone(alert.signal_type || '');
      if (ok) setSoundHint(false);
    }
  };

  const briefTone = alert?.brief_signal;
  const border =
    briefTone === 'BUY' ? 'border-emerald-500/40' :
    briefTone === 'SELL' ? 'border-rose-500/40' :
    'border-amber-500/40';

  return (
    <AnimatePresence>
      {alert && (
        <motion.div
          key={alert.id + alert.received_at}
          initial={{ opacity: 0, y: -24, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -12, scale: 0.98 }}
          transition={{ type: 'spring', stiffness: 380, damping: 28 }}
          className="fixed top-20 right-4 z-[80] w-[min(100vw-2rem,380px)]"
          role="alertdialog"
          aria-live="assertive"
        >
          <div className={`rounded-2xl border ${border} bg-[#0c1228]/95 backdrop-blur-xl shadow-2xl shadow-black/50 overflow-hidden`}>
            <div className="px-4 py-3 flex items-start gap-3 border-b border-white/[0.06] bg-gradient-to-r from-amber-500/10 to-transparent">
              <div className="mt-0.5 p-2 rounded-xl bg-white/[0.06]">
                <BellRing className="w-4 h-4 text-amber-300 animate-pulse" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-[10px] font-bold uppercase tracking-widest text-amber-300/90">
                  Alerte TradingView
                </p>
                <p className="text-sm font-bold text-white truncate">
                  {labelFor(alert.signal_type)}
                </p>
              </div>
              <div className="flex items-center gap-1 shrink-0">
                <button
                  type="button"
                  onClick={onToggleMute}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/[0.06]"
                  title={muted ? 'Activer le son' : 'Couper le son'}
                >
                  {muted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                </button>
                <button
                  type="button"
                  onClick={onDismiss}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/[0.06]"
                  title="Fermer"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="p-4 space-y-3">
              <div className="flex items-center gap-3">
                <IconFor signalType={alert.signal_type} />
                <div className="min-w-0">
                  <p className="text-xs text-slate-400">
                    {alert.ticker || 'ICEEUR:C1!'}
                    {alert.tf ? ` · ${alert.tf}` : ''}
                  </p>
                  {alert.price != null && (
                    <p className="text-lg font-black text-white font-mono-price">
                      {alert.price.toLocaleString('fr-FR', { maximumFractionDigits: 2 })}
                      {alert.change_pct != null && (
                        <span className={`ml-2 text-sm font-semibold ${alert.change_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {alert.change_pct >= 0 ? '+' : ''}
                          {alert.change_pct.toFixed(2)}%
                        </span>
                      )}
                    </p>
                  )}
                </div>
              </div>

              {(alert.trend || alert.momentum) && (
                <div className="flex flex-wrap gap-1.5 text-[10px]">
                  {alert.trend && (
                    <span className="px-2 py-0.5 rounded border border-white/[0.08] bg-white/[0.04] text-slate-300 capitalize">
                      Tendance {alert.trend}
                    </span>
                  )}
                  {alert.momentum && (
                    <span className="px-2 py-0.5 rounded border border-white/[0.08] bg-white/[0.04] text-slate-300 capitalize">
                      Momentum {alert.momentum.replace(/_/g, ' ')}
                    </span>
                  )}
                  {alert.brief_signal && (
                    <span className={`px-2 py-0.5 rounded border font-bold ${
                      alert.brief_signal === 'BUY' ? 'text-emerald-300 border-emerald-500/30 bg-emerald-500/10' :
                      alert.brief_signal === 'SELL' ? 'text-rose-300 border-rose-500/30 bg-rose-500/10' :
                      'text-amber-300 border-amber-500/30 bg-amber-500/10'
                    }`}>
                      Brief {alert.brief_signal}
                    </span>
                  )}
                </div>
              )}

              {alert.brief_summary && (
                <p className="text-xs text-slate-300 leading-relaxed line-clamp-4">
                  {alert.brief_summary}
                </p>
              )}

              {(alert.support != null || alert.resistance != null) && (
                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  {alert.support != null && (
                    <div className="rounded-lg bg-white/[0.03] p-2">
                      <p className="text-slate-500">Support</p>
                      <p className="font-mono-price text-emerald-400 font-semibold">
                        {alert.support.toLocaleString('fr-FR')}
                      </p>
                    </div>
                  )}
                  {alert.resistance != null && (
                    <div className="rounded-lg bg-white/[0.03] p-2">
                      <p className="text-slate-500">Résistance</p>
                      <p className="font-mono-price text-rose-400 font-semibold">
                        {alert.resistance.toLocaleString('fr-FR')}
                      </p>
                    </div>
                  )}
                </div>
              )}

              {soundHint && !muted && (
                <button
                  type="button"
                  onClick={() => void enableSound()}
                  className="w-full text-xs font-semibold py-2 rounded-lg border border-amber-500/30 bg-amber-500/10 text-amber-200 hover:bg-amber-500/20 transition-colors"
                >
                  Cliquer pour activer le son des alertes
                </button>
              )}

              <div className="flex items-center justify-between gap-2">
                <p className="text-[10px] text-slate-600">
                  Reçue {new Date(alert.received_at).toLocaleString('fr-FR', {
                    day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit',
                  })}
                </p>
                <button
                  type="button"
                  onClick={onDismiss}
                  className="text-[11px] font-semibold text-slate-300 hover:text-white px-2.5 py-1 rounded-lg border border-white/[0.08] hover:bg-white/[0.06]"
                >
                  Fermer
                </button>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
