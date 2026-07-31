'use client';

import { useState, useEffect, useCallback } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { api } from '@/lib/api';
import type {
  MarketIntelligenceResponse,
  PredictionResponse,
  ValidationMetricsResponse,
} from '@/types/api';
import { TradingViewEmbed } from '@/components/TradingViewEmbed';
import { MarketBrief } from '@/components/MarketBrief';
import { MarketKPIBar } from '@/components/dashboard/MarketKPIBar';
import { PredictionHorizonCard } from '@/components/dashboard/PredictionHorizonCard';
import { ForecastChart } from '@/components/dashboard/ForecastChart';
import { AnalysisPanel } from '@/components/dashboard/AnalysisPanel';
import { TradingViewAlertPopup } from '@/components/TradingViewAlertPopup';
import { useTradingViewAlertPoll } from '@/hooks/useTradingViewAlertPoll';
import {
  RefreshCw, AlertTriangle, BarChart3, Coffee, Bell,
} from 'lucide-react';

export interface MarketDashboardConfig {
  market: string;
  tradingViewDisplayName?: string;
  title: string;
  subtitle: string;
  accent: 'amber' | 'emerald';
  includeSentiment: boolean;
  unitLabel: string;
  priceSource: string;
  activeNav: 'cacao' | 'coffee';
  otherMarket: { href: string; label: string };
}

const TV_FALLBACKS: Record<string, {
  embedSymbol: string;
  chartSymbol: string;
  displayName: string;
  embedLabel: string;
}> = {
  ICE_NY: {
    embedSymbol: 'PEPPERSTONE:COCOA',
    chartSymbol: 'PEPPERSTONE:COCOA',
    displayName: 'Cocoa Cash Contract',
    embedLabel: 'Pepperstone CFD · USD/tonne',
  },
  COFFEE_ROBUSTA: {
    embedSymbol: 'ROBCOFFEE',
    chartSymbol: 'ICEEUR:RC1!',
    displayName: 'Robusta Coffee',
    embedLabel: 'CMC Markets CFD (embed) · ICE RC1! (futures)',
  },
};

const themes = {
  amber: {
    gradient: 'from-amber-400 to-orange-600',
    shadow: 'shadow-amber-500/20',
    chartStroke: '#f59e0b',
    chartGradient: 'gPriceAmber',
    navActive: 'bg-amber-500/15 text-amber-300',
    accentClass: 'text-amber-400',
    Icon: BarChart3,
  },
  emerald: {
    gradient: 'from-emerald-400 to-teal-600',
    shadow: 'shadow-emerald-500/20',
    chartStroke: '#10b981',
    chartGradient: 'gPriceEmerald',
    navActive: 'bg-emerald-500/15 text-emerald-300',
    accentClass: 'text-emerald-400',
    Icon: Coffee,
  },
};

export function SimplifiedMarketDashboard({ config }: { config: MarketDashboardConfig }) {
  const theme = themes[config.accent];
  const [data, setData] = useState<PredictionResponse | null>(null);
  const [intelligence, setIntelligence] = useState<MarketIntelligenceResponse | null>(null);
  const [validation, setValidation] = useState<ValidationMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [briefLoading, setBriefLoading] = useState(true);
  const [advancedLoading, setAdvancedLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [briefError, setBriefError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fallback = TV_FALLBACKS[config.market];
  const [tvEmbedSymbol, setTvEmbedSymbol] = useState<string | null>(fallback?.embedSymbol ?? null);
  const [tvChartSymbol, setTvChartSymbol] = useState<string | null>(fallback?.chartSymbol ?? null);
  const [tvEmbedLabel, setTvEmbedLabel] = useState<string | undefined>(fallback?.embedLabel);
  const [tvDisplayName, setTvDisplayName] = useState<string | undefined>(
    config.tradingViewDisplayName ?? fallback?.displayName,
  );

  useEffect(() => {
    const fb = TV_FALLBACKS[config.market];
    if (fb) {
      setTvEmbedSymbol(fb.embedSymbol);
      setTvChartSymbol(fb.chartSymbol);
      setTvEmbedLabel(fb.embedLabel);
      setTvDisplayName(config.tradingViewDisplayName ?? fb.displayName);
    }

    api.getMarkets()
      .then(res => {
        const info = res.markets.find(m => m.api_markets.includes(config.market));
        if (info) {
          setTvEmbedSymbol(info.tradingview_embed_symbol ?? fb?.embedSymbol ?? null);
          setTvChartSymbol(info.tradingview_symbol ?? fb?.chartSymbol ?? null);
          setTvEmbedLabel(info.tradingview_embed_label ?? fb?.embedLabel);
          setTvDisplayName(config.tradingViewDisplayName ?? info.display_name);
        }
      })
      .catch(() => { /* fallback déjà appliqué */ });
  }, [config.market, config.tradingViewDisplayName]);

  const loadBrief = useCallback(async (force = false) => {
    setBriefLoading(true);
    setBriefError(null);
    try {
      const res = await api.getMarketIntelligence({
        market: config.market,
        mode: 'standard',
        force_refresh: force,
      });
      setIntelligence(res);
    } catch (err: unknown) {
      const e = err as { response?: { status?: number; data?: { detail?: string; message?: string } } };
      const status = e.response?.status;
      const detail = e.response?.data?.detail;
      const message = e.response?.data?.message;
      let msg = (typeof detail === 'string' ? detail : null)
        || (typeof message === 'string' ? message : null)
        || 'Brief indisponible';
      if (status === 404) {
        msg = 'Endpoint brief introuvable — redémarrez l’API Docker';
      } else if (status === 503 && msg.includes('ANTHROPIC')) {
        msg = 'Claude non configuré — ajoutez ANTHROPIC_API_KEY dans .env';
      }
      setBriefError(msg);
    } finally {
      setBriefLoading(false);
    }
  }, [config.market]);

  const runAdvanced = async (question: string) => {
    setAdvancedLoading(true);
    setBriefError(null);
    try {
      const res = await api.postBrief({
        market: config.market,
        mode: 'advanced',
        question,
      });
      setIntelligence(res);
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string; message?: string } } };
      const msg = e.response?.data?.detail || e.response?.data?.message || 'Analyse Opus indisponible';
      setBriefError(typeof msg === 'string' ? msg : JSON.stringify(msg));
    } finally {
      setAdvancedLoading(false);
    }
  };

  const fetchAll = async () => {
    setLoading(true);
    setError(null);
    try {
      const [predRes, valRes] = await Promise.all([
        api.getPredictions({
          market: config.market,
          horizons: [1, 7, 30],
          include_sentiment: config.includeSentiment,
        }),
        api.getValidationMetrics(),
      ]);
      setData(predRes);
      setValidation(valRes);
      setLastUpdate(new Date());
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string; message?: string } } };
      setError(e.response?.data?.detail || e.response?.data?.message || 'Impossible de charger les prédictions');
    } finally {
      setLoading(false);
    }
    loadBrief();
  };

  useEffect(() => { fetchAll(); }, [config.market]);

  const onNewTvAlert = useCallback(() => {
    // Rafraîchir le brief (déjà régénéré par le webhook) sans forcer Claude
    void loadBrief(false);
  }, [loadBrief]);

  const {
    latestAlert,
    popupAlert,
    dismissPopup,
    muted,
    setMuted,
  } = useTradingViewAlertPoll(config.market, onNewTvAlert);

  const currentPrice = data?.current_price ?? 0;
  const garchVol = data?.predictions?.[0]?.components?.garch_annualized_volatility;
  const highVolRegime = Boolean(data?.predictions?.some(p => p.components?.high_volatility_regime));
  const val7 = validation?.xgb_metrics?.find(m => m.horizon === 7);

  return (
    <div className="min-h-screen bg-[#06091a] bg-grid">
      <header className="sticky top-0 z-50 glass-card !rounded-none border-x-0 border-t-0">
        <div className="max-w-[1400px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3 min-w-0">
            <Link href="/" className="shrink-0 rounded-xl overflow-hidden ring-1 ring-white/10 bg-black/40">
              <Image
                src="/logo.png"
                alt="SCPB Market Forecast"
                width={40}
                height={40}
                className="w-10 h-10 object-contain"
                priority
              />
            </Link>
            <div className="min-w-0">
              <h1 className="text-lg font-bold text-white tracking-tight truncate">
                SCPB Market Forecast
              </h1>
              <p className="text-[11px] text-slate-500 font-medium truncate">
                {config.title}
                <span className="text-slate-600"> · </span>
                {config.subtitle}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <nav className="flex items-center gap-1 bg-white/[0.04] rounded-xl p-1 border border-white/[0.06]">
              {config.activeNav === 'cacao' ? (
                <>
                  <span className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${theme.navActive}`}>Cacao</span>
                  <Link href={config.otherMarket.href} className="px-3 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-white hover:bg-white/[0.06] transition-all">
                    {config.otherMarket.label}
                  </Link>
                </>
              ) : (
                <>
                  <Link href={config.otherMarket.href} className="px-3 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-white hover:bg-white/[0.06] transition-all">
                    {config.otherMarket.label}
                  </Link>
                  <span className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${theme.navActive}`}>Café Robusta</span>
                </>
              )}
            </nav>
            <span
              className="hidden md:inline-flex items-center gap-1.5 text-[10px] text-slate-400 px-2 py-1 rounded-lg border border-white/[0.06] bg-white/[0.03]"
              title={latestAlert ? `Dernière alerte: ${latestAlert.signal_type}` : 'Écoute des alertes TradingView'}
            >
              <span className="relative flex h-1.5 w-1.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60" />
                <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-400" />
              </span>
              <Bell className="w-3 h-3" />
              Alertes live
            </span>
            {lastUpdate && (
              <span className="text-xs text-slate-500 hidden sm:block">
                {lastUpdate.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}
              </span>
            )}
            <button
              onClick={fetchAll}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold bg-white/[0.06] border border-white/[0.08] text-slate-300 hover:bg-white/[0.1] hover:text-white transition-all disabled:opacity-40"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              Actualiser
            </button>
          </div>
        </div>
      </header>

      <TradingViewAlertPopup
        alert={popupAlert}
        onDismiss={dismissPopup}
        muted={muted}
        onToggleMute={() => setMuted(!muted)}
      />

      <main className="max-w-[1400px] mx-auto px-6 py-8">
        {error && (
          <div className="glass-card p-4 !border-rose-500/20 mb-6">
            <p className="text-rose-400 text-sm flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" /> {error}
            </p>
          </div>
        )}

        {loading && !data && (
          <div className="grid lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-4">
              <div className="glass-card h-28 shimmer" />
              <div className="grid grid-cols-3 gap-4">
                {[0, 1, 2].map(i => <div key={i} className="glass-card h-36 shimmer" />)}
              </div>
              <div className="glass-card h-80 shimmer" />
              <div className="glass-card h-48 shimmer" />
            </div>
            <div className="space-y-4">
              <div className="glass-card h-[520px] shimmer" />
              <div className="glass-card h-72 shimmer" />
            </div>
          </div>
        )}

        {data && (
          <div className="grid lg:grid-cols-3 gap-6 items-start animate-fade-in">
            {/* Colonne gauche — contenu analytique principal */}
            <div className="lg:col-span-2 space-y-5">
              <MarketKPIBar
                data={data}
                unitLabel={config.unitLabel}
                includeSentiment={config.includeSentiment}
                priceSource={config.priceSource}
                lastUpdate={lastUpdate}
                accentClass={theme.accentClass}
              />

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {data.predictions.map(pred => (
                  <PredictionHorizonCard
                    key={pred.horizon}
                    pred={pred}
                    currentPrice={currentPrice}
                    briefSignal={intelligence?.brief?.signal}
                    validation={val7}
                  />
                ))}
              </div>

              <ForecastChart
                historical={data.historical_prices}
                predictions={data.predictions}
                currentPrice={currentPrice}
                chartStroke={theme.chartStroke}
                chartGradient={theme.chartGradient}
                accentClass={theme.accentClass}
                brief={intelligence?.brief}
              />

              <AnalysisPanel
                predictions={data.predictions}
                currentPrice={currentPrice}
                sentiment={data.sentiment_score}
                garchVol={garchVol}
                highVolRegime={highVolRegime}
                brief={intelligence?.brief}
                validation={validation}
                modelVersion={data.model_version}
              />
            </div>

            {/* Colonne droite — sticky : TradingView + Brief IA */}
            <div className="lg:col-span-1">
              <div className="lg:sticky lg:top-24 space-y-5">
                {tvEmbedSymbol && (
                  <TradingViewEmbed
                    symbol={tvEmbedSymbol}
                    displayName={tvDisplayName}
                    contractLabel={tvEmbedLabel}
                    chartLink={
                      tvChartSymbol && tvChartSymbol !== tvEmbedSymbol
                        ? `https://www.tradingview.com/chart/?symbol=${encodeURIComponent(tvChartSymbol)}`
                        : undefined
                    }
                    height={480}
                    accentClass={theme.accentClass}
                  />
                )}
                <MarketBrief
                  market={config.market}
                  accentClass="text-violet-400"
                  data={intelligence}
                  loading={briefLoading}
                  error={briefError}
                  onRefresh={loadBrief}
                  onAdvanced={runAdvanced}
                  advancedLoading={advancedLoading}
                  sticky
                />
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="mt-12 border-t border-white/[0.04] py-5">
        <p className="text-center text-xs text-slate-600">
          Afrexia · Prédictions ML + Brief Claude · © 2026
        </p>
      </footer>
    </div>
  );
}
