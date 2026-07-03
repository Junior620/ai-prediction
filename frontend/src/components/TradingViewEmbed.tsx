'use client';

import { useEffect, useRef, useState } from 'react';
import { BarChart2, AlertTriangle, ExternalLink } from 'lucide-react';

const WIDGET_SCRIPT =
  'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';

interface TradingViewEmbedProps {
  /** Symbole compatible widget embed (contrat mensuel ICE, pas le continu CC1!/RC1!) */
  symbol: string;
  displayName?: string;
  contractLabel?: string;
  /** Lien vers le graphique continu sur tradingview.com */
  chartLink?: string;
  height?: number;
  accentClass?: string;
}

/**
 * Widget Advanced Chart TradingView.
 * Cacao : PEPPERSTONE:COCOA (CFD embeddable). Robusta embed : ROBCOFFEE.
 */
export function TradingViewEmbed({
  symbol,
  displayName,
  contractLabel,
  chartLink,
  height = 520,
  accentClass = 'text-amber-400',
}: TradingViewEmbedProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [mounted, setMounted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted || !symbol || !containerRef.current) return;

    const container = containerRef.current;
    container.innerHTML = '';
    setError(null);

    try {
      const widgetDiv = document.createElement('div');
      widgetDiv.className = 'tradingview-widget-container__widget';
      widgetDiv.style.cssText = `height:${height}px;width:100%;min-height:${height}px;`;

      const script = document.createElement('script');
      script.type = 'text/javascript';
      script.src = WIDGET_SCRIPT;
      script.async = true;
      script.textContent = JSON.stringify({
        width: '100%',
        height,
        symbol,
        interval: 'D',
        timezone: 'Etc/UTC',
        theme: 'dark',
        style: '1',
        locale: 'fr',
        enable_publishing: false,
        allow_symbol_change: false,
        hide_side_toolbar: true,
        backgroundColor: '#06091a',
        gridColor: 'rgba(148, 163, 184, 0.08)',
        support_host: 'https://www.tradingview.com',
      });

      script.onerror = () => {
        setError('Graphique TradingView indisponible');
      };

      container.appendChild(widgetDiv);
      container.appendChild(script);
    } catch {
      setError('Graphique TradingView indisponible');
    }

    return () => {
      container.innerHTML = '';
    };
  }, [mounted, symbol, height]);

  return (
    <div className="glass-card p-5 overflow-hidden">
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-sm font-bold text-white flex items-center gap-2">
          <BarChart2 className={`w-4 h-4 ${accentClass}`} />
          Marché en direct
        </h2>
        <span className="text-[10px] text-slate-500 font-mono">{symbol}</span>
      </div>

      <div className="mb-3 space-y-1">
        {displayName && (
          <p className="text-[11px] text-slate-400">{displayName}</p>
        )}
        {contractLabel && (
          <p className="text-[10px] text-slate-500">{contractLabel}</p>
        )}
        {chartLink && (
          <a
            href={chartLink}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-[10px] text-amber-400/90 hover:text-amber-300 transition-colors"
          >
            <ExternalLink className="w-3 h-3" />
            Contrat continu sur TradingView
          </a>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-2 text-rose-400 text-xs mb-3">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div
        ref={containerRef}
        key={symbol}
        className="tradingview-widget-container w-full rounded-lg overflow-hidden"
        style={{ height: `${height}px`, minHeight: `${height}px` }}
      />
    </div>
  );
}
