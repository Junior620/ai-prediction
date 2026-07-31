'use client';

import { useState } from 'react';
import type { MarketBriefContent, MarketIntelligenceResponse } from '@/types/api';
import {
  Brain, RefreshCw, AlertTriangle, TrendingUp, TrendingDown, Minus,
  Sparkles, ChevronDown, ChevronUp,
} from 'lucide-react';

interface MarketBriefProps {
  market: string;
  accentClass?: string;
  data: MarketIntelligenceResponse | null;
  loading: boolean;
  error: string | null;
  onRefresh: (force?: boolean) => void;
  onAdvanced: (question: string) => Promise<void>;
  advancedLoading?: boolean;
  sticky?: boolean;
  /** Surbrillance temporaire après une alerte TradingView */
  highlight?: boolean;
}

const signalStyles = {
  BUY: { accent: 'text-emerald-400', bg: 'bg-emerald-500/15', border: 'border-emerald-500/30', Icon: TrendingUp },
  SELL: { accent: 'text-rose-400', bg: 'bg-rose-500/15', border: 'border-rose-500/30', Icon: TrendingDown },
  HOLD: { accent: 'text-amber-400', bg: 'bg-amber-500/15', border: 'border-amber-500/30', Icon: Minus },
};

function BriefBody({ brief }: { brief: MarketBriefContent }) {
  const sig = signalStyles[brief.signal] || signalStyles.HOLD;
  const support = brief.key_levels?.support;
  const resistance = brief.key_levels?.resistance;

  return (
    <div className="space-y-4">
      <div className={`flex items-center gap-3 p-3 rounded-xl border ${sig.bg} ${sig.border}`}>
        <sig.Icon className={`w-5 h-5 ${sig.accent}`} />
        <div>
          <span className={`text-xl font-black ${sig.accent}`}>{brief.signal}</span>
          <span className="text-xs text-slate-400 ml-2 capitalize">· confiance {brief.confidence}</span>
        </div>
        <span className="ml-auto text-xs text-slate-500 capitalize">{brief.trend}</span>
      </div>

      <p className="text-sm text-slate-300 leading-relaxed">{brief.summary}</p>

      {brief.outlook_7d && (
        <div>
          <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Vue 7 jours</p>
          <p className="text-sm text-slate-400">{brief.outlook_7d}</p>
        </div>
      )}

      {(support != null || resistance != null) && (
        <div className="grid grid-cols-2 gap-2 text-xs">
          {support != null && (
            <div className="bg-white/[0.03] rounded-lg p-2.5">
              <p className="text-slate-500 mb-0.5">Support</p>
              <p className="font-bold text-emerald-400 font-mono-price">${support.toLocaleString()}</p>
            </div>
          )}
          {resistance != null && (
            <div className="bg-white/[0.03] rounded-lg p-2.5">
              <p className="text-slate-500 mb-0.5">Résistance</p>
              <p className="font-bold text-rose-400 font-mono-price">${resistance.toLocaleString()}</p>
            </div>
          )}
        </div>
      )}

      {brief.risks?.length > 0 && (
        <div>
          <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">Risques</p>
          <ul className="space-y-1">
            {brief.risks.map((r, i) => (
              <li key={i} className="text-xs text-slate-400 flex gap-2">
                <span className="text-rose-400/70">•</span>{r}
              </li>
            ))}
          </ul>
        </div>
      )}

      {brief.recommendation && (
        <p className="text-xs text-slate-300 italic border-t border-white/[0.06] pt-3">
          {brief.recommendation}
        </p>
      )}
    </div>
  );
}

export function MarketBrief({
  accentClass = 'text-violet-400',
  data,
  loading,
  error,
  onRefresh,
  onAdvanced,
  advancedLoading = false,
  sticky = true,
  highlight = false,
}: MarketBriefProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [question, setQuestion] = useState('');

  const runAdvanced = async () => {
    if (!question.trim() || (data?.opus_remaining ?? 0) <= 0) return;
    await onAdvanced(question.trim());
    setShowAdvanced(false);
    setQuestion('');
  };

  return (
    <div
      className={`glass-card p-5 transition-all duration-500 ${
        sticky ? 'lg:shadow-xl lg:shadow-black/20' : ''
      } ${
        highlight
          ? 'ring-2 ring-amber-400/60 shadow-lg shadow-amber-500/20 scale-[1.01]'
          : ''
      }`}
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-bold text-white flex items-center gap-2">
          <Brain className={`w-4 h-4 ${accentClass}`} />
          Brief IA
          <span className="text-[10px] font-normal text-slate-500">Sonnet</span>
        </h2>
        <div className="flex items-center gap-2">
          {data?.cached && (
            <span className="text-[10px] text-slate-600">cache 24h</span>
          )}
          <button
            onClick={() => onRefresh(true)}
            disabled={loading}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/[0.06] transition-all disabled:opacity-40"
            title="Rafraîchir le brief"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {loading && !data && (
        <div className="space-y-3">
          <div className="h-16 shimmer rounded-xl" />
          <div className="h-4 shimmer rounded w-3/4" />
          <div className="h-4 shimmer rounded w-full" />
        </div>
      )}

      {error && (
        <div className="flex items-start gap-2 text-rose-400 text-xs mb-3">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {data?.brief && <BriefBody brief={data.brief} />}

      <div className="mt-4 pt-4 border-t border-white/[0.06]">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full flex items-center justify-between text-xs text-slate-400 hover:text-white transition-colors"
        >
          <span className="flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-violet-400" />
            Analyse approfondie (Opus)
            <span className="text-slate-600">
              · {data?.opus_remaining ?? 3} restant{(data?.opus_remaining ?? 3) !== 1 ? 's' : ''}/jour
            </span>
          </span>
          {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {showAdvanced && (
          <div className="mt-3 space-y-2">
            <textarea
              value={question}
              onChange={e => setQuestion(e.target.value)}
              placeholder="Ex: Le signal baissier est-il cohérent avec la volatilité GARCH ?"
              rows={3}
              className="w-full text-xs bg-white/[0.04] border border-white/[0.08] rounded-lg p-3 text-slate-300 placeholder:text-slate-600 resize-none focus:outline-none focus:border-violet-500/40"
            />
            <button
              onClick={runAdvanced}
              disabled={advancedLoading || !question.trim() || (data?.opus_remaining ?? 0) <= 0}
              className="w-full py-2 rounded-lg text-xs font-semibold bg-violet-500/20 text-violet-300 border border-violet-500/30 hover:bg-violet-500/30 transition-all disabled:opacity-40"
            >
              {advancedLoading ? 'Analyse en cours…' : 'Lancer Opus'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
