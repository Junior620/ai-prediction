export interface PredictionItem {
  horizon: number;
  price: number;
  confidence_interval: [number, number];
  confidence_level: number;
  timestamp: string;
  components?: {
    baseline?: number;
    nhits?: number | null;
    prophet?: number;
    ensemble?: number;
    residual?: number;
    sentiment?: number;
    garch_annualized_volatility?: number | null;
    high_volatility_regime?: boolean;
  };
}

export interface PredictionResponse {
  predictions: PredictionItem[];
  model_version: string;
  sentiment_score: number | null;
  market: string;
  current_price?: number;
  current_date?: string;
  historical_prices?: { date: string; price: number }[];
}

export interface PredictionRequest {
  market: string;
  horizons: number[];
  include_sentiment: boolean;
}

export interface ApiError {
  error: string;
  message: string;
  detail?: string;
}

export interface HorizonValidationMetrics {
  horizon: number;
  mape?: number | null;
  rmse?: number | null;
  mae?: number | null;
  directional_accuracy?: number | null;
  n_predictions?: number | null;
}

export interface ValidationMetricsResponse {
  report_timestamp?: string | null;
  validation_type: string;
  n_origins?: number | null;
  horizons: number[];
  xgb_metrics: HorizonValidationMetrics[];
  legacy_holdout_mape_1step?: number | null;
}

export interface MarketBriefContent {
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: 'low' | 'medium' | 'high';
  trend: 'bullish' | 'bearish' | 'neutral';
  summary: string;
  outlook_7d?: string;
  key_levels?: { support?: number | null; resistance?: number | null };
  risks: string[];
  recommendation?: string;
  disclaimer?: string;
  _meta?: Record<string, unknown>;
}

export interface MarketIntelligenceResponse {
  market: string;
  market_display_name: string;
  unit: string;
  tradingview_symbol?: string | null;
  current_price?: number | null;
  current_date?: string | null;
  model_version?: string | null;
  sentiment_score?: number | null;
  predictions?: PredictionItem[];
  brief: MarketBriefContent;
  mode: string;
  opus_remaining: number;
  generated_at: string;
  cached?: boolean;
}

export interface BriefRequest {
  market: string;
  mode?: 'standard' | 'advanced';
  question?: string;
  force_refresh?: boolean;
}

export interface MarketInfo {
  market_id: string;
  display_name: string;
  api_markets: string[];
  unit: string;
  source: string;
  contract_symbol?: string | null;
  garch_enabled: boolean;
  tradingview_symbol?: string | null;
  tradingview_embed_symbol?: string | null;
  tradingview_embed_label?: string | null;
  tradingview_alert_symbol?: string | null;
  available: boolean;
  model_version?: string | null;
}

export interface MarketsResponse {
  markets: MarketInfo[];
}

export interface LatestTradingViewAlert {
  id: string;
  market: string;
  signal_type: string;
  price?: number | null;
  tf?: string | null;
  ticker?: string | null;
  message?: string | null;
  trend?: string | null;
  momentum?: string | null;
  support?: number | null;
  resistance?: number | null;
  change_pct?: number | null;
  received_at: string;
  brief_signal?: string | null;
  brief_summary?: string | null;
}

export interface RecentTradingViewAlertsResponse {
  market: string;
  alerts: LatestTradingViewAlert[];
}

export interface DashboardNotification {
  id: string;
  market: string;
  source: string;
  kind: string;
  title: string;
  body?: string | null;
  payload?: Record<string, unknown>;
  is_read: boolean;
  created_at: string;
}

export interface NotificationsListResponse {
  market: string;
  notifications: DashboardNotification[];
  unread_count: number;
}

export interface FuturesHorizonPrediction {
  horizon: number;
  price: number;
  method: string;
  change_pct?: number | null;
}

export interface FuturesContractItem {
  contract: string;
  symbol: string;
  yahoo_symbol?: string | null;
  price_usd: number;
  change?: number | null;
  volume?: number | null;
  predictions: FuturesHorizonPrediction[];
}

export interface FuturesCurveResponse {
  contracts: FuturesContractItem[];
  collected_at?: string | null;
  source?: string | null;
  model_version?: string | null;
  spot_pct_by_horizon?: Record<string, number> | null;
}

export interface LondonHistoryPoint {
  date: string;
  price: number;
  volume?: number | null;
  open_interest?: number | null;
  source?: string | null;
}

export interface LondonTermContract {
  contract_rank: number;
  symbol: string;
  close: number;
  volume?: number | null;
  open_interest?: number | null;
  label: string;
}

export interface LondonMarketResponse {
  latest?: LondonHistoryPoint | null;
  history: LondonHistoryPoint[];
  term_structure: LondonTermContract[];
  term_date?: string | null;
  unit: string;
  source?: string | null;
}

export interface ModelComparisonMetric {
  model: string;
  label: string;
  horizon: number;
  mae: number;
  rmse: number;
  mape: number;
  n: number;
}

export interface ModelComparisonResponse {
  generated_at?: string | null;
  split_date?: string | null;
  n_train?: number | null;
  n_test?: number | null;
  metrics: ModelComparisonMetric[];
  note?: string | null;
}
