export type ForecastStatus = 'experimental' | 'validated' | 'unavailable';

export interface ForecastView {
  price?: number | null;
  lower?: number | null;
  upper?: number | null;
  target_date?: string | null;
  issued_date?: string | null;
  model_version?: string | null;
  status?: ForecastStatus | null;
}

export interface PredictionItem {
  horizon: number;
  price: number | null;
  confidence_interval: [number, number] | null;
  confidence_level: number | null;
  timestamp: string;
  status?: ForecastStatus;
  target_date?: string | null;
  reference_price?: number | null;
  initial_horizon?: number | null;
  sessions_remaining?: number | null;
  latest_price?: number | null;
  latest_date?: string | null;
  latest_gap?: number | null;
  latest_gap_pct?: number | null;
  revision_notice?: 'inversion' | 'revision' | null;
  has_revision?: boolean | null;
  latest_unavailable?: boolean | null;
  initial?: ForecastView | null;
  latest?: ForecastView | null;
  fresh?: ForecastView | null;
  components?: {
    baseline?: number;
    nhits?: number | null;
    prophet?: number;
    ensemble?: number;
    residual?: number;
    sentiment?: number;
    feature_failure?: boolean;
    status?: ForecastStatus;
    label?: string;
    band_label?: string;
    garch_annualized_volatility?: number | null;
    high_volatility_regime?: boolean;
    warning?: string;
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

export interface RevisionDelayStep {
  remaining: number;
  comparison: 'initial_vs_remaining' | 'successive';
  expected: number;
  available: number;
  abstentions: number;
  availability_rate?: number | null;
  mean_initial_error?: number | null;
  mean_revised_error?: number | null;
  mean_spot_error?: number | null;
  mean_gain?: number | null;
  improved_share?: number | null;
  gain_low?: number | null;
  gain_high?: number | null;
  verdict: 'etabli' | 'defavorable' | 'incertain';
  interval_level?: number | null;
  usable_blocks?: number | null;
  block?: number | null;
}

export interface RevisionValueHorizon {
  horizon: number;
  comparison?: string | null;
  verdict: 'etabli' | 'defavorable' | 'incertain';
  expiries: number;
  mean_initial_error?: number | null;
  mean_latest_error?: number | null;
  mean_gain?: number | null;
  improved_share?: number | null;
  worsened_share?: number | null;
  improvement_sum?: number | null;
  degradation_sum?: number | null;
  gain_low?: number | null;
  gain_high?: number | null;
  block?: number | null;
  interval_level?: number | null;
  usable_blocks?: number | null;
  steps?: RevisionDelayStep[];
}

export interface RevisionValueResponse {
  market: string;
  as_of?: string | null;
  note?: string | null;
  horizons: RevisionValueHorizon[];
}

export interface ValidationHistoryItem {
  report_id: string;
  timestamp?: string | null;
  kind?: string | null;
  period?: string | null;
  protocol?: string | null;
  counts?: string | null;
  compatible_with_current_lot: boolean;
  note: string;
}

export interface ValidationHistoryResponse {
  market: string;
  release_version?: string | null;
  reports: ValidationHistoryItem[];
}

export interface ForecastTrackingPoint {
  forecast_id?: string | null;
  origin_date?: string | null;
  origin_price?: number | null;
  target_date?: string | null;
  horizon?: number | null;
  price?: number | null;
  lower?: number | null;
  upper?: number | null;
  status?: ForecastStatus | string | null;
  model_version?: string | null;
  reference_price?: number | null;
  trace?: string | null;
  revises?: string[];
  error?: number | null;
}

export interface ForecastTrackingResponse {
  market: string;
  found?: boolean | null;
  initial_id?: string | null;
  initials?: ForecastTrackingPoint[];
  total_initials?: number | null;
  offset?: number | null;
  initial?: ForecastTrackingPoint | null;
  reference_price?: number | null;
  revisions?: ForecastTrackingPoint[];
  latest?: ForecastTrackingPoint | null;
  gap?: number | null;
  gap_pct?: number | null;
  direction?: string | null;
  direction_vs_emission_spot?: string | null;
  inversions?: number | null;
  points?: { date: string; price?: number | null; model_version?: string | null; gap?: boolean }[];
  remaining_horizon?: number | null;
  mature?: boolean | null;
  realized_price?: number | null;
  initial_error?: number | null;
  latest_error?: number | null;
  model_comparison?: {
    available?: boolean;
    comparison?: string;
    previous_model_version?: string | null;
    previous_price?: number | null;
    new_price?: number | null;
    note?: string | null;
  } | null;
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
  validated?: boolean;
  relative_gain?: number | null;
  gap_ci_low?: number | null;
  gap_ci_high?: number | null;
  n_effective?: number | null;
  fallback_rate?: number | null;
  measurement?: string | null;
  reason?: string | null;
  gap_reading?: string | null;
  available_mape?: number | null;
  naive_mape?: number | null;
  abstention_count?: number | null;
  abstention_rate?: number | null;
  coverage?: number | null;
  coverage_note?: string | null;
}

export interface ValidationMetricsResponse {
  report_timestamp?: string | null;
  validation_type: string;
  n_origins?: number | null;
  origin_start?: string | null;
  origin_end?: string | null;
  target_start?: string | null;
  target_end?: string | null;
  horizons: number[];
  xgb_metrics: HorizonValidationMetrics[];
  legacy_holdout_mape_1step?: number | null;
  report_id?: string | null;
  measurement_status?: string | null;
  partition?: string | null;
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
  unit?: string;
  currency?: string;
  london_contracts?: FuturesContractItem[];
  london_collected_at?: string | null;
  london_source?: string | null;
  london_unit?: string;
  london_currency?: string;
}

export interface PredictionHistoryItem {
  created_at: string;
  horizon: number;
  predicted_price: number;
  lower_bound?: number | null;
  upper_bound?: number | null;
  model_version?: string | null;
}

export interface PredictionHistoryResponse {
  predictions: PredictionHistoryItem[];
  count: number;
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

export interface CocoaWeatherLocation {
  location: string;
  country: string;
  temperature_c?: number | null;
  precipitation_mm?: number | null;
  humidity?: number | null;
  condition?: string | null;
  collected_at?: string | null;
}

export interface CocoaWeatherResponse {
  locations: CocoaWeatherLocation[];
  collected_at?: string | null;
  note: string;
}
