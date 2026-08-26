import axios from 'axios';
import type {
  PredictionRequest,
  PredictionResponse,
  ValidationMetricsResponse,
  MarketIntelligenceResponse,
  BriefRequest,
  MarketsResponse,
  LatestTradingViewAlert,
  RecentTradingViewAlertsResponse,
  DashboardNotification,
  NotificationsListResponse,
  FuturesCurveResponse,
} from '@/types/api';

/** Same-origin BFF — JWT stays on the server (API_TOKEN). */
const apiClient = axios.create({
  baseURL: '/api/backend',
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface PerformanceMetricsResponse {
  metrics?: Array<{
    model_type?: string;
    horizon?: number;
    mape?: number;
    rmse?: number;
    mae?: number;
    direction_accuracy?: number;
    period_start?: string;
    period_end?: string;
    [key: string]: unknown;
  }>;
  [key: string]: unknown;
}

export const api = {
  async getPredictions(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await apiClient.post<PredictionResponse>('/api/v1/predict', request);
    return response.data;
  },

  async healthCheck(): Promise<{ status: string }> {
    const response = await apiClient.get('/health');
    return response.data;
  },

  async getPredictionHistory(limit = 30, horizon?: number): Promise<any> {
    const params: any = { limit };
    if (horizon) params.horizon = horizon;
    const response = await apiClient.get('/api/v1/prediction-history', { params });
    return response.data;
  },

  async getFutures(includePredictions = true): Promise<FuturesCurveResponse> {
    const response = await apiClient.get<FuturesCurveResponse>('/api/v1/futures', {
      params: { include_predictions: includePredictions },
    });
    return response.data;
  },

  async getPerformance(params?: {
    start_date?: string;
    end_date?: string;
  }): Promise<PerformanceMetricsResponse | null> {
    try {
      const response = await apiClient.get<PerformanceMetricsResponse>('/api/v1/performance', {
        params,
      });
      return response.data;
    } catch {
      return null;
    }
  },

  async getValidationMetrics(): Promise<ValidationMetricsResponse | null> {
    try {
      const response = await apiClient.get<ValidationMetricsResponse>('/api/v1/validation/metrics');
      return response.data;
    } catch {
      return null;
    }
  },

  async getMarketIntelligence(params: {
    market: string;
    mode?: string;
    force_refresh?: boolean;
  }): Promise<MarketIntelligenceResponse> {
    const response = await apiClient.get<MarketIntelligenceResponse>(
      '/api/v1/market-intelligence',
      { params },
    );
    return response.data;
  },

  async postBrief(body: BriefRequest): Promise<MarketIntelligenceResponse> {
    const response = await apiClient.post<MarketIntelligenceResponse>('/api/v1/brief', body);
    return response.data;
  },

  async getMarkets(): Promise<MarketsResponse> {
    const response = await apiClient.get<MarketsResponse>('/api/v1/markets');
    return response.data;
  },

  async getLatestTradingViewAlert(market: string): Promise<LatestTradingViewAlert | null> {
    try {
      const response = await apiClient.get<LatestTradingViewAlert | null>(
        '/api/v1/tradingview/alerts/latest',
        { params: { market } },
      );
      return response.data ?? null;
    } catch {
      return null;
    }
  },

  async getRecentTradingViewAlerts(
    market: string,
    limit = 5,
  ): Promise<LatestTradingViewAlert[]> {
    try {
      const response = await apiClient.get<RecentTradingViewAlertsResponse>(
        '/api/v1/tradingview/alerts/recent',
        { params: { market, limit } },
      );
      return response.data?.alerts ?? [];
    } catch {
      return [];
    }
  },

  async getNotifications(
    market: string,
    opts?: { limit?: number; unread_only?: boolean },
  ): Promise<NotificationsListResponse> {
    const response = await apiClient.get<NotificationsListResponse>('/api/v1/notifications', {
      params: {
        market,
        limit: opts?.limit ?? 30,
        unread_only: opts?.unread_only ?? false,
      },
    });
    return response.data;
  },

  async markNotificationRead(id: string): Promise<void> {
    await apiClient.post(`/api/v1/notifications/${id}/read`);
  },

  async markAllNotificationsRead(market: string): Promise<void> {
    await apiClient.post('/api/v1/notifications/read-all', null, { params: { market } });
  },

  /** Fetch WS auth from server route, then build wss URL (token not in bundle). */
  async notificationsWsUrl(market: string): Promise<string | null> {
    try {
      const res = await fetch('/api/auth/ws-token', { cache: 'no-store' });
      if (!res.ok) return null;
      const data = (await res.json()) as { token?: string; apiUrl?: string };
      if (!data.token || !data.apiUrl) return null;
      const wsBase = data.apiUrl.startsWith('https')
        ? data.apiUrl.replace(/^https/, 'wss')
        : data.apiUrl.replace(/^http/, 'ws');
      const q = new URLSearchParams({ token: data.token, market });
      return `${wsBase}/api/v1/ws/notifications?${q.toString()}`;
    } catch {
      return null;
    }
  },
};
