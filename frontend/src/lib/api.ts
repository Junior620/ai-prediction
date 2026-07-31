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
} from '@/types/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_TOKEN = process.env.NEXT_PUBLIC_API_TOKEN;

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
    ...(API_TOKEN && { Authorization: `Bearer ${API_TOKEN}` }),
  },
});

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

  async getFutures(): Promise<any> {
    const response = await apiClient.get('/api/v1/futures');
    return response.data;
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

  notificationsWsUrl(market: string): string | null {
    if (!API_TOKEN) return null;
    const base = API_URL.replace(/\/$/, '');
    const wsBase = base.startsWith('https')
      ? base.replace(/^https/, 'wss')
      : base.replace(/^http/, 'ws');
    const q = new URLSearchParams({ token: API_TOKEN, market });
    return `${wsBase}/api/v1/ws/notifications?${q.toString()}`;
  },
};
