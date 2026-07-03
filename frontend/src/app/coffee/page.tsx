'use client';

import { SimplifiedMarketDashboard } from '@/components/SimplifiedMarketDashboard';

export default function CoffeeDashboard() {
  return (
    <SimplifiedMarketDashboard
      config={{
        market: 'COFFEE_ROBUSTA',
        tradingViewDisplayName: 'Robusta Coffee Futures',
        title: 'Coffee Intelligence · Robusta',
        subtitle: 'ICE London · Hybrid AI + GARCH',
        accent: 'emerald',
        includeSentiment: false,
        unitLabel: 'USD / tonne',
        priceSource: 'Investing.com (RCU6) · clôture quotidienne',
        activeNav: 'coffee',
        otherMarket: { href: '/', label: 'Cacao' },
      }}
    />
  );
}
