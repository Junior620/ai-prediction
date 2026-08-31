'use client';

import { SimplifiedMarketDashboard } from '@/components/SimplifiedMarketDashboard';

export default function Dashboard() {
  return (
    <SimplifiedMarketDashboard
      config={{
        market: 'ICE_NY',
        tradingViewDisplayName: 'Cocoa Cash Contract',
        title: 'Cacao',
        subtitle: 'ICE London · Databento · Hybrid AI',
        accent: 'amber',
        includeSentiment: true,
        unitLabel: 'GBP / tonne',
        priceCurrency: 'GBP',
        priceSource: 'ICE London · Databento OHLCV + OI',
        activeNav: 'cacao',
        otherMarket: { href: '/coffee', label: 'Café Robusta' },
      }}
    />
  );
}
