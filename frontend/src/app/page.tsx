'use client';

import { SimplifiedMarketDashboard } from '@/components/SimplifiedMarketDashboard';

export default function Dashboard() {
  return (
    <SimplifiedMarketDashboard
      config={{
        market: 'ICE_NY',
        tradingViewDisplayName: 'Cocoa Cash Contract',
        title: 'Cacao',
        subtitle: 'ICE London · Hybrid AI',
        accent: 'amber',
        includeSentiment: true,
        unitLabel: 'GBP / tonne',
        priceCurrency: 'GBP',
        priceSource: 'ICE London · clôture quotidienne',
        activeNav: 'cacao',
        otherMarket: { href: '/coffee', label: 'Café Robusta' },
      }}
    />
  );
}
