'use client';

import { SimplifiedMarketDashboard } from '@/components/SimplifiedMarketDashboard';

export default function Dashboard() {
  return (
    <SimplifiedMarketDashboard
      config={{
        market: 'ICE_NY',
        tradingViewDisplayName: 'Cocoa Cash Contract',
        title: 'Cacao',
        subtitle: 'ICE New York · Hybrid AI',
        accent: 'amber',
        includeSentiment: true,
        unitLabel: 'USD / tonne',
        priceSource: 'Yahoo Finance (CC=F) · clôture quotidienne',
        activeNav: 'cacao',
        otherMarket: { href: '/coffee', label: 'Café Robusta' },
      }}
    />
  );
}
