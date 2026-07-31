'use client';

import { Suspense } from 'react';
import NotificationsPageClient from './NotificationsPageClient';

export default function NotificationsPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-[#06091a] flex items-center justify-center text-slate-500 text-sm">
          Chargement des notifications…
        </div>
      }
    >
      <NotificationsPageClient />
    </Suspense>
  );
}
