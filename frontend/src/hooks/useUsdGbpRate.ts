'use client';

import { useEffect, useState } from 'react';

/**
 * Taux USD → GBP (combien de GBP pour 1 USD).
 * Source publique Frankfurter (BCE), rafraîchi ~1h côté client.
 */
export function useUsdGbpRate(): number | null {
  const [rate, setRate] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await fetch('https://api.frankfurter.app/latest?from=USD&to=GBP');
        if (!res.ok) return;
        const data = await res.json();
        const gbp = data?.rates?.GBP;
        if (!cancelled && typeof gbp === 'number' && gbp > 0) {
          setRate(gbp);
        }
      } catch {
        // silencieux : l'UI affichera "—"
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return rate;
}
