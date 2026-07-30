'use client';

import { useEffect, useState } from 'react';

/** Fallback si l’API FX est indisponible (~taux BCE récent). */
const FALLBACK_USD_GBP = 0.747;

/**
 * Taux USD → GBP (combien de GBP pour 1 USD).
 * Passe par /api/fx/usd-gbp (proxy Next) pour éviter CORS / adblock.
 */
export function useUsdGbpRate(): number | null {
  const [rate, setRate] = useState<number | null>(FALLBACK_USD_GBP);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await fetch('/api/fx/usd-gbp');
        if (!res.ok) return;
        const data = await res.json();
        const gbp = data?.rate;
        if (!cancelled && typeof gbp === 'number' && gbp > 0 && Number.isFinite(gbp)) {
          setRate(gbp);
        }
      } catch {
        // garde le fallback déjà affiché
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return rate;
}
