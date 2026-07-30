import { NextResponse } from 'next/server';

export const revalidate = 3600;

const FALLBACK_USD_GBP = 0.747;

/**
 * GET /api/fx/usd-gbp
 * Proxy serveur pour éviter CORS / bloqueurs côté navigateur.
 * Retourne { rate, source, date } où rate = GBP pour 1 USD.
 */
export async function GET() {
  try {
    const res = await fetch(
      'https://api.frankfurter.app/latest?from=USD&to=GBP',
      { next: { revalidate: 3600 } },
    );
    if (res.ok) {
      const data = await res.json();
      const rate = data?.rates?.GBP;
      if (typeof rate === 'number' && rate > 0) {
        return NextResponse.json({
          rate,
          source: 'frankfurter',
          date: data?.date ?? null,
        });
      }
    }
  } catch {
    // fallback below
  }

  return NextResponse.json({
    rate: FALLBACK_USD_GBP,
    source: 'fallback',
    date: null,
  });
}
