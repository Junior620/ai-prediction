import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

/** Identifie le déploiement en cours pour qu'un onglet ouvert recharge la page. */
export function GET() {
  const id = process.env.VERCEL_DEPLOYMENT_ID || 'local';
  return NextResponse.json(
    { id },
    { headers: { 'Cache-Control': 'no-store' } },
  );
}
