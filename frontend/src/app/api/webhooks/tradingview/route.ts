import { NextRequest, NextResponse } from 'next/server';

/**
 * Proxy public HTTPS pour TradingView.
 *
 * TradingView → https://<app>.vercel.app/api/webhooks/tradingview
 *            → API_BACKEND_URL/api/v1/webhooks/tradingview (FastAPI)
 *
 * Variables Vercel (server-only) :
 *   API_BACKEND_URL = https://ton-api-publique.com  (sans slash final)
 */
export const runtime = 'nodejs';
export const maxDuration = 60;
export const dynamic = 'force-dynamic';

function backendBase(): string {
  const base =
    process.env.API_BACKEND_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    '';
  return base.replace(/\/$/, '');
}

export async function POST(req: NextRequest) {
  const base = backendBase();
  if (!base) {
    return NextResponse.json(
      {
        error: 'API_BACKEND_URL non configuree sur Vercel',
        hint: 'Ajoute API_BACKEND_URL (URL publique de ton API FastAPI) dans les variables d’environnement Vercel.',
      },
      { status: 503 },
    );
  }

  if (base.includes('localhost') || base.includes('127.0.0.1')) {
    return NextResponse.json(
      {
        error: 'API_BACKEND_URL pointe vers localhost — inaccessible depuis Vercel',
        hint: 'Expose l’API (Railway, Render, Fly, ou ngrok) puis mets l’URL HTTPS publique.',
      },
      { status: 503 },
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'JSON invalide' }, { status: 400 });
  }

  const target = `${base}/api/v1/webhooks/tradingview`;

  try {
    const upstream = await fetch(target, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      // Claude peut prendre 30–60 s
      signal: AbortSignal.timeout(55_000),
    });

    const text = await upstream.text();
    const contentType = upstream.headers.get('content-type') || 'application/json';

    return new NextResponse(text, {
      status: upstream.status,
      headers: { 'Content-Type': contentType },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Proxy error';
    return NextResponse.json(
      {
        error: 'Echec du proxy vers l’API',
        detail: message,
        target,
      },
      { status: 502 },
    );
  }
}

export async function GET() {
  const base = backendBase();
  return NextResponse.json({
    service: 'tradingview-webhook-proxy',
    ready: Boolean(base) && !base.includes('localhost'),
    backend: base ? `${base}/api/v1/webhooks/tradingview` : null,
    webhook_url_for_tradingview:
      'https://<ton-projet>.vercel.app/api/webhooks/tradingview',
  });
}
