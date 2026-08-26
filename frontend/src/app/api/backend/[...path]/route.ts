import { NextRequest, NextResponse } from 'next/server';

/**
 * BFF proxy: browser → /api/backend/... → API_BACKEND_URL/...
 * JWT stays server-side (API_TOKEN), never NEXT_PUBLIC_*.
 */
export const runtime = 'nodejs';
export const maxDuration = 60;
export const dynamic = 'force-dynamic';

function backendBase(): string {
  const base =
    process.env.API_BACKEND_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    'http://localhost:8000';
  return base.replace(/\/$/, '');
}

function apiToken(): string | undefined {
  return process.env.API_TOKEN || undefined;
}

async function proxy(req: NextRequest, pathSegments: string[]) {
  const base = backendBase();
  const token = apiToken();
  if (!token) {
    return NextResponse.json(
      {
        error: 'API_TOKEN non configure (variable serveur, pas NEXT_PUBLIC_)',
      },
      { status: 503 },
    );
  }

  const subpath = pathSegments.join('/');
  const url = new URL(req.url);
  const target = `${base}/${subpath}${url.search}`;

  const headers: Record<string, string> = {
    Authorization: `Bearer ${token}`,
  };
  const contentType = req.headers.get('content-type');
  if (contentType) headers['Content-Type'] = contentType;

  const init: RequestInit = {
    method: req.method,
    headers,
    signal: AbortSignal.timeout(55_000),
  };

  if (req.method !== 'GET' && req.method !== 'HEAD') {
    const body = await req.text();
    if (body) init.body = body;
  }

  try {
    const upstream = await fetch(target, init);
    const text = await upstream.text();
    const ct = upstream.headers.get('content-type') || 'application/json';
    return new NextResponse(text, {
      status: upstream.status,
      headers: { 'Content-Type': ct },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Proxy error';
    return NextResponse.json(
      { error: 'Echec du proxy vers l’API', detail: message, target },
      { status: 502 },
    );
  }
}

type Ctx = { params: { path: string[] } };

export async function GET(req: NextRequest, ctx: Ctx) {
  return proxy(req, ctx.params.path);
}

export async function POST(req: NextRequest, ctx: Ctx) {
  return proxy(req, ctx.params.path);
}

export async function PUT(req: NextRequest, ctx: Ctx) {
  return proxy(req, ctx.params.path);
}

export async function PATCH(req: NextRequest, ctx: Ctx) {
  return proxy(req, ctx.params.path);
}

export async function DELETE(req: NextRequest, ctx: Ctx) {
  return proxy(req, ctx.params.path);
}
