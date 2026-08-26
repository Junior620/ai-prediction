import { NextResponse } from 'next/server';

/**
 * Returns the server-side API_TOKEN for WebSocket auth only.
 * Not baked into the client bundle; still visible in Network once fetched.
 */
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  const token = process.env.API_TOKEN;
  const apiUrl =
    process.env.NEXT_PUBLIC_API_URL ||
    process.env.API_BACKEND_URL ||
    '';

  if (!token) {
    return NextResponse.json(
      { error: 'API_TOKEN non configure' },
      { status: 503 },
    );
  }
  if (!apiUrl) {
    return NextResponse.json(
      { error: 'NEXT_PUBLIC_API_URL / API_BACKEND_URL manquant' },
      { status: 503 },
    );
  }

  return NextResponse.json({
    token,
    apiUrl: apiUrl.replace(/\/$/, ''),
  });
}
