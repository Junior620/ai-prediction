import { ImageResponse } from 'next/og';

export const runtime = 'edge';
export const alt = 'SCPB Market Forecast';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export default async function Image() {
  let logoData: ArrayBuffer | null = null;
  try {
    const res = await fetch('https://forecast.ste-scpb.com/logo.png');
    if (res.ok) logoData = await res.arrayBuffer();
  } catch {
    logoData = null;
  }

  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'linear-gradient(145deg, #06091a 0%, #0d1528 55%, #132033 100%)',
          position: 'relative',
        }}
      >
        <div
          style={{
            position: 'absolute',
            inset: 0,
            background:
              'radial-gradient(circle at 30% 20%, rgba(253,184,39,0.18), transparent 45%), radial-gradient(circle at 75% 80%, rgba(148,173,69,0.14), transparent 40%)',
          }}
        />
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 28,
            zIndex: 1,
          }}
        >
          {logoData ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={logoData as unknown as string}
              width={200}
              height={200}
              alt="SCPB"
              style={{
                borderRadius: 28,
                boxShadow: '0 20px 60px rgba(0,0,0,0.45)',
              }}
            />
          ) : (
            <div
              style={{
                width: 200,
                height: 200,
                borderRadius: 28,
                background: 'linear-gradient(160deg, #fdb827 0%, #94ad45 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#06091a',
                fontSize: 48,
                fontWeight: 800,
                letterSpacing: '0.04em',
                boxShadow: '0 20px 60px rgba(0,0,0,0.45)',
              }}
            >
              SCPB
            </div>
          )}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 12,
            }}
          >
            <div
              style={{
                fontSize: 64,
                fontWeight: 700,
                color: '#ffffff',
                letterSpacing: '-0.03em',
                lineHeight: 1.1,
              }}
            >
              SCPB Market Forecast
            </div>
            <div
              style={{
                fontSize: 28,
                color: '#94a3b8',
                letterSpacing: '0.02em',
              }}
            >
              Prévisions cacao & café robusta
            </div>
          </div>
        </div>
        <div
          style={{
            position: 'absolute',
            bottom: 40,
            display: 'flex',
            color: '#64748b',
            fontSize: 22,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
          }}
        >
          forecast.ste-scpb.com
        </div>
      </div>
    ),
    { ...size },
  );
}
