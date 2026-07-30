import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const siteUrl = "https://forecast.ste-scpb.com";
const siteName = "SCPB Market Forecast";
const siteDescription =
  "Prévisions professionnelles des prix cacao et café robusta — modèles hybrides IA (Prophet, XGBoost, N-HiTS, FinBERT).";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: siteName,
    template: `%s · ${siteName}`,
  },
  description: siteDescription,
  applicationName: siteName,
  keywords: [
    "SCPB",
    "SCPB Market Forecast",
    "cacao",
    "café",
    "robusta",
    "prédiction",
    "marché",
    "commodities",
    "ICE",
  ],
  authors: [{ name: "STE-SCPB" }],
  creator: "STE-SCPB",
  publisher: "STE-SCPB",
  robots: {
    index: true,
    follow: true,
  },
  icons: {
    icon: [{ url: "/logo.png", type: "image/png" }],
    apple: [{ url: "/logo.png", type: "image/png" }],
    shortcut: "/logo.png",
  },
  openGraph: {
    type: "website",
    locale: "fr_FR",
    url: siteUrl,
    siteName,
    title: siteName,
    description: siteDescription,
    images: [
      {
        url: "/opengraph-image",
        width: 1200,
        height: 630,
        alt: "SCPB Market Forecast — logo et prévisions marché",
      },
      {
        url: "/logo.png",
        width: 512,
        height: 512,
        alt: "Logo SCPB Market Forecast",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: siteName,
    description: siteDescription,
    images: ["/twitter-image", "/logo.png"],
  },
  alternates: {
    canonical: siteUrl,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fr" className={inter.variable}>
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
