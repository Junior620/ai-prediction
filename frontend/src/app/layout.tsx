import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Cocoa Intelligence Platform — AI-Powered Price Predictions",
  description:
    "Real-time cocoa price predictions powered by a hybrid AI model (Prophet + XGBoost + FinBERT). ICE New York market analytics dashboard.",
  keywords: ["cocoa", "prediction", "AI", "trading", "commodity", "dashboard"],
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
