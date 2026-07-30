import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatPrice(price: number): string {
  return new Intl.NumberFormat('fr-FR', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(price);
}

/** Convertit un prix USD/tonne en GBP/tonne (taux = GBP par 1 USD, i.e. USDGBP). */
export function usdToGbp(priceUsd: number, usdGbpRate: number): number {
  return priceUsd * usdGbpRate;
}

export function formatPriceGbp(priceUsd: number, usdGbpRate: number | null): string {
  if (usdGbpRate == null || !Number.isFinite(usdGbpRate) || usdGbpRate <= 0) {
    return 'n/d £';
  }
  // Format explicite « 3 844,12 £ » (évite ambiguïté $US / symbole locale)
  const gbp = usdToGbp(priceUsd, usdGbpRate);
  const amount = new Intl.NumberFormat('fr-FR', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(gbp);
  return `${amount} £`;
}

export function formatPercentage(value: number): string {
  return new Intl.NumberFormat('fr-FR', {
    style: 'percent',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
    signDisplay: 'always',
  }).format(value / 100);
}

export function getSentimentLabel(score: number | null): string {
  if (score === null) return 'N/A';
  if (score > 0.2) return 'Positif';
  if (score < -0.2) return 'Négatif';
  return 'Neutre';
}

export function getSentimentColor(score: number | null): string {
  if (score === null) return 'text-gray-500';
  if (score > 0.2) return 'text-green-600';
  if (score < -0.2) return 'text-red-600';
  return 'text-blue-600';
}
