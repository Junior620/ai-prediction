# M3 vs M3+USD/GBP (hors production)

- Généré : 2026-09-01T03:16:25.889370Z
- Split : 2025-01-01
- Marché : cocoa (GBP/MT)

| Modèle | Horizon | MAPE % | MAE | RMSE | n |
|---|---:|---:|---:|---:|---:|
| M3_OHLCV_OI | J+1 | 5.79 | 242.7 | 327.0 | 414 |
| M3_OHLCV_OI | J+7 | 8.75 | 360.8 | 463.4 | 407 |
| M3_OHLCV_OI | J+30 | 18.91 | 745.5 | 928.1 | 385 |
| M3_plus_USDGBP | J+1 | 6.32 | 261.6 | 344.6 | 414 |
| M3_plus_USDGBP | J+7 | 9.06 | 367.5 | 475.7 | 407 |
| M3_plus_USDGBP | J+30 | 19.21 | 746.7 | 930.5 | 385 |

## Décision

Pas de lift clair J+1 (M3=5.79%, M3+FX=6.32%) — prod reste M3 pur.

> Ce rapport n'active **pas** USD/GBP en production.
