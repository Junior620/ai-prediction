# Comparaison modeles cacao Londres (Databento)

- Split date : `2025-01-01`
- Train : 1526 jours | Test : 414 jours
- Genere : 2026-08-31T19:02:21.348840

| Modele | Horizon | MAE | RMSE | MAPE (%) | N |
|---|---:|---:|---:|---:|---:|
| M1_Prophet_Close | J+1 | 5465.80 | 6139.10 | 155.05 | 404 |
| M1_Prophet_Close | J+7 | 5607.59 | 6234.47 | 159.70 | 391 |
| M1_Prophet_Close | J+30 | 5790.54 | 6353.34 | 165.46 | 375 |
| M2_XGB_OHLCV | J+1 | 299.63 | 399.25 | 7.52 | 404 |
| M2_XGB_OHLCV | J+7 | 374.74 | 491.20 | 9.82 | 391 |
| M2_XGB_OHLCV | J+30 | 690.28 | 879.59 | 18.45 | 375 |
| M3_XGB_OHLCV_OI | J+1 | 287.19 | 384.88 | 6.96 | 404 |
| M3_XGB_OHLCV_OI | J+7 | 367.21 | 482.77 | 9.40 | 391 |
| M3_XGB_OHLCV_OI | J+30 | 698.72 | 888.63 | 18.62 | 375 |
| M4_XGB_Full | J+1 | 342.67 | 427.84 | 8.34 | 404 |
| M4_XGB_Full | J+7 | 391.76 | 508.27 | 10.22 | 391 |
| M4_XGB_Full | J+30 | 701.17 | 881.14 | 18.85 | 375 |

> M1=Prophet(Close) · M2=XGB(OHLCV) · M3=XGB(OHLCV+OI) · M4=XGB(complet+sentiment)