# Webhook TradingView -> Cocoa/Coffee Intelligence

TradingView **calcule** la lecture de marche (tendance, momentum, niveaux, volumes)
sur le **futures cacao Londres `ICEEUR:C1!`**.
Ton API **recoit** ces infos via webhook, puis Claude **interprete** en style financier
(sans jargon technique).

Separation volontaire :
- **Dashboard (graphique)** : `PEPPERSTONE:COCOA` (CFD, affichage uniquement)
- **Alertes Pine + Brief IA** : `ICEEUR:C1!` (Londres)

```
TradingView (Pine)  --HTTP POST-->  FastAPI /api/v1/webhooks/tradingview
                                         |
                    +--------------------+--------------------+
                    |                    |                    |
              Modele hybride      Lecture de marche      News / sentiment
              (prix anticipes)    (tendance, momentum)   (Supabase)
                    |                    |                    |
                    +--------------------+--------------------+
                                         |
                                         v
                                  Claude Sonnet / Opus
                              (interprete, ne predit pas)
                                         |
                                         v
                                      Brief IA
```

## 1. Configuration serveur

`.env` :

```
TRADINGVIEW_WEBHOOK_SECRET=<32+ caracteres aleatoires>
ANTHROPIC_API_KEY=sk-ant-...
```

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
docker-compose up -d --force-recreate api
```

## 2. Table Supabase

Executer `scripts/sql/create_tradingview_alerts_table.sql` dans Supabase
(cree la table + colonnes lecture de marche).

## 3. Payload enrichi

```json
{
  "secret": "VOTRE_SECRET",
  "market": "ICE_NY",
  "signal_type": "support_break",
  "price": 4995.50,
  "tf": "1D",
  "ticker": "ICEEUR:C1!",
  "indicator": "CocoaTrack Market Reading",
  "message": "Cours sous le seuil des 5000",
  "timestamp": "2026-07-03T16:00:00Z",
  "trend": "bearish",
  "momentum": "strong_sell",
  "change_pct": -1.8,
  "rsi": 28.5,
  "price_vs_ma": "below",
  "support": 4980.0,
  "resistance": 5120.0,
  "volume_ratio": 1.7,
  "mode": "standard",
  "force_refresh": true
}
```

| Champ | Requis | Description |
|---|---|---|
| `secret` | oui | = `TRADINGVIEW_WEBHOOK_SECRET` |
| `market` | oui | `ICE_NY`, `COFFEE_ROBUSTA`, ou ticker TV |
| `signal_type` | oui | `buy`, `sell`, `support_break`, `resistance_break`, `trend_change` |
| `price` | non | Cours au moment du signal |
| `trend` | non | `bullish` / `bearish` / `neutral` |
| `momentum` | non | `strong_buy` / `buy` / `neutral` / `sell` / `strong_sell` |
| `change_pct` | non | Variation de seance (%) |
| `rsi` | non | 0-100 (traduit en langage marche avant Claude) |
| `price_vs_ma` | non | `above` / `below` / `mixed` |
| `support` / `resistance` | non | Niveaux proches |
| `volume_ratio` | non | 1.0 = volume moyen |
| `mode` | non | `standard` (Sonnet) ou `advanced` (Opus) |

Le backend **traduit** ces champs avant Claude :

| Champ brut | Lecture envoyee a Claude |
|---|---|
| `trend=bearish` | Tendance de fond: baissiere |
| `momentum=strong_sell` | Momentum court terme: tres vendeur |
| `rsi=28` | Dynamique des prix: zone de survente |
| `price_vs_ma=below` | Cours sous les moyennes de reference |
| `volume_ratio=1.7` | Volumes nettement superieurs a la moyenne |

Claude ne doit **jamais** ecrire RSI, MA, TradingView, Pine dans le brief.

## 4. Script Pine (cacao Londres)

Fichier : `scripts/pine/cocoa_market_reading.pine`

1. Ouvre TradingView → graphique sur **`ICEEUR:C1!`** (pas Pepperstone).
2. Timeframe recommande : **1D**.
3. Pine Editor → colle le script.
4. Remplace `REMPLACER_PAR_VOTRE_SECRET` par ton `TRADINGVIEW_WEBHOOK_SECRET`.
5. Save → Add to chart.
6. Si le fond est rouge leger : mauvais symbole — repasse sur `ICEEUR:C1!`.
7. Cree une alerte : condition = `Any alert() function call`.
8. Webhook URL (recommande via Vercel) :
   `https://<ton-projet>.vercel.app/api/webhooks/tradingview`
   (proxy vers l'API — voir `docs/deploy-vercel.md`).

   Direct API (sans Vercel) :
   `https://<api-publique>/api/v1/webhooks/tradingview`
   (local : `ngrok http 8000`).

Le payload force `"ticker":"ICEEUR:C1!"` et `"market":"ICE_NY"` (brief cacao du projet).
Le widget du dashboard reste `PEPPERSTONE:COCOA`.

## 5. Test manuel (sans TradingView)

```powershell
$body = @'
{
  "secret": "VOTRE_SECRET",
  "market": "ICE_NY",
  "signal_type": "support_break",
  "price": 4995.5,
  "trend": "bearish",
  "momentum": "strong_sell",
  "change_pct": -1.8,
  "rsi": 28.5,
  "price_vs_ma": "below",
  "support": 4980,
  "resistance": 5120,
  "volume_ratio": 1.7,
  "message": "Cours sous le seuil des 5000",
  "mode": "standard",
  "force_refresh": true
}
'@
$body | Out-File -Encoding ascii payload.json -NoNewline
curl.exe -s -X POST http://localhost:8000/api/v1/webhooks/tradingview `
  -H "Content-Type: application/json" `
  --data-binary "@payload.json"
```

Le brief doit mentionner pression vendeuse, tendance baissiere, volumes, niveaux —
**sans** citer RSI ni TradingView.

## 6. Erreurs

| Code | Cause |
|---|---|
| 401 | Secret invalide |
| 422 | Marche non reconnu |
| 429 | Quota Opus epuise (`mode=advanced`) |
| 503 | Secret non configure ou BriefService indisponible |

## 7. Securite

- Secret compare en temps constant (`hmac.compare_digest`).
- Pas de JWT (TradingView ne peut pas en envoyer).
- HTTPS obligatoire en production.
