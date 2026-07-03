# Deploiement Vercel (frontend) + Webhook TradingView

## Architecture

```
Navigateur  ──►  Vercel (Next.js dashboard)
                      │
                      │ NEXT_PUBLIC_API_URL
                      ▼
                 API FastAPI (Railway / Render / Fly / VPS)
                      ▲
TradingView ──HTTPS───┘
   (ou via proxy Vercel : /api/webhooks/tradingview)
```

**Important :** Vercel heberge le **frontend uniquement**.
L’API ML (Docker, modeles, Redis, Claude) ne tourne **pas** sur Vercel
(trop lourde pour du serverless). Elle doit etre publique en HTTPS.

## 1. Preparer le frontend

Root Directory sur Vercel : `frontend`

Fichiers deja prets :
- `frontend/vercel.json`
- `frontend/src/app/api/webhooks/tradingview/route.ts` (proxy webhook)
- `frontend/.env.example`

## 2. Deployer sur Vercel

### Option A — CLI

```powershell
cd E:\prediction\frontend
npm i -g vercel
vercel login
vercel
```

Puis production :

```powershell
vercel --prod
```

### Option B — Dashboard

1. [vercel.com](https://vercel.com) → **Add New Project**
2. Importe le repo Git (ou upload)
3. **Root Directory** : `frontend`
4. Framework : Next.js (auto)
5. Ajoute les variables d’environnement (ci-dessous)
6. **Deploy**

## 3. Variables d’environnement Vercel

Project → Settings → Environment Variables :

| Variable | Exemple | Visible navigateur |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `https://api.ton-domaine.com` | oui |
| `NEXT_PUBLIC_API_TOKEN` | JWT long-lived | oui |
| `API_BACKEND_URL` | `https://api.ton-domaine.com` | non (server) |

`API_BACKEND_URL` sert au **proxy webhook**. Souvent identique a `NEXT_PUBLIC_API_URL`.

Apres modification des env : **Redeploy**.

## 4. URL Webhook TradingView

Une fois deploye, ton URL publique est :

```text
https://<ton-projet>.vercel.app/api/webhooks/tradingview
```

Exemple :

```text
https://cocoa-intelligence.vercel.app/api/webhooks/tradingview
```

Colle cette URL dans TradingView → Alerte → **Webhook URL**.

Verification rapide (navigateur ou curl) :

```text
GET https://<ton-projet>.vercel.app/api/webhooks/tradingview
```

Reponse attendue : `{ "ready": true, "backend": "https://.../api/v1/webhooks/tradingview" }`

Si `ready: false` → `API_BACKEND_URL` manque ou pointe vers localhost.

## 5. API FastAPI publique (obligatoire pour le webhook)

Tant que l’API est seulement sur `localhost:8000`, **ni Vercel ni TradingView** ne peuvent l’atteindre.

Options :

| Option | Usage |
|---|---|
| **ngrok** (dev) | `ngrok http 8000` → mets l’URL `https://xxx.ngrok-free.app` dans `API_BACKEND_URL` et `NEXT_PUBLIC_API_URL` |
| **Railway / Render / Fly.io** | Deploy Docker `cocoa-api` en prod |
| **VPS + HTTPS** | Nginx + Let’s Encrypt |

Pour un test rapide avec ngrok :

```powershell
ngrok http 8000
```

Puis dans Vercel :

```
NEXT_PUBLIC_API_URL=https://xxxx.ngrok-free.app
API_BACKEND_URL=https://xxxx.ngrok-free.app
```

Redeploy Vercel, puis webhook TradingView :

```
https://<ton-projet>.vercel.app/api/webhooks/tradingview
```

## 6. Timeouts

Le brief Claude peut prendre 30–60 s.

- Le proxy Vercel a `maxDuration = 60`
- Sur le plan **Hobby**, la limite peut etre plus basse (10 s) → le webhook peut timeout
- Plan **Pro** recommande pour les briefs via webhook

Si timeout : l’API peut quand meme finir le brief (cache Redis) ; TradingView affichera une erreur d’envoi.

## 7. Checklist

- [ ] Frontend deploye sur Vercel (`frontend` = root)
- [ ] `NEXT_PUBLIC_API_URL` = API HTTPS publique
- [ ] `NEXT_PUBLIC_API_TOKEN` = JWT valide
- [ ] `API_BACKEND_URL` = meme API HTTPS
- [ ] `GET /api/webhooks/tradingview` → `ready: true`
- [ ] Pine sur `ICEEUR:C1!` avec le secret
- [ ] Alerte TradingView : Any alert() function call + Webhook URL Vercel

## 8. Separation des symboles (rappel)

| Surface | Symbole |
|---|---|
| Graphique dashboard (Vercel) | `PEPPERSTONE:COCOA` |
| Pine / alertes / Brief IA | `ICEEUR:C1!` |
