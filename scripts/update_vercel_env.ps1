# Met à jour API_TOKEN (server-only) sur Vercel après rotation JWT.
# Prérequis: npm i -g vercel && vercel login && vercel link (dans frontend/)
#
# Usage (depuis la racine du repo):
#   powershell -File scripts\update_vercel_env.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$EnvFile = Join-Path $Root ".env"
$token = $null
Get-Content $EnvFile | ForEach-Object {
    if ($_ -match '^\s*API_TOKEN=(.+)$') { $token = $Matches[1].Trim().Trim('"') }
}
if (-not $token) { throw "API_TOKEN introuvable dans .env" }

Push-Location (Join-Path $Root "frontend")
try {
    Write-Host "Suppression ancienne NEXT_PUBLIC_API_TOKEN (si presente)..."
    npx --yes vercel env rm NEXT_PUBLIC_API_TOKEN production --yes 2>$null
    npx --yes vercel env rm NEXT_PUBLIC_API_TOKEN preview --yes 2>$null
    Write-Host "Ajout API_TOKEN (server)..."
    $token | npx --yes vercel env add API_TOKEN production
    $token | npx --yes vercel env add API_TOKEN preview
    Write-Host "Redeploy..."
    npx --yes vercel --prod --yes
} finally {
    Pop-Location
}
Write-Host "OK — verifiez https://vercel.com que API_TOKEN est set et NEXT_PUBLIC_API_TOKEN absente."
