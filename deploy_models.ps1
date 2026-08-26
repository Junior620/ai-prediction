# Deploy latest trained models to Contabo VPS and restart the API.
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\deploy_models.ps1
#   powershell -ExecutionPolicy Bypass -File .\deploy_models.ps1 -SkipRestart
#
# Config (optional file .env.deploy at repo root):
#   DEPLOY_VPS_HOST=169.58.99.28
#   DEPLOY_VPS_USER=root
#   DEPLOY_VPS_PATH=/opt/prediction
#   DEPLOY_API_HEALTH_URL=https://api.market.ste-scpb.com/health
#   DEPLOY_SSH_KEY=C:\Users\You\.ssh\id_ed25519   (optional)

param(
    [switch]$SkipRestart,
    [switch]$SkipHealth
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Load-DeployEnv {
    $envFile = Join-Path $Root ".env.deploy"
    if (-not (Test-Path $envFile)) { return }
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        $parts = $line -split "=", 2
        if ($parts.Count -ne 2) { return }
        $name = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        Set-Item -Path "Env:$name" -Value $value
    }
}

Load-DeployEnv

$HostName = if ($env:DEPLOY_VPS_HOST) { $env:DEPLOY_VPS_HOST } else { "169.58.99.28" }
$User = if ($env:DEPLOY_VPS_USER) { $env:DEPLOY_VPS_USER } else { "root" }
$RemotePath = if ($env:DEPLOY_VPS_PATH) { $env:DEPLOY_VPS_PATH } else { "/opt/prediction" }
$HealthUrl = if ($env:DEPLOY_API_HEALTH_URL) { $env:DEPLOY_API_HEALTH_URL } else { "https://api.market.ste-scpb.com/health" }
$SshTarget = "${User}@${HostName}"

$SshArgs = @(
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=30",
    "-o", "ServerAliveInterval=10",
    "-o", "ServerAliveCountMax=6",
    "-o", "IPQoS=none"
)
if ($env:DEPLOY_SSH_KEY -and (Test-Path $env:DEPLOY_SSH_KEY)) {
    Write-Host ('[INFO] Cle SSH: ' + $env:DEPLOY_SSH_KEY)
    $SshArgs = @("-i", $env:DEPLOY_SSH_KEY) + $SshArgs
} else {
    Write-Host '[AVERTISSEMENT] DEPLOY_SSH_KEY absent ou introuvable.'
    Write-Host '         Ajoute dans .env.deploy : DEPLOY_SSH_KEY=C:\Users\Christian\.ssh\id_ed25519'
}

function Invoke-Ssh([string]$RemoteCommand) {
    & ssh.exe @SshArgs $SshTarget $RemoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "SSH failed ($LASTEXITCODE): $RemoteCommand"
    }
}

function Invoke-Scp {
    param(
        [Parameter(Mandatory = $true)][string[]]$Sources,
        [Parameter(Mandatory = $true)][string]$Destination,
        [switch]$Recurse
    )
    # Un fichier a la fois : le scp Windows OpenSSH bloque souvent
    # sur le 2e fichier d'un transfert multiple (progress 0% ETA --).
    foreach ($src in $Sources) {
        $argsList = @("-O") + $SshArgs
        if ($Recurse) { $argsList += "-r" }
        $argsList += $src
        $argsList += "${SshTarget}:${Destination}"
        Write-Host ('[INFO] SCP ' + (Split-Path $src -Leaf) + ' -> ' + $Destination)
        & scp.exe @argsList
        if ($LASTEXITCODE -ne 0) {
            throw "SCP failed ($LASTEXITCODE) " + (Split-Path $src -Leaf) + " -> $Destination"
        }
    }
}

function Get-LatestFile([string]$Dir, [string]$Pattern) {
    if (-not (Test-Path $Dir)) { return $null }
    return Get-ChildItem -Path $Dir -Filter $Pattern -File -ErrorAction SilentlyContinue |
        Sort-Object Name |
        Select-Object -Last 1
}

function Get-LatestDir([string]$Dir, [string]$Pattern) {
    if (-not (Test-Path $Dir)) { return $null }
    return Get-ChildItem -Path $Dir -Directory -Filter $Pattern -ErrorAction SilentlyContinue |
        Sort-Object Name |
        Select-Object -Last 1
}

Write-Host ""
Write-Host "================================================================================"
Write-Host ("  DEPLOY MODELES -> VPS (" + $SshTarget + ")")
Write-Host "================================================================================"
Write-Host ""

Write-Host '[INFO] Test SSH...'
try {
    Invoke-Ssh "echo OK"
} catch {
    throw ("SSH vers " + $SshTarget + " impossible. Verifie la cle dans .env.deploy (DEPLOY_SSH_KEY) et: ssh " + $SshTarget)
}

# --- Discover latest artifacts ---
$cocoaDir = Join-Path $Root "models"
$robustaDir = Join-Path $Root "models\coffee_robusta"

$cocoaProphet = Get-LatestFile $cocoaDir "prophet_improved_*.pkl"
$cocoaXgb = Get-LatestFile $cocoaDir "xgboost_improved_*.pkl"
$cocoaNhits = Get-LatestDir $cocoaDir "nhits_*"

$robustaProphet = Get-LatestFile $robustaDir "prophet_improved_*.pkl"
$robustaXgb = Get-LatestFile $robustaDir "xgboost_improved_*.pkl"
$robustaNhits = Get-LatestDir $robustaDir "nhits_*"

if (-not $cocoaProphet -or -not $cocoaXgb) {
    throw "Modeles cacao introuvables (prophet/xgboost) dans models/"
}
if (-not $robustaProphet -or -not $robustaXgb) {
    throw "Modeles robusta introuvables (prophet/xgboost) dans models/coffee_robusta/"
}

Write-Host ('[INFO] Cacao Prophet : ' + $cocoaProphet.Name)
Write-Host ('[INFO] Cacao XGBoost: ' + $cocoaXgb.Name)
$cocoaNhitsLabel = if ($cocoaNhits) { $cocoaNhits.Name } else { '(aucun)' }
Write-Host ('[INFO] Cacao N-HiTS : ' + $cocoaNhitsLabel)
Write-Host ('[INFO] Robusta Prophet : ' + $robustaProphet.Name)
Write-Host ('[INFO] Robusta XGBoost: ' + $robustaXgb.Name)
$robustaNhitsLabel = if ($robustaNhits) { $robustaNhits.Name } else { '(aucun)' }
Write-Host ('[INFO] Robusta N-HiTS : ' + $robustaNhitsLabel)
Write-Host ""

# --- Ensure remote dirs ---
Write-Host '[INFO] Preparation des dossiers distants...'
Invoke-Ssh "mkdir -p $RemotePath/models/coffee_robusta $RemotePath/models/futures $RemotePath/config/coffee_robusta"

# --- Upload models ---
Write-Host '[INFO] Upload modeles cacao...'
Invoke-Scp -Sources @($cocoaProphet.FullName, $cocoaXgb.FullName) -Destination "$RemotePath/models/"
if ($cocoaNhits) {
    Invoke-Scp -Recurse -Sources @($cocoaNhits.FullName) -Destination "$RemotePath/models/"
}

Write-Host '[INFO] Upload modeles robusta...'
Invoke-Scp -Sources @($robustaProphet.FullName, $robustaXgb.FullName) -Destination "$RemotePath/models/coffee_robusta/"
if ($robustaNhits) {
    Invoke-Scp -Recurse -Sources @($robustaNhits.FullName) -Destination "$RemotePath/models/coffee_robusta/"
}

$futuresDir = Join-Path $Root "models\futures"
if (Test-Path $futuresDir) {
    Write-Host '[INFO] Upload modeles futures (courbe a terme)...'
    $futuresFiles = Get-ChildItem -Path $futuresDir -File -ErrorAction SilentlyContinue
    foreach ($ff in $futuresFiles) {
        Invoke-Scp -Sources @($ff.FullName) -Destination "$RemotePath/models/futures/"
    }
}

# --- Upload config weights / conformal (if present) ---
$configFiles = @(
    @{ Local = "config\ensemble_weights.json"; Remote = "$RemotePath/config/" },
    @{ Local = "config\conformal_intervals.json"; Remote = "$RemotePath/config/" },
    @{ Local = "config\coffee_robusta\ensemble_weights.json"; Remote = "$RemotePath/config/coffee_robusta/" },
    @{ Local = "config\coffee_robusta\conformal_intervals.json"; Remote = "$RemotePath/config/coffee_robusta/" }
)
foreach ($item in $configFiles) {
    $localPath = Join-Path $Root $item.Local
    if (Test-Path $localPath) {
        Write-Host ('[INFO] Upload ' + $item.Local)
        Invoke-Scp -Sources @($localPath) -Destination $item.Remote
    }
}

# --- Flush prediction cache + restart API ---
if (-not $SkipRestart) {
    Write-Host '[INFO] Flush Redis (predictions cache)...'
    try {
        Invoke-Ssh "cd $RemotePath && docker compose exec -T redis redis-cli FLUSHDB"
    } catch {
        Write-Host ('[AVERTISSEMENT] Flush Redis echoue (non bloquant): ' + $_.Exception.Message)
    }
    Write-Host '[INFO] Redemarrage API Docker sur le VPS...'
    Invoke-Ssh "cd $RemotePath && docker compose restart api"
    # FinBERT + modeles peuvent prendre 45-90s avant /health 200 via nginx
    Write-Host '[INFO] Attente demarrage API (45s)...'
    Start-Sleep -Seconds 45
}

# --- Health check (retries: 502 pendant le boot FinBERT est normal) ---
if (-not $SkipHealth) {
    Write-Host ('[INFO] Health check: ' + $HealthUrl)
    $healthOk = $false
    $lastErr = $null
    for ($i = 1; $i -le 12; $i++) {
        try {
            $health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 20
            $markets = ($health.markets_loaded -join ", ")
            if ($health.status -eq "healthy" -and $health.services.price_predictor) {
                Write-Host ('[OK] status=' + $health.status + '  markets=' + $markets + '  predictor=' + $health.services.price_predictor)
                $healthOk = $true
                break
            }
            Write-Host ('[INFO] API pas encore healthy... (' + $i + '/12)')
        } catch {
            $lastErr = $_.Exception.Message
            Write-Host ('[INFO] Health pas pret (' + $i + '/12): ' + $lastErr)
        }
        Start-Sleep -Seconds 10
    }
    if (-not $healthOk) {
        Write-Host ('[AVERTISSEMENT] Health check echoue apres retries: ' + $lastErr)
        Write-Host ('         Verifier: ssh ' + $SshTarget + " 'cd $RemotePath && docker compose logs api --tail 40'")
        exit 2
    }
}

Write-Host ""
Write-Host '[OK] Deploy VPS termine'
Write-Host "================================================================================"
exit 0
