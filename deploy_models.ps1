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
        $leaf = Split-Path $src -Leaf
        $ok = $false
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            Write-Host ('[INFO] SCP ' + $leaf + ' -> ' + $Destination + ' (essai ' + $attempt + '/3)')
            & scp.exe @argsList
            if ($LASTEXITCODE -eq 0) {
                $ok = $true
                break
            }
            Start-Sleep -Seconds (5 * $attempt)
        }
        if (-not $ok) {
            throw "SCP failed after retries: $leaf -> $Destination"
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
Invoke-Ssh "mkdir -p $RemotePath/models/coffee_robusta $RemotePath/models/futures $RemotePath/models/futures_london $RemotePath/models/futures_london_named $RemotePath/config/coffee_robusta $RemotePath/src/models"

# --- Upload models ---
Write-Host '[INFO] Upload modeles cacao...'
Invoke-Scp -Sources @($cocoaProphet.FullName, $cocoaXgb.FullName) -Destination "$RemotePath/models/"
$cocoaInfo = Get-LatestFile $cocoaDir "model_info_improved_*.json"
if ($cocoaInfo) {
    Invoke-Scp -Sources @($cocoaInfo.FullName) -Destination "$RemotePath/models/"
}
$cocoaH7 = Get-LatestFile $cocoaDir "xgboost_h7_*.pkl"
$cocoaH14 = Get-LatestFile $cocoaDir "xgboost_h14_*.pkl"
$cocoaH30 = Get-LatestFile $cocoaDir "xgboost_h30_*.pkl"
$cocoaDirectInfo = Get-LatestFile $cocoaDir "model_info_direct_horizon_*.json"
$directUploads = @()
if ($cocoaH7) { $directUploads += $cocoaH7.FullName }
if ($cocoaH14) { $directUploads += $cocoaH14.FullName }
if ($cocoaH30) { $directUploads += $cocoaH30.FullName }
if ($cocoaDirectInfo) { $directUploads += $cocoaDirectInfo.FullName }
if ($directUploads.Count -gt 0) {
    Write-Host '[INFO] Upload modeles direct-horizon cacao...'
    Invoke-Scp -Sources $directUploads -Destination "$RemotePath/models/"
}
if ($cocoaNhits) {
    Invoke-Scp -Recurse -Sources @($cocoaNhits.FullName) -Destination "$RemotePath/models/"
}

Write-Host '[INFO] Upload modeles robusta...'
Invoke-Scp -Sources @($robustaProphet.FullName, $robustaXgb.FullName) -Destination "$RemotePath/models/coffee_robusta/"
$robustaH7 = Get-LatestFile $robustaDir "xgboost_h7_*.pkl"
$robustaH14 = Get-LatestFile $robustaDir "xgboost_h14_*.pkl"
$robustaH30 = Get-LatestFile $robustaDir "xgboost_h30_*.pkl"
$robustaDirectInfo = Get-LatestFile $robustaDir "model_info_direct_horizon_*.json"
$robustaDirectUploads = @()
if ($robustaH7) { $robustaDirectUploads += $robustaH7.FullName }
if ($robustaH14) { $robustaDirectUploads += $robustaH14.FullName }
if ($robustaH30) { $robustaDirectUploads += $robustaH30.FullName }
if ($robustaDirectInfo) { $robustaDirectUploads += $robustaDirectInfo.FullName }
if ($robustaDirectUploads.Count -gt 0) {
    Write-Host '[INFO] Upload modeles direct-horizon robusta...'
    Invoke-Scp -Sources $robustaDirectUploads -Destination "$RemotePath/models/coffee_robusta/"
}
if ($robustaNhits) {
    Invoke-Scp -Recurse -Sources @($robustaNhits.FullName) -Destination "$RemotePath/models/coffee_robusta/"
}

$futuresDir = Join-Path $Root "models\futures"
if (Test-Path $futuresDir) {
    Write-Host '[INFO] Upload modeles futures NY (fallback)...'
    $futuresFiles = Get-ChildItem -Path $futuresDir -File -ErrorAction SilentlyContinue
    foreach ($ff in $futuresFiles) {
        Invoke-Scp -Sources @($ff.FullName) -Destination "$RemotePath/models/futures/"
    }
}

$futuresLondonDir = Join-Path $Root "models\futures_london"
if (Test-Path $futuresLondonDir) {
    Write-Host '[INFO] Upload modeles futures Londres (GBP)...'
    $flFiles = Get-ChildItem -Path $futuresLondonDir -File -ErrorAction SilentlyContinue
    foreach ($ff in $flFiles) {
        Invoke-Scp -Sources @($ff.FullName) -Destination "$RemotePath/models/futures_london/"
    }
}

$futuresNamedDir = Join-Path $Root "models\futures_london_named"
if (Test-Path $futuresNamedDir) {
    Write-Host '[INFO] Upload modeles futures Londres mois nommes (DEC26…)...'
    $fnFiles = Get-ChildItem -Path $futuresNamedDir -File -ErrorAction SilentlyContinue
    foreach ($ff in $fnFiles) {
        Invoke-Scp -Sources @($ff.FullName) -Destination "$RemotePath/models/futures_london_named/"
    }
}

# --- Upload model code needed for M3 / London futures ---
$modelCodeFiles = @(
    @{ Local = "src\models\futures_curve_predictor.py"; Remote = "$RemotePath/src/models/" },
    @{ Local = "src\models\improved_price_predictor.py"; Remote = "$RemotePath/src/models/" },
    @{ Local = "src\models\hybrid_features.py"; Remote = "$RemotePath/src/models/" },
    @{ Local = "src\models\hybrid_trainer.py"; Remote = "$RemotePath/src/models/" },
    @{ Local = "src\models\direct_horizon_trainer.py"; Remote = "$RemotePath/src/models/" },
    @{ Local = "src\models\multi_step_predictor.py"; Remote = "$RemotePath/src/models/" },
    @{ Local = "src\validation\report_loader.py"; Remote = "$RemotePath/src/validation/" },
    @{ Local = "docker-compose.yml"; Remote = "$RemotePath/" }
)
foreach ($item in $modelCodeFiles) {
    $localPath = Join-Path $Root $item.Local
    if (Test-Path $localPath) {
        Write-Host ('[INFO] Upload ' + $item.Local)
        Invoke-Scp -Sources @($localPath) -Destination $item.Remote
    }
}

# --- Upload config (market registry + weights / conformal) ---
$configFiles = @(
    @{ Local = "config\config.yaml"; Remote = "$RemotePath/config/" },
    @{ Local = "config\settings.py"; Remote = "$RemotePath/config/" },
    @{ Local = "config\ensemble_weights.json"; Remote = "$RemotePath/config/" },
    @{ Local = "config\conformal_intervals.json"; Remote = "$RemotePath/config/" },
    @{ Local = "config\model_comparison_latest.json"; Remote = "$RemotePath/config/" },
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

# --- Upload API modules (nouveaux endpoints dashboard) ---
$apiFiles = @(
    @{ Local = "src\api\app.py"; Remote = "$RemotePath/src/api/" },
    @{ Local = "src\api\models.py"; Remote = "$RemotePath/src/api/" }
)
foreach ($item in $apiFiles) {
    $localPath = Join-Path $Root $item.Local
    if (Test-Path $localPath) {
        Write-Host ('[INFO] Upload ' + $item.Local)
        Invoke-Scp -Sources @($localPath) -Destination $item.Remote
    }
}

# --- Upload walk-forward summaries (dashboard Performance) ---
$wfDir = Join-Path $Root "reports\walk_forward"
if (Test-Path $wfDir) {
    $summaryFiles = Get-ChildItem -Path $wfDir -Filter "*_summary.json" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 6
    if ($summaryFiles) {
        Write-Host '[INFO] Upload reports/walk_forward summaries...'
        Invoke-Ssh "mkdir -p $RemotePath/reports/walk_forward"
        Invoke-Scp -Sources @($summaryFiles.FullName) -Destination "$RemotePath/reports/walk_forward/"
    }
}

# --- Upload model comparison JSON (dashboard M1-M4) ---
$cmpLocal = Join-Path $Root "config\model_comparison_latest.json"
if (Test-Path $cmpLocal) {
    Write-Host '[INFO] Upload config/model_comparison_latest.json'
    Invoke-Scp -Sources @($cmpLocal) -Destination "$RemotePath/config/"
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
