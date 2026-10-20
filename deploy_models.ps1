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
    "-o", "TCPKeepAlive=yes",
    "-o", "IPQoS=none"
)
if ($env:DEPLOY_SSH_KEY -and (Test-Path $env:DEPLOY_SSH_KEY)) {
    Write-Host ('[INFO] Cle SSH: ' + $env:DEPLOY_SSH_KEY)
    $SshArgs = @("-i", $env:DEPLOY_SSH_KEY) + $SshArgs
} else {
    Write-Host '[AVERTISSEMENT] DEPLOY_SSH_KEY absent ou introuvable.'
    Write-Host '         Ajoute dans .env.deploy : DEPLOY_SSH_KEY=C:\Users\Christian\.ssh\id_ed25519'
}

function Format-ProcessArgs([string[]]$ArgumentList) {
    # Start-Process/ProcessStartInfo need a single argv string on Windows PS 5.1.
    ($ArgumentList | ForEach-Object {
        $a = [string]$_
        if ($a -match '[\s"]') {
            '"' + ($a.Replace('"', '\"')) + '"'
        } else {
            $a
        }
    }) -join ' '
}

function Invoke-ProcessWithTimeout {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [int]$TimeoutSec = 90
    )
    # System.Diagnostics.Process (pas Start-Process) : ExitCode fiable apres WaitForExit.
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $FilePath
    $psi.Arguments = Format-ProcessArgs $ArgumentList
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true

    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi
    [void]$p.Start()

    $outTask = $p.StandardOutput.ReadToEndAsync()
    $errTask = $p.StandardError.ReadToEndAsync()

    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        try { $p.Kill() } catch {}
        throw "Timeout apres ${TimeoutSec}s: $(Split-Path $FilePath -Leaf)"
    }

    $out = $outTask.Result
    $err = $errTask.Result
    if ($out) {
        ($out -split "`r?`n") | Where-Object { $_ -ne "" } | ForEach-Object { Write-Host $_ }
    }
    if ($err) {
        ($err -split "`r?`n") | Where-Object { $_ -ne "" } | ForEach-Object { Write-Host $_ }
    }
    return [int]$p.ExitCode
}

function Invoke-Ssh([string]$RemoteCommand, [int]$TimeoutSec = 180) {
    $last = "SSH failed: $RemoteCommand"
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            $code = Invoke-ProcessWithTimeout -FilePath "ssh.exe" `
                -ArgumentList ($SshArgs + @($SshTarget, $RemoteCommand)) `
                -TimeoutSec $TimeoutSec
            if ($code -eq 0) { return }
            $last = "SSH failed ($code): $RemoteCommand"
        } catch {
            $last = $_.Exception.Message
        }
        Write-Host ('[AVERTISSEMENT] ' + $last + ' (essai ' + $attempt + '/5)')
        if ($attempt -lt 5) {
            $wait = [Math]::Min(60, 8 * $attempt)
            Write-Host ("[INFO] Pause ${wait}s avant retry SSH...")
            Start-Sleep -Seconds $wait
        }
    }
    throw $last
}

function Invoke-Scp {
    param(
        [Parameter(Mandatory = $true)][string[]]$Sources,
        [Parameter(Mandatory = $true)][string]$Destination,
        [switch]$Recurse,
        [int]$TimeoutSec = 300,
        [switch]$AllowFail
    )
    # Un fichier a la fois : le scp Windows OpenSSH bloque souvent
    # sur le 2e fichier d'un transfert multiple (progress 0% ETA --).
    # Contabo coupe parfois la session (Connection reset) : retries + pause.
    foreach ($src in $Sources) {
        $argsList = @("-O") + $SshArgs
        if ($Recurse) { $argsList += "-r" }
        $argsList += $src
        $argsList += "${SshTarget}:${Destination}"
        $leaf = Split-Path $src -Leaf
        $ok = $false
        for ($attempt = 1; $attempt -le 5; $attempt++) {
            Write-Host ('[INFO] SCP ' + $leaf + ' -> ' + $Destination + ' (essai ' + $attempt + '/5)')
            try {
                $code = Invoke-ProcessWithTimeout -FilePath "scp.exe" `
                    -ArgumentList $argsList -TimeoutSec $TimeoutSec
                if ($code -eq 0) {
                    $ok = $true
                    Start-Sleep -Milliseconds 400
                    break
                }
                Write-Host ("[AVERTISSEMENT] scp exit code " + $code)
            } catch {
                Write-Host ('[AVERTISSEMENT] ' + $_.Exception.Message)
            }
            $wait = [Math]::Min(60, 8 * $attempt)
            Write-Host ("[INFO] Pause ${wait}s avant retry SCP...")
            Start-Sleep -Seconds $wait
        }
        if (-not $ok) {
            $msg = "SCP failed after retries: $leaf -> $Destination"
            if ($AllowFail) {
                Write-Host ('[AVERTISSEMENT] ' + $msg + ' (non bloquant)')
            } else {
                throw $msg
            }
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
    # Contabo peut etre lent (RTT 300ms+) ou temporairement sature : 90s
    Invoke-Ssh "echo OK" -TimeoutSec 90
} catch {
    throw ("SSH vers " + $SshTarget + " impossible (" + $_.Exception.Message + "). Verifie DEPLOY_SSH_KEY dans .env.deploy et: ssh " + $SshTarget)
}

# --- Artefacts nommés par le manifeste, jamais le fichier le plus récent ---
function Get-ActiveRelease([string]$Market) {
    $path = Join-Path $Root ("config\active_release_" + $Market + ".json")
    if (-not (Test-Path $path)) { throw ("Manifeste actif absent: " + $path) }
    return Get-Content $path -Raw -Encoding UTF8 | ConvertFrom-Json
}
function Resolve-ReleaseFile($Release, [string]$Key) {
    $relative = $Release.artifacts.$Key
    if (-not $relative) { throw ("Artefact absent du manifeste: " + $Key) }
    return Join-Path $Root (($relative -replace '/', '\'))
}
function Send-RepoFile([string]$Relative) {
    $normalized = ($Relative -replace '\\', '/').TrimStart('/')
    $local = Join-Path $Root ($normalized -replace '/', '\')
    if (-not (Test-Path $local)) { throw ("Fichier absent: " + $normalized) }
    $remoteParent = Split-Path $normalized -Parent
    $remoteDir = if ($remoteParent) { "$RemotePath/" + ($remoteParent -replace '\\', '/') } else { "$RemotePath" }
    if ($null -eq $script:PreparedRemoteDirs) { $script:PreparedRemoteDirs = @{} }
    if (-not $script:PreparedRemoteDirs.ContainsKey($remoteDir)) {
        Invoke-Ssh ("mkdir -p '" + $remoteDir + "'")
        $script:PreparedRemoteDirs[$remoteDir] = $true
    }
    Invoke-Scp -Sources @($local) -Destination ($remoteDir + "/")
}
$cocoaRelease = Get-ActiveRelease "cocoa"
$robustaRelease = Get-ActiveRelease "coffee_robusta"
$cocoaProphet = Get-Item (Resolve-ReleaseFile $cocoaRelease "prophet")
$cocoaXgb = Get-Item (Resolve-ReleaseFile $cocoaRelease "xgboost")
$cocoaNhitsPath = Resolve-ReleaseFile $cocoaRelease "nhits"
$cocoaNhits = if (Test-Path $cocoaNhitsPath) { Get-Item $cocoaNhitsPath } else { $null }
$robustaProphet = Get-Item (Resolve-ReleaseFile $robustaRelease "prophet")
$robustaXgb = Get-Item (Resolve-ReleaseFile $robustaRelease "xgboost")
$robustaNhitsPath = Resolve-ReleaseFile $robustaRelease "nhits"
$robustaNhits = if (Test-Path $robustaNhitsPath) { Get-Item $robustaNhitsPath } else { $null }
$cocoaDir = $cocoaProphet.Directory.FullName
$robustaDir = $robustaProphet.Directory.FullName

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
Invoke-Ssh "mkdir -p $RemotePath/models/coffee_robusta $RemotePath/models/futures $RemotePath/models/futures_london $RemotePath/models/futures_london_named $RemotePath/config/coffee_robusta $RemotePath/src/models $RemotePath/src/monitoring $RemotePath/src/validation $RemotePath/src/api $RemotePath/src/data_collection $RemotePath/data"
$script:PreparedRemoteDirs = @{
    "$RemotePath/models" = $true
    "$RemotePath/models/coffee_robusta" = $true
    "$RemotePath/models/futures" = $true
    "$RemotePath/models/futures_london" = $true
    "$RemotePath/models/futures_london_named" = $true
    "$RemotePath/config" = $true
    "$RemotePath/config/coffee_robusta" = $true
    "$RemotePath/src/models" = $true
    "$RemotePath/src/monitoring" = $true
    "$RemotePath/src/validation" = $true
    "$RemotePath/src/data_collection" = $true
    "$RemotePath/src/api" = $true
    "$RemotePath" = $true
}

# --- Mode de publication explicite sur le VPS ---
# docker-compose.yml refuse de demarrer sans PREDICTION_RELEASE_MODE : le site public sert le
# manifeste actif, jamais un mode retombe par defaut.
# grep : 0 ligne presente, 1 absente, 2 fichier illisible. SSH coupe = 255 ou delai depasse :
# on retente, le VPS peut etre lent (voir le test SSH a 90 s).
$modeCheck = $null
for ($attempt = 1; $attempt -le 3; $attempt++) {
    try {
        $modeCheck = Invoke-ProcessWithTimeout -FilePath "ssh.exe" `
            -ArgumentList ($SshArgs + @($SshTarget, "grep -qx 'PREDICTION_RELEASE_MODE=active' $RemotePath/.env")) `
            -TimeoutSec 120
    } catch {
        $modeCheck = $null
        Write-Host ('[AVERTISSEMENT] Controle du mode : ' + $_.Exception.Message + ' (essai ' + $attempt + '/3)')
    }
    if ($null -ne $modeCheck -and $modeCheck -ne 255) { break }
    if ($attempt -lt 3) { Start-Sleep -Seconds (10 * $attempt) }
}
if ($modeCheck -eq 1 -or $modeCheck -eq 2) {
    Write-Host "[ERREUR] $RemotePath/.env ne contient pas la ligne PREDICTION_RELEASE_MODE=active."
    Write-Host "         Ajoutez-la sur le VPS avant de deployer (rien n'a ete envoye)."
    exit 3
}
if ($modeCheck -ne 0) {
    Write-Host "[ERREUR] SSH vers le VPS coupe ou trop lent pendant le controle du mode (3 essais)."
    Write-Host "         Rien n'a ete envoye. Relancez deploy_models.bat plus tard."
    exit 5
}

# --- Upload models ---
Write-Host '[INFO] Upload modeles cacao...'
Send-RepoFile $cocoaRelease.artifacts.prophet
Send-RepoFile $cocoaRelease.artifacts.xgboost
$cocoaInfo = Get-Item (Resolve-ReleaseFile $cocoaRelease "improved_info")
if ($cocoaInfo) {
    Invoke-Scp -Sources @($cocoaInfo.FullName) -Destination "$RemotePath/models/"
}
$cocoaDirectInfo = Get-Item (Resolve-ReleaseFile $cocoaRelease "direct_info")
$cocoaDirectMeta = Get-Content $cocoaDirectInfo.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
Write-Host '[INFO] Upload modeles direct-horizon cacao (chemin du manifeste conserve)...'
foreach ($horizonKey in @("1", "7", "14", "30")) {
    Send-RepoFile $cocoaDirectMeta.horizons.$horizonKey.model_path
}
Send-RepoFile $cocoaRelease.artifacts.direct_info
if ($cocoaNhits) {
    Invoke-Scp -Recurse -Sources @($cocoaNhits.FullName) -Destination "$RemotePath/models/"
}

Write-Host '[INFO] Upload modeles robusta...'
Send-RepoFile $robustaRelease.artifacts.prophet
Send-RepoFile $robustaRelease.artifacts.xgboost
$robustaDirectInfo = Get-Item (Resolve-ReleaseFile $robustaRelease "direct_info")
$robustaDirectMeta = Get-Content $robustaDirectInfo.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
$robustaInfo = Get-Item (Resolve-ReleaseFile $robustaRelease "improved_info")
Invoke-Scp -Sources @($robustaInfo.FullName) -Destination "$RemotePath/models/coffee_robusta/"
Write-Host '[INFO] Upload modeles direct-horizon robusta (chemin du manifeste conserve)...'
foreach ($horizonKey in @("1", "7", "14", "30")) {
    Send-RepoFile $robustaDirectMeta.horizons.$horizonKey.model_path
}
Send-RepoFile $robustaRelease.artifacts.direct_info
if ($robustaNhits) {
    Invoke-Scp -Recurse -Sources @($robustaNhits.FullName) -Destination "$RemotePath/models/coffee_robusta/"
}

$futuresDir = Join-Path $Root "models\futures"
if (Test-Path $futuresDir) {
    Write-Host '[INFO] Upload modeles futures NY (fallback)...'
    $futuresFiles = Get-ChildItem -Path $futuresDir -File -ErrorAction SilentlyContinue
    foreach ($ff in $futuresFiles) {
        Invoke-Scp -Sources @($ff.FullName) -Destination "$RemotePath/models/futures/" -AllowFail
    }
}

$futuresLondonDir = Join-Path $Root "models\futures_london"
if (Test-Path $futuresLondonDir) {
    Write-Host '[INFO] Upload modeles futures Londres (GBP)...'
    $flFiles = Get-ChildItem -Path $futuresLondonDir -File -ErrorAction SilentlyContinue
    foreach ($ff in $flFiles) {
        Invoke-Scp -Sources @($ff.FullName) -Destination "$RemotePath/models/futures_london/" -AllowFail
    }
}

$futuresNamedDir = Join-Path $Root "models\futures_london_named"
if (Test-Path $futuresNamedDir) {
    Write-Host '[INFO] Upload modeles futures Londres mois nommes (DEC26…)...'
    $fnFiles = Get-ChildItem -Path $futuresNamedDir -File -ErrorAction SilentlyContinue
    foreach ($ff in $fnFiles) {
        Invoke-Scp -Sources @($ff.FullName) -Destination "$RemotePath/models/futures_london_named/" -AllowFail
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
    @{ Local = "src\models\served_forecast.py"; Remote = "$RemotePath/src/models/" },
    @{ Local = "src\models\release_manifest.py"; Remote = "$RemotePath/src/models/" },
    @{ Local = "src\models\release_calendar.py"; Remote = "$RemotePath/src/models/" },
    @{ Local = "src\models\revision_command.py"; Remote = "$RemotePath/src/models/" },
    @{ Local = "src\models\horizon_schedule.py"; Remote = "$RemotePath/src/models/" },
    @{ Local = "src\models\prediction_journal.py"; Remote = "$RemotePath/src/models/" },
    @{ Local = "src\models\nhits_replay.py"; Remote = "$RemotePath/src/models/" },
    @{ Local = "src\models\weather_features.py"; Remote = "$RemotePath/src/models/" },
    @{ Local = "src\models\ensemble_weights.py"; Remote = "$RemotePath/src/models/" },
    @{ Local = "src\models\conformal_intervals.py"; Remote = "$RemotePath/src/models/" },
    @{ Local = "src\validation\acceptance.py"; Remote = "$RemotePath/src/validation/" },
    @{ Local = "src\validation\served_backtest.py"; Remote = "$RemotePath/src/validation/" },
    @{ Local = "src\data_collection\cocoa_weather.py"; Remote = "$RemotePath/src/data_collection/" },
    @{ Local = "src\data_collection\cocoa_roll.py"; Remote = "$RemotePath/src/data_collection/" },
    # Série cacao sans saut : l'API la lit à chaque préparation des prix.
    @{ Local = "data\cocoa_roll_series.csv"; Remote = "$RemotePath/data/" },
    @{ Local = "src\monitoring\performance_monitor.py"; Remote = "$RemotePath/src/monitoring/" },
    # /health lit la fraicheur ; les alertes partent par webhook ou SMTP.
    @{ Local = "src\monitoring\freshness.py"; Remote = "$RemotePath/src/monitoring/" },
    @{ Local = "src\monitoring\alert_system.py"; Remote = "$RemotePath/src/monitoring/" },
    @{ Local = "src\validation\report_loader.py"; Remote = "$RemotePath/src/validation/" },
    # Importe par src/api/revision_jobs.py : sans lui le VPS calcule /revision-value avec l'ancienne version.
    @{ Local = "src\validation\revision_value.py"; Remote = "$RemotePath/src/validation/" },
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
    @{ Local = "config\acceptance.json"; Remote = "$RemotePath/config/" },
    @{ Local = "config\exchange_holidays.json"; Remote = "$RemotePath/config/" },
    @{ Local = "config\active_release_cocoa.json"; Remote = "$RemotePath/config/" },
    @{ Local = "config\active_release_coffee_robusta.json"; Remote = "$RemotePath/config/" },
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
    @{ Local = "src\api\revision_jobs.py"; Remote = "$RemotePath/src/api/" },
    @{ Local = "src\api\models.py"; Remote = "$RemotePath/src/api/" },
    @{ Local = "src\api\cache.py"; Remote = "$RemotePath/src/api/" }
)
foreach ($item in $apiFiles) {
    $localPath = Join-Path $Root $item.Local
    if (Test-Path $localPath) {
        Write-Host ('[INFO] Upload ' + $item.Local)
        Invoke-Scp -Sources @($localPath) -Destination $item.Remote
    }
}

# --- Upload walk-forward summaries (dashboard Performance) ---
# Non bloquant: un SSH/SCP hung ne doit pas empecher Redis flush + restart API.
$wfDir = Join-Path $Root "reports\walk_forward"
if (Test-Path $wfDir) {
    $summaryFiles = Get-ChildItem -Path $wfDir -Filter "*_summary.json" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 2
    if ($summaryFiles) {
        Write-Host '[INFO] Upload reports/walk_forward summaries...'
        try {
            Invoke-Ssh "mkdir -p $RemotePath/reports/walk_forward" -TimeoutSec 45
            Invoke-Scp -Sources @($summaryFiles.FullName) `
                -Destination "$RemotePath/reports/walk_forward/" -TimeoutSec 60
        } catch {
            Write-Host ('[AVERTISSEMENT] Upload reports ignore (non bloquant): ' + $_.Exception.Message)
        }
    }
    $coffeeWf = Join-Path $wfDir "coffee_robusta"
    if (Test-Path $coffeeWf) {
        $coffeeSummaries = Get-ChildItem -Path $coffeeWf -Filter "*_summary.json" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 2
        if ($coffeeSummaries) {
            Write-Host '[INFO] Upload reports/walk_forward/coffee_robusta summaries...'
            try {
                Invoke-Ssh "mkdir -p $RemotePath/reports/walk_forward/coffee_robusta" -TimeoutSec 45
                Invoke-Scp -Sources @($coffeeSummaries.FullName) `
                    -Destination "$RemotePath/reports/walk_forward/coffee_robusta/" -TimeoutSec 60
            } catch {
                Write-Host ('[AVERTISSEMENT] Upload reports cafe ignore (non bloquant): ' + $_.Exception.Message)
            }
        }
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
                $servedMode = $health.release_mode
                $servedCocoa = $health.serving.cocoa.model_version
                if ($servedMode -ne "active" -or $servedCocoa -ne $cocoaRelease.version) {
                    Write-Host ('[ERREUR] Le VPS sert mode=' + $servedMode + ' lot=' + $servedCocoa + ', attendu active / ' + $cocoaRelease.version)
                    exit 4
                }
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
