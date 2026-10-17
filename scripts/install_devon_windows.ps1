<#
.SYNOPSIS
    Devon (nomorals-2.0) Windows installer for RDP / EC2 / workstation hosts.

.DESCRIPTION
    Installs everything Devon needs on Windows, profile-aware:
      - Preflight: OS / CPU / RAM / disk / GPU -> LOW / MEDIUM / HIGH tier.
        Tier selects pip extras. It never gates capability:
        the full codebase runs on every tier. LLM is API-only on all
        tiers. LOW = CPU-only small host (e.g. t3.small/medium EC2, no GPU).
      - System deps via winget (Python 3.11, Git, ffmpeg, espeak-ng,
        VC++ redist, nano). No admin required where possible; winget
        itself may prompt for elevation for per-machine installs.
      - Python venv + pip extras by tier.
      - LLM via API keys ONLY (Groq / HF / OpenRouter / ...). NO GGUF,
        NO local LLM, NO llama.cpp, NO Ollama - owner decision: API keys
        for now, till better hosting is affordable.
      - Optional tiny local extras: Piper TTS (small, CPU) always;
        local Whisper-tiny only with -IncludeLocalWhisper (default OFF).
        XTTS / diffusion / RVC are SKIPPED on RDP - API fallback / deferred.
      - .env from a template with ALL config keys, placeholders only,
        never real secrets. Existing .env is never overwritten unless
        -Force (then it is backed up first).
      - Verification: `python -m nomorals --help` + `nm doctor`,
        then a capability matrix: what works / what is degraded.

.PARAMETER SkipLocalExtras
    Skip even the tiny local fallbacks (Piper TTS). API-only.

.PARAMETER IncludeLocalWhisper
    Also download local Whisper-tiny as an STT fallback. Default OFF;
    STT goes through API keys.

.PARAMETER SkipSystemDeps
    Skip winget installs (deps already present).


.PARAMETER Force
    Overwrite an existing .env (backed up first) and re-download models.

.PARAMETER DryRun
    Print what would happen, change nothing.

.PARAMETER RepoPath
    Path to the nomorals-2.0 checkout. Default: current directory.

.EXAMPLE
    .\scripts\install_devon_windows.ps1
    .\scripts\install_devon_windows.ps1 -DryRun
    .\scripts\install_devon_windows.ps1 -IncludeLocalWhisper

.NOTES
    After install, edit config with:  notepad .env
    (or, if nano was installed:    nano .env)
    Then start Devon:  .\.venv\Scripts\python.exe -m nomorals
#>
[CmdletBinding()]
param(
    [switch]$SkipLocalExtras,
    [switch]$IncludeLocalWhisper,
    [switch]$SkipSystemDeps,
    [switch]$Force,
    [switch]$DryRun,
    [string]$RepoPath = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# -- Tier thresholds (single source of truth; tests read these) -------------
$script:LOW_RAM_GB = 4.5      # <= this RAM (GB) -> LOW, unless GPU present
$script:MEDIUM_RAM_GB = 8.0    # < this RAM -> MEDIUM, >= HIGH_RAM_GB -> HIGH
$script:HIGH_RAM_GB = 16.0
$script:LOW_CPU = 2           # < this vCPU count -> LOW
$script:MIN_DISK_FREE_GB = 10 # local extras need <1 GB; warn below

# -- Local extras catalogue (API-first) --------------------------------------
# OWNER DECISION: NO GGUF, NO local LLM, NO llama.cpp, NO Ollama.
# LLM / STT / TTS / diffusion all go through API keys (.env) for now,
# till better hosting is affordable. Profile = API-first, online-first;
# offline is fallback, not primary.
# The only local downloads are tiny fallbacks:
#   Piper TTS (~0.1 GB, CPU) - always, so voice works with no keys.
#   Whisper-tiny (~0.1 GB) - ONLY with -IncludeLocalWhisper (default OFF).
# XTTS / diffusion / RVC / Demucs are NOT downloaded on RDP - they need
# better hosting. API fallback for now; the script says so, never silently.
$script:LocalExtras = @{
    piper   = @{ Id = 'rhasspy/piper-voices'; File = 'en/en_US/lessac/medium/en_US-lessac-medium.onnx'; SizeGB = 0.1; Role = 'TTS fallback (Piper, small, CPU)' }
    whisper = @{ Id = 'openai/whisper-tiny'; File = $null; SizeGB = 0.1; Role = 'STT fallback (Whisper-tiny, opt-in only)' }
}

$script:PipExtrasByTier = @{
    LOW    = 'fast,config,hub,media,docs,chat'
    MEDIUM = 'fast,config,hub,media,media-edit,docs,voice,chat,ta,datasci,genimg,browser'
    HIGH   = 'fast,config,hub,media,media-edit,docs,voice,chat,ta,datasci,genimg,browser,train,cloud,dev'
}

function Write-Step { param([string]$Text) Write-Host "`n==> $Text" -ForegroundColor Cyan }
function Write-Warn { param([string]$Text) Write-Host "WARNING: $Text" -ForegroundColor Yellow }
function Write-Ok   { param([string]$Text) Write-Host "OK: $Text" -ForegroundColor Green }

function Get-Preflight {
    $os = Get-CimInstance Win32_OperatingSystem
    $cs = Get-CimInstance Win32_ComputerSystem
    $ramGB = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
    $cpu = [Environment]::ProcessorCount
    $drive = Get-PSDrive -Name C -ErrorAction SilentlyContinue
    $diskFreeGB = if ($drive) { [math]::Round($drive.Free / 1GB, 1) } else { 0 }
    $gpu = $null; $gpuName = 'none'
    try {
        $nvidia = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
        if ($nvidia) { $gpuName = $nvidia.Trim(); $gpu = $gpuName }
    } catch { }
    if (-not $gpu) {
        try {
            $vc = Get-CimInstance Win32_VideoController | Select-Object -First 1 -ExpandProperty Name
            if ($vc -and $vc -notmatch 'Microsoft Basic|Standard VGA') { $gpuName = $vc }
        } catch { }
    }
    $pyVersion = $null
    try { $pyVersion = (& python --version 2>&1).ToString().Trim() } catch { }
    return [ordered]@{
        OS = "$($os.Caption) $($os.Version)"; CPU = $cpu; RAM_GB = $ramGB
        DiskFreeGB = $diskFreeGB; GPU = $gpuName; HasNvidia = [bool]$gpu
        Python = ($pyVersion -as [string]); Arch = [Environment]::Is64BitOperatingSystem
    }
}

function Get-Tier {
    param([int]$Cpu, [double]$RamGB, [bool]$HasGpu)
    # GPU lifts one tier (GPU host with 8GB RAM behaves like HIGH for models).
    if ($HasGpu -and $RamGB -ge $script:MEDIUM_RAM_GB) { return 'HIGH' }
    if ($Cpu -lt $script:LOW_CPU -or $RamGB -le $script:LOW_RAM_GB) { return 'LOW' }
    if ($RamGB -ge $script:HIGH_RAM_GB) { return 'HIGH' }
    return 'MEDIUM'
}

function Install-WingetPackage {
    param([string]$Id, [string]$Name)
    if ($DryRun) { Write-Host "  [dry-run] winget install --id $Id -e --accept-source-agreements --accept-package-agreements"; return }
    $present = winget list --id $Id -e 2>$null | Select-String -SimpleMatch $Id
    if ($present) { Write-Ok "$Name already installed"; return }
    Write-Host "  Installing $Name ($Id)..."
    winget install --id $Id -e --accept-source-agreements --accept-package-agreements
}

function New-EnvFile {
    param([string]$Repo, [switch]$ForceOverwrite)
    $envPath = Join-Path $Repo '.env'
    $examplePath = Join-Path $Repo '.env.example'
    if ((Test-Path $envPath) -and -not $ForceOverwrite) {
        Write-Ok ".env already exists (kept). Use -Force to regenerate (backed up first)."
        return
    }
    if (Test-Path $envPath) {
        $bak = "$envPath.bak-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
        Copy-Item $envPath $bak; Write-Host "  Backed up existing .env to $bak"
    }
    $header = @"
# Devon (.env) - generated by scripts/install_devon_windows.ps1
# Edit with:  notepad .env        (or:  nano .env  if nano is installed)
# NEVER put real secrets in chat, screenshots, or git. .env is gitignored.
# Put API keys in ONLY via the secure credential flow / this file on your
# own machine. Never paste a key into chat.
# API-FIRST: LLM / STT / TTS / diffusion all use API keys below.
# NO local LLM, NO GGUF - owner decision, till better hosting is affordable.

# -- API KEYS FIRST ---------------------------------------------------------
# Fill in only the keys you use. Devon boots against the Mock provider
# with none set. At least one LLM key is recommended on RDP (no local LLM).
# NM_LLM_PROVIDER=groq
# NM_GROQ_API_KEY=
# NM_GROQ_MODEL=openai/gpt-oss-120b
# NM_OPENROUTER_API_KEY=
# NM_OPENROUTER_MODEL=qwen/qwen3-8b:free
# NM_OPENAI_API_KEY=
# NM_OPENAI_BASE_URL=
# NM_OPENAI_MODEL=
# HF_TOKEN=
# NM_HF_MODEL=
# NM_LLM_FALLBACK_CHAIN=groq,openrouter
# NM_EMBED_PROVIDER=hashing
# NM_EMBED_MODEL=
# NM_VOICE_STT_PROVIDER=api
# NM_VOICE_TTS_PROVIDER=api
# NM_IMAGE_PROVIDER=api


"@
    $body = ''
    if (Test-Path $examplePath) { $body = Get-Content $examplePath -Raw }
    $extra = @"

# -- Chat: Telegram (gated bot + personal bot) ------------------------------
# NM_CHAT_TELEGRAM_ENABLED=0
# NM_CHAT_TELEGRAM_API_ID=
# NM_CHAT_TELEGRAM_API_HASH=
# NM_CHAT_TELEGRAM_SESSION=
# NM_CHAT_TELEGRAM_CHATS=
# NM_CHAT_TELEGRAM_BOT_ENABLED=1
# NM_CHAT_TELEGRAM_BOT_TOKEN=
# NM_CHAT_TELEGRAM_BOT_CHATS=
# NM_CHAT_TELEGRAM_BOT_PARTNER_CONTEXT=0
# NM_CHAT_TELEGRAM_PERSONAL_BOT_ENABLED=0
# NM_CHAT_TELEGRAM_PERSONAL_BOT_TOKEN=
# NM_CHAT_TELEGRAM_PERSONAL_BOT_CHATS=
# NM_CHAT_TELEGRAM_PERSONAL_BOT_PARTNER_CONTEXT=1
# NM_CHAT_DISCORD_ENABLED=0
# NM_CHAT_DISCORD_TOKEN=
# NM_CHAT_DISCORD_CHANNELS=
# NM_CHAT_WHATSAPP_OUTBOX_DIR=
# NM_CHAT_MEDIA_IN_GROUPS=1
# NM_CHAT_MEDIA_MAX_MB=20

# -- LLM: local + cloud providers -------------------------------------------
# NM_LLM_PROVIDER=mock
# NM_LLM_LOCAL_MODEL=
# NM_LLM_LOCAL_AUTO_START=0
# NM_LLM_LOCAL_HOST=127.0.0.1
# NM_LLM_LOCAL_PORT=8080
# NM_LLM_TIMEOUT=120
# NM_LLM_VERIFY_AT_STARTUP=0
# NM_LLM_CACHE_DIR=
# NM_LLM_ACTIVE_MODEL=

# -- Models: voice / vision / music -----------------------------------------
# NM_VOICE_TTS_ENGINE=piper
# NM_VOICE_STT_MODEL=tiny
# NM_MUSIC_MODEL=
# NM_IMAGE_MODEL=

# -- Trading (GoldenSwing lives in its own repo; Devon never trades live) --
# NM_TRADING_ENABLED=0
# NM_TRADING_MODE=paper
# NM_TRADING_ALLOW_LIVE=0

# -- Partner / owner --------------------------------------------------------
# NM_PARTNER_OWNER_CHATS=
# NM_PARTNER_PLATFORMS=telegram-bot
# NM_PARTNER_PROACTIVE_ENABLED=0
# NM_PARTNER_QUIET_START=23:00
# NM_PARTNER_QUIET_END=07:00
# NM_PARTNER_PERSONALITY=

# -- API / hub --------------------------------------------------------------
# NM_API_HOST=127.0.0.1
# NM_API_PORT=8731
# NM_API_TOKEN=
# NM_HUB_URL=
# NM_HUB_TOKEN=

# -- OSINT / security -------------------------------------------------------
# NM_OSINT_ENABLED=0
# NM_OSINT_SHODAN_KEY=
# NM_OSINT_ABUSEIPDB_KEY=
# NM_OSINT_HIBP_KEY=
# NM_NET_ENABLED=0
# NM_NET_PROXY=

# -- Offline / budgets ------------------------------------------------------
# NM_OFFLINE=0
# NM_BUDGET_WALL_SECONDS=3600
# NM_BUDGET_TOKENS=2000000
# NM_BUDGET_CHILDREN=256
"@
    $content = $header + $body + $extra
    if ($DryRun) { Write-Host "  [dry-run] would write .env ($($content.Length) chars)"; return }
    Set-Content -Path $envPath -Value $content -Encoding UTF8
    Write-Ok "Wrote .env (placeholders only). Edit with: notepad .env"
}

function New-ConfigTemplates {
    param([string]$Repo)
    $cfgDir = Join-Path $Repo 'config'
    if (-not (Test-Path $cfgDir)) { New-Item -ItemType Directory -Path $cfgDir | Out-Null }
    $templates = @{
        'models.yaml'  = "# API-FIRST (owner decision): NO local LLM, NO GGUF, till better hosting.`n# LLM / STT / TTS / diffusion all via API keys in .env.`n# Local fallbacks only: Piper TTS (always), Whisper-tiny (opt-in).`n# XTTS / diffusion / RVC: deferred - needs better hosting.`nllm:`n  provider: api  # groq | openrouter | openai | hf`n  local_llm: none  # deliberately none`nstt:`n  provider: api`n  local_fallback: whisper-tiny  # opt-in only`ntts:`n  provider: api`n  local_fallback: piper`ndiffusion:`n  provider: api  # local diffusion deferred`n"
        'chat.yaml'    = "# Chat adapters. Enable one at a time; Telegram bot is the default.`ntelegram_bot:`n  enabled: true`n  partner_context: false`ntelegram_personal_bot:`n  enabled: false`n  partner_context: true`ndiscord:`n  enabled: false`n"
        'trading.yaml' = "# Devon never trades live. Paper only, always.`n# Real trading lives in the GoldenSwing repo on the same host.`nenabled: false`nmode: paper`nallow_live: false  # NEVER set true from Devon`n"
    }
    foreach ($name in $templates.Keys) {
        $p = Join-Path $cfgDir $name
        if (-not (Test-Path $p)) { Set-Content $p $templates[$name] -Encoding UTF8; Write-Ok "Wrote config/$name" }
        else { Write-Ok "config/$name already exists (kept)" }
    }
}

function Invoke-Verification {
    param([string]$Repo, [string]$Tier)
    $venvPy = Join-Path $Repo '.venv\Scripts\python.exe'
    if (-not (Test-Path $venvPy)) { Write-Warn 'venv python not found - verification skipped.'; return }
    Write-Step 'Verification'
    try {
        $helpOut = & $venvPy -m nomorals --help 2>&1 | Select-Object -First 5
        Write-Ok "python -m nomorals --help works: $($helpOut -join ' ')"
    } catch { Write-Warn "python -m nomorals --help failed: $_" }
    try {
        & $venvPy -m nomorals.cli doctor 2>&1 | Select-Object -First 20 | ForEach-Object { Write-Host "  $_" }
    } catch { Write-Warn "nm doctor failed: $_ (Devon still runs; see logs)" }
    Write-Host ''
    Write-Host 'Capability matrix for this host:' -ForegroundColor Cyan
    $matrix = @(
        @{ Cap = 'Chat / tools / games / books / learn / legal / commitments / immune'; State = 'WORKS' }
        @{ Cap = 'Telegram / Discord / WhatsApp adapters'; State = 'WORKS once .env keys are set' }
        @{ Cap = 'Local LLM (GGUF / llama.cpp / Ollama)'; State = 'NOT INSTALLED - owner decision: API keys only' }
        @{ Cap = 'LLM via API (Groq / OpenRouter / OpenAI / HF)'; State = 'WORKS once .env keys are set' }
        @{ Cap = 'Speech-to-text / TTS'; State = 'API first; Piper local fallback; Whisper-tiny only if -IncludeLocalWhisper' }
        @{ Cap = 'Voice cloning (XTTS)'; State = 'SKIPPED on RDP - needs better hosting, API fallback for now' }
        @{ Cap = 'Image diffusion'; State = 'API / DEFERRED - local diffusion needs better hosting' }
        @{ Cap = 'Video generation'; State = 'DEFERRED - needs better hosting' }
        @{ Cap = 'RVC / Demucs / stems'; State = 'DEFERRED - needs better hosting' }
        @{ Cap = 'Trading'; State = 'PAPER ONLY - real trading lives in GoldenSwing, never Devon' }
    )
    foreach ($row in $matrix) { Write-Host ("  [{0}] {1}" -f $row.State, $row.Cap) }
}

# -- Main --------------------------------------------------------------------
Write-Host 'Devon (nomorals-2.0) Windows installer' -ForegroundColor Green
Write-Host "Repo: $RepoPath"
if ($DryRun) { Write-Host '[DRY RUN - nothing will change]' -ForegroundColor Yellow }

$pre = Get-Preflight
Write-Step 'Preflight'
Write-Host "  OS:   $($pre.OS)"
Write-Host "  CPU:  $($pre.CPU) vCPU   RAM: $($pre.RAM_GB) GB   Disk free (C:): $($pre.DiskFreeGB) GB"
Write-Host "  GPU:  $($pre.GPU)"
Write-Host "  Python: $($pre.Python)"
$tier = Get-Tier -Cpu $pre.CPU -RamGB $pre.RAM_GB -HasGpu $pre.HasNvidia
Write-Host "  Tier: $tier" -ForegroundColor Green
if ($pre.DiskFreeGB -lt $script:MIN_DISK_FREE_GB) { Write-Warn "Less than $($script:MIN_DISK_FREE_GB) GB free on C:. Local extras need <1 GB. API keys do the heavy lifting." }

# Local extras (tiny fallbacks only - API-first, no GGUF, no local LLM)
if (-not $SkipLocalExtras) {
    Write-Step 'Local extras (tiny fallbacks only - API-first)'
    Write-Host '  Piper TTS (~0.1 GB) - local fallback so voice works with no keys.'
    if ($IncludeLocalWhisper) { Write-Host '  Whisper-tiny (~0.1 GB) - opt-in local STT fallback.' }
    else { Write-Host '  Whisper-tiny: skipped (default OFF - STT via API). Use -IncludeLocalWhisper to add it.' }
    Write-Host '  XTTS / diffusion / RVC: SKIPPED - needs better hosting, API fallback for now.'
} else { Write-Host 'Local extras: skipped (-SkipLocalExtras)' }

# System deps
if (-not $SkipSystemDeps) {
    Write-Step 'System dependencies (winget)'
    try { $winget = Get-Command winget -ErrorAction Stop; Write-Ok "winget found: $($winget.Source)" }
    catch { Write-Warn 'winget not found. Install App Installer from the Microsoft Store, then re-run. Skipping system deps.'; $SkipSystemDeps = $true }
}
if (-not $SkipSystemDeps) {
    Install-WingetPackage -Id 'Python.Python.3.11' -Name 'Python 3.11'
    Install-WingetPackage -Id 'Git.Git' -Name 'Git'
    Install-WingetPackage -Id 'Gyan.FFmpeg' -Name 'ffmpeg'
    Install-WingetPackage -Id 'eSpeak-NG.eSpeak-NG' -Name 'espeak-ng'
    Install-WingetPackage -Id 'Microsoft.VCRedist.2015+.x64' -Name 'Visual C++ Redistributable'
    Install-WingetPackage -Id 'GNU.Nano' -Name 'nano (editor for .env)'
    Write-Host '  Refreshing PATH for this session...'
    $env:Path = [System.Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path', 'User')
}

# Python venv + deps
Write-Step 'Python venv + dependencies'
$venvPy = Join-Path $RepoPath '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) {
    if ($DryRun) { Write-Host '  [dry-run] python -m venv .venv' }
    else { & python -m venv (Join-Path $RepoPath '.venv'); Write-Ok 'Created .venv' }
} else { Write-Ok '.venv already exists' }
$extras = $script:PipExtrasByTier[$tier]
if ($DryRun) { Write-Host "  [dry-run] .venv\Scripts\python.exe -m pip install -e .[$extras]" }
else {
    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install -e ".[$extras]"
    Write-Ok "Installed nomorals[$extras]"
}

# Local extras download (Piper always, Whisper-tiny opt-in)
if (-not $SkipLocalExtras) {
    Write-Step 'Downloading local extras (Piper + opt-in Whisper-tiny)'
    $extrasDir = Join-Path $RepoPath 'models'
    if (-not (Test-Path $extrasDir)) { if (-not $DryRun) { New-Item -ItemType Directory -Path $extrasDir | Out-Null } }
    $toFetch = @($script:LocalExtras['piper'])
    if ($IncludeLocalWhisper) { $toFetch += $script:LocalExtras['whisper'] }
    foreach ($m in $toFetch) {
        $dest = Join-Path $extrasDir ($m.Id -replace '/', '__')
        if ($DryRun) { Write-Host "  [dry-run] hf download $($m.Id) -> $dest"; continue }
        if ((Test-Path $dest) -and -not $Force) { Write-Ok "$($m.Id) already present (use -Force to re-download)"; continue }
        Write-Host "  Downloading $($m.Id) (~$($m.SizeGB) GB)..."
        try {
            if ($m.File) { & $venvPy -m huggingface_hub.commands.huggingface_cli download $m.Id $m.File --local-dir $dest --local-dir-use-symlinks False }
            else { & $venvPy -m huggingface_hub.commands.huggingface_cli download $m.Id --local-dir $dest --local-dir-use-symlinks False }
            Write-Ok "Downloaded $($m.Id)"
        } catch {
            Write-Warn "Failed to download $($m.Id): $_. Continuing - Devon runs without it (API fallback), re-run to retry."
        }
    }
    $manifest = @{ tier = $tier; profile = 'api-first'; local_llm = 'none'; gguf = 'none'; downloaded_at = (Get-Date -Format 'o'); extras = $toFetch }
    if (-not $DryRun) { $manifest | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $extrasDir 'manifest.json') -Encoding UTF8 }
}

# Env + config templates
Write-Step 'Config (.env + config/*.yaml)'
if ($DryRun) { Write-Host '  [dry-run] would write .env + config templates' } else { New-EnvFile -Repo $RepoPath -ForceOverwrite:$Force; New-ConfigTemplates -Repo $RepoPath }

if (-not $DryRun) { Invoke-Verification -Repo $RepoPath -Tier $tier }

Write-Host ''
Write-Host 'Done. Next steps:' -ForegroundColor Green
Write-Host '  1. notepad .env        (or: nano .env)  - fill in only the keys you use.'
Write-Host '  2. .\.venv\Scripts\python.exe -m nomorals --help'
Write-Host '  3. .\.venv\Scripts\python.exe -m nomorals.cli doctor'
Write-Host '  4. Start Devon: .\.venv\Scripts\python.exe -m nomorals'
Write-Host '  LLM is API-only: set at least one key in .env (Groq / OpenRouter / OpenAI / HF).'
Write-Host '  Do NOT run GoldenSwing backtests and heavy Devon jobs at the same time on a small RDP host - one heavy job at a time.'
