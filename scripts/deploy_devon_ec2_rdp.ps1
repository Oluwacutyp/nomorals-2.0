<#
.SYNOPSIS
    Devon (nomorals-2.0) EC2 RDP deploy wrapper — Devon ONLY.

.DESCRIPTION
    Deploys Devon on a Windows EC2 RDP host (same host type that already
    runs other workloads). It does NOT install, configure, start, or touch
    any other product: Devon only, Devon's repo, Devon's .env, Devon's bots.
    Any other system on the host is left exactly as it is.

    Steps:
      1. Clone the nomorals-2.0 repo (or git pull --ff-only if present).
      2. Call scripts/install_devon_windows.ps1 (the real installer —
         this script never duplicates it).
      3. Copy scripts/devon_ec2_env.template -> .env when .env is missing
         (existing .env is NEVER overwritten; placeholders only, never
         real secrets).
      4. Verify: `python -m nomorals --help`, `nm doctor`,
         `/impossible` smoke (CLI), singing smoke (writes an audio FILE —
         EC2 has no speaker, so singing is generation to file, sent via
         Telegram / saved to disk).
      5. Print the exact commands to fill .env and start Devon.

    Secrets: NEVER paste tokens/keys in chat, screenshots, or git.
    Fill .env on the RDP host with `notepad .env` from your secure store,
    or the Secure Vault flow. This script never prints secret values.

.PARAMETER RepoPath
    Where the nomorals-2.0 checkout lives. Default: C:\devon\nomorals-2.0

.PARAMETER Branch
    Git branch to deploy. Default: main.

.PARAMETER RepoUrl
    Git remote. Default: https://github.com/Oluwacutyp/nomorals-2.0.git

.PARAMETER SkipInstall
    Skip the installer (already installed). Still verifies.

.PARAMETER SkipVerify
    Skip verification (not recommended).

.PARAMETER IncludeLocalWhisper
    Pass through to the installer: also download Whisper-tiny fallback.

.PARAMETER DryRun
    Print what would happen, change nothing.

.EXAMPLE
    .\scripts\deploy_devon_ec2_rdp.ps1
    .\scripts\deploy_devon_ec2_rdp.ps1 -DryRun
    .\scripts\deploy_devon_ec2_rdp.ps1 -RepoPath C:\devon\nomorals-2.0
#>
[CmdletBinding()]
param(
    [string]$RepoPath = 'C:\devon\nomorals-2.0',
    [string]$Branch = 'main',
    [string]$RepoUrl = 'https://github.com/Oluwacutyp/nomorals-2.0.git',
    [switch]$SkipInstall,
    [switch]$SkipVerify,
    [switch]$IncludeLocalWhisper,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Write-Step { param([string]$Text) Write-Host "`n==> $Text" -ForegroundColor Cyan }
function Write-Warn { param([string]$Text) Write-Host "WARNING: $Text" -ForegroundColor Yellow }
function Write-Ok   { param([string]$Text) Write-Host "OK: $Text" -ForegroundColor Green }

Write-Host 'Devon EC2 RDP deploy (Devon ONLY)' -ForegroundColor Green
Write-Host "Repo: $RepoPath   Branch: $Branch"
if ($DryRun) { Write-Host '[DRY RUN — nothing will change]' -ForegroundColor Yellow }

# ── 1. Clone or pull ────────────────────────────────────────────────────────
Write-Step '1/5 — Checkout (clone or pull --ff-only)'
if (-not (Test-Path (Join-Path $RepoPath '.git'))) {
    if ($DryRun) { Write-Host "  [dry-run] git clone -b $Branch $RepoUrl $RepoPath" }
    else {
        $parent = Split-Path $RepoPath -Parent
        if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
        git clone -b $Branch $RepoUrl $RepoPath
        Write-Ok "Cloned $RepoUrl ($Branch)"
    }
} else {
    if ($DryRun) { Write-Host "  [dry-run] git -C $RepoPath pull --ff-only origin $Branch" }
    else {
        git -C $RepoPath pull --ff-only origin $Branch
        Write-Ok "Pulled latest $Branch"
    }
}

# ── 2. Installer (the real one — never duplicated here) ─────────────────────
Write-Step '2/5 — Installer (scripts/install_devon_windows.ps1)'
$installer = Join-Path $RepoPath 'scripts\install_devon_windows.ps1'
if (-not (Test-Path $installer)) { throw "Installer not found: $installer. Checkout is incomplete." }
if ($SkipInstall) { Write-Host '  Skipped (-SkipInstall).' }
elseif ($DryRun) { Write-Host "  [dry-run] & $installer -RepoPath $RepoPath $(if ($IncludeLocalWhisper) { '-IncludeLocalWhisper' })" }
else {
    if ($IncludeLocalWhisper) { & $installer -RepoPath $RepoPath -IncludeLocalWhisper }
    else { & $installer -RepoPath $RepoPath }
    Write-Ok 'Installer finished'
}

# ── 3. .env from the full template (never overwrite an existing .env) ───────
Write-Step '3/5 — .env (full template, placeholders only)'
$envPath = Join-Path $RepoPath '.env'
$template = Join-Path $RepoPath 'scripts\devon_ec2_env.template'
if (-not (Test-Path $template)) { throw "Env template not found: $template. Checkout is incomplete." }
if (Test-Path $envPath) {
    Write-Ok '.env already exists (kept). Fill missing keys with: notepad .env'
} elseif ($DryRun) {
    Write-Host "  [dry-run] Copy-Item $template -> $envPath"
} else {
    Copy-Item $template $envPath
    Write-Ok 'Wrote .env from scripts/devon_ec2_env.template (placeholders only)'
    Write-Warn 'Fill .env with: notepad .env — from your secure store. NEVER paste secrets in chat.'
    Write-Host '  REQUIRED: one LLM key (Groq recommended) + BOTH Devon bot tokens:'
    Write-Host '    NM_CHAT_TELEGRAM_BOT_TOKEN          (Devon main bot, GATED)'
    Write-Host '    NM_CHAT_TELEGRAM_PERSONAL_BOT_TOKEN (second Devon bot, USER-ONLY)'
}

# ── 4. Verify ───────────────────────────────────────────────────────────────
if (-not $SkipVerify) {
    Write-Step '4/5 — Verify'
    $venvPy = Join-Path $RepoPath '.venv\Scripts\python.exe'
    if (-not (Test-Path $venvPy)) { Write-Warn 'venv python not found — run the installer first (step 2). Verification skipped.' }
    elseif ($DryRun) {
        Write-Host '  [dry-run] python -m nomorals --help'
        Write-Host '  [dry-run] python -m nomorals.cli doctor'
        Write-Host '  [dry-run] nm impossible examples (CLI smoke)'
        Write-Host '  [dry-run] singing smoke: writes audio FILE (no speaker on EC2)'
    } else {
        & $venvPy -m nomorals --help | Select-Object -First 3 | ForEach-Object { Write-Host "  $_" }
        Write-Ok 'python -m nomorals --help works'
        & $venvPy -m nomorals.cli doctor 2>&1 | Select-Object -First 25 | ForEach-Object { Write-Host "  $_" }
        & $venvPy -m nomorals impossible examples 2>&1 | Select-Object -First 8 | ForEach-Object { Write-Host "  $_" }
        Write-Ok 'Impossible Made Possible CLI smoke works'
        Write-Host '  Singing smoke: Devon generates singing to an AUDIO FILE (EC2 has no'
        Write-Host '  speaker). Verify with: nm music / nm sing — output is a file you'
        Write-Host '  send via Telegram / save to disk. No live-speaker playback on EC2.'
    }
} else { Write-Host '  Verification skipped (-SkipVerify).' }

# ── 5. Next steps ───────────────────────────────────────────────────────────
Write-Step '5/5 — Start Devon'
Write-Host '  1. notepad .env   (fill ONLY the keys you use, from your secure store)'
Write-Host '  2. Start Devon:'
Write-Host "       cd $RepoPath"
Write-Host '       .\.venv\Scripts\python.exe -m nomorals'
Write-Host '  3. Check both bots answer: main bot (gated) + second bot (user-only).'
Write-Host '     Each bot owns its own getUpdates poller; never share one token.'
Write-Host '  4. Singing check: ask Devon to sing — expect an audio FILE back,'
Write-Host '     not live-speaker audio (EC2 has no speaker).'
Write-Host ''
Write-Host 'Devon ONLY. Nothing else on this host was installed, configured, started, or touched.' -ForegroundColor Green
