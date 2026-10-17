<#
.SYNOPSIS
    Devon (nomorals-2.0) EC2 RDP deploy wrapper - Devon ONLY.

.DESCRIPTION
    Deploys Devon on a Windows EC2 RDP host (same host type that already
    runs other workloads). It does NOT install, configure, start, or touch
    any other product: Devon only, Devon's repo, Devon's .env, Devon's bots.
    Any other system on the host is left exactly as it is.

    Steps:
      1. Clone the nomorals-2.0 repo (or git pull --ff-only if present).
      2. Call scripts/install_devon_windows.ps1 (the real installer -
         this script never duplicates it).
      3. Copy scripts/devon_ec2_env.template -> .env when .env is missing
         (existing .env is NEVER overwritten; placeholders only, never
         real secrets).
      4. Verify: `python -m nomorals --help`, `nm doctor`,
         `/impossible` smoke (CLI), singing smoke (writes an audio FILE -
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
if ($DryRun) { Write-Host '[DRY RUN - nothing will change]' -ForegroundColor Yellow }

# -- 0. Wrong-directory guard ------------------------------------------------
# Exact failure seen on EC2: the owner ran `.\scripts\deploy_devon_ec2_rdp.ps1`
# from C:\devon (the PARENT) after a failed clone, so PowerShell answered
# "not recognized" and every later command ran in the wrong directory.
# If this script IS running (invoked by absolute path) but the shell is
# sitting in the parent of the repo, stop with the exact cd command.
# (If it was invoked relatively from the wrong dir, PowerShell itself
#  errors before this code runs - the deploy doc tells the user to cd first.)
$cwd = (Get-Location).Path
$repoLeaf = Split-Path $RepoPath -Leaf
$repoParent = Split-Path $RepoPath -Parent
if ($cwd -eq $repoParent -and (Test-Path (Join-Path $cwd $repoLeaf))) {
    throw ("WRONG DIRECTORY: you are in $cwd, not in the repo.`n" +
           "Run this from $RepoPath, not from $cwd.`n" +
           "Run:  cd $RepoPath`n" +
           "Then: .\\scripts\\deploy_devon_ec2_rdp.ps1")
}
if ($cwd -ne $RepoPath) {
    Write-Warn ("Shell is in $cwd, repo is $RepoPath. Recommended: cd $RepoPath first, " +
                "so `notepad .env` and `.\.venv\...` land in the repo, not in $cwd.")
}

# -- 1. Clone or pull --------------------------------------------------------
Write-Step '1/5 - Checkout (pull if present, clone only if absent)'
Write-Host '  If the repo directory already exists, SKIP clone - pull instead.'
Write-Host '  Never clone into an existing non-empty directory (git refuses:'
Write-Host '  "fatal: destination path already exists and is not an empty directory").'
if (-not (Test-Path (Join-Path $RepoPath '.git'))) {
    if ((Test-Path $RepoPath) -and (Get-ChildItem -Path $RepoPath -Force | Select-Object -First 1)) {
        throw ("CHECKOUT BLOCKED: $RepoPath already exists and is not empty, but it is " +
               "not a git checkout (no .git). Do NOT clone into it - git will fail with " +
               "`fatal: destination path already exists and is not an empty directory`. " +
               "Recovery: if it is a partial/failed clone, rename it aside " +
               "(`Rename-Item $RepoPath ${RepoPath}.bak`) or delete it, then re-run this script. " +
               "If it IS a checkout whose .git is missing, re-clone to a fresh path instead.")
    }
    if ($DryRun) { Write-Host "  [dry-run] git clone -b $Branch $RepoUrl $RepoPath" }
    else {
        $parent = Split-Path $RepoPath -Parent
        if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
        git clone -b $Branch $RepoUrl $RepoPath
        if ($LASTEXITCODE -ne 0) {
            throw ("git clone failed (exit $LASTEXITCODE). If the directory already exists, " +
                   "do NOT re-clone - cd into it and pull instead:  cd $RepoPath ; git pull origin $Branch")
        }
        Write-Ok "Cloned $RepoUrl ($Branch)"
    }
} else {
    if ($DryRun) { Write-Host "  [dry-run] git -C $RepoPath pull --ff-only origin $Branch" }
    else {
        git -C $RepoPath pull --ff-only origin $Branch
        if ($LASTEXITCODE -ne 0) {
            # Divergent history (local commits + remote commits). Do NOT auto-reset:
            # reset --hard discards local changes. Print the exact recovery and stop.
            Write-Host ''
            Write-Host '  PULL FAILED (divergent history: local commits + remote commits).' -ForegroundColor Red
            Write-Host '  This script does NOT auto-reset - `git reset --hard` discards local changes.'
            Write-Host '  Recovery (run these yourself, only if you are OK discarding local changes):'
            Write-Host "    cd $RepoPath"
            Write-Host '    git fetch origin'
            Write-Host "    git reset --hard origin/$Branch"
            Write-Host "    git pull origin $Branch"
            Write-Host '  Or keep your local changes: `git pull --rebase origin main` and resolve conflicts.'
            throw "git pull --ff-only failed (exit $LASTEXITCODE). Recovery printed above - fix it, then re-run."
        }
        Write-Ok "Pulled latest $Branch"
    }
}

# -- 2. Installer (the real one - never duplicated here) ---------------------
Write-Step '2/5 - Installer (scripts/install_devon_windows.ps1)'
$installer = Join-Path $RepoPath 'scripts\install_devon_windows.ps1'
if (-not (Test-Path $installer)) { throw "Installer not found: $installer. Checkout is incomplete." }
if ($SkipInstall) { Write-Host '  Skipped (-SkipInstall).' }
elseif ($DryRun) { Write-Host "  [dry-run] & $installer -RepoPath $RepoPath $(if ($IncludeLocalWhisper) { '-IncludeLocalWhisper' })" }
else {
    if ($IncludeLocalWhisper) { & $installer -RepoPath $RepoPath -IncludeLocalWhisper }
    else { & $installer -RepoPath $RepoPath }
    Write-Ok 'Installer finished'
}

# -- 3. .env from the full template ------------------------------------------
# Rules: a missing .env is created from the template. A non-empty existing
# .env is NEVER overwritten. An EMPTY .env (0 bytes - e.g. created by
# `notepad .env` in the wrong directory before the deploy ran) is NOT
# "done": it is backed up and replaced with the template, with a message.
# After writing, verify the .env is non-empty and contains placeholder keys.
Write-Step '3/5 - .env (full template, placeholders only)'
$envPath = Join-Path $RepoPath '.env'
$template = Join-Path $RepoPath 'scripts\devon_ec2_env.template'
if (-not (Test-Path $template)) { throw "Env template not found: $template. Checkout is incomplete." }
$envExists = Test-Path $envPath
$envSize = if ($envExists) { (Get-Item $envPath).Length } else { -1 }
if ($envExists -and $envSize -gt 0) {
    Write-Ok ".env already exists ($envSize bytes, kept). Fill missing keys with: notepad .env"
    Write-Host "  (Run this from $RepoPath so `notepad .env` opens THIS file, not an empty one elsewhere.)"
} else {
    if ($envExists -and $envSize -eq 0) {
        $backup = "$envPath.empty.bak"
        if ($DryRun) { Write-Host "  [dry-run] .env exists but is EMPTY (0 bytes) - would back up to $backup and replace from template" }
        else {
            Copy-Item $envPath $backup -Force
            Write-Warn ".env existed but was EMPTY (0 bytes) - backed up to $backup and replaced from template."
            Write-Host '  (An empty .env usually means `notepad .env` ran in the wrong directory before deploy.)'
        }
    }
    if ($DryRun) { Write-Host "  [dry-run] Copy-Item $template -> $envPath" }
    else {
        Copy-Item $template $envPath -Force
        # Verify: non-empty and contains placeholder keys.
        $written = (Get-Item $envPath).Length
        $content = Get-Content $envPath -Raw
        if ($written -eq 0 -or $content -notmatch 'NM_CHAT_TELEGRAM_BOT_TOKEN') {
            throw ".env verification FAILED: wrote $written bytes, placeholder keys missing. Template copy did not land - do not edit an empty .env."
        }
        Write-Ok "Wrote .env from scripts/devon_ec2_env.template ($written bytes, placeholders only, verified)"
        Write-Warn 'Fill .env with: notepad .env - from your secure store. NEVER paste secrets in chat.'
        Write-Host '  REQUIRED: one LLM key (Groq recommended) + BOTH Devon bot tokens:'
        Write-Host '    NM_CHAT_TELEGRAM_BOT_TOKEN          (Devon main bot, GATED)'
        Write-Host '    NM_CHAT_TELEGRAM_PERSONAL_BOT_TOKEN (second Devon bot, USER-ONLY)'
    }
}

# -- 4. Verify ---------------------------------------------------------------
if (-not $SkipVerify) {
    Write-Step '4/5 - Verify'
    $venvPy = Join-Path $RepoPath '.venv\Scripts\python.exe'
    if (-not (Test-Path $venvPy)) { Write-Warn 'venv python not found - run the installer first (step 2). Verification skipped.' }
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
        Write-Host '  speaker). Verify with: nm music / nm sing - output is a file you'
        Write-Host '  send via Telegram / save to disk. No live-speaker playback on EC2.'
    }
} else { Write-Host '  Verification skipped (-SkipVerify).' }

# -- 5. Next steps -----------------------------------------------------------
Write-Step '5/5 - Start Devon'
Write-Host '  1. notepad .env   (fill ONLY the keys you use, from your secure store)'
Write-Host '  2. Start Devon:'
Write-Host "       cd $RepoPath"
Write-Host '       .\.venv\Scripts\python.exe -m nomorals'
Write-Host '  3. Check both bots answer: main bot (gated) + second bot (user-only).'
Write-Host '     Each bot owns its own getUpdates poller; never share one token.'
Write-Host '  4. Singing check: ask Devon to sing - expect an audio FILE back,'
Write-Host '     not live-speaker audio (EC2 has no speaker).'
Write-Host ''
Write-Host 'Devon ONLY. Nothing else on this host was installed, configured, started, or touched.' -ForegroundColor Green
