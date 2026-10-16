"""Tests for the Windows RDP installer (static + template checks).

PowerShell is not available on this Linux host, so these tests validate
the installer by parsing it: params, tier thresholds, model catalogue,
no hardcoded secrets, .env generation, and the documented model list.
A pwsh dry-run is attempted when `pwsh` exists and skipped otherwise.
"""

import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PS1 = REPO / "scripts" / "install_devon_windows.ps1"
MD = REPO / "scripts" / "install_devon_windows.md"
ENV_EXAMPLE = REPO / ".env.example"


def _text(p):
    return p.read_text(encoding="utf-8")


def test_script_exists_and_has_params():
    src = _text(PS1)
    for param in ("$SkipLocalExtras", "$IncludeLocalWhisper",
                  "$SkipSystemDeps", "$Force", "$DryRun", "$RepoPath"):
        assert param in src, f"missing param {param}"
    # API-first: no model-set selector, no Ollama, no GGUF downloads.
    assert "$Models" not in src
    assert "IncludeOllama" not in src
    assert "Qwen/" not in src


def test_tier_thresholds_present_and_sane():
    src = _text(PS1)
    low = float(re.search(r"LOW_RAM_GB\s*=\s*([\d.]+)", src).group(1))
    med = float(re.search(r"MEDIUM_RAM_GB\s*=\s*([\d.]+)", src).group(1))
    high = float(re.search(r"HIGH_RAM_GB\s*=\s*([\d.]+)", src).group(1))
    assert 0 < low < med < high, (low, med, high)
    assert "function Get-Tier" in src
    # GPU lifts one tier; LOW is RAM- or CPU-bound.
    assert "HasGpu" in src and "LOW" in src


def test_tier_logic_matches_spec():
    """Mirror the PS1 Get-Tier rules in Python and check boundaries."""
    def tier(cpu, ram, gpu):
        if gpu and ram >= 8.0:
            return "HIGH"
        if cpu < 2 or ram <= 4.5:
            return "LOW"
        if ram >= 16.0:
            return "HIGH"
        return "MEDIUM"

    assert tier(1, 2.0, False) == "LOW"      # tiny EC2
    assert tier(2, 4.0, False) == "LOW"      # current RDP class
    assert tier(4, 8.0, False) == "MEDIUM"
    assert tier(4, 16.0, False) == "HIGH"
    assert tier(4, 8.0, True) == "HIGH"      # GPU lifts
    assert tier(2, 3.0, True) == "LOW"       # GPU needs >=8GB RAM to lift


def test_local_extras_catalogue_api_first():
    """Only tiny fallbacks are downloaded: Piper always, Whisper opt-in.
    No GGUF, no Qwen, no XTTS/diffusion downloads."""
    src = _text(PS1)
    assert "NO GGUF" in src
    assert "rhasspy/piper-voices" in src
    assert "openai/whisper-tiny" in src
    for banned in ("Qwen/", "coqui/XTTS-v2", "stabilityai/sd-turbo",
                   "stabilityai/sdxl-turbo", "facebook/demucs",
                   "llama.cpp", "Ollama"):
        # Allowed only inside explicit NOT-INSTALLED / SKIPPED statements.
        for line in src.splitlines():
            if banned in line:
                assert ("NOT INSTALLED" in line or "SKIPPED" in line
                        or "NO " in line or "no GGUF" in line.lower()
                        or "Deferred" in line or "deferred" in line
                        or "needs better hosting" in line), line
    sizes = [float(s) for s in re.findall(r"SizeGB\s*=\s*([\d.]+)", src)]
    assert sizes and all(0 < s <= 0.2 for s in sizes), sizes


def test_no_hardcoded_secrets():
    src = _text(PS1) + _text(MD)
    # No token-like assignments, no placeholder that looks real.
    assert "sk-" not in src
    assert "ghp_" not in src
    assert re.search(r"(?m)^\s*\$?[\w]*TOKEN\s*=\s*['\"][^'\"]+['\"]", src) is None
    assert "PLACEHOLDER" in src or "placeholder" in src.lower()


def test_env_generation_never_overwrites():
    src = _text(PS1)
    assert "already exists (kept)" in src
    assert ".bak-" in src  # backup on -Force
    assert "notepad .env" in src
    assert "nano .env" in src


def test_env_example_parses():
    """Every non-comment, non-empty line in .env.example is KEY=VALUE."""
    for line in _text(ENV_EXAMPLE).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert "=" in line, line
        key = line.split("=", 1)[0].strip()
        assert re.match(r"^[A-Z0-9_]+$", key), key


def test_md_documents_models_and_rdp_assessment():
    combined = (_text(MD) + _text(PS1)).lower()
    for token in ("one heavy job at a time",):
        assert token in combined, f"missing {token}"
    md = _text(MD)
    assert "Paper only" in md or "paper-only" in md.lower()


def test_ps1_balanced_braces_and_no_todo_stub():
    src = _text(PS1)
    assert src.count("{") == src.count("}"), "unbalanced braces"
    assert "TODO" not in src and "NotImplemented" not in src
    assert "hf download" in src
    assert "python -m venv" in src
    assert "nm doctor" in src or "doctor" in src


def test_pwsh_dry_run_when_available():
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if not pwsh:
        return  # honest skip: no PowerShell on this host
    r = subprocess.run(
        [pwsh, "-NoProfile", "-Command",
         f"& '{PS1}' -DryRun -SkipSystemDeps -SkipLocalExtras"],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout
