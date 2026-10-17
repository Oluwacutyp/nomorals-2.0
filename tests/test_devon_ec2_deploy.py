"""Devon EC2 RDP deploy: wrapper + full env template. Devon ONLY.

No other product is referenced in the new deploy files. Template must
cover every config key Devon code reads (config map + grep), with
placeholders only — never real secrets.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "scripts" / "deploy_devon_ec2_rdp.ps1"
TEMPLATE = REPO / "scripts" / "devon_ec2_env.template"
DOC = REPO / "scripts" / "DEVON_EC2_DEPLOY.md"
INSTALLER = REPO / "scripts" / "install_devon_windows.ps1"

FORBIDDEN_PRODUCT = ("goldenswing", "GoldenSwing", "GOLDENSWING")
SECRET_RE = re.compile(r"(?im)^\s*#?\s*(?:NM_[A-Z_0-9]+|[A-Z_][A-Z_0-9]*TOKEN|[A-Z_]*API_KEY|[A-Z_]*SECRET)\s*=\s*\S+")


def _text(p):
    return p.read_text(encoding="utf-8", errors="replace")


def test_deploy_script_exists_and_calls_installer():
    assert DEPLOY.exists(), "deploy wrapper missing"
    src = _text(DEPLOY)
    assert "install_devon_windows.ps1" in src
    assert "-DryRun" in src and "DryRun" in src
    assert "devon_ec2_env.template" in src
    assert "TODO" not in src and "NotImplementedError" not in src


def test_deploy_script_never_overwrites_env_and_no_secrets():
    src = _text(DEPLOY)
    assert "never overwritten" in src.lower() or "already exists (kept)" in src
    # No secret-looking assignment with a real value.
    for line in src.splitlines():
        assert not SECRET_RE.match(line), f"secret-looking line in deploy script: {line[:80]}"


def test_new_deploy_files_reference_no_other_product():
    for p in (DEPLOY, TEMPLATE, DOC):
        assert p.exists(), f"missing {p.name}"
        src = _text(p)
        for bad in FORBIDDEN_PRODUCT:
            assert bad not in src, f"{p.name} must not reference another product ({bad})"


def test_env_template_has_both_devon_bots_with_roles():
    src = _text(TEMPLATE)
    assert "NM_CHAT_TELEGRAM_BOT_TOKEN" in src
    assert "NM_CHAT_TELEGRAM_PERSONAL_BOT_TOKEN" in src
    assert "GATED" in src and "USER-ONLY" in src
    assert "never share one token" in src.lower()
    # Placeholders only: every KEY= line is blank.
    for line in src.splitlines():
        m = re.match(r"\s*#\s*([A-Z_][A-Z_0-9]*)\s*=\s*(.*)$", line)
        if m:
            assert m.group(2).strip() == "", f"template key {m.group(1)} has a non-blank value"


def test_env_template_covers_config_map_keys():
    """Every key in core/config.py _ENV map must appear in the template."""
    cfg = _text(REPO / "nomorals" / "core" / "config.py")
    keys = set(re.findall(r'"(NM_[A-Z_0-9]+)"\s*:', cfg))
    assert len(keys) >= 100, "config map unexpectedly small"
    tmpl = _text(TEMPLATE)
    missing = sorted(k for k in keys if k not in tmpl)
    assert not missing, f"template missing config-map keys: {missing[:20]}"


def test_env_template_covers_grepped_nm_keys():
    """Every NM_ key grepped in nomorals/ + scripts/ must appear in template."""
    found = set()
    for p in list((REPO / "nomorals").rglob("*.py")) + list((REPO / "scripts").glob("*.ps1")):
        try:
            found |= set(re.findall(r"NM_[A-Z_0-9]{3,}", p.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    # Drop test-only / prefix fragments the grep catches.
    found = {k for k in found if k not in {"NM_FOO_BAR", "NM_HUB_", "NM_RUNTIME_"} and not k.startswith("NM_CHAT_TELEGRAM_BOT_TOKEN")}
    tmpl = _text(TEMPLATE)
    missing = sorted(k for k in found if k not in tmpl)
    assert not missing, f"template missing grepped NM_ keys: {missing[:20]}"


def test_doc_has_exact_commands_and_singing():
    src = _text(DOC)
    assert "deploy_devon_ec2_rdp.ps1" in src
    assert "notepad .env" in src
    assert "python.exe -m nomorals" in src
    assert "audio FILE" in src or "audio file" in src
    assert "no speaker" in src.lower()
    assert "NM_CHAT_TELEGRAM_BOT_TOKEN" in src
    assert "NM_CHAT_TELEGRAM_PERSONAL_BOT_TOKEN" in src


def test_installer_still_present_and_untouched_by_wrapper():
    assert INSTALLER.exists()
    assert "install_devon_windows" in _text(DEPLOY)


def test_existing_dir_path_pulls_instead_of_clone():
    """Existing repo dir -> pull, never clone into it. Non-empty non-git dir -> clear refusal."""
    src = _text(DEPLOY)
    assert "SKIP clone" in src
    assert "pull --ff-only" in src
    assert "not an empty" in src  # names the exact git failure the user hit
    assert "already exists and is not empty" in src


def test_wrong_dir_guard_prints_cd_command():
    """Wrong-directory run fails with the exact cd command, not a bare CommandNotFound."""
    src = _text(DEPLOY)
    assert "WRONG DIRECTORY" in src
    assert "Run this from" in src
    assert r"cd $RepoPath" in src or "cd C:" in src


def test_pull_failure_prints_recovery_and_never_auto_resets():
    """Divergent pull -> exact recovery printed, no auto reset --hard."""
    src = _text(DEPLOY)
    assert "reset --hard" in src
    assert "does NOT auto-reset" in src or "does not auto-reset" in src.lower()
    assert "discards local changes" in src
    # The script must not execute reset --hard itself.
    assert "git reset --hard" in src  # only inside the printed recovery message
    # Every reset --hard occurrence must be inside a Write-Host recovery line or a comment.
    for ln in src.splitlines():
        if "reset --hard" in ln:
            assert ("Write-Host" in ln or ln.strip().startswith("#") or "Recovery" in ln), \
                f"reset --hard outside printed recovery: {ln[:100]}"


def test_empty_env_backed_up_and_replaced_then_verified():
    """Empty (0-byte) .env -> backup + template replacement + verification; never left empty."""
    src = _text(DEPLOY)
    assert ".empty.bak" in src
    assert "EMPTY (0 bytes)" in src
    assert "NM_CHAT_TELEGRAM_BOT_TOKEN" in src  # verification key check
    assert "verification FAILED" in src


def test_doc_recovery_section_has_exact_commands():
    """Doc: SKIP-clone first step + Recovery section with the exact commands."""
    src = _text(DOC)
    assert "SKIP clone" in src
    assert "Recovery" in src
    assert "git pull origin main" in src
    assert r"cd C:\devon\nomorals-2.0" in src
    assert ".env.empty.bak" in src
    assert "not recognized" in src
