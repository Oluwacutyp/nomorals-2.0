"""Tail-52 pins for nomorals/tools/captcha.py —
0 converts / 7 exclusions / 1 flagged (1198).

8 tests for 8 sites (counts equal). Each test pins one site
with a control half proving the guarded path was actually
exercised, asserting via the operation's own call log /
double that the failing operation itself was attempted.
The file contains no exception handlers, so it adds zero
census sites.

captcha.py is a FROZEN-FOUR file (owner ruling T2.2): this
batch is pins-only, no production change. Convert-shape
site 1198 is flagged in the tail-52 SURVEY.md; its pin
asserts OBSERVED behavior only.

captcha.py calls no failures-ledger record function
anywhere (grep-proven — its only record_* calls are the
rate limiter's own state methods and SolveBudget's
record_spend), so NOT-recording is pinned against the
module's own observability surfaces: the captcha state dir
(NOMORALS_CAPTCHA_DIR) file snapshot before/after — no
audit line written, no state file created or mutated by
any trigger — plus, for the two _load sites, byte-identical
state files after the read.

Verdict notes (full grounds in the tail-52 SURVEY.md):
- 234: _decode_data_uri malformed base64 -> b"", the
  documented genuine no-bytes value; detect's summary
  truthfully reports 0 bytes and byte-less solves raise.
- 386: detect remote-image fetch failure -> image_bytes
  b"", detect()'s own documented best-effort contract;
  the challenge and its URL stay true.
- 408: same documented shape on the audio leg.
- 526: _rate_limit_path settings.resolve failure -> the
  module default root, the same path settings=None gives.
- 575: SolverRateLimiter._load corrupt state -> {}, the
  documented fresh state ("missing/corrupt state is
  fine"); pacing state only, self-healing on record_*.
- 1143: _audit_path settings.resolve failure -> the
  module default root (mirror of 526).
- 1198: FLAGGED (convert-shape, freeze-barred) —
  SolveBudget._load corrupt ledger -> {} re-opens the
  can_spend money gate and renders $0 spent / full
  remaining. Pin asserts OBSERVED behavior only.
- 1285: _budget_path settings.resolve failure -> the
  module default root (mirror of 526).
"""

import base64
import json
import os
import time
from pathlib import Path

import pytest

import nomorals.tools.captcha as cap


def _snap(root):
    root = Path(root)
    if not root.exists():
        return {}
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


class _RaisingSettings:
    """settings double: resolve() logs the arg and raises."""

    def __init__(self):
        self.calls = []

    def resolve(self, rel):
        self.calls.append(rel)
        raise RuntimeError("settings resolve boom")


class _GoodSettings:
    def __init__(self, base):
        self.calls = []
        self._base = base

    def resolve(self, rel):
        self.calls.append(rel)
        return str(Path(self._base) / rel)


def _load_spy(log):
    real = json.load

    def spy(fh, *a, **k):
        log.append(fh.read())
        fh.seek(0)
        return real(fh, *a, **k)

    return spy


@pytest.fixture()
def capdir(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMORALS_CAPTCHA_DIR", str(tmp_path))
    return tmp_path


def test_234_decode_data_uri_malformed_base64_returns_empty(capdir, monkeypatch):
    before = _snap(capdir)
    calls = []
    real_b64 = base64.b64decode

    def b64_spy(payload, validate=False):
        calls.append((payload, validate))
        return real_b64(payload, validate=validate)

    monkeypatch.setattr(base64, "b64decode", b64_spy)
    assert cap._decode_data_uri("data:image/png;base64,!!!bogus!!!") == b""
    assert calls and calls[0][1] is True  # decode was attempted
    monkeypatch.undo()
    valid = "data:image/png;base64," + base64.b64encode(b"PNGDATA").decode()
    assert cap._decode_data_uri(valid) == b"PNGDATA"  # control
    assert cap._decode_data_uri("data:image/png,raw") == b""  # genuine value
    found = cap.detect('<img class="captcha" src="'
                       'data:image/png;base64,!!!bogus!!!">', "https://e.com/")
    assert len(found) == 1 and found[0].image_bytes == b""
    assert found[0].summary()["image_bytes"] == 0  # truthful consumer surface
    assert _snap(capdir) == before


def test_386_detect_image_fetch_failure_leaves_bytes_empty(capdir, monkeypatch):
    before = _snap(capdir)
    log = []

    def bad_fetch(url, timeout=0):
        log.append((url, timeout))
        raise RuntimeError("network down")

    monkeypatch.setattr(cap, "fetch_image_bytes", bad_fetch)
    found = cap.detect('<img id="captcha-img" src="/cap.png">',
                       "https://e.com/login", fetch_bytes=True)
    assert log == [("https://e.com/cap.png", 10.0)]  # fetch attempted
    assert len(found) == 1
    assert found[0].kind == cap.CaptchaKind.IMAGE_CAPTCHA
    assert found[0].image_url == "https://e.com/cap.png"
    assert found[0].image_bytes == b""  # documented best-effort fallback

    def good_fetch(url, timeout=0):
        log.append((url, timeout))
        return b"PNGBYTES"

    monkeypatch.setattr(cap, "fetch_image_bytes", good_fetch)
    found2 = cap.detect('<img id="captcha-img" src="/cap.png">',
                        "https://e.com/login", fetch_bytes=True)
    assert found2[0].image_bytes == b"PNGBYTES"  # control
    assert _snap(capdir) == before


def test_408_detect_audio_fetch_failure_leaves_bytes_empty(capdir, monkeypatch):
    before = _snap(capdir)
    log = []

    def bad_fetch(url, timeout=0):
        log.append((url, timeout))
        raise RuntimeError("network down")

    monkeypatch.setattr(cap, "fetch_image_bytes", bad_fetch)
    html = '<p>get audio challenge</p><audio src="/chal.mp3"></audio>'
    found = cap.detect(html, "https://e.com/login", fetch_bytes=True)
    assert log == [("https://e.com/chal.mp3", 10.0)]  # fetch attempted
    assert len(found) == 1
    assert found[0].kind == cap.CaptchaKind.AUDIO_CAPTCHA
    assert found[0].image_url == "https://e.com/chal.mp3"
    assert found[0].image_bytes == b""

    def good_fetch(url, timeout=0):
        log.append((url, timeout))
        return b"AUDIOBYTES"

    monkeypatch.setattr(cap, "fetch_image_bytes", good_fetch)
    found2 = cap.detect(html, "https://e.com/login", fetch_bytes=True)
    assert found2[0].image_bytes == b"AUDIOBYTES"  # control
    assert _snap(capdir) == before


def test_526_rate_limit_path_resolve_failure_falls_back(capdir):
    before = _snap(capdir)
    raising = _RaisingSettings()
    got = cap._rate_limit_path(raising)
    assert raising.calls == ["data/captcha/rate_limit.json"]  # attempted
    assert got == os.path.join(str(capdir), "rate_limit.json")
    good = _GoodSettings("/settings-root")
    assert cap._rate_limit_path(good) == \
        "/settings-root/data/captcha/rate_limit.json"  # control
    assert good.calls == ["data/captcha/rate_limit.json"]
    assert cap._rate_limit_path(None) == got  # settings=None: same fallback
    assert _snap(capdir) == before


def test_575_limiter_load_corrupt_state_returns_fresh(capdir, monkeypatch):
    lim = cap.SolverRateLimiter()
    Path(lim._path).write_text("{corrupt json", encoding="utf-8")
    before = _snap(capdir)
    log = []
    monkeypatch.setattr(cap.json, "load", _load_spy(log))
    assert lim._load() == {}
    assert log == ["{corrupt json"]  # the parse was attempted on the file
    monkeypatch.undo()
    st = lim.status()
    assert st["submits_last_24h"] == 0 and st["cooling_down"] is False
    assert st["consecutive_failures"] == 0 and st["hard_failure"] is False
    lim.check()  # OBSERVED: fresh snapshot passes the pacing gate
    assert _snap(capdir) == before  # read recorded nothing, file untouched
    now = time.time()
    Path(lim._path).write_text(json.dumps(
        {"submits": [now], "cooldown_until": now + 3600,
         "consecutive_failures": 2, "hard_failure": True}), encoding="utf-8")
    assert lim._load()["consecutive_failures"] == 2  # control: valid loads
    st2 = lim.status()
    assert st2["submits_last_24h"] == 1 and st2["cooling_down"] is True
    assert st2["hard_failure"] is True
    with pytest.raises(cap.CaptchaError):
        lim.check()  # control: a real cooldown is enforced


def test_1143_audit_path_resolve_failure_falls_back(capdir):
    before = _snap(capdir)
    raising = _RaisingSettings()
    got = cap._audit_path(raising)
    assert raising.calls == ["data/captcha/audit.jsonl"]  # attempted
    assert got == os.path.join(str(capdir), "audit.jsonl")
    good = _GoodSettings("/settings-root")
    assert cap._audit_path(good) == \
        "/settings-root/data/captcha/audit.jsonl"  # control
    assert good.calls == ["data/captcha/audit.jsonl"]
    assert _snap(capdir) == before


def test_1198_budget_load_corrupt_ledger_observed_only(capdir, monkeypatch):
    # FLAGGED site (convert-shape; frozen file, no in-batch fix):
    # this pin asserts OBSERVED behavior only — see SURVEY.md.
    bud = cap.SolveBudget(daily_cap_usd=0.50, monthly_cap_usd=5.00)
    bud.record_spend(0.499, "task-1", "recaptcha_v2")
    ok_before, _reason = bud.can_spend()
    assert ok_before is False  # control: real spend blocks at the cap
    assert bud.budget_status()["daily_spent_usd"] == 0.499  # control
    Path(bud._path).write_text("{corrupt json", encoding="utf-8")
    before = _snap(capdir)
    log = []
    monkeypatch.setattr(cap.json, "load", _load_spy(log))
    assert bud._load() == {}
    assert log == ["{corrupt json"]  # the parse was attempted on the file
    monkeypatch.undo()
    ok_after, reason = bud.can_spend()
    assert ok_after is True and reason == ""  # OBSERVED: gate re-opens
    bst = bud.budget_status()
    assert bst["daily_spent_usd"] == 0.0  # OBSERVED: $0 claim
    assert bst["daily_remaining_usd"] == 0.50  # OBSERVED: full remaining
    assert bst["monthly_spent_usd"] == 0.0
    assert bst["paid_solving_enabled"] is True
    assert _snap(capdir) == before  # read recorded nothing, file untouched


def test_1285_budget_path_resolve_failure_falls_back(capdir):
    before = _snap(capdir)
    raising = _RaisingSettings()
    got = cap._budget_path(raising)
    assert raising.calls == ["data/captcha/budget.json"]  # attempted
    assert got == os.path.join(str(capdir), "budget.json")
    good = _GoodSettings("/settings-root")
    assert cap._budget_path(good) == \
        "/settings-root/data/captcha/budget.json"  # control
    assert good.calls == ["data/captcha/budget.json"]
    assert _snap(capdir) == before
