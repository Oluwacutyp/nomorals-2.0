"""Tests for Step 8 Captcha: classification, non-captcha-first,
learned rates, per-origin ceiling, TrialVault identity binding,
local-first audio, and the staged pipeline.

No test touches a real solving service or the network: fetchers,
OCR/ASR/service/ping are all injected fakes.
"""

from __future__ import annotations

import inspect
import json
import random
from pathlib import Path

import pytest

from nomorals.agents.trial import identity as identity_mod
from nomorals.agents.trial.identity import (
    SYNTHETIC_DOMAINS,
    TrialIdentityBinder,
    generate_trial_identity,
)
from nomorals.agents.trial.vault import TrialVault
from nomorals.tools.captcha_flow import (
    CaptchaFlow,
    OriginCostCeiling,
    SuccessLedger,
    classify_challenge,
    non_captcha_candidates,
    resolve_without_captcha,
    solve_audio_local_first,
    solve_pipeline,
)

SITE = "https://example.com/login"


# ── classification ───────────────────────────────────────────────────────────

class TestClassify:
    def test_429_is_rate_limit_not_solve(self):
        v = classify_challenge(429, {"Retry-After": "120"}, "slow down")
        assert v["kind"] == "rate_limit"
        assert v["action"] == "backoff"
        assert v["retry_after_s"] == 120.0

    def test_429_without_header_defaults(self):
        v = classify_challenge(429, {}, "")
        assert v["kind"] == "rate_limit" and v["retry_after_s"] == 60.0

    def test_captcha_widget(self):
        html = '<div class="g-recaptcha" data-sitekey="abc"></div>'
        assert classify_challenge(200, {}, html)["kind"] == "captcha"

    def test_bot_wall_interstitial(self):
        html = "<title>Just a moment...</title> verifying you are human"
        v = classify_challenge(200, {}, html)
        assert v["kind"] == "bot_wall"
        assert v["action"] == "wait_or_reroute"

    def test_ok_and_403(self):
        assert classify_challenge(200, {}, "<p>hi</p>")["kind"] == "ok"
        assert classify_challenge(403, {}, "forbidden")["kind"] == "bot_wall"


# ── non-captcha path first ───────────────────────────────────────────────────

class TestNonCaptchaFirst:
    def _fetcher(self, mapping):
        def fetch(url):
            return mapping[url]
        return fetch

    def test_direct_win(self):
        fetch = self._fetcher({SITE: (200, {}, "<p>content</p>")})
        res = resolve_without_captcha(SITE, fetcher=fetch)
        assert res is not None and res["method"] == "direct"

    def test_archive_fallback_when_direct_is_walled(self):
        archive = f"https://web.archive.org/web/2id_/{SITE}"
        fetch = self._fetcher({
            SITE: (200, {}, "Just a moment... verifying you are human"),
            archive: (200, {}, "<p>archived</p>"),
        })
        res = resolve_without_captcha(SITE, fetcher=fetch)
        assert res is not None and res["method"] == "archive"

    def test_all_walled_returns_none(self):
        fetch = self._fetcher({
            SITE: (200, {}, "Just a moment..."),
            f"https://web.archive.org/web/2id_/{SITE}":
                (429, {}, "slow down"),
        })
        assert resolve_without_captcha(SITE, fetcher=fetch) is None

    def test_candidates_include_archive(self):
        urls = [c["url"] for c in non_captcha_candidates(SITE)]
        assert urls[0] == SITE
        assert "web.archive.org" in urls[1]


# ── learned success rates ────────────────────────────────────────────────────

class TestSuccessLedger:
    def test_rate_and_best_method(self, tmp_path):
        led = SuccessLedger(tmp_path / "ledger.json")
        assert led.rate("image_captcha", "example.com", "ocr") is None
        led.record("image_captcha", "example.com", "ocr", ok=True)
        led.record("image_captcha", "example.com", "ocr", ok=False)
        led.record("image_captcha", "example.com", "service", ok=True)
        assert led.rate("image_captcha", "example.com", "ocr") == 0.5
        assert led.rate("image_captcha", "example.com", "service") == 1.0
        method, rate = led.best_method("image_captcha", "example.com",
                                       ["ocr", "service"])
        assert (method, rate) == ("service", 1.0)

    def test_invariant_unknown_is_unknown_not_zero(self, tmp_path):
        """INVARIANT: an untried method has rate None, never 0.0."""
        led = SuccessLedger(tmp_path / "ledger.json")
        assert led.rate("x", "y", "never-tried") is None
        method, rate = led.best_method("x", "y", ["never-tried", "other"])
        assert (method, rate) == ("never-tried", None)

    def test_ledger_persists(self, tmp_path):
        path = tmp_path / "ledger.json"
        SuccessLedger(path).record("k", "s", "m", ok=True)
        assert SuccessLedger(path).rate("k", "s", "m") == 1.0


# ── per-origin cost ceiling ──────────────────────────────────────────────────

class TestOriginCostCeiling:
    def test_allows_within_cap(self, tmp_path):
        cap = OriginCostCeiling(tmp_path / "ceiling.json", daily_cap_usd=0.50)
        ok, _ = cap.can_spend(SITE, 0.01)
        assert ok is True

    def test_invariant_never_overruns_pings_owner(self, tmp_path):
        """INVARIANT: at cap → refused + owner pinged, spend unchanged."""
        pings: list = []
        cap = OriginCostCeiling(tmp_path / "ceiling.json", daily_cap_usd=0.05)
        cap.record_spend(SITE, 0.05, "image_captcha")
        res = cap.check_or_ping(SITE, 0.01,
                                ping_owner=lambda p: pings.append(p))
        assert res["allowed"] is False
        assert res["needs_owner_ping"] is True
        assert len(pings) == 1 and pings[0]["origin"] == "example.com"
        assert cap.spent_today(SITE) == pytest.approx(0.05)  # not overrun

    def test_per_origin_isolation(self, tmp_path):
        cap = OriginCostCeiling(tmp_path / "ceiling.json", daily_cap_usd=0.05)
        cap.record_spend("https://a.com/x", 0.05)
        ok, _ = cap.can_spend("https://b.com/x", 0.01)
        assert ok is True


# ── identity binding (TrialVault) ────────────────────────────────────────────

class TestTrialIdentityBinder:
    def test_one_per_service_reuse(self, tmp_path):
        binder = TrialIdentityBinder(tmp_path / "trial-home")
        first = binder.bind("example-service")
        second = binder.bind("example-service")
        assert first == second  # same identity, never a second one
        assert binder.has("example-service") is True
        assert binder.has("other-service") is False

    def test_invariant_cross_service_reuse_impossible(self, tmp_path):
        """INVARIANT: identity_for reads only its own service's entry."""
        binder = TrialIdentityBinder(tmp_path / "trial-home")
        a = binder.bind("service-a")
        b = binder.bind("service-b")
        assert a["email"] != b["email"]
        assert binder.identity_for("service-a")["service"] == "service-a"
        assert binder.identity_for("service-b")["service"] == "service-b"
        # tamper: an entry naming another service is refused, not coerced
        binder.vault.store("service-a", "trial-service-a-xxxx",
                           json.dumps({"service": "service-b",
                                       "email": "x@y.z"}))
        with pytest.raises(ValueError):
            binder.identity_for("service-a")

    def test_uses_trial_vault_not_accounts_vault(self, tmp_path):
        binder = TrialIdentityBinder(tmp_path / "trial-home")
        assert isinstance(binder.vault, TrialVault)
        binder.bind("svc")
        assert (tmp_path / "trial-home" / "trial_accounts.json").exists()

    def test_generator_synthetic_only(self):
        ident = generate_trial_identity("svc", rng=random.Random(1))
        assert ident["service"] == "svc"
        assert ident["disposable"] is True
        assert ident["email"].split("@")[1] in SYNTHETIC_DOMAINS
        assert ident["phone"].startswith("+234")
        # seeded → deterministic (except the wall-clock timestamp)
        again = generate_trial_identity("svc", rng=random.Random(1))
        again.pop("created_at")
        ident_cmp = dict(ident)
        ident_cmp.pop("created_at")
        assert again == ident_cmp

    def test_invariant_generator_has_no_real_detail_input(self):
        """INVARIANT (code-level): the generator cannot read real details —
        it has no parameter, import, or read path for them."""
        src = inspect.getsource(identity_mod)
        for token in ("owner_identity", "contacts", "real_email",
                      "real_phone", "get_owner", "CredentialVault"):
            assert token not in src, f"forbidden token in generator: {token}"
        assert "TrialVault" in src  # bound in the trial vault, nowhere else


# ── audio: local ASR first ───────────────────────────────────────────────────

class TestAudioLocalFirst:
    def test_local_asr_wins_and_service_not_called(self):
        calls: list = []
        res = solve_audio_local_first(
            b"audio-bytes",
            asr_fn=lambda audio, language="en": "hello world",
            service_fn=lambda audio: (calls.append("service"), {"ok": True,
                         "text": "other"})[1])
        assert res == {"ok": True, "backend": "whisper.cpp",
                       "text": "hello world", "local": True}
        assert calls == []

    def test_service_only_after_local_fails(self):
        res = solve_audio_local_first(
            b"audio-bytes",
            asr_fn=lambda audio, language="en": "",
            service_fn=lambda audio: {"ok": True, "text": "from-service",
                                      "cost_usd": 0.002})
        assert res["backend"] == "service" and res["local"] is False
        assert res["text"] == "from-service"

    def test_neither_available(self):
        res = solve_audio_local_first(
            b"audio-bytes",
            asr_fn=lambda audio, language="en": "",
            service_fn=None)
        assert res["ok"] is False


# ── staged pipeline ──────────────────────────────────────────────────────────

class TestCaptchaPipeline:
    def test_non_captcha_first_beats_solving(self):
        service_calls: list = []

        def fetcher(url):
            return (200, {}, "<p>content</p>")

        res = solve_pipeline(
            {"kind": "recaptcha_v2", "page_url": SITE},
            url=SITE, fetcher=fetcher,
            service_fn=lambda ch: service_calls.append(ch) or
            {"ok": True, "token": "tok"})
        assert res["stage"] == "non_captcha" and res["ok"] is True
        assert service_calls == []  # never reached the solver

    def test_429_short_circuits_to_backoff(self):
        res = solve_pipeline({"kind": "unknown", "http_status": 429,
                              "headers": {"Retry-After": "30"},
                              "page_url": SITE},
                             service_fn=lambda ch: {"ok": True,
                                                    "token": "tok"})
        assert res["stage"] == "rate_limit"
        assert res["action"] == "backoff"
        assert res["retry_after_s"] == 30.0

    def test_ocr_stage_before_service(self):
        service_calls: list = []
        res = solve_pipeline(
            {"kind": "image_captcha", "page_url": SITE,
             "image_bytes": b"img"},
            ocr_fn=lambda ch: {"ok": True, "text": "aB3d"},
            service_fn=lambda ch: service_calls.append(ch) or
            {"ok": True, "token": "tok"})
        assert res == {"ok": True, "stage": "ocr", "kind": "image_captcha",
                       "text": "aB3d"}
        assert service_calls == []

    def test_audio_stage_uses_local_first(self):
        res = solve_pipeline(
            {"kind": "audio_captcha", "page_url": SITE,
             "image_bytes": b"aud"},
            audio_fn=lambda audio: {"ok": True, "text": "seven",
                                    "local": True})
        assert res["stage"] == "audio" and res["text"] == "seven"

    def test_ceiling_blocks_service_and_pings_owner(self, tmp_path):
        pings: list = []
        service_calls: list = []
        ceiling = OriginCostCeiling(tmp_path / "ceiling.json",
                                    daily_cap_usd=0.01)
        ceiling.record_spend(SITE, 0.01)
        res = solve_pipeline(
            {"kind": "recaptcha_v2", "page_url": SITE},
            ceiling=ceiling, ping_owner=lambda p: pings.append(p),
            service_fn=lambda ch: service_calls.append(ch) or
            {"ok": True, "token": "tok"})
        assert res["stage"] == "ceiling"
        assert res["needs_owner_ping"] is True
        assert len(pings) == 1
        assert service_calls == []  # never spent past the ceiling

    def test_invariant_human_ping_only_when_stuck(self, tmp_path):
        """INVARIANT: the owner is pinged only after every auto stage fails."""
        pings: list = []
        audit = tmp_path / "audit.jsonl"
        res = solve_pipeline(
            {"kind": "image_captcha", "page_url": SITE,
             "image_bytes": b"img"},
            audit_path=audit,
            ocr_fn=lambda ch: {"ok": False, "text": ""},
            ping_owner=lambda p: pings.append(p))
        assert res["takeover"] is True and res["stage"] == "human"
        assert len(pings) == 1
        assert res["stages_tried"][-1] == "human"
        # audit rows carry cost/stage, never tokens or solved text
        rows = [json.loads(line) for line in audit.read_text().splitlines()]
        assert rows and all("token" not in r and "text" not in r
                            for r in rows)
        assert {r["stage"] for r in rows} >= {"passive", "ocr", "human"}

    def test_human_not_pinged_when_ocr_succeeds(self, tmp_path):
        pings: list = []
        res = solve_pipeline(
            {"kind": "image_captcha", "page_url": SITE,
             "image_bytes": b"img"},
            ocr_fn=lambda ch: {"ok": True, "text": "x9y2"},
            ping_owner=lambda p: pings.append(p))
        assert res["ok"] is True and pings == []

    def test_ledger_learns_from_pipeline(self, tmp_path):
        ledger = SuccessLedger(tmp_path / "ledger.json")
        flow = CaptchaFlow(ledger=ledger,
                           ocr_fn=lambda ch: {"ok": True, "text": "ab12"})
        flow.solve({"kind": "image_captcha", "page_url": SITE,
                    "image_bytes": b"img"})
        assert ledger.rate("image_captcha", "example.com", "ocr") == 1.0
