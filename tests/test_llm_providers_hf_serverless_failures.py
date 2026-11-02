"""Tail-114 pins: nomorals/llm/providers/hf_serverless.py census
sites (165, 232, 332). All three are held as exclusions — one
fallback ladder, three rungs:

165 — ``fetch_catalog``: the live-catalog request raising -> the
     curated ``ROUTER_FALLBACK_MODELS`` list, byte-identical to the
     fallback the same function returns for a non-ok response and
     for an empty catalog. The degradation is the documented shape
     of the function, not a fabricated catalog.
232 — ``health``: the liveness probe raising -> False (in-code
     comment: "liveness probe"). The probe's honest negative; the
     router fails over on it.
332 — ``_candidate_models``: the live-catalog rung raising -> fall
     through to rung 2, the curated list (in-code comment prints
     the ladder). The curated tail is appended either way; the
     crash only loses the live head.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from nomorals.llm.providers.hf_serverless import (
    ROUTER_FALLBACK_MODELS,
    HFServerlessProvider,
)


class _FakeHTTP:
    def __init__(self, behaviour: str):
        self.behaviour = behaviour

    def get(self, url, timeout=None):
        if self.behaviour == "raise":
            raise RuntimeError("net down")
        resp = MagicMock()
        if self.behaviour == "ok":
            resp.ok = True
            resp.json = lambda: {"data": [{"id": "m/one", "providers": []},
                                          {"id": "m/two", "providers": []}]}
        else:
            resp.ok = False
        return resp


def _provider(behaviour: str) -> HFServerlessProvider:
    provider = HFServerlessProvider(token="t")
    provider.http = _FakeHTTP(behaviour)
    return provider


# ── 165: catalog fetch crash -> curated fallback ─────────────────────

def test_165_fetch_crash_returns_curated_fallback():
    catalog = _provider("raise").fetch_catalog()
    assert [c["id"] for c in catalog] == list(ROUTER_FALLBACK_MODELS)


def test_165_non_ok_response_returns_the_same_fallback():
    catalog = _provider("notok").fetch_catalog()
    assert [c["id"] for c in catalog] == list(ROUTER_FALLBACK_MODELS)


def test_165_control_live_catalog_is_normalized():
    catalog = _provider("ok").fetch_catalog()
    assert [c["id"] for c in catalog] == ["m/one", "m/two"]


# ── 232: health probe crash -> False ─────────────────────────────────

def test_232_health_crash_returns_false():
    assert _provider("raise").health() is False


def test_232_control_health_ok_returns_true():
    assert _provider("ok").health() is True


# ── 332: candidate-models catalog rung crash -> curated rung ────────

def test_332_catalog_crash_falls_through_to_curated():
    provider = HFServerlessProvider(token="t")

    def _boom():
        raise RuntimeError("catalog down")

    provider.fetch_catalog = _boom
    out = provider._candidate_models(exclude=ROUTER_FALLBACK_MODELS[0])
    assert out == list(ROUTER_FALLBACK_MODELS[1:])


def test_332_control_live_catalog_heads_the_list():
    provider = HFServerlessProvider(token="t")
    provider.fetch_catalog = lambda: [{"id": "live/one"},
                                      {"id": "live/two"}]
    out = provider._candidate_models()
    assert out[:2] == ["live/one", "live/two"]
    assert list(ROUTER_FALLBACK_MODELS)[0] in out
