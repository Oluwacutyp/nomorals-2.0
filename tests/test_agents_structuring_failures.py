"""Tail-98 pins — nomorals/agents/structuring.py silent-failure sites.

213 E103 — ``_deterministic()`` tool matching: a crash in the
registry-matching block leaves ``tools`` at its initialized default
[] — the module documents tools as an enhancement over the
deterministic floor, and the rest of the brief is unaffected.
EXCLUSION.

231 E104 — ``_model_polish()`` permission gate: the
``reasoning_permitted`` check crashing returns None (keep the
deterministic brief) — the gate fails CLOSED, polish is refused,
nothing is fabricated. EXCLUSION.

307 E104 — ``_tool_catalog()`` primary build: registry crash falls
back to a names-only catalog (descriptions dropped, names kept).
Degradation inside a "never raises" helper. EXCLUSION.

310 E104 — ``_tool_catalog()`` fallback: the retry crashing yields
[] — the documented never-raises contract of the helper; the brief
then carries no tool suggestions. EXCLUSION.
"""
from types import SimpleNamespace

import nomorals.agents.reasoning as reasoning_mod
import nomorals.agents.structuring as S


def _arch(context):
    return S.PromptArchitect(context)


# ── 213: matching crash -> tools [] , brief intact ────────────────

class _IntNameRegistry:
    """Registry whose tool name is not a str: cataloguing succeeds,
    token matching on it crashes inside the 213 block."""

    def names(self):
        return [123]

    def get(self, name):
        return SimpleNamespace(description="deploy things")


def test_213_tool_matching_crash_leaves_tools_empty():
    ctx = SimpleNamespace(tools=_IntNameRegistry())
    out = _arch(ctx).structure("Build the thing and verify it works",
                               polish=False)
    assert out["tools"] == []
    assert out["intent"]  # deterministic floor unaffected
    assert "brief" in out


def test_213_control_matching_registry_suggests_tool():
    class _Reg:
        def names(self):
            return ["deploy_app"]

        def get(self, name):
            return SimpleNamespace(description="ship it")

    ctx = SimpleNamespace(tools=_Reg())
    out = _arch(ctx).structure("Deploy the app and verify it works",
                               polish=False)
    assert out["tools"] == ["deploy_app"]


# ── 231: polish gate crash -> None (fail-closed) ──────────────────

def test_231_polish_gate_crash_returns_none(monkeypatch):
    def _boom(context):
        raise RuntimeError("gate boom")

    monkeypatch.setattr(reasoning_mod, "reasoning_permitted", _boom)
    ctx = SimpleNamespace(router=object())
    base = {"intent": "i", "subgoals": ["s"], "inputs": [],
            "constraints": [], "acceptance": [], "tools": [],
            "risks": []}
    assert _arch(ctx)._model_polish("text", base) is None
    # control: gate answering False also keeps the deterministic brief
    monkeypatch.setattr(reasoning_mod, "reasoning_permitted",
                        lambda context: False)
    assert _arch(ctx)._model_polish("text", base) is None


# ── 307: catalog build crash -> names-only fallback ───────────────

class _FlakyRegistry:
    def __init__(self):
        self.calls = 0

    def names(self):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("registry boom")
        return ["alpha", "beta"]

    def get(self, name):
        return None


def test_307_catalog_crash_falls_back_to_names_only():
    assert S._tool_catalog(_FlakyRegistry()) == [("alpha", ""),
                                                 ("beta", "")]


# ── 310: catalog fallback crash -> [] ─────────────────────────────

class _DeadRegistry:
    def names(self):
        raise RuntimeError("registry dead")


def test_310_catalog_double_crash_returns_empty():
    assert S._tool_catalog(_DeadRegistry()) == []
    assert S._tool_catalog(None) == []
