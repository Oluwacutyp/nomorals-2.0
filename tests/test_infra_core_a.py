"""Section A (core runtime) — tests for the new infrastructure behavior.

Covers: config hot-reload + precedence fix + validated set()/diff/schema,
policy conditional rules + timed grants + child narrowing, cloud detection,
profile pinning, shutdown ordering, registry health/stats/describe, and the
runtune threads auto-tune fix.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest


# ── config ───────────────────────────────────────────────────────────────────

def _env(**kw):
    base = {"NM_PROFILE": "workstation", "NM_HOME": tempfile.mkdtemp()}
    base.update(kw)
    return base


def test_profile_preset_layers_under_toml():
    from nomorals.core.config import load_settings
    d = tempfile.mkdtemp()
    toml = Path(d) / "c.toml"
    # workstation preset sets concurrency.threads=16; TOML must win over it
    toml.write_text("[concurrency]\nthreads = 42\n")
    s = load_settings(config_file=str(toml), env=_env())
    assert s.concurrency.threads == 42


def test_settings_set_coerces_and_validates():
    from nomorals.core.config import load_settings
    from nomorals.core.errors import ConfigError
    s = load_settings(env=_env())
    s2 = s.set("llm.timeout", "33")
    assert s2.llm.timeout == 33.0 and isinstance(s2.llm.timeout, float)
    assert s.llm.timeout != 33.0  # original untouched
    with pytest.raises(ConfigError):
        s.set("bogus.path", 1)
    with pytest.raises(ConfigError):
        s.set("budget.wall_seconds", "-5")


def test_diff_settings_masks_secrets():
    from nomorals.core.config import diff_settings, load_settings
    s = load_settings(env=_env())
    s2 = s.set("llm.timeout", "31").set("llm.hf_token", "secret-abcdef-123456")
    changed = {c["path"]: c for c in diff_settings(s, s2)}
    assert changed["llm.timeout"]["new"] == 31.0
    assert changed["llm.hf_token"]["new"] != "secret-abcdef-123456"
    assert "…" in changed["llm.hf_token"]["new"]


def test_config_schema_shape():
    from nomorals.core.config import config_schema
    sc = config_schema()
    assert "llm" in sc["sections"]
    assert sc["sections"]["llm"]["fields"]["timeout"]["type"] == "float"
    assert sc["sections"]["llm"]["fields"]["hf_token"]["secret"] is True
    assert set(sc["profiles"]) == {"workstation", "laptop", "termux"}


def test_watcher_hot_reload_and_broken_keeps_old():
    from nomorals.core.config import watch_settings
    d = tempfile.mkdtemp()
    toml = Path(d) / "w.toml"
    toml.write_text("[llm]\ntimeout = 120.0\n")
    events = []
    s, w = watch_settings(config_file=str(toml), env=_env(),
                          poll_interval_s=0.2, debounce_s=0.1,
                          use_watchfiles=False,
                          on_change=lambda o, n, c: events.append(c))
    try:
        w.start()
        time.sleep(0.4)
        toml.write_text("[llm]\ntimeout = 44.0\n")
        time.sleep(1.5)
        assert w.current.llm.timeout == 44.0
        assert events and events[0][0]["path"] == "llm.timeout"
        toml.write_text("[[[broken")
        time.sleep(1.5)
        assert w.current.llm.timeout == 44.0  # validate-before-swap
        assert w.stats()["failures"] >= 1
    finally:
        w.stop()


# ── policy ───────────────────────────────────────────────────────────────────

def test_conditional_rule_resource_binding():
    from nomorals.core.policy import Policy, CapabilitySet
    p = Policy()
    p.deny("fs.write", note="writes stay in the workspace")
    p.allow("fs.write", priority=200,
            when=lambda ctx: str(ctx.get("path", "")).startswith("/ws/"))
    grant = CapabilitySet.of("fs.write")
    assert p.check("fs.write", grant=grant,
                   context={"path": "/ws/a.txt"}).allowed
    d = p.check("fs.write", grant=grant, context={"path": "/etc/x"})
    assert not d.allowed and "workspace" in d.reason


def test_broken_condition_fails_closed():
    from nomorals.core.policy import Policy, CapabilitySet
    p = Policy()
    p.allow("fs.write", when=lambda ctx: 1 / 0)
    d = p.check("fs.write", grant=CapabilitySet.of("fs.write"), context={})
    assert not d.allowed


def test_timed_grant_expiry_and_revoke():
    from nomorals.core.policy import Policy, TimedGrant
    p = Policy()
    g = p.issue_grant("fs.read", ttl_s=0.2, actor="agent1")
    assert g.grants("fs.read")
    assert p.check("fs.read", grant=g).allowed
    time.sleep(0.3)
    assert not g.grants("fs.read")
    assert not p.check("fs.read", grant=g).allowed
    g2 = p.issue_grant("fs.read", ttl_s=60)
    p.revoke_grant(g2)
    assert not g2.grants("fs.read")


def test_narrow_grant_never_widens():
    from nomorals.core.policy import CapabilitySet, TimedGrant, narrow_grant
    parent = CapabilitySet.of("fs.read", "net.out")
    child = narrow_grant(parent, "research")  # research wants fs.write too
    assert child.grants("fs.read")
    assert not child.grants("fs.write")  # parent didn't hold it
    tg = TimedGrant("fs.*", ttl_s=100)
    tc = narrow_grant(tg, "research")
    assert isinstance(tc, TimedGrant) and tc.remaining_s() <= 100


def test_proposal_store_pluggable():
    from nomorals.core.policy import PermissionGradient, Policy, PolicyStore
    seen: dict = {}

    class DictStore:
        def save_proposal(self, proposal):
            seen[proposal["proposal_id"]] = dict(proposal)
        def get_proposal(self, pid):
            return dict(seen[pid]) if pid in seen else None
        def pending_proposals(self):
            return [dict(p) for p in seen.values()
                    if p["status"] == "pending"]
        def update_proposal(self, pid, updates):
            if pid not in seen:
                return False
            seen[pid].update(updates)
            return True

    assert isinstance(DictStore(), PolicyStore)  # structural
    g = PermissionGradient(Policy(), store=DictStore())
    pr = g.propose("social.post", actor="a", draft="hi")
    assert g.get_proposal(pr["proposal_id"])["status"] == "pending"
    assert len(g.pending_proposals()) == 1
    assert g.reject_proposal(pr["proposal_id"], note="no")
    assert g.get_proposal(pr["proposal_id"])["status"] == "rejected"


# ── profile / profiles / runtune ─────────────────────────────────────────────

def test_detect_cloud_pin():
    from nomorals.core.profile import detect_cloud, detect_profile
    os.environ["NM_CLOUD"] = "aws"
    try:
        assert detect_cloud() == "aws"
        assert detect_profile().cloud == "aws"
        assert detect_profile().to_dict()["cloud"] == "aws"
    finally:
        del os.environ["NM_CLOUD"]
    assert detect_cloud() in ("none", "aws", "gcp", "azure", "other")


def test_get_profile_kind_pinned_arg():
    from nomorals.core.profiles import get_profile_kind, describe_profile
    assert get_profile_kind(pinned="termux") == "termux"
    assert get_profile_kind(pinned="workstation") == "workstation"
    d = describe_profile(pinned="termux")
    assert d["kind"] == "termux" and d["pin_source"] == "argument"
    assert "ctx_size" in d["values"]


def test_runtune_threads_auto_and_summary():
    from nomorals.core.config import Settings, load_settings
    from nomorals.core.runtune import build_tune
    s = load_settings(env=_env())
    assert s.runtime.threads == 0  # 0 = auto, not an override
    # explicit runtime.threads wins over everything
    t = build_tune(s.set("runtime.threads", "24"))
    assert t.threads == 24
    assert any("explicit runtime.threads" in n for n in t.notes)
    # bare defaults: nothing deliberate -> the auto path scales by CPU/RAM
    t2 = build_tune(Settings())
    assert any("(auto" in n for n in t2.notes if n.startswith("threads"))
    assert "cloud:" in t2.summary()
    assert t2.max_upload_mb > 0


# ── shutdown ─────────────────────────────────────────────────────────────────

def test_shutdown_ordering_and_timeout():
    from nomorals.core.shutdown import ShutdownCoordinator
    c = ShutdownCoordinator(name="t")
    order = []
    c.register("listen", lambda r: order.append("listen"), priority=100)
    c.register("drain", lambda: order.append("drain"), priority=50)
    c.register("slow", lambda: time.sleep(30), priority=0, timeout_s=0.2)
    c.register("beacon", lambda: order.append("beacon"), priority=-100)
    assert c.ready
    c.request_shutdown("test")
    assert not c.ready
    rep = c.run()
    assert order == ["listen", "drain", "beacon"]
    statuses = {h["name"]: h["status"] for h in rep["hooks"]}
    assert statuses["slow"] == "timeout"
    assert rep["stopped"] is True
    assert c.run() is rep  # idempotent
    assert c.beacon()["stopped"] is True


def test_shutdown_startup_ascending():
    from nomorals.core.shutdown import ShutdownCoordinator
    c = ShutdownCoordinator()
    seq = []
    c.register("a", lambda: seq.append("a"), priority=10, phase="startup")
    c.register("b", lambda: seq.append("b"), priority=1, phase="startup")
    c.run_startup()
    assert seq == ["b", "a"]


# ── registry ─────────────────────────────────────────────────────────────────

def _registry():
    from nomorals.tools.registry import ToolRegistry
    r = ToolRegistry(enforce=False)

    def add(a: int, b: int = 1):
        """Add."""
        return a + b

    def boom():
        """Boom."""
        raise RuntimeError("x")

    r.register("add", add, capability="exec.code", kind="compute",
               version="1.0")
    r.register("boom", boom, capability="exec.code")
    return r


def test_registry_per_tool_stats_and_describe():
    r = _registry()
    r.call("add", a=1)
    r.call("boom")
    stats = r.tool_stats("add")
    assert stats["calls"] == 1 and stats["last_status"] == "ok"
    assert r.tool_stats("boom")["errors"] == 1
    d = r.describe("add")
    assert d["version"] == "1.0" and d["capability"] == "exec.code"
    assert d["stats"]["calls"] == 1
    assert r.describe("nope") is None
    assert r.by_capability("exec.code") == ["add", "boom"]
    assert r.capabilities_used() == ["exec.code"]
    assert r.by_kind("compute") == ["add"]


def test_registry_alias_and_deprecate():
    r = _registry()
    assert r.alias("plus", "add")
    assert not r.alias("x", "missing")
    assert r.call("plus", a=1, b=2).value == 3
    assert r.describe("plus")["alias_of"] == "add"
    assert r.deprecate("add", replaced_by="plus")
    assert r.describe("add")["deprecated"] is True
    assert r.call("add", a=1).value == 2  # still callable


def test_registry_health_degraded_then_down():
    r = _registry()
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        raise RuntimeError("unhealthy")

    assert r.register_health("boom", probe)
    assert not r.register_health("missing", probe)
    assert r.check_health("boom").status == "degraded"
    assert r.check_health("boom").status == "degraded"
    h = r.check_health("boom")
    assert h.status == "down" and h.consecutive_failures == 3
    r.register_health("add", lambda: True)
    assert r.check_health("add").status == "ok"
    rep = r.health_report()
    assert rep["counts"] == {"down": 1, "ok": 1}
