"""Doc 17 Pass 4 — the /proxy status truth blocks (gap G6).

The shipped /proxy surface gains the two blocks it was missing,
rendered through its existing reply path:

(a) ACTIVATION — epoch, last merge (+added / already present /
    excluded), registry size and the corrupt-recovery note, read
    from P1's durable record; the states render DISTINCTLY:
    never-activated ≠ a failed attempt (the lab failure ledger is
    the record of attempts that never completed a run) ≠ current
    ≠ the named empty stall.
(b) RECEIPTS — the newest decisions from P2's journal, outcomes
    folded, bounded to a fixed short list plus a pointer at the
    full journal (no paged dumps), with no internal decision or
    proxy ids anywhere in the text.

Both blocks are fail-soft: a broken read drops the block, never
the status reply.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.partner.runtime_system import RuntimeSystemMixin
from nomorals.core.proxy_routing import RouteAuthority, RouteReceiptJournal
from nomorals.tools import proxy_sources, proxysources
from nomorals.tools.proxylab_failures import clear_failures, record_failure
from nomorals.tools.proxysources import SourceRegistry


class _FakeTools:
    """The one tool the status verb calls, with its shipped
    value shape."""

    def call(self, name: str, **kwargs):
        assert name == "proxy_status"
        return SimpleNamespace(
            ok=True, value={"known": [], "active": "direct",
                            "raw_sockets_routed": False})


def _runtime(home: Path) -> RuntimeSystemMixin:
    rt = RuntimeSystemMixin()
    rt.context = SimpleNamespace(
        settings=SimpleNamespace(home=str(home)), tools=_FakeTools())
    return rt


def _status(home: Path) -> str:
    return _runtime(home)._control_proxy("status")


def _activate(home: Path) -> None:
    proxies = home / "proxies"
    proxies.mkdir(parents=True, exist_ok=True)
    registry = SourceRegistry(proxies / "sources.json")
    proxy_sources.activate_sources(registry)


# ── (a) activation states ────────────────────────────────────────────────────

def test_never_activated_state(tmp_path):
    clear_failures()
    out = _status(tmp_path / "home")
    # the pre-existing status lines are untouched, the block appends
    assert out.startswith("outbound: direct\nknown: none\n")
    assert "not yet activated on this host" in out
    assert "catalogue activation: epoch" not in out
    assert "route receipts: none recorded yet" in out


def test_failed_attempt_reads_differently_from_never(tmp_path):
    clear_failures()
    record_failure("proxylab.activate_sources",
                   RuntimeError("registry write exploded"))
    out = _status(tmp_path / "home")
    assert "not yet activated on this host" in out
    assert "⚠ the latest activation attempt failed: " \
           "registry write exploded" in out


def test_current_state_after_activation(tmp_path):
    clear_failures()
    home = tmp_path / "home"
    _activate(home)
    out = _status(home)
    assert "🧬 catalogue activation: epoch 1 · registry holds " \
           "116 sources" in out
    assert "+15 added · 14 already present · 3 excluded" in out
    assert "not yet activated" not in out
    assert "activation attempt failed" not in out


def test_failed_attempt_after_a_recorded_run_still_warns(tmp_path):
    home = tmp_path / "home"
    _activate(home)
    clear_failures()
    record_failure("proxylab.activate_sources",
                   RuntimeError("disk full"))
    out = _status(home)
    assert "catalogue activation: epoch 1" in out
    assert "⚠ the latest activation attempt failed: disk full" in out


def test_empty_merge_is_a_named_stall(tmp_path):
    home = tmp_path / "home"
    proxies = home / "proxies"
    proxies.mkdir(parents=True)
    registry = SourceRegistry(proxies / "sources.json")
    with mock.patch.object(proxysources, "BUILT_IN_SOURCES", []), \
            mock.patch.object(proxy_sources, "SOURCES_2026", ()):
        proxy_sources.activate_sources(registry)
    clear_failures()
    out = _status(home)
    assert "catalogue activation: epoch 1" in out
    assert "left the registry EMPTY" in out


def test_corrupt_recovery_note_renders(tmp_path):
    home = tmp_path / "home"
    proxies = home / "proxies"
    proxies.mkdir(parents=True)
    (proxies / "sources.json").write_bytes(b"\x89not json at all")
    _activate(home)
    clear_failures()
    out = _status(home)
    assert "a corrupt registry was recovered on the way in" in out
    backups = sorted(proxies.glob("sources.json.corrupt-*.bak"))
    assert len(backups) == 1
    assert backups[0].name in out


# ── (b) receipts block ───────────────────────────────────────────────────────

def test_receipts_block_is_bounded_and_id_free(tmp_path):
    home = tmp_path / "home"
    journal = RouteReceiptJournal(home / "proxies"
                                  / "route_receipts.jsonl")
    authority = RouteAuthority(journal)
    minted = []
    for i in range(7):
        minted.append(authority.record(
            consumer="core_http", route="free",
            reason_code="strategy:round_robin",
            host=f"host{i}.example", source="proxylab",
            proxy_id=f"10.0.0.{i}:8080"))
    authority.record_outcome(minted[-1].decision_id, "failed",
                             "connection reset")
    clear_failures()

    out = _status(home)

    assert "🧾 route receipts — latest 5 of 7 decision(s):" in out
    assert "… 2 earlier decision(s) in the journal" in out
    assert "outcome: failed" in out
    assert "host6.example" in out       # newest shown
    assert "host0.example" not in out   # the bound drops the oldest
    for receipt in minted:
        assert receipt.decision_id not in out
    assert "10.0.0.6:8080" not in out   # proxy ids never print


def test_no_route_decision_renders_as_no_route(tmp_path):
    home = tmp_path / "home"
    journal = RouteReceiptJournal(home / "proxies"
                                  / "route_receipts.jsonl")
    RouteAuthority(journal).record(
        consumer="browser", route="none",
        reason_code="tier_gate_refusal", host="vault.example",
        source="proxypool")
    clear_failures()
    out = _status(home)
    assert "browser → NO ROUTE vault.example · tier_gate_refusal" \
        in out


# ── fail-soft ────────────────────────────────────────────────────────────────

def test_blocks_fail_soft_and_status_survives(tmp_path):
    rt = _runtime(tmp_path / "home")
    with mock.patch(
            "nomorals.tools.proxy_sources.activation_state_for_home",
            side_effect=OSError("store gone")), \
            mock.patch(
                "nomorals.core.proxy_routing."
                "RouteReceiptJournal.recent",
                side_effect=OSError("journal gone")):
        out = rt._control_proxy("status")
    assert out.startswith("outbound: direct\nknown: none\n")
    assert "catalogue activation" not in out
    assert "route receipts" not in out
    # home resolution follows the lab's idiom end to end
    assert os.path.basename(str(tmp_path))  # (fixture sanity)
