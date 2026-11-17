"""Gapfix R2-40 pins — net_leak_check tool surface.

The check itself (STUN parse, verdicts, headlines) is already pinned by
tests/test_security_sweep.py, tests/test_security_pass47.py and
tests/test_security_netleak_failures.py. This file pins only the TOOL
SURFACE: registration, capability, and the three verdicts as the brain
sees them through the tool call. Transport is faked at the same patch
points the sweep uses — no live STUN egress on this box.
"""

from __future__ import annotations

from unittest import mock

from nomorals.security import netleak as N


def _registry():
    from nomorals.tools.registry import ToolRegistry

    return ToolRegistry().register_builtins()


def _fake_transport(monkeypatch, *, http_ip, stun_ip, local_ips, v6):
    monkeypatch.setattr(N, "_http_text", lambda url, timeout=10: http_ip)
    monkeypatch.setattr(N, "stun_public_ip", lambda *a, **k: stun_ip)
    monkeypatch.setattr(N, "local_interface_ips", lambda: local_ips)
    monkeypatch.setattr(N, "ipv6_egress", lambda: v6)


class _CapFake:
    """Minimal registry double recording capabilities (pass47 pattern)."""

    def __init__(self):
        self.context = None
        self.caps: dict[str, object] = {}

    def register(self, name, description="", capability=None, parameters=None):
        def deco(fn):
            self.caps[name] = capability
            return fn

        return deco


def test_net_leak_check_registered():
    reg = _registry()
    assert "net_leak_check" in reg._tools


def test_net_leak_check_carries_net_out_capability():
    from nomorals.core.policy import Capability
    from nomorals.tools import security as security_mod

    fake = _CapFake()
    security_mod.register(fake)
    assert fake.caps["net_leak_check"] == Capability.NET_OUT


def test_net_leak_check_reports_leak_with_address_named(monkeypatch):
    _fake_transport(
        monkeypatch,
        http_ip="198.51.100.7",
        stun_ip="203.0.113.9",
        local_ips=[{"ip": "192.168.1.5", "class": "rfc1918-private"}],
        v6={"egress": "", "local": [], "supported": True},
    )
    out = _registry()._tools["net_leak_check"].fn()
    assert out["ok"] is False
    assert out["webrtc_leak"] is True
    assert out["verdict"] == "leak"
    # the observed address is named, not just a boolean
    assert out["stun_ip"] == "203.0.113.9"
    assert "203.0.113.9" in out["webrtc_detail"]
    assert out["http_public_ip"] == "198.51.100.7"


def test_net_leak_check_reports_clean(monkeypatch):
    _fake_transport(
        monkeypatch,
        http_ip="198.51.100.7",
        stun_ip="198.51.100.7",
        local_ips=[],
        v6={"egress": "", "local": [], "supported": False},
    )
    out = _registry()._tools["net_leak_check"].fn()
    assert out["ok"] is True
    assert out["webrtc_leak"] is False
    assert out["verdict"] == "clean"


def test_net_leak_check_inconclusive_when_stun_unreachable(monkeypatch):
    # STUN down AND no HTTP egress: nothing was measured on either path.
    # Must say so honestly — never a fake clean, never a fake leak.
    monkeypatch.setattr(
        N, "_http_text", mock.Mock(side_effect=OSError("network down")))
    monkeypatch.setattr(N, "stun_public_ip", lambda *a, **k: "")
    monkeypatch.setattr(N, "local_interface_ips", lambda: [])
    monkeypatch.setattr(
        N, "ipv6_egress", lambda: {"egress": "", "local": [], "supported": False})
    out = _registry()._tools["net_leak_check"].fn()
    assert out["verdict"] == "inconclusive"
    assert out["webrtc_leak"] is False
    assert "STUN" in out["note"]
    assert out["ok"] is False


def test_net_leak_check_http_only_is_inconclusive_not_clean(monkeypatch):
    # The subtle case: HTTP egress known, STUN unreachable. rep.ok reads
    # True (no leak found on the paths that ran) but the WebRTC verdict is
    # unknown — the tool must not let "ok" be read as "clean".
    _fake_transport(
        monkeypatch,
        http_ip="198.51.100.7",
        stun_ip="",
        local_ips=[],
        v6={"egress": "", "local": [], "supported": False},
    )
    out = _registry()._tools["net_leak_check"].fn()
    assert out["verdict"] == "inconclusive"
    assert "STUN" in out["note"]


def test_net_leak_check_ipv6_egress_is_advisory_not_conviction(monkeypatch):
    _fake_transport(
        monkeypatch,
        http_ip="198.51.100.7",
        stun_ip="198.51.100.7",
        local_ips=[],
        v6={"egress": "2001:db8::1", "local": ["2001:db8::1"],
            "supported": True},
    )
    out = _registry()._tools["net_leak_check"].fn()
    assert out["ipv6_leak"] is True
    assert out["ipv6_egress"] == "2001:db8::1"
    # advisory: surfaced, not a conviction — the WebRTC verdict stays clean
    assert out["verdict"] == "clean"
    assert out["webrtc_leak"] is False


def test_net_leak_check_expected_ip_strict_form(monkeypatch):
    _fake_transport(
        monkeypatch,
        http_ip="198.51.100.7",
        stun_ip="203.0.113.9",
        local_ips=[],
        v6={"egress": "", "local": [], "supported": True},
    )
    out = _registry()._tools["net_leak_check"].fn(
        expected_ip="198.51.100.7")
    assert out["webrtc_leak"] is True
    assert out["verdict"] == "leak"
    assert "198.51.100.7" in out["webrtc_detail"]


def test_net_leak_check_error_path_is_honest(monkeypatch):
    monkeypatch.setattr(
        N, "check_net_leak", mock.Mock(side_effect=RuntimeError("boom")))
    out = _registry()._tools["net_leak_check"].fn()
    assert out["ok"] is False
    assert "boom" in out["error"]


def test_netleak_report_verdict_matches_format_report_headline():
    # The verdict property is the single source for the headline rule:
    # leak / clean / inconclusive must agree with format_report.
    from nomorals.security.netleak import NetLeakReport, format_report

    leak = NetLeakReport(ok=False, http_public_ip="198.51.100.7",
                         stun_ip="203.0.113.9", webrtc_leak=True,
                         webrtc_detail="STUN sees 203.0.113.9")
    assert leak.verdict == "leak"
    assert "LEAK DETECTED" in format_report(leak)

    clean = NetLeakReport(ok=True, http_public_ip="198.51.100.7",
                          stun_ip="198.51.100.7")
    assert clean.verdict == "clean"
    assert "NO LEAK" in format_report(clean)

    unknown = NetLeakReport(ok=True, http_public_ip="198.51.100.7",
                            stun_ip="")
    assert unknown.verdict == "inconclusive"
    assert "INCONCLUSIVE" in format_report(unknown)
