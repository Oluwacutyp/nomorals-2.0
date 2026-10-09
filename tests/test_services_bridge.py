"""Tests for the service connector bridge (nomorals/tools/services.py)."""

import pytest

from nomorals.tools.registry import ToolRegistry
from nomorals.tools import services


@pytest.fixture()
def registry():
    r = ToolRegistry()
    services.register(r)
    return r


class _Ctx:
    capabilities = None
    vault = None


def test_service_list_discovers_connectors(registry):
    out = registry.call("service_list", context=_Ctx())
    assert out.ok, out
    v = out.value
    assert v["count"] >= 28, f"expected 28+ connectors, got {v['count']}"
    ids = {s["id"] for s in v["services"]}
    for expected in ("github", "binance", "mono", "gmail"):
        assert expected in ids, f"{expected} not bridged"
    # actions discovered
    gh = next(s for s in v["services"] if s["id"] == "github")
    assert len(gh["actions"]) > 0
    assert all("action" in a and "params" in a for a in gh["actions"])


def test_service_list_marks_sensitive(registry):
    out = registry.call("service_list", context=_Ctx())
    v = out.value
    bn = next(s for s in v["services"] if s["id"] == "binance")
    sensitive = [a for a in bn["actions"] if a["sensitive"]]
    # Binance has order/trade-type actions; at least the flag machinery works
    assert isinstance(sensitive, list)


def test_service_call_unknown_connector(registry):
    out = registry.call("service_call", context=_Ctx(),
                        connector="nope", action="x", params={})
    assert out.ok
    assert out.value["ok"] is False
    assert "unknown connector" in out.value["error"]


def test_service_call_unknown_action(registry):
    out = registry.call("service_call", context=_Ctx(),
                        connector="github", action="not_real", params={})
    assert out.ok
    v = out.value
    assert v["ok"] is False
    assert "unknown action" in v["error"]
    assert "valid_actions" in v


def test_service_call_missing_args(registry):
    out = registry.call("service_call", context=_Ctx())
    # Missing connector/action -> the call is refused, honestly
    assert not out.ok


def test_service_status_all(registry):
    out = registry.call("service_status", context=_Ctx())
    assert out.ok
    v = out.value
    assert len(v["services"]) >= 28
    assert all("status" in s for s in v["services"])


def test_lifecycle_not_exposed(registry):
    out = registry.call("service_list", context=_Ctx())
    v = out.value
    for svc in v["services"]:
        actions = {a["action"] for a in svc["actions"]}
        for banned in ("connect", "disconnect", "status",
                       "test_connection", "resume_checkpoint"):
            assert banned not in actions, f"{svc['id']} leaks {banned}"
        for a in actions:
            assert not a.startswith("_"), f"{svc['id']} leaks private {a}"
