"""Offline tests for build-map #92: multi-agent congestion prediction."""

import os
import tempfile

import pytest

from nomorals.planning.congestion import (
    ContentionMonitor,
    format_status,
    pre_fanout_check,
)


def _mon():
    return ContentionMonitor(db_path=os.path.join(tempfile.mkdtemp(), "c.db"))


def test_register_and_list():
    m = _mon()
    r = m.register("groq-key", kind="api_key", capacity=10)
    assert r is not None and r.capacity == 10 and r.kind == "api_key"
    assert any(x.name == "groq-key" for x in m.list_resources())


def test_register_bad_kind_falls_back():
    m = _mon()
    r = m.register("weird", kind="nonsense")
    assert r is not None and r.kind == "endpoint"


def test_acquire_release_depth():
    m = _mon()
    m.register("ep", capacity=4)
    assert m.queue_depth("ep") == 0
    m.acquire("ep", "agent-1")
    m.acquire("ep", "agent-2")
    assert m.queue_depth("ep") == 2
    assert m.release("ep", "agent-1") is True
    assert m.queue_depth("ep") == 1
    assert m.release("ep", "agent-1") is False  # already released


def test_acquire_unknown_resource():
    m = _mon()
    assert m.acquire("nope", "a1") is None
    assert m.queue_depth("nope") == 0


def test_predict_jam_unknown():
    m = _mon()
    prob, eta = m.predict_jam("nope")
    assert prob == 0.0 and eta is None


def test_predict_jam_clear():
    m = _mon()
    m.register("ep", capacity=8)
    prob, eta = m.predict_jam("ep")
    assert prob < 0.3


def test_predict_jam_congested():
    m = _mon()
    m.register("ep", capacity=2)
    m.acquire("ep", "a1")
    m.acquire("ep", "a2")
    m.acquire("ep", "a3")  # 3/2 = 150% utilization
    prob, eta = m.predict_jam("ep")
    assert prob >= 0.55


def test_predict_jam_never_raises():
    m = ContentionMonitor(db_path="/nonexistent-dir-xyz/c.db")
    prob, eta = m.predict_jam("anything")
    assert prob == 0.0


def test_advise_proceed_when_clear():
    m = _mon()
    m.register("ep", capacity=8)
    a = m.advise("ep")
    assert a["action"] == "proceed"


def test_advise_backoff_when_jammed():
    m = _mon()
    m.register("ep", capacity=2)
    for i in range(4):
        m.acquire("ep", f"a{i}")
    a = m.advise("ep", agents_waiting=4)
    assert a["action"] in ("backoff", "switch")


def test_advise_switch_to_alternate():
    m = _mon()
    m.register("key-a", kind="api_key", capacity=1)
    m.register("key-b", kind="api_key", capacity=8)
    m.register_alternate("key-a", "key-b")
    for i in range(3):
        m.acquire("key-a", f"a{i}")
    a = m.advise("key-a", agents_waiting=3)
    assert a["action"] == "switch"
    assert a["alternate"] == "key-b"


def test_advise_stagger_mid_load():
    m = _mon()
    m.register("ep", capacity=4)
    for i in range(3):  # 75% — below jam threshold but warming
        m.acquire("ep", f"a{i}")
    a = m.advise("ep", agents_waiting=3)  # waiting agents push pressure up
    assert a["action"] in ("stagger", "proceed", "backoff")


def test_advise_unknown_resource():
    m = _mon()
    a = m.advise("nope")
    assert a["action"] == "proceed"


def test_register_alternate_missing():
    m = _mon()
    m.register("key-a", kind="api_key")
    assert m.register_alternate("key-a", "key-missing") is False


def test_pre_fanout_check():
    m = _mon()
    m.register("ep1", capacity=8)
    m.register("ep2", capacity=2)
    m.acquire("ep2", "a1")
    m.acquire("ep2", "a2")
    out = pre_fanout_check(m, ["ep1", "ep2"], agents_waiting=2)
    assert out["ep1"]["action"] == "proceed"
    assert out["ep2"]["action"] in ("backoff", "switch", "stagger")


def test_pre_fanout_none_monitor():
    assert pre_fanout_check(None, ["ep1"]) == {}


def test_status_format():
    m = _mon()
    assert "no resources" in format_status(m)
    m.register("ep", capacity=4)
    m.acquire("ep", "a1")
    s = format_status(m)
    assert "ep" in s and "1/4" in s


def test_chat_status():
    from nomorals.planning.congestion import control_congestion
    out = control_congestion("status", monitor=_mon())
    assert "🚦" in out


def test_chat_register():
    from nomorals.planning.congestion import control_congestion
    m = _mon()
    out = control_congestion("register groq-key api_key 10", monitor=m)
    assert "registered" in out and "groq-key" in out


def test_chat_advise():
    from nomorals.planning.congestion import control_congestion
    m = _mon()
    m.register("ep", capacity=2)
    m.acquire("ep", "a1")
    m.acquire("ep", "a2")
    out = control_congestion("advise ep agents=3", monitor=m)
    assert "🚦" in out and ("BACKOFF" in out or "SWITCH" in out or "STAGGER" in out)


def test_chat_predict():
    from nomorals.planning.congestion import control_congestion
    m = _mon()
    m.register("ep", capacity=4)
    out = control_congestion("predict ep", monitor=m)
    assert "jam probability" in out


def test_chat_alternate():
    from nomorals.planning.congestion import control_congestion
    m = _mon()
    m.register("key-a")
    m.register("key-b")
    out = control_congestion("alternate key-a key-b", monitor=m)
    assert "failover" in out


def test_chat_usage_on_garbage():
    from nomorals.planning.congestion import control_congestion
    out = control_congestion("blorp", monitor=_mon())
    assert "usage" in out


def test_chat_never_raises():
    from nomorals.planning.congestion import control_congestion
    for tail in ["", "register", "advise", "predict", "alternate a", None, "x" * 500]:
        out = control_congestion(tail, monitor=_mon())
        assert isinstance(out, str) and out


def test_in_memory_monitor():
    m = ContentionMonitor(in_memory=True)
    r = m.register("ep", capacity=2)
    assert r is not None
    m.acquire("ep", "a1")
    assert m.queue_depth("ep") == 1
