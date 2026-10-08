"""Tests for build-map #98 — self-hosted send layer. All offline (mock providers)."""

import tempfile

from nomorals.marketing.send import (
    HARD_BOUNCE_CLASSES,
    SendEngine,
    control_send,
    parse_dsn,
    render_template,
)


def _tmpdb():
    return tempfile.mktemp(suffix=".db")


def _eng(**kw):
    e = SendEngine(db_path=_tmpdb(), **kw)
    e.register_provider("mock", kind="mock", rate_per_minute=1000,
                        sender=lambda to, subj, body: True)
    return e


# ── templates ─────────────────────────────────────────────────────────────

def test_render_template_vars():
    out = render_template("Hi {{ name }}, your {{ order.id }} ships {{ when }}.",
                          {"name": "Ada", "order": {"id": "A1"}, "when": "Friday"})
    assert out == "Hi Ada, your A1 ships Friday."


def test_render_template_conditional_true_false():
    body = "Hello{% if vip %} VIP{% endif %}!"
    assert render_template(body, {"vip": True}) == "Hello VIP!"
    assert render_template(body, {"vip": False}) == "Hello!"
    assert render_template(body, {}) == "Hello!"


def test_render_template_else_branch():
    body = "{% if paid %}thanks{% else %}please pay{% endif %}"
    assert render_template(body, {"paid": 1}) == "thanks"
    assert render_template(body, {"paid": 0}) == "please pay"


def test_render_template_never_raises():
    assert isinstance(render_template(None, None), str)
    assert render_template("{% if x %}", None) == "{% if x %}"
    assert render_template("{{ missing }}", {}) == ""


def test_save_and_list_templates():
    e = _eng()
    assert e.save_template("launch", "Hi {{ name }}!", "Launch day")
    tpls = e.list_templates()
    assert any(t["name"] == "launch" for t in tpls)
    got = e.get_template("launch")
    assert got["body"] == "Hi {{ name }}!"


# ── queue / process ───────────────────────────────────────────────────────

def test_enqueue_and_process_happy_path():
    e = _eng()
    e.save_template("hi", "Hello {{ name }}", "Hey")
    sid = e.enqueue("a@example.com", "hi", {"name": "Ada"}, "mock")
    assert sid.startswith("snd_")
    assert e.queue_depth() == 1
    stats = e.process()
    assert stats["sent"] == 1
    assert e.queue_depth() == 0


def test_enqueue_rejects_bad_input():
    e = _eng()
    assert e.enqueue("", "t", {}, "mock") == ""
    assert e.enqueue("a@b.com", "", {}, "mock") == ""
    assert e.enqueue("a@b.com", "t", {}, "") == ""


def test_unknown_provider_fails_send():
    e = _eng()
    e.save_template("hi", "hello", "")
    sid = e.enqueue("a@example.com", "hi", {}, "nosuch")
    assert sid
    stats = e.process()
    assert stats["failed"] == 1


def test_missing_template_fails_send():
    e = _eng()
    sid = e.enqueue("a@example.com", "nosuch", {}, "mock")
    stats = e.process()
    assert stats["failed"] == 1


# ── retry / backoff ───────────────────────────────────────────────────────

def test_retry_with_backoff_then_dead():
    e = _eng()
    e.save_template("hi", "hello", "")
    e.register_provider("flaky", kind="mock", rate_per_minute=1000,
                        sender=lambda to, s, b: (_ for _ in ()).throw(RuntimeError("boom")))
    e.enqueue("a@example.com", "hi", {}, "flaky")
    # Force attempts past the backoff window by resetting next_attempt_at.
    import time
    for i in range(4):
        stats = e.process()
        assert stats["deferred"] == 1, f"attempt {i}: {stats}"
        e._db.execute("UPDATE send_queue SET next_attempt_at = 0")
        e._db.commit()
    stats = e.process()  # 5th attempt → dead
    assert stats["failed"] == 1
    assert e.queue_depth("dead") == 1


def test_false_return_retries():
    e = _eng()
    e.save_template("hi", "hello", "")
    e.register_provider("liar", kind="mock", rate_per_minute=1000,
                        sender=lambda to, s, b: False)
    e.enqueue("a@example.com", "hi", {}, "liar")
    stats = e.process()
    assert stats["deferred"] == 1
    assert e.queue_depth() == 1


# ── rate limiting ─────────────────────────────────────────────────────────

def test_sliding_window_rate_limit():
    e = _eng()
    e.save_template("hi", "hello", "")
    e.register_provider("slow", kind="mock", rate_per_minute=2,
                        sender=lambda to, s, b: True)
    for i in range(4):
        e.enqueue(f"u{i}@x.com", "hi", {}, "slow")
    stats = e.process()
    assert stats["sent"] == 2
    assert stats["deferred"] == 2
    assert e.queue_depth() == 2


# ── bounces / blocklist ───────────────────────────────────────────────────

def test_parse_dsn_hard():
    p = parse_dsn("Delivery failed: 5.1.1 user unknown <gone@example.com>")
    assert p["kind"] == "hard"
    assert "gone@example.com" in p["addresses"]
    assert p["code"] == "5.1.1"


def test_parse_dsn_soft():
    p = parse_dsn("4.2.2 mailbox full for <full@example.com>, will retry")
    assert p["kind"] == "soft"
    assert "full@example.com" in p["addresses"]


def test_parse_dsn_never_raises():
    assert parse_dsn(None)["kind"] == "unknown"
    assert parse_dsn("")["addresses"] == []


def test_hard_bounce_blocklists():
    e = _eng()
    e.save_template("hi", "hello", "")
    res = e.process_bounce("5.1.1 user unknown <gone@example.com>")
    assert res["kind"] == "hard"
    assert "gone@example.com" in res["blocklisted"]
    assert e.is_blocklisted("gone@example.com")
    # Enqueue to a blocklisted address is refused.
    assert e.enqueue("gone@example.com", "hi", {}, "mock") == ""


def test_blocked_send_does_not_fire():
    e = _eng()
    e.save_template("hi", "hello", "")
    fired = []
    e.register_provider("spy", kind="mock", rate_per_minute=1000,
                        sender=lambda to, s, b: fired.append(to) or True)
    sid = e.enqueue("b@example.com", "hi", {}, "spy")
    assert sid  # queued…
    e.blocklist("b@example.com", "test")  # …then blocklisted
    stats = e.process()
    assert stats["blocked"] == 1
    assert fired == []


def test_unblocklist():
    e = _eng()
    e.blocklist("x@example.com")
    assert e.unblocklist("x@example.com")
    assert not e.is_blocklisted("x@example.com")


# ── cost routing ──────────────────────────────────────────────────────────

def test_estimate_campaign():
    e = _eng()
    e.register_provider("ses-main", kind="ses", rate_per_minute=60)
    est = e.estimate_campaign(100, "ses-main")
    assert est["count"] == 100
    assert est["total_kobo"] == 100 * 15
    assert "100" in est["text"]


def test_cost_recorded_on_send():
    e = _eng()
    e.save_template("hi", "hello", "")
    e.register_provider("ses-main", kind="ses", rate_per_minute=1000,
                        sender=lambda to, s, b: True)
    e.enqueue("a@example.com", "hi", {}, "ses-main")
    e.process()
    st = e.status()
    assert st["spent_24h_kobo"] == 15


# ── status ────────────────────────────────────────────────────────────────

def test_status_shape():
    e = _eng()
    st = e.status()
    assert st["queued"] == 0
    assert st["blocklisted"] == 0
    assert isinstance(st["providers"], list)
    assert any(p["name"] == "mock" for p in st["providers"])


def test_profile_gate_batch_size():
    e = SendEngine(db_path=_tmpdb(), profile="termux")
    assert e.batch_size == 10
    e2 = SendEngine(db_path=_tmpdb(), profile="workstation")
    assert e2.batch_size == 1000


# ── chat ──────────────────────────────────────────────────────────────────

def test_control_send_help():
    out = control_send("")
    assert "/send campaign" in out


def test_control_send_template_and_campaign():
    out = control_send("template add launch | Hi {{ name }}!")
    assert "saved" in out
    out = control_send("provider add m1 mock 100")
    assert "registered" in out
    out = control_send("campaign launch to a@x.com, b@y.com via m1")
    assert "queued" in out and "2/2" in out


def test_control_send_queue_status_process():
    e = _eng()
    control_send("template add t2 | hello", engine=e)
    control_send("provider add m2 mock 100", engine=e)
    control_send("campaign t2 to z@z.com via m2", engine=e)
    assert "1 send" in control_send("queue", engine=e)
    out = control_send("process", engine=e)
    assert "sent 1" in out
    assert "queued 0" in control_send("status", engine=e)


def test_control_send_bounce():
    out = control_send("bounce bad@x.com")
    assert "blocklisted" in out


def test_control_send_never_raises():
    assert isinstance(control_send(None), str)
    assert isinstance(control_send("campaign"), str)
    assert isinstance(control_send("provider add"), str)
    assert isinstance(control_send("template add x"), str)
