"""Own-domain temp-mail rung (10MinuteMail #50).

End-to-end: a real SMTP listener on an ephemeral port, a real
smtplib send, wait_code extraction, TTL sweep deleting rows (not
just hiding them), +10-minute extend, and the unconfigured rung
reporting unavailable by name while the cascade falls through.
"""
from __future__ import annotations

import smtplib
import sqlite3
import time
from types import SimpleNamespace

import pytest

from nomorals.accounts import temp_mail as tm
from nomorals.agents.partner.runtime_reach import (
    RuntimeReachMixin,
    _tempmail_dies_in,
)

DOMAIN = "example.test"


@pytest.fixture()
def env_domain(monkeypatch, tmp_path):
    """Configured own-domain rung backed by a scratch SQLite store."""
    db = tmp_path / "own_domain_mail.db"
    monkeypatch.setenv("NM_TEMPMAIL_DOMAIN", DOMAIN)
    monkeypatch.setenv("NM_TEMPMAIL_DB", str(db))
    monkeypatch.delenv("NM_HOME", raising=False)
    return db


@pytest.fixture()
def env_unconfigured(monkeypatch, tmp_path):
    """No domain at all — the rung must report unavailable."""
    monkeypatch.delenv("NM_TEMPMAIL_DOMAIN", raising=False)
    monkeypatch.setenv("NM_TEMPMAIL_DB", str(tmp_path / "mail.db"))
    return None


@pytest.fixture()
def store(env_domain):
    return tm.OwnDomainStore(path=str(env_domain))


@pytest.fixture()
def listener(store):
    """A real catch-all listener on 127.0.0.1, ephemeral port."""
    server = tm.start_own_domain_listener(
        domain=DOMAIN, store=store, host="127.0.0.1", port=0)
    assert server is not None
    port = server.server_address[1]
    yield server, port
    tm.stop_own_domain_listener(store=store)


def _send(port: int, to_addr: str, body: str,
          sender: str = "signup@service.test",
          subject: str = "Your verification code") -> None:
    msg = (f"From: {sender}\r\nTo: {to_addr}\r\n"
           f"Subject: {subject}\r\n\r\n{body}\r\n")
    with smtplib.SMTP("127.0.0.1", port, timeout=10) as client:
        client.sendmail(sender, [to_addr], msg)


def _row_count(db_path, table: str) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def _expire(db_path, address: str) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("UPDATE mailboxes SET expires_at = ? WHERE address = ?",
                     (time.time() - 60, address))
        conn.commit()
    finally:
        conn.close()


# ── config / probe ────────────────────────────────────────────────────

def test_expires_at_defaults_to_zero():
    assert tm.TempAddress().expires_at == 0.0


def test_unconfigured_probe_false_and_grab_names_setting(env_unconfigured):
    prov = tm.OwnDomainProvider()
    assert prov.configured() is False
    assert prov.probe() is False
    with pytest.raises(RuntimeError, match="NM_TEMPMAIL_DOMAIN"):
        prov.grab_address()
    status = tm.own_domain_status()
    assert status["configured"] is False
    assert "NM_TEMPMAIL_DOMAIN" in status["reason"]
    assert tm.start_own_domain_listener() is None


def test_grab_mints_address_with_10min_expiry(env_domain, store):
    before = time.time()
    addr = tm.get_provider("owndomain").grab_address()
    assert addr.address.endswith("@" + DOMAIN)
    assert addr.provider == "owndomain"
    assert before + 590 < addr.expires_at < before + 610
    assert store.mailbox(addr.address) is not None


def test_invalid_domain_is_not_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("NM_TEMPMAIL_DOMAIN", "not-a-domain")
    monkeypatch.setenv("NM_TEMPMAIL_DB", str(tmp_path / "mail.db"))
    prov = tm.OwnDomainProvider()
    assert prov.configured() is False
    assert prov.probe() is False


# ── end-to-end receive ────────────────────────────────────────────────

def test_e2e_send_real_message_wait_code_extracts(env_domain, store,
                                                 listener):
    _, port = listener
    addr = tm.get_provider("owndomain").grab_address()
    _send(port, addr.address, "Welcome! Your code is 482913. It expires soon.")
    code = tm.wait_code(addr, timeout_s=15, poll_s=1)
    assert code == "482913"
    msgs = tm.get_provider("owndomain").get_messages(addr)
    assert len(msgs) == 1
    assert msgs[0].sender == "signup@service.test"
    assert msgs[0].subject == "Your verification code"
    assert msgs[0].code == "482913"


def test_catchall_accepts_unminted_local_part(env_domain, store, listener):
    _, port = listener
    _send(port, "whoever@" + DOMAIN, "hello there 123456")
    rows = store.list_messages("whoever@" + DOMAIN)
    assert len(rows) == 1
    assert store.mailbox("whoever@" + DOMAIN) is not None
    assert rows[0]["body_text"].strip() == "hello there 123456"


def test_dot_stuffing_undone(env_domain, store, listener):
    _, port = listener
    addr = tm.get_provider("owndomain").grab_address()
    _send(port, addr.address, "line1\n.line2\nline3")
    rows = store.list_messages(addr.address)
    assert ".line2" in rows[0]["body_text"]
    assert "..line2" not in rows[0]["body_text"]


def test_rcpt_other_domain_rejected_and_nothing_stored(env_domain, store,
                                                       listener):
    _, port = listener
    with smtplib.SMTP("127.0.0.1", port, timeout=10) as client, \
            pytest.raises(smtplib.SMTPRecipientsRefused):
        client.sendmail("a@b.test", ["x@else.com"],
                        "Subject: hi\r\n\r\nhi\r\n")
    assert store.counts() == {"mailboxes": 0, "messages": 0}


def test_oversize_data_refused(env_domain, store):
    server = tm.start_own_domain_listener(
        domain=DOMAIN, store=store, host="127.0.0.1", port=0, max_bytes=1024)
    assert server is not None
    try:
        port = server.server_address[1]
        addr = tm.get_provider("owndomain").grab_address()
        with smtplib.SMTP("127.0.0.1", port, timeout=10) as client, \
                pytest.raises(smtplib.SMTPDataError):
            client.sendmail("a@b.test", [addr.address],
                            "Subject: big\r\n\r\n" + ("x" * 4096) + "\r\n")
        assert store.list_messages(addr.address) == []
    finally:
        tm.stop_own_domain_listener(store=store)


def test_malformed_message_still_lands():
    sender, subject, date, text, html = tm._message_parts(
        b"not an email at all 123456")
    assert "123456" in text
    assert sender == "" and subject == ""


# ── TTL / extend / sweep ──────────────────────────────────────────────

def test_ttl_sweep_deletes_rows_not_just_hides(env_domain, store, listener):
    _, port = listener
    addr = tm.get_provider("owndomain").grab_address()
    _send(port, addr.address, "code 777777")
    assert len(store.list_messages(addr.address)) == 1
    _expire(env_domain, addr.address)
    # Expired reads as empty even before the reaper runs.
    assert tm.get_provider("owndomain").get_messages(addr) == []
    out = tm.run_own_domain_mail_tick_once()
    assert out["ok"] is True
    assert out["swept_mailboxes"] == 1
    assert out["swept_messages"] == 1
    assert _row_count(env_domain, "mailboxes") == 0
    assert _row_count(env_domain, "messages") == 0
    assert store.mailbox(addr.address) is None


def test_extend_moves_expiry(env_domain, store):
    prov = tm.get_provider("owndomain")
    addr = prov.grab_address()
    before = addr.expires_at
    new_expires = prov.extend_address(addr)
    assert abs(new_expires - (before + 600)) < 5
    assert store.mailbox(addr.address)["expires_at"] == new_expires
    _expire(env_domain, addr.address)
    assert prov.extend_address(addr) == 0.0


def test_tick_sweeps_only_expired(env_domain, store):
    live = tm.get_provider("owndomain").grab_address()
    dead = tm.get_provider("owndomain").grab_address()
    _expire(env_domain, dead.address)
    out = tm.run_own_domain_mail_tick_once()
    assert out["swept_mailboxes"] == 1
    assert out["live_mailboxes"] == 1
    assert store.mailbox(live.address) is not None
    assert store.mailbox(dead.address) is None


def test_delete_message_removes_row(env_domain, store):
    prov = tm.get_provider("owndomain")
    addr = prov.grab_address()
    store.store_message(addr.address, sender="a@b", subject="s",
                        body_text="code 111111")
    msgs = prov.get_messages(addr)
    assert len(msgs) == 1
    assert prov.delete_message(addr, msgs[0]) is True
    assert prov.get_messages(addr) == []
    assert prov.delete_message(addr, msgs[0]) is False


def test_status_configured(env_domain, store, listener):
    addr = tm.get_provider("owndomain").grab_address()
    status = tm.own_domain_status()
    assert status["configured"] is True
    assert status["domain"] == DOMAIN
    assert status["listener"] is True
    assert status["live_mailboxes"] == 1
    assert status["next_expiry"] == addr.expires_at
    assert status["ttl_s"] == 600.0


# ── listener lifecycle ────────────────────────────────────────────────

def test_listener_start_stop_idempotent(env_domain, store):
    first = tm.start_own_domain_listener(
        domain=DOMAIN, store=store, host="127.0.0.1", port=0)
    assert first is not None
    second = tm.start_own_domain_listener(
        domain=DOMAIN, store=store, host="127.0.0.1", port=0)
    assert second is first
    assert tm.own_domain_listener_alive(store=store) is True
    assert tm.stop_own_domain_listener(store=store) is True
    assert tm.own_domain_listener_alive(store=store) is False
    assert tm.stop_own_domain_listener(store=store) is False


# ── cascade ───────────────────────────────────────────────────────────

class _FakeProvider(tm.TempMailProvider):
    name = "fake"
    calls = 0

    def probe(self) -> bool:
        return True

    def grab_address(self) -> tm.TempAddress:
        type(self).calls += 1
        return tm.TempAddress(address="u@fake.test", login="u",
                              domain="fake.test", provider=self.name)


def test_cascade_falls_through_when_unconfigured(env_unconfigured,
                                                 monkeypatch):
    monkeypatch.setitem(tm.PROVIDERS, "fake", _FakeProvider)
    _FakeProvider.calls = 0
    addr = tm.grab_address_cascade(providers=["owndomain", "fake"])
    assert addr.provider == "fake"
    assert addr.address == "u@fake.test"
    assert _FakeProvider.calls == 1


def test_cascade_prefers_own_domain_when_configured(env_domain,
                                                    monkeypatch):
    monkeypatch.setitem(tm.PROVIDERS, "fake", _FakeProvider)
    _FakeProvider.calls = 0
    addr = tm.grab_address_cascade(providers=["owndomain", "fake"])
    assert addr.provider == "owndomain"
    assert addr.address.endswith("@" + DOMAIN)
    assert _FakeProvider.calls == 0


def test_cascade_order_puts_own_domain_first():
    assert tm.CASCADE_PROVIDERS[0] == "owndomain"
    assert set(tm.CASCADE_PROVIDERS) == {
        "owndomain", "mailtm", "1secmail", "guerrillamail"}


def test_wait_code_resolves_owndomain_via_registry(env_domain, store):
    addr = tm.get_provider("owndomain").grab_address()
    store.store_message(addr.address, sender="noreply@x.test",
                        subject="code", body_text="your code is 246810")
    assert tm.wait_code(addr, timeout_s=5, poll_s=0.5) == "246810"


# ── scheduler job ─────────────────────────────────────────────────────

class _Sched:
    def __init__(self):
        self.jobs: list[dict] = []

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, kind, payload, **_kw):
        job = {"name": name, "spec": spec, "kind": kind,
               "payload": payload}
        self.jobs.append(job)
        return job

    def remove(self, job_id):
        self.jobs = [j for j in self.jobs if j.get("id") != job_id]


def test_job_registration_idempotent():
    sched = _Sched()
    assert tm.ensure_own_domain_mail_job(None) is None
    first = tm.ensure_own_domain_mail_job(sched)
    assert first["name"] == tm.OWN_DOMAIN_MAIL_JOB
    assert first["spec"] == tm.OWN_DOMAIN_MAIL_SPEC
    assert first["payload"] == {"tool": tm.OWN_DOMAIN_MAIL_TOOL,
                                "args": {}}
    assert tm.ensure_own_domain_mail_job(sched) is first


def test_tick_registers_under_tool_name():
    assert tm.OWN_DOMAIN_MAIL_TOOL == "__own_domain_mail_tick__"
    assert tm.OWN_DOMAIN_MAIL_SPEC == "*/5 * * * *"


# ── chat surface (mixin, stubbed KV) ──────────────────────────────────

class _Reach(RuntimeReachMixin):
    """Bare mixin with an in-memory temp-inbox record."""

    def __init__(self):
        self.context = SimpleNamespace(db=None)
        self._record: dict | None = None

    def _temp_save(self, kind, message, chat_key, record, ttl):
        self._record = dict(record)
        return True

    def _temp_load(self, kind, message, chat_key):
        return dict(self._record) if self._record else None

    def _temp_forget(self, kind, message, chat_key):
        self._record = None
        return True


def test_tempmail_new_status_extend_roundtrip(env_domain, store):
    reach = _Reach()
    new = reach._control_tempmail("new", chat_key="t")
    assert "via owndomain" in new
    assert "dies in" in new
    status = reach._control_tempmail("status", chat_key="t")
    assert "via owndomain" in status
    assert "dies in" in status
    assert "extend adds" in status
    extended = reach._control_tempmail("extend", chat_key="t")
    assert extended.startswith("extended —")
    assert "dies in" in extended
    assert reach._record["expires_at"] > time.time() + 600


def test_tempmail_status_names_public_rung_unchanged(env_unconfigured,
                                                     monkeypatch):
    reach = _Reach()
    reach._record = {"address": "u@1secmail.com", "login": "u",
                     "domain": "1secmail.com", "provider": "1secmail",
                     "token": "", "expires_at": 0.0}
    status = reach._control_tempmail("status", chat_key="t")
    assert "via 1secmail" in status
    assert "dies" not in status
    assert "only own-domain inboxes extend" in \
        reach._control_tempmail("extend", chat_key="t")


def test_tempmail_expired_reads_as_gone(env_domain, store):
    reach = _Reach()
    reach._control_tempmail("new", chat_key="t")
    address = reach._record["address"]
    _expire(env_domain, address)
    assert "already gone" in reach._control_tempmail("extend", chat_key="t")
    assert "is gone" in reach._control_tempmail("status", chat_key="t")
    assert "died" in reach._control_checkmail("", chat_key="t")


def test_tempmail_dies_in_shapes():
    now = time.time()
    # Second boundaries move underfoot — assert shapes, not exact
    # seconds (a 599s countdown reads 9m58s or 9m59s either way).
    assert _tempmail_dies_in(now + 599).startswith("in 9m")
    assert _tempmail_dies_in(now + 45).startswith("in ")
    assert _tempmail_dies_in(now + 45).endswith("s")
    assert _tempmail_dies_in(now + 3660).startswith("in 1h")
    assert _tempmail_dies_in(now - 90).endswith("ago")
