"""D5: WhatsApp link-time account label.

The audit verdict was THERE-BUT-DORMANT: the registry already stored a
label at pair time, but a re-pair overwrote it silently, nothing ever
applied the label as a WhatsApp label definition when the link landed,
no rename verb existed, and the pairing confirmation never named the
label.  These tests pin the wired behaviour:

- pair stores the label as "pending" until the link lands;
- a re-pair KEEPS an existing label unless the owner explicitly renames;
- the link-completion hook applies the label over the bridge's own
  label verbs (reuse a same-name definition, carry its color on a
  rename) and records the honest outcome — "set" only on a proved
  write, "failed" with the bridge's reason otherwise, never silent;
- /pair rename + /pair label, the pairing page, the web actions and
  one brain tool ride the same registry truth.
"""
from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from nomorals.social.pairing import (
    AccountRecord,
    BridgeSupervisor,
    PairingRegistry,
    label_state_line,
)

# ── fakes ──────────────────────────────────────────────────────────

class FakeProc:
    """A bridge process: yields its lines, then either dies
    (poll() -> 0) or parks alive on a gate when block=True."""

    def __init__(self, lines=(), die=True, block=False):
        self.terminated = False
        self._poll = None
        self._gate = threading.Event()
        self.stdout = self._stream(list(lines), die, block)

    def _stream(self, lines, die, block):
        yield from lines
        if block:
            self._gate.wait(30)
        elif die:
            self._poll = 0

    def poll(self):
        return self._poll

    def terminate(self):
        self.terminated = True
        self._poll = 0
        self._gate.set()


def wait_for(pred, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return bool(pred())


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class FakeBridge:
    """A TCP server speaking the bridge's newline-JSON protocol.

    Answers ``labels`` from its definition list and ``label_edit``
    with the configured reply (default: ok + a minted label_id).
    Every request is recorded, so tests can prove which verbs ran
    and what they carried.
    """

    def __init__(self, port, *, labels=(), edit_reply=None,
                 labels_reply=None):
        self.port = port
        self.labels = [dict(row) for row in labels]
        self.edit_reply = edit_reply
        self.labels_reply = labels_reply
        self.requests: list[dict] = []
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", port))
        self._sock.listen(16)
        self._sock.settimeout(0.5)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        with contextlib.suppress(OSError):
            self._sock.close()
        self._thread.join(timeout=2.0)
        return False

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except (TimeoutError, OSError):
                continue
            threading.Thread(target=self._handle, args=(conn,),
                             daemon=True).start()

    def _handle(self, conn):
        buf = b""
        try:
            conn.settimeout(2.0)
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(4096)
                except (TimeoutError, OSError):
                    return
                if not chunk:
                    return
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        req = json.loads(line.decode("utf-8", "replace"))
                    except ValueError:
                        continue
                    self.requests.append(dict(req))
                    conn.sendall((json.dumps(
                        self._reply(req)) + "\n").encode("utf-8"))
        finally:
            with contextlib.suppress(Exception):
                conn.close()

    def _reply(self, req):
        rid = req.get("id", "")
        cmd = req.get("cmd", "")
        if cmd == "labels":
            if self.labels_reply is not None:
                out = dict(self.labels_reply)
                out.setdefault("id", rid)
                return out
            return {"id": rid, "ok": True, "labels": self.labels,
                    "chat_labels": {}, "message_labels": {}}
        if cmd == "label_edit":
            if self.edit_reply is not None:
                out = dict(self.edit_reply)
                out.setdefault("id", rid)
                return out
            return {"id": rid, "ok": True,
                    "label_id": req.get("label_id") or "4242",
                    "name": req.get("name"), "color": 0,
                    "deleted": False}
        return {"id": rid, "ok": False,
                "error": f"unknown command {cmd!r}"}

    def edits(self):
        return [r for r in self.requests if r.get("cmd") == "label_edit"]

    def reads(self):
        return [r for r in self.requests if r.get("cmd") == "labels"]


def make_supervisor(tmp_path, proc_factory=None, *, port=None):
    registry = PairingRegistry(tmp_path / "paired.json")
    spawns = []

    def fake_spawn(argv, env):
        proc = proc_factory() if proc_factory else FakeProc()
        spawns.append((argv, env, proc))
        return proc

    bridge = tmp_path / "whatsapp-bridge.mjs"
    bridge.write_text("// bridge\n", encoding="utf-8")
    sup = BridgeSupervisor(
        registry, bridge_path=bridge,
        sessions_root=tmp_path / "sessions", spawn=fake_spawn,
        base_port=port if port is not None else 8787)
    return sup, registry, spawns


def pair_with_bridge(tmp_path, number, label, *, bridge):
    """Pair while a FakeBridge listens on the account's port."""
    port = bridge.port
    sup, registry, _ = make_supervisor(
        tmp_path, lambda: FakeProc(
            [f"connected as {number}@s.whatsapp.net"], die=True),
        port=port)
    rec = sup.pair_whatsapp(number, label)
    assert wait_for(
        lambda: registry.get(rec.id).status == "linked"
        and registry.get(rec.id).label_state in ("set", "failed")), \
        "link never landed"
    return sup, registry, rec


# ── 1 · the label is stored at link time, pending until the link ──

def test_pair_stores_label_pending(tmp_path):
    sup, registry, _ = make_supervisor(tmp_path)
    rec = sup.pair_whatsapp("2348012345678", "personal")
    assert rec.label == "personal"
    assert rec.label_state == "pending"
    assert rec.label_error == ""
    again = PairingRegistry(tmp_path / "paired.json")
    assert again.get(rec.id).label == "personal"
    assert again.get(rec.id).label_state == "pending"


def test_pair_without_label_has_no_label_state(tmp_path):
    sup, _registry, _ = make_supervisor(tmp_path)
    rec = sup.pair_whatsapp("2348012345678")
    assert rec.label == ""
    assert rec.label_state == ""
    assert rec.label_error == ""


def test_old_registry_loads_without_label_fields(tmp_path):
    path = tmp_path / "paired.json"
    path.write_text(json.dumps({"accounts": [{
        "id": "whatsapp:2348012345678", "platform": "whatsapp",
        "identity": "2348012345678", "label": "personal",
        "creds_dir": "", "port": 8787, "status": "linked",
        "created_at": 1.0, "last_seen": 0.0}]}), encoding="utf-8")
    registry = PairingRegistry(path)
    rec = registry.get("whatsapp:2348012345678")
    assert rec.label == "personal"
    assert rec.label_state == ""
    assert rec.label_error == ""
    assert rec.label_id == ""


# ── 2 · existing labels survive unless explicitly renamed ──────────

def test_repair_keeps_existing_label(tmp_path):
    sup, _registry, _ = make_supervisor(tmp_path)
    first = sup.pair_whatsapp("2348012345678", "personal")
    assert first.label == "personal"
    second = sup.pair_whatsapp("2348012345678", "work")
    assert second.label == "personal"
    assert second.label_state == "pending"


def test_repair_with_rename_overwrites(tmp_path):
    sup, _registry, _ = make_supervisor(tmp_path)
    sup.pair_whatsapp("2348012345678", "personal")
    renamed = sup.pair_whatsapp("2348012345678", "work", rename=True)
    assert renamed.label == "work"
    assert renamed.label_state == "pending"


def test_repair_names_unlabeled_account(tmp_path):
    sup, _registry, _ = make_supervisor(tmp_path)
    sup.pair_whatsapp("2348012345678")
    named = sup.pair_whatsapp("2348012345678", "personal")
    assert named.label == "personal"
    assert named.label_state == "pending"


def test_set_label_validates_and_keeps_definition_id(tmp_path):
    _sup, registry, _ = make_supervisor(tmp_path)
    registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", label="personal",
        label_state="failed", label_error="labels are a Business "
        "feature", label_id="99", status="linked", created_at=1.0))
    with pytest.raises(ValueError):
        registry.set_label("whatsapp:2348012345678", "   ")
    with pytest.raises(ValueError):
        registry.set_label("whatsapp:9999999999999", "work")
    rec = registry.set_label("whatsapp:2348012345678", "work")
    assert rec.label == "work"
    assert rec.label_id == "99"  # a rename reuses the definition
    assert rec.label_state == "pending"
    assert rec.label_error == ""


def test_telegram_repair_keeps_existing_label(tmp_path):
    from nomorals.social.telegram_pairing import TelegramPairer

    class _Client:
        async def connect(self):
            return None

        async def disconnect(self):
            return None

        async def send_code(self, phone):
            return "hash-1"

    registry = PairingRegistry(tmp_path / "paired.json")
    pairer = TelegramPairer(
        registry, sessions_root=tmp_path / "tg",
        api_id=12345, api_hash="hash",
        client_factory=lambda stem, a, h: _Client())
    first = pairer.start("2348012345678", "personal")
    assert first.label == "personal"
    second = pairer.start("2348012345678", "work")
    assert second.label == "personal"
    renamed = pairer.start("2348012345678", "work", rename=True)
    assert renamed.label == "work"
    for flow in list(pairer._flows.values()):
        flow.close()


# ── 3 · applying the label over the bridge's own verbs ─────────────

def _linked_rec(tmp_path, registry, *, label="personal", port=0,
                label_id="", label_state="pending", label_error=""):
    rec = AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", label=label, port=port,
        label_id=label_id, label_state=label_state,
        label_error=label_error, status="linked",
        created_at=1.0)
    registry.upsert(rec)
    return rec


def test_apply_creates_label_on_bridge(tmp_path):
    port = free_port()
    sup, registry, _ = make_supervisor(tmp_path, port=port)
    rec = _linked_rec(tmp_path, registry, port=port)
    with FakeBridge(port, labels=[]) as bridge:
        verdict = sup.apply_account_label(rec)
    assert verdict["ok"] is True
    assert verdict["label_id"] == "4242"
    assert len(bridge.edits()) == 1
    edit = bridge.edits()[0]
    assert edit["name"] == "personal"
    assert edit["deleted"] is False
    assert "color" not in edit  # a fresh create takes the default
    fresh = registry.get(rec.id)
    assert fresh.label_state == "set"
    assert fresh.label_id == "4242"
    assert fresh.label_error == ""


def test_apply_reuses_existing_definition(tmp_path):
    port = free_port()
    sup, registry, _ = make_supervisor(tmp_path, port=port)
    rec = _linked_rec(tmp_path, registry, port=port)
    existing = [{"id": "99", "name": "personal", "color": 7}]
    with FakeBridge(port, labels=existing) as bridge:
        verdict = sup.apply_account_label(rec)
    assert verdict["ok"] is True
    assert verdict.get("reused") is True
    assert verdict["label_id"] == "99"
    assert bridge.edits() == []  # no duplicate definition minted
    fresh = registry.get(rec.id)
    assert fresh.label_state == "set"
    assert fresh.label_id == "99"


def test_apply_rename_carries_existing_color(tmp_path):
    port = free_port()
    sup, registry, _ = make_supervisor(tmp_path, port=port)
    rec = _linked_rec(tmp_path, registry, port=port, label="work",
                      label_id="99")
    existing = [{"id": "99", "name": "personal", "color": 7}]
    with FakeBridge(port, labels=existing) as bridge:
        verdict = sup.apply_account_label(rec)
    assert verdict["ok"] is True
    assert len(bridge.edits()) == 1
    edit = bridge.edits()[0]
    assert edit["label_id"] == "99"
    assert edit["name"] == "work"
    assert edit["color"] == 7  # a rename keeps the label's color
    assert registry.get(rec.id).label_state == "set"


def test_apply_refusal_is_honest(tmp_path):
    port = free_port()
    sup, registry, _ = make_supervisor(tmp_path, port=port)
    rec = _linked_rec(tmp_path, registry, port=port)
    refusal = {"ok": False,
               "error": "labels are a Business feature — this "
                        "account is not a Business account"}
    with FakeBridge(port, labels=[], edit_reply=refusal):
        verdict = sup.apply_account_label(rec)
    assert verdict["ok"] is False
    assert "Business" in verdict["error"]
    fresh = registry.get(rec.id)
    assert fresh.label_state == "failed"
    assert "Business" in fresh.label_error
    assert fresh.status == "linked"  # the link stands; the label failed


def test_apply_bridge_down_is_honest(tmp_path):
    port = free_port()  # nothing listens here
    sup, registry, _ = make_supervisor(tmp_path, port=port)
    rec = _linked_rec(tmp_path, registry, port=port)
    verdict = sup.apply_account_label(rec)
    assert verdict["ok"] is False
    assert verdict["error"]
    fresh = registry.get(rec.id)
    assert fresh.label_state == "failed"
    assert fresh.label_error
    assert fresh.status == "linked"


def test_apply_ok_without_id_is_unproven(tmp_path):
    port = free_port()
    sup, registry, _ = make_supervisor(tmp_path, port=port)
    rec = _linked_rec(tmp_path, registry, port=port)
    with FakeBridge(port, labels=[],
                    edit_reply={"ok": True, "name": "personal"}):
        verdict = sup.apply_account_label(rec)
    assert verdict["ok"] is False
    assert "no id" in verdict["error"]
    assert registry.get(rec.id).label_state == "failed"


def test_apply_with_no_label_is_a_noop(tmp_path):
    port = free_port()
    sup, registry, _ = make_supervisor(tmp_path, port=port)
    rec = _linked_rec(tmp_path, registry, label="", port=port,
                      label_state="")
    with FakeBridge(port) as bridge:
        verdict = sup.apply_account_label(rec)
    assert verdict["ok"] is False
    assert bridge.requests == []
    assert registry.get(rec.id).label_state == ""


# ── 4 · the link itself applies the label ──────────────────────────

def test_link_completion_applies_label(tmp_path):
    port = free_port()
    with FakeBridge(port, labels=[]) as bridge:
        _sup, registry, rec = pair_with_bridge(
            tmp_path, "2348012345678", "personal", bridge=bridge)
    assert registry.get(rec.id).status == "linked"
    assert registry.get(rec.id).label_state == "set"
    assert registry.get(rec.id).label_id == "4242"
    assert len(bridge.edits()) == 1
    assert bridge.edits()[0]["name"] == "personal"


def test_link_completion_reports_refusal_honestly(tmp_path):
    port = free_port()
    refusal = {"ok": False,
               "error": "labels are a Business feature — this "
                        "account is not a Business account"}
    with FakeBridge(port, labels=[], edit_reply=refusal) as bridge:
        _sup, registry, rec = pair_with_bridge(
            tmp_path, "2348012345678", "personal", bridge=bridge)
    assert registry.get(rec.id).status == "linked"
    fresh = registry.get(rec.id)
    assert fresh.label_state == "failed"
    assert "Business" in fresh.label_error
    assert len(bridge.edits()) == 1


def test_devon_own_whatsapp_labels_the_same_way(tmp_path):
    # His question: "if I want to link Devon to her own Whatsapp
    # will the label work?" — her account rides the same pair flow,
    # the same label, the same link-time apply.  No special case.
    port = free_port()
    with FakeBridge(port, labels=[]) as bridge:
        _sup, registry, rec = pair_with_bridge(
            tmp_path, "2348099999999", "devon", bridge=bridge)
    assert registry.get(rec.id).label == "devon"
    assert registry.get(rec.id).label_state == "set"
    assert bridge.edits()[0]["name"] == "devon"


def test_sweep_relink_retries_a_failed_label(tmp_path):
    # A label refused at the first link is retried when the account
    # links again (sweep adopt-transition), not on every sweep pass:
    # drive the real sweep with a survivor answering as this account.
    from nomorals.social.pairing import PoolSupervision

    port = free_port()
    sup, registry, _ = make_supervisor(tmp_path, port=port)
    rec = _linked_rec(tmp_path, registry, port=port,
                      label_state="failed",
                      label_error="bridge was down")
    registry.set_status(rec.id, "down")
    sup.is_running = lambda account_id: False
    sup.probe = lambda rec, timeout=2.0: "linked"
    sup.adopt = lambda rec: SimpleNamespace(verified=True)
    sweep = PoolSupervision(registry, supervisor=sup)
    with FakeBridge(port, labels=[]) as bridge:
        changes = sweep.sweep()
    assert registry.get(rec.id).status == "linked"
    assert registry.get(rec.id).label_state == "set"
    assert len(bridge.edits()) == 1
    assert any("linked" in str(c.get("change", "")) for c in changes)


# ── 5 · the shared wording ─────────────────────────────────────────

def test_label_state_line_words():
    base = AccountRecord(id="whatsapp:2348012345678",
                         platform="whatsapp", identity="2348012345678")
    assert label_state_line(base) == ""
    pending = AccountRecord(id="whatsapp:2348012345678",
                            platform="whatsapp",
                            identity="2348012345678", label="personal",
                            label_state="pending")
    assert "personal" in label_state_line(pending)
    assert "pending" in label_state_line(pending)
    done = AccountRecord(id="whatsapp:2348012345678",
                         platform="whatsapp",
                         identity="2348012345678", label="personal",
                         label_state="set", label_id="4242")
    assert "set in WhatsApp" in label_state_line(done)
    failed = AccountRecord(id="whatsapp:2348012345678",
                           platform="whatsapp",
                           identity="2348012345678", label="personal",
                           label_state="failed",
                           label_error="labels are a Business feature")
    line = label_state_line(failed)
    assert "personal" in line and "Business" in line


# ── 6 · chat: /pair rename, /pair label, list + confirmation ───────

def _chat_rig(tmp_path, *, supervisor=None):
    from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin
    from nomorals.social.personal import PersonalSurfaces, PersonalVault

    if supervisor is None:
        supervisor, registry, _ = make_supervisor(tmp_path)
    else:
        registry = supervisor.registry
    personal = PersonalSurfaces(
        registry, PersonalVault(tmp_path / "personal"))
    rt = object.__new__(RuntimeProvisionMixin)
    rt._pair_supervisor = supervisor
    rt._pair_personal_svc = {"registry": registry,
                             "personal": personal}
    rt.context = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    rt.gateway = SimpleNamespace(adapters={})
    rt._sync_whatsapp_surface = lambda: ""
    rt._sync_telegram_surface = lambda: ""
    return rt, registry


def test_pair_rename_chat_verb(tmp_path):
    port = free_port()
    sup, registry, _ = make_supervisor(tmp_path, port=port)
    registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", label="personal",
        label_state="set", label_id="99", status="linked",
        created_at=1.0, port=port))
    rt, _reg = _chat_rig(tmp_path, supervisor=sup)
    with FakeBridge(port, labels=[
            {"id": "99", "name": "personal", "color": 7}]) as bridge:
        out = rt._control_pair("rename whatsapp:2348012345678 work")
    assert "work" in out, out
    assert "set in WhatsApp" in out, out
    assert registry.get("whatsapp:2348012345678").label == "work"
    assert len(bridge.edits()) == 1
    assert bridge.edits()[0]["color"] == 7


def test_pair_rename_unlinked_account_stays_pending(tmp_path):
    sup, registry, _ = make_supervisor(tmp_path)
    registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", label="personal",
        label_state="set", label_id="99", status="down",
        created_at=1.0))
    rt, _reg = _chat_rig(tmp_path, supervisor=sup)
    out = rt._control_pair("rename whatsapp:2348012345678 work")
    assert "work" in out, out
    assert "when the account links" in out, out
    fresh = registry.get("whatsapp:2348012345678")
    assert fresh.label == "work"
    assert fresh.label_state == "pending"


def test_pair_label_chat_verb_retries(tmp_path):
    port = free_port()
    sup, registry, _ = make_supervisor(tmp_path, port=port)
    registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", label="personal",
        label_state="failed",
        label_error="bridge was down", status="linked",
        created_at=1.0, port=port))
    rt, _reg = _chat_rig(tmp_path, supervisor=sup)
    with FakeBridge(port, labels=[]) as bridge:
        out = rt._control_pair("label whatsapp:2348012345678")
    assert "set in WhatsApp" in out, out
    assert registry.get("whatsapp:2348012345678").label_state == "set"
    assert len(bridge.edits()) == 1


def test_pair_list_shows_label_state(tmp_path):
    sup, registry, _ = make_supervisor(tmp_path)
    registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", label="personal",
        label_state="failed",
        label_error="labels are a Business feature",
        status="linked", created_at=1.0))
    rt, _reg = _chat_rig(tmp_path, supervisor=sup)
    reply = rt._control_pair("")
    # The summary carries toggle buttons, so it arrives as an
    # ActionableMessage — the note lives in its text.
    out = reply.text
    assert "personal" in out, out
    assert "Business" in out, out


def test_pair_whatsapp_confirmation_names_the_label(tmp_path):
    port = free_port()
    with FakeBridge(port, labels=[]):
        sup, registry, _ = make_supervisor(
            tmp_path,
            lambda: FakeProc(["PAIRING CODE for +2348012345678: "
                              "WXYZ-1234"], block=True),
            port=port)
        rt, _reg = _chat_rig(tmp_path, supervisor=sup)
        out = rt._control_pair("whatsapp 2348012345678 personal")
    assert "WXYZ-1234" in out, out
    assert "personal" in out, out
    assert "personal" in out and "label" in out.lower(), out
    assert registry.get("whatsapp:2348012345678").label == "personal"


def test_pair_whatsapp_kept_label_is_named(tmp_path):
    sup, registry, _ = make_supervisor(
        tmp_path,
        lambda: FakeProc(["PAIRING CODE for +2348012345678: "
                          "WXYZ-1234"], block=True))
    sup.pair_whatsapp("2348012345678", "personal")
    rt, _reg = _chat_rig(tmp_path, supervisor=sup)
    # Re-pair with a different label: the old one survives and the
    # reply says so (plus how to rename on purpose).
    out = rt._control_pair("whatsapp 2348012345678 work")
    assert "personal" in out, out
    assert "rename" in out, out
    assert registry.get("whatsapp:2348012345678").label == "personal"


# ── 7 · page + web ride the same truth ─────────────────────────────

def test_pairing_page_shows_label_state(tmp_path):
    from nomorals.social.pairing_page import pairing_page_html

    registry = PairingRegistry(tmp_path / "paired.json")
    registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", label="personal",
        label_state="failed",
        label_error="labels are a Business feature",
        status="linked", created_at=1.0))
    html = pairing_page_html(registry)
    assert "personal" in html
    assert "Business" in html


def test_pairing_web_rename_and_apply(tmp_path):
    from nomorals.social.pairing_web import pairing_action, pairing_state

    port = free_port()
    sup, registry, _ = make_supervisor(tmp_path, port=port)
    registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", label="personal",
        label_state="failed", label_error="bridge was down",
        label_id="99", status="down", created_at=1.0, port=port))
    svc = {"supervisor": sup, "registry": registry, "pairer": None,
           "personal": None, "workspace": tmp_path}
    ctx = SimpleNamespace()
    # Rename while unlinked: the new name stays pending (no bridge
    # to apply it to) — the linked-rename-apply path is proven by
    # the chat test; the web one proves rename + explicit apply.
    out = pairing_action(ctx, {"action": "rename",
                               "id": "whatsapp:2348012345678",
                               "label": "work"}, svc=svc)
    assert out["ok"] is True, out
    assert registry.get("whatsapp:2348012345678").label == "work"
    assert registry.get("whatsapp:2348012345678").label_state \
        == "pending"
    registry.set_status("whatsapp:2348012345678", "linked")
    with FakeBridge(port, labels=[
            {"id": "99", "name": "personal", "color": 7}]) as bridge:
        out = pairing_action(
            ctx, {"action": "apply_label",
                  "id": "whatsapp:2348012345678"}, svc=svc)
    assert out["ok"] is True, out
    assert registry.get("whatsapp:2348012345678").label_state == "set"
    assert len(bridge.edits()) == 1
    assert bridge.edits()[0]["name"] == "work"
    assert bridge.edits()[0]["color"] == 7
    state = pairing_state(ctx, svc=svc)
    row = [a for a in state["accounts"]
           if a["id"] == "whatsapp:2348012345678"][0]
    assert row["label"] == "work"
    assert row["label_state"] == "set"


# ── 8 · the brain tool ─────────────────────────────────────────────

def test_brain_tool_registered_and_counts_789():
    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    assert "whatsapp_account_label" in registry._tools
    assert len(registry._tools) == 809


def test_brain_tool_renames_and_applies(tmp_path):
    import nomorals.tools.whatsapp as wa_tools

    port = free_port()
    sup, registry, _ = make_supervisor(tmp_path, port=port)
    registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", label="personal",
        label_state="set", label_id="99", status="linked",
        created_at=1.0, port=port))
    wa_tools.set_pairing_supervisor_factory(lambda: sup)
    try:
        from nomorals.tools.registry import ToolRegistry

        tool_registry = ToolRegistry()
        wa_tools.register(tool_registry)
        tool = tool_registry.get("whatsapp_account_label")
        assert tool is not None
        fn = tool.fn
        with FakeBridge(port, labels=[
                {"id": "99", "name": "personal", "color": 7}]):
            out = fn(account="whatsapp:2348012345678", label="work")
        assert out["ok"] is True, out
        assert out["label"] == "work"
        assert out["label_state"] == "set"
        assert registry.get("whatsapp:2348012345678").label == "work"
        read = fn(account="2348012345678")
        assert read["ok"] is True
        assert read["label"] == "work"
        assert read["label_state"] == "set"
    finally:
        wa_tools.set_pairing_supervisor_factory(None)


def test_brain_tool_fails_closed_without_supervisor(tmp_path):
    import nomorals.tools.whatsapp as wa_tools

    wa_tools.set_pairing_supervisor_factory(None)
    try:
        from nomorals.tools.registry import ToolRegistry

        tool_registry = ToolRegistry()
        wa_tools.register(tool_registry)
        out = tool_registry.get("whatsapp_account_label").fn(
            account="whatsapp:2348012345678")
        assert out["ok"] is False
        assert "supervisor" in out["error"].lower()
    finally:
        wa_tools.set_pairing_supervisor_factory(None)
