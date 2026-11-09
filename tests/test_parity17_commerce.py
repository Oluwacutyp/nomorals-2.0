"""PARITY #17 — WHATSAPP COMMERCE LEFTOVERS (platform
parity census bundle #17, 2026-11-07).

The census graded four commerce rows ABSENT.  The
platform truth, verified from the actual libraries
before a line was built:

  * Broadcast lists — neither engine exposes them to a
    linked device: whatsmeow refuses non-status
    broadcast sends outright ("sending to non-status
    broadcast lists is not yet supported"), and Baileys
    has no broadcast-list API at all.
  * WhatsApp Pay — no payments surface in either
    library, and not offered in the owner's region.
  * Labels — REAL on both engines: whatsmeow app-state
    label patches (BuildLabelEdit / BuildLabelChat /
    BuildLabelMessage + SendAppState) and Baileys'
    addLabel / addChatLabel / addMessageLabel, with
    label state riding the app-state sync back.
  * Product catalog — REAL on Baileys only (getCatalog,
    productCreate / productUpdate / productDelete);
    whatsmeow implements no catalog API (its business
    surface reads orders and business profiles only), so
    the Go bridge refuses in words and that refusal is
    the answer.

This suite pins the build at every seam: both bridges'
verbs + the rebuilt binaries, the chatops catalog
registration, the formatters (the listing NUMBER is
the handle; internal ids never print), the adapter's
shapes and its read-modify-write product update, the
composed chatop chain over a recording fake bridge
(reads vs acts under the owner's standing stop,
journal rows), and the two honest closures — /broadcast
and /wpay answer the platform truth with zero commerce
traffic.
"""
from __future__ import annotations

import json
import socketserver
import threading
import time
from pathlib import Path

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

PORT = 19471
OWNER = "2348012345678"
PN_POSTER = "2347099999999@s.whatsapp.net"
GROUP = "120363001234567890@g.us"
ROOT = Path(__file__).resolve().parents[1]

GO_CATALOG_ERROR = (
    "the catalog is not reachable on the Go bridge — "
    "whatsmeow exposes no catalog API (its business "
    "surface reads orders and business profiles only; "
    "the w:biz:catalog namespace is not implemented). "
    "Catalog verbs ride the Baileys engine.")

LABEL_SEED = [
    {"id": "L1", "name": "VIP", "color": 5,
     "deleted": False, "predefined_id": None},
    {"id": "L2", "name": "Needs reply", "color": 3,
     "deleted": False, "predefined_id": None},
    {"id": "L3", "name": "Old label", "color": 1,
     "deleted": True, "predefined_id": None},
]
PRODUCT_SEED = [
    {"id": "P1", "name": "Ankara Gown", "price": 45000,
     "currency": "NGN", "description": "Hand-dyed, size M",
     "retailer_id": "", "url": "", "hidden": False,
     "availability": "in stock"},
    {"id": "P2", "name": "Leather Sandals",
     "price": 18500.5, "currency": "NGN", "description": "",
     "retailer_id": "SKU-2", "url": "", "hidden": False,
     "availability": "in stock"},
]
FOREIGN_PRODUCTS = [
    {"id": "F1", "name": "Foreign Widget", "price": 10,
     "currency": "USD", "description": "", "retailer_id": "",
     "url": "", "hidden": False, "availability": "in stock"},
]


# ── recording fake bridge (the bridges' wire protocol) ─────

class _BridgeHandler(socketserver.StreamRequestHandler):
    def handle(self):
        srv = self.server
        self.wfile.write(json.dumps({
            "type": "status", "state": "open", "user": OWNER,
            "user_name": "Owner"}).encode() + b"\n")
        self.wfile.flush()
        with srv.lock:
            srv.clients.append(self.wfile)
        try:
            for raw in self.rfile:
                line = raw.strip()
                if not line:
                    continue
                try:
                    cmd = json.loads(line)
                except ValueError:
                    continue
                with srv.lock:
                    srv.commands.append(cmd)
                cid = cmd.get("id")
                if not cid:
                    continue
                reply = srv.answer(cmd)
                reply["id"] = cid
                self.wfile.write(
                    json.dumps(reply).encode() + b"\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class FakeBridge(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, port=PORT):
        self.commands: list[dict] = []
        self.clients: list = []
        self.lock = threading.Lock()
        self.labels = [dict(r) for r in LABEL_SEED]
        self.chat_labels = {PN_POSTER: ["L2"]}
        self.message_labels = [
            {"chat": PN_POSTER, "message_id": "m9",
             "label_id": "L1"}]
        self.products = [dict(r) for r in PRODUCT_SEED]
        self.catalog_error = ""
        self.catalog_engine = "baileys"
        self._label_seq = 8
        self._product_seq = 8
        super().__init__(("127.0.0.1", port), _BridgeHandler)
        threading.Thread(target=self.serve_forever,
                         daemon=True).start()

    def answer(self, cmd):
        verb = cmd.get("cmd")
        if verb == "labels":
            return {"ok": True,
                    "labels": [dict(r) for r in self.labels],
                    "chat_labels": {
                        k: list(v) for k, v
                        in self.chat_labels.items()},
                    "message_labels": [
                        dict(r) for r in self.message_labels],
                    "coverage": "a test coverage line"}
        if verb == "label_edit":
            lid = str(cmd.get("label_id") or "")
            if cmd.get("deleted"):
                for row in self.labels:
                    if row["id"] == lid:
                        row["deleted"] = True
                return {"ok": True, "label_id": lid}
            for row in self.labels:
                if row["id"] == lid:
                    row["name"] = cmd.get("name") or ""
                    if "color" in cmd:
                        row["color"] = cmd["color"]
                    return {"ok": True, "label_id": lid}
            self._label_seq += 1
            lid = f"L{self._label_seq}"
            self.labels.append({
                "id": lid, "name": cmd.get("name") or "",
                "color": cmd.get("color"),
                "deleted": False, "predefined_id": None})
            return {"ok": True, "label_id": lid}
        if verb == "label_chat":
            lst = self.chat_labels.setdefault(
                cmd.get("chat") or "", [])
            if cmd.get("labeled"):
                if cmd.get("label_id") not in lst:
                    lst.append(cmd.get("label_id"))
            elif cmd.get("label_id") in lst:
                lst.remove(cmd.get("label_id"))
            return {"ok": True}
        if verb == "label_message":
            row = {"chat": cmd.get("chat") or "",
                   "message_id": cmd.get("message_id") or "",
                   "label_id": cmd.get("label_id") or ""}
            if cmd.get("labeled"):
                if row not in self.message_labels:
                    self.message_labels.append(row)
            elif row in self.message_labels:
                self.message_labels.remove(row)
            return {"ok": True}
        if verb == "catalog":
            if self.catalog_engine == "go":
                return {"ok": False, "error": GO_CATALOG_ERROR}
            if self.catalog_error:
                return {"ok": False,
                        "error": self.catalog_error}
            if cmd.get("jid"):
                return {"ok": True,
                        "products": [dict(r) for r
                                     in FOREIGN_PRODUCTS],
                        "next_cursor": "",
                        "collections": [],
                        "collections_error": ""}
            limit = max(1, int(cmd.get("limit") or 10))
            start = int(cmd.get("cursor") or 0)
            page = self.products[start:start + limit]
            end = start + len(page)
            return {"ok": True,
                    "products": [dict(r) for r in page],
                    "next_cursor": (str(end) if end
                                    < len(self.products)
                                    else ""),
                    "collections": [{
                        "id": "C1", "name": "New in",
                        "product_ids": [
                            p["id"] for p in self.products]}],
                    "collections_error": ""}
        if verb == "catalog_product":
            if self.catalog_engine == "go":
                return {"ok": False, "error": GO_CATALOG_ERROR}
            action = cmd.get("action")
            if action == "create":
                self._product_seq += 1
                row = {"id": f"P{self._product_seq}",
                       "name": cmd.get("name") or "",
                       "price": cmd.get("price"),
                       "currency": cmd.get("currency") or "",
                       "description":
                           cmd.get("description") or "",
                       "retailer_id":
                           cmd.get("retailer_id") or "",
                       "url": cmd.get("url") or "",
                       "hidden": False,
                       "availability": "in stock"}
                self.products.append(row)
                return {"ok": True, "product": dict(row)}
            if action == "update":
                for i, row in enumerate(self.products):
                    if row["id"] == cmd.get("product_id"):
                        merged = dict(row)
                        for key in ("name", "price",
                                    "currency", "description",
                                    "retailer_id", "url"):
                            if key in cmd:
                                merged[key] = cmd[key]
                        self.products[i] = merged
                        return {"ok": True,
                                "product": dict(merged)}
                return {"ok": False,
                        "error": "product not found"}
            if action == "delete":
                before = len(self.products)
                self.products = [
                    p for p in self.products
                    if p["id"] != cmd.get("product_id")]
                return {"ok": True,
                        "deleted": before - len(self.products)}
            return {"ok": False, "error": "unknown action"}
        return {"ok": True}

    def cmds(self, name):
        with self.lock:
            return [c for c in self.commands
                    if c.get("cmd") == name]

    def close(self):
        self.shutdown()
        self.server_close()


def wait_for(pred, timeout=25.0, step=0.1):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(step)
    return pred()


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings
    from nomorals.social.chat.gateway import ChatGateway
    from nomorals.social.chat.local import LocalAdapter
    from nomorals.social.pairing import AccountRecord

    tmp = tmp_path_factory.mktemp("parity17")
    bridge = FakeBridge()
    settings = load_settings(overrides={
        "home": str(tmp),
        "partner.platforms": "local",
        "partner.owner_sender_ids": OWNER,
        "chat.local_enabled": "true",
        "chat.whatsapp_enabled": "true",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    gateway = ChatGateway({"local": LocalAdapter()}, db=None)
    runtime = PartnerRuntime(context, gateway=gateway)
    sup = runtime._pairing_supervisor()
    ws = Path(settings.workspace_dir)
    rec = sup.registry.upsert(AccountRecord(
        id=f"whatsapp:{OWNER}", platform="whatsapp",
        identity=OWNER,
        creds_dir=str(ws / "data" / "whatsapp_sessions"
                      / f"wa-{OWNER}"),
        port=PORT, status="linked", created_at=1.0))

    def beat():
        runtime.on_message(ChatMessage(
            chat=ChatRef(platform="local", chat_id="console",
                         kind=ChatKind.DM, title="console",
                         peer="owner"),
            incoming=True, text="hello", sender="owner",
            sender_id="owner",
            message_id=f"beat-{time.time()}", ts=time.time()))

    beat()
    assert wait_for(
        lambda: runtime.gateway.adapters.get("whatsapp")
        is not None)
    adapter = runtime.gateway.adapters["whatsapp"]
    assert wait_for(lambda: adapter.connected.is_set())
    svc = runtime._pairing_personal()
    env = {"runtime": runtime, "bridge": bridge,
           "adapter": adapter, "sup": sup, "rec": rec,
           "svc": svc, "personal": svc["personal"],
           "vault": svc["personal"].vault, "tmp": tmp,
           "beat": beat}
    yield env
    runtime.stop()
    runtime.gateway.stop()
    bridge.close()


def _op(rig, text, *, platform="whatsapp", chat_id=PN_POSTER,
        kind=ChatKind.DM, title="", sender_id=None, mid="",
        reply_to=""):
    if sender_id is None:
        sender_id = OWNER
    chat = ChatRef(platform=platform, chat_id=chat_id,
                   kind=kind, title=title, peer=chat_id)
    msg = ChatMessage(
        chat=chat, incoming=True, text=text, sender="Owner",
        sender_id=sender_id,
        message_id=mid or f"op-{time.time()}",
        reply_to=reply_to, ts=time.time(),
        meta={"owner_command": True})
    return rig["runtime"].handle_control(
        text, chat.key, message=msg)


def _journal_rows(rig, kind):
    return [r for r in rig["vault"].journal(rig["rec"].id)
            if r.get("kind") == kind]


def _ordinal(rendered, name):
    """The listing number a rendered /labels or /catalog
    reply gives ``name`` — the handle the writes take."""
    for line in rendered.splitlines():
        head, _, rest = line.partition(". ")
        if head.isdigit() and name in rest:
            return int(head)
    raise AssertionError(f"{name!r} not in listing:\n"
                         f"{rendered}")


# ── bridge source pins (both engines + the binaries) ───────

def test_go_bridge_carries_the_commerce_verbs():
    dispatch = (ROOT / "bridge" / "gobridge"
                / "messages.go").read_text()
    for verb in ("labels", "label_edit", "label_chat",
                 "label_message", "catalog",
                 "catalog_product"):
        assert f'case "{verb}"' in dispatch, verb
    src = (ROOT / "bridge" / "gobridge"
           / "commerce.go").read_text()
    # labels ride whatsmeow's real app-state surface
    assert "appstate.BuildLabelEdit" in src
    assert "appstate.BuildLabelChat" in src
    assert "appstate.BuildLabelMessage" in src
    assert "SendAppState" in src
    assert "FetchAppState" in src
    # the read is the sync, stated in the coverage line
    assert "the sync is the read" in src
    # the catalog refusal names the platform truth
    assert "whatsmeow exposes no catalog API" in src


def test_baileys_bridge_carries_the_commerce_verbs():
    src = (ROOT / "bridge"
           / "whatsapp-bridge.mjs").read_text()
    for verb in ("labels", "label_edit", "label_chat",
                 "label_message", "catalog",
                 "catalog_product"):
        assert f"case '{verb}'" in src, verb
    # Baileys' own label + catalog calls
    assert "wa.addLabel" in src
    assert "wa.addChatLabel" in src
    assert "wa.addMessageLabel" in src
    assert "wa.getCatalog" in src
    assert "wa.productCreate" in src
    assert "wa.productUpdate" in src
    assert "wa.productDelete" in src
    # label state rides the app-state sync events back
    assert "wa.ev.on('labels.edit'" in src
    assert "wa.ev.on('labels.association'" in src


def test_rebuilt_binaries_carry_the_verbs():
    for name in ("whatsapp-bridge-linux",
                 "whatsapp-bridge.exe"):
        data = (ROOT / "bridge" / "bin" / name).read_bytes()
        assert b"the catalog is not reachable on the Go " \
               b"bridge" in data, name
        assert b"label colors run 0 to 19" in data, name


def test_catalog_registers_the_family():
    from nomorals.social.chat.chatops import CHATOPS_COMMANDS
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
        PUBLIC_COMMANDS,
    )

    for kind in ("labels", "catalog", "broadcast", "wpay"):
        assert kind in CHATOPS_COMMANDS
        assert kind in CONTROL_COMMANDS
        assert kind in COMMAND_DETAILS
        assert kind in LIST_ONELINERS
        assert any(kind in names for _t, names in LIST_GROUPS)
        # owner-only account surfaces — never public
        assert kind not in PUBLIC_COMMANDS


# ── the formatters (pure) ──────────────────────────────────

def _labels_verdict():
    return {"ok": True, "labels": [dict(r) for r in LABEL_SEED],
            "chat_labels": {PN_POSTER: ["L2"]},
            "message_labels": [
                {"chat": PN_POSTER, "message_id": "m9",
                 "label_id": "L1"}],
            "coverage": "a test coverage line"}


def test_format_labels_numbers_and_hides_ids():
    from nomorals.social.chat.chatops import format_labels

    out = format_labels(_labels_verdict())
    # name order: Needs reply (1), VIP (2); tombstone hidden
    assert "1. Needs reply · color 3 · 1 chat labeled" in out
    assert "2. VIP · color 5" in out
    assert "Old label" not in out
    assert "(1 message carrying a label)" in out
    assert "(read a test coverage line)" in out
    for raw_id in ("L1", "L2", "L3"):
        assert raw_id not in out


def test_format_labels_empty_and_refused():
    from nomorals.social.chat.chatops import format_labels

    out = format_labels({"ok": True, "labels": [],
                         "chat_labels": {},
                         "message_labels": [],
                         "coverage": "x"})
    assert "no labels on this account" in out
    assert "Business" in out
    out = format_labels({"ok": False,
                         "error": "bridge is not logged in"})
    assert out == ("the labels didn't load — "
                   "bridge is not logged in")


def test_format_catalog_numbers_prices_and_hides_ids():
    from nomorals.social.chat.chatops import format_catalog

    verdict = {"ok": True,
               "products": [dict(r) for r in PRODUCT_SEED],
               "next_cursor": "",
               "collections": [{"id": "C1", "name": "New in",
                                "product_ids": ["P1", "P2"]}],
               "collections_error": ""}
    out = format_catalog(verdict, own=True)
    assert "Your catalog:" in out
    assert "1. Ankara Gown · NGN 45000" in out
    # a fractional price prints as it is, no invented zeros
    assert "2. Leather Sandals · NGN 18500.5" in out
    assert "Hand-dyed, size M" in out
    assert "Collections: New in (2)" in out
    for raw_id in ("P1", "P2", "C1"):
        assert raw_id not in out
    foreign = format_catalog(
        {"ok": True, "products": FOREIGN_PRODUCTS,
         "next_cursor": "", "collections": [],
         "collections_error": ""}, own=False)
    assert "Their catalog:" in foreign
    assert "Foreign Widget · USD 10" in foreign


def test_format_catalog_cut_page_and_collections_error():
    from nomorals.social.chat.chatops import format_catalog

    out = format_catalog(
        {"ok": True, "products": [PRODUCT_SEED[0]],
         "next_cursor": "50", "collections": [],
         "collections_error": "iq timeout"}, own=True)
    assert "the catalog has more beyond this page" in out
    assert "the collections read failed: iq timeout" in out
    out = format_catalog({"ok": True, "products": [],
                          "next_cursor": "",
                          "collections": [],
                          "collections_error": ""})
    assert "is empty" in out
    out = format_catalog({"ok": False, "error": "nope"})
    assert out == "the catalog didn't load — nope"


def test_base_adapter_reports_the_truth():
    from nomorals.social.chat.base import ChatAdapter

    class _Bare(ChatAdapter):
        name = "bare"

        def run(self, handler):  # pragma: no cover - unused
            raise NotImplementedError

        def send(self, chat, text, **kwargs):  # pragma: no cover
            from nomorals.social.chat.base import SendResult

            return SendResult(ok=False, platform=self.name,
                              error="unused in this test")

    ad = _Bare()
    chat = ChatRef(platform="bare", chat_id="1")
    for verdict in (ad.get_labels(),
                    ad.edit_label("X"),
                    ad.label_chat(chat, "L1"),
                    ad.label_message(chat, "m1", "L1"),
                    ad.get_catalog(),
                    ad.create_product("X", 1, "NGN"),
                    ad.update_product("P1", name="Y"),
                    ad.delete_product("P1")):
        assert verdict["ok"] is False
        assert "not supported on this adapter" in (
            verdict["error"])


# ── adapter pins: WhatsApp ─────────────────────────────────

@pytest.fixture()
def wa_adapter(monkeypatch):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    ad = WhatsAppAdapter(port=19994)
    ad.connected.set()
    calls: list[dict] = []
    state = {"labels": [dict(r) for r in LABEL_SEED],
             "products": [dict(r) for r in PRODUCT_SEED],
             "no_id": False}

    def fake_send_cmd(payload, timeout=15):
        calls.append(payload)
        verb = payload.get("cmd")
        if verb == "labels":
            return {"ok": True, "labels": state["labels"],
                    "chat_labels": {}, "message_labels": [],
                    "coverage": "test"}
        if verb == "label_edit":
            if state["no_id"]:
                return {"ok": True}
            return {"ok": True, "label_id":
                    payload.get("label_id") or "L9"}
        if verb == "catalog":
            limit = int(payload.get("limit") or 10)
            start = int(payload.get("cursor") or 0)
            page = state["products"][start:start + limit]
            end = start + len(page)
            return {"ok": True, "products": page,
                    "next_cursor": (
                        str(end) if end < len(state["products"])
                        else ""),
                    "collections": [],
                    "collections_error": ""}
        if verb == "catalog_product":
            if state["no_id"]:
                return {"ok": True, "product": {}}
            if payload.get("action") == "create":
                return {"ok": True, "product": {
                    "id": "P9", "name": payload.get("name")}}
            return {"ok": True}
        return {"ok": True}

    monkeypatch.setattr(ad, "_send_cmd", fake_send_cmd)
    ad.calls = calls  # type: ignore[attr-defined]
    ad.state = state  # type: ignore[attr-defined]
    return ad


def _wa_chat(chat_id=PN_POSTER, kind=ChatKind.DM):
    return ChatRef(platform="whatsapp", chat_id=chat_id,
                   kind=kind, peer=chat_id)


def test_wa_label_payloads(wa_adapter):
    verdict = wa_adapter.get_labels()
    assert wa_adapter.calls[-1] == {"cmd": "labels"}
    assert verdict["ok"]
    verdict = wa_adapter.edit_label("School fees", color=7)
    assert wa_adapter.calls[-1] == {
        "cmd": "label_edit", "label_id": "",
        "name": "School fees", "deleted": False, "color": 7}
    assert verdict["ok"] and verdict["label_id"] == "L9"
    verdict = wa_adapter.edit_label(
        "VIP Gold", label_id="L1", color=5)
    assert wa_adapter.calls[-1]["label_id"] == "L1"
    verdict = wa_adapter.edit_label(label_id="L1",
                                    deleted=True)
    assert wa_adapter.calls[-1] == {
        "cmd": "label_edit", "label_id": "L1", "name": "",
        "deleted": True}
    verdict = wa_adapter.label_chat(_wa_chat(), "L1")
    assert wa_adapter.calls[-1] == {
        "cmd": "label_chat", "chat": PN_POSTER,
        "label_id": "L1", "labeled": True}
    verdict = wa_adapter.label_message(
        _wa_chat(), "m9", "L2", labeled=False)
    assert wa_adapter.calls[-1] == {
        "cmd": "label_message", "chat": PN_POSTER,
        "message_id": "m9", "label_id": "L2",
        "labeled": False}


def test_wa_label_shape_refusals_before_traffic(wa_adapter):
    cases = [
        (lambda: wa_adapter.edit_label(""), "a label needs a name"),
        (lambda: wa_adapter.edit_label("X", color=20),
         "colors run 0 to 19"),
        (lambda: wa_adapter.edit_label("X", color="blue"),
         "colors run 0 to 19"),
        (lambda: wa_adapter.edit_label(deleted=True),
         "deleting a label needs"),
        (lambda: wa_adapter.label_chat(_wa_chat(), ""),
         "labeling needs the label's id"),
        (lambda: wa_adapter.label_message(_wa_chat(), "", "L1"),
         "labeling a message needs the message's id"),
    ]
    for call, fragment in cases:
        verdict = call()
        assert verdict["ok"] is False, fragment
        assert fragment in verdict["error"], verdict
    assert wa_adapter.calls == []  # nothing reached the bridge


def test_wa_label_unproven_write_is_not_success(wa_adapter):
    wa_adapter.state["no_id"] = True
    verdict = wa_adapter.edit_label("Ghost")
    assert verdict["ok"] is False
    assert "unproven" in verdict["error"]


def test_wa_catalog_payloads(wa_adapter):
    verdict = wa_adapter.get_catalog()
    assert wa_adapter.calls[-1] == {"cmd": "catalog",
                                    "limit": 10}
    verdict = wa_adapter.get_catalog(
        "2347000000000@s.whatsapp.net", limit=50, cursor="50")
    assert wa_adapter.calls[-1] == {
        "cmd": "catalog", "limit": 50,
        "jid": "2347000000000@s.whatsapp.net",
        "cursor": "50"}
    verdict = wa_adapter.create_product(
        "Ankara Cap", 12000, "ngn", description="One size")
    assert wa_adapter.calls[-1] == {
        "cmd": "catalog_product", "action": "create",
        "name": "Ankara Cap", "price": 12000.0,
        "currency": "NGN", "description": "One size"}
    assert verdict["ok"]
    verdict = wa_adapter.delete_product("P1")
    assert wa_adapter.calls[-1] == {
        "cmd": "catalog_product", "action": "delete",
        "product_id": "P1"}


def test_wa_catalog_shape_refusals_before_traffic(wa_adapter):
    cases = [
        (lambda: wa_adapter.create_product("", 1, "NGN"),
         "a product needs a name"),
        (lambda: wa_adapter.create_product("X", "abc", "NGN"),
         "price must be a number"),
        (lambda: wa_adapter.create_product("X", -5, "NGN"),
         "price must be a number"),
        (lambda: wa_adapter.create_product("X", 1, "NG"),
         "three-letter code"),
        (lambda: wa_adapter.update_product("", name="Y"),
         "needs the product's id"),
        (lambda: wa_adapter.delete_product(""),
         "needs the product's id"),
    ]
    for call, fragment in cases:
        verdict = call()
        assert verdict["ok"] is False, fragment
        assert fragment in verdict["error"], verdict
    assert wa_adapter.calls == []  # nothing reached the bridge


def test_wa_update_product_is_read_modify_write(wa_adapter):
    verdict = wa_adapter.update_product("P2", price=19999)
    assert verdict["ok"]
    verbs = [c["cmd"] for c in wa_adapter.calls]
    assert verbs == ["catalog", "catalog_product"]
    sent = wa_adapter.calls[-1]
    assert sent["action"] == "update"
    assert sent["product_id"] == "P2"
    # the merged record carries the untouched fields as
    # the catalog itself reported them
    assert sent["name"] == "Leather Sandals"
    assert sent["price"] == 19999.0
    assert sent["currency"] == "NGN"
    assert sent["retailer_id"] == "SKU-2"


def test_wa_update_product_missing_refuses(wa_adapter):
    verdict = wa_adapter.update_product("P404", price=1)
    assert verdict["ok"] is False
    assert "not in your catalog" in verdict["error"]
    assert all(c["cmd"] != "catalog_product"
               for c in wa_adapter.calls)


def test_wa_create_product_unproven_is_not_success(wa_adapter):
    wa_adapter.state["no_id"] = True
    verdict = wa_adapter.create_product("Ghost", 1, "NGN")
    assert verdict["ok"] is False
    assert "unproven" in verdict["error"]


def test_wa_commerce_disconnected(monkeypatch):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    ad = WhatsAppAdapter(port=19993)
    verdict = ad.get_labels()
    assert verdict["ok"] is False
    assert verdict["error"] == "whatsapp bridge not connected"
    verdict = ad.get_catalog()
    assert verdict["ok"] is False
    assert verdict["error"] == "whatsapp bridge not connected"


# ── telegram surface: the truth in words ───────────────────

def _tg_message(platform="telegram"):
    chat = ChatRef(platform=platform, chat_id="4242",
                   kind=ChatKind.DM, peer="4242")
    return ChatMessage(chat=chat, incoming=True, text="x",
                       sender="Owner", sender_id="1")


def test_telegram_answers_commerce_truth_in_words():
    from nomorals.agents.partner.runtime_chatops import (
        RuntimeChatOpsMixin,
    )

    mixin = RuntimeChatOpsMixin()
    consulted: list = []
    mixin._adapter_for = (  # type: ignore[method-assign]
        lambda platform: consulted.append(platform))
    out = mixin._chatops_labels("", _tg_message())
    assert "WhatsApp Business surface" in out
    out = mixin._chatops_catalog("", _tg_message())
    assert "WhatsApp Business surface" in out
    out = mixin._chatops_broadcast("", _tg_message())
    assert "broadcast lists are a WhatsApp shape" in out
    out = mixin._chatops_wpay("", _tg_message())
    assert "WhatsApp Pay can't be reached" in out
    assert "Nigeria" in out
    assert consulted == []  # answered before any adapter work


# ── composed chain: /labels on WhatsApp ────────────────────

def test_labels_list_end_to_end(rig):
    out = _op(rig, "/labels")
    assert "Your WhatsApp labels:" in out
    assert "1. Needs reply · color 3 · 1 chat labeled" in out
    assert "2. VIP · color 5" in out
    assert "Old label" not in out
    assert "(read a test coverage line)" in out
    assert rig["bridge"].cmds("labels")


def test_labels_new_journals(rig):
    out = _op(rig, "/labels new School fees | 7")
    assert "“School fees” is created" in out
    sent = rig["bridge"].cmds("label_edit")[-1]
    assert sent["name"] == "School fees"
    assert sent["color"] == 7
    assert sent["deleted"] is False
    rows = _journal_rows(rig, "label_changed")
    assert rows and rows[-1]["field"] == "created"
    assert rows[-1]["text"] == "School fees"
    listing = _op(rig, "/labels")
    assert "School fees" in listing


def test_labels_rename_carries_the_current_color(rig):
    listing = _op(rig, "/labels")
    n = _ordinal(listing, "VIP")
    out = _op(rig, f"/labels rename {n} VIP Gold")
    assert "now “VIP Gold”" in out
    sent = rig["bridge"].cmds("label_edit")[-1]
    assert sent["label_id"] == "L1"
    assert sent["name"] == "VIP Gold"
    # a definition is one record: the rename carried the
    # color the fresh read reported, it did not reset it
    assert sent["color"] == 5
    rows = _journal_rows(rig, "label_changed")
    assert rows[-1]["field"] == "renamed"


def test_labels_delete_and_bad_number(rig):
    before = len(rig["bridge"].cmds("label_edit"))
    out = _op(rig, "/labels delete 99")
    assert "there is no label 99" in out
    assert len(rig["bridge"].cmds("label_edit")) == before
    listing = _op(rig, "/labels")
    n = _ordinal(listing, "Needs reply")
    out = _op(rig, f"/labels delete {n}")
    assert "“Needs reply” is deleted" in out
    sent = rig["bridge"].cmds("label_edit")[-1]
    assert sent["label_id"] == "L2" and sent["deleted"] is True
    rows = _journal_rows(rig, "label_changed")
    assert rows[-1]["field"] == "deleted"
    assert "Needs reply" not in _op(rig, "/labels")


def test_labels_chat_and_message(rig):
    listing = _op(rig, "/labels")
    vip = _ordinal(listing, "VIP Gold")
    out = _op(rig, f"/labels chat {vip}")
    assert "this chat is labeled “VIP Gold”" in out
    sent = rig["bridge"].cmds("label_chat")[-1]
    assert sent["chat"] == PN_POSTER
    assert sent["label_id"] == "L1"
    assert sent["labeled"] is True
    out = _op(rig, f"/labels chat {vip} off")
    assert "is off this chat" in out
    assert rig["bridge"].cmds("label_chat")[-1]["labeled"] \
        is False
    # a message label needs the reply it lands on
    before = len(rig["bridge"].cmds("label_message"))
    out = _op(rig, f"/labels msg {vip}")
    assert "reply to the message" in out
    assert len(rig["bridge"].cmds("label_message")) == before
    out = _op(rig, f"/labels msg {vip}", reply_to="m9")
    assert "that message is labeled “VIP Gold”" in out
    sent = rig["bridge"].cmds("label_message")[-1]
    assert sent["message_id"] == "m9"
    assert sent["label_id"] == "L1"
    rows = [r for r in _journal_rows(rig, "label_changed")
            if r.get("field") == "message_labeled"]
    assert rows and rows[-1]["source_id"] == "m9"


def test_labels_writes_stop_gated_reads_not(rig):
    rig["sup"].registry.set_owner_stopped("whatsapp", True)
    try:
        out = _op(rig, "/labels")
        assert "Your WhatsApp labels:" in out  # the read runs
        before = len(rig["bridge"].cmds("label_edit"))
        out = _op(rig, "/labels new Blocked Label")
        assert "standing stop" in out
        assert len(rig["bridge"].cmds("label_edit")) == before
    finally:
        rig["sup"].registry.set_owner_stopped(
            "whatsapp", False)


def test_labels_usage_for_unknown_verb(rig):
    out = _op(rig, "/labels frobnicate")
    assert "usage: /labels" in out


# ── composed chain: /catalog on WhatsApp ───────────────────

def test_catalog_read_end_to_end(rig):
    out = _op(rig, "/catalog")
    assert "Your catalog:" in out
    assert "1. Ankara Gown · NGN 45000" in out
    assert "2. Leather Sandals · NGN 18500.5" in out
    assert "Collections: New in (2)" in out
    assert "P1" not in out and "P2" not in out
    assert rig["bridge"].cmds("catalog")


def test_catalog_foreign_by_number(rig):
    out = _op(rig, "/catalog 2347099999999")
    assert "Their catalog:" in out
    assert "Foreign Widget · USD 10" in out
    sent = rig["bridge"].cmds("catalog")[-1]
    assert sent["jid"] == PN_POSTER


def test_catalog_add_then_delete_by_number(rig):
    out = _op(rig, "/catalog add Ankara Cap | 12000 | NGN")
    assert "“Ankara Cap” is in your catalog" in out
    sent = rig["bridge"].cmds("catalog_product")[-1]
    assert sent["action"] == "create"
    assert sent["price"] == 12000.0
    rows = _journal_rows(rig, "catalog_changed")
    assert rows and rows[-1]["field"] == "created"
    assert rows[-1]["text"] == "Ankara Cap"
    listing = _op(rig, "/catalog")
    n = _ordinal(listing, "Ankara Cap")
    out = _op(rig, f"/catalog del {n}")
    assert "“Ankara Cap” is out of your catalog" in out
    sent = rig["bridge"].cmds("catalog_product")[-1]
    assert sent["action"] == "delete"
    rows = _journal_rows(rig, "catalog_changed")
    assert rows[-1]["field"] == "deleted"
    assert "Ankara Cap" not in _op(rig, "/catalog")


def test_catalog_edit_price_keeps_the_record(rig):
    listing = _op(rig, "/catalog")
    n = _ordinal(listing, "Ankara Gown")
    out = _op(rig, f"/catalog edit {n} price 47000")
    assert "now has price 47000" in out
    sent = rig["bridge"].cmds("catalog_product")[-1]
    assert sent["action"] == "update"
    assert sent["price"] == 47000.0
    # read-modify-write: the rest of the record survived
    assert sent["name"] == "Ankara Gown"
    assert sent["currency"] == "NGN"
    assert sent["description"] == "Hand-dyed, size M"
    rows = _journal_rows(rig, "catalog_changed")
    assert rows[-1]["field"] == "updated"


def test_catalog_bad_add_and_bad_field(rig):
    before = len(rig["bridge"].cmds("catalog_product"))
    out = _op(rig, "/catalog add Only A Name")
    assert "usage: /catalog add" in out
    out = _op(rig, "/catalog add X | notanumber | NGN")
    assert "the price must be a number" in out
    listing = _op(rig, "/catalog")
    n = _ordinal(listing, "Ankara Gown")
    out = _op(rig, f"/catalog edit {n} weight 3")
    assert "the editable fields are" in out
    assert len(rig["bridge"].cmds("catalog_product")) == before


def test_catalog_writes_stop_gated_reads_not(rig):
    rig["sup"].registry.set_owner_stopped("whatsapp", True)
    try:
        out = _op(rig, "/catalog")
        assert "Your catalog:" in out  # the read runs
        before = len(rig["bridge"].cmds("catalog_product"))
        out = _op(rig, "/catalog add Blocked | 1 | NGN")
        assert "standing stop" in out
        assert len(rig["bridge"].cmds(
            "catalog_product")) == before
    finally:
        rig["sup"].registry.set_owner_stopped(
            "whatsapp", False)


def test_catalog_go_engine_refusal_surfaces(rig):
    rig["bridge"].catalog_engine = "go"
    try:
        out = _op(rig, "/catalog")
    finally:
        rig["bridge"].catalog_engine = "baileys"
    # the bridge's own refusal is the answer, in its words
    assert "whatsmeow exposes no catalog API" in out
    assert "Catalog verbs ride the Baileys engine" in out


def test_catalog_bridge_refusal_is_named(rig):
    rig["bridge"].catalog_error = "bridge is not logged in"
    try:
        out = _op(rig, "/catalog")
    finally:
        rig["bridge"].catalog_error = ""
    # the chatop read path answers refusals with the
    # family's verdict line (the formatter's own refusal
    # wording is pinned by the pure formatter tests)
    assert out == "couldn't do it — bridge is not logged in"


# ── the honest closures: /broadcast + /wpay ────────────────

COMMERCE_VERBS = ("labels", "label_edit", "label_chat",
                  "label_message", "catalog",
                  "catalog_product")


def test_broadcast_and_wpay_answer_without_traffic(rig):
    before = {v: len(rig["bridge"].cmds(v))
              for v in COMMERCE_VERBS}
    out = _op(rig, "/broadcast")
    assert "broadcast lists live on your phone only" in out
    assert "not yet supported" in out
    assert "Nothing was sent" in out
    assert "/poststatus" in out
    out = _op(rig, "/wpay")
    assert "WhatsApp Pay can't be reached" in out
    assert "Nigeria" in out
    assert "Nothing was sent or requested" in out
    after = {v: len(rig["bridge"].cmds(v))
             for v in COMMERCE_VERBS}
    assert after == before  # neither answer touched a bridge
