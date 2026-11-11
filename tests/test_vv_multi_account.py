"""VV MULTI-ACCOUNT — /viewonce · /vv per account.

The capture half always keyed vault rows by the account that
received them (PersonalVault.save_media writes ``account`` on
every row, per-account dirs, per-account pruning); the reveal
half read the vault UNFILTERED — every account's captures in
one merged list and one index space, no account ever named.
This suite pins the completed surface:

  * resolution order — an explicit account token (the /pair
    saved [id] · /pair journal [account] precedent: id, number,
    or unique id tail, any position; ≤3 digits is the index),
    else the chat's own account (the seat platform's only
    paired account), else the configured default
    (resolve_account: primary → linked → registered);
  * captures never cross accounts — list, index, bare and
    reply forms under one account's scope never surface
    another's, and the per-account index space is its own;
  * with more than one personal account the reply NAMES the
    account it acted for, and other accounts' holdings are
    named rather than silently absent;
  * accountless legacy rows attribute to the resolved
    account only when it is the registry's ONLY personal
    account (nothing else they could have come from); with
    several accounts no scoped view claims them;
  * single-account deployments keep the shipped surface
    byte for byte.
"""
from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

from nomorals.agents.partner.runtime_chatops import (
    RuntimeChatOpsMixin,
)
from nomorals.agents.partner.runtime_provision import (
    RuntimeProvisionMixin,
)
from nomorals.social.pairing import AccountRecord, PairingRegistry

A_ID = "whatsapp:2348011111111"
B_ID = "whatsapp:2348022222222"
C_ID = "whatsapp:2348033333333"
T_ID = "telegram:777"
A_NUM = "2348011111111"
B_NUM = "2348022222222"
C_NUM = "2348033333333"

TS = 1_700_000_000.0


def _when(ts):
    return datetime.fromtimestamp(float(ts)).strftime("%d %b %H:%M")


class _RT(RuntimeProvisionMixin, RuntimeChatOpsMixin):
    pass


def _rig(tmp_path, accounts):
    """accounts: (id, platform, identity, label, created_at,
    view_once, primary) tuples.  Returns (rt, svc, sent)."""
    rt = _RT.__new__(_RT)
    rt.context = SimpleNamespace(settings=SimpleNamespace(
        workspace_dir=str(tmp_path), chat=None))
    registry = PairingRegistry(tmp_path / "paired.json")
    for (rid, platform, identity, label, created,
         vo, primary) in accounts:
        registry.upsert(AccountRecord(
            id=rid, platform=platform, identity=identity,
            label=label, status="linked", created_at=created,
            primary=primary))
        registry.set_toggles(rid, view_once=vo)
    rt._pair_supervisor = SimpleNamespace(
        registry=registry, is_running=lambda rid: True)
    svc = rt._pairing_personal()
    sent: list = []

    def send_file(platform, chat, path, caption="", kind=""):
        sent.append({"platform": platform, "path": path,
                     "caption": caption, "kind": kind})
        return SimpleNamespace(ok=True)

    rt.gateway = SimpleNamespace(send_file=send_file)
    return rt, svc, sent


def _msg(platform="whatsapp", reply_to=""):
    return SimpleNamespace(
        reply_to=reply_to, sender_id="",
        meta={"self_chat": True},
        chat=SimpleNamespace(platform=platform, peer="",
                             chat_id=""))


def _seed(tmp_path, svc, *, platform, account_id, mid, poster,
          ts, name):
    src = tmp_path / name
    src.write_bytes(b"\xff\xd8" + name.encode())
    out = svc["personal"].observe_message(
        platform, chat_key=f"{platform}:{poster}999",
        counterpart_id=f"{poster}999@s.whatsapp.net",
        sender_id=f"{poster}999@s.whatsapp.net",
        sender_name=poster, message_id=mid, text="",
        media=[{"path": str(src), "kind": "image"}],
        view_once=True, account_id=account_id, ts=ts)
    assert len(out) == 1
    return out[0]


def _legacy_row(tmp_path, svc, *, poster, ts, mid):
    """A view-once row in the shape from before rows carried
    an account: every field save_media writes, minus
    ``account``."""
    f = tmp_path / f"legacy-{mid}.jpg"
    f.write_bytes(b"\xff\xd8legacy")
    row = {"ts": ts, "kind": "view_once", "platform": "whatsapp",
           "counterpart_id": "2347000000000",
           "poster_name": poster, "media_kind": "image",
           "source_id": mid, "path": str(f), "bytes": 8}
    vault = svc["personal"].vault
    with open(vault.journal_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def _reveals(svc):
    return [r for r in svc["personal"].vault.journal()
            if r.get("kind") == "viewonce_reveal"]


def _multi(tmp_path):
    """Two WhatsApp accounts holding captures + one holding
    none.  Capture timestamps interleave across accounts, so
    scoping can never be an accident of time order."""
    rt, svc, sent = _rig(tmp_path, [
        (A_ID, "whatsapp", A_NUM, "Main Line", 1.0, True, False),
        (B_ID, "whatsapp", B_NUM, "Second Line", 2.0, True, False),
        (C_ID, "whatsapp", C_NUM, "Quiet Line", 3.0, False, False),
    ])
    rows = {
        "a_old": _seed(tmp_path, svc, platform="whatsapp",
                       account_id=A_ID, mid="A-OLD",
                       poster="Ada One", ts=TS + 100,
                       name="a-old.jpg"),
        "b_old": _seed(tmp_path, svc, platform="whatsapp",
                       account_id=B_ID, mid="B-OLD",
                       poster="Bola One", ts=TS + 150,
                       name="b-old.jpg"),
        "a_new": _seed(tmp_path, svc, platform="whatsapp",
                       account_id=A_ID, mid="A-NEW",
                       poster="Ada Two", ts=TS + 200,
                       name="a-new.jpg"),
        "b_new": _seed(tmp_path, svc, platform="whatsapp",
                       account_id=B_ID, mid="B-NEW",
                       poster="Bola Two", ts=TS + 250,
                       name="b-new.jpg"),
    }
    return rt, svc, sent, rows


# ── store keying the surface stands on ─────────────────────

def test_vault_rows_stay_keyed_per_account(tmp_path):
    _rt, svc, _sent, _rows = _multi(tmp_path)
    a_rows = svc["personal"].vault.saved_entries(A_ID)
    b_rows = svc["personal"].vault.saved_entries(B_ID)
    assert {r["poster_name"] for r in a_rows
            if r["kind"] == "view_once"} == {"Ada One", "Ada Two"}
    assert {r["poster_name"] for r in b_rows
            if r["kind"] == "view_once"} == {"Bola One", "Bola Two"}


# ── resolution: default, naming, separation ────────────────

def test_bare_reveals_default_account_and_names_it(tmp_path):
    rt, svc, sent, rows = _multi(tmp_path)
    reply = rt._chatops_viewonce("viewonce", "", _msg())
    # the default is the earliest linked WhatsApp account (A);
    # its newest capture is Ada Two even though Bola Two is
    # newer in the merged pool.
    assert "view-once from Ada Two" in reply
    assert "(Main Line)" in reply
    assert sent[0]["path"] == rows["a_new"]["path"]
    # B's captures are not in A's listing — they are NAMED as
    # the other account's holdings, never silently absent.
    assert "Bola" not in reply.split("other accounts:")[0]
    assert "other accounts: Second Line (2 captures)" in reply
    assert _reveals(svc)[0]["account"] == A_ID
    assert _reveals(svc)[0]["outcome"] == "found"


def test_list_is_scoped_and_named(tmp_path):
    rt, svc, sent, rows = _multi(tmp_path)
    reply = rt._chatops_viewonce("vv", "list", _msg())
    assert "view-once captures on Main Line" in reply
    assert "Ada Two" in reply and "Ada One" in reply
    assert "Bola" not in reply.split("other accounts:")[0]
    assert "other accounts: Second Line (2 captures)" in reply
    assert sent == []          # a list sends nothing


def test_selector_by_number_and_by_id(tmp_path):
    rt, svc, sent, rows = _multi(tmp_path)
    reply = rt._chatops_viewonce("viewonce", B_NUM, _msg())
    assert "view-once from Bola Two" in reply
    assert "(Second Line)" in reply
    assert sent[-1]["path"] == rows["b_new"]["path"]
    reply = rt._chatops_viewonce("viewonce", B_ID, _msg())
    assert "view-once from Bola Two" in reply
    assert sent[-1]["path"] == rows["b_new"]["path"]


def test_selector_index_space_is_per_account(tmp_path):
    rt, svc, sent, rows = _multi(tmp_path)
    # index 2 in B's scope is B's older capture — in the old
    # merged pool index 2 was A-NEW (global newest-first ran
    # B-NEW, A-NEW, B-OLD, A-OLD).
    reply = rt._chatops_viewonce(
        "viewonce", f"{B_NUM} 2", _msg())
    assert "view-once from Bola One" in reply
    assert sent[-1]["path"] == rows["b_old"]["path"]
    # tokens in any order (the /pair journal rule)
    reply = rt._chatops_viewonce(
        "viewonce", f"list {B_NUM}", _msg())
    assert "view-once captures on Second Line" in reply
    assert "Bola One" in reply and "Ada" not in reply


def test_index_miss_is_scoped_and_journaled_per_account(tmp_path):
    rt, svc, sent, rows = _multi(tmp_path)
    reply = rt._chatops_viewonce("viewonce", f"{B_NUM} 9", _msg())
    assert "there are 2 captures on Second Line" in reply
    assert "pick 1–2" in reply
    rows_j = _reveals(svc)
    assert rows_j[-1]["outcome"] == "miss"
    assert rows_j[-1]["account"] == B_ID


def test_scoped_empty_names_that_account_not_a_merged_empty(
        tmp_path):
    rt, svc, sent, rows = _multi(tmp_path)
    reply = rt._chatops_viewonce("viewonce", C_NUM, _msg())
    assert "no view-once captures yet on Quiet Line" in reply
    assert "capture is OFF for that account" in reply
    assert "Right now:" not in reply        # not the merged text
    assert "Main Line (2 captures)" in reply
    assert "Second Line (2 captures)" in reply
    assert _reveals(svc)[-1]["outcome"] == "empty"
    assert _reveals(svc)[-1]["account"] == C_ID


def test_scoped_empty_with_capture_on_says_on(tmp_path):
    rt, svc, sent = _rig(tmp_path, [
        (A_ID, "whatsapp", A_NUM, "Main Line", 1.0, True, False),
        (B_ID, "whatsapp", B_NUM, "Second Line", 2.0, True, False),
    ])
    _seed(tmp_path, svc, platform="whatsapp", account_id=A_ID,
          mid="A-1", poster="Ada", ts=TS + 100, name="a.jpg")
    reply = rt._chatops_viewonce("viewonce", B_NUM, _msg())
    assert "no view-once captures yet on Second Line" in reply
    assert "capture is ON for that account" in reply
    assert "/pair toggle" not in reply     # already on: no lesson
    assert "Main Line (1 capture)" in reply


def test_reply_never_crosses_accounts(tmp_path):
    rt, svc, sent, rows = _multi(tmp_path)
    # B's capture asked for by reply under the default (A)
    # scope: a miss — B's media is never handed back.
    reply = rt._chatops_viewonce(
        "vv", "", _msg(reply_to="B-NEW"))
    assert "no view-once capture" in reply
    assert "in Main Line's vault" in reply
    assert sent == []
    assert _reveals(svc)[-1]["outcome"] == "miss"
    assert _reveals(svc)[-1]["account"] == A_ID
    # the same reply WITH B's selector opens B's capture.
    reply = rt._chatops_viewonce(
        "vv", B_NUM, _msg(reply_to="B-NEW"))
    assert "view-once from Bola Two" in reply
    assert sent[-1]["path"] == rows["b_new"]["path"]


def test_file_gone_names_the_account(tmp_path):
    rt, svc, sent, rows = _multi(tmp_path)
    from pathlib import Path
    Path(rows["b_new"]["path"]).unlink()
    reply = rt._chatops_viewonce("viewonce", B_NUM, _msg())
    assert "on Second Line is in the record" in reply
    assert _reveals(svc)[-1]["outcome"] == "file_gone"


# ── resolution: seat beats default, primary is default ─────

def test_chat_own_account_beats_configured_default(tmp_path):
    rt, svc, sent = _rig(tmp_path, [
        (A_ID, "whatsapp", A_NUM, "Main Line", 1.0, True, True),
        (T_ID, "telegram", "777", "TG Line", 2.0, True, False),
    ])
    _seed(tmp_path, svc, platform="whatsapp", account_id=A_ID,
          mid="A-1", poster="Ada", ts=TS + 100, name="a.jpg")
    tg = _seed(tmp_path, svc, platform="telegram",
               account_id=T_ID, mid="T-1", poster="Tega",
               ts=TS + 50, name="t.jpg")
    # the seat is Telegram: its ONLY account answers, even
    # though WhatsApp's account is the named primary.
    reply = rt._chatops_viewonce("viewonce", "", _msg("telegram"))
    assert "view-once from Tega" in reply
    assert "(TG Line)" in reply
    assert sent[-1]["path"] == tg["path"]


def test_configured_default_is_the_primary(tmp_path):
    rt, svc, sent = _rig(tmp_path, [
        (A_ID, "whatsapp", A_NUM, "Main Line", 1.0, True, False),
        (B_ID, "whatsapp", B_NUM, "Second Line", 2.0, True, True),
    ])
    _seed(tmp_path, svc, platform="whatsapp", account_id=A_ID,
          mid="A-1", poster="Ada", ts=TS + 300, name="a.jpg")
    b = _seed(tmp_path, svc, platform="whatsapp",
              account_id=B_ID, mid="B-1", poster="Bola",
              ts=TS + 100, name="b.jpg")
    # a bot seat names no account of its own: the primary (B)
    # answers — not the earliest, not the newest capture.
    reply = rt._chatops_viewonce(
        "viewonce", "", _msg("telegram-bot"))
    assert "view-once from Bola" in reply
    assert "(Second Line)" in reply
    assert sent[-1]["path"] == b["path"]


# ── legacy accountless rows ────────────────────────────────

def test_accountless_rows_unclaimed_when_accounts_are_many(
        tmp_path):
    rt, svc, sent, rows = _multi(tmp_path)
    _legacy_row(tmp_path, svc, poster="Legacy Ghost",
                ts=TS + 400, mid="LEG-1")
    reply = rt._chatops_viewonce("viewonce", "list", _msg())
    assert "Legacy Ghost" not in reply
    reply = rt._chatops_viewonce("viewonce", f"{B_NUM} list",
                                 _msg())
    assert "Legacy Ghost" not in reply
    # it is not deleted or hidden from the unscoped store view
    assert any(r.get("poster_name") == "Legacy Ghost"
               for r in svc["personal"].vault.saved_entries())


def test_accountless_rows_attribute_to_the_only_account(
        tmp_path):
    rt, svc, sent = _rig(tmp_path, [
        (A_ID, "whatsapp", A_NUM, "", 1.0, True, False),
    ])
    row = _legacy_row(tmp_path, svc, poster="Legacy Ghost",
                      ts=TS + 400, mid="LEG-1")
    reply = rt._chatops_viewonce("viewonce", "list", _msg())
    assert "Legacy Ghost" in reply
    reply = rt._chatops_viewonce("viewonce", "", _msg())
    assert "view-once from Legacy Ghost" in reply
    assert sent[-1]["path"] == row["path"]


# ── usage: not an attempt, grammar taught when multi ───────

def test_usage_error_journals_nothing_and_teaches_selector(
        tmp_path):
    rt, svc, sent, rows = _multi(tmp_path)
    reply = rt._chatops_viewonce("viewonce", "bogus", _msg())
    assert reply.startswith("usage:")
    assert "ACCOUNT" in reply
    assert _reveals(svc) == []


# ── single-account byte parity ─────────────────────────────

def _single(tmp_path):
    rt, svc, sent = _rig(tmp_path, [
        (A_ID, "whatsapp", A_NUM, "", 1.0, True, False),
    ])
    return rt, svc, sent


def test_single_account_replies_are_the_shipped_bytes(tmp_path):
    rt, svc, sent = _single(tmp_path)
    _seed(tmp_path, svc, platform="whatsapp", account_id=A_ID,
          mid="S-OLD", poster="Ada", ts=TS + 100, name="s1.jpg")
    _seed(tmp_path, svc, platform="whatsapp",
          account_id=A_ID, mid="S-NEW", poster="Ada",
          ts=TS + 200, name="s2.jpg")
    listing = (f"  1. {_when(TS + 200)} · Ada · image\n"
               f"  2. {_when(TS + 100)} · Ada · image")
    reply = rt._chatops_viewonce("viewonce", "list", _msg())
    assert reply == ("👁️ view-once captures (newest first) — "
                     f"/viewonce N opens one:\n{listing}")
    reply = rt._chatops_viewonce("viewonce", "", _msg())
    assert reply == (
        f"👁️ that's the view-once from Ada ({_when(TS + 200)}), "
        "straight from your vault."
        f"\nthe rest:\n{listing}\n/viewonce N opens another.")
    assert sent[-1]["caption"] == (
        f"👁️ view-once from Ada · {_when(TS + 200)} — kept in "
        "your vault before it could vanish")
    reply = rt._chatops_viewonce("viewonce", "9", _msg())
    assert reply == (f"there are 2 captures — pick 1–2:\n"
                     f"{listing}")
    reply = rt._chatops_viewonce("viewonce", "bogus", _msg())
    assert reply == ("usage: /viewonce · /viewonce N · "
                     "/viewonce list · or reply to the "
                     "view-once message with /viewonce")


def test_single_account_empty_vault_is_the_shipped_bytes(
        tmp_path):
    rt, svc, sent = _single(tmp_path)
    reply = rt._chatops_viewonce("viewonce", "", _msg())
    assert reply == (
        "no view-once captures yet — when the view_once "
        "toggle is on, view-once media sent to your personal "
        "accounts is kept in your vault before it vanishes, "
        "and /viewonce shows it back. Right now: "
        f"{A_ID}: capture ON. Turn it on with /pair toggle "
        "ACCOUNT view_once on.")
    rows = _reveals(svc)
    assert len(rows) == 1
    assert rows[0]["outcome"] == "empty"
    assert rows[0]["account"] == A_ID
