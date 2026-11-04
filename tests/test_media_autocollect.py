"""§21 auto-collect: stickers landing in a claimed room whose
admin enabled the mode join Devon's expression library,
passively — the stealer's second mode.  Pins: provenance,
the §33 claimed gate, the admin toggle (default OFF, per
§21's "a mode a group admin enables"), content-hash dedupe,
the media2 capture discipline (magic bytes, .tgs / no-bytes
by reference), failure isolation (a capture explosion never
breaks the room's message handling), and the per-room
passive bound.
"""

from __future__ import annotations

import gzip
from types import SimpleNamespace

import pytest

from nomorals.agents.partner import failures as partner_failures
from nomorals.agents.partner.runtime_expression import (
    RuntimeExpressionMixin)
from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat import expression
from nomorals.social.chat.base import ChatKind

BOT_KEY = "telegram-bot:-100123"
BOT_KEY2 = "telegram-bot:-100456"
USER_KEY = "telegram:-100999"


# ── the rig (the test_media2 pattern) ──────────────────────


class _Chat:
    def __init__(self, platform, key, kind=ChatKind.GROUP):
        self.platform = platform
        self.key = key
        self.chat_id = key.split(":", 1)[1]
        self.title = "Builders"
        self.kind = kind


class _Msg:
    def __init__(self, chat, **kw):
        self.chat = chat
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "7")
        self.sender = kw.get("sender", "Ada")
        self.sender_username = kw.get("sender_username", "ada")
        self.text = kw.get("text", "")
        self.message_id = kw.get("message_id", "55")
        self.reply_to = kw.get("reply_to", "")
        self.media = kw.get("media") or []


class _Adapter:
    def __init__(self, tmp_path, serve=None, admin_ids=(),
                 bot_status="administrator"):
        self.calls = []
        self.media_sent = []
        self.admin_ids = set(admin_ids)
        self._bot_id = 4242
        self.bot_status = bot_status
        self.tmp_path = tmp_path
        self.serve = serve or {}

    def admin_member(self, chat, user_id):
        if str(user_id) == str(self._bot_id):
            status = self.bot_status
        elif str(user_id) in self.admin_ids:
            status = "administrator"
        else:
            status = "member"
        return {"ok": True, "result": {"status": status}}

    def send_media(self, chat, ref):
        self.media_sent.append(ref)
        return SimpleNamespace(ok=True, message_id="9",
                               media_file_id="TGFILE1", error="")

    def download_file(self, file_id, fname="media"):
        payload = self.serve.get(file_id)
        if payload is None:
            return ""
        dest = self.tmp_path / f"dl_{file_id}_{fname}"
        dest.write_bytes(payload)
        return str(dest)


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent = []

    def send(self, platform, chat, text, **kw):
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, message_id="1")


class _Runtime(RuntimeExpressionMixin, RuntimeGroupMixin,
               RuntimeGamesMixin):
    def __init__(self, tmp_path, adapters, operators=()):
        self.sent = []
        self.gateway = _Gateway(adapters)
        self.operators = set(operators)
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))

    def _send(self, chat, text, **kw):
        self.sent.append(text)
        return SimpleNamespace(ok=True, message_id="1")

    def _is_operator(self, message):
        return str(message.sender_id or "") in self.operators


def _rig(tmp_path, serve=None, platform="telegram-bot",
         key=BOT_KEY, kind=ChatKind.GROUP, **kw):
    adapter = _Adapter(tmp_path, serve=serve, **kw)
    rt = _Runtime(tmp_path, {platform: adapter}, operators={"99"})
    return rt, adapter, _Chat(platform, key, kind=kind)


def _webp_bytes(tmp_path, seed=0):
    PIL = pytest.importorskip("PIL")
    from PIL import Image

    path = tmp_path / f"fixture_{seed}.webp"
    Image.new("RGBA", (128, 128),
              (seed * 37 % 255, 90, 160, 255)).save(
        path, "WEBP", lossless=True)
    return path.read_bytes()


def _sticker_file(tmp_path, seed=0):
    path = tmp_path / f"in_{seed}.webp"
    path.write_bytes(_webp_bytes(tmp_path, seed))
    return path


def _sticker_msg(chat, *, path="", file_id="", name="sticker.webp",
                 **kw):
    media = [SimpleNamespace(kind="sticker", path=path,
                             file_id=file_id, name=name)]
    return _Msg(chat, media=media, **kw)


def _enable(rt, key=BOT_KEY):
    surface = rt._group_surface()
    rt._expression_update(surface, key, auto_collect=True)
    return surface


def _library(rt):
    return rt._expression_library()


def _auto_items(rt, origin=None):
    return [i for i in _library(rt).items
            if i.get("source") == "auto-collect"
            and (origin is None or i.get("origin") == origin)]


# ── the happy path ─────────────────────────────────────────


def test_collect_lands_in_library_with_provenance(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    _enable(rt)
    src = _sticker_file(tmp_path, seed=1)
    msg = _sticker_msg(chat, path=str(src), file_id="FID1")
    rt._group_surface_turn(msg)
    items = _auto_items(rt)
    assert len(items) == 1
    item = items[0]
    assert item["kind"] == "sticker"
    assert item["origin"] == BOT_KEY
    assert item["by_reference"] is False
    assert item["sha"]
    from pathlib import Path
    held = Path(item["path"])
    assert held.parent.name == "expression_media"
    assert held.exists()
    assert held.read_bytes() == src.read_bytes()


def test_collect_downloads_when_only_a_file_id(tmp_path):
    payload = _webp_bytes(tmp_path, seed=2)
    rt, adapter, chat = _rig(tmp_path, serve={"FID2": payload})
    _enable(rt)
    msg = _sticker_msg(chat, path="", file_id="FID2")
    rt._auto_collect_media(rt._group_surface(), BOT_KEY, msg)
    items = _auto_items(rt)
    assert len(items) == 1
    assert items[0]["by_reference"] is False
    assert items[0]["file_ids"] == {"telegram-bot": "FID2"}


def test_webm_is_kept_as_webm(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    _enable(rt)
    src = tmp_path / "anim.webm"
    src.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 64)
    msg = _sticker_msg(chat, path=str(src), name="anim.webm")
    rt._auto_collect_media(rt._group_surface(), BOT_KEY, msg)
    items = _auto_items(rt)
    assert len(items) == 1
    assert items[0]["path"].endswith(".webm")


# ── scope: claimed + enabled, nothing else ────────────────


def test_toggle_off_by_default_collects_nothing(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    src = _sticker_file(tmp_path, seed=3)
    msg = _sticker_msg(chat, path=str(src), file_id="FID3")
    rt._auto_collect_media(rt._group_surface(), BOT_KEY, msg)
    assert _auto_items(rt) == []


def test_unclaimed_room_collects_nothing(tmp_path):
    rt, adapter, chat = _rig(tmp_path, platform="telegram",
                             key=USER_KEY)
    surface = rt._group_surface()
    # the toggle is ON for this room too — the §33 claim
    # gate, not the toggle, is what refuses.
    rt._expression_update(surface, USER_KEY, auto_collect=True)
    src = _sticker_file(tmp_path, seed=4)
    msg = _sticker_msg(chat, path=str(src), file_id="FID4")
    rt._auto_collect_media(surface, USER_KEY, msg)
    assert _auto_items(rt) == []


def test_dm_collects_nothing(tmp_path):
    rt, adapter, chat = _rig(tmp_path, kind=ChatKind.DM)
    _enable(rt)
    src = _sticker_file(tmp_path, seed=5)
    msg = _sticker_msg(chat, path=str(src), file_id="FID5")
    rt._auto_collect_media(rt._group_surface(), BOT_KEY, msg)
    assert _auto_items(rt) == []


def test_non_sticker_media_is_ignored(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    _enable(rt)
    src = _sticker_file(tmp_path, seed=6)
    msg = _Msg(chat, media=[SimpleNamespace(
        kind="image", path=str(src), file_id="FID6", name="p.webp")])
    rt._auto_collect_media(rt._group_surface(), BOT_KEY, msg)
    assert _auto_items(rt) == []


# ── dedupe ─────────────────────────────────────────────────


def test_duplicate_sticker_is_a_single_entry(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    _enable(rt)
    src = _sticker_file(tmp_path, seed=7)
    surface = rt._group_surface()
    rt._auto_collect_media(
        surface, BOT_KEY,
        _sticker_msg(chat, path=str(src), file_id="FID7"))
    # the same sticker lands again under a DIFFERENT path
    # (a fresh download) with the same bytes — the content
    # hash, not the path, says: already held.
    src2 = tmp_path / "redownloaded.webp"
    src2.write_bytes(src.read_bytes())
    rt._auto_collect_media(
        surface, BOT_KEY,
        _sticker_msg(chat, path=str(src2), file_id="FID7B"))
    assert len(_auto_items(rt)) == 1


def test_duplicate_reference_is_a_single_entry(tmp_path):
    rt, adapter, chat = _rig(tmp_path, serve={"FID8": None})
    _enable(rt)
    surface = rt._group_surface()
    for _ in range(2):
        rt._auto_collect_media(
            surface, BOT_KEY,
            _sticker_msg(chat, path="", file_id="FID8"))
    assert len(_auto_items(rt)) == 1


# ── by-reference honesty ───────────────────────────────────


def test_tgs_is_by_reference_with_the_reason(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    _enable(rt)
    src = tmp_path / "anim.tgs"
    src.write_bytes(gzip.compress(b'{"v":"5.6.0","fr":60}'))
    msg = _sticker_msg(chat, path=str(src), file_id="FID9",
                       name="anim.tgs")
    rt._auto_collect_media(rt._group_surface(), BOT_KEY, msg)
    items = _auto_items(rt)
    assert len(items) == 1
    assert items[0]["by_reference"] is True
    assert items[0]["path"] == ""
    assert "Lottie" in items[0]["note"]


def test_no_bytes_is_by_reference(tmp_path):
    rt, adapter, chat = _rig(tmp_path, serve={"FID10": None})
    _enable(rt)
    msg = _sticker_msg(chat, path="", file_id="FID10")
    rt._auto_collect_media(rt._group_surface(), BOT_KEY, msg)
    items = _auto_items(rt)
    assert len(items) == 1
    assert items[0]["by_reference"] is True
    assert items[0]["file_ids"] == {"telegram-bot": "FID10"}
    assert "reference" in items[0]["note"]


# ── failure isolation ──────────────────────────────────────


def test_capture_failure_never_breaks_the_turn(tmp_path,
                                              monkeypatch):
    rt, adapter, chat = _rig(tmp_path)
    surface = _enable(rt)

    def _boom(*a, **kw):
        raise RuntimeError("capture exploded")

    monkeypatch.setattr(expression, "capture_library_sticker",
                        _boom)
    before = partner_failures.failure_count()
    src = _sticker_file(tmp_path, seed=11)
    msg = _sticker_msg(chat, path=str(src), file_id="FID11")
    out = rt._group_surface_turn(msg)
    assert out is False  # the turn completed normally
    member = surface.store.group(BOT_KEY)["members"]["7"]
    assert member["message_count"] >= 1  # the message processed
    assert _auto_items(rt) == []
    sites = [f.site for f in partner_failures.recent_failures(50)]
    assert "expression.auto_collect" in sites
    assert partner_failures.failure_count() > before


# ── the per-room bound ─────────────────────────────────────


def test_room_bound_stops_cleanly_and_is_per_room(tmp_path,
                                                  monkeypatch):
    monkeypatch.setattr(expression, "AUTO_COLLECT_ROOM_CAP", 3)
    rt, adapter, chat = _rig(tmp_path)
    surface = _enable(rt)
    rt._expression_update(surface, BOT_KEY2, auto_collect=True)
    chat2 = _Chat("telegram-bot", BOT_KEY2)
    for seed in range(20, 25):
        src = _sticker_file(tmp_path, seed=seed)
        rt._auto_collect_media(
            surface, BOT_KEY,
            _sticker_msg(chat, path=str(src),
                         file_id=f"F{seed}"))
    assert len(_auto_items(rt, origin=BOT_KEY)) == 3
    # a different room has its own bound — it still collects.
    src = _sticker_file(tmp_path, seed=30)
    rt._auto_collect_media(
        surface, BOT_KEY2,
        _sticker_msg(chat2, path=str(src), file_id="F30"))
    assert len(_auto_items(rt, origin=BOT_KEY2)) == 1


# ── the admin grammar ──────────────────────────────────────


def test_gsettings_collect_grammar_flips_the_toggle(tmp_path):
    rt, adapter, chat = _rig(tmp_path, admin_ids={"7"})
    surface = rt._group_surface()
    msg = _Msg(chat, sender_id="7")
    out = rt._control_gsettings("expression collect on", msg)
    assert "auto-collect on" in str(out)
    assert surface.store.settings(
        BOT_KEY)["expression"]["auto_collect"] is True
    out = rt._control_gsettings("expression collect off", msg)
    assert "auto-collect off" in str(out)
    assert surface.store.settings(
        BOT_KEY)["expression"]["auto_collect"] is False
    out = rt._control_gsettings("expression bogus", msg)
    assert "collect on|off" in str(out)
