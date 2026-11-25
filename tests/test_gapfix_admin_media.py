"""Admin media theatre (audit N2, owner 2026-11-11).

"mute ban and the rest got no stickers nor gifs not even images and the
rests too" — and "the gifs command menu carries only anime images".

The pins in this file:

- The library never answers an admin moment with an unrelated item: a
  mute in an anime-only library is a miss, not a hug GIF.
- Every admin moment has its OWN studio design and renders its OWN
  animation — no more one happy face for mute/ban/kick/warn.
- Admin moments bypass the shared 25 s celebration cooldown and fire
  caption-capable kinds only (animation/gif/image/video — never a
  sticker, whose caption the Bot API drops).
- The typed admin commands answer WITH the media: the reply text rides
  the animation as its caption, so the handler returns "" (the media WAS
  the reply, the /gif + reaction contract). When the media cannot land,
  the text returns WITH the honest reason — never a silent bare line.
- The gifs menu (`/gif <word>`) serves the admin visuals from the same
  shared set.

Network-free: every transport is a fixture or the studio itself.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.chat import emoji_studio
from nomorals.social.chat.expression import (
    MOMENT_TAGS,
    ExpressionLibrary,
    ExpressionService,
)

try:
    # Added by this lane; absent on the pristine base, where the
    # tests below must FAIL (failing-first) rather than fail to
    # collect — ADMIN_MOMENTS is empty there by construction.
    from nomorals.social.chat.expression import ADMIN_MOMENTS
except ImportError:  # pragma: no cover - pristine base only
    ADMIN_MOMENTS = frozenset()
from nomorals.social.chat.control import LIST_ONELINERS

GIF_BYTES = b"GIF89a" + b"\x01\x00\x01\x00" + b"\x00" * 64

ADMIN_COMMAND_MOMENTS = (
    "mute", "unmute", "ban", "unban", "kick", "warn", "restrict",
    "reactwipe", "rmdeleted", "rminactive", "purge", "slowmode",
    "pin", "unpin",
)


def _ok_send(captured):
    def send(ref):
        captured.append(ref)
        return SimpleNamespace(ok=True, message_id="1",
                               media_file_id="", error="")
    return send


def _svc(tmp_path, **kw):
    lib = ExpressionLibrary(tmp_path / "lib.json")
    kw.setdefault("cooldown_s", 0)
    kw.setdefault("rng", random.Random(0))
    return ExpressionService(lib, **kw), lib


def _anime_library(tmp_path):
    _svc_obj, lib = _svc(tmp_path)
    for name in ("hug", "pat"):
        p = tmp_path / f"{name}.gif"
        p.write_bytes(GIF_BYTES)
        lib.add_item(kind="animation", path=str(p), tags=(name,),
                     moods=("happy",),
                     source="provider:waifu.pics")
    return lib


# ── the library: a miss is a miss ────────────────────────────────


def test_pick_returns_none_when_nothing_matches(tmp_path):
    lib = _anime_library(tmp_path)
    assert lib.pick(("mute", "quiet", "silence", "shush"),
                    ("strict",), room="room-1") is None


def test_pick_recent_window_still_yields_to_a_match(tmp_path):
    # The no-repeat rule yields only when a match EXISTS but is
    # recent — the fallback is about recency, never about
    # serving an unrelated item.
    _svc_obj, lib = _svc(tmp_path)
    p = tmp_path / "mute.gif"
    p.write_bytes(GIF_BYTES)
    item = lib.add_item(kind="animation", path=str(p),
                        tags=("mute",), source="test")
    lib.note_use(item["id"], room="room-1")
    picked = lib.pick(("mute", "quiet"), ("mute",), room="room-1")
    assert picked is not None and picked["id"] == item["id"]


# ── the studio: one design per admin moment ─────────────────────


def test_admin_moments_have_distinct_studio_designs():
    default = emoji_studio.mood_design("")
    designs = {}
    for moment in ADMIN_COMMAND_MOMENTS:
        design = emoji_studio.mood_design(moment)
        assert design != default, moment
        designs[moment] = design
    assert len({str(sorted(d.items())) for d in designs.values()}) \
        == len(ADMIN_COMMAND_MOMENTS)


def test_admin_moment_tag_sets_are_disjoint():
    # pick() scores tag overlap: a word shared by two admin
    # moments would let one act wear its sibling's visual
    # (ban's "goodbye" dressed a kick; unmute's "free" dressed
    # an unban).  The moment's own name is its first tag.
    seen: dict[str, str] = {}
    for moment in ADMIN_COMMAND_MOMENTS:
        tags, _moods = MOMENT_TAGS[moment]
        assert tags[0] == moment, moment
        for tag in tags:
            assert tag not in seen, (tag, seen[tag], moment)
            seen[tag] = moment


def test_admin_moments_are_registered():
    for moment in ADMIN_COMMAND_MOMENTS:
        assert moment in MOMENT_TAGS, moment
        assert moment in ADMIN_MOMENTS, moment


# ── the service: distinct animations, no cooldown, no stickers ─


def _fire_admin(svc, moment, tmp_path):
    captured: list = []
    out = svc.fire(moment, room="room-1", platform="telegram-bot",
                   settings={}, send=_ok_send(captured))
    assert out["sent"] is True, (moment, out)
    assert len(captured) == 1
    return captured[0], out


def test_admin_moments_self_create_distinct_animations(tmp_path):
    svc, _lib = _svc(tmp_path, media_dir=tmp_path / "m")
    refs = {}
    for moment in ADMIN_COMMAND_MOMENTS:
        ref, out = _fire_admin(svc, moment, tmp_path)
        assert ref.kind in ("animation", "gif"), (moment, ref.kind)
        assert out.get("kind") == ref.kind
        path = Path(ref.path)
        assert path.exists()
        assert path.read_bytes()[:6] in (b"GIF87a", b"GIF89a")
        refs[moment] = ref
    paths = [str(r.path) for r in refs.values()]
    assert len(set(paths)) == len(paths), paths


def test_admin_moment_bypasses_the_shared_cooldown(tmp_path):
    # Default 25 s cooldown: a welcome just fired in this room must
    # not eat the mute that follows it — enforcement is not ambience.
    lib = ExpressionLibrary(tmp_path / "lib.json")
    svc = ExpressionService(lib, providers=[],
                            media_dir=tmp_path / "m",
                            rng=random.Random(0))
    captured: list = []
    out = svc.fire("welcome", room="room-1", platform="telegram-bot",
                   settings={}, send=_ok_send(captured))
    assert out["sent"] is True
    ref, out = _fire_admin(svc, "mute", tmp_path)
    assert ref.kind in ("animation", "gif")


def test_admin_fire_never_picks_a_sticker(tmp_path):
    # A sticker shelved under an admin tag stays in the shared set
    # for /gif — but the admin reply never rides one, because the
    # Bot API drops a sticker's caption and the words would be lost.
    _svc_obj, lib = _svc(tmp_path, media_dir=tmp_path / "m")
    lib.add_item(kind="sticker", path="/mute.webp", tags=("mute",),
                 source="test")
    captured: list = []
    out = _svc_obj.fire(
        "mute", room="room-1", platform="telegram-bot", settings={},
        send=_ok_send(captured))
    assert out["sent"] is True, out
    assert captured[0].kind in ("animation", "gif"), captured[0].kind


def test_gif_for_admin_word_serves_the_admin_visual(tmp_path):
    svc, _lib = _svc(tmp_path, media_dir=tmp_path / "m")
    captured: list = []
    out = svc.gif_for("mute", room="room-1", platform="telegram-bot",
                      send=_ok_send(captured))
    assert out["sent"] is True, out
    assert captured[0].kind in ("animation", "gif")
    assert Path(captured[0].path).read_bytes()[:6] in (
        b"GIF87a", b"GIF89a")
    assert "mute" in (svc.library.items[0].get("tags") or [])


# ── the menu: the gifs listing serves admin visuals too ─────────


def test_gif_menu_names_the_admin_visuals():
    line = LIST_ONELINERS["gif"]
    for word in ("mute", "ban", "kick", "warn", "unmute", "unban"):
        assert word in line, line


def test_admin_moments_are_in_the_manifest():
    from nomorals.social.chat import surface_manifest as sm

    members = set(sm.surface_by_id("expression").members)
    for moment in ADMIN_COMMAND_MOMENTS:
        assert moment in members, moment


# ── the composed runtime: the reply carries its media ───────────


class _Chat:
    def __init__(self):
        from nomorals.social.chat.base import ChatKind

        self.platform = "telegram-bot"
        self.key = "telegram-bot:-100123"
        self.chat_id = "-100123"
        self.title = "Builders"
        self.kind = ChatKind.GROUP


class _Msg:
    def __init__(self, **kw):
        self.chat = _Chat()
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "7")
        self.sender = kw.get("sender", "Ada")
        self.sender_username = kw.get("sender_username", "ada")
        self.text = kw.get("text", "")
        self.message_id = kw.get("message_id", "55")
        self.reply_to = kw.get("reply_to", "")
        self.media = kw.get("media") or []


class _MediaAdapter:
    """The group's adapter: admin verbs succeed, media sends are
    recorded WITH their caption — the caption is the assertion."""

    def __init__(self, *, refuse_media=False):
        self.calls: list = []
        self.media_sent: list = []
        self.refuse_media = refuse_media
        self._bot_id = 4242

    def admin_member(self, chat, user_id):
        status = "administrator" if str(user_id) in (
            "4242", "7") else "member"
        return {"ok": True, "result": {"status": status}}

    def admin_restrict(self, chat, user_id, *, minutes=60):
        self.calls.append(("restrict", user_id))
        return {"ok": True}

    def admin_unrestrict(self, chat, user_id):
        self.calls.append(("unrestrict", user_id))
        return {"ok": True}

    def admin_ban(self, chat, user_id):
        self.calls.append(("ban", user_id))
        return {"ok": True}

    def admin_unban(self, chat, user_id):
        self.calls.append(("unban", user_id))
        return {"ok": True}

    def admin_ban_sender_chat(self, chat, sender_chat_id):
        self.calls.append(("ban_sender_chat", sender_chat_id))
        return {"ok": True}

    def admin_unban_sender_chat(self, chat, sender_chat_id):
        self.calls.append(("unban_sender_chat", sender_chat_id))
        return {"ok": True}

    def admin_reaction_wipe(self, chat, user_id):
        self.calls.append(("reaction_wipe", user_id))
        return {"ok": True}

    def admin_delete(self, chat, message_id):
        self.calls.append(("delete", message_id))
        return {"ok": True}

    def admin_pin(self, chat, message_id):
        self.calls.append(("pin", message_id))
        return {"ok": True}

    def admin_unpin(self, chat):
        self.calls.append(("unpin", ""))
        return {"ok": True}

    def admin_set_slow_mode(self, chat, seconds):
        self.calls.append(("slowmode", seconds))
        return {"ok": True}

    def send_media(self, chat, ref, caption="", buttons=None):
        self.media_sent.append((ref, caption))
        if self.refuse_media:
            return SimpleNamespace(ok=False, message_id="",
                                   media_file_id="", error="boom")
        return SimpleNamespace(ok=True, message_id="9",
                               media_file_id="F1", error="")


def _rig(tmp_path, *, refuse_media=False):
    from nomorals.agents.partner.runtime_expression import (
        RuntimeExpressionMixin,
    )
    from nomorals.agents.partner.runtime_group import (
        RuntimeGroupMixin,
    )

    adapter = _MediaAdapter(refuse_media=refuse_media)

    class _Runtime(RuntimeExpressionMixin, RuntimeGroupMixin):
        def __init__(self):
            self.gateway = SimpleNamespace(
                adapters={"telegram-bot": adapter})
            self.context = SimpleNamespace(
                settings=SimpleNamespace(
                    workspace_dir=str(tmp_path),
                    home_path=str(tmp_path)))
            self.sent: list = []

        def _send(self, chat, text, **kw):
            self.sent.append(text)
            return SimpleNamespace(ok=True, message_id="1")

        def _is_operator(self, message):
            return False

    rt = _Runtime()
    rt._group_surface().handle_message(
        "telegram-bot:-100123", "777", username="loud", text="hi")
    return rt, adapter


def _assert_captioned(rt, adapter, out, needle):
    """One captioned animation IS the reply: the handler returns
    "" and the words ride the media's caption."""
    assert out == "", out
    assert len(adapter.media_sent) == 1
    ref, caption = adapter.media_sent[0]
    assert ref.kind in ("animation", "gif"), ref.kind
    assert needle in caption, caption
    return ref, caption


@pytest.mark.parametrize("verb,call,needle", [
    ("mute", lambda rt: rt._control_mute("@loud 10m", _Msg()),
     "muted"),
    ("unmute", lambda rt: rt._control_unmute("@loud", _Msg()),
     "can speak again"),
    ("ban", lambda rt: rt._control_ban("@loud", _Msg()),
     "is banned"),
    ("unban", lambda rt: rt._control_unban("@loud", _Msg()),
     "is unbanned"),
    ("kick", lambda rt: rt._control_kick("@loud", _Msg()),
     "is out"),
    ("warn", lambda rt: rt._control_warn("@loud caps", _Msg()),
     "warning 1/3"),
    ("bansender", lambda rt: rt._control_bansender(
        "", _Msg(meta={"reply_to_sender_chat": {
            "id": "-100999", "title": "SpamChan"}})),
     "is banned"),
    ("unbansender", lambda rt: rt._control_bansender(
        "", _Msg(meta={"reply_to_sender_chat": {
            "id": "-100999", "title": "SpamChan"}}), unban=True),
     "may post here again"),
    ("reactwipe", lambda rt: rt._control_reactwipe("@loud", _Msg()),
     "wiped"),
    ("purge", lambda rt: rt._control_purge(
        "", _Msg(reply_to="123")), "gone"),
    ("pin", lambda rt: rt._control_pin("", _Msg(reply_to="123")),
     "pinned"),
    ("unpin", lambda rt: rt._control_unpin("", _Msg()),
     "unpinned"),
    ("slowmode", lambda rt: rt._control_slowmode("30s", _Msg()),
     "slow mode on"),
])
def test_typed_admin_commands_answer_with_captioned_media(
        tmp_path, verb, call, needle):
    rt, adapter = _rig(tmp_path)
    out = call(rt)
    _assert_captioned(rt, adapter, out, needle)


def test_warn_landing_on_the_mute_step_carries_the_mute_visual(
        tmp_path):
    # The default ladder mutes at 3: the third warn's reply is the
    # MUTE visual carrying the step's words — not a warn visual,
    # and not a bare line.
    rt, adapter = _rig(tmp_path)
    assert rt._control_warn("@loud caps", _Msg()) == ""
    assert rt._control_warn("@loud caps", _Msg()) == ""
    out = rt._control_warn("@loud caps", _Msg())
    assert out == "", out
    assert len(adapter.media_sent) == 3
    ref, caption = adapter.media_sent[2]
    assert ref.kind in ("animation", "gif")
    assert "that's the step" in caption and "muted" in caption


def test_cleanup_removal_carries_media_only_when_someone_was_removed(
        tmp_path, monkeypatch):
    from nomorals.agents.partner.runtime_group import (
        RuntimeGroupMixin,
    )

    rt, adapter = _rig(tmp_path)
    msg = _Msg()
    # No targets: a report, not an act — text only, no media.
    monkeypatch.setattr(
        RuntimeGroupMixin, "_cleanup_scan",
        lambda self, surface, key, message, days: (
            {"deleted": [], "inactive": []},
            {"rows": [], "probed": 0}))
    out = rt._control_rmdeleted("", msg)
    assert "nothing to remove" in out
    assert adapter.media_sent == []
    # One removal: the act carries its visual, captioned.
    monkeypatch.setattr(
        RuntimeGroupMixin, "_cleanup_scan",
        lambda self, surface, key, message, days: (
            {"deleted": [{"label": "Ghost", "user_id": "777"}],
             "inactive": []},
            {"rows": [{"user_id": "777"}], "probed": 1}))
    monkeypatch.setattr(
        RuntimeGroupMixin, "_cleanup_act_on",
        lambda self, surface, key, message, adapter, finding,
        which: ("removed", ""))
    out = rt._control_rmdeleted("", msg)
    _assert_captioned(rt, adapter, out, "removed")


# ── honest degradation: a dropped visual is NAMED ───────────────


def test_admin_reply_names_the_missing_visual_when_send_fails(
        tmp_path):
    rt, adapter = _rig(tmp_path, refuse_media=True)
    out = rt._control_mute("@loud 10m", _Msg())
    assert "muted" in out
    assert "(no mute visual here:" in out, out
    # The send was attempted (and refused) — the attempt is
    # recorded, the verdict still landed in words.
    assert len(adapter.media_sent) == 1
    _ref, caption = adapter.media_sent[0]
    assert "muted" in caption


def test_admin_theatre_reports_the_gate_reason(tmp_path):
    rt, _adapter = _rig(tmp_path)
    rt._fire_expression = (  # type: ignore[method-assign]
        lambda message, moment, caption="": {
            "sent": False,
            "reason": "gifs off in this group"})
    out = rt._admin_theatre(_Msg(), "mute", "🔇 Loud muted for 10m")
    assert out == ("🔇 Loud muted for 10m\n"
                   "(no mute visual here: gifs off in this group)")


def test_admin_theatre_returns_empty_when_the_media_lands(tmp_path):
    rt, _adapter = _rig(tmp_path)
    seen: dict = {}

    def _fire(message, moment, caption=""):
        seen["moment"] = moment
        seen["caption"] = caption
        return {"sent": True, "moment": moment}

    rt._fire_expression = _fire  # type: ignore[method-assign]
    out = rt._admin_theatre(_Msg(), "ban", "🔨 Loud is banned")
    assert out == ""
    assert seen == {"moment": "ban", "caption": "🔨 Loud is banned"}


def test_long_admin_reply_fires_uncaptioned_and_keeps_its_text(
        tmp_path):
    # Past the caption budget the media still fires, but the words
    # stay a text reply — truncating the verdict to fit a caption
    # would lose the record.
    rt, _adapter = _rig(tmp_path)
    seen: dict = {}

    def _fire(message, moment, caption=""):
        seen["caption"] = caption
        return {"sent": True, "moment": moment}

    rt._fire_expression = _fire  # type: ignore[method-assign]
    text = "🧹 deleted account cleanup: 1 removed · 0 failed · " \
        "0 skipped\n" + ("✓ member — removed\n" * 60)
    assert len(text) > 900
    out = rt._admin_theatre(_Msg(), "rmdeleted", text)
    assert out == text
    assert seen["caption"] == ""
