"""DESIGN §21 + §22 — the media expression organ.

Pinned here, at the organ level (transport-free):

- the ONE button vocabulary: labels are emoji-led, labelling
  is idempotent, per-group presentation overrides win, and
  the notice keyboard inherits the same map;
- the full Unicode emoji table: search by meaning finds the
  tone variants, the ZWJ families, the flags and the rare
  faces — coverage follows Unicode, not a phone's font;
- her custom emoji set: refs where a platform carries them,
  sticker art next, the unicode face everywhere else;
- the library: dedupe, per-room no-immediate-repeat rotation,
  file_id capture after a first send and reuse by handle;
- moments: warnings stay DRY at every setting, ban theatre
  obeys the frequency ladder, an empty world ships text-only
  (sent=False with a reason — never a fake send);
- providers: the health breaker stands a failing source down
  after 3 failures and a fetched GIF JOINS the library;
- conversions ride the house ffmpeg seam (fake runner for the
  argv shapes, a real-binary smoke test when ffmpeg exists);
- the maker produces a REAL 512×512 WebP sticker;
- adapter reaction verbs: Telegram bot setMessageReaction,
  Discord place + read.
"""

from __future__ import annotations

import asyncio
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.chat import expression as expr
from nomorals.social.chat.expression import (
    BUTTON_EMOJI,
    ExpressionLibrary,
    ExpressionService,
    ProviderHealth,
    button_label,
    buttons_from_spec,
    classify_notice_moment,
    emoji_info,
    pick_reaction,
    reaction_allowed,
    search_emoji,
)
from nomorals.social.chat.group_notices import notice_keyboard


# ── the button vocabulary ─────────────────────────────────────


def test_button_vocabulary_labels_and_idempotence():
    assert button_label("undo", "Undo") == "↩️ Undo"
    assert button_label("ban", "Ban") == "🔨 Ban"
    assert button_label("approve", "Approve") == "✅ Approve"
    # already led (a hand-rolled site mid-migration): untouched
    assert button_label("undo", "↩️ Undo") == "↩️ Undo"
    assert button_label("rules", "📜 Rules") == "📜 Rules"
    # unknown action: the text stands as authored
    assert button_label("frobnicate", "Frob") == "Frob"
    # every action the brief names has a glyph
    for action in ("undo", "warn", "mute", "ban", "kick",
                   "approve", "decline", "settings", "back",
                   "home", "rules", "games", "balance", "quests",
                   "economy", "estate", "defense", "acceptance",
                   "presentation", "welcome", "leaderboard",
                   "steal", "gif", "sticker", "convert", "react"):
        assert action in BUTTON_EMOJI, action


def test_button_vocabulary_group_overrides():
    settings = {"presentation": {
        "button_emoji": {"undo": "⏪"}}}
    assert button_label("undo", "Undo", settings=settings) \
        == "⏪ Undo"
    off = {"presentation": {"emoji_buttons": False}}
    assert button_label("undo", "Undo", settings=off) == "Undo"


def test_buttons_from_spec_mixed_rows():
    rows = buttons_from_spec([
        [("approve", "Approve", "joinq approve 1"),
         ("decline", "Decline", "joinq decline 1")],
        [("📋 The queue", "joinq")],  # ready pair passes through
    ])
    assert rows[0] == [("✅ Approve", "joinq approve 1"),
                       ("🚫 Decline", "joinq decline 1")]
    assert rows[1] == [("📋 The queue", "joinq")]


def test_notice_keyboard_inherits_the_vocabulary():
    kb = notice_keyboard({"id": "n7"})
    assert kb == [[("↩️ Undo", "undonotice n7")]]
    assert notice_keyboard({"id": "n7", "undone": True}) is None


# ── the full emoji space ──────────────────────────────────────


def test_emoji_search_across_the_full_table():
    assert search_emoji("melting", limit=1)[0].char == "🫠"
    assert search_emoji("grinning face", limit=1)[0].char == "😀"
    tones = search_emoji("thumbs up medium", limit=2)
    assert tones[0].char == "👍🏽"
    fams = search_emoji("family", limit=6)
    assert any("‍" in e.char for e in fams)  # ZWJ chains found
    flags = search_emoji("flag ng", limit=4)
    assert "🇳🇬" in [e.char for e in flags]
    rocket = search_emoji("rocket moon", limit=3)
    assert "🚀" in [e.char for e in rocket]


def test_emoji_info_names_anything():
    assert emoji_info("❤️")["name"] == "heavy black heart"
    fam = emoji_info("👨‍👩‍👧")
    assert "family" in fam["name"]
    # an UNLISTED ZWJ chain is still described, from its parts
    info = emoji_info("🧑‍🧑‍🧒‍🧒")
    assert info is not None and "sequence" in info["name"]


def test_reaction_rules_and_platform_sets():
    # the Bot API set is a real limit; userbot/WA/Discord are open
    assert reaction_allowed("telegram-bot", "🔥") is True
    assert reaction_allowed("telegram-bot", "🫠") is False
    assert reaction_allowed("telegram", "🫠") is True
    assert reaction_allowed("discord", "🫠") is True
    assert pick_reaction("we won the jackpot!",
                         platform="telegram-bot") == "🏆"
    assert pick_reaction("welcome to the group") == "👋"
    assert pick_reaction("that joke landed 😂") == "😂"


def test_custom_emoji_registry_render_order():
    from nomorals.social.chat.expression import CustomEmojiRegistry

    reg = CustomEmojiRegistry()
    reg.add("devon", unicode="🥷")
    assert reg.render("devon", "whatsapp") == {
        "kind": "unicode", "value": "🥷", "name": "devon"}
    reg.add("devon", platform="telegram-bot", ref="FID55")
    # expression completion (2026-11-07): a telegram ref IS a
    # custom_emoji_id from her own packs — the render kind
    # says so (was "sticker_ref" when refs were file_ids).
    assert reg.render("devon", "telegram-bot")["kind"] \
        == "custom_ref"
    assert reg.render("devon", "telegram-bot")["value"] == "FID55"
    # a platform without the ref still gets the face
    assert reg.render("devon", "discord")["value"] == "🥷"
    assert reg.render("nope", "discord")["kind"] == "missing"


# ── the library + moments ─────────────────────────────────────


def _stock(tmp_path, n=3, tags=("welcome",)):
    lib = ExpressionLibrary(tmp_path / "lib.json")
    items = []
    media = tmp_path / "media"
    media.mkdir(exist_ok=True)
    for i in range(n):
        p = media / f"g{i}.gif"
        p.write_bytes(b"GIF89a fake")
        items.append(lib.add_item(kind="gif", path=str(p),
                                  tags=list(tags), source="test"))
    return lib, items


def _ok_send(captured):
    def send(ref):
        captured.append(ref)
        return SimpleNamespace(ok=True, message_id="1",
                               media_file_id="",
                               error="")
    return send


def test_moment_rotation_has_no_immediate_repeats(tmp_path):
    lib, _items = _stock(tmp_path)
    svc = ExpressionService(lib, cooldown_s=0,
                            rng=random.Random(7))
    seen = []
    for _ in range(3):
        out = svc.fire("welcome", room="r1", platform="discord",
                       settings={}, send=_ok_send([]))
        assert out["sent"] is True
        seen.append(out["item_id"])
    assert seen[0] != seen[1] and seen[1] != seen[2]


def test_warning_is_dry_at_every_setting(tmp_path):
    lib, _items = _stock(tmp_path)
    svc = ExpressionService(lib, cooldown_s=0)
    captured: list = []
    out = svc.fire("warn", room="r1", platform="discord",
                   settings={"enabled": True, "gifs": "normal"},
                   send=_ok_send(captured))
    assert out["sent"] is False
    assert out["reason"] == "dry by design"
    assert captured == []


def test_ban_theatre_obeys_the_frequency_ladder(tmp_path):
    lib, _items = _stock(tmp_path, tags=("ban", "hammer"))
    svc = ExpressionService(lib, cooldown_s=0)
    low = svc.fire("ban", room="r1", platform="discord",
                   settings={"gifs": "low"}, send=_ok_send([]))
    assert low["sent"] is False and "low" in low["reason"]
    off = svc.fire("ban", room="r1", platform="discord",
                   settings={"ban_theatre": False},
                   send=_ok_send([]))
    assert off["sent"] is False
    on = svc.fire("ban", room="r1", platform="discord",
                  settings={"gifs": "normal"}, send=_ok_send([]))
    assert on["sent"] is True


def test_empty_world_ships_text_only_never_fake(tmp_path):
    lib = ExpressionLibrary(tmp_path / "lib.json")
    svc = ExpressionService(lib, cooldown_s=0)
    out = svc.fire("welcome", room="r1", platform="discord",
                   settings={}, send=_ok_send([]))
    assert out == {"sent": False, "moment": "welcome",
                   "reason": "no media available — text only"}


def test_file_id_captured_then_resent_by_handle(tmp_path):
    lib, items = _stock(tmp_path, n=1)
    svc = ExpressionService(lib, cooldown_s=0)
    refs: list = []

    def send(ref):
        refs.append(ref)
        return SimpleNamespace(ok=True, message_id="1",
                               media_file_id="TGFILE1", error="")

    out = svc.fire("welcome", room="room-a",
                   platform="telegram-bot", settings={}, send=send)
    assert out["sent"] is True and out["via"] == "upload"
    assert refs[0].kind == "animation"  # gif → sendAnimation kind
    assert items[0]["file_ids"]["telegram-bot"] == "TGFILE1"
    # a second room, same platform: the handle is reused, and
    # the path is not even consulted
    out2 = svc.fire("welcome", room="room-b",
                    platform="telegram-bot", settings={}, send=send)
    assert out2["via"] == "file_id"
    assert refs[1].file_id == "TGFILE1"


def test_library_dedupes_and_survives_corruption(tmp_path):
    lib = ExpressionLibrary(tmp_path / "lib.json")
    a = lib.add_item(kind="sticker", path="/x.webp",
                     file_ids={"telegram-bot": "F1"})
    b = lib.add_item(kind="sticker", path="/y.webp",
                     file_ids={"telegram-bot": "F1"})
    assert a["id"] == b["id"] and len(lib) == 1
    (tmp_path / "lib.json").write_text("{not json")
    lib2 = ExpressionLibrary(tmp_path / "lib.json")
    assert lib2.recovered is True and len(lib2) == 0
    assert list(tmp_path.glob("lib.corrupt-*"))


def test_provider_health_stands_down_and_library_grows(tmp_path,
                                                      monkeypatch):
    lib = ExpressionLibrary(tmp_path / "lib.json")

    class Dead:
        name = "dead"
        calls = 0

        def search(self, query, limit=4):
            Dead.calls += 1
            raise OSError("boom")

    # The stand-down story runs with her own renderer
    # arranged down too (the every-rung-down case), so
    # provider health is the whole story.
    from nomorals.social.chat import emoji_studio

    def _boom(*a, **k):
        raise RuntimeError("renderer down")

    monkeypatch.setattr(emoji_studio, "mood_animation", _boom)
    monkeypatch.setattr(emoji_studio, "sticker_master_png",
                        _boom)
    svc = ExpressionService(lib, providers=[Dead()],
                            media_dir=tmp_path / "m", cooldown_s=0)
    for _ in range(3):
        out = svc.fire("welcome", room="r", platform="discord",
                       settings={}, send=_ok_send([]))
        assert out["sent"] is False
    assert svc.health.status("dead")["fails"] == 3
    svc.fire("welcome", room="r", platform="discord",
             settings={}, send=_ok_send([]))
    assert Dead.calls == 3  # stood down: not even attempted
    monkeypatch.undo()

    # Expression completion (2026-11-07): with her renderer
    # UP, a dead provider no longer means silence — she
    # CREATES the moment herself and the piece joins the
    # library with its provenance.
    svc_self = ExpressionService(lib, providers=[Dead()],
                                 media_dir=tmp_path / "m",
                                 cooldown_s=0)
    out = svc_self.fire("welcome", room="r9", platform="discord",
                        settings={}, send=_ok_send([]))
    assert out["sent"] is True
    assert any(item["source"] == "self:render"
               for item in lib.items)

    class Live:
        name = "live"

        def search(self, query, limit=4):
            return [{"url": "https://x.test/party.gif",
                     "title": "party", "kind": "gif"}]

    def fake_download(url, dest_dir, **kw):
        dest = Path(dest_dir) / "got.gif"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"GIF89a")
        return dest

    monkeypatch.setattr(expr, "download_to_library", fake_download)
    lib2 = ExpressionLibrary(tmp_path / "lib2.json")
    svc2 = ExpressionService(lib2, providers=[Live()],
                             media_dir=tmp_path / "m", cooldown_s=0)
    out = svc2.fire("quest_complete", room="r2",
                    platform="discord", settings={},
                    send=_ok_send([]))
    assert out["sent"] is True
    assert len(lib2) == 1
    assert lib2.items[0]["source"] == "provider:live"


def test_provider_health_semantics():
    health = ProviderHealth()
    assert health.available("x", now=100.0)
    for i in range(3):
        health.note("x", False, now=100.0)
    assert not health.available("x", now=100.0)
    assert health.available("x", now=100.0 + 301)
    health.note("x", True, now=100.0)
    assert health.status("x")["fails"] == 0


def test_classify_notice_moment():
    assert classify_notice_moment(
        "🎉 group quest complete: Loud Week") == "quest_complete"
    assert classify_notice_moment(
        "🏆 season 12 champions") == "season_podium"
    assert classify_notice_moment("⚠️ a warning") == ""
    assert classify_notice_moment("🪙 balance moved") == ""


# ── conversions + the maker ───────────────────────────────────


def _fake_runner(calls):
    def run(args, **kw):
        calls.append(list(args))
        Path(args[-1]).write_bytes(b"converted")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return run


def test_conversion_argv_shapes_via_the_ffmpeg_seam(tmp_path):
    from nomorals.social.chat import media_convert

    src = tmp_path / "in.webp"
    src.write_bytes(b"webp")
    calls: list = []
    out = media_convert.sticker_to_mp4(
        src, out_dir=tmp_path, runner=_fake_runner(calls))
    assert out.suffix == ".mp4" and out.exists()
    assert "-loop" in calls[0] and "libx264" in calls[0]

    vid = tmp_path / "in.mp4"
    vid.write_bytes(b"mp4")
    calls.clear()
    out = media_convert.mp4_to_mp3(
        vid, out_dir=tmp_path, runner=_fake_runner(calls))
    assert out.suffix == ".mp3" and "libmp3lame" in calls[0]

    aud = tmp_path / "in.mp3"
    aud.write_bytes(b"mp3")
    calls.clear()
    out = media_convert.mp3_to_video(
        aud, out_dir=tmp_path, runner=_fake_runner(calls))
    assert out.suffix == ".mp4"
    assert any("showwaves" in a for a in calls[0])

    with pytest.raises(media_convert.ConversionError):
        media_convert.convert_media("nope", vid,
                                    runner=_fake_runner([]))


def test_image_to_sticker_and_text_sticker_are_real(tmp_path):
    PIL = pytest.importorskip("PIL")
    from PIL import Image

    from nomorals.social.chat import media_convert

    src = tmp_path / "pic.png"
    Image.new("RGB", (800, 400), (200, 40, 40)).save(src)
    out = media_convert.image_to_sticker(src, out_dir=tmp_path)
    img = Image.open(out)
    assert img.size == (512, 512) and img.format == "WEBP"

    made = media_convert.make_text_sticker(
        "no wahala", out_dir=tmp_path)
    assert made["raster"], made["note"]
    img = Image.open(made["raster"])
    assert img.size == (512, 512) and img.format == "WEBP"
    assert made["spec"].id  # the SVG spec remains of record


@pytest.mark.skipif(not shutil.which("ffmpeg"),
                    reason="ffmpeg binary absent on this machine")
def test_real_ffmpeg_smoke(tmp_path):
    import wave

    from nomorals.social.chat import media_convert

    wav = tmp_path / "tone.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x10" * 4000)
    out = media_convert.mp3_to_video(wav, out_dir=tmp_path)
    assert out.exists() and out.stat().st_size > 1000


# ── adapter reaction verbs ────────────────────────────────────


def test_telegram_bot_react_uses_set_message_reaction():
    from nomorals.social.chat.base import ChatKind, ChatRef
    from nomorals.social.chat.telegram import TelegramBotAdapter

    adapter = TelegramBotAdapter.__new__(TelegramBotAdapter)
    adapter.name = "telegram-bot"
    adapter.stats = {"sent": 0, "send_errors": 0}
    calls: list = []
    adapter._api = lambda method, **kw: calls.append((method, kw)) or True
    chat = ChatRef(platform="telegram-bot", chat_id="-1001",
                   kind=ChatKind.GROUP)
    out = adapter.react(chat, "55", "🔥")
    assert out.ok
    method, kw = calls[0]
    assert method == "setMessageReaction"
    assert kw["message_id"] == 55
    import json as _json
    assert _json.loads(kw["reaction"]) == [
        {"type": "emoji", "emoji": "🔥"}]


def test_telegram_bot_send_media_kinds_and_file_id_capture(
        tmp_path):
    from nomorals.social.chat.base import (
        ChatKind, ChatRef, MediaRef)
    from nomorals.social.chat.telegram import TelegramBotAdapter

    adapter = TelegramBotAdapter.__new__(TelegramBotAdapter)
    adapter.name = "telegram-bot"
    adapter.stats = {"sent": 0, "send_errors": 0}
    uploads: list = []
    apis: list = []

    def fake_upload(method, data, field, path, **kw):
        uploads.append((method, field))
        return {"message_id": 9,
                "sticker": {"file_id": "STK1"},
                "animation": {"file_id": "ANI1"}}

    adapter._api_upload = fake_upload
    adapter._api = lambda method, **kw: apis.append((method, kw)) \
        or {"message_id": 10, "animation": {"file_id": "ANI1"}}
    chat = ChatRef(platform="telegram-bot", chat_id="-1001",
                   kind=ChatKind.GROUP)
    sticker = tmp_path / "s.webp"
    sticker.write_bytes(b"webp")
    out = adapter.send_media(chat, MediaRef(
        path=str(sticker), kind="sticker"))
    assert out.ok and uploads[0] == ("sendSticker", "sticker")
    assert out.media_file_id == "STK1"
    # by handle: no upload happens at all
    out2 = adapter.send_media(chat, MediaRef(
        path="", kind="animation", file_id="ANI1"))
    assert out2.ok and len(uploads) == 1
    assert apis[0][0] == "sendAnimation"
    assert apis[0][1]["animation"] == "ANI1"


def test_discord_react_and_read_reactions():
    from nomorals.social.chat.base import ChatKind, ChatRef
    from nomorals.social.chat.discord import DiscordAdapter

    class FakeReaction:
        def __init__(self, emoji, count):
            self.emoji, self.count = emoji, count

    class FakeMessage:
        def __init__(self):
            self.reactions = [FakeReaction("🔥", 3),
                              FakeReaction("👍", 1)]
            self.added: list = []

        async def add_reaction(self, emoji):
            self.added.append(emoji)

    class FakeChannel:
        def __init__(self, message):
            self._message = message

        async def fetch_message(self, mid):
            return self._message

    message = FakeMessage()
    channel = FakeChannel(message)
    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter.name = "discord"
    adapter.stats = {"sent": 0, "send_errors": 0}
    adapter._dc = lambda: object()
    adapter._channel = lambda chat: asyncio.sleep(0, result=channel)
    adapter._run_on_loop = lambda coro: asyncio.run(coro)
    chat = ChatRef(platform="discord", chat_id="777",
                   kind=ChatKind.GROUP)
    out = adapter.react(chat, "42", "🔥")
    assert out.ok and message.added == ["🔥"]
    counts = adapter.read_reactions(chat, "42")
    assert counts == {"🔥": 3, "👍": 1}
