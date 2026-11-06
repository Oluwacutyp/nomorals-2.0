"""Reply-quoted media must reach the media commands — every surface.

The live bug (owner screenshot, 2026-11-06): in a WhatsApp group
a member replied to a PHOTO with /sticker and got the usage text
back.  The quoted photo never left the reply's contextInfo: both
WhatsApp bridges emitted only the stanza id, the WhatsApp adapter
never stamped replied-media meta, and the runtime resolver —
which every reply-media command shares (/sticker, /steal,
/convert, /whatanime, /removebg, /qr read, and /sham's audio) —
had no WhatsApp source to read.  The Telegram userbot had the
same class of drop (replied message never fetched), and Discord
never populated reply_to, leaving its fetch_media seam dead.

These tests drive the PRODUCTION paths with real payload shapes:
bridge event dicts exactly as the Go and Baileys bridges emit
them (including an LID-addressed sender), the WhatsApp adapter's
own dispatch, the runtime's real handlers, and a Telethon-shaped
inbound for the userbot.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nomorals.agents.partner.runtime_expression import RuntimeExpressionMixin
from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.whatsapp import WhatsAppAdapter

GROUP_JID = "120363012345678901@g.us"
DM_JID = "2348012345678@s.whatsapp.net"
FINN_JID = "2349024392318@s.whatsapp.net"
FINN_LID = "998877665544332211@lid"


def _photo(tmp_path: Path) -> Path:
    p = tmp_path / "quoted_photo.jpg"
    Image.new("RGB", (320, 240), (200, 40, 40)).save(p, "JPEG")
    return p


def _wa_deliver(obj: dict) -> list:
    adapter = WhatsAppAdapter()
    msgs: list = []
    adapter._dispatch(obj, msgs.append)
    return msgs


def _go_event(tmp_path: Path, *, chat_id: str = GROUP_JID,
              kind: str = "group", quoted_kind: str = "image",
              quoted_path: Path | None = None) -> dict:
    """The Go bridge's message event for Finn's /sticker reply
    (post-fix shape: quoted media downloaded by the bridge)."""
    mime = {"image": "image/jpeg", "video": "video/mp4"}[quoted_kind]
    return {
        "type": "message", "id": "CMD1",
        "chat": {"id": chat_id, "kind": kind,
                 "title": "BOUESTI 2026/2027"},
        "from": {"id": FINN_JID, "name": "Finn"},
        "text": "/sticker",
        "media": [],
        "reply_to": "PHOTO1",
        "mentioned": False, "mentions": [],
        "quoted_text": "If you never enter school. No enter",
        "quoted_media": [{"path": str(quoted_path or ""),
                          "mime": mime, "kind": quoted_kind}],
        "view_once": False,
        "ts": int(time.time() * 1000),
    }


# ── WhatsApp adapter: the quoted payload lands on the message ──


def test_go_bridge_reply_to_photo_stamps_quoted_media(tmp_path):
    photo = _photo(tmp_path)
    msgs = _wa_deliver(_go_event(tmp_path, quoted_path=photo))
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg.reply_to == "PHOTO1"
    local = msg.meta.get("replied_media_local") or {}
    assert local.get("path") == str(photo)
    assert local.get("kind") == "image"
    assert (msg.meta.get("quoted_text") or "").startswith(
        "If you never enter school")
    assert msg.meta["envelope"]["quoted_chars"] > 0


def test_baileys_lid_reply_to_video_stamps_quoted_media(tmp_path):
    clip = tmp_path / "quoted_clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    event = _go_event(tmp_path, quoted_kind="video",
                      quoted_path=clip)
    # The Baileys rung's sender shape on an LID-addressed account:
    # from.id is the LID, the phone JID rides as "alt".
    event["from"] = {"id": FINN_LID, "name": "Finn",
                     "alt": FINN_JID}
    msgs = _wa_deliver(event)
    assert len(msgs) == 1
    msg = msgs[0]
    local = msg.meta.get("replied_media_local") or {}
    assert local.get("path") == str(clip)
    assert local.get("kind") == "video"
    assert msg.meta.get("sender_alt_id") == FINN_JID


def test_text_reply_stamps_no_media(tmp_path):
    event = _go_event(tmp_path)
    event["quoted_media"] = []
    msgs = _wa_deliver(event)
    assert len(msgs) == 1
    assert "replied_media_local" not in msgs[0].meta
    assert msgs[0].meta.get("quoted_text")


def test_plain_message_with_own_media_unchanged(tmp_path):
    photo = _photo(tmp_path)
    msgs = _wa_deliver({
        "type": "message", "id": "OWN1",
        "chat": {"id": DM_JID, "kind": "dm", "title": "Ada"},
        "from": {"id": DM_JID, "name": "Ada"},
        "text": "/sticker",
        "media": [{"path": str(photo), "mime": "image/jpeg",
                   "kind": "image"}],
        "reply_to": "", "mentioned": False,
        "ts": int(time.time() * 1000),
    })
    assert len(msgs) == 1
    assert msgs[0].first_media.path == str(photo)
    assert "replied_media_local" not in msgs[0].meta


def test_pathless_own_media_entry_leaves_no_phantom_ref(tmp_path):
    """Bridge download failure emits the media entry with an
    EMPTY path (the trace that media arrived, unseen) instead of
    omitting it.  The adapter must not mint a media ref out of
    thin air: no path, no MediaRef — the photofix half stamps
    such entries media_unavailable at the meta level."""
    msgs = _wa_deliver({
        "type": "message", "id": "FAIL1",
        "chat": {"id": DM_JID, "kind": "dm", "title": "Ada"},
        "from": {"id": DM_JID, "name": "Ada"},
        "text": "look at this",
        "media": [{"path": "", "mime": "image/jpeg",
                   "kind": "image"}],
        "reply_to": "", "mentioned": False,
        "ts": int(time.time() * 1000),
    })
    assert len(msgs) == 1
    assert msgs[0].media == []
    assert msgs[0].first_media is None


def test_wa_failed_own_photo_and_quoted_photo_compose(tmp_path):
    """Composition invariant (photofix x stickerfix on ONE
    message): the message's OWN photo failed to download at the
    bridge (pathless entry) while the QUOTED photo's bytes
    arrived fine.  meta must carry BOTH stamps —
    media_unavailable for the own photo (it arrived, unseen)
    and replied_media_local for the quoted one (its bytes are
    here for the reply-media commands)."""
    photo = _photo(tmp_path)
    event = _go_event(tmp_path, quoted_path=photo)
    event["media"] = [{"path": "", "mime": "image/jpeg",
                       "kind": "image"}]
    msgs = _wa_deliver(event)
    assert len(msgs) == 1
    msg = msgs[0]
    missed = msg.meta.get("media_unavailable") or []
    assert any(m.get("kind") == "image" for m in missed)
    local = msg.meta.get("replied_media_local") or {}
    assert local.get("path") == str(photo)
    assert local.get("kind") == "image"
    assert msg.media == []


# ── the runtime: /sticker and /convert on the delivered message ──


class _Adapter:
    def __init__(self):
        self.media_sent = []

    def send_media(self, chat, ref):
        self.media_sent.append(ref)
        return SimpleNamespace(
            ok=True, message_id="9", media_file_id="WA1", error="")


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters

    def send(self, platform, chat, text, **kw):
        return SimpleNamespace(ok=True, message_id="1")


class _Runtime(RuntimeExpressionMixin, RuntimeGroupMixin):
    def __init__(self, tmp_path, adapters):
        self.gateway = _Gateway(adapters)
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))

    def _send(self, chat, text, **kw):
        return SimpleNamespace(ok=True, message_id="1")


def _dm_sticker_msg(tmp_path: Path, quoted_path: Path,
                    quoted_kind: str = "image"):
    msgs = _wa_deliver(_go_event(
        tmp_path, chat_id=DM_JID, kind="dm",
        quoted_kind=quoted_kind, quoted_path=quoted_path))
    assert len(msgs) == 1
    return msgs[0]


def test_sticker_from_wa_replied_photo_end_to_end(tmp_path):
    photo = _photo(tmp_path)
    msg = _dm_sticker_msg(tmp_path, photo)
    assert msg.chat.kind == ChatKind.DM
    adapter = _Adapter()
    rt = _Runtime(tmp_path, {"whatsapp": adapter})
    out = rt._control_sticker("", msg)
    assert out == ""
    assert len(adapter.media_sent) == 1
    ref = adapter.media_sent[0]
    assert ref.kind == "sticker"
    produced = Path(ref.path)
    assert produced.exists() and produced.suffix == ".webp"
    with Image.open(produced) as im:
        assert max(im.size) == 512


@pytest.mark.skipif(shutil.which("ffmpeg") is None,
                    reason="ffmpeg not available")
def test_sticker_from_wa_replied_video_end_to_end(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         "testsrc=duration=1:size=320x240:rate=10",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True, capture_output=True)
    msg = _dm_sticker_msg(tmp_path, clip, quoted_kind="video")
    adapter = _Adapter()
    rt = _Runtime(tmp_path, {"whatsapp": adapter})
    out = rt._control_sticker("", msg)
    assert out == ""
    assert len(adapter.media_sent) == 1
    ref = adapter.media_sent[0]
    assert ref.kind == "sticker"
    assert Path(ref.path).exists()


def test_convert_from_wa_replied_photo(tmp_path):
    photo = _photo(tmp_path)
    msg = _dm_sticker_msg(tmp_path, photo)
    adapter = _Adapter()
    rt = _Runtime(tmp_path, {"whatsapp": adapter})
    out = rt._control_convert("image-to-sticker", msg)
    assert out == ""
    assert len(adapter.media_sent) == 1
    assert adapter.media_sent[0].kind == "sticker"


def test_resolver_serves_every_reply_media_command(tmp_path):
    """_replied_media_local is the ONE seam /steal, /sticker,
    /convert, /whatanime, /removebg and /qr read all resolve
    through — the WhatsApp quoted path must answer there."""
    photo = _photo(tmp_path)
    msg = _dm_sticker_msg(tmp_path, photo)
    rt = _Runtime(tmp_path, {"whatsapp": _Adapter()})
    found = rt._replied_media_local(msg)
    assert found is not None
    assert found[0] == str(photo)
    assert found[1] == "image"


def test_sticker_text_reply_still_usage(tmp_path):
    event = _go_event(tmp_path, chat_id=DM_JID, kind="dm")
    event["quoted_media"] = []
    msgs = _wa_deliver(event)
    rt = _Runtime(tmp_path, {"whatsapp": _Adapter()})
    out = rt._control_sticker("", msgs[0])
    assert out.startswith("usage: /sticker")


def test_sticker_no_reply_still_usage(tmp_path):
    msgs = _wa_deliver({
        "type": "message", "id": "PLAIN1",
        "chat": {"id": DM_JID, "kind": "dm", "title": "Ada"},
        "from": {"id": DM_JID, "name": "Ada"},
        "text": "/sticker", "media": [], "reply_to": "",
        "mentioned": False, "ts": int(time.time() * 1000),
    })
    rt = _Runtime(tmp_path, {"whatsapp": _Adapter()})
    out = rt._control_sticker("", msgs[0])
    assert out.startswith("usage: /sticker")


def test_sticker_failed_quoted_download_still_usage(tmp_path):
    """A quoted photo whose bridge download FAILED arrives as a
    pathless quoted_media entry: the bytes genuinely are not
    there, so /sticker keeps the honest usage fallback — no
    replied-media stamp, no fake sticker, no crash.  (_go_event
    with no quoted_path emits exactly the pathless shape.)"""
    event = _go_event(tmp_path, chat_id=DM_JID, kind="dm")
    msgs = _wa_deliver(event)
    assert len(msgs) == 1
    assert "replied_media_local" not in msgs[0].meta
    rt = _Runtime(tmp_path, {"whatsapp": _Adapter()})
    out = rt._control_sticker("", msgs[0])
    assert out.startswith("usage: /sticker")


# ── Telegram userbot: the replied message is downloaded at inbound ──


class _TgMessage:
    def __init__(self, **kw):
        self._reply = None
        self.__dict__.update(kw)

    async def get_reply_message(self):
        return self._reply


class _TgClient:
    def __init__(self, path):
        self.path = path

    async def download_media(self, message, file=None):
        return self.path


def test_userbot_replied_photo_is_downloaded(tmp_path):
    from nomorals.social.chat.telegram import TelegramAdapter

    photo = _photo(tmp_path)
    adapter = TelegramAdapter(
        api_id=123, api_hash="hash",
        session_path=str(tmp_path / "tg.session"),
        media_dir=str(tmp_path / "media"))
    user = SimpleNamespace(id=999, first_name="Ada", last_name="",
                           username="ada", bot=False, photo=None)
    replied = SimpleNamespace(
        id=41, media=SimpleNamespace(
            document=SimpleNamespace(
                size=1000, mime_type="image/jpeg", attributes=[])),
        voice=None)
    message = _TgMessage(
        id=42, raw_text="/sticker", media=None, voice=None,
        date=datetime.now(UTC),
        reply_to=SimpleNamespace(id=41, reply_to_msg_id=41),
        from_id=SimpleNamespace(user_id=999), peer_id=None,
        action=None)
    message._reply = replied
    event = SimpleNamespace(
        message=message, chat_id=999, sender_id=999,
        is_group=False, is_channel=False, chat=user, sender=user,
        input_chat=None, input_sender=None)
    got: list = []
    asyncio.new_event_loop().run_until_complete(
        adapter._handle_inbound(event, _TgClient(str(photo)),
                                got.append, allow_check=False))
    assert len(got) == 1
    local = got[0].meta.get("replied_media_local") or {}
    assert local.get("path") == str(photo)
    assert local.get("kind") == "image"
