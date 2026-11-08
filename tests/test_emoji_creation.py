"""/emoji creation rung (owner ask, 2026-11-06): the command
was lookup-only — '/emoji family rocket' handed back two
separate Unicode emoji and a word list, because the table
holds the parts and never the whole.  Now: confident lookups
answer exactly as before (pins), '/emoji make <words>'
creates, and a search no single emoji confidently covers
creates too — generated through the image pipeline's rung,
post-processed into a real sticker, sent through the maker's
own seam.  No rung configured earns one honest line, never a
fabrication and never a near-match passed off as the answer.

EXPRESSION COMPLETION (owner correction, 2026-11-07 —
"Not stickers. Emoji is different from stickers!!!"): the
2026-11-06 shape above was itself the mislabel.  /emoji make
now authors a REAL custom emoji (her studio's 100×100 art,
registered into her Telegram pack where the surface allows,
living in /customemoji); the sticker maker survives under
its honest name (/emoji sticker), and create_emoji_sticker
gained her own local renderer as its floor rung — no
configured backend no longer means no creation.  The pins
below were updated to the corrected contract, with causes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from nomorals.agents.partner.runtime_expression import (
    RuntimeExpressionMixin,
)
from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.expression import (
    EmojiCreateUnavailable,
    confident_emoji_match,
    create_emoji_sticker,
    emoji_prompt_for,
)

BOT_KEY = "telegram-bot:-100123"


class _Chat:
    def __init__(self, platform, key, title="Builders"):
        self.platform = platform
        self.key = key
        self.chat_id = key.split(":", 1)[1]
        self.title = title
        self.kind = ChatKind.GROUP


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
    def __init__(self, tmp_path, admin_ids=(),
                 bot_status="administrator"):
        self.media_sent = []
        self.admin_ids = set(admin_ids)
        self._bot_id = 4242
        self.bot_status = bot_status
        self.tmp_path = tmp_path

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


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent = []

    def send(self, platform, chat, text, **kw):
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, message_id="1")


class _Runtime(RuntimeExpressionMixin, RuntimeGroupMixin):
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


def _rig(tmp_path, platform="telegram-bot", key=BOT_KEY, **kw):
    adapter = _Adapter(tmp_path, **kw)
    rt = _Runtime(tmp_path, {platform: adapter}, operators={"99"})
    return rt, adapter, _Chat(platform, key)


class _FakeBackend:
    """The image rung, faked: records the prompt, returns one
    deliberately non-square picture."""

    name = "fake"

    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        return [Image.new("RGB", (800, 600), (200, 40, 40))]


@pytest.fixture
def fake_backend(monkeypatch):
    fake = _FakeBackend()
    monkeypatch.setattr(
        "nomorals.media_edit.generate.get_backend",
        lambda *a, **k: fake)
    return fake


@pytest.fixture
def no_backend(monkeypatch):
    from nomorals.media_edit.generate import GenerativeEditError

    def _raise(*a, **k):
        raise GenerativeEditError(
            "no generative backend is configured (fake)",
            kind="unavailable")

    monkeypatch.setattr(
        "nomorals.media_edit.generate.get_backend", _raise)


# ── the lookup pins (unchanged behavior, both trees) ─────────


def test_lookup_pins_unchanged(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    out = rt._control_emoji("melting face", _Msg(chat))
    assert "🫠" in str(out)
    out = rt._control_emoji("👨‍👩‍👧", _Msg(chat))
    assert "family" in str(out)
    out = rt._control_emoji("rocket", _Msg(chat))
    assert "🚀" in str(out)
    assert adapter.media_sent == []  # lookups never send media


def test_confidence_covers_whole_queries_only():
    assert confident_emoji_match("melting face") is not None
    assert confident_emoji_match("melting face").char == "🫠"
    assert confident_emoji_match("thumbs up medium") is not None
    # the owner's live test: parts exist, the whole doesn't
    assert confident_emoji_match("family rocket") is None
    assert confident_emoji_match("") is None


def test_prompt_is_emoji_art():
    prompt = emoji_prompt_for("a cat riding a rocket")
    assert "a cat riding a rocket" in prompt
    assert "emoji" in prompt and "no text" in prompt


# ── the creation core ────────────────────────────────────────


def test_create_core_makes_a_sticker_shaped_artifact(tmp_path):
    fake = _FakeBackend()
    path = create_emoji_sticker(
        "a cat riding a rocket", tmp_path, backend=fake)
    assert path.exists() and path.suffix == ".webp"
    with Image.open(path) as img:
        assert img.size == (512, 512)
        assert img.mode == "RGBA"
    assert len(fake.prompts) == 1
    prompt = fake.prompts[0]
    assert "a cat riding a rocket" in prompt
    assert "emoji" in prompt and "outline" in prompt


def test_create_core_without_a_backend_uses_her_own_hand(
        tmp_path, no_backend):
    # Expression completion (2026-11-07): no configured image
    # backend no longer means no creation — her local studio
    # renderer is the floor rung and the sticker still lands.
    path = create_emoji_sticker("a gold heart", tmp_path)
    assert path.exists() and path.suffix == ".webp"
    with Image.open(path) as img:
        assert img.size == (512, 512)


def test_create_core_every_rung_down_is_honest(
        tmp_path, no_backend, monkeypatch):
    # The honest line survives for the case where EVERY rung
    # fails — backend unconfigured AND her renderer broken.
    from nomorals.social.chat import emoji_studio

    def _boom(*a, **k):
        raise RuntimeError("renderer exploded")

    monkeypatch.setattr(emoji_studio, "sticker_master_png",
                        _boom)
    with pytest.raises(EmojiCreateUnavailable) as caught:
        create_emoji_sticker("anything", tmp_path)
    assert "no generative backend" in str(caught.value)
    assert "renderer exploded" in str(caught.value)


# ── the command ──────────────────────────────────────────────


def test_make_creates_a_real_custom_emoji(tmp_path, fake_backend):
    # Expression completion (2026-11-07): /emoji make authors
    # a REAL custom emoji — her studio's 100×100 art, joining
    # her /customemoji set.  This rig's adapter carries no
    # pack verb, so no registration happens here and the art
    # travels as a sticker WITH its honest note (the pack
    # path itself is pinned in test_expression_completion).
    rt, adapter, chat = _rig(tmp_path)
    out = str(rt._control_emoji("make a cat riding a rocket",
                                _Msg(chat)))
    registry = rt._expression_library().custom_emoji
    names = registry.names()
    assert len(names) == 1
    entry = registry.get(names[0])
    assert entry["unicode"]  # a fallback face exists
    assert entry["image_path"].endswith(".png")
    with Image.open(entry["image_path"]) as img:
        assert img.size == (100, 100)  # custom-emoji format
    assert len(adapter.media_sent) == 1
    assert adapter.media_sent[0].kind == "sticker"
    assert "custom set" in out
    # the image backend is NOT the author of a custom emoji —
    # her studio is (the backend ladder serves /emoji sticker)
    assert fake_backend.prompts == []


def test_sticker_head_makes_a_sticker(tmp_path, fake_backend):
    # The maker under its honest name: /emoji sticker goes
    # through the image rung and sends a real sticker; the
    # sticker WAS the reply ("").
    rt, adapter, chat = _rig(tmp_path)
    out = rt._control_emoji("sticker a cat riding a rocket",
                            _Msg(chat))
    assert out == ""
    assert len(adapter.media_sent) == 1
    ref = adapter.media_sent[0]
    assert ref.kind == "sticker"
    assert ref.path.endswith(".webp")
    assert fake_backend.prompts  # the rung was actually asked
    assert "a cat riding a rocket" in fake_backend.prompts[0]


def test_uncovered_search_creates_instead_of_dumping(
        tmp_path, fake_backend):
    # the owner's exact live test: '/emoji family rocket' —
    # still a creation, never a dump; now a REAL custom emoji
    # (see test_make_creates_a_real_custom_emoji).
    rt, adapter, chat = _rig(tmp_path)
    out = str(rt._control_emoji("family rocket", _Msg(chat)))
    registry = rt._expression_library().custom_emoji
    assert len(registry.names()) == 1
    assert len(adapter.media_sent) == 1
    assert adapter.media_sent[0].kind == "sticker"
    assert "custom set" in out


def test_create_grammar_needs_a_description(tmp_path, fake_backend):
    rt, adapter, chat = _rig(tmp_path)
    out = rt._control_emoji("make", _Msg(chat))
    assert "usage:" in str(out)
    assert adapter.media_sent == []


def test_no_backend_configured_still_creates(
        tmp_path, no_backend):
    # The EC2 case that was dead (2026-11-07): no image
    # backend configured — creation used to be one honest
    # refusal.  Her own renderer is the floor now: the emoji
    # is created for real and joins her set.
    rt, adapter, chat = _rig(tmp_path)
    out = str(rt._control_emoji(
        "make a dragon made of lightning", _Msg(chat)))
    registry = rt._expression_library().custom_emoji
    assert len(registry.names()) == 1
    assert len(adapter.media_sent) == 1
    assert "custom set" in out


def test_creation_failure_is_one_honest_line(
        tmp_path, monkeypatch):
    # When even her renderer fails, the refusal is one plain
    # journaled line — never a fabrication.
    from nomorals.agents.partner.failures import recent_failures
    from nomorals.social.chat import emoji_studio

    def _boom(*a, **k):
        raise RuntimeError("renderer exploded")

    monkeypatch.setattr(emoji_studio, "render_master", _boom)
    rt, adapter, chat = _rig(tmp_path)
    out = str(rt._control_emoji("make a dragon", _Msg(chat)))
    assert "my own renderer failed" in out
    assert adapter.media_sent == []
    sites = [f.site for f in recent_failures(200)]
    assert "expression.emoji_create.render" in sites


def test_creation_respects_the_group_gate(tmp_path, fake_backend):
    # an unclaimed room (userbot surface) gets no creation post
    rt, adapter, chat = _rig(tmp_path, platform="telegram",
                             key="telegram:-100999")
    out = rt._control_emoji("make a moon fox", _Msg(chat))
    assert "isn't claimed" in str(out)
    assert adapter.media_sent == []
    assert fake_backend.prompts == []  # gate first: no spend


# ── the brain's seam: the registry the tool loop lists ───────


def test_emoji_create_registered_for_the_brain(
        tmp_path, fake_backend):
    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry(
        context=SimpleNamespace(workspace=str(tmp_path)))
    registry.register_builtins()
    assert "emoji_create" in registry._tools
    assert "emoji_create" in registry.prompt_listing()
    tool = registry._tools["emoji_create"]
    result = tool.fn("a snail with a jetpack")
    assert result["kind"] == "sticker"
    path = result["path"]
    assert path.endswith(".webp")
    with Image.open(path) as img:
        assert img.size == (512, 512)
    assert "a snail with a jetpack" in fake_backend.prompts[0]
