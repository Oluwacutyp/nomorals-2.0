"""Tail-106 pins: nomorals/social/chat/discord.py census sites
(93, 111, 195, 358).

All four sites are held as EXCLUSIONS:

93 — ``_dc`` is an import probe for the optional discord.py
dependency: the import IS the check and None is the honest
negative. Every caller handles None explicitly (``send`` /
``send_media`` return ``SendResult(ok=False, error="discord.py is
not installed")``). Pinned with the module forced unimportable,
plus the cached-module control.

111 — ``_mentions_me``'s first guard: a mentions list whose
iteration raises falls through (``pass``) to the designed second
signal — the raw @username text check — which then decides. The
cascade is the feature (user clients often ship an empty/broken
mentions array); a miss degrades to False, never to a fabricated
mention. Recording asymmetry is pinned honestly: this guard does
NOT call ``record_failure`` while its sibling text guard DOES —
the pins assert both sides of that asymmetry rather than papering
over it (observability sketch territory, not a convert: the
verdict is identical either way).

195 — ``preflight`` abstains on a non-JWT token: the check raises
only on positive evidence (a decoded payload with ``bot: true``).
An undecodable token returns silently so ``connect()`` surfaces
the real error, per the printed contract ("not a JWT; let
connect() surface it"). Pinned: bot JWT raises, user JWT and
garbage tokens do not.

358 — ``_channel``'s DM cascade: ``fetch_channel`` raising falls
back to the user interpretation (``fetch_user`` + ``create_dm``)
for refs built by hand from a user id — the module header comment
documents exactly this. The fallback is not a swallow: if the id
is neither a channel nor a user, the second lookup raises and
``send`` reports ``ok=False`` with that error. Pinned both ways.
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

import nomorals.social.chat.discord as dc_mod
from nomorals.core.errors import ValidationError
from nomorals.social.chat.base import ChatKind, ChatRef
from nomorals.social.chat.discord import DiscordAdapter


def _adapter(tmp_path, token="aaa.bbb.ccc"):
    return DiscordAdapter(token=token, media_dir=str(tmp_path))


def _jwt(payload: dict) -> str:
    body = base64.urlsafe_b64encode(
        json.dumps(payload).encode()).decode().rstrip("=")
    return f"x.{body}.y"


# ── 93: _dc import probe ─────────────────────────────────────────

def test_93_missing_dependency_returns_none(tmp_path):
    adapter = _adapter(tmp_path)
    assert adapter._discord_mod is None
    with mock.patch.dict(sys.modules, {"discord": None}):
        assert adapter._dc() is None
    assert adapter._discord_mod is None  # failure is not cached


def test_93_cached_module_is_returned_without_import(tmp_path):
    adapter = _adapter(tmp_path)
    sentinel = SimpleNamespace(name="fake-discord")
    adapter._discord_mod = sentinel
    assert adapter._dc() is sentinel


# ── 111: mentions guard falls through to the text signal ─────────

class _RaisingMentions:
    def __iter__(self):
        raise RuntimeError("mentions boom")


def test_111_unreadable_mentions_fall_through_to_text(tmp_path):
    adapter = _adapter(tmp_path)
    adapter._me = SimpleNamespace(id=42, name="death")
    calls: list = []
    with mock.patch.object(dc_mod, "record_failure",
                           lambda *a, **k: calls.append(a)):
        hit = SimpleNamespace(mentions=_RaisingMentions(),
                              content="hey @death hi")
        miss = SimpleNamespace(mentions=_RaisingMentions(),
                               content="nothing here")
        assert adapter._mentions_me(hit) is True   # text signal decides
        assert adapter._mentions_me(miss) is False  # never a fabricated hit
    assert calls == []  # this guard records nothing (pinned asymmetry)


def test_111_sibling_text_guard_does_record(tmp_path):
    adapter = _adapter(tmp_path)
    adapter._me = SimpleNamespace(id=42, name="death")

    class _Msg:
        mentions = ()

        @property
        def content(self):
            raise RuntimeError("content boom")

    calls: list = []
    with mock.patch.object(dc_mod, "record_failure",
                           lambda *a, **k: calls.append(a)):
        assert adapter._mentions_me(_Msg()) is False
    assert len(calls) == 1  # the sibling guard records; 111 does not


# ── 195: preflight abstains unless the bot evidence is positive ──

def test_195_non_jwt_token_abstains(tmp_path):
    adapter = _adapter(tmp_path, token="not-a-jwt")
    assert adapter.preflight() is None  # connect() will surface it


def test_195_user_jwt_passes(tmp_path):
    adapter = _adapter(tmp_path, token=_jwt({"id": "123"}))
    assert adapter.preflight() is None


def test_195_bot_jwt_raises(tmp_path):
    adapter = _adapter(tmp_path, token=_jwt({"bot": True}))
    with pytest.raises(ValidationError):
        adapter.preflight()


# ── 358: DM channel cascade — channel id first, user id fallback ─

class _FakeDM:
    id = 999


class _FakeUser:
    async def create_dm(self):
        return _FakeDM()


def test_358_channel_miss_falls_back_to_user_dm(tmp_path):
    adapter = _adapter(tmp_path)

    class _Bot:
        async def fetch_channel(self, cid):
            raise RuntimeError("unknown channel")

        async def fetch_user(self, uid):
            return _FakeUser()

    adapter._bot = _Bot()
    ref = ChatRef(platform="discord", chat_id="123", kind=ChatKind.DM)
    assert asyncio.run(adapter._channel(ref)).id == 999


def test_358_channel_hit_never_touches_user_lookup(tmp_path):
    adapter = _adapter(tmp_path)
    channel = SimpleNamespace(id=555)

    class _Bot:
        async def fetch_channel(self, cid):
            return channel

        async def fetch_user(self, uid):
            raise AssertionError("user lookup must not run")

    adapter._bot = _Bot()
    ref = ChatRef(platform="discord", chat_id="123", kind=ChatKind.DM)
    assert asyncio.run(adapter._channel(ref)).id == 555
