"""Moderation theatre GIFs (gapfix167): the /mute, /kick, /ban
reply used the same studio render every time — one
deterministic file per moment, shelved once and picked
forever. Now a moment searches Tenor/GIPHY with a phrase
proven against Tenor's live index ("shush", "kick out",
"ban hammer", ...), the shelf grows to three distinct GIFs
before the provider stops being asked, and the no-key floor
renders three palette variants of the studio animation
instead of one file. Transport is faked in every test: no
live HTTP anywhere. The caption contract is untouched — the
text still rides as the animation's caption, exactly as the
/admin theatre path sends it.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat import expression as expr
from nomorals.social.chat.expression import (
    ExpressionLibrary,
    ExpressionService,
    provider_query_for,
)


def _ok_send(captured):
    def send(ref):
        captured.append(ref)
        return SimpleNamespace(ok=True, message_id="1",
                               media_file_id="", error="")
    return send


class _Tenor:
    """Fake Tenor: records the query, returns four distinct
    URLs every time (Tenor search is deterministic per query,
    so the same four come back on every fire — the rotation
    has to come from the shelf, not from the provider)."""
    name = "tenor"

    def __init__(self):
        self.queries = []

    def search(self, query, limit=4):
        self.queries.append(query)
        slug = query.replace(" ", "-")
        return [{"url": f"https://tenor.test/{slug}-{i}.gif",
                 "title": f"{query} {i}", "kind": "gif"}
                for i in range(4)]


class _Dead:
    """A provider that answers nothing, but records being asked."""
    name = "dead"

    def __init__(self):
        self.queries = []

    def search(self, query, limit=4):
        self.queries.append(query)
        return []


def _fake_download(url, dest_dir, **kw):
    name = url.rsplit("/", 1)[-1]
    dest = Path(dest_dir) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"GIF89a:" + name.encode())
    return dest


def _service(tmp_path, providers, rng=None):
    lib = ExpressionLibrary(tmp_path / "lib.json")
    svc = ExpressionService(
        lib, providers=list(providers), media_dir=tmp_path / "media",
        cooldown_s=0, rng=rng or random.Random(0))
    return svc, lib


def _fire_mute(svc, room="r1"):
    return svc.fire("mute", room=room, platform="telegram-bot",
                    settings={}, send=_ok_send([]))


# ── the query table ──────────────────────────────────────────


def test_provider_query_table_is_the_proven_phrases():
    assert provider_query_for("mute", ("mute", "quiet")) == "shush"
    assert provider_query_for("kick", ("kick", "boot")) == "kick out"
    assert provider_query_for("ban", ("ban", "hammer")) == "ban hammer"
    assert provider_query_for("unmute", ("unmute", "speak")) == "speak up"
    assert provider_query_for("unban", ("unban", "return")) \
        == "welcome back"
    assert provider_query_for("warn", ("warn", "warning")) == "watch out"
    assert provider_query_for("slowmode", ("slowmode", "slow")) \
        == "slow sloth"
    # No override: the bare tag join stands — no phrase is
    # claimed without a live index behind it.
    assert provider_query_for("restrict", ("restrict", "locked")) \
        == "restrict locked"
    assert provider_query_for("purge", ("purge", "delete")) \
        == "purge delete"
    assert provider_query_for("unknown", ("foo", "bar")) == "foo bar"


# ── the Tenor rung ───────────────────────────────────────────


def test_tenor_rung_searches_the_proven_query(tmp_path, monkeypatch):
    tenor = _Tenor()
    svc, lib = _service(tmp_path, [tenor])
    monkeypatch.setattr(expr, "download_to_library", _fake_download)
    out = _fire_mute(svc)
    assert out["sent"] is True
    assert tenor.queries == ["shush"]
    assert lib.items[0]["source"] == "provider:tenor"
    assert lib.items[0]["origin"] == "https://tenor.test/shush-3.gif"


def test_kick_and_ban_use_their_queries(tmp_path, monkeypatch):
    tenor = _Tenor()
    svc, _lib = _service(tmp_path, [tenor])
    monkeypatch.setattr(expr, "download_to_library", _fake_download)
    svc.fire("kick", room="r1", platform="telegram-bot",
             settings={}, send=_ok_send([]))
    svc.fire("ban", room="r1", platform="telegram-bot",
             settings={}, send=_ok_send([]))
    assert tenor.queries == ["kick out", "ban hammer"]


def test_tenor_rotation_no_immediate_repeat(tmp_path, monkeypatch):
    tenor = _Tenor()
    svc, lib = _service(tmp_path, [tenor], rng=random.Random(0))
    monkeypatch.setattr(expr, "download_to_library", _fake_download)
    seen = []
    for _ in range(3):
        out = _fire_mute(svc)
        assert out["sent"] is True
        seen.append(out["item_id"])
    assert len(set(seen)) == 3
    assert len(lib.items) == 3
    assert all(item["source"] == "provider:tenor"
               for item in lib.items)


# ── the no-key floor ─────────────────────────────────────────


def test_studio_floor_rotates_without_a_key(tmp_path):
    svc, lib = _service(tmp_path, [], rng=random.Random(0))
    seen = []
    for _ in range(3):
        out = _fire_mute(svc)
        assert out["sent"] is True
        seen.append(out["item_id"])
    assert len(set(seen)) == 3
    assert len(lib.items) == 3
    assert all(item["source"] == "self:render" for item in lib.items)
    assert all(item["kind"] == "animation" for item in lib.items)


def test_studio_floor_grows_an_old_single_render(tmp_path):
    # His library already holds the old single moment_mute.gif
    # from before this ship: the first fire must grow the shelf
    # to the variants instead of serving that one file forever.
    lib = ExpressionLibrary(tmp_path / "lib.json")
    media = tmp_path / "media"
    media.mkdir(exist_ok=True)
    old = media / "moment_mute.gif"
    old.write_bytes(b"GIF89a old")
    lib.add_item(kind="animation", path=str(old),
                 tags=["mute", "quiet", "silence", "shush"],
                 source="self:render")
    svc = ExpressionService(lib, providers=[], media_dir=media,
                            cooldown_s=0, rng=random.Random(0))
    seen = []
    for _ in range(3):
        out = _fire_mute(svc)
        assert out["sent"] is True
        seen.append(out["item_id"])
    assert len(set(seen)) == 3
    assert len(lib.items) == 4  # the old file + three variants


# ── degradation ──────────────────────────────────────────────


def test_studio_floor_never_dilutes_a_real_gif(tmp_path):
    # A shelf holding one real GIF (stocked, stolen, provider)
    # keeps serving it: the studio floor grows only its own
    # output. Pinned by the two test_group_expression file_id
    # tests, which the first version of this growth broke —
    # the re-pick after growing could send a render instead of
    # the stocked item, and its file_id was never captured.
    lib = ExpressionLibrary(tmp_path / "lib.json")
    media = tmp_path / "media"
    media.mkdir(exist_ok=True)
    real = media / "welcome.gif"
    real.write_bytes(b"GIF89a real")
    item = lib.add_item(kind="gif", path=str(real),
                        tags=["welcome", "wave", "hello", "hi"],
                        source="test")
    svc = ExpressionService(lib, providers=[], media_dir=media,
                            cooldown_s=0, rng=random.Random(0))
    seen = []
    for _ in range(3):
        out = svc.fire("welcome", room="r1",
                       platform="telegram-bot", settings={},
                       send=_ok_send([]))
        assert out["sent"] is True
        seen.append(out["item_id"])
    assert seen == [item["id"]] * 3
    assert len(lib.items) == 1


def test_provider_failure_still_lands_the_studio_render(tmp_path):
    dead = _Dead()
    svc, lib = _service(tmp_path, [dead])
    out = _fire_mute(svc)
    assert out["sent"] is True
    assert dead.queries == ["shush"]  # asked with the proven
    # phrase, answered nothing, fell back to her own hand
    assert lib.items[0]["source"] == "self:render"


def test_gif_command_query_untouched(tmp_path, monkeypatch):
    # The moment overrides must not leak into /gif: it searches
    # the ask's own words.
    tenor = _Tenor()
    svc, _lib = _service(tmp_path, [tenor])
    monkeypatch.setattr(expr, "download_to_library", _fake_download)
    out = svc.gif_for("kick boot", room="r1",
                      platform="telegram-bot", send=_ok_send([]))
    assert out["sent"] is True
    assert tenor.queries == ["kick boot"]
