"""Expression repair F2 (owner, 2026-11-11) — the media floor.

From his own screen on current main: "/gif menu carries only
anime images; mute ban and the rest got no stickers nor gifs
not even images."  The audit (~/workspace/expression-audit)
traced why; this suite pins the repairs:

- /gif has the floor the moments always had: library → the
  provider chain → the unified pool's "gif" kind → her own
  studio, which renders a REAL animated GIF for the ask.
- The typed moderation commands ride the moment path: /ban
  fires the same ban moment the automod path fires, /mute
  and /kick have moments of their own, and /warn carries one
  too — the "warnings stay dry" line traced to pass-era
  text, never to an owner order (§31 quotes him only that
  commiseration on bans stays text; his 2026-11-11 words are
  the latest and they win).
- Sources are no longer frozen at code-time: a persisted
  registry in her data dir joins the pool walks and the
  expression chain as ONE list with the seeds and the
  settings hook; a registered source that keeps failing is
  journaled, stood down, then disabled — skipped, never
  retried forever.
- Reactions keep their studio floor, and the general chain
  gains a verified keyless seed (cataas tag search — only
  rows the catalogue marks image/gif are ever offered).

Network-free: every transport/fetcher is a fixture with the
real APIs' verified shapes.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.anime.registered_sources import SourceRegistry
from nomorals.anime.sources import (
    IMAGE_POOLS,
    ImagePoolClient,
    SourceFailure,
)
from nomorals.anime.transport import HttpResult, TransportError
from nomorals.social.chat import emoji_studio
from nomorals.social.chat.expression import (
    MOMENT_TAGS,
    CataasSearchProvider,
    ExpressionLibrary,
    ExpressionService,
    RegisteredSearchProvider,
    WaifuPicsProvider,
    default_providers,
)

GIF_BYTES = b"GIF89a" + b"\x01\x00\x01\x00" + b"\x00" * 64


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


# ── /gif: the floor ───────────────────────────────────────────


def test_gif_for_self_creates_when_chain_and_library_empty(
        tmp_path, monkeypatch):
    """Empty library, empty chain, no pool: she renders the
    ask herself — the same studio generator the moments use —
    and what is sent is a real animated GIF file."""
    calls = []

    def fake_generator(mood, out_dir, *, name="moment"):
        calls.append((mood, name))
        path = Path(out_dir) / f"{name}.gif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(GIF_BYTES)
        return path

    monkeypatch.setattr(emoji_studio, "mood_animation",
                        fake_generator)
    svc, lib = _svc(tmp_path, providers=[],
                    media_dir=tmp_path / "m")
    sent: list = []
    out = svc.gif_for("underwater basketweaving", room="r",
                      platform="whatsapp", send=_ok_send(sent))
    assert out["sent"] is True
    assert calls, "the studio generator was never reached"
    assert sent[0].kind == "animation"  # delivered AS animation
    item = lib.items[0]
    assert item["source"] == "self:render"
    assert set(item["tags"]) == {"underwater", "basketweaving"}
    assert Path(item["path"]).read_bytes()[:6] == b"GIF89a"


def test_gif_for_self_create_is_a_real_gif_with_the_real_studio(
        tmp_path):
    """Unpatched: the studio's own render is a multi-frame
    GIF on disk, not a still wearing a .gif name."""
    from PIL import Image

    svc, lib = _svc(tmp_path, providers=[],
                    media_dir=tmp_path / "m")
    out = svc.gif_for("celebration", room="r",
                      platform="whatsapp", send=_ok_send([]))
    assert out["sent"] is True
    with Image.open(lib.items[0]["path"]) as img:
        assert img.format == "GIF"
        assert getattr(img, "n_frames", 1) > 1


def test_gif_for_prefers_library_over_creating(tmp_path,
                                               monkeypatch):
    svc, lib = _svc(tmp_path, providers=[],
                    media_dir=tmp_path / "m")
    stocked = tmp_path / "shelf.gif"
    stocked.write_bytes(GIF_BYTES)
    lib.add_item(kind="gif", path=str(stocked), tags=["dance"],
                 source="test")

    def forbidden(*a, **kw):
        raise AssertionError("created despite a library hit")

    monkeypatch.setattr(emoji_studio, "mood_animation",
                        forbidden)
    sent: list = []
    out = svc.gif_for("dance", room="r", platform="whatsapp",
                      send=_ok_send(sent))
    assert out["sent"] is True
    assert sent[0].path == str(stocked)


def test_gif_for_prefers_pool_hit_over_creating(tmp_path,
                                                monkeypatch):
    svc, lib = _svc(tmp_path, providers=[],
                    media_dir=tmp_path / "m")
    pool_gif = tmp_path / "pool.gif"
    pool_gif.write_bytes(GIF_BYTES)

    class FakePool:
        def fetch_media(self, kind, rng=None):
            assert kind == "gif"
            return SimpleNamespace(
                url="", source="her own studio",
                local_path=str(pool_gif))

    svc.pool_client = FakePool()

    def forbidden(*a, **kw):
        raise AssertionError("created despite a pool hit")

    monkeypatch.setattr(emoji_studio, "mood_animation",
                        forbidden)
    sent: list = []
    out = svc.gif_for("dogs", room="r", platform="whatsapp",
                      send=_ok_send(sent))
    assert out["sent"] is True
    assert sent[0].path == str(pool_gif)
    assert lib.items[0]["source"] == "self:render"


def test_gif_for_prefers_provider_chain_over_pool_and_studio(
        tmp_path, monkeypatch):
    svc, lib = _svc(tmp_path, media_dir=tmp_path / "m")

    class ChainHit:
        name = "chainhit"

        def search(self, query, *, limit=4):
            return [{"url": "https://x.test/party.gif",
                     "title": "party", "kind": "gif"}]

    svc.providers = [ChainHit()]

    def fake_download(url, dest_dir):
        dest = Path(dest_dir) / "dl.gif"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(GIF_BYTES)
        return dest

    monkeypatch.setattr(
        "nomorals.social.chat.expression.download_to_library",
        fake_download)

    def forbidden(*a, **kw):
        raise AssertionError("a lower rung was reached")

    monkeypatch.setattr(emoji_studio, "mood_animation",
                        forbidden)
    sent: list = []
    out = svc.gif_for("party", room="r", platform="whatsapp",
                      send=_ok_send(sent))
    assert out["sent"] is True
    assert lib.items[0]["source"] == "provider:chainhit"


def test_gif_for_stays_honest_with_no_media_home(tmp_path):
    svc, _lib = _svc(tmp_path, providers=[], media_dir=None)
    out = svc.gif_for("anything", room="r", platform="whatsapp",
                      send=_ok_send([]))
    assert out == {"sent": False,
                   "reason": "nothing in my library for that yet"}


# ── admin theatre: the moments exist and fire ─────────────────


def test_moderation_moments_exist():
    for moment in ("mute", "kick", "warn", "ban"):
        assert moment in MOMENT_TAGS, moment


def test_mute_kick_warn_moments_fire_from_the_library(tmp_path):
    svc, lib = _svc(tmp_path, providers=[],
                    media_dir=tmp_path / "m")
    for i, moment in enumerate(("mute", "kick", "warn")):
        p = tmp_path / f"{moment}.gif"
        p.write_bytes(GIF_BYTES)
        lib.add_item(kind="gif", path=str(p),
                     tags=[MOMENT_TAGS[moment][0][0]],
                     source="test")
        captured: list = []
        out = svc.fire(moment, room=f"r{i}", platform="discord",
                       settings={}, send=_ok_send(captured))
        assert out["sent"] is True, (moment, out)
        assert captured


# ── the typed commands fire them (handler wiring) ─────────────


def _bare_mod_rig(tmp_path, *, refuse=False):
    from nomorals.agents.partner.runtime_group import (
        RuntimeGroupMixin,
    )
    from nomorals.social.chat.group_surface import (
        GroupStore,
        GroupSurface,
    )

    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    calls: list = []

    class _Adapter:
        def admin_member(self, chat, user_id):
            return {"status": "administrator"}

        def admin_restrict(self, chat, user_id, *, minutes=60):
            calls.append(("restrict", user_id))
            return {"ok": not refuse}

        def admin_unrestrict(self, chat, user_id):
            calls.append(("unrestrict", user_id))
            return {"ok": not refuse}

        def admin_ban(self, chat, user_id):
            calls.append(("ban", user_id))
            return {"ok": not refuse,
                    "error": "nope" if refuse else ""}

        def admin_unban(self, chat, user_id):
            calls.append(("unban", user_id))
            return {"ok": not refuse}

    adapter = _Adapter()
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore[method-assign]
    mixin._group_adapter = lambda platform: adapter  # type: ignore[method-assign]
    fired: list = []
    mixin._fire_expression = (  # type: ignore[method-assign]
        lambda message, moment: fired.append(moment))
    surface.handle_message("telegram-bot:-1001", "777",
                           username="loud", text="hi")
    msg = SimpleNamespace(
        chat=SimpleNamespace(key="telegram-bot:-1001",
                             platform="telegram-bot",
                             kind="group", title="Builders"),
        sender_id="admin1", sender="Admin",
        sender_username="boss", text="", meta={}, reply_to="",
        message_id="900")
    return mixin, fired, calls, msg


def test_typed_ban_fires_the_ban_moment(tmp_path):
    mixin, fired, _calls, msg = _bare_mod_rig(tmp_path)
    out = mixin._control_ban("@loud", msg)
    assert "is banned" in out
    assert fired == ["ban"]


def test_typed_ban_refusal_fires_nothing(tmp_path):
    mixin, fired, _calls, msg = _bare_mod_rig(tmp_path,
                                              refuse=True)
    out = mixin._control_ban("@loud", msg)
    assert "didn't go through" in out
    assert fired == []


def test_typed_mute_fires_the_mute_moment(tmp_path):
    mixin, fired, _calls, msg = _bare_mod_rig(tmp_path)
    out = mixin._control_mute("@loud 10m", msg)
    assert "muted" in out
    assert fired == ["mute"]


def test_typed_kick_fires_the_kick_moment(tmp_path):
    mixin, fired, calls, msg = _bare_mod_rig(tmp_path)
    out = mixin._control_kick("@loud", msg)
    assert "is out" in out
    assert [c[0] for c in calls] == ["ban", "unban"]
    assert fired == ["kick"]


def test_typed_warn_fires_the_warn_moment(tmp_path):
    mixin, fired, _calls, msg = _bare_mod_rig(tmp_path)
    out = mixin._control_warn("@loud caps", msg)
    assert "1/3" in out
    assert fired == ["warn"]


# ── the composed runtime: media actually lands ────────────────


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


class _FullAdapter:
    def __init__(self):
        self.calls: list = []
        self.media_sent: list = []
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

    def send_media(self, chat, ref):
        self.media_sent.append(ref)
        return SimpleNamespace(ok=True, message_id="9",
                               media_file_id="F1", error="")

    def react(self, chat, message_id, emoji):
        return SimpleNamespace(ok=True, message_id=message_id,
                               error="")


def _full_rig(tmp_path):
    from nomorals.agents.partner.runtime_expression import (
        RuntimeExpressionMixin,
    )
    from nomorals.agents.partner.runtime_group import (
        RuntimeGroupMixin,
    )

    adapter = _FullAdapter()

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
        "telegram-bot:-100123", "777", username="loud",
        text="hi")
    return rt, adapter


@pytest.mark.parametrize("verb,call", [
    ("ban", lambda rt: rt._control_ban("@loud", _Msg())),
    ("mute", lambda rt: rt._control_mute("@loud 10m", _Msg())),
    ("kick", lambda rt: rt._control_kick("@loud", _Msg())),
    ("warn", lambda rt: rt._control_warn("@loud caps", _Msg())),
])
def test_typed_admin_commands_carry_media(tmp_path, verb, call):
    """Empty library, no keys: the studio floor still puts a
    sticker/GIF on the action — the owner's exact complaint,
    closed at the composed seam."""
    rt, adapter = _full_rig(tmp_path)
    out = call(rt)
    assert out  # the text reply still lands
    assert len(adapter.media_sent) == 1, verb
    assert adapter.media_sent[0].kind in ("sticker", "animation",
                                          "gif")


def test_automod_ban_moment_still_fires(tmp_path):
    """The sensor path that always had theatre keeps it —
    unchanged by the command-path wiring."""
    rt, adapter = _full_rig(tmp_path)
    rt._send_action_notice(
        rt._group_surface(), "telegram-bot:-100123", _Msg(),
        {"kind": "ban", "warning": "🔨 banned by the filter"})
    assert len(adapter.media_sent) == 1
    assert rt.sent  # the notice text posted too


# ── the source registry ───────────────────────────────────────


class _Transport:
    def __init__(self, routes):
        self.routes = routes
        self.calls: list[str] = []

    def request(self, method, url, params=None, timeout=None,
                **kw):
        self.calls.append(url)
        for needle, answer in self.routes:
            if needle in url:
                if isinstance(answer, Exception):
                    raise answer
                return answer
        raise TransportError("no route")


def _json(data):
    return HttpResult(200, data=data)


def test_registered_source_joins_the_next_pool_walk(tmp_path):
    reg = SourceRegistry(tmp_path / "reg.json")
    reg.register({"name": "my-gifs",
                  "url": "https://gifs.example.test/api",
                  "shape": "results-url", "kinds": ["waifu"],
                  "query": "cats"})
    transport = _Transport([
        ("gifs.example.test", _json(
            {"results": [{"url": "https://gifs.example.test/a.gif"}]})),
    ])
    client = ImagePoolClient(transport, registry=reg,
                             generator=None)
    hit = client.fetch_media("waifu", rng=random.Random(1))
    assert hit.source == "my-gifs"
    assert hit.url == "https://gifs.example.test/a.gif"
    # a success is journaled as health, the streak stays clean
    assert reg.registered()[0]["fails"] == 0


def test_registered_source_failure_journaled_then_skipped(
        tmp_path):
    reg = SourceRegistry(tmp_path / "reg.json")
    reg.register({"name": "dead-one",
                  "url": "https://dead.example.test/x",
                  "shape": "results-url", "kinds": ["waifu"]})
    transport = _Transport([])
    client = ImagePoolClient(transport, registry=reg,
                             generator=None)

    def walk(seed):
        with pytest.raises(SourceFailure):
            client.fetch_media("waifu", rng=random.Random(seed))

    walk(1)
    row = reg.registered()[0]
    assert row["fails"] == 1
    assert row["last_error"]
    assert any(j["event"] == "failure" and j["source"] == "dead-one"
               for j in reg.journal())
    walk(2)
    walk(3)
    row = reg.registered()[0]
    assert row["fails"] == 3  # stood down after the third
    calls_at_stood_down = len(
        [u for u in transport.calls if "dead.example" in u])
    assert calls_at_stood_down == 3
    walk(4)  # inside the spell: the walk never dials it
    assert len([u for u in transport.calls
                if "dead.example" in u]) == calls_at_stood_down
    # the spell expires; the source fails its way to disabled
    # (each post-spell failure re-stands it down — the house
    # breaker semantics — so each walk needs its own expiry)
    def expire_spell():
        data = json.loads((tmp_path / "reg.json").read_text())
        data["sources"][0]["stood_down_until"] = 0.0
        (tmp_path / "reg.json").write_text(json.dumps(data))
        reg.load()

    for seed in (5, 6, 7):
        expire_spell()
        walk(seed)
    row = reg.registered()[0]
    assert row["disabled"] is True
    assert any(j["event"] == "disabled" for j in reg.journal())
    calls_at_disabled = len(
        [u for u in transport.calls if "dead.example" in u])
    walk(8)  # disabled: skipped for good, not retried forever
    assert len([u for u in transport.calls
                if "dead.example" in u]) == calls_at_disabled


def test_registry_never_touches_the_seed_table(tmp_path):
    before = {kind: [(s.name, s.url) for s in chain]
              for kind, chain in IMAGE_POOLS.items()}
    reg = SourceRegistry(tmp_path / "reg.json")
    reg.register({"name": "extra-one",
                  "url": "https://extra.example.test/a",
                  "shape": "url-list", "kinds": ["neko"]})
    transport = _Transport([
        ("extra.example.test", _json(
            ["https://extra.example.test/cat.jpg"])),
    ])
    client = ImagePoolClient(transport, registry=reg,
                             generator=None)
    hit = client.fetch_media("neko", rng=random.Random(2))
    assert hit.source == "extra-one"
    after = {kind: [(s.name, s.url) for s in chain]
             for kind, chain in IMAGE_POOLS.items()}
    assert before == after  # seeds are code; the file grew, not the table


def test_keyed_registered_source_without_its_key_never_dials(
        tmp_path):
    reg = SourceRegistry(tmp_path / "reg.json")
    reg.register({"name": "keyed-one",
                  "url": "https://keyed.example.test/s",
                  "shape": "tenor", "kinds": ["waifu"],
                  "key": "tenor", "query": "cats"})
    transport = _Transport([
        ("keyed.example.test", _json({"results": []})),
        ("api.waifu.pics", _json(
            {"url": "https://i.waifu.pics/x.gif"})),
    ])
    client = ImagePoolClient(transport, registry=reg)  # no keys
    hit = client.fetch_media("waifu", rng=random.Random(0))
    assert hit.source != "keyed-one"
    assert not any("keyed.example" in u for u in transport.calls)


def test_pool_gif_kind_floors_to_her_studio(tmp_path):
    transport = _Transport([])
    made = tmp_path / "studio.gif"
    made.write_bytes(GIF_BYTES)
    client = ImagePoolClient(
        transport, generator=lambda kind: str(made))
    hit = client.fetch_media("gif", rng=random.Random(0))
    assert hit.local_path == str(made)
    assert hit.source_class == "studio"
    classes = client.available_classes("gif")
    assert "animals" in classes and "studio" in classes
    keyed = ImagePoolClient(transport, tenor_key="K",
                            giphy_key="G",
                            generator=lambda kind: str(made))
    assert "general" in keyed.available_classes("gif")


def test_settings_providers_and_registry_form_one_chain(tmp_path):
    reg = SourceRegistry(tmp_path / "reg.json")
    reg.register({"name": "gifbox",
                  "url": "https://gifbox.test/search",
                  "shape": "tenor", "kinds": ["gif"],
                  "key": "tenor"})
    extra = SimpleNamespace(name="owner-extra")
    chain = default_providers(extra=[extra], registry=reg)
    names = [getattr(p, "name", "?") for p in chain]
    assert names == ["owner-extra", "gifbox", "cataas"]


def test_registered_provider_searches_and_journals(tmp_path):
    reg = SourceRegistry(tmp_path / "reg.json")
    reg.register({"name": "gifbox",
                  "url": "https://gifbox.test/search",
                  "shape": "tenor", "kinds": ["gif"],
                  "key": "tenor"})
    spec = reg.specs("gif")[0]
    seen: list[str] = []

    def fetcher(url):
        seen.append(url)
        return {"results": [{
            "title": "a hug",
            "media_formats": {
                "tinymp4": {"url": "https://gifbox.test/hug.mp4"}}}]}

    provider = RegisteredSearchProvider(
        spec, keys={"tenor": "K"}, fetcher=fetcher,
        note=reg.note_result)
    hits = provider.search("hug")
    assert hits == [{"url": "https://gifbox.test/hug.mp4",
                     "title": "a hug", "kind": "gif"}]
    assert "q=hug" in seen[0] and "key=K" in seen[0]

    def dead(url):
        raise RuntimeError("host down")

    provider2 = RegisteredSearchProvider(
        spec, keys={"tenor": "K"}, fetcher=dead,
        note=reg.note_result)
    with pytest.raises(RuntimeError):
        provider2.search("hug")
    row = reg.registered()[0]
    assert row["fails"] == 1
    assert any(j["event"] == "failure" for j in reg.journal())


def test_registry_rejects_bad_specs_at_the_door(tmp_path):
    reg = SourceRegistry(tmp_path / "reg.json")
    with pytest.raises(ValueError):
        reg.register({"name": "x", "url": "ftp://nope",
                      "shape": "tenor"})
    with pytest.raises(ValueError):
        reg.register({"name": "x", "url": "https://ok.test",
                      "shape": "carrier-pigeon"})
    with pytest.raises(ValueError):
        reg.register({"name": "x", "url": "https://ok.test",
                      "shape": "tenor", "kinds": ["mars"]})
    assert reg.registered() == []


# ── reactions: the general seed + the studio floor ────────────


def test_cataas_seed_offers_only_gif_rows():
    rows = [
        {"id": "g1", "tags": ["hug", "cute"],
         "mimetype": "image/gif"},
        {"id": "s1", "tags": ["hug"], "mimetype": "image/jpeg"},
        {"id": "g2", "tags": ["hug", "black"],
         "mimetype": "image/gif"},
    ]
    provider = CataasSearchProvider(fetcher=lambda url: rows)
    hits = provider.search("hug")
    assert [h["url"] for h in hits] == [
        "https://cataas.com/cat/g1", "https://cataas.com/cat/g2"]
    assert all(h["kind"] == "gif" for h in hits)


def test_reaction_reaches_the_studio_floor_when_anime_is_down(
        tmp_path):
    dead = WaifuPicsProvider(
        fetcher=lambda url: (_ for _ in ()).throw(
            RuntimeError("down")))
    svc, lib = _svc(tmp_path, providers=[],
                    media_dir=tmp_path / "m",
                    reaction_providers=[dead])
    out = svc.reaction_gif("hug", room="r", platform="whatsapp",
                           send=_ok_send([]))
    assert out["sent"] is True
    assert lib.items[0]["source"] == "self:render"


def test_reaction_general_rung_uses_the_chain(tmp_path,
                                              monkeypatch):
    """Beyond-anime: with the anime seeds silent, the service's
    general chain (here the cataas seed, faked) answers the
    command word and the reaction lands."""
    rows = [{"id": "g1", "tags": ["hug"],
             "mimetype": "image/gif"}]
    svc, lib = _svc(
        tmp_path,
        providers=[CataasSearchProvider(fetcher=lambda url: rows)],
        media_dir=tmp_path / "m", reaction_providers=[])

    def fake_download(url, dest_dir):
        dest = Path(dest_dir) / "dl.gif"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(GIF_BYTES)
        return dest

    monkeypatch.setattr(
        "nomorals.social.chat.expression.download_to_library",
        fake_download)
    out = svc.reaction_gif("hug", room="r", platform="whatsapp",
                           send=_ok_send([]))
    assert out["sent"] is True
    assert lib.items[0]["source"] == "provider:cataas"
