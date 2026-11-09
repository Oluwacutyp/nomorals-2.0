"""Living Chat Experience — Phase 3 gates (doc 2 P3 + doc 6
procedural creation).

Contracts pinned here:

1. **Procedural creation** — a creation request becomes an
   EXECUTABLE DESCRIPTION (an emoji-studio composition, a
   Scene3D document, a motion/explainer spec) that the existing
   renderers execute; the pipeline INSPECTS the real output
   (decodability, dimensions, non-blankness, probe facts) and
   REPAIRS the description — bounded attempts, every repair
   journaled with what was wrong and what changed. Strict mode:
   the procedural path NEVER reaches a generation model; a
   description that cannot be made to render fails honestly
   with the inspection evidence, never with a fabricated asset.
2. **Expression unification** — the expression organs (custom
   emoji, stickers, reactions, media moments) are registered
   in the P1 surface manifest, and the manifest's member set is
   proven complete against the organs' own live registries.
3. **Owner style-pinning** — pins (typography intent/face,
   palette/theme traits) persist, apply globally and per
   surface family, outrank semantic defaults at the ONE
   selection seam (the typography service + the chat theme
   resolver), are named where they are in force (selection,
   resolved style, living cards), and unsetting restores the
   default.
4. **Numbered fallback for every living card** (P2 remainder
   #1) — a living card delivered in the numbered form registers
   its actions into the OperationStore AT SEND TIME under the
   card's own identity, so a bare-number reply resolves against
   that card's operation, whoever produced the card.
"""

from __future__ import annotations

import ast
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import pytest

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None


def _tmpdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="nm-test-", dir=_TEST_TMPDIR)


@pytest.fixture(autouse=True)
def _pin_store_isolation():
    """Every test runs against a throwaway pin store; the real
    home store is never touched and never leaks in."""
    from nomorals.core import style_pins

    tmp = _tmpdir()
    previous = style_pins._STORE
    style_pins.set_pin_store(style_pins.PinStore(
        path=Path(tmp.name) / "pins.json"))
    yield
    style_pins.set_pin_store(previous)
    tmp.cleanup()


# ── 3a. the pin store ───────────────────────────────────────────

from nomorals.core.style_pins import (  # noqa: E402
    PinStore,
    effective_pin,
    get_pin_store,
)


class TestPinStore:
    def test_round_trip_and_persistence(self, tmp_path):
        path = tmp_path / "pins.json"
        store = PinStore(path=path)
        store.set("", intent="display", theme="house")
        assert store.get("").intent == "display"
        again = PinStore(path=path)
        assert again.get("").intent == "display"
        assert again.get("").theme == "house"

    def test_global_scope_alias(self):
        store = PinStore(path=None)
        store.set("global", intent="mono")
        assert store.get("").intent == "mono"
        assert effective_pin(store, "").intent == "mono"

    def test_family_overrides_global_per_field(self):
        store = PinStore(path=None)
        store.set("", intent="display", theme="neon")
        store.set("creation", intent="playful")
        eff = effective_pin(store, "creation")
        assert eff is not None
        assert eff.intent == "playful"      # family wins its field
        assert eff.theme == "neon"          # global fills the rest
        assert "creation" in eff.label and "global" in eff.label

    def test_no_pins_no_effective(self):
        store = PinStore(path=None)
        assert effective_pin(store, "creation") is None
        assert effective_pin(store, "") is None

    def test_clear_restores_default(self):
        store = PinStore(path=None)
        store.set("", intent="display")
        store.set("creation", intent="playful")
        store.clear("creation")
        eff = effective_pin(store, "creation")
        assert eff is not None and eff.intent == "display"
        store.clear("")
        assert effective_pin(store, "creation") is None

    def test_clear_one_field_keeps_others(self):
        store = PinStore(path=None)
        store.set("", intent="display", theme="neon")
        store.clear("", field="intent")
        eff = effective_pin(store, "")
        assert eff is not None
        assert eff.intent == "" and eff.theme == "neon"

    def test_traits_ride_along(self):
        store = PinStore(path=None)
        store.set("", trait={"accent": "gold"})
        eff = effective_pin(store, "")
        assert eff is not None and eff.traits.get("accent") == "gold"

    def test_singleton_is_settable(self):
        store = get_pin_store()
        assert isinstance(store, PinStore)


# ── 3b. pins at the selection seams ─────────────────────────────

ft = pytest.importorskip("fontTools")


def _build_font(path: Path, family: str, codepoints) -> Path:
    from fontTools.fontBuilder import FontBuilder
    from fontTools.ttLib.tables._g_l_y_f import Glyph

    glyphs = [".notdef"] + [f"u{cp:04X}" for cp in codepoints]
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(glyphs)
    fb.setupCharacterMap({cp: f"u{cp:04X}" for cp in codepoints})
    fb.setupGlyf({g: Glyph() for g in glyphs})
    fb.setupHorizontalMetrics({g: (600, 0) for g in glyphs})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({
        "familyName": family, "styleName": "Regular",
        "uniqueFontIdentifier": f"{family} Regular fixture",
        "fullName": f"{family} Regular",
        "psName": (family + "Regular").replace(" ", ""),
        "version": "1.000",
    })
    fb.setupOS2()
    fb.setupPost()
    fb.save(str(path))
    return path


@pytest.fixture()
def typo_service(tmp_path):
    from nomorals.media.typography import TypographyService

    latin = list(range(0x20, 0x100))
    _build_font(tmp_path / "AlphaSans.ttf", "Alpha Sans", latin)
    _build_font(tmp_path / "BetaSerif.ttf", "Beta Serif", latin)
    return TypographyService(roots=[tmp_path],
                             store_dir=tmp_path / "store")


class TestTypographyPinSeam:
    def test_family_pin_outranks_default_selection(self, typo_service):
        from nomorals.core import style_pins

        base = typo_service.select("text")
        assert base.pin == ""
        style_pins.get_pin_store().set("", font_family="Beta Serif")
        pinned = typo_service.select("text")
        assert pinned.face is not None
        assert pinned.face.family == "Beta Serif"
        assert "global" in pinned.pin
        assert any("pin" in note for note in pinned.notes)

    def test_intent_pin_outranks_semantic_default(self, typo_service):
        from nomorals.core import style_pins

        style_pins.get_pin_store().set("", intent="mono")
        sel = typo_service.select("text")
        assert sel.intent == "mono"
        assert sel.pin

    def test_family_scoped_pin_applies_only_there(self, typo_service):
        from nomorals.core import style_pins

        style_pins.get_pin_store().set("posters",
                                       font_family="Beta Serif")
        plain = typo_service.select("text")
        assert plain.pin == ""
        scoped = typo_service.select("text", surface_family="posters")
        assert scoped.face is not None
        assert scoped.face.family == "Beta Serif"
        assert "posters" in scoped.pin

    def test_unpin_restores_default(self, typo_service):
        from nomorals.core import style_pins

        store = style_pins.get_pin_store()
        store.set("", font_family="Beta Serif")
        assert typo_service.select("text").face.family == "Beta Serif"
        store.clear("")
        sel = typo_service.select("text")
        assert sel.pin == ""


class TestChatThemePinSeam:
    def test_resolve_style_honors_theme_pin(self):
        from nomorals.core import style_pins
        from nomorals.social.chat.typography import THEMES, resolve_style

        base = resolve_style({}, "chat")
        other = next(name for name in THEMES if name != base.theme)
        style_pins.get_pin_store().set("chat", theme=other)
        pinned = resolve_style({}, "chat")
        assert pinned.theme == other
        assert pinned.pin

    def test_resolve_style_pin_is_family_scoped(self):
        from nomorals.core import style_pins
        from nomorals.social.chat.typography import THEMES, resolve_style

        base = resolve_style({}, "chat")
        other = next(name for name in THEMES if name != base.theme)
        style_pins.get_pin_store().set("briefing", theme=other)
        # a pin for another surface class does not leak into chat
        assert resolve_style({}, "chat").theme == base.theme
        assert resolve_style({}, "briefing").theme == other


# ── 1. the procedural creation pipeline ─────────────────────────

from nomorals.social.chat import creation  # noqa: E402
from nomorals.social.chat.creation import (  # noqa: E402
    CreationSpec,
    author_motion_spec,
    author_scene_spec,
    author_studio_spec,
    create_procedural,
)
from nomorals.storage.db import Database  # noqa: E402


def _studio_spec(**payload_over) -> CreationSpec:
    spec = author_studio_spec("a gold heart with sparkles")
    payload = dict(spec.payload)
    payload.update(payload_over)
    expect = dict(spec.expect)
    if "form" in payload_over:
        expect["format"] = {"png": "png", "sticker": "webp",
                            "emoji": "png"}[payload_over["form"]]
        if payload_over["form"] == "emoji":
            expect["width"] = expect["height"] = 100
    return CreationSpec(kind="studio", title=spec.title,
                        payload=payload, expect=expect)


class TestProceduralStudio:
    def test_happy_path_renders_real_artifact(self):
        with _tmpdir() as tmp:
            result = create_procedural(
                author_studio_spec("a gold heart with sparkles"),
                Path(tmp) / "out")
            assert result.ok, result.findings
            assert result.path is not None and result.path.exists()
            assert len(result.attempts) == 1
            assert result.attempts[0].findings == ()
            from PIL import Image
            with Image.open(result.path) as img:
                img.verify()
            with Image.open(result.path) as img:
                assert img.size == (512, 512)

    def test_broken_description_is_repaired_and_journaled(self):
        with _tmpdir() as tmp:
            db = Database(os.path.join(tmp, "journal.db"))
            spec = _studio_spec(form="png")
            spec.payload["design"] = {
                "shape": "heart", "palette": "neon", "face": "",
                "letter": "", "sparkles": False, "bg": ""}
            spec.payload["size"] = 5000
            result = create_procedural(spec, Path(tmp) / "out", db=db)
            assert result.ok, result.findings
            assert len(result.attempts) >= 2
            first = result.attempts[0]
            codes = {f.code for f in first.findings}
            assert "unknown-palette" in codes
            assert "size-out-of-range" in codes
            changes = " ".join(first.changes)
            assert "palette" in changes and "size" in changes
            rows = db.query(
                "SELECT findings, changes FROM procedural_creations "
                "ORDER BY attempt")
            assert len(rows) >= 2
            blob = " ".join(str(r["findings"]) + str(r["changes"])
                            for r in rows)
            assert "unknown-palette" in blob
            from PIL import Image
            with Image.open(result.path) as img:
                assert img.size == (512, 512)

    def test_inspection_dimension_mismatch_is_repaired(self):
        with _tmpdir() as tmp:
            spec = _studio_spec(form="png")
            spec.expect = {**spec.expect, "width": 256, "height": 256}
            result = create_procedural(spec, Path(tmp) / "out")
            assert result.ok, result.findings
            codes = {f.code for a in result.attempts
                     for f in a.findings}
            assert "dimension-mismatch" in codes
            from PIL import Image
            with Image.open(result.path) as img:
                assert img.size == (256, 256)

    def test_renderer_error_fails_honestly_without_repairs(self,
                                                          monkeypatch):
        def _boom(spec, out_dir, name):
            raise RuntimeError("renderer exploded")

        monkeypatch.setattr(creation, "_render_studio", _boom)
        with _tmpdir() as tmp:
            result = create_procedural(
                author_studio_spec("a gold heart"), Path(tmp) / "out")
            assert not result.ok
            assert result.path is None
            assert len(result.attempts) == 1
            assert result.attempts[0].findings[0].code == "renderer-error"
            assert "renderer exploded" in \
                result.attempts[0].findings[0].detail
            assert result.attempts[0].changes == ()

    def test_strict_mode_never_reaches_a_generation_model(self,
                                                          monkeypatch):
        # source-level: the pipeline module carries no generation
        # import and never routes through the ladder helper that
        # tries a generation backend first.
        src = Path(creation.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        assert not any("generate" in name or "imggen" in name
                       for name in imported), imported
        assert "create_emoji_sticker" not in src
        # runtime-level: the generation backend entrypoint is a
        # landmine — a procedural creation must still succeed.
        import nomorals.media_edit.generate as gen

        def _landmine(*a, **k):
            raise AssertionError("generation backend reached")

        monkeypatch.setattr(gen, "get_backend", _landmine)
        with _tmpdir() as tmp:
            result = create_procedural(
                author_studio_spec("a purple laughing cat"),
                Path(tmp) / "out")
            assert result.ok, result.findings


class _FakeSceneRenderer:
    """A registry citizen writing REAL PNG frames (the pipeline
    contract), so inspection runs on genuine pixels."""

    name = "p3fake"

    def capabilities(self):
        from nomorals.media.edit_engine.scene3d import RendererCapabilities
        return RendererCapabilities(
            surfaces=("scene3d",), node_kinds=("group", "mesh", "light"),
            mesh_kinds=("primitive", "explicit"),
            primitives=("cube", "sphere", "plane", "cylinder", "cone"),
            material_kinds=("flat", "lambert", "phong"),
            light_kinds=("ambient", "directional", "point", "spot"),
            animated_props=("position", "rotation", "scale"),
            max_width=7680, max_height=4320, deterministic=True)

    def probe(self):
        from nomorals.media.edit_engine.scene3d import RendererProbe
        return RendererProbe(True, "")

    def render_frames(self, scene, *, frames, workdir):
        from PIL import Image
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in frames:
            p = workdir / f"f_{f:04d}.png"
            img = Image.new("RGB", (scene.output["width"],
                                    scene.output["height"]),
                            (24, 28, 44))
            for x in range(8, 40):
                for y in range(8, 40):
                    img.putpixel((x, y), (220, 60, 50))
            img.save(str(p))
            paths.append(str(p))
        return {"frames_dir": str(workdir), "pattern": "f_%04d.png",
                "framerate": float(scene.output.get("fps", 30)),
                "frames": paths, "width": scene.output["width"],
                "height": scene.output["height"]}


class TestProceduralScene:
    def test_scene_renders_through_registered_renderer(self):
        from nomorals.media.edit_engine import scene3d

        fake = _FakeSceneRenderer()
        scene3d.register_renderer(fake)
        try:
            with _tmpdir() as tmp:
                spec = author_scene_spec("a red cube")
                spec.payload["renderer"] = "p3fake"
                result = create_procedural(spec, Path(tmp) / "out")
                assert result.ok, result.findings
                from PIL import Image
                with Image.open(result.path) as img:
                    assert img.size == (320, 320)
        finally:
            scene3d.unregister_renderer("p3fake")

    def test_invalid_scene_document_fails_honestly(self):
        with _tmpdir() as tmp:
            spec = author_scene_spec("a red cube")
            doc = dict(spec.payload["document"])
            nodes = [dict(n) for n in doc["nodes"]]
            nodes.append(dict(nodes[0]))  # duplicate node name
            doc["nodes"] = nodes
            spec.payload["document"] = doc
            result = create_procedural(spec, Path(tmp) / "out")
            assert not result.ok
            assert result.path is None
            codes = {f.code for f in result.findings}
            assert "scene-invalid" in codes
            assert result.attempts[-1].changes == ()

    @pytest.mark.skipif(
        not shutil.which("ffprobe"), reason="no ffprobe on this host")
    def test_real_renderer_when_available(self):
        from nomorals.media.edit_engine import scene3d

        try:
            from nomorals.media.edit_engine import render3d_blender, render3d_rust  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - rung import is evidence
            assert isinstance(exc, Exception)
        available = [n for n in scene3d.registered_renderers()
                     if scene3d.get_renderer(n).probe().available]
        if not available:
            pytest.skip("no scene renderer available on this host")
        with _tmpdir() as tmp:
            spec = author_scene_spec("a blue sphere")
            doc = dict(spec.payload["document"])
            doc["output"] = {"width": 128, "height": 128, "fps": 30}
            spec.payload["document"] = doc
            spec.expect = {**spec.expect, "width": 128, "height": 128}
            result = create_procedural(spec, Path(tmp) / "out")
            assert result.ok, result.findings


@pytest.mark.skipif(not shutil.which("ffmpeg"),
                    reason="no ffmpeg on this host")
class TestProceduralMotion:
    def test_explainer_renders_and_probes(self):
        with _tmpdir() as tmp:
            spec = author_motion_spec("how rain forms")
            result = create_procedural(spec, Path(tmp) / "out")
            assert result.ok, result.findings
            assert result.path is not None
            assert result.path.suffix == ".mp4"
            import json
            import subprocess
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", str(result.path)],
                capture_output=True, text=True, timeout=60)
            data = json.loads(probe.stdout)
            video = [s for s in data["streams"]
                     if s["codec_type"] == "video"]
            assert video and video[0]["width"] == 320
            assert float(data["format"]["duration"]) >= 0.8

    def test_unknown_engine_is_unrepairable(self):
        with _tmpdir() as tmp:
            spec = author_motion_spec("how rain forms")
            spec.payload["scenes"] = [{
                "engine": "hologram", "title": "x", "duration": 1.0}]
            result = create_procedural(spec, Path(tmp) / "out")
            assert not result.ok
            codes = {f.code for f in result.findings}
            assert "unknown-engine" in codes
            assert result.attempts[-1].changes == ()


# ── 2. expression unification: the manifest ─────────────────────

from nomorals.social.chat import surface_manifest as sm  # noqa: E402
from nomorals.social.chat.control import parse_control  # noqa: E402
from nomorals.social.chat.expression import (  # noqa: E402
    EXPRESSION_COMMANDS,
    MOMENT_TAGS,
    REACTION_GIF_COMMANDS,
)


def _resolves(data: str) -> bool:
    if data.startswith(("http://", "https://")):
        return True
    return parse_control("/" + data) is not None


class TestExpressionManifest:
    def test_expression_members_match_live_registries(self):
        surface = sm.surface_by_id("expression")
        live = set(EXPRESSION_COMMANDS) | set(REACTION_GIF_COMMANDS) \
            | set(MOMENT_TAGS)
        assert set(surface.members) == live, (
            f"manifest-only={sorted(set(surface.members) - live)} "
            f"live-only={sorted(live - set(surface.members))}")

    def test_expression_commands_are_registered_commands(self):
        from nomorals.social.chat.control import CONTROL_COMMANDS
        for kind in EXPRESSION_COMMANDS:
            assert kind in CONTROL_COMMANDS

    def test_new_surfaces_present_and_resolvable(self):
        for sid in ("expression", "creation", "style-pins"):
            surface = sm.surface_by_id(sid)
            assert surface.actions, f"{sid} declares no actions"
            for ref in surface.actions:
                kind = ref.handler.split(":", 1)[1]
                from nomorals.social.chat.control import CONTROL_COMMANDS
                assert kind in CONTROL_COMMANDS, (
                    f"{sid}/{ref.action_id}: no control kind {kind!r}")
                if ref.data:
                    assert _resolves(ref.data), (
                        f"{sid}/{ref.action_id}: payload {ref.data!r} "
                        "does not resolve")


# ── 3c. pins are named on living cards; the creation family ─────

from nomorals.social.chat.action_schema import Action  # noqa: E402
from nomorals.social.chat.living_cards import (  # noqa: E402
    Card,
    render_card,
)


def _creation_card() -> Card:
    return Card(card_id="cafe0001", family="creation",
                chat_key="local:room1", owner_key="owner1",
                platform="local", chat_id="room1",
                title="Creating — a gold heart", body="working",
                actions=(Action(label="Again", data="create gold heart"),))


class TestCardPinNaming:
    def test_creation_family_registered(self):
        from nomorals.social.chat import living_cards
        assert living_cards.FAMILY_CREATION == "creation"
        assert "creation" in living_cards.RENDERERS
        assert "creation" in living_cards.CARD_FAMILIES

    def test_card_names_an_in_force_pin(self):
        from nomorals.core import style_pins

        plain = render_card(_creation_card(), "local")
        assert "style pin" not in plain.text
        style_pins.get_pin_store().set("creation", intent="playful")
        pinned = render_card(_creation_card(), "local")
        assert "style pin" in pinned.text
        assert "creation" in pinned.text


# ── 4. numbered registration at send time ───────────────────────

from nomorals.agents.partner.button_picks import ButtonPickStore  # noqa: E402
from nomorals.core.operations import PickStatus  # noqa: E402
from nomorals.social.chat.living_cards import CardHost, CardStore  # noqa: E402


def _host_with_picks(picks: ButtonPickStore, *, send_ok: bool = True):
    sent: list = []

    def _send(platform, chat, text, buttons=None):
        sent.append((platform, text, buttons))
        return (send_ok, "m1" if send_ok else "")

    def _edit(chat, message_id, text, buttons=None):
        return False

    host = CardHost(
        CardStore(), send=_send, edit=_edit,
        register_picks=lambda chat_key, commands, owner_key,
        operation_id: picks.register_operation(
            chat_key, commands, owner_key=owner_key,
            operation_id=operation_id, kind="living-card"))
    return host, sent


def _numbered_card(**kw) -> Card:
    base = dict(card_id="cafe0002", family="generic",
                chat_key="discord:room9", owner_key="alice",
                platform="discord", chat_id="room9",
                title="Deploy", body="pick one",
                actions=(Action(label="Platforms", data="platforms"),
                         Action(label="Status", data="status")))
    base.update(kw)
    return Card(**base)


class TestNumberedRegistration:
    def test_publish_registers_under_card_identity(self):
        picks = ButtonPickStore()
        host, _sent = _host_with_picks(picks)
        card = _numbered_card()
        host.publish(card, object())
        res = picks.take_result("discord:room9", 1, owner_key="alice",
                                operation_id="cafe0002")
        assert res.status is PickStatus.OK
        assert res.command == "platforms"

    def test_bare_number_resolves_the_card(self):
        picks = ButtonPickStore()
        host, _sent = _host_with_picks(picks)
        host.publish(_numbered_card(), object())
        res = picks.take_result("discord:room9", 2, owner_key="alice")
        assert res.status is PickStatus.OK
        assert res.command == "status"

    def test_failed_send_registers_nothing(self):
        picks = ButtonPickStore()
        host, _sent = _host_with_picks(picks, send_ok=False)
        host.publish(_numbered_card(), object())
        assert not picks.has("discord:room9", owner_key="alice")

    def test_revise_refreshes_the_registration(self):
        picks = ButtonPickStore()
        host, _sent = _host_with_picks(picks)
        card = _numbered_card()
        host.publish(card, object())
        host.revise(card, object(),
                    actions=(Action(label="Doctor", data="doctor"),))
        res = picks.take_result("discord:room9", 1, owner_key="alice",
                                operation_id="cafe0002")
        assert res.status is PickStatus.OK
        assert res.command == "doctor"

    def test_button_surface_registers_nothing(self):
        picks = ButtonPickStore()
        host, _sent = _host_with_picks(picks)
        card = _numbered_card(platform="telegram-bot",
                              chat_key="telegram-bot:room9")
        host.publish(card, object())
        assert not picks.has("telegram-bot:room9", owner_key="alice")


# ── 4b. runtime end-to-end: a bare number runs the card's action ─

from nomorals.agents.context import build_context  # noqa: E402
from nomorals.agents.partner_runtime import PartnerRuntime  # noqa: E402
from nomorals.core.config import load_settings  # noqa: E402
from nomorals.social.chat.base import (  # noqa: E402
    ChatKind,
    ChatMessage,
    ChatRef,
)


def _make_runtime():
    tmp = _tmpdir()
    settings = load_settings(overrides={
        "home": tmp.name,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
        "partner.enabled": "false",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    runtime = PartnerRuntime(context)
    runtime._typing_enabled = lambda chat: False
    return runtime, tmp


class RuntimeCardPickTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime, self.tmp = _make_runtime()

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        self.tmp.cleanup()

    def test_bare_number_executes_living_card_action(self) -> None:
        from types import SimpleNamespace

        from nomorals.agents.partner.runtime import _pick_owner_key

        chat = ChatRef(platform="local", chat_id="room1",
                       kind=ChatKind.DM, title="room", peer="room1")
        msg = ChatMessage(chat=chat, incoming=True, text="hi",
                          sender="owner1", sender_id="owner1")
        self.runtime.gateway = SimpleNamespace(
            send=lambda platform, c, text, buttons=None:
            SimpleNamespace(ok=True, message_id="m1"))
        card = Card(
            card_id="cafe0003", family="generic",
            chat_key=chat.key, owner_key=_pick_owner_key(msg),
            platform="local", chat_id="room1",
            title="Deploy", body="pick one",
            actions=(Action(label="Platforms", data="platforms"),))
        self.runtime._living_card_host().publish(card, chat)
        out = self.runtime._button_pick_hook(
            ChatMessage(chat=chat, incoming=True, text="1",
                        sender="owner1", sender_id="owner1"))
        self.assertIsNotNone(out)


# ── 1+3. the chat command surface ───────────────────────────────

class TestCommandSurface:
    def test_kinds_parse(self):
        cmd = parse_control("/create a gold heart")
        assert cmd is not None and cmd.kind == "create"
        cmd = parse_control("/stylepin set global intent display")
        assert cmd is not None and cmd.kind == "stylepin"

    def test_create_without_message_answers_in_words(self):
        runtime, tmp = _make_runtime()
        try:
            out = runtime._control_create("a gold heart", None)
            assert isinstance(out, str) and out
        finally:
            runtime.stop()
            tmp.cleanup()

    def test_stylepin_handler_flow(self):
        from nomorals.core import style_pins

        runtime, tmp = _make_runtime()
        try:
            out = runtime._control_stylepin(
                "set global intent display", None)
            assert "display" in out
            eff = style_pins.effective_pin(style_pins.get_pin_store(), "")
            assert eff is not None and eff.intent == "display"
            shown = runtime._control_stylepin("", None)
            assert "display" in shown
            bad = runtime._control_stylepin(
                "set global intent banana", None)
            assert "not a typography intent" in bad
            assert "display" in shown  # the real pin survived
            assert "banana" not in str(
                style_pins.get_pin_store().get(""))
            cleared = runtime._control_stylepin("clear global", None)
            assert cleared
            assert style_pins.effective_pin(
                style_pins.get_pin_store(), "") is None
        finally:
            runtime.stop()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
