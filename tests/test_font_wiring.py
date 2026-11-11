"""Pins for FontLibrary F2 — the wiring programme (font
census §7 W1–W3): every render surface resolves fonts
through the ONE registry door (the typography service's
context policy over the stocked FontLibrary), the context
policy table is data, caller choice runs over the real
catalogue, and a family that cannot resolve is a NAMED
outcome at each surface's error seam — never a silent
slide to Pillow's bitmap default.

The lights-up pin is mechanical, per the census: a family
registered in a synthetic library lights up on multiple
surfaces with zero surface-code changes.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from nomorals.media.typography import (
    CONTEXT_POLICIES,
    ContextFontPolicy,
    FontNotStocked,
    context_policy,
    emoji_font_path,
    get_service,
    pil_font_for,
    reset_service,
)
from nomorals.media.typography.fontlib import FontLibrary
from nomorals.media.typography.selection import INTENTS

_DEJAVU = Path("/usr/share/fonts/truetype/dejavu")
pytestmark = pytest.mark.skipif(
    not (_DEJAVU / "DejaVuSans.ttf").is_file(),
    reason="host DejaVu faces are the fixture source")

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None


@pytest.fixture(autouse=True)
def _pin_store_isolation():
    """Every test runs against a throwaway pin store (the
    living-chat p3 isolation idiom)."""
    from nomorals.core import style_pins

    tmp = tempfile.TemporaryDirectory(
        prefix="nm-test-", dir=_TEST_TMPDIR)
    previous = style_pins._STORE
    style_pins.set_pin_store(style_pins.PinStore(
        path=Path(tmp.name) / "pins.json"))
    yield
    style_pins.set_pin_store(previous)
    tmp.cleanup()


# ── the policy table as data ─────────────────────────────


def test_policy_rows_name_real_intents_and_stocked_families():
    library = FontLibrary.discover()
    for context, policy in CONTEXT_POLICIES.items():
        assert policy.intent in INTENTS, context
        for family in policy.families:
            assert library.stocked(family), (
                f"context {context} names unstocked {family}")


def test_context_policy_rows_carry_the_briefs_taste():
    assert context_policy("quote").families[0] == \
        "Playfair Display"
    assert context_policy("quote").intent == "editorial"
    assert "script" in context_policy("quote_script").intent \
        or context_policy("quote_script").families
    assert context_policy("game").intent == "playful"
    assert context_policy("poster").weight == 700
    assert context_policy("screen_whatsapp").families[0] == \
        "Roboto"
    assert context_policy("an-unknown-context").intent == "text"


# ── contexts resolve to the stocked faces ────────────────


@pytest.mark.parametrize("context,family", [
    ("quote", "Playfair Display"),
    ("quote_script", "Great Vibes"),
    ("motivation", "Montserrat"),
    ("poster", "Anton"),
    ("meme", "Anton"),
    ("game", "Poppins"),
    ("game_title", "Bebas Neue"),
    ("scoreboard", "JetBrains Mono"),
    ("card", "Inter"),
    ("card_title", "Montserrat"),
    ("brandmark", "Montserrat"),
    ("caption", "Inter"),
    ("mono", "JetBrains Mono"),
    ("cover", "Anton"),
    ("screen", "Roboto"),
    ("screen_whatsapp", "Roboto"),
    ("screen_telegram", "Roboto"),
    ("screen_sms", "Roboto"),
    ("screen_instagram", "Inter"),
    ("screen_messenger", "Inter"),
    ("screen_snapchat", "Inter"),
    ("screen_imessage", "Inter"),
])
def test_context_resolves_to_its_stocked_family(context, family):
    font = pil_font_for(get_service(), context, 24)
    assert font.getname()[0] == family


def test_variable_weight_axis_is_applied():
    service = get_service()
    probe = "Typography weight probe WW"
    widths = {
        w: pil_font_for(service, "card", 40, weight=w
                        ).getlength(probe)
        for w in (400, 700, 900)}
    assert widths[400] < widths[700] < widths[900]


def test_intent_door_is_byte_stable_for_legacy_callers():
    """A caller naming only an intent gets exactly the face
    the service itself selects — the wiring changed the
    ladder underneath, not the selection."""
    from nomorals.social.chat import makers

    via_makers = makers._font(36, intent="display")
    via_service = get_service().pil_font("display", 36,
                                         weight=400)
    assert via_makers.getname() == via_service.getname()
    probe = "byte stability probe"
    assert via_makers.getlength(probe) == \
        via_service.getlength(probe)


# ── named misses at the surfaces' error seams ────────────


def test_named_miss_raises_from_the_door():
    with pytest.raises(FontNotStocked) as caught:
        pil_font_for(get_service(), "card", 20,
                     family="No Such Family ZZ")
    assert "No Such Family ZZ" in str(caught.value)
    assert "not in the stocked library" in str(caught.value)


def test_named_miss_raises_from_makers():
    from nomorals.social.chat import makers

    with pytest.raises(FontNotStocked):
        makers._font(30, family="No Such Family ZZ")


def test_studio_miss_is_named_and_lands_on_a_stocked_face(
        caplog):
    """studio.find_font's pinned contract is never-fail (an
    unloadable path and an unknown name both return a
    working font — tests/test_media_edit_studio_failures
    test_112): the miss is named in the log and the
    fall-through lands on the poster context's stocked
    face, not the old ladder's DejaVu/bitmap slide."""
    import logging

    from nomorals.media_edit import studio

    with caplog.at_level(logging.WARNING):
        font = studio.find_font("No Such Family ZZ", 20)
    assert font is not None
    assert font.getname()[0] == "Anton"
    assert "No Such Family ZZ" in caplog.text


def test_named_miss_is_a_motion_error_in_motion_studio():
    from nomorals.media.motion_studio._core import (
        MotionStudioError,
        find_font,
    )

    with pytest.raises(MotionStudioError) as caught:
        find_font(613, "No Such Family ZZ")
    assert "No Such Family ZZ" in str(caught.value)


def test_named_miss_is_a_mock_screens_error():
    from nomorals.social.chat.mock_screens import render as R
    from nomorals.social.chat.mock_screens.render import (
        MockScreensError,
        _Canvas,
    )
    from nomorals.social.chat.mock_screens.schema import (
        DeviceChrome,
        Participant,
        Screen,
    )

    screen = Screen(
        platform="whatsapp", group=False, title="Ada",
        participants=[Participant("me", "You"),
                      Participant("ada", "Ada")],
        messages=[],
        chrome=DeviceChrome(style="android", clock="9:41",
                            battery=62, network="5G"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(R.CHAT_DESIGNS["whatsapp"], "family",
                   "No Such Family ZZ")
        canvas = _Canvas(screen)
        with pytest.raises(MockScreensError) as caught:
            canvas.font(14)
    assert "No Such Family ZZ" in str(caught.value)


# ── per-surface faces (the door, per surface) ────────────


def test_surface_faces_come_from_the_registry():
    from nomorals.media import cover_video
    from nomorals.media_edit import images as edit_images
    from nomorals.social.chat import (
        brandmark,
        emoji_studio,
        media_convert,
        whatsapp_render,
    )

    assert brandmark._font(40).getname()[0] == "Montserrat"
    assert cover_video._font(40).getname()[0] == "Anton"
    assert edit_images._load_font(40).getname()[0] == "Anton"
    assert media_convert._load_font(40).getname()[0] == "Anton"
    assert emoji_studio._font(40).getname()[0] == "Anton"
    assert "Mono" in whatsapp_render._font().getname()[0]


def test_motion_unnamed_font_is_the_poster_context():
    from nomorals.media.motion_studio._core import find_font

    assert find_font(617).getname()[0] == "Anton"


def test_studio_discovery_and_default_run_over_the_catalogue():
    from nomorals.media_edit import studio

    found = studio.discover_fonts(refresh=True)
    assert "inter" in found  # a stocked core family
    assert studio.find_font(None, 24) is not None
    assert studio.find_font("Inter", 24).getname()[0] == "Inter"


def test_owner_signature_stamp_applies_with_a_registry_face():
    from PIL import Image

    from nomorals.social.chat.owner_system import (
        apply_media_watermark,
    )

    buf = io.BytesIO()
    Image.new("RGB", (400, 300), (30, 60, 90)).save(
        buf, format="PNG")
    out, applied = apply_media_watermark(
        buf.getvalue(), "Powered by peacethefirst")
    assert applied is True
    assert out != buf.getvalue()


def test_upgrade_composite_accepts_a_family_name():
    from PIL import Image

    from nomorals.media.upgrade.typography import (
        TextComposite,
        composite_text,
    )

    img = Image.new("RGB", (320, 200), (10, 10, 10))
    out = composite_text(img, [TextComposite(
        text="portable spec", font_path="Playfair Display",
        size=28)])
    assert out.size == img.size
    # an existing file path still loads directly
    core_file = str(
        Path(__file__).resolve().parent.parent
        / "nomorals" / "media" / "typography" / "fonts"
        / "ofl" / "inter" / "Inter[opsz,wght].ttf")
    assert Path(core_file).is_file()
    out2 = composite_text(img, [TextComposite(
        text="file spec", font_path=core_file, size=28)])
    assert out2.size == img.size
    # a family nothing holds: the layer falls back, named in
    # the log — the composite itself never dies over a font
    out3 = composite_text(img, [TextComposite(
        text="missing spec", font_path="No Such Family ZZ",
        size=28)])
    assert out3.size == img.size


def test_emoji_face_comes_through_the_door():
    from nomorals.social.chat.mock_screens import render as R

    assert emoji_font_path(get_service()) is not None
    assert R._emoji_tile("🔥", 24) is not None


# ── /stylepin validates against the real catalogue ───────


def _make_runtime():
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    tmp = tempfile.TemporaryDirectory(
        prefix="nm-test-", dir=_TEST_TMPDIR)
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


def test_stylepin_font_validates_at_set_time():
    from nomorals.core import style_pins

    runtime, tmp = _make_runtime()
    try:
        ok = runtime._control_stylepin(
            "set global font Playfair Display", None)
        assert "pinned" in ok
        eff = style_pins.effective_pin(
            style_pins.get_pin_store(), "")
        assert eff is not None
        assert eff.font_family == "Playfair Display"
        refused = runtime._control_stylepin(
            "set global font No Such Family ZZ", None)
        assert "not a font family" in refused
        assert "/fonts" in refused
        eff = style_pins.effective_pin(
            style_pins.get_pin_store(), "")
        assert eff.font_family == "Playfair Display"
    finally:
        runtime.stop()
        tmp.cleanup()


# ── the lights-up pin (mechanical, census W1(c)) ─────────


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_synthetic_pack(root: Path) -> Path:
    """A two-family synthetic pack: a family no real tier
    stocks (its face file is a serif font, so a render
    through it is identifiable) and a colour-emoji family
    (path-resolution only)."""
    root.mkdir(parents=True, exist_ok=True)

    def add_family(name, category, rel, src):
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return {
            "name": name, "category": category,
            "license": "OFL-1.1", "license_file": "",
            "reserved_font_name": False,
            "dir": str(Path(rel).parent), "subsets": ["latin"],
            "faces": [{
                "file": rel, "weight": 400, "style": "normal",
                "variable": False, "sha256": _sha(dest),
                "bytes": dest.stat().st_size,
                "scripts": ["Latin"],
            }],
        }

    families = [
        add_family("Lights Up Test Sans", "Sans Serif",
                   "ofl/lightsup/LightsUp.ttf",
                   _DEJAVU / "DejaVuSerif.ttf"),
        add_family("Noto Color Emoji", "Emoji",
                   "ofl/lightsup/Emoji.ttf",
                   _DEJAVU / "DejaVuSans.ttf"),
    ]
    manifest = {
        "format": 1, "kind": "full",
        "source": {"repo": "google/fonts",
                   "commit": "testcommit"},
        "family_count": len(families),
        "file_count": sum(len(f["faces"]) for f in families),
        "total_bytes": sum(fa["bytes"] for f in families
                           for fa in f["faces"]),
        "families": families,
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    return root


@pytest.fixture()
def synthetic_service(tmp_path, monkeypatch):
    """A typography service whose world is: the synthetic
    pack in a throwaway home + the real vendored core; host
    font roots are patched OUT, so a face can only have come
    through the registry."""
    home = tmp_path / "home"
    _write_synthetic_pack(home / "fonts" / "library")
    monkeypatch.setenv("NM_HOME", str(home))
    monkeypatch.setenv("NM_FONT_DIRS", "")
    monkeypatch.setattr(
        "nomorals.media.typography.service.default_roots",
        lambda: [])
    reset_service()
    service = get_service()
    yield service
    reset_service()


def test_registered_family_lights_up_everywhere(
        synthetic_service, monkeypatch):
    from nomorals.media_edit import studio
    from nomorals.social.chat import makers
    from nomorals.social.chat.mock_screens.render import (
        _Canvas,
    )
    from nomorals.social.chat.mock_screens.schema import (
        DeviceChrome,
        Participant,
        Screen,
    )

    service = synthetic_service
    # the registry itself resolves the new family, and the
    # emoji helper finds the pack's emoji family first
    resolved = service.font_library().resolve(
        "Lights Up Test Sans")
    assert "library" in str(resolved)
    emoji = emoji_font_path(service)
    assert emoji is not None and "library" in str(emoji)

    # the explicit-family door: two surfaces, no new code
    via_makers = makers._font(30, family="Lights Up Test Sans")
    assert via_makers.getname()[0] == "DejaVu Serif"
    via_studio = studio.find_font("Lights Up Test Sans", 30)
    assert via_studio.getname()[0] == "DejaVu Serif"

    # the policy door: naming the family in policy DATA (no
    # surface code) lights it up on the card surface and on
    # the mock-screen surface alike
    monkeypatch.setitem(
        CONTEXT_POLICIES, "card",
        ContextFontPolicy(families=("Lights Up Test Sans",),
                          intent="text"))
    monkeypatch.setitem(
        CONTEXT_POLICIES, "screen_whatsapp",
        ContextFontPolicy(families=("Lights Up Test Sans",),
                          intent="text"))
    assert makers._font(30, context="card"
                        ).getname()[0] == "DejaVu Serif"
    screen = Screen(
        platform="whatsapp", group=False, title="Ada",
        participants=[Participant("me", "You"),
                      Participant("ada", "Ada")],
        messages=[],
        chrome=DeviceChrome(style="android", clock="9:41",
                            battery=62, network="5G"))
    canvas = _Canvas(screen)
    assert canvas.font(14).getname()[0] == "DejaVu Serif"
