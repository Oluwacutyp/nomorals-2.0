"""Pass #36 — image generation + vision surfaces.

Defects pinned here (all reproduced on pristine HEAD cebdeaa7):

D1  ``/look`` called a tool named ``vision_screen`` that no longer exists
    in the registry — every /look reply was "vision failed: unknown tool".
D2  The brain-facing native image tools pass a *directory* as ``save_to``
    but ``Studio._finalize`` treated it as a file path, so every native
    generation through the tools died in PIL ("unknown file extension").
D3  ``aspect`` advertised by ``image_generate``/``image_batch`` was
    silently dropped by ``Studio.generate`` (not a PipelineConfig field,
    never mapped to width/height): 16:9 rendered as 64x64 square.
D4  ``image_outpaint``'s ``direction`` was echoed in the result but never
    translated into canvas extensions — the outpaint extended nothing.
D5  ``Studio._finalize`` name collisions: batch images sharing one file
    ``save_to`` overwrote each other; default names had 1-second
    granularity so back-to-back generations overwrote each other (and
    the sidecar prompt).
D6  ``/image`` generation saved to ``img_<unix-seconds>.png`` — two
    generations in the same second (any chats) overwrote each other.
D7  ``vision_to_base64`` labelled bytes with the default/caller mime
    ("image/png") regardless of content — JPEG bytes came back as
    ``data:image/png;base64,/9j/...``.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


# ── helpers ──────────────────────────────────────────────────────────────


def _png(color=(255, 0, 0), size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def _jpg(color=(0, 255, 0), size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


class _StubUsage:
    def as_dict(self) -> dict:
        return {"total_tokens": 7}


class _StubResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.provider = "stub-provider"
        self.model = "stub-vision-1"
        self.usage = _StubUsage()


class _StubRouter:
    """Router whose describe_image answers by instruction content."""

    def describe_image(self, payload: bytes, instruction: str) -> _StubResponse:
        if "Transcribe" in instruction:
            return _StubResponse("HELLO SCREEN TEXT")
        return _StubResponse("A stub screen description.")


def _vision_registry(tmp_path: Path):
    from nomorals.tools import vision as vision_tools
    from nomorals.tools.registry import ToolRegistry

    settings = SimpleNamespace(workspace_dir=str(tmp_path))
    context = SimpleNamespace(router=_StubRouter(), settings=settings,
                              extras={})
    reg = ToolRegistry()
    reg.context = context
    vision_tools.register(reg)
    return reg, context


def _fake_studio_pipeline(studio, captured: dict | None = None):
    class FakePipe:
        def generate(self, prompt, cfg):
            if captured is not None:
                captured.update(width=cfg.width, height=cfg.height,
                                seed=cfg.seed, steps=cfg.steps)
            return [Image.new("RGB", (8, 8), (10, 20, 30))]

    studio.pipeline = lambda: FakePipe()
    return studio


# ── D1: vision_screen exists and serves the /look contract ───────────────


def test_vision_screen_registered():
    from nomorals.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register_builtins()
    assert "vision_screen" in reg._tools


def test_vision_screen_returns_look_contract(tmp_path):
    reg, _ctx = _vision_registry(tmp_path)
    (tmp_path / "shot.png").write_bytes(_png())
    outcome = reg.call("vision_screen", path="shot.png",
                       focus="the error banner")
    assert outcome.ok, getattr(outcome.error, "message", outcome.error)
    value = outcome.value
    assert value["description"] == "A stub screen description."
    assert value["ocr_text"] == "HELLO SCREEN TEXT"
    assert value["format"] == "png"
    assert value["width"] == 8 and value["height"] == 8
    assert value["provider"] == "stub-provider"


def test_control_look_end_to_end(tmp_path):
    """The /look chat handler itself: was 'vision failed: unknown tool'."""
    from nomorals.agents.partner.runtime_voice import RuntimeVoiceMixin
    from nomorals.tools.registry import ToolRegistry

    reg, context = _vision_registry(tmp_path)
    (tmp_path / "shot.png").write_bytes(_png())
    context.tools = reg

    runtime = object.__new__(RuntimeVoiceMixin)
    runtime.context = context
    runtime._ref_from_key = lambda key: SimpleNamespace(platform="test")
    runtime._send_long_checked = lambda platform, chat, text: text

    reply = runtime._control_look("shot.png", chat_key="chat:1")
    assert "vision failed" not in reply
    assert "A stub screen description." in reply
    assert "HELLO SCREEN TEXT" in reply
    assert "stub-provider" in reply


# ── D2/D3/D5: Studio save_to + aspect + naming ───────────────────────────


def test_studio_generate_save_to_directory(tmp_path):
    from nomorals.media.imggen.studio import Studio

    studio = _fake_studio_pipeline(Studio())
    out_dir = tmp_path / "images"
    out_dir.mkdir()
    paths = studio.generate("a cat", save_to=str(out_dir))
    assert len(paths) == 1
    target = Path(paths[0])
    assert target.parent == out_dir
    assert target.is_file()
    with Image.open(target) as im:
        assert im.format == "PNG"
    assert (str(target) + ".txt") and Path(str(target) + ".txt").read_text() == "a cat"


def test_studio_generate_aspect_ratio_applied():
    from nomorals.media.imggen.studio import Studio

    captured: dict = {}
    studio = _fake_studio_pipeline(Studio(), captured)
    # keep the real generate() (that's what translates aspect) but stub
    # out the disk write
    studio._finalize = lambda images, prompt, save_to: ["x.png"]
    studio.generate("a cat", aspect_ratio="16:9")
    assert captured["width"] > captured["height"]


def test_studio_generate_explicit_size_wins_over_aspect():
    from nomorals.media.imggen.studio import Studio

    captured: dict = {}
    studio = _fake_studio_pipeline(Studio(), captured)
    studio._finalize = lambda images, prompt, save_to: ["x.png"]
    studio.generate("a cat", aspect_ratio="16:9", width=64, height=64)
    assert (captured["width"], captured["height"]) == (64, 64)


def test_finalize_batch_file_target_keeps_every_image(tmp_path):
    from nomorals.media.imggen.studio import Studio

    studio = Studio()
    target = tmp_path / "one.png"
    imgs = [Image.new("RGB", (4, 4), (255, 0, 0)),
            Image.new("RGB", (4, 4), (0, 0, 255))]
    paths = studio._finalize(imgs, "p", str(target))
    assert len(paths) == 2
    assert paths[0] != paths[1]
    pixels = []
    for p in paths:
        assert Path(p).is_file()
        with Image.open(p) as im:
            pixels.append(im.convert("RGB").getpixel((0, 0)))
    assert pixels[0] == (255, 0, 0)
    assert pixels[1] == (0, 0, 255)


def test_finalize_default_names_do_not_collide(tmp_path, monkeypatch):
    from nomorals.media.imggen.studio import Studio

    monkeypatch.setenv("HOME", str(tmp_path))
    studio = Studio()
    first = studio._finalize([Image.new("RGB", (4, 4), (255, 0, 0))],
                             "first prompt", None)
    second = studio._finalize([Image.new("RGB", (4, 4), (0, 0, 255))],
                              "second prompt", None)
    assert first[0] != second[0]
    assert Path(first[0]).is_file() and Path(second[0]).is_file()
    with Image.open(first[0]) as im:
        assert im.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
    assert Path(first[0] + ".txt").read_text() == "first prompt"
    assert Path(second[0] + ".txt").read_text() == "second prompt"


def test_tool_generate_writes_into_out_dir(tmp_path, monkeypatch):
    """The image_generate tool end-to-end with a stubbed Studio pipeline."""
    from nomorals.media.imggen import studio as studio_mod

    captured: dict = {}
    real_studio = studio_mod.Studio

    def factory(*a, **k):
        return _fake_studio_pipeline(real_studio(), captured)

    monkeypatch.setattr(studio_mod, "Studio", factory)
    context = SimpleNamespace(workspace=str(tmp_path))
    from nomorals.tools.registry import ToolRegistry
    from nomorals.tools import media_gen

    reg = ToolRegistry()
    reg.context = context
    media_gen.register(reg)
    outcome = reg.call("image_generate", prompt="a lighthouse",
                       aspect="16:9")
    assert outcome.ok, getattr(outcome.error, "message", outcome.error)
    paths = outcome.value["paths"]
    assert len(paths) == 1 and Path(paths[0]).is_file()
    assert Path(paths[0]).parent == tmp_path / "generated" / "images"
    assert captured["width"] > captured["height"]


# ── D4: image_outpaint direction ─────────────────────────────────────────


def test_outpaint_direction_reaches_studio(tmp_path, monkeypatch):
    from nomorals.media.imggen import studio as studio_mod

    calls: dict = {}

    class RecordingStudio:
        def outpaint(self, image_path, prompt, **kwargs):
            calls.update(kwargs)
            calls["image_path"] = image_path
            return ["/tmp/out.png"]

    monkeypatch.setattr(studio_mod, "Studio", RecordingStudio)
    src = tmp_path / "in.png"
    Image.new("RGB", (40, 20), (1, 2, 3)).save(src)
    context = SimpleNamespace(
        workspace=str(tmp_path),
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))

    from nomorals.tools.registry import ToolRegistry
    from nomorals.tools import media_gen

    reg = ToolRegistry()
    reg.context = context
    media_gen.register(reg)
    outcome = reg.call("image_outpaint", image_path="in.png",
                       prompt="more sky", direction="left")
    assert outcome.ok, getattr(outcome.error, "message", outcome.error)
    assert calls.get("left") == 40  # extended by the image's own width
    assert not calls.get("right") and not calls.get("top") \
        and not calls.get("bottom")


def test_outpaint_unknown_direction_rejected(tmp_path, monkeypatch):
    from nomorals.media.imggen import studio as studio_mod

    monkeypatch.setattr(studio_mod, "Studio", object)
    src = tmp_path / "in.png"
    Image.new("RGB", (10, 10)).save(src)
    context = SimpleNamespace(
        workspace=str(tmp_path),
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    from nomorals.tools.registry import ToolRegistry
    from nomorals.tools import media_gen

    reg = ToolRegistry()
    reg.context = context
    media_gen.register(reg)
    outcome = reg.call("image_outpaint", image_path="in.png",
                       prompt="more", direction="sideways")
    assert not outcome.ok


# ── D6: /image generate filename isolation ───────────────────────────────


def test_image_generate_replies_name_distinct_files(tmp_path, monkeypatch):
    import nomorals.media_edit.generate as gen_mod
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    class FakeBackend:
        def generate(self, prompt):
            return [Image.new("RGB", (6, 6), (9, 9, 9))]

        def describe(self):
            return "fake-backend"

    monkeypatch.setattr(gen_mod, "get_backend", lambda *a, **k: FakeBackend())
    settings = SimpleNamespace(
        resolve=lambda rel: str(tmp_path / rel))
    runtime = object.__new__(RuntimeMediaMixin)
    runtime.context = SimpleNamespace(settings=settings)

    reply1 = runtime._image_generate("first prompt")
    reply2 = runtime._image_generate("second prompt")
    saved1 = [ln for ln in reply1.splitlines() if "saved:" in ln]
    saved2 = [ln for ln in reply2.splitlines() if "saved:" in ln]
    assert saved1 and saved2
    assert saved1 != saved2
    gen_dir = tmp_path / "data" / "media" / "generated"
    assert len(list(gen_dir.glob("*.png"))) == 2


# ── D7: vision_to_base64 sniffs the real format ──────────────────────────


def test_vision_to_base64_labels_jpeg_as_jpeg(tmp_path):
    reg, _ctx = _vision_registry(tmp_path)
    (tmp_path / "photo.jpg").write_bytes(_jpg())
    outcome = reg.call("vision_to_base64", path="photo.jpg")
    assert outcome.ok
    assert outcome.value["mime"] == "image/jpeg"
    assert outcome.value["data_url"].startswith("data:image/jpeg;base64,/9j/")


def test_vision_to_base64_labels_png_as_png(tmp_path):
    reg, _ctx = _vision_registry(tmp_path)
    (tmp_path / "photo.png").write_bytes(_png())
    outcome = reg.call("vision_to_base64", path="photo.png")
    assert outcome.ok
    assert outcome.value["mime"] == "image/png"
    assert outcome.value["data_url"].startswith("data:image/png;base64,")


def test_vision_to_base64_fallback_mime_for_unknown_bytes(tmp_path):
    reg, _ctx = _vision_registry(tmp_path)
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02\x03 not an image")
    outcome = reg.call("vision_to_base64", path="blob.bin",
                       mime="application/octet-stream")
    assert outcome.ok
    assert outcome.value["mime"] == "application/octet-stream"
