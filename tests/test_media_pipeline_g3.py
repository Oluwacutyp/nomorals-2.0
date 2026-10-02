"""Wave G3 — MEDIA_PIPELINE + ARCHIVE hardening tests.

Covers the G3 deliverables:

1. SIZE CAPS: ``max_mb`` (env ``NM_MEDIA_MAX_MB``, default 200) — over-cap
   inputs are refused with the cap and the actual size named; image/video/
   audio kinds get one shrink attempt first.
2. COMPRESS DEFAULTS per kind are real parameters with documented defaults
   (image 1600px/q80, video CRF 28/veryfast, audio 128k) and env overrides.
3. CLEAR ERRORS: every failure names its stage
   (download|compress|send|zip|unzip).
4. STREAMING: archive extraction never read()s a whole entry into memory.
5. ARCHIVE guards: zip-slip (several traversal shapes) and zip-bomb
   (entry-count cap + total-size cap) are exercised by real malicious zips.
6. HYGIENE: the pipeline removes its compressed intermediate after a
   successful send and keeps everything inside the workspace (no fixed
   /tmp paths).
"""

from __future__ import annotations

import inspect
import math
import os
import random
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from PIL import Image

from nomorals.core.errors import MediaError, ToolError
from nomorals.tools import archive as A
from nomorals.tools import compress as C
from nomorals.tools import media_pipeline as P


# ── fixtures ─────────────────────────────────────────────────────────────────


def _context(**over: Any) -> SimpleNamespace:
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=tempfile.mkdtemp()),
        extras={},
        router=None,
    )
    for key, value in over.items():
        setattr(ctx, key, value)
    return ctx


def _make_downloader(fixture: Path):
    def _dl(url: str, dest_dir: Any, **kw: Any) -> dict[str, Any]:
        dest = Path(dest_dir) / fixture.name
        shutil.copyfile(fixture, dest)
        return {"path": str(dest), "bytes": dest.stat().st_size,
                "title": fixture.stem, "extractor": "test"}
    return _dl


def _make_sender(captured: list):
    def _send(context: Any, platform: str, chat_id: str, path: str,
              caption: str = "") -> dict[str, Any]:
        captured.append({"platform": platform, "chat": chat_id, "path": path,
                         "caption": caption,
                         "bytes_at_send": Path(path).stat().st_size})
        return {"sent": True, "platform": platform,
                "chat": f"{platform}:{chat_id}", "message_id": "m1",
                "path": path}
    return _send


def _noisy_jpeg(path: Path, width: int = 2000, height: int = 1500,
                quality: int = 95) -> Path:
    rng = random.Random(42)
    img = Image.new("RGB", (width, height))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                 for _ in range(width * height)])
    img.save(path, "JPEG", quality=quality)
    return path


def _wav(path: Path, seconds: float = 2.0, rate: int = 44100) -> Path:
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(20000 * math.sin(2 * math.pi * 440 * i / rate)))
            for i in range(n))
        wf.writeframes(frames)
    return path


def _tiny_mp4(path: Path, seconds: int = 4) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"testsrc=size=640x480:rate=24:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k",
         "-shortest", str(path)],
        check=True, timeout=120)
    return path


# ── 1. size caps ─────────────────────────────────────────────────────────────


class SizeCapTests(unittest.TestCase):
    def test_over_cap_pdf_refused_names_cap_and_size(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        fixture = root / "doc.pdf"
        fixture.write_bytes(b"%PDF-1.4 fake " + b"x" * 5000)  # ~5 KB
        with self.assertRaises(MediaError) as cm:
            P.run_pipeline(
                ctx, "http://example.invalid/doc.pdf", "telegram", "123",
                max_mb=0.001,  # ~1 KB cap
                downloader=_make_downloader(fixture),
                sender=_make_sender([]),
            )
        msg = str(cm.exception)
        self.assertIn("size gate", msg)
        self.assertIn("0.001", msg)          # the cap
        self.assertIn("NM_MEDIA_MAX_MB", msg)  # where to change it
        self.assertIn(str(fixture.stat().st_size), msg)  # actual size

    def test_over_cap_image_shrunk_under_cap_succeeds(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        fixture = _noisy_jpeg(root / "big.jpg")
        original = fixture.stat().st_size
        # measure what the gate's aggressive pass produces, then set the cap
        # between the shrunk size and the original
        shrunk = P.compress_image_for_send(
            fixture, max_width=800, quality=70)
        self.assertTrue(shrunk["ok"], shrunk.get("reason"))
        shrunk_bytes = shrunk["new_bytes"]
        Path(shrunk["path"]).unlink()
        self.assertLess(shrunk_bytes, original)
        cap_mb = (shrunk_bytes * 1.5) / (1024 * 1024)
        self.assertLess(cap_mb * 1024 * 1024, original)
        captured: list = []
        report = P.run_pipeline(
            ctx, "http://example.invalid/big.jpg", "telegram", "123",
            max_mb=cap_mb,
            downloader=_make_downloader(fixture),
            sender=_make_sender(captured),
        )
        self.assertTrue(report["ok"])
        self.assertTrue(report["size_gate_shrunk"])
        self.assertIn("size_gate", report["stages"][0])
        self.assertEqual(len(captured), 1)

    def test_over_cap_image_still_too_big_refused_after_shrink(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        fixture = _noisy_jpeg(root / "big.jpg")
        with self.assertRaises(MediaError) as cm:
            P.run_pipeline(
                ctx, "http://example.invalid/big.jpg", "telegram", "123",
                max_mb=0.0001,  # nothing fits
                downloader=_make_downloader(fixture),
                sender=_make_sender([]),
            )
        msg = str(cm.exception)
        self.assertIn("size gate", msg)
        self.assertIn("NM_MEDIA_MAX_MB", msg)
        self.assertIn("even after one compression pass", msg)
        # the failed shrink intermediate must not litter the workspace
        leftovers = list(root.rglob("*.send.jpg"))
        self.assertEqual(leftovers, [])

    def test_max_mb_env_override(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        fixture = root / "doc.pdf"
        fixture.write_bytes(b"%PDF-1.4 fake " + b"x" * 5000)
        with mock.patch.dict(os.environ, {"NM_MEDIA_MAX_MB": "0.001"}):
            with self.assertRaises(MediaError) as cm:
                P.run_pipeline(
                    ctx, "http://example.invalid/doc.pdf", "telegram", "123",
                    downloader=_make_downloader(fixture),
                    sender=_make_sender([]),
                )
        self.assertIn("NM_MEDIA_MAX_MB", str(cm.exception))

    def test_non_positive_max_mb_rejected(self) -> None:
        ctx = _context()
        with self.assertRaises(ToolError):
            P.run_pipeline(ctx, "http://example.invalid/x.jpg", "telegram",
                           "123", max_mb=0)


# ── 2. per-kind compress defaults ────────────────────────────────────────────


class CompressDefaultsTests(unittest.TestCase):
    def test_defaults_are_documented_params(self) -> None:
        sig = inspect.signature(P.run_pipeline)
        params = sig.parameters
        self.assertEqual(params["image_max_width"].default, 1600)
        self.assertEqual(params["image_quality"].default, 80)
        self.assertEqual(params["video_crf"].default, 28)
        self.assertEqual(params["video_preset"].default, "veryfast")
        self.assertEqual(params["audio_bitrate"].default, "128k")
        # module-level defaults exist for env wiring
        self.assertEqual(P.DEFAULT_IMAGE_MAX_WIDTH, 1600)
        self.assertEqual(P.DEFAULT_IMAGE_QUALITY, 80)
        self.assertEqual(P.DEFAULT_VIDEO_CRF, 28)
        self.assertEqual(P.DEFAULT_AUDIO_BITRATE, "128k")
        self.assertEqual(P.DEFAULT_MAX_MB, 200.0)

    def test_image_default_downscale_applied(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        fixture = _noisy_jpeg(root / "wide.jpg", width=2400, height=1600)
        seen_widths: list = []

        def spy_send(context: Any, platform: str, chat_id: str, path: str,
                     caption: str = "") -> dict[str, Any]:
            with Image.open(path) as img:
                seen_widths.append(img.size[0])
            return {"sent": True, "platform": platform,
                    "chat": f"{platform}:{chat_id}", "message_id": "m1"}

        report = P.run_pipeline(
            ctx, "http://example.invalid/wide.jpg", "telegram", "123",
            downloader=_make_downloader(fixture), sender=spy_send)
        self.assertTrue(report["ok"])
        self.assertTrue(report["compressed"])
        self.assertEqual(seen_widths, [1600])  # default max width applied
        compress = report["stages"][1]
        self.assertTrue(compress["ok"])
        self.assertIn("pillow-reencode", compress["method"])

    def test_video_crf_threaded_to_compress_file(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        fixture = root / "clip.mp4"
        fixture.write_bytes(b"\x00" * 2048)
        calls: list = []
        real = C.compress_file

        def spy(path: Any, **kw: Any) -> dict[str, Any]:
            calls.append(kw)
            return real(path)

        with mock.patch.object(C, "compress_file", spy):
            P.run_pipeline(
                ctx, "http://example.invalid/clip.mp4", "telegram", "123",
                video_crf=18, downloader=_make_downloader(fixture),
                sender=_make_sender([]))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].get("video_crf"), 18)

    def test_audio_compress_uses_bitrate_default_and_override(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")
        root = Path(tempfile.mkdtemp())
        src = _wav(root / "tone.wav")
        original = src.stat().st_size
        report = P.compress_audio_for_send(src)  # default 128k
        self.assertTrue(report["ok"], report.get("reason"))
        self.assertIn("128k", report["method"])
        self.assertTrue(report["path"].endswith(".send.mp3"))
        self.assertLess(report["new_bytes"], original)
        Path(report["path"]).unlink()
        report2 = P.compress_audio_for_send(src, bitrate="64k")
        self.assertTrue(report2["ok"], report2.get("reason"))
        self.assertIn("64k", report2["method"])
        Path(report2["path"]).unlink()

    def test_audio_pipeline_stage_uses_ffmpeg(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        fixture = _wav(root / "tone.wav")
        captured: list = []
        report = P.run_pipeline(
            ctx, "http://example.invalid/tone.wav", "telegram", "123",
            downloader=_make_downloader(fixture),
            sender=_make_sender(captured))
        self.assertTrue(report["ok"])
        compress = report["stages"][1]
        self.assertTrue(compress["ok"], compress.get("reason"))
        self.assertIn("ffmpeg-audio", compress["method"])
        self.assertTrue(captured[0]["path"].endswith(".send.mp3"))

    def test_compress_file_video_dest_never_clobbers_source(self) -> None:
        """Regression: compress_file used dest == src for video, so ffmpeg
        truncated the input mid-read."""
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")
        root = Path(tempfile.mkdtemp())
        src = _tiny_mp4(root / "clip.mp4")
        before = src.stat().st_size
        report = C.compress_file(src)
        self.assertTrue(report["ok"], report.get("reason"))
        self.assertNotEqual(Path(report["path"]), src)
        self.assertEqual(src.stat().st_size, before)  # source untouched
        self.assertTrue(Path(report["path"]).is_file())


# ── 3. stage-named errors ────────────────────────────────────────────────────


class StageErrorTests(unittest.TestCase):
    def test_download_error_names_stage(self) -> None:
        ctx = _context()

        def boom(url: str, dest_dir: Any, **kw: Any) -> dict[str, Any]:
            raise MediaError("network down")

        with self.assertRaises(MediaError) as cm:
            P.run_pipeline(ctx, "http://example.invalid/x.jpg", "telegram",
                           "123", downloader=boom, sender=_make_sender([]))
        self.assertIn("download", str(cm.exception))

    def test_compress_error_names_stage(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        fixture = root / "pic.jpg"
        _noisy_jpeg(fixture, width=64, height=48)
        with mock.patch.object(
                P, "_compress_stage",
                side_effect=RuntimeError("pillow exploded")):
            with self.assertRaises(MediaError) as cm:
                P.run_pipeline(
                    ctx, "http://example.invalid/pic.jpg", "telegram", "123",
                    downloader=_make_downloader(fixture),
                    sender=_make_sender([]))
        self.assertIn("compress", str(cm.exception))
        self.assertIn("pillow exploded", str(cm.exception))

    def test_send_error_names_stage(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        fixture = root / "pic.jpg"
        _noisy_jpeg(fixture, width=64, height=48)

        def bad_send(context: Any, platform: str, chat_id: str, path: str,
                     caption: str = "") -> dict[str, Any]:
            raise ToolError("gateway offline")

        with self.assertRaises(ToolError) as cm:
            P.run_pipeline(
                ctx, "http://example.invalid/pic.jpg", "telegram", "123",
                compress=False, downloader=_make_downloader(fixture),
                sender=bad_send)
        self.assertIn("send", str(cm.exception))
        self.assertIn("gateway offline", str(cm.exception))

    def test_zip_errors_name_stage(self) -> None:
        ctx = _context()
        with self.assertRaises(ToolError) as cm:
            A.zip_create(ctx, [])
        self.assertIn("zip", str(cm.exception))

    def test_unzip_errors_name_stage(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        (root / "plain.txt").write_text("not a zip")
        with self.assertRaises(ToolError) as cm:
            A.zip_extract(ctx, "plain.txt")
        self.assertIn("unzip", str(cm.exception))


# ── 4. streaming extraction ──────────────────────────────────────────────────


class StreamingTests(unittest.TestCase):
    def test_extract_never_reads_whole_entry(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        payload = bytes(random.Random(7).randrange(256) for _ in range(3_000_000))
        zpath = root / "big.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("blob.bin", payload)
        whole_reads: list = []
        real_read = zipfile.ZipExtFile.read

        def spy_read(self: Any, size: Any = -1) -> bytes:
            if size in (-1, None):
                whole_reads.append(size)
            return real_read(self, size)

        with mock.patch.object(zipfile.ZipExtFile, "read", spy_read):
            out = A.zip_extract(ctx, "big.zip", destination="out")
        self.assertEqual(whole_reads, [])
        self.assertEqual(out["files"], 1)
        self.assertEqual((Path(out["destination"]) / "blob.bin").read_bytes(),
                         payload)

    def test_pipeline_downloads_inside_workspace(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        fixture = root / "notes.txt"
        fixture.write_text("hello\n" * 10)
        report = P.run_pipeline(
            ctx, "http://example.invalid/notes.txt", "telegram", "123",
            downloader=_make_downloader(fixture), sender=_make_sender([]))
        self.assertTrue(
            Path(report["downloaded_path"]).resolve().is_relative_to(
                root.resolve()))


# ── 5. zip-slip ──────────────────────────────────────────────────────────────


class ZipSlipTests(unittest.TestCase):
    EVIL = ["../escape.txt", "/absolute.txt", "sub/../../escape2.txt",
            "..\\win.txt", "a/b/../../../escape3.txt"]

    def _evil_zip(self, root: Path, name: str, entries: list[str]) -> None:
        with zipfile.ZipFile(root / name, "w") as zf:
            for i, entry in enumerate(entries):
                zf.writestr(entry, f"pwned-{i}")

    def test_zip_slip_shapes_rejected(self) -> None:
        for entry in self.EVIL:
            with self.subTest(entry=entry):
                ctx = _context()
                root = Path(ctx.settings.workspace_dir)
                self._evil_zip(root, "evil.zip", [entry])
                with self.assertRaises(ToolError) as cm:
                    A.zip_extract(ctx, "evil.zip")
                self.assertIn("zip-slip", str(cm.exception))
                self.assertFalse((root / "escape.txt").exists())
                self.assertFalse((root / "escape2.txt").exists())
                self.assertFalse((root / "escape3.txt").exists())
                self.assertFalse((root / "absolute.txt").exists())
                self.assertFalse((root / "win.txt").exists())

    def test_zip_slip_does_not_write_before_rejecting(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        self._evil_zip(root, "evil.zip", ["good.txt", "../escape.txt"])
        with self.assertRaises(ToolError):
            A.zip_extract(ctx, "evil.zip")
        self.assertFalse((root / "escape.txt").exists())
        # the good entry must not have been written either (validate first)
        self.assertFalse((root / "extracted").exists())


# ── 6. zip-bomb guards ───────────────────────────────────────────────────────


class ZipBombTests(unittest.TestCase):
    def test_entry_count_cap_rejects(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        zpath = root / "many.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            for i in range(6):
                zf.writestr(f"f{i}.txt", "x")
        with mock.patch.object(A, "MAX_EXTRACT_FILES", 5):
            with self.assertRaises(ToolError) as cm:
                A.zip_extract(ctx, "many.zip")
        msg = str(cm.exception)
        self.assertIn("unzip", msg)
        self.assertIn("5", msg)
        self.assertIn("zip-bomb", msg)

    def test_total_size_cap_rejects(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        zpath = root / "fat.zip"
        rng = random.Random(11)
        with zipfile.ZipFile(zpath, "w") as zf:
            for i in range(3):
                # random bytes: declared uncompressed size ~50 each
                zf.writestr(f"f{i}.bin",
                            bytes(rng.randrange(256) for _ in range(50)))
        with mock.patch.object(A, "MAX_EXTRACT_BYTES", 100):
            with self.assertRaises(ToolError) as cm:
                A.zip_extract(ctx, "fat.zip")
        msg = str(cm.exception)
        self.assertIn("unzip", msg)
        self.assertIn("zip-bomb", msg)
        self.assertFalse((root / "extracted").exists())

    def test_caps_are_env_overridable(self) -> None:
        import importlib

        try:
            with mock.patch.dict(os.environ, {"NM_ARCHIVE_MAX_EXTRACT_FILES": "7",
                                              "NM_ARCHIVE_MAX_EXTRACT_MB": "1"}):
                mod = importlib.reload(A)
                self.assertEqual(mod.MAX_EXTRACT_FILES, 7)
                self.assertEqual(mod.MAX_EXTRACT_BYTES, 1024 * 1024)
        finally:
            importlib.reload(A)  # restore real env-derived constants
        self.assertEqual(A.MAX_EXTRACT_FILES, 10_000)
        self.assertEqual(A.MAX_EXTRACT_BYTES, 2048 * 1024 * 1024)


# ── 7. hygiene ───────────────────────────────────────────────────────────────


class HygieneTests(unittest.TestCase):
    def test_intermediate_removed_after_successful_send(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        fixture = _noisy_jpeg(root / "big.jpg")
        captured: list = []
        report = P.run_pipeline(
            ctx, "http://example.invalid/big.jpg", "telegram", "123",
            image_max_width=800, image_quality=60,
            downloader=_make_downloader(fixture),
            sender=_make_sender(captured))
        self.assertTrue(report["ok"])
        self.assertTrue(report["compressed"])
        sent_path = captured[0]["path"]
        self.assertTrue(sent_path.endswith(".send.jpg"))
        # intermediate gone, original download kept
        self.assertFalse(Path(sent_path).exists())
        self.assertTrue(Path(report["downloaded_path"]).is_file())
        self.assertEqual(list(root.rglob("*.send.jpg")), [])

    def test_intermediate_kept_and_named_on_send_failure(self) -> None:
        ctx = _context()
        root = Path(ctx.settings.workspace_dir)
        fixture = _noisy_jpeg(root / "big.jpg")

        def bad_send(context: Any, platform: str, chat_id: str, path: str,
                     caption: str = "") -> dict[str, Any]:
            raise ToolError("gateway offline")

        with self.assertRaises(ToolError) as cm:
            P.run_pipeline(
                ctx, "http://example.invalid/big.jpg", "telegram", "123",
                image_max_width=800, image_quality=60,
                downloader=_make_downloader(fixture), sender=bad_send)
        self.assertIn("gateway offline", str(cm.exception))
        leftovers = list(root.rglob("*.send.jpg"))
        self.assertEqual(len(leftovers), 1)
        self.assertIn(str(leftovers[0]), str(cm.exception))


if __name__ == "__main__":
    unittest.main()
