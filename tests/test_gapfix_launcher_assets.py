"""Launcher asset pipeline (lane 129, census §3B.8): the pictures on
an app's details page.

Pattern port of the ArtCraft Launcher's picture handling — upstream
``crates/core/src/details.rs`` (the pinned-tag URL shape, the JPEG
checks, the budgets), ``crates/native/src/screenshots.rs`` (fetch
once, cache, re-validate a saved copy) and ``crates/native/src/
paths.rs`` (the cache layout), plus the publishing rules and the
per-picture terms in the pictures repository's README and
``ATTRIBUTION.md``. Devon's own code throughout: no upstream file or
text is vendored or copied, and no upstream JPEG is vendored at all —
the launcher downloads pictures at runtime, exactly as upstream
does, so the repository stays small.

Failing-first: every test here names a real behaviour of the port.
The gap this fills: Devon's launcher organ (lane 112) had the
verify-then-install pipeline, the catalog and the trust rules, but
no picture handling at all — no shot catalog, no fetch, no JPEG
validation, no cache, no manifest, no per-picture terms. That is
what these tests pin down.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.os.launcher.screenshots import (
    IDEAL_MAX_BYTES,
    MAX_SCREENSHOT_BYTES,
    MAX_SCREENSHOT_PIXELS,
    SCREENSHOTS_BASE,
    SCREENSHOTS_REV,
    SHOTS,
    NetworkFailure,
    PictureUnavailable,
    ScreenshotError,
    caption_for,
    check_use,
    fetch_all,
    jpeg_dimensions,
    looks_like_jpeg,
    prepare_picture,
    read_manifest,
    resize_picture,
    saved_copy,
    screenshot_path,
    screenshot_url,
    screenshot_within_budget,
    strip_metadata,
    terms_for,
    write_atomic,
)
from nomorals.tools.registry import ToolRegistry

CLONE = (Path.home() / "workspace" / "echelon-mining" / "repos"
         / "storytold" / "craft-launcher-assets")


def _jpeg(width: int, height: int, *, pad: int = 0,
          marker: int = 0xC0) -> bytes:
    """A minimal JPEG: SOI, an APP0 segment, then a SOFn frame
    declaring ``width`` x ``height`` (the upstream test helper's
    shape, rebuilt here)."""
    data = bytearray([0xFF, 0xD8])  # SOI
    data += bytes([0xFF, 0xE0, 0x00, 0x04, 0x00, 0x00])  # APP0
    data += bytes([0xFF, marker, 0x00, 0x11, 0x08])  # SOFn, len 17
    data += height.to_bytes(2, "big") + width.to_bytes(2, "big")
    data += bytes([0x03, 0x01, 0x11, 0x00])  # one component
    data += bytes(i % 251 for i in range(pad))
    return bytes(data)


def _registry() -> ToolRegistry:
    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    return registry


def _call(registry, **kwargs):
    outcome = registry.call("craftlauncher", **kwargs)
    if hasattr(outcome, "value"):
        return outcome.value
    return outcome


# ---------------------------------------------------------------------------
# JPEG checks (upstream details.rs: looks_like_jpeg, jpeg_dimensions,
# screenshot_within_budget)
# ---------------------------------------------------------------------------


def test_only_jpeg_files_are_accepted() -> None:
    assert looks_like_jpeg(bytes([0xFF, 0xD8, 0xFF, 0xE0, 0, 0x10]))
    assert not looks_like_jpeg(b"")
    assert not looks_like_jpeg(bytes([0xFF, 0xD8]))
    assert not looks_like_jpeg(b"<!DOCTYPE html>")
    # A PNG's signature.
    assert not looks_like_jpeg(
        bytes([0x89, ord("P"), ord("N"), ord("G"), 0x0D, 0x0A, 0x1A, 0x0A]))


def test_reads_jpeg_frame_dimensions() -> None:
    assert jpeg_dimensions(_jpeg(1200, 750)) == (1200, 750)
    assert jpeg_dimensions(_jpeg(65535, 65535)) == (65535, 65535)
    # Not a JPEG, or no frame header present.
    assert jpeg_dimensions(b"<html>") is None
    assert jpeg_dimensions(bytes([0xFF, 0xD8])) is None
    # Progressive (SOF2) reads the same way.
    assert jpeg_dimensions(_jpeg(800, 600, marker=0xC2)) == (800, 600)


def test_unsupported_frame_marker_is_rejected() -> None:
    # Lossless (SOF3) is rejected (CWE-789).
    assert jpeg_dimensions(_jpeg(100, 100, marker=0xC3)) is None


def test_multiple_frame_markers_are_rejected() -> None:
    data = bytearray([0xFF, 0xD8])
    for width, height in ((100, 100), (9000, 9000)):
        data += bytes([0xFF, 0xC0, 0x00, 0x0B, 0x08])
        data += height.to_bytes(2, "big") + width.to_bytes(2, "big")
        data += bytes([0x01, 0x00, 0x00, 0x00])
    assert jpeg_dimensions(bytes(data)) is None


def test_oversized_pictures_are_refused_before_decoding() -> None:
    assert screenshot_within_budget(_jpeg(1200, 750))
    assert screenshot_within_budget(_jpeg(8192, 8192))
    # The decompression-bomb header: a tiny file declaring billions
    # of pixels.
    assert not screenshot_within_budget(_jpeg(65535, 65535))
    assert not screenshot_within_budget(_jpeg(8193, 8192))
    assert not screenshot_within_budget(_jpeg(0, 100))
    # A non-JPEG (or a JPEG with no readable frame) is refused
    # rather than trusted.
    assert not screenshot_within_budget(b"<html>")
    assert MAX_SCREENSHOT_PIXELS == 8192 * 8192


# ---------------------------------------------------------------------------
# Addresses come only from the page's own pictures (details.rs)
# ---------------------------------------------------------------------------


def test_picture_addresses_come_only_from_the_pages_own_pictures() -> None:
    assert screenshot_url("photocraft", "demo") == (
        "https://raw.githubusercontent.com/storytold/"
        "craft-launcher-assets/v1/photocraft/demo.jpg")
    for key in ["", "nope", "../vectorcraft/neon", "demo/../masks",
                "demo.jpg", "/etc/passwd", "demo?x=1", "demo#"]:
        assert screenshot_url("photocraft", key) is None, key
    assert screenshot_url("no-such-app", "demo") is None
    # Every picture of every page has an address of the one shape,
    # with nothing to escape.
    for app, shots in SHOTS.items():
        for shot in shots:
            url = screenshot_url(app, shot.key)
            assert url is not None
            path = url[len(f"{SCREENSHOTS_BASE}/{SCREENSHOTS_REV}/"):]
            assert path == f"{app}/{shot.key}.jpg"
            assert all(ch.isascii() and (ch.islower() or ch.isdigit()
                       or ch in "/-.")
                       for ch in path), path


def test_shot_catalog_covers_eleven_apps_and_39_pictures() -> None:
    assert len(SHOTS) == 11
    assert sum(len(shots) for shots in SHOTS.values()) == 39
    for app, shots in SHOTS.items():
        assert shots, app
        assert len({shot.key for shot in shots}) == len(shots), app


def test_shot_catalog_matches_the_pictures_repository() -> None:
    """Every catalogued shot is a real asset file, and every asset
    file is catalogued — the catalog and the repository agree."""
    if not (CLONE / "ATTRIBUTION.md").is_file():
        pytest.skip("pictures repository clone is not on disk")
    on_disk = set()
    for app_dir in CLONE.iterdir():
        if not app_dir.is_dir() or app_dir.name.startswith("."):
            continue
        for picture in app_dir.glob("*.jpg"):
            on_disk.add((app_dir.name, picture.stem))
    catalogued = {(app, shot.key)
                  for app, shots in SHOTS.items() for shot in shots}
    assert catalogued == on_disk
    # And every asset file is really 1200x750 and under the 150 KB
    # ideal, as the publishing rules require.
    for app, key in sorted(on_disk):
        data = (CLONE / app / f"{key}.jpg").read_bytes()
        assert jpeg_dimensions(data) == (1200, 750), f"{app}/{key}"
        assert len(data) <= IDEAL_MAX_BYTES, f"{app}/{key}"
        assert len(data) <= MAX_SCREENSHOT_BYTES


def test_devon_component_mapping_points_at_real_modules() -> None:
    import importlib

    for app, shots in SHOTS.items():
        for shot in shots:
            if shot.devon_component is None:
                continue
            module = importlib.import_module(shot.devon_component)
            assert module is not None, f"{app}/{shot.key}"


def test_captions_derive_from_keys() -> None:
    assert caption_for("layer-styles") == "Layer styles"
    assert caption_for("demo") == "Demo"
    assert caption_for("3d") == "3d"


# ---------------------------------------------------------------------------
# Cache (paths.rs + screenshots.rs: saved_copy, write_atomic)
# ---------------------------------------------------------------------------


def test_saved_copy_is_revalidated_on_every_read(tmp_path: Path) -> None:
    path = tmp_path / "pic.jpg"
    assert saved_copy(path) is None  # nothing saved
    path.write_bytes(b"<html>oops</html>")
    assert saved_copy(path) is None  # not a JPEG
    path.write_bytes(b"\xff" * (MAX_SCREENSHOT_BYTES + 1))
    assert saved_copy(path) is None  # too big
    path.write_bytes(_jpeg(65535, 65535))
    assert saved_copy(path) is None  # over the pixel budget
    good = _jpeg(1200, 750)
    path.write_bytes(good)
    assert saved_copy(path) == good


def test_write_atomic_replaces_and_keeps_no_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "pic.jpg"
    write_atomic(path, b"one")
    assert path.read_bytes() == b"one"
    write_atomic(path, b"two")
    assert path.read_bytes() == b"two"
    assert list(tmp_path.rglob(".screenshot-*")) == []


def test_screenshot_path_is_scoped_by_revision(tmp_path: Path) -> None:
    path = screenshot_path(tmp_path, "photocraft", "demo")
    assert path == (tmp_path / "cache" / "screenshots" / "v1"
                    / "photocraft" / "demo.jpg")
    assert screenshot_path(tmp_path, "photocraft", "demo",
                           rev="v2") != path


# ---------------------------------------------------------------------------
# Terms (ATTRIBUTION.md: every asset gets an attribution row)
# ---------------------------------------------------------------------------


def test_terms_cover_every_picture() -> None:
    for app, shots in SHOTS.items():
        for shot in shots:
            row = terms_for(app, shot.key)
            assert row is not None, f"{app}/{shot.key}"
            assert row["ui_licence"], f"{app}/{shot.key}"
            assert row["artwork_licence"], f"{app}/{shot.key}"
            assert row["source"], f"{app}/{shot.key}"
    assert terms_for("photocraft", "demo")["ui_licence"] == "Apache-2.0"
    assert terms_for("vectorcraft", "neon")["ui_licence"] == (
        "MIT OR Apache-2.0")
    assert terms_for("nope", "demo") is None
    assert terms_for("photocraft", "nope") is None


def test_use_without_an_attribution_row_is_refused() -> None:
    assert check_use("photocraft", "demo", "details-page")["ok"] is True
    refused = check_use("nope", "demo", "details-page")
    assert refused["ok"] is False
    assert "no attribution row" in refused["reason"]
    refused = check_use("photocraft", "demo", "")
    assert refused["ok"] is False
    assert "purpose" in refused["reason"]
    # Every catalogued picture may be used for its stated purpose:
    # the details page.
    for app, shots in SHOTS.items():
        for shot in shots:
            assert check_use(app, shot.key, "details-page")["ok"] is True


# ---------------------------------------------------------------------------
# Fetch (screenshots.rs: fetch_all)
# ---------------------------------------------------------------------------


class _Server:
    """A loopback pictures server: path -> (status, bytes)."""

    def __init__(self, routes: dict[str, tuple[int, bytes]]) -> None:
        self.routes = routes
        self.hits: list[str] = []
        self._httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> _Server:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib name
                server.hits.append(self.path)
                status, body = server.routes.get(
                    self.path, (404, b"not found"))
                self.send_response(status)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()

    @property
    def base(self) -> str:
        assert self._httpd is not None
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"


def _fixture_server() -> dict[str, tuple[int, bytes]]:
    """Every photocraft picture served; vectorcraft/neon missing."""
    routes: dict[str, tuple[int, bytes]] = {}
    for key in ("demo", "adjustments", "layer-styles", "masks"):
        routes[f"/v1/photocraft/{key}.jpg"] = (
            200, _jpeg(1200, 750, pad=5000))
    return routes


def test_fetch_downloads_caches_and_manifests(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _Server(_fixture_server()) as server:
        result = fetch_all(home, "photocraft", base_url=server.base,
                           fetcher=_loopback_fetcher(server.base))
        assert result["fetched"] == 4
        assert result["cached"] == 0
        assert result["failed"] == 0
        assert all(s["status"] == "downloaded"
                   for s in result["shots"])
        for shot in result["shots"]:
            path = screenshot_path(home, "photocraft", shot["key"])
            assert path.is_file()
            assert saved_copy(path) is not None
        manifest = read_manifest(home)
        assert len(manifest["assets"]) == 4
        row = manifest["assets"]["photocraft/demo"]
        assert row["ui_licence"] == "Apache-2.0"
        assert row["artwork_licence"] == "public domain"
        assert row["width"] == 1200 and row["height"] == 750
        assert row["used_for"] == "download"
        assert row["sha256"] == hashlib.sha256(
            _jpeg(1200, 750, pad=5000)).hexdigest()
        # Second fetch: everything comes from the cache, no network.
        hits = len(server.hits)
        result = fetch_all(home, "photocraft", base_url=server.base,
                           fetcher=_loopback_fetcher(server.base))
        assert result["cached"] == 4
        assert result["fetched"] == 0
        assert len(server.hits) == hits


def _loopback_fetcher(base: str):
    """A fetcher over the loopback server (the tool's HttpClient
    fetcher, exercised end to end through the tool below)."""
    from nomorals.core.http import HttpClient

    client = HttpClient(allow_private_ips=True, force_direct=True)

    def fetch(url: str) -> bytes | None:
        from nomorals.core.errors import http_status_of

        try:
            resp = client.get(url)
        except Exception as exc:  # noqa: BLE001 - mapped below
            if http_status_of(exc) == 404:
                return None
            raise NetworkFailure(str(exc)) from exc
        if resp.status == 404:
            return None
        if not resp.ok:
            raise NetworkFailure(f"HTTP {resp.status}")
        return resp.body

    return fetch


def test_a_missing_picture_does_not_hold_the_others_back(
        tmp_path: Path) -> None:
    home = tmp_path / "home"
    routes = _fixture_server()
    # vectorcraft/neon is not published; the rest are.
    for key in ("ribbons", "sheet", "bezier"):
        routes[f"/v1/vectorcraft/{key}.jpg"] = (
            200, _jpeg(1200, 750, pad=100))
    with _Server(routes) as server:
        result = fetch_all(home, "vectorcraft", base_url=server.base,
                           fetcher=_loopback_fetcher(server.base))
        by_key = {s["key"]: s for s in result["shots"]}
        assert by_key["neon"]["status"] == "missing"
        assert "isn't published" in by_key["neon"]["error"]
        for key in ("ribbons", "sheet", "bezier"):
            assert by_key[key]["status"] == "downloaded"


def test_what_is_not_a_picture_is_not_kept(tmp_path: Path) -> None:
    home = tmp_path / "home"
    routes = {"/v1/photocraft/demo.jpg": (200, b"<html>Sign in</html>"),
              "/v1/photocraft/adjustments.jpg": (200, _jpeg(1200, 750)),
              "/v1/photocraft/layer-styles.jpg": (
                  200, _jpeg(65535, 65535)),
              "/v1/photocraft/masks.jpg": (200, _jpeg(1200, 750))}
    with _Server(routes) as server:
        result = fetch_all(home, "photocraft", base_url=server.base,
                           fetcher=_loopback_fetcher(server.base))
        by_key = {s["key"]: s for s in result["shots"]}
        assert by_key["demo"]["status"] == "invalid"
        assert "isn't a JPEG" in by_key["demo"]["error"]
        assert not screenshot_path(
            home, "photocraft", "demo").exists()
        assert by_key["layer-styles"]["status"] == "invalid"
        assert "too large" in by_key["layer-styles"]["error"]
        assert not screenshot_path(
            home, "photocraft", "layer-styles").exists()
        assert by_key["adjustments"]["status"] == "downloaded"
        assert by_key["masks"]["status"] == "downloaded"


def test_a_network_failure_ends_the_run_but_saved_copies_serve(
        tmp_path: Path) -> None:
    home = tmp_path / "home"
    # The second picture is saved already; the rest need the network,
    # which is not there.
    second = screenshot_path(home, "photocraft", "adjustments")
    write_atomic(second, _jpeg(1200, 750, pad=100))

    def failing(url: str) -> bytes | None:
        raise NetworkFailure("connection refused")

    result = fetch_all(home, "photocraft", fetcher=failing)
    keys = [s["key"] for s in result["shots"]]
    assert keys == ["demo", "adjustments", "layer-styles", "masks"]
    by_key = {s["key"]: s for s in result["shots"]}
    assert by_key["adjustments"]["status"] == "cached"
    for key in ("demo", "layer-styles", "masks"):
        assert by_key[key]["status"] == "network"
        assert "connection refused" in by_key[key]["error"]


def test_a_damaged_saved_copy_is_fetched_again(tmp_path: Path) -> None:
    home = tmp_path / "home"
    path = screenshot_path(home, "photocraft", "demo")
    write_atomic(path, b"<html>oops</html>")
    with _Server(_fixture_server()) as server:
        result = fetch_all(home, "photocraft", base_url=server.base,
                           fetcher=_loopback_fetcher(server.base))
        by_key = {s["key"]: s for s in result["shots"]}
        assert by_key["demo"]["status"] == "downloaded"
        assert saved_copy(path) is not None


def test_fetch_unknown_app_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ScreenshotError, match="unknown app"):
        fetch_all(tmp_path, "nope", fetcher=lambda url: None)


def test_manifest_survives_corruption(tmp_path: Path) -> None:
    home = tmp_path / "home"
    assert read_manifest(home) == {"assets": {}}
    manifest = home / "cache" / "screenshots" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{not json", encoding="utf-8")
    assert read_manifest(home) == {"assets": {}}


# ---------------------------------------------------------------------------
# Publish side (README: changing a picture)
# ---------------------------------------------------------------------------


def test_strip_metadata_removes_exif_and_comments() -> None:
    data = bytearray([0xFF, 0xD8])
    # APP1 (EXIF) then COM, then SOF0, then SOS + scan data.
    data += bytes([0xFF, 0xE1, 0x00, 0x06]) + b"Exif"
    data += bytes([0xFF, 0xFE, 0x00, 0x06]) + b"note"
    data += bytes([0xFF, 0xC0, 0x00, 0x11, 0x08])
    data += (750).to_bytes(2, "big") + (1200).to_bytes(2, "big")
    data += bytes([0x03, 0x01, 0x11, 0x00])
    data += bytes([0xFF, 0xDA, 0x00, 0x08]) + b"\x01\x02\x03\xff\x00\x04"
    cleaned, stripped = strip_metadata(bytes(data))
    assert stripped == ["APP1 (EXIF/XMP)", "COM (comment)"]
    assert b"Exif" not in cleaned and b"note" not in cleaned
    assert jpeg_dimensions(cleaned) == (1200, 750)
    assert cleaned.endswith(b"\x01\x02\x03\xff\x00\x04")
    with pytest.raises(PictureUnavailable):
        strip_metadata(b"<html>")


def test_prepare_validates_dimensions_and_strips(tmp_path: Path) -> None:
    result = prepare_picture(_jpeg(1200, 750, pad=1000),
                             app="photocraft", key="demo")
    assert result["ok"] is True
    assert (result["width"], result["height"]) == (1200, 750)
    assert result["bytes"] == len(result["data"])
    assert result["sha256"] == hashlib.sha256(
        result["data"]).hexdigest()
    assert result["attribution"]["path"] == "photocraft/demo.jpg"
    assert result["attribution"]["ui_licence"] == "Apache-2.0"
    # Wrong dimensions are refused, naming the actual size.
    with pytest.raises(PictureUnavailable, match="must be 1200x750"):
        prepare_picture(_jpeg(800, 600), app="photocraft", key="demo")
    # Not a JPEG, or no readable frame, is refused.
    with pytest.raises(PictureUnavailable, match="not a JPEG"):
        prepare_picture(b"<html>", app="photocraft", key="demo")
    # An unknown app is refused.
    with pytest.raises(ScreenshotError, match="unknown app"):
        prepare_picture(_jpeg(1200, 750), app="nope", key="demo")


def test_prepare_refuses_an_oversized_picture() -> None:
    big = _jpeg(1200, 750, pad=MAX_SCREENSHOT_BYTES)
    with pytest.raises(PictureUnavailable, match="over the"):
        prepare_picture(big, app="photocraft", key="demo")


@pytest.mark.skipif(shutil.which("ffmpeg") is None,
                    reason="ffmpeg is not installed")
def test_prepare_resizes_with_ffmpeg(tmp_path: Path) -> None:
    # A real JPEG, made by ffmpeg itself (a hand-built header-only
    # JPEG has no scan data and no decoder will open it).
    source = tmp_path / "source.jpg"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         "testsrc=duration=1:size=800x600", "-frames:v", "1",
         "-q:v", "2", str(source)],
        check=True, timeout=60)
    assert jpeg_dimensions(source.read_bytes()) == (800, 600)
    result = prepare_picture(source.read_bytes(),
                             app="photocraft", key="demo", resize=True)
    assert result["ok"] is True
    assert (result["width"], result["height"]) == (1200, 750)
    assert jpeg_dimensions(result["data"]) == (1200, 750)


def test_resize_without_ffmpeg_is_an_honest_refusal(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ScreenshotError, match="ffmpeg is not installed"):
        resize_picture(_jpeg(800, 600))


# ---------------------------------------------------------------------------
# Tool surface + registry pin
# ---------------------------------------------------------------------------


def test_tool_registered_and_counts_789() -> None:
    registry = _registry()
    spec = registry.get("craftlauncher")
    assert spec is not None
    assert "shots" in spec.description
    assert "fetch" in spec.description
    # 749 after the mocap port (lane 125) added the mocap tool, and
    # 774 after the secprobe wire-up added 25 tools; this
    # lane adds no tool — the new actions ride the existing
    # action-dispatched craftlauncher tool.
    # 775 after this lane (B.19) added scene_spawn. 776 after this lane (B.20) added cloudcam. 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet). 801 after gapfix124 added browser_watch. 802 after this lane (SongRecon surface) added songrecon. 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 807 after this lane (agents192) added fleet.
    assert len(registry._tools) == 807


def test_tool_shots_lists_the_catalog() -> None:
    registry = _registry()
    shots = _call(registry, action="shots")
    assert shots["rev"] == SCREENSHOTS_REV
    assert shots["total"] == 39
    assert len(shots["apps"]) == 11
    assert shots["apps"]["photocraft"][0]["key"] == "demo"
    assert shots["apps"]["photocraft"][0]["url"] == (
        "https://raw.githubusercontent.com/storytold/"
        "craft-launcher-assets/v1/photocraft/demo.jpg")
    one = _call(registry, action="shots", value="vectorcraft")
    assert list(one["apps"]) == ["vectorcraft"]
    assert one["total"] == 4
    bad = _call(registry, action="shots", value="nope")
    assert bad["ok"] is False


def test_tool_fetch_end_to_end_over_loopback(tmp_path: Path) -> None:
    registry = _registry()
    home = tmp_path / "home"
    with _Server(_fixture_server()) as server:
        result = _call(
            registry, action="fetch", value="photocraft",
            params={"home": str(home), "base_url": server.base})
        assert result["fetched"] == 4
        assert result["failed"] == 0
        again = _call(
            registry, action="fetch", value="photocraft",
            params={"home": str(home), "base_url": server.base})
        assert again["cached"] == 4
        manifest = _call(registry, action="manifest",
                         params={"home": str(home)})
        assert len(manifest["assets"]) == 4
    # A non-loopback base URL is refused (loopback mirror only).
    refused = _call(registry, action="fetch", value="photocraft",
                    params={"base_url": "https://example.com"})
    assert refused["ok"] is False
    assert "loopback" in refused["error"]


def test_tool_terms_and_prepare(tmp_path: Path) -> None:
    registry = _registry()
    row = _call(registry, action="terms", value="photocraft/demo")
    assert row["ui_licence"] == "Apache-2.0"
    assert row["artwork_licence"] == "public domain"
    bad = _call(registry, action="terms", value="nope/demo")
    assert bad["ok"] is False
    src = tmp_path / "candidate.jpg"
    src.write_bytes(_jpeg(1200, 750, pad=500))
    out = tmp_path / "clean.jpg"
    result = _call(registry, action="prepare", sources=[str(src)],
                   output=str(out),
                   params={"app": "photocraft", "key": "demo"})
    assert result["ok"] is True
    assert result["output"] == str(out)
    assert out.is_file()
    assert jpeg_dimensions(out.read_bytes()) == (1200, 750)
    wrong = _call(registry, action="prepare", sources=[str(src)],
                  params={"app": "photocraft", "key": "demo"})
    # The default output lands beside the source.
    assert wrong["ok"] is True
    assert Path(wrong["output"]).name == "candidate.1200x750.jpg"
