"""52-websites #27 (Archive.ph): page snapshots with hashes as provenance.

Proof the brief demands, each pinned here:
- a fixture page mutated after capture → her copy still shows the
  original and the hash verifies;
- a JS-rendered fixture snapshot matches the rendered state, not
  the raw HTML;
- a seed-source snapshot is labelled as a reference copy;
- retention prune removes old snapshots and says so.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.browser.snapshot import (
    SnapshotStore,
    capture,
    cite_line,
    format_snapshot_reply,
    parse_snapshot_request,
    prune,
    seed,
    set_retention,
    snapshot_text,
    verify,
)

RAW_HTML = ("<html><head><title>Raw</title></head><body>"
            "<p>before JS ran</p></body></html>")
RENDERED_DOM = ("<html><head><title>Rendered</title></head><body>"
                "<p>Rendered by JS</p>"
                "<img src=\"/img/logo.png\">"
                "<img src=\"https://example.com/img/dead.png\">"
                "</body></html>")
ARCHIVED_HTML = ("<html><head><title>Old</title></head><body>"
                 "<p>the page as it was</p></body></html>")
SHOT_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class FakeRenderer:
    """Renders the JS-mutated DOM, never the raw HTML."""

    def __init__(self, dom: str = RENDERED_DOM,
                 title: str = "Rendered") -> None:
        self.dom = dom
        self.title = title

    def render(self, url: str, *, screenshot_path) -> dict:
        Path(screenshot_path).write_bytes(SHOT_BYTES)
        return {"dom": self.dom, "title": self.title,
                "screenshot_path": str(screenshot_path)}


class FailingRenderer:
    def render(self, url: str, *, screenshot_path) -> dict:
        raise RuntimeError("chromium crashed")


def _fetcher_for(mapping: dict) -> object:
    def fetch(url: str) -> bytes:
        if url in mapping:
            value = mapping[url]
            if isinstance(value, Exception):
                raise value
            return value
        raise urllib.error.URLError(f"no fixture for {url}")
    return fetch


@pytest.fixture()
def home(tmp_path):
    return tmp_path


@pytest.fixture()
def store(home):
    return SnapshotStore(home)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── capture: rendered state, not raw HTML ───────────────────────────────


def test_capture_stores_rendered_dom_not_raw(store):
    snap = capture("https://example.com/page",
                   store=store, renderer=FakeRenderer())
    assert snap.rendered is True
    assert snap.render_reason == ""
    assert snap.title == "Rendered"
    stored = Path(snap.dom_path).read_bytes()
    assert b"Rendered by JS" in stored
    assert b"before JS ran" not in stored
    assert snap.content_hash == _sha(stored)
    assert snap.render_hash == _sha(stored + SHOT_BYTES)
    assert Path(snap.screenshot_path).is_file()
    assert Path(snap.screenshot_path).read_bytes() == SHOT_BYTES
    assert snap.kind == "capture"
    assert snap.source == "devon-browser"


def test_mutated_fixture_still_shows_original(store):
    renderer = FakeRenderer()
    snap = capture("https://example.com/page",
                   store=store, renderer=renderer)
    renderer.dom = ("<html><body><p>the page changed</p></body></html>")
    stored = Path(snap.dom_path).read_text(encoding="utf-8")
    assert "Rendered by JS" in stored
    assert "the page changed" not in stored
    result = verify(store, snap.id)
    assert result["ok"] is True
    assert result["content_hash"] == snap.content_hash


def test_verify_catches_tampered_dom(store):
    snap = capture("https://example.com/page",
                   store=store, renderer=FakeRenderer())
    dom_path = Path(snap.dom_path)
    dom_path.write_bytes(dom_path.read_bytes() + b"tampered")
    result = verify(store, snap.id)
    assert result["ok"] is False
    assert "hash" in result["error"]


def test_verify_catches_missing_screenshot(store):
    snap = capture("https://example.com/page",
                   store=store, renderer=FakeRenderer())
    Path(snap.screenshot_path).unlink()
    result = verify(store, snap.id)
    assert result["ok"] is False
    assert "screenshot" in result["error"]


def test_verify_unknown_id(store):
    result = verify(store, "no-such-id")
    assert result["ok"] is False


# ── raw fallback: honest, labelled ──────────────────────────────────────


def test_raw_fallback_when_no_renderer(store):
    fetcher = _fetcher_for(
        {"https://example.com/page": RAW_HTML.encode("utf-8")})
    snap = capture("https://example.com/page",
                   store=store, renderer=None, fetcher=fetcher)
    assert snap.rendered is False
    assert "playwright" in snap.render_reason
    assert snap.screenshot_path == ""
    assert snap.render_hash == snap.content_hash
    stored = Path(snap.dom_path).read_bytes()
    assert b"before JS ran" in stored
    assert verify(store, snap.id)["ok"] is True


def test_render_failure_falls_back_to_raw(store):
    fetcher = _fetcher_for(
        {"https://example.com/page": RAW_HTML.encode("utf-8")})
    snap = capture("https://example.com/page", store=store,
                   renderer=FailingRenderer(), fetcher=fetcher)
    assert snap.rendered is False
    assert "render failed" in snap.render_reason
    assert b"before JS ran" in Path(snap.dom_path).read_bytes()


def test_capture_rejects_non_http(store):
    with pytest.raises(ValueError):
        capture("ftp://example.com/file", store=store)
    with pytest.raises(ValueError):
        capture("not a url", store=store)


def test_capture_fetch_failure_is_a_value_error(store):
    fetcher = _fetcher_for({})
    with pytest.raises(ValueError, match="could not fetch"):
        capture("https://example.com/page", store=store,
                renderer=None, fetcher=fetcher)


# ── asset bundle ────────────────────────────────────────────────────────


def test_asset_bundle_records_missing(store):
    mapping = {"https://example.com/img/logo.png": b"logo-bytes",
               "https://example.com/img/dead.png":
                   urllib.error.URLError("404")}
    snap = capture("https://example.com/page", store=store,
                   renderer=FakeRenderer(),
                   fetcher=_fetcher_for(mapping))
    by_url = {asset["url"]: asset for asset in snap.assets}
    assert len(snap.assets) == 2
    assert by_url["https://example.com/img/logo.png"]["ok"] is True
    assert by_url["https://example.com/img/logo.png"]["bytes"] == 10
    assert by_url["https://example.com/img/logo.png"]["sha256"] == \
        _sha(b"logo-bytes")
    assert by_url["https://example.com/img/dead.png"]["ok"] is False
    assert by_url["https://example.com/img/dead.png"]["error"]


# ── seed sources: reference copies, never sole evidence ─────────────────


def test_seed_wayback_is_a_reference_copy(store):
    mapping = {
        "https://archive.org/wayback/available?url="
        "https%3A%2F%2Fexample.com%2Fpage": json.dumps({
            "archived_snapshots": {"closest": {
                "url": "https://web.archive.org/web/20260101000000/"
                       "https://example.com/page",
                "timestamp": "20260101000000"}}}).encode("utf-8"),
        "https://web.archive.org/web/20260101000000/"
        "https://example.com/page": ARCHIVED_HTML.encode("utf-8"),
    }
    snap = seed("https://example.com/page", source="wayback",
                store=store, fetcher=_fetcher_for(mapping))
    assert snap.kind == "reference"
    assert snap.source == "wayback"
    assert snap.rendered is False
    assert "never sole evidence" in snap.render_reason
    assert b"the page as it was" in Path(snap.dom_path).read_bytes()
    assert verify(store, snap.id)["ok"] is True
    receipt = format_snapshot_reply(snap)
    assert "reference copy from wayback" in receipt


def test_seed_archiveph_is_a_reference_copy(store):
    mapping = {
        "https://archive.ph/newest/https%3A%2F%2Fexample.com%2Fpage":
            ARCHIVED_HTML.encode("utf-8"),
    }
    snap = seed("https://example.com/page", source="archive.ph",
                store=store, fetcher=_fetcher_for(mapping))
    assert snap.kind == "reference"
    assert snap.source == "archive.ph"
    assert "never sole evidence" in snap.render_reason


def test_seed_wayback_with_no_snapshot_refuses(store):
    mapping = {
        "https://archive.org/wayback/available?url="
        "https%3A%2F%2Fexample.com%2Fpage": json.dumps(
            {"archived_snapshots": {}}).encode("utf-8"),
    }
    with pytest.raises(ValueError, match="no snapshot"):
        seed("https://example.com/page", source="wayback",
             store=store, fetcher=_fetcher_for(mapping))


def test_seed_unknown_source_refused(store):
    with pytest.raises(ValueError, match="wayback or archive.ph"):
        seed("https://example.com/page", source="google-cache",
             store=store)


# ── retention ───────────────────────────────────────────────────────────


def test_prune_removes_old_and_reports(store):
    old = capture("https://example.com/old", store=store,
                  renderer=FakeRenderer())
    fresh = capture("https://example.com/fresh", store=store,
                    renderer=FakeRenderer())
    old.captured_at = time.time() - 100 * 86400
    store.add(old)
    old_dir = store.snapshot_dir(old.id)
    assert old_dir.is_dir()
    result = prune(store, older_than_days=30)
    assert result["ok"] is True
    assert result["removed"] == [old.id]
    assert result["kept"] == 1
    assert store.get(old.id) is None
    assert store.get(fresh.id) is not None
    assert not old_dir.exists()


def test_prune_keeps_fresh(store):
    capture("https://example.com/fresh", store=store,
            renderer=FakeRenderer())
    result = prune(store, older_than_days=30)
    assert result["removed"] == []
    assert result["kept"] == 1


def test_set_retention(store):
    result = set_retention(store, 7)
    assert result["retention_days"] == 7
    assert SnapshotStore(store.path.parent.parent).retention_days == 7
    with pytest.raises(ValueError):
        set_retention(store, 0)


# ── store durability ────────────────────────────────────────────────────


def test_corrupt_index_quarantined_aside(home):
    store = SnapshotStore(home)
    capture("https://example.com/page", store=store,
            renderer=FakeRenderer())
    store.path.write_text("{not json", encoding="utf-8")
    reopened = SnapshotStore(home)
    assert reopened.list() == []
    quarantined = list(home.glob("data/snapshots.corrupt-*"))
    assert len(quarantined) == 1


def test_store_round_trip(home):
    first = SnapshotStore(home)
    snap = capture("https://example.com/page", store=first,
                   renderer=FakeRenderer())
    second = SnapshotStore(home)
    loaded = second.get(snap.id)
    assert loaded is not None
    assert loaded.url == snap.url
    assert loaded.content_hash == snap.content_hash
    assert second.find_by_url("https://example.com/page") is not None


# ── replies and citations ───────────────────────────────────────────────


def test_receipt_names_rendered_state(store):
    rendered = capture("https://example.com/a", store=store,
                       renderer=FakeRenderer())
    receipt = format_snapshot_reply(rendered)
    assert "rendered: yes" in receipt
    assert rendered.content_hash[:16] in receipt
    assert rendered.id in receipt
    raw = capture("https://example.com/b", store=store, renderer=None,
                  fetcher=_fetcher_for(
                      {"https://example.com/b":
                       RAW_HTML.encode("utf-8")}))
    assert "rendered: no" in format_snapshot_reply(raw)


def test_cite_line(store):
    snap = capture("https://example.com/page", store=store,
                   renderer=FakeRenderer())
    line = cite_line(snap)
    assert line.startswith("as captured on ")
    assert snap.url in line
    assert snap.id in line
    assert snap.content_hash[:16] in line


def test_snapshot_text_strips_markup():
    text = snapshot_text(RENDERED_DOM)
    assert "Rendered by JS" in text
    assert "<p>" not in text
    assert "<html>" not in text


def test_parse_snapshot_request():
    parsed = parse_snapshot_request(
        "snapshot this page https://example.com/page")
    assert parsed == {"url": "https://example.com/page"}
    parsed = parse_snapshot_request(
        "take a snapshot of https://example.com/page.")
    assert parsed == {"url": "https://example.com/page"}
    assert parse_snapshot_request("hello there") is None
    assert parse_snapshot_request(
        "/snapshot https://example.com/page") is None
    assert parse_snapshot_request("snapshot this") is None
    assert parse_snapshot_request("") is None


# ── the brain tool ──────────────────────────────────────────────────────


def _tool(home):
    from nomorals.tools import snapshot as snap_mod
    from nomorals.tools.registry import ToolRegistry

    context = SimpleNamespace(
        settings=SimpleNamespace(home=str(home)))
    reg = ToolRegistry(context=context)
    snap_mod.register(reg)
    return reg._tools["snapshot"].fn


def test_tool_capture_end_to_end(home, monkeypatch):
    from nomorals.tools import snapshot as snap_mod

    tool = _tool(home)
    monkeypatch.setattr(
        snap_mod, "capture",
        lambda url, store: capture(url, store=store,
                                   renderer=FakeRenderer()))
    result = tool(action="capture", url="https://example.com/page")
    assert result["ok"] is True
    assert "content hash" in result["receipt"]
    assert result["cite"].startswith("as captured on ")
    assert result["snapshot"]["rendered"] is True


def test_tool_capture_needs_url(home):
    tool = _tool(home)
    assert tool(action="capture", url="")["ok"] is False


def test_tool_seed_and_show_and_verify(home, monkeypatch):
    from nomorals.tools import snapshot as snap_mod

    tool = _tool(home)
    mapping = {
        "https://archive.org/wayback/available?url="
        "https%3A%2F%2Fexample.com%2Fpage": json.dumps({
            "archived_snapshots": {"closest": {
                "url": "https://web.archive.org/web/20260101000000/"
                       "https://example.com/page"}}}).encode("utf-8"),
        "https://web.archive.org/web/20260101000000/"
        "https://example.com/page": ARCHIVED_HTML.encode("utf-8"),
    }
    monkeypatch.setattr(snap_mod, "seed",
                        lambda url, source, store: seed(
                            url, source=source, store=store,
                            fetcher=_fetcher_for(mapping)))
    seeded = tool(action="seed", url="https://example.com/page",
                  source="wayback")
    assert seeded["ok"] is True
    assert seeded["snapshot"]["kind"] == "reference"
    snap_id = seeded["snapshot"]["id"]
    shown = tool(action="show", id=snap_id)
    assert shown["ok"] is True
    assert shown["snapshot"]["id"] == snap_id
    assert tool(action="verify", id=snap_id)["ok"] is True
    cited = tool(action="cite", id=snap_id)
    assert cited["ok"] is True
    assert "the page as it was" in cited["text"]
    assert cited["cite"].startswith("as captured on ")


def test_tool_list_prune_status_retention(home, monkeypatch):
    from nomorals.tools import snapshot as snap_mod

    tool = _tool(home)
    monkeypatch.setattr(
        snap_mod, "capture",
        lambda url, store: capture(url, store=store,
                                   renderer=FakeRenderer()))
    tool(action="capture", url="https://example.com/a")
    tool(action="capture", url="https://example.com/b")
    listed = tool(action="list")
    assert listed["count"] == 2
    assert listed["retention_days"] == 90
    status = tool(action="status")
    assert status["count"] == 2
    assert status["bytes"] > 0
    retained = tool(action="retention", days=45)
    assert retained["retention_days"] == 45
    assert tool(action="retention", days="soon")["ok"] is False
    pruned = tool(action="prune", days=30)
    assert pruned["removed"] == []
    assert pruned["kept"] == 2
    assert tool(action="bogus")["ok"] is False
    assert tool(action="show", id="nope")["ok"] is False


def test_tool_registered_by_builtins():
    from nomorals.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register_builtins()
    assert "snapshot" in reg._tools
    assert "__snapshot_prune_tick__" in reg._tools
