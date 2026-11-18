"""Shows, wired into production.

Audit gap R1-4: the domain (nomorals/shows/composer.py — Show/Episode,
compose_episode, publish_episode) existed and was unit-pinned, but
nothing in production reached it — no durable store, no chat command,
no way to create, keep, or publish a show. Every test here drives the
PRODUCTION wiring: the durable ShowStore under the home's data/
directory and the /show control — never the dormant organ in
isolation.

Teeth: on the pristine base each test fails for the missing wiring
(no _control_show, no "show" command kind).

Composing never generates scripts — the caller supplies the segments.
Publishing delegates to the composer's publicface gate, so an episode
stays a draft unless the owner publishes it AND it carries no
private-data markers.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from nomorals.shows.composer import Show, compose_episode
from nomorals.shows.store import ShowStore
from nomorals.social.chat.control import (
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
    caller_may_run,
    detailed_help,
    help_text,
)
from nomorals.social.chat.personal_commands import PERSONAL_OWNER_COMMANDS


def _reach_host(tmp_path):
    """Production control host on a throwaway home (assets harness)."""
    from nomorals.agents.partner.runtime_reach import RuntimeReachMixin

    host = RuntimeReachMixin()
    host.context = SimpleNamespace(
        settings=SimpleNamespace(home_path=Path(tmp_path)))
    return host


def _make_show(host, title="Evening Pulse", rest=""):
    out = host._control_show(f"create {title}{rest}")
    assert out.startswith("show created: "), out
    return out.split(":")[1].strip().split()[0]


# ── durable store ───────────────────────────────────────────────────


def test_store_persists_across_instances(tmp_path):
    path = tmp_path / "shows.json"
    store = ShowStore(path)
    show = store.put_show(Show(title="Evening Pulse",
                               schedule="daily 23:00",
                               voices=["devon"]).to_dict())
    episode = store.put_episode(compose_episode(
        Show(**show),
        [{"kind": "script", "text": "good evening Lagos"},
         {"kind": "ad", "text": "buy yams"}],
        title="Pilot").to_dict())
    assert episode["word_count"] == 5
    assert episode["duration_seconds"] == 2.0
    assert "<item>" in episode["rss_item"]

    fresh = ShowStore(path)
    assert [s["title"] for s in fresh.list_shows()] == ["Evening Pulse"]
    eps = fresh.list_episodes(show["id"])
    assert [e["id"] for e in eps] == [episode["id"]]
    assert eps[0]["script_text"] == "good evening Lagos\n\nbuy yams"
    assert (tmp_path / "shows.json").exists()


def test_store_corrupt_file_quarantined_and_raises(tmp_path):
    path = tmp_path / "shows.json"
    path.write_text("{not json", encoding="utf-8")
    try:
        ShowStore(path)
    except ValueError as exc:
        assert "quarantined" in str(exc)
    else:
        raise AssertionError("corrupt store must raise")
    assert not path.exists()
    quarantined = list(tmp_path.glob("shows.json.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "{not json"
    # A bad row inside valid JSON is skipped, not fatal.
    path.write_text(json.dumps(
        {"shows": [{"id": "bad"}, {"id": "ok", "title": "Good"}],
         "episodes": []}), encoding="utf-8")
    store = ShowStore(path)
    assert [s["id"] for s in store.list_shows()] == ["ok"]


# ── /show control: shows ────────────────────────────────────────────


def test_show_empty_answers(tmp_path):
    host = _reach_host(tmp_path)
    assert "no shows yet" in host._control_show("")
    assert "no shows yet" in host._control_show("list")


def test_show_create_list_show(tmp_path):
    host = _reach_host(tmp_path)
    show_id = _make_show(
        host, rest=" | daily 23:00 | devon,ada | bed.mp3")
    listed = host._control_show("list")
    assert "Evening Pulse" in listed and show_id in listed
    assert "daily 23:00" in listed
    shown = host._control_show(f"show {show_id}")
    assert "Evening Pulse" in shown
    assert "schedule: daily 23:00" in shown
    assert "voices: devon, ada" in shown
    assert "bed track: bed.mp3" in shown
    assert "no episodes yet" in shown
    # Unique id-prefix resolves; an unknown id says so plainly.
    assert "Evening Pulse" in host._control_show(
        f"show {show_id[:6]}")
    assert "no show matching" in host._control_show("show nope")
    assert "usage:" in host._control_show("create")
    assert "usage:" in host._control_show("frobnicate")


def test_show_delete_removes_episodes(tmp_path):
    host = _reach_host(tmp_path)
    show_id = _make_show(host)
    host._control_show(
        f"episode {show_id} Pilot | hello world this is a test")
    assert "Pilot" in host._control_show(f"episodes {show_id}")
    out = host._control_show(f"delete {show_id}")
    assert out == "deleted show: Evening Pulse", out
    assert "no shows yet" in host._control_show("list")
    assert "no show matching" in host._control_show(
        f"episodes {show_id}")
    assert "no show matching" in host._control_show("delete nope")


# ── /show control: episodes ────────────────────────────────────────


def test_show_episode_compose_round_trip(tmp_path):
    host = _reach_host(tmp_path)
    show_id = _make_show(host)
    out = host._control_show(
        f"episode {show_id} Pilot | good evening Lagos | "
        "ad: buy yams | bed: bed.mp3")
    assert out.startswith("episode composed: "), out
    assert "5 words" in out and "~2.0s" in out
    episode_id = out.split(":")[1].strip().split()[0]
    listed = host._control_show(f"episodes {show_id}")
    assert "Pilot" in listed and episode_id in listed
    rss = host._control_show(f"rss {episode_id}")
    assert "<item>" in rss and "<title>Pilot</title>" in rss
    assert "good evening Lagos" in rss
    # The bed segment carries no spoken words but stays on the record.
    store = host._show_store()
    stored = store.get_episode(episode_id)
    assert [s["kind"] for s in stored["segments"]] == [
        "script", "ad", "bed"]
    assert stored["segments"][2]["ref"] == "bed.mp3"
    # A bare field is a script segment.
    out = host._control_show(
        f"episode {show_id} Second | hello there world from Lagos")
    assert "episode composed:" in out
    stored2 = host._show_store().list_episodes(show_id)[1]
    assert stored2["segments"] == [
        {"kind": "script", "text": "hello there world from Lagos"}]
    # Unknown kind is an error in words, never a silent script —
    # even a prose "Note:" prefix reads as a kind attempt.
    out = host._control_show(
        f"episode {show_id} Third | Note: hello there")
    assert "unknown segment kind 'Note'" in out
    assert "kinds: script, voice, bed, ad" in out
    assert "usage:" in host._control_show(f"episode {show_id}")
    assert "no show matching" in host._control_show(
        "episode nope Title | hello")
    assert "no episode matching" in host._control_show("rss nope")
    # Remove deletes one episode and leaves the rest.
    out = host._control_show(f"remove {episode_id}")
    assert out == "removed episode: Pilot", out
    assert "Pilot" not in host._control_show(f"episodes {show_id}")
    assert "Second" in host._control_show(f"episodes {show_id}")
    assert "no episode matching" in host._control_show(
        f"remove {episode_id}")


def test_show_episode_needs_segments(tmp_path):
    host = _reach_host(tmp_path)
    show_id = _make_show(host)
    out = host._control_show(f"episode {show_id} Empty |")
    assert "at least one segment" in out
    assert "no episodes for" in host._control_show(
        f"episodes {show_id}")


# ── /show control: publish ─────────────────────────────────────────


def test_show_publish_clean_goes_public(tmp_path):
    host = _reach_host(tmp_path)
    show_id = _make_show(host)
    out = host._control_show(
        f"episode {show_id} Pilot | good evening Lagos")
    episode_id = out.split(":")[1].strip().split()[0]
    out = host._control_show(f"publish {episode_id}")
    assert out.startswith("published: Pilot — live at /p/show-"), out
    stored = host._show_store().get_episode(episode_id)
    assert stored["published"] is True
    assert stored["publish_link"] == f"/p/show-{episode_id}"
    assert "[published]" in host._control_show(f"episodes {show_id}")
    # The published page reaches the publicface store the /p/ route
    # reads — otherwise /p/show-<id> would still 404.
    from nomorals.publicface.publish import PublicPageStore

    pf = PublicPageStore(str(Path(tmp_path) / "data" / "publicface.json"))
    page = pf.get_by_slug(f"show-{episode_id}")
    assert page is not None and page.status == "public"
    assert page.content == "good evening Lagos"


def test_show_publish_twice_keeps_one_public_page(tmp_path):
    host = _reach_host(tmp_path)
    show_id = _make_show(host)
    out = host._control_show(
        f"episode {show_id} Pilot | good evening Lagos")
    episode_id = out.split(":")[1].strip().split()[0]
    host._control_show(f"publish {episode_id}")
    host._control_show(f"publish {episode_id}")
    from nomorals.publicface.publish import PublicPageStore

    pf = PublicPageStore(str(Path(tmp_path) / "data" / "publicface.json"))
    rows = [p for p in pf.pages.values()
            if p.slug == f"show-{episode_id}"]
    assert len(rows) == 1 and rows[0].status == "public"


def test_show_remove_episode_drops_public_page(tmp_path):
    host = _reach_host(tmp_path)
    show_id = _make_show(host)
    out = host._control_show(
        f"episode {show_id} Pilot | good evening Lagos")
    episode_id = out.split(":")[1].strip().split()[0]
    host._control_show(f"publish {episode_id}")
    from nomorals.publicface.publish import PublicPageStore

    pf = PublicPageStore(str(Path(tmp_path) / "data" / "publicface.json"))
    assert pf.get_by_slug(f"show-{episode_id}") is not None
    out = host._control_show(f"remove {episode_id}")
    assert out.startswith("removed episode: "), out
    pf = PublicPageStore(str(Path(tmp_path) / "data" / "publicface.json"))
    assert pf.get_by_slug(f"show-{episode_id}") is None


def test_show_delete_show_drops_public_pages(tmp_path):
    host = _reach_host(tmp_path)
    show_id = _make_show(host)
    out = host._control_show(
        f"episode {show_id} Pilot | good evening Lagos")
    episode_id = out.split(":")[1].strip().split()[0]
    host._control_show(f"publish {episode_id}")
    from nomorals.publicface.publish import PublicPageStore

    pf = PublicPageStore(str(Path(tmp_path) / "data" / "publicface.json"))
    assert pf.get_by_slug(f"show-{episode_id}") is not None
    out = host._control_show(f"delete {show_id}")
    assert out.startswith("deleted show: "), out
    pf = PublicPageStore(str(Path(tmp_path) / "data" / "publicface.json"))
    assert pf.get_by_slug(f"show-{episode_id}") is None


def test_show_publish_private_markers_stay_draft(tmp_path):
    host = _reach_host(tmp_path)
    show_id = _make_show(host)
    out = host._control_show(
        f"episode {show_id} Leaky | the secret recipe for yam")
    episode_id = out.split(":")[1].strip().split()[0]
    out = host._control_show(f"publish {episode_id}")
    assert out.startswith("left as draft: Leaky — "), out
    assert "secret" in out
    stored = host._show_store().get_episode(episode_id)
    assert stored["published"] is False
    assert "no episode matching" in host._control_show("publish nope")


# ── registry ────────────────────────────────────────────────────────


def test_show_registered():
    assert "show" in CONTROL_COMMANDS
    slots = [title for title, kinds in LIST_GROUPS if "show" in kinds]
    assert slots == ["memory & thinking"]
    assert "show" in LIST_ONELINERS and "/show" in LIST_ONELINERS["show"]
    assert "/show" in help_text()
    page = detailed_help("show")
    assert "try:" in page and "related:" in page
    assert "/show create" in page


def test_show_owner_only_off_menu():
    # Owner-only: absent from PUBLIC_COMMANDS, so caller_may_run denies
    # non-owners before the handler runs (assets/legacy/family pattern).
    assert "show" not in PUBLIC_COMMANDS
    assert caller_may_run("show", is_owner=False) is False
    assert caller_may_run("show", is_owner=True) is True
    # Off the BotFather menu by design: the 100-command cap is full
    # (48 owner pins) — registered, runnable, /list-ed, like the
    # reachability wave's off-menu owner commands.
    assert "show" not in PERSONAL_OWNER_COMMANDS
