"""Life archive + preferences, wired into production.

Audit gaps R1-17/R1-18: both organs existed and were unit-pinned,
but nothing in production reached them — LifeArchive had no way in
or out (no /life, no durable path), and PreferenceStore had zero
consumers (no /prefs, no ranking, no signal recording). Every test
here drives the PRODUCTION wiring: the durable archive/store under
the home's data/ directory and the /life + /prefs controls — never
the dormant organs in isolation.

Teeth: on the pristine base each handler test fails for the missing
wiring (no _control_life / _control_prefs, no "life"/"prefs"
command kinds).

LifeArchive is append-only by design — there is no delete verb.
Preferences start at zero and only move on explicit approve /
disapprove signals; a fresh store scores everything 0 and preserves
the caller's order.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nomorals.life.archive import LifeArchive
from nomorals.preferences.feedback import record_signal
from nomorals.preferences.store import DOMAIN_DESCRIPTORS, PreferenceStore
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


def _add_life(host, kind="milestone", text="shipped the first build"):
    out = host._control_life(f"add {kind} | {text}")
    assert out.startswith("recorded: "), out
    return out.split(":")[1].strip().split()[0]


# ── life archive: durable organ ─────────────────────────────────────


def test_life_archive_persists_across_instances(tmp_path):
    path = tmp_path / "life.sqlite"
    archive = LifeArchive(str(path))
    event_id = archive.record_event(
        "journal", "milestone", "shipped the first build", owner=True)
    assert archive.get(event_id, owner=True)["text"] == "shipped the first build"
    assert archive.search("first build", owner=True)[0]["id"] == event_id

    fresh = LifeArchive(str(path))
    rows = fresh.timeline(owner=True)
    assert [r["id"] for r in rows] == [event_id]
    assert rows[0]["kind"] == "milestone"
    assert rows[0]["source"] == "journal"
    assert path.exists()


def test_life_archive_owner_gate():
    archive = LifeArchive()
    try:
        archive.record_event("journal", "note", "x")
    except PermissionError:
        pass
    else:
        raise AssertionError("record without owner=True must refuse")
    try:
        archive.timeline()
    except PermissionError:
        pass
    else:
        raise AssertionError("timeline without owner=True must refuse")


# ── /life control ───────────────────────────────────────────────────


def test_life_empty_answers(tmp_path):
    host = _reach_host(tmp_path)
    assert "no events yet" in host._control_life("")
    assert "no events yet" in host._control_life("list")


def test_life_add_list_show_search(tmp_path):
    host = _reach_host(tmp_path)
    event_id = _add_life(host)
    listed = host._control_life("list")
    assert "shipped the first build" in listed
    assert event_id in listed
    assert "milestone" in listed
    shown = host._control_life(f"show {event_id}")
    assert "shipped the first build" in shown
    assert "milestone" in shown
    assert "source: chat" in shown
    found = host._control_life("search first build")
    assert event_id in found
    assert "no events matching" in host._control_life("search zebra")
    # Unique id-prefix resolves; an unknown id says so plainly.
    assert "shipped the first build" in host._control_life(
        f"show {event_id[:6]}")
    assert "no event matching" in host._control_life("show nope")
    assert "usage:" in host._control_life("add")
    assert "usage:" in host._control_life("add milestone")
    assert "usage:" in host._control_life("frobnicate")


def test_life_add_with_source_and_date(tmp_path):
    host = _reach_host(tmp_path)
    out = host._control_life(
        "add milestone | won the court case | journal | court:2026-11-13"
        " | 2026-11-13")
    assert out.startswith("recorded: "), out
    shown = host._control_life(f"show {out.split(':')[1].strip().split()[0]}")
    assert "2026-11-13" in shown
    assert "source: journal" in shown
    assert "provenance: court:2026-11-13" in shown
    bad = host._control_life("add milestone | text | chat |  | 13/11/2026")
    assert "YYYY-MM-DD" in bad


def test_life_list_filters_by_kind(tmp_path):
    host = _reach_host(tmp_path)
    _add_life(host, kind="milestone")
    _add_life(host, kind="note", text="bought yams")
    assert "bought yams" not in host._control_life("list milestone")
    assert "bought yams" in host._control_life("list note")
    # A numeric limit caps the newest-first list: 1 shows only the
    # newest event (the note), 2 shows both.
    assert "bought yams" in host._control_life("list 1")
    assert "shipped the first build" not in host._control_life("list 1")
    assert "shipped the first build" in host._control_life("list 2")


# ── preferences: durable organ ──────────────────────────────────────


def test_prefs_store_persists_across_instances(tmp_path):
    path = tmp_path / "preferences.sqlite"
    store = PreferenceStore(str(path))
    result = record_signal(store, "music",
                           {"genre": "afrobeats", "energy": "high"},
                           True, source="chat")
    assert result["recorded"] is True
    assert result["updates"]["genre"]["afrobeats"] == 1.0

    fresh = PreferenceStore(str(path))
    assert fresh.get_weight("music", "genre", "afrobeats") == 1.0
    assert fresh.get_weight("music", "genre", "unseen") == 0.0
    assert len(fresh.signals("music")) == 1
    assert path.exists()


def test_prefs_store_starts_at_zero():
    store = PreferenceStore()
    assert store.get_weights("music") == {}
    assert store.get_weight("music", "genre", "afrobeats") == 0.0
    assert store.domains() == sorted(DOMAIN_DESCRIPTORS)


# ── /prefs control ──────────────────────────────────────────────────


def test_prefs_empty_answers(tmp_path):
    host = _reach_host(tmp_path)
    listed = host._control_prefs("")
    assert "music" in listed
    assert "no learned weights yet" in listed
    assert "no learned weights yet" in host._control_prefs("show music")
    assert "unknown domain" in host._control_prefs("show nope")


def test_prefs_like_dislike_rank(tmp_path):
    host = _reach_host(tmp_path)
    out = host._control_prefs(
        "like music | genre:afrobeats | energy:high")
    assert "recorded" in out
    assert "afrobeats" in out
    shown = host._control_prefs("show music")
    assert "afrobeats" in shown
    assert "+1.0" in shown
    ranked = host._control_prefs(
        "rank music | Sunrise=genre:afrobeats,energy:high"
        " | Midnight=genre:afrobeats,energy:low")
    assert ranked.index("Sunrise") < ranked.index("Midnight")
    assert "+1.0" in ranked or "1.0" in ranked
    host._control_prefs("dislike music | genre:afrobeats")
    assert host._control_prefs(
        "show music").count("afrobeats") >= 1
    # A fresh store preserves the caller's order (all scores 0).
    fresh = host._control_prefs(
        "rank books | Alpha=genre:fantasy | Beta=genre:romance")
    assert fresh.index("Alpha") < fresh.index("Beta")
    assert "usage:" in host._control_prefs("like music")
    assert "unknown domain" in host._control_prefs("like nope | genre:x")
    assert "usage:" in host._control_prefs("rank music")
    assert "usage:" in host._control_prefs("frobnicate")


def test_prefs_signals_and_learn(tmp_path):
    host = _reach_host(tmp_path)
    host._control_prefs("like music | genre:afrobeats")
    host._control_prefs("dislike food | spice:hot")
    signals = host._control_prefs("signals")
    assert "afrobeats" in signals
    assert "spice" in signals
    assert "approve" in signals and "disapprove" in signals
    assert "afrobeats" in host._control_prefs("signals music")
    assert "spice" not in host._control_prefs("signals music")
    learned = host._control_prefs("learn")
    assert "recorded 0" in learned
    assert "skipped" in learned


# ── registry ────────────────────────────────────────────────────────


def test_life_registered():
    assert "life" in CONTROL_COMMANDS
    slots = [title for title, kinds in LIST_GROUPS if "life" in kinds]
    assert slots == ["memory & thinking"]
    assert "life" in LIST_ONELINERS and "/life" in LIST_ONELINERS["life"]
    assert "/life" in help_text()
    page = detailed_help("life")
    assert "try:" in page and "related:" in page
    assert "/life add" in page


def test_prefs_registered():
    assert "prefs" in CONTROL_COMMANDS
    slots = [title for title, kinds in LIST_GROUPS if "prefs" in kinds]
    assert slots == ["memory & thinking"]
    assert "prefs" in LIST_ONELINERS and "/prefs" in LIST_ONELINERS["prefs"]
    assert "/prefs" in help_text()
    page = detailed_help("prefs")
    assert "try:" in page and "related:" in page
    assert "/prefs like" in page


def test_life_prefs_owner_only_off_menu():
    # Owner-only: absent from PUBLIC_COMMANDS, so caller_may_run denies
    # non-owners before the handler runs (assets/legacy/family pattern).
    for kind in ("life", "prefs"):
        assert kind not in PUBLIC_COMMANDS
        assert caller_may_run(kind, is_owner=False) is False
        assert caller_may_run(kind, is_owner=True) is True
        # Off the BotFather menu by design: the 100-command cap is full
        # (48 owner pins) — registered, runnable, /list-ed, like the
        # reachability wave's off-menu owner commands.
        assert kind not in PERSONAL_OWNER_COMMANDS
