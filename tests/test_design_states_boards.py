"""Design absorption #6 — board/menu state coverage.

The audit, pinned: every board answers for the states it can
reach — loading / empty / error / partial / success — in ONE
grammar that lives in the shared organs (present's Section
states + state_line/board_placeholder; typography's
Card.state_section), never hand-rolled per surface.

- present: a non-ready section renders its designed state line
  in its place on every dialect; a Doc mixing ready and state
  sections is the partial board; the stored Doc form round-
  trips the state; an unknown state fails fast at construction.
- loading is a placeholder SHAPE (board_placeholder), sent only
  by a producer that edits the same message into the board —
  every board adopted here renders synchronously and sends no
  placeholder; the shape is pinned so a future async board uses
  this one, not an invention.
- leaderboards (GroupSurface.render_board): empty renders the
  designed empty state naming what fills the board; a peek at a
  games board with no provider renders the unavailable board
  (publish still skips with None); a raising read renders the
  error board (publish still skips); an unknown id is the ONLY
  None; the success render is unchanged.
- /menu: synchronous registry render — no loading, never empty
  (validate_menu), groups with nothing visible are omitted.
- /pair journal: the button-degradation section marks itself
  unavailable when the failure ledger cannot be read, instead
  of silently dropping off the board.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from nomorals.social.chat.present import (
    KV,
    Doc,
    Section,
    board_placeholder,
    doc_from_json,
    doc_to_json,
    render_text,
    state_line,
)
from nomorals.social.chat.typography import Card, render_card

G = "telegram-bot:-100111"


# ── present: the section-state machinery ─────────────────────

def test_section_states_render_on_every_dialect():
    import html as _html
    for platform in ("telegram-bot", "whatsapp", "plain"):
        out = render_text(Doc(title="B", sections=(
            Section.empty("Quests",
                          detail="quests land when one is posted"),
            Section.unavailable(
                "Games",
                detail="the games spine is not feeding here"),
            Section.error("Store",
                          detail="the read failed — try again"),
        )), platform)
        # Telegram's engine entity-escapes prose (clients render
        # the entity as the character) — assert on the text the
        # reader actually sees.
        out = _html.unescape(out)
        assert "nothing here yet" in out
        assert "quests land when one is posted" in out
        assert "unavailable" in out
        assert "the games spine is not feeding here" in out
        assert "couldn't load" in out
        assert "the read failed — try again" in out
        assert "Quests" in out and "Games" in out


def test_state_section_replaces_its_content():
    # A producer that declares a section unreadable must not
    # leak stale rows beside the state line.
    doc = Doc(title="B", sections=(
        Section("Rows", rows=[KV("Secret", "stale")],
                state="unavailable", state_detail="gone"),
    ))
    out = render_text(doc, "telegram-bot")
    assert "unavailable" in out
    assert "stale" not in out


def test_ready_section_renders_as_before():
    doc = Doc(title="B", sections=(
        Section("Rows", rows=[KV("Sparks", "12")]),))
    out = render_text(doc, "telegram-bot")
    assert "Sparks" in out and "12" in out
    assert "nothing here yet" not in out


def test_partial_doc_good_sections_render_missing_marked():
    doc = Doc(title="Status", sections=(
        Section("Account", rows=[KV("Balance", "10")]),
        Section.unavailable("Feed", detail="bridge is down"),
        Section.empty("Alerts", detail="alerts land on a hit"),
    ))
    out = render_text(doc, "whatsapp")
    assert "Balance" in out and "10" in out
    assert "unavailable" in out and "bridge is down" in out
    assert "nothing here yet" in out


def test_unknown_section_state_fails_fast():
    with pytest.raises(ValueError):
        Section(state="loading-ish")
    with pytest.raises(ValueError):
        state_line("bogus")


def test_state_line_shapes():
    assert state_line("empty") == "🗒 nothing here yet"
    assert state_line("unavailable", "no feed").startswith(
        "⚠️ unavailable — ")
    assert state_line("error", "boom").startswith(
        "❌ couldn't load — ")


def test_doc_json_roundtrip_preserves_state():
    doc = Doc(title="B", sections=(
        Section("A", rows=[KV("x", "1")]),
        Section.unavailable("G", detail="not here"),
    ))
    back = doc_from_json(doc_to_json(doc))
    assert back is not None
    assert back.sections[0].state == "ready"
    assert back.sections[1].state == "unavailable"
    assert back.sections[1].state_detail == "not here"
    assert "unavailable" in render_text(back, "telegram-bot")


def test_doc_json_without_state_keys_reads_ready():
    # A payload stored before board states existed has no state
    # keys; it must read back as a plain ready Doc.
    import json
    payload = {"v": 1, "title": "B", "icon": "", "tone": "info",
               "summary": "", "footer": "", "footer_unix": None,
               "actions": [], "overflow_text": "",
               "sections": [{"title": "A", "rows": [],
                             "lines": ["hi"], "quote": [],
                             "expandable": False, "pre": []}]}
    back = doc_from_json(json.dumps(payload))
    assert back is not None
    assert back.sections[0].state == "ready"
    assert "hi" in render_text(back, "plain")


def test_board_placeholder_is_the_one_loading_shape():
    assert board_placeholder("🏆 sparks") == "🏆 sparks\n…"
    assert board_placeholder() == "…"


# ── typography: cards inherit the states ─────────────────────

def test_card_state_section_renders_with_stamp():
    card = Card("leaderboard", title="quests", icon="🏆")
    card.state_section(
        "empty",
        detail="quests land when the first quest is posted.")
    out = render_card(card, platform="telegram-bot",
                      settings=None).text
    assert "nothing here yet" in out
    assert "quests land when the first quest is posted." in out
    assert "Powered by peacethefirst" in out


def test_card_partial_mix_renders_both_halves():
    card = Card("game", title="room", icon="🎲")
    card.stats([("Ada", 3, True)], title="players")
    card.state_section("unavailable", title="spectators",
                       detail="the room feed is not readable")
    out = render_card(card, platform="telegram-bot",
                      settings=None).text
    assert "Ada" in out
    assert "unavailable" in out
    assert "the room feed is not readable" in out


# ── leaderboards: the adopted board ──────────────────────────

def _surface(tmp_path):
    from nomorals.social.chat.group_surface import GroupStore, GroupSurface
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def test_leaderboard_empty_state_names_what_fills_it(tmp_path):
    surface = _surface(tmp_path)
    out = surface.render_board("community:other:sparks")
    assert "no standings yet" in out
    assert "nothing here yet" in out
    assert "carrying this community tag" in out
    out = surface.render_board("group:sparks", group_key=G)
    assert "no standings yet" in out
    assert "in this group" in out
    out = surface.render_board(
        "games:xp",
        games_rows=[])
    assert "no standings yet" in out
    assert "games are played" in out


def test_leaderboard_unavailable_games_board(tmp_path):
    surface = _surface(tmp_path)
    # A peek answers as a board: which board, and why no rows.
    out = surface.render_board("games:xp")
    assert out is not None
    assert "unavailable" in out
    assert "games spine is not feeding" in out
    # The publish path keeps its skip contract: no state board
    # is ever posted as a scheduled board.
    assert surface.render_board("games:xp", publish=True) is None


def test_leaderboard_error_board_and_publish_skip(tmp_path):
    surface = _surface(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("store gone")

    surface.board_rows = _boom
    out = surface.render_board("group:sparks", group_key=G)
    assert out is not None
    import html as _html
    assert "couldn't load" in _html.unescape(out)
    assert "RuntimeError" in out
    assert surface.render_board(
        "group:sparks", group_key=G, publish=True) is None


def test_leaderboard_unknown_id_is_the_only_none(tmp_path):
    surface = _surface(tmp_path)
    assert surface.render_board("nonsense", group_key=G) is None
    assert surface.render_board(
        "group:reputation", group_key=G) is None


def test_leaderboard_success_render_unchanged(tmp_path):
    surface = _surface(tmp_path)
    surface.handle_message(G, "55", username="bola", name="Bola",
                           text="hello group")
    surface.game_award(G, "55", 100, "ttt", now=time.time())
    out = surface.render_board("group:sparks", group_key=G,
                               publish=True)
    assert "Bola" in out
    assert "nothing here yet" not in out
    assert "Powered by peacethefirst" in out


# ── /menu: the synchronous board ─────────────────────────────

def test_menu_success_and_omission_contract():
    # RE-POINTED (presentation Phase 2, 2026-11-11): telegram_menu's
    # own text renderer was a dead second menu and is removed; the
    # contract now pins the LIVE composition (control.menu_home_text)
    # against the same coverage invariant.
    from nomorals.social.chat.control import _menu_session_sections, menu_home_text
    from nomorals.social.chat.telegram_menu import validate_menu
    ok, problems = validate_menu()
    assert ok and problems == []
    for public_only in (False, True):
        text = menu_home_text("telegram", public_only=public_only)
        assert text.strip()
        # every section visible to this caller renders under its
        # head with its commands — a section with nothing visible
        # to the caller is omitted, never blank.
        for emoji, title, items in _menu_session_sections(
                "telegram", public_only):
            assert f"{emoji} {title}" in text
            for cmd, _desc in items:
                assert cmd in text


# ── /pair journal: the partial section ───────────────────────

def _pair_runtime(tmp_path):
    from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin
    from nomorals.social.pairing import AccountRecord, PairingRegistry
    from nomorals.social.personal import PersonalSurfaces, PersonalVault, _append_jsonl
    acc = "whatsapp:2348012345678"
    registry = PairingRegistry(tmp_path / "paired.json")
    registry.upsert(AccountRecord(
        id=acc, platform="whatsapp", identity="2348012345678",
        status="linked", created_at=1.0))
    personal = PersonalSurfaces(
        registry, PersonalVault(tmp_path / "personal"))
    rt = object.__new__(RuntimeProvisionMixin)
    rt._pair_supervisor = SimpleNamespace(registry=registry)
    rt._pair_personal_svc = {"registry": registry,
                             "personal": personal}
    rt.context = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    row = {"ts": 1762000100.0, "kind": "status_react",
           "account": acc, "platform": "whatsapp",
           "counterpart_id": "", "poster_name": "Ada",
           "media_kind": "", "source_id": "", "path": "",
           "bytes": 0, "note": "reacted 🔥"}
    _append_jsonl(personal.vault.journal_path, row)
    return rt


def test_pair_journal_marks_unreadable_section(tmp_path,
                                               monkeypatch):
    rt = _pair_runtime(tmp_path)
    healthy = rt._control_pair("journal")
    assert "button sends degraded" in healthy

    import nomorals.social.chat.failures as failures

    def _boom(*args, **kwargs):
        raise RuntimeError("ledger gone")

    monkeypatch.setattr(failures, "recent_failures", _boom)
    out = rt._control_pair("journal")
    # the journal itself still renders; the one unreadable
    # section is marked in place, not silently dropped.
    assert "reacted 🔥" in out
    assert "button sends degraded" not in out
    assert "unavailable" in out
    assert "button-degradation count" in out
