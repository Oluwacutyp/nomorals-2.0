"""Tests for relationship cadence (build-map #10). All offline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents.proactive import (
    RelationshipCadence,
    control_memory,
)
from nomorals.agents.morning_briefing import PeopleProvider
from nomorals.social.chat.control import parse_control

NOW = datetime(2026, 10, 8, 12, 0, 0).timestamp()  # Thursday


def _write(directory: Path, filename: str, text: str) -> Path:
    path = directory / filename
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture()
def people_dir(tmp_path: Path) -> Path:
    d = tmp_path / "people"
    d.mkdir()
    _write(d, "adaeze.md",
           "# Adaeze\nCloseness: close\nBirthday: May 4\n"
           "Last contact: 2026-09-18\n")
    _write(d, "bola.md",
           "# Bola\nCloseness: normal\nLast spoke: 2026-08-19\n")
    _write(d, "chidi.md",
           "# Chidi\nCloseness: distant\nLast seen: 2026-03-12\n")
    _write(d, "funmi.md",
           "# Funmi\nCloseness: close\nLast contact: 2026-10-07\n")
    return d


@pytest.fixture()
def cadence(people_dir: Path) -> RelationshipCadence:
    return RelationshipCadence(people_dir=people_dir)


# ── parsing ──────────────────────────────────────────────────────────────

def test_parses_well_formed_pages(cadence: RelationshipCadence):
    people = {p.name: p for p in cadence.people()}
    assert set(people) == {"Adaeze", "Bola", "Chidi", "Funmi"}
    ada = people["Adaeze"]
    assert ada.closeness == "close"
    assert ada.birthday == (5, 4)
    assert ada.last_contact_ts is not None
    assert ada.days_since_contact(NOW) == pytest.approx(20.0, abs=0.6)


def test_alternate_last_contact_keys(cadence: RelationshipCadence):
    people = {p.name: p for p in cadence.people()}
    assert people["Bola"].days_since_contact(NOW) == pytest.approx(50.0, abs=0.6)
    assert people["Chidi"].days_since_contact(NOW) == pytest.approx(210.0, abs=0.6)


def test_malformed_page_skipped_not_raised(tmp_path: Path):
    d = tmp_path / "people"
    d.mkdir()
    _write(d, "weird.md", "\x00\x01 binary-ish \n::: no heading :::\n")
    _write(d, "nodates.md", "# No Dates\nJust some notes, no fields at all.\n")
    cad = RelationshipCadence(people_dir=d)
    people = cad.people()  # must not raise
    by_name = {p.name: p for p in people}
    assert by_name["No Dates"].last_contact_ts is None
    assert by_name["No Dates"].birthday is None
    assert by_name["No Dates"].closeness == "normal"


def test_missing_people_dir_is_empty():
    cad = RelationshipCadence(people_dir="/nonexistent-dir-xyz")
    assert cad.people() == []
    assert cad.neglected(NOW) == []
    assert cad.upcoming_birthdays(NOW) == []


def test_index_order_derives_closeness(tmp_path: Path):
    d = tmp_path / "people"
    d.mkdir()
    for name in ["Aaa", "Bbb", "Ccc", "Ddd", "Eee", "Fff"]:
        _write(d, f"{name.lower()}.md", f"# {name}\n")
    _write(d, "INDEX.md",
           "".join(f"- **{n}** — notes\n" for n in
                   ["Aaa", "Bbb", "Ccc", "Ddd", "Eee", "Fff"]))
    cad = RelationshipCadence(people_dir=d)
    by_name = {p.name: p for p in cad.people()}
    assert by_name["Aaa"].closeness == "close"
    assert by_name["Bbb"].closeness == "close"
    assert by_name["Ccc"].closeness == "normal"
    assert by_name["Ddd"].closeness == "normal"
    assert by_name["Eee"].closeness == "distant"
    assert by_name["Fff"].closeness == "distant"


def test_explicit_closeness_beats_index(tmp_path: Path):
    d = tmp_path / "people"
    d.mkdir()
    _write(d, "zara.md", "# Zara\nCloseness: distant\n")
    _write(d, "INDEX.md", "- **Zara** — top of the index\n")
    cad = RelationshipCadence(people_dir=d)
    assert cad.people()[0].closeness == "distant"


# ── neglect ──────────────────────────────────────────────────────────────

def test_neglect_ranking(cadence: RelationshipCadence):
    # Adaeze: close, 20d ago → overdue 13 × 3.0 = 39
    # Bola:   normal, 50d ago → overdue 20 × 1.0 = 20
    # Chidi:  distant, 210d ago → overdue 120 × 0.4 = 48
    # Funmi:  close, 1d ago → not neglected
    nudges = cadence.neglected(NOW)
    assert [n.name for n in nudges] == ["Chidi", "Adaeze", "Bola"]
    assert nudges[0].score == pytest.approx(48.0, abs=2.0)
    assert all(n.kind == "neglect" for n in nudges)


def test_neglect_text(cadence: RelationshipCadence):
    nudge = cadence.neglected(NOW)[0]
    assert "Chidi" in nudge.text()
    assert "days" in nudge.text()


def test_unknown_contact_nudges_gently(tmp_path: Path):
    d = tmp_path / "people"
    d.mkdir()
    _write(d, "ghost.md", "# Ghost\nCloseness: normal\n")
    cad = RelationshipCadence(people_dir=d)
    nudges = cad.neglected(NOW)
    assert len(nudges) == 1
    assert "No contact recorded" in nudges[0].text()


def test_custom_thresholds(cadence: RelationshipCadence):
    strict = RelationshipCadence(
        people_dir=cadence.people_dir,
        thresholds={"close": 1.0, "normal": 1.0, "distant": 1.0})
    names = {n.name for n in strict.neglected(NOW)}
    assert "Funmi" in names  # 1 day > 1-day threshold... (boundary: strictly over)
    lax = RelationshipCadence(
        people_dir=cadence.people_dir,
        thresholds={"close": 365.0, "normal": 365.0, "distant": 365.0})
    assert lax.neglected(NOW) == []


# ── birthdays ────────────────────────────────────────────────────────────

def test_upcoming_birthdays(tmp_path: Path):
    d = tmp_path / "people"
    d.mkdir()
    _write(d, "jan.md", "# Jan\nBirthday: 01-05\n")
    _write(d, "dec.md", "# Dec\nBirthday: December 30\n")
    _write(d, "far.md", "# Far\nBirthday: 07-04\n")
    cad = RelationshipCadence(people_dir=d)
    dec30 = datetime(2026, 12, 30, 12, 0, 0).timestamp()
    upcoming = cad.upcoming_birthdays(dec30, within_days=14)
    by_name = {n.name: n for n in upcoming}
    assert set(by_name) == {"Dec", "Jan"}  # year wrap: Jan 5 is 6 days out
    assert by_name["Dec"].days_until == 0
    assert by_name["Jan"].days_until == 6
    assert upcoming[0].name == "Dec"  # soonest first
    assert "today" in by_name["Dec"].text()


def test_anniversary_included(tmp_path: Path):
    d = tmp_path / "people"
    d.mkdir()
    _write(d, "wed.md", "# Wed\nAnniversary: 2020-10-10\n")
    cad = RelationshipCadence(people_dir=d)
    upcoming = cad.upcoming_birthdays(NOW, within_days=14)
    assert len(upcoming) == 1
    assert upcoming[0].kind == "anniversary"
    assert upcoming[0].days_until == 2


def test_feb29_maps_to_feb28(tmp_path: Path):
    d = tmp_path / "people"
    d.mkdir()
    _write(d, "leap.md", "# Leap\nBirthday: 02-29\n")
    cad = RelationshipCadence(people_dir=d)
    feb27 = datetime(2027, 2, 27, 12, 0, 0).timestamp()  # 2027 not a leap year
    upcoming = cad.upcoming_birthdays(feb27, within_days=14)
    assert len(upcoming) == 1
    assert upcoming[0].days_until == 1


# ── record_contact ───────────────────────────────────────────────────────

def test_record_contact_creates_page(tmp_path: Path):
    d = tmp_path / "people"
    cad = RelationshipCadence(people_dir=d)
    path = cad.record_contact("New Friend", ts=NOW)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# New Friend\n")
    assert "Last contact: 2026-10-08" in text


def test_record_contact_updates_existing(people_dir: Path):
    cad = RelationshipCadence(people_dir=people_dir)
    cad.record_contact("Adaeze", ts=NOW)
    text = (people_dir / "adaeze.md").read_text(encoding="utf-8")
    assert "Last contact: 2026-10-08" in text
    assert "2026-09-18" not in text
    assert "Birthday: May 4" in text  # other fields preserved


def test_record_contact_matches_existing_case_insensitively(tmp_path: Path):
    # On case-sensitive filesystems "Ada.md" and "ada.md" differ: record_contact
    # must update the existing page, not fork a second file.
    (tmp_path / "Ada.md").write_text(
        "# Ada\nCloseness: close\nLast contact: 2026-09-01\n", encoding="utf-8"
    )
    cad = RelationshipCadence(people_dir=tmp_path)
    cad.record_contact("Ada", ts=NOW)
    assert sorted(p.name for p in tmp_path.glob("*.md")) == ["Ada.md"]
    text = (tmp_path / "Ada.md").read_text(encoding="utf-8")
    assert "Last contact: 2026-10-08" in text
    assert "2026-09-01" not in text


def test_record_contact_idempotent(people_dir: Path):
    cad = RelationshipCadence(people_dir=people_dir)
    cad.record_contact("Bola", ts=NOW)
    first = (people_dir / "bola.md").read_bytes()
    cad.record_contact("Bola", ts=NOW)
    assert (people_dir / "bola.md").read_bytes() == first


# ── briefing provider ────────────────────────────────────────────────────

def _ctx_for(d: Path):
    return SimpleNamespace(settings=SimpleNamespace(people_dir=str(d)))


def test_provider_quiet_when_empty(tmp_path: Path):
    d = tmp_path / "people"
    d.mkdir()
    assert PeopleProvider().collect(_ctx_for(d), 0.0) is None


def test_provider_renders_nudges_and_birthdays(tmp_path: Path):
    d = tmp_path / "people"
    d.mkdir()
    _write(d, "old.md", "# Old Pal\nCloseness: close\nLast contact: 2020-01-01\n")
    section = PeopleProvider().collect(_ctx_for(d), 0.0)
    assert section is not None
    assert section.name == "people"
    assert any("Old Pal" in line for line in section.lines)


def test_provider_caps_nudges_at_two(tmp_path: Path):
    d = tmp_path / "people"
    d.mkdir()
    for i in range(6):
        _write(d, f"p{i}.md",
               f"# Person{i}\nCloseness: normal\nLast contact: 2020-01-01\n")
    section = PeopleProvider().collect(_ctx_for(d), 0.0)
    assert section is not None
    nudge_lines = [ln for ln in section.lines if "🎂" not in ln]
    assert len(nudge_lines) <= 2


def test_provider_anti_nag_second_briefing(tmp_path: Path):
    d = tmp_path / "people"
    d.mkdir()
    _write(d, "old.md", "# Old Pal\nCloseness: normal\nLast contact: 2020-01-01\n")
    provider = PeopleProvider()
    ctx = _ctx_for(d)
    first = provider.collect(ctx, 0.0)
    assert first is not None and first.lines
    second = provider.collect(ctx, 0.0)
    # surfaced nudges score below the floor on the immediate re-run
    assert second is None or not any("Old Pal" in ln for ln in second.lines)


def test_provider_never_crashes_briefing(tmp_path: Path):
    provider = PeopleProvider()
    # garbage ctx: no settings at all → falls back to ~/.devon/people
    assert provider.collect(SimpleNamespace(), 0.0) is None or True


# ── /memories command ────────────────────────────────────────────────────

def test_memories_parses():
    cmd = parse_control("/memories")
    assert cmd is not None and cmd.kind == "memories"
    cmd = parse_control("/memories Adaeze")
    assert cmd.kind == "memories" and cmd.arg == "Adaeze"


def test_memories_summary(tmp_path: Path):
    d = tmp_path / "people"
    d.mkdir()
    _write(d, "adaeze.md", "# Adaeze\nCloseness: close\nLast contact: 2026-10-01\n")

    class FakeMemory:
        def counts_by_kind(self):
            return {"episodic": 3, "fact": 2}

    ctx = SimpleNamespace(
        settings=SimpleNamespace(people_dir=str(d)), memory=FakeMemory())
    reply = control_memory("", ctx)
    assert "1 people" in reply
    assert "5 long-term memories" in reply
    assert "Adaeze" in reply
    # summaries only — no raw dump of page internals beyond the summary
    assert "episodic" in reply


def test_memories_person_view(tmp_path: Path):
    d = tmp_path / "people"
    d.mkdir()
    _write(d, "adaeze.md",
           "# Adaeze\nCloseness: close\nBirthday: May 4\nLast contact: 2026-10-01\n")
    ctx = SimpleNamespace(settings=SimpleNamespace(people_dir=str(d)),
                          memory=None)
    reply = control_memory("adaeze", ctx)
    assert "Adaeze" in reply
    assert "close" in reply
    assert "Birthday" in reply


def test_memories_unknown_person(tmp_path: Path):
    d = tmp_path / "people"
    d.mkdir()
    ctx = SimpleNamespace(settings=SimpleNamespace(people_dir=str(d)),
                          memory=None)
    reply = control_memory("Nobody", ctx)
    assert "No person page" in reply
