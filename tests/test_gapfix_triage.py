"""Smart notification triage, wired into production.

Audit gap R1-3: ``nomorals/social/triage.py`` scored every inbound
message on paper and nobody called it — ``TriageLog`` was in-memory
only, so even the scoring had nowhere to land.  The wiring:

* ``_triage_inbound`` (runtime_reach, next to ``_presence_watch_event``)
  scores every message that flows on to an answer in
  ``_process_inner``, records it in the durable log, and buzzes the
  owner through the notifier only when the scorer says so.
* ``TriageLogStore`` persists the log under ``<home>/data/triage.json``
  (atomic tmp+replace, corrupt → quarantined — the PresenceWatchStore
  house pattern).
* ``TriageProvider`` surfaces ``render_digest()`` in the morning
  briefing (priority 30, beside People 25 / radar 35); absent when
  the log holds nothing.

Every test here drives the PRODUCTION wiring — the durable store
under a throwaway home and the reach-mixin hook — never the organ
in isolation where a seam exists.  Teeth: on the pristine base
each wiring test fails for the missing store (``ModuleNotFoundError``)
and the missing hook (``AttributeError``); the pure scorer tests pin
the organ's contract, which this lane does not change.

Closeness honesty (read from the real repo before building):
``RelationshipTracker.note_inbound`` has NO production caller, and
``PersonStore.record_inbound`` has none either — the trackers are
read but never fed, so a read-only closeness lookup would score
everyone 0 forever.  The hook therefore feeds the tracker with the
same call the tracker documents ("They messaged. Bumps closeness
by recency-weighted frequency") and reads the recomputed closeness
back.  The person key is ``<platform>:<sender_id>`` — stable per
sender, never a display-name guess.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.social.chat.base import (
    ChatKind,
    ChatMessage,
    ChatRef,
)
from nomorals.social.triage import (
    TriageLogStore,
    render_digest,
    triage_message,
)


def _store(tmp_path, *, max_entries=500):
    return TriageLogStore(
        Path(tmp_path) / "data" / "triage.json",
        max_entries=max_entries)


def _reach_host(tmp_path):
    """Production reach host on a throwaway home (family harness)."""
    from nomorals.agents.partner.runtime_reach import RuntimeReachMixin

    host = RuntimeReachMixin()
    host.context = SimpleNamespace(
        settings=SimpleNamespace(home_path=Path(tmp_path)))
    return host


def _msg(*, platform="telegram", chat_id="987654", kind=ChatKind.DM,
         text="", sender="Zoey", sender_id="987654", mentioned=False,
         reply_to="", meta=None):
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=chat_id, kind=kind,
                     title="", peer=chat_id),
        incoming=True, text=text, sender=sender, sender_id=sender_id,
        mentioned=mentioned, reply_to=reply_to, meta=meta or {})


def _fake_registry(is_owner=False):
    registry = mock.Mock()
    registry.is_owner.return_value = is_owner
    return registry


def _seed_close_contact(host, *, platform="telegram", sender_id="987654",
                        name="Zoey"):
    """Drive the tracker to close-contact closeness the honest way:
    owner-initiated outbound (dominates the formula) + inbound."""
    tracker = host._triage_tracker()
    person_id = f"{platform}:{sender_id}"
    for _ in range(5):
        tracker.note_outbound(person_id, owner_initiated=True,
                              display_name=name)
    for _ in range(10):
        tracker.note_inbound(person_id, display_name=name)
    rel = tracker.get(person_id)
    assert rel is not None and rel.closeness >= 0.6
    return rel.closeness


# ── pure scorer (contract pinned; this lane does not change it) ──────


def test_close_contact_urgent_ask_is_critical_and_buzzes():
    score = triage_message(
        _msg(text="Mum is in hospital, can you come by 5pm?",
             mentioned=True, reply_to="m1"),
        is_owner=False, closeness=0.9, kind="dm",
        mentioned=True, reply_to_owner=True)
    assert score.tier == "critical"
    assert score.buzz is True
    assert score.score >= 0.75


def test_important_held_in_quiet_hours_and_buzzes_by_day():
    # 03:00 local: important waits for morning; 12:00: it buzzes.
    night = datetime.datetime(2026, 11, 16, 3, 0).timestamp()
    day = datetime.datetime(2026, 11, 16, 12, 0).timestamp()
    held = triage_message(
        _msg(text="can we talk tomorrow?"), is_owner=False,
        closeness=0.8, kind="dm", now=night)
    assert held.tier == "important"
    assert held.buzz is False
    assert "held for morning" in " ".join(held.reasons)
    buzzing = triage_message(
        _msg(text="can we talk tomorrow?"), is_owner=False,
        closeness=0.8, kind="dm", now=day)
    assert buzzing.tier == "important"
    assert buzzing.buzz is True


def test_group_ack_is_noise():
    score = triage_message(
        _msg(text="ok"), is_owner=False, closeness=0.0, kind="group")
    assert score.tier == "noise"
    assert score.buzz is False


def test_stranger_question_is_not_important():
    score = triage_message(
        _msg(text="who is this?"), is_owner=False, closeness=0.0,
        kind="dm")
    assert score.tier == "noise"
    assert score.buzz is False


# ── durable store ───────────────────────────────────────────────────


def test_log_persists_across_instances(tmp_path):
    store = _store(tmp_path)
    store.record("telegram:1", "critical", 0.9, sender="Zoey")
    store.record("telegram:2", "noise", 0.05, sender="Spammer")

    fresh = _store(tmp_path)
    digest = fresh.digest(since_hours=24.0)
    assert digest["total"] == 2
    assert digest["by_tier"] == {"critical": 1, "noise": 1}
    assert len(digest["needs_attention"]) == 1
    assert digest["needs_attention"][0]["sender"] == "Zoey"


def test_corrupt_file_quarantined_starts_empty(tmp_path):
    path = Path(tmp_path) / "data" / "triage.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    store = _store(tmp_path)
    assert store.digest(since_hours=24.0)["total"] == 0
    assert store.corrupted is not None
    assert Path(store.corrupted).exists()


def test_entries_trim_to_max(tmp_path):
    store = _store(tmp_path, max_entries=5)
    for i in range(8):
        store.record(f"telegram:{i}", "routine", 0.3, sender=f"p{i}")
    assert store.digest(since_hours=24.0)["total"] == 5


def test_render_digest_names_attention(tmp_path):
    store = _store(tmp_path)
    store.record("telegram:1", "critical", 0.9, sender="Zoey")
    text = render_digest(store.digest(since_hours=24.0))
    assert "Zoey" in text
    assert "critical" in text


# ── hook (production wiring) ────────────────────────────────────────


def test_hook_records_routine_message_quietly(tmp_path):
    host = _reach_host(tmp_path)
    host._owner_registry = lambda: _fake_registry(False)
    delivered = []
    with mock.patch(
            "nomorals.agents.notifier.notify",
            side_effect=lambda ctx, kind, title, body, **kw: (
                delivered.append((kind, title, body)), {})):
        result = host._triage_inbound(_msg(text="hello there"))
    assert result is not None
    assert result["tier"] in ("routine", "noise")
    assert result["buzz"] is False
    assert delivered == []
    digest = _store(tmp_path).digest(since_hours=24.0)
    assert digest["total"] == 1


def test_hook_buzzes_critical_from_close_contact(tmp_path):
    host = _reach_host(tmp_path)
    host._owner_registry = lambda: _fake_registry(False)
    _seed_close_contact(host)
    delivered = []
    with mock.patch(
            "nomorals.agents.notifier.notify",
            side_effect=lambda ctx, kind, title, body, **kw: (
                delivered.append((kind, title, body)), {})):
        result = host._triage_inbound(_msg(
            text="Mum is in hospital, can you come by 5pm?",
            mentioned=True, reply_to="m1"))
    assert result is not None
    assert result["tier"] == "critical"
    assert result["buzz"] is True
    assert len(delivered) == 1
    assert delivered[0][0] == "triage"
    assert "Zoey" in delivered[0][1]
    digest = _store(tmp_path).digest(since_hours=24.0)
    assert digest["by_tier"].get("critical") == 1
    assert digest["needs_attention"][0]["sender"] == "Zoey"


def test_hook_skips_owner_messages(tmp_path):
    host = _reach_host(tmp_path)
    host._owner_registry = lambda: _fake_registry(True)
    delivered = []
    with mock.patch(
            "nomorals.agents.notifier.notify",
            side_effect=lambda ctx, kind, title, body, **kw: (
                delivered.append((kind, title, body)), {})):
        result = host._triage_inbound(_msg(text="urgent, call me now?"))
    assert result is None
    assert delivered == []
    assert not (Path(tmp_path) / "data" / "triage.json").exists()


def test_hook_skips_own_message_meta(tmp_path):
    host = _reach_host(tmp_path)
    host._owner_registry = lambda: _fake_registry(False)
    result = host._triage_inbound(
        _msg(text="urgent, call me now?", meta={"own_message": True}))
    assert result is None
    assert not (Path(tmp_path) / "data" / "triage.json").exists()


def test_hook_group_message_diluted_no_buzz(tmp_path):
    host = _reach_host(tmp_path)
    host._owner_registry = lambda: _fake_registry(False)
    _seed_close_contact(host)
    delivered = []
    with mock.patch(
            "nomorals.agents.notifier.notify",
            side_effect=lambda ctx, kind, title, body, **kw: (
                delivered.append((kind, title, body)), {})):
        result = host._triage_inbound(_msg(
            text="can someone cover my shift tomorrow?",
            kind=ChatKind.GROUP, chat_id="g1"))
    assert result is not None
    assert result["tier"] in ("routine", "important")
    assert result["buzz"] is False
    assert delivered == []


def test_hook_tracker_fed_closeness_grows(tmp_path):
    host = _reach_host(tmp_path)
    host._owner_registry = lambda: _fake_registry(False)
    first = host._triage_inbound(_msg(text="hi"))
    second = host._triage_inbound(_msg(text="hi again"))
    assert first is not None and second is not None
    assert second["score"] > first["score"]
    tracker = host._triage_tracker()
    rel = tracker.get("telegram:987654")
    assert rel is not None
    assert rel.inbound_count == 2
    assert rel.display_name == "Zoey"


def test_hook_never_raises_on_corrupt_store(tmp_path):
    host = _reach_host(tmp_path)
    host._owner_registry = lambda: _fake_registry(False)
    path = Path(tmp_path) / "data" / "triage.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    result = host._triage_inbound(_msg(text="hello there"))
    assert result is not None
    assert result["tier"] in ("routine", "noise")


# ── briefing provider ───────────────────────────────────────────────


def test_provider_absent_without_home():
    from nomorals.agents.morning_briefing import TriageProvider

    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir="/tmp/x"))
    assert TriageProvider().collect(ctx, 0) is None


def test_provider_absent_without_log(tmp_path):
    from nomorals.agents.morning_briefing import TriageProvider

    ctx = SimpleNamespace(
        settings=SimpleNamespace(home_path=Path(tmp_path)))
    assert TriageProvider().collect(ctx, 0) is None


def test_provider_renders_digest(tmp_path):
    from nomorals.agents.morning_briefing import TriageProvider

    store = _store(tmp_path)
    store.record("telegram:1", "critical", 0.9, sender="Zoey")
    store.record("telegram:2", "routine", 0.3, sender="Ada")
    ctx = SimpleNamespace(
        settings=SimpleNamespace(home_path=Path(tmp_path)))
    section = TriageProvider().collect(ctx, 0)
    assert section is not None
    assert section.name == "triage"
    assert section.priority == 30
    text = "\n".join(section.lines)
    assert "Zoey" in text
    assert "critical" in text


def test_provider_registered_in_composer():
    from nomorals.agents.morning_briefing import BriefingComposer

    names = [p.name for p in BriefingComposer().providers]
    assert "triage" in names


def test_provider_empty_log_absent(tmp_path):
    from nomorals.agents.morning_briefing import TriageProvider

    path = Path(tmp_path) / "data" / "triage.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"version": 1, "entries": []}),
                    encoding="utf-8")
    ctx = SimpleNamespace(
        settings=SimpleNamespace(home_path=Path(tmp_path)))
    assert TriageProvider().collect(ctx, 0) is None
