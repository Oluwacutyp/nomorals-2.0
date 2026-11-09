"""WS3 P7 — repurposing, the GS desk landing, the stamp root fix.

Pins, per the pass brief:
(a) a posted draft repurposes into a short draft + a long-form
    draft, each pending_review with provenance, source unmutated;
    the skeptic blocks an adaptation stating untraced numbers;
(b) a day record repurposes into a short draft;
(c) a desk bundle (contract-faithful fixture in the GS writer's
    shape) lands exactly one draft with the PNG attached as-is;
    a tampered bundle lands none + a named failure; a non-desk
    bundle lands none; the gateway inbound seam lands it too;
(d) typography: a message QUOTING the stamp text mid-body still
    gets its footer; a message carrying the exact footer gets no
    second one; ordinary messages unchanged;
(e) /repurpose is catalogued owner-only (the P6 pin shape).
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import pytest

from nomorals.agents.dayrecord import DayStore
from nomorals.agents.desk_drafts import (
    GS_DESK_BUNDLE_KIND,
    inspect_desk_bundle,
    land_desk_draft,
)
from nomorals.agents.repurpose import (
    FORM_POST,
    FORM_SHORT_SCRIPT,
    FORM_THREAD,
    day_source,
    posted_draft_sources,
    repurpose_source,
)
from nomorals.social.chat.typography import (
    DEFAULT_STAMP_TEXT,
    STAMP_HTML,
    apply_stamp,
)
from nomorals.social.drafts import (
    PENDING_REVIEW,
    POSTED,
    DraftQueue,
)
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database


@pytest.fixture()
def db():
    database = Database(":memory:")
    database.migrate()
    return database


@pytest.fixture()
def queue(tmp_path):
    q = DraftQueue(db_path=str(tmp_path / "drafts.db"))
    yield q
    q.close()


@pytest.fixture()
def store(db, tmp_path):
    return ArtifactStore(db, BlobStore(db, tmp_path / "blobs"))


POSTED_TEXT = (
    "Shipped the new desk card today.\n\n"
    "Gold closed the day up 12 dollars and the desk took 4 trades"
    " at a 75% win rate.\n\n"
    "Equity sits at 1,250 USD and the month is young.")


def _posted(queue, text=POSTED_TEXT):
    draft = queue.create_draft(text, platforms=["x", "linkedin"])
    queue.propose(draft.id)
    queue.approve(draft.id)
    queue.mark_posted(draft.id)
    return queue.get(draft.id)


# ── (a) posted draft → other forms ───────────────────────────────────

def test_posted_draft_repurposes_to_short_and_thread(db, queue):
    source_draft = _posted(queue)
    before = queue.get(source_draft.id)
    sources = posted_draft_sources(queue)
    assert len(sources) == 1 and sources[0].kind == "posted_draft"

    short = repurpose_source(db, queue, sources[0], FORM_SHORT_SCRIPT)
    assert short["ok"], short
    short_draft = queue.get(short["draft_id"])
    assert short_draft.status == PENDING_REVIEW
    assert short_draft.id != source_draft.id
    meta = short_draft.metadata
    assert meta["origin"] == "repurpose"
    assert meta["source_kind"] == "posted_draft"
    assert meta["source_ref"] == source_draft.id
    assert meta["form"] == "short_script"
    assert meta["contract_id"] == short["contract_id"]
    assert meta["drafter"] in ("brain", "template")
    # the source's facts ride along, traced
    assert "75%" in short_draft.content

    thread = repurpose_source(db, queue, sources[0], FORM_THREAD)
    assert thread["ok"], thread
    thread_draft = queue.get(thread["draft_id"])
    assert thread_draft.status == PENDING_REVIEW
    assert thread_draft.metadata["form"] == "thread"
    assert thread_draft.content.startswith("1/ ")
    assert thread_draft.platforms == ["x", "linkedin"]

    # the source is never mutated
    after = queue.get(source_draft.id)
    assert after.status == POSTED
    assert after.content == before.content
    assert after.metadata == before.metadata

    # receipts: one per landed piece, chain facts recorded
    from nomorals.agents.content_cell import ReceiptStore

    receipts = ReceiptStore(db).for_contract(short["contract_id"])
    assert len(receipts) == 1
    assert receipts[0]["kind"] == "repurpose"
    assert receipts[0]["draft_id"] == short["draft_id"]
    assert receipts[0]["chain"]["source_kind"] == "posted_draft"


def test_skeptic_blocks_untraced_numbers(db, queue):
    _posted(queue)
    source = posted_draft_sources(queue)[0]
    pending_before = len(queue.list_by_status(PENDING_REVIEW,
                                              limit=50))

    def inventive_llm(prompt: str) -> str:
        return "Engagement grew 900% overnight, a record month."

    result = repurpose_source(db, queue, source, FORM_THREAD,
                              llm_fn=inventive_llm)
    assert not result["ok"]
    assert "900" in result["objection"]
    assert len(queue.list_by_status(PENDING_REVIEW, limit=50)) \
        == pending_before


def test_brain_rung_is_recorded(db, queue):
    _posted(queue)
    source = posted_draft_sources(queue)[0]
    result = repurpose_source(
        db, queue, source, FORM_SHORT_SCRIPT,
        llm_fn=lambda prompt: "Hook line from the brain.\n"
                              "Gold moved 12 dollars today.")
    assert result["ok"], result
    draft = queue.get(result["draft_id"])
    assert draft.metadata["drafter"] == "brain"
    assert "12" in draft.content


def test_wrong_form_for_kind_is_refused(db, queue):
    _posted(queue)
    source = posted_draft_sources(queue)[0]
    result = repurpose_source(db, queue, source, FORM_POST)
    assert not result["ok"] and "does not repurpose" in result["error"]


# ── (b) day record → short ───────────────────────────────────────────

def test_day_record_repurposes_to_short(db, queue):
    store = DayStore(db)
    store.add_fragment(
        text="Closed 4 trades before lunch, all gold, all calm.",
        day="2026-11-09")
    store.add_fragment(
        text="Wrote the week up in the evening.",
        day="2026-11-09")
    source = day_source(db, "2026-11-09")
    assert source is not None and source.kind == "day"
    result = repurpose_source(db, queue, source, FORM_SHORT_SCRIPT)
    assert result["ok"], result
    draft = queue.get(result["draft_id"])
    assert draft.status == PENDING_REVIEW
    assert draft.metadata["source_kind"] == "day"
    assert draft.metadata["source_ref"] == "2026-11-09"
    assert "4 trades" in draft.content


def test_empty_day_has_no_source(db):
    assert day_source(db, "2026-11-09") is None


def test_short_repurposes_to_text_post(db, queue):
    from nomorals.agents.repurpose import SourceMaterial

    source = SourceMaterial(
        kind="short", ref="post_abc",
        label="Why gold respects levels",
        text="Why gold respects levels\n"
             "Gold moved 12 dollars into the level and held.",
        basis="Why gold respects levels\n"
              "Gold moved 12 dollars into the level and held.",
        platforms=["x"])
    result = repurpose_source(db, queue, source, FORM_POST)
    assert result["ok"], result
    draft = queue.get(result["draft_id"])
    assert draft.status == PENDING_REVIEW
    assert draft.metadata["source_kind"] == "short"
    assert draft.metadata["form"] == "post"
    assert draft.platforms == ["x"]


# ── (c) the desk bundle landing ──────────────────────────────────────

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"goldenswing-desk-card" * 8
CAPTION = "🖥️ Desk 2026-11-09: +84.20 USD · 75% win rate · 4 closed"


def _bundle_bytes(members: list[tuple[str, bytes]], *, kind: str,
                  created_by: str = "goldenswing") -> bytes:
    """A contract-faithful bundle in the GS writer's shape:
    members, then bundle.manifest.json LAST (Devon's own
    bundle_manifest builds the contract dict)."""
    from nomorals.archives import bundle_manifest

    entries = [{"name": name, "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "is_dir": False}
               for name, data in members]
    manifest = bundle_manifest(entries, created_by=created_by,
                               kind=kind)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members:
            zf.writestr(name, data)
        zf.writestr("bundle.manifest.json", json.dumps(
            manifest, indent=2, ensure_ascii=False).encode("utf-8"))
    return buf.getvalue()


def _desk_bundle_bytes(day: str = "2026-11-09") -> bytes:
    family = {"bundle": "goldenswing desk card", "day": day,
              "caption": CAPTION,
              "stats": {"pnl": 84.2, "currency": "USD", "trades": 4,
                        "win_rate": 75.0, "equity": 1250.0,
                        "halted": False}}
    return _bundle_bytes(
        [("desk.png", PNG_BYTES),
         ("caption.txt", CAPTION.encode("utf-8")),
         ("manifest.json", json.dumps(
             family, indent=2, ensure_ascii=False).encode("utf-8"))],
        kind=GS_DESK_BUNDLE_KIND)


def test_desk_bundle_lands_one_draft_with_media(db, queue, store):
    result = land_desk_draft(db, queue, store, _desk_bundle_bytes(),
                             source="test")
    assert result["landed"], result
    draft = queue.get(result["draft_id"])
    assert draft.status == PENDING_REVIEW
    assert draft.content == CAPTION
    assert draft.metadata["origin"] == "gs_desk_bundle"
    assert draft.metadata["gs_day"] == "2026-11-09"
    media = draft.metadata["media"]
    assert len(media) == 1 and media[0]["mime"] == "image/png"
    art = store.get(media[0]["artifact"])
    assert art is not None
    # the GS-rendered PNG rides as-is — byte for byte
    assert store.blobs.get_bytes(art.content_hash) == PNG_BYTES

    again = land_desk_draft(db, queue, store, _desk_bundle_bytes(),
                            source="test")
    assert not again["landed"] and again["already"]
    assert again["draft_id"] == result["draft_id"]
    assert len(queue.list_by_status(PENDING_REVIEW, limit=50)) == 1


def test_tampered_desk_bundle_lands_none(db, queue, store):
    good = _desk_bundle_bytes()
    # the ORIGINAL contract manifest over a forged PNG member —
    # the sha check must refuse it before any member is trusted
    family = json.loads(zipfile.ZipFile(
        io.BytesIO(good)).read("manifest.json"))
    orig_manifest = zipfile.ZipFile(io.BytesIO(good)).read(
        "bundle.manifest.json")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("desk.png", PNG_BYTES + b"forged")
        zf.writestr("caption.txt", CAPTION.encode("utf-8"))
        zf.writestr("manifest.json", json.dumps(family).encode())
        zf.writestr("bundle.manifest.json", orig_manifest)
    result = land_desk_draft(db, queue, store, buf.getvalue(),
                             source="test")
    assert not result["landed"]
    assert "verification" in result["error"]
    assert "desk.png" in result["error"]
    assert queue.list_by_status(PENDING_REVIEW, limit=50) == []


def test_non_desk_bundle_lands_none(db, queue, store):
    other = _bundle_bytes([("journal.jsonl", b"{}\n")],
                          kind="journal-state")
    result = land_desk_draft(db, queue, store, other, source="test")
    assert result == {"landed": False, "desk": False}
    assert queue.list_by_status(PENDING_REVIEW, limit=50) == []
    assert inspect_desk_bundle(b"not a zip at all") == {"desk": False}


def test_gateway_inbound_seam_lands_the_draft(db, queue, store,
                                              tmp_path):
    from nomorals.social.chat.base import (
        ChatKind, ChatMessage, ChatRef, MediaRef)
    from nomorals.social.chat.gateway import ChatGateway
    from tests.test_gateway_inbound import _FakeAdapter

    bundle_file = tmp_path / "goldenswing-desk-2026-11-09.zip"
    bundle_file.write_bytes(_desk_bundle_bytes())

    def hook(art, data):
        return land_desk_draft(db, queue, store, data,
                               source="chat_inbound")

    gw = ChatGateway({"fake": _FakeAdapter()}, db=db,
                     artifact_store=store,
                     inbound_artifact_hook=hook)
    msg = ChatMessage(
        chat=ChatRef(platform="tg", chat_id="1", kind=ChatKind.DM),
        incoming=True, text="", sender="owner",
        media=[MediaRef(path=str(bundle_file), kind="file",
                        name=bundle_file.name)],
        reply_to="", mentioned=False, ts=time.time(),
        message_id="m-desk")
    gw._register_inbound_artifacts(msg)
    drafts = queue.list_by_status(PENDING_REVIEW, limit=50)
    assert len(drafts) == 1
    assert drafts[0].metadata["gs_day"] == "2026-11-09"
    assert msg.meta["artifact_ids"]  # registration itself intact


# ── (d) typography: the footer is a position, not a substring ────────

def test_quoted_stamp_text_still_gets_footer():
    text = f'he ended his post with "{DEFAULT_STAMP_TEXT}" again'
    out = apply_stamp(text)
    assert out == text + "\n\n" + STAMP_HTML
    assert out.count(DEFAULT_STAMP_TEXT) == 2


def test_stamp_text_as_trailing_sentence_still_gets_footer():
    text = f"this whole thing is {DEFAULT_STAMP_TEXT}"
    out = apply_stamp(text)
    assert out.endswith("\n\n" + STAMP_HTML)


def test_exact_footer_gets_no_second():
    body = "a presented card"
    for stamped in (
            body + "\n\n" + STAMP_HTML,
            body + "\n" + STAMP_HTML,
            body + "\n\n_" + DEFAULT_STAMP_TEXT + "_",
            body + "\n\n" + DEFAULT_STAMP_TEXT,
            body + "\n\n" + STAMP_HTML + "\n"):
        assert apply_stamp(stamped) == stamped
        assert apply_stamp(stamped, light=True) == stamped


def test_ordinary_messages_unchanged():
    assert apply_stamp("hello") == "hello\n\n" + STAMP_HTML
    assert apply_stamp("hello", light=True) == "hello\n" + STAMP_HTML
    assert apply_stamp("") == ""
    assert apply_stamp("   ") == "   "
    once = apply_stamp("body")
    assert apply_stamp(once) == once


def test_repurpose_surface_board_and_verbs(db, queue, tmp_path):
    """The /repurpose surface end-to-end (stub mixin, real queue):
    the board numbers its sources, the verbs resolve against the
    same lists, and a day verb drafts from today's record."""
    from types import SimpleNamespace

    from nomorals.agents.dayrecord import today_key
    from nomorals.agents.partner.runtime_system import RuntimeSystemMixin

    _posted(queue)
    store = DayStore(db)
    store.add_fragment(text="Closed 4 trades before lunch.",
                       day=today_key())

    class _Fake(RuntimeSystemMixin):
        pass

    fake = _Fake()
    fake.context = SimpleNamespace(db=db)
    fake._draft_queue = lambda: DraftQueue(
        db_path=str(tmp_path / "drafts.db"))

    board = fake._control_repurpose("")
    assert "posted drafts" in board and "Shipped the new desk" in board

    out = fake._control_repurpose("post 1 thread")
    assert "waiting on /drafts" in out
    out = fake._control_repurpose("day short")
    assert "waiting on /drafts" in out
    out = fake._control_repurpose("day 1999-01-01 short")
    assert "empty" in out
    forms = sorted(
        (d.metadata or {}).get("form")
        for d in queue.list_by_status(PENDING_REVIEW, limit=50)
        if (d.metadata or {}).get("origin") == "repurpose")
    assert forms == ["short_script", "thread"]


# ── (e) the queue read + the catalog pins ────────────────────────────

def test_list_by_status_reads_terminal_states(queue):
    draft = _posted(queue)
    assert [d.id for d in queue.list_by_status(POSTED)] == [draft.id]
    assert queue.list_by_status(PENDING_REVIEW) == []


def test_repurpose_command_is_catalogued_owner_only():
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
        PUBLIC_COMMANDS,
        detailed_help,
        help_text,
    )

    assert "repurpose" in CONTROL_COMMANDS
    assert "repurpose" in COMMAND_DETAILS
    grouped = [c for _, cs in LIST_GROUPS for c in cs]
    assert grouped.count("repurpose") == 1
    assert "repurpose" in LIST_ONELINERS
    assert "repurpose" not in PUBLIC_COMMANDS
    assert "/repurpose" in help_text()
    assert "/repurpose" in detailed_help("repurpose")
