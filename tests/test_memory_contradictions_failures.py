"""Pass tail-117 pins: nomorals/memory/contradictions.py — 3 silent handlers.

- 301 E104 ``_eligible``: metadata/kind access crash -> False; the
  record is excluded from the scan (fail-closed), identical to the
  documented superseded/episode exclusions.
- 308 E104 ``_is_private``: metadata access crash -> False, i.e. the
  record is treated as NOT private and participates in detection.
  POSTURE-QUESTION CANDIDATE (parent/owner list): an unfenced
  fail-open on a privacy gate, sibling of persona-60 — no downstream
  re-check exists in detect_against/resolve, records can be
  duck-typed adapters (books/library), and hit notes are surfaced by
  agents/proactive. Reachable only when a record's metadata mapping
  itself raises; honest metadata gates correctly (pinned below).
- 371 E104 per-pair guard in ``detect_against``: one bad pair is
  skipped, the scan continues — the printed contract ("one bad pair
  never kills the scan").

None of the three guards records anything; the module's debug log
fires only on the outer detect/detect_for guards, not these.
"""

from types import SimpleNamespace

from nomorals.memory import contradictions as cx
from nomorals.memory.base import MemoryKind


class BadMeta(dict):
    """A metadata mapping whose .get raises (broken adapter record)."""

    def get(self, *args, **kwargs):
        raise RuntimeError("meta boom")


def rec(rid, kind, content, metadata=None):
    return SimpleNamespace(id=rid, kind=kind, content=content,
                           metadata={} if metadata is None else metadata)


OLD = rec("o1", MemoryKind.PREFERENCE, "I prefer dark coffee in the morning")
NEW = rec("n1", MemoryKind.PREFERENCE, "I prefer light coffee in the morning")


# ── 301: _eligible ───────────────────────────────────────────────────

def test_eligible_fact_record():
    assert cx._eligible(rec("1", MemoryKind.FACT, "x")) is True


def test_eligible_episode_excluded():
    assert cx._eligible(rec("1", MemoryKind.EPISODE, "x")) is False


def test_eligible_superseded_excluded():
    assert cx._eligible(
        rec("1", MemoryKind.FACT, "x", {"superseded_by": "9"})) is False


def test_eligible_metadata_crash_excludes():
    assert cx._eligible(
        rec("1", MemoryKind.FACT, "x", BadMeta(private=True))) is False


# ── 308: _is_private ─────────────────────────────────────────────────

def test_is_private_true_when_marked():
    assert cx._is_private(
        rec("1", MemoryKind.FACT, "x", {"private": True})) is True


def test_is_private_false_when_unmarked():
    assert cx._is_private(rec("1", MemoryKind.FACT, "x")) is False


def test_is_private_metadata_crash_fails_open():
    # the pinned posture shape: a raising metadata mapping reads as
    # "not private" — the gate's honest-metadata behaviour is pinned
    # by the two tests above; this is the crash-path remainder.
    assert cx._is_private(
        rec("1", MemoryKind.FACT, "x", BadMeta(private=True))) is False


def test_detect_gates_private_record_with_honest_metadata():
    priv_new = rec("n2", MemoryKind.PREFERENCE,
                   "I prefer light coffee in the morning",
                   {"private": True})
    assert cx.detect_against(priv_new, [OLD]) == []


# ── 371: per-pair guard ──────────────────────────────────────────────

def test_detect_healthy_pair_fires():
    hits = cx.detect_against(NEW, [OLD])
    assert len(hits) == 1 and hits[0].confidence >= 0.5


class BadContent:
    id = "bad1"
    kind = MemoryKind.FACT
    metadata = {}

    @property
    def content(self):
        raise RuntimeError("content boom")


def test_detect_one_bad_pair_never_kills_the_scan():
    hits = cx.detect_against(NEW, [BadContent(), OLD])
    assert len(hits) == 1 and hits[0].older_id == "o1"
