"""Tail-108 pins: nomorals/memory/base.py census sites.

The module's printed laws: ``infer_trust`` "Never raises" and its
no-evidence default is TRUSTED ("anything else (including legacy
records with an empty source) defaults to trusted so existing recall
behaviour is unchanged"); formatting "never breaks the surface".
Every site is held as an exclusion:

* 112 — ``infer_trust`` crash -> TRUSTED. The crash return is the
  function's own documented default, reached when the marker scan
  itself cannot run (non-string input). FLAGGED for the owner as a
  posture-question candidate: the result is persisted at write time
  (memory/manager.py) and drives ``requires_confirmation`` /
  cross-session downranking, so a crash fails the provenance gate
  OPEN — but the fail-open value is the printed contract, and no
  within-contract fix exists (returning UNTRUSTED on crash would
  contradict the documented default and reclassify every legacy
  record whose fields are merely odd).
* 388 — ``format_record`` outer guard -> the record's OWN content
  (or id); presentational, never invents text.
* 391 — inner fallback guard -> "": the record is omitted from the
  render, never misdescribed.
* 409 — ``format_recall`` guard -> "", the function's own
  documented value for an empty set.

The module has no logger and no failure ledger, so non-recording at
388/391/409 is structural; at 112 it is the adjudicated gap above.
"""

from nomorals.memory import base as mb


# ── 112: infer_trust ─────────────────────────────────────────────

def test_infer_trust_explicit_wins_both_ways():
    assert mb.infer_trust("user", "", explicit="untrusted") == mb.UNTRUSTED
    assert mb.infer_trust("tool:x", "", explicit="trusted") == mb.TRUSTED


def test_infer_trust_marker_source_and_origin_are_untrusted():
    assert mb.infer_trust("browser scrape", "") == mb.UNTRUSTED
    assert mb.infer_trust("", "web page") == mb.UNTRUSTED


def test_infer_trust_no_evidence_defaults_trusted():
    assert mb.infer_trust("owner", "") == mb.TRUSTED
    assert mb.infer_trust() == mb.TRUSTED


def test_infer_trust_crash_returns_documented_default():
    # A non-string explicit value raises inside the guard
    # (``(123).strip``); the pin documents the fail-open shape:
    # the marker scan never ran, yet the verdict is TRUSTED.
    assert mb.infer_trust("browser", "", explicit=123) == mb.TRUSTED
    assert mb.infer_trust(object(), "") == mb.TRUSTED


# ── 388 / 391: format_record ─────────────────────────────────────

def _record():
    return mb.MemoryRecord(id="m1", kind="fact", content="hello world")


def test_format_record_healthy_paths_render_content():
    assert "hello world" in mb.format_record(_record(), "detailed")
    assert "hello world" in mb.format_record(_record(), "compact")


class _ExplodingKind:
    id = "mx"
    content = "rescued content"
    score = 0.0
    trust = ""
    access_count = 0
    tags = ""
    requires_confirmation = False
    is_trusted = True

    @property
    def kind(self):
        raise RuntimeError("kind boom")


def test_format_record_crash_falls_back_to_own_content():
    assert mb.format_record(_ExplodingKind(), "detailed") == "rescued content"


class _AllExploding:
    @property
    def content(self):
        raise RuntimeError("content boom")

    @property
    def id(self):
        raise RuntimeError("id boom")


def test_format_record_total_crash_renders_empty():
    assert mb.format_record(_AllExploding(), "detailed") == ""


# ── 409: format_recall ───────────────────────────────────────────

def test_format_recall_healthy_and_empty():
    assert "hello world" in mb.format_recall([_record()])
    assert mb.format_recall([]) == ""


class _BadIter:
    def __iter__(self):
        raise RuntimeError("iter boom")


def test_format_recall_crash_returns_empty_string():
    assert mb.format_recall(_BadIter()) == ""
