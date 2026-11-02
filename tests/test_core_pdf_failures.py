"""Tail-102 pins: nomorals/core/pdf.py census sites — all inside the
best-effort PDF text-extraction parser, all held as exclusions.

The parser's contract is extraction under malformed input: a bad
operator or CMap entry costs its own glyph/position and nothing
else. Every site degrades to ABSENCE (an unmapped glyph decodes to
"", a skipped move keeps the last position, an ignored kerning
token inserts no space) — never to invented content — and the
extraction of everything around the defect proceeds. Consumers
(documents reading via runtime_media / search engine / BookForge)
receive partial text, which is the honest outcome for a malformed
file. All four triggers were probed live before pinning.

843 — ``_parse_cmap`` bfrange array: a malformed hex entry
(odd-length, ``bytes.fromhex`` raises) -> that gid stays unmapped;
neighbouring entries still land. The site carries its own in-code
rationale (``malformed cmap entry skipped``).
911 — ``_draws_in_stream`` Tm operands: non-numeric operands
(``float("-")`` raises) -> keep last position; the draw is still
emitted at the kept position.
918 — ``_draws_in_stream`` Td/TD operands: same shape as 911.
937 — ``_draws_in_stream`` TJ kerning token: a malformed kerning
number is ignored -> no word space inserted; a valid large
negative kern still inserts the space (control below).

NOT-recording is pinned with the snapshot method: the module
calls no failures-ledger record function anywhere (and has no
logger), so both general ledgers must be unchanged across each
trigger.
"""
from __future__ import annotations

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.core import pdf


def _snapshot():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def test_843_malformed_cmap_entry_skipped_neighbours_land():
    before = _snapshot()
    cmap, _width = pdf._parse_cmap(
        b"beginbfrange\n<0000> <0001> [<0041> <041>] endbfrange")
    assert cmap == {0: "A"}  # gid 1 (odd-length hex) unmapped, not invented
    assert _snapshot() == before


def test_911_malformed_tm_operands_keep_last_position():
    before = _snapshot()
    assert pdf._draws_in_stream(b"- - - - - - Tm (Hi) Tj") == [(0.0, 0.0, "Hi")]
    # control: well-formed Tm moves the origin
    assert pdf._draws_in_stream(b"1 0 0 1 30 40 Tm (Hi) Tj") == [(30.0, 40.0, "Hi")]
    assert _snapshot() == before


def test_918_malformed_td_operands_keep_last_position():
    before = _snapshot()
    assert pdf._draws_in_stream(b"- - Td (Hi) Tj") == [(0.0, 0.0, "Hi")]
    # control: well-formed Td moves the line start
    assert pdf._draws_in_stream(b"10 20 Td (Hi) Tj") == [(10.0, 20.0, "Hi")]
    assert _snapshot() == before


def test_937_malformed_kerning_number_ignored():
    before = _snapshot()
    assert pdf._draws_in_stream(b"[(A) - (B)] TJ") == [(0.0, 0.0, "AB")]
    # control: a valid big negative kern still splits the word
    assert pdf._draws_in_stream(b"[(A) -100 (B)] TJ") == [(0.0, 0.0, "A B")]
    assert _snapshot() == before
