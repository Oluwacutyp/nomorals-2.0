"""Doc 24 SR4 — samples/lineage + originality bands (classes 4/5/7).

Pins the SR4 doctrine end to end over synthesized fixtures:

* a host track with an excerpt of an enrolled work spliced in
  (mid-track, gain-shifted) reports that work in its lineage
  with the region (both time ranges) as the evidence;
* a two-source host reports both sources with disjoint regions;
* a clean self-composed host reports NO lineage in catalog-scope
  wording — "no enrolled source found", never an absolute
  originality claim;
* a below-floor fragment (1.2 s) is NOT claimed as a sample: it
  is named as sub-floor evidence and its span stays unaccounted;
* originality bands derive from measured coverage at scan time:
  a band label planted into a receipt dict is never read back —
  re-derivation from the receipt's own seconds disagrees with it;
* a query that IS an enrolled recording (or a frame-exact
  re-performance the class-1 engine identifies as it) is
  excluded as its own source;
* lineage also runs between enrolled works, from a track's
  stored fingerprint.

Fixture scope: every track here is synthesized with the SR2
bench generator. These tests pin behaviour and fixture-scope
detection (on the calibration pool: 7/7 true regions found
within ±1.5 s, 0 false regions); real-library sample detection
accuracy stays unproven until a ground-truth set over real
recordings runs. All stores live under per-test tmp paths.
"""

from __future__ import annotations

from array import array

import pytest

from nomorals.audio.fingerprint import FingerprintDB
from nomorals.audio.lineage import (
    BAND_MAJORITY_ACCOUNTED,
    BAND_MINORITY_ACCOUNTED,
    BAND_NONE_ACCOUNTED,
    LINEAGE_SCHEMA,
    MATCH_CLASS_SAMPLE_LINEAGE,
    band_from_receipt,
    originality_band,
    render_lineage_lines,
    scan_lineage_db,
    scan_lineage_hashes,
)
from nomorals.audio.songrecon_bench import (
    apply_gain,
    excerpt,
    synth_sibling,
    synth_track,
    write_wav,
)

SR = 11025

# 12 s sources: at fixture landmark density the shipped
# 4,000-hash enrollment cap stores ~12.6 s, so a 12 s source is
# enrolled whole (its stored fingerprint covers the track).
SOURCE_A = synth_track(101, 12.0)
SOURCE_B = synth_track(202, 12.0)
SOURCE_C = synth_track(303, 12.0)


def splice(base, seg, at_s, gain):
    """Replace ``base`` samples at ``at_s`` with ``seg`` x gain."""
    host = list(base)
    seg = apply_gain(seg, gain)
    a = int(at_s * SR)
    for i, v in enumerate(seg):
        if a + i < len(host):
            host[a + i] = v
    return array("d", host)


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    """One shared read-only catalog (scans never mutate it)."""
    work = tmp_path_factory.mktemp("sr4-catalog")
    db = FingerprintDB(str(work / "catalog.db"))
    for name, samples, title in (("a.wav", SOURCE_A, "Source A"),
                                 ("b.wav", SOURCE_B, "Source B"),
                                 ("c.wav", SOURCE_C, "Source C")):
        path = write_wav(work / name, samples)
        res = db.add_track(path, title=title, artist="Fixture")
        assert res["ok"], res
    yield db
    db.close()


@pytest.fixture()
def wav(tmp_path):
    def _write(name, samples):
        return write_wav(tmp_path / name, samples)
    return _write


def _by_title(result, title):
    for src in result["sources"]:
        if src["title"] == title:
            return src
    return None


# ---------------------------------------------------------------------------
# the claims
# ---------------------------------------------------------------------------

def test_embedded_excerpt_claimed_with_region(catalog, wav):
    host = wav("h1.wav", splice(synth_track(900, 30.0),
                                excerpt(SOURCE_A, 3.0, 8.0), 10.0, 0.8))
    res = scan_lineage_db(host, db=catalog)
    assert res["ok"] and res["status"] == "lineage_found", res
    src = _by_title(res, "Source A")
    assert src is not None
    assert _by_title(res, "Source B") is None
    assert _by_title(res, "Source C") is None
    (region,) = src["regions"]
    # anchor spans sit inside the true segment's edges
    assert abs(region["query_start_s"] - 10.0) <= 1.5
    assert abs(region["query_end_s"] - 18.0) <= 1.5
    assert abs(region["reference_start_s"] - 3.0) <= 1.5
    assert abs(region["reference_end_s"] - 11.0) <= 1.5
    strength = region["match_strength"]
    assert strength["name"] == "regional_coherent_fraction"
    assert strength["numerator"] == region["coherent_votes"]
    assert "not a probability" in strength["calibration"]
    # the band reflects the covered share (8 s of 30 s)
    assert res["originality_band"] == BAND_MINORITY_ACCOUNTED
    assert 0.15 < res["accounted_share"] < 0.35
    assert res["unaccounted_s"] == pytest.approx(
        30.0 - res["accounted_s"], abs=0.01)
    receipt = res["receipt"]
    assert receipt["schema"] == LINEAGE_SCHEMA
    assert receipt["match_class"] == MATCH_CLASS_SAMPLE_LINEAGE
    assert receipt["sources"][0]["regions"], receipt
    assert band_from_receipt(receipt) == receipt["originality_band"]


def test_two_sources_disjoint_regions(catalog, wav):
    host = wav("h2.wav", splice(
        splice(synth_track(901, 30.0), excerpt(SOURCE_A, 1.0, 7.0),
               3.0, 0.6),
        excerpt(SOURCE_B, 4.0, 7.0), 18.0, 0.9))
    res = scan_lineage_db(host, db=catalog)
    assert res["status"] == "lineage_found", res
    a = _by_title(res, "Source A")
    b = _by_title(res, "Source B")
    assert a is not None and b is not None
    (ra,) = a["regions"]
    (rb,) = b["regions"]
    assert ra["query_end_s"] <= rb["query_start_s"]  # disjoint
    assert abs(ra["query_start_s"] - 3.0) <= 1.5
    assert abs(rb["query_start_s"] - 18.0) <= 1.5
    assert res["accounted_share"] == pytest.approx(14.0 / 30.0,
                                                   abs=0.09)


def test_majority_band_when_most_of_work_is_accounted(catalog, wav):
    host = wav("hm.wav", splice(synth_track(906, 16.0),
                                excerpt(SOURCE_A, 0.5, 11.0), 2.0, 0.9))
    res = scan_lineage_db(host, db=catalog)
    assert _by_title(res, "Source A") is not None
    assert res["accounted_share"] > 0.5
    assert res["originality_band"] == BAND_MAJORITY_ACCOUNTED


# ---------------------------------------------------------------------------
# the honesty pins
# ---------------------------------------------------------------------------

def test_clean_work_reports_no_enrolled_source_not_originality(
        catalog, wav):
    host = wav("clean.wav", synth_track(999, 30.0))
    res = scan_lineage_db(host, db=catalog)
    assert res["ok"] and res["status"] == "no_lineage_found", res
    assert res["sources"] == []
    assert res["originality_band"] == BAND_NONE_ACCOUNTED
    assert res["accounted_s"] == 0.0
    text = "\n".join(render_lineage_lines(res))
    assert "no enrolled source found" in text
    # catalog scope rides on the answer; absence of lineage is
    # never worded as a finding of originality
    assert "not a finding that it is original" in text
    assert "never a copyright verdict" in text
    assert "is original" not in text.replace(
        "not a finding that it is original", "")


def test_below_floor_fragment_is_evidence_not_a_claim(catalog, wav):
    host = wav("frag.wav", splice(synth_track(905, 30.0),
                                  excerpt(SOURCE_A, 4.0, 1.2), 8.0, 0.9))
    res = scan_lineage_db(host, db=catalog)
    assert res["status"] == "no_lineage_found", res
    assert _by_title(res, "Source A") is None
    evidence = [e for e in res["sub_floor_evidence"]
                if e["title"] == "Source A"]
    assert evidence, res["sub_floor_evidence"]
    (ev,) = evidence
    assert abs(ev["query_start_s"] - 8.0) <= 1.5
    assert any("span" in f and "sample floor" in f
               for f in ev["failed_floors"])
    # the fragment's time is unaccounted, not silently absorbed
    assert res["accounted_s"] == 0.0
    text = "\n".join(render_lineage_lines(res))
    assert "below the sample floor (not claimed as a sample)" in text


def test_planted_band_label_is_never_read(catalog, wav):
    host = wav("h1.wav", splice(synth_track(900, 30.0),
                                excerpt(SOURCE_A, 3.0, 8.0), 10.0, 0.8))
    res = scan_lineage_db(host, db=catalog)
    receipt = dict(res["receipt"])
    assert receipt["originality_band"] == BAND_MINORITY_ACCOUNTED
    receipt["originality_band"] = "near_fully_accounted"  # planted
    assert band_from_receipt(receipt) == BAND_MINORITY_ACCOUNTED


def test_band_boundaries_are_pure_coverage():
    assert originality_band(0.0) == BAND_NONE_ACCOUNTED
    assert originality_band(0.25) == BAND_MINORITY_ACCOUNTED
    assert originality_band(0.5) == BAND_MAJORITY_ACCOUNTED
    assert originality_band(0.9) == "near_fully_accounted"
    assert originality_band(1.0) == "near_fully_accounted"


def test_enrolled_recording_is_its_own_identity_not_a_source(
        catalog, wav):
    own = wav("own.wav", SOURCE_A)
    res = scan_lineage_db(own, db=catalog)
    assert res["query_identity"].get("title") == "Source A", res
    assert _by_title(res, "Source A") is None


def test_reperformance_defers_to_class1_identity(catalog, wav):
    # A frame-exact re-performance of the whole work: the class-1
    # engine identifies it as the recording, so lineage names it
    # as the query identity instead of claiming it as a sample.
    sib = wav("sib.wav", synth_sibling(101, 12.0))
    res = scan_lineage_db(sib, db=catalog)
    assert res["query_identity"].get("title") == "Source A", res
    assert _by_title(res, "Source A") is None


def test_lineage_between_enrolled_works(tmp_path):
    db = FingerprintDB(str(tmp_path / "cat.db"))
    try:
        a_path = write_wav(tmp_path / "a.wav", SOURCE_A)
        assert db.add_track(a_path, title="Source A",
                            artist="Fixture")["ok"]
        # The host's stored fingerprint is the shipped enrollment
        # window (its first ~12 s), so the excerpt sits early.
        host_path = write_wav(tmp_path / "host.wav", splice(
            synth_track(900, 30.0), excerpt(SOURCE_A, 3.0, 8.0),
            2.0, 0.8))
        added = db.add_track(host_path, title="Host",
                             artist="Fixture")
        assert added["ok"], added
        host_id = added["track_id"]
        hashes = db.track_hashes(host_id)
        assert hashes
        res = scan_lineage_hashes(hashes, duration_s=30.0, db=db,
                                  exclude_track_id=host_id)
        assert res["status"] == "lineage_found", res
        assert _by_title(res, "Source A") is not None
        assert _by_title(res, "Host") is None
    finally:
        db.close()


def test_render_wording_carries_evidence_and_scope(catalog, wav):
    host = wav("h1.wav", splice(synth_track(900, 30.0),
                                excerpt(SOURCE_A, 3.0, 8.0), 10.0, 0.8))
    res = scan_lineage_db(host, db=catalog)
    text = "\n".join(render_lineage_lines(res))
    assert "sampled in 1 region(s)" in text
    assert "regional coherent fraction" in text
    assert "not a probability" in text
    assert "originality band: minority accounted" in text
    assert "unaccounted:" in text
    assert "confidence" not in text  # SR1 vocabulary discipline


def test_scan_of_missing_file_is_unavailable(catalog, tmp_path):
    res = scan_lineage_db(tmp_path / "nope.wav", db=catalog)
    assert res["ok"] is False
    assert res["status"] == "unavailable"
    assert res["receipt"]["match_class"] == MATCH_CLASS_SAMPLE_LINEAGE


def test_cli_lineage_action_parses_and_dispatches(catalog, wav,
                                                  monkeypatch, capsys):
    import argparse

    from nomorals.cmdline.commands.audio import _cmd_audio
    from nomorals.cmdline.parser import _parser

    args = _parser().parse_args(
        ["audio", "lineage", "x.wav", "--json"])
    assert args.command == "audio"
    assert args.audio_action == "lineage"

    host = wav("h1.wav", splice(synth_track(900, 30.0),
                                excerpt(SOURCE_A, 3.0, 8.0), 10.0, 0.8))
    monkeypatch.setattr(
        "nomorals.audio.lineage.default_fingerprint_db",
        lambda db_path="": catalog)
    rc = _cmd_audio(argparse.Namespace(
        audio_action="lineage", file=host, json=True), None)
    assert rc == 0
    import json
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "lineage_found"
    assert out["receipt"]["match_class"] == MATCH_CLASS_SAMPLE_LINEAGE
