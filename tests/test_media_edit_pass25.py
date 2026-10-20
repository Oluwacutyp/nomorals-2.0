"""Pass #25 — media edit engine + documents + datasci.

Each test pins one verified defect (all were red on pristine HEAD):

1.  documents ``Document.chunks()`` merged a sibling section's text into
    the previous chunk, mislabelling it with the wrong heading_path —
    the docstring promises section boundaries are never crossed.
2.  documents ``Table.to_records()`` / ``stats()`` silently dropped a
    column when two headers shared a name.
3.  documents ``to_markdown`` truncated table rows wider than the
    header row (cells vanished; ``to_pdf`` rides on markdown).
4.  edit_engine ``layer_to_ass`` wrote the ASS ``box`` flag into the
    Alignment field — BorderStyle stayed 1 and ``box_color`` was
    emitted as a comment that the code then stripped: boxes never
    rendered.
5.  edit_engine ``assemble_segments`` computed the flash/dip head fade
    and threw it away — only the outgoing tail fade ever rendered.
6.  edit_engine ``Document.to_dict()`` never wrote ``doc_hash`` —
    ``copy()`` / round-trips lost the sealed hash despite the
    "round-trip exactly" contract.
7.  edit_engine ``Document.restore_removed()`` popped the removed
    range *before* validating the video track — a failed restore
    destroyed the "recoverable, never destroyed" range.
8.  datasci missing-value labels: ``astype(str).fillna("∅")`` never
    fires (astype stringifies NaN first), so missing values showed as
    "nan"/"None" instead of the intended "∅" bucket.
9.  edit_engine ``KeyframeTrack.from_dict`` kept dict order — an
    unsorted payload made ``sample()`` return the wrong value.
10. quality_gate ``auto_fix`` claimed "captions shifted" but only moved
    ``layer.start`` — the word timings (what actually renders) and
    ``layer.end`` never moved.
"""

from __future__ import annotations

import importlib

import pytest

from nomorals.documents.convert import to_markdown
from nomorals.documents.model import Document as DocModel
from nomorals.documents.model import Section, Table
from nomorals.media.edit_engine.document import Document, DocTrack
from nomorals.media.edit_engine.keyframes import Keyframe, KeyframeTrack
from nomorals.media.edit_engine.text import TextLayer, ass_color, layer_to_ass
from nomorals.media.edit_engine.timeline import Clip
from nomorals.media_edit.videos import MediaEditError

render_mod = importlib.import_module("nomorals.media.edit_engine.render")


# ── 1. chunks never cross section boundaries ─────────────────────────

def test_chunks_do_not_merge_across_sections():
    doc = DocModel(title="T", sections=[
        Section(level=1, heading="Alpha", text="alpha body words", page=1),
        Section(level=1, heading="Beta", text="beta body words", page=1),
    ])
    chunks = doc.chunks(max_words=300)
    by_path = {c["heading_path"]: c["text"] for c in chunks}
    assert "Alpha" in by_path and "Beta" in by_path
    assert "beta" not in by_path["Alpha"]
    assert "alpha" not in by_path["Beta"]


def test_chunks_still_merge_paragraphs_within_one_section():
    doc = DocModel(title="T", sections=[
        Section(level=1, heading="Alpha",
                text="first para words\n\nsecond para words", page=1),
    ])
    chunks = doc.chunks(max_words=300)
    assert len(chunks) == 1
    assert "first para" in chunks[0]["text"]
    assert "second para" in chunks[0]["text"]


# ── 2. duplicate table headers lose no data ─────────────────────────

def test_to_records_keeps_duplicate_header_columns():
    table = Table(headers=["a", "a"], rows=[["1", "2"], ["3", "4"]])
    records = table.to_records()
    assert records[0] == {"a": "1", "a.1": "2"}
    assert records[1] == {"a": "3", "a.1": "4"}


def test_table_stats_keys_unique_with_duplicate_headers():
    table = Table(headers=["a", "a"], rows=[["1", "2"]])
    stats = table.stats()
    assert len(stats) == 2
    assert set(stats) == {"a", "a.1"}


# ── 3. markdown tables keep wide rows ─────────────────────────────────

def test_markdown_table_keeps_cells_beyond_header_width():
    doc = DocModel(title="T", tables=[
        Table(headers=["x", "y"], rows=[["1", "2", "3EXTRA"]])])
    lines = [ln for ln in to_markdown(doc).splitlines()
             if ln.startswith("|")]
    assert any("3EXTRA" in ln for ln in lines)


# ── 4. ASS box lands in BorderStyle, box_color in BackColour ─────────

def _style_fields(ass: str) -> list[str]:
    line = next(ln for ln in ass.splitlines() if ln.startswith("Style:"))
    return [f.strip() for f in line[len("Style:"):].split(",")]


def test_layer_box_uses_borderstyle_3_and_box_color():
    fields = _style_fields(layer_to_ass(
        TextLayer(text="hi", box=True, box_color="#FF0000")))
    # ASS field order after "Style:": Name(0), Fontname(1), Fontsize(2),
    # Primary(3), Secondary(4), Outline(5), Back(6), Bold(7), Italic(8),
    # Underline(9), StrikeOut(10), ScaleX(11), ScaleY(12), Spacing(13),
    # Angle(14), BorderStyle(15), Outline(16), Shadow(17),
    # Alignment(18), MarginL(19), MarginR(20), MarginV(21), Encoding(22).
    assert fields[15] == "3"                      # BorderStyle
    assert fields[6] == ass_color("#FF0000")      # BackColour
    assert fields[18] == "2"                      # Alignment = bottom


def test_layer_no_box_keeps_borderstyle_1():
    fields = _style_fields(layer_to_ass(TextLayer(text="hi")))
    assert fields[15] == "1"
    assert fields[18] == "2"


# ── 5. flash/dip transitions fade BOTH sides of the boundary ─────────

def _flash_graph(monkeypatch, kind):
    graphs = []

    def fake_run(cmd, **kw):
        if "-filter_complex" in cmd:
            graphs.append(cmd[cmd.index("-filter_complex") + 1])
        with open(cmd[-1], "wb") as fh:
            fh.write(b"x")
        return {"seconds": 0.1}

    monkeypatch.setattr(render_mod, "run_ffmpeg", fake_run)
    monkeypatch.setattr(
        render_mod, "video_probe",
        lambda p: {"duration": 4.0, "width": 320, "height": 240,
                   "streams": [{"type": "video"}]})
    clips = [Clip(path="/tmp/pass25-a.mp4", start=0, end=4),
             Clip(path="/tmp/pass25-b.mp4", start=0, end=4)]
    for c in clips:  # _clip_window requires the files to exist
        with open(c.path, "wb") as fh:
            fh.write(b"x")
    render_mod.assemble_segments(
        clips, segments=[(0, 2.0), (1, 2.0)], transitions=kind,
        out="/tmp/pass25-out.mp4")
    return graphs[0]


def test_flash_transition_applies_head_fade_to_next_segment(monkeypatch):
    graph = _flash_graph(monkeypatch, "flash")
    assert "fade=t=out" in graph          # tail of segment 0
    assert "fade=t=in" in graph           # head of segment 1
    head = [p for p in graph.split(";") if "fade=t=in" in p]
    assert head and head[0].startswith("[sg1")


def test_dip_transition_applies_head_fade_to_next_segment(monkeypatch):
    graph = _flash_graph(monkeypatch, "dip")
    assert "fade=t=in" in graph


# ── 6. Document round-trips its doc_hash ──────────────────────────────

def test_document_round_trip_preserves_doc_hash():
    doc = Document(
        tracks=[DocTrack(kind="video",
                         clips=[Clip(path="a.mp4", start=0, end=5)])],
        title="x")
    doc.seal()
    assert doc.doc_hash
    assert doc.to_dict()["doc_hash"] == doc.doc_hash
    back = Document.from_dict(doc.to_dict())
    assert back.doc_hash == doc.doc_hash
    assert doc.copy().doc_hash == doc.doc_hash
    # the stored hash must not feed back into the hash computation
    assert back.compute_hash() == doc.doc_hash


# ── 7. failed restore keeps the removed range ─────────────────────────

def test_restore_removed_failure_keeps_entry():
    doc = Document(tracks=[], removed=[
        {"start": 1.0, "end": 2.0, "reason": "r", "words": []}])
    with pytest.raises(MediaEditError):
        doc.restore_removed()
    assert len(doc.removed) == 1


# ── 8. datasci missing values share the ∅ bucket ─────────────────────

def test_value_counts_labels_missing_with_empty_marker():
    pd = pytest.importorskip("pandas")
    from nomorals.datasci import DataWorkspace
    ws = DataWorkspace()
    ws.load_frame("d", pd.DataFrame(
        {"c": ["a", None, "a", float("nan")]}))
    vc = {row["value"]: row["count"] for row in ws.value_counts("d", "c")}
    assert vc == {"a": 2, "∅": 2}


def test_pie_labels_missing_with_empty_marker():
    pytest.importorskip("matplotlib")
    pd = pytest.importorskip("pandas")
    from nomorals.datasci.plots import _render_pie
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame({"c": ["a", None, "b", None]})
    fig, ax = plt.subplots()
    try:
        _render_pie(ax, frame, "c", 20)
        labels = [t.get_text() for t in ax.texts]
    finally:
        plt.close(fig)
    assert "∅" in labels
    assert "None" not in labels and "nan" not in labels


# ── 9. keyframe tracks sample correctly from unsorted payloads ──────

def test_keyframe_track_from_dict_sorts_keys():
    track = KeyframeTrack.from_dict({
        "clip_id": "c", "effect": "zoom", "param": "z",
        "keys": [{"time": 2.0, "value": 10.0},
                 {"time": 0.0, "value": 0.0},
                 {"time": 1.0, "value": 5.0}]})
    assert track.sample(0.0) == 0.0
    assert track.sample(2.0) == 10.0
    assert 0.0 < track.sample(0.5) < 5.0
    assert 5.0 < track.sample(1.5) < 10.0


def test_keyframe_track_direct_construction_sorts_keys():
    track = KeyframeTrack(clip_id="c", effect="zoom", param="z", keys=[
        Keyframe(2.0, 10.0), Keyframe(0.0, 0.0), Keyframe(1.0, 5.0)])
    assert track.sample(0.0) == 0.0
    assert track.sample(2.0) == 10.0


# ── 10. auto_fix caption shift moves words AND end ────────────────────

def test_auto_fix_caption_shift_moves_word_timings():
    from nomorals.media.edit_engine.quality_gate import QualityGate, auto_fix
    doc = Document(
        tracks=[DocTrack(kind="video",
                         clips=[Clip(path="a.mp4", start=0, end=10)])],
        text_layers=[TextLayer(
            words=[{"text": "hi", "start": 1.0, "end": 1.5}],
            start=1.0, end=2.0)])
    report = QualityGate().check(doc, {"caption_drift_s": 0.4})
    out = auto_fix(doc, report, {"caption_drift_s": 0.4})
    layer = out["document"].text_layers[0]
    assert layer.start == pytest.approx(0.6)
    assert layer.end == pytest.approx(1.6)
    assert layer.words[0]["start"] == pytest.approx(0.6)
    assert layer.words[0]["end"] == pytest.approx(1.1)
