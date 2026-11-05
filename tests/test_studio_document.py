"""Echelon Wave 1 · D1 acceptance: EditStudio is a façade over Document.

D1 (INTEGRATION_PLAN §2.0): ``edit_engine.Document`` is THE canonical
edit document; ``media_edit.EditStudio`` is a session façade over it.
These tests pin the agreement:

* an edit made through the façade is a Document edit — the session's
  Document changes with every push/undo/redo;
* the compilation is total — segment ops become clips, every other op
  becomes an Effect descriptor with its params verbatim, in order;
* sessions rebuild from Documents (``from_document``) and round-trip;
* structure a session cannot express raises — never a silent drop.
"""

import pytest

from nomorals.media.edit_engine.document import Document, image_document
from nomorals.media.edit_engine.timeline import Clip, Effect
from nomorals.media_edit import MediaEditError
from nomorals.media_edit.studio import EditStudio


# ── image sessions ────────────────────────────────────────────────────

def test_image_session_ops_are_document_effects():
    st = EditStudio("in.png", kind="image", name="poster")
    st.filter("vivid", strength=0.5).grade(exposure=0.3).text("HELLO")
    doc = st.to_document()
    assert isinstance(doc, Document)
    assert doc.title == "poster"
    track = doc.track("image")
    assert track is not None and len(track.clips) == 1
    clip = track.clips[0]
    assert clip.path == "in.png"
    assert [(e.name, e.params) for e in clip.effects] == [
        ("filter", {"preset": "vivid", "strength": 0.5}),
        ("grade", {"exposure": 0.3}),
        ("text_layer", {"text": "HELLO"}),
    ]
    # the document is sealed: hash + version track the content
    assert doc.doc_hash == doc.compute_hash()
    assert doc.version >= 2


def test_document_property_tracks_every_façade_edit():
    st = EditStudio("in.png", kind="image", name="x")
    d0 = st.document
    assert d0.track("image").clips[0].effects == ()
    st.filter("noir")
    d1 = st.document
    assert [e.name for e in d1.track("image").clips[0].effects] == ["filter"]
    assert d1.doc_hash != d0.doc_hash
    st.undo()
    d2 = st.document
    assert d2.track("image").clips[0].effects == ()
    assert d2.doc_hash == d0.doc_hash
    st.redo()
    assert st.document.doc_hash == d1.doc_hash
    st.clear()
    assert st.document.doc_hash == d0.doc_hash


def test_image_round_trip_session_document_session():
    st = EditStudio("in.png", kind="image", name="rt")
    st.filter("vivid", strength=0.25).smart_crop("9:16")
    doc = st.to_document()
    st2 = EditStudio.from_document(doc)
    assert st2.source == "in.png"
    assert st2.name == "rt"
    assert [(o["op"], o["params"]) for o in st2.ops] == [
        ("filter", {"preset": "vivid", "strength": 0.25}),
        ("smart_crop", {"aspect": "9:16", "mode": "saliency"}),
    ]
    # and the rebuilt session compiles to the same document content
    assert st2.to_document().to_dict()["tracks"] == \
        doc.to_dict()["tracks"]


# ── video sessions ────────────────────────────────────────────────────

def test_video_session_segments_and_session_ops():
    st = EditStudio("a.mp4", kind="video", name="cut1")
    st.add_clip("b.mp4", start=2.0, end=9.0, speed=1.5)
    st.add_image("logo.png", duration=2.5)
    st.transition("fade", 0.4).speed(2.0).chapter("intro", at=0.0)
    doc = st.to_document()
    video = doc.track("video")
    assert [c.path for c in video.clips] == ["a.mp4", "b.mp4"]
    b = video.clips[1]
    assert (b.start, b.end, b.speed) == (2.0, 9.0, 1.5)
    # session-level ops ride the first clip as Effect descriptors,
    # in session order, params verbatim
    assert [(e.name, e.params) for e in video.clips[0].effects] == [
        ("v_transition", {"type": "fade", "duration": 0.4}),
        ("v_speed", {"factor": 2.0, "start": None, "end": None}),
        ("v_chapter", {"name": "intro", "at": 0.0, "end": None}),
    ]
    image = doc.track("image")
    assert [(c.path, c.duration) for c in image.clips] == [("logo.png", 2.5)]


def test_video_round_trip():
    st = EditStudio("a.mp4", kind="video", name="vrt")
    st.add_clip("b.mp4", start=1.0, end=4.0)
    st.transition("fade", 0.5)
    doc = st.to_document()
    st2 = EditStudio.from_document(doc)
    assert st2.source == "a.mp4"
    assert [(o["op"], o["params"]) for o in st2.ops] == [
        ("v_transition", {"type": "fade", "duration": 0.5}),
        ("v_clip", {"path": "b.mp4", "start": 1.0, "end": 4.0,
                    "speed": 1.0}),
    ]
    assert st2.to_document().to_dict()["tracks"] == \
        doc.to_dict()["tracks"]


def test_document_is_the_same_truth_for_engine_consumers():
    # quality_gate.auto_fix / video_four.tighten consume Document —
    # a façade-compiled document must be a real one for them.
    st = EditStudio("in.png", kind="image", name="qc")
    st.filter("vivid")
    doc = st.document
    assert doc.compute_hash() == doc.doc_hash
    clone = Document.from_dict(doc.to_dict())
    assert clone.to_dict() == doc.to_dict()


# ── honest refusals: nothing is silently dropped ─────────────────────

def test_from_document_rejects_unrepresentable_structure():
    doc = image_document("x.png")
    doc.tracks[0].clips[0] = Clip(path="x.png", gain=0.5)
    with pytest.raises(MediaEditError, match="gain"):
        EditStudio.from_document(doc)

    doc2 = image_document("x.png")
    doc2.add_clip("image", Clip(path="y.png"))
    with pytest.raises(MediaEditError, match="exactly one"):
        EditStudio.from_document(doc2)

    doc3 = Document(tracks=[], title="empty")
    with pytest.raises(MediaEditError, match="no video or image track"):
        EditStudio.from_document(doc3)


def test_session_rejects_the_other_kinds_ops_in_document():
    st = EditStudio("a.mp4", kind="video", name="bad")
    st.op("filter", preset="vivid")  # an image op in a video session
    with pytest.raises(MediaEditError, match="not a video op"):
        st.to_document()
