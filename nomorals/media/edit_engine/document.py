"""One unified edit document (EDIT_GEN_UPGRADE 2.1 + 2.3).

A :class:`Document` is the single serializable truth for every edit:

* tracks — video, image, audio, voice, text, music (one model, not six)
* clips with in/out, layers/blend, keyframes (see :mod:`keyframes`)
* effects as descriptors (name + params, validated by :mod:`effects`)
* transcript — word timings the timeline references (see :mod:`transcript`)
* loudness / stems / spatial — doc properties (see :mod:`audio_ops`)
* identity + look locks — project-wide (see :mod:`videogen.consistency`)
* output — width/height/fps/profile/platform (a render hint, not identity)

Specialisations are *views* of one document, never separate engines:

* image  = one-clip, one-frame timeline (a single image track clip)
* video  = multi-track timeline
* audio  = track set + automation (audio/voice/music tracks, no picture)
* voice-over = voice track + transcript
* transcript edit = timeline whose keeps reference word timings

Cross-modal edits are one document: "cut video where he says 'never
again', use that audio as music bed" = a transcript cut on the video
track + the cut audio re-placed as a music-bed layer.

Non-destructive (2.3): the document is truth; renders are derivatives
keyed by ``doc_hash`` + ``doc_version``. :class:`UndoStack` gives
infinite undo/redo over document dicts. :meth:`Document.with_output`
re-renders at a new resolution/profile without touching the edit.
:meth:`Document.ab_variant` forks two docs that differ only in look
pack, so A/B renders come from the same edit.

Serialisation: :meth:`Document.to_dict` / :meth:`from_dict` round-trip
exactly (JSON-safe). :meth:`Document.doc_hash` is the SHA-256 of the
canonical dict (excluding the hash itself) — the render-cache identity
alongside :func:`pure_render.render_cache_key`.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from ...media_edit.videos import MediaEditError
from .audio import AudioLayer, AudioMix
from .spec import EditSpec
from .text import TextLayer
from .timeline import Clip, Effect, Timeline, Track
from .failures import record_failure

__all__ = [
    "TRACK_KINDS",
    "DocTrack",
    "Document",
    "UndoStack",
    "diff_documents",
    "image_document",
    "audio_document",
    "voiceover_document",
    "transcript_document",
]

TRACK_KINDS = ("video", "image", "audio", "voice", "text", "music")

#: output keys that are render hints, not edit identity.
_OUTPUT_KEYS = ("width", "height", "fps", "profile", "platform", "crf",
                "preset", "out")


@dataclass
class DocTrack:
    """One track in a Document. ``kind`` is one of :data:`TRACK_KINDS`.

    Media tracks hold :class:`Clip` objects (in/out, at, speed, effects,
    gain). The ``text`` track holds clips whose ``path`` is empty and
    whose effect list carries ``{"name": "text", "params": {...}}`` —
    or, preferably, callers use :attr:`Document.text_layers` for
    captions; the text track exists so a document can carry ordered
    title cards as first-class timeline items.
    """
    kind: str
    clips: list[Clip] = field(default_factory=list)
    label: str = ""
    muted: bool = False
    blend: str = "normal"  # normal | screen | multiply | overlay | add
    opacity: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in TRACK_KINDS:
            raise MediaEditError(
                f"track kind must be one of {TRACK_KINDS}, got {self.kind!r}")
        if self.blend not in ("normal", "screen", "multiply", "overlay",
                              "add"):
            raise MediaEditError(f"unknown blend {self.blend!r}")
        if not 0.0 <= float(self.opacity) <= 1.0:
            raise MediaEditError("track opacity must be in [0, 1]")
        self.opacity = float(self.opacity)
        self.clips = [c if isinstance(c, Clip) else Clip.from_dict(dict(c))
                      for c in self.clips]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind,
                "clips": [c.to_dict() for c in self.clips],
                "label": self.label, "muted": self.muted,
                "blend": self.blend, "opacity": self.opacity}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocTrack":
        return cls(kind=str(data.get("kind", "video")),
                   clips=[Clip.from_dict(c) for c in data.get("clips", [])],
                   label=str(data.get("label", "")),
                   muted=bool(data.get("muted", False)),
                   blend=str(data.get("blend", "normal")),
                   opacity=float(data.get("opacity", 1.0)))


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonical(value[k])
                for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _canonical(value.to_dict())
    raise MediaEditError(
        f"not JSON-serializable in Document: {type(value).__name__}")


@dataclass
class Document:
    """One edit document — the single truth renders derive from."""
    tracks: list[DocTrack] = field(default_factory=list)
    text_layers: list[TextLayer] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    audio_mix: AudioMix | None = None
    loudness: dict[str, Any] = field(default_factory=dict)
    stems: dict[str, str] = field(default_factory=dict)
    identity_locks: list[dict[str, Any]] = field(default_factory=list)
    look_lock: dict[str, Any] | None = None
    keyframes: list[dict[str, Any]] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)
    title: str = ""
    version: int = 1
    doc_hash: str = ""
    removed: list[dict[str, Any]] = field(default_factory=list)
    # ^ recoverable removed ranges (pacing/transcript cuts): each
    #   {"start","end","reason","words"} — nothing is ever destroyed.

    def __post_init__(self) -> None:
        self.tracks = [t if isinstance(t, DocTrack)
                       else DocTrack.from_dict(dict(t))
                       for t in self.tracks]
        self.text_layers = [
            l if isinstance(l, TextLayer) else TextLayer.from_dict(dict(l))
            for l in self.text_layers]
        if self.audio_mix is not None and not isinstance(self.audio_mix,
                                                         AudioMix):
            self.audio_mix = AudioMix.from_dict(dict(self.audio_mix))
        self.output = dict(self.output or {})
        self.output.setdefault("width", 1080)
        self.output.setdefault("height", 1920)
        self.output.setdefault("fps", 30)
        self.output.setdefault("profile", "workstation")
        self.output.setdefault("platform", "")
        self.version = int(self.version)
        self.transcript = [dict(w) for w in self.transcript]

    # -- identity ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        data = {
            "tracks": [t.to_dict() for t in self.tracks],
            "text_layers": [t.to_dict() for t in self.text_layers],
            "transcript": [dict(w) for w in self.transcript],
            "audio_mix": self.audio_mix.to_dict() if self.audio_mix else None,
            "loudness": dict(self.loudness),
            "stems": dict(self.stems),
            "identity_locks": copy.deepcopy(self.identity_locks),
            "look_lock": copy.deepcopy(self.look_lock),
            "keyframes": copy.deepcopy(self.keyframes),
            "output": dict(self.output),
            "title": self.title,
            "version": self.version,
            "doc_hash": self.doc_hash,
            "removed": copy.deepcopy(self.removed),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(
            tracks=[DocTrack.from_dict(t) for t in data.get("tracks", [])],
            text_layers=[TextLayer.from_dict(t)
                         for t in data.get("text_layers", [])],
            transcript=[dict(w) for w in data.get("transcript", [])],
            audio_mix=(AudioMix.from_dict(data["audio_mix"])
                       if data.get("audio_mix") else None),
            loudness=dict(data.get("loudness") or {}),
            stems=dict(data.get("stems") or {}),
            identity_locks=copy.deepcopy(data.get("identity_locks") or []),
            look_lock=copy.deepcopy(data.get("look_lock")),
            keyframes=copy.deepcopy(data.get("keyframes") or []),
            output=dict(data.get("output") or {}),
            title=str(data.get("title", "")),
            version=int(data.get("version", 1)),
            doc_hash=str(data.get("doc_hash", "")),
            removed=copy.deepcopy(data.get("removed") or []))

    def compute_hash(self) -> str:
        """SHA-256 of the canonical document (hash field excluded)."""
        raw = self.to_dict()
        raw.pop("doc_hash", None)  # the hash never hashes itself
        data = _canonical(raw)
        blob = json.dumps(data, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def seal(self) -> "Document":
        """Bump version + stamp hash. Call after any edit."""
        self.version = int(self.version) + 1
        self.doc_hash = self.compute_hash()
        return self

    def copy(self) -> "Document":
        return Document.from_dict(self.to_dict())

    # -- tracks --------------------------------------------------------
    def track(self, kind: str) -> DocTrack | None:
        for t in self.tracks:
            if t.kind == kind:
                return t
        return None

    def tracks_of(self, kind: str) -> list[DocTrack]:
        return [t for t in self.tracks if t.kind == kind]

    def add_clip(self, kind: str, clip: Clip | dict[str, Any],
                 *, label: str = "") -> "Document":
        """Append a clip to the first track of ``kind`` (creating it)."""
        tr = self.track(kind)
        if tr is None:
            tr = DocTrack(kind=kind, label=label)
            self.tracks.append(tr)
        tr.clips.append(clip if isinstance(clip, Clip)
                        else Clip.from_dict(dict(clip)))
        return self

    @property
    def duration(self) -> float:
        """Longest media-track span (probe-free estimate)."""
        total = 0.0
        for tr in self.tracks:
            if tr.kind == "text":
                continue
            for c in tr.clips:
                try:
                    total = max(total, c.at + c.play_duration())
                except ValueError as e:
                    record_failure("document.duration", e)
                    pass
        return total

    # -- specialisations (views, not separate engines) -----------------
    def to_edit_spec(self) -> EditSpec:
        """Document → :class:`EditSpec` for the video renderer."""
        video = self.track("video")
        clips = list(video.clips) if video else []
        image = self.track("image")
        if not clips and image is not None:
            clips = list(image.clips)
        return EditSpec(
            clips=clips, text_layers=list(self.text_layers),
            audio=self.audio_mix,
            fps=int(self.output.get("fps", 30)),
            width=int(self.output.get("width", 1080)),
            height=int(self.output.get("height", 1920)),
            profile=str(self.output.get("profile", "workstation")),
            preset=str(self.output.get("preset", "")) or None,
            crf=self.output.get("crf"))

    @classmethod
    def from_edit_spec(cls, spec: EditSpec, **kw: Any) -> "Document":
        doc = cls(tracks=[DocTrack(kind="video", clips=list(spec.clips))],
                  text_layers=list(spec.text_layers),
                  audio_mix=spec.audio,
                  output={"width": spec.width, "height": spec.height,
                          "fps": spec.fps, "profile": spec.profile or
                          "workstation", "preset": spec.preset,
                          "crf": spec.crf, "out": spec.out},
                  **kw)
        return doc

    def to_timeline(self) -> Timeline:
        """Document → :class:`Timeline` (video track + text + audio)."""
        return self.to_edit_spec().to_timeline()

    # -- cross-modal ops (2.1) ------------------------------------------
    def cut_where_said(self, phrase: str, *,
                       context_s: float = 0.0) -> list[tuple[float, float]]:
        """Ranges where the transcript says ``phrase`` (case-insensitive).

        Uses :mod:`transcript_edit` range math via the transcript ops
        compiler — the same "what does this phrase cover" as every other
        transcript edit, so cuts and captions can never disagree.
        """
        from ...media_edit.transcript_edit import TimestampedTranscript
        tt = TimestampedTranscript.from_words(self.transcript)
        hits = tt.ranges_for(phrase)
        if context_s:
            out = []
            for s, e in hits:
                out.append((max(0.0, s - context_s), e + context_s))
            return out
        return [(float(s), float(e)) for s, e in hits]

    def cut_video_where_said(self, phrase: str, *, keep: bool = True,
                             context_s: float = 0.0) -> "Document":
        """Cut (or keep-only) the video track where ``phrase`` is said.

        ``keep=True`` keeps only the phrase ranges; ``keep=False`` cuts
        them out. Removed ranges land in :attr:`removed` with their
        words — recoverable, never destroyed.
        """
        from .transcript import keeps_for
        dur = self.duration or 0.0
        if not self.transcript:
            raise MediaEditError("cut_video_where_said needs a transcript")
        if dur <= 0:
            raise MediaEditError("cut_video_where_said needs clip durations")
        hits = self.cut_where_said(phrase, context_s=context_s)
        if not hits:
            raise MediaEditError(f"phrase {phrase!r} never said")
        if keep:
            keeps = [(s, e) for s, e in hits]
            removed = []
            cur = 0.0
            for s, e in sorted(keeps):
                if s > cur:
                    removed.append((cur, s))
                cur = max(cur, e)
            if cur < dur:
                removed.append((cur, dur))
        else:
            keeps = keeps_for(self.transcript, duration=dur,
                              ops=[{"op": "cut_ranges", "ranges": [list(r) for r in hits]}])
            removed = hits
        video = self.track("video")
        if video is None or not video.clips:
            raise MediaEditError("cut_video_where_said needs a video track")
        src = video.clips[0].path
        new_clips = [Clip(path=src, start=s, end=e) for s, e in keeps]
        words = self.transcript
        for s, e in removed:
            inside = [w for w in words
                      if float(w.get("start", 0)) >= s
                      and float(w.get("end", 0)) <= e]
            self.removed.append({"start": s, "end": e,
                                 "reason": f"cut_where_said {phrase!r}",
                                 "words": inside})
        video.clips = new_clips
        return self.seal()

    def use_audio_as_music_bed(self, source_clip_path: str, *,
                               start: float = 0.0, end: float | None = None,
                               volume: float = 0.35,
                               duck_key: int = 0) -> "Document":
        """Cross-modal: take audio (e.g. the cut phrase's audio) and lay
        it under the edit as a music bed. One document, no re-export."""
        layers = list(self.audio_mix.layers) if self.audio_mix else []
        layers.append(AudioLayer(path=source_clip_path, start=0.0,
                                 volume=volume, offset=start,
                                 duration=(end - start) if end else None,
                                 loop=False))
        if self.audio_mix is None:
            self.audio_mix = AudioMix(layers=layers, ducking="sidechain"
                                      if layers else "none",
                                      duck_key=duck_key)
        else:
            self.audio_mix.layers = layers
            if len(layers) > 1:
                self.audio_mix.ducking = "sidechain"
        music = self.track("music")
        if music is None:
            self.tracks.append(DocTrack(kind="music", clips=[
                Clip(path=source_clip_path, start=start, end=end,
                     gain=volume)]))
        else:
            music.clips.append(Clip(path=source_clip_path, start=start,
                                    end=end, gain=volume))
        return self.seal()

    def restore_removed(self, index: int = -1) -> "Document":
        """Recover a removed range back into the video track."""
        if not self.removed:
            raise MediaEditError("nothing removed to restore")
        # validate BEFORE popping: a failed restore must not destroy
        # the recoverable range it was asked to bring back.
        video = self.track("video")
        if video is None or not video.clips:
            raise MediaEditError("no video track to restore into")
        entry = self.removed.pop(index)
        src = video.clips[0].path
        video.clips.append(Clip(path=src, start=float(entry["start"]),
                                end=float(entry["end"])))
        video.clips.sort(key=lambda c: c.start)
        return self.seal()

    # -- non-destructive variants (2.3) ----------------------------------
    def with_output(self, **overrides: Any) -> "Document":
        """Copy with new output hints (res/profile/platform). The edit
        itself is untouched — same tracks, same cuts, new derivative."""
        doc = self.copy()
        for k, v in overrides.items():
            if k not in _OUTPUT_KEYS and k != "look_pack":
                raise MediaEditError(
                    f"with_output takes output keys {sorted(_OUTPUT_KEYS)}, "
                    f"got {k!r}")
            doc.output[k] = v
        return doc.seal()

    def ab_variant(self, look_pack: dict[str, Any]) -> "Document":
        """A/B fork: same edit, different look pack (grade/palette/grain).

        The look lives in :attr:`look_lock` (data, not code) — swapping
        it never touches cuts, timing, or transcript.
        """
        doc = self.copy()
        doc.look_lock = copy.deepcopy(look_pack)
        doc.output["look_pack"] = copy.deepcopy(look_pack)
        return doc.seal()

    def derivative(self, *, profile: str, output_path: str) -> dict[str, Any]:
        """Manifest for one render derivative of this document."""
        return {"doc_hash": self.doc_hash or self.compute_hash(),
                "doc_version": self.version,
                "profile": profile, "output": output_path,
                "width": self.output.get("width"),
                "height": self.output.get("height"),
                "fps": self.output.get("fps"),
                "look_lock": copy.deepcopy(self.look_lock)}


class UndoStack:
    """Infinite undo/redo over document dicts.

    Stores deep copies of :meth:`Document.to_dict` — no cap, because
    "infinite undo" means the stack never silently drops history.
    Callers own memory bounds (a document dict is small JSON).
    """

    def __init__(self, initial: Document | dict[str, Any] | None = None):
        self._past: list[dict[str, Any]] = []
        self._future: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        if initial is not None:
            self._current = self._as_dict(initial)

    @staticmethod
    def _as_dict(doc: Document | dict[str, Any]) -> dict[str, Any]:
        return doc.to_dict() if isinstance(doc, Document) else copy.deepcopy(doc)

    def push(self, doc: Document | dict[str, Any]) -> None:
        if self._current is not None:
            self._past.append(copy.deepcopy(self._current))
        self._current = self._as_dict(doc)
        self._future.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._past)

    @property
    def can_redo(self) -> bool:
        return bool(self._future)

    @property
    def depth(self) -> int:
        return len(self._past)

    def undo(self) -> Document:
        if not self._past:
            raise MediaEditError("nothing to undo")
        self._future.append(copy.deepcopy(self._current))
        self._current = self._past.pop()
        return Document.from_dict(copy.deepcopy(self._current))

    def redo(self) -> Document:
        if not self._future:
            raise MediaEditError("nothing to redo")
        self._past.append(copy.deepcopy(self._current))
        self._current = self._future.pop()
        return Document.from_dict(copy.deepcopy(self._current))

    def current(self) -> Document:
        if self._current is None:
            raise MediaEditError("undo stack is empty")
        return Document.from_dict(copy.deepcopy(self._current))


def diff_documents(a: Document | dict[str, Any],
                   b: Document | dict[str, Any]) -> dict[str, Any]:
    """One diff for one document: which top-level sections changed.

    Returns ``{"changed": [...], "added": {...}, "removed": {...},
    "before_hash", "after_hash"}`` — the "one diff" for review/undo.
    """
    da = a.to_dict() if isinstance(a, Document) else dict(a)
    db = b.to_dict() if isinstance(b, Document) else dict(b)
    before_hash, after_hash = da.get("doc_hash"), db.get("doc_hash")
    da.pop("doc_hash", None)  # reported separately below, not a "change"
    db.pop("doc_hash", None)
    changed = [k for k in sorted(set(da) | set(db))
               if _canonical(da.get(k)) != _canonical(db.get(k))]
    return {"changed": changed,
            "before_version": da.get("version"),
            "after_version": db.get("version"),
            "before_hash": before_hash,
            "after_hash": after_hash}


# -- specialisation constructors -----------------------------------------

def image_document(path: str, *, width: int = 1080, height: int = 1080,
                   title: str = "") -> Document:
    """Image = one-clip, one-frame timeline."""
    return Document(
        tracks=[DocTrack(kind="image", clips=[Clip(path=path)])],
        output={"width": width, "height": height, "fps": 30,
                "profile": "workstation"},
        title=title or path)


def audio_document(paths: list[str], *, title: str = "") -> Document:
    """Audio = track set + automation, no picture track."""
    tracks = [DocTrack(kind="audio",
                       clips=[Clip(path=p) for p in paths])]
    return Document(tracks=tracks, title=title or "audio edit")


def voiceover_document(audio_path: str,
                       transcript: list[dict[str, Any]], *,
                       title: str = "") -> Document:
    """Voice-over = voice track + transcript."""
    return Document(
        tracks=[DocTrack(kind="voice", clips=[Clip(path=audio_path)])],
        transcript=[dict(w) for w in transcript],
        title=title or audio_path)


def transcript_document(source: str,
                        transcript: list[dict[str, Any]], *,
                        title: str = "") -> Document:
    """Transcript edit = timeline referencing word timings."""
    return Document(
        tracks=[DocTrack(kind="video", clips=[Clip(path=source)])],
        transcript=[dict(w) for w in transcript],
        title=title or source)
