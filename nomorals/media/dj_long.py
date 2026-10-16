"""Long-form DJ: 30–60 min mixes, stems, drop-aware transitions, live DJ.

Extends :mod:`nomorals.media.dj_engine` (short beatmatched transitions)
to the long form:

* **Long mixes** — a whole set planned as one descriptor (order, energy
  arc, transitions, timestamps). Planning is a pure function of the
  descriptor: same descriptor → same mix, byte for byte (``plan_long_mix``
  seeds its RNG from a SHA-256 of the canonical descriptor JSON).
* **Stems, not just crossfades** — each track carries per-stem levels
  (bass / drums / vocal / melody). Transitions pick a stem technique:
  ``bass_swap``, ``filter_sweep``, ``acapella_overlay``, ``drop_cut`` or
  plain ``crossfade``.
* **Drop-aware** — :func:`detect_drop` finds the drop (explicit
  ``drop_at_s``, else the steepest rise of the energy curve) and the
  planner mixes *to* the drop instead of into a random bar.
* **Resumable render** — rendering writes one chunk file per segment
  plus a checkpoint; :func:`resume_render` picks up where it stopped.
  Rendering is honest: without an injected audio ``renderer`` callable
  the chunks are plan-only (``audio_rendered=False``) — no fake WAVs.
  With a renderer, each chunk carries the renderer's artifact path.
* **Opus for WhatsApp** — :func:`opus_profile` gives the WhatsApp-safe
  Opus target (48 kHz mono, bounded bitrate).
* **Taste model** — :class:`TasteProfile` learns from skip / like /
  rewind events and re-ranks candidates. Deterministic, no ML deps.
* **Live DJ** — :class:`LiveDJ` is ``/mix live``: now-playing, vote-next,
  and a requests queue. Pure state machine; sending is the caller's job.
* **Artifacts** — timestamped tracklist text plus a cover: a real image
  when the caller injects an ``imggen`` callable, otherwise an honest
  brief-only dict (``rendered=False``) — never a fake image path.

Only stdlib + ``nomorals.core``. No audio is read or written here.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "LongTrack", "StemLevels", "Transition", "LongMixPlan", "MixChunk",
    "MixArtifact", "TasteProfile", "LiveDJ",
    "detect_drop", "plan_long_mix", "canonical_descriptor",
    "render_long_mix", "resume_render", "opus_profile",
    "tracklist_text", "make_cover_brief", "make_cover",
    "TRANSITION_TYPES", "OPUS_PROFILE",
]

TRANSITION_TYPES = ("bass_swap", "filter_sweep", "acapella_overlay",
                    "drop_cut", "crossfade")

#: WhatsApp-safe Opus target.
OPUS_PROFILE: dict[str, Any] = {
    "codec": "opus", "sample_rate_hz": 48000, "channels": 1,
    "bitrate_kbps": 64, "max_bitrate_kbps": 128,
    "container": "ogg", "note": "WhatsApp voice-note compatible: Opus in Ogg, mono.",
}


def opus_profile() -> dict[str, Any]:
    """Return a copy of the WhatsApp-safe Opus target profile."""
    return dict(OPUS_PROFILE)


# ───────────────────────── tracks / stems ───────────────────────────────────

@dataclass
class StemLevels:
    """Per-stem presence 0..1 for one track (descriptive, not audio)."""

    bass: float = 0.8
    drums: float = 0.8
    vocal: float = 0.5
    melody: float = 0.6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StemLevels":
        return cls(bass=float(d.get("bass", 0.8)), drums=float(d.get("drums", 0.8)),
                   vocal=float(d.get("vocal", 0.5)), melody=float(d.get("melody", 0.6)))


@dataclass
class LongTrack:
    """One track in a long mix. All fields are descriptor data."""

    id: str
    title: str
    artist: str = ""
    bpm: float | None = None
    key: str = ""
    camelot: str = ""
    energy: float = 0.5
    duration_s: float = 180.0
    drop_at_s: float | None = None
    stems: StemLevels = field(default_factory=StemLevels)
    energy_curve: list[float] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LongTrack":
        stems = d.get("stems") or {}
        return cls(
            id=str(d["id"]), title=str(d.get("title", d["id"])),
            artist=str(d.get("artist", "")), bpm=d.get("bpm"),
            key=str(d.get("key", "")), camelot=str(d.get("camelot", "")),
            energy=float(d.get("energy", 0.5)),
            duration_s=float(d.get("duration_s", 180.0)),
            drop_at_s=d.get("drop_at_s"),
            stems=StemLevels.from_dict(stems) if isinstance(stems, dict) else StemLevels(),
            energy_curve=[float(x) for x in d.get("energy_curve", [])],
            tags=[str(x) for x in d.get("tags", [])])


def detect_drop(track: LongTrack) -> float:
    """Drop position in seconds.

    Explicit ``drop_at_s`` wins. Otherwise the steepest rise of the
    energy curve (largest positive delta between adjacent samples),
    mapped onto the track duration. No curve → midpoint.
    """
    if track.drop_at_s is not None:
        return max(0.0, min(float(track.drop_at_s), track.duration_s))
    curve = track.energy_curve
    if len(curve) >= 2 and track.duration_s > 0:
        best_i, best_d = 1, float("-inf")
        for i in range(1, len(curve)):
            delta = curve[i] - curve[i - 1]
            if delta > best_d:
                best_d, best_i = delta, i
        frac = best_i / (len(curve) - 1)
        return round(frac * track.duration_s, 1)
    return round(track.duration_s * 0.5, 1)


# ───────────────────────── plan ─────────────────────────────────────────────

@dataclass
class Transition:
    """One planned transition between two tracks."""

    from_id: str
    to_id: str
    kind: str                       # one of TRANSITION_TYPES
    start_s: float                  # offset in the mix where blend starts
    length_s: float
    to_drop_s: float = 0.0          # drop of the incoming track
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LongMixPlan:
    """A complete long-mix plan. Pure data — reproducible from descriptor."""

    tracks: list[LongTrack]
    transitions: list[Transition]
    target_minutes: float
    seed: int
    total_duration_s: float = 0.0
    created_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"tracks": [t.to_dict() for t in self.tracks],
                "transitions": [t.to_dict() for t in self.transitions],
                "target_minutes": self.target_minutes, "seed": self.seed,
                "total_duration_s": self.total_duration_s,
                "created_ts": self.created_ts}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LongMixPlan":
        return cls(tracks=[LongTrack.from_dict(t) for t in d.get("tracks", [])],
                   transitions=[Transition(**t) for t in d.get("transitions", [])],
                   target_minutes=float(d.get("target_minutes", 0)),
                   seed=int(d.get("seed", 0)),
                   total_duration_s=float(d.get("total_duration_s", 0)),
                   created_ts=float(d.get("created_ts", 0)))

    def descriptor(self) -> dict[str, Any]:
        return self.to_dict()


def canonical_descriptor(tracks: list[LongTrack | dict[str, Any]],
                         target_minutes: float) -> dict[str, Any]:
    """Canonical descriptor JSON for a mix request (seed source)."""
    norm = [t if isinstance(t, dict) else t.to_dict() for t in tracks]
    return {"tracks": norm, "target_minutes": target_minutes}


def _seed_from(descriptor: dict[str, Any]) -> int:
    blob = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha256(blob.encode()).hexdigest()[:8], 16)


def _pick_transition(out_t: LongTrack, in_t: LongTrack,
                    rng: random.Random) -> tuple[str, str]:
    """Choose a stem transition + human-readable note."""
    drop = detect_drop(in_t)
    # Drop-cut wins when the incoming drop is early and energies align.
    if drop <= 45.0 and abs(in_t.energy - out_t.energy) <= 0.35:
        return "drop_cut", f"cut straight into the drop at {drop:.0f}s"
    # Acapella overlay: outgoing is instrumental-ish, incoming has vocals.
    if out_t.stems.vocal <= 0.25 and in_t.stems.vocal >= 0.6:
        return "acapella_overlay", "incoming vocal rides over outgoing instrumental"
    # Bass swap: both tracks have strong bass — swap the low end.
    if out_t.stems.bass >= 0.6 and in_t.stems.bass >= 0.6:
        return "bass_swap", "low ends swapped at the phrase boundary"
    # Filter sweep: big energy jump — sweep the filter open.
    if in_t.energy - out_t.energy >= 0.25:
        return "filter_sweep", "filter sweeps open into the energy lift"
    # Fallback: plain crossfade, phrase-aligned.
    return "crossfade", "phrase-aligned crossfade"


def plan_long_mix(tracks: list[LongTrack | dict[str, Any]],
                  target_minutes: float = 45.0,
                  seed: int | None = None,
                  taste: "TasteProfile | None" = None) -> LongMixPlan:
    """Plan a long mix. Deterministic: same descriptor → same plan.

    Ordering follows an energy arc (warm-up → peak → cool-down) with a
    seeded shuffle inside each third so the arc holds but the order is
    not just chart order. ``taste`` re-ranks candidates when given.
    Transitions are stem-aware and drop-aware (see module docstring).
    """
    norm = [t if isinstance(t, LongTrack) else LongTrack.from_dict(t) for t in tracks]
    if not norm:
        raise ValueError("plan_long_mix needs at least one track")
    descriptor = canonical_descriptor(norm, target_minutes)
    eff_seed = seed if seed is not None else _seed_from(descriptor)
    rng = random.Random(eff_seed)

    # Energy arc: split into thirds, shuffle within each third.
    ordered = sorted(norm, key=lambda t: t.energy)
    n = len(ordered)
    thirds = [ordered[: max(1, n // 3)], ordered[max(1, n // 3): max(1, 2 * n // 3)],
              ordered[max(1, 2 * n // 3):]]
    # Peak in the middle: reorder thirds warm → peak → cool.
    thirds[1] = sorted(thirds[1], key=lambda t: -t.energy)
    arc: list[LongTrack] = []
    for third in thirds:
        part = list(third)
        rng.shuffle(part)
        arc.extend(part)
    if taste is not None:
        # Stable re-rank inside each third by taste score (arc preserved).
        arc = sorted(arc, key=lambda t: -taste.score(t.id))
        # Re-impose arc: sort by (third index, -taste) — arc thirds first.
        def _third_idx(t: LongTrack) -> int:
            pos = ordered.index(t)
            return 0 if pos < max(1, n // 3) else (1 if pos < max(1, 2 * n // 3) else 2)
        arc = sorted(arc, key=lambda t: (_third_idx(t), -taste.score(t.id)))

    # Trim / extend to target: trim from the end (cool-down) first.
    target_s = target_minutes * 60.0
    total = sum(t.duration_s for t in arc)
    chosen = list(arc)
    while len(chosen) > 1 and total > target_s * 1.15:
        dropped = chosen.pop()
        total -= dropped.duration_s
    # If far under target, loop the arc (a real DJ re-plays the vibe).
    guard = 0
    while total < target_s * 0.8 and guard < 4:
        chosen.extend(list(arc))
        total = sum(t.duration_s for t in chosen)
        guard += 1

    transitions: list[Transition] = []
    cursor = 0.0
    for i in range(len(chosen) - 1):
        out_t, in_t = chosen[i], chosen[i + 1]
        kind, notes = _pick_transition(out_t, in_t, rng)
        blend = 16.0 if kind != "drop_cut" else 4.0
        start = cursor + out_t.duration_s - blend
        transitions.append(Transition(
            from_id=out_t.id, to_id=in_t.id, kind=kind,
            start_s=round(start, 1), length_s=blend,
            to_drop_s=detect_drop(in_t), notes=notes))
        cursor += out_t.duration_s - (blend * 0.5 if kind != "drop_cut" else 0.0)
    total_out = cursor + (chosen[-1].duration_s if chosen else 0.0)
    return LongMixPlan(tracks=chosen, transitions=transitions,
                       target_minutes=target_minutes, seed=eff_seed,
                       total_duration_s=round(total_out, 1),
                       created_ts=0.0)  # 0 = plan time; keeps plans reproducible


# ───────────────────────── tracklist / cover ────────────────────────────────

def tracklist_text(plan: LongMixPlan) -> str:
    """Timestamped tracklist, e.g. ``00:00 — Artist — Title [drop_cut]``."""
    lines = [f"🎧 long mix — {len(plan.tracks)} tracks, "
             f"{plan.total_duration_s / 60:.1f} min"]
    cursor = 0.0
    for i, track in enumerate(plan.tracks):
        mm, ss = divmod(int(cursor), 60)
        trans = ""
        if i < len(plan.transitions):
            trans = f" → {plan.transitions[i].kind}"
        lines.append(f"{mm:02d}:{ss:02d} — {track.artist} — {track.title}{trans}")
        cursor += track.duration_s
        if i < len(plan.transitions):
            cursor -= plan.transitions[i].length_s * 0.5
    return "\n".join(lines)


def make_cover_brief(plan: LongMixPlan) -> str:
    """Honest cover brief (no image claimed). Used when no imggen exists."""
    titles = ", ".join(t.title for t in plan.tracks[:3])
    peak = max(plan.tracks, key=lambda t: t.energy) if plan.tracks else None
    return (
        f"Cover brief for a {plan.total_duration_s / 60:.0f}-minute DJ mix. "
        f"Vibe: warm-up → peak → cool-down energy arc; "
        f"peak track: {peak.title if peak else 'n/a'} ({peak.artist if peak else ''}). "
        f"Opening tracks: {titles}. Style: bold title typography, "
        f"dark dancefloor background, no text other than the mix title."
    )


def make_cover(plan: LongMixPlan,
               imggen: Callable[[str], Any] | None = None) -> dict[str, Any]:
    """Cover art: real image via injected ``imggen(brief)``, else brief-only.

    Never returns a fake path. ``{"rendered": False, "brief": ...}`` is
    the honest answer when no generator is injected.
    """
    brief = make_cover_brief(plan)
    if imggen is None:
        return {"rendered": False, "brief": brief, "artifact": None}
    try:
        artifact = imggen(brief)
    except Exception as exc:  # noqa: BLE001 — imggen is caller code
        _log.warning("dj_long cover imggen failed: %s", exc)
        return {"rendered": False, "brief": brief, "artifact": None,
                "error": str(exc)}
    return {"rendered": True, "brief": brief, "artifact": artifact}


# ───────────────────────── resumable render ─────────────────────────────────

@dataclass
class MixChunk:
    index: int
    track_id: str
    transition_kind: str
    start_s: float
    duration_s: float
    audio_rendered: bool = False
    artifact: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MixArtifact:
    plan: LongMixPlan
    chunks: list[MixChunk]
    out_dir: str
    complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"plan": self.plan.to_dict(),
                "chunks": [c.to_dict() for c in self.chunks],
                "out_dir": self.out_dir, "complete": self.complete}


def _checkpoint_path(out_dir: Path) -> Path:
    return out_dir / "mix_checkpoint.json"


def render_long_mix(plan: LongMixPlan, out_dir: str | Path,
                    renderer: Callable[[LongTrack, Transition | None], Any] | None = None,
                    resume: bool = True) -> MixArtifact:
    """Render a mix plan into per-segment chunk files (resumable).

    Each chunk is one track segment (+ its outgoing transition). Chunk
    files are JSON descriptors; ``renderer`` (optional, injected) turns
    a segment into real audio and its return value is stored as the
    chunk artifact. Without a renderer, chunks are plan-only and
    ``audio_rendered`` stays False — the manifest is still complete and
    resumable. Re-running with ``resume=True`` skips already-written
    chunks (checkpoint file) instead of redoing them.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    if resume and _checkpoint_path(out).exists():
        try:
            ck = json.loads(_checkpoint_path(out).read_text(encoding="utf-8"))
            done = {int(i) for i in ck.get("done", [])}
        except (json.JSONDecodeError, OSError, ValueError):
            done = set()
    chunks: list[MixChunk] = []
    for i, track in enumerate(plan.tracks):
        if i in done and (out / f"chunk_{i:03d}.json").exists():
            chunks.append(MixChunk(**json.loads(
                (out / f"chunk_{i:03d}.json").read_text(encoding="utf-8"))))
            continue
        trans = plan.transitions[i - 1] if i > 0 else None
        artifact, rendered = None, False
        if renderer is not None:
            try:
                artifact = renderer(track, trans)
                rendered = True
            except Exception as exc:  # noqa: BLE001 — renderer is caller code
                _log.warning("dj_long renderer failed on %s: %s", track.id, exc)
        chunk = MixChunk(index=i, track_id=track.id,
                         transition_kind=trans.kind if trans else "intro",
                         start_s=0.0, duration_s=track.duration_s,
                         audio_rendered=rendered, artifact=artifact)
        (out / f"chunk_{i:03d}.json").write_text(
            json.dumps(chunk.to_dict(), indent=1), encoding="utf-8")
        done.add(i)
        chunks.append(chunk)
    _checkpoint_path(out).write_text(
        json.dumps({"done": sorted(done),
                    "plan_seed": plan.seed,
                    "complete": len(done) >= len(plan.tracks)}, indent=1),
        encoding="utf-8")
    return MixArtifact(plan=plan, chunks=chunks, out_dir=str(out),
                       complete=len(done) >= len(plan.tracks))


def resume_render(plan: LongMixPlan, out_dir: str | Path,
                  renderer: Callable[[LongTrack, Transition | None], Any] | None = None) -> MixArtifact:
    """Resume an interrupted render (same as render with resume=True)."""
    return render_long_mix(plan, out_dir, renderer=renderer, resume=True)


# ───────────────────────── taste model ──────────────────────────────────────

@dataclass
class TasteProfile:
    """Learned taste from skip / like / rewind events.

    Weights are plain counts → scores, fully deterministic. ``skip``
    counts against a track, ``like`` for it, ``rewind`` (replayed part)
    counts double for it. Unknown tracks score 0.5 (neutral).
    """

    likes: dict[str, int] = field(default_factory=dict)
    skips: dict[str, int] = field(default_factory=dict)
    rewinds: dict[str, int] = field(default_factory=dict)

    def record(self, event: str, track_id: str) -> None:
        event = event.strip().lower()
        if event == "like":
            self.likes[track_id] = self.likes.get(track_id, 0) + 1
        elif event == "skip":
            self.skips[track_id] = self.skips.get(track_id, 0) + 1
        elif event == "rewind":
            self.rewinds[track_id] = self.rewinds.get(track_id, 0) + 1
        else:
            raise ValueError(f"unknown taste event {event!r} (like/skip/rewind)")

    def score(self, track_id: str) -> float:
        like = self.likes.get(track_id, 0) + 2 * self.rewinds.get(track_id, 0)
        skip = self.skips.get(track_id, 0)
        total = like + skip
        if total == 0:
            return 0.5
        return round(like / total, 3)

    def top(self, track_ids: list[str], n: int = 5) -> list[str]:
        return sorted(track_ids, key=lambda t: (-self.score(t), t))[:n]

    def to_dict(self) -> dict[str, Any]:
        return {"likes": self.likes, "skips": self.skips, "rewinds": self.rewinds}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TasteProfile":
        return cls(likes=dict(d.get("likes", {})), skips=dict(d.get("skips", {})),
                   rewinds=dict(d.get("rewinds", {})))


# ───────────────────────── live DJ ──────────────────────────────────────────

@dataclass
class LiveDJ:
    """``/mix live`` session: now-playing + vote-next + requests queue."""

    mix_id: str
    queue: list[str] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    votes: dict[str, dict[str, int]] = field(default_factory=dict)
    now_playing: str | None = None
    played: list[str] = field(default_factory=list)
    started_ts: float = 0.0

    def start(self, first_track: str) -> str:
        self.now_playing = first_track
        self.played.append(first_track)
        self.started_ts = self.started_ts or time.time()
        return f"▶️ now playing: {first_track}"

    def request(self, user: str, track_id: str) -> str:
        self.requests.append({"user": user, "track_id": track_id,
                              "at": time.time()})
        if track_id not in self.queue:
            self.queue.append(track_id)
        return f"🎵 {user} requested {track_id} (queue: {len(self.queue)})"

    def vote_next(self, user: str, track_id: str) -> str:
        per = self.votes.setdefault(track_id, {})
        if user in per:
            return f"{user} already voted for {track_id}"
        per[user] = 1
        return f"🗳️ {user} voted for {track_id} ({len(per)} votes)"

    def vote_counts(self) -> dict[str, int]:
        return {t: len(v) for t, v in self.votes.items()}

    def next_track(self) -> str | None:
        """Highest-voted queued track wins; ties → earliest queued."""
        if not self.queue:
            return None
        counts = self.vote_counts()
        winner = max(self.queue, key=lambda t: (counts.get(t, 0), -self.queue.index(t)))
        self.queue.remove(winner)
        self.votes.pop(winner, None)
        self.now_playing = winner
        self.played.append(winner)
        return winner

    def now_playing_text(self) -> str:
        counts = self.vote_counts()
        leading = max(counts, key=lambda t: counts[t]) if counts else "—"
        return (f"▶️ now playing: {self.now_playing or '—'}\n"
                f"🗳️ leading next: {leading} "
                f"({counts.get(leading, 0)} votes)\n"
                f"🎵 queue: {', '.join(self.queue) or 'empty'}\n"
                f"📩 requests: {len(self.requests)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
