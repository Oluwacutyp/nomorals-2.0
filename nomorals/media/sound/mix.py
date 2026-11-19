"""SoundCraft mixer — strips, pan law, sends, meters (lane 107).

Ported from storytold/soundcraft's ``crates/mix`` (strip, pan,
routing, meter) into Devon's Python. Sample-accurate: every track
is summed per sample, pan is the equal-power law (centre = -3 dB
per side, the same law upstream's pan.rs uses), sends are
post-fader by default, and clipping is reported honestly — the
master peak and the first clipping sample are in the report, never
hidden by a silent limiter.
"""

from __future__ import annotations

import math

import numpy as np

from .automation import AutomationLane
from .model import SessionDocument

__all__ = ["MixEngine", "pan_gains"]

_MIDI_A4 = 440.0


def pan_gains(pan: float) -> tuple[float, float]:
    """Equal-power stereo pan: -1 left, 0 centre, +1 right."""
    pan = max(-1.0, min(1.0, float(pan)))
    angle = (pan + 1.0) * math.pi / 4.0
    return math.cos(angle), math.sin(angle)


def _note_freq(pitch: int) -> float:
    return _MIDI_A4 * (2.0 ** ((int(pitch) - 69) / 12.0))


def _render_notes(notes: list[dict], start: int, count: int,
                  sample_rate: int, offset: int = 0) -> np.ndarray:
    """Render MIDI notes as sine tones with an ADSR envelope.

    ``offset`` is the clip start in session samples; note starts are
    relative to the clip.
    """
    out = np.zeros(count, dtype=np.float64)
    attack = max(1, int(0.005 * sample_rate))
    release = max(1, int(0.100 * sample_rate))
    for note in notes:
        ns = int(note["start"]) + offset
        nd = int(note["duration"])
        vel = float(note.get("velocity", 0.8))
        freq = _note_freq(int(note["pitch"]))
        lo = max(ns, start)
        hi = min(ns + nd, start + count)
        if hi <= lo:
            continue
        idx = np.arange(lo - ns, hi - ns, dtype=np.float64)
        env = np.ones(hi - lo)
        a = min(attack, hi - lo)
        env[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
        r = min(release, hi - lo)
        env[-r:] = np.linspace(1.0, 0.0, r, endpoint=False)
        env *= 0.7  # sustain level
        tone = vel * env * np.sin(2.0 * math.pi * freq * idx / sample_rate)
        out[lo - start:hi - start] += tone
    return out


class MixEngine:
    """Mix a session document to stereo, sample-accurately."""

    def __init__(self, doc: SessionDocument, block_size: int = 512):
        self.doc = doc
        self.sample_rate = doc.sample_rate
        self.block_size = int(block_size)
        self._inserts: dict[int, list] = {}
        for track in doc.tracks:
            self._inserts[track.id] = []
        self._resolve_inserts()

    def _resolve_inserts(self) -> None:
        from .effects import Compressor, GainPlugin, Limiter, ParametricEQ
        for track in self.doc.tracks:
            chain = []
            for spec in track.inserts:
                kind = spec.get("kind")
                params = spec.get("params", {})
                if kind == "gain":
                    chain.append(GainPlugin(float(params.get("gain_db", 0.0))))
                elif kind == "eq":
                    chain.append(ParametricEQ(
                        self.sample_rate, float(params.get("freq", 1000.0)),
                        float(params.get("gain_db", 0.0)),
                        float(params.get("q", 1.0)),
                        str(params.get("eq_kind", "peak"))))
                elif kind == "compressor":
                    chain.append(Compressor(
                        self.sample_rate,
                        float(params.get("threshold_db", -18.0)),
                        float(params.get("ratio", 4.0)),
                        float(params.get("attack_s", 0.005)),
                        float(params.get("release_s", 0.1))))
                elif kind == "limiter":
                    chain.append(Limiter(float(params.get("ceiling", 0.95))))
            self._inserts[track.id] = chain

    def insert_plugin(self, track_id: int, plugin: object) -> None:
        """Attach a live plugin object to a track's insert chain
        (in-process plugins and quarantined instances alike)."""
        self._inserts.setdefault(track_id, []).append(plugin)

    def _track_signal(self, track, start: int, count: int) -> np.ndarray:
        out = np.zeros(count, dtype=np.float64)
        for clip in self.doc.clips:
            if clip.track_id != track.id:
                continue
            lo = max(clip.start, start)
            hi = min(clip.start + clip.duration, start + count)
            if hi <= lo:
                continue
            if clip.notes:
                sig = _render_notes(clip.notes, start, count,
                                    self.sample_rate, offset=clip.start)
                out += sig * (10.0 ** (clip.gain_db / 20.0))
                continue
            if clip.source_id is None:
                continue
            source = self.doc.source_samples(clip.source_id)
            src_lo = clip.offset + (lo - clip.start)
            src_hi = src_lo + (hi - lo)
            chunk = source[src_lo:src_hi]
            if len(chunk) < hi - lo:
                chunk = np.pad(chunk, (0, hi - lo - len(chunk)))
            out[lo - start:hi - start] += (
                chunk * (10.0 ** (clip.gain_db / 20.0)))
        return out

    def _track_gain(self, track, start: int, count: int) -> np.ndarray:
        base = 10.0 ** (track.gain_db / 20.0)
        lane_raw = track.automation.get("gain_db")
        if not lane_raw:
            return np.full(count, base)
        lane = AutomationLane.from_list("gain_db", lane_raw)
        db = lane.values(start, count)
        return (10.0 ** (db / 20.0)) * base

    def render(self, start: int, count: int) -> tuple[np.ndarray, np.ndarray, dict]:
        """Render ``count`` samples from ``start`` to stereo.

        Returns ``(left, right, report)``; the report carries the
        master peak, per-strip peaks, per-bus peaks, and the first
        clipping sample when the master exceeds 0 dBFS.
        """
        left = np.zeros(count, dtype=np.float64)
        right = np.zeros(count, dtype=np.float64)
        strip_peaks: dict[int, float] = {}
        bus_peaks: dict[str, float] = {}
        bus_names = {b.id: b.name for b in self.doc.buses}
        any_solo = any(t.solo for t in self.doc.tracks)
        for track in self.doc.tracks:
            if track.mute and not track.solo:
                strip_peaks[track.id] = 0.0
                continue
            if any_solo and not track.solo:
                strip_peaks[track.id] = 0.0
                continue
            sig = self._track_signal(track, start, count)
            for plugin in self._inserts.get(track.id, []):
                sig = np.asarray(plugin.process(sig, self.sample_rate),
                                 dtype=np.float64)
            sig = sig * self._track_gain(track, start, count)
            strip_peaks[track.id] = float(
                np.max(np.abs(sig))) if count else 0.0
            for send in track.sends:
                bus_id = int(send["bus_id"])
                send_gain = 10.0 ** (float(send.get("gain_db", 0.0)) / 20.0)
                name = bus_names.get(bus_id, str(bus_id))
                bus_peaks[name] = max(
                    bus_peaks.get(name, 0.0),
                    float(np.max(np.abs(sig * send_gain))) if count else 0.0)
            gl, gr = pan_gains(track.pan)
            left += sig * gl
            right += sig * gr
        master_peak = float(max(
            np.max(np.abs(left)) if count else 0.0,
            np.max(np.abs(right)) if count else 0.0))
        bus_peaks["Master"] = master_peak
        clipped = master_peak > 1.0
        clip_first = None
        if clipped:
            over = np.where((np.abs(left) > 1.0) | (np.abs(right) > 1.0))[0]
            if len(over):
                clip_first = int(start + int(over[0]))
        report = {
            "master_peak": master_peak,
            "clipped": clipped,
            "clip_first_sample": clip_first,
            "strip_peaks": strip_peaks,
            "bus_peaks": bus_peaks,
        }
        return left, right, report
