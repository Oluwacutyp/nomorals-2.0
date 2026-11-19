"""SoundCraft parity — machine-checked audio parity (lane 107).

Reads ``nomorals/media/sound-parity.toml``: each case names a real
fixture and an operation with expected outcomes. Missing fixture =
fail, never skip. ``run_parity`` returns ``{"passed": <count>,
"failed": [<case ids>], "passed_ids": [...], "failures": [{"id",
"reason"}]}`` — the same shape as the other organs' parity
runners, with ``skipped`` always 0.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from .effects import Compressor, Limiter, ParametricEQ
from .engine import RealtimeEngine
from .errors import SoundError
from .hosts import HostKind, PluginHost
from .mix import MixEngine
from .model import SessionDocument

__all__ = ["run_parity"]

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "media" / "sound-parity.toml"


def _fixture_path(name: str) -> Path:
    return (Path(__file__).resolve().parents[3] / "tests" / "fixtures"
            / "sound" / name)


def _load(name: str) -> SessionDocument:
    path = _fixture_path(name)
    if not path.exists():
        raise SoundError(f"missing fixture {name}", kind="not_found")
    return SessionDocument.from_bytes(path.read_bytes())


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _dominant_freq(x: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64)))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(freqs[int(np.argmax(spec))])


def _run_case(case: dict[str, Any]) -> None:
    operation = str(case.get("operation", ""))
    fixture = str(case.get("fixture", ""))
    expect = case.get("expect", {})
    if operation == "roundtrip":
        doc = _load(fixture)
        if doc.to_bytes() != _fixture_path(fixture).read_bytes():
            raise AssertionError("round trip bytes differ")
    elif operation == "mix":
        doc = _load(fixture)
        _, _, report = MixEngine(doc).render(0, doc.sample_rate)
        if "master_peak" in expect:
            got = report["master_peak"]
            want = float(expect["master_peak"])
            tol = float(expect.get("tolerance", 0.02))
            if abs(got - want) > tol:
                raise AssertionError(f"master_peak: got {got}, want {want}")
        if expect.get("clipped") is False and report["clipped"]:
            raise AssertionError("unexpected clipping")
    elif operation == "pan_left":
        from .edit import SessionEditor
        doc = _load(fixture)
        ed = SessionEditor(doc)
        for track in doc.tracks:
            ed.set_track_pan(track.id, -1.0)
        _, right, _ = MixEngine(doc).render(0, doc.sample_rate)
        if float(np.max(np.abs(right))) > 1e-9:
            raise AssertionError("right channel not silent under hard left pan")
    elif operation == "eq_boost":
        import wave
        path = _fixture_path(fixture)
        with wave.open(str(path), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        eq = ParametricEQ(48000, freq=440.0, gain_db=12.0, q=1.0)
        ratio = _rms(eq.process(data)) / _rms(data)
        want = float(expect.get("rms_ratio", 3.98))
        if abs(ratio - want) > float(expect.get("tolerance", 0.35)):
            raise AssertionError(f"eq rms ratio: got {ratio}, want {want}")
    elif operation == "compressor":
        loud = np.full(48000, 0.9)
        comp = Compressor(48000, threshold_db=-18.0, ratio=4.0)
        reduction = comp.gain_reduction_db(loud)
        if reduction < float(expect.get("min_reduction_db", 3.0)):
            raise AssertionError(f"gain reduction {reduction} dB too small")
    elif operation == "limiter":
        hot = np.concatenate([np.full(24000, 0.5), np.full(24000, 2.0)])
        out = Limiter(ceiling=0.95).process(hot)
        if float(np.max(np.abs(out))) > 0.95 + 1e-6:
            raise AssertionError("limiter exceeded its ceiling")
    elif operation == "automation":
        doc = _load(fixture)
        left, _, _ = MixEngine(doc).render(0, doc.sample_rate)
        last = left[3 * 48000 // 4:]
        # Last quarter runs at -12 dB (0.25x) of the 0.5-amp sine,
        # centre-panned (1/sqrt(2)): rms = 0.5/sqrt(2)/sqrt(2)*0.25
        # = 0.0625.
        want = float(expect.get("last_quarter_rms", 0.0625))
        if abs(_rms(last) - want) > float(expect.get("tolerance", 0.02)):
            raise AssertionError(f"automated rms: got {_rms(last)}, want {want}")
    elif operation == "midi_tone":
        doc = _load(fixture)
        left, _, _ = MixEngine(doc).render(0, doc.sample_rate)
        freq = _dominant_freq(left[:24000], 48000)
        if abs(freq - 440.0) > 8.0:
            raise AssertionError(f"dominant freq: got {freq}, want 440")
    elif operation == "quarantine":
        host = PluginHost(HostKind.CLAP)
        inst = host.load("nomorals.media.sound.plugins.crash:CrashPlugin")
        try:
            out = inst.process(np.full(512, 0.5), 48000)
            if float(np.max(np.abs(out))) > 1e-9:
                raise AssertionError("crashed plugin did not go silent")
            if inst.state != "quarantined":
                raise AssertionError(f"state: got {inst.state}")
            # The engine is still alive afterwards.
            doc = _load("simple.scraft")
            left, _, report = MixEngine(doc).render(0, 48000)
            if _rms(left) <= 0.1 or report["clipped"]:
                raise AssertionError("engine dead after quarantine")
        finally:
            inst.close()
    elif operation == "open_fails":
        path = _fixture_path(fixture)
        if not path.exists():
            raise SoundError(f"missing fixture {fixture}", kind="not_found")
        try:
            SessionDocument.from_bytes(path.read_bytes())
        except SoundError:
            return
        raise AssertionError("broken fixture opened without error")
    elif operation == "realtime":
        doc = _load(fixture)
        engine = RealtimeEngine(doc, block_size=512)
        engine.play()
        engine.drive(20)
        if engine.beats != 20 or engine.underruns != 0 or engine.faults:
            raise AssertionError(
                f"beats={engine.beats} underruns={engine.underruns} "
                f"faults={engine.faults}")
        engine.stop()
    else:
        raise SoundError(f"unknown parity operation {operation!r}",
                         kind="invalid_argument")


def run_parity(path: str | Path | None = None) -> dict[str, Any]:
    toml_path = Path(path) if path is not None else _DEFAULT_PATH
    if not toml_path.exists():
        return {"passed": 0, "failed": ["<toml>"],
                "passed_ids": [],
                "failures": [{"id": "<toml>",
                              "reason": f"missing {toml_path}"}],
                "skipped": 0}
    with toml_path.open("rb") as handle:
        data = tomllib.load(handle)
    passed_ids: list[str] = []
    failures: list[dict[str, str]] = []
    for case in data.get("cases", []):
        case_id = str(case.get("id", "<unnamed>"))
        try:
            _run_case(case)
        except Exception as exc:  # noqa: BLE001 — one case's failure
            failures.append({"id": case_id,
                             "reason": f"{type(exc).__name__}: {exc}"})
        else:
            passed_ids.append(case_id)
    return {"passed": len(passed_ids),
            "failed": [f["id"] for f in failures],
            "passed_ids": passed_ids,
            "failures": failures,
            "skipped": 0}
