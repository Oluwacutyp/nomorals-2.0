"""Voice cache (census §3B.14, discord-vocodes) — the cache the upstream
never had.

The clone is a 233-line Rust bot: prefix command picks a fixed
speaker, text goes to a TTS endpoint, audio comes back attached.
Every repeat re-fetches and the model reloads per request — no
cache dir, no key, no memo anywhere. Devon's version: converted
audio on disk keyed by model + input + settings (with integrity
checks and LRU eviction), and loaded models shared in memory so a
repeat load never re-runs torch.jit.load.

Covers: key stability, store/lookup round-trip, hit/miss
accounting (persisted across instances), eviction by count and by
bytes, corrupt-entry refusal naming the entry, clear, the model
identity rules, the shared model-load cache (torch tests skip
without torch), convert_file wired through the cache (miss → hit,
different settings → miss, corrupt → recover naming it, store
failure never fails the conversion), the CLI (cache stats/clear,
convert --no-cache), and the chat verbs (/voice convert from
cache, /voice cache stats/clear).
"""

import array
import hashlib
import importlib.util
import json
import math
import os
import wave

import pytest

from nomorals.voice import cache as vc
from nomorals.voice import realtime as rt

TORCH = importlib.util.find_spec("torch") is not None
SCIPY = importlib.util.find_spec("scipy.signal") is not None

torch_only = pytest.mark.skipif(not (TORCH and SCIPY),
                                reason="needs torch + scipy")


def _sine_frames(freq=220.0, seconds=0.5, rate=16000, amp=0.4):
    return [amp * math.sin(2.0 * math.pi * freq * i / rate)
            for i in range(int(rate * seconds))]


def _write_wav(path, frames, rate=16000):
    samples = array.array(
        "h", (max(-32768, min(32767, int(round(s * 32768.0))))
              for s in frames))
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(samples.tobytes())
    return str(path)


def _read_wav_bytes(path):
    with wave.open(str(path), "rb") as fh:
        return fh.readframes(fh.getnframes()), fh.getframerate()


def _sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        digest.update(fh.read())
    return digest.hexdigest()


class _CountingIdentityExecutor:
    """Fake conversion model: magnitude in, magnitude out, counted."""

    def __init__(self):
        self.calls = 0

    def execute(self, inputs):
        self.calls += 1
        return dict(inputs)


def _store(cache, key, wav_path, *, model="none", sha="x" * 64,
           rate=16000, params=None):
    return cache.store(
        key, wav_path, model_identity=model, input_sha256=sha,
        input_rate=rate, params=params or rt.ConversionParams(),
        input_samples=8000, output_samples=8000, output_rate=rate)


# -- keys -------------------------------------------------------------------

def test_key_stable_for_identical_inputs(tmp_path):
    cache = vc.ConversionCache(str(tmp_path))
    params = rt.ConversionParams(direction="A2B")
    first = cache.key_for(model_identity="m", input_sha256="a" * 64,
                          input_rate=16000, params=params)
    second = cache.key_for(model_identity="m", input_sha256="a" * 64,
                           input_rate=16000, params=params)
    assert first == second and len(first) == 64


def test_key_moves_with_input_model_and_settings(tmp_path):
    cache = vc.ConversionCache(str(tmp_path))
    base = dict(model_identity="m", input_sha256="a" * 64,
                input_rate=16000, params=rt.ConversionParams())
    key = cache.key_for(**base)
    assert cache.key_for(**{**base, "input_sha256": "b" * 64}) != key
    assert cache.key_for(**{**base, "model_identity": "n"}) != key
    assert cache.key_for(**{**base, "input_rate": 8000}) != key
    assert cache.key_for(
        **{**base,
           "params": rt.ConversionParams(direction="B2A")}) != key
    assert cache.key_for(
        **{**base,
           "params": rt.ConversionParams(pitch_shift_semitones=2.0)}) != key
    # A plain dict of the same fields keys identically.
    assert cache.key_for(
        model_identity="m", input_sha256="a" * 64, input_rate=16000,
        params={"model_rate": 16000, "output_rate": 0,
                "buffer_size_minimum": 70000, "direction": "A2B",
                "pitch_shift_semitones": 0.0, "f0_mean_log_src": None,
                "f0_std_log_src": None, "f0_mean_log_target": None,
                "f0_std_log_target": None, "skip_vocode": False,
                "discard_vocoded_audio": False, "n_fft": 1024,
                "hop_length": 256, "griffin_iters": 0}) == key


# -- result cache -----------------------------------------------------------

def test_store_lookup_round_trip(tmp_path):
    cache = vc.ConversionCache(str(tmp_path))
    src = _write_wav(tmp_path / "in.wav", _sine_frames())
    key = cache.key_for(model_identity="m", input_sha256=_sha_file(src),
                        input_rate=16000, params=rt.ConversionParams())
    assert cache.lookup(key) is None
    assert cache.status()["misses"] == 1
    _store(cache, key, src)
    hit = cache.lookup(key)
    assert hit is not None
    assert hit["path"].endswith(".wav")
    from pathlib import Path
    assert Path(hit["path"]).read_bytes() == Path(src).read_bytes()
    assert hit["output_samples"] == 8000
    assert cache.status()["hits"] == 1


def test_accounting_persists_across_instances(tmp_path):
    first = vc.ConversionCache(str(tmp_path))
    src = _write_wav(tmp_path / "in.wav", _sine_frames())
    key = "k" * 64
    _store(first, key, src)
    first.lookup(key)
    second = vc.ConversionCache(str(tmp_path))
    status = second.status()
    assert status["stores"] == 1
    assert status["hits"] == 1
    assert status["entries"] == 1
    assert status["bytes"] > 0


def test_eviction_by_count_is_lru(tmp_path):
    cache = vc.ConversionCache(str(tmp_path), max_entries=2)
    src = _write_wav(tmp_path / "in.wav", _sine_frames())
    for key in ("a" * 64, "b" * 64, "c" * 64):
        _store(cache, key, src)
    status = cache.status()
    assert status["entries"] == 2
    assert status["evictions"] == 1
    assert cache.lookup("a" * 64) is None  # oldest evicted
    assert cache.lookup("c" * 64) is not None


def test_eviction_by_bytes(tmp_path):
    cache = vc.ConversionCache(str(tmp_path), max_entries=100,
                               max_bytes=40_000)
    src = _write_wav(tmp_path / "in.wav", _sine_frames(seconds=1.0))
    _store(cache, "a" * 64, src)
    _store(cache, "b" * 64, src)
    status = cache.status()
    assert status["bytes"] <= 40_000
    assert status["evictions"] >= 1
    assert cache.lookup("b" * 64) is not None  # newest survives


def test_corrupt_entry_refuses_by_name(tmp_path):
    cache = vc.ConversionCache(str(tmp_path))
    src = _write_wav(tmp_path / "in.wav", _sine_frames())
    key = "d" * 64
    _store(cache, key, src)
    cached = os.path.join(str(tmp_path), key + ".wav")
    with open(cached, "r+b") as fh:
        fh.seek(50)
        fh.write(b"\x00\x00")
    with pytest.raises(vc.CorruptCacheEntry) as caught:
        cache.lookup(key)
    assert key in str(caught.value)
    assert not os.path.exists(cached)
    assert cache.status()["corrupt"] == 1
    assert cache.lookup(key) is None  # next lookup is a clean miss


def test_clear_drops_entries_and_resets_counters(tmp_path):
    cache = vc.ConversionCache(str(tmp_path))
    src = _write_wav(tmp_path / "in.wav", _sine_frames())
    _store(cache, "a" * 64, src)
    _store(cache, "b" * 64, src)
    cache.lookup("a" * 64)
    assert cache.clear() == 2
    status = cache.status()
    assert status["entries"] == 0
    assert status["hits"] == 0 and status["stores"] == 0


def test_corrupt_stats_file_does_not_break_cache(tmp_path):
    with open(tmp_path / "stats.json", "w") as fh:
        fh.write("{not json")
    cache = vc.ConversionCache(str(tmp_path))
    assert cache.status()["hits"] == 0
    src = _write_wav(tmp_path / "in.wav", _sine_frames())
    _store(cache, "a" * 64, src)
    assert cache.lookup("a" * 64) is not None


def test_default_cache_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_VOICE_CACHE_DIR", str(tmp_path / "mine"))
    assert vc.default_cache_dir() == str(tmp_path / "mine")
    cache = vc.default_conversion_cache()
    assert cache.cache_dir == str(tmp_path / "mine")


# -- model identity ---------------------------------------------------------

def test_executor_identity(tmp_path):
    assert vc.executor_identity(None) == "none"

    class _WithPath:
        path = str(tmp_path / "model.pt")

    path = tmp_path / "model.pt"
    path.write_bytes(b"fake-model-bytes")
    identity = vc.executor_identity(_WithPath())
    assert identity.startswith("torchscript:")
    assert str(tmp_path / "model.pt") in identity
    path.write_bytes(b"fake-model-bytes-longer")
    assert vc.executor_identity(_WithPath()) != identity

    class _WithRef:
        ref = "voice@1.0"
        services_dir = str(tmp_path)

    assert vc.executor_identity(_WithRef()).startswith("service:")
    assert "voice@1.0" in vc.executor_identity(_WithRef())

    class _Missing:
        path = str(tmp_path / "nope.pt")

    assert vc.executor_identity(_Missing()).endswith(":missing")

    class _Unknown:
        pass

    assert vc.executor_identity(_Unknown()).startswith("unknown:")


# -- shared model-load cache ------------------------------------------------

def test_model_load_cache_shares_servers(tmp_path):
    cache = vc.ModelLoadCache()
    first = cache.server(str(tmp_path / "a"))
    second = cache.server(str(tmp_path / "a"))
    assert first is second
    assert cache.server(str(tmp_path / "b")) is not first
    status = cache.status()
    assert status["torchscript"]["loaded"] == 0
    assert len(status["services"]) == 2


def test_model_load_cache_unregistered_service_refuses(tmp_path):
    cache = vc.ModelLoadCache()
    with pytest.raises(Exception, match="ghost"):
        cache.load_service("ghost@9.9.9",
                           services_dir=str(tmp_path / "empty"))


@torch_only
def test_torchscript_load_shared_across_executors(tmp_path, monkeypatch):
    import torch
    import torch.nn as nn

    class _Tiny(nn.Module):
        def forward(self, x):
            return x + 1.0

    model_path = str(tmp_path / "tiny.pt")
    torch.jit.save(torch.jit.script(_Tiny()), model_path)
    loads = {"n": 0}
    real_load = torch.jit.load

    def _counting_load(*args, **kwargs):
        loads["n"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch.jit, "load", _counting_load)
    first = rt.TorchScriptExecutor(model_path)
    second = rt.TorchScriptExecutor(model_path)
    import numpy as np
    mag = np.ones((1, 4, 4), dtype="float32")
    assert first.execute({"magnitude": mag})["magnitude"].shape == (1, 4, 4)
    assert second.execute({"magnitude": mag})["magnitude"].shape == (1, 4, 4)
    assert loads["n"] == 1  # second executor reused the shared load
    # A replaced file (new mtime/size) reloads instead of serving stale.
    torch.jit.save(torch.jit.script(_Tiny()), model_path)
    third = rt.TorchScriptExecutor(model_path)
    third.execute({"magnitude": mag})
    assert loads["n"] == 2


# -- convert_file through the cache -----------------------------------------

def test_convert_file_miss_then_hit(tmp_path):
    cache = vc.ConversionCache(str(tmp_path / "cache"))
    src = _write_wav(tmp_path / "in.wav", _sine_frames(seconds=0.5))
    executor = _CountingIdentityExecutor()
    params = rt.ConversionParams(skip_vocode=False)
    out1 = str(tmp_path / "out1.wav")
    first = rt.convert_file(src, out1, executor=executor, params=params,
                            cache=cache)
    assert first["cached"] is False
    assert executor.calls == 1
    out2 = str(tmp_path / "out2.wav")
    second = rt.convert_file(src, out2, executor=executor, params=params,
                             cache=cache)
    assert second["cached"] is True
    assert second["cache_key"] == first["cache_key"]
    assert executor.calls == 1  # no re-run on the hit
    from pathlib import Path
    assert Path(out1).read_bytes() == Path(out2).read_bytes()
    assert cache.status()["hits"] == 1


def test_convert_file_different_settings_miss(tmp_path):
    cache = vc.ConversionCache(str(tmp_path / "cache"))
    src = _write_wav(tmp_path / "in.wav", _sine_frames(seconds=0.5))
    executor = _CountingIdentityExecutor()
    out = str(tmp_path / "out.wav")
    rt.convert_file(src, out, executor=executor,
                    params=rt.ConversionParams(direction="A2B"),
                    cache=cache)
    receipt = rt.convert_file(src, out, executor=executor,
                              params=rt.ConversionParams(direction="B2A"),
                              cache=cache)
    assert receipt["cached"] is False
    assert executor.calls == 2


def test_convert_file_skip_vocode_cached(tmp_path):
    cache = vc.ConversionCache(str(tmp_path / "cache"))
    src = _write_wav(tmp_path / "in.wav", _sine_frames(seconds=0.5))
    out = str(tmp_path / "out.wav")
    params = rt.ConversionParams(skip_vocode=True)
    first = rt.convert_file(src, out, executor=None, params=params,
                            cache=cache)
    second = rt.convert_file(src, out, executor=None, params=params,
                             cache=cache)
    assert first["cached"] is False and second["cached"] is True


def test_convert_file_corrupt_entry_recovers_and_names_it(tmp_path):
    cache = vc.ConversionCache(str(tmp_path / "cache"))
    src = _write_wav(tmp_path / "in.wav", _sine_frames(seconds=0.5))
    executor = _CountingIdentityExecutor()
    params = rt.ConversionParams()
    out = str(tmp_path / "out.wav")
    first = rt.convert_file(src, out, executor=executor, params=params,
                            cache=cache)
    key = first["cache_key"]
    cached = os.path.join(str(tmp_path / "cache"), key + ".wav")
    with open(cached, "r+b") as fh:
        fh.seek(60)
        fh.write(b"\xff\xff")
    second = rt.convert_file(src, out, executor=executor, params=params,
                             cache=cache)
    assert second["cached"] is False
    assert second.get("cache_corrupt") == key
    assert executor.calls == 2  # re-ran after refusing the corrupt entry
    third = rt.convert_file(src, out, executor=executor, params=params,
                            cache=cache)
    assert third["cached"] is True  # fresh entry replaced the corrupt one


def test_convert_file_without_cache_unchanged(tmp_path):
    src = _write_wav(tmp_path / "in.wav", _sine_frames(seconds=0.5))
    out = str(tmp_path / "out.wav")
    receipt = rt.convert_file(src, out, executor=None,
                              params=rt.ConversionParams(skip_vocode=True))
    assert receipt["ok"] and receipt["cached"] is False
    assert receipt["cache_key"] == ""


def test_convert_file_store_failure_never_fails_conversion(
        tmp_path, monkeypatch):
    cache = vc.ConversionCache(str(tmp_path / "cache"))
    src = _write_wav(tmp_path / "in.wav", _sine_frames(seconds=0.5))
    out = str(tmp_path / "out.wav")

    def _boom(*args, **kwargs):
        raise OSError("disk full (test)")

    monkeypatch.setattr(cache, "store", _boom)
    receipt = rt.convert_file(
        src, out, executor=None,
        params=rt.ConversionParams(skip_vocode=True), cache=cache)
    assert receipt["ok"]
    assert "disk full" in receipt.get("cache_store_error", "")


# -- CLI ----------------------------------------------------------------------

def test_cli_parser_cache_and_no_cache():
    from nomorals.cmdline.parser import _parser
    parser = _parser()
    args = parser.parse_args(["voice", "cache", "stats", "--json"])
    assert args.voice_action == "cache"
    assert args.voice_cache_action == "stats"
    assert args.json
    args = parser.parse_args(["voice", "cache", "clear"])
    assert args.voice_cache_action == "clear"
    args = parser.parse_args(["voice", "convert", "clip.wav", "--no-cache"])
    assert args.no_cache


def _fake_context(tmp_path):
    from types import SimpleNamespace
    return SimpleNamespace(
        settings=SimpleNamespace(
            audio=SimpleNamespace(voice_data_dir=str(tmp_path))))


def test_cli_cache_stats_and_clear(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NM_VOICE_CACHE_DIR", str(tmp_path / "cache"))
    from nomorals.cmdline.commands.voice import _cmd_voice_cache
    from nomorals.cmdline.parser import _parser
    cache = vc.ConversionCache(str(tmp_path / "cache"))
    src = _write_wav(tmp_path / "in.wav", _sine_frames())
    _store(cache, "e" * 64, src)
    args = _parser().parse_args(["voice", "cache", "stats"])
    assert _cmd_voice_cache(args, _fake_context(tmp_path)) == 0
    text = capsys.readouterr().out
    assert "voice cache:" in text
    assert "1 entries" in text
    args = _parser().parse_args(["voice", "cache", "stats", "--json"])
    assert _cmd_voice_cache(args, _fake_context(tmp_path)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cache"]["entries"] == 1
    args = _parser().parse_args(["voice", "cache", "clear"])
    assert _cmd_voice_cache(args, _fake_context(tmp_path)) == 0
    assert "1 cached conversion(s) dropped" in capsys.readouterr().out
    assert vc.ConversionCache(str(tmp_path / "cache")).status()["entries"] == 0


def test_cli_convert_reports_from_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NM_VOICE_CACHE_DIR", str(tmp_path / "cache"))
    from nomorals.cmdline.commands.voice import _cmd_voice_convert
    from nomorals.cmdline.parser import _parser
    src = _write_wav(tmp_path / "in.wav", _sine_frames(seconds=0.5))
    out = str(tmp_path / "out.wav")
    args = _parser().parse_args(
        ["voice", "convert", str(src), "--out", out, "--skip-vocode"])
    assert _cmd_voice_convert(args, _fake_context(tmp_path)) == 0
    assert "from cache" not in capsys.readouterr().out
    assert _cmd_voice_convert(args, _fake_context(tmp_path)) == 0
    assert "from cache" in capsys.readouterr().out
    # --no-cache re-converts even on a hit.
    args = _parser().parse_args(
        ["voice", "convert", str(src), "--out", out, "--skip-vocode",
         "--no-cache"])
    assert _cmd_voice_convert(args, _fake_context(tmp_path)) == 0
    assert "from cache" not in capsys.readouterr().out


def test_cli_convert_status_names_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NM_VOICE_CACHE_DIR", str(tmp_path / "cache"))
    from nomorals.cmdline.commands.voice import _cmd_voice_convert
    from nomorals.cmdline.parser import _parser
    args = _parser().parse_args(["voice", "convert", "--status"])
    assert _cmd_voice_convert(args, _fake_context(tmp_path)) == 0
    assert "cache:" in capsys.readouterr().out


# -- chat ---------------------------------------------------------------------

def test_control_convert_serves_from_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_VOICE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NOMORALS_VOICE_SERVICES_DIR", str(tmp_path))
    executor = _CountingIdentityExecutor()
    monkeypatch.setattr(rt, "resolve_executor",
                        lambda **kwargs: executor)
    src = _write_wav(tmp_path / "note.wav", _sine_frames(seconds=0.2))
    first = rt.control_convert(str(src), data_dir=str(tmp_path / "d1"))
    assert first["ok"]
    assert "from cache" not in first["text"]
    assert executor.calls == 1
    second = rt.control_convert(str(src), data_dir=str(tmp_path / "d2"))
    assert second["ok"]
    assert "from cache" in second["text"]
    assert executor.calls == 1
    third = rt.control_convert(str(src) + " --no-cache",
                               data_dir=str(tmp_path / "d3"))
    assert third["ok"]
    assert "from cache" not in third["text"]
    assert executor.calls == 2


def test_chat_cache_stats_and_clear(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv("NM_VOICE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NOMORALS_VOICES_DIR", str(tmp_path / "voices"))
    from nomorals.agents.partner_runtime import PartnerRuntime
    runtime = PartnerRuntime.__new__(PartnerRuntime)
    runtime.context = SimpleNamespace(extras={}, tools=None)
    cache = vc.ConversionCache(str(tmp_path / "cache"))
    src = _write_wav(tmp_path / "in.wav", _sine_frames())
    _store(cache, "f" * 64, src)
    reply = runtime._control_voice("cache stats", "chat:x")
    assert "voice cache:" in reply
    assert "1 entries" in reply
    reply = runtime._control_voice("cache clear", "chat:x")
    assert "1 cached conversion(s) dropped" in reply
    assert vc.ConversionCache(str(tmp_path / "cache")).status()["entries"] == 0
    assert "cache stats|clear" in runtime._control_voice("bogus", "chat:x")
