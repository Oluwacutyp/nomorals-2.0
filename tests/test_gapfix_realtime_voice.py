"""Realtime voice conversion (census §3B.12) — the mined pattern in Devon.

Covers the proto3 wire codec, the frame queue, the speech gate, the
converter (skip-vocode + model paths), the sidecar server, all three
transports, the live loop against fake hardware at the MicSource /
SpeakerSink boundary, file conversion, the chat verb, and the CLI.
Torch tests skip when torch/scipy/librosa are absent; the ZeroMQ
round-trip skips when pyzmq is absent (the refusal test runs instead).
"""

import array
import importlib.util
import math
import os
import socket
import threading
import time
import wave

import pytest

from nomorals.voice import realtime as rt

TORCH = importlib.util.find_spec("torch") is not None
SCIPY = importlib.util.find_spec("scipy.signal") is not None
LIBROSA = importlib.util.find_spec("librosa") is not None
PYZMQ = importlib.util.find_spec("zmq") is not None
SERVICES = importlib.util.find_spec("nomorals.voice.services") is not None

torch_only = pytest.mark.skipif(not (TORCH and SCIPY),
                                reason="needs torch + scipy")
librosa_only = pytest.mark.skipif(not (TORCH and SCIPY and LIBROSA),
                                  reason="needs torch + scipy + librosa")
zmq_only = pytest.mark.skipif(not PYZMQ, reason="needs pyzmq")
no_zmq = pytest.mark.skipif(PYZMQ, reason="needs pyzmq absent")


# -- helpers ---------------------------------------------------------------

def _sine_frames(freq=220.0, seconds=1.0, rate=16000, amp=0.4):
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


def _read_wav(path):
    with wave.open(str(path), "rb") as fh:
        rate = fh.getframerate()
        raw = fh.readframes(fh.getnframes())
    samples = array.array("h")
    samples.frombytes(raw)
    return [s / 32768.0 for s in samples], rate


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _sine_chunk(amp=0.5, freq=220.0, n=480, rate=16000):
    return array.array(
        "h", (int(amp * 32767 * math.sin(2 * math.pi * freq * i / rate))
              for i in range(n))).tobytes()


class FakeMic:
    sample_rate = 16000

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read_chunk(self):
        if self._chunks:
            return self._chunks.pop(0)
        time.sleep(0.01)
        return None

    def close(self):
        pass


class FakeSpeaker:
    def __init__(self):
        self.played = []

    def play(self, wav_path):
        _, rate = _read_wav(wav_path)
        with wave.open(wav_path, "rb") as fh:
            count = fh.getnframes()
        token = len(self.played)
        self.played.append((wav_path, count, rate))
        return token

    def playing(self, token):
        return False

    def stop(self, token):
        pass

    def close(self):
        pass


class _BoomExecutor:
    def execute(self, inputs):
        raise RuntimeError("model exploded")


# -- the proto3 wire codec ---------------------------------------------------

def test_proto_request_round_trip_all_fields():
    request = rt.VocodeAudioRequest(
        float_audio=[0.1, -0.2, 0.3],
        request_batch_number=7,
        vocode_params=rt.VocodeParams(
            initial_sample_rate=44100, original_source_rate=44100,
            original_source_save_file=True, pre_convert_resample=True,
            pre_convert_resample_rate=16000,
            pre_convert_resample_save_file=True,
            model_hyperparameter_sampling_rate=16000,
            model_save_file=True, post_convert_resample=True,
            post_convert_resample_rate=48000,
            post_convert_resample_save_file=True),
        skip_vocode=True, buffer_size_minimum=70000,
        discard_vocoded_audio=True)
    back = rt.VocodeAudioRequest.from_bytes(request.to_bytes())
    assert len(back.float_audio) == 3
    assert all(abs(a - b) < 1e-7
               for a, b in zip(back.float_audio, [0.1, -0.2, 0.3], strict=True))
    assert back.request_batch_number == 7
    assert back.buffer_size_minimum == 70000
    assert back.skip_vocode and back.discard_vocoded_audio
    params = back.vocode_params
    assert params.initial_sample_rate == 44100
    assert params.original_source_rate == 44100
    assert params.original_source_save_file
    assert params.pre_convert_resample
    assert params.pre_convert_resample_rate == 16000
    assert params.pre_convert_resample_save_file
    assert params.model_hyperparameter_sampling_rate == 16000
    assert params.model_save_file
    assert params.post_convert_resample
    assert params.post_convert_resample_rate == 48000
    assert params.post_convert_resample_save_file


def test_proto_request_byte_exact_head():
    # Hand-computed: field 1 packed (tag 0x0A, length 4), one f32 0.5
    # (0x0000003F little-endian), then field 2 (tag 0x10) varint 1.
    request = rt.VocodeAudioRequest(
        float_audio=[0.5], request_batch_number=1,
        vocode_params=rt.VocodeParams(
            model_hyperparameter_sampling_rate=0))
    raw = request.to_bytes()
    assert raw[:8] == bytes([0x0A, 0x04, 0x00, 0x00, 0x00, 0x3F,
                             0x10, 0x01])
    # The params field (field 10, tag 0x52) follows, length 0: the
    # default params encode to nothing.
    assert raw[8:10] == bytes([0x52, 0x00])


def test_proto_unknown_field_tolerated():
    request = rt.VocodeAudioRequest(float_audio=[0.25],
                                    request_batch_number=3)
    raw = request.to_bytes()
    # Append field 99 (varint, tag 0x98 0x06) with value 42.
    raw += bytes([0x98, 0x06, 0x2A])
    back = rt.VocodeAudioRequest.from_bytes(raw)
    assert back.request_batch_number == 3
    assert abs(back.float_audio[0] - 0.25) < 1e-7


def test_proto_response_round_trip():
    response = rt.VocodeAudioResponse(float_audio=[-0.5, 0.5],
                                      response_batch_number=9)
    back = rt.VocodeAudioResponse.from_bytes(response.to_bytes())
    assert back.response_batch_number == 9
    assert all(abs(a - b) < 1e-7
               for a, b in zip(back.float_audio, [-0.5, 0.5], strict=True))


def test_proto_truncated_raises():
    request = rt.VocodeAudioRequest(float_audio=[0.1, 0.2])
    # Chop inside the packed float payload: the length says 8 bytes,
    # only 6 remain.
    with pytest.raises(ValueError):
        rt.VocodeAudioRequest.from_bytes(request.to_bytes()[:-4])
    with pytest.raises(ValueError):
        rt.VocodeAudioRequest.from_bytes(b"\x0a\x08\x00\x00")


def test_proto_params_round_trip():
    params = rt.VocodeParams(original_source_rate=88000,
                             model_hyperparameter_sampling_rate=16000)
    back = rt.VocodeParams.from_bytes(params.to_bytes())
    assert back.original_source_rate == 88000
    assert back.model_hyperparameter_sampling_rate == 16000
    assert not back.pre_convert_resample


# -- the frame queue ---------------------------------------------------------

def test_queue_drain_size_none_when_short():
    queue = rt.AudioFrameQueue()
    queue.extend([1.0, 2.0])
    assert queue.drain_size(3) is None
    assert len(queue) == 2
    assert queue.drain_size(2) == [1.0, 2.0]
    assert len(queue) == 0


def test_queue_bound_drops_oldest_and_counts():
    queue = rt.AudioFrameQueue(max_samples=10)
    queue.extend([1.0] * 8)
    assert queue.drain_size(5) == [1.0] * 5
    queue.extend([2.0] * 8)  # 3 left + 8 = 11 > 10: drops 1 oldest
    assert queue.dropped == 1
    assert queue.drain() == [1.0] * 2 + [2.0] * 8


def test_queue_wait_size_times_out():
    queue = rt.AudioFrameQueue()
    assert queue.wait_size(4, timeout=0.05) is None
    queue.extend([1.0, 2.0, 3.0, 4.0])
    assert queue.wait_size(4, timeout=0.05) == [1.0, 2.0, 3.0, 4.0]


# -- the speech gate ---------------------------------------------------------

def test_gate_silence_never_activates():
    gate = rt.SpeechGate()
    for _ in range(50):
        assert gate.feed([0.0] * 5000) == []
    assert not gate.active
    assert gate.activations == 0


def test_gate_all_negative_is_silence_not_crash():
    # The upstream divided by the count of positive samples here.
    gate = rt.SpeechGate()
    assert not rt.SpeechGate.is_speaking([-0.5] * 5000)
    for _ in range(10):
        assert gate.feed([-0.5] * 5000) == []
    assert not gate.active


def test_gate_threshold():
    assert not rt.SpeechGate.is_speaking([0.005] * 5000)
    assert rt.SpeechGate.is_speaking([0.007] * 5000)
    assert not rt.SpeechGate.is_speaking([0.0] * 5000)


def test_gate_activates_after_three_and_replays_edge():
    gate = rt.SpeechGate()
    speech = [0.5] * 5000
    assert gate.feed(speech) == []
    assert gate.feed(speech) == []
    assert gate.feed(speech) == []
    # 4th speech frame: the count passes 3, the gate activates and the
    # 4-frame ring replays (the upstream's "> 3" rule, faithfully).
    out = gate.feed(speech)
    assert len(out) == 4 * 5000
    assert gate.active
    assert gate.activations == 1
    # While active, each new frame passes straight through.
    assert gate.feed(speech) == speech


def test_gate_deactivates_after_twenty_silent():
    gate = rt.SpeechGate()
    speech = [0.5] * 5000
    silence = [0.0] * 5000
    for _ in range(5):
        gate.feed(speech)
    assert gate.active
    for _ in range(20):
        assert gate.feed(silence) == silence
    assert gate.active  # 20 silent frames: not yet (> 20 to drop)
    assert gate.feed(silence) == []
    assert not gate.active


# -- resampling ---------------------------------------------------------------

def test_resample_identity_and_halving():
    frames = [0.5] * 16000
    assert rt._resample(frames, 16000, 16000) == frames
    assert rt._resample([], 16000, 8000) == []
    out = rt._resample(frames, 16000, 8000)
    assert 7900 < len(out) < 8100
    with pytest.raises(ValueError):
        rt._resample(frames, 0, 8000)


# -- the converter ------------------------------------------------------------

def test_converter_skip_vocode_passthrough():
    frames = _sine_frames(seconds=0.5)
    converter = rt.Converter(None, rt.ConversionParams(skip_vocode=True))
    out = converter.convert(frames, 16000)
    assert len(out) == len(frames)
    assert max(abs(a - b) for a, b in zip(out, frames, strict=True)) < 1e-6


def test_converter_skip_vocode_resamples():
    frames = _sine_frames(seconds=1.0)
    converter = rt.Converter(None, rt.ConversionParams(skip_vocode=True))
    # 16000 samples at 8 kHz is 2.0 s; at the 16 kHz model rate it is
    # ~32000 samples.
    out = converter.convert(frames, 8000)
    assert 31000 < len(out) < 33000


def test_converter_no_model_raises_honest():
    converter = rt.Converter(None, rt.ConversionParams())
    with pytest.raises(rt.ConversionUnavailable) as caught:
        converter.convert(_sine_frames(seconds=0.2), 16000)
    assert "--model" in str(caught.value)


def test_converter_empty_input():
    converter = rt.Converter(None, rt.ConversionParams(skip_vocode=True))
    assert converter.convert([], 16000) == []


def test_converter_discard_returns_model_rate_input():
    frames = _sine_frames(seconds=0.5)
    converter = rt.Converter(
        _IdentityExecutor(),
        rt.ConversionParams(discard_vocoded_audio=True))
    out = converter.convert(frames, 16000)
    assert len(out) == len(frames)
    assert max(abs(a - b) for a, b in zip(out, frames, strict=True)) < 1e-6


def test_conversion_params_pitch_from_f0_stats():
    params = rt.ConversionParams(
        f0_mean_log_src=5.0, f0_std_log_src=0.2,
        f0_mean_log_target=5.0 + math.log(2) / 2, f0_std_log_target=0.2)
    # Half an octave up in log-F0 mean = +6 semitones.
    assert abs(params.pitch_shift() - 6.0) < 1e-9
    assert rt.ConversionParams(pitch_shift_semitones=3.0).pitch_shift() == 3.0
    assert rt.ConversionParams().pitch_shift() == 0.0


def test_params_wire_round_trip():
    params = rt.ConversionParams(
        model_rate=16000, output_rate=48000, buffer_size_minimum=70000,
        skip_vocode=True, discard_vocoded_audio=True)
    wire = params.to_vocode_params(source_rate=44100)
    assert wire.original_source_rate == 44100
    assert wire.pre_convert_resample
    assert wire.pre_convert_resample_rate == 16000
    assert wire.post_convert_resample
    assert wire.post_convert_resample_rate == 48000
    request = rt.VocodeAudioRequest(
        buffer_size_minimum=70000, skip_vocode=True,
        discard_vocoded_audio=True, vocode_params=wire)
    back = rt.ConversionParams.from_vocode_params(wire, request)
    assert back.model_rate == 16000
    assert back.output_rate == 48000
    assert back.skip_vocode and back.discard_vocoded_audio


class _IdentityExecutor:
    def execute(self, inputs):
        return dict(inputs)


@torch_only
def test_probe_export_and_execute(tmp_path):
    out_path = str(tmp_path / "probe.pt")
    receipt = rt.export_probe_model(out_path)
    assert os.path.isfile(out_path)
    assert os.path.isfile(receipt["receipt"])
    assert receipt["inputs"][0]["name"] == "magnitude"
    executor = rt.TorchScriptExecutor(out_path)
    import numpy as np
    magnitude = np.abs(np.random.randn(1, 513, 64)).astype("float32")
    result = executor.execute({"magnitude": magnitude})
    assert result["magnitude"].shape == magnitude.shape
    assert bool(np.isfinite(result["magnitude"]).all())
    assert bool((result["magnitude"] != magnitude).any())


@torch_only
def test_converter_full_path_probe():
    executor = rt.TorchScriptExecutor(_export_probe_to_tmp())
    frames = _sine_frames(seconds=1.0)
    converted = rt.Converter(executor, rt.ConversionParams()).convert(
        frames, 16000)
    assert len(converted) == len(frames)
    assert all(math.isfinite(x) for x in converted)


@torch_only
def test_converter_identity_round_trip():
    frames = _sine_frames(seconds=1.0)
    same = rt.Converter(_IdentityExecutor(),
                        rt.ConversionParams()).convert(frames, 16000)
    assert len(same) == len(frames)
    assert max(abs(a - b) for a, b in zip(same, frames, strict=True)) < 1e-9


@librosa_only
def test_converter_pitch_shift():
    executor = rt.TorchScriptExecutor(_export_probe_to_tmp())
    frames = _sine_frames(seconds=1.0)
    shifted = rt.Converter(
        executor,
        rt.ConversionParams(pitch_shift_semitones=4.0)).convert(
            frames, 16000)
    assert len(shifted) == len(frames)
    assert all(math.isfinite(x) for x in shifted)


@torch_only
def test_converter_griffinlim():
    executor = rt.TorchScriptExecutor(_export_probe_to_tmp())
    frames = _sine_frames(seconds=1.0)
    out = rt.Converter(
        executor, rt.ConversionParams(griffin_iters=8)).convert(
            frames, 16000)
    assert len(out) == len(frames)
    assert all(math.isfinite(x) for x in out)


_PROBE_TMP = []


def _export_probe_to_tmp():
    import tempfile
    if not _PROBE_TMP:
        path = os.path.join(tempfile.mkdtemp(prefix="rt-probe-"),
                            "probe.pt")
        rt.export_probe_model(path)
        _PROBE_TMP.append(path)
    return _PROBE_TMP[0]


# -- the sidecar server -------------------------------------------------------

def _skip_server():
    converter = rt.Converter(
        None, rt.ConversionParams(skip_vocode=True, buffer_size_minimum=10))
    return rt.ConversionServer(converter)


def test_server_buffers_then_converts():
    server = _skip_server()
    first = rt.VocodeAudioRequest(
        float_audio=[0.1] * 5, request_batch_number=1,
        buffer_size_minimum=10, skip_vocode=True)
    assert server.handle(first.to_bytes()) == b"OK"
    second = rt.VocodeAudioRequest(
        float_audio=[0.2] * 6, request_batch_number=2,
        buffer_size_minimum=10, skip_vocode=True)
    response = rt.VocodeAudioResponse.from_bytes(
        server.handle(second.to_bytes()))
    assert len(response.float_audio) == 11
    assert response.response_batch_number == 1
    status = server.status()
    assert status["request_batches"] == 2
    assert status["response_batches"] == 1
    assert status["samples_in"] == 11


def test_server_bad_request_raises():
    server = _skip_server()
    with pytest.raises(ValueError):
        server.handle(b"\x0a\x08\x00\x00")


def test_server_conversion_error_records_and_replies_ok():
    server = rt.ConversionServer(
        rt.Converter(_BoomExecutor(),
                     rt.ConversionParams(buffer_size_minimum=1500)))
    request = rt.VocodeAudioRequest(
        float_audio=[0.1] * 2000, request_batch_number=1,
        buffer_size_minimum=1500)
    assert server.handle(request.to_bytes()) == b"OK"
    status = server.status()
    assert status["errors"] == 1
    assert "exploded" in status["last_error"]


def test_server_honours_request_flags_over_base():
    # The base converter would convert; the request says skip.
    server = rt.ConversionServer(
        rt.Converter(_BoomExecutor(),
                     rt.ConversionParams(buffer_size_minimum=5)))
    request = rt.VocodeAudioRequest(
        float_audio=[0.1] * 6, request_batch_number=1,
        buffer_size_minimum=5, skip_vocode=True)
    response = rt.VocodeAudioResponse.from_bytes(
        server.handle(request.to_bytes()))
    assert len(response.float_audio) == 6
    assert server.status()["errors"] == 0


# -- transports ---------------------------------------------------------------

def test_inprocess_round_trip():
    server = _skip_server()
    client = rt.InProcessClient(server)
    request = rt.VocodeAudioRequest(
        float_audio=[0.1] * 11, request_batch_number=1,
        buffer_size_minimum=10, skip_vocode=True)
    response = rt.VocodeAudioResponse.from_bytes(
        client.request(request.to_bytes()))
    assert len(response.float_audio) == 11
    client.close()


def test_tcp_round_trip():
    server = _skip_server()
    port = _free_port()
    stop = threading.Event()
    thread = threading.Thread(
        target=rt.serve_tcp,
        args=(server, "127.0.0.1", port),
        kwargs={"stop_event": stop}, daemon=True)
    thread.start()
    try:
        client = rt.TcpClient("127.0.0.1", port)
        first = rt.VocodeAudioRequest(
            float_audio=[0.1] * 5, request_batch_number=1,
            buffer_size_minimum=10, skip_vocode=True)
        assert client.request(first.to_bytes()) == b"OK"
        second = rt.VocodeAudioRequest(
            float_audio=[0.2] * 6, request_batch_number=2,
            buffer_size_minimum=10, skip_vocode=True)
        response = rt.VocodeAudioResponse.from_bytes(
            client.request(second.to_bytes()))
        assert len(response.float_audio) == 11
        client.close()
    finally:
        stop.set()
        thread.join(timeout=5)


@no_zmq
def test_zeromq_absent_refusal():
    with pytest.raises(rt.ConversionUnavailable) as caught:
        rt.ZeroMqClient("tcp://127.0.0.1:5555")
    assert "pyzmq" in str(caught.value)


@zmq_only
def test_zeromq_round_trip():
    server = _skip_server()
    stop = threading.Event()
    thread = threading.Thread(
        target=rt.serve_zeromq,
        args=(server, "tcp://127.0.0.1:5562"),
        kwargs={"stop_event": stop}, daemon=True)
    thread.start()
    try:
        client = rt.ZeroMqClient("tcp://127.0.0.1:5562")
        first = rt.VocodeAudioRequest(
            float_audio=[0.1] * 5, request_batch_number=1,
            buffer_size_minimum=10, skip_vocode=True)
        assert client.request(first.to_bytes()) == b"OK"
        second = rt.VocodeAudioRequest(
            float_audio=[0.2] * 6, request_batch_number=2,
            buffer_size_minimum=10, skip_vocode=True)
        response = rt.VocodeAudioResponse.from_bytes(
            client.request(second.to_bytes()))
        assert len(response.float_audio) == 11
        assert response.response_batch_number == 1
        client.close()
    finally:
        stop.set()
        thread.join(timeout=5)


def test_make_client_unknown_transport():
    with pytest.raises(ValueError):
        rt.make_client("smoke-signals")


def test_make_client_inproc_needs_server():
    with pytest.raises(ValueError):
        rt.make_client("inproc")


# -- executors ------------------------------------------------------------------

def test_resolve_executor_no_model(monkeypatch, tmp_path):
    monkeypatch.setenv("NOMORALS_VOICE_SERVICES_DIR", str(tmp_path))
    with pytest.raises(rt.ConversionUnavailable) as caught:
        rt.resolve_executor()
    assert "--model" in str(caught.value)


def test_resolve_executor_prefers_explicit_model(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"fake")
    executor = rt.resolve_executor(model_path=str(path))
    assert isinstance(executor, rt.TorchScriptExecutor)


def test_torchscript_executor_missing_file():
    executor = rt.TorchScriptExecutor("/nonexistent/model.pt")
    with pytest.raises(Exception) as caught:
        executor.execute({"magnitude": [[0.0]]})
    assert "model.pt" in str(caught.value) or "torch" in str(caught.value)


@pytest.mark.skipif(SERVICES, reason="needs the registry module absent")
def test_registry_executor_absent_services():
    executor = rt.RegistryExecutor("voice-convert-probe@0.1.0")
    with pytest.raises(rt.ConversionUnavailable) as caught:
        executor.execute({"magnitude": [[0.0]]})
    assert "registry" in str(caught.value)


# -- file conversion + chat + CLI ---------------------------------------------

def test_convert_file_skip_vocode(tmp_path):
    src = _write_wav(tmp_path / "in.wav", _sine_frames(seconds=0.5))
    out = str(tmp_path / "out.wav")
    receipt = rt.convert_file(
        src, out, executor=None,
        params=rt.ConversionParams(skip_vocode=True))
    assert receipt["ok"]
    assert receipt["input_samples"] == 8000
    assert receipt["output_samples"] == 8000
    frames, rate = _read_wav(out)
    assert rate == 16000
    assert len(frames) == 8000


def test_convert_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rt.convert_file(str(tmp_path / "nope.wav"),
                        str(tmp_path / "out.wav"),
                        executor=None,
                        params=rt.ConversionParams(skip_vocode=True))


def test_control_convert_status():
    result = rt.control_convert("status")
    assert result["ok"]
    assert "torch" in result["text"]
    assert result["path"] == ""


def test_control_convert_usage():
    result = rt.control_convert("")
    assert not result["ok"]
    assert "usage" in result["text"]


def test_control_convert_missing_file():
    result = rt.control_convert("/nonexistent/clip.wav")
    assert not result["ok"]
    assert "no such audio file" in result["text"]


def test_control_convert_no_model_refuses_not_passthrough(
        monkeypatch, tmp_path):
    monkeypatch.setenv("NOMORALS_VOICE_SERVICES_DIR", str(tmp_path))
    src = _write_wav(tmp_path / "in.wav", _sine_frames(seconds=0.2))
    result = rt.control_convert(str(src))
    assert not result["ok"]
    assert result["path"] == ""
    assert "no conversion model" in result["text"]


def test_control_convert_attached_media(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMORALS_VOICE_SERVICES_DIR", str(tmp_path))
    src = _write_wav(tmp_path / "note.wav", _sine_frames(seconds=0.2))

    class _Media:
        kind = "voice"
        path = str(src)

    class _Message:
        media = [_Media()]

    result = rt.control_convert("", message=_Message())
    assert not result["ok"]
    assert "no conversion model" in result["text"]


def test_cli_parser_convert():
    from nomorals.cmdline.parser import _parser
    parser = _parser()
    args = parser.parse_args(
        ["voice", "convert", "clip.wav", "--model", "m.pt", "--json"])
    assert args.voice_action == "convert"
    assert args.file == "clip.wav"
    assert args.model == "m.pt"
    assert args.json
    args = parser.parse_args(["voice", "convert", "--live", "--seconds", "5"])
    assert args.live and args.seconds == 5.0
    args = parser.parse_args(["voice", "convert", "--serve", "--port", "5560",
                              "--transport", "tcp"])
    assert args.serve and args.port == 5560
    args = parser.parse_args(["voice", "convert", "--status"])
    assert args.status


def _fake_context(tmp_path):
    from types import SimpleNamespace
    return SimpleNamespace(
        settings=SimpleNamespace(
            audio=SimpleNamespace(voice_data_dir=str(tmp_path))))


def test_cli_convert_file_mode(tmp_path, monkeypatch):
    from nomorals.cmdline.commands.voice import _cmd_voice_convert
    from nomorals.cmdline.parser import _parser
    # §3B.14: the convert path now caches to the default voice cache
    # dir — point it at scratch so the test never writes the real one.
    monkeypatch.setenv("NM_VOICE_CACHE_DIR", str(tmp_path / "cache"))
    src = _write_wav(tmp_path / "in.wav", _sine_frames(seconds=0.5))
    out = str(tmp_path / "out.wav")
    args = _parser().parse_args(
        ["voice", "convert", str(src), "--out", out, "--skip-vocode"])
    assert _cmd_voice_convert(args, _fake_context(tmp_path)) == 0
    assert os.path.isfile(out)
    frames, _ = _read_wav(out)
    assert len(frames) == 8000


def test_cli_convert_file_mode_json(tmp_path, monkeypatch, capsys):
    from nomorals.cmdline.commands.voice import _cmd_voice_convert
    from nomorals.cmdline.parser import _parser
    # §3B.14: see test_cli_convert_file_mode — cache at scratch.
    monkeypatch.setenv("NM_VOICE_CACHE_DIR", str(tmp_path / "cache"))
    src = _write_wav(tmp_path / "in.wav", _sine_frames(seconds=0.5))
    args = _parser().parse_args(
        ["voice", "convert", str(src), "--skip-vocode", "--json"])
    assert _cmd_voice_convert(args, _fake_context(tmp_path)) == 0
    import json
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["ok"] and receipt["input_samples"] == 8000


def test_cli_convert_status(tmp_path, capsys):
    from nomorals.cmdline.commands.voice import _cmd_voice_convert
    from nomorals.cmdline.parser import _parser
    args = _parser().parse_args(["voice", "convert", "--status"])
    assert _cmd_voice_convert(args, _fake_context(tmp_path)) == 0
    assert "realtime voice conversion" in capsys.readouterr().out


def test_cli_convert_no_model(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMORALS_VOICE_SERVICES_DIR", str(tmp_path))
    from nomorals.cmdline.commands.voice import _cmd_voice_convert
    from nomorals.cmdline.parser import _parser
    src = _write_wav(tmp_path / "in.wav", _sine_frames(seconds=0.2))
    args = _parser().parse_args(["voice", "convert", str(src)])
    assert _cmd_voice_convert(args, _fake_context(tmp_path)) == 1


# -- the live loop --------------------------------------------------------------

def test_live_loop_fake_hardware(tmp_path):
    chunks = [_sine_chunk() for _ in range(200)]
    mic = FakeMic(chunks)
    speaker = FakeSpeaker()
    params = rt.ConversionParams(skip_vocode=True, buffer_size_minimum=5000)
    server = rt.ConversionServer(rt.Converter(None, params))
    loop = rt.RealtimeConverter(
        mic=mic, speaker=speaker, client=rt.InProcessClient(server),
        params=params, data_dir=str(tmp_path))
    report = loop.run(seconds=6.0)
    assert report["batches_sent"] > 0
    assert report["batches_received"] == report["batches_sent"]
    assert report["transport_errors"] == 0
    assert report["activations"] >= 1
    assert len(speaker.played) > 0
    assert sum(n for _, n, _ in speaker.played) > 0
    # Fakes run faster than real time, so the mic's 6 s of audio exhausts
    # before the 6 s wall-clock deadline.
    assert report["end_reason"] == "mic-ended"


class _InfiniteMic:
    """Never ends — the deadline is the only way the loop stops."""

    def read_chunk(self):
        return _sine_chunk()


def test_live_loop_deadline_stops(tmp_path):
    params = rt.ConversionParams(skip_vocode=True, buffer_size_minimum=5000)
    server = rt.ConversionServer(rt.Converter(None, params))
    loop = rt.RealtimeConverter(
        mic=_InfiniteMic(), speaker=FakeSpeaker(),
        client=rt.InProcessClient(server),
        params=params, data_dir=str(tmp_path))
    report = loop.run(seconds=0.5)
    assert report["end_reason"] == "time"
    assert report["batches_sent"] > 0


def test_live_loop_mic_ended(tmp_path):
    chunks = [_sine_chunk() for _ in range(60)]
    mic = FakeMic(chunks)
    speaker = FakeSpeaker()
    params = rt.ConversionParams(skip_vocode=True, buffer_size_minimum=5000)
    server = rt.ConversionServer(rt.Converter(None, params))
    loop = rt.RealtimeConverter(
        mic=mic, speaker=speaker, client=rt.InProcessClient(server),
        params=params, data_dir=str(tmp_path))
    report = loop.run()
    assert report["end_reason"] == "mic-ended"
    assert report["batches_received"] > 0
    assert len(speaker.played) > 0


def test_live_loop_no_mic_raises(tmp_path):
    params = rt.ConversionParams(skip_vocode=True)
    server = rt.ConversionServer(rt.Converter(None, params))
    loop = rt.RealtimeConverter(
        mic=None, speaker=FakeSpeaker(),
        client=rt.InProcessClient(server), params=params,
        data_dir=str(tmp_path))
    with pytest.raises(rt.ConversionUnavailable):
        loop.run(seconds=1.0)


def test_live_loop_no_speaker_raises(tmp_path):
    params = rt.ConversionParams(skip_vocode=True)
    server = rt.ConversionServer(rt.Converter(None, params))
    loop = rt.RealtimeConverter(
        mic=FakeMic([_sine_chunk()]), speaker=None,
        client=rt.InProcessClient(server), params=params,
        data_dir=str(tmp_path))
    with pytest.raises(rt.ConversionUnavailable):
        loop.run(seconds=1.0)


def test_live_loop_silence_sends_nothing(tmp_path):
    silence = array.array("h", [0] * 480).tobytes()
    mic = FakeMic([silence] * 100)
    speaker = FakeSpeaker()
    params = rt.ConversionParams(skip_vocode=True, buffer_size_minimum=5000)
    server = rt.ConversionServer(rt.Converter(None, params))
    loop = rt.RealtimeConverter(
        mic=mic, speaker=speaker, client=rt.InProcessClient(server),
        params=params, data_dir=str(tmp_path))
    report = loop.run()
    assert report["batches_sent"] == 0
    assert report["activations"] == 0
    assert len(speaker.played) == 0


def test_conversion_status_shape():
    status = rt.conversion_status()
    assert set(status) == {"torch", "librosa", "scipy", "pyzmq",
                           "probe_model", "probe_path", "services",
                           # §3B.14: the cache is part of the status now —
                           # conversion_status() reports it by design.
                           "cache"}
    assert isinstance(status["torch"], bool)
