"""Open-ended motion generation: any describable action -> pose track."""

import numpy as np
import pytest

from nomorals.media.directed.motion_score import (
    _extract_json,
    compile_score,
    generate_motion_score,
    generate_track,
    kimodo_status,
)

_SCORE = (
    '{"action": "shadow box", "implausible": false, "phases": ['
    '{"name": "guard", "t": [0.0, 0.3], "easing": "ease_out", "moves": ['
    '{"joint": "r_elbow", "to": [0.40, 0.38]},'
    '{"joint": "r_wrist", "to": [0.44, 0.30]},'
    '{"hand": "right", "pose": "fist"}]},'
    '{"name": "jab", "t": [0.3, 0.6], "easing": "ease_out", "repeat": 2,'
    ' "moves": [{"joint": "r_wrist", "to": [0.58, 0.28]}]},'
    '{"name": "recover", "t": [0.6, 1.0], "easing": "ease_in_out",'
    ' "moves": []}]}'
)


def _mock_suggest(prompt: str) -> str:
    return _SCORE


def test_open_generation_any_action():
    track, meta = generate_track(
        "shadow box", suggest=_mock_suggest, n_frames=20)
    assert track.frames.shape == (20, 60, 2)
    assert meta["source"] == "generated"


def test_preset_fast_path_no_model():
    track, meta = generate_track("wave", suggest=None, n_frames=20)
    assert meta["source"] == "preset"
    assert meta["action"] == "wave"
    assert track.frames.shape[0] == 20


def test_unknown_action_no_model_honest_error():
    with pytest.raises(RuntimeError, match="no model"):
        generate_track("do a backflip", suggest=None, n_frames=10)


def test_implausible_action_rejected():
    def bad(prompt: str) -> str:
        return ('{"action": "fly", "implausible": true, '
                '"notes": "humans cannot fly", "phases": []}')
    with pytest.raises(RuntimeError, match="implausible"):
        generate_track("fly like a bird", suggest=bad, n_frames=10)


def test_travel_clamp_noted():
    def far(prompt: str) -> str:
        return ('{"action": "stretch", "phases": [{"name": "x", '
                '"t": [0,1], "easing": "linear", "moves": '
                '[{"joint": "r_wrist", "to": [0.99, 0.01]}]}]}')
    _, meta = generate_track("super stretch", suggest=far, n_frames=10)
    assert any("clamped" in n for n in meta["notes"])


def test_motion_is_directed():
    track, _ = generate_track(
        "shadow box", suggest=_mock_suggest, n_frames=20)
    guard = track.frames[2, 4]   # r_wrist early
    jab = track.frames[10, 4]    # r_wrist mid
    assert float(np.linalg.norm(jab - guard)) > 0.05


def test_fenced_json_extraction():
    assert _extract_json("```json\n" + _SCORE + "\n```")["action"] == "shadow box"


def test_empty_description_rejected():
    with pytest.raises(ValueError):
        generate_track("", suggest=_mock_suggest)


def test_kimodo_status_shape():
    st = kimodo_status()
    assert {"available", "torch", "cuda", "reason"} <= set(st)
    assert isinstance(st["available"], bool)


def test_compile_unknown_joint_ignored():
    score = {"action": "x", "phases": [
        {"name": "p", "t": [0, 1], "easing": "linear",
         "moves": [{"joint": "tentacle", "to": [0.5, 0.5]},
                   {"joint": "r_wrist", "to": [0.5, 0.4]}]}]}
    track, _ = compile_score(score, n_frames=8)
    assert track.frames.shape == (8, 60, 2)


def test_repeat_phase():
    score = {"action": "x", "phases": [
        {"name": "p", "t": [0, 1], "easing": "linear", "repeat": 3,
         "moves": [{"joint": "r_wrist", "to": [0.6, 0.3]}]}]}
    track, _ = compile_score(score, n_frames=30)
    # wrist x should oscillate (3 repeats) rather than sit still
    xs = track.frames[:, 4, 0]
    assert xs.max() - xs.min() > 0.02
