"""Lane B.20 — census §3B.20: Devon's own streamed point-cloud organ
(CloudCamPlugin pattern, pattern only, no upstream code).

Proves: the wire protocol (encode/decode round trip, begin vs append,
strict refusal of malformed packets), the thread-safe ring buffer
(drop counting, backpressure named), the four sources (callable, file
JSON/PLY, mocap state store, ZMQ refusal when pyzmq is missing), the
texture-upload staging (hidden sentinel, placeholder, bounds), the
PLY/PNG renderers, the session worker lifecycle, the session registry,
the cloudcam brain tool, the `nm cloudcam` CLI, and the /cloudcam chat
verb — with the tool count pinned at 786 (775 after B.19 added
scene_spawn; 776 after this lane (B.20) added cloudcam). 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet). 801 after gapfix124 added browser_watch. 802 after this lane (SongRecon surface) added songrecon. 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 807 after this lane (agents192) added fleet. 808 after this lane (F3) added control.
"""

from __future__ import annotations

import json
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.media.cloudcam.errors import CloudCamError
from nomorals.media.cloudcam.protocol import (
    COMMAND_APPEND,
    COMMAND_BEGIN,
    CloudCamPoint,
    decode_data_packet,
    decode_length_packet,
    encode_data_packet,
    encode_length_packet,
)
from nomorals.media.cloudcam.registry import (
    get_session,
    list_sessions,
    load_session,
    reset_sessions,
)
from nomorals.media.cloudcam.render import render_projection, write_ply
from nomorals.media.cloudcam.ring import RingBuffer
from nomorals.media.cloudcam.session import CloudCamSession
from nomorals.media.cloudcam.sources import (
    CallableSource,
    FileSource,
    StoreSource,
    ZmqSource,
    depth_shade,
)
from nomorals.media.cloudcam.stage import (
    HIDDEN_POSITION,
    frame_bounds,
    placeholder_grid,
    stage_frame,
)
from nomorals.tools.registry import ToolRegistry

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "kinect"


@pytest.fixture(autouse=True)
def _clean_sessions():
    reset_sessions()
    yield
    reset_sessions()


def _pts(*coords):
    return [CloudCamPoint(float(coords[i]), float(coords[i + 1]),
                          float(coords[i + 2]), 10, 20, 30, 255)
            for i in range(0, len(coords), 3)]


def _frame_points(count=12):
    return [CloudCamPoint(float(i), float(i) * 2.0, float(i) * 3.0,
                          i % 256, 100, 200, 255)
            for i in range(count)]


def _err(outcome) -> str:
    return str(getattr(outcome.error, "message", outcome.error))


def _tool(registry, **kwargs):
    outcome = registry.call("cloudcam", **kwargs)
    if getattr(outcome, "ok", False):
        return outcome.value
    raise AssertionError(
        f"cloudcam call failed: {getattr(outcome, 'error', outcome)}")


def _registry(tmp_path):
    registry = ToolRegistry(SimpleNamespace(home=str(tmp_path)))
    registry.register_builtins()
    return registry


def test_tool_count_is_786_after_cloudcam():
    # 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet). 801 after gapfix124 added browser_watch. 802 after this lane (SongRecon surface) added songrecon. 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 807 after this lane (agents192) added fleet. 808 after this lane (F3) added control.
    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    assert len(registry._tools) == 808
    assert "cloudcam" in registry._tools


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

def test_protocol_round_trip_and_commands():
    pts = _frame_points(50)
    blob = encode_data_packet(pts)
    command, back = decode_data_packet(blob)
    assert command == COMMAND_BEGIN == 2
    assert back == pts
    blob2 = encode_data_packet(pts, command=COMMAND_APPEND)
    command, back = decode_data_packet(blob2)
    assert command == COMMAND_APPEND == 1
    assert back == pts
    assert struct.unpack("<ii", blob[:8]) == (2, len(pts))
    assert struct.unpack("<ii", blob2[:8]) == (1, len(pts))


def test_protocol_length_packet_round_trip():
    blob = encode_length_packet(COMMAND_BEGIN, 1234)
    assert decode_length_packet(blob) == (COMMAND_BEGIN, 1234)
    with pytest.raises(CloudCamError) as exc:
        decode_length_packet(b"\x01\x02")
    assert exc.value.kind == "bad_packet"
    with pytest.raises(CloudCamError) as exc:
        decode_length_packet(encode_length_packet(COMMAND_BEGIN, -1))
    assert "negative" in str(exc.value)


def test_protocol_refuses_malformed():
    with pytest.raises(CloudCamError) as exc:
        decode_data_packet(b"")
    assert exc.value.kind == "bad_packet"
    with pytest.raises(CloudCamError) as exc:
        decode_data_packet(b"\x02\x00\x00\x00\x05\x00\x00\x00")
    assert "but carries" in str(exc.value)
    with pytest.raises(CloudCamError) as exc:
        decode_data_packet(struct.pack("<ii", 2, -1))
    assert "negative" in str(exc.value)
    with pytest.raises(CloudCamError) as exc:
        decode_data_packet(struct.pack("<ii", 2, 500_001))
    assert "over the" in str(exc.value)
    with pytest.raises(CloudCamError) as exc:
        decode_data_packet(struct.pack("<ii", 2, 1) + b"\x00" * 15)
    assert "but carries" in str(exc.value)


def test_protocol_refuses_bad_points():
    with pytest.raises(CloudCamError) as exc:
        CloudCamPoint(float("nan"), 0.0, 0.0)
    assert exc.value.kind == "bad_point"
    with pytest.raises(CloudCamError) as exc:
        CloudCamPoint(0.0, 0.0, 0.0, r=300)
    assert exc.value.kind == "bad_point"
    p = CloudCamPoint(1.0, 2.0, 3.0, 4, 5, 6, 7)
    assert p.as_tuple() == (1.0, 2.0, 3.0, 4, 5, 6, 7)


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------

def test_ring_new_append_latest_and_counts():
    ring = RingBuffer(capacity=3)
    assert ring.latest() is None
    assert ring.has_points() is False
    ring.new_frame(_pts(1, 0, 0), source="test")
    assert ring.frames_received == 1
    assert ring.latest().points == _pts(1, 0, 0)
    ring.append_to_frame(_pts(2, 0, 0))
    assert ring.latest().points == _pts(1, 0, 0) + _pts(2, 0, 0)
    assert ring.has_points() is True
    assert ring.dropped_frames == 0


def test_ring_drop_counting_is_honest():
    ring = RingBuffer(capacity=2)
    ring.new_frame(_pts(1, 0, 0))
    ring.new_frame(_pts(2, 0, 0))
    ring.new_frame(_pts(3, 0, 0))  # no reader: the oldest frame drops
    assert ring.dropped_frames == 1
    assert ring.frames_received == 3
    ring.new_frame(_pts(4, 0, 0))  # overwrites frame 2, never read
    assert ring.dropped_frames == 2
    assert ring.latest().points == _pts(4, 0, 0)


def test_ring_read_marks_seen_so_superseded_frames_do_not_count():
    ring = RingBuffer(capacity=2)
    ring.new_frame(_pts(1, 0, 0))
    ring.new_frame(_pts(2, 0, 0))
    ring.latest()  # the reader saw frame 2; frame 1 is superseded
    ring.new_frame(_pts(3, 0, 0))
    ring.new_frame(_pts(4, 0, 0))
    assert ring.dropped_frames == 0


def test_ring_append_opens_a_frame_when_none_is_open():
    ring = RingBuffer(capacity=2)
    seq = ring.append_to_frame(_pts(1, 0, 0))
    assert seq == 1
    assert ring.frames_received == 1
    assert ring.latest().points == _pts(1, 0, 0)


def test_ring_refuses_bad_capacity_and_oversize_frames():
    with pytest.raises(CloudCamError) as exc:
        RingBuffer(capacity=0)
    assert exc.value.kind == "bad_capacity"
    ring = RingBuffer(capacity=2)
    with pytest.raises(CloudCamError) as exc:
        ring.new_frame([CloudCamPoint(0.0, 0.0, 0.0)] * 1_000_001)
    assert exc.value.kind == "frame_too_large"


def test_ring_thread_safety():
    ring = RingBuffer(capacity=8)
    errors = []

    def writer():
        try:
            for i in range(200):
                ring.new_frame(_pts(i, 0, 0))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    def reader():
        try:
            for _ in range(200):
                ring.latest()
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(2)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors
    assert ring.frames_received == 400
    assert ring.dropped_frames >= 0


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def test_callable_source_frames_then_exhausts():
    frames = [_frame_points(3), _frame_points(5)]
    source = CallableSource(lambda: frames.pop(0) if frames else None)
    events = source.poll()
    assert len(events) == 1
    assert events[0][0] is True and len(events[0][1]) == 3
    events = source.poll()
    assert events[0][0] is True and len(events[0][1]) == 5
    assert source.poll() == []
    assert source.poll() == []


def test_callable_source_coerces_shapes():
    source = CallableSource(lambda: [[1, 2, 3], {"x": 4, "y": 5, "z": 6}])
    events = source.poll()
    assert events[0][1][0] == CloudCamPoint(1.0, 2.0, 3.0)
    assert events[0][1][1] == CloudCamPoint(4.0, 5.0, 6.0)


def test_depth_shade_names_its_derivation():
    shaded = depth_shade([(0.0, 0.0, 0.0), (1.0, 1.0, 1000.0)])
    assert shaded[0].r == 0 and shaded[0].b == 255  # near = blue
    assert shaded[1].r == 255 and shaded[1].b == 0  # far = red
    assert shaded[0].g == 64 and shaded[0].a == 255
    assert depth_shade([]) == []


def test_file_source_json(tmp_path):
    path = tmp_path / "cloud.json"
    path.write_text(json.dumps(
        {"frames": [[list(p.as_tuple()) for p in _frame_points(3)],
                    [list(p.as_tuple()) for p in _frame_points(4)]]}),
        encoding="utf-8")
    source = FileSource(path)
    events = source.poll()
    assert events[0][0] is True and len(events[0][1]) == 3
    events = source.poll()
    assert events[0][0] is True and len(events[0][1]) == 4
    assert source.poll() == []
    assert source.poll() == []


def test_file_source_ply(tmp_path):
    path = tmp_path / "cloud.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n1 2 3 10 20 30\n4 5 6 40 50 60\n",
        encoding="utf-8")
    source = FileSource(path)
    events = source.poll()
    assert events[0][0] is True
    assert events[0][1][0] == CloudCamPoint(1.0, 2.0, 3.0, 10, 20, 30, 255)
    assert events[0][1][1] == CloudCamPoint(4.0, 5.0, 6.0, 40, 50, 60, 255)
    assert source.poll() == []


def test_file_source_refusals(tmp_path):
    with pytest.raises(CloudCamError) as exc:
        FileSource(tmp_path / "missing.json")
    assert exc.value.kind == "missing_file"
    bad = tmp_path / "cloud.bin"
    bad.write_bytes(b"\x00")
    with pytest.raises(CloudCamError) as exc:
        FileSource(bad)
    assert exc.value.kind == "bad_file"
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(CloudCamError):
        FileSource(bad_json)
    bad_ply = tmp_path / "bad.ply"
    bad_ply.write_text("ply\nformat binary_little_endian 1.0\n",
                       encoding="utf-8")
    with pytest.raises(CloudCamError) as exc:
        FileSource(bad_ply)
    assert "binary" in str(exc.value)


def test_store_source_publishes_kinect_pointcloud():
    from nomorals.media.mocap import KinectProducer
    from nomorals.media.mocap.state import MocapStateStore

    store = MocapStateStore()
    KinectProducer(store).publish_fixture(FIXTURES)
    source = StoreSource(store)
    events = source.poll()
    assert events
    assert events[0][0] is True
    assert len(events[0][1]) > 0
    assert all(isinstance(p, CloudCamPoint) for p in events[0][1])
    # Second poll: nothing new since the last frame.
    assert source.poll() == []
    assert source.poll() == []


def test_store_source_resync_after_missed_frames():
    from nomorals.media.mocap import KinectProducer
    from nomorals.media.mocap.state import MocapStateStore

    store = MocapStateStore()
    producer = KinectProducer(store)
    producer.publish_fixture(FIXTURES)
    producer.publish_fixture(FIXTURES)
    source = StoreSource(store)
    # Two frames landed before the source subscribed; it re-syncs to
    # the newest one instead of claiming both.
    events = source.poll()
    assert events
    assert events[0][0] is True and len(events[0][1]) > 0
    assert source.poll() == []


def test_zmq_source_refuses_without_pyzmq():
    try:
        import zmq  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("pyzmq is installed — the live ZMQ path needs "
                    "a real socket test, not this refusal test")
    with pytest.raises(CloudCamError) as exc:
        ZmqSource("tcp://127.0.0.1:8888")
    assert exc.value.kind == "missing_pyzmq"
    assert "pyzmq" in str(exc.value)


# ---------------------------------------------------------------------------
# Staging + rendering
# ---------------------------------------------------------------------------

def test_stage_packs_and_hides():
    grid = stage_frame(_pts(1, 2, 3), 2, 2)
    assert grid.packed == 1
    assert grid.hidden == 3
    assert grid.dropped == 0
    assert grid.placeholder is False
    assert grid.point_count == 1
    assert grid.cells == 4
    x, y, z, w = struct.unpack_from("<ffff", grid.positions, 0)
    assert (x, y, z, w) == (1.0, 2.0, 3.0, 1.0)
    hx, hy, hz, _ = struct.unpack_from("<ffff", grid.positions, 16)
    assert (hx, hy, hz) == (HIDDEN_POSITION, HIDDEN_POSITION,
                            HIDDEN_POSITION)
    assert grid.colors[4:8] == bytes((0, 0, 0, 0))
    assert grid.colors[3] == 255  # texel 0 keeps the point's alpha


def test_stage_placeholder_and_bounds():
    grid = placeholder_grid(2, 2)
    assert grid.packed == 0
    assert grid.hidden == 4
    assert grid.placeholder is True
    assert grid.point_count == 0
    x, y, z, _ = struct.unpack_from("<ffff", grid.positions, 0)
    assert (x, y, z) == (-500.0, -500.0, 0.0)
    assert grid.colors[3] == 10
    bounds = frame_bounds(_pts(-5, 1, 2, 3, 4, 5))
    assert bounds["count"] == 2
    assert bounds["min"] == [-5.0, 1.0, 2.0]
    assert bounds["max"] == [3.0, 4.0, 5.0]
    assert bounds["centroid"] == [-1.0, 2.5, 3.5]
    assert frame_bounds([])["count"] == 0


def test_stage_drops_beyond_capacity_by_name():
    grid = stage_frame(_frame_points(10), 2, 2)
    assert grid.packed == 4
    assert grid.dropped == 6
    assert grid.point_count == 10


def test_render_ply_writes_a_readable_file(tmp_path):
    pts = _frame_points(5)
    out = tmp_path / "cloud.ply"
    result = write_ply(pts, out)
    assert result["points"] == 5
    assert result["path"] == str(out.resolve())
    text = out.read_text(encoding="utf-8")
    assert "element vertex 5" in text
    assert "property uchar alpha" in text
    body = text.split("end_header\n", 1)[1].strip().splitlines()
    assert len(body) == 5
    first = body[0].split()
    assert float(first[0]) == 0.0 and int(first[3]) == 0
    # The file round-trips back through the file source.
    events = FileSource(out).poll()
    assert len(events[0][1]) == 5


def test_render_projection_png(tmp_path):
    out = tmp_path / "view.png"
    rendered = render_projection(_frame_points(20), out, view="top")
    assert rendered["points"] == 20
    assert rendered["plotted"] == 20
    assert rendered["empty"] is False
    assert rendered["path"] == str(out.resolve())
    assert out.stat().st_size > 100
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (rendered["width"], rendered["height"]) == (640, 360)
    empty = render_projection([], tmp_path / "empty.png", view="front")
    assert empty["empty"] is True
    assert empty["points"] == 0
    assert empty["plotted"] == 0
    with pytest.raises(CloudCamError) as exc:
        render_projection(_frame_points(2), tmp_path / "x.png",
                          view="side")
    assert exc.value.kind == "bad_view"


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def test_session_tick_placeholder_then_frame():
    # The placeholder is deterministic before the poller ever runs:
    # a never-started session stages the gradient plane.
    idle = CloudCamSession(name="t0", source=CallableSource(lambda: None),
                           width=4, height=4, capacity=4)
    receipt = idle.tick()
    assert receipt["placeholder"] is True
    assert receipt["frame_seq"] == 0
    assert receipt["point_count"] == 0

    calls = {"n": 0}

    def feed():
        calls["n"] += 1
        return _frame_points(6) if calls["n"] == 1 else None

    session = CloudCamSession(name="t1", source=CallableSource(feed),
                              width=4, height=4, capacity=4)
    started = session.start()
    assert started["running"] is True
    assert started["name"] == "t1"
    with pytest.raises(CloudCamError) as exc:
        session.start()
    assert exc.value.kind == "already_running"
    deadline = time.time() + 5
    while time.time() < deadline:
        receipt = session.tick()
        if receipt["point_count"] > 0:
            break
        time.sleep(0.02)
    assert receipt["placeholder"] is False
    assert receipt["point_count"] == 6
    assert receipt["packed"] == 6
    assert receipt["frame_seq"] >= 1
    assert receipt["bounds"]["count"] == 6
    assert session.status()["ring_frames_received"] >= 1
    assert session.running is True
    session.stop()
    assert session.running is False
    assert session.status()["running"] is False


def test_session_drop_accounting_with_tiny_ring():
    # The feed stays silent until the test says go, so start()'s
    # own status() read cannot mark the first frame seen — the
    # drop count below is then exact, not a race.
    go = threading.Event()
    calls = {"n": 0}

    def feed():
        if not go.is_set():
            return None
        calls["n"] += 1
        return _frame_points(2) if calls["n"] <= 4 else None

    session = CloudCamSession(name="t2", source=CallableSource(feed),
                              capacity=1)
    session.start()
    time.sleep(0.05)
    go.set()
    time.sleep(0.3)
    session.stop()
    status = session.status()
    assert status["ring_frames_received"] == 4
    assert status["ring_dropped_frames"] == 3  # capacity 1, nobody read
    assert status["latest_points"] == 2


def test_session_records_source_errors_and_keeps_running():
    def boom():
        raise CloudCamError("boom", kind="source")

    session = CloudCamSession(name="t3", source=CallableSource(boom))
    session.start()
    time.sleep(0.2)
    status = session.status()
    assert status["error_count"] >= 1
    assert status["last_error"] == "source: boom"
    assert status["running"] is True
    session.stop()


def test_session_stop_joins_the_worker():
    session = CloudCamSession(name="t4",
                              source=CallableSource(lambda: None))
    session.start()
    assert session.running is True
    session.stop()
    assert session.running is False
    session.stop()  # stopping a stopped session is a no-op


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------

def test_registry_load_get_list_reset():
    a = load_session("a", source="idle")
    b = load_session("b", source="idle")
    assert get_session("a") is a
    assert get_session("b") is b
    names = [s["name"] for s in list_sessions()]
    assert set(names) == {"default", "a", "b"}
    with pytest.raises(CloudCamError) as exc:
        get_session("ghost")
    assert exc.value.kind == "unknown_session"
    assert "no cloudcam session" in str(exc.value)
    reset_sessions()
    # list_sessions creates the default on first use.
    assert [s["name"] for s in list_sessions()] == ["default"]
    with pytest.raises(CloudCamError) as exc:
        load_session("a", source="nonsense-source")
    assert exc.value.kind == "bad_source"


def test_registry_replace_stops_the_old_poller():
    old = load_session("x", source="idle")
    old.start()
    assert old.running is True
    new = load_session("x", source="idle")
    assert new is not old
    assert old.running is False
    new.stop()


# ---------------------------------------------------------------------------
# Brain tool
# ---------------------------------------------------------------------------

def test_tool_status_list_start_tick_stop(tmp_path):
    registry = _registry(tmp_path)
    status = _tool(registry, action="status")
    assert status["name"] == "default"
    assert status["running"] is False
    sessions = _tool(registry, action="list")["sessions"]
    assert any(s["name"] == "default" for s in sessions)
    started = _tool(registry, action="start", node="demo")
    assert started["ok"] is True
    assert started["name"] == "demo"
    assert started["running"] is True
    assert started["source"] == "idle"
    # A second tool start on the same name is a load-or-replace
    # (the registry stops the old poller and opens a fresh session),
    # not an error — the already-running refusal lives at session
    # level and is proven in test_session_tick_placeholder_then_frame.
    again = _tool(registry, action="start", node="demo")
    assert again["ok"] is True
    assert again["name"] == "demo"
    assert again["running"] is True
    assert again["ring_frames_received"] == 0
    status = _tool(registry, action="status", node="demo")
    assert status["name"] == "demo"
    receipt = _tool(registry, action="tick", node="demo")
    assert receipt["placeholder"] is True
    stopped = _tool(registry, action="stop", node="demo")
    assert stopped["ok"] is True
    assert stopped["running"] is False


def test_tool_unknown_action_and_missing_session(tmp_path):
    registry = _registry(tmp_path)
    bad = registry.call("cloudcam", action="teleport")
    assert not bad.ok
    assert "unknown cloudcam action" in _err(bad)
    missing = registry.call("cloudcam", action="tick", node="ghost")
    assert not missing.ok
    assert "no cloudcam session" in _err(missing)


def test_tool_export_and_render_with_file_source(tmp_path):
    registry = _registry(tmp_path)
    cloud = tmp_path / "cloud.json"
    cloud.write_text(json.dumps(
        {"frames": [[list(p.as_tuple()) for p in _frame_points(8)]]}),
        encoding="utf-8")
    _tool(registry, action="start", node="filedemo",
          params={"source": "file", "path": str(cloud)})
    deadline = time.time() + 5
    receipt = {}
    while time.time() < deadline:
        receipt = _tool(registry, action="tick", node="filedemo")
        if receipt["point_count"] > 0:
            break
        time.sleep(0.02)
    assert receipt["point_count"] == 8
    out = tmp_path / "out.ply"
    exported = _tool(registry, action="export", node="filedemo",
                     output=str(out))
    assert exported["points"] == 8
    assert exported["session"] == "filedemo"
    assert "element vertex 8" in out.read_text(encoding="utf-8")
    png = tmp_path / "view.png"
    rendered = _tool(registry, action="render", node="filedemo",
                     output=str(png))
    assert rendered["plotted"] == 8
    assert png.exists()
    _tool(registry, action="stop", node="filedemo")


def test_tool_export_refuses_without_a_frame(tmp_path):
    registry = _registry(tmp_path)
    _tool(registry, action="start", node="empty")
    missing = registry.call("cloudcam", action="export", node="empty",
                            output=str(tmp_path / "x.ply"))
    assert not missing.ok
    assert "has staged no frame yet" in _err(missing)
    _tool(registry, action="stop", node="empty")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_list_status_start_tick_export_stop(tmp_path):
    from nomorals.cmdline.dispatch import main

    cloud = tmp_path / "cloud.json"
    cloud.write_text(json.dumps(
        {"frames": [[list(p.as_tuple()) for p in _frame_points(6)]]}),
        encoding="utf-8")
    assert main(["cloudcam", "list"]) == 0
    assert main(["cloudcam", "status"]) == 0
    assert main(["cloudcam", "start", "--session", "cli",
                 "--source", "file", "--path", str(cloud)]) == 0
    deadline = time.time() + 5
    staged = 0
    while time.time() < deadline:
        assert main(["cloudcam", "tick", "--session", "cli"]) == 0
        if get_session("cli").latest_points():
            staged = len(get_session("cli").latest_points())
            break
        time.sleep(0.02)
    assert staged == 6
    out = tmp_path / "cli.ply"
    assert main(["cloudcam", "export", "--session", "cli",
                 "--out", str(out)]) == 0
    assert "element vertex 6" in out.read_text(encoding="utf-8")
    png = tmp_path / "cli.png"
    assert main(["cloudcam", "render", "--session", "cli",
                 "--out", str(png), "--view", "front"]) == 0
    assert png.exists()
    assert main(["cloudcam", "stop", "--session", "cli"]) == 0
    assert main(["cloudcam", "status", "--session", "cli"]) == 0


def test_cli_bad_source_fails(tmp_path):
    from nomorals.cmdline.dispatch import main

    # argparse choices: an unknown source is a usage error (rc 2).
    assert main(["cloudcam", "start", "--session", "bad",
                 "--source", "nope"]) == 2
    assert main(["cloudcam", "tick", "--session", "ghost"]) == 1


# ---------------------------------------------------------------------------
# Chat verb
# ---------------------------------------------------------------------------

def test_chat_control_cloudcam():
    from nomorals.agents.partner.runtime_mocap import RuntimeMocapMixin

    mixin = RuntimeMocapMixin()
    help_text = mixin._control_cloudcam("", "test")
    assert "/cloudcam start" in help_text
    assert "cloudcam sessions:" in mixin._control_cloudcam("list", "test")
    started = mixin._control_cloudcam("start", "test")
    assert "streaming" in started
    ticked = mixin._control_cloudcam("tick", "test")
    assert "placeholder" in ticked
    status = mixin._control_cloudcam("status", "test")
    assert '"name": "default"' in status
    assert "stopped" in mixin._control_cloudcam("stop", "test")
    assert "/cloudcam start" in mixin._control_cloudcam("bogus", "test")


def test_chat_control_mocap_still_works():
    from nomorals.agents.partner.runtime_mocap import RuntimeMocapMixin

    mixin = RuntimeMocapMixin()
    assert "rig list" in mixin._control_mocap("", "test")
