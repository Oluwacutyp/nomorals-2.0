"""Clicky interaction mechanisms on the scene organ (scout 17, A9).

The interaction half of the Farza/Clicky visual surface, built natively
on the scene-spec organ lane 1 shipped (``nomorals/visuals/``):

- the [POINT] tag protocol + the full tag vocabulary, parsed out of
  the brain's reply text and stripped before it is spoken or shown;
- the quadratic-Bezier flight, as deterministic pure data;
- annotation lifecycle objects (armed -> completed | missed) on a
  durable board;
- the authenticated localhost bridge control plane (token-gated,
  host-checked, rate-limited);
- the idle trigger (danpeg's acted-since-last-check gate);
- the interactive live page, where the flight is visible.

The draft's section 8 DO NOT COPY list binds throughout: annotation
is never actuation (a tag places a marker; nothing here moves a mouse
or clicks), the bridge never trusts URL obscurity, nothing is
cloud-only, and the surface is profile-runged by the organ's own
render ladder.
"""

from __future__ import annotations

import json
import math
import socket
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from nomorals.visuals import annotations as ann_mod
from nomorals.visuals import bridge as bridge_mod
from nomorals.visuals import flight as flight_mod
from nomorals.visuals import idle as idle_mod
from nomorals.visuals import interactive as inter_mod
from nomorals.visuals import tags as tags_mod
from nomorals.visuals.spec import SceneSpecError, validate_scene_spec

# ── tag protocol ─────────────────────────────────────────────────────


class TagProtocolTests(unittest.TestCase):
    def test_point_tag_round_trip(self):
        parsed = tags_mod.parse_tags(
            "Look at the button [POINT:640,360:the button] right there",
            width=1280, height=720)
        assert parsed.text == "Look at the button  right there"
        assert len(parsed.tags) == 1
        tag = parsed.tags[0]
        assert tag.kind == "point"
        assert tag.x == pytest.approx(0.5)
        assert tag.y == pytest.approx(0.5)
        assert tag.label == "the button"
        assert tag.screen == 1
        assert parsed.none is False

    def test_point_none_sentinel(self):
        parsed = tags_mod.parse_tags(
            "Nothing worth pointing at [POINT:none]", width=1280,
            height=720)
        assert parsed.none is True
        assert parsed.tags == []
        assert "[POINT" not in parsed.text

    def test_point_screen_suffix(self):
        parsed = tags_mod.parse_tags(
            "[POINT:400,300:terminal:screen2]", width=1280, height=720,
            screen_count=2)
        assert parsed.tags[0].screen == 2
        assert parsed.tags[0].label == "terminal"

    def test_point_screen_out_of_range_clamps_to_cursor_screen(self):
        parsed = tags_mod.parse_tags(
            "[POINT:100,100:x:screen9]", width=1280, height=720,
            screen_count=2)
        assert parsed.tags[0].screen == 1

    def test_rect_highlight_scribble_shape_caption(self):
        parsed = tags_mod.parse_tags(
            "[RECT:0,0,640,360:the panel] "
            "[HIGHLIGHT:640,0,640,360:look] "
            "[SCRIBBLE:10,10;20,30;40,30:the path] "
            "[SHAPE:arrow:0,0:640,360:flow] "
            "[CAPTION:640,700:done]",
            width=1280, height=720)
        kinds = [t.kind for t in parsed.tags]
        assert kinds == ["rect", "highlight", "scribble", "shape",
                         "caption"]
        rect = parsed.tags[0]
        assert rect.w == pytest.approx(0.5)
        assert rect.h == pytest.approx(0.5)
        scribble = parsed.tags[2]
        assert len(scribble.points) == 3
        assert scribble.points[1] == pytest.approx(
            (20 / 1280, 30 / 720))
        shape = parsed.tags[3]
        assert shape.shape == "arrow"
        assert shape.x == pytest.approx(0.0)
        assert shape.y == pytest.approx(0.0)

    def test_multiple_tags_anywhere_in_text(self):
        parsed = tags_mod.parse_tags(
            "First [POINT:10,10:a] middle [POINT:20,20:b] end",
            width=1280, height=720)
        assert len(parsed.tags) == 2
        assert parsed.text == "First  middle  end"

    def test_normalized_coords_without_frame(self):
        parsed = tags_mod.parse_tags("[POINT:0.25,0.75:quarter]")
        assert parsed.tags[0].x == pytest.approx(0.25)
        assert parsed.tags[0].y == pytest.approx(0.75)

    def test_malformed_tags_left_in_text_never_crash(self):
        text = "Broken [POINT:abc,def:x] and [POINT:1.5,0.5:out] kept"
        parsed = tags_mod.parse_tags(text)
        assert parsed.tags == []
        assert parsed.text == text

    def test_pixel_coords_clamp_into_frame(self):
        parsed = tags_mod.parse_tags(
            "[POINT:99999,99999:far]", width=1280, height=720)
        assert parsed.tags[0].x == pytest.approx(1.0)
        assert parsed.tags[0].y == pytest.approx(1.0)

    def test_strip_tags(self):
        assert tags_mod.strip_tags(
            "hi [POINT:0.5,0.5:x] there [RECT:0,0,1,1:y]") == "hi  there "

    def test_tags_to_annotations_feed_the_spec_validator(self):
        parsed = tags_mod.parse_tags(
            "[POINT:640,360:the button] "
            "[RECT:0,0,640,360:the panel] "
            "[SCRIBBLE:10,10;20,30:the path]",
            width=1280, height=720)
        spec = validate_scene_spec({
            "kind": "overlay", "title": "round trip",
            "annotations": tags_mod.tags_to_annotations(parsed.tags),
        })
        assert spec["kind"] == "overlay"
        assert len(spec["annotations"]) == 3
        assert spec["annotations"][0]["kind"] == "cursor"
        assert spec["annotations"][1]["kind"] == "rect"
        assert spec["annotations"][1]["w"] == pytest.approx(0.5)
        assert spec["annotations"][2]["kind"] == "scribble"
        assert len(spec["annotations"][2]["points"]) == 2


# ── flight ───────────────────────────────────────────────────────────


class FlightTests(unittest.TestCase):
    def test_flight_is_deterministic(self):
        first = flight_mod.flight_frames((0.1, 0.1), (0.9, 0.8))
        second = flight_mod.flight_frames((0.1, 0.1), (0.9, 0.8))
        assert first == second

    def test_flight_starts_and_ends_exactly(self):
        frames = flight_mod.flight_frames((0.1, 0.2), (0.8, 0.7))
        assert frames[0]["x"] == pytest.approx(0.1)
        assert frames[0]["y"] == pytest.approx(0.2)
        assert frames[-1]["x"] == pytest.approx(0.8)
        assert frames[-1]["y"] == pytest.approx(0.7)
        assert frames[0]["t"] == pytest.approx(0.0)
        assert frames[-1]["t"] == pytest.approx(
            frames[-1]["t"])  # last frame carries the duration

    def test_flight_duration_scales_with_distance_and_clamps(self):
        short = flight_mod.flight_duration((0.5, 0.5), (0.51, 0.5))
        assert short == pytest.approx(0.6)  # floor
        long = flight_mod.flight_duration((0.0, 0.0), (1.0, 1.0))
        assert long == pytest.approx(1.4)  # ceiling
        mid = flight_mod.flight_duration((0.0, 0.0), (0.5, 0.0))
        assert 0.6 < mid < 1.4

    def test_flight_arcs_off_the_straight_line(self):
        frames = flight_mod.flight_frames((0.1, 0.5), (0.9, 0.5))
        mid = frames[len(frames) // 2]
        # The arc lifts the midpoint off the straight horizontal line.
        assert mid["y"] < 0.5
        assert mid["scale"] > 1.0  # scale pulse mid-flight
        assert frames[0]["scale"] == pytest.approx(1.0)
        assert frames[-1]["scale"] == pytest.approx(1.0)

    def test_flight_angle_follows_the_tangent(self):
        frames = flight_mod.flight_frames((0.1, 0.1), (0.9, 0.1))
        # At the top of the arc the tangent is horizontal: moving
        # right means 90deg (the glyph points up at 0deg).
        mid = frames[len(frames) // 2]
        assert mid["angle"] == pytest.approx(90.0, abs=2.0)
        # At launch the tangent tilts up into the arc.
        assert 70.0 < frames[1]["angle"] < 90.0

    def test_flight_frames_are_json_clean(self):
        frames = flight_mod.flight_frames((0.2, 0.2), (0.7, 0.6))
        text = json.dumps(frames)
        assert json.loads(text) == frames
        assert all(math.isfinite(f["x"]) and math.isfinite(f["y"])
                   for f in frames)


# ── annotation lifecycle ─────────────────────────────────────────────


class AnnotationLifecycleTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_create_update_complete_remove(self):
        board = ann_mod.AnnotationBoard(home=self.tmp_path)
        ann = board.create(kind="cursor", x=0.5, y=0.5, text="here")
        assert ann.state == "armed"
        assert ann.timeout_ms == 5000
        board.update(ann.id, text="there")
        assert board.get(ann.id).text == "there"
        board.complete(ann.id)
        assert board.get(ann.id).state == "completed"
        assert board.remove(ann.id) is True
        assert board.get(ann.id) is None
        assert board.remove(ann.id) is False

    def test_sweep_expires_armed_only(self):
        board = ann_mod.AnnotationBoard(home=self.tmp_path)
        old = board.create(kind="rect", x=0.1, y=0.1, timeout_ms=100)
        done = board.create(kind="cursor", x=0.2, y=0.2, timeout_ms=100)
        board.complete(done.id)
        fresh = board.create(kind="cursor", x=0.3, y=0.3,
                             timeout_ms=60000)
        expired = board.sweep(now=time.time() + 10)
        assert [a.id for a in expired] == [old.id]
        assert board.get(old.id).state == "missed"
        assert board.get(done.id).state == "completed"
        assert board.get(fresh.id).state == "armed"

    def test_update_unknown_id_raises(self):
        board = ann_mod.AnnotationBoard(home=self.tmp_path)
        with pytest.raises(KeyError):
            board.update("no-such-id", text="x")

    def test_board_is_durable_across_instances(self):
        first = ann_mod.AnnotationBoard(home=self.tmp_path)
        ann = first.create(kind="highlight", x=0.4, y=0.4, text="keep")
        second = ann_mod.AnnotationBoard(home=self.tmp_path)
        loaded = second.get(ann.id)
        assert loaded is not None
        assert loaded.text == "keep"
        assert loaded.kind == "highlight"

    def test_to_overlay_spec_only_live_annotations(self):
        board = ann_mod.AnnotationBoard(home=self.tmp_path)
        board.create(kind="cursor", x=0.5, y=0.5, text="live",
                     timeout_ms=60000)
        dead = board.create(kind="rect", x=0.1, y=0.1, timeout_ms=1)
        board.sweep(now=time.time() + 10)
        assert board.get(dead.id).state == "missed"
        spec = board.to_overlay_spec("live board")
        assert spec is not None
        assert len(spec["annotations"]) == 1
        assert spec["annotations"][0]["text"] == "live"

    def test_to_overlay_spec_empty_board_is_none(self):
        board = ann_mod.AnnotationBoard(home=self.tmp_path)
        assert board.to_overlay_spec("empty") is None

    def test_move_cursor_leaves_no_marker(self):
        board = ann_mod.AnnotationBoard(home=self.tmp_path)
        board.move_cursor(0.2, 0.8, 1)
        assert board.cursor == (0.2, 0.8, 1)
        assert board.list() == []
        reloaded = ann_mod.AnnotationBoard(home=self.tmp_path)
        assert reloaded.cursor == (0.2, 0.8, 1)

    def test_board_cursor_tracks_the_last_point(self):
        board = ann_mod.AnnotationBoard(home=self.tmp_path)
        assert board.cursor == (0.5, 0.5, 1)
        board.point(0.2, 0.8, label="there", screen=1)
        assert board.cursor == (0.2, 0.8, 1)
        reloaded = ann_mod.AnnotationBoard(home=self.tmp_path)
        assert reloaded.cursor == (0.2, 0.8, 1)


# ── interactive live page ────────────────────────────────────────────


class InteractivePageTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_page_carries_flight_and_annotations(self):
        spec = validate_scene_spec({
            "kind": "overlay", "title": "flight demo",
            "annotations": [{"kind": "cursor", "x": 0.5, "y": 0.5,
                             "text": "here"}],
        })
        frames = flight_mod.flight_frames((0.1, 0.1), (0.5, 0.5))
        html = inter_mod.render_interactive_page(
            spec, flight=frames, title="flight demo")
        assert "flight demo" in html
        assert "requestAnimationFrame" in html
        assert '"t": 0.0' in html or '"t":0.0' in html or "0.0" in html
        assert "here" in html
        path = inter_mod.write_interactive_page(
            spec, self.tmp_path / "page.html", flight=frames)
        assert Path(path).exists()
        assert Path(path).read_text(encoding="utf-8") == html

    def test_page_without_flight_still_renders(self):
        spec = validate_scene_spec({
            "kind": "diagram", "title": "plain",
            "nodes": [{"id": "a", "label": "a"}],
            "edges": [],
        })
        html = inter_mod.render_interactive_page(spec)
        assert "plain" in html
        assert "<canvas" in html

    def test_page_rejects_bad_spec(self):
        with pytest.raises(SceneSpecError):
            inter_mod.render_interactive_page({"kind": "nope"})


# ── bridge control plane ─────────────────────────────────────────────


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _call(port: int, method: str, path: str, body=None, token=None,
          headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        payload = exc.read() or b"{}"
        try:
            return exc.code, json.loads(payload)
        except ValueError:
            return exc.code, {"raw": payload.decode("utf-8", "replace")}


class BridgeTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_health_open_control_token_gated(self):
        port = _free_port()
        server = bridge_mod.start_bridge(home=self.tmp_path, port=port)
        try:
            status, body = _call(port, "GET", "/health")
            assert status == 200
            assert body["ok"] is True
            status, _ = _call(port, "GET", "/state")
            assert status == 401
            status, _ = _call(port, "POST", "/cursor",
                              {"x": 0.5, "y": 0.5}, token="wrong")
            assert status == 401
            status, _ = _call(port, "POST", "/cursor",
                              {"x": 0.5, "y": 0.5}, token="wrong")
            assert status == 401
        finally:
            bridge_mod.stop_bridge(server)

    def test_good_token_drives_cursor_and_annotations(self):
        port = _free_port()
        server = bridge_mod.start_bridge(home=self.tmp_path, port=port)
        try:
            token = bridge_mod.load_or_create_token(home=self.tmp_path)
            status, body = _call(port, "POST", "/cursor",
                                 {"x": 0.8, "y": 0.2, "label": "there"},
                                 token=token)
            assert status == 200
            assert body["ok"] is True
            assert body["flight"][0]["x"] == pytest.approx(0.5)
            assert body["flight"][-1]["x"] == pytest.approx(0.8)
            status, body = _call(port, "POST", "/annotations",
                                 {"kind": "rect", "x": 0.1, "y": 0.1,
                                  "w": 0.4, "h": 0.3, "text": "panel"},
                                 token=token)
            assert status == 200
            ann_id = body["annotation"]["id"]
            status, body = _call(port, "GET", "/annotations", token=token)
            assert status == 200
            assert any(a["id"] == ann_id for a in body["annotations"])
            status, body = _call(port, "DELETE",
                                 f"/annotations/{ann_id}", token=token)
            assert status == 200
            assert body["ok"] is True
        finally:
            bridge_mod.stop_bridge(server)

    def test_point_route_parses_tags_onto_the_board(self):
        port = _free_port()
        server = bridge_mod.start_bridge(home=self.tmp_path, port=port)
        try:
            token = bridge_mod.load_or_create_token(home=self.tmp_path)
            status, body = _call(
                port, "POST", "/point",
                {"text": "see [POINT:640,360:the button] here",
                 "width": 1280, "height": 720},
                token=token)
            assert status == 200
            assert body["text"] == "see  here"
            assert len(body["placed"]) == 1
            status, body = _call(port, "GET", "/state", token=token)
            assert len(body["annotations"]) == 1
            assert body["annotations"][0]["text"] == "the button"
        finally:
            bridge_mod.stop_bridge(server)

    def test_host_header_spoof_refused(self):
        import http.client

        port = _free_port()
        server = bridge_mod.start_bridge(home=self.tmp_path, port=port)
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port,
                                              timeout=5)
            conn.request("GET", "/health",
                         headers={"Host": "evil.example"})
            resp = conn.getresponse()
            assert resp.status == 403
            conn.close()
        finally:
            bridge_mod.stop_bridge(server)

    def test_token_file_created_once_and_stable(self):
        first = bridge_mod.load_or_create_token(home=self.tmp_path)
        second = bridge_mod.load_or_create_token(home=self.tmp_path)
        assert first == second
        assert len(first) >= 32
        token_file = bridge_mod.token_path(home=self.tmp_path)
        assert token_file.exists()
        import stat as _stat
        assert _stat.S_IMODE(token_file.stat().st_mode) == 0o600

    def test_rate_limit_blocks_a_flood(self):
        port = _free_port()
        server = bridge_mod.start_bridge(home=self.tmp_path, port=port)
        server.rate_limit = 3  # tiny limit, test-only
        try:
            for _ in range(3):
                status, _ = _call(port, "GET", "/health")
                assert status == 200
            status, _ = _call(port, "GET", "/health")
            assert status == 429
        finally:
            bridge_mod.stop_bridge(server)


# ── idle trigger ─────────────────────────────────────────────────────


class IdleTriggerTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_fires_after_threshold_only_if_acted(self):
        trigger = idle_mod.IdleTrigger(threshold_seconds=3.0)
        assert trigger.check(now=100.0) is False  # never acted: AFK
        trigger.note_activity(now=100.0)
        assert trigger.check(now=102.0) is False  # too soon
        assert trigger.check(now=103.5) is True

    def test_never_fires_twice_on_the_same_pause(self):
        trigger = idle_mod.IdleTrigger(threshold_seconds=3.0)
        trigger.note_activity(now=100.0)
        assert trigger.begin_observation(now=104.0) is True
        assert trigger.begin_observation(now=105.0) is False
        assert trigger.check(now=106.0) is False
        trigger.observation_complete()
        assert trigger.check(now=107.0) is False  # same pause, spent
        trigger.note_activity(now=108.0)
        assert trigger.check(now=112.0) is True  # new activity, new pause

    def test_new_activity_resets_the_clock(self):
        trigger = idle_mod.IdleTrigger(threshold_seconds=3.0)
        trigger.note_activity(now=100.0)
        trigger.note_activity(now=102.5)
        assert trigger.check(now=104.0) is False
        assert trigger.check(now=106.0) is True

    def test_disarmed_trigger_never_fires(self):
        trigger = idle_mod.IdleTrigger(threshold_seconds=3.0)
        trigger.disarm()
        trigger.note_activity(now=100.0)
        assert trigger.check(now=200.0) is False
        trigger.arm()
        assert trigger.check(now=200.0) is True

    def test_idle_tick_notifies_once_about_open_work(self):
        from nomorals.agents.workflow.store import WorkflowStore, default_workflows_path

        store = WorkflowStore(default_workflows_path(self.tmp_path))
        store.save_workflow(name="demo", task="do the thing",
                            script="print(1)")
        trigger = idle_mod.IdleTrigger(threshold_seconds=3.0)
        trigger.note_activity(now=100.0)
        notes = []

        class _Notifier:
            def publish(self, kind, title, body, **kwargs):
                notes.append((kind, title, body))

        result = idle_mod.run_visual_idle_tick_once(
            self.tmp_path, notifier=_Notifier(), trigger=trigger, now=104.0)
        assert result["ok"] is True
        assert result["fired"] is True
        assert len(notes) == 1
        assert notes[0][0] == "visual_idle"
        assert "demo" in notes[0][1] or "demo" in notes[0][2]
        # Second tick on the same pause: no second note.
        result = idle_mod.run_visual_idle_tick_once(
            self.tmp_path, notifier=_Notifier(), trigger=trigger, now=105.0)
        assert result["fired"] is False

    def test_idle_tick_silent_when_nothing_open(self):
        trigger = idle_mod.IdleTrigger(threshold_seconds=3.0)
        trigger.note_activity(now=100.0)
        notes = []

        class _Notifier:
            def publish(self, kind, title, body, **kwargs):
                notes.append((kind, title, body))

        result = idle_mod.run_visual_idle_tick_once(
            self.tmp_path, notifier=_Notifier(), trigger=trigger, now=104.0)
        assert result["ok"] is True
        assert result["fired"] is True
        assert notes == []


# ── overlay spec extension ───────────────────────────────────────────


class OverlaySpecExtensionTests(unittest.TestCase):
    def test_sized_annotations_validate(self):
        spec = validate_scene_spec({
            "kind": "overlay", "title": "sized",
            "annotations": [
                {"kind": "rect", "x": 0.1, "y": 0.1, "w": 0.4, "h": 0.3,
                 "text": "panel", "screen": 1},
                {"kind": "scribble", "x": 0.2, "y": 0.2,
                 "points": [[0.2, 0.2], [0.3, 0.4]], "text": "path"},
                {"kind": "shape", "x": 0.1, "y": 0.1, "shape": "arrow",
                 "points": [[0.1, 0.1], [0.5, 0.5]], "text": "flow"},
            ],
        })
        assert spec["annotations"][0]["w"] == pytest.approx(0.4)
        assert spec["annotations"][1]["points"][1][0] == pytest.approx(0.3)
        assert spec["annotations"][2]["shape"] == "arrow"

    def test_bad_size_or_shape_is_an_authoring_error(self):
        with pytest.raises(SceneSpecError):
            validate_scene_spec({
                "kind": "overlay",
                "annotations": [{"kind": "rect", "x": 0.1, "y": 0.1,
                                 "w": 1.5, "h": 0.3}],
            })
        with pytest.raises(SceneSpecError):
            validate_scene_spec({
                "kind": "overlay",
                "annotations": [{"kind": "shape", "x": 0.1, "y": 0.1,
                                 "shape": "zigzag"}],
            })
