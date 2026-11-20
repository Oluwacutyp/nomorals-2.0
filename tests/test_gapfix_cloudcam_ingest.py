"""Lane B.21 — census §3B.21: Devon's own mocap ingestion pipeline
(Kinect → point cloud → ZeroMQ), storytold/cloudcam pattern only.

Proves: the clone's chunked wire byte-for-byte (commands 1/2/3,
3000 points per packet, 16-byte little-endian points), the
decoder counting malformed input instead of raising or
swallowing it, conversion of a capture (zero-depth skipped,
culls and offsets applied, colour sampled or honestly uniform),
the session's frame receipts and drop counters, the ZeroMQ
publish round-tripping real points, named refusals for every
source that cannot run here (no SDK, no pyzmq, unknown source,
missing fixture), and the tool + CLI + chat wiring — with the
tool count pinned at 776 (B.20's cloudcam tool, a583b718).
"""

from __future__ import annotations

import contextlib
import io
import json
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.partner.runtime_mocap import RuntimeMocapMixin
from nomorals.cmdline.commands.mocap import _cmd_mocap
from nomorals.cmdline.parser import _parser
from nomorals.hardware.kinect import (
    Calibration,
    Capture,
    Frame,
    ImageFormat,
    KinectError,
    NoDeviceError,
    NoSdkError,
    parse_raw_calibration,
)
from nomorals.media.mocap import (
    DATA_LENGTH_COMMAND,
    MAX_SEND_POINTS_PER_PACKET,
    POINT_DATA_BEGIN_PAYLOAD_COMMAND,
    POINT_DATA_CONTINUE_PAYLOAD_COMMAND,
)
from nomorals.media.mocap.errors import MocapError
from nomorals.media.mocap.ingest import (
    CaptureSource,
    FileKinectSource,
    IngestOptions,
    IngestSession,
    ZmqPublisher,
    capture_to_points,
    drop_session,
    get_session,
    list_sessions,
    open_session,
    open_source,
    publish_one_frame,
)
from nomorals.media.mocap.state import MocapStateStore
from nomorals.media.mocap.wire import (
    FrameDecoder,
    Point,
    WireError,
    decode_frame,
    encode_data_length,
    encode_frame,
)
from nomorals.tools.registry import ToolRegistry

try:
    import zmq
    _ZMQ_AVAILABLE = True
except ImportError:
    zmq = None  # type: ignore[assignment]
    _ZMQ_AVAILABLE = False

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "kinect"

# A 4x3 depth grid: two zero pixels (skipped), the rest real.
_TINY_ROWS = [
    [1000, 0, 1200, 1100],
    [900, 950, 0, 1050],
    [1000, 1000, 1000, 1000],
]


def _fixture_calibration() -> Calibration:
    return parse_raw_calibration(
        (FIXTURES / "calibration.json").read_text(encoding="utf-8"))


def _tiny_capture(with_color: str = "") -> Capture:
    depth = Frame(fmt=ImageFormat.DEPTH16, width=4, height=3,
                  stride=8,
                  data=b"".join(struct.pack("<H", v)
                                for row in _TINY_ROWS for v in row))
    color = None
    if with_color == "bgra":
        # One distinct colour per pixel: (x*40, y*80, 200, 255).
        data = bytearray()
        for y in range(3):
            for x in range(4):
                data += bytes([200, y * 80, x * 40, 255])
        color = Frame(fmt=ImageFormat.COLOR_BGRA32, width=4,
                      height=3, stride=16, data=bytes(data))
    elif with_color == "mjpg":
        from PIL import Image
        image = Image.new("RGB", (2, 2), (10, 20, 30))
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        color = Frame(fmt=ImageFormat.COLOR_MJPG, width=2, height=2,
                      stride=0, data=buf.getvalue())
    return Capture(color=color, depth=depth, temperature_c=0.0)


class _TinySource(CaptureSource):
    """A source that returns the same tiny capture forever."""

    name = "tiny"

    def __init__(self, *, with_color: str = "",
                 capture: Capture | None = None) -> None:
        self._capture = capture if capture is not None \
            else _tiny_capture(with_color)
        self._calibration = _fixture_calibration()
        self._closed = False

    @property
    def calibration(self) -> Calibration:
        return self._calibration

    def next_capture(self, timeout_ms: int) -> Capture | None:
        if self._closed:
            raise MocapError("tiny source is closed",
                             kind="source_closed")
        return self._capture

    def close(self) -> None:
        self._closed = True


class _NoneSource(CaptureSource):
    """A source that always times out."""

    name = "none"

    def __init__(self) -> None:
        self._calibration = _fixture_calibration()

    @property
    def calibration(self) -> Calibration:
        return self._calibration

    def next_capture(self, timeout_ms: int) -> Capture | None:
        return None

    def close(self) -> None:
        pass


def _drop_all_sessions() -> None:
    for entry in list_sessions():
        with contextlib.suppress(MocapError, KinectError):
            drop_session(entry["name"])


def _points(n: int) -> list[Point]:
    return [Point(x=float(i), y=float(-i), z=float(i) * 0.5,
                  r=i % 256, g=(i * 2) % 256, b=(i * 3) % 256,
                  a=255)
            for i in range(n)]


def _tiny_fixture_dir(tmp: Path) -> Path:
    """A recorded-frames directory with one tiny depth frame."""
    root = tmp / "tiny-kinect"
    root.mkdir()
    (root / "calibration.json").write_text(
        (FIXTURES / "calibration.json").read_text(encoding="utf-8"),
        encoding="utf-8")
    (root / "depth_geometry.json").write_text(
        json.dumps({"width": 4, "height": 3}), encoding="utf-8")
    (root / "depth_frame_0.raw").write_bytes(
        b"".join(struct.pack("<H", v)
                 for row in _TINY_ROWS for v in row))
    return root


class WireTest(unittest.TestCase):
    def test_protocol_constants_match_the_clone(self):
        # main_zeromq.rs: DATA_LENGTH = 1, POINT_DATA_BEGIN = 2,
        # POINT_DATA_CONTINUE = 3, max 3000 points per packet.
        self.assertEqual(DATA_LENGTH_COMMAND, 1)
        self.assertEqual(POINT_DATA_BEGIN_PAYLOAD_COMMAND, 2)
        self.assertEqual(POINT_DATA_CONTINUE_PAYLOAD_COMMAND, 3)
        self.assertEqual(MAX_SEND_POINTS_PER_PACKET, 3000)

    def test_empty_frame_is_a_bare_begin(self):
        # One BEGIN packet carrying zero points: the 8-byte
        # header (command 2, count 0), no point bytes.
        self.assertEqual(encode_frame([]),
                         [struct.pack("<II", 2, 0)])

    def test_single_packet_roundtrip(self):
        pts = _points(5)
        messages = encode_frame(pts)
        self.assertEqual(len(messages), 1)
        self.assertEqual(decode_frame(messages), pts)

    def test_encode_matches_clone_bytes(self):
        # One point, hand-packed the way the clone's
        # send_point_data does: an 8-byte header (command u32,
        # count u32, both little-endian) + 3xf32 LE + 4xu8.
        point = Point(x=1.5, y=-2.5, z=3.5, r=10, g=20, b=30, a=255)
        want = (struct.pack("<II", 2, 1)
                + struct.pack("<fffBBBB", 1.5, -2.5, 3.5,
                              10, 20, 30, 255))
        self.assertEqual(encode_frame([point]), [want])
        self.assertEqual(decode_frame([want]), [point])

    def test_encode_chunks_at_3000(self):
        pts = _points(3001)
        messages = encode_frame(pts)
        self.assertEqual(len(messages), 2)
        self.assertEqual(struct.unpack_from("<I", messages[0], 0)[0],
                         2)  # BEGIN
        self.assertEqual(struct.unpack_from("<I", messages[1], 0)[0],
                         3)  # CONTINUE
        self.assertEqual(len(messages[0]), 8 + 3000 * 16)
        self.assertEqual(len(messages[1]), 8 + 1 * 16)
        self.assertEqual(decode_frame(messages), pts)
        self.assertEqual(len(encode_frame(_points(6000))), 2)
        self.assertEqual(len(encode_frame(_points(6001))), 3)

    def test_decode_checks_length_preamble(self):
        pts = _points(4)
        messages = [encode_data_length(4)] + encode_frame(pts)
        self.assertEqual(decode_frame(messages), pts)
        # A preamble that disagrees with the payload is refused
        # by name — the frame is not half-returned.
        messages = [encode_data_length(999)] + encode_frame(pts)
        with self.assertRaises(WireError):
            decode_frame(messages)

    def test_decode_malformed_counts_never_raises(self):
        dec = FrameDecoder()
        self.assertEqual(dec.feed(b"\x02\x00\x00"), [])  # truncated
        self.assertEqual(dec.feed(b"\x09\x00\x00"), [])  # bad cmd
        self.assertEqual(dec.feed(b"\x02" + b"\x00" * 5), [])
        self.assertEqual(dec.feed(b"\x03" + b"\x00" * 16), [])
        self.assertGreaterEqual(len(dec.errors), 4)
        # Still usable after the garbage.
        pts = _points(3)
        out: list[list[Point]] = []
        for message in encode_frame(pts):
            out.extend(dec.feed(message))
        self.assertEqual(out, [pts])

    def test_decoder_splits_two_frames(self):
        dec = FrameDecoder()
        out: list[list[Point]] = []
        for message in (encode_frame(_points(2))
                        + encode_frame(_points(3))):
            out.extend(dec.feed(message))
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]), 2)
        self.assertEqual(len(out[1]), 3)

    def test_decoder_short_packet_completes_frame(self):
        dec = FrameDecoder()
        out = dec.feed(encode_frame(_points(2))[0])
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 2)

    def test_decoder_full_packet_waits_for_continue(self):
        dec = FrameDecoder()
        self.assertEqual(
            dec.feed(encode_frame(_points(3000))[0]), [])
        # A bare CONTINUE of 1 point finishes the frame.
        cont = (struct.pack("<II", 3, 1)
                + _points(1)[0].to_bytes())
        out = dec.feed(cont)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 3001)

    def test_point_float32_precision(self):
        point = Point(x=1234.5678, y=-0.001, z=98765.4321,
                      r=1, g=2, b=3, a=4)
        back = decode_frame(encode_frame([point]))[0]
        self.assertAlmostEqual(back.x, point.x, places=3)
        self.assertAlmostEqual(back.z, point.z, places=0)

    def test_wire_decodes_under_cloudcam_protocol(self):
        # The seam to B.20 (shipped a583b718): this lane's
        # chunked stream must drive media/cloudcam's poller
        # unchanged — its ZmqSource treats command 2 as "open a
        # frame" and anything else as "append", over the same
        # 8-byte header + 16-byte points.
        from nomorals.media.cloudcam.protocol import (
            COMMAND_BEGIN,
            decode_data_packet,
            encode_data_packet,
        )
        pts = _points(3001)
        frames: list[list[Point]] = []
        current: list[Point] = []

        def as_points(chunk):
            return [Point(x=p.x, y=p.y, z=p.z, r=p.r, g=p.g, b=p.b,
                          a=p.a)
                    for p in chunk]

        for message in encode_frame(pts):
            command, chunk = decode_data_packet(message)
            self.assertEqual(len(chunk),
                             struct.unpack_from("<I", message, 4)[0])
            if command == COMMAND_BEGIN:
                if current:
                    frames.append(current)
                current = as_points(chunk)
            else:
                current.extend(as_points(chunk))
        if current:
            frames.append(current)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0], pts)
        # And the reverse: one packet the cloudcam side encodes
        # decodes as one frame on this side.
        packet = encode_data_packet(
            [Point(x=1.0, y=2.0, z=3.0, r=4, g=5, b=6, a=7)])
        self.assertEqual(decode_frame([packet]),
                         [Point(x=1.0, y=2.0, z=3.0, r=4, g=5,
                                b=6, a=7)])


class ConversionTest(unittest.TestCase):
    def test_zero_depth_skipped_and_counted(self):
        converted = capture_to_points(
            _tiny_capture(), _fixture_calibration(), IngestOptions())
        self.assertEqual(converted.zero_pixels, 2)
        self.assertEqual(converted.valid_pixels, 10)
        self.assertEqual(len(converted.points), 10)
        self.assertEqual(converted.culled, 0)
        self.assertTrue(all(p.z != 0.0 for p in converted.points))

    def test_depth_cull_drops_far_points(self):
        options = IngestOptions(depth_cull_mm=1000)
        converted = capture_to_points(
            _tiny_capture(), _fixture_calibration(), options)
        # Depths over 1000: 1200, 1100, 1050 — three culled.
        self.assertEqual(converted.culled, 3)
        self.assertEqual(len(converted.points), 7)
        self.assertTrue(all(p.z <= 1000 for p in converted.points))

    def test_offsets_shift_every_point(self):
        base = capture_to_points(
            _tiny_capture(), _fixture_calibration(), IngestOptions())
        shifted = capture_to_points(
            _tiny_capture(), _fixture_calibration(),
            IngestOptions(xoff_mm=10, yoff_mm=-20, zoff_mm=30))
        self.assertEqual(len(shifted.points), len(base.points))
        for a, b in zip(base.points, shifted.points, strict=True):
            self.assertAlmostEqual(b.x - a.x, 10.0)
            self.assertAlmostEqual(b.y - a.y, -20.0)
            self.assertAlmostEqual(b.z - a.z, 30.0)

    def test_left_right_cull_direction(self):
        # The clone's semantics, inverted because its camera is
        # inverted: left_cull drops x > left_cull, right_cull
        # drops x < right_cull.
        none = IngestOptions(left_cull_mm=100000)
        self.assertEqual(len(capture_to_points(
            _tiny_capture(), _fixture_calibration(), none).points), 10)
        all_left = IngestOptions(left_cull_mm=-100000)
        self.assertEqual(len(capture_to_points(
            _tiny_capture(), _fixture_calibration(),
            all_left).points), 0)
        none_right = IngestOptions(right_cull_mm=-100000)
        self.assertEqual(len(capture_to_points(
            _tiny_capture(), _fixture_calibration(),
            none_right).points), 10)

    def test_no_depth_refused_by_name(self):
        capture = Capture(color=None, depth=None, temperature_c=0.0)
        with self.assertRaises(MocapError) as ctx:
            capture_to_points(capture, _fixture_calibration(),
                              IngestOptions())
        self.assertEqual(ctx.exception.kind, "no_depth")

    def test_uniform_color_without_color_frame(self):
        converted = capture_to_points(
            _tiny_capture(), _fixture_calibration(), IngestOptions())
        self.assertEqual(converted.color, "uniform")
        self.assertIn("no colour frame", converted.color_detail)
        self.assertTrue(all((p.r, p.g, p.b, p.a)
                            == (255, 255, 255, 255)
                            for p in converted.points))

    def test_bgra_direct_sampling(self):
        converted = capture_to_points(
            _tiny_capture("bgra"), _fixture_calibration(),
            IngestOptions())
        self.assertEqual(converted.color, "bgra")
        # Pixel (0, 0) is BGRA (200, 0, 0, 255) -> RGB (0, 0, 200).
        first = converted.points[0]
        self.assertEqual((first.r, first.g, first.b), (0, 0, 200))
        # Pixel (3, 0): BGRA (200, 0, 120, 255) -> RGB (120, 0, 200).
        third = [p for p in converted.points
                 if p.r == 120 and p.b == 200]
        self.assertTrue(third)

    def test_mjpg_transform_path(self):
        converted = capture_to_points(
            _tiny_capture("mjpg"), _fixture_calibration(),
            IngestOptions())
        # Colour geometry (2x2) differs from depth (4x3), so the
        # calibration transform path runs, not direct sampling.
        self.assertEqual(converted.color, "mjpg")
        self.assertEqual(len(converted.points), 10)

    def test_options_validation(self):
        bad = [IngestOptions(fps=0), IngestOptions(fps=-1),
               IngestOptions(stride=0), IngestOptions(max_frames=-1),
               IngestOptions(zmq_mode="sideways"),
               IngestOptions(timeout_ms=0),
               IngestOptions(device_index=-1)]
        for options in bad:
            with self.assertRaises(MocapError) as ctx:
                options.validate()
            self.assertEqual(ctx.exception.kind, "bad_option")


class SourceTest(unittest.TestCase):
    def test_fixture_source_loads(self):
        source = FileKinectSource(str(FIXTURES))
        try:
            self.assertIsNotNone(source.calibration)
            capture = source.next_capture(100)
            assert capture is not None
            self.assertTrue(capture.has_depth)
            self.assertTrue(capture.has_color)
            assert capture.depth is not None
            self.assertEqual((capture.depth.width,
                              capture.depth.height), (640, 576))
        finally:
            source.close()

    def test_fixture_source_missing_dir_refused(self):
        with self.assertRaises(MocapError) as ctx:
            FileKinectSource("/nonexistent/kinect-fixtures")
        self.assertEqual(ctx.exception.kind, "missing_fixture")

    def test_fixture_source_bad_calibration_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "calibration.json").write_text(
                "not json at all", encoding="utf-8")
            (root / "depth_nfov_640x576.raw").write_bytes(b"\x00" * 8)
            with self.assertRaises(MocapError) as ctx:
                FileKinectSource(str(root))
            self.assertEqual(ctx.exception.kind, "bad_fixture")

    def test_file_source_unknown_geometry_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "calibration.json").write_text(
                (FIXTURES / "calibration.json").read_text(
                    encoding="utf-8"), encoding="utf-8")
            (root / "depth_frame_0.raw").write_bytes(b"\x00" * 100)
            with self.assertRaises(MocapError) as ctx:
                FileKinectSource(str(root))
            self.assertEqual(ctx.exception.kind, "unknown_geometry")

    def test_file_source_sidecar_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "calibration.json").write_text(
                (FIXTURES / "calibration.json").read_text(
                    encoding="utf-8"), encoding="utf-8")
            (root / "depth_geometry.json").write_text(
                json.dumps({"width": 4, "height": 3}),
                encoding="utf-8")
            (root / "depth_frame_0.raw").write_bytes(b"\x00" * 8)
            with self.assertRaises(MocapError) as ctx:
                FileKinectSource(str(root))
            self.assertEqual(ctx.exception.kind, "bad_fixture")

    def test_open_source_unknown_refused(self):
        with self.assertRaises(MocapError) as ctx:
            open_source("sideways", IngestOptions())
        self.assertEqual(ctx.exception.kind, "unknown_source")

    def test_open_source_file_needs_dir(self):
        with self.assertRaises(MocapError) as ctx:
            open_source("file", IngestOptions())
        self.assertEqual(ctx.exception.kind, "missing_source_dir")

    def test_live_source_no_device_refused(self):
        # No libk4a on this box: the refusal names the SDK.
        with self.assertRaises((NoSdkError, NoDeviceError)) as ctx:
            publish_one_frame(MocapStateStore(), "live",
                              IngestOptions(publish_zmq=False))
        self.assertIn("libk4a", str(ctx.exception))


class SessionTest(unittest.TestCase):
    def setUp(self):
        _drop_all_sessions()

    def tearDown(self):
        _drop_all_sessions()

    def test_session_tick_receipt_and_store(self):
        store = MocapStateStore()
        session = IngestSession(
            _TinySource(), IngestOptions(publish_zmq=False),
            store=store, name="tick-test")
        first = session.tick()
        self.assertEqual(first["frame"], 1)
        self.assertEqual(first["points"], 10)
        self.assertEqual(first["color"], "uniform")
        self.assertIsNotNone(first["store"])
        self.assertEqual(first["store"]["channel"], "pointcloud")
        second = session.tick()
        self.assertEqual(second["frame"], 2)
        status = session.status()
        self.assertEqual(status["frames"], 2)
        self.assertEqual(status["points_total"], 20)
        self.assertEqual(status["drops"], {
            "timeouts": 0, "conversion_errors": 0,
            "store_errors": 0, "zmq_errors": 0})

    def test_session_timeout_drop_counted(self):
        session = IngestSession(
            _NoneSource(), IngestOptions(publish_zmq=False),
            store=MocapStateStore(), name="timeout-test")
        receipt = session.tick()
        self.assertIsNone(receipt.get("frame"))
        self.assertEqual(receipt["dropped"], "timeouts")
        status = session.status()
        self.assertEqual(status["drops"]["timeouts"], 1)
        self.assertIn("no capture", status["last_error"])

    def test_session_without_store_refused(self):
        with self.assertRaises(MocapError) as ctx:
            IngestSession(_TinySource(), IngestOptions(),
                          store=None, name="no-store")
        self.assertEqual(ctx.exception.kind, "no_store")

    def test_session_registry(self):
        session = open_session(
            "reg-test", _TinySource(), IngestOptions(publish_zmq=False),
            store=MocapStateStore())
        self.assertIs(get_session("reg-test"), session)
        self.assertEqual([s["name"] for s in list_sessions()],
                         ["reg-test"])
        with self.assertRaises(MocapError) as ctx:
            open_session("reg-test", _TinySource(),
                         IngestOptions(publish_zmq=False),
                         store=MocapStateStore())
        self.assertEqual(ctx.exception.kind, "duplicate_session")
        with self.assertRaises(MocapError) as ctx:
            get_session("no-such-session")
        self.assertEqual(ctx.exception.kind, "unknown_session")
        dropped = drop_session("reg-test")
        self.assertTrue(dropped["ok"])
        self.assertEqual(list_sessions(), [])
        with self.assertRaises(MocapError) as ctx:
            drop_session("reg-test")
        self.assertEqual(ctx.exception.kind, "unknown_session")

    def test_session_threaded_start_stop(self):
        session = open_session(
            "thread-test", _TinySource(),
            IngestOptions(publish_zmq=False, fps=60.0, max_frames=3),
            store=MocapStateStore())
        started = session.start()
        self.assertEqual(started["state"], "running")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if session.status()["state"] == "stopped":
                break
            time.sleep(0.05)
        status = session.status()
        self.assertEqual(status["state"], "stopped")
        self.assertEqual(status["frames"], 3)
        # stop() again is idempotent.
        self.assertEqual(session.stop()["state"], "stopped")

    def test_tick_while_running_refused(self):
        session = open_session(
            "running-test", _TinySource(),
            IngestOptions(publish_zmq=False, fps=60.0),
            store=MocapStateStore())
        session.start()
        try:
            with self.assertRaises(MocapError) as ctx:
                session.tick()
            self.assertEqual(ctx.exception.kind, "running")
        finally:
            session.stop()

    def test_zmq_failure_counted_store_still_published(self):
        # pyzmq blocked: the ZeroMQ side fails by name, the
        # store publish still lands, and the drop is counted.
        import unittest.mock as mock
        store = MocapStateStore()
        session = IngestSession(
            _TinySource(), IngestOptions(publish_zmq=True),
            store=store, name="zmq-fail-test")
        with mock.patch.dict(sys.modules, {"zmq": None}):
            receipt = session.tick()
        self.assertEqual(receipt["frame"], 1)
        self.assertEqual(receipt["points"], 10)
        self.assertIsNotNone(receipt["store"])
        self.assertIn("pyzmq", receipt["zmq_error"])
        self.assertEqual(session.status()["drops"]["zmq_errors"], 1)


@unittest.skipUnless(_ZMQ_AVAILABLE, "pyzmq not installed")
class ZmqTest(unittest.TestCase):
    def setUp(self):
        _drop_all_sessions()

    def tearDown(self):
        _drop_all_sessions()

    def test_zmq_publisher_refuses_without_pyzmq(self):
        import unittest.mock as mock
        with mock.patch.dict(sys.modules, {"zmq": None}), \
                self.assertRaises(MocapError) as ctx:
            ZmqPublisher("tcp://127.0.0.1:18991")
        self.assertEqual(ctx.exception.kind, "missing_pyzmq")
        self.assertIn("pyzmq", str(ctx.exception))

    def test_zmq_publish_roundtrip(self):
        assert zmq is not None
        publisher = ZmqPublisher("tcp://127.0.0.1:*")
        ctx = zmq.Context()
        pull = ctx.socket(zmq.PULL)
        pull.connect(publisher.last_endpoint)
        pull.setsockopt(zmq.RCVTIMEO, 5000)
        try:
            pts = _points(3001)
            sent = publisher.send_frame(pts)
            self.assertEqual(sent["messages"], 2)
            self.assertEqual(sent["points"], 3001)
            dec = FrameDecoder()
            frames: list[list[Point]] = []
            while not frames:
                frames.extend(dec.feed(pull.recv()))
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0], pts)
        finally:
            pull.close()
            ctx.term()
            publisher.close()

    def test_zmq_send_without_consumer_times_out(self):
        # A bare PUSH send blocks forever with no peer — the
        # publisher bounds it, and the failure names the
        # endpoint instead of wedging the caller.
        assert zmq is not None
        publisher = ZmqPublisher("tcp://127.0.0.1:*",
                                 send_timeout_ms=200)
        started = time.monotonic()
        try:
            with self.assertRaises(MocapError) as ctx:
                publisher.send_frame(_points(3))
            self.assertEqual(ctx.exception.kind, "zmq_send")
            self.assertLess(time.monotonic() - started, 10.0)
        finally:
            publisher.close()

    def test_session_tick_publishes_over_zmq(self):
        assert zmq is not None
        # The PULL binds first and the session's publisher
        # connects to it — a peer is present before the first
        # send, so the send never blocks on a missing consumer.
        ctx = zmq.Context()
        pull = ctx.socket(zmq.PULL)
        pull.bind("tcp://127.0.0.1:18993")
        pull.setsockopt(zmq.RCVTIMEO, 5000)
        store = MocapStateStore()
        session = IngestSession(
            _TinySource(),
            IngestOptions(publish_store=True, publish_zmq=True,
                          zmq_endpoint="tcp://127.0.0.1:18993",
                          zmq_mode="connect"),
            store=store, name="zmq-tick-test")
        try:
            receipt = session.tick()
            dec = FrameDecoder()
            frames: list[list[Point]] = []
            while not frames:
                frames.extend(dec.feed(pull.recv()))
            self.assertEqual(len(frames[0]), receipt["points"])
            self.assertEqual(receipt["zmq"]["points"],
                             receipt["points"])
        finally:
            pull.close()
            ctx.term()
            session.stop()


class FixtureEndToEndTest(unittest.TestCase):
    def test_publish_one_frame_fixture_full(self):
        # The real recorded frame, end to end: conversion,
        # store snapshot, and the wire. Slow on purpose — pure
        # Python over 640x576 plus the colour transform.
        store = MocapStateStore()
        receipt = publish_one_frame(
            store, "fixture",
            IngestOptions(publish_store=True, publish_zmq=False))
        self.assertEqual(receipt["frame"], 1)
        self.assertEqual(receipt["points"], 368540)
        self.assertEqual(receipt["valid_pixels"], 368540)
        self.assertEqual(receipt["zero_pixels"], 100)
        self.assertEqual(receipt["culled"], 0)
        self.assertEqual(receipt["color"], "mjpg")
        self.assertIsNotNone(receipt["store"])
        self.assertEqual(receipt["store"]["points_total"], 368640)
        self.assertEqual(receipt["store"]["points_sampled"], 1440)

    @unittest.skipUnless(_ZMQ_AVAILABLE, "pyzmq not installed")
    def test_fixture_frame_zmq_roundtrip(self):
        assert zmq is not None
        store = MocapStateStore()
        ctx = zmq.Context()
        pull = ctx.socket(zmq.PULL)
        pull.connect("tcp://127.0.0.1:18898")
        pull.setsockopt(zmq.RCVTIMEO, 60000)
        try:
            receipt = publish_one_frame(
                store, "fixture",
                IngestOptions(publish_store=False, publish_zmq=True,
                              zmq_endpoint="tcp://127.0.0.1:18898"))
            self.assertEqual(receipt["points"], 368540)
            self.assertEqual(receipt["zmq"]["messages"], 123)
            dec = FrameDecoder()
            frames: list[list[Point]] = []
            while sum(len(f) for f in frames) < 368540:
                frames.extend(dec.feed(pull.recv()))
            self.assertEqual(sum(len(f) for f in frames), 368540)
            self.assertEqual(len(frames), 1)
        finally:
            pull.close()
            ctx.term()


class WiringTest(unittest.TestCase):
    def setUp(self):
        _drop_all_sessions()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixture_dir = _tiny_fixture_dir(Path(self._tmp.name))

    def tearDown(self):
        _drop_all_sessions()

    def _registry(self):
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        return registry

    def _call(self, **kwargs):
        outcome = self._registry().call("mocap", **kwargs)
        if getattr(outcome, "ok", False):
            return outcome.value
        raise AssertionError(
            f"mocap call failed: {getattr(outcome, 'error', outcome)}")

    def test_tool_count_is_779_after_scene_project(self):
        # 778 after B.26 shipped the scene_project tool; 779 after B.23 added facemcap.
        # (777 after B.22 shipped the show tool on 147c5b5e;
        # 776 after B.20 shipped cloudcam): no new tool here — the
        # ingest actions ride the existing mocap tool, so the
        # registry count moves past B.22's 777 only by B.26's
        # scene_project.
        registry = self._registry()
        self.assertEqual(len(registry._tools), 779)
        self.assertIn("mocap", registry._tools)
        self.assertIn("cloudcam", registry._tools)

    def test_tool_ingest_publish(self):
        out = self._call(
            action="ingest_publish",
            sources=[str(self.fixture_dir)],
            params={"source": "file", "publish_zmq": False})
        self.assertEqual(out["frame"], 1)
        self.assertEqual(out["points"], 10)
        self.assertEqual(out["color"], "uniform")

    def test_tool_ingest_start_status_stop(self):
        started = self._call(
            action="ingest_start", node="tool-session",
            sources=[str(self.fixture_dir)],
            params={"source": "file", "publish_zmq": False,
                    "fps": 60.0})
        self.assertEqual(started["state"], "running")
        listed = self._call(action="ingest_status")
        self.assertEqual([s["name"] for s in listed["sessions"]],
                         ["tool-session"])
        one = self._call(action="ingest_status", node="tool-session")
        self.assertEqual(one["name"], "tool-session")
        stopped = self._call(action="ingest_stop",
                             node="tool-session")
        self.assertTrue(stopped["ok"])

    def test_tool_ingest_tick(self):
        # Start a session that stops itself after one frame, then
        # tick it by hand through the tool — the worker owns the
        # source only while it runs.
        self._call(
            action="ingest_start", node="tick-session",
            sources=[str(self.fixture_dir)],
            params={"source": "file", "publish_zmq": False,
                    "fps": 60.0, "max_frames": 1})
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self._call(action="ingest_status",
                          node="tick-session")["state"] == "stopped":
                break
            time.sleep(0.05)
        out = self._call(action="ingest_tick", node="tick-session",
                         params={"frames": 2})
        self.assertEqual(len(out["receipts"]), 2)
        self.assertEqual(out["receipts"][0]["frame"], 2)
        self.assertEqual(out["receipts"][1]["frame"], 3)
        self._call(action="ingest_stop", node="tick-session")

    def test_tool_ingest_unknown_key_refused(self):
        outcome = self._registry().call(
            "mocap", action="ingest_publish",
            params={"source": "fixture", "bogus": 1})
        self.assertFalse(getattr(outcome, "ok", True))
        self.assertIn("unknown keys",
                      str(getattr(outcome, "error", "")))

    def test_tool_ingest_unknown_action_refused(self):
        outcome = self._registry().call(
            "mocap", action="ingest_frobnicate")
        self.assertFalse(getattr(outcome, "ok", True))

    def test_cli_ingest_publish_parses_and_runs(self):
        args = _parser().parse_args(
            ["mocap", "ingest", "publish", "--source", "file",
             "--dir", str(self.fixture_dir), "--no-zmq"])
        self.assertEqual(args.mocap_ingest_verb, "publish")
        self.assertEqual(_cmd_mocap(args, None), 0)

    def test_cli_ingest_start_status_stop(self):
        args = _parser().parse_args(
            ["mocap", "ingest", "start", "--source", "file",
             "--dir", str(self.fixture_dir), "--no-zmq",
             "--session", "cli-session", "--fps", "60",
             "--max-frames", "2"])
        self.assertEqual(_cmd_mocap(args, None), 0)
        args = _parser().parse_args(
            ["mocap", "ingest", "status", "--session", "cli-session"])
        self.assertEqual(_cmd_mocap(args, None), 0)
        args = _parser().parse_args(
            ["mocap", "ingest", "status"])
        self.assertEqual(_cmd_mocap(args, None), 0)
        args = _parser().parse_args(
            ["mocap", "ingest", "stop", "--session", "cli-session"])
        self.assertEqual(_cmd_mocap(args, None), 0)
        self.assertEqual(list_sessions(), [])

    def test_cli_ingest_unknown_session_status_fails(self):
        args = _parser().parse_args(
            ["mocap", "ingest", "status", "--session", "nope"])
        self.assertEqual(_cmd_mocap(args, None), 1)

    def test_chat_ingest_help_lists_verbs(self):
        mixin = RuntimeMocapMixin()
        help_text = mixin._control_mocap("", "chat-key")
        self.assertIn("ingest start", help_text)
        self.assertIn("ingest publish", help_text)

    def test_chat_ingest_status_empty(self):
        mixin = RuntimeMocapMixin()
        self.assertEqual(
            mixin._control_mocap("ingest status", "chat-key"),
            "no ingest sessions running")

    def test_chat_ingest_file_without_dir_refused_in_words(self):
        mixin = RuntimeMocapMixin()
        reply = mixin._control_mocap("ingest publish file", "chat-key")
        self.assertIn("ingest:", reply)
        self.assertIn("directory", reply)

    def test_chat_ingest_live_without_device_refused_in_words(self):
        mixin = RuntimeMocapMixin()
        reply = mixin._control_mocap("ingest publish live", "chat-key")
        self.assertIn("ingest:", reply)
        self.assertIn("libk4a", reply)

    def test_chat_ingest_stop_unknown_refused_in_words(self):
        mixin = RuntimeMocapMixin()
        reply = mixin._control_mocap("ingest stop nope", "chat-key")
        self.assertIn("ingest:", reply)
        self.assertIn("no ingest session 'nope'", reply)


if __name__ == "__main__":
    unittest.main()
