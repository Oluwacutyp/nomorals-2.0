"""Kinect → point cloud → ZeroMQ ingestion for Devon.

The pipeline is storytold/cloudcam's ``main_zeromq.rs`` (pattern
only, no upstream code): poll a Kinect, turn each depth frame
into a point cloud, and push the cloud over ZeroMQ to a consumer.
Devon's version lives inside the mocap organ it already has
instead of beside it — the capture comes from
:mod:`nomorals.hardware.kinect`, the cloud is published into the
mocap store through the existing :class:`KinectProducer`, and the
wire is :mod:`nomorals.media.mocap.wire` (the clone's chunked
protocol, byte for byte).

One session is one source + one store + one socket, direct — no
hub, no shared queue. Two devices means two sessions, each
independent.

What the clone does that Devon does not have yet, and this module
adds:

* a **running session** — the clone polls in a loop with a
  messaging state machine (grab → BEGIN → CONTINUE); Devon only
  had one-shot ``publish_capture`` / ``publish_fixture``. The
  session polls on a worker thread at a target fps, paces
  best-effort, and counts every failure by name (capture
  timeouts, conversion errors, store refusals, ZeroMQ send
  errors) instead of printing and continuing like the clone;
* the **ZeroMQ publish side** of the CloudCam wire — Devon's
  repo had no point-cloud publisher at all (``voice/realtime.py``
  has REQ/REP only). The publisher binds ``tcp://*:8888`` by
  default, the address the clone's Unreal consumer connects to,
  and the ``media/cloudcam`` organ (B.20, unshipped at this
  base) consumes exactly this wire — the protocol module is the
  seam between the two;
* the clone's **culls and offsets** (``depth_cull`` /
  ``left_cull`` / ``right_cull`` / ``xoff`` / ``yoff`` /
  ``zoff``) as ingestion options, with the clone's exact
  semantics (documented where they bite — left/right are
  inverted because the clone's camera is inverted);
* the clone's **fake device** as a first-class source: recorded
  frames replayed at fps, so the whole pipeline proves itself
  with no hardware attached.

Honest limits, stated here:

* there is no Kinect hardware on this box and pyzmq is not in
  system python — the ``live`` source refuses by name when the
  SDK sees no device, and the ZeroMQ side refuses by name when
  pyzmq is missing (the same wording as
  ``voice/realtime.py::_require_zmq``);
* the conversion is pure Python over every valid pixel, so a
  full 640x576 frame takes seconds, not milliseconds — the
  session reports its achieved fps and never pretends to run at
  30 when it does not. The SDK transform rung
  (``Transformation.depth_to_point_cloud``) is the fast path for
  a live box; it is not wired in here;
* per-point colour follows the clone's depth→colour transform
  (unproject in depth space, transform to colour space, project,
  sample) when the colour frame's geometry differs from the
  depth frame's, and direct sampling when the geometries match.
  A colour frame that cannot be decoded (MJPG without Pillow)
  degrades to a uniform colour, and the frame receipt says so.
"""

from __future__ import annotations

import contextlib
import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...hardware.kinect import (
    Calibration,
    CalibrationType,
    Capture,
    ColorResolution,
    DepthMode,
    DeviceConfig,
    Fps,
    Frame,
    ImageFormat,
    KinectDevice,
    KinectError,
    parse_raw_calibration,
    point_cloud,
)
from .errors import MocapError
from .kinect_bridge import KinectProducer
from .state import MocapStateStore
from .wire import Point, encode_frame

__all__ = [
    "DEFAULT_FPS",
    "DEFAULT_ZMQ_ENDPOINT",
    "ConvertedFrame",
    "FileKinectSource",
    "IngestOptions",
    "IngestSession",
    "LiveKinectSource",
    "ZmqPublisher",
    "capture_to_points",
    "drop_session",
    "get_session",
    "list_sessions",
    "publish_one_frame",
]

#: The ZeroMQ endpoint the clone's consumer connects to.
DEFAULT_ZMQ_ENDPOINT: str = "tcp://*:8888"
#: Target frames per second for a running session.
DEFAULT_FPS: float = 30.0
#: Stride the store snapshot samples at (the bridge default).
DEFAULT_STRIDE: int = 16
#: The vendored fixture directory (the SDK's own unit-test
#: calibration plus a recorded NFOV depth frame).
_FIXTURES = (Path(__file__).resolve().parents[3]
             / "tests" / "fixtures" / "kinect")

#: Uniform colour for points with no colour sample (white, opaque).
_UNIFORM_COLOR: tuple[int, int, int, int] = (255, 255, 255, 255)


def _require_zmq(what: str) -> Any:
    """pyzmq on demand, refused by name when it is missing — the
    same wording as ``voice/realtime.py::_require_zmq``."""
    try:
        import zmq
    except ImportError as exc:
        raise MocapError(
            f"{what} needs pyzmq, which is not installed on this "
            "machine — pip install pyzmq (the ZeroMQ transport is "
            "the CloudCam backend; the store publish works "
            "without it)",
            kind="missing_pyzmq") from exc
    return zmq


@dataclass
class IngestOptions:
    """One session's shape — the clone's command-line args as
    data (``main_zeromq.rs``: num_cameras, wide, depth_cull,
    right_cull, left_cull, xoff, yoff, zoff), plus Devon's
    sinks."""

    fps: float = DEFAULT_FPS
    wide: bool = False
    depth_cull_mm: int = 0
    left_cull_mm: int = 0
    right_cull_mm: int = 0
    xoff_mm: int = 0
    yoff_mm: int = 0
    zoff_mm: int = 0
    stride: int = DEFAULT_STRIDE
    max_frames: int = 0  # 0 = run until stopped
    zmq_endpoint: str = DEFAULT_ZMQ_ENDPOINT
    zmq_mode: str = "bind"  # "bind" (server) or "connect"
    zmq_send_timeout_ms: int = 1000
    publish_store: bool = True
    publish_zmq: bool = True
    device_index: int = 0
    timeout_ms: int = 1000

    def validate(self) -> None:
        """Refuse bad options by name before anything starts."""
        if self.fps <= 0:
            raise MocapError(
                f"ingest fps must be positive, got {self.fps}",
                kind="bad_option")
        if self.stride < 1:
            raise MocapError(
                f"ingest stride must be >= 1, got {self.stride}",
                kind="bad_option")
        if self.max_frames < 0:
            raise MocapError(
                f"ingest max_frames must be >= 0, got "
                f"{self.max_frames}",
                kind="bad_option")
        if self.zmq_mode not in ("bind", "connect"):
            raise MocapError(
                f"ingest zmq_mode must be 'bind' or 'connect', got "
                f"{self.zmq_mode!r}",
                kind="bad_option")
        if self.zmq_send_timeout_ms < 0:
            raise MocapError(
                f"ingest zmq_send_timeout_ms must be >= 0, got "
                f"{self.zmq_send_timeout_ms}",
                kind="bad_option")
        if self.timeout_ms <= 0:
            raise MocapError(
                f"ingest timeout_ms must be positive, got "
                f"{self.timeout_ms}",
                kind="bad_option")
        if self.device_index < 0:
            raise MocapError(
                f"ingest device_index must be >= 0, got "
                f"{self.device_index}",
                kind="bad_option")


@dataclass
class ConvertedFrame:
    """One capture converted: the wire points plus the counts
    that prove what happened to every pixel."""

    points: list[Point]
    width: int
    height: int
    valid_pixels: int
    zero_pixels: int
    culled: int
    color: str  # "bgra" | "mjpg" | "uniform"
    color_detail: str = ""


def _color_grid(color: Frame) -> tuple[int, int, bytes, str] | None:
    """A colour frame as samplable bytes, or None when it cannot
    be decoded on this box. Returns (width, height, rgb_bytes,
    kind) with rgb_bytes in row-major RGB order.

    BGRA32 is read directly (stdlib). MJPG is decoded with
    Pillow when it is installed; without Pillow the frame is
    undecodable and the caller falls back to uniform colour.
    """
    if color.fmt is ImageFormat.COLOR_BGRA32:
        if len(color.data) < color.stride * color.height:
            return None
        rgb = bytearray(color.width * color.height * 3)
        for y in range(color.height):
            base = y * color.stride
            for x in range(color.width):
                off = base + x * 4
                b, g, r = color.data[off], color.data[off + 1], \
                    color.data[off + 2]
                dst = (y * color.width + x) * 3
                rgb[dst], rgb[dst + 1], rgb[dst + 2] = r, g, b
        return color.width, color.height, bytes(rgb), "bgra"
    if color.fmt is ImageFormat.COLOR_MJPG:
        try:
            from PIL import Image
        except ImportError:
            return None
        try:
            import io
            image = Image.open(io.BytesIO(color.data)).convert("RGB")
        except Exception:
            return None
        return image.width, image.height, image.tobytes(), "mjpg"
    return None


def capture_to_points(capture: Capture, calibration: Calibration,
                      options: IngestOptions) -> ConvertedFrame:
    """One capture → wire points, the clone's
    ``calculate_point_cloud3`` in Python.

    Zero-depth pixels are skipped (the clone skips them; Devon's
    ``point_cloud`` maps them to the origin, which is not a
    point). The culls then run on the unprojected millimetre
    coordinates, offsets last. Colour comes from the colour
    frame when it can be sampled — directly when its geometry
    matches the depth frame, through the calibration transform
    (depth space → colour space → project → sample) when it does
    not — and falls back to uniform white, named in the receipt,
    when there is no samplable colour frame.
    """
    if not capture.has_depth or capture.depth is None:
        raise MocapError(
            "capture has no depth frame to convert",
            kind="no_depth")
    depth = capture.depth
    rows = depth.depth_values()
    height = len(rows)
    width = len(rows[0]) if rows else 0
    zero_pixels = sum(1 for row in rows for d in row if d == 0)
    valid_pixels = width * height - zero_pixels
    cloud = point_cloud(calibration, rows)
    grid = None
    grid_kind = "uniform"
    grid_detail = "no colour frame on the capture"
    if capture.color is not None:
        grid = _color_grid(capture.color)
        if grid is not None:
            grid_kind = grid[3]
            grid_detail = f"{grid[3]} {grid[0]}x{grid[1]}"
        else:
            grid_detail = (
                f"colour frame {capture.color.fmt.name} "
                "could not be decoded on this box")
    points: list[Point] = []
    culled = 0
    for i, (x, y, z) in enumerate(cloud):
        if z == 0.0:
            continue
        px, py, pz = float(x), float(y), float(z)
        if options.depth_cull_mm != 0 and pz > options.depth_cull_mm:
            culled += 1
            continue
        if options.left_cull_mm != 0 and px > options.left_cull_mm:
            culled += 1
            continue
        if options.right_cull_mm != 0 and px < options.right_cull_mm:
            culled += 1
            continue
        r, g, b, a = _UNIFORM_COLOR
        if grid is not None:
            gw, gh, rgb, _kind = grid
            col = i % width
            row = i // width
            if gw == width and gh == height:
                dst = (row * gw + col) * 3
                r, g, b = rgb[dst], rgb[dst + 1], rgb[dst + 2]
            else:
                sample = _sample_color_via_transform(
                    calibration, (px, py, pz), grid)
                if sample is not None:
                    r, g, b = sample
        px += options.xoff_mm
        py += options.yoff_mm
        pz += options.zoff_mm
        points.append(Point(x=px, y=py, z=pz, r=r, g=g, b=b, a=a))
    return ConvertedFrame(
        points=points, width=width, height=height,
        valid_pixels=valid_pixels, zero_pixels=zero_pixels,
        culled=culled, color=grid_kind, color_detail=grid_detail)


def _sample_color_via_transform(
        calibration: Calibration,
        point_mm: tuple[float, float, float],
        grid: tuple[int, int, bytes, str]) -> tuple[int, int, int] | None:
    """Sample the colour grid for a depth-space point: transform
    to colour space, project to colour pixels, sample. Returns
    None when the point does not land on the colour frame."""
    gw, gh, rgb, _kind = grid
    try:
        in_color = calibration.transform_3d_to_3d(
            point_mm, CalibrationType.DEPTH, CalibrationType.COLOR)
        (u, v), valid = calibration.project_3d_to_2d(
            in_color, CalibrationType.COLOR)
    except (KinectError, ValueError, ArithmeticError):
        return None
    if not valid:
        return None
    cx, cy = int(u), int(v)
    if cx < 0 or cy < 0 or cx >= gw or cy >= gh:
        return None
    dst = (cy * gw + cx) * 3
    return rgb[dst], rgb[dst + 1], rgb[dst + 2]


# -- sources -------------------------------------------------------

class CaptureSource:
    """One capture at a time, on demand. A source owns its
    device or files and releases them on :meth:`close`."""

    name: str = "source"

    @property
    def calibration(self) -> Calibration:
        """The calibration the source's frames are measured in."""
        raise NotImplementedError

    def next_capture(self, timeout_ms: int) -> Capture | None:
        """The next capture, or None on timeout. A source that is
        exhausted (a finite recording) returns None forever."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the device or files (idempotent)."""


class LiveKinectSource(CaptureSource):
    """The live device, polled like the clone's ``main_zeromq``
    loop: open → start cameras → ``get_capture`` with a timeout.

    The configuration is the clone's: NFOV unbinned depth at 30
    fps, BGRA32 colour at 720p — or WFOV unbinned at 15 fps when
    ``wide`` is set (wide FOV lowers the possible frame rate, as
    the clone notes). No device or no SDK is refused by name at
    open time, never at first capture.
    """

    name = "live"

    def __init__(self, options: IngestOptions) -> None:
        self._options = options
        self._device = KinectDevice.open(options.device_index)
        depth_mode = (DepthMode.WFOV_UNBINNED if options.wide
                      else DepthMode.NFOV_UNBINNED)
        fps = Fps.FPS_15 if options.wide else Fps.FPS_30
        config = DeviceConfig(
            color_format=ImageFormat.COLOR_BGRA32,
            color_resolution=ColorResolution.RES_720P,
            depth_mode=depth_mode,
            camera_fps=fps)
        self._depth_mode = depth_mode
        self._device.start_cameras(config)
        self._calibration = self._device.calibration(
            depth_mode, ColorResolution.RES_720P)
        self._closed = False

    @property
    def calibration(self) -> Calibration:
        return self._calibration

    def next_capture(self, timeout_ms: int) -> Capture | None:
        if self._closed:
            raise MocapError("live source is closed",
                             kind="source_closed")
        return self._device.get_capture(timeout_ms)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._device.close()


class FileKinectSource(CaptureSource):
    """Recorded frames replayed at fps — the clone's
    ``FakeDeviceCaptureProvider`` as a real source: the same
    recorded depth frame(s) on every tick, so the whole pipeline
    proves itself with no hardware attached.

    Layout (the vendored fixture's): ``calibration.json`` plus
    ``depth_nfov_640x576.raw`` (one 640x576 DEPTH16 frame), or
    ``depth_frame_*.raw`` frames cycled in name order. A
    ``color_mjpg.jpg`` beside them is carried as the colour
    frame. A missing file or an unreadable calibration is
    refused by name at open time. Geometry comes from a
    ``depth_geometry.json`` sidecar (``{"width": N, "height":
    N}``) when one is present, else from the file size against
    the Kinect's known depth geometries (640x576, 512x512,
    1024x1024, 320x288) — a size that matches none of them is
    refused by name.
    """

    #: Known Kinect depth geometries, by DEPTH16 byte count.
    _KNOWN_GEOMETRIES = {
        640 * 576 * 2: (640, 576),
        512 * 512 * 2: (512, 512),
        1024 * 1024 * 2: (1024, 1024),
        320 * 288 * 2: (320, 288),
    }

    name = "file"

    def __init__(self, directory: str | Path) -> None:
        root = Path(directory).expanduser()
        calib_path = root / "calibration.json"
        if not calib_path.exists():
            raise MocapError(
                f"no Kinect calibration fixture at {calib_path}",
                kind="missing_fixture")
        try:
            self._calibration = parse_raw_calibration(
                calib_path.read_text(encoding="utf-8"))
        except (ValueError, KinectError) as exc:
            raise MocapError(
                f"fixture calibration is unreadable: {exc}",
                kind="bad_fixture") from exc
        frames = sorted(root.glob("depth_frame_*.raw"))
        if not frames:
            single = root / "depth_nfov_640x576.raw"
            if not single.exists():
                raise MocapError(
                    f"no Kinect depth fixture at {single}",
                    kind="missing_fixture")
            frames = [single]
        self._directory = root
        self._frames = frames
        self._index = 0
        self._depth: Frame | None = None
        self._color: Frame | None = None
        self._geometry = self._read_geometry_sidecar(root)
        self._load_frame(0)
        self._closed = False

    @staticmethod
    def _read_geometry_sidecar(
            root: Path) -> tuple[int, int] | None:
        path = root / "depth_geometry.json"
        if not path.exists():
            return None
        try:
            import json
            raw = json.loads(path.read_text(encoding="utf-8"))
            width, height = int(raw["width"]), int(raw["height"])
        except (ValueError, KeyError, TypeError) as exc:
            raise MocapError(
                f"depth_geometry.json is unreadable: {exc}",
                kind="bad_fixture") from exc
        if width < 1 or height < 1:
            raise MocapError(
                "depth_geometry.json needs positive width/height",
                kind="bad_fixture")
        return width, height

    def _geometry_for(self, raw: bytes) -> tuple[int, int]:
        if self._geometry is not None:
            width, height = self._geometry
            if width * height * 2 != len(raw):
                raise MocapError(
                    f"depth_geometry.json says {width}x{height} "
                    f"({width * height * 2} bytes) but the frame "
                    f"carries {len(raw)} bytes",
                    kind="bad_fixture")
            return width, height
        try:
            return self._KNOWN_GEOMETRIES[len(raw)]
        except KeyError as exc:
            raise MocapError(
                f"cannot infer depth geometry from {len(raw)} "
                "bytes — no known Kinect depth frame is that size "
                "(want a depth_geometry.json sidecar)",
                kind="unknown_geometry") from exc

    @property
    def calibration(self) -> Calibration:
        return self._calibration

    @property
    def directory(self) -> Path:
        """Where this source's frames live."""
        return self._directory

    def _load_frame(self, index: int) -> None:
        path = self._frames[index % len(self._frames)]
        raw = path.read_bytes()
        width, height = self._geometry_for(raw)
        self._depth = Frame(
            fmt=ImageFormat.DEPTH16, width=width, height=height,
            stride=width * 2, data=raw)
        color_path = self._directory / "color_mjpg.jpg"
        self._color = None
        if color_path.exists():
            data = color_path.read_bytes()
            width_c, height_c = self._color_geometry(data)
            self._color = Frame(
                fmt=ImageFormat.COLOR_MJPG, width=width_c,
                height=height_c, stride=0, data=data)

    @staticmethod
    def _color_geometry(data: bytes) -> tuple[int, int]:
        """MJPG geometry via Pillow when it is there; (0, 0)
        when it is not — the colour frame is then undecodable
        and the conversion falls back to uniform colour."""
        try:
            from PIL import Image
            with Image.open(io.BytesIO(data)) as image:
                return image.width, image.height
        except Exception:
            return 0, 0

    def next_capture(self, timeout_ms: int) -> Capture | None:
        if self._closed:
            raise MocapError("file source is closed",
                             kind="source_closed")
        self._load_frame(self._index)
        self._index += 1
        return Capture(color=self._color, depth=self._depth,
                       temperature_c=0.0)

    def close(self) -> None:
        self._closed = True


def open_source(name: str, options: IngestOptions, *,
                source_dir: str = "") -> CaptureSource:
    """A source by name: ``live`` (the device), ``fixture`` (the
    vendored recorded frame), ``file`` (a directory of recorded
    frames — ``source_dir`` required). Anything else is refused
    by name."""
    key = (name or "").strip().lower()
    if key == "live":
        return LiveKinectSource(options)
    if key == "fixture":
        return FileKinectSource(source_dir or str(_FIXTURES))
    if key == "file":
        if not (source_dir or "").strip():
            raise MocapError(
                "the file source needs a directory of recorded "
                "frames (sources=[dir])",
                kind="missing_source_dir")
        return FileKinectSource(source_dir)
    raise MocapError(
        f"unknown ingest source {name!r} (want live, fixture or "
        "file)",
        kind="unknown_source")


# -- ZeroMQ publisher ----------------------------------------------

class ZmqPublisher:
    """The CloudCam wire over a ZeroMQ PUSH socket.

    The socket binds by default (the server side, like the
    clone's Unreal consumer expects) or connects when
    ``mode="connect"``. pyzmq is imported lazily — without it
    the constructor refuses by name, and the session keeps
    publishing to the store. ``send_timeout_ms`` bounds every
    send: a consumer that is absent or too slow turns the frame
    into a counted drop, never a wedged worker — a bare PUSH
    socket blocks forever on ``send`` with no peer connected.
    """

    def __init__(self, endpoint: str = DEFAULT_ZMQ_ENDPOINT,
                 *, mode: str = "bind",
                 send_timeout_ms: int = 1000) -> None:
        if mode not in ("bind", "connect"):
            raise MocapError(
                f"zmq mode must be 'bind' or 'connect', got "
                f"{mode!r}",
                kind="bad_option")
        if send_timeout_ms < 0:
            raise MocapError(
                f"zmq send_timeout_ms must be >= 0, got "
                f"{send_timeout_ms}",
                kind="bad_option")
        zmq = _require_zmq("the CloudCam publisher")
        self.endpoint = endpoint
        self.mode = mode
        self.send_timeout_ms = send_timeout_ms
        self._zmq = zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUSH)
        self._sock.setsockopt(zmq.SNDTIMEO, send_timeout_ms)
        if mode == "bind":
            self._sock.bind(endpoint)
            self.last_endpoint = self._sock.getsockopt(
                zmq.LAST_ENDPOINT).decode("utf-8", "replace")
        else:
            self._sock.connect(endpoint)
            self.last_endpoint = endpoint
        self._closed = False
        self.messages_sent = 0
        self.points_sent = 0
        self.bytes_sent = 0

    def send_frame(self, points: list[Point]) -> dict[str, Any]:
        """Publish one frame: BEGIN + CONTINUEs, chunked at 3000
        points. A send failure raises :class:`MocapError`
        (kind ``zmq_send``) naming the endpoint — the caller
        counts it, the publisher stays usable."""
        if self._closed:
            raise MocapError("ZeroMQ publisher is closed",
                             kind="zmq_closed")
        try:
            messages = encode_frame(points)
            total = 0
            for message in messages:
                self._sock.send(message)
                total += len(message)
        except Exception as exc:
            raise MocapError(
                f"ZeroMQ send to {self.endpoint} failed: {exc}",
                kind="zmq_send") from exc
        self.messages_sent += 1
        self.points_sent += len(points)
        self.bytes_sent += total
        return {"messages": len(messages),
                "points": len(points), "bytes": total}

    def close(self) -> None:
        """Close the socket and context (idempotent). Linger is
        zero: a session that stops with an unconsumed queue must
        never wedge the process waiting for a consumer that is
        gone."""
        if not self._closed:
            self._closed = True
            with contextlib.suppress(Exception):
                self._sock.setsockopt(self._zmq.LINGER, 0)
            try:
                self._sock.close()
            finally:
                self._ctx.term()


# -- session --------------------------------------------------------

@dataclass
class _Counters:
    frames: int = 0
    points_total: int = 0
    points_last: int = 0
    timeouts: int = 0
    conversion_errors: int = 0
    store_errors: int = 0
    zmq_errors: int = 0


class IngestSession:
    """One running ingestion: one source, polled on a worker
    thread, each frame converted and published to the mocap
    store and/or over ZeroMQ. Every failure is counted by name
    in :meth:`status` — nothing is printed and continued like
    the clone, and nothing is silently dropped."""

    def __init__(self, source: CaptureSource,
                 options: IngestOptions, *,
                 store: MocapStateStore | None = None,
                 name: str = "default") -> None:
        options.validate()
        if options.publish_store and store is None:
            raise MocapError(
                "ingest with publish_store needs a mocap store",
                kind="no_store")
        self.name = name
        self.source = source
        self.options = options
        self._store = store
        self._producer = (KinectProducer(store)
                          if store is not None else None)
        self._publisher: ZmqPublisher | None = None
        self._counters = _Counters()
        self._lock = threading.Lock()
        self._state = "idle"
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started_at: float | None = None
        self._last_error: str = ""
        self._last_receipt: dict[str, Any] = {}

    # -- one frame, synchronous ------------------------------------

    def tick(self) -> dict[str, Any]:
        """Poll once, convert, publish. The receipt names the
        frame, the point counts, and every drop. A tick on a
        running session is refused — the worker owns the source
        while it runs."""
        if self._state == "running":
            raise MocapError(
                f"ingest session {self.name!r} is running — stop "
                "it before ticking by hand",
                kind="running")
        return self._tick_once()

    def _tick_once(self) -> dict[str, Any]:
        """One poll → convert → publish, no state guard — the
        worker thread's path (and the one :meth:`tick` delegates
        to)."""
        started = time.monotonic()
        try:
            capture = self.source.next_capture(
                self.options.timeout_ms)
        except (KinectError, MocapError) as exc:
            return self._drop("timeouts", f"capture failed: {exc}",
                              started)
        if capture is None:
            return self._drop(
                "timeouts",
                f"no capture within {self.options.timeout_ms} ms",
                started)
        try:
            converted = capture_to_points(
                capture, self.source.calibration, self.options)
        except (KinectError, MocapError) as exc:
            return self._drop("conversion_errors",
                              f"conversion failed: {exc}", started)
        receipt: dict[str, Any] = {
            "frame": self._counters.frames + 1,
            "points": len(converted.points),
            "valid_pixels": converted.valid_pixels,
            "zero_pixels": converted.zero_pixels,
            "culled": converted.culled,
            "color": converted.color,
            "color_detail": converted.color_detail,
            "store": None,
            "zmq": None,
        }
        if self.options.publish_store and self._producer is not None:
            try:
                receipt["store"] = self._producer.publish_capture(
                    capture, self.source.calibration,
                    stride=self.options.stride)
            except (KinectError, MocapError) as exc:
                with self._lock:
                    self._counters.store_errors += 1
                    self._last_error = f"store publish failed: {exc}"
                receipt["store_error"] = str(exc)
        if self.options.publish_zmq:
            try:
                publisher = self._publisher_for_send()
                receipt["zmq"] = publisher.send_frame(
                    converted.points)
            except MocapError as exc:
                with self._lock:
                    self._counters.zmq_errors += 1
                    self._last_error = str(exc)
                receipt["zmq_error"] = str(exc)
        with self._lock:
            self._counters.frames += 1
            self._counters.points_total += len(converted.points)
            self._counters.points_last = len(converted.points)
            self._last_receipt = dict(receipt)
        receipt["duration_s"] = round(time.monotonic() - started, 3)
        return receipt

    def _drop(self, kind: str, reason: str,
              started: float) -> dict[str, Any]:
        with self._lock:
            setattr(self._counters, kind,
                    getattr(self._counters, kind) + 1)
            self._last_error = reason
        return {"frame": None, "dropped": kind, "reason": reason,
                "duration_s": round(time.monotonic() - started, 3)}

    def _publisher_for_send(self) -> ZmqPublisher:
        if self._publisher is None:
            self._publisher = ZmqPublisher(
                self.options.zmq_endpoint, mode=self.options.zmq_mode,
                send_timeout_ms=self.options.zmq_send_timeout_ms)
        return self._publisher

    # -- running ----------------------------------------------------

    def start(self) -> dict[str, Any]:
        """Start polling on a worker thread. A session already
        running is refused by name."""
        if self._state == "running":
            raise MocapError(
                f"ingest session {self.name!r} is already running",
                kind="already_running")
        self.options.validate()
        self._stop.clear()
        self._started_at = time.monotonic()
        self._state = "running"
        self._thread = threading.Thread(
            target=self._loop, name=f"ingest-{self.name}",
            daemon=True)
        self._thread.start()
        return self.status()

    def _loop(self) -> None:
        interval = 1.0 / self.options.fps
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._tick_once()
            except MocapError as exc:
                with self._lock:
                    self._last_error = str(exc)
            except Exception as exc:  # noqa: BLE001 - worker never dies
                # Unexpected: the tick's own guards cover the known
                # failures, so this is a bug surfacing — record it
                # and keep polling instead of killing the thread.
                with self._lock:
                    self._last_error = (
                        f"unexpected ingest error: {exc!r}")
            if (self.options.max_frames
                    and self._counters.frames
                    >= self.options.max_frames):
                break
            wait = interval - (time.monotonic() - started)
            if wait > 0:
                self._stop.wait(wait)
        with self._lock:
            if self._state == "running":
                self._state = "stopped"

    def stop(self) -> dict[str, Any]:
        """Stop the worker (idempotent), close the socket and the
        source, and report the final status."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        if self._publisher is not None:
            self._publisher.close()
            self._publisher = None
        with contextlib.suppress(KinectError, MocapError):
            self.source.close()
        with self._lock:
            if self._state == "running":
                self._state = "stopped"
        return self.status()

    def run(self, frames: int) -> list[dict[str, Any]]:
        """``frames`` synchronous ticks — the testable path and
        the CLI's one-shot shape. Refused while the session is
        running."""
        if frames < 1:
            raise MocapError(
                f"ingest run needs at least 1 frame, got {frames}",
                kind="bad_option")
        return [self.tick() for _ in range(frames)]

    # -- status ------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """The session as it stands: state, frames, points,
        every drop counted by name, the last error, and the
        achieved frame rate (the target fps is a wish; the
        achieved one is the truth)."""
        with self._lock:
            counters = self._counters
            elapsed = (time.monotonic() - self._started_at
                       if self._started_at is not None else 0.0)
            return {
                "name": self.name,
                "state": self._state,
                "source": self.source.name,
                "frames": counters.frames,
                "points_total": counters.points_total,
                "points_last": counters.points_last,
                "drops": {
                    "timeouts": counters.timeouts,
                    "conversion_errors": counters.conversion_errors,
                    "store_errors": counters.store_errors,
                    "zmq_errors": counters.zmq_errors,
                },
                "last_error": self._last_error,
                "last_receipt": dict(self._last_receipt),
                "fps_target": self.options.fps,
                "fps_achieved": (
                    round(counters.frames / elapsed, 2)
                    if elapsed > 0 and counters.frames else 0.0),
                "zmq_endpoint": self.options.zmq_endpoint,
                "zmq_mode": self.options.zmq_mode,
                "publish_store": self.options.publish_store,
                "publish_zmq": self.options.publish_zmq,
            }


# -- session registry -----------------------------------------------

_SESSIONS: dict[str, IngestSession] = {}
_SESSIONS_LOCK = threading.Lock()


def open_session(name: str, source: CaptureSource,
                 options: IngestOptions, *,
                 store: MocapStateStore | None = None) -> IngestSession:
    """Create a session under ``name``. A live session with the
    same name is refused — stop and drop it first."""
    with _SESSIONS_LOCK:
        if name in _SESSIONS:
            raise MocapError(
                f"ingest session {name!r} already exists — stop "
                "and drop it first",
                kind="duplicate_session")
        session = IngestSession(source, options, store=store, name=name)
        _SESSIONS[name] = session
        return session


def get_session(name: str) -> IngestSession:
    """A live session by name, or a refusal naming it."""
    with _SESSIONS_LOCK:
        try:
            return _SESSIONS[name]
        except KeyError as exc:
            raise MocapError(
                f"no ingest session {name!r}",
                kind="unknown_session") from exc


def list_sessions() -> list[dict[str, Any]]:
    """Every live session's status, in creation order."""
    with _SESSIONS_LOCK:
        return [session.status() for session in _SESSIONS.values()]


def drop_session(name: str) -> dict[str, Any]:
    """Stop a session and forget it. Unknown names are refused."""
    session = get_session(name)
    status = session.stop()
    with _SESSIONS_LOCK:
        _SESSIONS.pop(name, None)
    return {"ok": True, "name": name, "status": status}


def publish_one_frame(store: MocapStateStore | None,
                      source_name: str, options: IngestOptions, *,
                      source_dir: str = "") -> dict[str, Any]:
    """One frame from a source, no session: open the source,
    convert, publish to the store and/or ZeroMQ, close. The
    one-shot path the tool and the CLI share."""
    source = open_source(source_name, options, source_dir=source_dir)
    try:
        session = IngestSession(source, options, store=store)
        return session.tick()
    finally:
        source.close()
