"""Streamed point clouds — Devon's own CloudCam organ.

Pattern from storytold/cloudcamplugin (PATTERN ONLY, no code
copied): a worker thread polls a frame source into a ring
buffer, and each tick stages the latest frame into position +
colour texture buffers a renderer consumes. Sources: ZeroMQ
(the clone's CloudCam wire), the mocap store's pointcloud
channel (Kinect depth, no device needed), a JSON/PLY file, or
any callable. Each session is independent and direct — no
central hub, no shared queue.
"""

from __future__ import annotations

from .errors import CloudCamError
from .protocol import (
    COMMAND_APPEND,
    COMMAND_BEGIN,
    CloudCamPoint,
    decode_data_packet,
    decode_length_packet,
    encode_data_packet,
    encode_length_packet,
)
from .registry import (
    get_session,
    list_sessions,
    load_session,
    reset_sessions,
)
from .render import render_projection, write_ply
from .ring import PointCloudFrame, RingBuffer
from .session import CloudCamSession
from .sources import (
    CallableSource,
    FileSource,
    FrameSource,
    StoreSource,
    ZmqSource,
    depth_shade,
)
from .stage import (
    StagedFrame,
    frame_bounds,
    placeholder_grid,
    stage_frame,
)

__all__ = [
    "COMMAND_APPEND",
    "COMMAND_BEGIN",
    "CallableSource",
    "CloudCamError",
    "CloudCamPoint",
    "CloudCamSession",
    "FileSource",
    "FrameSource",
    "PointCloudFrame",
    "RingBuffer",
    "StagedFrame",
    "StoreSource",
    "ZmqSource",
    "decode_data_packet",
    "decode_length_packet",
    "depth_shade",
    "encode_data_packet",
    "encode_length_packet",
    "frame_bounds",
    "get_session",
    "list_sessions",
    "load_session",
    "placeholder_grid",
    "render_projection",
    "reset_sessions",
    "stage_frame",
    "write_ply",
]
