"""Cross-engine mocap state for Devon.

One shared store every engine reads and writes (``state``), a router
that picks the engine for a category and fails honestly when it is
not configured (``router``), BVH read/write for the mocap
interchange format game engines read (``bvh``), bridges from
Devon's engines into the store (``kinect_bridge``), and the
Kinect → point cloud → ZeroMQ ingestion pipeline (``ingest``
over the clone's chunked wire in ``wire``) — storytold/cloudcam's
pattern, no upstream code (see ``SURVEY.md`` in
``~/workspace/cloudcam-ingest-verify/``).

The pattern is artcraftx's cross-engine tasks table + provider
router, re-expressed for live mocap state — pattern only, no
upstream code (see ``MOCAP_STATE_SURVEY.md`` in
``~/workspace/mocap-state-verify/``).
"""

from __future__ import annotations

from .bvh import (
    Joint,
    Skeleton,
    SkeletonFrame,
    hierarchy_joints,
    parse_bvh,
    skeleton_from_dict,
    skeleton_to_dict,
    write_bvh,
)
from .errors import (
    ChannelOwnershipError,
    EngineNotConfigured,
    MocapError,
    StaleChannelError,
    UnknownChannelError,
    UnknownEngineError,
    UnknownSubscriberError,
)
from .ingest import (
    DEFAULT_FPS,
    DEFAULT_ZMQ_ENDPOINT,
    ConvertedFrame,
    FileKinectSource,
    IngestOptions,
    IngestSession,
    LiveKinectSource,
    ZmqPublisher,
    capture_to_points,
    drop_session,
    get_session,
    list_sessions,
    open_session,
    open_source,
    publish_one_frame,
)
from .kinect_bridge import (
    BvhExporter,
    BvhPlaybackProducer,
    KinectProducer,
    load_bvh_json,
)
from .router import MocapRouter, default_router
from .state import MocapStateStore
from .wire import (
    DATA_LENGTH_COMMAND,
    MAX_SEND_POINTS_PER_PACKET,
    POINT_DATA_BEGIN_PAYLOAD_COMMAND,
    POINT_DATA_CONTINUE_PAYLOAD_COMMAND,
    FrameDecoder,
    Point,
    WireError,
    decode_frame,
    encode_data_length,
    encode_frame,
)

__all__ = [
    "BvhExporter",
    "BvhPlaybackProducer",
    "ChannelOwnershipError",
    "ConvertedFrame",
    "DATA_LENGTH_COMMAND",
    "DEFAULT_FPS",
    "DEFAULT_ZMQ_ENDPOINT",
    "EngineNotConfigured",
    "FileKinectSource",
    "FrameDecoder",
    "IngestOptions",
    "IngestSession",
    "Joint",
    "KinectProducer",
    "LiveKinectSource",
    "MAX_SEND_POINTS_PER_PACKET",
    "MocapError",
    "MocapRouter",
    "MocapStateStore",
    "POINT_DATA_BEGIN_PAYLOAD_COMMAND",
    "POINT_DATA_CONTINUE_PAYLOAD_COMMAND",
    "Point",
    "Skeleton",
    "SkeletonFrame",
    "StaleChannelError",
    "UnknownChannelError",
    "UnknownEngineError",
    "UnknownSubscriberError",
    "WireError",
    "ZmqPublisher",
    "capture_to_points",
    "decode_frame",
    "default_router",
    "drop_session",
    "encode_data_length",
    "encode_frame",
    "get_session",
    "hierarchy_joints",
    "list_sessions",
    "load_bvh_json",
    "open_session",
    "open_source",
    "parse_bvh",
    "publish_one_frame",
    "skeleton_from_dict",
    "skeleton_to_dict",
    "write_bvh",
]
