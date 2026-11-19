"""Mocap: cross-engine motion-capture state (artcraftx pattern port).

A thin surface over :mod:`nomorals.media.mocap` — the organ owns the
shared state store, the engine router, the BVH codec and the Kinect
bridge; the tool only routes arguments and shapes the reply.

The store is process-level shared state: every engine in this
process reads and writes the same store, and it lives as long as the
process does. ``save``/``load`` persist it as JSON for durability
across restarts. Writes output files, so FS_WRITE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..media.mocap import (
    BvhExporter,
    KinectProducer,
    MocapStateStore,
    default_router,
    load_bvh_json,
)

__all__ = ["register"]

_ACTIONS = ("status", "engines", "route", "register", "publish", "read",
            "snapshot", "claim", "release", "subscribe", "poll",
            "unsubscribe", "bvh_write", "bvh_read", "kinect_fixture",
            "kinect_status", "save", "load")

_STORE = MocapStateStore()
_ROUTER = default_router()
_EXPORTER = BvhExporter()

_FIXTURES = (Path(__file__).resolve().parents[2]
             / "tests" / "fixtures" / "kinect")


def _params(params: dict | None) -> dict:
    return dict(params or {})


def _json_arg(value: Any, *, what: str) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except ValueError as exc:
        raise ValueError(f"{what} must be JSON: {exc}") from exc


def register(registry: Any) -> None:
    """Attach the mocap family to a registry."""
    @registry.register(
        "mocap",
        description=(
            "Cross-engine motion-capture state — one shared store "
            "every engine reads and writes, with ownership, "
            "staleness flags and subscriber polls (artcraftx "
            "cross-engine pattern, pattern only). Publish Kinect "
            "depth/point-cloud frames or BVH skeleton takes, read "
            "channels fresh or stale-flagged, route a category to "
            "its engine, export a skeleton take as BVH for game "
            "engines. Mocap actions: status=<store + router state>; "
            "engines=<list the router's engines>; route=<pick the "
            "engine for a category, value=<category>>; register=<an "
            "engine joins the store, node=<engine>, params=<{produces, "
            "consumes, note}>>; publish=<write "
            "a channel, node=<channel>, value=<engine>, params="
            "<{payload, kind, max_age_s}>>; read=<read a channel, "
            "node=<channel>, params=<{since_seq, require_fresh, "
            "max_age_s}>>; snapshot=<late-joiner view of every "
            "channel>; claim=<take write ownership, node=<channel>, "
            "value=<engine>>; release=<give it up, node=<channel>, "
            "value=<engine>>; subscribe=<watch channels, node="
            "<subscriber id>, params=<{channels, payload}>>; poll=<"
            "changes since last poll, node=<subscriber id>>; "
            "unsubscribe=<stop watching, node=<subscriber id>>; "
            "bvh_write=<write a BVH file from a take, sources=[take "
            "JSON], output=<path.bvh>>; bvh_read=<parse a BVH file, "
            "sources=[path.bvh]>; kinect_fixture=<publish the "
            "vendored Kinect fixture as depth + pointcloud, sources="
            "[fixtures dir]>; kinect_status=<probe the SDK/device>; "
            "save=<persist the store, output=<path>>; load=<restore "
            "a saved store, sources=[path]>. action=<status|engines|"
            "route|register|publish|read|snapshot|claim|release|"
            "subscribe|poll|unsubscribe|bvh_write|bvh_read|"
            "kinect_fixture|kinect_status|save|load>; sources=<list "
            "of paths>; output=<path to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def mocap_tool(action: str = "status",
                   sources: list[str] | None = None,
                   output: str = "", kind: str = "",
                   node: str = "", value: Any = None,
                   command: str = "",
                   params: dict | None = None,
                   title: str = "",
                   ) -> dict[str, Any]:
        global _STORE
        action = (action or "status").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown mocap action {action!r}; "
                             f"use one of {', '.join(_ACTIONS)}")
        args = _params(params)
        sources = list(sources or [])
        if action == "status":
            return {
                "channels": _STORE.channels(),
                "engines": _STORE.engines(),
                "subscribers": _STORE.subscribers(),
                "router_categories": _ROUTER.categories(),
            }
        if action == "engines":
            return {"engines": _ROUTER.engines()}
        if action == "route":
            category = str(value or "").strip()
            if not category:
                raise ValueError("route needs value=<category>")
            return _ROUTER.route(category)
        if action == "register":
            engine = (node or "").strip()
            if not engine:
                raise ValueError("register needs node=<engine>")
            return _STORE.register_engine(
                engine,
                produces=tuple(args.get("produces") or ()),
                consumes=tuple(args.get("consumes") or ()),
                configured=bool(args.get("configured", True)),
                note=str(args.get("note", "")))
        if action == "publish":
            channel = (node or "").strip()
            engine = str(value or "").strip()
            if not channel or not engine:
                raise ValueError(
                    "publish needs node=<channel> and value=<engine>")
            payload = args.get("payload")
            if payload is None:
                raise ValueError("publish needs params.payload")
            return _STORE.write(
                channel, engine, payload,
                kind=str(args.get("kind", "snapshot")),
                max_age_s=args.get("max_age_s"))
        if action == "read":
            channel = (node or "").strip()
            if not channel:
                raise ValueError("read needs node=<channel>")
            return _STORE.read(
                channel,
                since_seq=args.get("since_seq"),
                require_fresh=bool(args.get("require_fresh", False)),
                max_age_s=args.get("max_age_s"))
        if action == "snapshot":
            return _STORE.snapshot()
        if action == "claim":
            channel = (node or "").strip()
            engine = str(value or "").strip()
            if not channel or not engine:
                raise ValueError(
                    "claim needs node=<channel> and value=<engine>")
            return _STORE.claim(channel, engine)
        if action == "release":
            channel = (node or "").strip()
            engine = str(value or "").strip()
            if not channel or not engine:
                raise ValueError(
                    "release needs node=<channel> and value=<engine>")
            return _STORE.release(channel, engine)
        if action == "subscribe":
            subscriber = (node or "").strip()
            channels = args.get("channels") or []
            if not subscriber or not channels:
                raise ValueError(
                    "subscribe needs node=<subscriber id> and "
                    "params.channels")
            return _STORE.subscribe(subscriber, list(channels),
                                    payload=args.get("payload"))
        if action == "poll":
            subscriber = (node or "").strip()
            if not subscriber:
                raise ValueError("poll needs node=<subscriber id>")
            return _STORE.poll(subscriber)
        if action == "unsubscribe":
            subscriber = (node or "").strip()
            if not subscriber:
                raise ValueError(
                    "unsubscribe needs node=<subscriber id>")
            _STORE.unsubscribe(subscriber)
            return {"ok": True, "subscriber": subscriber}
        if action == "bvh_write":
            if not sources:
                raise ValueError("bvh_write needs sources=[take JSON]")
            if not output:
                raise ValueError("bvh_write needs output=<path.bvh>")
            take = load_bvh_json(sources[0])
            return _EXPORTER.export_frames(
                take["skeleton"], take["frames"], output, store=_STORE)
        if action == "bvh_read":
            if not sources:
                raise ValueError("bvh_read needs sources=[path.bvh]")
            return _EXPORTER.read_bvh(sources[0])
        if action == "kinect_fixture":
            fixtures = sources[0] if sources else str(_FIXTURES)
            producer = KinectProducer(_STORE)
            return producer.publish_fixture(fixtures)
        if action == "kinect_status":
            from ..hardware.kinect import probe
            report = probe()
            return {
                "sdk_present": report.sdk_present,
                "device_count": report.device_count,
                "summary": report.summary,
            }
        if action == "save":
            if not output:
                raise ValueError("save needs output=<path>")
            return _STORE.save(output)
        if action == "load":
            if not sources:
                raise ValueError("load needs sources=[path]")
            _STORE = MocapStateStore.load(sources[0])
            return {"ok": True, "path": str(Path(sources[0]).resolve()),
                    "channels": _STORE.channels()}
        raise ValueError(f"unhandled mocap action {action!r}")
