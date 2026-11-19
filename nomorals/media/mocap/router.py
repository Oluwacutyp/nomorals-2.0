"""Cross-engine routing for mocap work.

The pattern is artcraftx's ``MultiRouterClient`` (see
``MOCAP_STATE_SURVEY.md`` — pattern only, no upstream code): one
optional engine per provider, and an engine that is not configured
is an honest error naming the engine, never a silent fallback to
another engine.

An engine registers with the categories it serves (``capture``,
``pointcloud``, ``depth``, ``mocap``, ``bvh_write``, ``bvh_read``,
…). Routing a category picks the first configured engine that
serves it, in registration order. When no engine serves a category,
or none of the engines that serve it is configured, the error names
the category and the engines that were considered.
"""

from __future__ import annotations

from typing import Any

from .errors import EngineNotConfigured, MocapError

__all__ = [
    "MocapRouter",
    "default_router",
]


class MocapRouter:
    """Registry of mocap engines and the categories they serve."""

    def __init__(self) -> None:
        self._engines: dict[str, dict[str, Any]] = {}

    def register(self, name: str, categories: tuple[str, ...] | list[str],
                 *, configured: bool = True, note: str = "",
                 handler: Any = None) -> dict[str, Any]:
        """Register an engine. A duplicate name is refused; a
        category list that is empty is refused — an engine that
        serves nothing cannot be routed to."""
        if not name:
            raise MocapError("engine name must not be empty",
                             kind="bad_engine")
        if name in self._engines:
            raise MocapError(f"engine {name!r} is already registered",
                             kind="duplicate_engine")
        if not categories:
            raise MocapError(
                f"engine {name!r} serves no categories",
                kind="bad_engine")
        record = {
            "name": name,
            "categories": list(categories),
            "configured": bool(configured),
            "note": note,
            "handler": handler,
        }
        self._engines[name] = record
        return self._public(record)

    def unregister(self, name: str) -> dict[str, Any]:
        """Remove an engine. Unknown engines are refused by name."""
        try:
            return self._public(self._engines.pop(name))
        except KeyError as exc:
            raise MocapError(f"unknown engine {name!r}",
                             kind="unknown_engine") from exc

    def route(self, category: str) -> dict[str, Any]:
        """Pick the engine for a category: the first configured
        engine that serves it, in registration order."""
        serving = [record for record in self._engines.values()
                   if category in record["categories"]]
        if not serving:
            raise MocapError(
                f"no engine serves category {category!r}",
                kind="unknown_category")
        for record in serving:
            if record["configured"]:
                return self._public(record)
        names = ", ".join(repr(r["name"]) for r in serving)
        raise EngineNotConfigured(
            serving[0]["name"],
            reason=(f"category {category!r} is served only by "
                    f"unconfigured engines: {names}"))

    def engines(self) -> list[dict[str, Any]]:
        """Every registered engine, in registration order."""
        return [self._public(record)
                for record in self._engines.values()]

    def categories(self) -> list[str]:
        """Every category at least one engine serves, sorted."""
        found: set[str] = set()
        for record in self._engines.values():
            found.update(record["categories"])
        return sorted(found)

    @staticmethod
    def _public(record: dict[str, Any]) -> dict[str, Any]:
        out = dict(record)
        out["handler"] = record["handler"] is not None
        return out


def default_router() -> MocapRouter:
    """The router Devon ships with: the local BVH engine (always
    configured — it is pure Python) and the Kinect engine
    (configured only when the SDK sees a device; the reason is
    carried on the record so a refusal names it)."""
    from .kinect_bridge import BvhExporter

    router = MocapRouter()
    router.register(
        "bvh",
        ("mocap", "bvh_write", "bvh_read"),
        configured=True,
        note="local BVH reader/writer — pure Python, always available",
        handler=BvhExporter)
    try:
        from ...hardware.kinect import probe as kinect_probe
        report = kinect_probe()
        if not report.sdk_present:
            configured = False
            note = ("Kinect SDK unavailable: "
                    f"{report.sdk_error or 'libk4a not found'}")
        elif report.device_count > 0:
            configured = True
            note = f"{report.device_count} device(s) connected"
        else:
            configured = False
            note = "no Kinect device connected — connect one to capture"
    except Exception as exc:  # SDK missing or unusable: honest absence
        configured = False
        note = f"Kinect SDK unavailable: {exc}"
    router.register(
        "kinect",
        ("capture", "pointcloud", "depth"),
        configured=configured,
        note=note)
    return router
