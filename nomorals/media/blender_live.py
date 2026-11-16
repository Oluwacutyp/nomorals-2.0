"""Live Blender steering via the MCP for Blender server (Blender MCP).

Devon already renders 3D in batch — ``render3d_blender.py`` drives
Blender headless as a render substrate (a Scene3D document becomes a
bpy script, rendered to frames). This organ is the LIVE rung beside
it: it talks to a Blender the operator is running, through the MCP
for Blender server (ahujasid/mcp-for-blender, MIT) over Devon's MCP
client (``nomorals/tools/mcp_client.py``, server ``"blender"`` —
``uvx mcp-for-blender``).

The ten tools the server exposes are ``execute_blender_code`` (run
Blender Python in the live session), ``look`` (screenshot from the
viewport, the camera, several angles, or animation frames), ``
get_scene_info`` (a compact scene summary), ``generate_3d`` /
``generate_gaussian_world`` / ``search_assets`` / ``import_asset``
(operator-keyed generators and asset libraries), ``get_addon_status``
(Blender + add-on versions, what is switched on), and the telemetry
controls (``disable_telemetry``, ``record_trajectory_feedback``). The
README says "nine" — a miscount; the server code exposes ten distinct
tools. This organ uses the four live-steering tools; the
generators need the operator's own keys and are left to the server.

The add-on must be installed in Blender and its server started
(``uvx mcp-for-blender install-addon``, then enable the add-on in
Blender). Devon NEVER starts Blender herself — she connects to the
running session, and says so plainly when none is found. The live
session is bounded: every call carries the MCP client's per-call
timeout, and :meth:`BlenderLiveSession.iterate` caps its iterations.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..tools.mcp_client import (
    MCPClientError,
    MCPClientManager,
)

#: The MCP server name in Devon's registry (``DEFAULT_MCP_SERVERS``).
BLENDER_MCP_SERVER = "blender"

#: The ten tools the MCP for Blender server exposes (the README
#: says "nine" — a miscount; the server code exposes ten distinct
#: tools).
BLENDER_MCP_TOOLS: tuple[str, ...] = (
    "execute_blender_code",
    "look",
    "get_scene_info",
    "generate_3d",
    "generate_gaussian_world",
    "search_assets",
    "import_asset",
    "get_addon_status",
    "disable_telemetry",
    "record_trajectory_feedback",
)

#: The live-steering tools this organ uses (the rest need the
#: operator's own generator/asset keys and are left to the server).
LIVE_TOOLS: tuple[str, ...] = (
    "execute_blender_code",
    "look",
    "get_scene_info",
    "get_addon_status",
)

#: Default cap on iterate-loop passes (run -> screenshot -> score ->
#: adjust). A task that has not converged by then is reported, not
#: run forever.
DEFAULT_MAX_ITERATIONS = 3

#: The score at which the iterate loop stops early (the scene matches
#: the goal well enough to call it done).
GOAL_SCORE_THRESHOLD = 0.8

#: Words too common to count as goal evidence in the scene summary.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "with",
    "make", "create", "add", "put", "show", "give", "get", "set",
    "use", "using", "for", "from", "by", "at", "is", "are", "be",
    "it", "its", "this", "that", "into", "over", "under", "new",
})

#: How the operator gets a live Blender going (the organ never does
#: this itself).
ADDON_SETUP = (
    "No live Blender found. Start Blender, install the MCP for "
    "Blender add-on (`uvx mcp-for-blender install-addon`), enable "
    "the add-on in Blender, and make sure its server is running — "
    "then try again. Devon never starts Blender herself."
)


def _score_scene_against_goal(goal: str, scene_summary: str) -> tuple[float, str]:
    """How well the scene summary covers the goal's named things.

    A deterministic, honest check: the goal's content words (lowercased,
    stopwords dropped) that appear in the scene summary, as a fraction
    of the goal's content words. A goal with no content words scores 0
    — it names nothing to check.
    """
    words = [w for w in re.findall(r"[a-z0-9]+", goal.lower())
             if w not in _STOPWORDS]
    if not words:
        return 0.0, "the goal names nothing to check in the scene"
    summary = scene_summary.lower()
    hits = [w for w in words if w in summary]
    score = len(hits) / len(words)
    if hits:
        note = f"scene covers {len(hits)}/{len(words)} goal words: {', '.join(hits)}"
    else:
        note = f"scene covers none of the {len(words)} goal words"
    return score, note


def _extract_image(result: dict[str, Any]) -> dict[str, Any]:
    """Pull the first image out of an MCP tool result, if there is one.

    The ``look`` tool returns its screenshot as a content item of type
    ``image`` (base64 ``data`` + ``mimeType``) beside the text caption.
    Returns ``{"image_b64": ..., "mime": ...}`` or ``{}`` when the
    result carries no image.
    """
    for item in result.get("content") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "image":
            continue
        data = item.get("data") or ""
        if not data:
            continue
        try:
            base64.b64decode(data, validate=True)
        except Exception:  # noqa: BLE001 - not base64, not an image
            continue
        return {"image_b64": str(data),
                "mime": str(item.get("mimeType") or "image/png")}
    return {}


def _extract_text(result: dict[str, Any]) -> str:
    """The text content of an MCP tool result, concatenated."""
    parts = []
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "\n".join(parts)


class BlenderLiveSession:
    """One live Blender session, reached through the MCP client.

    Constructing the session connects nothing; :meth:`connect` starts
    the MCP server subprocess and handshakes. Every call goes through
    the MCP client's per-call timeout. Close with :meth:`close` (or
    use the session as a context manager).
    """

    def __init__(self, manager: MCPClientManager | None = None, *,
                 settings: Any = None,
                 max_iterations: int = DEFAULT_MAX_ITERATIONS) -> None:
        self._manager = manager or MCPClientManager.from_settings(settings)
        self.max_iterations = max(1, int(max_iterations))
        self.connected = False
        self.server_info: dict[str, Any] = {}
        self.tools: list[str] = []

    # ── lifecycle ────────────────────────────────────────────────────

    def connect(self) -> dict[str, Any]:
        """Connect to the running Blender via the MCP server.

        Returns ``{"ok": True, "server": ..., "tools": [...]}`` on
        success, or ``{"ok": False, "error": ...}`` when no live
        Blender is reachable — the operator's step, stated plainly.
        """
        if self.connected:
            return {"ok": True, "server": self.server_info,
                    "tools": list(self.tools)}
        try:
            client = self._manager.connect(BLENDER_MCP_SERVER)
            tools = [t.get("name") for t in client.list_tools()
                     if isinstance(t, dict)]
        except MCPClientError as exc:
            return {"ok": False, "error": f"{ADDON_SETUP} ({exc})"}
        except Exception as exc:  # noqa: BLE001 - connect must not raise
            return {"ok": False, "error": f"{ADDON_SETUP} ({exc})"}
        self.connected = True
        self.server_info = dict(client.server_info or {})
        self.tools = [t for t in tools if t]
        return {"ok": True, "server": self.server_info,
                "tools": list(self.tools)}

    def close(self) -> None:
        """Shut the MCP connection down (the Blender session is the
        operator's — closing here only ends Devon's side)."""
        if self.connected:
            self._manager.close(BLENDER_MCP_SERVER)
        self.connected = False
        self.server_info = {}
        self.tools = []

    def __enter__(self) -> BlenderLiveSession:
        result = self.connect()
        if not result.get("ok"):
            raise MCPClientError(str(result.get("error") or "no live Blender"))
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── the four live-steering tools ─────────────────────────────────

    def _call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one Blender MCP tool, returning the raw MCP result.

        Raises :class:`MCPClientError` when no session is connected
        (call :meth:`connect` first) or the transport fails.
        """
        if not self.connected:
            result = self.connect()
            if not result.get("ok"):
                raise MCPClientError(str(result.get("error") or "no live Blender"))
        return self._manager.call_tool(BLENDER_MCP_SERVER, tool, arguments)

    def status(self) -> dict[str, Any]:
        """The connected Blender: version, add-on version, what
        libraries and generators are switched on (``get_addon_status``).
        """
        result = self._call("get_addon_status", {})
        if result.get("isError"):
            return {"ok": False, "error": _extract_text(result) or "addon status failed"}
        return {"ok": True, "text": _extract_text(result)}

    def scene_summary(self, *, fields: list[str] | None = None,
                      query: str | None = None,
                      root: str | None = None,
                      limit: int = 20) -> dict[str, Any]:
        """A compact scene summary, one line per object
        (``get_scene_info``). ``fields`` picks what each line shows
        (placement, materials, topology, weights, ...); ``query``
        filters by name, ``root`` drills into one object's hierarchy.
        """
        arguments: dict[str, Any] = {"limit": max(1, int(limit))}
        if fields:
            arguments["fields"] = list(fields)
        if query:
            arguments["query"] = query
        if root:
            arguments["root"] = root
        result = self._call("get_scene_info", arguments)
        if result.get("isError"):
            return {"ok": False, "error": _extract_text(result) or "scene info failed"}
        return {"ok": True, "text": _extract_text(result)}

    def run_code(self, code: str, user_prompt: str = "") -> dict[str, Any]:
        """Run Blender Python in the live session
        (``execute_blender_code``). Whatever the code prints is
        returned. Work in small steps and print what you need to know.
        """
        if not isinstance(code, str) or not code.strip():
            return {"ok": False, "error": "no Blender Python code given"}
        arguments: dict[str, Any] = {"code": code}
        if user_prompt:
            arguments["user_prompt"] = user_prompt
        result = self._call("execute_blender_code", arguments)
        if result.get("isError"):
            return {"ok": False, "error": _extract_text(result) or "code failed in Blender"}
        return {"ok": True, "text": _extract_text(result)}

    def screenshot(self, *, mode: str = "angles",
                   views: list[str] | None = None,
                   shading: str = "material",
                   target: list[str] | None = None,
                   distance: float | None = None,
                   frame_count: int = 6) -> dict[str, Any]:
        """See the live scene (``look``).

        ``mode`` is ``viewport``, ``camera``, ``angles`` (an
        auto-framed multi-view sheet — the default, so one call shows
        the scene from several sides) or ``frames`` (an animation
        strip). ``views`` picks the angles (front, back, left, right,
        top, three_quarter); ``shading`` is solid, material, rendered,
        wireframe or xray. Returns the screenshot bytes (base64) plus
        the caption text.
        """
        arguments: dict[str, Any] = {"mode": mode, "shading": shading,
                                     "frame_count": max(1, int(frame_count))}
        if views:
            arguments["views"] = list(views)
        if target:
            arguments["target"] = list(target)
        if distance is not None:
            arguments["distance"] = float(distance)
        result = self._call("look", arguments)
        if result.get("isError"):
            return {"ok": False, "error": _extract_text(result) or "look failed"}
        out: dict[str, Any] = {"ok": True,
                               "text": _extract_text(result)}
        out.update(_extract_image(result))
        return out

    # ── the iterate loop ─────────────────────────────────────────────

    def motion_target(self, goal: str,
                      suggest: Callable[[str], str]) -> dict[str, Any] | None:
        """The motion score for an animation goal, from Devon's
        existing motion-score rung (``media/directed/motion_score.py``).

        Returns the score dict, or None when the goal is not a motion
        goal (implausible / unparseable) — the iterate loop then falls
        back to the scene-summary score. The import is lazy: the rung
        is only loaded when a motion goal is actually directed.
        """
        from ..directed.motion_score import generate_motion_score

        try:
            return generate_motion_score(goal, suggest)
        except Exception:  # noqa: BLE001 - not a motion goal, fall back
            return None

    def iterate(self, goal: str,
                code_fn: Callable[..., str], *,
                max_iterations: int | None = None,
                suggest: Callable[[str], str] | None = None,
                score_fn: Callable[..., tuple[float, str]] | None = None
                ) -> dict[str, Any]:
        """Run -> screenshot -> score -> adjust, bounded.

        ``code_fn(goal, feedback, motion_target)`` returns the Blender
        Python for this pass; ``feedback`` is the previous pass's score
        note (empty on the first pass) and ``motion_target`` is the
        motion score from :meth:`motion_target` when ``suggest`` (the
        brain) is given, else None. After each pass the scene is
        screenshotted (angles) and summarised; the score is
        ``score_fn(scene_summary, screenshot, motion_target)`` when
        given, else the built-in scene-summary check. The loop stops
        early at :data:`GOAL_SCORE_THRESHOLD` and never exceeds
        ``max_iterations`` (default the session's cap). Returns the
        passes, the final score, the last screenshot, and whether the
        goal was met.
        """
        limit = self.max_iterations if max_iterations is None else max(1, int(max_iterations))
        target = self.motion_target(goal, suggest) if suggest is not None else None
        passes: list[dict[str, Any]] = []
        feedback = ""
        final_score = 0.0
        last_shot: dict[str, Any] = {}
        last_summary = ""
        for number in range(1, limit + 1):
            code = code_fn(goal, feedback, target)
            run = self.run_code(code, user_prompt=goal)
            shot = self.screenshot()
            summary = self.scene_summary()
            last_shot = shot
            last_summary = summary.get("text", "") if summary.get("ok") else ""
            if score_fn is not None:
                score, note = score_fn(last_summary, shot, target)
            else:
                score, note = _score_scene_against_goal(goal, last_summary)
            final_score = float(score)
            passes.append({
                "pass": number,
                "ok": bool(run.get("ok")),
                "run": run.get("text", "") if run.get("ok") else run.get("error", ""),
                "screenshot_ok": bool(shot.get("ok")),
                "score": final_score,
                "note": note,
            })
            if not run.get("ok"):
                feedback = f"pass {number} failed in Blender: {run.get('error', '')}"
                continue
            if final_score >= GOAL_SCORE_THRESHOLD:
                break
            feedback = f"pass {number} scored {final_score:.2f}: {note}"
        return {
            "ok": True,
            "goal": goal,
            "met": final_score >= GOAL_SCORE_THRESHOLD,
            "score": final_score,
            "passes": passes,
            "screenshot": last_shot,
            "scene_summary": last_summary,
        }


# ── tool registration ────────────────────────────────────────────────

def register(registry: Any) -> None:
    """Register the live-Blender steering tool on the spine (audit gap R2-25).

    One session per call: connect (honest refusal when no live Blender is
    reachable — Devon never starts Blender herself), run the action, close.
    The ``iterate`` loop is NOT a tool action: its ``code_fn`` comes from
    the brain, so the brain calls :meth:`BlenderLiveSession.iterate`
    directly instead of through this tool.
    """
    from ..core.policy import Capability

    @registry.register(
        "blender_live",
        description=(
            "Steer the operator's live Blender session through the MCP for "
            "Blender server (action, code, path, query, mode). Actions: "
            "connect (handshake with the running Blender), status (versions "
            "+ add-on state), scene (compact scene summary, optional query), "
            "look (viewport screenshot), code (run Blender Python in the "
            "live session), import (pull a GLB/FBX into the live scene). "
            "Devon never starts Blender herself — when no live session is "
            "reachable the tool says so and names the setup step."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "action": ("str (optional) — connect | status | scene | look | "
                       "code | import"),
            "code": "str (optional) — Blender Python for the code action",
            "path": "str (optional) — .glb/.fbx path for the import action",
            "query": "str (optional) — scene filter for the scene action",
            "mode": "str (optional) — look mode: viewport | camera | angles | frames",
        },
    )
    def blender_live(action: str = "status", code: str = "",
                     path: str = "", query: str = "",
                     mode: str = "angles") -> dict[str, Any]:
        session = BlenderLiveSession()
        try:
            connected = session.connect()
            if not connected.get("ok"):
                return {"ok": False,
                        "error": str(connected.get("error")
                                     or "no live Blender")}
            if action == "connect":
                return connected
            if action == "status":
                return session.status()
            if action == "scene":
                return session.scene_summary(query=query or None)
            if action == "look":
                return session.screenshot(mode=mode)
            if action == "code":
                return session.run_code(code)
            if action == "import":
                suffix = Path(path).suffix.lower()
                if suffix == ".glb":
                    from .trellis import blender_import_snippet
                elif suffix == ".fbx":
                    from .unirig import blender_import_snippet
                else:
                    return {"ok": False,
                            "error": (f"cannot import {suffix!r} into "
                                      "Blender — use a .glb or .fbx")}
                return session.run_code(blender_import_snippet(path))
            return {"ok": False,
                    "error": (f"unknown blender_live action {action!r} — "
                              "use connect|status|scene|look|code|import")}
        finally:
            session.close()
