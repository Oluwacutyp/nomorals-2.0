"""The native Rust 3D renderer rung (plan §1.5e, engine 1 of 2).

Owner decision 2026-11-05: both §1.5e rungs are built — this one
and the Blender-class external driver — and the profile runtime
picks. This rung's bar, in the owner's words, is to surpass
Blender on Devon's axes: determinism (same document + frame →
byte-identical PNG: single-threaded f64 rasterizer, fixed draw
order, a pinned stored-block PNG encoder, no wall clock anywhere),
agent-drivability (a scene is *data* — the core's evaluated frame
state in, pixels out; there is no GUI-shaped API to steer), and
render speed on Devon's scene classes (a small zero-dependency
binary: no interpreter, no scene-file translation layer).

Division of labor, by the spine's contract: the Python core's
:func:`~.scene3d.scene_state_at` is the one scene semantics — this
module ships its per-frame state records to the Rust binary
(crate: ``nomorals/media/render3d_rs/``) and the binary only
rasterizes the states it is given. It never re-evaluates a scene.

Self-provisioning (the owner's standing doctrine — Devon installs
her own runtimes): :func:`ensure_binary` walks a recorded ladder —
installed binary → cargo build from the in-repo crate → rustup
toolchain provisioned into the workspace tools dir, then build.
Every step lands in ``build_record.json`` beside the binary; a
rung that cannot serve says why, by name, in its probe. Nothing
here runs at import: registration is pure data, provisioning
waits for the first probe.

Env overrides (tests + host control):
``NM_RENDER3D_BIN`` (binary path), ``NM_RENDER3D_TOOLS_DIR``,
``NM_RENDER3D_SOURCE_DIR`` (crate), ``NM_RENDER3D_NO_PROVISION=1``
(never download or build — probe reports the absence by name).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from .scene3d import (
    RendererCapabilities,
    RendererProbe,
    Scene3D,
    Scene3DError,
    register_renderer,
    registered_renderers,
    scene_state_dict,
)

__all__ = [
    "RUNG_NAME",
    "RustSceneRenderer",
    "register_rust_renderer",
    "ensure_binary",
    "provisioning_status",
    "binary_path",
    "crate_dir",
    "tools_root",
]

RUNG_NAME = "rust3d"
_ENGINE_NAME = "devon-render3d"
_PINNED_TOOLCHAIN = "1.85.0"
_RUSTUP_URL = "https://sh.rustup.rs"
_PROVISION_TIMEOUT = 1800
_RENDER_TIMEOUT_FLOOR = 600


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def tools_root() -> Path:
    """The workspace tools dir (env-overridable)."""
    override = os.environ.get("NM_RENDER3D_TOOLS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / "workspace" / "tools"


def crate_dir() -> Path:
    """The in-repo crate (env-overridable for tests)."""
    override = os.environ.get("NM_RENDER3D_SOURCE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent.parent / "render3d_rs"


def _exe_suffix() -> str:
    return ".exe" if os.name == "nt" else ""


def binary_path() -> Path:
    override = os.environ.get("NM_RENDER3D_BIN")
    if override:
        return Path(override).expanduser()
    return tools_root() / "render3d" / f"render3d_rs{_exe_suffix()}"


def _rustup_home() -> Path:
    return Path(os.environ.get("RUSTUP_HOME") or tools_root() / "rustup")


def _cargo_home() -> Path:
    return Path(os.environ.get("CARGO_HOME") or tools_root() / "cargo")


def _cargo_bin() -> Path | None:
    candidate = _cargo_home() / "bin" / f"cargo{_exe_suffix()}"
    if candidate.exists():
        return candidate
    found = shutil.which("cargo")
    return Path(found) if found else None


def _record_path() -> Path:
    return tools_root() / "render3d" / "build_record.json"


def _crate_fingerprint(source: Path) -> str:
    """SHA-256 over the crate's sources — the build record's
    answer to "what exactly was compiled"."""
    digest = hashlib.sha256()
    files = [source / "Cargo.toml"] + sorted(
        (source / "src").glob("*.rs"))
    for path in files:
        if path.exists():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# the provisioning ladder
# ---------------------------------------------------------------------------

_ensure_lock = threading.Lock()
_ensure_cache: dict[tuple[str, str], dict[str, Any]] = {}


def _run(cmd: list[str], *, timeout: int,
         env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, env=env, check=False)


def _binary_runs(path: Path) -> str | None:
    """The binary's --version line, or None when it cannot run."""
    try:
        proc = _run([str(path), "--version"], timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _toolchain_env() -> dict[str, str]:
    env = dict(os.environ)
    env["RUSTUP_HOME"] = str(_rustup_home())
    env["CARGO_HOME"] = str(_cargo_home())
    env["PATH"] = str(_cargo_home() / "bin") + os.pathsep + env.get("PATH", "")
    return env


def _provision_toolchain(record: dict[str, Any]) -> str | None:
    """Install rustup + the pinned toolchain into the tools dir.
    Returns None on success, else the named failure."""
    if os.environ.get("NM_RENDER3D_NO_PROVISION") == "1":
        return ("cargo is absent and self-provisioning is disabled "
                "(NM_RENDER3D_NO_PROVISION=1)")
    tools = tools_root()
    script = tools / "render3d" / "rustup-init.sh"
    try:
        script.parent.mkdir(parents=True, exist_ok=True)
        try:
            proc = _run(["curl", "-sSf", _RUSTUP_URL, "-o", str(script)],
                        timeout=120)
            if proc.returncode != 0:
                raise OSError(proc.stderr.strip()[:200])
        except (OSError, subprocess.TimeoutExpired):
            with urllib.request.urlopen(_RUSTUP_URL, timeout=120) as resp:
                script.write_bytes(resp.read())
    except Exception as exc:  # noqa: BLE001 — the reason is the record
        return f"rustup download failed: {exc}"
    try:
        proc = _run(
            ["sh", str(script), "-y", "--no-modify-path",
             "--profile", "minimal",
             "--default-toolchain", _PINNED_TOOLCHAIN],
            timeout=_PROVISION_TIMEOUT, env=_toolchain_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"rustup install failed: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        return "rustup install failed: " + " | ".join(tail)
    record["toolchain_provisioned"] = _PINNED_TOOLCHAIN
    return None


def _build_binary(record: dict[str, Any]) -> dict[str, Any]:
    """cargo build the in-repo crate and install the artifact."""
    source = crate_dir()
    if not (source / "Cargo.toml").exists():
        return {"status": "unavailable",
                "reason": f"render3d crate source not found at {source}"}
    cargo = _cargo_bin()
    if cargo is None:
        failure = _provision_toolchain(record)
        if failure is not None:
            return {"status": "unavailable", "reason": failure}
        cargo = _cargo_bin()
        if cargo is None:
            return {"status": "unavailable",
                    "reason": "rustup provisioned but cargo is still "
                              "absent from " + str(_cargo_home() / "bin")}
    target_dir = tools_root() / "render3d-target"
    env = _toolchain_env()
    env["CARGO_TARGET_DIR"] = str(target_dir)
    try:
        proc = _run([str(cargo), "build", "--release",
                     "--manifest-path", str(source / "Cargo.toml")],
                    timeout=_PROVISION_TIMEOUT, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable",
                "reason": f"render3d cargo build failed: {exc}"}
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        return {"status": "unavailable",
                "reason": "render3d cargo build failed: "
                          + " | ".join(tail)}
    artifact = target_dir / "release" / f"render3d_rs{_exe_suffix()}"
    if not artifact.exists():
        return {"status": "unavailable",
                "reason": f"cargo build produced no binary at {artifact}"}
    dest = binary_path()
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, dest)
        if os.name != "nt":
            dest.chmod(0o755)
    except OSError as exc:
        return {"status": "unavailable",
                "reason": f"render3d binary install failed: {exc}"}
    version = _binary_runs(dest)
    if version is None:
        return {"status": "unavailable",
                "reason": f"installed render3d binary at {dest} "
                          f"does not run"}
    rustc = _cargo_home() / "bin" / f"rustc{_exe_suffix()}"
    rustc_version = ""
    if rustc.exists():
        try:
            rustc_version = _run([str(rustc), "--version"], timeout=30,
                                 env=_toolchain_env()).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            rustc_version = ""
    record.update({
        "status": "built",
        "binary": str(dest),
        "engine_version": version,
        "rustc": rustc_version,
        "toolchain_pin": _PINNED_TOOLCHAIN,
        "crate_source": str(source),
        "crate_fingerprint": _crate_fingerprint(source),
        "built_at_unix": time.time(),
    })
    try:
        _record_path().parent.mkdir(parents=True, exist_ok=True)
        _record_path().write_text(json.dumps(record, indent=2),
                                  encoding="utf-8")
        record["record_path"] = str(_record_path())
    except OSError:
        record["record_path"] = None
    return record


def ensure_binary() -> dict[str, Any]:
    """The provisioning ladder, as data. Memoized per resolved
    binary/tools paths (env overrides bust the memo).

    * binary present and running → ``present``
    * source + toolchain (or a provisionable one) → ``built``
      (the build is recorded in ``build_record.json``)
    * otherwise → ``unavailable`` with the named reason.
    """
    key = (str(binary_path()), str(tools_root()))
    with _ensure_lock:
        if key in _ensure_cache:
            return dict(_ensure_cache[key])
        record: dict[str, Any] = {"binary": str(binary_path())}
        dest = binary_path()
        if dest.exists():
            version = _binary_runs(dest)
            if version is not None:
                record.update({"status": "present",
                               "engine_version": version})
                _ensure_cache[key] = record
                return dict(record)
            record["stale_binary"] = (
                f"{dest} exists but does not run; rebuilding")
        record = _build_binary(record)
        _ensure_cache[key] = record
        return dict(record)


def provisioning_status() -> dict[str, Any]:
    """The rung's provisioning facts WITHOUT triggering a build:
    what exists, what the last recorded build says."""
    dest = binary_path()
    cargo = _cargo_bin()
    record: dict[str, Any] = {}
    if _record_path().exists():
        try:
            record = json.loads(_record_path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = {}
    return {
        "binary": str(dest),
        "binary_present": dest.exists(),
        "cargo": str(cargo) if cargo else None,
        "crate_source": str(crate_dir()),
        "crate_present": (crate_dir() / "Cargo.toml").exists(),
        "toolchain_pin": _PINNED_TOOLCHAIN,
        "last_build_record": record,
    }


# ---------------------------------------------------------------------------
# the rung
# ---------------------------------------------------------------------------

class RustSceneRenderer:
    """The native Rust renderer behind the scene3d registry."""

    name = RUNG_NAME

    def capabilities(self) -> RendererCapabilities:
        return RendererCapabilities(
            surfaces=("scene3d", "motion_camera", "format3d"),
            node_kinds=("group", "mesh", "light"),
            mesh_kinds=("primitive", "explicit"),
            primitives=("cube", "sphere", "plane", "cylinder", "cone"),
            material_kinds=("flat", "lambert", "phong"),
            light_kinds=("ambient", "directional", "point", "spot"),
            animated_props=("position", "rotation", "scale",
                            "intensity", "fov", "near", "far",
                            "color", "opacity"),
            max_width=3840,
            max_height=2160,
            deterministic=True,
            notes=("native software rasterizer (devon-render3d, "
                   "zero-dependency Rust): single-threaded f64, "
                   "pinned stored-block PNG encoder — same state, "
                   "same bytes. Max canvas is a declared memory "
                   "bound (f64 framebuffers), not a scene limit."))

    def probe(self) -> RendererProbe:
        record = ensure_binary()
        if record["status"] == "present":
            return RendererProbe(True, "")
        if record["status"] == "built":
            return RendererProbe(
                True,
                "binary self-provisioned by the Devon toolchain "
                f"ladder (build record: {record.get('record_path')})")
        return RendererProbe(False, str(record.get("reason", "")))

    def render_frames(self, scene: Scene3D, *,
                      frames: Any, workdir: Any) -> dict[str, Any]:
        record = ensure_binary()
        if record["status"] == "unavailable":
            raise Scene3DError(
                f"rust3d cannot render: {record.get('reason')}")
        out_dir = Path(workdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        width = int(scene.output.get("width", 1920))
        height = int(scene.output.get("height", 1080))
        job = {
            "width": width,
            "height": height,
            "out_dir": str(out_dir),
            "pattern": "f_%04d.png",
            "frames": [{"frame": int(f),
                        "state": scene_state_dict(scene, int(f))}
                       for f in frames],
        }
        job_path = out_dir / "render3d_job.json"
        job_path.write_text(json.dumps(job), encoding="utf-8")
        timeout = max(_RENDER_TIMEOUT_FLOOR, 120 * max(1, len(job["frames"])))
        try:
            proc = _run([str(binary_path()), str(job_path)],
                        timeout=min(timeout, 3600))
        except subprocess.TimeoutExpired as exc:
            raise Scene3DError(
                f"rust3d render timed out after {exc.timeout}s") from exc
        except OSError as exc:
            raise Scene3DError(
                f"rust3d binary could not run: {exc}") from exc
        try:
            result = json.loads(proc.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise Scene3DError(
                f"rust3d returned no result record "
                f"(rc={proc.returncode}): {proc.stderr.strip()[:300]}"
            ) from exc
        if proc.returncode != 0 or not result.get("ok"):
            raise Scene3DError(
                f"rust3d render failed: {result.get('error', proc.stderr.strip()[:300])}")
        paths = []
        for entry in result["frames"]:
            path = out_dir / entry["file"]
            if not path.exists():
                raise Scene3DError(
                    f"rust3d reported {entry['file']} but wrote no "
                    f"file at {path}")
            paths.append(str(path))
        return {"frames_dir": str(out_dir),
                "pattern": "f_%04d.png",
                "framerate": float(scene.output.get("fps", 30)),
                "frames": paths,
                "width": width,
                "height": height,
                "engine": {"name": _ENGINE_NAME,
                           "version": result.get("version"),
                           "binary": str(binary_path()),
                           "total_ms": result.get("total_ms"),
                           "frames_ms": [e.get("render_ms")
                                         for e in result["frames"]]}}


_RENDERER = RustSceneRenderer()


def register_rust_renderer() -> RustSceneRenderer:
    """Register the rung (idempotent — engines register at import
    time; a duplicate registration is a no-op here, not an error)."""
    if RUNG_NAME not in registered_renderers():
        register_renderer(_RENDERER)
    return _RENDERER
