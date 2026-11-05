"""The Blender-class 3D rung (§1.5e, engine 2 of 2).

Devon drives Blender headless as a substrate — the ffmpeg pattern:
a :class:`~.scene3d.Scene3D` document plus a frame range become a
self-contained Blender Python (``bpy``) script, and
``blender --background --python <script>`` renders the contract's
``f_%04d.png`` image sequence. Owner decision 2026-11-05: both
§1.5e rungs build (this one and the native Rust renderer); the
profile runtime picks per device.

Honesty rules, inherited from the §1.1e contract:

* **Animation is baked, never re-derived.** The generated script
  carries the core's evaluated frame states (``scene_state_at``
  output for every frame in the bake range, embedded as data) and
  sets object/camera/light/material state from them per frame —
  Blender's own interpolation never sees an easing curve, so the
  two engines cannot diverge on animation math.
* **The mapping table is the capabilities declaration.** The
  module-level ``PRIMITIVE_MAP`` / ``MATERIAL_MAP`` / ``LIGHT_MAP``
  tables drive both the script generator and
  :meth:`BlenderRenderer.capabilities`, so what the registry is
  told the rung can do is by construction what the driver emits.
  Approximations (unitless intensity → Blender energy, Blinn-Phong
  shininess → GGX roughness, ambient → world background) are named
  in the tables, not buried in code.
* **The probe self-provisions, once.** :func:`probe_blender`
  locates a binary (``NM_BLENDER_BIN``, PATH,
  ``~/workspace/tools/blender``); if none exists it attempts ONE
  provisioning route — the official portable tarball from
  download.blender.org into ``~/workspace/tools/blender`` — and
  records the outcome in a marker file there. A failed route is
  never retried silently: the probe reports unavailable with the
  recorded reason (devon's self-provisioning doctrine: a missing
  tool is her job to fix; the owner's hand is only for genuine
  walls, and a dead download route is reported, not hidden).
  ``NM_BLENDER_NO_PROVISION=1`` disables the attempt (managed
  hosts); ``NM_BLENDER_REPROVISION=1`` clears the marker.
* **Determinism is the frame function's, not Blender's.** The
  generated script is byte-for-byte deterministic for a given
  document + frame set, and Cycles runs with a fixed seed and
  denoising off — but Blender's pixel output can still vary by
  version and host, so the rung declares ``deterministic=False``
  and the render's tie to determinism is the per-frame state
  hashes :func:`~.scene3d.render_scene` attaches.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from .scene3d import (
    NODE_KINDS,
    RendererCapabilities,
    RendererProbe,
    Scene3D,
    Scene3DError,
    get_renderer,
    register_renderer,
    registered_renderers,
    scene_state_at,
)

__all__ = [
    "RENDERER_NAME",
    "PRIMITIVE_MAP", "MATERIAL_MAP", "LIGHT_MAP",
    "BLENDER_VERSION_PIN", "BLENDER_TARBALL_URL",
    "BlenderLocator", "default_locator",
    "find_blender", "blender_version", "provision_blender",
    "probe_blender", "generate_bpy_script",
    "BlenderRenderer", "register_blender_renderer",
]

RENDERER_NAME = "blender"

#: the portable build the provisioning route fetches (an LTS line;
#: the driver itself accepts any Blender >= 3.0 on the host).
BLENDER_VERSION_PIN = "4.2.3"
BLENDER_TARBALL_URL = (
    "https://download.blender.org/release/Blender4.2/"
    f"blender-{BLENDER_VERSION_PIN}-linux-x64.tar.xz")

_DONE_MARKER = "DEVON3D_DONE "
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# the mapping tables — capabilities are derived from these, and the
# script generator emits from these; they cannot drift apart.
# ---------------------------------------------------------------------------

#: core primitive → the bpy creation op + the param translation.
#: ``params`` maps core param names (with fallbacks in order) onto
#: the op's kwargs; every approximation is named in ``note``.
PRIMITIVE_MAP: dict[str, dict[str, Any]] = {
    "cube": {
        "op": "bpy.ops.mesh.primitive_cube_add",
        "kwargs": {"size": ("size", 2.0)},
        "note": "core size is the cube's edge length, as is "
                "Blender's size",
    },
    "sphere": {
        "op": "bpy.ops.mesh.primitive_uv_sphere_add",
        "kwargs": {"radius": ("radius", 1.0),
                   "segments": ("segments", 32),
                   "ring_count": ("ring_count", 16)},
        "note": "UV sphere; segments/ring_count are tessellation, "
                "not shape",
    },
    "plane": {
        "op": "bpy.ops.mesh.primitive_plane_add",
        "kwargs": {"size": ("size", 2.0)},
        "note": "a zero-thickness XY sheet, as authored",
    },
    "cylinder": {
        "op": "bpy.ops.mesh.primitive_cylinder_add",
        "kwargs": {"radius": ("radius", 1.0),
                   "depth": ("height|depth", 2.0),
                   "vertices": ("segments", 32)},
        "note": "the core's height is Blender's depth (the Z "
                "extent); segments tessellate the caps' rim",
    },
    "cone": {
        "op": "bpy.ops.mesh.primitive_cone_add",
        "kwargs": {"radius1": ("radius", 1.0),
                   "radius2": ("radius2", 0.0),
                   "depth": ("height|depth", 2.0),
                   "vertices": ("segments", 32)},
        "note": "radius2 defaults to a true cone point (0.0); "
                "height is Blender's depth",
    },
}

#: core material kind → the Blender shading construction.
MATERIAL_MAP: dict[str, dict[str, Any]] = {
    "flat": {
        "construction": "emission",
        "note": "unlit color is an Emission shader (strength 1) — "
                "truly unlit in Cycles/Eevee; opacity rides a "
                "transparent mix",
    },
    "lambert": {
        "construction": "principled",
        "note": "Principled BSDF with roughness 1.0 and specular "
                "0.0 is Blender's diffuse-only configuration; "
                "emissive maps to Principled's emission inputs",
    },
    "phong": {
        "construction": "principled",
        "note": "Blinn-Phong shininess converts to GGX roughness "
                "as sqrt(2 / (shininess + 2)) — the standard "
                "conversion; the lobe is GGX, not Phong",
    },
}

#: core light kind → the Blender light/world construction.
LIGHT_MAP: dict[str, dict[str, Any]] = {
    "ambient": {
        "blender": "WORLD",
        "note": "Blender has no ambient light object: ambient "
                "nodes sum into the world background (color x "
                "intensity, added to the scene background)",
    },
    "directional": {
        "blender": "SUN",
        "note": "energy = core intensity 1:1 (see below)",
    },
    "point": {
        "blender": "POINT",
        "note": "energy = core intensity 1:1 — the core's "
                "intensity is unitless and Blender's energy is "
                "not, so absolute brightness is a mapping "
                "approximation; falloff shape is the engine's",
    },
    "spot": {
        "blender": "SPOT",
        "note": "spot_angle -> spot_size (radians), "
                "spot_penumbra -> spot_blend; energy as point",
    },
}


# ---------------------------------------------------------------------------
# locating + provisioning Blender
# ---------------------------------------------------------------------------

def _default_tools_dir() -> Path:
    return Path.home() / "workspace" / "tools" / "blender"


class BlenderLocator:
    """Finds (and, once, provisions) a Blender binary.

    Every external seam is injectable for tests: the environment
    mapping, the tools dir, the downloader and the runner. The
    provisioning attempt is marker-gated: ``<tools>/.provision.json``
    records the outcome, and a recorded failure is reported, never
    silently retried.
    """

    def __init__(self, *,
                 tools_dir: str | os.PathLike | None = None,
                 environ: dict[str, str] | None = None,
                 downloader: Callable[[str, Path], None] | None = None,
                 runner: Callable[..., subprocess.CompletedProcess]
                 | None = None) -> None:
        self.tools_dir = Path(tools_dir) if tools_dir is not None \
            else _default_tools_dir()
        self.environ = dict(os.environ if environ is None else environ)
        self._downloader = downloader or _download
        self._runner = runner or subprocess.run
        self._found: tuple[Path, str] | None = None

    # -- finding -----------------------------------------------------
    def find(self) -> Path | None:
        """The binary, if one exists on this host right now."""
        if self._found is not None:
            return self._found[0]
        env_bin = self.environ.get("NM_BLENDER_BIN", "").strip()
        if env_bin:
            p = Path(env_bin).expanduser()
            if p.is_file() and os.access(p, os.X_OK):
                return p
            return None  # an explicit path that is wrong is a fact
        on_path = shutil.which("blender", path=self.environ.get("PATH"))
        if on_path:
            return Path(on_path)
        for candidate in self._tools_candidates():
            return candidate
        return None

    def _tools_candidates(self) -> list[Path]:
        out: list[Path] = []
        root = self.tools_dir
        if not root.is_dir():
            return out
        direct = root / "blender"
        if direct.is_file() and os.access(direct, os.X_OK):
            out.append(direct)
        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).relative_to(root).parts)
            if depth > 3:
                dirnames[:] = []
                continue
            if "blender" in filenames:
                p = Path(dirpath) / "blender"
                if os.access(p, os.X_OK):
                    out.append(p)
        out.sort()
        return out

    def version(self, binary: Path) -> str:
        """``blender --version``'s headline, or "" if unreadable."""
        if self._found is not None and self._found[0] == binary:
            return self._found[1]
        try:
            proc = self._runner([str(binary), "--version"],
                                capture_output=True, text=True,
                                timeout=60)
            first = (proc.stdout or "").splitlines()[0].strip()
            version = first if first.lower().startswith("blender") \
                else ""
        except Exception:  # noqa: BLE001 — an unrunnable binary
            version = ""        # is probed as found-but-unreadable
        if version:
            self._found = (binary, version)
        return version

    # -- provisioning --------------------------------------------------
    @property
    def _marker_path(self) -> Path:
        return self.tools_dir / ".provision.json"

    def _read_marker(self) -> dict[str, Any] | None:
        try:
            return json.loads(self._marker_path.read_text(
                encoding="utf-8"))
        except Exception:  # noqa: BLE001 — no marker, no opinion
            return None

    def _write_marker(self, data: dict[str, Any]) -> None:
        # the marker is a cache: failing to write it must never
        # crash a probe, so the write is best-effort by design.
        with suppress(Exception):
            self.tools_dir.mkdir(parents=True, exist_ok=True)
            self._marker_path.write_text(
                json.dumps(data, indent=2, sort_keys=True),
                encoding="utf-8")

    def provision(self) -> tuple[Path | None, str]:
        """The ONE self-provision route: the official portable
        tarball into the tools dir. Returns (binary, reason) —
        exactly one of the two is meaningful."""
        if platform.system() != "Linux" \
                or platform.machine() not in ("x86_64", "AMD64"):
            return None, (f"portable-tarball provisioning is only "
                          f"viable on linux/x86_64 (this host: "
                          f"{platform.system()}/{platform.machine()})")
        archive = self.tools_dir / f"blender-{BLENDER_VERSION_PIN}.tar.xz"
        try:
            self.tools_dir.mkdir(parents=True, exist_ok=True)
            self._downloader(BLENDER_TARBALL_URL, archive)
            with tarfile.open(archive, "r:xz") as tar:
                try:
                    tar.extractall(self.tools_dir, filter="data")
                except TypeError:  # older Pythons: no filter kwarg
                    tar.extractall(self.tools_dir)
                except Exception:  # noqa: BLE001 — the data filter
                    # rejects some of Blender's internal relative
                    # symlinks; the archive is the pinned official
                    # release over HTTPS, so fall back to a plain
                    # extraction of this one known artifact.
                    tar.extractall(self.tools_dir)
        except Exception as exc:  # noqa: BLE001 — named, not hidden
            return None, f"blender provisioning failed: {exc}"
        finally:
            with suppress(Exception):
                archive.unlink(missing_ok=True)
        for candidate in self._tools_candidates():
            if self.version(candidate):
                return candidate, ""
        return None, ("blender provisioning downloaded and "
                      "extracted the tarball but no runnable "
                      "blender binary was found under "
                      f"{self.tools_dir}")

    # -- the probe ------------------------------------------------------
    def probe(self) -> RendererProbe:
        binary = self.find()
        if binary is not None:
            version = self.version(binary)
            if version:
                return RendererProbe(
                    True, f"{version} at {binary}")
            return RendererProbe(
                False, f"blender binary at {binary} did not answer "
                       f"--version")
        if self.environ.get("NM_BLENDER_BIN", "").strip():
            return RendererProbe(
                False, f"NM_BLENDER_BIN="
                       f"{self.environ['NM_BLENDER_BIN']!r} does not "
                       f"name a runnable blender binary")
        if self.environ.get("NM_BLENDER_NO_PROVISION") == "1":
            return RendererProbe(
                False, "no blender binary found (PATH / "
                       f"{self.tools_dir}) and provisioning is "
                       f"disabled (NM_BLENDER_NO_PROVISION=1)")
        marker = None if self.environ.get(
            "NM_BLENDER_REPROVISION") == "1" else self._read_marker()
        if marker is not None and marker.get("status") == "failed":
            return RendererProbe(
                False, "no blender binary found and the one "
                       "provisioning attempt already failed: "
                       f"{marker.get('reason', 'unknown reason')} "
                       f"(NM_BLENDER_REPROVISION=1 retries)")
        binary, reason = self.provision()
        if binary is not None:
            self._write_marker({"status": "ok",
                                "binary": str(binary),
                                "version": self.version(binary)})
            return RendererProbe(
                True, f"{self.version(binary)} at {binary} "
                      f"(self-provisioned portable build)")
        self._write_marker({"status": "failed", "reason": reason})
        return RendererProbe(False, reason)


_DEFAULT_LOCATOR: BlenderLocator | None = None


def default_locator() -> BlenderLocator:
    global _DEFAULT_LOCATOR
    if _DEFAULT_LOCATOR is None:
        _DEFAULT_LOCATOR = BlenderLocator()
    return _DEFAULT_LOCATOR


def find_blender(locator: BlenderLocator | None = None) -> Path | None:
    return (locator or default_locator()).find()


def blender_version(locator: BlenderLocator | None = None) -> str:
    loc = locator or default_locator()
    binary = loc.find()
    return loc.version(binary) if binary is not None else ""


def provision_blender(locator: BlenderLocator | None = None
                      ) -> tuple[Path | None, str]:
    return (locator or default_locator()).provision()


def probe_blender(locator: BlenderLocator | None = None
                  ) -> RendererProbe:
    return (locator or default_locator()).probe()


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` (the provisioning downloader)."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "devon-render3d-blender"})
    with urllib.request.urlopen(request, timeout=180) as response, \
            open(dest, "wb") as fh:
        shutil.copyfileobj(response, fh, length=1024 * 1024)


# ---------------------------------------------------------------------------
# the bpy script generator — deterministic, state baked from the core
# ---------------------------------------------------------------------------

def _f(value: float) -> str:
    return repr(float(value))


def _vec(values: Sequence[float]) -> str:
    return "(" + ", ".join(_f(v) for v in values) + ")"


def _frame_table(scene: Scene3D, frames: Sequence[int]
                 ) -> dict[str, Any]:
    """``scene_state_at`` output, compacted, for every frame in
    the bake range — the script's ONLY source of animation truth."""
    lo, hi = min(frames), max(frames)
    table: dict[str, Any] = {}
    for f in range(lo, hi + 1):
        state = scene_state_at(scene, f)
        nodes: dict[str, Any] = {}
        for name, entry in state.nodes.items():
            row: dict[str, Any] = {"m": entry["world_matrix"],
                                   "v": bool(entry["visible"])}
            light = entry.get("light")
            if light is not None:
                row["e"] = float(light["intensity"])
                if light["kind"] == "ambient":
                    row["amb"] = [float(c) * float(light["intensity"])
                                  for c in light["color"]]
            nodes[name] = row
        mats = {name: {"color": [float(c) for c in m["color"]],
                       "opacity": float(m["opacity"])}
                for name, m in state.materials.items()}
        cam = state.camera
        table[str(f)] = {
            "nodes": nodes, "mats": mats,
            "cam": {"m": cam["world_matrix"], "fov": cam["fov"],
                    "near": cam["near"], "far": cam["far"]}}
    return table


def generate_bpy_script(scene: Scene3D, frames: Sequence[int], *,
                        engine: str = "CYCLES", samples: int = 32,
                        out_dir_name: str = "frames") -> str:
    """The bpy script for ``scene`` + ``frames`` — a pure function
    of the document, the frame set and the render settings, so two
    calls with equal inputs are byte-identical.

    ``out_dir_name`` is the directory (relative to the script's own
    folder at run time, absolute when the driver rewrites it — see
    :meth:`BlenderRenderer.render_frames`) the PNGs land in; the
    script renders exactly ``frames`` as ``f_%04d.png`` there.
    """
    frame_list = sorted({int(f) for f in frames})
    if not frame_list:
        raise Scene3DError("generate_bpy_script needs at least "
                           "one frame")
    table = _frame_table(scene, frame_list)
    width = int(scene.output.get("width", 1920))
    height = int(scene.output.get("height", 1080))
    fps = float(scene.output.get("fps", 30))
    cam_name = scene.camera_name()
    lines: list[str] = []
    add = lines.append
    add("# Devon Scene3D -> Blender bake (render3d_blender)")
    add(f"# scene: {scene.title!r}  doc_hash: "
        f"{scene.doc_hash or scene.compute_hash()}")
    add(f"# frames: {frame_list}  baked: "
        f"{frame_list[0]}..{frame_list[-1]}")
    add("# animation state is baked from scene_state_at — this "
        "script derives nothing")
    add("import json, math, os, sys")
    add("import bpy")
    add("from mathutils import Matrix")
    add("")
    add(f"OUT_DIR = {out_dir_name!r}")
    add(f"FRAMES = {frame_list!r}")
    add("TABLE = json.loads(")
    add(repr(json.dumps(table, sort_keys=True,
                        separators=(",", ":"))))
    add(")")
    add("")
    add("def _mat(m):")
    add("    return Matrix((tuple(m[0:4]), tuple(m[4:8]), "
        "tuple(m[8:12]), tuple(m[12:16])))")
    add("")
    add("def _set(obj, matrix):")
    add("    obj.matrix_world = _mat(matrix)")
    add("")
    add("def _keys(obj, frame):")
    add("    obj.keyframe_insert('location', frame=frame)")
    add("    obj.keyframe_insert('rotation_quaternion', frame=frame)")
    add("    obj.keyframe_insert('scale', frame=frame)")
    add("    obj.keyframe_insert('hide_render', frame=frame)")
    add("")
    add("bpy.ops.wm.read_factory_settings(use_empty=True)")
    add("S = bpy.context.scene")
    add(f"S.render.engine = {engine!r}")
    if engine == "CYCLES":
        add("S.cycles.samples = " + str(int(samples)))
        add("S.cycles.seed = 0")
        add("S.cycles.use_denoising = False")
        add("S.cycles.device = 'CPU'")
    add(f"S.render.resolution_x = {width}")
    add(f"S.render.resolution_y = {height}")
    add("S.render.resolution_percentage = 100")
    fps_int = max(1, int(round(fps)))
    add(f"S.render.fps = {fps_int}")
    add(f"S.render.fps_base = {_f(fps / fps_int)}")
    add("S.render.image_settings.file_format = 'PNG'")
    add(f"S.frame_start = {frame_list[0]}")
    add(f"S.frame_end = {frame_list[-1]}")
    add("")
    # world: scene background + ambient sum baked per frame
    bg = scene.background
    add("W = bpy.data.worlds.new('World')")
    add("S.world = W")
    add("W.use_nodes = True")
    add("BG = W.node_tree.nodes['Background']")
    add(f"BG.inputs[0].default_value = ({_f(bg[0])}, {_f(bg[1])}, "
        f"{_f(bg[2])}, 1.0)")
    add("BG.inputs[1].default_value = 1.0")
    add(f"BG_BASE = ({_f(bg[0])}, {_f(bg[1])}, {_f(bg[2])})")
    add("")
    # materials, sorted by name (creation order is deterministic)
    add("MATS = {}")
    add("MAT_SOCKS = {}")
    for name in sorted(scene.materials):
        mat = scene.materials[name]
        construction = MATERIAL_MAP[mat.kind]["construction"]
        add(f"# material {name!r}: {mat.kind} -> {construction}"
            f" — {MATERIAL_MAP[mat.kind]['note']}")
        add(f"M = bpy.data.materials.new({name!r})")
        add("M.use_nodes = True")
        add("NT = M.node_tree")
        add("NT.nodes.clear()")
        add("OUT = NT.nodes.new('ShaderNodeOutputMaterial')")
        add("OUT.location = (300, 0)")
        if construction == "emission":
            add("EM = NT.nodes.new('ShaderNodeEmission')")
            add(f"EM.inputs[0].default_value = "
                f"({_f(mat.color[0])}, {_f(mat.color[1])}, "
                f"{_f(mat.color[2])}, 1.0)")
            add("EM.inputs[1].default_value = 1.0")
            if mat.opacity < 1.0:
                add("TR = NT.nodes.new('ShaderNodeBsdfTransparent')")
                add("MIX = NT.nodes.new('ShaderNodeMixShader')")
                add("NT.links.new(TR.outputs[0], MIX.inputs[1])")
                add("NT.links.new(EM.outputs[0], MIX.inputs[2])")
                add("NT.links.new(MIX.outputs[0], OUT.inputs[0])")
                add(f"MAT_SOCKS[{name!r}] = (EM.inputs[0], "
                    f"MIX.inputs[0])")
                add(f"MIX.inputs[0].default_value = "
                    f"{_f(mat.opacity)}")
            else:
                add("NT.links.new(EM.outputs[0], OUT.inputs[0])")
                add(f"MAT_SOCKS[{name!r}] = (EM.inputs[0], None)")
        else:
            add("BS = NT.nodes.new('ShaderNodeBsdfPrincipled')")
            add(f"BS.inputs['Base Color'].default_value = "
                f"({_f(mat.color[0])}, {_f(mat.color[1])}, "
                f"{_f(mat.color[2])}, 1.0)")
            if mat.kind == "lambert":
                add("BS.inputs['Roughness'].default_value = 1.0")
                specular = 0.0
            else:  # phong: GGX roughness from Blinn-Phong shininess
                rough = math.sqrt(2.0 / (mat.shininess + 2.0))
                add(f"BS.inputs['Roughness'].default_value = "
                    f"{_f(rough)}")
                specular = 0.5
            add("for _n in ('Specular IOR Level', 'Specular'):")
            add("    if _n in BS.inputs:")
            add(f"        BS.inputs[_n].default_value = "
                f"{_f(specular)}")
            add("for _n in ('Emission Color', 'Emission'):")
            add("    if _n in BS.inputs:")
            add(f"        BS.inputs[_n].default_value = "
                f"({_f(mat.emissive[0])}, {_f(mat.emissive[1])}, "
                f"{_f(mat.emissive[2])}, 1.0)")
            add("if 'Emission Strength' in BS.inputs:")
            add("    BS.inputs['Emission Strength'].default_value"
                " = 1.0")
            add(f"BS.inputs['Alpha'].default_value = "
                f"{_f(mat.opacity)}")
            add("NT.links.new(BS.outputs[0], OUT.inputs[0])")
            add(f"MAT_SOCKS[{name!r}] = (BS.inputs['Base Color'], "
                f"BS.inputs['Alpha'])")
        add(f"MATS[{name!r}] = M")
        add("")
    # nodes, in document order (fixed by the document)
    add("OBJS = {}")
    add("LDATA = {}")
    for node in scene.nodes:
        add(f"# node {node.name!r} ({node.kind})")
        if node.mesh is not None:
            mesh = node.mesh
            if mesh.kind == "primitive":
                spec = PRIMITIVE_MAP[mesh.primitive]
                kwargs: dict[str, Any] = {}
                for kwarg, (param_spec, default) in \
                        spec["kwargs"].items():
                    value: Any = default
                    for pname in param_spec.split("|"):
                        if pname in mesh.params:
                            value = mesh.params[pname]
                            break
                    if kwarg in ("segments", "ring_count", "vertices"):
                        value = int(value)
                    kwargs[kwarg] = value
                kw_parts = []
                for k, v in kwargs.items():
                    v_text = _f(v) if isinstance(v, float) \
                        else repr(v)
                    kw_parts.append(f"{k}={v_text}")
                kw_text = ", ".join(kw_parts)
                add(f"{spec['op']}({kw_text})")
                add("O = bpy.context.active_object")
            else:
                verts = [list(v) for v in mesh.vertices]
                faces = [list(f) for f in mesh.faces]
                add(f"MD = bpy.data.meshes.new({node.name!r})")
                add(f"MD.from_pydata({verts!r}, [], {faces!r})")
                add("MD.update()")
                if mesh.normals:
                    normals = [list(n) for n in mesh.normals]
                    add("if hasattr(MD, 'normals_split_custom_set"
                        "_from_vertices'):")
                    add(f"    MD.normals_split_custom_set_from"
                        f"_vertices({normals!r})")
                add(f"O = bpy.data.objects.new({node.name!r}, MD)")
                add("bpy.context.collection.objects.link(O)")
            if node.material is not None:
                add(f"O.data.materials.append(MATS[{node.material!r}])")
        elif node.light is not None:
            light = node.light
            btype = LIGHT_MAP[light.kind]["blender"]
            if btype == "WORLD":
                add("O = bpy.data.objects.new("
                    f"{node.name!r}, None)  # ambient: no object — "
                    "world background (see LIGHT_MAP)")
                add("bpy.context.collection.objects.link(O)")
            else:
                add(f"LD = bpy.data.lights.new({node.name!r}, "
                    f"{btype!r})")
                add(f"LD.color = {_vec(light.color)}")
                add(f"LD.energy = {_f(light.intensity)}")
                if light.kind == "spot":
                    add(f"LD.spot_size = "
                        f"{_f(math.radians(light.spot_angle))}")
                    add(f"LD.spot_blend = {_f(light.spot_penumbra)}")
                add(f"O = bpy.data.objects.new({node.name!r}, LD)")
                add("bpy.context.collection.objects.link(O)")
                add(f"LDATA[{node.name!r}] = LD")
        else:
            add(f"O = bpy.data.objects.new({node.name!r}, None)")
            add("bpy.context.collection.objects.link(O)")
        add(f"O.name = {node.name!r}")
        add("O.rotation_mode = 'QUATERNION'")
        add(f"OBJS[{node.name!r}] = O")
        add("")
    # the active camera (the core evaluates through it alone)
    cam = scene.cameras[cam_name]
    add(f"# camera {cam_name!r} (the scene's active camera)")
    add(f"CD = bpy.data.cameras.new({cam_name!r})")
    add("CD.sensor_fit = 'VERTICAL'")
    add("CD.sensor_height = 24.0")
    # Blender's angle is derived, not animatable — the fov rides
    # the focal length: lens = sensor / (2 tan(fov/2)), fov being
    # the VERTICAL angle under sensor_fit='VERTICAL'.
    lens0 = 12.0 / math.tan(math.radians(cam.fov) / 2.0)
    add(f"CD.lens = {_f(lens0)}")
    add(f"CD.clip_start = {_f(cam.near)}")
    add(f"CD.clip_end = {_f(cam.far)}")
    add(f"CAM = bpy.data.objects.new({cam_name!r}, CD)")
    add("bpy.context.collection.objects.link(CAM)")
    add("CAM.rotation_mode = 'QUATERNION'")
    add("S.camera = CAM")
    add("")
    # the bake: per frame, state from the table only
    add("for _f_no in range("
        f"{frame_list[0]}, {frame_list[-1]} + 1):")
    add("    ROW = TABLE[str(_f_no)]")
    add("    for _name, _row in ROW['nodes'].items():")
    add("        _o = OBJS[_name]")
    add("        _set(_o, _row['m'])")
    add("        _o.hide_render = not _row['v']")
    add("        _keys(_o, _f_no)")
    add("        if _name in LDATA and 'e' in _row:")
    add("            LDATA[_name].energy = _row['e']")
    add("            LDATA[_name].keyframe_insert('energy', "
        "frame=_f_no)")
    add("    _amb = [0.0, 0.0, 0.0]")
    add("    for _row in ROW['nodes'].values():")
    add("        if 'amb' in _row:")
    add("            for _i in range(3):")
    add("                _amb[_i] += _row['amb'][_i]")
    add("    BG.inputs[0].default_value = (BG_BASE[0] + _amb[0], "
        "BG_BASE[1] + _amb[1], BG_BASE[2] + _amb[2], 1.0)")
    add("    BG.inputs[0].keyframe_insert('default_value', "
        "frame=_f_no)")
    add("    for _name, _m in ROW['mats'].items():")
    add("        _c, _a = MAT_SOCKS[_name]")
    add("        _c.default_value = (_m['color'][0], "
        "_m['color'][1], _m['color'][2], 1.0)")
    add("        _c.keyframe_insert('default_value', frame=_f_no)")
    add("        if _a is not None:")
    add("            _a.default_value = _m['opacity']")
    add("            _a.keyframe_insert('default_value', "
        "frame=_f_no)")
    add("    _set(CAM, ROW['cam']['m'])")
    add("    _keys(CAM, _f_no)")
    add("    CD.lens = 12.0 / math.tan("
        "math.radians(ROW['cam']['fov']) / 2.0)")
    add("    CD.keyframe_insert('lens', frame=_f_no)")
    add("    CD.clip_start = ROW['cam']['near']")
    add("    CD.keyframe_insert('clip_start', frame=_f_no)")
    add("    CD.clip_end = ROW['cam']['far']")
    add("    CD.keyframe_insert('clip_end', frame=_f_no)")
    add("")
    add("os.makedirs(OUT_DIR, exist_ok=True)")
    add("for _f_no in FRAMES:")
    add("    S.frame_set(_f_no)")
    add("    S.render.filepath = os.path.join("
        "OUT_DIR, 'f_%04d' % _f_no)")
    add("    bpy.ops.render.render(write_still=True)")
    add("    print('DEVON3D_FRAME', _f_no)")
    add(f"print({_DONE_MARKER!r} + json.dumps("
        "{'blender_version': bpy.app.version_string, "
        f"'engine': {engine!r}, 'frames': len(FRAMES)}}))")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the renderer
# ---------------------------------------------------------------------------

class BlenderRenderer:
    """The Blender-class rung: Scene3D documents rendered by a
    headless Blender subprocess (see the module docstring)."""

    name = RENDERER_NAME

    def __init__(self, *, locator: BlenderLocator | None = None,
                 engine: str = "CYCLES", samples: int = 32,
                 timeout: float | None = None) -> None:
        self._locator = locator or default_locator()
        self.engine = engine
        self.samples = int(samples)
        if timeout is None:
            raw = os.environ.get("NM_BLENDER_TIMEOUT", "")
            timeout = float(raw) if raw else 1800.0
        self.timeout = float(timeout)

    def capabilities(self) -> RendererCapabilities:
        animated = sorted(
            {"position", "rotation", "scale", "intensity",
             "fov", "near", "far", "color", "opacity"})
        return RendererCapabilities(
            surfaces=("scene3d",),
            node_kinds=tuple(NODE_KINDS),
            mesh_kinds=("primitive", "explicit"),
            primitives=tuple(PRIMITIVE_MAP),
            material_kinds=tuple(MATERIAL_MAP),
            light_kinds=tuple(LIGHT_MAP),
            animated_props=tuple(animated),
            max_width=0, max_height=0,
            deterministic=False,
            notes="blender-class external engine (headless bpy "
                  "subprocess); animation baked from scene_state_at"
                  "; pixel output varies by blender version/host — "
                  "frame states are the determinism surface. "
                  "Mapping approximations: "
                  + "; ".join(f"{k}: {v['note']}"
                              for k, v in sorted(MATERIAL_MAP.items()))
                  + "; "
                  + "; ".join(f"{k}: {v['note']}"
                              for k, v in sorted(LIGHT_MAP.items())))

    def probe(self) -> RendererProbe:
        return self._locator.probe()

    def render_frames(self, scene: Scene3D, *,
                      frames: Sequence[int],
                      workdir: str | os.PathLike) -> dict[str, Any]:
        frame_list = sorted({int(f) for f in frames})
        if not frame_list:
            raise Scene3DError("blender rung: no frames requested")
        probe = self.probe()
        if not probe.available:
            raise Scene3DError(
                f"blender rung unavailable: {probe.reason}")
        binary = self._locator.find()
        if binary is None:  # probe said yes; the host disagrees
            raise Scene3DError(
                "blender rung: probe reported available but no "
                "binary can be located now")
        out = Path(workdir)
        out.mkdir(parents=True, exist_ok=True)
        frames_dir = out / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        script = generate_bpy_script(
            scene, frame_list, engine=self.engine,
            samples=self.samples, out_dir_name=str(frames_dir))
        script_path = out / "scene3d_blender.py"
        script_path.write_text(script, encoding="utf-8")
        manifest = {
            "renderer": self.name, "scene_hash":
                scene.doc_hash or scene.compute_hash(),
            "frames": frame_list, "frames_dir": str(frames_dir),
            "pattern": "f_%04d.png",
            "width": int(scene.output.get("width", 1920)),
            "height": int(scene.output.get("height", 1080)),
            "framerate": scene.output.get("fps", 30),
            "engine": self.engine, "samples": self.samples}
        manifest_path = out / "render_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8")
        command = [str(binary), "--background", "--python",
                   str(script_path), "--", str(manifest_path)]
        try:
            proc = subprocess.run(
                command, capture_output=True, text=True,
                timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise Scene3DError(
                f"blender rung: render timed out after "
                f"{self.timeout:.0f}s") from exc
        except OSError as exc:
            raise Scene3DError(
                f"blender rung: could not launch {binary}: {exc}"
            ) from exc
        done: dict[str, Any] | None = None
        for line in (proc.stdout or "").splitlines():
            if line.startswith(_DONE_MARKER):
                try:
                    done = json.loads(line[len(_DONE_MARKER):])
                except ValueError:
                    done = None
        tail = "\n".join(
            ((proc.stdout or "") + "\n" + (proc.stderr or "")
             ).splitlines()[-12:])
        if proc.returncode != 0:
            raise Scene3DError(
                f"blender rung: blender exited {proc.returncode}; "
                f"output tail:\n{tail}")
        if done is None:
            raise Scene3DError(
                "blender rung: render produced no completion "
                f"marker; output tail:\n{tail}")
        files: list[str] = []
        for f in frame_list:
            path = frames_dir / f"f_{f:04d}.png"
            if not path.is_file() or path.stat().st_size == 0:
                raise Scene3DError(
                    f"blender rung: frame {f} missing from the "
                    f"render output ({path})")
            with open(path, "rb") as fh:
                if fh.read(8) != _PNG_SIGNATURE:
                    raise Scene3DError(
                        f"blender rung: frame {f} at {path} is "
                        f"not a PNG")
            files.append(path.name)
        return {
            "frames_dir": str(frames_dir),
            "pattern": "f_%04d.png",
            "framerate": scene.output.get("fps", 30),
            "frames": len(files),
            "files": files,
            "width": int(scene.output.get("width", 1920)),
            "height": int(scene.output.get("height", 1080)),
            "blender_version": done.get(
                "blender_version") or self._locator.version(binary),
            "engine": self.engine,
            "settings": {"engine": self.engine,
                         "samples": self.samples,
                         "seed": 0, "denoising": False,
                         "device": "CPU"},
            "script": str(script_path),
            "manifest": str(manifest_path),
            "command": command,
        }


def register_blender_renderer(renderer: BlenderRenderer | None = None
                              ) -> BlenderRenderer:
    """Register the Blender rung (idempotent — engines register at
    import/wiring time; a second registration of the same rung is
    the same rung)."""
    renderer = renderer or BlenderRenderer()
    if renderer.name in registered_renderers():
        existing = get_renderer(renderer.name)
        assert isinstance(existing, BlenderRenderer)
        return existing
    register_renderer(renderer)
    return renderer


# engines register at their own import/wiring time (the registry
# docstring's contract): importing this module makes the rung
# resolvable; its probe decides availability per host.
register_blender_renderer()
