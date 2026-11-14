"""Motion-studio rungs for Devon's web surfaces (Phase 2 port).

Wires four of the motion-studio repos from the @0xMovez post into
Devon as CDN pins + page generators — no new dependency, no React,
no new web framework:

- **Lottie** (airbnb/lottie-web, MIT) — Devon already *generates*
  Lottie (``nomorals/social/chat/emoji_anim.py`` builds Lottie dicts
  + ``.tgs``); :func:`lottie_player_page` closes the loop so her
  pages *play* it.
- **GSAP** (GreenSock, free-including-commercial under GreenSock's
  own standard license — **not** an OSI license; recorded on the
  pin) — the standard animation library for her pages.
- **three.js** (mrdoob, MIT) — the 3D web foundation. A scene spec
  (plain dict) becomes a full three.js page via
  :func:`three_scene_page`; the live host is the dependency-free
  Node server ``bridge/motion-server.mjs``.
- **three.quarks** (Alchemist0823, MIT) + **Tweakpane** (cocopon,
  MIT) — on top of three.js: the particle/VFX system and the live
  parameter-tuning pane for Devon's 3D pages.

All CDN pins are verified live (HTTP 200) at build time; the
versions are pinned, never ``@latest``. The scene-spec schema is
the contract between this module and the Node host — both sides
validate it, and the tests pin the CDN URLs on both sides.
"""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any

__all__ = [
    "MOTION_CDN",
    "MOTION_PORT_DEFAULT",
    "default_scene_spec",
    "gsap_demo_page",
    "lottie_player_page",
    "motion_head",
    "motion_port",
    "three_scene_page",
    "validate_scene_spec",
    "write_motion_page",
]

#: Default port for the motion host (``bridge/motion-server.mjs``).
#: Adjacent to the WhatsApp bridge's 8787; binds 127.0.0.1 only.
#: Overridden by the ``NM_MOTION_PORT`` environment variable.
MOTION_PORT_DEFAULT = 8788

#: Pinned CDN registry — the single source of truth for every motion
#: rung's script URL. ``license`` is the grant the pin relies on;
#: GSAP's is GreenSock's own standard license, free including
#: commercial use, NOT an OSI license — that is recorded, not hidden.
MOTION_CDN: dict[str, dict[str, str]] = {
    "lottie": {
        "url": "https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.13.0/lottie.min.js",
        "version": "5.13.0",
        "license": "MIT",
        "license_note": "airbnb/lottie-web.",
    },
    "gsap": {
        "url": "https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js",
        "version": "3.12.5",
        "license": "GreenSock standard license",
        "license_note": (
            "Free including commercial use under GreenSock's own "
            "standard license (https://gsap.com/standard-license) — "
            "NOT an OSI license."
        ),
    },
    "three": {
        "url": "https://cdn.jsdelivr.net/npm/three@0.187.0/build/three.module.js",
        "version": "0.187.0",
        "license": "MIT",
        "license_note": "mrdoob/three.js r187.",
    },
    "three-addons": {
        "url": "https://cdn.jsdelivr.net/npm/three@0.187.0/examples/jsm/",
        "version": "0.187.0",
        "license": "MIT",
        "license_note": "three.js addons (OrbitControls et al.).",
    },
    "three-quarks": {
        "url": "https://cdn.jsdelivr.net/npm/three.quarks@0.17.1/dist/three.quarks.esm.js",
        "version": "0.17.1",
        "license": "MIT",
        "license_note": (
            "Alchemist0823/three.quarks. Peer three >= 0.182.0 — "
            "the pinned three 0.187.0 satisfies it."
        ),
    },
    "quarks-core": {
        "url": "https://cdn.jsdelivr.net/npm/quarks.core@0.17.1/dist/quarks.core.esm.js",
        "version": "0.17.1",
        "license": "MIT",
        "license_note": "three.quarks' core (imported by its ESM build).",
    },
    "tweakpane": {
        "url": "https://cdn.jsdelivr.net/npm/tweakpane@4.0.5/dist/tweakpane.min.js",
        "version": "4.0.5",
        "license": "MIT",
        "license_note": "cocopon/tweakpane (UMD global ``Tweakpane``).",
    },
}

#: Classic-script (UMD/global) libs — a plain <script src> tag.
_UMD_LIBS = ("lottie", "gsap", "tweakpane")

#: Libs that need the ES-module import map.
_IMPORTMAP_LIBS = ("three", "three-addons", "three-quarks", "quarks-core")

_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_PAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

_SCENE_OBJECT_TYPES = ("box", "sphere", "plane", "cylinder", "torus", "cone")
_SCENE_LIGHT_TYPES = ("ambient", "directional", "point")
_PARTICLE_SHAPES = ("point", "cone")


def motion_port() -> int:
    """The motion host's port (``NM_MOTION_PORT``, else the default)."""
    try:
        return int(os.environ.get("NM_MOTION_PORT", "") or MOTION_PORT_DEFAULT)
    except ValueError:
        return MOTION_PORT_DEFAULT


def motion_head(*libs: str) -> str:
    """HTML ``<head>`` fragment loading the named motion libs.

    UMD libs become plain ``<script src>`` tags; any of the three.js
    family pulls in the ES-module import map (``three``,
    ``three/addons/``, ``quarks.core``, ``three.quarks``). Raises
    :class:`ValueError` naming an unknown lib.
    """
    unknown = [lib for lib in libs if lib not in MOTION_CDN]
    if unknown:
        raise ValueError(
            "unknown motion lib(s): " + ", ".join(unknown)
            + " — known: " + ", ".join(sorted(MOTION_CDN))
        )
    lines: list[str] = []
    if any(lib in _IMPORTMAP_LIBS for lib in libs):
        imports = {
            "three": MOTION_CDN["three"]["url"],
            "three/addons/": MOTION_CDN["three-addons"]["url"],
            "quarks.core": MOTION_CDN["quarks-core"]["url"],
            "three.quarks": MOTION_CDN["three-quarks"]["url"],
        }
        lines.append(
            '<script type="importmap">'
            + json.dumps({"imports": imports})
            + "</script>"
        )
    for lib in libs:
        if lib in _UMD_LIBS:
            lines.append(
                '<script src="' + MOTION_CDN[lib]["url"] + '"></script>'
            )
    return "\n".join(lines)


# ── Lottie ────────────────────────────────────────────────────────

_LOTTIE_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
body { margin: 0; background: #14141c; color: #eee; font-family: system-ui, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; }
#stage { width: __WIDTH__px; height: __HEIGHT__px; }
#controls { display: flex; gap: 8px; margin-top: 16px; align-items: center; }
button { background: #2a2a38; color: #eee; border: 1px solid #555; border-radius: 6px; padding: 8px 14px; cursor: pointer; }
button:hover { background: #3a3a4c; }
select { background: #2a2a38; color: #eee; border: 1px solid #555; border-radius: 6px; padding: 8px; }
#status { margin-top: 10px; font-size: 13px; color: #999; }
</style>
__HEAD__
</head>
<body>
<div id="stage"></div>
<div id="controls">
<button id="play">Pause</button>
<button id="restart">Restart</button>
<label>Speed <select id="speed"><option>0.5</option><option selected>1</option><option>1.5</option><option>2</option></select></label>
</div>
<div id="status">Powered by peacethefirst</div>
<script type="application/json" id="lottie-data">__ANIMATION_JSON__</script>
<script>
var anim = lottie.loadAnimation({
  container: document.getElementById('stage'),
  renderer: 'svg',
  loop: __LOOP__,
  autoplay: __AUTOPLAY__,
  animationData: JSON.parse(document.getElementById('lottie-data').textContent)
});
var playBtn = document.getElementById('play');
playBtn.addEventListener('click', function () {
  if (anim.isPaused) { anim.play(); playBtn.textContent = 'Pause'; }
  else { anim.pause(); playBtn.textContent = 'Play'; }
});
document.getElementById('restart').addEventListener('click', function () {
  anim.goToAndPlay(0); playBtn.textContent = 'Pause';
});
document.getElementById('speed').addEventListener('change', function (e) {
  anim.setSpeed(parseFloat(e.target.value));
});
anim.addEventListener('DOMLoaded', function () {
  document.getElementById('status').textContent = 'Playing — Powered by peacethefirst';
});
</script>
</body>
</html>
"""


def lottie_player_page(animation: dict[str, Any], *, title: str = "Lottie",
                       autoplay: bool = True, loop: bool = True,
                       speed: float = 1.0, width: int = 300,
                       height: int = 300) -> str:
    """A standalone HTML page that plays one Lottie animation dict.

    The animation is embedded as JSON (never a URL fetch), the
    lottie-web player comes from the pinned CDN, and the page carries
    play/pause, restart and speed controls. Raises
    :class:`ValueError` when ``animation`` is not a Lottie
    composition (a dict with a ``layers`` list).
    """
    if not isinstance(animation, dict) or not isinstance(
            animation.get("layers"), list):
        raise ValueError(
            "not a Lottie composition: expected a dict with a "
            "'layers' list"
        )
    if width <= 0 or height <= 0:
        raise ValueError("player width/height must be positive")
    blob = json.dumps(animation, ensure_ascii=False).replace("</", "<\\/")
    page = _LOTTIE_PAGE
    page = page.replace("__TITLE__", html.escape(title))
    page = page.replace("__HEAD__", motion_head("lottie"))
    page = page.replace("__WIDTH__", str(width))
    page = page.replace("__HEIGHT__", str(height))
    page = page.replace("__LOOP__", "true" if loop else "false")
    page = page.replace("__AUTOPLAY__", "true" if autoplay else "false")
    page = page.replace("__ANIMATION_JSON__", blob)
    return page


# ── GSAP ──────────────────────────────────────────────────────────

_GSAP_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
body { margin: 0; background: #14141c; color: #eee; font-family: system-ui, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; overflow: hidden; }
h1 { font-size: 64px; margin: 0; letter-spacing: 2px; }
h1 span { display: inline-block; }
p { color: #999; font-size: 18px; }
button { background: #2a2a38; color: #eee; border: 1px solid #555; border-radius: 6px; padding: 10px 18px; cursor: pointer; margin-top: 24px; }
button:hover { background: #3a3a4c; }
#cue { position: fixed; bottom: 24px; color: #666; font-size: 13px; }
</style>
__HEAD__
</head>
<body>
<h1 id="hero"><span>D</span><span>e</span><span>v</span><span>o</span><span>n</span></h1>
<p id="sub">motion, by Devon</p>
<button id="replay">Replay</button>
<div id="cue">Powered by peacethefirst</div>
<script>
function playIntro() {
  gsap.fromTo('#hero span', { y: 80, opacity: 0, rotate: 12 },
    { y: 0, opacity: 1, rotate: 0, duration: 0.7, stagger: 0.07, ease: 'back.out(1.7)' });
  gsap.fromTo('#sub', { opacity: 0 }, { opacity: 1, duration: 0.6, delay: 0.5 });
  gsap.fromTo('#replay', { opacity: 0 }, { opacity: 1, duration: 0.4, delay: 0.7 });
}
playIntro();
document.getElementById('replay').addEventListener('click', playIntro);
</script>
</body>
</html>
"""


def gsap_demo_page(*, title: str = "GSAP demo") -> str:
    """A standalone HTML demo page for the GSAP rung.

    The hero letters stagger in with a back-out ease and the Replay
    button re-runs the intro — the whole point of the page is to
    prove the GSAP pin loads and animates, so a broken CDN shows up
    as a static page instead of a silent pass.
    """
    page = _GSAP_PAGE
    page = page.replace("__TITLE__", html.escape(title))
    page = page.replace("__HEAD__", motion_head("gsap"))
    return page


# ── three.js scene spec ───────────────────────────────────────────

#: Scene-spec schema (the contract between this module and the Node
#: host — both validate it). All colours are ``#rrggbb``; all
#: positions/rotations are ``[x, y, z]`` numbers.
#:
#: {
#:   "camera": {"position": [0, 3, 8], "fov": 50},
#:   "background": "#101018",
#:   "objects": [{"type": "box", "size": [1, 1, 1] | 1,
#:                "position": [0, 0, 0], "rotation": [0, 0, 0],
#:                "color": "#ffcc33", "roughness": 0.4,
#:                "metalness": 0.2,
#:                "spin": {"x": 0, "y": 0.6, "z": 0}}],
#:   "lights": [{"type": "ambient"|"directional"|"point",
#:               "color": "#ffffff", "intensity": 1.0,
#:               "position": [5, 8, 4]}],
#:   "particles": {"rate": 40, "life": [0.8, 1.6], "speed": 2.0,
#:                 "size": 0.25, "color": "#ffcc33",
#:                 "position": [0, 0.5, 0], "shape": "point"|"cone",
#:                 "max": 500, "looping": true, "duration": 2.0},
#:   "tweak": {"speed": 1.0, "particleRate": 40},
#:   "controls": {"orbit": true, "autoRotate": false},
#: }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_number(value: Any, field: str) -> float:
    if not _is_number(value):
        raise ValueError("scene spec: " + field + " must be a number")
    return float(value)


def _check_vec3(value: Any, field: str) -> list[float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or not all(_is_number(v) for v in value)):
        raise ValueError(
            "scene spec: " + field + " must be [x, y, z] numbers")
    return [float(v) for v in value]


def _check_color(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HEX_COLOR_RE.match(value):
        raise ValueError(
            "scene spec: " + field + " must be a #rrggbb colour")
    return value


def _check_size(obj_type: str, value: Any, field: str) -> Any:
    if obj_type in ("box", "plane"):
        want = 3 if obj_type == "box" else 2
        if _is_number(value):
            return [float(value)] * want
        if (not isinstance(value, (list, tuple)) or len(value) != want
                or not all(_is_number(v) for v in value)):
            raise ValueError(
                "scene spec: " + field + " must be a list of "
                + str(want) + " numbers for a " + obj_type)
        return [float(v) for v in value]
    if obj_type in ("cylinder", "torus"):
        if _is_number(value):
            return [float(value), float(value)]
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or not all(_is_number(v) for v in value)):
            raise ValueError(
                "scene spec: " + field + " must be [a, b] numbers "
                "for a " + obj_type)
        return [float(v) for v in value]
    return _check_number(value, field)


def validate_scene_spec(spec: Any) -> dict[str, Any]:
    """Validate a scene spec and return a normalised copy.

    Defaults are filled in; every bad field raises
    :class:`ValueError` naming it. The input dict is never mutated.
    """
    if not isinstance(spec, dict):
        raise ValueError("scene spec: expected a dict")
    out: dict[str, Any] = {}

    camera = spec.get("camera") or {}
    if not isinstance(camera, dict):
        raise ValueError("scene spec: 'camera' must be a dict")
    position = _check_vec3(camera.get("position", [0, 3, 8]), "camera.position")
    fov = _check_number(camera.get("fov", 50), "camera.fov")
    if not 10 <= fov <= 120:
        raise ValueError("scene spec: camera.fov must be 10..120")
    out["camera"] = {"position": position, "fov": fov}

    out["background"] = _check_color(
        spec.get("background", "#101018"), "background")

    objects: list[dict[str, Any]] = []
    raw_objects = spec.get("objects") or []
    if not isinstance(raw_objects, list):
        raise ValueError("scene spec: 'objects' must be a list")
    for i, raw in enumerate(raw_objects):
        field = "objects[" + str(i) + "]"
        if not isinstance(raw, dict):
            raise ValueError("scene spec: " + field + " must be a dict")
        obj_type = raw.get("type")
        if obj_type not in _SCENE_OBJECT_TYPES:
            raise ValueError(
                "scene spec: " + field + ".type must be one of "
                + ", ".join(_SCENE_OBJECT_TYPES))
        size = _check_size(obj_type, raw.get("size", 1), field + ".size")
        spin: dict[str, float] | bool = False
        raw_spin = raw.get("spin")
        if raw_spin is True:
            spin = {"x": 0.0, "y": 0.6, "z": 0.0}
        elif isinstance(raw_spin, dict):
            spin = {
                "x": _check_number(raw_spin.get("x", 0), field + ".spin.x"),
                "y": _check_number(raw_spin.get("y", 0), field + ".spin.y"),
                "z": _check_number(raw_spin.get("z", 0), field + ".spin.z"),
            }
        elif raw_spin not in (None, False):
            raise ValueError(
                "scene spec: " + field + ".spin must be true/false "
                "or an {x, y, z} dict")
        objects.append({
            "type": obj_type,
            "size": size,
            "position": _check_vec3(raw.get("position", [0, 0, 0]),
                                    field + ".position"),
            "rotation": _check_vec3(raw.get("rotation", [0, 0, 0]),
                                    field + ".rotation"),
            "color": _check_color(raw.get("color", "#ffffff"),
                                  field + ".color"),
            "roughness": _check_number(raw.get("roughness", 0.6),
                                       field + ".roughness"),
            "metalness": _check_number(raw.get("metalness", 0.1),
                                       field + ".metalness"),
            "spin": spin,
        })
    out["objects"] = objects

    lights: list[dict[str, Any]] = []
    raw_lights = spec.get("lights") or []
    if not isinstance(raw_lights, list):
        raise ValueError("scene spec: 'lights' must be a list")
    for i, raw in enumerate(raw_lights):
        field = "lights[" + str(i) + "]"
        if not isinstance(raw, dict):
            raise ValueError("scene spec: " + field + " must be a dict")
        light_type = raw.get("type")
        if light_type not in _SCENE_LIGHT_TYPES:
            raise ValueError(
                "scene spec: " + field + ".type must be one of "
                + ", ".join(_SCENE_LIGHT_TYPES))
        light: dict[str, Any] = {
            "type": light_type,
            "color": _check_color(raw.get("color", "#ffffff"),
                                  field + ".color"),
            "intensity": _check_number(raw.get("intensity", 1.0),
                                       field + ".intensity"),
        }
        if light_type in ("directional", "point"):
            light["position"] = _check_vec3(
                raw.get("position", [5, 8, 4]), field + ".position")
        if light_type == "point":
            light["distance"] = _check_number(
                raw.get("distance", 0), field + ".distance")
        lights.append(light)
    if not lights:
        lights = [{"type": "ambient", "color": "#ffffff",
                   "intensity": 0.7},
                  {"type": "directional", "color": "#ffffff",
                   "intensity": 1.2, "position": [5, 8, 4]}]
    out["lights"] = lights

    raw_particles = spec.get("particles")
    if raw_particles is None:
        out["particles"] = None
    else:
        if not isinstance(raw_particles, dict):
            raise ValueError("scene spec: 'particles' must be a dict")
        shape = raw_particles.get("shape", "point")
        if shape not in _PARTICLE_SHAPES:
            raise ValueError(
                "scene spec: particles.shape must be one of "
                + ", ".join(_PARTICLE_SHAPES))
        life = raw_particles.get("life", [0.8, 1.6])
        if (not isinstance(life, (list, tuple)) or len(life) != 2
                or not all(_is_number(v) for v in life)):
            raise ValueError(
                "scene spec: particles.life must be [min, max] numbers")
        out["particles"] = {
            "rate": _check_number(raw_particles.get("rate", 40),
                                  "particles.rate"),
            "life": [float(life[0]), float(life[1])],
            "speed": _check_number(raw_particles.get("speed", 2.0),
                                   "particles.speed"),
            "size": _check_number(raw_particles.get("size", 0.25),
                                  "particles.size"),
            "color": _check_color(raw_particles.get("color", "#ffcc33"),
                                  "particles.color"),
            "position": _check_vec3(
                raw_particles.get("position", [0, 0.5, 0]),
                "particles.position"),
            "shape": shape,
            "max": int(_check_number(raw_particles.get("max", 500),
                                     "particles.max")),
            "looping": bool(raw_particles.get("looping", True)),
            "duration": _check_number(
                raw_particles.get("duration", 2.0), "particles.duration"),
        }

    raw_tweak = spec.get("tweak") or {}
    if not isinstance(raw_tweak, dict):
        raise ValueError("scene spec: 'tweak' must be a dict")
    tweak: dict[str, Any] = {}
    for key, value in raw_tweak.items():
        if not isinstance(key, str) or not key:
            raise ValueError("scene spec: tweak keys must be non-empty strings")
        if not (_is_number(value) or isinstance(value, (bool, str))):
            raise ValueError(
                "scene spec: tweak." + key
                + " must be a number, string or boolean")
        tweak[key] = value
    if "speed" not in tweak:
        tweak["speed"] = 1.0
    out["tweak"] = tweak

    raw_controls = spec.get("controls") or {}
    if not isinstance(raw_controls, dict):
        raise ValueError("scene spec: 'controls' must be a dict")
    out["controls"] = {
        "orbit": bool(raw_controls.get("orbit", True)),
        "autoRotate": bool(raw_controls.get("autoRotate", False)),
    }
    return out


def default_scene_spec() -> dict[str, Any]:
    """The demo scene: a spinning gold box over a dark plane, lit
    warm, with a gold particle fountain — and a Tweakpane speed +
    particle-rate pair to play with."""
    return validate_scene_spec({
        "camera": {"position": [0, 2.5, 7], "fov": 50},
        "background": "#101018",
        "objects": [
            {"type": "box", "size": [1.6, 1.6, 1.6],
             "position": [0, 1.2, 0], "color": "#ffcc33",
             "roughness": 0.35, "metalness": 0.4,
             "spin": {"x": 0.2, "y": 0.6, "z": 0.0}},
            {"type": "plane", "size": [12, 12],
             "position": [0, 0, 0], "rotation": [-1.5708, 0, 0],
             "color": "#1c1c28", "roughness": 0.9, "metalness": 0.0},
        ],
        "lights": [
            {"type": "ambient", "color": "#ffffff", "intensity": 0.5},
            {"type": "directional", "color": "#fff2cc",
             "intensity": 1.4, "position": [5, 8, 4]},
            {"type": "point", "color": "#ffcc33", "intensity": 12,
             "position": [0, 3, 2], "distance": 12},
        ],
        "particles": {"rate": 40, "life": [0.8, 1.6], "speed": 2.0,
                      "size": 0.25, "color": "#ffcc33",
                      "position": [0, 0.2, 0], "shape": "cone",
                      "max": 500, "looping": True, "duration": 2.0},
        "tweak": {"speed": 1.0, "particleRate": 40},
        "controls": {"orbit": True, "autoRotate": False},
    })


#: The generic three.js scene builder. It reads the embedded SPEC
#: (a validated scene-spec dict) and PARAMS (the tweak dict) and
#: nothing else — the same builder runs in the Python-generated
#: pages and in the Node host (``bridge/motion-server.mjs`` mirrors
#: it; the tests pin the shared markers on both sides). Written
#: against the verified APIs: three r187, three.quarks 0.17.1
#: (``ParticleSystem`` + ``BatchedParticleRenderer`` +
#: ``ConstantValue``/``IntervalValue``/``ConstantColor`` +
#: ``PointEmitter``/``ConeEmitter``) and Tweakpane v4
#: (``pane.addBinding`` + ``binding.on('change')``).
THREE_BUILDER_JS = """
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  ParticleSystem, BatchedParticleRenderer, ConstantValue, IntervalValue,
  ConstantColor, PointEmitter, ConeEmitter,
} from 'three.quarks';

const SPEC = __SPEC_JSON__;
const PARAMS = __PARAMS_JSON__;

const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
const scene = new THREE.Scene();
scene.background = new THREE.Color(SPEC.background);
const camera = new THREE.PerspectiveCamera(
  SPEC.camera.fov, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(SPEC.camera.position[0], SPEC.camera.position[1], SPEC.camera.position[2]);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = !!SPEC.controls.autoRotate;

const spinners = [];
function makeGeometry(o) {
  const s = o.size;
  if (o.type === 'box') return new THREE.BoxGeometry(s[0], s[1], s[2]);
  if (o.type === 'sphere') return new THREE.SphereGeometry(s, 32, 24);
  if (o.type === 'plane') return new THREE.PlaneGeometry(s[0], s[1]);
  if (o.type === 'cylinder') return new THREE.CylinderGeometry(s[0], s[1], s[2], 32);
  if (o.type === 'torus') return new THREE.TorusGeometry(s[0], s[1], 16, 64);
  if (o.type === 'cone') return new THREE.ConeGeometry(s[0], s[1], 32);
  throw new Error('unknown object type ' + o.type);
}
for (const o of SPEC.objects) {
  const mesh = new THREE.Mesh(
    makeGeometry(o),
    new THREE.MeshStandardMaterial({ color: o.color, roughness: o.roughness, metalness: o.metalness }));
  mesh.position.set(o.position[0], o.position[1], o.position[2]);
  mesh.rotation.set(o.rotation[0], o.rotation[1], o.rotation[2]);
  if (o.spin) { mesh.userData.spin = o.spin; spinners.push(mesh); }
  scene.add(mesh);
}
for (const l of SPEC.lights) {
  let light;
  if (l.type === 'ambient') light = new THREE.AmbientLight(l.color, l.intensity);
  else if (l.type === 'directional') light = new THREE.DirectionalLight(l.color, l.intensity);
  else light = new THREE.PointLight(l.color, l.intensity, l.distance || 0);
  if (l.position) light.position.set(l.position[0], l.position[1], l.position[2]);
  scene.add(light);
}

let particleSystem = null;
let batchRenderer = null;
if (SPEC.particles) {
  const p = SPEC.particles;
  const c = new THREE.Color(p.color);
  batchRenderer = new BatchedParticleRenderer();
  scene.add(batchRenderer);
  particleSystem = new ParticleSystem({
    duration: p.duration, looping: p.looping,
    startLife: new IntervalValue(p.life[0], p.life[1]),
    startSpeed: new ConstantValue(p.speed),
    startSize: new ConstantValue(p.size),
    startColor: new ConstantColor(new THREE.Vector4(c.r, c.g, c.b, 1)),
    worldSpace: true,
    maxParticle: p.max,
    emissionOverTime: new ConstantValue(p.rate),
    shape: p.shape === 'cone' ? new ConeEmitter({ angle: 0.35, radius: 0.2 }) : new PointEmitter(),
    material: new THREE.MeshBasicMaterial({ color: p.color, transparent: true, opacity: 0.9 }),
  });
  batchRenderer.addSystem(particleSystem);
  particleSystem.emitter.position.set(p.position[0], p.position[1], p.position[2]);
  scene.add(particleSystem.emitter);
}

const pane = new window.Tweakpane.Pane({ title: 'Scene' });
for (const [key, value] of Object.entries(PARAMS)) {
  let binding;
  if (typeof value === 'number') {
    binding = pane.addBinding(PARAMS, key, { min: 0, max: Math.max(1, value * 3), step: 0.01 });
  } else if (typeof value === 'boolean') {
    binding = pane.addBinding(PARAMS, key);
  } else if (typeof value === 'string' && /^#[0-9a-fA-F]{6}$/.test(value)) {
    binding = pane.addBinding(PARAMS, key, { view: 'color' });
  } else {
    binding = pane.addBinding(PARAMS, key);
  }
  if (key === 'particleRate' && particleSystem) {
    binding.on('change', (ev) => {
      try { particleSystem.emissionOverTime.value = ev.value; } catch (e) {}
    });
  }
  if (key === 'background') {
    binding.on('change', (ev) => {
      try { scene.background = new THREE.Color(ev.value); } catch (e) {}
    });
  }
}

const clock = new THREE.Clock();
function tick() {
  const delta = clock.getDelta();
  const speed = typeof PARAMS.speed === 'number' ? PARAMS.speed : 1;
  for (const m of spinners) {
    const s = m.userData.spin;
    m.rotation.x += (s.x || 0) * delta * speed;
    m.rotation.y += (s.y || 0) * delta * speed;
    m.rotation.z += (s.z || 0) * delta * speed;
  }
  if (batchRenderer) batchRenderer.update(delta);
  controls.update();
  renderer.render(scene, camera);
}
renderer.setAnimationLoop(tick);
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
"""

_THREE_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
body { margin: 0; background: #101018; color: #eee; font-family: system-ui, sans-serif; overflow: hidden; }
#scene { display: block; width: 100vw; height: 100vh; }
#hint { position: fixed; left: 16px; bottom: 14px; font-size: 13px; color: #999; }
#hint b { color: #eee; }
</style>
__HEAD__
</head>
<body>
<canvas id="scene"></canvas>
<div id="hint"><b>Devon 3D</b> — drag to orbit, scroll to zoom. Powered by peacethefirst</div>
<script type="module">
__BUILDER_JS__
</script>
</body>
</html>
"""


def three_scene_page(spec: dict[str, Any], *,
                     title: str = "Devon 3D scene") -> str:
    """A scene spec → a standalone three.js HTML page.

    The page carries the import map (three + addons + quarks.core +
    three.quarks), the Tweakpane UMD script, and the generic scene
    builder with the validated spec embedded. Raises
    :class:`ValueError` on a bad spec (see
    :func:`validate_scene_spec`).
    """
    valid = validate_scene_spec(spec)
    builder = THREE_BUILDER_JS
    builder = builder.replace(
        "__SPEC_JSON__", json.dumps(valid, ensure_ascii=False))
    builder = builder.replace(
        "__PARAMS_JSON__", json.dumps(valid["tweak"], ensure_ascii=False))
    page = _THREE_PAGE
    page = page.replace("__TITLE__", html.escape(title))
    page = page.replace(
        "__HEAD__",
        motion_head("three", "three-quarks", "tweakpane"))
    page = page.replace("__BUILDER_JS__", builder)
    return page


def write_motion_page(html_text: str, name: str, *,
                      out_dir: str | Path | None = None) -> Path:
    """Write a generated motion page to disk and return its path.

    ``name`` is a bare page name (letters, digits, ``-``/``_`` — no
    separators, no extension); the file lands as ``<name>.html``
    under ``~/.nomorals/motion/`` unless ``out_dir`` is given.
    """
    if not _PAGE_NAME_RE.match(name):
        raise ValueError(
            "page name must be a bare name (letters, digits, -/_): "
            + repr(name))
    base = (Path(out_dir) if out_dir is not None
            else Path.home() / ".nomorals" / "motion")
    base.mkdir(parents=True, exist_ok=True)
    path = base / (name + ".html")
    path.write_text(html_text, encoding="utf-8")
    return path
