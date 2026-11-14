"""Motion-studio rungs (Phase 2 port): Lottie player, GSAP, three.js,
three.quarks + Tweakpane — CDN pins, page generators, scene-spec
validation, and the Node motion host.
"""

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from nomorals.media import motion_rungs as mr

REPO_ROOT = Path(__file__).resolve().parent.parent
MOTION_SERVER = REPO_ROOT / "bridge" / "motion-server.mjs"


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class CdnRegistryTests(unittest.TestCase):
    def test_every_rung_pinned(self):
        for lib in ("lottie", "gsap", "three", "three-addons",
                    "three-quarks", "quarks-core", "tweakpane"):
            with self.subTest(lib=lib):
                pin = mr.MOTION_CDN[lib]
                self.assertTrue(pin["url"].startswith("https://"),
                                pin["url"])
                self.assertTrue(pin["version"])
                self.assertTrue(pin["license"])

    def test_gsap_license_is_recorded_as_not_osi(self):
        pin = mr.MOTION_CDN["gsap"]
        self.assertIn("GreenSock", pin["license"])
        self.assertIn("NOT an OSI", pin["license_note"])

    def test_three_satisfies_quarks_peer_range(self):
        # three.quarks 0.17.1 needs three >= 0.182.0.
        three = mr.MOTION_CDN["three"]["version"]
        major, minor, _patch = (int(p) for p in three.split("."))
        self.assertEqual(major, 0)
        self.assertGreaterEqual(minor, 182)

    def test_motion_head_unknown_lib_names_it(self):
        with self.assertRaisesRegex(ValueError, "frobnicate"):
            mr.motion_head("frobnicate")

    def test_motion_head_umd_and_importmap(self):
        head = mr.motion_head("lottie", "gsap")
        self.assertIn(mr.MOTION_CDN["lottie"]["url"], head)
        self.assertIn(mr.MOTION_CDN["gsap"]["url"], head)
        self.assertNotIn("importmap", head)
        head3 = mr.motion_head("three", "three-quarks", "tweakpane")
        self.assertIn("importmap", head3)
        self.assertIn('"three"', head3)
        self.assertIn("three/addons/", head3)
        self.assertIn("quarks.core", head3)
        self.assertIn("three.quarks", head3)
        self.assertIn(mr.MOTION_CDN["tweakpane"]["url"], head3)


class LottiePlayerTests(unittest.TestCase):
    def test_page_plays_a_composition(self):
        animation = {"v": "5.7.4", "fr": 60, "ip": 0, "op": 120,
                     "w": 100, "h": 100, "nm": "test-anim",
                     "layers": [{"nm": "one"}]}
        page = mr.lottie_player_page(animation, title="Test anim")
        self.assertIn(mr.MOTION_CDN["lottie"]["url"], page)
        self.assertIn("lottie.loadAnimation", page)
        self.assertIn('"nm": "test-anim"', page)
        self.assertIn("Test anim", page)
        self.assertIn('id="play"', page)
        self.assertIn('id="restart"', page)
        self.assertIn('id="speed"', page)

    def test_page_rejects_non_composition(self):
        with self.assertRaisesRegex(ValueError, "Lottie composition"):
            mr.lottie_player_page({"v": "5.7.4"})
        with self.assertRaisesRegex(ValueError, "Lottie composition"):
            mr.lottie_player_page([{"layers": []}])

    def test_closes_devons_own_lottie_loop(self):
        # Devon GENERATES Lottie (emoji_anim.py) — the player page
        # must play exactly what her generator emits.
        from nomorals.social.chat.emoji_anim import lottie_for_design
        animation = lottie_for_design({"palette": "gold"})
        self.assertEqual(animation["nm"], "devon-emoji")
        page = mr.lottie_player_page(animation)
        self.assertIn('"nm": "devon-emoji"', page)
        self.assertIn("lottie.loadAnimation", page)


class GsapDemoTests(unittest.TestCase):
    def test_page_animates(self):
        page = mr.gsap_demo_page(title="GSAP check")
        self.assertIn(mr.MOTION_CDN["gsap"]["url"], page)
        self.assertIn("gsap.fromTo", page)
        self.assertIn("GSAP check", page)
        self.assertIn('id="replay"', page)


class SceneSpecTests(unittest.TestCase):
    def test_default_spec_validates(self):
        spec = mr.default_scene_spec()
        again = mr.validate_scene_spec(spec)
        self.assertEqual(again, spec)

    def test_defaults_are_filled(self):
        spec = mr.validate_scene_spec({})
        self.assertEqual(spec["camera"]["position"], [0, 3, 8])
        self.assertEqual(spec["background"], "#101018")
        self.assertEqual(spec["objects"], [])
        self.assertEqual(len(spec["lights"]), 2)
        self.assertIsNone(spec["particles"])
        self.assertEqual(spec["tweak"]["speed"], 1.0)
        self.assertTrue(spec["controls"]["orbit"])

    def test_input_is_not_mutated(self):
        raw = {"objects": [{"type": "box"}]}
        mr.validate_scene_spec(raw)
        self.assertEqual(raw, {"objects": [{"type": "box"}]})

    def test_bad_specs_name_the_field(self):
        cases = [
            ("not a dict", "expected a dict"),
            ({"camera": {"fov": 5}}, "camera.fov"),
            ({"camera": {"position": [0, 1]}}, "camera.position"),
            ({"background": "red"}, "background"),
            ({"objects": [{"type": "pyramid"}]}, "objects\\[0\\].type"),
            ({"objects": [{"type": "box", "color": "gold"}]},
             "objects\\[0\\].color"),
            ({"objects": [{"type": "sphere", "size": "big"}]},
             "objects\\[0\\].size"),
            ({"lights": [{"type": "laser"}]}, "lights\\[0\\].type"),
            ({"particles": {"shape": "spiral"}}, "particles.shape"),
            ({"particles": {"life": [1]}}, "particles.life"),
            ({"tweak": {"speed": [1]}}, "tweak.speed"),
        ]
        for spec, pattern in cases:
            with self.subTest(spec=spec), self.assertRaisesRegex(
                    ValueError, pattern):
                mr.validate_scene_spec(spec)


class ThreeScenePageTests(unittest.TestCase):
    def test_builder_js_parses(self):
        # The embedded scene builder is real JavaScript — catch a
        # typo here, not in a browser. (Skipped when node is absent.)
        if not shutil.which("node"):
            self.skipTest("node not on PATH")
        builder = mr.THREE_BUILDER_JS.replace(
            "__SPEC_JSON__", json.dumps(mr.default_scene_spec()))
        builder = builder.replace(
            "__PARAMS_JSON__", json.dumps({"speed": 1.0}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "builder.mjs"
            path.write_text(builder, encoding="utf-8")
            proc = subprocess.run(
                ["node", "--check", str(path)],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_default_scene_page(self):
        page = mr.three_scene_page(mr.default_scene_spec())
        # Import map: three + addons + quarks.
        self.assertIn('"three"', page)
        self.assertIn("three/addons/", page)
        self.assertIn("quarks.core", page)
        self.assertIn("three.quarks", page)
        self.assertIn(mr.MOTION_CDN["three"]["url"], page)
        self.assertIn(mr.MOTION_CDN["three-quarks"]["url"], page)
        self.assertIn(mr.MOTION_CDN["tweakpane"]["url"], page)
        # The scene's own content is embedded, not a fetch.
        self.assertIn("#ffcc33", page)
        self.assertIn("OrbitControls", page)
        self.assertIn("BatchedParticleRenderer", page)
        self.assertIn("ParticleSystem", page)
        self.assertIn("ConeEmitter", page)
        self.assertIn("Tweakpane.Pane", page)
        self.assertIn("addBinding", page)
        self.assertIn('"speed": 1.0', page)

    def test_no_particles_means_no_particle_system(self):
        # The generic builder is always present; what changes is the
        # embedded SPEC — no particles means a null particles entry.
        page = mr.three_scene_page({"objects": [{"type": "sphere"}]})
        self.assertIn('"particles": null', page)
        self.assertIn('"type": "sphere"', page)

    def test_every_object_type_generates(self):
        for obj_type in ("box", "sphere", "plane", "cylinder",
                         "torus", "cone"):
            with self.subTest(obj_type=obj_type):
                page = mr.three_scene_page(
                    {"objects": [{"type": obj_type}]})
                self.assertIn(f'"type": "{obj_type}"', page)

    def test_bad_spec_raises_before_any_page(self):
        with self.assertRaisesRegex(ValueError, "objects"):
            mr.three_scene_page({"objects": [{"type": "nope"}]})

    def test_page_is_standalone_html(self):
        page = mr.three_scene_page(mr.default_scene_spec())
        self.assertTrue(page.startswith("<!DOCTYPE html>"))
        self.assertIn("</html>", page)


class WriteMotionPageTests(unittest.TestCase):
    def test_writes_and_returns_the_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = mr.write_motion_page(
                mr.gsap_demo_page(), "gsap-demo", out_dir=tmp)
            self.assertEqual(path, Path(tmp) / "gsap-demo.html")
            self.assertIn("gsap.fromTo", path.read_text(encoding="utf-8"))

    def test_name_sanitisation_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ("../escape", "a/b", ".hidden", "with space"):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    mr.write_motion_page("<html/>", bad, out_dir=tmp)


@unittest.skipUnless(shutil.which("node"), "node not on PATH")
class MotionServerTests(unittest.TestCase):
    """The Node host rung, live: spawn it on a scratch port and
    exercise every route over real HTTP."""

    def setUp(self):
        self.port = _free_port()
        env = dict(os.environ, NM_MOTION_PORT=str(self.port))
        self.proc = subprocess.Popen(
            ["node", str(MOTION_SERVER)], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.base = f"http://127.0.0.1:{self.port}"
        deadline = time.time() + 20
        while True:
            try:
                self._get("/health")
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.2)

    def tearDown(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8")

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8")

    def test_health(self):
        status, body = self._get("/health")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertTrue(data["ok"])
        self.assertEqual(data["service"], "devon-motion")
        self.assertEqual(data["port"], self.port)

    def test_index_lists_the_rungs(self):
        status, body = self._get("/")
        self.assertEqual(status, 200)
        for route in ("/lottie", "/gsap", "/three"):
            self.assertIn(route, body)
        self.assertIn("GreenSock standard license", body)

    def test_lottie_route_serves_the_player(self):
        status, body = self._get("/lottie")
        self.assertEqual(status, 200)
        self.assertIn(mr.MOTION_CDN["lottie"]["url"], body)
        self.assertIn("lottie.loadAnimation", body)
        self.assertIn("devon-pulse", body)

    def test_gsap_route_serves_the_demo(self):
        status, body = self._get("/gsap")
        self.assertEqual(status, 200)
        self.assertIn(mr.MOTION_CDN["gsap"]["url"], body)
        self.assertIn("gsap.fromTo", body)

    def test_three_route_serves_the_viewer(self):
        status, body = self._get("/three")
        self.assertEqual(status, 200)
        self.assertIn(mr.MOTION_CDN["three"]["url"], body)
        self.assertIn(mr.MOTION_CDN["three-quarks"]["url"], body)
        self.assertIn(mr.MOTION_CDN["tweakpane"]["url"], body)
        self.assertIn("BatchedParticleRenderer", body)
        self.assertIn("Tweakpane.Pane", body)
        self.assertIn("#ffcc33", body)

    def test_three_route_with_a_spec(self):
        import urllib.parse
        spec = {"objects": [{"type": "sphere", "color": "#00ff88"}],
                "tweak": {"speed": 2.0}}
        path = "/three?scene=" + urllib.parse.quote(json.dumps(spec))
        status, body = self._get(path)
        self.assertEqual(status, 200)
        self.assertIn("#00ff88", body)
        self.assertIn("SphereGeometry", body)

    def test_three_route_rejects_a_bad_spec(self):
        import urllib.parse
        spec = {"objects": [{"type": "pyramid"}]}
        path = "/three?scene=" + urllib.parse.quote(json.dumps(spec))
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get(path)
        self.assertEqual(ctx.exception.code, 400)
        self.assertIn("objects[0].type",
                      ctx.exception.read().decode("utf-8"))

    def test_post_scene_renders_a_page(self):
        status, body = self._post(
            "/scene", {"objects": [{"type": "torus", "color": "#8855ff"}]})
        self.assertEqual(status, 200)
        self.assertIn("TorusGeometry", body)
        self.assertIn("#8855ff", body)

    def test_served_three_page_js_parses(self):
        # The page the host actually serves must be valid JS — this
        # catches a typo in the Node-side builder, not in a browser.
        import re
        _status, body = self._get("/three")
        match = re.search(
            r'<script type="module">(.*?)</script>', body, re.DOTALL)
        self.assertIsNotNone(match, "no module script in /three page")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "served.mjs"
            path.write_text(match.group(1), encoding="utf-8")
            proc = subprocess.run(
                ["node", "--check", str(path)],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_unknown_route_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/nope")
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
