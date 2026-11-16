"""Gap R2-25: 3D run/steer wiring — tools + chat controls behind GPU gates.

Failing-first: the three wrappers (``media.trellis.image_to_3d``,
``media.unirig.rig_mesh``, ``media.blender_live.BlenderLiveSession``)
exist and are tested, but nothing calls them — no tool, no chat command.
These tests pin the wiring: tool registration, the GPU gate firing before
the wrapper, the live-Blender connect honesty, the command registration,
and the chat controls reaching the tools.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

NO_GPU = {"has_gpu": False, "name": "", "vram_gb": 0.0, "source": "",
          "detail": "no NVIDIA GPU detected on this box",
          "forward_url": "", "attested": False}
BIG_GPU = {"has_gpu": True, "name": "RTX 4090", "vram_gb": 24.0,
           "source": "nvidia-smi", "detail": "RTX 4090 with 24.0 GB",
           "forward_url": "", "attested": False}
SMALL_GPU = dict(BIG_GPU, name="RTX 3060", vram_gb=12.0,
                 detail="RTX 3060 with 12.0 GB")


def _registry():
    from nomorals.tools.registry import ToolRegistry

    return ToolRegistry().register_builtins()


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _registry()

    def test_three_3d_tools_registered(self):
        for name in ("trellis_image_to_3d", "unirig_rig_mesh",
                     "blender_live"):
            with self.subTest(tool=name):
                self.assertIn(name, self.r.names())

    def test_trellis_tool_is_fs_write(self):
        self.assertEqual(
            self.r._tools["trellis_image_to_3d"].capability, "fs.write")

    def test_unirig_tool_is_fs_write(self):
        self.assertEqual(
            self.r._tools["unirig_rig_mesh"].capability, "fs.write")

    def test_blender_tool_is_net_out(self):
        self.assertEqual(
            self.r._tools["blender_live"].capability, "net.out")


class TrellisToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _registry()

    def test_tool_refuses_without_gpu_and_never_calls_wrapper(self):
        with patch("nomorals.media.trellis.probe_gpu",
                   return_value=dict(NO_GPU)), \
                patch("nomorals.media.trellis.image_to_3d") as run:
            out = self.r.call("trellis_image_to_3d",
                              image="photo.jpg", out="out.glb")
        self.assertTrue(out.ok)
        data = out.value
        self.assertFalse(data["ok"])
        self.assertEqual(data["status"], "gpu-refused")
        self.assertIn("24", data["detail"])
        self.assertIn("no NVIDIA GPU", data["detail"])
        run.assert_not_called()

    def test_tool_refuses_small_gpu(self):
        with patch("nomorals.media.trellis.probe_gpu",
                   return_value=dict(SMALL_GPU)), \
                patch("nomorals.media.trellis.image_to_3d") as run:
            out = self.r.call("trellis_image_to_3d",
                              image="photo.jpg", out="out.glb")
        data = out.value
        self.assertFalse(data["ok"])
        self.assertEqual(data["status"], "gpu-refused")
        run.assert_not_called()

    def test_tool_reaches_wrapper_when_gate_passes(self):
        made = {"ok": True, "status": "ok", "detail": "GLB written",
                "glb": "out.glb", "blender_snippet": "import bpy"}
        with patch("nomorals.media.trellis.probe_gpu",
                   return_value=dict(BIG_GPU)), \
                patch("nomorals.media.trellis.image_to_3d",
                      return_value=made) as run:
            out = self.r.call("trellis_image_to_3d", image="photo.jpg",
                              out="out.glb", resolution="512", seed=7)
        self.assertTrue(out.ok)
        self.assertEqual(out.value, made)
        run.assert_called_once()
        _, kwargs = run.call_args
        self.assertEqual(kwargs["resolution"], "512")
        self.assertEqual(kwargs["seed"], 7)


class UnirigToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _registry()

    def test_tool_refuses_without_gpu_and_never_calls_wrapper(self):
        with patch("nomorals.media.unirig._trellis_probe_gpu",
                   return_value=dict(NO_GPU)), \
                patch("nomorals.media.unirig.rig_mesh") as run:
            out = self.r.call("unirig_rig_mesh",
                              mesh="mesh.glb", out="rigged.fbx")
        data = out.value
        self.assertFalse(data["ok"])
        self.assertEqual(data["status"], "gpu-refused")
        self.assertIn("8", data["detail"])
        run.assert_not_called()

    def test_tool_reaches_wrapper_when_gate_passes(self):
        made = {"ok": True, "status": "ok", "detail": "rigged",
                "rigged": "rigged.fbx", "blender_snippet": "import bpy"}
        with patch("nomorals.media.unirig._trellis_probe_gpu",
                   return_value=dict(SMALL_GPU)), \
                patch("nomorals.media.unirig.rig_mesh",
                      return_value=made) as run:
            out = self.r.call("unirig_rig_mesh", mesh="mesh.glb",
                              out="rigged.fbx", seed=99)
        self.assertTrue(out.ok)
        self.assertEqual(out.value, made)
        run.assert_called_once()
        _, kwargs = run.call_args
        self.assertEqual(kwargs["seed"], 99)


class _FakeBlenderClient:
    """Stand-in for the MCP client behind a fake session."""

    def __init__(self, calls):
        self.calls = calls

    def list_tools(self):
        return [{"name": "get_addon_status"},
                {"name": "get_scene_info"},
                {"name": "execute_blender_code"},
                {"name": "look"}]

    @property
    def server_info(self):
        return {"name": "Blender 4.2", "version": "4.2.0"}


class _FakeSession:
    def __init__(self, calls, *, connect_ok=True):
        self.calls = calls
        self.connect_ok = connect_ok
        self.closed = False

    def connect(self):
        self.calls.append("connect")
        if not self.connect_ok:
            return {"ok": False,
                    "error": "No live Blender found. Start Blender, "
                             "install the MCP for Blender add-on"}
        return {"ok": True, "server": {"name": "Blender 4.2"},
                "tools": ["get_addon_status"]}

    def close(self):
        self.closed = True

    def status(self):
        self.calls.append("status")
        return {"ok": True, "text": "Blender 4.2 · add-on 1.0 · on"}

    def scene_summary(self, **_kw):
        self.calls.append("scene")
        return {"ok": True, "text": "Cube · Sphere"}

    def run_code(self, code, user_prompt=""):
        self.calls.append(("code", code))
        return {"ok": True, "text": "ran ok"}

    def screenshot(self, **_kw):
        self.calls.append("look")
        return {"ok": True, "text": "viewport", "image_b64": "AAA=",
                "mime": "image/png"}


class BlenderToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _registry()

    def _patch_session(self, calls, *, connect_ok=True):
        return patch("nomorals.media.blender_live.BlenderLiveSession",
                     lambda *a, **k: _FakeSession(calls,
                                                 connect_ok=connect_ok))

    def test_connect_refuses_without_live_blender(self):
        with self._patch_session([], connect_ok=False):
            out = self.r.call("blender_live", action="connect")
        data = out.value
        self.assertFalse(data["ok"])
        self.assertIn("No live Blender found", data["error"])

    def test_status_uses_live_session(self):
        calls: list = []
        with self._patch_session(calls):
            out = self.r.call("blender_live", action="status")
        data = out.value
        self.assertTrue(data["ok"])
        self.assertIn("Blender 4.2", data["text"])
        self.assertIn("connect", calls)
        self.assertIn("status", calls)

    def test_scene_action(self):
        calls: list = []
        with self._patch_session(calls):
            out = self.r.call("blender_live", action="scene")
        self.assertTrue(out.value["ok"])
        self.assertIn("Cube", out.value["text"])

    def test_import_runs_the_glb_snippet(self):
        calls: list = []
        with self._patch_session(calls):
            out = self.r.call("blender_live", action="import",
                              path="chair.glb")
        self.assertTrue(out.value["ok"])
        codes = [c[1] for c in calls if isinstance(c, tuple)]
        self.assertEqual(len(codes), 1)
        self.assertIn("import_scene.gltf", codes[0])
        self.assertIn("chair.glb", codes[0])

    def test_import_runs_the_fbx_snippet(self):
        calls: list = []
        with self._patch_session(calls):
            out = self.r.call("blender_live", action="import",
                              path="rigged.fbx")
        self.assertTrue(out.value["ok"])
        codes = [c[1] for c in calls if isinstance(c, tuple)]
        self.assertIn("import_scene.fbx", codes[0])

    def test_unknown_action_refuses(self):
        with self._patch_session([]):
            out = self.r.call("blender_live", action="dance")
        self.assertFalse(out.value["ok"])


class CommandRegistryTests(unittest.TestCase):
    def test_commands_registered(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            LIST_ONELINERS,
            list_catalog,
        )

        for kind in ("trellis", "unirig", "blender"):
            with self.subTest(kind=kind):
                self.assertIn(kind, CONTROL_COMMANDS)
                self.assertIn(kind, COMMAND_DETAILS)
                self.assertIn(kind, LIST_ONELINERS)
                grouped = [k for _, names in LIST_GROUPS for k in names]
                self.assertIn(kind, grouped)
        catalog = list_catalog()
        for kind in ("trellis", "unirig", "blender"):
            self.assertIn(f"/{kind}", catalog)

    def test_commands_are_owner_only(self):
        from nomorals.social.chat.control import PUBLIC_COMMANDS

        for kind in ("trellis", "unirig", "blender"):
            self.assertNotIn(kind, PUBLIC_COMMANDS)


class _FakeTools:
    def __init__(self, results):
        self.calls: list = []
        self.results = results

    def call(self, name, **kwargs):
        self.calls.append((name, kwargs))
        value = self.results.get(name, {"ok": True})
        return SimpleNamespace(ok=True, value=value, error=None)


def _mixin(tmp_path, *, operator=True, results=None):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    mixin = RuntimeMediaMixin()
    mixin._is_operator = lambda message: operator  # type: ignore[method-assign]
    mixin.context = SimpleNamespace(
        tools=_FakeTools(results or {}),
        settings=SimpleNamespace(
            resolve=lambda key: str(tmp_path / key)))
    return mixin


def _msg(*, media=None):
    return SimpleNamespace(media=media or [])


def _tmpdir(tmp_path):
    """A disk-backed scratch dir: pytest's tmp_path when injected, else
    the verify dir (never /tmp — a 512 MB tmpfs)."""
    if tmp_path is not None:
        return Path(tmp_path)
    import tempfile

    return Path(tempfile.mkdtemp(
        dir="/home/hatch/workspace/gapfix25-3d-verify"))


class ChatControlTests(unittest.TestCase):
    def test_trellis_usage_on_empty(self):
        mixin = _mixin(Path("/tmp/x3d"))
        out = mixin._control_trellis("", message=_msg())
        self.assertIn("/trellis", out)

    def test_trellis_refuses_non_owner(self):
        mixin = _mixin(Path("/tmp/x3d"), operator=False)
        out = mixin._control_trellis("photo.jpg", message=_msg())
        self.assertEqual(out, "that one's just for the owner.")

    def test_trellis_reaches_tool_and_names_the_glb(self, tmp_path=None):
        tmp = _tmpdir(tmp_path)
        img = tmp / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")
        mixin = _mixin(tmp, results={
            "trellis_image_to_3d": {
                "ok": True, "status": "ok", "detail": "GLB written",
                "glb": str(tmp / "photo.glb"),
                "blender_snippet": "import bpy"}})
        out = mixin._control_trellis(str(img), message=_msg())
        self.assertIn("photo.glb", out)
        name, kwargs = mixin.context.tools.calls[0]
        self.assertEqual(name, "trellis_image_to_3d")
        self.assertEqual(kwargs["image"], str(img))

    def test_trellis_attached_image_wins_over_path_arg(self, tmp_path=None):
        tmp = _tmpdir(tmp_path)
        img = tmp / "attached.png"
        img.write_bytes(b"\x89PNGfake")
        mixin = _mixin(tmp, results={
            "trellis_image_to_3d": {
                "ok": True, "status": "ok", "detail": "GLB written",
                "glb": str(tmp / "attached.glb"),
                "blender_snippet": ""}})
        attached = SimpleNamespace(kind="image", path=str(img))
        out = mixin._control_trellis("other.jpg", message=_msg(media=[attached]))
        self.assertIn("attached.glb", out)
        _, kwargs = mixin.context.tools.calls[0]
        self.assertEqual(kwargs["image"], str(img))

    def test_trellis_gpu_refusal_surfaces(self, tmp_path=None):
        tmp = _tmpdir(tmp_path)
        img = tmp / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")
        mixin = _mixin(tmp, results={
            "trellis_image_to_3d": {
                "ok": False, "status": "gpu-refused",
                "detail": "TRELLIS.2 needs an NVIDIA GPU with 24 GB+ "
                          "VRAM — no NVIDIA GPU detected on this box",
                "glb": None, "blender_snippet": ""}})
        out = mixin._control_trellis(str(img), message=_msg())
        self.assertIn("24 GB", out)
        self.assertNotIn("photo.glb", out)

    def test_unirig_reaches_tool(self, tmp_path=None):
        tmp = _tmpdir(tmp_path)
        mesh = tmp / "mesh.glb"
        mesh.write_bytes(b"glTFfake")
        mixin = _mixin(tmp, results={
            "unirig_rig_mesh": {
                "ok": True, "status": "ok", "detail": "rigged",
                "rigged": str(tmp / "rigged.fbx"),
                "blender_snippet": ""}})
        out = mixin._control_unirig(str(mesh), message=_msg())
        self.assertIn("rigged.fbx", out)
        name, kwargs = mixin.context.tools.calls[0]
        self.assertEqual(name, "unirig_rig_mesh")
        self.assertEqual(kwargs["mesh"], str(mesh))

    def test_blender_status_through_control(self):
        mixin = _mixin(Path("/tmp/x3d"), results={
            "blender_live": {"ok": True,
                             "text": "Blender 4.2 · add-on 1.0 · on"}})
        out = mixin._control_blender("status", message=_msg())
        self.assertIn("Blender 4.2", out)
        name, kwargs = mixin.context.tools.calls[0]
        self.assertEqual(name, "blender_live")
        self.assertEqual(kwargs["action"], "status")

    def test_blender_usage_on_empty(self):
        mixin = _mixin(Path("/tmp/x3d"))
        out = mixin._control_blender("", message=_msg())
        self.assertIn("/blender", out)


if __name__ == "__main__":
    unittest.main()
