"""B.26 — scene projects (storytold/streamtwo, PATTERN ONLY).

Devon's own project shape: one directory tying scenes, takes,
rigs and assets together, with a sealed manifest, load/save,
validation and named refusals. The Unreal project shape is the
pattern — the layout, the content tree, the module/plugin
structure, the manifest — not the engine: nothing here compiles
a .uproject, and an import of a real .uproject maps what Devon
has an analogue for and names what it does not.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.partner.runtime_mocap import RuntimeMocapMixin
from nomorals.cmdline.commands.sceneproject import _cmd_sceneproject
from nomorals.cmdline.parser import _parser
from nomorals.media.edit_engine.scene3d import (
    Camera,
    Rotation,
    Scene3D,
    SceneNode,
    Transform,
)
from nomorals.media.mocap.bvh import (
    Joint,
    Skeleton,
    SkeletonFrame,
    skeleton_to_dict,
    write_bvh,
)
from nomorals.media.project import (
    Project,
    load_project,
    reset_projects,
)
from nomorals.media.project.errors import ProjectError
from nomorals.media.project.manifest import ProjectManifest
from nomorals.media.rig import default_rig
from nomorals.media.show import drop_show
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    help_text,
    list_catalog,
    parse_control,
)
from nomorals.tools.mocap import _STORE
from nomorals.tools.registry import ToolRegistry


def _scene() -> Scene3D:
    return Scene3D(
        title="project-test",
        cameras={"main": Camera(
            transform=Transform(
                position=(0.0, 0.0, 5.0),
                rotation=Rotation.look_at((0.0, 0.0, 0.0))))},
        active_camera="main",
        nodes=[SceneNode(
            name="anchor",
            transform=Transform(position=(1.0, 2.0, 0.0)))],
    )


def _scene_file(tmp: Path, name: str = "level1.json") -> Path:
    path = tmp / name
    _scene().save(path)
    return path


def _take_skeleton() -> Skeleton:
    head = Joint(name="Head", offset=(0.0, 10.0, 0.0),
                 channels=("Zrotation", "Xrotation", "Yrotation"))
    spine = Joint(name="Spine", offset=(0.0, 10.0, 0.0), children=[head])
    root = Joint(name="Hips", offset=(0.0, 0.0, 0.0),
                 channels=("Xposition", "Yposition", "Zposition",
                           "Zrotation", "Xrotation", "Yrotation"),
                 children=[spine])
    return Skeleton(root=root, frame_time=0.1)


def _take_frames(n: int = 4) -> list[SkeletonFrame]:
    return [SkeletonFrame(values={
        "Hips": (0.0, float(i), 0.0, 0.0, 0.0, 0.0),
        "Spine": (0.0, float(i), 0.0),
        "Head": (0.0, float(10 * i), 0.0),
    }) for i in range(n)]


def _bvh_file(tmp: Path, name: str = "take1.bvh") -> Path:
    path = tmp / name
    path.write_text(write_bvh(_take_skeleton(), _take_frames()),
                    encoding="utf-8")
    return path


def _json_take_file(tmp: Path, name: str = "take2.json") -> Path:
    path = tmp / name
    path.write_text(json.dumps({
        "skeleton": skeleton_to_dict(_take_skeleton()),
        "frames": [
            {"Hips": [0.0, float(i), 0.0, 0.0, 0.0, 0.0],
             "Spine": [0.0, float(i), 0.0],
             "Head": [0.0, float(10 * i), 0.0]}
            for i in range(3)],
    }), encoding="utf-8")
    return path


def _rig_file(tmp: Path, name: str = "rig1.json") -> Path:
    path = tmp / name
    path.write_text(json.dumps(default_rig().to_dict()),
                    encoding="utf-8")
    return path


def _uproject_file(tmp: Path, name: str = "StreamTwo.uproject",
                   **overrides) -> Path:
    """A .uproject shaped like the StreamTwo clone's own file."""
    data = {
        "FileVersion": 3,
        "EngineAssociation": "4.25",
        "Category": "",
        "Description": "second Twitch stream",
        "Modules": [{"Name": "StreamTwo", "Type": "Runtime",
                     "LoadingPhase": "Default"}],
        "Plugins": [
            {"Name": "RuntimeAudioImporter", "Enabled": True},
            {"Name": "BlueprintWebSocket", "Enabled": True},
            {"Name": "VocodesPlugin", "Enabled": True},
            {"Name": "RuntimeFilesDownloader", "Enabled": True},
            {"Name": "MediaPlayerPlugin", "Enabled": False},
            {"Name": "EventCenterPlugin", "Enabled": True},
        ],
        "TargetPlatforms": ["Win64"],
    }
    data.update(overrides)
    path = tmp / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _registry(tmp: Path) -> ToolRegistry:
    registry = ToolRegistry(SimpleNamespace(home=str(tmp)))
    registry.register_builtins()
    return registry


def _tool(registry: ToolRegistry, **kwargs: object) -> dict:
    outcome = registry.call("scene_project", **kwargs)
    if getattr(outcome, "ok", False):
        return outcome.value
    raise AssertionError(
        f"scene_project call failed: "
        f"{getattr(outcome, 'error', outcome)}")


def _raw(registry: ToolRegistry, **kwargs: object):
    return registry.call("scene_project", **kwargs)


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name)

    def test_create_round_trip(self):
        project = Project.create("zelda", self.tmp)
        self.assertTrue(project.manifest_path.exists())
        self.assertTrue((project.root / "Config"
                         / "DefaultProject.ini").exists())
        for sub in ("Scenes", "Takes", "Rigs", "Assets"):
            self.assertTrue((project.root / "Content" / sub).is_dir())
        self.assertTrue((project.root / "Source").is_dir())
        again = Project.open(project.manifest_path)
        self.assertEqual(again.name, "zelda")
        self.assertEqual(again.status()["seal"],
                         project.status()["seal"])
        self.assertEqual(again.validate(), [])

    def test_create_refuses_bad_name(self):
        with self.assertRaises(ProjectError) as caught:
            Project.create("a b/c", self.tmp)
        self.assertIn("a b/c", str(caught.exception))

    def test_create_refuses_existing(self):
        Project.create("zelda", self.tmp)
        with self.assertRaises(ProjectError) as caught:
            Project.create("zelda", self.tmp)
        self.assertEqual(caught.exception.kind, "already_exists")

    def test_manifest_tamper_refused(self):
        project = Project.create("zelda", self.tmp)
        data = json.loads(project.manifest_path.read_text())
        data["description"] = "tampered"
        project.manifest_path.write_text(json.dumps(data))
        with self.assertRaises(ProjectError) as caught:
            Project.open(project.manifest_path)
        self.assertIn("seal", str(caught.exception))

    def test_unknown_module_refused(self):
        with self.assertRaises(ProjectError) as caught:
            ProjectManifest(name="x", modules=[
                {"name": "notanorgan", "type": "runtime",
                 "loading_phase": "default", "depends_on": []}])
        self.assertIn("notanorgan", str(caught.exception))

    def test_duplicate_module_refused(self):
        with self.assertRaises(ProjectError) as caught:
            ProjectManifest(name="x", modules=[
                {"name": "scene", "type": "runtime",
                 "loading_phase": "default", "depends_on": []},
                {"name": "scene", "type": "runtime",
                 "loading_phase": "default", "depends_on": []}])
        self.assertIn("scene", str(caught.exception))

    def test_cyclic_module_deps_refused(self):
        with self.assertRaises(ProjectError) as caught:
            ProjectManifest(name="x", modules=[
                {"name": "scene", "type": "runtime",
                 "loading_phase": "default", "depends_on": ["rig"]},
                {"name": "rig", "type": "runtime",
                 "loading_phase": "default",
                 "depends_on": ["scene"]}])
        self.assertIn("cycle", str(caught.exception))

    def test_unknown_plugin_refused(self):
        with self.assertRaises(ProjectError) as caught:
            ProjectManifest(name="x", plugins=[
                {"name": "notaplugin", "enabled": True}])
        self.assertIn("notaplugin", str(caught.exception))

    def test_media_player_disabled_by_default(self):
        project = Project.create("zelda", self.tmp)
        status = project.status()
        self.assertIn("media_player", status["plugins_disabled"])
        self.assertNotIn("media_player", status["plugins"])


class ProjectEntriesTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name)
        reset_projects()
        self.addCleanup(reset_projects)
        self.project = Project.create("zelda", self.tmp)
        load_project(self.project.manifest_path)

    def test_add_scene_copies_and_validates(self):
        source = _scene_file(self.tmp)
        receipt = self.project.add_scene("level1", source)
        self.project.save()
        self.assertEqual(receipt["stored"], "copy")
        self.assertTrue(
            (self.project.root / "Content" / "Scenes"
             / "level1.json").exists())
        scene = self.project.open_scene("level1")
        self.assertEqual(scene.title, "project-test")
        self.assertEqual([n.name for n in scene.nodes], ["anchor"])
        self.assertEqual(self.project.status()["scenes"], ["level1"])

    def test_add_scene_refuses_broken_file(self):
        bad = self.tmp / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ProjectError) as caught:
            self.project.add_scene("bad", bad)
        self.assertIn("bad", str(caught.exception))
        self.assertEqual(self.project.status()["scenes"], [])

    def test_add_take_bvh_and_json(self):
        self.project.add_take("take1", _bvh_file(self.tmp))
        self.project.add_take("take2", _json_take_file(self.tmp))
        self.project.save()
        self.assertEqual(self.project.status()["takes"],
                         ["take1", "take2"])
        self.assertTrue(self.project.take_path("take1").exists())
        self.assertTrue(self.project.take_path("take2").exists())

    def test_add_take_refuses_garbage(self):
        bad = self.tmp / "bad.bvh"
        bad.write_text("HIERARCHY\nnonsense\n", encoding="utf-8")
        with self.assertRaises(ProjectError):
            self.project.add_take("bad", bad)

    def test_add_rig_validates(self):
        receipt = self.project.add_rig("rig1", _rig_file(self.tmp))
        self.project.save()
        self.assertEqual(receipt["stored"], "copy")
        self.assertEqual(self.project.status()["rigs"], ["rig1"])
        bad = self.tmp / "bad.json"
        bad.write_text(json.dumps({"name": "x", "bones": []}),
                       encoding="utf-8")
        with self.assertRaises(ProjectError) as caught:
            self.project.add_rig("bad", bad)
        self.assertIn("bones", str(caught.exception))

    def test_add_asset_copy_and_link(self):
        source = self.tmp / "notes.txt"
        source.write_text("hello", encoding="utf-8")
        copied = self.project.add_asset("notes", source)
        self.assertEqual(copied["stored"], "copy")
        self.assertEqual(
            self.project.entries["assets"]["notes"]["path"],
            "Content/Assets/notes.txt")
        linked = self.project.add_asset("notes-link", source, copy=False)
        self.assertEqual(linked["stored"], "link")
        self.assertEqual(
            self.project.entries["assets"]["notes-link"]["path"],
            str(source.resolve()))
        self.assertEqual(self.project.status()["assets"],
                         ["notes", "notes-link"])

    def test_duplicate_entry_refused(self):
        self.project.add_scene("level1", _scene_file(self.tmp))
        with self.assertRaises(ProjectError) as caught:
            self.project.add_scene("level1", _scene_file(self.tmp))
        self.assertEqual(caught.exception.kind, "duplicate_entry")

    def test_bad_entry_name_refused(self):
        with self.assertRaises(ProjectError):
            self.project.add_scene("../escape", _scene_file(self.tmp))

    def test_set_default_scene_requires_known_scene(self):
        with self.assertRaises(ProjectError) as caught:
            self.project.set_default_scene("missing")
        self.assertEqual(caught.exception.kind, "unknown_entry")
        self.project.add_scene("level1", _scene_file(self.tmp))
        self.project.set_default_scene("level1")
        self.project.save()
        self.assertEqual(
            self.project.status()["default_scene"], "level1")

    def test_remove_entry_and_delete_file(self):
        self.project.add_scene("level1", _scene_file(self.tmp))
        self.project.save()
        receipt = self.project.remove("scenes", "level1",
                                      delete_file=True)
        self.assertEqual(receipt["removed"], "level1")
        self.assertFalse(
            (self.project.root / "Content" / "Scenes"
             / "level1.json").exists())
        self.assertEqual(self.project.status()["scenes"], [])

    def test_remove_link_never_deletes_target(self):
        source = self.tmp / "notes.txt"
        source.write_text("hello", encoding="utf-8")
        self.project.add_asset("notes", source, copy=False)
        self.project.save()
        self.project.remove("assets", "notes", delete_file=True)
        self.assertTrue(source.exists())

    def test_validate_reports_missing_file(self):
        self.project.add_scene("level1", _scene_file(self.tmp))
        self.project.save()
        (self.project.root / "Content" / "Scenes"
         / "level1.json").unlink()
        problems = self.project.validate()
        self.assertEqual(len(problems), 1)
        self.assertIn("level1", problems[0])

    def test_export_bundle(self):
        self.project.add_scene("level1", _scene_file(self.tmp))
        self.project.add_take("take1", _bvh_file(self.tmp))
        self.project.set_default_scene("level1")
        self.project.save()
        out = self.tmp / "bundle"
        receipt = self.project.export_bundle(out)
        self.assertGreater(receipt["files"], 3)
        bundle = json.loads((out / "bundle.json").read_text())
        self.assertEqual(bundle["project"], "zelda")
        self.assertIn("zelda.dvproject", bundle["files"])
        for name, digest in bundle["files"].items():
            self.assertEqual(len(digest), 64)
            self.assertTrue((out / name).exists())

    def test_bind_show(self):
        self.project.add_take("take1", _bvh_file(self.tmp))
        self.project.save()
        try:
            status = self.project.bind_show("zelda-show", _STORE)
            self.assertEqual(status["project"], "zelda")
            self.assertEqual(status["project_take"], "take1")
            self.assertEqual(status["project_default_scene"], "")
        finally:
            # Cleanup only: a show that never loaded is not an error.
            with contextlib.suppress(Exception):
                drop_show("zelda-show")

    def test_bind_show_refuses_no_takes(self):
        with self.assertRaises(ProjectError) as caught:
            self.project.bind_show("zelda-show", _STORE)
        self.assertEqual(caught.exception.kind, "unknown_entry")


class UprojectImportTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name)
        reset_projects()
        self.addCleanup(reset_projects)

    def test_import_streamtwo_shape(self):
        path = _uproject_file(self.tmp)
        project, receipt = Project.from_uproject(path, self.tmp)
        self.assertEqual(project.name, "StreamTwo")
        self.assertEqual(receipt["source_engine"], "4.25")
        self.assertEqual(receipt["modules"], ["scene"])
        self.assertEqual(sorted(receipt["plugins"]),
                         ["audio_import", "files_downloader",
                          "media_player", "vocodes", "websocket"])
        self.assertEqual(receipt["unmapped_plugins"],
                         ["EventCenterPlugin"])
        status = project.status()
        self.assertIn("media_player", status["plugins_disabled"])
        self.assertEqual(status["unmapped_plugins"],
                         ["EventCenterPlugin"])
        self.assertEqual(project.validate(), [])

    def test_import_refuses_unknown_plugin(self):
        path = _uproject_file(
            self.tmp, Plugins=[{"Name": "NoSuchPlugin",
                                "Enabled": True}])
        with self.assertRaises(ProjectError) as caught:
            Project.from_uproject(path, self.tmp)
        self.assertEqual(caught.exception.kind, "unknown_plugin")
        self.assertIn("NoSuchPlugin", str(caught.exception))

    def test_import_refuses_unknown_module(self):
        path = _uproject_file(
            self.tmp,
            Modules=[{"Name": "StreamTwo", "Type": "Runtime",
                      "LoadingPhase": "Default"},
                     {"Name": "ZeroMQ", "Type": "Runtime",
                      "LoadingPhase": "Default"}])
        with self.assertRaises(ProjectError) as caught:
            Project.from_uproject(path, self.tmp)
        self.assertEqual(caught.exception.kind, "unknown_module")
        self.assertIn("ZeroMQ", str(caught.exception))

    def test_import_maps_known_module(self):
        path = _uproject_file(
            self.tmp,
            Modules=[{"Name": "StreamTwo", "Type": "Runtime",
                      "LoadingPhase": "Default"},
                     {"Name": "rig", "Type": "Runtime",
                      "LoadingPhase": "Default"}])
        project, receipt = Project.from_uproject(path, self.tmp)
        self.assertEqual(sorted(receipt["modules"]),
                         ["rig", "scene"])

    def test_import_refuses_missing_file(self):
        with self.assertRaises(ProjectError) as caught:
            Project.from_uproject(self.tmp / "nope.uproject",
                                  self.tmp)
        self.assertEqual(caught.exception.kind, "missing_file")


class ToolTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name)
        reset_projects()
        self.addCleanup(reset_projects)
        self.registry = _registry(self.tmp)

    def test_tool_count_is_786_after_scene_project(self):
        # 778 after this lane (B.26) added scene_project; 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh
        # (777 after B.22 added show).
        self.assertEqual(len(self.registry._tools), 800)
        self.assertIn("scene_project", self.registry._tools)

    def test_tool_create_list_status(self):
        created = _tool(self.registry, action="create",
                        node="zelda", params={"root": str(self.tmp)})
        self.assertTrue(created["created"])
        listed = _tool(self.registry, action="list")
        self.assertEqual([p["name"] for p in listed["projects"]],
                         ["zelda"])
        status = _tool(self.registry, action="status", node="zelda")
        self.assertEqual(status["name"], "zelda")
        self.assertEqual(status["scenes"], [])

    def test_tool_add_scene_end_to_end(self):
        _tool(self.registry, action="create", node="zelda",
              params={"root": str(self.tmp)})
        scene = _scene_file(self.tmp)
        receipt = _tool(self.registry, action="add_scene",
                        node="zelda", sources=[str(scene)],
                        params={"entry": "level1"})
        self.assertEqual(receipt["entry"], "level1")
        status = _tool(self.registry, action="status", node="zelda")
        self.assertEqual(status["scenes"], ["level1"])
        self.assertEqual(
            _tool(self.registry, action="validate", node="zelda"),
            {"project": "zelda", "valid": True, "problems": []})

    def test_tool_rejects_unknown_action(self):
        outcome = _raw(self.registry, action="nope")
        self.assertFalse(outcome.ok)
        self.assertIn("nope", str(outcome.error))

    def test_tool_open_and_drop(self):
        project = Project.create("zelda", self.tmp)
        opened = _tool(self.registry, action="open",
                       sources=[str(project.manifest_path)])
        self.assertEqual(opened["name"], "zelda")
        dropped = _tool(self.registry, action="drop", node="zelda")
        self.assertEqual(dropped["dropped"], "zelda")
        with self.assertRaises(AssertionError):
            _tool(self.registry, action="status", node="zelda")

    def test_tool_import_uproject(self):
        path = _uproject_file(self.tmp)
        receipt = _tool(self.registry, action="import_uproject",
                        sources=[str(path)],
                        params={"root": str(self.tmp)})
        self.assertEqual(receipt["name"], "StreamTwo")
        self.assertEqual(receipt["unmapped_plugins"],
                         ["EventCenterPlugin"])


class CliTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name)
        reset_projects()
        self.addCleanup(reset_projects)

    def test_parser_sceneproject(self):
        args = _parser().parse_args(
            ["sceneproject", "create", "zelda",
             "--root", str(self.tmp)])
        self.assertEqual(args.sceneproject_action, "create")
        self.assertEqual(args.project_name, "zelda")
        self.assertEqual(args.root, str(self.tmp))

    def test_parser_alias_sproj(self):
        args = _parser().parse_args(["sproj", "list"])
        self.assertEqual(args.sceneproject_action, "list")

    def test_cmd_create_list_status(self):
        args = _parser().parse_args(
            ["sceneproject", "create", "zelda",
             "--root", str(self.tmp)])
        self.assertEqual(_cmd_sceneproject(args, None), 0)
        args = _parser().parse_args(["sceneproject", "list"])
        self.assertEqual(_cmd_sceneproject(args, None), 0)
        args = _parser().parse_args(
            ["sceneproject", "--json", "status", "--name", "zelda"])
        self.assertEqual(_cmd_sceneproject(args, None), 0)

    def test_cmd_add_scene_and_validate(self):
        args = _parser().parse_args(
            ["sceneproject", "create", "zelda",
             "--root", str(self.tmp)])
        self.assertEqual(_cmd_sceneproject(args, None), 0)
        scene = _scene_file(self.tmp)
        args = _parser().parse_args(
            ["sceneproject", "add-scene", str(scene),
             "--name", "zelda", "--entry", "level1"])
        self.assertEqual(_cmd_sceneproject(args, None), 0)
        args = _parser().parse_args(
            ["sceneproject", "validate", "--name", "zelda"])
        self.assertEqual(_cmd_sceneproject(args, None), 0)

    def test_cmd_unknown_project_fails(self):
        args = _parser().parse_args(
            ["sceneproject", "status", "--name", "nope"])
        self.assertEqual(_cmd_sceneproject(args, None), 2)


class ChatTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name)
        reset_projects()
        self.addCleanup(reset_projects)
        self.mixin = RuntimeMocapMixin()

    def test_chat_project_registered_in_the_catalog(self):
        self.assertIn("project", CONTROL_COMMANDS)
        self.assertEqual(CONTROL_COMMANDS["project"], (0, None))
        details = COMMAND_DETAILS["project"]
        self.assertTrue(details["what"])
        self.assertTrue(details["usage"])
        self.assertTrue(details["example"])
        self.assertIn("/project", details["usage"])
        self.assertIn("project", LIST_ONELINERS)
        self.assertIn("/project", LIST_ONELINERS["project"])
        slots = [group for group in LIST_GROUPS
                 if "project" in group[1]]
        self.assertEqual(len(slots), 1)  # exactly one slot
        self.assertIn("project", help_text())
        self.assertIn("project", list_catalog())

    def test_chat_project_dispatches_through_parse_control(self):
        command = parse_control("/project list")
        self.assertIsNotNone(command)
        self.assertEqual(command.kind, "project")
        self.assertIn("projects:",
                      self.mixin._control_project("list", "test"))

    def test_chat_project_create_list_status(self):
        created = self.mixin._control_project(
            f"create zelda {self.tmp}", "test")
        self.assertIn("zelda", created)
        self.assertIn("zelda", self.mixin._control_project(
            "list", "test"))
        status = self.mixin._control_project("status zelda", "test")
        self.assertIn('"name": "zelda"', status)

    def test_chat_project_add_scene_and_validate(self):
        self.mixin._control_project(f"create zelda {self.tmp}", "test")
        scene = _scene_file(self.tmp)
        added = self.mixin._control_project(
            f"add-scene zelda {scene}", "test")
        self.assertIn("level1", added)
        self.assertIn("whole", self.mixin._control_project(
            "validate zelda", "test"))

    def test_chat_project_help(self):
        self.assertIn("/project create",
                      self.mixin._control_project("", "test"))
        self.assertIn("/project create",
                      self.mixin._control_project("bogus", "test"))


if __name__ == "__main__":
    unittest.main()
