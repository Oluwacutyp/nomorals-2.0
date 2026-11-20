"""Lane B.18 — census §3B.18: Devon's own VTuber rig (HyperJail pattern).

The study (storytold/hyperjail, pattern only, no upstream code):
a UE5 VTubing experiment whose project is an empty shell — the rig
IS the configuration: body mocap in, face blendshapes in, voice
visemes in, a named bone-remap table between the tracking skeleton
and the avatar skeleton, the driven model out.

Devon's rig is the same shape in Python, wired into the organs
that already produce those inputs: the mocap organ's BVH skeletons
and state store, the directed lip-sync organ's visemes, and the
LiveKit video session's frame sources.

Proves: the rig definition loads + validates (bad rigs refused by
name), the remap applies (unmapped reported, merges named), the
driver poses from a mocap frame (rotation lands, smoothing holds,
stale carried), face blendshapes follow voice input (explicit
viseme, text G2P, acoustic heuristic labeled), the renderer emits
exact RGB24 bytes that change with the viseme and the blink, a
full driven frame round-trips through the LiveKit frame source and
through BVH export, the mocap tool's rig actions, the CLI parses,
and the chat control kind — with the tool count pinned at 782.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.partner.runtime_mocap import RuntimeMocapMixin
from nomorals.cmdline.commands.mocap import _cmd_mocap
from nomorals.cmdline.parser import _parser
from nomorals.media.mocap import BvhExporter, MocapStateStore
from nomorals.media.mocap.bvh import (
    Joint,
    Skeleton,
    SkeletonFrame,
    parse_bvh,
    skeleton_to_dict,
    write_bvh,
)
from nomorals.media.mocap.errors import MocapError
from nomorals.media.rig import (
    FaceInput,
    RigDefinition,
    RigDriver,
    RigError,
    RigSession,
    default_remap,
    default_rig,
    get_session,
    list_sessions,
    load_session,
    load_take_bvh,
    reset_sessions,
    take_from_store,
)
from nomorals.media.rig.remap import BoneRemap
from nomorals.media.rig.render import render_pose
from nomorals.media.rig.rig import Bone
from nomorals.social.chat.control import parse_control
from nomorals.tools.registry import ToolRegistry
from nomorals.voice.livekit_call import LiveKitUnavailable, rig_frame_source


def _frame(**overrides) -> SkeletonFrame:
    """One BVH-dialect mocap frame: Hips (6ch) + Head (3ch)."""
    values: dict[str, tuple[float, ...]] = {
        "Hips": (0.0, 0.0, 0.0, 10.0, 20.0, 30.0),
        "Head": (0.0, 45.0, 0.0),
    }
    values.update(overrides)
    return SkeletonFrame(values=values)


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


def _write_bvh(tmp: Path, n: int = 4) -> Path:
    skeleton = _take_skeleton()
    path = tmp / "take.bvh"
    path.write_text(write_bvh(skeleton, _take_frames(n)),
                    encoding="utf-8")
    return path


class _Clock:
    """A controllable monotonic clock for staleness tests."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class RigDefinitionTest(unittest.TestCase):
    def test_default_rig_loads(self):
        rig = default_rig()
        self.assertEqual(rig.name, "devon-default")
        self.assertEqual(len(rig.bones), 60)
        self.assertEqual(len(rig.blendshapes), 52)
        self.assertEqual(len(rig.visemes), 21)
        roots = [bone for bone in rig.bones if bone.parent is None]
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0].name, "root")

    def test_to_dict_from_dict_round_trip(self):
        rig = default_rig()
        again = RigDefinition.from_dict(rig.to_dict())
        self.assertEqual(again.name, rig.name)
        self.assertEqual([bone.name for bone in again.bones],
                         [bone.name for bone in rig.bones])
        self.assertEqual(again.blendshapes, rig.blendshapes)
        self.assertEqual(again.visemes, rig.visemes)

    def test_hierarchy_order_is_parent_before_children(self):
        rig = default_rig()
        order = [bone.name for bone in rig.hierarchy_order()]
        seen: set[str] = set()
        for bone in rig.bones:
            if bone.parent is not None:
                self.assertIn(bone.parent, seen)
            seen.add(bone.name)
        self.assertEqual(len(order), 60)

    def test_duplicate_bone_refused_by_name(self):
        with self.assertRaises(RigError) as caught:
            RigDefinition(name="dup", bones=(
                Bone(name="root", parent=None),
                Bone(name="spine", parent="root"),
                Bone(name="spine", parent="root"),
            ))
        self.assertIn("spine", str(caught.exception))
        self.assertIn("duplicate", str(caught.exception))

    def test_unknown_parent_refused_by_name(self):
        with self.assertRaises(RigError) as caught:
            RigDefinition(name="orphan", bones=(
                Bone(name="root", parent=None),
                Bone(name="arm", parent="shoulder"),
            ))
        self.assertIn("shoulder", str(caught.exception))

    def test_cycle_refused(self):
        with self.assertRaises(RigError) as caught:
            RigDefinition(name="loop", bones=(
                Bone(name="root", parent=None),
                Bone(name="a", parent="b"),
                Bone(name="b", parent="a"),
            ))
        self.assertIn("cycle", str(caught.exception))

    def test_two_roots_refused(self):
        with self.assertRaises(RigError) as caught:
            RigDefinition(name="two", bones=(
                Bone(name="root", parent=None),
                Bone(name="root2", parent=None),
            ))
        self.assertIn("exactly one root", str(caught.exception))

    def test_non_finite_offset_refused(self):
        with self.assertRaises(RigError) as caught:
            RigDefinition(name="nan", bones=(
                Bone(name="root", parent=None,
                     offset=(float("nan"), 0.0, 0.0)),
            ))
        self.assertIn("non-finite", str(caught.exception))

    def test_empty_name_refused(self):
        with self.assertRaises(RigError):
            RigDefinition(name="  ", bones=(
                Bone(name="root", parent=None),))

    def test_unknown_bone_refused_by_name(self):
        rig = default_rig()
        with self.assertRaises(RigError) as caught:
            rig.bone("not_a_bone")
        self.assertIn("not_a_bone", str(caught.exception))

    def test_missing_rig_file_refused_by_name(self):
        with self.assertRaises(RigError) as caught:
            RigDefinition.from_json("/no/such/rig.json")
        self.assertIn("/no/such/rig.json", str(caught.exception))

    def test_malformed_rig_file_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(RigError):
                RigDefinition.from_json(path)


class BoneRemapTest(unittest.TestCase):
    def test_both_dialects_check_against_default_rig(self):
        rig = default_rig()
        default_remap("bvh").check_against(rig)
        default_remap("xsens").check_against(rig)

    def test_apply_maps_and_reports_unmapped(self):
        remap = default_remap("bvh")
        result = remap.apply(_frame(UnknownJoint=(1.0, 2.0, 3.0)))
        self.assertEqual(result.values["root"],
                         (0.0, 0.0, 0.0, 10.0, 20.0, 30.0))
        self.assertEqual(result.values["head"], (0.0, 45.0, 0.0))
        self.assertEqual(result.unmapped, ("UnknownJoint",))
        self.assertEqual(result.merged, ())

    def test_merge_names_the_later_source(self):
        remap = default_remap("xsens")
        result = remap.apply(SkeletonFrame(values={
            "T12": (1.0, 2.0, 3.0),
            "T8": (4.0, 5.0, 6.0),
        }))
        # T12 and T8 both map to spine2; the later frame entry wins.
        self.assertEqual(result.values["spine2"], (4.0, 5.0, 6.0))
        self.assertEqual(result.merged, ("T8->spine2",))

    def test_remap_to_unknown_rig_bone_refused_at_load(self):
        rig = default_rig()
        bad = BoneRemap(dialect="test", mapping={"Hips": "nope"})
        with self.assertRaises(RigError) as caught:
            bad.check_against(rig)
        self.assertIn("nope", str(caught.exception))

    def test_unknown_dialect_refused(self):
        with self.assertRaises(RigError) as caught:
            default_remap("not_a_dialect")
        self.assertIn("not_a_dialect", str(caught.exception))

    def test_remap_round_trip(self):
        remap = default_remap("bvh")
        again = BoneRemap.from_dict(remap.to_dict())
        self.assertEqual(again.dialect, "bvh")
        self.assertEqual(again.mapping, remap.mapping)


class RigDriverTest(unittest.TestCase):
    def _driver(self, **kwargs) -> RigDriver:
        return RigDriver(default_rig(), default_remap("bvh"),
                         smooth=1.0, **kwargs)

    def test_head_rotation_lands_on_first_frame(self):
        pose = self._driver().drive(_frame())
        self.assertEqual(pose.seq, 0)
        self.assertEqual(pose.rotations["root"], (10.0, 20.0, 30.0))
        self.assertEqual(pose.rotations["head"][1], 45.0)
        self.assertEqual(pose.unmapped, ())
        self.assertFalse(pose.stale)

    def test_unmapped_joint_reported_on_pose(self):
        pose = self._driver().drive(
            _frame(UnknownJoint=(1.0, 2.0, 3.0)))
        self.assertEqual(pose.unmapped, ("UnknownJoint",))

    def test_smoothing_lerps_toward_target(self):
        driver = RigDriver(default_rig(), default_remap("bvh"),
                           smooth=0.5)
        first = driver.drive(_frame())
        self.assertEqual(first.rotations["head"][1], 45.0)
        second = driver.drive(SkeletonFrame(values={
            "Head": (0.0, 90.0, 0.0)}))
        # Halfway from 45 toward 90.
        self.assertAlmostEqual(second.rotations["head"][1], 67.5)

    def test_smooth_zero_holds_previous_pose(self):
        driver = RigDriver(default_rig(), default_remap("bvh"),
                           smooth=0.0)
        first = driver.drive(_frame())
        second = driver.drive(SkeletonFrame(values={
            "Head": (0.0, 90.0, 0.0)}))
        self.assertEqual(second.rotations["head"],
                         first.rotations["head"])

    def test_frame_none_holds_last_pose(self):
        driver = self._driver()
        first = driver.drive(_frame())
        held = driver.drive(None)
        self.assertEqual(held.rotations, first.rotations)
        self.assertEqual(held.seq, 1)

    def test_stale_flag_carried_on_pose(self):
        pose = self._driver().drive(_frame(), stale=True)
        self.assertTrue(pose.stale)

    def test_bad_smooth_refused(self):
        with self.assertRaises(RigError):
            RigDriver(default_rig(), default_remap("bvh"), smooth=2.0)

    def test_pose_to_dict_round_trips(self):
        pose = self._driver().drive(_frame())
        blob = pose.to_dict()
        self.assertEqual(blob["seq"], 0)
        self.assertEqual(blob["rotations"]["root"], [10.0, 20.0, 30.0])
        self.assertEqual(json.loads(json.dumps(blob)), blob)


class FaceTest(unittest.TestCase):
    def _driver(self) -> RigDriver:
        return RigDriver(default_rig(), default_remap("bvh"),
                         smooth=1.0)

    def test_blendshapes_follow_voice_input(self):
        pose = self._driver().drive(
            _frame(), FaceInput(blendshapes={"eyeBlinkLeft": 1.0,
                                             "jawOpen": 0.5}))
        self.assertEqual(pose.blendshapes["eyeBlinkLeft"], 1.0)
        self.assertEqual(pose.blendshapes["jawOpen"], 0.5)

    def test_blendshape_clamped_with_note(self):
        pose = self._driver().drive(
            _frame(), FaceInput(blendshapes={"eyeBlinkLeft": 1.5}))
        self.assertEqual(pose.blendshapes["eyeBlinkLeft"], 1.0)
        self.assertTrue(any("clamped" in note
                            for note in pose.face_notes))

    def test_unknown_blendshape_refused_by_name(self):
        with self.assertRaises(RigError) as caught:
            self._driver().drive(
                _frame(), FaceInput(blendshapes={"notARealShape": 1.0}))
        self.assertIn("notARealShape", str(caught.exception))

    def test_explicit_viseme_lands_with_mouth_shape(self):
        pose = self._driver().drive(_frame(), FaceInput(viseme="AA"))
        self.assertEqual(pose.viseme, "AA")
        self.assertEqual(len(pose.mouth), 3)
        silent = self._driver().drive(_frame(), FaceInput(viseme="sil"))
        self.assertNotEqual(pose.mouth, silent.mouth)

    def test_unknown_viseme_refused_by_name(self):
        with self.assertRaises(RigError) as caught:
            self._driver().drive(_frame(), FaceInput(viseme="zzz"))
        self.assertIn("zzz", str(caught.exception))

    def test_text_drives_visemes_through_g2p(self):
        pose = self._driver().drive(_frame(), FaceInput(text="hello"))
        self.assertIn(pose.viseme, default_rig().visemes)

    def test_acoustic_heuristic_is_labeled(self):
        pose = self._driver().drive(
            _frame(), FaceInput(openness=0.9, centroid=0.5))
        self.assertIn(pose.viseme, default_rig().visemes)
        self.assertTrue(any("acoustic heuristic" in note
                            for note in pose.face_notes))

    def test_laughter_clamped_with_note(self):
        pose = self._driver().drive(_frame(), FaceInput(laughter=2.0))
        self.assertEqual(pose.laughter, 1.0)
        self.assertTrue(any("laughter" in note
                            for note in pose.face_notes))

    def test_silence_without_face_input(self):
        pose = self._driver().drive(_frame())
        self.assertEqual(pose.viseme, "sil")
        self.assertEqual(pose.blendshapes, {})


class RenderTest(unittest.TestCase):
    def test_render_returns_exact_rgb_bytes(self):
        driver = RigDriver(default_rig(), default_remap("bvh"),
                           smooth=1.0)
        pose = driver.drive(_frame())
        rgb = render_pose(default_rig(), pose, width=320, height=180)
        self.assertEqual(len(rgb), 320 * 180 * 3)

    def test_viseme_changes_pixels(self):
        rig = default_rig()
        driver = RigDriver(rig, default_remap("bvh"), smooth=1.0)
        open_pose = driver.drive(_frame(), FaceInput(viseme="AA"))
        driver.reset()
        shut_pose = driver.drive(_frame(), FaceInput(viseme="sil"))
        self.assertNotEqual(
            render_pose(rig, open_pose, width=320, height=180),
            render_pose(rig, shut_pose, width=320, height=180))

    def test_blink_changes_pixels(self):
        rig = default_rig()
        driver = RigDriver(rig, default_remap("bvh"), smooth=1.0)
        blink = driver.drive(
            _frame(), FaceInput(blendshapes={"eyeBlinkLeft": 1.0,
                                             "eyeBlinkRight": 1.0}))
        driver.reset()
        open_eyes = driver.drive(_frame())
        self.assertNotEqual(
            render_pose(rig, blink, width=320, height=180),
            render_pose(rig, open_eyes, width=320, height=180))

    def test_bad_size_refused(self):
        rig = default_rig()
        driver = RigDriver(rig, default_remap("bvh"), smooth=1.0)
        pose = driver.drive(_frame())
        with self.assertRaises(RigError):
            render_pose(rig, pose, width=10, height=10)


class SessionTest(unittest.TestCase):
    def setUp(self):
        reset_sessions()

    def _session(self, **kwargs) -> RigSession:
        return RigSession(default_rig(), default_remap("bvh"),
                          width=320, height=180, **kwargs)

    def test_drive_take_with_text(self):
        session = self._session()
        posed = session.drive_take(_take_skeleton(), _take_frames(4),
                                   text="hello world")
        self.assertEqual(len(posed), 4)
        self.assertEqual([pose.seq for pose in posed], [0, 1, 2, 3])
        visemes = default_rig().visemes
        self.assertTrue(all(pose.viseme in visemes for pose in posed))

    def test_drive_take_text_and_face_refused(self):
        session = self._session()
        with self.assertRaises(RigError):
            session.drive_take(_take_skeleton(), _take_frames(2),
                               text="hi", face=FaceInput(viseme="AA"))

    def test_drive_take_empty_frames_refused(self):
        session = self._session()
        with self.assertRaises(RigError):
            session.drive_take(_take_skeleton(), [])

    def test_render_take_in_memory_and_to_pngs(self):
        session = self._session()
        posed = session.drive_take(_take_skeleton(), _take_frames(3))
        frames = session.render_take(posed)
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(len(rgb) == 320 * 180 * 3
                            for rgb in frames))
        with tempfile.TemporaryDirectory() as tmp:
            paths = session.render_take(posed, out_dir=tmp)
            self.assertEqual(len(paths), 3)
            for index, path in enumerate(paths):
                self.assertTrue(Path(path).exists())
                self.assertEqual(Path(path).name,
                                 f"frame-{index:04d}.png")

    def test_frame_source_frames_then_none_then_loops(self):
        session = self._session()
        source = session.frame_source(_take_skeleton(), _take_frames(4))
        first = source(0.0)
        self.assertIsNotNone(first)
        self.assertEqual(len(first[0]), 320 * 180 * 3)
        self.assertEqual((first[1], first[2]), (320, 180))
        last = source(0.35)
        self.assertIsNotNone(last)
        self.assertNotEqual(first[0], last[0])
        self.assertIsNone(source(0.41))
        looping = session.frame_source(_take_skeleton(),
                                       _take_frames(4), loop=True)
        self.assertIsNotNone(looping(0.45))

    def test_export_bvh_round_trip(self):
        session = self._session()
        posed = session.drive_take(_take_skeleton(), _take_frames(4))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "driven.bvh"
            result = session.export_bvh(posed, out)
            self.assertEqual(result["frames"], 4)
            self.assertEqual(result["rig"], "devon-default")
            skeleton, frames = parse_bvh(
                out.read_text(encoding="utf-8"))
            self.assertEqual(len(frames), 4)
            self.assertEqual(skeleton.root.name, "root")
            from nomorals.media.mocap.bvh import hierarchy_joints
            self.assertEqual(len(hierarchy_joints(skeleton)), 60)

    def test_export_bvh_shares_channel_with_mocap_exporter(self):
        # The ``bvh`` channel has ONE writer (engine "bvh"). A rig
        # export and a plain mocap export must both succeed in
        # either order on the same store — the rig records through
        # the BVH writer, never as a second engine claiming the
        # channel. (Regression: the rig first wrote as engine
        # "rig", so a later bvh_write died with ChannelOwnershipError
        # — and vice versa.)
        session = self._session()
        posed = session.drive_take(_take_skeleton(), _take_frames(2))
        exporter = BvhExporter()
        with tempfile.TemporaryDirectory() as tmp:
            out_rig = Path(tmp) / "rig.bvh"
            out_plain = Path(tmp) / "plain.bvh"
            store = MocapStateStore()
            session.export_bvh(posed, out_rig, store=store)
            receipt = exporter.export_frames(
                _take_skeleton(), _take_frames(2), out_plain,
                store=store)
            self.assertEqual(receipt["frames"], 2)
            view = store.read("bvh")
            self.assertEqual(view["payload"]["path"], str(out_plain))
            store2 = MocapStateStore()
            exporter.export_frames(
                _take_skeleton(), _take_frames(2), out_plain,
                store=store2)
            result = session.export_bvh(posed, out_rig, store=store2)
            self.assertEqual(result["frames"], 2)
            view2 = store2.read("bvh")
            self.assertEqual(view2["payload"]["rig"], "devon-default")
            self.assertEqual(view2["payload"]["path"], str(out_rig))

    def test_load_take_bvh_missing_refused_by_name(self):
        with self.assertRaises(RigError) as caught:
            load_take_bvh("/no/such/take.bvh")
        self.assertIn("/no/such/take.bvh", str(caught.exception))

    def test_load_take_bvh_malformed_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.bvh"
            path.write_text("not a bvh file", encoding="utf-8")
            with self.assertRaises(RigError):
                load_take_bvh(path)

    def test_take_from_store_and_drive_from_store(self):
        clock = _Clock()
        store = MocapStateStore(clock=clock)
        store.register_engine("bvh-playback", produces=("skeleton",))
        skeleton = _take_skeleton()
        store.write("skeleton", "bvh-playback", {
            "skeleton": skeleton_to_dict(skeleton),
            "frames": [{name: list(values)
                        for name, values in frame.values.items()}
                       for frame in _take_frames(2)],
        })
        take = take_from_store(store)
        self.assertEqual(len(take["frames"]), 2)
        self.assertFalse(take["stale"])
        session = self._session()
        pose = session.drive_from_store(store)
        self.assertEqual(pose.rotations["head"][1], 10.0)

    def test_drive_from_store_require_fresh_refuses_stale(self):
        clock = _Clock()
        store = MocapStateStore(clock=clock)
        store.register_engine("bvh-playback", produces=("skeleton",))
        skeleton = _take_skeleton()
        store.write("skeleton", "bvh-playback", {
            "skeleton": skeleton_to_dict(skeleton),
            "frames": [{name: list(values)
                        for name, values in frame.values.items()}
                       for frame in _take_frames(1)],
        }, max_age_s=0.5)
        clock.now = 2000.0  # the write is now 1000s old
        session = self._session()
        with self.assertRaises(MocapError):
            session.drive_from_store(store, require_fresh=True)
        # Without require_fresh the stale frame still drives, flagged.
        pose = session.drive_from_store(store)
        self.assertTrue(pose.stale)

    def test_registry_load_get_list_reset(self):
        reset_sessions()
        sessions = list_sessions()
        self.assertEqual([entry["name"] for entry in sessions],
                         ["default"])
        load_session("second")
        names = [entry["name"] for entry in list_sessions()]
        self.assertEqual(names, ["default", "second"])
        session = get_session("second")
        self.assertEqual(session.rig.name, "devon-default")
        with self.assertRaises(RigError) as caught:
            get_session("nope")
        self.assertIn("nope", str(caught.exception))
        reset_sessions()
        self.assertEqual([entry["name"] for entry in list_sessions()],
                         ["default"])


class ToolTest(unittest.TestCase):
    def setUp(self):
        reset_sessions()

    def _call(self, **kwargs):
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        outcome = registry.call("mocap", **kwargs)
        if getattr(outcome, "ok", False):
            return outcome.value
        raise AssertionError(
            f"mocap call failed: {getattr(outcome, 'error', outcome)}")

    def _raw_call(self, **kwargs):
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        return registry.call("mocap", **kwargs)

    def test_tool_count_is_782(self):
        # 774 at B.18 (this lane added the mocap tool, no new tool
        # count move of its own); 775 after B.19 added scene_spawn
        # (the B.19 rebase carried this file in with the pin still
        # at 774 — stale on main, fixed here); 776 after B.20 added
        # cloudcam; 777 after B.22 added show; 778 after B.26 added scene_project; 779 after this lane
        # (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p.
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        self.assertEqual(len(registry._tools), 782)
        self.assertIn("mocap", registry._tools)

    def test_rig_list_load_status_pose(self):
        listed = self._call(action="rig_list")
        self.assertEqual(listed["sessions"][0]["name"], "default")
        loaded = self._call(action="rig_load", node="tool-rig-18")
        self.assertTrue(loaded["ok"])
        self.assertEqual(loaded["bones"], 60)
        status = self._call(action="rig_status", node="tool-rig-18")
        self.assertEqual(status["rig"], "devon-default")
        pose = self._call(action="rig_pose", node="tool-rig-18",
                          params={"frame": {
                              "Hips": [0, 0, 0, 10, 20, 30],
                              "Head": [0, 45, 0]},
                                  "face": {"viseme": "AA"}})
        self.assertEqual(pose["rotations"]["root"], [10.0, 20.0, 30.0])
        self.assertEqual(pose["viseme"], "AA")

    def test_rig_pose_bad_face_key_refused(self):
        outcome = self._raw_call(
            action="rig_pose", node="tool-rig-18",
            params={"face": {"surprise": 1.0}})
        self.assertFalse(getattr(outcome, "ok", True))

    def test_rig_drive_and_export_and_render(self):
        self._call(action="rig_load", node="tool-rig-18")
        with tempfile.TemporaryDirectory() as tmp:
            take = _write_bvh(Path(tmp))
            driven = self._call(action="rig_drive", node="tool-rig-18",
                                sources=[str(take)],
                                params={"text": "hello"})
            self.assertEqual(driven["frames"], 4)
            self.assertEqual(len(driven["visemes"]), 4)
            out_bvh = Path(tmp) / "out.bvh"
            exported = self._call(action="rig_export",
                                  node="tool-rig-18",
                                  sources=[str(take)],
                                  output=str(out_bvh))
            self.assertEqual(exported["frames"], 4)
            self.assertTrue(out_bvh.exists())
            out_png = Path(tmp) / "pose.png"
            rendered = self._call(action="rig_render",
                                  node="tool-rig-18",
                                  output=str(out_png))
            self.assertTrue(out_png.exists())
            self.assertEqual(rendered["session"], "tool-rig-18")


class CliTest(unittest.TestCase):
    def test_parser_mocap_rig_list(self):
        args = _parser().parse_args(["mocap", "--json", "rig", "list"])
        self.assertEqual(args.mocap_action, "rig")
        self.assertEqual(args.mocap_rig_verb, "list")
        self.assertTrue(args.json)

    def test_parser_mocap_rig_drive(self):
        args = _parser().parse_args(
            ["mocap", "rig", "drive", "--bvh", "x.bvh",
             "--text", "hello"])
        self.assertEqual(args.mocap_rig_verb, "drive")
        self.assertEqual(args.bvh, "x.bvh")
        self.assertEqual(args.text, "hello")

    def test_parser_live_video_rig_source(self):
        args = _parser().parse_args(
            ["voice", "live", "video", "start", "--room", "r1",
             "--source", "rig", "--bvh", "x.bvh"])
        self.assertEqual(args.voice_live_video_action, "start")
        self.assertEqual(args.source, "rig")
        self.assertEqual(args.bvh, "x.bvh")

    def test_cmd_mocap_rig_list_runs(self):
        reset_sessions()
        args = _parser().parse_args(["mocap", "--json", "rig", "list"])
        self.assertEqual(_cmd_mocap(args, None), 0)

    def test_cmd_mocap_rig_drive_runs(self):
        reset_sessions()
        with tempfile.TemporaryDirectory() as tmp:
            take = _write_bvh(Path(tmp))
            args = _parser().parse_args(
                ["mocap", "rig", "drive", "--bvh", str(take),
                 "--text", "hello"])
            self.assertEqual(_cmd_mocap(args, None), 0)


class ChatTest(unittest.TestCase):
    def test_parse_control_mocap_kind(self):
        self.assertEqual(parse_control("/mocap rig list").kind, "mocap")
        self.assertEqual(parse_control("/mocap status").kind, "mocap")

    def test_mixin_rig_list_and_status(self):
        reset_sessions()
        mixin = RuntimeMocapMixin()
        listed = mixin._control_mocap("rig list", "chat-key")
        self.assertIn("rig sessions:", listed)
        self.assertIn("devon-default", listed)
        status = mixin._control_mocap("rig status", "chat-key")
        self.assertIn("devon-default", status)

    def test_mixin_rig_pose_and_help(self):
        reset_sessions()
        mixin = RuntimeMocapMixin()
        pose = mixin._control_mocap("rig pose", "chat-key")
        self.assertIn("pose f:0 viseme sil", pose)
        self.assertIn("/mocap status", mixin._control_mocap("", "chat-key"))
        self.assertIn("/mocap status",
                      mixin._control_mocap("bogus", "chat-key"))

    def test_mixin_unknown_session_refused_in_words(self):
        reset_sessions()
        mixin = RuntimeMocapMixin()
        reply = mixin._control_mocap("rig status nope", "chat-key")
        self.assertTrue(reply.startswith("rig: "))
        self.assertIn("nope", reply)


class LiveKitTest(unittest.TestCase):
    def test_rig_frame_source_frames_then_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            take = _write_bvh(Path(tmp))
            source = rig_frame_source(None, take)
            first = source(0.0)
            self.assertIsNotNone(first)
            self.assertEqual(len(first[0]), 640 * 360 * 3)
            self.assertEqual((first[1], first[2]), (640, 360))
            self.assertIsNone(source(60.0))

    def test_rig_frame_source_with_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            take = _write_bvh(Path(tmp), n=8)
            source = rig_frame_source(None, take, text="hello",
                                      width=320, height=180)
            frame = source(0.0)
            self.assertIsNotNone(frame)
            self.assertEqual(len(frame[0]), 320 * 180 * 3)

    def test_rig_frame_source_missing_take_refused_by_name(self):
        with self.assertRaises(LiveKitUnavailable) as caught:
            rig_frame_source(None, "/no/such/take.bvh")
        self.assertIn("/no/such/take.bvh", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
