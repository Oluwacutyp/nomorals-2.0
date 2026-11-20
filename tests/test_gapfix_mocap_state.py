"""Lane 125 — census §3B.2: cross-engine mocap/state patterns
(storytold/artcraftx, pattern only) into nomorals/media/mocap/.

The port mines artcraftx's design — one shared tasks table as the
cross-engine state, one optional client per engine with honest
absence, subscribers whose opaque payload is re-emitted on poll,
and BVH as the mocap interchange format for game engines — and
implements it as Devon's own code. No upstream file or text is
vendored or copied.

Proves: the store (write/read, ownership, staleness, deltas +
missed, subscribers, late-joiner snapshot, save/load), the router
(configured/unconfigured/unknown), the BVH round trip, the Kinect
fixture bridge, the BVH export loop, and the mocap brain tool — with
the tool count pinned at 782 (748 at the lane's base 0733f79a, +1
mocap tool, +25 secprobe tools, +1 scene_spawn (B.19), +1 cloudcam
(B.20), +1 show (B.22), +1 scene_project (B.26), +1 facemcap (B.23) — the B.19 sed had mangled this line,
rewritten by B.20).
No existing tool reaches the cross-engine
state store — Kinect has no tool, anicraft is 2D vector animation,
media is downloads).
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.hardware.kinect import parse_raw_calibration
from nomorals.media.mocap import (
    BvhExporter,
    BvhPlaybackProducer,
    KinectProducer,
    MocapRouter,
    MocapStateStore,
    default_router,
)
from nomorals.media.mocap.bvh import (
    Joint,
    Skeleton,
    SkeletonFrame,
    parse_bvh,
    skeleton_from_dict,
    skeleton_to_dict,
    write_bvh,
)
from nomorals.media.mocap.errors import (
    ChannelOwnershipError,
    EngineNotConfigured,
    MocapError,
    StaleChannelError,
)
from nomorals.tools.registry import ToolRegistry

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "kinect"


class _Clock:
    """A controllable monotonic clock for staleness tests."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _store(clock: _Clock | None = None, **kwargs) -> MocapStateStore:
    return MocapStateStore(clock=clock, **kwargs)


def _skeleton() -> Skeleton:
    tip = Joint(name="End Site", offset=(0.0, 5.0, 0.0),
                channels=(), is_end_site=True)
    spine = Joint(name="Spine", offset=(0.0, 10.0, 0.0),
                  children=[tip])
    root = Joint(name="Hips", offset=(0.0, 0.0, 0.0),
                 channels=("Xposition", "Yposition", "Zposition",
                           "Zrotation", "Xrotation", "Yrotation"),
                 children=[spine])
    return Skeleton(root=root, frame_time=1.0 / 30.0)


def _frames(n: int = 3) -> list[SkeletonFrame]:
    out = []
    for i in range(n):
        out.append(SkeletonFrame(values={
            "Hips": (0.0, float(100 + i), 0.0, 0.0, 0.0, 0.0),
            "Spine": (0.0, float(i), 0.0),
        }))
    return out


class StoreTest(unittest.TestCase):
    def test_write_then_read_round_trip(self):
        store = _store()
        store.register_engine("writer-a", produces=("skeleton",))
        receipt = store.write("skeleton", "writer-a",
                              {"joints": ["Hips"]})
        self.assertEqual(receipt["seq"], 1)
        view = store.read("skeleton")
        self.assertEqual(view["payload"], {"joints": ["Hips"]})
        self.assertEqual(view["seq"], 1)
        self.assertEqual(view["writer"], "writer-a")
        self.assertFalse(view["stale"])
        self.assertIsNone(view["max_age_s"])

    def test_read_unknown_channel_is_refused_by_name(self):
        store = _store()
        with self.assertRaises(MocapError) as ctx:
            store.read("nope")
        self.assertIn("nope", str(ctx.exception))

    def test_read_empty_channel_is_refused(self):
        store = _store()
        store.register_engine("writer-a")
        store.claim("skeleton", "writer-a")
        with self.assertRaises(MocapError):
            store.read("skeleton")

    def test_write_claims_a_free_channel(self):
        store = _store()
        store.register_engine("writer-a")
        store.write("skeleton", "writer-a", {"x": 1})
        self.assertEqual(store.owner("skeleton"), "writer-a")

    def test_non_owner_write_is_refused_and_recorded(self):
        store = _store()
        store.register_engine("writer-a")
        store.register_engine("writer-b")
        store.claim("skeleton", "writer-a")
        with self.assertRaises(ChannelOwnershipError) as ctx:
            store.write("skeleton", "writer-b", {"x": 1})
        self.assertIn("writer-a", str(ctx.exception))
        failures = store.failures("skeleton")
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["type"], "ownership")
        self.assertIn("writer-a", failures[0]["message"])

    def test_release_by_non_owner_is_refused(self):
        store = _store()
        store.register_engine("writer-a")
        store.register_engine("writer-b")
        store.claim("skeleton", "writer-a")
        with self.assertRaises(ChannelOwnershipError):
            store.release("skeleton", "writer-b")
        view = store.release("skeleton", "writer-a")
        self.assertIsNone(view["owner"])
        store.write("skeleton", "writer-b", {"x": 2})
        self.assertEqual(store.read("skeleton")["writer"], "writer-b")

    def test_unregister_releases_claims(self):
        store = _store()
        store.register_engine("writer-a")
        store.register_engine("writer-b")
        store.claim("skeleton", "writer-a")
        store.unregister_engine("writer-a")
        self.assertIsNone(store.owner("skeleton"))
        store.write("skeleton", "writer-b", {"x": 1})

    def test_duplicate_engine_is_refused(self):
        store = _store()
        store.register_engine("writer-a")
        with self.assertRaises(MocapError):
            store.register_engine("writer-a")

    def test_non_json_payload_is_refused(self):
        store = _store()
        store.register_engine("writer-a")
        with self.assertRaises(MocapError):
            store.write("skeleton", "writer-a", {"bad": object()})

    def test_stale_data_is_flagged_never_silently_served(self):
        clock = _Clock()
        store = _store(clock)
        store.register_engine("writer-a")
        store.write("pointcloud", "writer-a", {"n": 1}, max_age_s=1.0)
        clock.now += 0.5
        self.assertFalse(store.read("pointcloud")["stale"])
        clock.now += 1.0
        view = store.read("pointcloud")
        self.assertTrue(view["stale"])
        self.assertAlmostEqual(view["age_s"], 1.5, places=3)
        with self.assertRaises(StaleChannelError) as ctx:
            store.read("pointcloud", require_fresh=True)
        self.assertIn("stale", str(ctx.exception))

    def test_delta_catch_up_and_missed_count(self):
        store = _store(max_deltas=3)
        store.register_engine("writer-a")
        store.write("skeleton", "writer-a", {"take": 0},
                    kind="snapshot")
        for i in range(1, 7):
            store.write("skeleton", "writer-a", {"frame": i},
                        kind="delta")
        view = store.read("skeleton", since_seq=1)
        self.assertEqual([d["seq"] for d in view["changes"]],
                         [5, 6, 7])
        # Writes 2, 3, 4 rotated out of the bounded log.
        self.assertEqual(view["missed"], 3)
        self.assertEqual(view["payload"], {"take": 0})
        self.assertEqual(view["latest"], {"frame": 6})
        fresh = store.read("skeleton", since_seq=7)
        self.assertEqual(fresh["changes"], [])
        self.assertEqual(fresh["missed"], 0)

    def test_subscriber_poll_reemits_payload_and_tracks(self):
        store = _store()
        store.register_engine("writer-a")
        store.write("skeleton", "writer-a", {"take": 0})
        store.subscribe("anim", ["skeleton"],
                        payload={"seat": "front-row"})
        first = store.poll("anim")
        self.assertEqual(first["subscriber_payload"],
                         {"seat": "front-row"})
        self.assertEqual(first["updates"]["skeleton"]["changes"], [])
        store.write("skeleton", "writer-a", {"frame": 1}, kind="delta")
        second = store.poll("anim")
        self.assertEqual(
            [d["seq"] for d in
             second["updates"]["skeleton"]["changes"]], [2])
        third = store.poll("anim")
        self.assertEqual(third["updates"]["skeleton"]["changes"], [])

    def test_subscriber_missed_writes_are_reported(self):
        store = _store(max_deltas=2)
        store.register_engine("writer-a")
        store.write("skeleton", "writer-a", {"take": 0})
        store.subscribe("anim", ["skeleton"])
        for i in range(1, 5):
            store.write("skeleton", "writer-a", {"frame": i},
                        kind="delta")
        view = store.poll("anim")
        self.assertGreater(
            view["updates"]["skeleton"]["missed"], 0)

    def test_unknown_subscriber_poll_is_refused_by_name(self):
        store = _store()
        with self.assertRaises(MocapError) as ctx:
            store.poll("ghost")
        self.assertIn("ghost", str(ctx.exception))

    def test_snapshot_is_the_late_joiner_view(self):
        store = _store()
        store.register_engine("writer-a")
        store.write("skeleton", "writer-a", {"take": 1})
        store.write("pointcloud", "writer-a", {"n": 9})
        snap = store.snapshot()
        self.assertEqual(
            snap["channels"]["skeleton"]["payload"], {"take": 1})
        self.assertEqual(
            snap["channels"]["pointcloud"]["payload"], {"n": 9})
        self.assertEqual(len(snap["engines"]), 1)

    def test_save_and_load_round_trip(self):
        clock = _Clock()
        store = _store(clock)
        store.register_engine("writer-a")
        store.write("skeleton", "writer-a", {"take": 7})
        store.subscribe("anim", ["skeleton"], payload={"seat": 1})
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store.json"
            saved = store.save(path)
            self.assertTrue(Path(saved["path"]).exists())
            restored = MocapStateStore.load(path, clock=clock)
        self.assertEqual(
            restored.read("skeleton")["payload"], {"take": 7})
        self.assertEqual(restored.subscribers(), ["anim"])
        polled = restored.poll("anim")
        self.assertEqual(polled["subscriber_payload"], {"seat": 1})

    def test_load_missing_file_is_refused_by_name(self):
        with self.assertRaises(MocapError) as ctx:
            MocapStateStore.load("/tmp/definitely-not-a-store-125.json")
        self.assertIn("definitely-not-a-store-125", str(ctx.exception))


class RouterTest(unittest.TestCase):
    def test_route_picks_the_first_configured_engine(self):
        router = MocapRouter()
        router.register("alpha", ("mocap",), configured=False)
        router.register("beta", ("mocap",), configured=True)
        self.assertEqual(router.route("mocap")["name"], "beta")

    def test_route_to_unconfigured_engine_is_honest(self):
        router = MocapRouter()
        router.register("alpha", ("mocap",), configured=False,
                        note="no key")
        with self.assertRaises(EngineNotConfigured) as ctx:
            router.route("mocap")
        self.assertIn("alpha", str(ctx.exception))

    def test_route_unknown_category_is_refused(self):
        router = MocapRouter()
        router.register("alpha", ("mocap",))
        with self.assertRaises(MocapError) as ctx:
            router.route("teleport")
        self.assertIn("teleport", str(ctx.exception))

    def test_register_without_categories_is_refused(self):
        router = MocapRouter()
        with self.assertRaises(MocapError):
            router.register("alpha", ())

    def test_default_router_bvh_configured_kinect_not(self):
        from nomorals.hardware.kinect import probe as kinect_probe

        report = kinect_probe()
        router = default_router()
        bvh = router.route("mocap")
        self.assertEqual(bvh["name"], "bvh")
        self.assertTrue(bvh["configured"])
        with self.assertRaises(EngineNotConfigured) as ctx:
            router.route("capture")
        self.assertIn("kinect", str(ctx.exception))
        engines = {e["name"]: e for e in router.engines()}
        # The router must report what the probe reports: SDK absent
        # means the note says so; SDK present with no device means
        # the note says no device. Either way it is not configured.
        self.assertFalse(engines["kinect"]["configured"])
        if report.sdk_present:
            self.assertIn("no Kinect device", engines["kinect"]["note"])
        else:
            self.assertIn("SDK unavailable",
                          engines["kinect"]["note"])


class BvhTest(unittest.TestCase):
    def test_write_parse_round_trip_is_exact(self):
        skeleton = _skeleton()
        frames = _frames(3)
        text = write_bvh(skeleton, frames)
        self.assertIn("HIERARCHY", text)
        self.assertIn("ROOT Hips", text)
        self.assertIn("End Site", text)
        self.assertIn("Frames: 3", text)
        parsed, parsed_frames = parse_bvh(text)
        # BVH writes the frame time at 6 decimals, so the parse is
        # exact to 1e-6, not bit-exact.
        self.assertAlmostEqual(parsed.frame_time, skeleton.frame_time,
                               places=5)
        self.assertEqual(
            [j.name for j in parsed.root.children[0].children],
            ["End Site"])
        self.assertEqual(len(parsed_frames), 3)
        for want, got in zip(frames, parsed_frames, strict=True):
            for joint, values in want.values.items():
                for a, b in zip(values, got.values[joint], strict=True):
                    self.assertAlmostEqual(a, b, places=5)

    def test_skeleton_dict_round_trip(self):
        skeleton = _skeleton()
        rebuilt = skeleton_from_dict(skeleton_to_dict(skeleton))
        self.assertEqual(
            rebuilt.root.children[0].name, "Spine")
        self.assertEqual(
            rebuilt.root.children[0].children[0].is_end_site, True)

    def test_frame_missing_a_joint_is_refused(self):
        skeleton = _skeleton()
        bad = SkeletonFrame(values={"Hips": (0, 0, 0, 0, 0, 0)})
        with self.assertRaises(MocapError) as ctx:
            write_bvh(skeleton, [bad])
        self.assertIn("Spine", str(ctx.exception))

    def test_malformed_bvh_is_refused(self):
        for text in ("", "HIERARCHY\nROOT Hips\n", "not bvh at all"):
            with self.assertRaises(MocapError, msg=text[:20]):
                parse_bvh(text)

    def test_end_site_carries_no_channels(self):
        skeleton = _skeleton()
        text = write_bvh(skeleton, _frames(1))
        parsed, _ = parse_bvh(text)
        tip = parsed.root.children[0].children[0]
        self.assertEqual(tip.channels, ())
        self.assertTrue(tip.is_end_site)


class BridgeTest(unittest.TestCase):
    def test_fixture_publishes_depth_and_pointcloud(self):
        store = _store()
        producer = KinectProducer(store)
        receipt = producer.publish_fixture(FIXTURES)
        self.assertEqual(receipt["points_total"], 640 * 576)
        self.assertGreater(receipt["points_sampled"], 1000)
        depth = store.read("depth")
        self.assertEqual(depth["payload"]["valid_pixels"], 368540)
        self.assertEqual(depth["payload"]["zero_pixels"], 100)
        self.assertAlmostEqual(depth["payload"]["mean_mm"], 1071.6,
                               delta=1.0)
        cloud = store.read("pointcloud")
        self.assertEqual(cloud["payload"]["count"], 640 * 576)
        self.assertEqual(cloud["payload"]["source"], "fixture")
        self.assertEqual(len(cloud["payload"]["points"][0]), 3)

    def test_fixture_missing_dir_is_refused_by_name(self):
        store = _store()
        producer = KinectProducer(store)
        with self.assertRaises(MocapError) as ctx:
            producer.publish_fixture("/tmp/no-such-fixtures-125")
        self.assertIn("no-such-fixtures-125", str(ctx.exception))

    def test_bvh_take_round_trips_through_the_store(self):
        store = _store()
        playback = BvhPlaybackProducer(store)
        skeleton = _skeleton()
        playback.load_take(skeleton, _frames(3))
        take = store.read("skeleton")
        self.assertEqual(take["payload"]["frame_count"], 3)
        self.assertEqual(take["payload"]["dropped_frames"], 0)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "take.bvh"
            exporter = BvhExporter()
            result = exporter.export_from_store(store, out)
            self.assertEqual(result["frames"], 3)
            self.assertTrue(Path(result["path"]).is_absolute())
            self.assertTrue(out.exists())
            back = exporter.read_bvh(out)
            self.assertEqual(back["frames"], 3)
            self.assertEqual(back["joints"],
                             ["Hips", "Spine", "End Site"])
            bvh_channel = store.read("bvh")
            self.assertEqual(bvh_channel["payload"]["path"],
                             str(out.resolve()))
            # A second export on the same store must not fail on a
            # duplicate engine registration.
            exporter.export_from_store(store, out)

    def test_take_bound_drops_oldest_frames_and_says_so(self):
        store = _store()
        playback = BvhPlaybackProducer(store, max_frames=2)
        playback.load_take(_skeleton(), _frames(3))
        self.assertEqual(playback.frame_count, 2)
        self.assertEqual(playback.dropped_frames, 1)
        take = store.read("skeleton")
        self.assertEqual(take["payload"]["dropped_frames"], 1)

    def test_export_without_a_take_is_refused(self):
        store = _store()
        exporter = BvhExporter()
        import tempfile
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(MocapError):
            exporter.export_from_store(store, Path(tmp) / "x.bvh")

    def test_calibration_fixture_parses(self):
        calib = parse_raw_calibration(
            (FIXTURES / "calibration.json").read_text(encoding="utf-8"))
        self.assertIsNotNone(calib)


class ToolTest(unittest.TestCase):
    def test_tool_count_is_782_after_mocap(self):
        # 774 after the secprobe wire-up (749 after the mocap tool): no existing tool reaches the
        # cross-engine state store (Kinect has no tool, anicraft is
        # 2D vector animation, media is downloads), so the store
        # needs its own action-dispatched entry.
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        # 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p.
        self.assertEqual(len(registry._tools), 782)
        self.assertIn("mocap", registry._tools)

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

    def test_tool_publish_read_snapshot_round_trip(self):
        channel = "tool-skeleton-125"
        self._call(action="register", node="tool-writer",
                   params={"produces": [channel]})
        out = self._call(action="publish", node=channel,
                         value="tool-writer",
                         params={"payload": {"take": 3}})
        self.assertEqual(out["seq"], 1)
        view = self._call(action="read", node=channel)
        self.assertEqual(view["payload"], {"take": 3})
        snap = self._call(action="snapshot")
        self.assertIn(channel, snap["channels"])

    def test_tool_subscribe_and_poll(self):
        channel = "tool-poll-125"
        self._call(action="register", node="tool-writer-poll",
                   params={"produces": [channel]})
        self._call(action="publish", node=channel, value="tool-writer-poll",
                   params={"payload": {"v": 0}})
        self._call(action="subscribe", node="tool-sub-125",
                   params={"channels": [channel],
                           "payload": {"seat": "a"}})
        polled = self._call(action="poll", node="tool-sub-125")
        self.assertEqual(polled["subscriber_payload"], {"seat": "a"})
        self.assertEqual(
            polled["updates"][channel]["changes"], [])

    def test_tool_route_and_unknown_action(self):
        routed = self._call(action="route", value="mocap")
        self.assertEqual(routed["name"], "bvh")
        outcome = self._raw_call(action="nope")
        self.assertFalse(getattr(outcome, "ok", True))

    def test_tool_kinect_fixture_and_status(self):
        receipt = self._call(action="kinect_fixture")
        self.assertEqual(receipt["points_total"], 640 * 576)
        status = self._call(action="kinect_status")
        self.assertEqual(status["device_count"], 0)

    def test_tool_bvh_write_and_read(self):
        import tempfile
        skeleton = _skeleton()
        take = {
            "skeleton": skeleton_to_dict(skeleton),
            "frames": [{name: list(values)
                        for name, values in frame.values.items()}
                       for frame in _frames(2)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            take_path = Path(tmp) / "take.json"
            take_path.write_text(json.dumps(take), encoding="utf-8")
            out_path = Path(tmp) / "out.bvh"
            written = self._call(action="bvh_write",
                                 sources=[str(take_path)],
                                 output=str(out_path))
            self.assertEqual(written["frames"], 2)
            self.assertTrue(out_path.exists())
            back = self._call(action="bvh_read",
                              sources=[str(out_path)])
            self.assertEqual(back["frames"], 2)
            self.assertEqual(back["joints"],
                             ["Hips", "Spine", "End Site"])


if __name__ == "__main__":
    unittest.main()
