"""Lane B.22 — census §3B.22: Devon's own Bevy-style show organ
(storyteller-bevy patterns, pattern only, no upstream code).

The clone (storytold/storyteller-bevy) is a Bevy 0.13.2 studio
monorepo: a Bevy app (`studio/`) + a BVH mocap crate
(`studio-bvh/`), compiled to wasm32. The patterns taken are
Bevy's own: an App plugins extend, named States with
OnEnter/OnExit hooks, a Schedule of ordered phases, typed
Resources, and Events drained at a schedule boundary.

Devon already HAS the organs those patterns would coordinate —
the mocap state store (one store, ownership, staleness), the
VTuber rig sessions, the cloudcam sessions — but nothing drives
them together through explicit, named states. A show is started
by hand, in any order, with no guard: nothing refuses "go live
with no rig", nothing runs capture → drive → render → publish
in a fixed order, nothing records one show's receipts.

Proves: the state machine (every legal transition works, every
illegal one is refused naming both states), the plugin protocol
(a plugin that needs a missing resource fails at build by name),
phase order (capture → drive → render → publish, proven by a
recording plugin), the take player (a BVH take steps
frame-by-frame into the store's skeleton channel and the rig
drives it), pause holds the pose, stop exports BVH + releases
the skeleton claim + stops the show-owned cloudcam poller, the
bounded event journal (drops counted, never silent), the show
brain tool, the `nm show` CLI, and the /live chat verb — with
the tool count pinned at 786 (776 after B.20 added cloudcam;
777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber; 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class); 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue).
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.media.mocap import MocapStateStore
from nomorals.media.mocap.bvh import (
    Joint,
    Skeleton,
    SkeletonFrame,
    parse_bvh,
    skeleton_to_dict,
    write_bvh,
)
from nomorals.media.rig import default_remap, default_rig
from nomorals.media.rig.session import RigSession
from nomorals.media.show import (
    PHASES,
    Show,
    ShowError,
    ShowState,
    TakePlayer,
    drop_show,
    get_show,
    list_shows,
    load_show,
    reset_shows,
)
from nomorals.media.show.states import TRANSITIONS, legal
from nomorals.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _clean_shows():
    reset_shows()
    from nomorals.media.cloudcam.registry import (
        reset_sessions as reset_cloudcam,
    )
    from nomorals.media.rig.registry import (
        reset_sessions as reset_rig,
    )
    reset_cloudcam()
    reset_rig()
    yield
    for entry in list_shows():
        with contextlib.suppress(Exception):
            drop_show(entry["name"])
    reset_shows()
    reset_cloudcam()
    reset_rig()


def _take_skeleton() -> Skeleton:
    head = Joint(name="Head", offset=(0.0, 10.0, 0.0),
                 channels=("Zrotation", "Xrotation", "Yrotation"),
                 children=[])
    hips = Joint(name="Hips", offset=(0.0, 0.0, 0.0),
                 channels=("Xposition", "Yposition", "Zposition",
                           "Zrotation", "Xrotation", "Yrotation"),
                 children=[head])
    return Skeleton(root=hips)


def _take_frames() -> list[SkeletonFrame]:
    return [
        SkeletonFrame(values={
            "Hips": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "Head": (0.0, float(i * 15.0), 0.0)})
        for i in range(4)
    ]


def _write_bvh(tmp_path: Path) -> Path:
    path = tmp_path / "take.bvh"
    path.write_text(write_bvh(_take_skeleton(), _take_frames()),
                    encoding="utf-8")
    return path


def _rig() -> RigSession:
    return RigSession(default_rig(), default_remap("bvh"),
                      width=160, height=90, fps=15.0)


def _cloudcam(name: str = "cam"):
    from nomorals.media.cloudcam.registry import load_session
    return load_session(name, source="idle")


def _show(name: str = "demo", *, take: bool = True,
          rig: bool = True, cloudcam: bool = True,
          store: MocapStateStore | None = None) -> Show:
    store = store if store is not None else MocapStateStore()
    show = Show(name, store,
                rig=_rig() if rig else None,
                cloudcam=_cloudcam() if cloudcam else None)
    if take:
        show.load_take_frames(_take_skeleton(), _take_frames())
    return show


def _err(outcome) -> str:
    return str(getattr(outcome.error, "message", outcome.error))


def _tool(registry, **kwargs):
    outcome = registry.call("show", **kwargs)
    if getattr(outcome, "ok", False):
        return outcome.value
    raise AssertionError(
        f"show call failed: {getattr(outcome, 'error', outcome)}")


def _registry(tmp_path):
    registry = ToolRegistry(SimpleNamespace(home=str(tmp_path)))
    registry.register_builtins()
    return registry


def test_tool_count_is_786_after_show():
    # 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue).
    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    assert len(registry._tools) == 794
    assert "show" in registry._tools


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

def test_state_table_is_exact():
    assert set(TRANSITIONS) == set(ShowState)
    assert TRANSITIONS[ShowState.IDLE] == {ShowState.ARMING}
    assert TRANSITIONS[ShowState.ARMING] == {
        ShowState.LIVE, ShowState.STOPPING, ShowState.FAILED}
    assert TRANSITIONS[ShowState.LIVE] == {
        ShowState.PAUSED, ShowState.STOPPING, ShowState.FAILED}
    assert TRANSITIONS[ShowState.PAUSED] == {
        ShowState.LIVE, ShowState.STOPPING, ShowState.FAILED}
    assert TRANSITIONS[ShowState.STOPPING] == {
        ShowState.DONE, ShowState.FAILED}
    assert TRANSITIONS[ShowState.DONE] == {ShowState.ARMING}
    assert TRANSITIONS[ShowState.FAILED] == {ShowState.ARMING}
    assert legal(ShowState.IDLE, ShowState.ARMING) is True
    assert legal(ShowState.LIVE, ShowState.IDLE) is False
    assert legal(ShowState.DONE, ShowState.LIVE) is False


def test_illegal_request_refused_naming_both_states():
    show = _show()
    assert show.state is ShowState.IDLE
    with pytest.raises(ShowError) as exc:
        show.request(ShowState.LIVE)
    assert exc.value.kind == "bad_transition"
    assert "IDLE" in str(exc.value) and "LIVE" in str(exc.value)
    assert "ARMING" in str(exc.value)  # the legal exit is named
    assert show.state is ShowState.IDLE
    assert show.pending is None


def test_request_queues_and_step_applies_at_the_boundary():
    show = _show()
    show.request(ShowState.ARMING)
    assert show.state is ShowState.IDLE  # not applied yet
    assert show.pending is ShowState.ARMING
    receipt = show.step()
    assert show.state is ShowState.ARMING
    assert show.pending is None
    assert receipt["state"] == "arming"
    assert receipt["tick"] == 1


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_start_runs_arming_then_live():
    show = _show()
    status = show.start()
    assert status["state"] == "live"
    assert show.state is ShowState.LIVE
    assert status["tick"] >= 2
    # Arming claimed the skeleton channel for the take player.
    assert show.store.owner("skeleton") == "show:demo"


def test_start_refuses_without_a_rig_by_name():
    show = _show(rig=False)
    with pytest.raises(ShowError) as exc:
        show.start()
    assert exc.value.kind == "no_rig"
    assert "no rig" in str(exc.value)
    assert show.state is ShowState.FAILED
    assert show.last_error is not None
    assert "no rig" in show.last_error


def test_start_twice_and_pause_resume_guards():
    show = _show()
    show.start()
    with pytest.raises(ShowError) as exc:
        show.start()
    assert exc.value.kind == "bad_transition"
    assert "LIVE" in str(exc.value)
    show.pause()
    assert show.state is ShowState.PAUSED
    with pytest.raises(ShowError):
        show.pause()
    show.resume()
    assert show.state is ShowState.LIVE
    with pytest.raises(ShowError):
        show.resume()
    show.stop()
    assert show.state is ShowState.DONE


def test_stop_from_idle_refused():
    show = _show()
    with pytest.raises(ShowError) as exc:
        show.stop()
    assert exc.value.kind == "bad_transition"
    assert "IDLE" in str(exc.value)


def test_stop_releases_the_skeleton_claim():
    show = _show()
    show.start()
    assert show.store.owner("skeleton") == "show:demo"
    show.stop()
    assert show.store.owner("skeleton") is None
    assert "show:demo" not in {
        e["name"] for e in show.store.engines()}


def test_stop_stops_a_show_owned_cloudcam_poller_only():
    show = _show()
    cam = show.resource("cloudcam")
    cam.start()
    assert cam.running is True
    show.start()
    show.stop()
    # The poller was running before the show started it: the show
    # leaves what it did not start alone.
    assert cam.running is True
    cam.stop()

    show2 = _show("demo2")
    cam2 = show2.resource("cloudcam")
    assert cam2.running is False
    show2.start()  # arming starts the attached cloudcam
    assert cam2.running is True
    show2.stop()
    assert cam2.running is False


def test_failed_show_can_arm_again_after_a_rig_is_attached():
    show = _show(rig=False)
    with pytest.raises(ShowError):
        show.start()
    assert show.state is ShowState.FAILED
    show.attach_rig(_rig())
    show.start()
    assert show.state is ShowState.LIVE
    show.stop()


def test_done_show_can_run_again():
    show = _show()
    show.start()
    show.stop()
    assert show.state is ShowState.DONE
    show.start()
    assert show.state is ShowState.LIVE
    show.stop()


# ---------------------------------------------------------------------------
# Phases run in order, systems are plugins
# ---------------------------------------------------------------------------

def test_phases_run_capture_drive_render_publish_in_order():
    seen: list[str] = []

    class Recorder:
        def __init__(self, phase: str) -> None:
            self.phase = phase

        def build(self, show: Show) -> None:
            show.system(self.phase,
                        lambda s: seen.append(self.phase))

    show = _show(take=False, cloudcam=False)
    for phase in PHASES:
        show.add_plugin(Recorder(phase))
    show.start()
    show.step()
    # start() runs one live tick itself, then step() runs another.
    assert seen == ["capture", "drive", "render", "publish"] * 2
    show.stop()


def test_unknown_phase_refused_by_name():
    show = _show()
    with pytest.raises(ShowError) as exc:
        show.system("encore", lambda s: None)
    assert exc.value.kind == "bad_phase"
    assert "encore" in str(exc.value)


def test_plugin_needing_a_missing_resource_fails_at_build():
    class Needy:
        def build(self, show: Show) -> None:
            show.resource("spotlight")  # nothing provides it

    show = _show()
    with pytest.raises(ShowError) as exc:
        show.add_plugin(Needy())
    assert exc.value.kind == "missing_resource"
    assert "spotlight" in str(exc.value)


def test_resources_add_set_and_refuse_duplicates():
    show = _show()
    assert show.resource("store") is show.store
    show.add_resource("spotlight", object())
    with pytest.raises(ShowError) as exc:
        show.add_resource("spotlight", object())
    assert exc.value.kind == "duplicate_resource"
    replacement = object()
    show.set_resource("spotlight", replacement)
    assert show.resource("spotlight") is replacement
    with pytest.raises(ShowError) as exc:
        show.resource("ghost")
    assert exc.value.kind == "missing_resource"


def test_system_error_is_an_event_not_a_dead_show():
    def boom(show: Show) -> None:
        raise ShowError("boom in the encore", kind="encore")

    show = _show(take=False, cloudcam=False)
    show.system("capture", boom)
    show.start()
    receipt = show.step()
    assert show.state is ShowState.LIVE
    assert receipt["phases"]["capture"]["errors"] == ["encore: boom "
                                                      "in the encore"]
    kinds = [e["kind"] for e in show.events()]
    assert "system_error" in kinds
    show.stop()


# ---------------------------------------------------------------------------
# Take player + drive + render + publish
# ---------------------------------------------------------------------------

def test_take_player_steps_frames_into_the_store():
    store = MocapStateStore()
    show = _show(store=store)
    show.start()
    assert show.take is not None
    assert show.take.total == 4
    assert show.take.index == 1  # the start's live step played frame 0
    show.step()
    assert show.take.index == 2
    view = store.read("skeleton")
    assert view["payload"]["frames"][0]["Head"] == [0.0, 15.0, 0.0]
    show.run(3)
    assert show.take.exhausted is True
    assert show.take.index == 4
    kinds = [e["kind"] for e in show.events()]
    assert "take_finished" in kinds
    show.stop()


def test_take_player_from_bvh_file(tmp_path):
    path = _write_bvh(tmp_path)
    player = TakePlayer.from_bvh(path)
    assert player.total == 4
    assert player.index == 0
    first = player.next()
    assert first is not None
    assert first.values["Head"] == (0.0, 0.0, 0.0)
    assert player.index == 1
    for _ in range(3):
        assert player.next() is not None
    assert player.exhausted is True
    assert player.next() is None
    player.reset()
    assert player.exhausted is False
    assert player.index == 0
    with pytest.raises(ShowError) as exc:
        TakePlayer.from_bvh(tmp_path / "missing.bvh")
    assert exc.value.kind == "missing_take"


def test_take_player_from_json(tmp_path):
    blob = {"skeleton": skeleton_to_dict(_take_skeleton()),
            "frames": [{name: list(values)
                        for name, values in frame.values.items()}
                       for frame in _take_frames()]}
    path = tmp_path / "take.json"
    path.write_text(json.dumps(blob), encoding="utf-8")
    player = TakePlayer.from_json(path)
    assert player.total == 4
    assert player.next() is not None


def test_live_tick_drives_renders_and_publishes():
    show = _show()
    show.start()
    receipt = show.step()
    assert receipt["state"] == "live"
    assert receipt["pose"]["seq"] >= 0
    assert receipt["pose"]["viseme"] == "sil"  # no face input: silence
    assert receipt["rendered_bytes"] == 160 * 90 * 3
    assert receipt["cloudcam"]["placeholder"] is True
    assert receipt["cloudcam"]["point_count"] == 0
    assert show.posed_count >= 1
    assert show.last_frame is not None
    assert len(show.last_frame) == 160 * 90 * 3
    show.stop()


def test_drive_holds_when_the_take_is_done_and_no_live_frame():
    show = _show()
    show.start()
    show.run(5)  # take has 4 frames; the 5th tick holds
    assert show.take.exhausted is True
    kinds = [e["kind"] for e in show.events()]
    assert "take_finished" in kinds
    assert "drive_held" in kinds
    assert show.state is ShowState.LIVE
    show.stop()


def test_drive_holds_with_no_take_and_no_channel():
    show = _show(take=False)
    show.start()
    receipt = show.step()
    assert receipt["pose"]["seq"] >= 0
    assert receipt["held"] is True
    show.stop()


def test_run_is_bounded_and_stops_early_on_failure(tmp_path):
    show = _show(rig=False)
    with pytest.raises(ShowError):
        show.run(10)
    assert show.state is ShowState.FAILED
    with pytest.raises(ShowError) as exc:
        _show().run(0)
    assert exc.value.kind == "bad_run"
    with pytest.raises(ShowError) as exc:
        _show().run(3601)
    assert exc.value.kind == "bad_run"


def test_export_writes_real_bvh_and_records_it(tmp_path):
    store = MocapStateStore()
    show = _show(store=store)
    show.start()
    show.run(4)
    out = tmp_path / "show.bvh"
    result = show.export(out)
    assert result["frames"] == show.posed_count
    assert result["path"] == str(out.resolve())
    skeleton, frames = parse_bvh(out.read_text(encoding="utf-8"))
    assert len(frames) == show.posed_count
    # The export carries the RIG's skeleton, not the take's: the
    # root is the rig's root bone.
    rig = show.resource("rig")
    root_bone = next(b for b in rig.rig.bones if b.parent is None)
    assert skeleton.root.name == root_bone.name
    # The export is recorded on the store's bvh channel.
    view = store.read("bvh")
    assert view["payload"]["rig"] == show.resource("rig").rig.name
    show.stop()


def test_export_refuses_with_no_poses(tmp_path):
    show = _show()
    with pytest.raises(ShowError) as exc:
        show.export(tmp_path / "empty.bvh")
    assert exc.value.kind == "no_poses"


def test_stop_auto_exports_when_export_path_configured(tmp_path):
    out = tmp_path / "auto.bvh"
    show = _show()
    show.export_path = out
    show.start()
    show.run(3)
    posed = show.posed_count  # 1 live tick from start + 3 from run
    show.stop()
    assert out.exists()
    _, frames = parse_bvh(out.read_text(encoding="utf-8"))
    assert len(frames) == posed


def test_render_dir_writes_a_png_per_tick(tmp_path):
    show = _show()
    show.render_dir = tmp_path / "frames"
    show.start()
    show.run(2)
    written = sorted((tmp_path / "frames").glob("frame-*.png"))
    assert len(written) == 3  # 1 live tick from start + 2 from run
    assert written[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    show.stop()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def test_events_journal_is_bounded_and_counts_drops():
    show = _show()
    show.start()
    show.run(3)
    events = show.events()
    kinds = [e["kind"] for e in events]
    assert "state_enter" in kinds
    assert "state_exit" in kinds
    assert all("tick" in e and "at" in e for e in events)
    show.stop()
    limited = show.events(limit=2)
    assert len(limited) == 2
    assert limited == show.events()[-2:]


def test_event_journal_drops_past_the_cap():
    show = _show()
    for _ in range(300):
        show._record("spam", {"n": 1})
    assert len(show.events()) == 256
    assert show.events_dropped == 44


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_load_get_list_drop_reset():
    a = load_show("a", MocapStateStore())
    b = load_show("b", MocapStateStore())
    assert get_show("a") is a
    assert get_show("b") is b
    assert {s["name"] for s in list_shows()} == {"a", "b"}
    with pytest.raises(ShowError) as exc:
        get_show("ghost")
    assert exc.value.kind == "unknown_show"
    assert "no show" in str(exc.value)
    drop_show("a")
    assert {s["name"] for s in list_shows()} == {"b"}
    reset_shows()
    assert list_shows() == []


def test_registry_replace_stops_the_old_show():
    old = load_show("x", MocapStateStore(), rig=_rig())
    old.start()
    assert old.state is ShowState.LIVE
    new = load_show("x", MocapStateStore())
    assert new is not old
    assert old.state is ShowState.DONE
    assert new.state is ShowState.IDLE


def test_registry_drop_stops_a_live_show_first():
    show = load_show("y", MocapStateStore(), rig=_rig())
    show.start()
    # Stopping is fine, so the drop succeeds after stopping it.
    drop_show("y")
    assert show.state is ShowState.DONE
    assert list_shows() == []


def test_load_show_needs_a_store():
    with pytest.raises(ShowError) as exc:
        load_show("z", None)  # type: ignore[arg-type]
    assert exc.value.kind == "bad_show"


def test_load_show_refuses_an_empty_name():
    with pytest.raises(ShowError) as exc:
        load_show("", MocapStateStore())
    assert exc.value.kind == "bad_show"


# ---------------------------------------------------------------------------
# Brain tool
# ---------------------------------------------------------------------------

def test_tool_create_list_status_start_tick_stop(tmp_path):
    registry = _registry(tmp_path)
    created = _tool(registry, action="create", node="tooldemo")
    assert created["name"] == "tooldemo"
    assert created["state"] == "idle"
    names = [s["name"] for s in _tool(registry, action="list")["shows"]]
    assert "tooldemo" in names
    status = _tool(registry, action="status", node="tooldemo")
    assert status["name"] == "tooldemo"
    assert status["state"] == "idle"
    started = _tool(registry, action="start", node="tooldemo")
    assert started["state"] == "live"
    receipt = _tool(registry, action="tick", node="tooldemo")
    assert receipt["state"] == "live"
    assert receipt["pose"]["seq"] >= 0
    stopped = _tool(registry, action="stop", node="tooldemo")
    assert stopped["state"] == "done"
    dropped = _tool(registry, action="drop", node="tooldemo")
    assert dropped["ok"] is True


def test_tool_unknown_action_and_missing_show(tmp_path):
    registry = _registry(tmp_path)
    bad = registry.call("show", action="teleport")
    assert not bad.ok
    assert "unknown show action" in _err(bad)
    missing = registry.call("show", action="status", node="ghost")
    assert not missing.ok
    assert "no show" in _err(missing)


def test_tool_take_run_export_events(tmp_path):
    registry = _registry(tmp_path)
    bvh = _write_bvh(tmp_path)
    _tool(registry, action="create", node="take",
          params={"render_dir": str(tmp_path / "frames")})
    _tool(registry, action="load_take", node="take",
          sources=[str(bvh)])
    started = _tool(registry, action="start", node="take")
    assert started["state"] == "live"
    ran = _tool(registry, action="run", node="take",
                params={"ticks": 4})
    assert ran["state"] == "live"
    assert ran["tick"] >= 4
    events = _tool(registry, action="events", node="take")
    assert any(e["kind"] == "take_finished"
               for e in events["events"])
    out = tmp_path / "take.bvh"
    exported = _tool(registry, action="export", node="take",
                     output=str(out))
    assert exported["frames"] >= 4
    assert out.exists()
    assert len(list((tmp_path / "frames").glob("frame-*.png"))) >= 4
    _tool(registry, action="stop", node="take")
    _tool(registry, action="drop", node="take")


def test_tool_attach_rig_and_cloudcam(tmp_path):
    registry = _registry(tmp_path)
    from nomorals.media.rig.registry import (
        load_session as load_rig,
    )
    load_rig("custom")
    _tool(registry, action="create", node="bare")
    attached = _tool(registry, action="attach_rig", node="bare",
                     value="custom")
    assert attached["ok"] is True
    # The return names the RIG, not the session: "custom" is the
    # session name; the rig it loads is the default rig.
    assert attached["rig"] == "devon-default"
    started = _tool(registry, action="start", node="bare")
    assert started["state"] == "live"
    # The rig was attached already, so start auto-attached only
    # the idle cloudcam — and says so.
    assert started["auto_attached"] == ["cloudcam:idle"]
    assert started["cloudcam"]["source"] == "idle"
    attached = _tool(registry, action="attach_cloudcam", node="bare",
                     value="default")
    assert attached["ok"] is True
    _tool(registry, action="stop", node="bare")
    _tool(registry, action="drop", node="bare")


def test_tool_start_auto_attaches_defaults_and_names_them(tmp_path):
    registry = _registry(tmp_path)
    _tool(registry, action="create", node="auto")
    started = _tool(registry, action="start", node="auto")
    assert started["state"] == "live"
    assert started["auto_attached"] == ["rig:default",
                                        "cloudcam:idle"]
    _tool(registry, action="stop", node="auto")
    _tool(registry, action="drop", node="auto")


def test_tool_create_with_take_and_cloudcam_source(tmp_path):
    registry = _registry(tmp_path)
    bvh = _write_bvh(tmp_path)
    created = _tool(registry, action="create", node="full",
                    params={"take": str(bvh), "cloudcam_source": "idle"})
    assert created["state"] == "idle"
    assert created["take"]["total"] == 4
    assert created["cloudcam"]["source"] == "idle"
    _tool(registry, action="start", node="full")
    _tool(registry, action="stop", node="full")
    _tool(registry, action="drop", node="full")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_create_list_status_start_tick_stop(tmp_path):
    from nomorals.cmdline.dispatch import main

    assert main(["show", "list"]) == 0
    assert main(["show", "create", "--session", "cli"]) == 0
    assert main(["show", "status", "--session", "cli"]) == 0
    assert main(["show", "start", "--session", "cli"]) == 0
    assert main(["show", "tick", "--session", "cli"]) == 0
    assert main(["show", "events", "--session", "cli"]) == 0
    assert main(["show", "stop", "--session", "cli"]) == 0
    assert main(["show", "drop", "--session", "cli"]) == 0
    assert main(["show", "status", "--session", "ghost"]) == 1


def test_cli_run_exports_bvh(tmp_path):
    from nomorals.cmdline.dispatch import main

    bvh = _write_bvh(tmp_path)
    out = tmp_path / "cli.bvh"
    assert main(["show", "create", "--session", "clirun",
                 "--take", str(bvh)]) == 0
    assert main(["show", "run", "--session", "clirun",
                 "--ticks", "4", "--export", str(out)]) == 0
    assert out.exists()
    _, frames = parse_bvh(out.read_text(encoding="utf-8"))
    assert len(frames) >= 4
    assert main(["show", "stop", "--session", "clirun"]) == 0
    assert main(["show", "drop", "--session", "clirun"]) == 0


def test_cli_bad_ticks_fails():
    from nomorals.cmdline.dispatch import main

    assert main(["show", "create", "--session", "bad"]) == 0
    assert main(["show", "run", "--session", "bad",
                 "--ticks", "0"]) == 1
    assert main(["show", "drop", "--session", "bad"]) == 0


# ---------------------------------------------------------------------------
# Chat verb
# ---------------------------------------------------------------------------

def test_chat_control_live():
    from nomorals.agents.partner.runtime_mocap import RuntimeMocapMixin

    mixin = RuntimeMocapMixin()
    help_text = mixin._control_live("", "test")
    assert "/live create" in help_text
    assert "shows:" in mixin._control_live("list", "test")
    created = mixin._control_live("create chatdemo", "test")
    assert "created" in created
    started = mixin._control_live("start chatdemo", "test")
    assert "live" in started
    ticked = mixin._control_live("tick chatdemo", "test")
    assert "tick" in ticked
    status = mixin._control_live("status chatdemo", "test")
    assert '"name": "chatdemo"' in status
    assert "stopped" in mixin._control_live("stop chatdemo", "test")
    assert "dropped" in mixin._control_live("drop chatdemo", "test")
    assert "/live create" in mixin._control_live("bogus", "test")


def test_chat_control_mocap_and_cloudcam_still_work():
    from nomorals.agents.partner.runtime_mocap import RuntimeMocapMixin

    mixin = RuntimeMocapMixin()
    assert "rig list" in mixin._control_mocap("", "test")
    assert "/cloudcam start" in mixin._control_cloudcam("", "test")


def test_chat_live_registered_in_the_catalog():
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
        help_text,
        list_catalog,
    )

    assert "live" in CONTROL_COMMANDS
    assert CONTROL_COMMANDS["live"] == (0, None)
    details = COMMAND_DETAILS["live"]
    assert details["what"] and details["usage"] and details["example"]
    assert "/live" in details["usage"]
    assert "live" in LIST_ONELINERS
    assert "/live" in LIST_ONELINERS["live"]
    slots = [group for group in LIST_GROUPS
             if "live" in group[1]]
    assert len(slots) == 1  # exactly one LIST_GROUPS slot
    assert "live" in help_text()
    assert "live" in list_catalog()


def test_chat_live_dispatches_through_handle_control():
    from nomorals.agents.partner.runtime_mocap import RuntimeMocapMixin
    from nomorals.social.chat.control import parse_control

    command = parse_control("/live list")
    assert command is not None
    assert command.kind == "live"
    # The mixin method is the dispatch target; the full
    # handle_control probe for every registered kind runs in
    # test_command_registry_coverage.
    mixin = RuntimeMocapMixin()
    assert "shows:" in mixin._control_live("list", "test")
