"""Actor spawning (§3B.19, actorspawningplugin PATTERN): per-scene
spawn into Scene3D documents — placement, lifecycle, budget,
cleanup, and the CLI + tool surfaces.

The registry counts 776 tools: 774 after the secprobe wire-up
added 25 tools, 775 after this lane (B.19) added scene_spawn. 776 after this lane (B.20) added cloudcam. 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet). 801 after gapfix124 added browser_watch. 802 after this lane (SongRecon surface) added songrecon. 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 807 after this lane (agents192) added fleet.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.media.edit_engine.scene3d import (
    AnimKey,
    AnimTrack,
    Camera,
    Mesh,
    Rotation,
    Scene3D,
    Scene3DError,
    SceneNode,
    Transform,
)
from nomorals.media.edit_engine.spawn import (
    Placement,
    SceneSpawner,
    SpawnBudgetError,
    SpawnError,
    SpawnRequest,
    actors_in_file,
    despawn_from_file,
    spawn_into_file,
    tick_file,
)
from nomorals.tools.registry import ToolRegistry

MINIMAL_BVH = """HIERARCHY
ROOT hips
{
    OFFSET 0.0 1.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT spine
    {
        OFFSET 0.0 0.2 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
            OFFSET 0.0 0.3 0.0
        }
    }
}
MOTION
Frames: 1
Frame Time: 0.033333
0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
"""


def _scene() -> Scene3D:
    return Scene3D(
        title="spawn-test",
        cameras={"main": Camera(
            transform=Transform(
                position=(0.0, 0.0, 5.0),
                rotation=Rotation.look_at((0.0, 0.0, 0.0))))},
        active_camera="main",
        nodes=[SceneNode(
            name="anchor",
            transform=Transform(position=(1.0, 2.0, 0.0)))],
    )


def _scene_file(tmp_path: Path) -> str:
    path = tmp_path / "scene.json"
    _scene().save(path)
    return str(path)


def _registry(tmp_path: Path) -> ToolRegistry:
    registry = ToolRegistry(SimpleNamespace(home=str(tmp_path)))
    registry.register_builtins()
    return registry


def _tool(registry: ToolRegistry, **kwargs: object) -> dict:
    # The registry wraps results: unwrap Ok, fail loudly on Err.
    outcome = registry.call("scene_spawn", **kwargs)
    if getattr(outcome, "ok", False):
        return outcome.value
    raise AssertionError(
        f"scene_spawn call failed: {getattr(outcome, 'error', outcome)}")


def _raw(registry: ToolRegistry, **kwargs: object):
    return registry.call("scene_spawn", **kwargs)


def _err(outcome) -> str:
    return str(getattr(outcome.error, "message", outcome.error))


# ---------------------------------------------------------------------------
# tool count pin
# ---------------------------------------------------------------------------

def test_tool_count_is_787_after_actorspawn():
    # 774 after the secprobe wire-up added 25 tools. 775 after this
    # lane (B.19) added scene_spawn. 776 after B.20 added cloudcam.
    # 777 after B.22 added show. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet). 801 after gapfix124 added browser_watch. 802 after this lane (SongRecon surface) added songrecon. 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 807 after this lane (agents192) added fleet.
    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    assert len(registry._tools) == 807
    assert "scene_spawn" in registry._tools


# ---------------------------------------------------------------------------
# spawning + placement
# ---------------------------------------------------------------------------

def test_spawn_prop_at_explicit_position():
    scene = _scene()
    spawner = SceneSpawner(scene)
    state = spawner.spawn(SpawnRequest(
        kind="prop", name="crate",
        placement=Placement.at((1.0, 2.0, 0.0)),
        primitive="cube", params={"size": 1.5}))
    assert state.status == "spawned"
    assert state.position == (1.0, 2.0, 0.0)
    node = scene.node("crate")
    assert node.kind == "mesh"
    assert node.mesh.primitive == "cube"
    assert node.mesh.params == {"size": 1.5}
    assert node.transform.position == (1.0, 2.0, 0.0)
    assert not scene.is_dirty()  # spawn seals the scene


def test_spawn_auto_names_and_counts_budget():
    scene = _scene()
    spawner = SceneSpawner(scene, max_actors=2)
    first = spawner.spawn(SpawnRequest(kind="prop"))
    second = spawner.spawn(SpawnRequest(kind="light"))
    assert first.name == "spawn-prop-1"
    assert second.name == "spawn-light-2"
    assert spawner.budget == {"max_actors": 2, "alive": 2, "free": 0}


def test_placement_offset_from_anchor_node():
    scene = _scene()
    spawner = SceneSpawner(scene)
    state = spawner.spawn(SpawnRequest(
        kind="prop", name="satellite",
        placement=Placement.offset_from("anchor", (0.5, 0.0, 1.0))))
    assert state.position == (1.5, 2.0, 1.0)


def test_placement_offset_from_camera():
    scene = _scene()
    spawner = SceneSpawner(scene)
    state = spawner.spawn(SpawnRequest(
        kind="prop", name="near-cam",
        placement=Placement.offset_from("camera", (0.0, 0.0, -1.0))))
    assert state.position == (0.0, 0.0, 4.0)


def test_placement_in_front_of_anchor_on_the_arc():
    # The clone's TODO made real: on the camera -> anchor arc.
    scene = _scene()
    spawner = SceneSpawner(scene)
    state = spawner.spawn(SpawnRequest(
        kind="prop", name="hero",
        placement=Placement.in_front_of("anchor", distance=2.0)))
    cam = (0.0, 0.0, 5.0)
    anchor = (1.0, 2.0, 0.0)
    direction = tuple((a - c) / math.sqrt(sum((a - c) ** 2 for a, c in zip(anchor, cam, strict=True))) for a, c in zip(anchor, cam, strict=True))
    expected = tuple(a + d * 2.0 for a, d in zip(anchor, direction, strict=True))
    assert state.position == pytest.approx(expected)


def test_placement_random_around_is_bounded_and_seeded():
    scene = _scene()
    spawner = SceneSpawner(scene)
    bounds = {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 1.0]}
    first = spawner.spawn(SpawnRequest(
        kind="prop", name="a",
        placement=Placement.random_around(
            "anchor", bounds=bounds, seed=7)))
    second = spawner.spawn(SpawnRequest(
        kind="prop", name="b",
        placement=Placement.random_around(
            "anchor", bounds=bounds, seed=7)))
    assert first.position == second.position  # same seed, same scatter
    assert 0.0 <= first.position[0] <= 2.0
    assert 1.0 <= first.position[1] <= 3.0
    assert 0.0 <= first.position[2] <= 1.0


def test_placement_rotation_and_look_at():
    scene = _scene()
    spawner = SceneSpawner(scene)
    spawner.spawn(SpawnRequest(
        kind="prop", name="tilted",
        placement=Placement.at((0, 0, 0)),
        rotation=(0.0, 0.0, 45.0)))
    assert scene.node("tilted").transform.rotation.euler == (0.0, 0.0, 45.0)
    spawner.spawn(SpawnRequest(
        kind="prop", name="watcher",
        placement=Placement.at((0, 0, 0)),
        look_at=(1.0, 0.0, 0.0)))
    assert scene.node("watcher").transform.rotation.kind == "lookat"


# ---------------------------------------------------------------------------
# lifecycle + cleanup
# ---------------------------------------------------------------------------

def test_tick_activates_and_lifetime_culls():
    scene = _scene()
    spawner = SceneSpawner(scene)
    state = spawner.spawn(SpawnRequest(
        kind="prop", name="brief", lifetime_s=60.0), now=1000.0)
    assert state.status == "spawned"
    assert spawner.tick(now=1010.0) == []
    assert spawner.state("brief").status == "active"
    assert spawner.tick(now=1060.0) == ["brief"]
    assert spawner.state("brief").status == "despawned"
    assert spawner.state("brief").despawn_reason == "lifetime expired"
    with pytest.raises(Scene3DError, match="no scene node 'brief'"):
        scene.node("brief")


def test_despawn_removes_node_and_its_tracks_and_frees_the_slot():
    scene = _scene()
    scene.nodes.append(SceneNode(
        name="mover", transform=Transform(position=(0, 0, 0)),
        mesh=Mesh(kind="primitive", primitive="cube")))
    scene.tracks.append(AnimTrack(
        target="nodes/mover/position",
        keys=[AnimKey(frame=0, values=(0.0, 0.0, 0.0)),
              AnimKey(frame=30, values=(1.0, 0.0, 0.0))]))
    scene.seal()
    spawner = SceneSpawner.from_scene(scene)
    # The pre-existing node is not a spawned actor; spawn one, track
    # it, then despawn and prove the slot (name + tracks) is freed.
    spawner.spawn(SpawnRequest(kind="prop", name="guest"))
    scene.tracks.append(AnimTrack(
        target="nodes/guest/position",
        keys=[AnimKey(frame=0, values=(0.0, 0.0, 0.0))]))
    scene.seal()
    spawner.despawn("guest")
    assert [n.name for n in scene.nodes] == ["anchor", "mover"]
    assert [t.target for t in scene.tracks] == ["nodes/mover/position"]
    assert not scene.is_dirty()
    # The freed name spawns again.
    spawner.spawn(SpawnRequest(kind="prop", name="guest"))
    assert scene.node("guest").name == "guest"


def test_despawn_twice_is_refused_by_name():
    scene = _scene()
    spawner = SceneSpawner(scene)
    spawner.spawn(SpawnRequest(kind="prop", name="once"))
    spawner.despawn("once")
    with pytest.raises(SpawnError, match="already despawned"):
        spawner.despawn("once")


def test_sound_actor_despawns_when_its_clip_finishes():
    scene = _scene()
    spawner = SceneSpawner(scene)
    spawner.spawn(SpawnRequest(
        kind="sound", name="chime", sound_clip="chime.wav",
        destroy_after_play=True, duration_s=10.0), now=500.0)
    assert spawner.tick(now=509.0) == []
    assert spawner.tick(now=510.0) == ["chime"]
    assert spawner.state("chime").despawn_reason == "sound finished"


def test_sound_actor_without_destroy_after_play_survives_its_clip():
    scene = _scene()
    spawner = SceneSpawner(scene)
    spawner.spawn(SpawnRequest(
        kind="sound", name="loop", sound_clip="loop.wav",
        destroy_after_play=False, duration_s=10.0), now=500.0)
    assert spawner.tick(now=9999.0) == []
    assert spawner.state("loop").status == "active"


# ---------------------------------------------------------------------------
# budget + refusals
# ---------------------------------------------------------------------------

def test_budget_refusal_names_the_cap():
    scene = _scene()
    spawner = SceneSpawner(scene, max_actors=1)
    spawner.spawn(SpawnRequest(kind="prop", name="only"))
    with pytest.raises(SpawnBudgetError, match=r"cap 1"):
        spawner.spawn(SpawnRequest(kind="prop", name="second"))


def test_invalid_requests_are_refused_by_name():
    scene = _scene()
    spawner = SceneSpawner(scene)
    with pytest.raises(SpawnError, match="spawn kind must be one of"):
        SpawnRequest(kind="dragon")
    with pytest.raises(SpawnError, match="placement mode must be one of"):
        Placement(mode="teleport")
    with pytest.raises(SpawnError, match="anchor 'ghost'"):
        spawner.spawn(SpawnRequest(
            kind="prop",
            placement=Placement.offset_from("ghost", (0, 0, 0))))
    with pytest.raises(SpawnError, match="parent 'ghost'"):
        spawner.spawn(SpawnRequest(kind="prop", parent="ghost"))
    with pytest.raises(SpawnError, match="material 'gold'"):
        spawner.spawn(SpawnRequest(kind="prop", material="gold"))
    with pytest.raises(SpawnError, match="character spawn needs"):
        spawner.spawn(SpawnRequest(kind="character"))
    with pytest.raises(SpawnError, match="sound spawn needs"):
        spawner.spawn(SpawnRequest(kind="sound"))
    with pytest.raises(SpawnError, match="primitive must be one of"):
        SpawnRequest(kind="prop", primitive="torus")
    spawner.spawn(SpawnRequest(kind="prop", name="taken"))
    with pytest.raises(SpawnError, match="already has a live actor"):
        spawner.spawn(SpawnRequest(kind="prop", name="taken"))
    with pytest.raises(SpawnError, match="no spawned actor 'nobody'"):
        spawner.state("nobody")
    with pytest.raises(SpawnError, match="BVH file not found"):
        spawner.spawn(SpawnRequest(
            kind="character", bvh_path="/no/such/file.bvh"))


def test_in_front_of_without_a_camera_is_refused():
    scene = Scene3D(title="no-camera")
    spawner = SceneSpawner(scene)
    with pytest.raises(SpawnError, match="needs a scene camera"):
        spawner.spawn(SpawnRequest(
            kind="prop",
            placement=Placement.in_front_of("anchor")))


# ---------------------------------------------------------------------------
# character + file round-trip
# ---------------------------------------------------------------------------

def test_character_spawn_from_bvh_carries_the_skeleton(tmp_path):
    bvh = tmp_path / "take.bvh"
    bvh.write_text(MINIMAL_BVH, encoding="utf-8")
    scene = _scene()
    spawner = SceneSpawner(scene)
    state = spawner.spawn(SpawnRequest(
        kind="character", name="dancer", bvh_path=str(bvh)))
    node = scene.node("dancer")
    assert node.mesh.primitive == "cylinder"
    skeleton = node.extras["skeleton"]
    assert skeleton["root"]["name"] == "hips"
    assert state.kind == "character"


def test_character_spawn_from_inline_skeleton():
    scene = _scene()
    spawner = SceneSpawner(scene)
    skeleton = {"root": {"name": "hips", "offset": [0, 1, 0],
                         "channels": [], "children": []},
                "frame_time": 1 / 30}
    spawner.spawn(SpawnRequest(
        kind="character", name="puppet", skeleton=skeleton))
    assert scene.node("puppet").extras["skeleton"]["root"]["name"] == "hips"


def test_file_round_trip_rebuilds_the_spawner(tmp_path):
    path = _scene_file(tmp_path)
    result = spawn_into_file(path, SpawnRequest(
        kind="prop", name="crate",
        placement=Placement.at((1.0, 0.0, 0.0)), lifetime_s=120.0))
    assert result["actor"]["name"] == "crate"
    assert result["budget"] == {"max_actors": 32, "alive": 1, "free": 31}
    loaded = actors_in_file(path)
    assert [a["name"] for a in loaded["actors"]] == ["crate"]
    assert loaded["actors"][0]["status"] == "spawned"
    assert loaded["actors"][0]["lifetime_s"] == 120.0
    # A reloaded scene still validates and seals.
    scene = Scene3D.load(path)
    assert not scene.is_dirty()
    out = despawn_from_file(path, "crate")
    assert out["actor"]["status"] == "despawned"
    assert actors_in_file(path)["actors"] == []


def test_tick_file_culls_expired_and_saves(tmp_path):
    path = _scene_file(tmp_path)
    spawn_into_file(path, SpawnRequest(
        kind="prop", name="brief", lifetime_s=10.0), now=1000.0)
    result = tick_file(path, now=1011.0)
    assert result["culled"] == ["brief"]
    assert Scene3D.load(path).is_dirty() is False
    assert [n.name for n in Scene3D.load(path).nodes] == ["anchor"]


# ---------------------------------------------------------------------------
# CLI + tool surfaces
# ---------------------------------------------------------------------------

def test_cli_spawn_list_despawn(tmp_path):
    from nomorals.cmdline.dispatch import main

    path = _scene_file(tmp_path)
    assert main(["scene", "spawn", "--scene", path, "--kind", "prop",
                 "--name", "crate", "--at", "1,2,0",
                 "--primitive", "cube"]) == 0
    assert main(["scene", "list", "--scene", path]) == 0
    assert main(["scene", "status", "--scene", path,
                 "--name", "crate"]) == 0
    assert main(["scene", "despawn", "--scene", path,
                 "--name", "crate"]) == 0
    assert [n.name for n in Scene3D.load(path).nodes] == ["anchor"]


def test_cli_bad_spawn_fails(tmp_path):
    from nomorals.cmdline.dispatch import main

    path = _scene_file(tmp_path)
    assert main(["scene", "spawn", "--scene", path, "--kind", "prop",
                 "--name", "x", "--offset-from", "ghost",
                 "--offset", "0,0,0"]) == 1
    assert main(["scene", "spawn", "--scene", "/no/such/scene.json",
                 "--kind", "prop"]) == 1


def test_tool_spawn_list_status_despawn(tmp_path):
    registry = _registry(tmp_path)
    scene = tmp_path / "scenes" / "room.json"
    scene.parent.mkdir(parents=True)
    _scene().save(scene)
    rel = "scenes/room.json"
    out = _tool(registry, action="spawn", scene=rel, kind="prop", name="crate",
               placement={"mode": "at", "position": [1, 2, 0]})
    assert out["actor"]["name"] == "crate"
    assert out["actor"]["position"] == [1.0, 2.0, 0.0]
    listed = _tool(registry, action="list", scene=rel)
    assert [a["name"] for a in listed["actors"]] == ["crate"]
    status = _tool(registry, action="status", scene=rel, name="crate")
    assert status["actor"]["kind"] == "prop"
    gone = _tool(registry, action="despawn", scene=rel, name="crate")
    assert gone["actor"]["status"] == "despawned"
    assert _tool(registry, action="list", scene=rel)["actors"] == []


def test_tool_rejects_unknown_action_and_missing_scene(tmp_path):
    # Refusals surface as Err outcomes, not raises: the registry
    # catches the tool's error and hands it back as a value.
    registry = _registry(tmp_path)
    bad = _raw(registry, action="explode", scene="x.json")
    assert not bad.ok
    assert "unknown scene_spawn action" in _err(bad)
    missing = _raw(registry, action="list", scene="")
    assert not missing.ok
    assert "needs a scene" in _err(missing)
    unloadable = _raw(registry, action="list", scene="nope.json")
    assert not unloadable.ok
    assert "could not load scene" in _err(unloadable)


def test_tool_budget_refusal_names_the_cap(tmp_path):
    # The budget refusal travels through the tool unchanged: spawn
    # through the file API with a 1-actor cap, then the second spawn
    # refuses naming the cap.
    path = _scene_file(tmp_path)
    spawn_into_file(path, SpawnRequest(kind="prop", name="one"),
                    max_actors=1)
    with pytest.raises(SpawnBudgetError, match=r"cap 1"):
        spawn_into_file(path, SpawnRequest(kind="prop", name="two"),
                        max_actors=1)
    # And through the tool, despawning an unknown actor refuses
    # naming it.
    registry = _registry(tmp_path)
    ghost = _raw(registry, action="despawn", scene=str(path), name="ghost")
    assert not ghost.ok
    assert "no spawned actor 'ghost'" in _err(ghost)
