"""Lane B.27 — census §3B.27: Devon's own VTuber show
(storytold/animationtest pattern, pattern only, no upstream code).

The clone (storytold/AnimationTest) is a UE5 5.0 shell: README "UE5
VTuber show HyperJail", .uproject plugins LiveLinkMvnPlugin +
LiveLink + AppleARKitFaceSupport + AppleARKit, stock GameModeBase.
The pattern taken: a VTuber show is a live-driven character
performance on a stage, coordinated by a director, with body
capture (LiveLink MVN) + face capture (ARKit) as the live inputs.

Devon already has the inputs (rig B.18, show coordinator B.22, face
takes B.23, MVN B.24, project B.26) but no performance: no scripted
act, no character, no camera direction, no audience loop, and no
owned live-capture performance. This lane adds the organ
``nomorals/media/vtuber/``: characters (named expressions +
gestures over the rig), acts (timed beats: say / expression /
gesture / camera / take / face_take), cameras (shots as crops of
the rendered frame), an act player that performs the act through
the show's own phases, audience cues, and a live-MVN attach that
holds the recorded take while the character is driven live.

Proves: character + act round-trips and validation, camera crop
math, a full act performed end to end (beats fire in order,
expression blendshapes and gesture rotations land on the posed
frames, camera cuts change the frame size, a say beat drives a
non-silence viseme, an audience cue lands at the next tick, the
BVH export covers the performed poses), live MVN holding the take
and driving the rig from a real encoded datagram, the vtuber brain
tool, the `nm vtuber` CLI, and the /vtuber chat verb — with the
tool count pinned at 788 (779 after B.23 added facemcap; 780 after this lane (B.25) added facemesh; 781 after this lane adds vtuber; 782 after this lane (C.2) registers tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class); 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet). 801 after gapfix124 added browser_watch. 802 after this lane (SongRecon surface) added songrecon. 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.media.mocap import MocapStateStore
from nomorals.media.mocap.bvh import (
    Joint,
    Skeleton,
    SkeletonFrame,
)
from nomorals.media.rig import default_remap, default_rig
from nomorals.media.rig.session import RigSession
from nomorals.media.show import (
    Show,
    reset_shows,
)
from nomorals.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _clean_all():
    reset_shows()
    from nomorals.media.cloudcam.registry import (
        reset_sessions as reset_cloudcam,
    )
    from nomorals.media.rig.registry import (
        reset_sessions as reset_rig,
    )
    from nomorals.media.vtuber.registry import reset_all as reset_vt
    reset_cloudcam()
    reset_rig()
    reset_vt()
    yield
    reset_shows()
    reset_cloudcam()
    reset_rig()
    reset_vt()


def _rig() -> RigSession:
    return RigSession(default_rig(), default_remap("bvh"),
                      width=160, height=90, fps=15.0)


def _cloudcam(name: str = "cam"):
    from nomorals.media.cloudcam.registry import load_session
    return load_session(name, source="idle")


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


def _show(name: str = "demo", *, take: bool = False) -> Show:
    store = MocapStateStore()
    show = Show(name, store, rig=_rig(), cloudcam=_cloudcam())
    if take:
        show.load_take_frames(_take_skeleton(), _take_frames())
    return show


def _character_doc() -> dict:
    return {
        "name": "devon",
        "rig": "default",
        "expressions": {
            "smile": {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8,
                      "cheekSquintLeft": 0.4, "cheekSquintRight": 0.4},
            "laugh": {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0,
                      "jawOpen": 0.6, "eyeSquintLeft": 0.5,
                      "eyeSquintRight": 0.5},
        },
        "gestures": {
            "wave": {"shoulder_R": (0.0, 0.0, -120.0),
                     "upper_arm_R": (0.0, 0.0, -30.0),
                     "forearm_R": (0.0, 0.0, -20.0)},
            "bow": {"spine": (25.0, 0.0, 0.0),
                    "spine1": (15.0, 0.0, 0.0),
                    "neck": (-10.0, 0.0, 0.0)},
        },
    }


def _act_doc() -> dict:
    return {
        "name": "opener",
        "character": "devon",
        "fps": 15.0,
        "beats": [
            {"at": 0.0, "say": "Hello everyone",
             "expression": "smile"},
            {"at": 1.0, "gesture": "wave", "camera": "close"},
            {"at": 2.0, "expression": "laugh", "camera": "face"},
        ],
    }


def _write_json(tmp_path: Path, name: str, doc: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return path


def _tool(registry, **kwargs):
    outcome = registry.call("vtuber", **kwargs)
    if getattr(outcome, "ok", False):
        return outcome.value
    raise AssertionError(
        f"vtuber call failed: {getattr(outcome, 'error', outcome)}")


def _registry(tmp_path):
    registry = ToolRegistry(SimpleNamespace(home=str(tmp_path)))
    registry.register_builtins()
    return registry


# ---------------------------------------------------------------------------
# Tool count pin
# ---------------------------------------------------------------------------

def test_tool_count_is_786_after_vtuber():
    # 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet). 801 after gapfix124 added browser_watch. 802 after this lane (SongRecon surface) added songrecon. 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist.
    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    assert len(registry._tools) == 805


# ---------------------------------------------------------------------------
# Character
# ---------------------------------------------------------------------------

def test_character_builtins_load_and_validate(tmp_path):
    from nomorals.media.vtuber.character import (
        BUILTIN_EXPRESSIONS,
        BUILTIN_GESTURES,
        Character,
    )
    rig = default_rig()
    assert "smile" in BUILTIN_EXPRESSIONS
    assert "laugh" in BUILTIN_EXPRESSIONS
    assert "wave" in BUILTIN_GESTURES
    assert "bow" in BUILTIN_GESTURES
    char = Character.from_dict(_character_doc(), rig)
    assert char.name == "devon"
    assert char.expression("smile")["mouthSmileLeft"] == 0.8
    assert char.gesture("wave")["shoulder_R"] == (0.0, 0.0, -120.0)
    round_tripped = Character.from_dict(char.to_dict(), rig)
    assert round_tripped.to_dict() == char.to_dict()


def test_character_unknown_blendshape_refused_by_name():
    from nomorals.media.vtuber.character import Character
    from nomorals.media.vtuber.errors import VtuberError
    rig = default_rig()
    doc = _character_doc()
    doc["expressions"]["bogus"] = {"notARealMorph": 1.0}
    with pytest.raises(VtuberError) as outcome:
        Character.from_dict(doc, rig)
    assert "notARealMorph" in str(outcome.value)
    assert "blendshape" in str(outcome.value)


def test_character_unknown_bone_refused_by_name():
    from nomorals.media.vtuber.character import Character
    from nomorals.media.vtuber.errors import VtuberError
    rig = default_rig()
    doc = _character_doc()
    doc["gestures"]["bogus"] = {"notARealBone": (0.0, 0.0, 10.0)}
    with pytest.raises(VtuberError) as outcome:
        Character.from_dict(doc, rig)
    assert "notARealBone" in str(outcome.value)


def test_character_unknown_expression_and_gesture_refused():
    from nomorals.media.vtuber.character import Character
    from nomorals.media.vtuber.errors import VtuberError
    char = Character.from_dict(_character_doc(), default_rig())
    with pytest.raises(VtuberError):
        char.expression("nope")
    with pytest.raises(VtuberError):
        char.gesture("nope")


def test_character_registry(tmp_path):
    from nomorals.media.vtuber.errors import VtuberError
    from nomorals.media.vtuber.registry import (
        drop_character,
        get_character,
        list_characters,
        load_character,
    )
    path = _write_json(tmp_path, "devon.json", _character_doc())
    char = load_character("devon", path)
    assert char.name == "devon"
    assert get_character("devon") is char
    assert [c["name"] for c in list_characters()] == ["devon"]
    with pytest.raises(VtuberError):
        get_character("ghost")
    assert drop_character("devon")["ok"] is True
    assert list_characters() == []


# ---------------------------------------------------------------------------
# Act
# ---------------------------------------------------------------------------

def test_act_round_trip_and_beats_sorted(tmp_path):
    from nomorals.media.vtuber.act import Act
    doc = _act_doc()
    doc["beats"] = list(reversed(doc["beats"]))
    act = Act.from_dict(doc)
    assert [b.at for b in act.beats] == [0.0, 1.0, 2.0]
    assert act.beats[0].say == "Hello everyone"
    assert act.beats[1].gesture == "wave"
    assert act.beats[1].camera == "close"
    assert act.beats[2].expression == "laugh"
    assert Act.from_dict(act.to_dict()).to_dict() == act.to_dict()
    path = _write_json(tmp_path, "opener.vtact.json", _act_doc())
    loaded = Act.from_file(path)
    assert loaded.name == "opener"
    assert len(loaded.beats) == 3


def test_act_negative_time_and_bad_beat_refused():
    from nomorals.media.vtuber.act import Act
    from nomorals.media.vtuber.errors import VtuberError
    doc = _act_doc()
    doc["beats"] = [{"at": -1.0, "say": "too early"}]
    with pytest.raises(VtuberError):
        Act.from_dict(doc)
    doc["beats"] = [{"at": 0.0}]
    with pytest.raises(VtuberError):
        Act.from_dict(doc)


def test_act_unknown_names_refused_at_attach():
    from nomorals.media.vtuber.act import Act
    from nomorals.media.vtuber.character import Character
    from nomorals.media.vtuber.errors import VtuberError
    from nomorals.media.vtuber.player import ActPlayer
    char = Character.from_dict(_character_doc(), default_rig())
    doc = _act_doc()
    doc["beats"] = [{"at": 0.0, "expression": "nope"}]
    act = Act.from_dict(doc)
    with pytest.raises(VtuberError):
        ActPlayer(act, char, default_rig())
    doc["beats"] = [{"at": 0.0, "camera": "nope"}]
    act = Act.from_dict(doc)
    with pytest.raises(VtuberError):
        ActPlayer(act, char, default_rig())


def test_act_registry(tmp_path):
    from nomorals.media.vtuber.errors import VtuberError
    from nomorals.media.vtuber.registry import (
        drop_act,
        get_act,
        list_acts,
        load_act,
    )
    path = _write_json(tmp_path, "opener.vtact.json", _act_doc())
    act = load_act("opener", path)
    assert act.name == "opener"
    assert get_act("opener") is act
    assert [a["name"] for a in list_acts()] == ["opener"]
    with pytest.raises(VtuberError):
        get_act("ghost")
    assert drop_act("opener")["ok"] is True
    assert list_acts() == []


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def test_camera_shots_and_crop_math():
    from nomorals.media.vtuber.camera import SHOTS, apply_shot
    from nomorals.media.vtuber.errors import VtuberError
    assert set(SHOTS) == {"wide", "medium", "close", "face"}
    w, h = 160, 90
    frame = bytes([7] * (w * h * 3))
    rgb, cw, ch = apply_shot(frame, w, h, "wide")
    assert (cw, ch) == (w, h)
    assert rgb == frame
    prev = (w, h)
    for shot in ("medium", "close", "face"):
        rgb, cw, ch = apply_shot(frame, w, h, shot)
        assert len(rgb) == cw * ch * 3
        assert (cw, ch) < prev
        prev = (cw, ch)
    with pytest.raises(VtuberError):
        apply_shot(frame, w, h, "nope")


# ---------------------------------------------------------------------------
# End-to-end performance
# ---------------------------------------------------------------------------

def _perform(tmp_path, *, act_doc=None):
    from nomorals.media.vtuber.act import Act
    from nomorals.media.vtuber.character import Character
    from nomorals.media.vtuber.plugin import attach_performance
    show = _show("demo")
    char = Character.from_dict(_character_doc(), default_rig())
    act = Act.from_dict(act_doc or _act_doc())
    player = attach_performance(show, act, char)
    return show, player


def test_act_performs_end_to_end(tmp_path):
    show, player = _perform(tmp_path)
    started = show.start()
    assert started["state"] == "live"
    # 3.5s of ticks at 15fps covers beats at 0s, 1s, 2s.
    show.run(52)
    state = player.state()
    assert state["beats_fired"] == 3
    assert state["beats_total"] == 3
    assert state["current_shot"] == "face"
    kinds = [e["kind"] for e in show.events()
             if e["kind"].startswith("vtuber_beat")]
    assert kinds == ["vtuber_beat_say", "vtuber_beat_gesture",
                     "vtuber_beat_expression"]
    # The smile beat's blendshapes landed on a posed frame.
    posed = show._posed
    assert posed, "no posed frames buffered"
    smiled = [p for p in posed
              if p.blendshapes.get("mouthSmileLeft", 0.0) >= 0.8]
    assert smiled, "no posed frame carries the smile"
    # The wave beat's rotations landed on a posed frame.
    waved = [p for p in posed
             if p.rotations.get("shoulder_R", (0.0, 0.0, 0.0))[2]
             <= -50.0]
    assert waved, "no posed frame carries the wave"
    # The say beat drove a non-silence viseme.
    spoken = [p for p in posed if p.viseme != "sil"]
    assert spoken, "no posed frame carries a spoken viseme"
    # The face shot cropped the last rendered frame.
    assert show.last_frame is not None
    assert len(show.last_frame) < 160 * 90 * 3
    assert show._last_shot == "face"


def test_audience_cue_lands_at_the_next_tick(tmp_path):
    show, player = _perform(tmp_path)
    show.start()
    show.run(5)
    player.cue("expression", "laugh")
    show.run(3)
    state = player.state()
    assert state["cues_applied"] == 1
    laughed = [p for p in show._posed
               if p.blendshapes.get("jawOpen", 0.0) >= 0.5]
    assert laughed, "the cued laugh never landed on a pose"
    player.cue("say", "thanks chat")
    show.run(3)
    assert player.state()["cues_applied"] == 2
    player.cue("camera", "close")
    show.run(1)
    assert player.state()["current_shot"] == "close"
    from nomorals.media.vtuber.errors import VtuberError
    with pytest.raises(VtuberError):
        player.cue("expression", "nope")
    with pytest.raises(VtuberError):
        player.cue("dance", "wave")


def test_performance_export_covers_the_act(tmp_path):
    show, player = _perform(tmp_path)
    show.start()
    show.run(52)
    out = tmp_path / "opener.bvh"
    result = show.export(out)
    assert result["frames"] >= 50
    assert out.exists()
    assert result["rig"] == "devon-default"


def test_performance_detach_stops_the_act(tmp_path):
    from nomorals.media.vtuber.plugin import detach_performance
    show, player = _perform(tmp_path)
    show.start()
    show.run(5)
    assert detach_performance(show)["ok"] is True
    assert player.state()["attached"] is False
    show.run(5)
    assert player.state()["beats_fired"] == 1


# ---------------------------------------------------------------------------
# Live MVN attach
# ---------------------------------------------------------------------------

def _mvn_datagram(segment_degrees: float = 0.0) -> bytes:
    from nomorals.media.mocap.mvn import (
        MVN_NAMES,
        MvnHeader,
        MvnSegmentItem,
        encode_datagram,
    )
    items = tuple(
        MvnSegmentItem(
            segment_id=i,
            position=(0.0, 0.0, 0.0),
            quaternion=(1.0, 0.0, 0.0, 0.0))
        for i in range(len(MVN_NAMES)))
    header = MvnHeader(
        msg_type="02", sample_counter=1, datagram_counter=0,
        last=True, item_count=len(items), time_code_ms=8,
        avatar_id=0)
    return encode_datagram(header, items)


def test_live_attach_holds_the_take_and_drives_live(tmp_path):
    from nomorals.media.mocap.mvn import MvnReceiver
    from nomorals.media.vtuber.plugin import (
        attach_live,
        detach_live,
    )
    show = _show("demo", take=True)
    show.start()
    assert show.take.index == 1  # start's live tick played frame 0
    receiver = MvnReceiver(show.store, legacy=True)
    attach_live(show, receiver, "mvn-1")
    show.run(5)
    assert show.take.index == 1, "the take stepped under live"
    assert show._hold_take is True
    # A real encoded datagram through the receiver drives the rig.
    receipt = receiver.feed(_mvn_datagram())
    assert receipt["counts"]["frames"] == 1
    show.run(2)
    assert show._last_pose is not None
    assert show._last_pose["rotations"], "live frame drove nothing"
    assert detach_live(show)["ok"] is True
    assert show._hold_take is False
    show.run(2)
    assert show.take.index > 1, "the take did not resume"


def test_live_attach_refuses_an_unknown_receiver(tmp_path):
    from nomorals.media.vtuber.errors import VtuberError
    from nomorals.media.vtuber.plugin import attach_live
    show = _show("demo")
    with pytest.raises(VtuberError):
        attach_live(show, None, "ghost")


# ---------------------------------------------------------------------------
# Brain tool
# ---------------------------------------------------------------------------

def test_tool_character_act_perform_cue_status_stop(tmp_path):
    registry = _registry(tmp_path)
    char_path = _write_json(tmp_path, "devon.json", _character_doc())
    act_path = _write_json(tmp_path, "opener.vtact.json", _act_doc())
    loaded = _tool(registry, action="character", sources=[str(char_path)],
                   node="devon")
    assert loaded["name"] == "devon"
    assert "smile" in loaded["expressions"]
    acts = _tool(registry, action="act", sources=[str(act_path)],
                 node="opener")
    assert acts["name"] == "opener"
    assert acts["beats"] == 3
    assert _tool(registry, action="character")["characters"] == ["devon"]
    assert _tool(registry, action="act")["acts"] == ["opener"]
    performed = _tool(registry, action="perform", node="demo",
                      params={"act": "opener", "character": "devon"})
    assert performed["state"] == "live"
    assert performed["act"] == "opener"
    assert performed["character"] == "devon"
    ticked = _tool(registry, action="tick", node="demo")
    assert ticked["state"] == "live"
    cued = _tool(registry, action="cue", node="demo",
                 params={"kind": "expression", "value": "laugh"})
    assert cued["cues_applied"] == 1
    status = _tool(registry, action="status", node="demo")
    assert status["act"] == "opener"
    assert status["beats_total"] == 3
    stopped = _tool(registry, action="stop", node="demo")
    assert stopped["ok"] is True
    assert stopped["state"] == "done"
    dropped = _tool(registry, action="drop", node="demo")
    assert dropped["ok"] is True
    with pytest.raises(AssertionError):
        _tool(registry, action="perform", node="demo",
              params={"act": "ghost", "character": "devon"})


def test_tool_unknown_action_refused():
    registry = _registry(Path("/tmp"))
    outcome = registry.call("vtuber", action="dance")
    assert not getattr(outcome, "ok", False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_vtuber_commands(tmp_path):
    import argparse

    from nomorals.cmdline.commands.vtuber import _cmd_vtuber
    char_path = _write_json(tmp_path, "devon.json", _character_doc())
    act_path = _write_json(tmp_path, "opener.vtact.json", _act_doc())

    def args(**kwargs):
        base = {"vtuber_action": "status", "session": "demo",
                "json": True, "character": "", "act": "",
                "path": "", "kind": "", "value": "",
                "duration": 0.0, "receiver": "", "ticks": 30,
                "interval": 0.0}
        base.update(kwargs)
        return argparse.Namespace(**base)

    assert _cmd_vtuber(args(vtuber_action="character",
                            path=str(char_path)), None) == 0
    assert _cmd_vtuber(args(vtuber_action="act",
                            path=str(act_path)), None) == 0
    assert _cmd_vtuber(args(vtuber_action="perform", act="opener",
                            character="devon"), None) == 0
    assert _cmd_vtuber(args(vtuber_action="cue", kind="expression",
                            value="laugh"), None) == 0
    assert _cmd_vtuber(args(vtuber_action="status"), None) == 0
    assert _cmd_vtuber(args(vtuber_action="stop"), None) == 0
    assert _cmd_vtuber(args(vtuber_action="drop"), None) == 0
    assert _cmd_vtuber(args(vtuber_action="bogus"), None) == 2


def test_cli_vtuber_registered_in_parser_and_dispatch():
    from nomorals.cmdline.dispatch import _COMMANDS
    from nomorals.cmdline.parser import CLI_ALIASES
    assert "vtuber" in CLI_ALIASES
    assert "vtuber" in _COMMANDS


# ---------------------------------------------------------------------------
# Chat verb
# ---------------------------------------------------------------------------

def test_chat_control_vtuber(tmp_path):
    from nomorals.agents.partner.runtime_mocap import RuntimeMocapMixin
    char_path = _write_json(tmp_path, "devon.json", _character_doc())
    act_path = _write_json(tmp_path, "opener.vtact.json", _act_doc())
    mixin = RuntimeMocapMixin()
    help_text = mixin._control_vtuber("", "test")
    assert "/vtuber perform" in help_text
    loaded = mixin._control_vtuber(f"character {char_path}", "test")
    assert "devon" in loaded
    loaded = mixin._control_vtuber(f"act {act_path}", "test")
    assert "opener" in loaded
    assert "characters:" in mixin._control_vtuber("list", "test")
    started = mixin._control_vtuber("perform demo opener devon", "test")
    assert "live" in started
    ticked = mixin._control_vtuber("tick demo", "test")
    assert "tick" in ticked
    cued = mixin._control_vtuber("cue demo expression laugh", "test")
    assert "cue" in cued
    status = mixin._control_vtuber("status demo", "test")
    assert '"act": "opener"' in status
    assert "stopped" in mixin._control_vtuber("stop demo", "test")
    assert "dropped" in mixin._control_vtuber("drop demo", "test")
    assert "/vtuber perform" in mixin._control_vtuber("bogus", "test")


def test_chat_vtuber_registered_in_the_catalog():
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
        help_text,
        list_catalog,
    )
    assert "vtuber" in CONTROL_COMMANDS
    assert CONTROL_COMMANDS["vtuber"] == (0, None)
    details = COMMAND_DETAILS["vtuber"]
    assert details["what"] and details["usage"] and details["example"]
    assert "/vtuber" in details["usage"]
    assert "vtuber" in LIST_ONELINERS
    assert "/vtuber" in LIST_ONELINERS["vtuber"]
    slots = [group for group in LIST_GROUPS
             if "vtuber" in str(group)]
    assert slots, "vtuber missing from the catalog groups"
    assert "vtuber" in help_text()
    assert "vtuber" in list_catalog()
