"""Lane B.23 — census §3B.23: ARKit LiveLink CSV facial capture,
as face takes (storytold/bevy-mocap patterns, pattern only, no
upstream code).

The clone (storytold/bevy-mocap, bevy 0.12) is an asset loader
converting Unreal LiveLink CSV recordings into Bevy
``AnimationClip``s: a SMPTE timecode, a blendshape count, one
float column per ARKit morph; a mapping of ``[mapping,
csv_name]`` pairs — a morph name (reorder to mesh order,
missing → 0.0) or a bone object ``{path, axis, offset?,
scale?}`` with angle ``value * 180° * scale + offset``,
composed per bone as sequential axis rotations (quaternions,
mapping order); a clip-info ``.morph_anim.json`` of ``{csv,
root_name, fps?, mapping?}``.

Devon's organ is the same pipeline in Python: the CSV parse
(:mod:`nomorals.media.facemcap.csv`), the mapping
(:mod:`nomorals.media.facemcap.mapping`), the take and its
player (:mod:`nomorals.media.facemcap.take`), the manifest
loader (:mod:`nomorals.media.facemcap.loader`), the registry
(:mod:`nomorals.media.facemcap.registry`) — wired into the
show's capture/drive phases as a face plugin, the ``facemcap``
brain tool, ``nm facemcap``, and the ``/facemcap`` chat verb.

Proves: the CSV parse (header, skipped count column, ragged
rows refused, fps deduced, SMPTE segments, the >255 refusal
the clone wraps silently), the mapping (morph rename,
missing → 0.0, bone offset/scale, the clone's own bone case
reproduced exactly, the Euler round-trip), the manifest load
(relative paths, fps override, missing-file refusals), the
take (player steps, skeleton-frame shape, JSON round-trip),
the drive through a rig session (blendshapes land, the head
lands — the rig-named bone bypasses the source remap, the
defect the first drive had), the show seam (face-only show,
overlay, unknown bone recorded once), and the wiring (tool
count pinned at 787 — 777 after B.22 added show; 778 after
B.26 added scene_project; 779 after this lane (B.23) added
facemcap; 780 after this lane (B.25) added facemesh; 781 after this lane (B.27) added vtuber; 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 804 after gapfix129 added group_cleanup — plus the
CLI and the chat verb).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.media.facemcap import (
    FaceMocapError,
    default_arkit_mapping,
    default_mapping_path,
    drive_take,
    drop_take,
    get_take,
    list_takes,
    load_csv,
    load_manifest,
    load_take,
    load_take_json,
    render_frame,
    reset_takes,
)
from nomorals.media.facemcap.csv import CSVAnimation, Timestamp
from nomorals.media.facemcap.mapping import (
    FaceMapping,
    _euler_zxy_to_quat,
    _quat_to_euler_zxy,
)
from nomorals.media.facemcap.take import FaceTake
from nomorals.media.mocap import MocapStateStore
from nomorals.media.rig import default_remap, default_rig
from nomorals.media.rig.session import RigSession
from nomorals.media.show import Show, reset_shows
from nomorals.tools.registry import ToolRegistry

_CSV = """timecode, blendshapecount, EyeBlinkLeft, JawOpen, HeadZ, HeadX
00:00:00:00, 4, 0.5, 0.4, -0.5, 0.5
00:00:00:01, 4, 1.0, 0.8, -0.7, 0.8
00:00:00:02, 4, 0.0, 0.0, 0.0, 0.0
"""

_MAPPING = [
    ["eyeBlinkLeft", "EyeBlinkLeft"],
    ["jawOpen", "JawOpen"],
    [{"path": ["head"], "axis": "Z"}, "HeadZ"],
    [{"path": ["head"], "axis": "X"}, "HeadX"],
]


@pytest.fixture(autouse=True)
def _clean_takes():
    reset_takes()
    reset_shows()
    from nomorals.media.rig.registry import (
        reset_sessions as reset_rig,
    )
    reset_rig()
    yield
    reset_takes()
    reset_shows()
    reset_rig()


def _write_fixtures(tmp_path: Path) -> Path:
    """The smile recording: CSV + mapping + manifest, in one
    dir, the manifest pointing at its neighbours by relative
    path."""
    (tmp_path / "smile.csv").write_text(_CSV, encoding="utf-8")
    (tmp_path / "smile-mapping.json").write_text(
        json.dumps(_MAPPING), encoding="utf-8")
    manifest = tmp_path / "smile.morph_anim.json"
    manifest.write_text(json.dumps({
        "csv": "smile.csv", "root_name": "smile",
        "mapping": "smile-mapping.json"}), encoding="utf-8")
    return manifest


def _rig() -> RigSession:
    return RigSession(default_rig(), default_remap("bvh"),
                      width=160, height=90, fps=15.0)


# ---------------------------------------------------------------------------
# CSV parse
# ---------------------------------------------------------------------------

def test_csv_parses_header_rows_and_deduced_fps():
    anim = CSVAnimation.parse(_CSV)
    assert anim.morph_names == [
        "EyeBlinkLeft", "JawOpen", "HeadZ", "HeadX"]
    assert anim.fps == 3.0
    assert len(anim.frames) == 3
    assert anim.frames[0].morphs == (0.5, 0.4, -0.5, 0.5)
    assert anim.frames[1].morphs == (1.0, 0.8, -0.7, 0.8)
    assert anim.frames[2].morphs == (0.0, 0.0, 0.0, 0.0)
    assert anim.frames[0].timestamp == Timestamp(
        hour=0, minute=0, second=0, frame=0)
    assert anim.frames[1].timestamp.frame == 1
    assert anim.duration_s == pytest.approx(2 / 3)


def test_csv_skips_blendshape_count_column():
    # The count column says 99; the parse ignores it — the
    # header's morph columns are what count.
    text = _CSV.replace("00:00:00:00, 4,", "00:00:00:00, 99,")
    anim = CSVAnimation.parse(text)
    assert len(anim.frames[0].morphs) == 4


def test_csv_timestamp_segments():
    assert Timestamp.parse("00:00:00:04") == Timestamp(
        hour=0, minute=0, second=0, frame=4)
    assert Timestamp.parse("01:04") == Timestamp(
        hour=0, minute=0, second=1, frame=4)
    assert Timestamp.parse("02:03:04") == Timestamp(
        hour=0, minute=2, second=3, frame=4)
    assert Timestamp.parse("00:00:00:04.9").frame == 4
    assert Timestamp.parse("00:00:01:00").to_seconds(30) == 1.0
    assert Timestamp.parse("00:00:00:15").to_seconds(30) == 0.5


@pytest.mark.parametrize("bad", [
    "", "00:00:00:00:00", "00:00:00:xx", "00:00:00:256",
    "00:00:00:-1",
])
def test_csv_timestamp_refusals(bad):
    with pytest.raises(FaceMocapError) as exc:
        Timestamp.parse(bad)
    assert exc.value.kind == "bad_timecode"


def test_csv_ragged_row_refused():
    text = _CSV + "00:00:00:03, 4, 0.1, 0.2\n"
    with pytest.raises(FaceMocapError) as exc:
        CSVAnimation.parse(text)
    assert exc.value.kind == "invalid_csv"


def test_csv_bad_morph_refused():
    text = _CSV.replace("0.5, 0.4,", "abc, 0.4,", 1)
    with pytest.raises(FaceMocapError) as exc:
        CSVAnimation.parse(text)
    assert exc.value.kind == "bad_morph"


@pytest.mark.parametrize("text", ["", "timecode, blendshapecount\n"])
def test_csv_empty_refused(text):
    with pytest.raises(FaceMocapError):
        CSVAnimation.parse(text)


def test_csv_header_with_no_morphs_refused():
    with pytest.raises(FaceMocapError) as exc:
        CSVAnimation.parse("timecode, blendshapecount\n"
                           "00:00:00:00, 0\n")
    assert exc.value.kind == "no_morphs"


def test_csv_no_frames_refused():
    with pytest.raises(FaceMocapError) as exc:
        CSVAnimation.parse(
            "timecode, blendshapecount, EyeBlinkLeft\n")
    assert exc.value.kind == "no_frames"


# ---------------------------------------------------------------------------
# Mapping — the math
# ---------------------------------------------------------------------------

def test_mapping_morph_rename_and_missing_defaults_zero():
    mapping = FaceMapping.from_json([
        ["eyeBlinkLeft", "EyeBlinkLeft"],
        ["jawOpen", "MissingColumn"],
    ])
    out = mapping.apply(CSVAnimation.parse(_CSV))
    assert out["blendshapes"][0] == {
        "eyeBlinkLeft": 0.5, "jawOpen": 0.0}
    assert out["blendshapes"][1] == {
        "eyeBlinkLeft": 1.0, "jawOpen": 0.0}
    assert out["unmapped"] == ["JawOpen", "HeadZ", "HeadX"]


def test_mapping_bone_clone_case():
    # The clone's own bone test case: Z −0.5 → −90°, then X
    # 0.5 → 90° — the composed rotation is exactly
    # Euler (−90, 0, 90), and that Euler converts back to the
    # clone's asserted quaternion (0.5, 0.5, −0.5) xyz.
    mapping = FaceMapping.from_json(_MAPPING)
    out = mapping.apply(CSVAnimation.parse(_CSV))
    z, x, y = out["bones"][0]["head"]
    assert z == pytest.approx(-90.0, abs=1e-9)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(90.0, abs=1e-9)
    quat = _euler_zxy_to_quat((z, x, y))
    # _Quat is (w, x, y, z): the clone's asserted xyz
    # (0.5, 0.5, -0.5) are indices 1, 2, 3, with w == 0.5.
    assert quat[0] == pytest.approx(0.5, abs=1e-9)
    assert quat[1] == pytest.approx(0.5, abs=1e-9)
    assert quat[2] == pytest.approx(0.5, abs=1e-9)
    assert quat[3] == pytest.approx(-0.5, abs=1e-9)


def test_mapping_bone_offset_and_scale():
    # angle = value * 180° * scale + offset: 0.25 * 180 * 2 + 10
    mapping = FaceMapping.from_json([
        [{"path": ["head"], "axis": "Z", "offset": 10.0,
          "scale": 2.0}, "HeadZ"],
    ])
    out = mapping.apply(CSVAnimation.parse(
        "timecode, blendshapecount, HeadZ\n"
        "00:00:00:00, 1, 0.25\n"))
    z, x, y = out["bones"][0]["head"]
    assert z == pytest.approx(100.0, abs=1e-9)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("angles", [
    (0.0, 0.0, 0.0), (30.0, 0.0, 0.0), (0.0, 45.0, 0.0),
    (0.0, 0.0, -20.0), (10.0, -25.0, 40.0),
    (-90.0, 0.0, 90.0), (170.0, 80.0, -170.0),
    (5.0, 89.0, 5.0), (-45.0, -89.0, 45.0),
])
def test_euler_round_trip(angles):
    back = _quat_to_euler_zxy(_euler_zxy_to_quat(angles))
    for got, want in zip(back, angles, strict=True):
        assert got == pytest.approx(want, abs=1e-9)


def test_euler_principal_axes_exact():
    for angles in [(30.0, 0.0, 0.0), (0.0, 45.0, 0.0),
                   (0.0, 0.0, -20.0)]:
        back = _quat_to_euler_zxy(_euler_zxy_to_quat(angles))
        assert back == pytest.approx(angles, abs=1e-12)


def test_mapping_unknown_blendshape_refused_when_rig_given():
    mapping = FaceMapping.from_json([["nope", "EyeBlinkLeft"]])
    with pytest.raises(FaceMocapError) as exc:
        mapping.apply(CSVAnimation.parse(_CSV),
                      rig_blendshapes=("eyeBlinkLeft",))
    assert exc.value.kind == "bad_mapping"


def test_mapping_malformed_entries_refused():
    for bad in [[["only-one"]], [[123, "EyeBlinkLeft"]],
                [[{"path": [], "axis": "Z"}, "HeadZ"]],
                [[{"path": ["head"]}, "HeadZ"]],
                [[{"path": ["head"], "axis": "Q"}, "HeadZ"]]]:
        with pytest.raises(FaceMocapError) as exc:
            FaceMapping.from_json(bad)
        assert exc.value.kind == "bad_mapping"


def test_default_arkit_mapping_has_52():
    assert default_mapping_path().exists()
    mapping = default_arkit_mapping()
    assert len(mapping.entries) == 52
    renames = {csv: morph for morph, csv in
               [(m.blendshape, c) for m, c in mapping.entries]}
    assert renames["EyeBlinkLeft"] == "eyeBlinkLeft"
    assert renames["JawOpen"] == "jawOpen"
    assert renames["MouthSmileLeft"] == "mouthSmileLeft"


# ---------------------------------------------------------------------------
# Loader + registry
# ---------------------------------------------------------------------------

def test_load_csv_direct_identity(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text(
        "timecode, blendshapecount, eyeBlinkLeft, jawOpen\n"
        "00:00:00:00, 2, 0.5, 0.4\n"
        "00:00:00:01, 2, 1.0, 0.8\n", encoding="utf-8")
    take = load_csv(path, name="raw")
    assert take.name == "raw"
    assert take.root_name == "raw"
    assert take.fps == 2.0
    assert take.total == 2
    assert take.frames[0].blendshapes == {
        "eyeBlinkLeft": 0.5, "jawOpen": 0.4}
    assert take.frames[0].bones == {}


def test_load_csv_arkit_mapping(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text(
        "timecode, blendshapecount, EyeBlinkLeft\n"
        "00:00:00:00, 1, 0.7\n", encoding="utf-8")
    take = load_csv(path, name="raw", mapping="arkit")
    assert take.frames[0].blendshapes["eyeBlinkLeft"] == 0.7
    assert len(take.frames[0].blendshapes) == 52


def test_load_manifest_with_mapping_and_fps(tmp_path):
    manifest = _write_fixtures(tmp_path)
    take = load_manifest(manifest, name="smile")
    assert take.name == "smile"
    assert take.root_name == "smile"
    assert take.fps == 3.0
    assert take.total == 3
    assert take.frames[0].blendshapes == {
        "eyeBlinkLeft": 0.5, "jawOpen": 0.4}
    assert take.frames[0].bones["head"][0] == pytest.approx(
        -90.0, abs=1e-9)


def test_load_manifest_fps_override(tmp_path):
    manifest = _write_fixtures(tmp_path)
    blob = json.loads(manifest.read_text(encoding="utf-8"))
    blob["fps"] = 24.0
    manifest.write_text(json.dumps(blob), encoding="utf-8")
    take = load_manifest(manifest, name="smile")
    assert take.fps == 24.0
    assert take.duration_s == pytest.approx(2 / 24)


def test_load_manifest_relative_paths_from_any_cwd(tmp_path,
                                                  monkeypatch):
    # The CSV and mapping resolve against the manifest's dir,
    # not the process cwd.
    manifest = _write_fixtures(tmp_path)
    monkeypatch.chdir("/")
    take = load_manifest(manifest, name="smile")
    assert take.total == 3


@pytest.mark.parametrize("blob", [
    {"root_name": "smile"},
    {"csv": "smile.csv"},
    {"csv": "smile.csv", "root_name": "smile", "fps": "fast"},
    {"csv": "smile.csv", "root_name": "smile",
     "mapping": "ghost.json"},
    {"csv": "ghost.csv", "root_name": "smile"},
])
def test_load_manifest_refusals(tmp_path, blob):
    _write_fixtures(tmp_path)
    manifest = tmp_path / "bad.morph_anim.json"
    manifest.write_text(json.dumps(blob), encoding="utf-8")
    with pytest.raises(FaceMocapError):
        load_manifest(manifest, name="bad")


def test_load_manifest_missing_file_refused(tmp_path):
    with pytest.raises(FaceMocapError) as exc:
        load_manifest(tmp_path / "ghost.morph_anim.json")
    assert exc.value.kind == "missing_manifest"


def test_load_csv_missing_file_refused(tmp_path):
    with pytest.raises(FaceMocapError) as exc:
        load_csv(tmp_path / "ghost.csv")
    assert exc.value.kind == "missing_csv"


def test_registry_load_list_get_drop(tmp_path):
    manifest = _write_fixtures(tmp_path)
    take = load_take("smile", manifest)
    assert take.name == "smile"
    assert get_take("smile") is take
    entries = list_takes()
    assert [e["name"] for e in entries] == ["smile"]
    assert entries[0]["frames"] == 3
    dropped = drop_take("smile")
    assert dropped["frames"] == 3
    assert list_takes() == []
    with pytest.raises(FaceMocapError) as exc:
        get_take("smile")
    assert exc.value.kind == "unknown_take"


def test_registry_rejects_unknown_suffix(tmp_path):
    path = tmp_path / "take.txt"
    path.write_text("nothing", encoding="utf-8")
    with pytest.raises(FaceMocapError) as exc:
        load_take("bad", path)
    assert exc.value.kind == "bad_take"


def test_registry_json_dispatch_neither_shape(tmp_path):
    path = tmp_path / "neither.json"
    path.write_text(json.dumps({"hello": "world"}),
                    encoding="utf-8")
    with pytest.raises(FaceMocapError) as exc:
        load_take("bad", path)
    assert exc.value.kind == "bad_take"


# ---------------------------------------------------------------------------
# Take: player, frames, JSON round-trip
# ---------------------------------------------------------------------------

def test_take_player_steps_and_payload(tmp_path):
    take = load_manifest(_write_fixtures(tmp_path), name="smile")
    player = take.player()
    assert player.total == 3
    assert not player.exhausted
    frame = player.next()
    assert frame is not None
    payload = player.payload(frame)
    assert payload["blendshapes"] == {
        "eyeBlinkLeft": 0.5, "jawOpen": 0.4}
    assert payload["bones"]["head"][0] == pytest.approx(
        -90.0, abs=1e-9)
    assert payload["frame"] == 0
    assert payload["time_s"] == 0.0
    assert player.index == 1
    player.next()
    player.next()
    assert player.exhausted
    assert player.next() is None
    player.reset()
    assert player.index == 0
    assert not player.exhausted


def test_take_to_skeleton_frames_and_face_inputs(tmp_path):
    take = load_manifest(_write_fixtures(tmp_path), name="smile")
    skeletons = take.to_skeleton_frames()
    assert len(skeletons) == 3
    assert skeletons[0].values["head"][0] == pytest.approx(
        -90.0, abs=1e-9)
    faces = take.to_face_inputs()
    assert len(faces) == 3
    assert faces[0].blendshapes == {
        "eyeBlinkLeft": 0.5, "jawOpen": 0.4}
    assert faces[1].blendshapes == {
        "eyeBlinkLeft": 1.0, "jawOpen": 0.8}


def test_take_json_round_trip(tmp_path):
    take = load_manifest(_write_fixtures(tmp_path), name="smile")
    path = tmp_path / "smile.json"
    path.write_text(json.dumps(take.to_json()), encoding="utf-8")
    back = load_take_json(path, name="smile-again")
    assert back.name == "smile-again"
    assert back.root_name == take.root_name
    assert back.fps == take.fps
    assert back.total == take.total
    assert back.frames[0].blendshapes == \
        take.frames[0].blendshapes
    assert back.frames[0].bones["head"] == pytest.approx(
        take.frames[0].bones["head"], abs=1e-12)
    # And through the registry: a .json with "frames" loads as
    # a face take, not a manifest.
    again = load_take("smile-copy", path)
    assert again.total == 3
    assert again.frames[1].blendshapes["jawOpen"] == 0.8


@pytest.mark.parametrize("blob", [
    {"format": "face-take-v0", "fps": 3.0, "morphs": [],
     "frames": [{"time_s": 0.0, "blendshapes": {},
                 "bones": {}}]},
    {"format": "face-take-v1", "fps": 3.0, "morphs": [],
     "frames": []},
    {"format": "face-take-v1", "fps": 0.0, "morphs": [],
     "frames": [{"time_s": 0.0, "blendshapes": {},
                 "bones": {}}]},
    {"format": "face-take-v1", "fps": 3.0, "morphs": [],
     "frames": [{"time_s": 0.0, "blendshapes": {},
                 "bones": {"head": [1.0, 2.0]}}]},
])
def test_take_json_refusals(blob):
    with pytest.raises(FaceMocapError) as exc:
        FaceTake.from_json(blob, where="take")
    assert exc.value.kind == "bad_take"


def test_take_needs_frames_and_positive_fps():
    from nomorals.media.facemcap.take import FaceFrame
    with pytest.raises(FaceMocapError):
        FaceTake(name="empty", root_name="empty", fps=3.0,
                 morph_names=[], frames=[])
    with pytest.raises(FaceMocapError):
        FaceTake(name="bad", root_name="bad", fps=0.0,
                 morph_names=[], frames=[
                     FaceFrame(time_s=0.0, blendshapes={},
                               bones={})])


# ---------------------------------------------------------------------------
# Drive through a rig session
# ---------------------------------------------------------------------------

def test_drive_take_through_rig_session(tmp_path):
    from nomorals.media.rig import get_session
    take = load_manifest(_write_fixtures(tmp_path), name="smile")
    session = get_session("default")
    posed = drive_take(session, take)
    assert len(posed) == 3
    head0 = posed[0].rotations["head"]
    assert head0[0] == pytest.approx(-90.0, abs=1e-9)
    assert head0[1] == pytest.approx(0.0, abs=1e-9)
    assert head0[2] == pytest.approx(90.0, abs=1e-9)
    assert posed[0].blendshapes == {
        "eyeBlinkLeft": 0.5, "jawOpen": 0.4}
    assert posed[0].unmapped == ()
    assert posed[1].blendshapes == {
        "eyeBlinkLeft": 1.0, "jawOpen": 0.8}
    assert posed[1].rotations["head"] != pytest.approx(
        head0, abs=1e-6)
    assert posed[2].rotations["head"] == pytest.approx(
        (0.0, 0.0, 0.0), abs=1e-9)
    assert posed[2].blendshapes == {
        "eyeBlinkLeft": 0.0, "jawOpen": 0.0}
    # A second drive plays the same poses — the driver resets.
    again = drive_take(session, take)
    assert [p.rotations["head"] for p in again] == [
        p.rotations["head"] for p in posed]
    assert [p.blendshapes for p in again] == [
        p.blendshapes for p in posed]


def test_drive_take_unknown_bone_named_on_pose(tmp_path):
    from nomorals.media.rig import get_session
    mapping = FaceMapping.from_json([
        [{"path": ["ghost"], "axis": "Z"}, "HeadZ"],
    ])
    anim = CSVAnimation.parse(_CSV)
    from nomorals.media.facemcap.loader import _build
    take = _build("ghost", anim, root_name="ghost", fps=3.0,
                  mapping=mapping, source="test")
    posed = drive_take(get_session("default"), take)
    assert "ghost" in posed[0].unmapped
    assert "ghost" not in posed[0].rotations


def test_render_frame_writes_png(tmp_path):
    from nomorals.media.rig import get_session
    take = load_manifest(_write_fixtures(tmp_path), name="smile")
    out = tmp_path / "frame0.png"
    result = render_frame(get_session("default"), take, 0, out)
    assert result["path"] == str(out.resolve())
    assert result["frame"] == 0
    assert out.exists() and out.stat().st_size > 0
    with pytest.raises(FaceMocapError) as exc:
        render_frame(get_session("default"), take, 99, out)
    assert exc.value.kind == "bad_frame"


# ---------------------------------------------------------------------------
# Show seam
# ---------------------------------------------------------------------------

def _show(name: str = "demo", *, rig: bool = True) -> Show:
    return Show(name, MocapStateStore(),
                rig=_rig() if rig else None)


def test_show_face_take_drives_blendshapes_and_head(tmp_path):
    manifest = _write_fixtures(tmp_path)
    show = _show()
    loaded = show.load_face_take(manifest)
    # The show registers its face take under "face:<show>", so
    # it never collides with a tool-loaded take of the same
    # file — the player carries that registry name.
    assert loaded["name"] == "face:demo"
    show.start()
    # start() runs the first live tick: frame 0 is in.
    status = show.status()
    assert status["state"] == "live"
    assert status["face_take"]["name"] == "face:demo"
    assert status["face"]["attached"] is True
    assert status["face"]["frame"] == 0
    pose = show._last_posed
    assert pose is not None
    assert pose.blendshapes["eyeBlinkLeft"] == 0.5
    assert pose.blendshapes["jawOpen"] == 0.4
    assert pose.rotations["head"][0] == pytest.approx(
        -90.0, abs=1e-9)
    view = show.store.read("face")
    assert view["payload"]["blendshapes"] == {
        "eyeBlinkLeft": 0.5, "jawOpen": 0.4}
    # Tick 2: frame 1 — the blendshapes move, the head moves.
    receipt = show.step()
    assert receipt["face"]["attached"] is True
    assert receipt["face"]["frame"] == 1
    assert show._last_posed.blendshapes["eyeBlinkLeft"] == 1.0
    assert show._last_posed.rotations["head"] != pytest.approx(
        (-90.0, 0.0, 90.0), abs=1e-6)
    show.stop()
    assert show.status()["state"] == "done"


def test_show_face_only_no_skeleton_take(tmp_path):
    # A show with a face take and no skeleton take still runs:
    # the rig holds the body at rest and the face drives.
    show = _show()
    show.load_face_take(_write_fixtures(tmp_path))
    show.start()
    receipt = show.step()
    assert receipt["state"] == "live"
    assert receipt["face"]["attached"] is True
    assert receipt["phases"]["drive"]["errors"] == []
    assert receipt["phases"]["capture"]["errors"] == []
    assert show._last_posed is not None
    show.stop()


def test_show_without_face_take_unchanged(tmp_path):
    # The seam no-ops: a show with no face take drives exactly
    # as before, and says so in the receipt.
    show = _show()
    show.start()
    receipt = show.step()
    assert receipt["face"] == {"attached": False}
    assert receipt["state"] == "live"
    assert show._last_posed is not None
    show.stop()


def test_show_face_unknown_bone_recorded_once(tmp_path):
    (tmp_path / "ghost.csv").write_text(_CSV, encoding="utf-8")
    (tmp_path / "ghost-mapping.json").write_text(
        json.dumps([["eyeBlinkLeft", "EyeBlinkLeft"],
                    [{"path": ["ghost"], "axis": "Z"}, "HeadZ"]]),
        encoding="utf-8")
    manifest = tmp_path / "ghost.morph_anim.json"
    manifest.write_text(json.dumps({
        "csv": "ghost.csv", "root_name": "ghost",
        "mapping": "ghost-mapping.json"}), encoding="utf-8")
    show = _show()
    show.load_face_take(manifest)
    show.start()
    show.step()
    show.step()
    show.step()
    unknown = [e for e in show.events()
               if e.get("kind") == "face_bone_unknown"]
    assert len(unknown) == 1
    assert unknown[0]["detail"]["bones"] == ["ghost"]
    show.stop()


# ---------------------------------------------------------------------------
# Wiring: tool, CLI, chat
# ---------------------------------------------------------------------------

def _registry(tmp_path):
    registry = ToolRegistry(SimpleNamespace(home=str(tmp_path)))
    registry.register_builtins()
    return registry


def _tool(registry, **kwargs):
    outcome = registry.call("facemcap", **kwargs)
    if getattr(outcome, "ok", False):
        return outcome.value
    raise AssertionError(
        f"facemcap call failed: {getattr(outcome, 'error', outcome)}")


def test_tool_count_is_787_after_facemcap():
    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    assert len(registry._tools) == 811
    assert "facemcap" in registry._tools


def test_tool_facemcap_actions(tmp_path):
    manifest = _write_fixtures(tmp_path)
    registry = _registry(tmp_path)
    assert _tool(registry, action="list") == {"takes": []}
    loaded = _tool(registry, action="load", node="smile",
                   sources=[str(manifest)])
    assert loaded["ok"] is True
    assert loaded["frames"] == 3
    status = _tool(registry, action="status", node="smile")
    assert status["name"] == "smile"
    assert status["fps"] == 3.0
    frame = _tool(registry, action="frame", node="smile",
                  params={"index": 0})
    assert frame["blendshapes"] == {
        "eyeBlinkLeft": 0.5, "jawOpen": 0.4}
    assert frame["bones"]["head"][0] == pytest.approx(
        -90.0, abs=1e-9)
    driven = _tool(registry, action="drive", node="smile",
                   value="default")
    assert driven["frames"] == 3
    assert driven["rig_session"] == "default"
    assert driven["rig"] == "devon-default"
    out = tmp_path / "smile.json"
    exported = _tool(registry, action="export", node="smile",
                     output=str(out))
    assert exported["frames"] == 3
    assert out.exists()
    back = load_take_json(out, name="smile-back")
    assert back.total == 3
    rendered = _tool(registry, action="render", node="smile",
                     value="default", output=str(
                         tmp_path / "f0.png"),
                     params={"index": 0})
    assert (tmp_path / "f0.png").exists()
    assert rendered["frame"] == 0
    listed = _tool(registry, action="list")
    assert [t["name"] for t in listed["takes"]] == ["smile"]
    dropped = _tool(registry, action="drop", node="smile")
    assert dropped["frames"] == 3
    assert _tool(registry, action="list") == {"takes": []}


def test_tool_facemcap_refusals(tmp_path):
    registry = _registry(tmp_path)
    outcome = registry.call("facemcap", action="bogus")
    assert not outcome.ok
    outcome = registry.call("facemcap", action="load")
    assert not outcome.ok
    outcome = registry.call("facemcap", action="status",
                            node="ghost")
    assert not outcome.ok


def test_cli_facemcap(tmp_path):
    from nomorals.cmdline.dispatch import main
    manifest = _write_fixtures(tmp_path)
    assert main(["facemcap", "list"]) == 0
    assert main(["facemcap", "load", "--session", "smile",
                 "--path", str(manifest)]) == 0
    assert main(["facemcap", "status", "--session", "smile"]) == 0
    assert main(["facemcap", "frame", "--session", "smile",
                 "--index", "0"]) == 0
    assert main(["facemcap", "drive", "--session", "smile",
                 "--rig", "default"]) == 0
    out = tmp_path / "smile.json"
    assert main(["facemcap", "export", "--session", "smile",
                 "--out", str(out)]) == 0
    assert out.exists()
    assert main(["facemcap", "render", "--session", "smile",
                 "--rig", "default", "--index", "1",
                 "--out", str(tmp_path / "f1.png")]) == 0
    assert (tmp_path / "f1.png").exists()
    assert main(["facemcap", "frame", "--session", "smile",
                 "--index", "99"]) == 2
    assert main(["facemcap", "drop", "--session", "smile"]) == 0
    assert main(["facemcap", "list"]) == 0


def test_cli_facemcap_alias(tmp_path):
    from nomorals.cmdline.dispatch import main
    assert main(["fc", "list"]) == 0


def test_chat_facemcap(tmp_path, monkeypatch):
    from nomorals.agents.partner.runtime_mocap import (
        RuntimeMocapMixin,
    )
    monkeypatch.chdir(tmp_path)
    manifest = _write_fixtures(tmp_path)
    mixin = RuntimeMocapMixin()
    assert "/facemcap load" in mixin._control_facemcap("", "test")
    assert "facemcap takes:" in mixin._control_facemcap(
        "list", "test")
    loaded = mixin._control_facemcap(
        f"load {manifest} smile", "test")
    assert "smile loaded: 3 frames" in loaded
    status = mixin._control_facemcap("status smile", "test")
    assert json.loads(status)["frames"] == 3
    frame = mixin._control_facemcap("frame smile 0", "test")
    assert json.loads(frame)["blendshapes"] == {
        "eyeBlinkLeft": 0.5, "jawOpen": 0.4}
    driven = mixin._control_facemcap("drive smile", "test")
    assert "driven through rig" in driven
    exported = mixin._control_facemcap("export smile", "test")
    assert "facemcap-smile.json" in exported
    assert (tmp_path / "facemcap-smile.json").exists()
    rendered = mixin._control_facemcap("render smile 1", "test")
    assert "facemcap-smile-1.png" in rendered
    assert (tmp_path / "facemcap-smile-1.png").exists()
    assert "smile dropped" in mixin._control_facemcap(
        "drop smile", "test")
    assert "/facemcap load" in mixin._control_facemcap(
        "bogus", "test")


def test_chat_facemcap_attach(tmp_path):
    from nomorals.agents.partner.runtime_mocap import (
        RuntimeMocapMixin,
    )
    from nomorals.media.show import load_show
    manifest = _write_fixtures(tmp_path)
    mixin = RuntimeMocapMixin()
    mixin._control_facemcap(f"load {manifest} smile", "test")
    show = load_show("demo", MocapStateStore(), rig=_rig())
    attached = mixin._control_facemcap(
        "attach smile demo", "test")
    assert "attached to show demo" in attached
    assert show.status()["face_take"]["name"] == "smile"


def test_control_command_facemcap_registered():
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
    )
    assert "facemcap" in CONTROL_COMMANDS
    assert "facemcap" in COMMAND_DETAILS
    assert "facemcap" in LIST_ONELINERS
    assert any("facemcap" in group
               for _, group in LIST_GROUPS)
