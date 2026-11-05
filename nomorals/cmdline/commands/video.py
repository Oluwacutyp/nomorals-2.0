"""``nm video`` — Devon Studio on the command line.

    nm video generate "a neon alley in rain" [--backend auto|ltx|wan|motion]
    nm video motion lyric --audio song.mp3 --lyrics "..." [--preset neon_pop]
    nm video motion visual --audio song.mp3 [--preset phonk] [--images cover.png]
    nm video motion slideshow --images a.png,b.png [--audio song.mp3]
    nm video motion trailer --clips a.mp4,b.mp4 --title "MIDNIGHT RUN"
    nm video motion grade --file clip.mp4 --preset cinematic
    nm video motion export --file clip.mp4 --format 9:16
    nm video chain scenes.json [--backend auto] [--style cinematic]
    nm video formats [list|show <name>|scaffold <name>]
    nm video capability
    nm video list [--json]

Neural generation needs CUDA ≥ 8GB VRAM (LTX) / ≥ 16GB (Wan) plus the
``diffusers`` package; otherwise generation routes honestly to the
motion studio. Nothing here raises a traceback — failures print cleanly.
"""

from __future__ import annotations

import argparse
import json
import sys


def _err(msg: str) -> int:
    print(f"video: {msg}", file=sys.stderr)
    return 1


def _ok(msg: str) -> int:
    print(msg)
    return 0


def _cmd_generate(args: argparse.Namespace, context) -> int:
    from ...media.videogen import generate, VideogenError
    try:
        res = generate(
            args.prompt, backend=args.backend, mode=args.mode,
            image=args.image or None, duration_s=args.seconds,
            seed=args.seed, out=args.out or None,
            model=getattr(args, "model", "") or None,
            lora=getattr(args, "lora", None) or None)
    except VideogenError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        return _err(str(exc))
    return _ok(res.message())


def _cmd_capability(args: argparse.Namespace, context) -> int:
    from ...media.videogen import capability_report
    return _ok(capability_report(prefer=getattr(args, "backend", "auto")))


def _cmd_list(args: argparse.Namespace, context) -> int:
    from ...media.motion_studio._core import ledger_status, read_ledger
    if args.json:
        print(json.dumps(read_ledger(), indent=2))
        return 0
    ledger = read_ledger(limit=20)
    if not ledger:
        return _ok("no studio renders yet — try `nm video motion slideshow --help`")
    st = ledger_status()
    header = f"🎬 studio renders: {st['total_entries']} total"
    if st["archived_entries"]:
        header += (f" ({st['archived_entries']} in the archive beyond the "
                   f"live window)")
    header += f" — showing the latest {len(ledger)}"
    print(header)
    for e in ledger:
        print(f"{e.get('kind', '?'):22} {e.get('path', '?')}")
    return 0


def _cmd_chain(args: argparse.Namespace, context) -> int:
    from ...media.videogen import chain_scenes, VideogenError
    try:
        scenes = json.loads(open(args.scenes_file).read())
    except Exception as exc:  # noqa: BLE001
        return _err(f"could not read scenes file: {exc}")
    try:
        report = chain_scenes(
            scenes, out=args.out or None, backend=args.backend,
            style=args.style, transition=args.transition,
            grade_preset=args.grade, format=args.format, seed=args.seed)
    except VideogenError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    return _ok(report.summary())


def _motion_lyric(args) -> int:
    from ...media.motion_studio import studio, MotionStudioError
    lyrics = args.lyrics
    if lyrics.startswith("@"):
        try:
            lyrics = open(lyrics[1:]).read()
        except OSError as exc:
            return _err(f"could not read lyrics file: {exc}")
    try:
        path = studio.make_lyric_video(
            args.audio, lyrics, out=args.out or None, preset=args.preset,
            image=args.image or None, grade_preset=args.grade,
            format=args.format)
    except MotionStudioError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    return _ok(f"🎤 lyric video → {path}")


def _motion_visual(args) -> int:
    from ...media.motion_studio import studio, MotionStudioError
    try:
        path = studio.make_music_visualizer(
            args.audio, out=args.out or None,
            images=[i for i in (args.images or "").split(",") if i],
            preset=args.preset,
            duration=args.seconds or None,
            grade_preset=args.grade, format=args.format or "")
    except MotionStudioError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    return _ok(f"🎛️ visualizer → {path}")


def _motion_slideshow(args) -> int:
    from ...media.motion_studio import studio, MotionStudioError
    images = [i for i in (args.images or "").split(",") if i]
    try:
        path = studio.make_slideshow(
            images, out=args.out or None, audio=args.audio or None,
            per_image=args.per_image, move=args.move,
            transition=args.transition, grade_preset=args.grade,
            format=args.format)
    except MotionStudioError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    return _ok(f"🖼️ slideshow → {path}")


def _motion_trailer(args) -> int:
    from ...media.motion_studio import studio, MotionStudioError
    clips = [c for c in (args.clips or "").split(",") if c]
    try:
        path = studio.make_trailer(
            clips, out=args.out or None, title=args.title,
            tagline=args.tagline, audio=args.audio or None,
            clip_len=args.clip_len, transition=args.transition,
            grade_preset=args.grade, format=args.format)
    except MotionStudioError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    return _ok(f"🎬 trailer → {path}")


def _motion_grade(args) -> int:
    from ...media.motion_studio import grading, MotionStudioError
    try:
        path = grading.grade(args.file, out=args.out or None,
                             preset=args.preset)
    except MotionStudioError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    return _ok(f"🎨 graded → {path}")


def _motion_export(args) -> int:
    from ...media.motion_studio import grading, MotionStudioError
    try:
        path = grading.export(args.file, out=args.out or None,
                              format=args.format)
    except MotionStudioError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    return _ok(f"📐 exported ({args.format}) → {path}")


def _motion_project(args) -> int:
    """Render a saved MotionProject document (Echelon Wave 4)."""
    from ...media.motion_studio import MotionStudioError
    from ...media.motion_studio.project import MotionProject
    fmt_note = ""
    try:
        proj = MotionProject.load(args.file)
        fmt_name = getattr(args, "format", "") or ""
        if fmt_name:
            from ...media.edit_engine.formats import resolve_format
            resolved = resolve_format(fmt_name)
            proj.output.update({"width": resolved.width,
                                "height": resolved.height,
                                "fps": resolved.fps})
            fmt_note = (f" [format {resolved.name}: "
                        f"{resolved.width}×{resolved.height}"
                        f"@{resolved.fps}]")
        res = proj.render(out=args.out or None)
    except MotionStudioError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    rec = res.get("motion_project", {})
    return _ok(f"🎬 motion project {rec.get('project_hash', '')[:12]} "
               f"({len(rec.get('segments', []))} segments) → "
               f"{res['output']}{fmt_note}")


def _cmd_formats(args: argparse.Namespace, context) -> int:
    """``nm video formats`` — the §1.5b format registry on the CLI."""
    from ...media.edit_engine.formats import (
        FormatError, formats_data, resolve_format, scaffold_format)
    action = getattr(args, "formats_action", None) or "list"
    try:
        if action == "list":
            data = formats_data()
            if getattr(args, "json", False):
                print(json.dumps(data, indent=2))
                return 0
            print(f"📐 formats: {data['count']} registered")
            for f in data["formats"]:
                groups = ",".join(f["groups"]) or "—"
                print(f"  {f['name']:18} {f['label']}  [{groups}]")
            for issue in data["bootstrap_issues"]:
                print(f"  ⚠ {issue}")
            return 0
        if action == "show":
            resolved = resolve_format(args.name)
            if getattr(args, "json", False):
                print(json.dumps(resolved.to_dict(), indent=2))
                return 0
            d = resolved.to_dict()
            print(f"📐 {d['name']} — {d['label']}")
            print(f"  canvas:   {d['canvas']['width']}×"
                  f"{d['canvas']['height']}@{d['canvas']['fps']}")
            print(f"  safe:     {d['safe']}")
            print(f"  captions: {d['captions']}")
            print(f"  grade:    {d['grade_pack']}")
            print(f"  loudness: {d['loudness']}")
            print(f"  design:   {d['design_pack']}")
            print(f"  renderer: {d['renderer']}")
            print(f"  provenance: {d['provenance']}")
            return 0
        if action == "scaffold":
            path = scaffold_format(args.name,
                                   dest=args.out or None)
            return _ok(
                f"📐 scaffold → {path}\n"
                f"fill in the TODOs, then load it in code with "
                f"load_format_file() or point NM_FORMAT_DIRS at its "
                f"directory")
    except FormatError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        return _err(str(exc))
    return _err("usage: nm video formats list|show <name>|scaffold <name>")


def _motion_moves(args) -> int:
    from ...media.motion_studio.kenburns import list_moves
    for m in list_moves():
        print(f"{m['name']:12} {m['label']}")
    return 0


def _motion_presets(args) -> int:
    from ...media.motion_studio.typography import list_presets as lyric_p
    from ...media.motion_studio.visualizer import list_presets as viz_p
    from ...media.motion_studio.grading import list_grades
    print("lyric presets:")
    for p in lyric_p():
        print(f"  {p['name']:16} {p['label']}")
    print("visualizer presets:")
    for p in viz_p():
        print(f"  {p['name']:16} style={p['style']} palette={p['palette']}")
    print("grades:")
    for g in list_grades():
        print(f"  {g['name']:16} {g['label']}")
    return 0


_MOTION_DISPATCH = {
    "lyric": _motion_lyric,
    "visual": _motion_visual,
    "slideshow": _motion_slideshow,
    "trailer": _motion_trailer,
    "grade": _motion_grade,
    "export": _motion_export,
    "project": _motion_project,
    "moves": _motion_moves,
    "presets": _motion_presets,
}


def cmd_video(args: argparse.Namespace, context) -> int:
    action = getattr(args, "video_action", "")
    if action == "generate":
        return _cmd_generate(args, context)
    if action == "motion":
        fn = _MOTION_DISPATCH.get(getattr(args, "motion_action", ""))
        if fn is None:
            return _err("usage: nm video motion "
                        "lyric|visual|slideshow|trailer|grade|export|project|moves|presets")
        return fn(args)
    if action == "chain":
        return _cmd_chain(args, context)
    if action == "formats":
        return _cmd_formats(args, context)
    if action == "capability":
        return _cmd_capability(args, context)
    if action == "list":
        return _cmd_list(args, context)
    return _err("usage: nm video generate|motion|chain|capability|list ...")
