"""``nm animato`` — Animato Studio in Devon: spark → build → share.

    nm animato new <name> [--niche N] [--style S] [--voice V] [--music M]
                          [--channel C] [--schedule S]
    nm animato list [--json]
    nm animato show <project> [--json]
    nm animato spark "<idea>" [--project NAME] [--publish youtube] [--private]
                              [--style S] [--voice V]
    nm animato queue <project> "<topic>"
    nm animato post <project> <episode> [--private]
    nm animato export <project> <episode>
    nm animato status [project]

A spark runs one episode end to end through the shorts factory and
either posts it straight to YouTube or hands back the finished file.
Nothing here raises a traceback — failures print cleanly.
"""

from __future__ import annotations

import argparse
import json
import sys


def _err(msg: str) -> int:
    print(f"animato: {msg}", file=sys.stderr)
    return 1


def _ok(msg: str) -> int:
    print(msg)
    return 0


def _runner():
    from ...media.contentops.studio import StudioRunner

    return StudioRunner()


def _project_arg(args) -> str:
    return (getattr(args, "project", "") or "").strip()


def _cmd_new(args: argparse.Namespace, context) -> int:
    try:
        runner = _runner()
        project = runner.store.create(
            args.name, niche=args.niche, style=args.style or "",
            voice=args.voice or "edge", music=args.music or "",
            channel=args.channel or "", schedule=args.schedule or "")
        return _ok(f"created studio project {project.slug} "
                   f"({project.niche} — {project.name})")
    except ValueError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        return _err(str(exc))


def _cmd_list(args: argparse.Namespace, context) -> int:
    try:
        projects = _runner().store.list()
        if args.json:
            print(json.dumps([p.to_dict() for p in projects], indent=2,
                             default=str))
            return 0
        if not projects:
            return _ok("no studio projects yet — "
                       "`nm animato new <name>` to start one")
        for p in projects:
            print(f"{p.slug:24} {p.niche:14} {len(p.episodes)} episodes  "
                  f"{len(p.topic_queue)} queued  {p.name[:40]}")
        return 0
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def _cmd_show(args: argparse.Namespace, context) -> int:
    try:
        project = _runner().store.load(_project_arg(args) or args.name)
        if args.json:
            print(json.dumps(project.to_dict(), indent=2, default=str))
            return 0
        print(f"{project.slug} — {project.name}")
        print(f"  niche: {project.niche}  style: {project.style or '(default)'}"
              f"  voice: {project.voice}  footage: {project.footage}")
        print(f"  music: {project.music or '(none)'}  "
              f"channel: {project.channel or '(not set)'}  "
              f"schedule: {project.schedule or '(not set)'}")
        if project.cast:
            print("  cast:")
            for c in project.cast:
                print(f"    • {c.name} (voice: {c.voice}"
                      f"{', ref: ' + c.face_ref if c.face_ref else ''})")
        if project.topic_queue:
            print("  queued topics:")
            for t in project.topic_queue:
                print(f"    • {t}")
        if project.episodes:
            print("  episodes:")
            for e in project.episodes[-10:]:
                print(f"    • {e.id}  {e.status:10} {e.topic[:50]}"
                      f"{'  → ' + e.video_path if e.video_path else ''}")
        return 0
    except FileNotFoundError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def _cmd_spark(args: argparse.Namespace, context) -> int:
    try:
        runner = _runner()
        slug = _project_arg(args)
        if slug:
            project = runner.store.load(slug)
        else:
            from ...media.contentops.studio import slugify

            slug = slugify(args.idea)
            try:
                project = runner.store.load(slug)
            except FileNotFoundError:
                project = runner.store.create(
                    args.idea, niche="did_you_know",
                    style=args.style or "", voice=args.voice or "edge")
        if args.style:
            project.style = args.style
        if args.voice:
            project.voice = args.voice
        if args.style or args.voice:
            runner.store.save(project)
        result = runner.spark(project, args.idea,
                              publish=args.publish or "",
                              private=bool(args.private))
        if not result.get("ok"):
            for note in result.get("notes", []):
                print(f"animato: {note}", file=sys.stderr)
            return 1
        out = f"episode {result['episode_id']} → {result['video_path']}"
        if result.get("published"):
            out += f"  posted: {result['published']}"
        return _ok(out)
    except FileNotFoundError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def _cmd_queue(args: argparse.Namespace, context) -> int:
    try:
        runner = _runner()
        project = runner.store.queue_topic(_project_arg(args) or args.name,
                                           args.topic)
        return _ok(f"queued in {project.slug}: {args.topic[:60]} "
                   f"({len(project.topic_queue)} queued)")
    except FileNotFoundError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def _cmd_post(args: argparse.Namespace, context) -> int:
    try:
        runner = _runner()
        result = runner.publish_episode(_project_arg(args) or args.name,
                                        args.episode,
                                        private=bool(args.private))
        if not result.get("ok"):
            return _err(result.get("reason", "publish failed"))
        return _ok(f"posted episode {result['episode_id']} → "
                   f"{result.get('video_id', '')}")
    except (FileNotFoundError, KeyError) as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def _cmd_export(args: argparse.Namespace, context) -> int:
    try:
        runner = _runner()
        result = runner.export_episode(_project_arg(args) or args.name,
                                       args.episode)
        if not result.get("ok"):
            return _err(result.get("reason", "export failed"))
        return _ok(result["path"])
    except (FileNotFoundError, KeyError) as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def _cmd_status(args: argparse.Namespace, context) -> int:
    try:
        runner = _runner()
        slug = _project_arg(args)
        projects = ([runner.store.load(slug)] if slug
                    else runner.store.list())
        rows = []
        for p in projects:
            for e in p.episodes:
                rows.append((p.slug, e))
        if args.json:
            print(json.dumps(
                [{"project": s, **e.to_dict()} for s, e in rows],
                indent=2, default=str))
            return 0
        if not rows:
            return _ok("no episodes yet — `nm animato spark \"<idea>\"` "
                       "to make one")
        for s, e in rows[-20:]:
            print(f"{s:24} {e.id}  {e.status:10} {e.topic[:50]}")
        return 0
    except FileNotFoundError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


HANDLERS = {
    "new": _cmd_new,
    "list": _cmd_list,
    "show": _cmd_show,
    "spark": _cmd_spark,
    "queue": _cmd_queue,
    "post": _cmd_post,
    "export": _cmd_export,
    "status": _cmd_status,
}


def cmd_animato(args: argparse.Namespace, context) -> int:
    handler = HANDLERS.get(args.animato_action)
    if handler is None:
        return _err(f"unknown animato action {args.animato_action!r}")
    return handler(args, context)
