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
    nm animato cast <project> <name> [--voice V] [--face-ref PATH]
                [--character-id ID]  — add a cast member; the face
                reference locks that character's face into every
                scene of every episode (port step 4)
    nm animato status [project]
    nm animato templates [--json]
    nm animato template <genre> <project> "<seed>" [--count N]
    nm animato autopost start|stop|status|run [--project NAME] [--force]

A spark runs one episode end to end through the shorts factory and
either posts it straight to YouTube or hands back the finished file.
The autopost mission runs the same loop on each project's schedule
and posts itself — start it once, it runs every 30 minutes.
A template genres a project (niche, style, voice, music bed) and
queues episodes from a seed — the mission picks them up on schedule.
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
    from ...media.contentops.publish.youtube import YouTubePublisher
    from ...media.contentops.studio import StudioRunner

    # L7 may import the L5 publisher directly — pass the class as the
    # factory so the L4 runner never imports it itself.
    return StudioRunner(publisher_factory=YouTubePublisher)


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


def _cmd_cast(args: argparse.Namespace, context) -> int:
    """Add (or replace by name) a cast member on a project.

    The face reference is validated here — a missing file is a
    plain error at add time, not a failed episode later. The spark
    re-validates (the file may have moved since) and fails closed.
    """
    from pathlib import Path

    from ...media.contentops.studio import CastMember

    face_ref = (getattr(args, "face_ref", "") or "").strip()
    if face_ref and not Path(face_ref).expanduser().is_file():
        return _err(f"no face reference image at {face_ref!r}")
    try:
        runner = _runner()
        project = runner.store.add_cast(
            _project_arg(args) or args.name,
            CastMember(name=args.member, voice=args.voice or "edge",
                       face_ref=face_ref,
                       character_id=args.character_id or ""))
        member = next(c for c in project.cast if c.name == args.member)
        return _ok(f"cast {member.name} → {project.slug} "
                   f"(voice: {member.voice}"
                   f"{', ref: ' + member.face_ref if member.face_ref else ', no reference'})")
    except FileNotFoundError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
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


def _cmd_templates(args: argparse.Namespace, context) -> int:
    try:
        from ...media.contentops.templates import list_templates

        templates = list_templates()
        if args.json:
            print(json.dumps([t.to_dict() for t in templates], indent=2,
                             default=str))
            return 0
        for t in templates:
            print(f"{t.id:10} {t.name:14} {t.niche:14} {t.description}")
        return 0
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def _cmd_template(args: argparse.Namespace, context) -> int:
    from ...media.contentops.templates import apply_template

    try:
        runner = _runner()
        result = apply_template(
            runner.store, (getattr(args, "genre", "") or "").strip(),
            _project_arg(args) or args.name,
            getattr(args, "seed", "") or "",
            count=getattr(args, "count", 1) or 1)
        lines = [f"applied {result['template']} to {result['project']} "
                 f"({result['niche']} — {result['style']} — "
                 f"music {result['music'] or 'ambient'})"]
        for field_name, change in result.get("changes", {}).items():
            lines.append(f"  {field_name}: {change}")
        lines.append(f"  queued {len(result['topics'])} topic(s) "
                     f"({result['queued']} in queue):")
        for topic in result["topics"]:
            lines.append(f"    • {topic}")
        return _ok("\n".join(lines))
    except (FileNotFoundError, ValueError) as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def _cmd_autopost(args: argparse.Namespace, context) -> int:
    from ...media.contentops.autopost import (
        AutopostStore,
        autopost_status,
        run_autopost,
    )

    try:
        runner = _runner()
        store = AutopostStore(runner.store.root)
        action = (args.autopost_action or "").strip().lower()
        if action == "start":
            store.set_enabled(True)
            return _ok("animato autopost started — due projects post "
                       "themselves every 30 minutes")
        if action == "stop":
            store.set_enabled(False)
            return _ok("animato autopost stopped — the schedule will "
                       "not fire (manual `autopost run --force` still works)")
        if action == "status":
            result = autopost_status(runner=runner, store=store)
            if args.json:
                print(json.dumps(result, indent=2, default=str))
                return 0
            print(f"autopost: {'on' if result['enabled'] else 'off'}")
            for row in result["projects"]:
                state = ("DUE" if row["due"] else
                         f"last {row['last_status'] or 'never'} "
                         f"{row['last_run_at'] or ''}").strip()
                print(f"  {row['project']:24} {row['schedule']:28} "
                      f"{row['queued_topics']} queued  {state}")
            for run in result["recent"]:
                print(f"  {run['at']}  {run['project']:24} "
                      f"{run['status']:8} {run.get('topic', '')[:40]}"
                      f"{' — ' + run['reason'] if run.get('reason') else ''}")
            return 0
        if action == "run":
            result = run_autopost(
                runner=runner, store=store,
                project=(getattr(args, "project", "") or "").strip(),
                force=bool(args.force))
            if args.json:
                print(json.dumps(result, indent=2, default=str))
                return 0 if result.get("ok") else 1
            if not result.get("ok"):
                return _err(result.get("reason", "autopost run failed"))
            if not result.get("enabled") and not args.force:
                return _ok("autopost is off — `animato autopost start` "
                           "to run the schedule")
            print(f"autopost: ran {result['ran']} project(s), "
                  f"{result['posted']} posted, {result['failed']} failed, "
                  f"{result['skipped']} skipped")
            for o in result["outcomes"]:
                line = (f"  {o['project']:24} {o['status']:8} "
                        f"{o.get('topic', '')[:40]}")
                if o.get("video_id"):
                    line += f"  → {o['video_id']}"
                if o.get("reason"):
                    line += f"  — {o['reason']}"
                print(line)
            return 0
        return _err(f"unknown autopost action {action!r}")
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
    "cast": _cmd_cast,
    "post": _cmd_post,
    "export": _cmd_export,
    "status": _cmd_status,
    "templates": _cmd_templates,
    "template": _cmd_template,
    "autopost": _cmd_autopost,
}


def cmd_animato(args: argparse.Namespace, context) -> int:
    handler = HANDLERS.get(args.animato_action)
    if handler is None:
        return _err(f"unknown animato action {args.animato_action!r}")
    return handler(args, context)
