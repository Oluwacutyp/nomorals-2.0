"""``nm shorts`` — Devon's short-form content empire on the command line.

    nm shorts make <niche> "<topic>" [--platforms youtube,tiktok] [--now]
    nm shorts queue <niche> "<topic>" [--at "2026-10-10 18:00"] [--platforms ...]
    nm shorts status [job_id]
    nm shorts resume <run_id>
    nm shorts post <job_id> <platform> [--at "2026-10-10 18:00"]
    nm shorts niches
    nm shorts niche add <name>
    nm shorts calendar [--due]
    nm shorts ledger [--platform youtube]
    nm shorts estimate <niche> "<topic>"

Renders are heavyweight (imggen + ffmpeg); every stage is logged and
resumable. Nothing here raises a traceback — failures print cleanly.
"""

from __future__ import annotations

import argparse
import json
import sys


def _err(msg: str) -> int:
    print(f"shorts: {msg}", file=sys.stderr)
    return 1


def _ok(msg: str) -> int:
    print(msg)
    return 0


def _pipeline(args):
    from ...media.contentops.pipeline import ShortPipeline

    return ShortPipeline()


def _cmd_make(args: argparse.Namespace, context) -> int:
    try:
        pipe = _pipeline(args)
        platforms = [p.strip() for p in (args.platforms or "").split(",") if p.strip()]
        job = pipe.plan(args.niche, args.topic, platforms=platforms or None)
        if args.now:
            result = pipe.run(job)
            if result.ok:
                return _ok(f"rendered {job.id} → {result.final_path}")
            return _err(f"render failed: {result.error}")
        return _ok(f"queued {job.id} ({args.niche} — {args.topic[:60]})")
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        return _err(str(exc))


def _cmd_queue(args: argparse.Namespace, context) -> int:
    try:
        pipe = _pipeline(args)
        platforms = [p.strip() for p in (args.platforms or "").split(",") if p.strip()]
        post = pipe.calendar.add(args.niche, args.topic,
                                 platforms=platforms or None,
                                 scheduled_for=args.at or "")
        when = f" at {args.at}" if args.at else ""
        return _ok(f"scheduled {post.id}{when} ({args.niche} — {args.topic[:60]})")
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def _cmd_status(args: argparse.Namespace, context) -> int:
    try:
        pipe = _pipeline(args)
        jobs = pipe.jobs.list()
        if args.job_id:
            jobs = [j for j in jobs if j.id == args.job_id or j.id.startswith(args.job_id)]
            if not jobs:
                return _err(f"unknown job {args.job_id!r}")
        if args.json:
            print(json.dumps([j.__dict__ for j in jobs], indent=2, default=str))
            return 0
        if not jobs:
            return _ok("no jobs yet — `nm shorts make <niche> \"<topic>\"` to start")
        for j in jobs[-20:]:
            print(f"{j.id}  {j.status:10} {j.niche:14} {j.topic[:50]}"
                  f"{'  run=' + j.run_id if j.run_id else ''}")
        return 0
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def _cmd_resume(args: argparse.Namespace, context) -> int:
    try:
        pipe = _pipeline(args)
        result = pipe.resume(args.run_id)
        if result.ok:
            return _ok(f"resumed → {result.final_path}")
        return _err(f"failed: {result.error}")
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def _load_post_copy(pipe, job):
    """Title/description/hashtags the pipeline wrote for this job."""
    from pathlib import Path

    if job.run_id:
        pc = Path(pipe.runs_root) / job.run_id / "post_copy.json"
        if pc.exists():
            try:
                return json.loads(pc.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
    return {"title": job.topic, "description": "", "hashtags": []}


def _platform_publisher(name: str):
    """Lazy platform → publisher class (honest import errors)."""
    if name == "youtube":
        from ...media.contentops.publish.youtube import YouTubePublisher
        return YouTubePublisher
    if name == "tiktok":
        from ...media.contentops.publish.tiktok import TikTokPublisher
        return TikTokPublisher
    if name in ("instagram", "reels"):
        from ...media.contentops.publish.meta import MetaPublisher
        return MetaPublisher
    if name == "facebook":
        from ...media.contentops.publish.meta import MetaPublisher
        return MetaPublisher
    if name == "x":
        from ...media.contentops.publish.x import XPublisher
        return XPublisher
    return None


def _cmd_post(args: argparse.Namespace, context) -> int:
    try:
        from ...media.contentops.pipeline import ShortPipeline

        pipe = ShortPipeline()
        jobs = [j for j in pipe.jobs.list()
                if j.id == args.job_id or j.id.startswith(args.job_id)]
        if not jobs:
            return _err(f"unknown job {args.job_id!r}")
        job = jobs[0]
        if job.status != "ready":
            return _err(f"job {job.id} is '{job.status}', not ready — render it first")
        pub_cls = _platform_publisher(args.platform)
        if pub_cls is None:
            return _err("unknown platform %r (youtube|tiktok|instagram|facebook|x)"
                        % args.platform)
        video_path = job.final_path or ""
        if not video_path:
            return _err(f"job {job.id} has no rendered video")
        copy = _load_post_copy(pipe, job)
        title = copy.get("title", job.topic)
        description = copy.get("description", "")
        tags = copy.get("hashtags", []) or []
        publisher = pub_cls()
        plat = args.platform
        if plat == "youtube":
            from datetime import datetime
            publish_at = None
            if args.at:
                try:
                    publish_at = datetime.fromisoformat(args.at)
                except ValueError:
                    return _err(f"bad --at time {args.at!r} (use ISO: 2026-10-10T18:00:00)")
            result = publisher.publish(
                video_path, title, description=description, tags=tags,
                publish_at=publish_at,
                privacy="private" if publish_at else "public",
                confirmed=True)
        elif plat == "tiktok":
            result = publisher.post_video(
                video_path, caption=f"{title} {description}".strip(), tags=tags)
        elif plat in ("instagram", "reels"):
            # Meta fetches from a public URL — gate honestly on local files.
            return _err("instagram reels need a public HTTPS URL for the video "
                        "(Meta fetches it); upload the file somewhere public first, "
                        "then use the publisher API directly")
        elif plat == "facebook":
            return _err("facebook video needs a public HTTPS URL for the video "
                        "or a page upload flow; use the publisher API directly")
        elif plat == "x":
            result = publisher.post_video(video_path, title, tags=tags)
        else:
            return _err(f"unhandled platform {plat!r}")
        return _ok(f"posted → {json.dumps(result, default=str)[:300]}")
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def _cmd_niches(args: argparse.Namespace, context) -> int:
    try:
        from ...media.contentops.niches import list_niches, get_niche

        for name in list_niches():
            plugin = get_niche(name)
            thesis = getattr(plugin, "thesis", "")
            cadence = getattr(plugin, "cadence", "?")
            print(f"{name:14} {cadence}/day  {thesis[:70]}")
        return 0
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def _cmd_niche_add(args: argparse.Namespace, context) -> int:
    try:
        from ...media.contentops.niches.scaffold import scaffold

        path = scaffold(args.name)
        return _ok(f"scaffolded niche at {path} — edit it, it self-registers")
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def _cmd_calendar(args: argparse.Namespace, context) -> int:
    try:
        pipe = _pipeline(args)
        posts = pipe.calendar.due() if args.due else pipe.calendar.list()
        if args.json:
            print(json.dumps([p.__dict__ for p in posts], indent=2, default=str))
            return 0
        if not posts:
            return _ok("calendar empty")
        for p in posts:
            print(f"{p.id}  {p.status:10} {p.niche:14} {p.topic[:50]}"
                  f"{'  at ' + p.scheduled_for if p.scheduled_for else ''}")
        return 0
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def _cmd_ledger(args: argparse.Namespace, context) -> int:
    try:
        from ...media.contentops.publish.ledger import PublishLedger

        ledger = PublishLedger()
        entries = ledger.list(platform=args.platform or "")
        if args.json:
            print(json.dumps(entries, indent=2, default=str))
            return 0
        if not entries:
            return _ok("nothing posted yet")
        for e in entries[-20:]:
            print(f"{e.get('at', '')}  {e.get('platform', ''):10} "
                  f"{e.get('status', ''):8} {e.get('video_id', '')} "
                  f"{str(e.get('title', ''))[:50]}")
        return 0
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def _cmd_estimate(args: argparse.Namespace, context) -> int:
    try:
        pipe = _pipeline(args)
        est = pipe.estimate(args.niche, args.topic)
        print(json.dumps(est, indent=2, default=str))
        return 0
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


HANDLERS = {
    "make": _cmd_make,
    "queue": _cmd_queue,
    "status": _cmd_status,
    "resume": _cmd_resume,
    "post": _cmd_post,
    "niches": _cmd_niches,
    "niche": _cmd_niche_add,
    "calendar": _cmd_calendar,
    "ledger": _cmd_ledger,
    "estimate": _cmd_estimate,
}


def cmd_shorts(args: argparse.Namespace, context) -> int:
    handler = HANDLERS.get(args.shorts_action)
    if handler is None:
        return _err(f"unknown shorts action {args.shorts_action!r}")
    return handler(args, context)
