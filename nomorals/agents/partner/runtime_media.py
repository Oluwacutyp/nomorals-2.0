"""RuntimeMediaMixin: PartnerRuntime command group (media)."""

from __future__ import annotations

import os

class RuntimeMediaMixin:
    """RuntimeMediaMixin for :class:`PartnerRuntime`."""


    # ── wave 72 systems: media · execution · archives · builders ───────────

    def _control_music(self, tail: str) -> str:
        """/music <topic> [style] | /music styles | /music song [slug]."""
        from ...media.music import STYLES, MusicCreator

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /music <topic> [style]  |  /music styles  |  "
                    "/music song [slug]\nstYLES: " + ", ".join(STYLES))
        words = tail.split()
        if words[0].lower() == "styles":
            return ("styles:\n" + "\n".join(
                f"  {k:12s} {v.label}  ({v.mode}, {v.tempo[0]}-{v.tempo[1]} bpm)"
                for k, v in STYLES.items()))
        if words[0].lower() == "song":
            from ...media.music import _saved_songs

            lookup = " ".join(words[1:]).strip()
            out = _saved_songs(self.context, lookup)
            if lookup:
                if out.get("found"):
                    s = out["song"]
                    return f"found “{s['title']}” → {s['midi']}\n/play it to play."
                names = ", ".join(x["title"] for x in out.get("songs", [])[:8])
                return (f"no saved song matching {lookup!r}. Saved: "
                        f"{names or '(none)'}")
            names = [x["title"] for x in out.get("songs", [])]
            return ("saved songs:\n" + "\n".join(f"  {n}" for n in names)
                    or "saved songs: (none)\n/music <topic> composes one.")
        topic = tail
        style = "pop"
        if words and words[-1].lower() in STYLES and len(words) > 1:
            style = words[-1].lower()
            topic = " ".join(words[:-1])
        if not topic:
            return "usage: /music <topic> [style]"
        try:
            song = MusicCreator(self.context).compose(topic, style=style)
        except Exception as exc:  # noqa: BLE001
            return f"music error: {exc}"
        n_lines = sum(len(sec.lyrics) for sec in song.sections)
        text = (f"🎵 “{song.title}”  [{song.style}, {song.key}, {song.tempo} bpm]\n"
                f"{n_lines} lyric lines across {len(song.sections)} sections\n"
                f"{song.midi_path}\nqueue it with: /play {song.midi_path}")
        return text

    def _control_play(self, tail: str) -> str:
        """/play <paths…> | status | queue | pause | … (transport)."""
        from ...media.playback import PlaybackEngine

        tail = (tail or "").strip()
        actions = {"add", "pause", "resume", "stop", "seek", "volume",
                   "next", "prev", "queue", "remove", "clear", "status"}
        words = tail.split()
        action = words[0].lower() if words and words[0].lower() in actions \
            else "play"
        rest = words[1:] if action != "play" else words
        try:
            engine = PlaybackEngine(self.context)
            if action in ("play", "add"):
                if not rest:
                    st = engine.status()
                    return (f"queue ({st['queue']}): "
                            + (f"now {st['current']}" if st["queue"] else "empty")
                            + "\n/play <path-or-url…> to queue and play")
                added = []
                for ref in rest:
                    res = engine.add(ref)
                    added.extend(res.get("added", []))
                queue_len = len(engine.queue())
                out = (f"queued {len(added)} (queue {queue_len}):\n"
                       + "\n".join(f"  - {a.get('title') or a['path']}"
                                   for a in added))
                if action == "play":
                    st = engine.play()
                    if st.get("status") == "playing":
                        out += f"\n▶ playing “{st.get('current', '')}” " \
                               f"[{st.get('backend')}]"
                    else:
                        out += (f"\nstatus: {st.get('status')} "
                                f"{st.get('error') or st.get('hint', '')}")
                return out
            if action == "seek" and rest:
                return f"seeked to {rest[0]}s: {engine.seek(float(rest[0]))}"
            if action == "volume" and rest:
                return f"volume: {engine.volume(int(rest[0]))}"
            if action == "remove" and rest:
                return f"removed: {engine.remove(int(rest[0]))}"
            if action == "queue":
                items = engine.queue()
                return (f"queue ({len(items)}):\n"
                        + "\n".join(f"  {i:2d}. {it['title']} [{it['kind']}]"
                                    for i, it in enumerate(items))
                        or "queue (0): empty")
            if action in ("pause", "resume", "stop", "next", "prev", "clear"):
                return f"{action}: {getattr(engine, action)()}"
            st = engine.status()
            return (f"backend: {st['backend'].get('name') if isinstance(st['backend'], dict) else st['backend']}\n"
                    f"playing: {st['playing']}  paused: {st['paused']}\n"
                    f"current: {st['current'] or '(none)'} "
                    f"[{st['position']}/{st['queue']} in queue]\n"
                    f"volume: {st['volume']}")
        except Exception as exc:  # noqa: BLE001
            return f"play error: {exc}"

    def _control_video(self, tail: str) -> str:
        """/video <query> [platform] | /video download <url> [audio] | platforms."""
        from ...media.video import _PLATFORM_SITES, VideoFinder

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /video <query> [platform]  |  "
                    "/video download <url> [audio]  |  /video platforms")
        words = tail.split()
        if words[0].lower() == "platforms":
            return "platforms: " + ", ".join(sorted(_PLATFORM_SITES))
        if words[0].lower() == "download":
            rest = words[1:]
            audio = any(w.lower() == "audio" for w in rest)
            url = " ".join(w for w in rest if w.lower() != "audio").strip()
            if not url:
                return "usage: /video download <url> [audio]"
            try:
                out = VideoFinder(self.context).download(url,
                                                         audio_only=audio)
                return f"⬇ downloaded: {out.get('path', out)}"
            except Exception as exc:  # noqa: BLE001
                return f"download failed: {exc}"
        query = tail
        platform = ""
        if words[-1].lower() in _PLATFORM_SITES and len(words) > 1:
            platform = words[-1].lower()
            query = " ".join(words[:-1])
        try:
            out = VideoFinder(self.context).find(query, max_results=8,
                                                 platform=platform)
        except Exception as exc:  # noqa: BLE001
            return f"video error: {exc}"
        if not out["results"]:
            return (f"no video results for “{query}”"
                    + (f" on {platform}" if platform else "")
                    + " — try a broader query or another platform.")
        lines = [f"🎬 {out['count']} result(s) for “{query}”:"]
        for i, r in enumerate(out["results"][:6], 1):
            title = (r.get("title") or r["url"])[:64]
            dur = f"  [{r['duration']}]" if r.get("duration") else ""
            extra = f" · {r['author']}" if r.get("author") else ""
            lines.append(f"  {i}. {title}{dur}{extra}\n     {r['url']}")
        lines.append("download: /video download <url>")
        return "\n".join(lines)

    def _control_image(self, tail: str) -> str:
        ref = (tail or "").strip()
        if not ref:
            return "usage: /image <path-or-url>"
        outcome = self.context.tools.call("image_lookup", path=ref)
        data, error = self._tool_data(outcome)
        if data is None:
            return f"image lookup failed: {error}"
        lines = [f"🖼 {data.get('format', '?')} · {data.get('width') or '?'}×{data.get('height') or '?'} · "
                 f"{data.get('bytes', 0) // 1024} KB"]
        if data.get("seen_before"):
            where = f" (last: {data['seen_where']})" if data.get("seen_where") else ""
            lines.append(f"seen before{data.get('seen_days', '')}{where}")
        near = data.get("near_duplicates") or []
        if near:
            lines.append("near duplicates: " + "; ".join(near[:3]))
        return "\n".join(lines)

    def _control_lens(self, tail: str) -> str:
        ref = (tail or "").strip()
        if not ref:
            return "usage: /lens <path-or-url>"
        outcome = self.context.tools.call("reverse_image_search", path=ref)
        data, error = self._tool_data(outcome)
        if data is None:
            return f"reverse search failed: {error}"
        lines = ["🔍 reverse image search:"]
        for name, link in (data.get("lookup_links") or {}).items():
            lines.append(f"  {name}: {link}")
        matches = data.get("unverified_matches") or []
        if matches:
            lines.append("possible related (unverified, from Bing):")
            lines.extend(f"  {m}" for m in matches[:5])
        if data.get("note"):
            lines.append(data["note"])
        return "\n".join(lines)

    def _control_gen(self, tail: str, chat_key: str = "") -> str:
        parts = (tail or "").split(None, 2)
        if len(parts) < 2:
            outcome = self.context.tools.call("script_kinds")
            kinds = outcome.value["kinds"] if outcome.ok else {}
            lines = ["usage: /gen <kind> <name> [json config]"]
            for kind, cfg in sorted(kinds.items()):
                lines.append(f"  {kind}: " + ", ".join(cfg))
            return "\n".join(lines)
        kind, name = parts[0], parts[1]
        config = parts[2].strip() if len(parts) > 2 else ""
        outcome = self.context.tools.call("script_gen", kind=kind, name=name, config=config)
        if not outcome.ok:
            return f"gen failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        return (f"✅ generated {v['path']} ({v['bytes']} B, validated {v['ext']})\n"
                f"{v['preview']}")

    def _control_apps(self, tail: str) -> str:
        """/apps [list|stacks|build <name> --stack …|info <name>]."""
        from ...builders import AppBuilder, STACKS

        tail = (tail or "").strip()
        tokens = tail.split() if tail else []
        action = tokens[0].lower() if tokens and tokens[0].lower() in (
            "list", "stacks", "build", "info", "serve", "stop", "served",
            "deploy", "stop_deploy", "deployed") else "list"
        rest = tokens[1:] if action != "list" else tokens
        def _port_flag():
            if "--port" in rest:
                i = rest.index("--port")
                if i + 1 < len(rest):
                    try:
                        return int(rest[i + 1])
                    except ValueError:
                        return 0
            return 0
        try:
            b = AppBuilder(self.context)
            if action == "served":
                out = b.served()
                entries = out.get("served") or []
                if not entries:
                    return "no apps currently serving"
                return (f"serving ({out['count']}):\n" + "\n".join(
                    f"  {e['app']}  {e['url']}  (pid {e['pid']}, "
                    f"alive={e['alive']})" for e in entries))
            if action == "serve":
                if not rest:
                    return "usage: /apps serve <name> [--port N]"
                out = b.serve(rest[0], port=_port_flag())
                health = out.get("health") or {}
                text = (f"🖥 serving {out['app']} ({out['stack']}) → "
                        f"{out['url']}  (pid {out.get('pid')}, "
                        f"health={'ok' if health.get('ok') else health.get('error', 'unknown')})")
                if out.get("note"):
                    text += f"\n{out['note']}"
                return text
            if action == "stop":
                if not rest:
                    return "usage: /apps stop <name>"
                out = b.stop(rest[0])
                return (f"stopped {out['app']} (was pid {out['was_pid']}) "
                        f"→ {'stopped' if out['stopped'] else 'already dead'}")
            if action == "stacks":
                return "stacks: " + ", ".join(STACKS)
            if action == "info":
                if not rest:
                    return "usage: /apps info <name>"
                out = b.info(rest[0])
                return (f"{out.get('name')} ({out.get('stack')}) · "
                        f"{len(out.get('files', []))} files\nrun: {out.get('run')}")
            if action == "build":
                name = " ".join(
                    w for w in rest
                    if not w.startswith("--")).strip()
                def _flag(flag, default=""):
                    if flag in rest:
                        i = rest.index(flag)
                        if i + 1 < len(rest):
                            return rest[i + 1]
                    return default
                if not name or name.startswith("--"):
                    return ("usage: /apps build <name> --stack "
                            "static|flask|fastapi|express|react-vite|cli-python|"
                            "django|nextjs|bot-telegram|go-cli "
                            "[--title T] [--features a,b,c]")
                stack = _flag("--stack", "static")
                feats = [f.strip() for f in _flag("--features", "").split(",")
                         if f.strip()]
                out = b.build({"name": name, "stack": stack,
                               "title": _flag("--title", name),
                               "features": feats,
                               "overwrite": True})
                v = out["validation"]
                return (f"🏗 built {out['app']} ({out['stack']}) → {out['dir']}\n"
                        f"files: {len(out['files'])} · validation: "
                        f"{'OK' if v['ok'] else 'FAILED ' + str(v['failed'])}\n"
                        f"run: {out['run']}")
            if action == "deployed":
                out = b.deployed()
                entries = out.get("deployments") or []
                if not entries:
                    return "no apps currently deployed"
                return (f"deployed ({out['count']}):\n" + "\n".join(
                    f"  {e['app']}  {e['url']}  (proxy pid {e['pid']}, "
                    f"→ :{e.get('backend_port')}, alive={e['alive']})"
                    for e in entries))
            if action == "deploy":
                def _flag(flag, default=""):
                    if flag in rest:
                        i = rest.index(flag)
                        if i + 1 < len(rest):
                            return rest[i + 1]
                    return default
                name = " ".join(
                    w for w in rest
                    if not w.startswith("--")).strip()
                if not name:
                    return ("usage: /apps deploy <name> [--domain d] "
                            "[--path /x] [--port N]")
                out = b.deploy(name, host=_flag("--host", "0.0.0.0"),
                               port=int(_flag("--port", "0") or 0),
                               domain=_flag("--domain"),
                               path=_flag("--path"))
                health = out.get("health") or {}
                text = (f"🌐 deployed {out['app']} → {out['url']}\n"
                        f"  real reverse proxy (pid {out.get('pid')}) "
                        f"→ 127.0.0.1:{out.get('backend_port')}\n"
                        f"  domain: {out.get('domain') or '(none)'}   "
                        f"path: {out.get('path') or '/'}   "
                        f"health={'ok' if health.get('ok') else health.get('error', 'unknown')}")
                if out.get("note"):
                    text += f"\n{out['note']}"
                return text
            if action == "stop_deploy":
                if not rest:
                    return "usage: /apps stop_deploy <name>"
                out = b.stop_deploy(rest[0])
                return (f"stopped proxy for {out['app']} "
                        f"({'stopped' if out['stopped'] else 'already dead'}) "
                        f"was {out.get('url', '')}")
            out = b.list_apps()
            if not out["apps"]:
                return ("no apps yet — /apps build myapp --stack flask "
                        "(stacks: " + ", ".join(STACKS) + ")")
            return (f"apps ({out['count']}):\n" + "\n".join(
                f"  {a['name']}  [{a['stack']}]  →  {a['run']}"
                for a in out["apps"]))
        except Exception as exc:  # noqa: BLE001
            return f"apps error: {exc}"

    def _control_podcast(self, tail: str, chat_key: str = "") -> str:
        """/podcast <query…> [platform] — the podcast pipeline, one call."""
        return self._control_hub(f"podcast {tail}", chat_key=chat_key)

    def _control_record(self, tail: str, chat_key: str = "") -> str:
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "status"
        if verb == "start":
            if len(parts) < 2:
                return "usage: /record start <name>"
            outcome = self.context.tools.call("record_start", name=parts[1])
            if not outcome.ok:
                return f"record failed: {getattr(outcome.error, 'message', outcome.error)}"
            return (f"● recording {parts[1]} — every tool call you send me is captured. "
                    f"/record stop when done; it becomes /macro {parts[1]}.")
        if verb == "stop":
            outcome = self.context.tools.call("record_stop")
            if not outcome.ok:
                return f"record failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            return (f"■ saved macro {v['steps']} steps → call it with "
                    f"/macro {v['saved']} (or ask me to run it)")
        if verb == "step":
            if len(parts) < 2:
                return "usage: /record step <tool> [json args]"
            args = " ".join(parts[2:]) if len(parts) > 2 else ""
            outcome = self.context.tools.call("record_step", tool=parts[1], args=args)
            if not outcome.ok:
                return f"record failed: {getattr(outcome.error, 'message', outcome.error)}"
            return f"step {outcome.value['steps']} added ({parts[1]})"
        if verb == "status":
            outcome = self.context.tools.call("record_status")
            if not outcome.ok:
                return f"record failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            if not v.get("recording"):
                return "not recording — /record start <name>"
            return (f"● recording {v['name']}: {v['steps']} steps "
                    f"({v['seconds']}s, last: {v.get('last')})")
        return "usage: /record start <name> | stop | step <tool> [json] | status"

    def _control_publish(self, tail: str, chat_key: str) -> str:
        parts = (tail or "").split()
        if len(parts) < 3:
            return "usage: /publish <platform> <chat_id> <markdown file> [format]"
        platform, chat_id, path = parts[0], parts[1], parts[2]
        fmt = parts[3].lower() if len(parts) > 3 else "pdf"
        try:
            from ...tools.filesystem import safe_path

            source = safe_path(self.context, path, must_exist=True)
        except Exception as exc:  # noqa: BLE001
            return f"publish failed: cannot read {path!r} ({exc})"
        content = source.read_text(encoding="utf-8", errors="replace")
        title = source.stem.replace("-", " ").replace("_", " ").strip() or "Report"
        outcome = self.context.tools.call(
            "report_publish", content=content, title=title, platform=platform,
            chat_id=chat_id, format=fmt, name=source.stem)
        if not outcome.ok:
            return (f"publish failed: {getattr(outcome.error, 'message', outcome.error)}")
        v = outcome.value
        size = v.get("bytes", 0)
        where = f" → {platform}:{chat_id}" if v.get("sent") else " (not sent)"
        return (f"📄 published {os.path.basename(v['path'])} "
                f"({size // 1024} KB, {v['format']}){where}")
