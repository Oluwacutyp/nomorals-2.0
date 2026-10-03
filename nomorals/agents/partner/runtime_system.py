"""RuntimeSystemMixin: PartnerRuntime command group (system)."""

from __future__ import annotations

import json
import os

class RuntimeSystemMixin:
    """RuntimeSystemMixin for :class:`PartnerRuntime`."""


    def _control_exec(self, tail: str) -> str:
        """/exec <code> | /exec until_green <code|file|project> [lang] | /exec languages."""
        from ...execbox import CodeRunner

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /exec <code> [lang]  |  "
                    "/exec until_green <code|file|project>  |  /exec languages")
        if tail.lower() == "languages":
            box = CodeRunner(self.context)
            langs = box.languages()
            avail = ", ".join(l.name for l in langs.values() if l.available)
            return f"installed: {avail}\nmissing: " \
                   f"{', '.join(l.name for l in langs.values() if not l.available) or '(none)'}"
        if tail.split()[0].lower() == "until_green" and len(tail.split()) > 1:
            target = " ".join(tail.split()[1:])
            try:
                out = CodeRunner(self.context).run_project_until_green(target)
            except Exception as exc:  # noqa: BLE001
                return f"until_green error: {exc}"
            mark = "🟢 GREEN" if out["green"] else "🔴 RED"
            text = (f"{mark} — {out['target']}\n"
                    f"  criterion: {out['criterion']}  "
                    f"({out['rounds']}/{out['max_rounds']} rounds)")
            if out.get("files_changed"):
                text += ("\n  files rewritten by model: "
                         + ", ".join(out["files_changed"][:6]))
            fr = out.get("final_run") or {}
            if fr.get("stderr"):
                text += "\n(stderr tail)\n" + fr["stderr"][-600:]
            elif fr.get("stdout"):
                text += "\n(stdout tail)\n" + fr["stdout"][-400:]
            return text
        code = tail
        lang = ""
        words = tail.split()
        known = set(CodeRunner(self.context).languages())
        if len(words) > 1 and words[-1].lower() in known:
            lang = words[-1].lower()
            code = " ".join(words[:-1])
        try:
            out = CodeRunner(self.context).run(code, lang=lang)
        except Exception as exc:  # noqa: BLE001
            return f"exec error: {exc}"
        mark = "✅" if out["ok"] else "❌"
        text = (f"{mark} [{out['language']}] exit={out['exit_code']} "
                f"{out['seconds']}s" + ("  ⏱ timed out" if out["timed_out"] else ""))
        if out["stdout"]:
            text += "\n" + out["stdout"].rstrip()
        if out["stderr"]:
            text += "\n(stderr)\n" + out["stderr"].rstrip()
        if out["files"]:
            text += "\nwrote: " + ", ".join(f["path"] for f in out["files"][:8])
        return text

    def _control_zip(self, tail: str) -> str:
        """/zip <paths…> --dest x.zip | /zip list|info|extract <archive> | compress <file> [fmt]."""
        from ...archives import Archivist

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /zip <paths…> --dest x.zip [--fmt tar.gz]  |  "
                    "/zip list|info|extract <archive>  |  /zip compress <file> [fmt]")
        tokens = tail.split()
        dest = ""
        fmt = "zip"
        if "--dest" in tokens:
            i = tokens.index("--dest")
            if i + 1 < len(tokens):
                dest = tokens[i + 1]
            tokens = tokens[:i] + tokens[i + 2:]
        if "--fmt" in tokens:
            i = tokens.index("--fmt")
            if i + 1 < len(tokens):
                fmt = tokens[i + 1]
            tokens = tokens[:i] + tokens[i + 2:]
        try:
            a = Archivist(self.context)
            if tokens and tokens[0].lower() in ("list", "info", "extract",
                                                 "digest", "compress") and len(tokens) > 1:
                action, path = tokens[0].lower(), tokens[1]
                if action == "info":
                    out = a.info(path)
                    return (f"{out['path']} → {out['format']} · "
                            f"{out.get('entries', '?')} entries · "
                            f"{out['compressed_bytes']} B"
                            + (f" · {out['note']}" if "note" in out else ""))
                if action == "list":
                    out = a.list(path)
                    return (f"{out['format']}, {out['count']} entries:\n"
                            + "\n".join(f"  {e['name']} ({e['size']} B)"
                                        for e in out["entries"][:30]))
                if action == "extract":
                    out = a.extract(path)
                    return (f"extracted {out['extracted']} file(s) → {out['dest']}"
                            + (f"\nskipped unsafe: {', '.join(out['skipped'][:5])}"
                               if out["skipped"] else ""))
                if action == "digest":
                    from ...tools.filesystem import safe_path

                    rp = safe_path(self.context, path, must_exist=True)
                    if rp.is_dir():
                        out = a.digest_directory(str(rp))
                    else:
                        out = a.digest(str(rp))
                    kg = out.get("kg", {})
                    lines = [f"📚 {out['path']} → {out['format']}: "
                             f"{out['text_files']} text file(s), {out['chars']:,} chars"]
                    lines.append(f"  knowledge graph: +{kg.get('added_nodes', 0)} "
                                 f"nodes, +{kg.get('added_links', 0)} links")
                    lines.append(f"  memory: {'episode stored' if out.get('memory_stored') else 'not stored'}")
                    if out.get("extracted_to"):
                        lines.append(f"  extracted to: {out['extracted_to']}")
                    if out.get("kg_error"):
                        lines.append(f"  kg error: {out['kg_error']}")
                    return "\n".join(lines)
                if action == "compress":
                    f2 = tokens[2] if len(tokens) > 2 else "gz"
                    out = a.compress(path, fmt=f2)
                    return f"compressed → {out['path']} ({out['compressed_bytes']} B)"
            if not dest:
                return ("usage: /zip <paths…> --dest x.zip [--fmt zip|tar.gz]  |  "
                        "/zip digest <archive>")
            out = a.create(tokens, dest, fmt=fmt)
            return (f"📦 created {out['path']} — {out['entries']} entries, "
                    f"{out['compressed_bytes']} B")
        except Exception as exc:  # noqa: BLE001
            return f"zip error: {exc}"

    # ── wave 84: the virtual CPU farm ─────────────────────────────────────────
    def _control_workspace(self, tail: str) -> str:
        from ...workspace import Workspace
        ws = getattr(self.context, "workspace", None)
        if ws is None:
            # Cache on the context so context.close() can shut it down;
            # spawning one per call would leak a thread fleet per message.
            ws = Workspace(self.context)
            try:
                setattr(self.context, "workspace", ws)
            except Exception:  # noqa: BLE001
                pass
        parts = [t for t in (tail or "").split() if t]
        verb = (parts[0].lower() if parts else "status")
        value = parts[1] if len(parts) > 1 else ""
        if verb == "scale" and value.isdigit():
            return f"farm resized to {ws.scale_to(int(value))} vcpu(s) — {ws.summary_line()}"
        if verb == "up":
            return f"farm: {ws.summary_line()}" if ws.scale_up(1) == len(ws.vcpus()) \
                else f"vcpu added — {ws.summary_line()}"
        if verb == "down":
            ws.scale_down(1)
            return f"vcpu retired (drained) — {ws.summary_line()}"
        if verb in {"pause", "resume"} and value:
            vcpu = ws.get(value)
            if vcpu is None:
                return f"no core named {value!r} — {ws.summary_line()}"
            (vcpu.pause() if verb == "pause" else vcpu.resume())
            return f"{value} is now {vcpu.status} — {ws.summary_line()}"
        if verb == "add" and value in {"io", "cpu", "balanced"}:
            v = ws.add_vcpu(kind=value)
            return f"attached {v.name} ({v.kind}) — {ws.summary_line()}"
        if verb == "remove" and value:
            ok = ws.remove_vcpu(value)
            return (f"{value} detached — {ws.summary_line()}" if ok
                    else f"could not remove {value!r} (unknown, or below the "
                         f"farm minimum)")
        st = ws.status()
        prof = st["profile"]
        lines = [f"🖥 {ws.summary_line()}",
                 f"profile {prof['kind']} ({prof['detail']}) — envelope "
                 f"{st['min_vcpus']}-{st['target_vcpus']}-{st['max_vcpus']}, "
                 f"autoscale {'on' if st['autoscale'] else 'off'}"]
        for d in st["details"]:
            flag = "⏸" if d["status"] == "paused" else \
                   "⛔" if d["status"] == "error" else "  "
            lines.append(f"  {flag}{d['id']:8s} [{d['status']:6s}] {d['kind']:8s} "
                         f"load {d['load']:.0%} queue {d['queue']} "
                         f"tasks {d['stats']['tasks_run']}")
        if st["last_scale"]:
            lines.append(f"last scale: {st['last_scale']}")
        lines.append("controls: /workspace scale <n> · up · down · "
                     "pause|resume <vcpu> · add <io|cpu|balanced> · remove <vcpu>")
        return "\n".join(lines)

    def _control_proxy(self, tail: str, chat_key: str = "") -> str:
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "status"
        if verb == "status":
            outcome = self.context.tools.call("proxy_status")
            if not outcome.ok:
                return f"proxy failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            known = ", ".join(v["known"]) or "none"
            raw = "raw sockets routed too" if v.get("raw_sockets_routed") \
                else ("socks needs PySocks (pip install pysocks)"
                      if v["active"].lower().startswith(("socks5", "socks5h"))
                      else "http proxies route urllib traffic too")
            return (f"outbound: {v['active']}\nknown: {known}\n"
                    f"{raw} — all clients covered")
        if verb == "list":
            outcome = self.context.tools.call("proxy_list")
            if not outcome.ok:
                return f"proxy failed: {getattr(outcome.error, 'message', outcome.error)}"
            proxies = outcome.value["proxies"]
            return "proxies: " + (", ".join(proxies) if proxies else "none — /proxy set <url>")
        if verb == "test":
            return self._tool_reply("proxy_test", chat_key, proxy=" ".join(parts[1:]))
        if verb == "set":
            url = " ".join(parts[1:])
            if not url:
                return "usage: /proxy set http://host:port | socks5://host:port"
            return self._tool_reply("proxy_set", chat_key, proxy=url)
        if verb == "clear":
            return self._tool_reply("proxy_clear", chat_key)
        # ── proxy lab: free-proxy discovery + SSH tunnels ──────────────────
        if verb == "scrape":
            outcome = self.context.tools.call("proxy_scrape")
            if not outcome.ok:
                return f"scrape failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            lines = [f"🕸️ scraped {v['total']} unique proxies "
                     f"({v['seconds']}s, {len(v['by_source'])} sources)"]
            for src, n in sorted(v["by_source"].items(), key=lambda kv: -kv[1]):
                lines.append(f"  {src}: {n}")
            for src, err in list(v["errors"].items())[:4]:
                lines.append(f"  {src}: FAILED ({err[:60]})")
            disabled = v.get("disabled_sources") or []
            if disabled:
                lines.append(f"  (retired for 24h, dead too often: "
                             f"{', '.join(disabled[:4])})")
            stats = v.get("source_stats") or {}
            if stats.get("discovered"):
                lines.append(f"  sources: {stats['active']} active "
                             f"({stats['discovered']} learned from the "
                             f"internet)")
            if len("\n".join(lines)) > 900:
                return self._send_long_checked(self._ref_from_key(chat_key).platform,
                                self._ref_from_key(chat_key), "\n".join(lines))
            return "\n".join(lines)
        if verb == "discover":
            seeds = " ".join(parts[1:])
            outcome = self.context.tools.call(
                "proxy_discover", seeds=seeds)
            if not outcome.ok:
                return (f"discover failed: "
                        f"{getattr(outcome.error, 'message', outcome.error)}")
            v = outcome.value
            reg = v.get("registered") or []
            lines = [f"🔎 mined {v.get('endpoints_tested', 0)} endpoint(s) "
                     f"from {len(v.get('seeds') or [])} seed page(s) — "
                     f"registered {len(reg)} new source(s)"]
            for r in reg[:8]:
                lines.append(f"  + {r['name']} ({r['kind']}, "
                             f"{r['found']} proxies)")
            for seed, err in list((v.get("seed_errors") or {}).items())[:3]:
                lines.append(f"  {seed.split('/')[2] if seed.startswith('http') else seed}: "
                             f"FAILED ({err[:60]})")
            if not reg:
                lines.append("  no new working list endpoints found this "
                             "run — the catalog keeps what already works")
            stats = v.get("source_stats") or {}
            if stats:
                lines.append(f"  catalog now: {stats.get('active', 0)} "
                             f"active ({stats.get('discovered', 0)} learned)")
            return "\n".join(lines)
        if verb == "sources":
            outcome = self.context.tools.call("proxy_sources")
            if not outcome.ok:
                return f"sources failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            rows = v.get("sources") or []
            stats = v.get("stats") or {}
            lines = [f"proxy sources: {stats.get('active', len(rows))} active, "
                     f"{stats.get('disabled', 0)} retired "
                     f"({stats.get('discovered', 0)} learned)"]
            for row in rows[:12]:
                flag = "⛔" if row.get("disabled") else "  "
                found = row.get("last_found")
                lines.append(f"  {flag}{row['name']:28s} "
                             f"tests={row.get('tests', 0)} "
                             f"last={'dead' if row.get('fails') else str(found or 0) + ' found'}")
            if len(rows) > 12:
                lines.append(f"  … {len(rows) - 12} more")
            return "\n".join(lines)
        if verb == "refresh":
            return self._tool_reply("proxy_refresh", chat_key)
        if verb == "pool":
            scheme = parts[1] if len(parts) > 1 else ""
            outcome = self.context.tools.call(
                "proxy_pool", action="urls", scheme=scheme, limit="8")
            if not outcome.ok:
                return f"pool failed: {getattr(outcome.error, 'message', outcome.error)}"
            urls = outcome.value["urls"]
            if not urls:
                return "pool empty — /proxy refresh to scrape+test"
            lines = [f"🧱 working proxies ({outcome.value['count']}):"]
            lines += [f"  {u}" for u in urls]
            lines.append("use: /proxy set <url>  (routes all outbound traffic)")
            return "\n".join(lines)
        if verb == "file":
            # /proxy file — send a CLEAN file of the working proxies.
            outcome = self.context.tools.call("proxy_file")
            if not outcome.ok:
                return f"proxy file failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            if not v.get("written"):
                return f"no working proxies right now — {v.get('note', '/proxy refresh first')}"
            chat = self._ref_from_key(chat_key)
            sent = False
            try:
                result = self.gateway.send_file(chat.platform, chat, v["path"],
                                                caption=f"{v['count']} working proxies "
                                                        f"(fastest first)")
                sent = bool(getattr(result, "ok", False))
            except Exception:  # noqa: BLE001 - console adapter
                sent = False
            if sent:
                return (f"🧱 sent the clean file — {v['count']} working proxies "
                        f"(fastest: {v.get('fastest', '?')})")
            return (f"🧱 clean file ready ({v['count']} proxies): {v['path']} "
                    f"\n(plain chat — no file sending on this adapter)")
        if verb == "rotate":
            sub = parts[1].lower() if len(parts) > 1 else "status"
            if sub in {"on", "start"}:
                strategy = parts[2] if len(parts) > 2 else ""
                return self._tool_reply("proxy_rotate", chat_key,
                                        action="start", strategy=strategy)
            if sub in {"off", "stop"}:
                return self._tool_reply("proxy_rotate", chat_key,
                                        action="stop")
            if sub == "next":
                return self._tool_reply("proxy_rotate", chat_key,
                                        action="next")
            return self._tool_reply("proxy_rotate", chat_key, action="status")
        if verb == "ssh":
            sub = parts[1].lower() if len(parts) > 1 else "list"
            if sub == "start":
                # /proxy ssh start <name> <host> <user> [key-path] [port]
                if len(parts) < 5:
                    return ("usage: /proxy ssh start <name> <host> <user> "
                            "[key-path] [port]")
                return self._tool_reply("ssh_socks", chat_key,
                                        action="start",
                                        name=parts[2], host=parts[3],
                                        user=parts[4],
                                        key=parts[5] if len(parts) > 5 else "",
                                        port=parts[6] if len(parts) > 6 else "")
            return self._tool_reply("ssh_socks", chat_key,
                                    action=sub, name=parts[1] if len(parts) > 1 else "")
        return ("usage: /proxy status | list | test [url] | set <url> | clear |"
                " scrape | refresh | pool [scheme] | file | rotate on [strategy]"
                " | ssh start <name> <host> <user>")

    def _control_macro(self, tail: str, chat_key: str) -> str:
        parts = (tail or "").split(None, 1)
        if not parts or parts[0].lower() in {"list", ""}:
            outcome = self.context.tools.call("macro_list")
            if not outcome.ok:
                return f"macro failed: {getattr(outcome.error, 'message', outcome.error)}"
            macros = outcome.value["macros"]
            if not macros:
                return "no macros yet — /record start <name>, do things, /record stop"
            lines = ["macros:"]
            for m in macros:
                tools = ", ".join(m["tools"])
                lines.append(f"  {m['name']} — {m['steps']} steps ({tools}) "
                             f"[runs: {m['runs']}]")
            return "\n".join(lines)
        name = parts[0]
        overrides = parts[1].strip() if len(parts) > 1 else ""
        outcome = self.context.tools.call("macro_run", name=name, overrides=overrides)
        if not outcome.ok:
            return f"macro failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        lines = [f"▶ {v['macro']}: {v['summary']} ({v['seconds']}s)"]
        for step in v["steps"]:
            mark = "✓" if step["ok"] else "✗"
            lines.append(f"  {mark} {step['step']}: {step['tool']} — {step['result'][:120]}")
        chat = self._ref_from_key(chat_key)
        return self._send_long_checked(chat.platform, chat, "\n".join(lines))

    def _control_file(self, tail: str, chat_key: str) -> str:
        parts = (tail or "").split()
        if len(parts) < 3:
            return "usage: /file <platform> <chat_id> <path> [caption]"
        platform, chat_id, path = parts[0], parts[1], parts[2]
        caption = " ".join(parts[3:]) if len(parts) > 3 else ""
        outcome = self.context.tools.call(
            "file_send", platform=platform, chat_id=chat_id,
            path=path, caption=caption)
        if not outcome.ok:
            return f"file send failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        return (f"📎 sent {os.path.basename(v['path'])} "
                f"({v['bytes'] // 1024} KB) → {platform}:{chat_id}")

    # ── database: /db ────────────────────────────────────────────────────────
    def _control_db(self, tail: str) -> str:
        parts = (tail or "").split(None, 1)
        verb = parts[0].lower() if parts else "counts"
        tools = self.context.tools
        if verb == "tables":
            outcome = tools.call("db_tables")
            if not outcome.ok:
                return f"db failed: {getattr(outcome.error, 'message', outcome.error)}"
            value = outcome.value
            lines = [f"tables ({value['count']}):"]
            for table in value["tables"][:100]:
                lines.append(f"  {table['name']} — {table['rows']} rows")
            return "\n".join(lines)
        if verb == "schema":
            if len(parts) < 2:
                return "usage: /db schema <table>"
            outcome = tools.call("db_schema", table=parts[1].strip())
            if not outcome.ok:
                return f"db failed: {getattr(outcome.error, 'message', outcome.error)}"
            value = outcome.value
            lines = [f"{value['table']} ({value['rows']} rows):"]
            for col in value["columns"]:
                flags = "".join(flag for cond, flag in (
                    (col["pk"], "PK"), (col["not_null"], "NN"),
                ) if cond)
                default = f" default={col['default']}" if col.get("default") is not None else ""
                lines.append(f"  {col['name']} {col['type']} {flags}{default}")
            return "\n".join(lines)
        if verb == "query":
            if len(parts) < 2:
                return "usage: /db query <select sql>"
            outcome = tools.call("db_query", sql=parts[1].strip())
            if not outcome.ok:
                return f"db failed: {getattr(outcome.error, 'message', outcome.error)}"
            value = outcome.value
            rows = value["data"][:20]
            if not rows:
                return f"{value['sql']}\n(no rows)"
            cols = value["columns"]
            lines = [" ".join(cols)]
            for row in rows:
                lines.append(" | ".join(str(row.get(c, ""))[:40] for c in cols))
            more = "" if len(rows) == value["rows"] else f"\n… {value['rows'] - len(rows)} more (limit forced ≤ 200)"
            return "\n".join(lines) + more
        if verb == "counts":
            outcome = tools.call("db_counts")
            if not outcome.ok:
                return f"db failed: {getattr(outcome.error, 'message', outcome.error)}"
            lines = ["largest tables:"]
            for table in outcome.value["tables"]:
                lines.append(f"  {table['rows']:>8,}  {table['name']}")
            return "\n".join(lines)
        return "usage: /db tables | schema <table> | query <select sql> | counts"

    # ── api connectors: /api ─────────────────────────────────────────────────
    def _control_api(self, tail: str) -> str:
        parts = (tail or "").split(None, 1)
        if not parts or parts[0].lower() == "list":
            outcome = self.context.tools.call("api_list")
            if not outcome.ok:
                return f"api failed: {getattr(outcome.error, 'message', outcome.error)}"
            lines = ["API connectors:"]
            for conn in outcome.value["connectors"]:
                params = ", ".join(conn["params"]) or "no params"
                lines.append(f"  {conn['name']} — {conn['description'][:70]}\n      params: {params[:120]}")
            return "\n".join(lines)[:4000]
        name = parts[0].strip()
        params = parts[1].strip() if len(parts) > 1 else ""
        outcome = self.context.tools.call("api_call", connector=name, params=params)
        if not outcome.ok:
            return f"api failed: {getattr(outcome.error, 'message', outcome.error)}"
        return f"{name}:\n" + json.dumps(outcome.value, default=str, indent=1)[:4000]

    # ── wave 73: media hub / podcast / CI fix ──────────────────────────────
    def _control_hub(self, tail: str, chat_key: str = "") -> str:
        """/hub song <topic…> [style] | video <query…> [platform] |
        podcast <query…> | status — the one-call media orchestrator."""
        from ...media import MediaHub

        tail = (tail or "").strip()
        tokens = tail.split() if tail else []
        if tokens and tokens[0].lower() in ("song", "video", "podcast",
                                            "status"):
            action = tokens[0].lower()
            rest = " ".join(tokens[1:]).strip()
        else:
            action = "song"
            rest = tail
        try:
            hub = MediaHub(self.context)
            if action == "status":
                out = hub.status()
                backend = out.get("backend") or {}
                bname = (backend.get("name")
                         if isinstance(backend, dict) else str(backend))
                lines = [f"player: {bname or 'unknown'}  "
                         f"playing={out.get('playing')}"]
                if out.get("path"):
                    lines.append(f"  path: {out['path']}")
                if out.get("current"):
                    lines.append(f"  current: {out['current']}")
                return "\n".join(lines)
            if not rest:
                return (f"usage: /hub {action} <topic…>  |  /hub video "
                        "<query…> [platform]  |  /hub podcast <query…>  |  "
                        "/hub status")
            send_to = (chat_key.split(":", 1)[0] if ":" in chat_key else "",
                       chat_key.split(":", 1)[1] if ":" in chat_key else "")
            if action == "song":
                # Smart style detection: only treat the last word as a style
                # if it actually resolves to one (exact, alias, or fuzzy).
                # Otherwise the whole input is the topic ("lucid dream" ->
                # topic="lucid dream", style="pop").
                from ...media.music import resolve_style
                words = rest.split()
                style = "pop"
                topic = rest
                if len(words) > 1:
                    try:
                        resolve_style(words[-1])
                        style = words[-1]
                        topic = " ".join(words[:-1])
                    except Exception:  # noqa: BLE001
                        # Last word isn't a style — whole input is the topic.
                        pass
                elif words:
                    topic = words[0]
                out = hub.run("song", topic=topic, style=style)
                song = out.get("song", {})
                lines = [f"🎵 {song.get('title', topic)}  "
                         f"({song.get('style', style)})"]
                if song.get("midi_path"):
                    lines.append(f"  midi: {song['midi_path']}")
                pb = out.get("playback")
                if pb:
                    lines.append(f"  playback: {pb.get('status', pb)}")
                return "\n".join(lines)
            if action == "video":
                words = rest.split()
                platform = ""
                if len(words) > 1 and words[-1].lower() in (
                        "youtube", "vimeo", "tiktok", "dailymotion",
                        "twitch", "rumble"):
                    platform = words[-1].lower()
                    query = " ".join(words[:-1])
                else:
                    query = rest
                out = hub.run("video", query=query, platform=platform)
                pick = out.get("pick") or {}
                fc = (out.get("found_count")
                      or (out.get("found") or {}).get("count", "?"))
                lines = [f"🎬 found {fc}, picked "
                         f"“{(pick.get('title') or pick.get('url') or query)[:70]}”"]
                if out.get("download", {}).get("path"):
                    lines.append(f"  file: {out['download']['path']}")
                if out.get("playback"):
                    lines.append(f"  playback: "
                                 f"{out['playback'].get('status', out['playback'])}")
                if out.get("error"):
                    lines.append(f"  {out['error']}")
                return "\n".join(lines)
            # podcast — transcript auto-sends to your newest live chat on
            # any connected platform (send_transcript=None = auto)
            out = hub.run("podcast", query=rest, send_transcript=None,
                          send_to=send_to)
            pick = out.get("pick") or {}
            fc = (out.get("found_count")
                  or (out.get("found") or {}).get("count", "?"))
            lines = [f"🎙 {rest}: found {fc}, picked "
                     f"“{(pick.get('title') or pick.get('url') or rest)[:70]}”"]
            if out.get("download", {}).get("path"):
                lines.append(f"  file: {out['download']['path']}")
            if out.get("transcript"):
                lines.append(f"  transcript: {len(out['transcript'])} chars "
                             f"(stt: {out.get('stt_provider', 'n/a')})")
            if out.get("summary"):
                lines.append(f"  summary: {out['summary'][:160]}")
            if out.get("chapters"):
                lines.append(f"  chapters: {len(out['chapters'])}")
                for ch in out["chapters"][:8]:
                    lines.append(f"    [{ch['start']}-{ch['end']}] {ch['title']}")
            if out.get("transcript_path"):
                lines.append(f"  saved: {out['transcript_path']}")
            s = out.get("send")
            if isinstance(s, dict):
                lines.append(f"  transcript → {s.get('platform')}:"
                             f"{s.get('chat')} "
                             f"({'sent' if s.get('ok') else 'FAILED: ' + str(s.get('error'))})")
            elif s:
                lines.append(f"  transcript: {s}")
            if out.get("stt_error"):
                lines.append(f"  stt: {out['stt_error']}")
            if out.get("note"):
                lines.append(f"  note: {out['note']}")
            if out.get("error"):
                lines.append(f"  {out['error']}")
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            return f"hub error: {exc}"

    def _control_data(self, tail: str, chat_key: str) -> str:
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "list"
        if verb == "mine":
            name = parts[1] if len(parts) > 1 else ""
            outcome = self.context.tools.call("train_mine", name=name)
            if not outcome.ok:
                return (f"mine failed: {getattr(outcome.error, 'message', outcome.error)}")
            v = outcome.value
            stats = v.get("stats", {})
            hist = v.get("score_histogram", {})
            text = (f"🧠 mined {v['examples']} training pairs from {stats.get('conversations', 0)} "
                    f"conversations ({stats.get('pairs_seen', 0)} pairs seen, "
                    f"noise {stats.get('noise_dropped', 0)}, dupes {stats.get('duplicates', 0)}, "
                    f"below-score {stats.get('below_score', 0)})\n"
                    f"quality: high {hist.get('high', 0)} / mid {hist.get('mid', 0)} / low {hist.get('low', 0)}\n"
                    f"bundle: {v['dir']}/{v['name']}.* "
                    f"(alpaca + sharegpt + chatml)\n"
                    f"dataset id: {v.get('dataset_id') or '(not registered)'}")
            return self._send_long_checked(self._ref_from_key(chat_key).platform,
                            self._ref_from_key(chat_key), text)
        if verb in {"list", ""}:
            outcome = self.context.tools.call("train_datasets")
            if not outcome.ok:
                return (f"data failed: {getattr(outcome.error, 'message', outcome.error)}")
            v = outcome.value
            stats = v.get("registry_stats", {})
            lines = [f"datasets: {stats.get('datasets', 0)} registered "
                     f"({stats.get('rows', 0)} rows, {stats.get('bytes', 0) // 1024} KB)"]
            for row in v.get("datasets", [])[:12]:
                lines.append(f"  {row['id']} — {row['name']} "
                             f"({row['rows']} rows, {row['bytes'] // 1024} KB)")
            mined = v.get("recently_mined", [])
            if mined:
                lines.append(f"recently mined: {mined[0]['name']} "
                             f"({mined[0]['examples'] if 'examples' in mined[0] else ''})")
            free = v.get("free_catalog", [])
            if free:
                lines.append("free datasets (fetch with /data fetch <name>):")
                for e in free[:8]:
                    lines.append(f"  {e['name']} — {e['kind']} · {e['license']}")
            return "\n".join(lines)
        if verb == "fetch" and len(parts) >= 2:
            ref = parts[1]
            max_rows = parts[2] if len(parts) > 2 else ""
            outcome = self.context.tools.call(
                "dataset_fetch", ref=ref, max_rows=max_rows)
            if not outcome.ok:
                return f"fetch failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            return (f"📦 fetched {v['rows']} rows from {v['source']} "
                    f"({v.get('license', '?')}) → {v['path']}\n"
                    f"registered as {v['name']}-fetched — nm train can use it")
        if verb == "mix" and len(parts) >= 2:
            # /data mix <name1,name2,...> [rows] — the fine-tune bundle
            sources = parts[1]
            rows = parts[2] if len(parts) > 2 else ""
            outcome = self.context.tools.call(
                "train_mix", sources=sources, rows=rows)
            if not outcome.ok:
                return f"mix failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            per_src = ", ".join(f"{n}: {d.get('rows', 0)}"
                                for n, d in v.get("per_source", {}).items())
            text = (f"🧬 persona mix ready — {v['rows']} rows "
                    f"({per_src})\n"
                    f"bundle: {v['outputs']['messages_jsonl']} "
                    f"(+ alpaca/sharegpt/chatml)\n")
            if v.get("colab_script"):
                text += (f"colab script: {v['colab_script']} — upload it + "
                         f"the .jsonl to Colab free and run (auto-resumes "
                         f"after kills)\n")
            if v.get("missing_sources"):
                text += f"not fetched (skipped): {', '.join(v['missing_sources'])}\n"
            text += ("next: /file <platform> <chat> <path> to get the files, "
                     "or nm data on the console")
            return self._send_long_checked(self._ref_from_key(chat_key).platform,
                            self._ref_from_key(chat_key), text)
        return ("usage: /data mine [name] | /data list | /data fetch <name> [rows] "
                "| /data mix <name1,name2> [rows]")

    def _control_fix(self, tail: str) -> str:
        """/fix <code> [lang] [--rounds N] — CI loop until green."""
        from ...execbox import CodeRunner

        tail = (tail or "").strip()
        if not tail:
            return "usage: /fix <code> [lang] [--rounds N]"
        tokens = tail.split()
        rounds = 4
        if "--rounds" in tokens:
            i = tokens.index("--rounds")
            if i + 1 < len(tokens):
                try:
                    rounds = int(tokens[i + 1])
                except ValueError:
                    rounds = 4
            tokens = tokens[:i] + tokens[i + 2:]
        known = set(CodeRunner(self.context).languages())
        lang = ""
        if len(tokens) > 1 and tokens[-1].lower() in known:
            lang = tokens[-1].lower()
            tokens = tokens[:-1]
        code = " ".join(tokens)
        try:
            out = CodeRunner(self.context).run_until_green(
                code, lang=lang, max_rounds=rounds)
        except Exception as exc:  # noqa: BLE001
            return f"fix error: {exc}"
        mark = "🟢 GREEN" if out["green"] else "🔴 RED"
        text = f"{mark} after {out['rounds']}/{out['max_rounds']} rounds"
        for h in out["history"]:
            line = (f"  {'✓' if h['ok'] else '✗'} round {h['round']}: "
                    f"exit={h['exit_code']} in {h['seconds']}s")
            if not h["ok"] and h.get("fix_note"):
                line += f"  → {h['fix_note']}"
            if not h["ok"] and not h.get("fix_note") and h.get("stderr_tail"):
                last = h["stderr_tail"].strip().splitlines()
                if last:
                    line += f"\n      {last[-1][:120]}"
            text += line + "\n"
        fr = out["final_run"] or {}
        if fr.get("stdout"):
            text += "--- stdout ---\n" + fr["stdout"].rstrip()
        if fr.get("stderr"):
            text += "--- stderr ---\n" + fr["stderr"].rstrip()
        if out["note"]:
            text += f"\n{out['note']}"
        return text

    # ── feature flags ────────────────────────────────────────────────────────
    def _control_features(self, tail: str) -> str:
        from ..features import FeatureRegistry

        reg = FeatureRegistry(self.context.db)
        if not tail:
            lines = ["features — toggle with /features <name> on|off:"]
            for f in reg.list():
                lines.append(f"  {'✓' if f['on'] else '✗'} {f['name']} — {f['description']}")
            return "\n".join(lines)
        parts = tail.split()
        if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
            names = ", ".join(f["name"] for f in reg.list())
            return f"usage: /features <name> on|off — names: {names}"
        name = parts[0].strip().lower()
        on = parts[1].lower() == "on"
        if not reg.set(name, on):
            names = ", ".join(f["name"] for f in reg.list())
            return f"unknown feature {name!r}. available: {names}"
        return f"{name}: {'on' if on else 'off'}"
