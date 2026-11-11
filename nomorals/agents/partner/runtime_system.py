"""RuntimeSystemMixin: PartnerRuntime command group (system)."""

from __future__ import annotations

import json
import os
import time
from typing import Any
from .failures import record_failure


class _ConnectorsChatError(Exception):
    """User-facing /connectors failure (never a traceback in chat)."""


def _render_runs_stop(row: dict[str, Any], result: dict[str, Any]) -> str:
    """Render one facade stop for /runs (public-surface fix 5).

    The facade's doctrine, in the owner's words: the organ's native
    mechanism ran first; the receipt (when one could be minted)
    records its answer. A stop that landed natively but could not
    be receipted says exactly that — the receipt never pretends to
    be the cancel, and its absence never hides one.
    """
    label = row["label"]
    receipt = result.get("receipt")
    receipt_error = result.get("receipt_error") or ""
    word = result.get("native_state") or ""
    if receipt is not None:
        state = receipt.get("receipt_state") or ""
        if state == "confirmed":
            return (f"⏹ stopped — {label}: the cancel landed and "
                    "stands on the organ's own record. Receipt: "
                    "confirmed — it rides the ticket now (the "
                    "ticket's cancel field shows it).")
        if state == "requested":
            return (f"⏹ stop requested — {label}: the organ took "
                    "the request; the work's own record carries "
                    "the final word. Receipt: requested.")
        if state == "not_found":
            return (f"{label}: nothing to stop — the organ has "
                    "no live record of it (already gone or never "
                    "started). Receipt: not_found.")
        if state == "unhonored":
            return (f"{label}: a stop was requested before, but "
                    "the work ended on its own terms anyway. "
                    "Receipt: unhonored.")
        return (f"{label}: the stop was NOT honored (receipt: "
                f"{state or 'refused'}) — "
                f"{receipt.get('detail') or 'the organ declined the cancel'}")
    landed = (word == "cancelled" or word.endswith("->cancelled")
              or word in ("cancelled_queued", "cancelled_running",
                          "requested"))
    if landed:
        return (f"⏹ stopped — {label}: the cancel landed natively, "
                f"but no receipt could be minted: {receipt_error}")
    return (f"{label}: nothing was cancelled — "
            f"{receipt_error or 'the facade returned no answer'}")


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
            except Exception as exc:  # noqa: BLE001
                record_failure("runtime_system._control_workspace", exc)
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

    def _proxy_home(self) -> str:
        """The Devon home, resolved exactly as the ProxyLab
        resolves it — so the surface reads the lab's own stores."""
        settings = getattr(self.context, "settings", None)
        if settings is not None:
            return str(getattr(settings, "home", "~/.nomorals"))
        return "~/.nomorals"

    def _proxy_activation_lines(self) -> list[str]:
        """Doc 17 P4 — the activation truth block for /proxy
        status, read from P1's durable record (never from logs):
        epoch, last merge, registry size, corrupt-recovery note.
        The states stay distinguishable: never-activated differs
        from a failed attempt (the lab's failure ledger is the
        record of attempts that never completed a run) and from a
        current record. Fail-soft: any read error drops the block
        and is ledger-recorded."""
        try:
            from ...tools.proxy_sources import activation_state_for_home
            from ...tools.proxylab_failures import recent_failures

            record = activation_state_for_home(self._proxy_home())
            failures = [f for f in recent_failures(50)
                        if f.site == "proxylab.activate_sources"]
            latest_fail = failures[0] if failures else None

            def _fail_line() -> str:
                when = time.strftime("%Y-%m-%d %H:%M",
                                     time.localtime(latest_fail.ts))
                return (f"  ⚠ the latest activation attempt failed: "
                        f"{latest_fail.message} ({when})")

            if not record or not int(record.get("epoch") or 0):
                lines = [
                    "🧬 catalogue activation: not yet activated on "
                    "this host — the next /proxy refresh merges the "
                    "verified catalogue (or run: nm proxy sources "
                    "sync)"]
                if latest_fail is not None:
                    lines.append(_fail_line())
                return lines
            last = record.get("last_run") or {}
            head = (f"🧬 catalogue activation: epoch "
                    f"{int(record.get('epoch') or 0)}")
            if last.get("registry_after") is not None:
                head += (f" · registry holds "
                         f"{int(last.get('registry_after') or 0)} "
                         f"sources")
            lines = [head]
            if last:
                when = (time.strftime("%Y-%m-%d %H:%M",
                                      time.localtime(
                                          float(last.get("at"))))
                        if last.get("at") else "never")
                lines.append(
                    f"  last merge {when}: "
                    f"+{len(last.get('added') or [])} added · "
                    f"{int(last.get('already_present') or 0)} already "
                    f"present · "
                    f"{len(last.get('excluded_unverified') or [])} "
                    f"excluded (unverified / no URL)")
                if str(last.get("status") or "") == "empty":
                    lines.append(
                        "  ⚠ that merge left the registry EMPTY — "
                        "no sources to scrape (a named stall, not a "
                        "silent no-op)")
                recovery = last.get("corrupt_recovery") or {}
                if recovery:
                    lines.append(
                        "  ♻ a corrupt registry was recovered on the "
                        "way in — the original bytes are preserved "
                        f"as {recovery.get('backup', '?')}")
            if latest_fail is not None and float(latest_fail.ts) > \
                    float(last.get("at") or 0):
                lines.append(_fail_line())
            return lines
        except Exception as exc:  # noqa: BLE001 - block never breaks status
            record_failure("runtime_system._proxy_activation_lines", exc)
            return []

    def _proxy_receipt_lines(self, limit: int = 5) -> list[str]:
        """Doc 17 P4 — the receipts truth block for /proxy status:
        the newest route decisions from P2's journal (outcomes
        folded), bounded to the surface's message discipline —
        a fixed short list plus a pointer at the full journal,
        never a paged dump. Hosts, consumers, reasons and outcomes
        only: no internal decision ids. Fail-soft like the
        activation block."""
        try:
            from ...core.proxy_routing import RouteReceiptJournal

            journal = RouteReceiptJournal(os.path.join(
                os.path.expanduser(self._proxy_home()), "proxies",
                "route_receipts.jsonl"))
            rows = journal.recent(500)
            if not rows:
                return ["🧾 route receipts: none recorded yet — "
                        "decisions land here as traffic routes"]
            shown = list(reversed(rows[-limit:]))
            lines = [f"🧾 route receipts — latest {len(shown)} of "
                     f"{len(rows)} decision(s):"]
            for row in shown:
                route = str(row.get("route") or "")
                label = "NO ROUTE" if route == "none" else route
                host = str(row.get("host") or "")[:48] or "—"
                when = (time.strftime("%Y-%m-%d %H:%M",
                                      time.localtime(
                                          float(row.get("at"))))
                        if row.get("at") else "?")
                line = (f"  {when} {row.get('consumer') or ''} → "
                        f"{label} {host} · "
                        f"{row.get('reason_code') or ''}")
                if row.get("outcome"):
                    line += f" · outcome: {row['outcome']}"
                lines.append(line)
            if len(rows) > len(shown):
                lines.append(
                    f"  … {len(rows) - len(shown)} earlier decision(s) "
                    f"in the journal — nm proxy receipts")
            return lines
        except Exception as exc:  # noqa: BLE001 - block never breaks status
            record_failure("runtime_system._proxy_receipt_lines", exc)
            return []

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
            lines = [f"outbound: {v['active']}", f"known: {known}",
                     f"{raw} — all clients covered"]
            # doc 17 P4: the two truth blocks the surface was
            # missing — catalogue activation state (P1's record)
            # and the latest route receipts (P2's journal). Each
            # block is fail-soft: a broken read drops the block,
            # never the status reply.
            lines += self._proxy_activation_lines()
            lines += self._proxy_receipt_lines()
            return "\n".join(lines)
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
        if verb == "health":
            # /proxy health — the dashboard, formatted for chat.
            outcome = self.context.tools.call("proxy_health")
            if not outcome.ok:
                return f"health failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            p = v.get("pool", {})
            lines = ["🏥 <b>proxy pool health</b>"]
            lines.append(
                f"  routable: <b>{p.get('routable', 0)}</b> "
                f"(fresh 24h: {p.get('fresh_24h', 0)}, "
                f"alive: {p.get('alive_total', 0)})")
            if p.get("in_backoff"):
                lines.append(f"  ⏳ in failure backoff: {p['in_backoff']}")
            if p.get("dead"):
                lines.append(f"  ☠ dead records: {p['dead']}")
            by_scheme = p.get("by_scheme") or {}
            if by_scheme:
                lines.append("  by protocol: " + ", ".join(
                    f"{k}={n}" for k, n in by_scheme.items()))
            by_anon = p.get("by_anonymity") or {}
            if by_anon:
                lines.append("  by anonymity: " + ", ".join(
                    f"{k}={n}" for k, n in by_anon.items()))
            top = p.get("top_countries") or {}
            if top:
                lines.append("  top countries: " + ", ".join(
                    f"{k}={n}" for k, n in list(top.items())[:8]))
            if p.get("ng_or_eu"):
                lines.append(f"  🇳🇬/🇪🇺 presence: {p['ng_or_eu']} proxies")
            if p.get("avg_latency_ms"):
                lines.append(
                    f"  avg latency (decayed): {p['avg_latency_ms']}ms")
            lines.append(f"  last refresh: {p.get('last_refresh', 'never')}")
            rot = v.get("rotation", {})
            lines.append(
                f"  rotation: {'on (' + rot['strategy'] + ')' if rot.get('enabled') else 'off'}"
                + (f", cooling {rot['cooling']}" if rot.get("cooling") else "")
                + (f", direct fallbacks {rot['direct_fallbacks']}"
                   if rot.get("direct_fallbacks") else ""))
            lines.append(f"  domain affinity: {v.get('affinity_bindings', 0)} "
                         f"pinned site(s)")
            sched = v.get("schedule") or {}
            lines.append("  auto-refresh: " + (
                f"{sched['spec']} (active)" if sched.get("active")
                else "off — /proxy schedule on to re-check automatically"))
            if v.get("guidance"):
                lines.append("")
                lines.append("⚠ " + v["guidance"])
            return "\n".join(lines)
        if verb == "pick":
            # /proxy pick [udp] [country=XX] [anon=elite] [target=url]
            kwargs: dict[str, str] = {}
            for tok in parts[1:]:
                low = tok.lower()
                if low in {"udp", "socks5"}:
                    kwargs["needs_udp"] = "true"
                elif low.startswith("country="):
                    kwargs["country"] = tok.split("=", 1)[1]
                elif low.startswith("anon="):
                    kwargs["min_anonymity"] = tok.split("=", 1)[1]
                elif low.startswith("target="):
                    kwargs["target"] = tok.split("=", 1)[1]
            outcome = self.context.tools.call("proxy_pick", **kwargs)
            if not outcome.ok:
                return f"pick failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            if v.get("proxy"):
                return (f"🎯 {v['proxy']}\n  {v.get('why', '')}\n"
                        f"  use: /proxy set {v['proxy']}")
            return "⚠ " + v.get("note", "no proxy available")
        if verb == "affinity":
            # /proxy affinity [bind <domain> <url> [ttl_h] | lookup <domain>
            #                | release <domain> | list]
            sub = parts[1].lower() if len(parts) > 1 else "list"
            if sub == "bind" and len(parts) >= 4:
                return self._tool_reply(
                    "proxy_affinity", chat_key, action="bind",
                    domain=parts[2], proxy_url=parts[3],
                    ttl_hours=parts[4] if len(parts) > 4 else "")
            if sub == "lookup" and len(parts) >= 3:
                return self._tool_reply(
                    "proxy_affinity", chat_key, action="lookup",
                    domain=parts[2])
            if sub == "release" and len(parts) >= 3:
                return self._tool_reply(
                    "proxy_affinity", chat_key, action="release",
                    domain=parts[2])
            if sub == "list":
                outcome = self.context.tools.call("proxy_affinity",
                                                  action="list")
                if not outcome.ok:
                    return f"affinity failed: {getattr(outcome.error, 'message', outcome.error)}"
                rows = outcome.value.get("bindings") or []
                if not rows:
                    return ("no pinned sites — /proxy affinity bind "
                            "<domain> <proxy_url>  (or /proxy pick "
                            "target=<url> pins automatically)")
                lines = [f"📌 pinned sites ({len(rows)}):"]
                for r in rows[:12]:
                    lines.append(f"  {r['domain']} → {r['proxy_url']} "
                                 f"({r['age_hours']}h of {r['ttl_hours']}h)")
                return "\n".join(lines)
            return ("usage: /proxy affinity bind <domain> <proxy_url> [ttl_h]"
                    " | lookup <domain> | release <domain> | list")
        if verb == "schedule":
            # /proxy schedule on [minutes] | off — auto-refresh cadence
            sub = parts[1].lower() if len(parts) > 1 else "status"
            if sub in {"on", "start"}:
                minutes = parts[2] if len(parts) > 2 else ""
                return self._tool_reply("proxy_schedule", chat_key,
                                        enabled="true",
                                        every_minutes=minutes)
            if sub in {"off", "stop"}:
                return self._tool_reply("proxy_schedule", chat_key,
                                        enabled="false")
            return self._tool_reply("proxy_schedule", chat_key,
                                    enabled="true")
        if verb == "pool":
            scheme = parts[1] if len(parts) > 1 else ""
            outcome = self.context.tools.call(
                "proxy_pool", action="urls", scheme=scheme, limit="8")
            if not outcome.ok:
                return f"pool failed: {getattr(outcome.error, 'message', outcome.error)}"
            urls = outcome.value["urls"]
            if not urls:
                from ...social.chat.tgbot_buttons import actionable

                return actionable(
                    "pool empty.",
                    [[("🔄 Refresh pool", "proxy refresh")]])
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
            except Exception as exc:  # noqa: BLE001 - console adapter
                record_failure("runtime_system._control_proxy", exc)
                sent = False
            if sent:
                # The result line is the content; "sent the clean
                # file" was narration about the delivery (owner
                # no-meta-narration rule, 2026-11-01) — the file
                # itself already landed with this as its caption.
                return (f"🧱 {v['count']} working proxies "
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
                " scrape | refresh | health | pool [scheme] | pick [udp] "
                "[country=XX] [anon=elite] [target=url] | file | affinity "
                "bind|lookup|release|list | rotate on [strategy] | schedule "
                "on [min] | ssh start <name> <host> <user>")

    def _control_toolhub(self, tail: str, chat_key: str = "") -> Any:
        """Tool hub: /toolhub | /toolhub proxy [run] | /toolhub bin [digits] | /toolhub help.

        (Named toolhub because /hub is already the media hub — see
        control.py. The card itself still says "Devon Hub".)

        Buttons: on Telegram the adapter derives each screen's
        keyboard from the reply text (header markers) / the toolhub
        origin. On WhatsApp there is no derivation, so the handler
        returns the screen's keyboard explicitly (ActionableMessage
        via :meth:`_hub_wrap`) — register #27: the card claimed
        buttons that were silently dropped. Gating is the existing
        grant: ``toolhub`` is not a public kind, so non-owners are
        denied before dispatch.
        """
        from ...social.chat.hub import (bin_usage_text, format_bin_result,
                                        hub_help_text, hub_text,
                                        keyboard_for_screen, lookup_bin,
                                        proxy_intro_text)

        parts = (tail or "").split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""
        if verb == "help":
            return self._hub_wrap(chat_key, hub_help_text(), "help")
        if verb == "bin":
            if not rest:
                return self._hub_wrap(chat_key, bin_usage_text(), "bin")
            # The result screen carries the bin screen's keyboard
            # too (Back / Hub) — an unwrapped result stranded the
            # WhatsApp + userbot surfaces with no way back.
            return self._hub_wrap(
                chat_key, format_bin_result(lookup_bin(rest)), "bin")
        if verb == "proxy":
            # Stack: /toolhub proxy IS the tool screen (intro + Refresh +
            # Back); only "run" executes the flow.
            if not rest:
                return self._hub_wrap(
                    chat_key, proxy_intro_text(), "proxy")
            if rest == "run":
                return self._hub_proxy_flow(chat_key)
            from ...social.chat.tgbot_buttons import actionable

            return actionable(
                "unknown proxy action.",
                [[("🧰 Proxy screen", "toolhub proxy")],
                 [("🔄 Refresh the pool", "toolhub proxy run")]])
        if verb:
            return ("usage: /toolhub | /toolhub proxy [run] | /toolhub bin <digits> "
                    "| /toolhub help")
        return self._hub_wrap(
            chat_key, hub_text(proxy_line=self._hub_proxy_line()), "hub",
            keyboard=keyboard_for_screen("hub"))

    def _hub_wrap(self, chat_key: str, text: str, screen: str,
                  keyboard: Any = None) -> Any:
        """Attach a hub screen's buttons where the surface needs them.

        Register #27 (2026-11-05): the hub copy says "tap a button
        below" — on Telegram the bot adapter derives the keyboard
        from the text, but WhatsApp has no derivation, so the reply
        went out buttonless while claiming buttons. On WhatsApp the
        handler returns an ActionableMessage carrying the screen's
        real keyboard (the bridges render it natively); every other
        surface keeps the plain-string derivation path. If the
        platform cannot take buttons at send time,
        ``_send_control_reply``'s honesty rule rewrites the copy to
        a numbered fallback — the claim is never left standing
        without the buttons.

        AA-77 (2026-11-05): the Telegram USERBOT surface has no
        derivation either — it now renders real inline keyboards
        (TelegramAdapter._inline_markup + its CallbackQuery
        handler), so it takes the authored keyboard here exactly
        like WhatsApp.  The Bot API surface (telegram-bot) keeps
        the derivation path: it derives this same keyboard from
        the origin command.
        """
        platform = (chat_key or "").partition(":")[0].strip().lower()
        if platform not in ("whatsapp", "telegram"):
            return text
        from ...social.chat.hub import keyboard_for_screen
        from ...social.chat.tgbot_buttons import actionable
        # Hub screens are a live session (presentation
        # architecture): on the surfaces that take the authored
        # keyboard here, navigation renders edit the ONE screen
        # message in place — the same live-edit the Bot API
        # surface's nav-edit seam gives its hub. Continuation
        # renders belong in the recorded screen message.
        return actionable(text, keyboard or keyboard_for_screen(screen),
                          session="hub", session_edit=True)

    def _hub_home(self) -> Any:
        """Devon's home dir (cooldown store lives under it)."""
        from pathlib import Path

        home = getattr(getattr(self.context, "settings", None),
                       "home", "") or "~/.nomorals"
        return Path(os.path.expanduser(home))

    def _hub_proxy_line(self) -> str | None:
        """Live pool state for the hub card (best-effort; None →
        the card says the pool is unknown instead of a stale
        number)."""
        try:
            from ...tools.proxylab import ProxyLab

            st = ProxyLab(self.context).status()
            total = int(st.get("working_total", 0))
            fresh = int(st.get("fresh_24h", 0))
            if total <= 0:
                return "empty — tap Proxy to refresh"
            return f"{total} working ({fresh} fresh <24h)"
        except Exception as exc:  # noqa: BLE001 - best-effort card state
            record_failure("runtime_system._hub_proxy_line", exc)
            return None

    def _hub_proxy_flow(self, chat_key: str) -> str:
        """Proxy screen: cooldown → live progress → file → summary.

        Cooldown is checked FIRST (no wasted scrape on a cooled-down
        user). Progress edits are best-effort: if the progress
        message or adapter edit fails, the flow continues and the
        summary still returns — progress never blocks results.
        """
        from ...social.chat.hub import (PROXY_COOLDOWN_S, CooldownStore,
                                        proxy_intro_text)
        from ...social.chat.progress import ProgressReporter
        from ...tools.proxylab import ProxyLab

        store = CooldownStore(self._hub_home() / "hub" / "cooldowns.json")
        left = store.remaining(chat_key, "proxy", PROXY_COOLDOWN_S)
        if left > 0:
            return ("🌐 Proxy lab\n\n"
                    f"⏳ cooling down — try again in "
                    f"{CooldownStore.remaining_text(left)}.\n"
                    "(one generation per hour per chat)")
        store.mark(chat_key, "proxy")

        chat = self._ref_from_key(chat_key)
        progress: ProgressReporter | None = None
        try:
            sent = self.gateway.send(chat.platform, chat,
                                     proxy_intro_text())
            message_id = getattr(sent, "message_id", None)
            if message_id:
                adapter = self.gateway._adapter_for(chat.platform)

                def _emit(text: str, _a=adapter, _c=chat,
                          _m=message_id) -> None:
                    _a.edit_progress(_c, _m, text)

                # Titled → the status grammar (design absorption
                # #1): the trace folds into one ✓/✗ line at the
                # end; steps below are bare step labels, the
                # reporter owns the title + glyph + elapsed.
                progress = ProgressReporter(
                    _emit, title="🌐 Proxy lab", tick_s=5.0)
        except Exception as exc:  # noqa: BLE001 - progress best-effort
            record_failure("runtime_system._hub_proxy_flow", exc)
            progress = None

        lab = ProxyLab(self.context)
        try:
            if progress is not None:
                progress.milestone("scraping free lists…")

            def _on_source(name: str, added: int, err: str) -> None:
                if progress is None:
                    return
                if err:
                    progress.milestone(
                        f"{name} failed ({err[:60]})")
                else:
                    progress.milestone(
                        f"scraped {name}: {added} candidates")

            def _on_test(done: int, total: int, _p: Any) -> None:
                if progress is not None:
                    progress.count(done, total, "testing")

            report = lab.refresh(on_scrape_source=_on_source,
                                 on_test_progress=_on_test)
        except Exception as exc:  # noqa: BLE001 - named, not swallowed
            if progress is not None:
                progress.fail(f"failed: {exc}")
            return f"🌐 Proxy lab — refresh failed: {exc}"

        file_out = lab.active_file()
        if not file_out.get("written"):
            if progress is not None:
                progress.finish("no working proxies")
            return ("🌐 Proxy lab — no working proxies right now\n"
                    f"({file_out.get('note', 'refresh found nothing alive')})\n"
                    "Free proxies rot fast — try again later.")
        sent = False
        try:
            result = self.gateway.send_file(
                chat.platform, chat, file_out["path"],
                caption=f"{file_out['count']} working proxies "
                        "(fastest first)")
            sent = bool(getattr(result, "ok", False))
        except Exception as exc:  # noqa: BLE001 - console adapter
            record_failure("runtime_system._hub_proxy_flow", exc)
            sent = False
        if progress is not None:
            progress.finish("done")

        schemes = ", ".join(
            f"{s}×{n}" for s, n in
            sorted((report.get("by_scheme") or {}).items()))
        countries: list[str] = []
        for item in report.get("best") or []:
            code = item.get("country")
            if code and code not in countries:
                countries.append(code)
            if len(countries) >= 5:
                break
        lines = [
            f"🌐 Proxy lab — {file_out['count']} working proxies",
            f"tested: {report.get('tested', '?')} candidates",
        ]
        if schemes:
            lines.append(f"protocols: {schemes}")
        if countries:
            lines.append(f"countries: {', '.join(countries)}")
        lines.append("Free proxies rot fast — expect tens of minutes "
                     "of life; refresh when they die.")
        if sent:
            lines.append("File sent above ☝️")
        else:
            lines.append(f"file: {file_out['path']} "
                         "(plain chat — no file sending on this adapter)")
        return "\n".join(lines)

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

    # ── artifacts: /artifact ─────────────────────────────────────────────────
    def _artifact_store(self) -> Any:
        """ArtifactStore over the context db, or None when no db is attached.

        Same construction the missions runner uses: the context's database
        plus a BlobStore in a ``blobs`` directory beside the db file.
        The documents engine's body-text extractor is wired in so the
        store's content search indexes document-format bodies too
        (the store is L2 and cannot import the engine itself).
        """
        from pathlib import Path

        from ...documents import artifact_body_text
        from ...storage.artifacts import ArtifactStore
        from ...storage.blob import BlobStore

        db = getattr(self.context, "db", None)
        if db is None:
            return None
        db_path = getattr(db, "path", None)
        blob_dir = (Path(db_path).parent / "blobs" if db_path
                    else Path("data/blobs"))
        return ArtifactStore(db, BlobStore(db, blob_dir),
                             text_extractor=artifact_body_text)

    @staticmethod
    def _resolve_artifact(store: Any, ref: str) -> Any:
        """Resolve an artifact id, an ``artifact://`` / ``artm://`` URI,
        or an alias name to the artifact (None when nothing matches)."""
        ref = (ref or "").strip()
        if not ref:
            return None
        if ref.lower().startswith("artm://"):
            ref = "artifact://" + ref[len("artm://"):]
        art = store.resolve(ref)
        if art is None:
            art = store.resolve_alias(ref)
        return art

    def _delivery_record_store(self) -> Any:
        """The delivery record registry over the context db (doc 15
        transfer/delivery records), or None when no db is attached."""
        db = getattr(self.context, "db", None)
        if db is None:
            return None
        from ...storage.delivery_records import DeliveryRecordStore

        return DeliveryRecordStore(db)

    def _begin_artifact_transfer(self, art: Any, *, kind: str,
                                 chat: Any) -> Any:
        """Journal an artifact hand-off BEFORE it is attempted
        (doc 15 delivery records: a begun transfer is a queryable
        record, so a send whose outcome is never recorded is a
        question the steward can ask). Fail-open: a registry failure
        is recorded and the send goes ahead unrecorded — recording
        never blocks a delivery. Returns the entry, or None."""
        try:
            records = self._delivery_record_store()
            if records is None:
                return None
            from ...storage.delivery_records import artifact_label

            return records.begin_transfer(
                artifact_id=art.id,
                artifact_label=artifact_label(art), kind=kind,
                from_surface="artifact store",
                platform=str(getattr(chat, "platform", "") or ""),
                chat_key=str(getattr(chat, "key", "") or ""))
        except Exception as exc:  # noqa: BLE001 - recording fails open
            record_failure("runtime_system._begin_artifact_transfer",
                           exc)
            return None

    def _complete_artifact_transfer(self, entry: Any, *, ok: bool,
                                    send_ref: str = "",
                                    error: str = "") -> None:
        """Land a begun transfer's outcome — on success the store
        mints the delivery receipt the steward cites. Fail-open
        like the begin: the send already happened either way."""
        if entry is None:
            return
        try:
            records = self._delivery_record_store()
            if records is None:
                return
            records.complete_transfer(entry.id, ok=ok,
                                      send_ref=send_ref, error=error)
        except Exception as exc:  # noqa: BLE001 - recording fails open
            record_failure(
                "runtime_system._complete_artifact_transfer", exc)

    def _artifact_unknown(self, store: Any, ref: str) -> str:
        """Honest unknown-ref reply: name the miss, show what exists."""
        lines = [f"unknown artifact {ref!r} — no id, URI, or alias matches."]
        recent = store.search(limit=5)
        if recent:
            lines.append("recent artifacts:")
            for art in recent:
                lines.append(f"  {art.id} — {art.type} ({art.size} bytes)")
        aliases = store.aliases()
        if aliases:
            lines.append("aliases: " + ", ".join(sorted(aliases)))
        if not recent and not aliases:
            lines.append("the store is empty — nothing has been saved yet.")
        return "\n".join(lines)

    def _artifact_export_path(self, store: Any, art: Any) -> Any:
        """Where a send materializes the artifact's bytes: a file named
        from its metadata (filename/name/title) or ``<type>-<id>``, with
        an extension from its mime, under ``artifact_exports`` beside
        the database (the file_send tool needs a real path)."""
        import mimetypes
        import re
        from pathlib import Path

        name = ""
        for key in ("filename", "name", "title"):
            value = (art.metadata or {}).get(key)
            if isinstance(value, str) and value.strip():
                name = value.strip()
                break
        if not name:
            name = f"{art.type}-{art.id}"
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or art.id
        if "." not in name:
            name += mimetypes.guess_extension(art.mime or "") or ".bin"
        db = getattr(self.context, "db", None)
        db_path = getattr(db, "path", None)
        base = Path(db_path).parent if db_path else Path("data")
        return base / "artifact_exports" / name

    #: Send-time conversion formats for /artifact send … as <fmt>:
    #: name → (extension, mime).  Aliases (md/txt) normalize to the
    #: canonical name in _ARTIFACT_CONVERT_ALIASES.
    _ARTIFACT_CONVERT_FORMATS = {
        "markdown": (".md", "text/markdown"),
        "text": (".txt", "text/plain"),
        "html": (".html", "text/html"),
        "pdf": (".pdf", "application/pdf"),
        "json": (".json", "application/json"),
        "csv": (".csv", "text/csv"),
        "docx": (".docx", "application/vnd.openxmlformats-officedocument"
                 ".wordprocessingml.document"),
        "epub": (".epub", "application/epub+zip"),
        "xlsx": (".xlsx", "application/vnd.openxmlformats-officedocument"
                 ".spreadsheetml.sheet"),
        "pptx": (".pptx", "application/vnd.openxmlformats-officedocument"
                 ".presentationml.presentation"),
    }
    _ARTIFACT_CONVERT_ALIASES = {"md": "markdown", "txt": "text"}

    def _artifact_convert(self, art: Any, data: bytes,
                          fmt_raw: str) -> tuple[bytes, str, str]:
        """Convert an artifact body through the documents engine.

        Returns (converted bytes, canonical format, extension).  The
        stored artifact is never touched — this is a derived send.
        Raises ValueError with a user-facing message for an unknown
        format, an unparseable body, or an engine refusal (e.g. csv
        from a document with no table, docx without python-docx).
        """
        from ...documents import (
            DocumentError,
            parse_bytes,
            to_csv,
            to_docx,
            to_epub,
            to_html,
            to_json,
            to_markdown,
            to_pdf,
            to_pptx,
            to_text,
            to_xlsx,
        )

        fmt = self._ARTIFACT_CONVERT_ALIASES.get(fmt_raw, fmt_raw)
        if fmt not in self._ARTIFACT_CONVERT_FORMATS:
            supported = ", ".join(self._ARTIFACT_CONVERT_FORMATS)
            raise ValueError(
                f"unsupported format {fmt_raw!r} — supported: {supported} "
                "(aliases: md, txt). Nothing was sent.")
        filename = ""
        for key in ("filename", "name", "title"):
            value = (art.metadata or {}).get(key)
            if isinstance(value, str) and value.strip():
                filename = value.strip()
                break
        try:
            doc = parse_bytes(data, filename=filename, mime=art.mime)
        except DocumentError as exc:
            raise ValueError(
                f"couldn't convert artifact {art.id}: its body isn't a "
                f"readable document ({exc}) — nothing was sent.") from exc
        renderers = {
            "markdown": lambda: to_markdown(doc).encode("utf-8"),
            "text": lambda: to_text(doc).encode("utf-8"),
            "html": lambda: to_html(doc).encode("utf-8"),
            "pdf": lambda: to_pdf(doc),
            "json": lambda: to_json(doc).encode("utf-8"),
            "csv": lambda: to_csv(doc).encode("utf-8"),
            "docx": lambda: to_docx(doc),
            "epub": lambda: to_epub(doc),
            "xlsx": lambda: to_xlsx(doc),
            "pptx": lambda: to_pptx(doc),
        }
        try:
            converted = renderers[fmt]()
        except DocumentError as exc:
            raise ValueError(
                f"couldn't convert artifact {art.id} to {fmt}: {exc} "
                "— nothing was sent.") from exc
        ext = self._ARTIFACT_CONVERT_FORMATS[fmt][0]
        return converted, fmt, ext

    def _control_artifact(self, tail: str, chat_key: str = "") -> str:
        """Saved artifacts: list, inspect, search, send, or export.

        /artifact list                 — most recent artifacts + aliases
        /artifact info <ref>           — metadata card for one artifact
        /artifact search <words>       — full-text search over bodies
        /artifact send <ref>           — send its file to the current chat
        /artifact send <ref> as <fmt>  — send converted via the documents
                                         engine (original unchanged)
        /artifact export <ref>         — send a bundle of one artifact
        /artifact export type:<t> | creator:<c> | mission:<id> | task:<id>
                                       — send a bundle of a filtered set

        <ref> is an artifact id, an artifact:// (or artm://) URI, or an
        alias. Owner-only (the handle_control grant: /artifact is not a
        public command). Failures are honest: an unknown ref lists what
        exists, an unreadable blob reports the error — nothing is ever
        reported sent that was not.
        """
        import time

        parts = (tail or "").split(None, 1)
        verb = parts[0].lower() if parts else "list"
        ref = parts[1].strip() if len(parts) > 1 else ""
        store = self._artifact_store()
        if store is None:
            return ("artifacts unavailable: no database is attached to "
                    "this chat context — artifacts live in the database.")
        if verb in ("list", "ls"):
            arts = store.search(limit=20)
            if not arts:
                return "no artifacts saved yet."
            lines = [f"artifacts ({len(arts)} most recent):"]
            for art in arts:
                title = str((art.metadata or {}).get("title")
                            or (art.metadata or {}).get("name") or "")
                label = f" — {title}" if title else (
                    f" — by {art.creator}" if art.creator else "")
                when = (time.strftime("%Y-%m-%d %H:%M",
                                      time.localtime(art.created_at))
                        if art.created_at else "")
                lines.append(f"  {art.id} · {art.type} · {art.size} bytes"
                             f"{label}" + (f" · {when}" if when else ""))
            aliases = store.aliases()
            if aliases:
                lines.append("aliases:")
                for name, target in sorted(aliases.items()):
                    lines.append(f"  {name} → {target}")
            return "\n".join(lines)
        if verb == "info":
            if not ref:
                return "usage: /artifact info <id|alias|artifact://uri>"
            art = self._resolve_artifact(store, ref)
            if art is None:
                return self._artifact_unknown(store, ref)
            return store.format_card(art)
        if verb in ("search", "find"):
            if not ref:
                return "usage: /artifact search <words>"
            hits = store.search_content(ref, limit=10)
            stats = store.content_index_stats()
            if not stats.get("available"):
                census = ("content index unavailable on this database — "
                          "body search is off, nothing was hidden.")
            else:
                kinds = ", ".join(
                    f"{kind} ×{n}" for kind, n in
                    sorted((stats.get("skipped_kinds") or {}).items()))
                census = (f"content index: {stats['indexed']} indexed · "
                          f"{stats['skipped']} skipped"
                          + (f" ({kinds})" if kinds else "")
                          + f" · {stats['superseded']} superseded")
            if not hits:
                return f"no artifact bodies match {ref!r}.\n{census}"
            lines = [f"artifact bodies matching {ref!r}:"]
            for hit in hits:
                snippet = " ".join((hit.snippet or "").split())
                lines.append(f"  {hit.artifact.id} · {hit.artifact.type}"
                             f" · {snippet}")
            lines.append(census)
            return "\n".join(lines)
        if verb == "export":
            if not ref:
                return ("usage: /artifact export <id|alias|uri> | "
                        "export type:<type> | creator:<who> | "
                        "mission:<id> | task:<id>")
            facet_key, _, facet_value = ref.partition(":")
            facet_fields = {"type": "type", "creator": "creator",
                            "mission": "mission_id", "task": "task_id"}
            if facet_key.lower() in facet_fields and facet_value.strip():
                facet_value = facet_value.strip()
                arts = store.search(
                    **{facet_fields[facet_key.lower()]: facet_value},
                    limit=500)
                if not arts:
                    return (f"no artifacts match "
                            f"{facet_key.lower()}:{facet_value} — "
                            "nothing to export.")
                ids = [a.id for a in arts]
                export_arts = arts
                label = f"{facet_key.lower()}:{facet_value}"
            else:
                art = self._resolve_artifact(store, ref)
                if art is None:
                    return self._artifact_unknown(store, ref)
                ids = [art.id]
                export_arts = [art]
                label = art.id
            import re
            from pathlib import Path

            db = getattr(self.context, "db", None)
            db_path = getattr(db, "path", None)
            base = Path(db_path).parent if db_path else Path("data")
            safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
            dest = (base / "artifact_exports" /
                    f"bundle-{safe_label}-{int(time.time())}.tar.gz")
            try:
                bundle = store.export_bundle(ids, dest)
            except Exception as exc:  # noqa: BLE001 - honest failure
                return (f"artifact export failed for {label}: {exc} "
                        "— nothing was sent.")
            chat = self._ref_from_key(chat_key)
            transfers = [self._begin_artifact_transfer(
                a, kind="export", chat=chat) for a in export_arts]
            outcome = self.context.tools.call(
                "file_send", platform=chat.platform, chat_id=chat.chat_id,
                path=str(bundle),
                caption=(f"artifact bundle — {len(ids)} artifact(s) "
                         f"({label})"))
            if not outcome.ok:
                error = str(getattr(outcome.error, "message",
                                     outcome.error))
                for entry in transfers:
                    self._complete_artifact_transfer(
                        entry, ok=False, error=error)
                return (f"artifact export failed ({label}): "
                        f"{getattr(outcome.error, 'message', outcome.error)}")
            value = getattr(outcome, "value", None)
            send_ref = (str(value.get("message_id") or "")
                        if isinstance(value, dict) else "")
            for entry in transfers:
                self._complete_artifact_transfer(
                    entry, ok=True, send_ref=send_ref)
            # Unlike a plain send (whose file IS the whole answer),
            # an export CREATES a new bundle file — the reply names
            # what was made and how much it carries.
            return (f"artifact bundle exported: {len(ids)} artifact(s) "
                    f"({label}) — {bundle.name}")
        if verb == "send":
            tokens = ref.split()
            fmt_raw = ""
            if len(tokens) >= 3 and tokens[-2].lower() in ("as", "to"):
                fmt_raw = tokens[-1].lower().lstrip(".")
                ref = " ".join(tokens[:-2])
            if not ref:
                return ("usage: /artifact send <id|alias|artifact://uri> "
                        "[as <format>]")
            art = self._resolve_artifact(store, ref)
            if art is None:
                return self._artifact_unknown(store, ref)
            try:
                data = store.read(art.id)
            except Exception as exc:  # noqa: BLE001 - honest, never a fake send
                return (f"couldn't read artifact {art.id} "
                        f"({art.size} bytes): {exc} — nothing was sent.")
            path = self._artifact_export_path(store, art)
            caption = f"artifact {art.id} ({art.type})"
            converted_note = ""
            if fmt_raw:
                try:
                    data, fmt_name, ext = self._artifact_convert(
                        art, data, fmt_raw)
                except ValueError as exc:
                    return str(exc)
                path = path.with_suffix(ext)
                caption = (f"artifact {art.id} — converted to {fmt_name} "
                           "(stored original unchanged)")
                converted_note = (
                    f"sent artifact {art.id} converted to {fmt_name} — "
                    "the stored original is unchanged.")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            chat = self._ref_from_key(chat_key)
            transfer = self._begin_artifact_transfer(
                art, kind="send", chat=chat)
            outcome = self.context.tools.call(
                "file_send", platform=chat.platform, chat_id=chat.chat_id,
                path=str(path), caption=caption)
            if not outcome.ok:
                self._complete_artifact_transfer(
                    transfer, ok=False,
                    error=str(getattr(outcome.error, "message",
                                       outcome.error)))
                return (f"artifact send failed ({len(data)} bytes): "
                        f"{getattr(outcome.error, 'message', outcome.error)}")
            value = getattr(outcome, "value", None)
            send_ref = (str(value.get("message_id") or "")
                        if isinstance(value, dict) else "")
            self._complete_artifact_transfer(
                transfer, ok=True, send_ref=send_ref)
            # The artifact file IS the delivery — no "📎 sent …"
            # receipt beside it in the same chat (owner
            # no-meta-narration rule, 2026-11-01).  A converted send
            # is the one exception: the reply names the conversion,
            # because the bytes delivered are NOT the stored bytes.
            return converted_note
        return ("usage: /artifact list | info <id|alias|uri> | "
                "search <words> | send <id|alias|uri> [as <format>] | "
                "export <id|alias|uri> | export type:<t> | creator:<c> | "
                "mission:<id> | task:<id>")

    # ── goal classes: /goals ───────────────────────────────────────────────
    def _goal_tracker(self) -> Any:
        """GoalTracker over the context db, or None when no db is attached.

        The user-goal store (nomorals/goals) — NOT the finance
        savings GoalStore behind /goal and NOT the agent GoalSystem
        behind the brain's goal tool.  The owner's rows live under
        the ``owner`` user id (the nm idea CLI default).
        """
        from ...goals import GoalTracker

        db = getattr(self.context, "db", None)
        if db is None:
            return None
        return GoalTracker(db)

    def _control_goals(self, tail: str, chat_key: str = "") -> Any:
        """/goals — the goal-class board and the tracker's briefings.

        /goals                       — board: goals grouped by class
        /goals list [class]          — the board, or one class's section
        /goals briefing              — progress briefing (get_briefing)
        /goals review                — weekly review (weekly_review)
        /goals focus                 — today's focus ranking (today_focus)
        /goals progress <goal>       — one goal: class reading + pace
        /goals add <title> [in <class>] | add <class>: <title>
                                     — create, filed under a class
        /goals class <goal> <class>  — re-file a goal under another class
        /goals classes               — the class picker as text

        <goal> is a goal id (or unique prefix), #n from the board,
        or a unique part of its title.  Owner-only (the handle_control
        grant: /goals is not a public command).  /goal stays the
        finance savings surface — this is the user-goal tracker.
        Progress always renders with its basis: measured (with the
        provider's source) or self-reported — never indistinguishable.
        The board also returns a class-picker keyboard (grid-packed);
        surfaces that cannot take buttons get the numbered fallback
        from _send_control_reply, and the text alone always works.
        """
        import re

        from ...goals import surface
        from ...goals.classes import GoalClassError, list_classes

        tracker = self._goal_tracker()
        if tracker is None:
            return ("goals unavailable: no database is attached to this "
                    "chat context — goals live in the database.")
        user = surface.OWNER_USER_ID
        parts = (tail or "").split(None, 1)
        verb = parts[0].lower() if parts else "list"
        rest = parts[1].strip() if len(parts) > 1 else ""

        if verb in ("list", "board", ""):
            only = None
            if rest:
                try:
                    from ...goals.classes import get_class
                    only = get_class(rest.split()[0]).id
                except GoalClassError as exc:
                    return str(exc)
            text = surface.render_class_board(tracker, user, only_class=only)
            if only is not None:
                if only == "careers":
                    # The class board carries the measured reading;
                    # the pipeline board below it is the drill-down
                    # (the careers organ's own surface).
                    from ...careers.surface import render_pipeline_board
                    text += "\n\n" + render_pipeline_board(
                        self._careers_store())
                return text
            from ...social.chat.tgbot_buttons import actionable, grid_layout
            buttons = [(c.title, f"/goals list {c.id}")
                       for c in list_classes()]
            buttons += [("Briefing", "/goals briefing"),
                        ("Review", "/goals review")]
            return actionable(text, grid_layout([buttons]))
        if verb == "briefing":
            return surface.render_briefing(tracker, user)
        if verb == "review":
            return surface.render_review(tracker, user)
        if verb == "focus":
            return surface.render_focus(tracker, user)
        if verb == "progress":
            if not rest:
                return "usage: /goals progress <goal id|#n|title part>"
            return surface.render_goal_progress(tracker, user, rest)
        if verb == "classes":
            return surface.render_classes()
        if verb == "add":
            if not rest:
                return ("usage: /goals add <title> [in <class>] | "
                        "/goals add <class>: <title>")
            title, class_arg = rest, None
            m = re.match(r"^([A-Za-z][\w-]*)\s*:\s*(.+)$", rest)
            if m:
                class_arg, title = m.group(1), m.group(2).strip()
            else:
                m = re.search(r"\s+in\s+([A-Za-z][\w-]*)\s*$", rest)
                if m:
                    class_arg = m.group(1)
                    title = rest[:m.start()].strip()
            if not title:
                return "a goal needs a title — /goals add <title> [in <class>]"
            try:
                goal, class_id, suggested = surface.create_goal_filed(
                    tracker, user, title, class_arg)
            except GoalClassError as exc:
                return str(exc)
            except ValueError as exc:
                return f"couldn't create that goal: {exc}"
            from ...goals.classes import get_class
            descriptor = get_class(class_id)
            note = (f" — filed under {descriptor.title} from the title; "
                    f"/goals class {goal.goal_id} <class> moves it"
                    if suggested else "")
            if class_id == "interests":
                watch = surface.interest_watch_note(tracker, goal)
                if watch:
                    note += f"\n  {watch}"
            return (f"filed **{goal.title}** under {descriptor.title} "
                    f"(`{class_id}`){note}\n  goal id: {goal.goal_id}")
        if verb == "class":
            words = rest.rsplit(None, 1)
            if len(words) != 2:
                return "usage: /goals class <goal id|#n|title part> <class>"
            ref, class_arg = words
            try:
                return surface.refile_goal(tracker, user, ref, class_arg)
            except GoalClassError as exc:
                return str(exc)
        return ("usage: /goals [list [class]] | briefing | review | focus | "
                "progress <goal> | add <title> [in <class>] | "
                "class <goal> <class> | classes")

    # ── careers: /careers ─────────────────────────────────────────────
    def _careers_store(self) -> Any:
        """The career pipeline store (nomorals/careers), resolved via
        the process settings — the same resolution the careers goal
        provider uses, so surface and measurement read one file."""
        from ...careers.store import CareerStore, careers_path
        from ...core.config import get_settings

        return CareerStore(careers_path(get_settings()))

    def _control_careers(self, tail: str, chat_key: str = "") -> Any:
        """/careers — the career pipeline board and its verbs.

        /careers                          — pipeline board (attention
                                            call-outs, then by stage)
        /careers add <company> | <role> [| <source or link>]
                                          — file an application (saved)
        /careers move <app> <stage> [note]
                                          — move it, history appended
        /careers next <app> | <action> [by YYYY-MM-DD | in Nd]
                                          — set the next action + due
        /careers note <app> | <text>      — append a note
        /careers attach <app> <artifact>  — record the CV artifact sent
        /careers remove <app>             — delete the application
        /careers stages                   — the stage vocabulary

        <app> is an application id (or unique prefix), #n from the
        board, or a unique part of "company role".  Owner-only.
        Careers-class goals measure from this pipeline (the
        application_pipeline provider) — see /goals list careers.
        """
        import re
        from datetime import UTC, datetime

        from ...careers import surface as careers_surface
        from ...careers.store import STAGES, CareerError

        store = self._careers_store()
        parts = (tail or "").split(None, 1)
        verb = parts[0].lower() if parts else "list"
        rest = parts[1].strip() if len(parts) > 1 else ""

        def _resolve(ref: str) -> Any:
            app, candidates = careers_surface.resolve_application(
                store, ref)
            if app is None:
                return None, careers_surface.unknown_application_text(
                    ref, candidates)
            return app, None

        if verb in ("list", "board", ""):
            return careers_surface.render_pipeline_board(store)
        if verb == "stages":
            return careers_surface.render_stages()
        if verb == "add":
            fields = [f.strip() for f in rest.split("|")]
            if len(fields) < 2 or not fields[0] or not fields[1]:
                return ("usage: /careers add <company> | <role> "
                        "[| <source or link>]")
            source, link = "", ""
            for extra in fields[2:]:
                if extra.startswith(("http://", "https://")):
                    link = extra
                elif extra:
                    source = extra
            try:
                app = store.add(fields[0], fields[1],
                                source=source, link=link)
            except CareerError as exc:
                return str(exc)
            return (f"filed **{app.company} — {app.role}** at saved "
                    f"· `{app.id}`\n  move it: /careers move "
                    f"{app.id} applied")
        if verb == "move":
            tokens = rest.split()
            stage_at = next(
                (i for i in range(1, len(tokens))
                 if tokens[i].lower() in STAGES), None)
            if stage_at is None:
                return ("usage: /careers move <app> <stage> [note] — "
                        f"stages: {', '.join(STAGES)}")
            ref = " ".join(tokens[:stage_at])
            stage = tokens[stage_at].lower()
            note = " ".join(tokens[stage_at + 1:]).lstrip("—- ").strip()
            app, error = _resolve(ref)
            if app is None:
                return error
            try:
                moved = store.move(app.id, stage, note=note)
            except CareerError as exc:
                return str(exc)
            return (f"moved **{moved.company} — {moved.role}** to "
                    f"{moved.stage}")
        if verb in ("next", "note"):
            if "|" in rest:
                ref, payload = (p.strip() for p in rest.split("|", 1))
            else:
                head = rest.split(None, 1)
                ref = head[0] if head else ""
                payload = head[1].strip() if len(head) > 1 else ""
            app, error = _resolve(ref)
            if app is None:
                return error
            if verb == "note":
                try:
                    store.add_note(app.id, payload)
                except CareerError as exc:
                    return str(exc)
                return f"noted on **{app.company} — {app.role}**."
            due = None
            m = re.search(r"\s+by\s+(\d{4})-(\d{2})-(\d{2})\s*$", payload)
            if m:
                try:
                    due = datetime(int(m.group(1)), int(m.group(2)),
                                   int(m.group(3)), 12, 0,
                                   tzinfo=UTC).timestamp()
                except ValueError:
                    return (f"that date doesn't exist: "
                            f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
                payload = payload[:m.start()].strip()
            else:
                m = re.search(r"\s+in\s+(\d+)\s*d\s*$", payload)
                if m:
                    import time as _time
                    due = _time.time() + int(m.group(1)) * 86400
                    payload = payload[:m.start()].strip()
            try:
                updated = store.set_next_action(app.id, payload, due)
            except CareerError as exc:
                return str(exc)
            if not updated.next_action:
                return (f"next action cleared for "
                        f"**{app.company} — {app.role}**.")
            due_note = ""
            if updated.next_action_due is not None:
                due_note = (" · due " + datetime.fromtimestamp(
                    updated.next_action_due, tz=UTC
                ).strftime("%Y-%m-%d"))
            return (f"next for **{app.company} — {app.role}**: "
                    f"{updated.next_action}{due_note}")
        if verb == "attach":
            head = rest.split(None, 1)
            if len(head) < 2:
                return ("usage: /careers attach <app> "
                        "<artifact id|uri|alias>")
            app, error = _resolve(head[0])
            if app is None:
                return error
            artifacts = self._artifact_store()
            if artifacts is None:
                return ("careers attach unavailable: no database is "
                        "attached to this chat context — artifacts "
                        "live in the database.")
            art = self._resolve_artifact(artifacts, head[1].strip())
            if art is None:
                return self._artifact_unknown(artifacts, head[1].strip())
            store.attach_cv(app.id, art.id)
            return (f"attached 📎 {art.id} ({art.type}) to "
                    f"**{app.company} — {app.role}** — the CV of "
                    "record for this application.")
        if verb == "remove":
            app, error = _resolve(rest)
            if app is None:
                return error
            store.remove(app.id)
            return f"removed **{app.company} — {app.role}** ({app.id})."
        return ("usage: /careers [list] | add <company> | <role> | "
                "move <app> <stage> | next <app> | <action> [by <date>] | "
                "note <app> | <text> | attach <app> <artifact> | "
                "remove <app> | stages")

    # ── ideas: /ideas (WS3 P5) ────────────────────────────────────────
    def _idea_tracker(self) -> Any:
        """IdeaTracker over the context db, or None when no db is
        attached — the same store the ``nm idea`` CLI drives."""
        from ...goals import IdeaTracker

        db = getattr(self.context, "db", None)
        if db is None:
            return None
        return IdeaTracker(db)

    def _control_ideas(self, tail: str, chat_key: str = "") -> Any:
        """/ideas — raw ideas, captured verbatim, and the development
        cell that grows them (expansion → connections → provenanced
        research → verified verdict: promote-in-class / draft /
        mission / park-with-trigger).

        The raw capture never changes: development lives in its own
        record beside the idea.  Handles are #n in list order —
        internal ids never print.
        """
        import re as _re

        from ...goals.surface import OWNER_USER_ID, run_sync
        from ..idea_development import (
            EXIT_ALIASES,
            IdeaDevelopment,
            make_brain_expand_fn,
            make_research_fn,
            render_record,
            resolve_idea,
        )

        tracker = self._idea_tracker()
        if tracker is None:
            return ("ideas unavailable: no database is attached to "
                    "this chat context — ideas live in the database.")
        user = OWNER_USER_ID
        text = (tail or "").strip()
        verb, _, rest = text.partition(" ")
        verb = verb.lower()
        rest = rest.strip()

        def _board() -> str:
            ideas = run_sync(tracker.list_ideas(user, limit=100))
            cell = IdeaDevelopment(self.context.db)
            lines = ["💡 your ideas — " + str(len(ideas)) + " captured"]
            for pos, idea in enumerate(ideas[:15], start=1):
                record = cell.latest_for(idea.idea_id)
                grown = ""
                if record:
                    exit_name = (record.get("verdict") or {}).get(
                        "exit", "")
                    grown = (" · developed → " + exit_name
                             if exit_name else " · developed")
                ice = (" · ICE " + str(idea.ice_score)
                       if idea.ice_score is not None else "")
                lines.append("#" + str(pos) + " " + idea.title
                             + " (" + idea.status + ")" + ice + grown)
            if len(ideas) > 15:
                lines.append("…and " + str(len(ideas) - 15) + " more")
            lines.append("grow one: /ideas develop <#> — the raw "
                         "words never change")
            return "\n".join(lines)

        if verb in ("", "list"):
            return _board()
        if verb == "add":
            if not rest:
                return ("usage: /ideas add <title> [| <detail>] — the "
                        "words are stored exactly as you give them")
            title, sep, description = rest.partition("|")
            title = title.strip()
            if not title:
                return "the idea needs at least a title"
            created = run_sync(tracker.create(
                user, title,
                description=description.strip() if sep else "",
                source="chat"))
            ideas = run_sync(tracker.list_ideas(user, limit=100))
            pos = next((n for n, i in enumerate(ideas, start=1)
                        if i.idea_id == created.idea_id), 1)
            return ("💡 idea #" + str(pos) + " captured: "
                    + created.title
                    + "\ngrow it when ready: /ideas develop "
                    + str(pos))
        if verb == "show":
            idea, problem = resolve_idea(tracker, user, rest)
            if idea is None:
                return problem
            cell = IdeaDevelopment(self.context.db)
            record = cell.latest_for(idea.idea_id)
            head = ("💡 " + idea.title + " (" + idea.status + ")"
                    + (("\n" + idea.description)
                       if idea.description else ""))
            if record is None:
                return (head + "\n\nnot developed yet — /ideas develop "
                        "grows it (expansion, connections, research, "
                        "a verdict)")
            return head + "\n\n" + render_record(record)
        if verb == "develop":
            trigger = ""
            match = _re.search(r"\bwhen\s+(.+)$", rest,
                               flags=_re.IGNORECASE)
            if match:
                trigger = match.group(1).strip()
                rest = rest[:match.start()].strip()
            class_arg = ""
            match = _re.search(r"\bin\s+([a-z0-9_-]+)\s*$", rest,
                               flags=_re.IGNORECASE)
            if match:
                class_arg = match.group(1)
                rest = rest[:match.start()].strip()
            exit_arg = ""
            match = _re.search(
                r"\bas\s+(goal|draft|mission|park)\s*$", rest,
                flags=_re.IGNORECASE)
            if match:
                exit_arg = EXIT_ALIASES.get(match.group(1).lower(),
                                            match.group(1).lower())
                rest = rest[:match.start()].strip()
            idea, problem = resolve_idea(tracker, user, rest)
            if idea is None:
                return problem
            from ...missions.mission import MissionStore

            cell = IdeaDevelopment(self.context.db)
            try:
                record = cell.develop(
                    idea.idea_id, exit_override=exit_arg,
                    class_arg=class_arg, trigger=trigger,
                    expand_fn=make_brain_expand_fn(self.context),
                    research_fn=make_research_fn(
                        self.context.db,
                        getattr(self.context, "tools", None)),
                    draft_queue=self._draft_queue(),
                    mission_store=MissionStore(self.context.db))
            except (KeyError, ValueError) as exc:
                return "couldn't develop that idea: " + str(exc)
            return render_record(record)
        return ("usage: /ideas [list] | add <title> [| <detail>] | "
                "show <#> | develop <#> [as goal|draft|mission|park] "
                "[in <class>] [when <trigger>]")

    # ── post drafts: /drafts + draft: button taps ─────────────────────
    def _draft_queue(self) -> Any:
        """The post DraftQueue — the same default store the #43
        override path opens (social/drafts.py)."""
        from ...social.drafts import DraftQueue

        return DraftQueue()

    def _draft_callback(self, data: str, chat_key: str = "",
                        message: Any = None, *,
                        schedule_answer: float | None = None) -> str:
        """Route one ``draft:<action>:<id>`` review-button tap.

        The tap re-entered as the text ``/draft:<action>:<id>`` (the
        adapters' callback convention) and ``handle_control`` stripped
        it to here. The drafts organ's own router
        (:func:`...social.drafts.handle_draft_callback`) does the
        act; this method only supplies the runtime's queue, publisher
        and scheduler. A not-yet-decided draft (still ``draft``, or
        ``scheduled`` by the weekly pipeline at an audience window)
        is re-presented for review first — the queue's graph sends
        both back to pending_review, where the callback's approve is
        legal — because a card tap on it means "post it now", not
        "wait for the window". Never raises: a tap always answers.

        Schedule and Edit are two-step acts: the organ's reply to
        the tap is a question ("when should I post it?") or a
        prompt ("send me the revised text") — so the tap ARMS an
        ask for this chat (see :meth:`_note_draft_ask`) and the
        owner's next message completes the act through
        :meth:`_draft_ask_answer`. ``schedule_answer`` is that
        completion re-entering: the parsed time rides the organ's
        own ``ask_schedule_time`` seam into ``schedule_post``, so
        the stored schedule is the tap's answer, not a default.
        """
        from ...social.drafts import (
            DRAFT,
            SCHEDULED,
            handle_draft_callback,
        )

        payload = (data or "").strip().lstrip("/")
        parts = payload.split(":")
        if len(parts) != 3 or parts[0] != "draft":
            return ""
        _, action, draft_id = parts
        queue = self._draft_queue()
        try:
            draft = queue.get(draft_id)
            if draft is not None and action == "post" \
                    and draft.status in (DRAFT, SCHEDULED):
                queue.propose(draft_id)
            if draft is not None and action == "schedule" \
                    and schedule_answer is not None \
                    and draft.status in (DRAFT, SCHEDULED):
                # The completing step of a Schedule tap moves the
                # draft to SCHEDULED, a transition the graph only
                # allows from review — same re-presentation as the
                # post tap above, done at answer time (the asking
                # tap itself moves nothing).
                queue.propose(draft_id)
            manager = None
            if action == "post":
                try:
                    from ...social import SocialManager
                    manager = SocialManager(
                        self.context).register_builtins()
                except Exception as exc:  # noqa: BLE001
                    return f"can't post — no publisher wired: {exc}"
            scheduler = getattr(self, "_scheduler", None) or \
                self.context.extras.get("scheduler")
            reply = handle_draft_callback(
                queue, payload, manager=manager, scheduler=scheduler,
                ask_schedule_time=(
                    (lambda _id: schedule_answer)
                    if schedule_answer is not None else None))
            if chat_key and draft is not None:
                # Arm the second step (see the docstring): a
                # Schedule tap that just asked its question, and
                # every Edit tap, wait on the owner's next message
                # in THIS chat. A completed schedule (an answer
                # re-entering) never re-arms.
                if action == "schedule" and schedule_answer is None:
                    self._note_draft_ask(chat_key, "schedule", draft_id)
                elif action == "edit":
                    self._note_draft_ask(chat_key, "edit", draft_id)
            return reply
        except Exception as exc:  # noqa: BLE001 — a tap must answer
            return f"draft action failed: {exc}"
        finally:
            queue.close()

    # ── draft asks: the Schedule/Edit second step ─────────────────

    #: How long a Schedule/Edit ask stays armed for its answer —
    #: long enough for the owner to type a time or a revision,
    #: short enough that an unrelated message much later is never
    #: captured as one.
    _DRAFT_ASK_TTL_S = 900.0

    def _draft_ask_store(self) -> dict[str, dict[str, Any]]:
        store = getattr(self, "_draft_asks", None)
        if store is None:
            store = self._draft_asks = {}
        return store

    def _note_draft_ask(self, chat_key: str, action: str,
                        draft_id: str) -> None:
        """Arm one chat's draft ask (single slot per chat — a new
        tap's ask replaces the old, the pending-ask idiom)."""
        self._draft_ask_store()[chat_key] = {
            "action": action, "draft_id": draft_id, "at": time.time()}

    @staticmethod
    def _parse_draft_schedule_answer(text: str) -> float | None:
        """A schedule answer → a timestamp, or None when the text
        carries no readable time. The drafts organ's own grammar
        first ("post at 6" — the exact phrase its question
        suggests), then the scheduler's natural-datetime parser
        for the bare forms ("tomorrow 9am", "in 2 hours",
        "9:30pm")."""
        from ...social.drafts import parse_post_time

        when = parse_post_time(text)
        if when is not None:
            return when
        try:
            from ...scheduler.recurrence import parse_natural_datetime

            return parse_natural_datetime(text).timestamp()
        except Exception:  # noqa: BLE001 - unparseable is an outcome
            return None

    def _draft_ask_answer(self, message: Any) -> str | None:
        """Complete an armed draft Schedule/Edit ask from the
        owner's next message in the same chat.

        Called from ``_process_inner`` for incoming operator
        messages that are not commands. Returns the reply to send,
        or None when no live ask is armed for this chat (the
        message is ordinary chat and flows on untouched). An
        unparseable schedule answer does NOT consume the ask — the
        owner is told how to phrase it and the ask stays armed;
        "cancel" disarms either ask with the draft untouched.
        Never raises.
        """
        chat_key = getattr(getattr(message, "chat", None), "key", "") or ""
        store = self._draft_ask_store()
        ask = store.get(chat_key)
        if not ask:
            return None
        if time.time() - float(ask.get("at") or 0) > self._DRAFT_ASK_TTL_S:
            store.pop(chat_key, None)
            return None
        text = (getattr(message, "text", "") or "").strip()
        if not text:
            return None
        action = str(ask.get("action") or "")
        draft_id = str(ask.get("draft_id") or "")
        if text.lower() in ("cancel", "never mind", "nevermind"):
            store.pop(chat_key, None)
            return "okay — dropped that; the draft stays as it was."
        if action == "schedule":
            when = self._parse_draft_schedule_answer(text)
            if when is None:
                return ("I couldn't read a time in that — say e.g. "
                        "'post at 6' or 'tomorrow 9am', or 'cancel' "
                        "to drop the scheduling.")
            store.pop(chat_key, None)
            return self._draft_callback(
                f"draft:schedule:{draft_id}", chat_key,
                message=message, schedule_answer=when)
        if action == "edit":
            store.pop(chat_key, None)
            from ...social.drafts import DISCARDED, POSTED

            queue = self._draft_queue()
            try:
                draft = queue.get(draft_id)
                if draft is None:
                    return "that draft is gone."
                if draft.status in (POSTED, DISCARDED):
                    # Settled between the tap and the answer: a
                    # late answer must never rewrite history.
                    return (f"that draft is already {draft.status} — "
                            "it can't be edited now.")
                queue.update_content(draft_id, text)
            except Exception as exc:  # noqa: BLE001 - answered in words
                return f"couldn't update the draft: {exc}"
            finally:
                queue.close()
            preview = text[:200] + ("…" if len(text) > 200 else "")
            return f"✏️ draft updated — it now reads:\n\n{preview}"
        store.pop(chat_key, None)
        return None

    def _control_drafts(self, tail: str, chat_key: str = "",
                        message: Any = None) -> Any:
        """/drafts — the post-draft review board (WS3 P2).

        /drafts                — the review batch: every draft
                                 awaiting a decision, numbered; tap a
                                 number to open its card, or use the
                                 text verbs below
        /drafts show <n>       — one draft's review card + buttons
        /drafts post <n>       — post THAT draft (the one named)
        /drafts discard <n>    — discard THAT draft

        <n> is the number from the board (a full draft id also
        resolves). The text path is complete without buttons: every
        button action has a verb above, and verbs and taps run the
        same callback route. Owner-only (the handle_control grant:
        /drafts is not a public command). Nothing posts without the
        owner's tap, verb, or explicit 'post this' — the #43 rule,
        unchanged.
        """
        from ...social.drafts import (
            DRAFT,
            propose_post,
            render_review_batch,
            review_buttons,
            review_card,
            reviewable_drafts,
        )

        queue = self._draft_queue()
        target: tuple[str, str] | None = None
        try:
            drafts = reviewable_drafts(queue)
            parts = (tail or "").split(None, 1)
            verb = parts[0].lower() if parts else "list"
            rest = parts[1].strip() if len(parts) > 1 else ""

            if verb in ("list", ""):
                if not drafts:
                    return "📝 no drafts waiting for review."
                text = render_review_batch(drafts)
                text += ("\n\nopen one: /drafts show <n> · post it: "
                         "/drafts post <n> · discard: "
                         "/drafts discard <n>")
                from ...social.chat.tgbot_buttons import actionable, grid_layout
                buttons = []
                for i, d in enumerate(drafts[:8], 1):
                    label = d.content[:24].replace("\n", " ")
                    if len(d.content) > 24:
                        label += "…"
                    buttons.append((f"{i}. {label}", f"drafts show {i}"))
                return actionable(text, grid_layout([buttons]))

            if verb in ("show", "post", "discard"):
                if not rest:
                    return (f"usage: /drafts {verb} <n> — the number "
                            "from /drafts")
                token = rest.split()[0]
                probe = token[1:] if token.startswith("#") else token
                picked: tuple[int, Any] | None = None
                if probe.isdigit():
                    idx = int(probe)
                    if 1 <= idx <= len(drafts):
                        picked = (idx, drafts[idx - 1])
                else:
                    for i, d in enumerate(drafts, 1):
                        if d.id == token:
                            picked = (i, d)
                            break
                if picked is None:
                    if not drafts:
                        return "📝 no drafts waiting for review."
                    return (f"no draft {token} on the board — "
                            f"{len(drafts)} waiting; /drafts lists them.")
                idx, draft = picked
                if verb in ("post", "discard"):
                    # Act through the callback route — but only after
                    # this queue closes (the route opens its own).
                    target = (verb, draft.id)
                else:
                    from ...social.chat.tgbot_buttons import actionable, grid_layout
                    if draft.status == DRAFT:
                        # Opening an unproposed draft IS proposing
                        # it: the standing-question form is its card.
                        queue.propose(draft.id)
                        draft = queue.get(draft.id) or draft
                        text, buttons = propose_post(draft)
                    else:
                        text = review_card(draft)
                        buttons = review_buttons(draft)
                    text += (f"\n\nor type: /drafts post {idx} · "
                             f"/drafts discard {idx}")
                    return actionable(text, grid_layout([buttons]))
            else:
                return ("usage: /drafts [list] | show <n> | post <n> | "
                        "discard <n> — <n> is the number from /drafts")
        finally:
            queue.close()
        if target is not None:
            action, draft_id = target
            return self._draft_callback(
                f"draft:{action}:{draft_id}", chat_key, message=message)
        return "📝 no drafts waiting for review."

    # ── the day record: /day ─────────────────────────────────────────────
    def _control_day(self, tail: str, chat_key: str = "") -> Any:
        """/day — the day record + evening derivation (WS3 P4).

        /day                   — today, whole: your words, then what
                                 the organs saw (goals, trading,
                                 missions, files), then what was made
        /day tell <text>       — add a note to today, in your words
        /day yesterday         — yesterday's record
        /day show <YYYY-MM-DD> — any day's record
        /day make              — make something of today: draft
                                 posts land in /drafts for review
        /day skip              — tonight's automatic derivation
                                 leaves today alone (recorded on the
                                 record; /day unskip undoes it)
        /day export            — file today's record as an artifact

        Owner-only (not a public command).  Narration in ordinary
        chat reaches the same record through the brain's ``day``
        tool — this surface is the deliberate case.  Nothing here
        posts: drafts land in the review queue, and the automatic
        19:00 run is the same derivation, fired by the durable job.
        """
        from datetime import datetime as _dt

        from ..dayrecord import (
            DayStore,
            brain_llm_fn,
            derive_day,
            export_day,
            render_day,
            today_key,
            yesterday_key,
        )

        db = getattr(self.context, "db", None)
        if db is None:
            return ("🌅 the day record is unavailable — no database"
                    " attached to this chat")
        store = DayStore(db)
        parts = (tail or "").split(None, 1)
        verb = parts[0].lower() if parts else "show"
        rest = parts[1].strip() if len(parts) > 1 else ""

        def _day_arg() -> str | None:
            if not rest:
                return today_key()
            try:
                _dt.strptime(rest, "%Y-%m-%d")
            except ValueError:
                return None
            return rest

        if verb == "tell":
            if not rest:
                return ("usage: /day tell <what you did> — it goes"
                        " into today's record, in your words")
            frag = store.add_fragment(text=rest, source="chat")
            count = len(store.fragments(frag["day"]))
            return (f"🗣 kept in today's record ({count} note(s) so"
                    " far). /day shows the day; /day make turns it"
                    " into drafts.")
        if verb in ("show", "list", ""):
            day = _day_arg()
            if day is None:
                return "usage: /day show <YYYY-MM-DD>"
            return render_day(self.context, day,
                              tracker=self._goal_tracker(),
                              artifact_store=self._artifact_store())
        if verb == "yesterday":
            return render_day(self.context, yesterday_key(),
                              tracker=self._goal_tracker(),
                              artifact_store=self._artifact_store())
        if verb in ("make", "derive"):
            day = _day_arg()
            if day is None:
                return "usage: /day make [YYYY-MM-DD]"
            queue = self._draft_queue()
            try:
                result = derive_day(
                    self.context, day, queue=queue,
                    llm_fn=brain_llm_fn(self.context),
                    tracker=self._goal_tracker(),
                    artifact_store=self._artifact_store())
            finally:
                queue.close()
            if result.drafts:
                return f"✍️ {result.note}\nreview them: /drafts"
            return f"🌅 {result.note}"
        if verb == "skip":
            day = _day_arg()
            if day is None:
                return "usage: /day skip [YYYY-MM-DD]"
            store.set_skip(day, skip=True,
                           note="skipped at the owner's word")
            return (f"⏭ tonight's automatic derivation will skip"
                    f" {day}. /day unskip undoes it; /day make still"
                    " works whenever you ask.")
        if verb == "unskip":
            day = _day_arg()
            if day is None:
                return "usage: /day unskip [YYYY-MM-DD]"
            store.set_skip(day, skip=False)
            return f"▶️ {day} is back on for the evening derivation."
        if verb == "export":
            day = _day_arg()
            if day is None:
                return "usage: /day export [YYYY-MM-DD]"
            art = export_day(self.context, day,
                             tracker=self._goal_tracker(),
                             artifact_store=self._artifact_store())
            if art is None:
                return ("the day could not be filed — no artifact"
                        " store attached")
            return (f"📎 {day} is filed as an artifact ({art.uri})"
                    " — /artifact can send it anywhere")
        return ("usage: /day [show] | tell <text> | yesterday |"
                " make | skip | unskip | export")

    # ── content calendar: /calendar ─────────────────────────────────────
    def _control_calendar(self, tail: str, chat_key: str = "") -> Any:
        """/calendar — the unified content calendar + series (WS3 P6).

        One board: drafts awaiting a decision, the forward timeline
        (scheduled drafts + due shorts + scheduled posts together)
        and the series — a series is a named, ordered run of pieces
        (a campaign is a series with an objective). Pieces attach by
        the numbers the boards show; internal ids never print.
        Owner-only via CONTROL_COMMANDS, like /drafts and /day.
        """
        from ...social.drafts import reviewable_drafts
        from ...social.series import (
            KIND_DRAFT, KIND_POST, KIND_SHORT, SeriesStore,
            render_calendar, stamp_draft)

        db = self.context.db
        queue = self._draft_queue()
        store = SeriesStore(db)

        def _shorts() -> list:
            # The planned shorts, read lazily and contained — a
            # broken calendar file never breaks the board.
            try:
                from ...media.contentops.pipeline import ShortPipeline

                return ShortPipeline().calendar.list()
            except Exception:  # noqa: BLE001
                return []

        def _series_named(name: str):
            for s in store.list_series():
                if s["name"].lower() == name.strip().lower():
                    return s
            return None

        try:
            text = (tail or "").strip()
            low = text.lower()
            if not text or low == "list":
                return render_calendar(db, queue, shorts=_shorts(),
                                       store=store)
            if low == "series":
                series = store.list_series()
                if not series:
                    return ("no series yet — start one:"
                            " /calendar new <name> [| <objective>]")
                lines = ["📚 **your series**", ""]
                for s in series:
                    members = store.members(s["id"])
                    obj = (f" — 🎯 {s['objective']}"
                           if s.get("objective") else "")
                    lines.append(
                        f"  • **{s['name']}**{obj} —"
                        f" {len(members)} piece(s)")
                return "\n".join(lines)
            if low.startswith("series "):
                name = text[len("series "):].strip()
                s = _series_named(name)
                if s is None:
                    return f"no series named {name!r}."
                members = store.members(s["id"])
                lines = [f"📚 **{s['name']}**"
                         + (f" — 🎯 {s['objective']}"
                            if s.get("objective") else "")]
                if not members:
                    lines.append("  (no pieces yet — attach with"
                                 " /calendar add)")
                for i, m in enumerate(members, 1):
                    label = self._calendar_piece_label(
                        m, queue=queue, shorts=_shorts(), db=db)
                    lines.append(f"  {i}. {label}")
                return "\n".join(lines)
            if low.startswith("new "):
                body = text[len("new "):].strip()
                name, _, objective = body.partition("|")
                name = name.strip()
                if not name:
                    return ("usage: /calendar new <name>"
                            " [| <objective>]")
                try:
                    s = store.create_series(
                        name, objective=objective.strip())
                except ValueError as exc:
                    return str(exc)
                kind_word = ("campaign" if s["objective"]
                             else "series")
                return (f"📚 {kind_word} **{s['name']}** started"
                        + (f" — 🎯 {s['objective']}"
                           if s["objective"] else "")
                        + "\nattach pieces: /calendar add"
                          f" {s['name']} draft|short|post <n>")
            if low.startswith("add "):
                parts = text.split()
                # add <series words…> <kind> <n> — the kind is the
                # second-to-last token, the number the last.
                if len(parts) < 4:
                    return ("usage: /calendar add <series>"
                            " draft|short|post <n>")
                kind_word = parts[-2].lower()
                num_word = parts[-1]
                name = " ".join(parts[1:-2])
                s = _series_named(name)
                if s is None:
                    return f"no series named {name!r}."
                if not num_word.isdigit():
                    return "the piece number must be a number — the"
                    " boards show them."
                n = int(num_word)
                if kind_word == "draft":
                    waiting = reviewable_drafts(queue)
                    if not 1 <= n <= len(waiting):
                        return (f"there are {len(waiting)} draft(s)"
                                " on the board — pick one of those"
                                " numbers (see /drafts).")
                    draft = waiting[n - 1]
                    store.attach(s["id"], KIND_DRAFT, draft.id)
                    stamp_draft(queue, draft, s["name"])
                    preview = " ".join(draft.content.split())[:60]
                    return (f"📚 added draft #{n} to"
                            f" **{s['name']}**: {preview}")
                if kind_word == "short":
                    shorts = _shorts()
                    if not 1 <= n <= len(shorts):
                        return (f"there are {len(shorts)} planned"
                                " short(s) — pick one of those"
                                " numbers (see /shorts calendar).")
                    post = shorts[n - 1]
                    store.attach(s["id"], KIND_SHORT, post.id)
                    return (f"📚 added short #{n} to"
                            f" **{s['name']}**: {post.topic}")
                if kind_word == "post":
                    rows = db.query(
                        "SELECT id, platform, content FROM"
                        " social_posts WHERE status = 'scheduled'"
                        " ORDER BY scheduled_at LIMIT 20")
                    if not 1 <= n <= len(rows):
                        return (f"there are {len(rows)} scheduled"
                                " post(s) — see /calendar for the"
                                " timeline.")
                    row = rows[n - 1]
                    store.attach(s["id"], KIND_POST, str(row["id"]))
                    preview = " ".join(
                        str(row["content"]).split())[:60]
                    return (f"📚 added scheduled post #{n} to"
                            f" **{s['name']}**:"
                            f" [{row['platform']}] {preview}")
                return ("kind must be draft, short or post:"
                        " /calendar add <series> draft|short|post <n>")
            return ("usage: /calendar [list] | series [<name>] |"
                    " new <name> [| <objective>] |"
                    " add <series> draft|short|post <n>")
        finally:
            queue.close()

    # ── repurposing: /repurpose ──────────────────────────────────────────────
    def _control_repurpose(self, tail: str, chat_key: str = "") -> Any:
        """/repurpose — one asset, other forms (WS3 P7).

        A posted draft spawns a short script or a thread; a day
        record spawns an adapted draft; a short spawns a text
        post.  Every adaptation runs through the content cell
        (source as evidence, the skeptic number-traces the draft
        against the source) and lands as a NEW draft at
        pending_review on /drafts — the source is never mutated
        and nothing posts (#43).  The board numbers its own
        sources; the verbs resolve against the same lists the
        board shows.  Owner-only via CONTROL_COMMANDS.
        """
        from ..dayrecord import brain_llm_fn, today_key
        from ..repurpose import (
            FORM_POST, FORM_SHORT_SCRIPT, FORM_THREAD,
            day_source, posted_draft_sources, repurpose_source,
            short_sources)

        db = self.context.db
        queue = self._draft_queue()
        form_words = {"short": FORM_SHORT_SCRIPT,
                      "thread": FORM_THREAD, "post": FORM_POST}

        def _run(source, form: str) -> str:
            result = repurpose_source(
                db, queue, source, form,
                llm_fn=brain_llm_fn(self.context))
            label = {"short_script": "short script",
                     "thread": "thread",
                     "post": "text post"}[form]
            if result.get("ok"):
                draft = queue.get(result["draft_id"])
                preview = (" ".join(draft.content.split())[:70]
                           if draft else "")
                return (f"♻️ repurposed {source.label} → {label}"
                        " draft, waiting on /drafts for your"
                        f" review:\n{preview}")
            if result.get("objection"):
                return ("🛑 the skeptic stopped that adaptation —"
                        f" {result['objection']}  Nothing was"
                        " queued; the source stands as it was.")
            return (f"repurposing failed: "
                    f"{result.get('error') or 'unknown error'}")

        try:
            text = (tail or "").strip()
            low = text.lower()
            if not text or low == "list":
                posted = posted_draft_sources(queue)
                shorts = short_sources()
                lines = ["♻️ **repurpose — one asset, other forms**",
                         ""]
                lines.append("📤 **posted drafts** (→ short script"
                             " or thread)")
                if posted:
                    for i, s in enumerate(posted, 1):
                        lines.append(f"  {i}. {s.label}")
                else:
                    lines.append("  (nothing posted yet)")
                lines.append("")
                lines.append("🎬 **shorts** (→ text post)")
                if shorts:
                    for i, s in enumerate(shorts, 1):
                        lines.append(f"  {i}. {s.label}")
                else:
                    lines.append("  (no shorts planned yet)")
                lines.append("")
                lines.append("📔 **day record** — /repurpose day"
                             " [YYYY-MM-DD] short|thread|post"
                             " (today if no date)")
                lines.append("")
                lines.append("verbs: /repurpose post <n>"
                             " short|thread · /repurpose short <n>"
                             " post · every result lands on"
                             " /drafts for review — nothing posts"
                             " itself.")
                return "\n".join(lines)
            parts = text.split()
            head = parts[0].lower()
            if head == "post" and len(parts) == 3:
                if not parts[1].isdigit():
                    return "usage: /repurpose post <n> short|thread"
                form = form_words.get(parts[2].lower())
                if form not in (FORM_SHORT_SCRIPT, FORM_THREAD):
                    return ("a posted draft repurposes into a"
                            " short script or a thread:"
                            " /repurpose post <n> short|thread")
                posted = posted_draft_sources(queue)
                n = int(parts[1])
                if not 1 <= n <= len(posted):
                    return (f"there are {len(posted)} posted"
                            " draft(s) on the board — pick one of"
                            " those numbers.")
                return _run(posted[n - 1], form)
            if head == "short" and len(parts) == 3:
                if not parts[1].isdigit():
                    return "usage: /repurpose short <n> post"
                if parts[2].lower() != "post":
                    return ("a short repurposes into a text post:"
                            " /repurpose short <n> post")
                shorts = short_sources()
                n = int(parts[1])
                if not 1 <= n <= len(shorts):
                    return (f"there are {len(shorts)} short(s) on"
                            " the board — pick one of those"
                            " numbers.")
                return _run(shorts[n - 1], FORM_POST)
            if head == "day":
                day = today_key()
                form_word = "short"
                for token in parts[1:]:
                    if token.lower() in form_words:
                        form_word = token.lower()
                    elif len(token) == 10 and token[4] == "-":
                        day = token
                source = day_source(db, day)
                if source is None:
                    return (f"the {day} day record is empty —"
                            " nothing to repurpose yet.")
                return _run(source, form_words[form_word])
            return ("usage: /repurpose [list] | post <n>"
                    " short|thread | short <n> post |"
                    " day [YYYY-MM-DD] short|thread|post")
        finally:
            queue.close()

    def _control_engage(self, tail: str, chat_key: str = "") -> Any:
        """/engage — replies & comments on his posts (WS3 P8).

        /engage                — the board: what needs him first
                                 (new items, oldest first), then
                                 what he has in hand; platforms
                                 whose replies cannot be read are
                                 NAMED unavailable, never silent
        /engage show <n>       — one item in full (marks it seen)
        /engage reply <n>      — draft a reply onto /drafts at
                                 pending_review — it posts only on
                                 his approval there (#43, unchanged)
        /engage ignore <n>     — dismiss the item
        /engage done <n>       — he handled it himself

        <n> is the number from the board — the one ordering
        board_items() defines, shared with the morning briefing.
        Owner-only via CONTROL_COMMANDS (never PUBLIC_COMMANDS).
        The poll that fills this board is the durable 'social
        engagement poll' job; this surface reads the store and
        reconciles replied state from the queue's own truth.
        """
        from ...social.engagement import (
            STATE_DONE,
            STATE_IGNORED,
            STATE_NEW,
            STATE_TRIAGED,
            EngagementStore,
            board_items,
            draft_reply,
            platform_truth,
            reconcile_replied,
            render_board,
            render_item,
        )
        from ..dayrecord import brain_llm_fn

        db = self.context.db
        store = EngagementStore(db)
        queue = self._draft_queue()
        try:
            reconcile_replied(store, queue)
            items = board_items(store)
            parts = (tail or "").split(None, 1)
            verb = parts[0].lower() if parts else "list"
            rest = parts[1].strip() if len(parts) > 1 else ""

            if verb in ("list", ""):
                truth = None
                try:
                    from ...social.manager import SocialManager

                    manager = SocialManager(self.context)
                    manager.register_builtins()
                    truth = platform_truth(manager.adapters)
                except Exception:  # noqa: BLE001 - board stands alone
                    truth = None
                return render_board(store, truth=truth)

            if verb in ("show", "reply", "ignore", "done"):
                if not rest:
                    return (f"usage: /engage {verb} <n> — the "
                            "number from /engage")
                token = rest.split()[0]
                probe = token[1:] if token.startswith("#") else token
                if not probe.isdigit():
                    return (f"usage: /engage {verb} <n> — the "
                            "number from /engage")
                idx = int(probe)
                if not 1 <= idx <= len(items):
                    if not items:
                        return ("💬 nothing is waiting on you — "
                                "/engage shows the board.")
                    return (f"no item {token} on the board — "
                            f"{len(items)} waiting; /engage lists "
                            "them.")
                item = items[idx - 1]
                if verb == "show":
                    if item["state"] == STATE_NEW:
                        store.set_state(item["id"], STATE_TRIAGED)
                        item = store.get(item["id"]) or item
                    return render_item(item, idx)
                if verb == "ignore":
                    store.set_state(item["id"], STATE_IGNORED)
                    author = item.get("author") or "that item"
                    return (f"ignored — {author} won't come back "
                            "to the board.")
                if verb == "done":
                    store.set_state(item["id"], STATE_DONE)
                    return ("done — marked as handled by you. It "
                            "leaves the board.")
                # reply: draft through the queue; the review flow
                # on /drafts owns every outward step from here.
                from ...social.manager import SocialManager

                manager = SocialManager(self.context)
                manager.register_builtins()
                result = draft_reply(
                    store, queue, item["id"],
                    llm_fn=brain_llm_fn(self.context),
                    adapters=manager.adapters)
                if not result.get("ok"):
                    return f"couldn't draft that reply: "\
                           f"{result.get('error') or 'unknown error'}"
                note = ("threads onto the original post"
                        if result["reply_native"] else
                        "posts as a standalone post — that "
                        "platform has no threaded reply")
                return (f"✍️ reply drafted ({result['drafter']} "
                        f"rung; {note}) — waiting on /drafts for "
                        f"your review:\n{result['text']}")
            return ("usage: /engage [list] | show <n> | reply <n> "
                    "| ignore <n> | done <n> — <n> is the number "
                    "from /engage")
        finally:
            queue.close()

    def _control_cases(self, tail: str) -> str:
        """/cases — outcome stewardship, owner surface (doc 15 P6).

        /cases              — the board: cases needing attention
                              first (oldest first), closed cases,
                              then acknowledged ones quiet at the
                              tail. Numbers are case_board()'s —
                              the one ordering every surface shares
        /cases <n>          — one case in full, with its evidence
                              trail (who checked, from which source,
                              what the evidence established)
        /cases verify <n>   — re-check case <n> NOW: the steward's
                              real investigation cycle for that case,
                              through the delegation substrate — not
                              a shortcut around it
        /cases ack <n>      — mark case <n> seen/accepted: the ack
                              is recorded on the case and the sweep
                              stops re-running it while it is
                              unchanged; acknowledgement NEVER
                              resolves a case — only cited evidence
                              does (P4's resolve path, unchanged)

        Owner-only via CONTROL_COMMANDS (never PUBLIC_COMMANDS).
        Internal ids never print — ordinals and subjects only.
        """
        from ..delegation import DelegationStore
        from ..outcome_cases import OutcomeCaseStore
        from ..outcome_steward import (
            OutcomeSteward,
            case_board,
            render_case_board,
            render_case_detail,
        )

        db = self.context.db
        store = OutcomeCaseStore(db)
        parts = (tail or "").split(None, 1)
        verb = parts[0].lower() if parts else "list"
        rest = parts[1].strip() if len(parts) > 1 else ""

        def resolve_ordinal(word: str):
            board = case_board(store)
            if not word.isdigit():
                return None, ("usage: /cases [list] | <n> | "
                              "verify <n> | ack <n> — <n> is the "
                              "number from /cases")
            n = int(word)
            if not 1 <= n <= len(board):
                return None, (f"case #{n} isn't on the board — "
                              f"{len(board)} case(s) listed")
            return board[n - 1], ""

        def detail(case) -> str:
            contract = None
            if case.contract_id:
                record = DelegationStore(db).get_contract(
                    case.contract_id)
                if record is not None:
                    contract = DelegationStore(db).summary(record)
            return render_case_detail(
                case, store.receipts(case.id), contract)

        if verb in ("list", ""):
            return render_case_board(store)
        if verb == "show" or verb.isdigit():
            case, err = resolve_ordinal(rest if verb == "show"
                                        else verb)
            if case is None:
                return err
            return detail(case)
        if verb == "verify":
            case, err = resolve_ordinal(rest)
            if case is None:
                return err
            if case.state in ("resolved", "closed-unknown"):
                return (f"that case is already closed ({case.state}"
                        f"{', verdict ' + case.verdict if case.state == 'resolved' else ''})"
                        " — a closed-unknown case reopens on its own "
                        "when the work is seen again; nothing to "
                        "re-check by hand.\n" + detail(case))
            result = OutcomeSteward(db).run_case(case.id)
            outcome = result.get("outcome", "")
            if outcome == "resolved":
                head = (f"re-checked — resolved: "
                        f"{result.get('verdict', '')} on the cited "
                        "evidence")
            elif outcome == "closed-unknown":
                head = ("re-checked — the outcome is still honestly "
                        "unknown; the checks are recorded on the case")
            else:
                shown = outcome or "incomplete"
                head = (f"re-check ran — outcome: {shown}; the case "
                        "stays open")
            return head + "\n" + detail(store.get(case.id))
        if verb == "ack":
            case, err = resolve_ordinal(rest)
            if case is None:
                return err
            before = (case.state, case.verdict)
            store.acknowledge(case.id, actor="owner")
            after = store.get(case.id)
            assert (after.state, after.verdict) == before
            return (f"acknowledged — {after.subject} recorded as "
                    f"seen ({after.state}). The sweep will leave it "
                    "quiet unless it changes; acknowledgement does "
                    "not resolve it.")
        return ("usage: /cases [list] | <n> | verify <n> | ack <n> "
                "— <n> is the number from /cases")

    def _runs_board(self) -> dict[str, Any]:
        """Compose the ticket board from the live stores (fix 5).

        Every input is one the projections were built to read: the
        durable records on the context db, the shared job registry
        (the projections' own default), this runtime's tool registry
        and CoreMind when present, the error system's incident
        journal at its production path, the live per-chat immune
        states, and the shared antibody store. A source that cannot
        be built degrades to the projection's own honesty (an
        unread source is NAMED by the coverage, never faked).
        """
        from ..ticket_board import board_rows

        db = self.context.db
        journal = None
        try:
            from ...core.incidents import IncidentJournal
            from ...immune.wiring import default_journal_path

            journal = IncidentJournal(
                path=str(default_journal_path(self.context)))
        except Exception:  # noqa: BLE001 - healing reads degrade named
            journal = None
        immune_states = None
        try:
            from ...immune import chat as _immune_chat

            immune_states = list(_immune_chat._STATES.values())
        except Exception:  # noqa: BLE001 - states stay unread, named
            immune_states = None
        # The healing projection reads an ITERABLE of antibody
        # stores: each live chat state's own store, plus the shared
        # organism store (usually the same object — deduped).
        antibody_stores: list[Any] = []
        seen_stores: set[int] = set()
        for state in immune_states or []:
            store = getattr(state, "store", None)
            if store is not None and id(store) not in seen_stores:
                seen_stores.add(id(store))
                antibody_stores.append(store)
        try:
            from ...immune.wiring import shared_antibody_store

            shared = shared_antibody_store()
            if id(shared) not in seen_stores:
                antibody_stores.append(shared)
        except Exception:  # noqa: BLE001 - store stays unread, named
            pass
        return board_rows(
            db,
            tool_registry=getattr(self, "tools", None),
            coremind=getattr(self, "mind", None),
            journal=journal, immune_states=immune_states,
            antibody_stores=antibody_stores)

    def _control_runs(self, tail: str) -> str:
        """/runs — the live-work ticket board, owner surface
        (public-surface fix 5; doc 18 P2 + doc 20 P1 projections,
        composed by agents/ticket_board.py).

        /runs               — the board: every execution ticket
                              (delegations, missions, registry jobs,
                              browser runs, media jobs, computer
                              watches, mind queue)
                              and healing ticket (incidents, heals,
                              fever cases, antibodies, receipts,
                              chat state) over the live stores —
                              live work first, settled work after
        /runs <n>           — one ticket in full: every field the
                              projection computed, and every absent
                              field WITH its reason (the honesty
                              design — never papered over)
        /runs stop <n> [why] — stop ticket <n> through the ONE
                              cancellation facade (doc 18 P5): the
                              organ's native mechanism runs first
                              and a CancelReceipt is minted from its
                              answer. Healing tickets are not
                              stoppable work — they close through
                              the immune loop, and the surface says
                              so rather than inventing a cancel
        /runs coverage      — the projections' own field-coverage
                              rollup: what the records can and
                              cannot say, per field and source

        Owner-only via CONTROL_COMMANDS (never PUBLIC_COMMANDS).
        Internal ids never print — board ordinals only.
        """
        from ...core.cancellation import cancel_with_receipt
        from ..ticket_board import (
            render_board,
            render_coverage,
            render_ticket,
        )

        db = self.context.db
        board = self._runs_board()
        rows = board["rows"]
        parts = (tail or "").split(None, 1)
        verb = parts[0].lower() if parts else "list"
        rest = parts[1].strip() if len(parts) > 1 else ""

        def resolve_ordinal(word: str):
            if not word.isdigit():
                return None, ("usage: /runs [list] | <n> | stop <n> "
                              "[reason] | coverage — <n> is the "
                              "number from /runs")
            n = int(word)
            if not 1 <= n <= len(rows):
                return None, (f"ticket #{n} isn't on the board — "
                              f"{len(rows)} ticket(s) listed")
            return rows[n - 1], ""

        if verb in ("list", ""):
            return render_board(board)
        if verb == "coverage":
            return render_coverage(board)
        if verb == "show" or verb.isdigit():
            row, err = resolve_ordinal(rest if verb == "show" else verb)
            if row is None:
                return err
            return render_ticket(row)
        if verb == "stop":
            sub = rest.split(None, 1)
            row, err = resolve_ordinal(sub[0] if sub else "")
            if row is None:
                return err
            if row["family"] != "run":
                return (f"{row['label']} is a healing ticket — "
                        "incidents and cases aren't stoppable work; "
                        "they close through the immune loop "
                        "(/immune). Nothing was cancelled.")
            reason = sub[1].strip() if len(sub) > 1 else ""
            browser_manager = None
            jobs_fn = getattr(self, "_browser_run_jobs", None)
            if jobs_fn is not None:
                try:
                    browser_manager = jobs_fn()
                except Exception:  # noqa: BLE001 - facade answers honestly
                    browser_manager = None
            media_manager = None
            try:
                from ...media_edit.jobs import get_manager

                media_manager = get_manager(db)
            except Exception:  # noqa: BLE001 - facade answers honestly
                media_manager = None
            watch_manager = None
            watch_fn = getattr(self, "_computer_watch_manager", None)
            if watch_fn is not None:
                try:
                    watch_manager = watch_fn()
                except Exception:  # noqa: BLE001 - facade answers honestly
                    watch_manager = None
            from ...core.jobs import shared_registry
            from ...missions import MissionStore
            from ..delegation import DelegationStore

            try:
                result = cancel_with_receipt(
                    db, row["ref"], requested_by="owner",
                    reason=reason or "stopped from /runs",
                    delegation=DelegationStore(db),
                    missions=MissionStore(db),
                    registry=shared_registry(),
                    browser_manager=browser_manager,
                    media_manager=media_manager,
                    coremind=getattr(self, "mind", None),
                    watch_manager=watch_manager)
            except Exception as exc:  # noqa: BLE001 - the facade
                # receipts a classifiable organ refusal and then
                # re-raises the organ's own exception unchanged
                # (its doctrine) — the surface renders the organ's
                # words; the receipt, when one was minted, is
                # already durable and rides the ticket.
                return (f"{row['label']}: the organ answered the "
                        f"stop in its own words: {exc} — where the "
                        "refusal could be classified, its receipt "
                        "is on the ticket.")
            return _render_runs_stop(row, result)
        return ("usage: /runs [list] | <n> | stop <n> [reason] | "
                "coverage — <n> is the number from /runs")

    # ── world model: /worldmodel ─────────────────────────────────────
    def _control_worldmodel(self, tail: str) -> str:
        """/worldmodel — the world model board, owner surface (the
        readers pass, 2026-11-10): the provenance-tagged, versioned
        facts the system holds about its codebase, runtime and work
        state (agents/worldmodel.py, Doc 5 §4), fed at the plan
        seam as work settles and read here from the model's own
        persistent board.  (The bare name /world belongs to the
        town game; this board takes /worldmodel.)

        /worldmodel          — the whole map: per-area facts with
                          their kind, source, version and age; a
                          fact unrefreshed for a day reads (stale);
                          an area with no facts does not appear —
                          an absence of record is never rendered as
                          a claim about the world
        /worldmodel <topic>  — every fact under one key prefix, in
                          full, with its version history depth

        Owner-only via CONTROL_COMMANDS (never PUBLIC_COMMANDS).
        Internal ids never print — work facts render by the goal
        and step their values carry."""
        from ..world_board import render_topic, render_world
        from ..worldmodel import default_world

        world = default_world(self.context)
        topic = (tail or "").strip()
        if topic:
            return render_topic(world, topic)
        return render_world(world)

    # ── craft cooperative: /craft ────────────────────────────────────
    def _craft_stores(self) -> tuple[Any, Any]:
        """The craft surface's stores: the filebook library and
        the competence-grant store, at their shipped default
        homes. Built per call, so a pack or grant recorded since
        the last call is read fresh (tests inject seeded ones)."""
        from ...craft.temporary import CompetenceGrantStore
        from ...skills.filebook import FileSkillLibrary

        library = getattr(self, "_craft_library", None)
        if library is None:
            library = FileSkillLibrary()
        grants = getattr(self, "_craft_grants", None)
        if grants is None:
            grants = CompetenceGrantStore()
        return library, grants

    def _control_craft(self, tail: str) -> str:
        """/craft — the craft cooperative board (docs 22+23, the
        chat surface over the shipped craft machinery).

        /craft                — the board: every known domain with
                                the capability level its records
                                DERIVE (levels are computed at read
                                from packs, contracts and receipts
                                — never assigned, never read from a
                                stored token), the facts that earn
                                it, pack states, open items, and
                                absences named in words (a domain
                                known only to the taxonomy seed
                                reads greenfield)
        /craft coverage       — the full measured coverage detail
                                behind the board
        /craft intake <goal>  — a READ-ONLY hire preview: the plan
                                the composer assembles for the goal
                                (the pipeline pattern, or a leading
                                preset name: "intake studio …"),
                                its decomposition, its team (role
                                contracts), the shipped
                                consequential rule's verdict per
                                step IN THE SEAM'S OWN WORDS, risks
                                and cost. It opens nothing, settles
                                nothing and runs nothing — execution
                                stays with the surfaces that run work
        /craft status <field> — one field's capability claim, read
                                from records now: carried by a live
                                pack, carried temporarily by a
                                recorded grant (its expiry and the
                                benchmark it cites named), or unmade
                                with the reason

        Owner-only via CONTROL_COMMANDS (never PUBLIC_COMMANDS).
        Internal ids never print — domains, packs and ordinals
        only. Every render is the craft machinery's own; this
        handler composes the stores and routes the verbs, nothing
        more.
        """
        from ...craft import (
            capability_status,
            coverage_report,
            render_board,
            render_capability_status,
            render_coverage,
        )

        text = (tail or "").strip()
        parts = text.split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""
        library, grants = self._craft_stores()
        db = getattr(self.context, "db", None)
        if verb in ("", "board", "list"):
            return render_board(coverage_report(
                db, library=library, grants=grants))
        if verb == "coverage":
            return render_coverage(coverage_report(
                db, library=library, grants=grants))
        if verb == "status":
            if not rest:
                return ("usage: /craft status <field> — one "
                        "field's capability claim, read from its "
                        "records: a live pack, a temporary grant, "
                        "or unmade with the reason named")
            return render_capability_status(capability_status(
                library, rest, grants=grants))
        if verb == "intake":
            if not rest:
                return ("usage: /craft intake <goal> — a "
                        "read-only preview of the plan the "
                        "composer assembles for the goal: its "
                        "steps, its team, the gate's verdict per "
                        "step, risks and cost. Nothing is opened "
                        "or run")
            return self._craft_intake(rest, library=library,
                                      grants=grants)
        return (f"unknown /craft verb {verb!r} — usage: /craft "
                "[board] | coverage | intake <goal> | "
                "status <field>")

    # ── translation glossaries: /glossary ───────────────────────────
    def _glossary_store(self) -> Any:
        """The shipped glossary store, built the way the
        translator's ticket path builds it (context db, else the
        JSON document under <home>/translate/).  Built per call so
        an edit is read fresh; tests inject a seeded store."""
        store = getattr(self, "_glossary_store_instance", None)
        if store is not None:
            return store
        import os

        from ...llm.glossary import GlossaryStore

        db = getattr(self.context, "db", None)
        settings = getattr(self.context, "settings", None)
        home = (getattr(settings, "home", "")
                or os.path.expanduser("~/.nomorals"))
        return GlossaryStore(db, home)

    def _control_glossary(self, tail: str) -> str:
        """/glossary — the translation glossary board (docs 19+21
        T2's management surface over the shipped store).

        /glossary             — the board: every glossary in the
                                store, its entry count per scope
                                (owner / domain / global — an empty
                                scope is named empty), and where the
                                entries live
        /glossary list [glossary] [scope]
                              — the entries themselves, term →
                                translation, grouped by glossary and
                                scope; a lone scope name lists that
                                scope across every glossary
        /glossary set <glossary> <scope> <term> = <translation>
                              — add or replace one entry through
                                the store's own write path
        /glossary del <glossary> [scope] <term>
                              — remove one entry through the store;
                                an absent term answers in words

        The store's own model, exactly: a glossary is a named set
        (the translator enforces the one a ticket names); inside
        it, a term sits at one scope, and when two rows name the
        same term for a pair the owner scope beats domain beats
        global.  Entries the surface writes apply to any language
        pair.  Every mutation goes through the store — no parallel
        file, no cache edits — and its reply names what changed,
        including the honest consequence of an edit: the glossary's
        fingerprint changes, so translations cached under the old
        fingerprint stop matching and are translated fresh.

        Owner-only via CONTROL_COMMANDS (never PUBLIC_COMMANDS).
        Glossary entries are terms, not ids — nothing internal
        prints.
        """
        store = self._glossary_store()
        text = (tail or "").strip()
        parts = text.split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""
        usage = ("usage: /glossary [list [glossary] [scope]] | "
                 "set <glossary> <scope> <term> = <translation> | "
                 "del <glossary> [scope] <term> — scope is one of "
                 "owner, domain, global")
        if verb in ("", "board"):
            return self._glossary_board(store)
        if verb == "list":
            return self._glossary_list(store, rest)
        if verb == "set":
            return self._glossary_set(store, rest)
        if verb == "del":
            return self._glossary_del(store, rest)
        return f"unknown /glossary verb {verb!r} — {usage}"

    def _glossary_board(self, store: Any) -> str:
        from ...llm.glossary import SCOPES

        lines = ["📖 Translation glossaries",
                 f"entries live in {store.describe_storage()}."]
        ids = store.glossary_ids()
        if not ids:
            lines.append(
                "no glossaries yet — the store is empty. Add the "
                "first entry with /glossary set <glossary> <scope> "
                "<term> = <translation>.")
            return "\n".join(lines)
        total = 0
        for gid in ids:
            rows = store.entries(gid)
            total += len(rows)
            counts = {scope: 0 for scope in SCOPES}
            for row in rows:
                counts[row.scope] = counts.get(row.scope, 0) + 1
            per_scope = " · ".join(
                f"{scope}: {counts[scope]}"
                + (" (empty)" if counts[scope] == 0 else "")
                for scope in ("owner", "domain", "global"))
            noun = "entry" if len(rows) == 1 else "entries"
            lines.append(f"\n{gid} — {len(rows)} {noun}\n  {per_scope}")
        lines.append(
            f"\n{len(ids)} "
            + ("glossary" if len(ids) == 1 else "glossaries")
            + f" · {total} "
            + ("entry" if total == 1 else "entries")
            + " in all. /glossary list shows every entry.")
        return "\n".join(lines)

    def _glossary_list(self, store: Any, rest: str) -> str:
        from ...llm.glossary import SCOPES

        want_glossary = ""
        want_scope = ""
        for token in rest.split():
            if token.lower() in SCOPES and not want_scope:
                want_scope = token.lower()
            elif not want_glossary:
                want_glossary = token
            else:
                return ("usage: /glossary list [glossary] [scope] "
                        "— at most one glossary name and one scope")
        ids = store.glossary_ids()
        if want_glossary and want_glossary not in ids:
            known = ", ".join(ids) if ids else "none yet"
            return (f"no glossary named {want_glossary!r} — the "
                    f"store holds: {known}")
        groups: list[tuple[str, str, list[Any]]] = []
        for gid in ids:
            if want_glossary and gid != want_glossary:
                continue
            rows = store.entries(gid)
            for scope in ("owner", "domain", "global"):
                if want_scope and scope != want_scope:
                    continue
                scoped = [r for r in rows if r.scope == scope]
                if scoped:
                    groups.append((gid, scope, scoped))
        if not groups:
            if want_scope and want_glossary:
                return (f"the {want_glossary} glossary has no "
                        f"entries at {want_scope} scope — that "
                        "scope is empty.")
            if want_scope:
                return (f"no entries at {want_scope} scope in any "
                        "glossary — that scope is empty.")
            if want_glossary:
                return (f"the {want_glossary} glossary holds no "
                        "entries yet.")
            return ("no glossary entries yet — the store is empty. "
                    "Add one with /glossary set <glossary> <scope> "
                    "<term> = <translation>.")
        lines = ["📖 Glossary entries"]
        for gid, scope, rows in groups:
            lines.append(f"\n{gid} · {scope} scope")
            for row in rows:
                pair = ""
                if row.source_lang or row.target_lang:
                    pair = (f"  ({row.source_lang or 'any'}→"
                            f"{row.target_lang or 'any'})")
                lines.append(
                    f"  {row.source_term} → {row.target_term}{pair}")
        return "\n".join(lines)

    def _glossary_set(self, store: Any, rest: str) -> str:
        from ...llm.glossary import SCOPES

        left, sep, target = rest.partition("=")
        target = target.strip()
        tokens = left.split()
        if not sep or len(tokens) < 3 or not target:
            return ("usage: /glossary set <glossary> <scope> "
                    "<term> = <translation> — for example: "
                    "/glossary set legal owner bank = Banco "
                    "Central")
        gid, scope = tokens[0], tokens[1].lower()
        term = " ".join(tokens[2:])
        if scope not in SCOPES:
            return (f"{tokens[1]!r} is not a glossary scope — the "
                    "scopes are owner, domain and global (when two "
                    "rows name the same term, owner beats domain "
                    "beats global). Nothing was written.")
        before = store.fingerprint(gid, "", "")
        old = next((r for r in store.entries(gid)
                    if r.source_term == term
                    and not r.source_lang and not r.target_lang),
                   None)
        if not store.put_term(gid, term, target, scope=scope):
            return (f"the glossary store did not save {term!r} — "
                    "nothing was written and nothing changed.")
        changed = store.fingerprint(gid, "", "") != before
        if old is not None:
            head = (f"replaced in the {gid} glossary ({scope} "
                    f"scope): {term} → {target} (was "
                    f"{old.target_term}, at {old.scope} scope)")
        else:
            head = (f"set in the {gid} glossary ({scope} scope): "
                    f"{term} → {target}")
        tail_note = (
            " The glossary's fingerprint changed with this edit, "
            "so translations cached under the old fingerprint stop "
            "matching and will be translated fresh."
            if changed else
            " The resolved terms are unchanged, so cached "
            "translations still match.")
        return head + "." + tail_note

    def _glossary_del(self, store: Any, rest: str) -> str:
        from ...llm.glossary import SCOPES

        tokens = rest.split()
        if len(tokens) < 2:
            return ("usage: /glossary del <glossary> [scope] "
                    "<term> — for example: /glossary del legal "
                    "owner bank")
        gid = tokens[0]
        scope = ""
        if tokens[1].lower() in SCOPES:
            scope = tokens[1].lower()
            term_tokens = tokens[2:]
        else:
            term_tokens = tokens[1:]
        if not term_tokens:
            return ("usage: /glossary del <glossary> [scope] "
                    "<term> — name the term to remove")
        term = " ".join(term_tokens)
        rows = [r for r in store.entries(gid)
                if r.source_term == term]
        if not rows:
            return (f"there is no {term!r} in the {gid} glossary "
                    "— nothing was removed.")
        if scope and not any(r.scope == scope for r in rows):
            present = ", ".join(sorted({r.scope for r in rows}))
            return (f"{term!r} is in the {gid} glossary at "
                    f"{present} scope, not {scope} — nothing was "
                    "removed.")
        any_pair = [r for r in rows
                    if not r.source_lang and not r.target_lang]
        if not any_pair:
            pairs = ", ".join(sorted(
                f"{r.source_lang or 'any'}→{r.target_lang or 'any'}"
                for r in rows))
            return (f"{term!r} in the {gid} glossary exists only "
                    f"for specific language pairs ({pairs}) — "
                    "/glossary del removes the any-pair entry, so "
                    "nothing was removed.")
        if scope and not any(r.scope == scope for r in any_pair):
            present = ", ".join(sorted({r.scope for r in any_pair}))
            return (f"the any-pair entry for {term!r} in the "
                    f"{gid} glossary sits at {present} scope, not "
                    f"{scope} — nothing was removed.")
        before = store.fingerprint(gid, "", "")
        removed = any_pair[0]
        if not store.remove_term(gid, removed.source_term):
            return (f"the glossary store did not remove {term!r} "
                    "— nothing changed.")
        changed = store.fingerprint(gid, "", "") != before
        head = (f"removed from the {gid} glossary "
                f"({removed.scope} scope): {removed.source_term} "
                f"→ {removed.target_term} is gone")
        tail_note = (
            " The glossary's fingerprint changed with this "
            "removal, so translations cached under the old "
            "fingerprint stop matching and will be translated "
            "fresh."
            if changed else
            " The resolved terms are unchanged, so cached "
            "translations still match.")
        return head + "." + tail_note

    def _craft_intake(self, goal_text: str, *, library: Any,
                      grants: Any) -> str:
        """Compose (never open) the intake preview for one goal.

        The domain is read off the goal's own words against the
        coverage board's domains; a pack recorded for that domain
        governs the composition and must validate — the composer
        refuses in the schema's own words, carried verbatim. The
        preview composes and renders only: no contract is opened,
        nothing settles, no handler runs."""
        import re as _re

        from ...craft import (
            ComposerError,
            compose_plan,
            coverage_report,
            preset_names,
            render_intake,
        )
        from ...skills.packs import load_pack, pack_docs

        def _slug(value: str) -> str:
            return _re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")

        goal = goal_text.strip()
        preset = "pipeline"
        head, sep, tail = goal.partition(" ")
        if sep and head.rstrip(":").lower() in preset_names():
            preset = head.rstrip(":").lower()
            goal = tail.strip()
        db = getattr(self.context, "db", None)
        report = coverage_report(db, library=library, grants=grants)
        tokens = set(_re.findall(r"[a-z0-9]+", goal.lower()))
        row = None
        for candidate in report.domains:
            names = set(candidate.domain.split("-"))
            names |= {w for w in _re.findall(
                r"[a-z0-9]+", candidate.title.lower())
                if len(w) >= 4}
            if names & tokens:
                row = candidate
                break
        pack = None
        if row is not None:
            for doc in pack_docs(library):
                if _slug(str(doc.domain or "")) == row.domain:
                    pack = load_pack(library, doc.id)
                    break
        try:
            plan = compose_plan(
                goal, preset=preset, pack=pack,
                domain=row.domain if row is not None else "")
        except ComposerError as exc:
            return str(exc)
        text = render_intake(plan, row, pack=pack)
        return (text + "\npreview only — this composed the plan "
                "and read the gate's verdicts; it opened no "
                "contract, settled nothing and ran nothing. "
                "Execution stays with the surfaces that run work "
                "(/mission, /delegate).")

    def _desk_bundle_hook(self) -> Any:
        """The gateway's inbound-artifact hook (WS3 P7): when a
        file Devon just stored is a GoldenSwing desk bundle, land
        its one draft.  Recognition rides the bundle contract
        inside agents/desk_drafts — this builder only wires the
        live stores.  None when there is no context db (no
        contracts, no landing)."""
        db = getattr(self.context, "db", None)
        if db is None:
            return None

        def _hook(art: Any, data: bytes) -> Any:
            from ..desk_drafts import land_desk_draft

            store = self._artifact_store()
            if store is None:
                return None
            queue = self._draft_queue()
            try:
                return land_desk_draft(
                    db, queue, store, data, source="chat_inbound")
            finally:
                queue.close()

        return _hook

    def _calendar_piece_label(self, member: dict, *, queue: Any,
                              shorts: list, db: Any) -> str:
        """One member of a series, named by what it IS — never by
        its internal id. A piece that no longer exists is named
        honestly as gone."""
        kind, ref = member["kind"], member["ref"]
        if kind == "draft":
            draft = queue.get(ref)
            if draft is None:
                return "📝 draft — gone"
            preview = " ".join(draft.content.split())[:60]
            return f"📝 draft [{draft.status}] — {preview}"
        if kind == "short":
            for p in shorts:
                if p.id == ref:
                    return f"🎬 short [{p.status}] — {p.topic}"
            return "🎬 short — gone"
        if kind == "post":
            rows = db.query(
                "SELECT platform, content, status FROM social_posts"
                " WHERE id = ?", (ref,))
            if not rows:
                return "📤 post — gone"
            r = rows[0]
            preview = " ".join(str(r["content"]).split())[:60]
            return f"📤 {r['platform']} [{r['status']}] — {preview}"
        return f"{kind} — gone"

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
        from ...social.chat.present import canonical_text
        from ..job_reports import tool_result_doc

        return canonical_text(tool_result_doc(name, outcome.value))

    # ── connector management: /connectors ────────────────────────────────
    def _connectors_vault(self) -> Any:
        """The encrypted vault for chat-side connector management.

        Same vault `nm connectors` uses.  Opens with ANY owner
        credential — the power key (partner.owner_key), a presented
        owner seal passphrase, or the legacy NM_VAULT_PASSPHRASE —
        via :func:`nomorals.accounts.vault.open_vault`.  Raises a
        plain, actionable message (naming every source checked) when
        no credential can open it.
        """
        from ...accounts.vault import VaultLockedError, open_vault

        db = getattr(self.context, "db", None)
        if db is None:
            raise _ConnectorsChatError(
                "no database is attached to this chat context — connector "
                "management needs the vault database")
        try:
            return open_vault(db, context=self.context,
                              actor="connectors-chat")
        except VaultLockedError as exc:
            raise _ConnectorsChatError(str(exc)) from exc

    def _control_connectors(self, tail: str) -> str:
        """Manage the service connectors from chat (whatever the registry holds).

        /connectors                  — status table (vault state, no network)
        /connectors list             — same
        /connectors status <name>    — live status check for one connector
        /connectors connect <name>   — guided connect flow

        Never raises: every failure becomes a chat-readable message. Secrets
        are never echoed: the guided flow resolves keys from environment
        variables only — the key never travels through chat.
        """
        import difflib
        import io
        from contextlib import redirect_stdout

        def _usage() -> str:
            return (
                "/connectors [list] — every connector + connected state\n"
                "/connectors status <name> — live status check\n"
                "/connectors connect <name> — guided connect flow\n"
                "keys come from environment variables (never paste them in "
                "chat); e.g. export LEONARDO_API_KEY=… then "
                "/connectors connect leonardo")

        try:
            from ...connectors import create_connector, list_connectors
            from ...connectors.base import ConnectorError
            from ...core.http import HttpClient
        except Exception as exc:  # noqa: BLE001 - surfaced as text
            return f"connectors unavailable: {exc}"

        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else "list"
        name = parts[1].strip() if len(parts) > 1 else ""
        if verb not in ("list", "status", "connect"):
            # tolerate "/connectors <name>" as a status shortcut
            if not name:
                name, verb = verb, "status"
            else:
                return _usage()

        try:
            vault = self._connectors_vault()
        except _ConnectorsChatError as exc:
            return f"connectors: {exc}"
        except Exception as exc:  # noqa: BLE001 - surfaced as text
            return f"connectors: could not open the vault: {exc}"

        try:
            infos = list_connectors()
        except Exception as exc:  # noqa: BLE001 - surfaced as text
            return f"connectors: could not list connectors: {exc}"
        known = [c["id"] for c in infos]

        def _resolve(target: str) -> str:
            t = (target or "").strip().lower()
            if t in known:
                return t
            hint = difflib.get_close_matches(t, known, n=1, cutoff=0.6)
            raise _ConnectorsChatError(
                f"unknown connector {target!r}"
                + (f" — did you mean {hint[0]!r}?" if hint else
                   "") + f" — /connectors list shows all {len(known)}")

        http = HttpClient()
        if verb == "list":
            lines = [f"connectors ({len(infos)}):"]
            for info in infos:
                cid = info["id"]
                try:
                    conn = create_connector(cid, vault, http=http)
                    cred = conn._load_credential()
                    mark = f"✅ {cred.username}" if cred else "❌"
                except Exception as exc:  # noqa: BLE001 - one bad adapter ≠ dead table
                    record_failure("runtime_system._control_connectors:859", exc)
                    mark = "⚠️"
                lines.append(f"  {mark:24s} {cid:18s} {info['name']}")
            lines.append("detail: /connectors status <name> · "
                         "connect: /connectors connect <name>")
            return "\n".join(lines)

        # status / connect need a concrete connector
        if not name:
            return f"usage: /connectors {verb} <name>"
        try:
            cid = _resolve(name)
        except _ConnectorsChatError as exc:
            return f"connectors: {exc}"
        try:
            conn = create_connector(cid, vault, http=http)
        except ConnectorError as exc:
            return f"connectors: {exc}"
        except Exception as exc:  # noqa: BLE001 - surfaced as text
            return f"connectors: could not build {cid}: {exc}"

        if verb == "status":
            try:
                st = conn.status()
            except Exception as exc:  # noqa: BLE001 - surfaced as text
                return f"{cid}: status check failed: {exc}"
            state = ("connected" + (f" as {st.account}" if st.account else "")
                     if st.connected else "not connected")
            detail = f" — {st.detail}" if st.detail else ""
            scopes = (f"  scopes: {', '.join(st.scopes)}"
                      if st.scopes else "")
            return f"{conn.name} ({cid}): {state}{detail}\n{scopes}".rstrip()

        # verb == "connect": the guided flow
        try:
            existing = conn._load_credential()
        except Exception as exc:  # noqa: BLE001 - treat as not connected
            record_failure("runtime_system._control_connectors:895", exc)
            existing = None
        if existing is not None:
            return (f"{conn.name} is already connected as {existing.username} "
                    f"— one account per service. Disconnect first "
                    f"(`nm connectors disconnect --name {cid}`) to switch.")
        # Connectors resolve their own secrets: env var first, else a clear
        # non-TTY error naming the exact variable to set. Some (OAuth) print
        # a grant guide — capture it into the reply instead of stdout.
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                result = conn.connect()
        except ConnectorError as exc:
            guide = buf.getvalue().strip()
            msg = str(exc)
            if guide:
                msg += "\n" + guide[:1500]
            return (f"connect {cid} needs input I can't take in chat:\n{msg}\n"
                    f"set the key in the bot's environment, then run "
                    f"/connectors connect {cid} again — the key is never "
                    f"echoed and never travels through chat.")
        except Exception as exc:  # noqa: BLE001 - surfaced as text
            return f"connect {cid} failed: {exc}"
        guide = buf.getvalue().strip()
        if result.ok:
            return (f"✅ {result.message or f'connected as {result.account}'}"
                    + (f"\n{guide[:1500]}" if guide else ""))
        # ok=False: the connector parked the flow on the owner (e.g. OAuth
        # grant guide) — relay its instructions verbatim.
        out = result.message or f"{cid}: connect needs owner input"
        if guide:
            out += "\n" + guide[:1500]
        return out + ("\nfinish it on the terminal with "
                      f"`nm connectors connect --name {cid}` if the chat "
                      "can't complete it.")

    # ── shared connector builder for the chat commands below ───────────
    def _chat_connector(self, connector_id: str) -> Any:
        """A connector instance for chat commands; clear message when the
        vault is locked or the connector isn't connected."""
        from ...connectors import create_connector
        from ...connectors.base import ConnectorError
        from ...core.http import HttpClient

        vault = self._connectors_vault()
        try:
            conn = create_connector(connector_id, vault,
                                    http=HttpClient())
        except ConnectorError as exc:
            raise _ConnectorsChatError(str(exc)) from exc
        try:
            if conn._load_credential() is None:
                raise _ConnectorsChatError(
                    f"{connector_id} is not connected — run "
                    f"/connectors connect {connector_id} first")
        except _ConnectorsChatError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as text
            raise _ConnectorsChatError(
                f"{connector_id}: credential check failed: {exc}") from exc
        return conn

    # ── notion: /notion ────────────────────────────────────────────────
    def _control_notion(self, tail: str) -> str:
        """/notion dbs [query] | query <db> [text] | add <page-id> <title> [| <body>]

        Reads and page-writes through the Notion connector. Writes are the
        owner's explicit command (confirmed=True); reads need no approval.
        Never raises — every failure becomes a chat-readable message.
        """
        def _usage() -> str:
            return (
                "/notion dbs [query] — databases shared with the integration\n"
                "/notion query <database-id-or-name> [text] — rows (first 10)\n"
                "/notion add <parent-page-id> <title> [| <body>] — new page\n"
                "share pages/databases with the integration in Notion "
                "(⋯ → Connections) or they won't appear")

        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""
        if verb not in ("dbs", "query", "add"):
            return _usage()
        try:
            conn = self._chat_connector("notion")
        except _ConnectorsChatError as exc:
            return f"notion: {exc}"
        try:
            if verb == "dbs":
                dbs = conn.list_databases(query=rest)
                if not dbs:
                    return ("no databases visible — share them with the "
                            "integration in Notion (⋯ → Connections)")
                lines = ["notion databases:"]
                for db in dbs[:20]:
                    title = self._notion_title_of(db)
                    lines.append(f"  {db.get('id', '?')[:8]}…  {title}")
                return "\n".join(lines)
            if verb == "query":
                toks = rest.split(None, 1)
                if not toks:
                    return "usage: /notion query <database-id-or-name> [text]"
                db_id = self._notion_resolve_db(conn, toks[0])
                text = toks[1] if len(toks) > 1 else ""
                # schema varies per database — filter client-side on title
                data = conn.query_database(db_id, page_size=10)
                rows = data.get("results", []) if isinstance(data, dict) else []
                if text:
                    low = text.lower()
                    rows = [r for r in rows
                            if low in self._notion_title_of(r).lower()]
                if not rows:
                    return "no rows" + (f" matching {text!r}" if text else "")
                lines = [f"notion rows ({len(rows)}):"]
                for r in rows:
                    lines.append(f"  • {self._notion_title_of(r)}")
                return "\n".join(lines)
            # verb == "add"
            head, *body_segs = [s.strip() for s in rest.split("|")]
            hparts = head.split(None, 1)
            if len(hparts) < 2 or not hparts[0] or not hparts[1].strip():
                return ("usage: /notion add <parent-page-id> <title> "
                        "[| <body>] — separate body paragraphs with |")
            parent_id, title = hparts[0], hparts[1].strip()
            children = [self._notion_paragraph(p) for p in body_segs
                        if p.strip()]
            # the owner typed the exact page — explicit approval
            page = conn.create_page(parent_page_id=parent_id, title=title,
                                    children=children, confirmed=True)
            url = page.get("url", "")
            return (f"📝 notion page created: {title}"
                    + (f"\n{url}" if url else ""))
        except _ConnectorsChatError as exc:
            return f"notion: {exc}"
        except Exception as exc:  # noqa: BLE001 - surfaced as text
            return f"notion failed: {exc}"

    @staticmethod
    def _notion_title_of(obj: dict[str, Any]) -> str:
        """Best-effort title of a Notion page/database object.

        Systematic: database objects carry a plain "title" list; pages
        carry it under properties → first property of type "title".
        """
        try:
            title = obj.get("title")
            if isinstance(title, list):
                text = "".join(t.get("plain_text", "") for t in title
                               if isinstance(t, dict)).strip()
                if text:
                    return text
            props = obj.get("properties") or {}
            for _name, prop in props.items():
                if isinstance(prop, dict) and prop.get("type") == "title":
                    text = "".join(t.get("plain_text", "")
                                   for t in prop.get("title", [])
                                   if isinstance(t, dict)).strip()
                    if text:
                        return text
            # fall back to the first non-empty rich text we can find
            for prop in (props.values() if isinstance(props, dict) else []):
                if isinstance(prop, dict) and prop.get("type") == "rich_text":
                    text = "".join(t.get("plain_text", "")
                                   for t in prop.get("rich_text", [])
                                   if isinstance(t, dict)).strip()
                    if text:
                        return text[:80]
        except Exception as exc:  # noqa: BLE001 - best effort only
            record_failure("runtime_system._notion_title_of", exc)
        return "(untitled)"

    def _notion_resolve_db(self, conn: Any, ref: str) -> str:
        """Database id, or unique name match (case-insensitive)."""
        ref = (ref or "").strip()
        if not ref:
            raise _ConnectorsChatError("database id or name is required")
        dbs = conn.list_databases(query="")
        for db in dbs:
            if db.get("id", "").replace("-", "") == ref.replace("-", ""):
                return db["id"]
        matches = [db for db in dbs
                   if self._notion_title_of(db).lower() == ref.lower()]
        if not matches:
            matches = [db for db in dbs
                       if ref.lower() in self._notion_title_of(db).lower()]
        if len(matches) == 1:
            return matches[0]["id"]
        if len(matches) > 1:
            names = ", ".join(self._notion_title_of(m) for m in matches[:5])
            raise _ConnectorsChatError(
                f"{ref!r} matches several databases: {names} — be specific")
        raise _ConnectorsChatError(
            f"no database {ref!r} — /notion dbs lists what's shared")

    @staticmethod
    def _notion_paragraph(text: str) -> dict[str, Any]:
        """One Notion paragraph block (the documented block shape)."""
        return {
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [
                {"type": "text", "text": {"content": text[:2000]}}]},
        }

    # ── google calendar: /gcal ─────────────────────────────────────────
    def _control_gcal(self, tail: str) -> str:
        """/gcal | agenda [days] | add <summary> | <start> | <end-or-minutes>

        Reads and event-writes through the Google Calendar connector.
        Times: "YYYY-MM-DD HH:MM" or "YYYY-MM-DD"; naive times are read in
        the owner's timezone. Writes are the owner's explicit command.
        Never raises.
        """
        def _usage() -> str:
            return (
                "/gcal — today's agenda\n"
                "/gcal agenda [days] — upcoming days (default 7)\n"
                "/gcal add <summary> | <YYYY-MM-DD HH:MM> | <end or minutes>\n"
                '  e.g. /gcal add Dentist | 2026-10-10 14:00 | 60')

        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""
        if verb in ("", "agenda"):
            days = 7
            if verb == "agenda" and rest.strip():
                try:
                    days = max(1, min(int(rest.strip().split()[0]), 31))
                except ValueError:
                    return "usage: /gcal agenda [days]"
            return self._gcal_agenda(0 if verb == "" else days,
                                     today_only=(verb == ""))
        if verb == "add":
            return self._gcal_add(rest)
        return _usage()

    def _gcal_agenda(self, days: int, today_only: bool) -> str:
        try:
            conn = self._chat_connector("gcalendar")
        except _ConnectorsChatError as exc:
            return f"gcal: {exc}"
        try:
            from ..weather import now_in_tz, owner_tz
            tz_name = owner_tz(getattr(self.context, "settings", None))
            now = now_in_tz(tz_name)
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if today_only:
                end = start.replace(hour=23, minute=59, second=59)
                label = "today"
            else:
                from datetime import timedelta
                end = start + timedelta(days=days)
                label = f"next {days} day(s)"
            data = conn.list_events(
                time_min=start.isoformat(), time_max=end.isoformat(),
                max_results=50)
            events = data.get("events", []) if isinstance(data, dict) else []
            if not events:
                return f"📅 nothing on the calendar {label}"
            lines = [f"📅 {label}:"]
            for ev in events[:20]:
                lines.append("  " + self._gcal_line(ev, tz_name))
            if len(events) > 20:
                lines.append(f"  … {len(events) - 20} more")
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001 - surfaced as text
            return f"gcal failed: {exc}"

    @staticmethod
    def _gcal_line(ev: dict[str, Any], tz_name: str) -> str:
        """One agenda line: 'Mon 14:00 — summary' in the owner's tz."""
        summary = str(ev.get("summary") or "(no title)")
        start = (ev.get("start") or {})
        when = start.get("dateTime") or start.get("date") or ""
        try:
            from datetime import datetime
            from ..weather import _zone
            dt = datetime.fromisoformat(str(when).replace("Z", "+00:00"))
            if dt.tzinfo is not None:
                dt = dt.astimezone(_zone(tz_name))
            when = dt.strftime("%a %H:%M")
        except Exception as exc:  # noqa: BLE001 - keep the raw value
            record_failure("runtime_system._gcal_line", exc)
        loc = f" @ {ev['location']}" if ev.get("location") else ""
        return f"{when} — {summary}{loc}"

    def _gcal_add(self, rest: str) -> str:
        segs = [s.strip() for s in (rest or "").split("|")]
        if len(segs) < 3 or not all(segs[:3]):
            return ("usage: /gcal add <summary> | <YYYY-MM-DD HH:MM> | "
                    "<end YYYY-MM-DD HH:MM or duration minutes>")
        summary, start_raw, end_raw = segs[0], segs[1], segs[2]
        try:
            from ..weather import owner_tz
            tz_name = owner_tz(getattr(self.context, "settings", None))
            start = self._gcal_parse_time(start_raw, tz_name)
            try:
                minutes = int(end_raw)
                from datetime import timedelta
                end = start + timedelta(minutes=minutes)
            except ValueError:
                end = self._gcal_parse_time(end_raw, tz_name)
            if end <= start:
                return "gcal: the end must be after the start"
        except _ConnectorsChatError as exc:
            return f"gcal: {exc}"
        try:
            conn = self._chat_connector("gcalendar")
        except _ConnectorsChatError as exc:
            return f"gcal: {exc}"
        try:
            # the owner typed the exact event — explicit approval
            ev = conn.create_event(
                summary,
                {"dateTime": start.isoformat()},
                {"dateTime": end.isoformat()},
                confirmed=True)
            link = ev.get("htmlLink", "")
            return (f"📅 event created: {summary}\n"
                    f"{self._gcal_line(ev, tz_name)}"
                    + (f"\n{link}" if link else ""))
        except Exception as exc:  # noqa: BLE001 - surfaced as text
            return f"gcal: could not create the event: {exc}"

    @staticmethod
    def _gcal_parse_time(raw: str, tz_name: str):
        """Parse "YYYY-MM-DD [HH:MM]"; naive → owner's timezone."""
        from datetime import datetime
        from ..weather import _zone
        raw = (raw or "").strip()
        dt = None
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            raise _ConnectorsChatError(
                f"can't parse time {raw!r} — use YYYY-MM-DD HH:MM")
        return dt.replace(tzinfo=_zone(tz_name))

    # ── trello: /trello ────────────────────────────────────────────────
    def _control_trello(self, tail: str) -> str:
        """/trello boards | lists <board> | cards <list> | add <list> | <name> [| <desc>]

        Boards/lists/cards read, card writes through the Trello connector.
        <board> and <list> accept ids or names (unique match). Writes are
        the owner's explicit command. Never raises.
        """
        def _usage() -> str:
            return (
                "/trello boards — your boards\n"
                "/trello lists <board-id-or-name> — lists on a board\n"
                "/trello cards <list-id-or-name> — cards on a list\n"
                "/trello add <list-id-or-name> | <card name> [| <desc>]")

        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""
        if verb not in ("boards", "lists", "cards", "add"):
            return _usage()
        try:
            conn = self._chat_connector("trello")
        except _ConnectorsChatError as exc:
            return f"trello: {exc}"
        try:
            if verb == "boards":
                boards = conn.list_boards()
                if not boards:
                    return "no open boards"
                lines = ["trello boards:"]
                for b in boards[:20]:
                    lines.append(f"  {b.get('id', '?')}  {b.get('name', '?')}")
                return "\n".join(lines)
            if verb == "lists":
                if not rest.strip():
                    return "usage: /trello lists <board-id-or-name>"
                board_id = self._trello_resolve_board(conn, rest.strip())
                lists = conn.list_lists(board_id)
                if not lists:
                    return "no lists on that board"
                lines = ["trello lists:"]
                for li in lists:
                    lines.append(
                        f"  {li.get('id', '?')}  {li.get('name', '?')}")
                return "\n".join(lines)
            if verb == "cards":
                if not rest.strip():
                    return "usage: /trello cards <list-id-or-name>"
                list_id = self._trello_resolve_list(conn, rest.strip())
                cards = conn.list_cards(list_id)
                if not cards:
                    return "no cards on that list"
                lines = ["trello cards:"]
                for c in cards[:20]:
                    due = f" (due {c['due'][:10]})" if c.get("due") else ""
                    lines.append(f"  • {c.get('name', '?')}{due}")
                return "\n".join(lines)
            # verb == "add"
            segs = [s.strip() for s in rest.split("|")]
            if len(segs) < 2 or not segs[0] or not segs[1]:
                return ("usage: /trello add <list-id-or-name> | <card name> "
                        "[| <desc>]")
            list_id = self._trello_resolve_list(conn, segs[0])
            desc = segs[2] if len(segs) > 2 else ""
            # the owner typed the exact card — explicit approval
            card = conn.create_card(list_id, segs[1], description=desc,
                                    confirmed=True)
            url = card.get("url", "")
            return (f"📌 trello card created: {segs[1]}"
                    + (f"\n{url}" if url else ""))
        except _ConnectorsChatError as exc:
            return f"trello: {exc}"
        except Exception as exc:  # noqa: BLE001 - surfaced as text
            return f"trello failed: {exc}"

    def _trello_resolve_board(self, conn: Any, ref: str) -> str:
        """Board id, or unique name match (case-insensitive)."""
        ref = (ref or "").strip()
        boards = conn.list_boards()
        for b in boards:
            if b.get("id") == ref:
                return b["id"]
        matches = [b for b in boards
                   if str(b.get("name", "")).lower() == ref.lower()]
        if not matches:
            matches = [b for b in boards
                       if ref.lower() in str(b.get("name", "")).lower()]
        if len(matches) == 1:
            return matches[0]["id"]
        if len(matches) > 1:
            names = ", ".join(str(m.get("name")) for m in matches[:5])
            raise _ConnectorsChatError(
                f"{ref!r} matches several boards: {names} — be specific")
        raise _ConnectorsChatError(
            f"no board {ref!r} — /trello boards lists them")

    def _trello_resolve_list(self, conn: Any, ref: str) -> str:
        """List id, or unique name match across all boards."""
        ref = (ref or "").strip()
        if not ref:
            raise _ConnectorsChatError("list id or name is required")
        # fast path: looks like a Trello id — try it directly
        if len(ref) == 24 and all(c in "0123456789abcdefABCDEF"
                                 for c in ref):
            return ref
        found: list[tuple[str, str]] = []
        for b in conn.list_boards():
            try:
                lists = conn.list_lists(b.get("id", ""))
            except Exception as exc:  # noqa: BLE001 - skip unreadable boards
                record_failure("runtime_system._trello_resolve_list", exc)
                continue
            for li in lists:
                name = str(li.get("name", ""))
                if name.lower() == ref.lower() or ref.lower() in name.lower():
                    found.append((li.get("id", ""), name))
        exact = [f for f in found if f[1].lower() == ref.lower()]
        pool = exact or found
        if len(pool) == 1:
            return pool[0][0]
        if len(pool) > 1:
            names = ", ".join(f[1] for f in pool[:5])
            raise _ConnectorsChatError(
                f"{ref!r} matches several lists: {names} — use the list id")
        raise _ConnectorsChatError(
            f"no list {ref!r} — /trello lists <board> shows ids")

    # ── exness: /exness ────────────────────────────────────────────────
    def _control_exness(self, tail: str) -> str:
        """/exness balance | positions | price <symbol> [timeframe] |
        buy <symbol> <lots> [sl] [tp] | sell <symbol> <lots> [sl] [tp] |
        close <position-id> | closeall

        Trading reads + order placement through the Exness connector.
        Orders are the owner's explicit command (confirmed=True); reads
        need no approval. Never raises — failures become chat text.
        """
        def _usage() -> str:
            return (
                "/exness balance — account balance / equity / margin\n"
                "/exness positions — open positions + pending orders\n"
                "/exness price <symbol> [timeframe] — latest candles\n"
                "/exness buy <symbol> <lots> [sl] [tp] — market buy\n"
                "/exness sell <symbol> <lots> [sl] [tp] — market sell\n"
                "/exness close <position-id> — close one position\n"
                "/exness closeall — close EVERYTHING (asks first)")

        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""
        if verb not in ("balance", "positions", "price", "buy", "sell",
                        "close", "closeall"):
            return _usage()
        try:
            conn = self._chat_connector("exness")
        except _ConnectorsChatError as exc:
            return f"exness: {exc}"
        try:
            if verb == "balance":
                snap = conn.get_snapshot()
                state = snap.get("account_state") or {}
                return (
                    "exness balance:\n"
                    f"  balance: {state.get('balance', '?')}\n"
                    f"  equity: {state.get('equity', '?')}\n"
                    f"  used margin: {state.get('used_margin', '?')}")
            if verb == "positions":
                snap = conn.get_snapshot()
                lines = ["exness positions:"]
                positions = snap.get("positions") or []
                orders = snap.get("orders") or []
                if not positions and not orders:
                    return "no open positions or pending orders"
                for p in positions:
                    lines.append(
                        f"  #{p.get('position_id')} "
                        f"{p.get('direction', '?').upper()} "
                        f"{p.get('volume', '?')} {p.get('instrument', '?')} "
                        f"@ {p.get('open_price', '?')}")
                for o in orders[:10]:
                    lines.append(
                        f"  ⏳ {o.get('order_id')} {o.get('order_type', '?')} "
                        f"{o.get('direction', '?')} "
                        f"{o.get('instrument', '?')} @ {o.get('price', '?')}")
                return "\n".join(lines)
            if verb == "price":
                toks = rest.split()
                if not toks:
                    return "usage: /exness price <symbol> [timeframe]"
                symbol = toks[0].upper()
                tf = (toks[1] if len(toks) > 1 else "H1").upper()
                candles = conn.get_candles(symbol, tf, count=3)
                if not candles:
                    return f"no candles for {symbol} {tf}"
                last = candles[-1]
                return (
                    f"{symbol} {tf}: {last.get('close')} "
                    f"(o {last.get('open')} h {last.get('high')} "
                    f"l {last.get('low')}) @ {last.get('time', '?')}")
            if verb in ("buy", "sell"):
                toks = rest.split()
                if len(toks) < 2:
                    return f"usage: /exness {verb} <symbol> <lots> [sl] [tp]"
                symbol = toks[0].upper()
                try:
                    lots = float(toks[1])
                except ValueError:
                    return f"lots must be a number, got {toks[1]!r}"
                sl = float(toks[2]) if len(toks) > 2 else None
                tp = float(toks[3]) if len(toks) > 3 else None
                # the owner typed the exact order — explicit approval
                out = conn.open_position(symbol, verb, lots, stop_loss=sl,
                                         take_profit=tp, confirmed=True)
                op = out.get("operation_id", "?")
                return (f"📈 exness {verb} {lots} {symbol} accepted "
                        f"(operation {op}) — poll with "
                        f"`nm connectors` or /exness positions")
            if verb == "close":
                pid = rest.strip()
                if not pid:
                    return "usage: /exness close <position-id>"
                out = conn.close_position(pid, confirmed=True)
                return (f"✅ exness close {pid} accepted "
                        f"(operation {out.get('operation_id', '?')})")
            # verb == "closeall" — destructive: require the explicit word
            if rest.strip().lower() != "confirm":
                return ("closing ALL positions is destructive — type "
                        "`/exness closeall confirm` to do it for real.")
            out = conn.close_all_positions(confirmed=True)
            return (f"🔥 exness closed all positions "
                    f"(operation {out.get('operation_id', '?')})")
        except _ConnectorsChatError as exc:
            return f"exness: {exc}"
        except Exception as exc:  # noqa: BLE001 - surfaced as text
            return f"exness failed: {exc}"

    # ── stripe: /stripe ────────────────────────────────────────────────
    def _control_stripe(self, tail: str) -> str:
        """/stripe balance | customers [email] | charges | link <amount>
        <currency> <label>

        Payments reads + checkout-link creation through the Stripe
        connector. Link creation is the owner's explicit command
        (confirmed where the connector requires it); reads need no
        approval. Never raises.
        """
        def _usage() -> str:
            return (
                "/stripe balance — available + pending balances\n"
                "/stripe customers [email] — customer list\n"
                "/stripe charges — recent charges\n"
                "/stripe link <amount> <currency> <label> — hosted "
                "checkout link (amount in major units, e.g. 25.50)")

        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""
        if verb not in ("balance", "customers", "charges", "link"):
            return _usage()
        try:
            conn = self._chat_connector("stripe")
        except _ConnectorsChatError as exc:
            return f"stripe: {exc}"
        try:
            if verb == "balance":
                bal = conn.get_balance()
                lines = ["stripe balance:"]
                for bucket in ("available", "pending"):
                    for b in bal.get(bucket, []):
                        lines.append(
                            f"  {bucket}: {b.get('amount', 0) / 100:,.2f} "
                            f"{b.get('currency', '').upper()}")
                return "\n".join(lines)
            if verb == "customers":
                customers = conn.list_customers(
                    limit=10, email=rest.strip())
                if not customers:
                    return "no customers found"
                lines = ["stripe customers:"]
                for c in customers:
                    lines.append(
                        f"  {c.get('id')}  "
                        f"{c.get('name') or c.get('email') or '?'}")
                return "\n".join(lines)
            if verb == "charges":
                charges = conn.list_charges(limit=10)
                if not charges:
                    return "no charges yet"
                lines = ["stripe charges:"]
                for ch in charges:
                    lines.append(
                        f"  {ch.get('id')}  "
                        f"{ch.get('amount', 0) / 100:,.2f} "
                        f"{ch.get('currency', '').upper()}  "
                        f"{ch.get('status', '?')}")
                return "\n".join(lines)
            # verb == "link"
            toks = rest.split(None, 2)
            if len(toks) < 3:
                return ("usage: /stripe link <amount> <currency> <label> "
                        "— e.g. /stripe link 25.50 usd logo gig")
            try:
                major = float(toks[0])
            except ValueError:
                return f"amount must be a number, got {toks[0]!r}"
            if major <= 0:
                return "amount must be positive"
            currency, label = toks[1].lower(), toks[2].strip()
            minor = int(round(major * 100))
            link = conn.create_payment_link([{
                "price_data": {
                    "currency": currency,
                    "unit_amount": minor,
                    "product_data": {"name": label},
                },
                "quantity": 1,
            }])
            return (f"🔗 stripe checkout link ({major:,.2f} "
                    f"{currency.upper()} — {label}):\n{link.get('url', '?')}")
        except _ConnectorsChatError as exc:
            return f"stripe: {exc}"
        except Exception as exc:  # noqa: BLE001 - surfaced as text
            return f"stripe failed: {exc}"

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
