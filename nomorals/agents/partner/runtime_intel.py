"""RuntimeIntelMixin: PartnerRuntime command group (intel)."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable
from ...core.text import truncate


#: natural-language duration: "in the next 2 minutes", "for 1 hour", "next 30s"
_NL_DURATION_RE = re.compile(
    r"\b(?:in\s+the\s+next|for(?:\s+the\s+next)?|next)\s+"
    r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)\b",
    re.I,
)
#: natural-language interval: "every 30 seconds", "each 5 min"
_NL_INTERVAL_RE = re.compile(
    r"\bevery\s+(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b",
    re.I,
)
#: leading verbs to strip: "watch btc", "monitor eth price", "track gold"
_NL_LEAD_VERB_RE = re.compile(
    r"^(?:please\s+)?(?:watch|monitor|track|keep\s+(?:an\s+)?eye\s+on|follow)\b\s*",
    re.I,
)
#: trailing filler words that describe the watch, not the target
_NL_FILLER_RE = re.compile(
    r"\s+(movement|movements|changes?|updates?|activity)$", re.I)

_TIME_UNIT_S = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}


def _nl_seconds(amount: str, unit: str) -> float:
    unit = unit.lower().rstrip("s")
    # normalize plurals: "seconds" -> "second" etc. via prefix match
    for key, mult in _TIME_UNIT_S.items():
        if key.startswith(unit) or unit.startswith(key):
            return float(amount) * mult
    return float(amount) * 60.0


def parse_monitor_nl(tail: str) -> dict[str, Any] | None:
    """Parse natural-language monitor requests.

    "btc price movement in the next 2 minutes"
      → {"target": "btc price", "duration_s": 120.0, "interval_s": 30.0}
    "watch eth every 30 seconds"
      → {"target": "eth", "interval_s": 30.0}
    Returns None when the text doesn't look like a monitor request.
    """
    text = (tail or "").strip()
    if not text or text.startswith("/"):
        return None
    # must contain a monitor-ish verb or a duration/interval phrase
    has_verb = bool(_NL_LEAD_VERB_RE.search(text))
    dur_m = _NL_DURATION_RE.search(text)
    int_m = _NL_INTERVAL_RE.search(text)
    if not (has_verb or dur_m or int_m):
        return None

    duration_s = _nl_seconds(dur_m.group(1), dur_m.group(2)) if dur_m else 0.0
    interval_s = _nl_seconds(int_m.group(1), int_m.group(2)) if int_m else 0.0

    # strip verb, duration phrase, interval phrase → target
    # (remove by matched text, not offsets — offsets shift after each cut)
    target = _NL_LEAD_VERB_RE.sub("", text)
    if dur_m:
        target = target.replace(dur_m.group(0), " ")
    if int_m:
        target = target.replace(int_m.group(0), " ")
    target = _NL_FILLER_RE.sub("", target)
    target = re.sub(r"\s+", " ", target).strip(" -–—:,.")
    # re-apply filler strip after whitespace normalization (trailing
    # space from phrase removal blocks the $-anchored filler regex)
    target = _NL_FILLER_RE.sub("", target).strip()

    if not target or len(target) < 2:
        return None
    # default interval: frequent enough to catch movement inside the
    # duration, clamped to the monitor's 30s minimum
    if not interval_s:
        interval_s = max(30.0, min(duration_s / 4.0, 300.0)) if duration_s else 300.0
    return {"target": target, "duration_s": duration_s,
            "interval_s": interval_s}

class RuntimeIntelMixin:
    """RuntimeIntelMixin for :class:`PartnerRuntime`."""


    # ── universal decoder ────────────────────────────────────────────────────
    def _control_decode(self, tail: str, chat_key: str) -> str:
        """/decode <data> | /decode file:<path> | /decode hash <digest> | decoders.

        Runs the Universal Decoder; if the winning decode is binary it is
        saved and sent to the chat as a file."""
        tail = (tail or "").strip()
        if not tail:
            return ("usage: /decode <data>  |  /decode file:<path>  |  "
                    "/decode hash <digest>  |  /decode decoders  |  "
                    "/decode history [query]  |  /decode show <report-id>")
        from ..decoder import DecoderAgent

        if tail == "decoders":
            from ...core.decoder import DECODERS
            names = ", ".join(d.name for d in DECODERS)
            return f"{len(DECODERS)} decoders: {names}"
        if tail.startswith("history") or tail.startswith("show "):
            from ...core.decoder import decode_history, get_report

            if tail.startswith("show "):
                rid = tail[5:].strip()
                row = get_report(getattr(self.context, "db", None), rid)
                if row is None:
                    return f"no report {rid!r} — /decode history"
                rep = row.get("report") or {}
                best = rep.get("best") or {}
                return (f" {row['id']} · {row['source']} · "
                        f"{best.get('chain') if isinstance(best, dict) else '—'}\n"
                        f"  input: {row['input_head'][:160]}")
            query = tail[7:].strip()
            rows = decode_history(getattr(self.context, "db", None),
                                  query=query, limit=10)
            if not rows:
                return "no archived decodes yet"
            lines = [f"📜 decode archive ({len(rows)} most recent):"]
            for r in rows:
                lines.append(
                    f"  {r['id']} {time.strftime('%m-%d %H:%M', time.localtime(r['ts']))} "
                    f"{r['source'][:20]:<20} {r['best_name'] or '—':<9} "
                    f"{(r['input_head'] or '').replace(chr(10), ' ')[:40]}")
            lines.append("  · /decode show <id>")
            return "\n".join(lines)
        if tail.startswith("hash "):
            from ...core.decoder import (identify_hash,
                                        known_hash_lookup_chained)
            digest = tail[5:].strip()
            cands = identify_hash(digest)
            known = (known_hash_lookup_chained(
                getattr(self.context, "db", None), digest)
                if cands else None)
            line = (f"{digest[:16]}… → {', '.join(cands) or 'unknown'}")
            if known:
                line += f"  ·  KNOWN: {known['algorithm']} of {known['plaintext']!r}"
            return line

        spec: dict[str, Any] = {"save": True, "explain": True}
        if tail.startswith("file:"):
            spec["path"] = tail[5:].strip()
        else:
            spec["data"] = tail
        agent = DecoderAgent(context=self.context, name="chat-decoder")
        result = agent.run(spec)
        if not result.ok:
            return f"decode failed: {result.error}"
        v = result.output
        best = v["report"].get("best") or {}
        out = best.get("output")
        text = v["explanation"]
        if isinstance(out, str) and out:
            snippet = truncate(out, 800)
            text += f"\ndecoded:\n{snippet}"
        sent = ""
        if v.get("saved_to"):
            chat = self._ref_from_key(chat_key)
            try:
                res = self.gateway.send_file(
                    chat.platform, chat, v["saved_to"],
                    caption=f"decoded from {v['report'].get('target')!r}"
                    f" → {best.get('output', {}).get('magic')}"
                    if isinstance(best.get("output"), dict) else "decoded file")
                if getattr(res, "ok", False):
                    sent = " (file sent)"
            except Exception:  # noqa: BLE001
                sent = f" (saved at {v['saved_to']})"
        return text + sent

    def _control_cookies(self, tail: str) -> str:
        """``/cookies <header>`` / ``/cookies file:<path>`` /
        ``/cookies ingest <header>`` — the CookieLab report."""
        tail = (tail or "").strip()
        if not tail:
            return "usage: /cookies <cookie-header>  |  /cookies file:<path>" \
                   "  |  /cookies ingest <cookie-header>"
        ingest = False
        if tail.lower().startswith("ingest "):
            ingest = True
            tail = tail[7:].strip()
        try:
            from pathlib import Path

            from ...core.cookies import CookieLab
            lab = CookieLab()
            raw = tail
            if raw.startswith("file:"):
                p = Path(raw[5:])
                if not p.is_file():
                    return f"no such file: {p}"
                raw = p.read_text(encoding="utf-8", errors="replace")
            rep = lab.report(raw)
        except Exception as exc:  # noqa: BLE001
            return f"cookie analysis failed: {exc}"
        lines = [f"cookies: {rep['count']} parsed"]
        if rep["services"]:
            lines.append("services: " + ", ".join(rep["services"]))
        kinds = rep.get("kinds", {})
        if kinds:
            lines.append("kinds: " + ", ".join(f"{k}={n}"
                                               for k, n in kinds.items()))
        for c in rep.get("cookies", []):
            flags = c.get("flags", {})
            f_txt = ""
            if flags:
                f_txt = " [" + ", ".join(k for k in
                                         ("httponly", "secure", "samesite")
                                         if k in flags) + "]"
            dec = c.get("decoded_value")
            d_txt = ""
            if dec is not None:
                d_txt = f"  decoded({c.get('decode_via')}): {str(dec)[:80]}"
            svc = f" <{c['service']}>" if c.get("service") else ""
            lines.append(f"  {c['name']} = {str(c['value'])[:48]}  "
                         f"[{c.get('kind')}]{svc}{f_txt}{d_txt}")
        sec = rep.get("security", {})
        warns = sec.get("plaintext_auth") or []
        if warns:
            lines.append("security: " + ", ".join(warns)
                         + " lack HttpOnly+Secure")
        if ingest:
            try:
                from ..kg import KnowledgeGraph
                ing = lab.ingest(self.context, raw, source="chat-cookies",
                                 graph=KnowledgeGraph(self.context.db))
                lines.append(f"ingested: {ing.get('nodes', 0)} graph nodes"
                             + (f", report {ing.get('report_id')}"
                                if ing.get("report_id") else ""))
            except Exception as exc:  # noqa: BLE001
                lines.append(f"ingest failed: {exc}")
        return "\n".join(lines)

    def _control_structure(self, tail: str) -> str:
        """``/structure <objective>`` — the structuring sub-agent's brief."""
        tail = (tail or "").strip()
        if not tail:
            return "usage: /structure <objective>"
        try:
            from ...agents.structuring import structure_text
            brief = structure_text(self.context, tail, for_="chat",
                                   polish=True)
        except Exception as exc:  # noqa: BLE001
            return f"structuring failed: {exc}"
        return brief.get("brief", "(no brief)")

    def _control_investigate(self, tail: str) -> str:
        """/investigate <artifact> [file] — decode→crack→OSINT→KG in one pass."""
        from ..investigate import InvestigateAgent

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /investigate <hash|jwt|cookie|url|blob|file> "
                    "[file]  — one pass: decode → crack → OSINT → knowledge "
                    "graph")
        is_file = tail.split()[-1].lower() == "file"
        art = tail[:-4].strip() if is_file else tail
        try:
            rep = InvestigateAgent(self.context).run(
                art, file=is_file, source="chat-investigate")
        except Exception as exc:  # noqa: BLE001
            return f"investigate error: {exc}"
        if not rep.get("ok", True):
            return f"investigate failed: {rep.get('error')}"
        lines = [f"🔬 {rep['kind']}: {rep['artifact_head'][:70]}"]
        lines.append("  " + " → ".join(rep["steps"]))
        for d, p_ in (rep.get("cracked") or {}).items():
            lines.append(f"  cracked: {d[:24]}… = {p_!r}")
        o = rep.get("osint") or {}
        if isinstance(o, dict) and o.get("persons_found"):
            lines.append(f"  identities: {', '.join(o['persons'][:6])}")
        if isinstance(o, dict) and o.get("domains"):
            lines.append(f"  domains:    {', '.join(o['domains'][:6])}")
        kg = rep.get("kg") or {}
        if isinstance(kg, dict) and kg.get("added_nodes") is not None:
            lines.append(f"  knowledge graph: +{kg.get('added_nodes', 0)} "
                         f"nodes, +{kg.get('added_links', 0)} links")
        if rep.get("report_id"):
            lines.append(f"  report: {rep['report_id']}  "
                         "(/decode show <id>)")
        return "\n".join(lines)

    def _control_monitor(self, tail: str) -> str:
        """/monitor add <target> [every Ns] | list | tick | rm <ref> | status."""
        tail = (tail or "").strip()
        from ..monitor import MonitorAgent
        from ..notifier import Notifier

        agent = MonitorAgent(self.context, notifier=Notifier(
            self.context))
        if not tail or tail == "list":
            rows = agent.list()
            if not rows:
                return "no monitors — /monitor add <url-or-file> [every 300s] [--webhook URL] [--min-gap 60]"
            lines = [f"{r['target'][:40]} · {r['kind']} · every "
                     f"{r['interval_s']:.0f}s · {'on' if r['enabled'] else 'off'}"
                     + (f" · webhook→{r['webhook_url'][-30:]}"
                        if r.get("webhook_url") else "")
                     + (f" · gap {r['min_alert_gap_s']:.0f}s"
                        if r.get("min_alert_gap_s") else "")
                     for r in rows]
            return f"{len(rows)} monitor(s):\n" + "\n".join(lines)
        if tail.startswith(("add ", "alert ", "webhook-test ")):
            verb, body = tail.split(None, 1)
            body = body.strip()
            import re

            def _flag(flag: str):
                m = re.search(rf"(?<![\w./-]){flag}\s+(\S+)", body)
                if not m:
                    return None
                body_ = (body[:m.start()] + " " + body[m.end():]).strip()
                return m.group(1), body_

            def _flag_on(flag: str):
                m = re.search(rf"(?<![\w./-]){flag}(?:\s|$)", body)
                if not m:
                    return False, body
                body_ = (body[:m.start()] + " " + body[m.end():]).strip()
                return True, body_

            webhook = ""
            m = _flag("--webhook")
            if m:
                webhook, body = m
            secret = ""
            m = _flag("--secret")
            if m:
                secret, body = m
            no_decode, body = _flag_on("--no-decode")
            min_gap = -1.0
            m = _flag("--min-gap")
            if m:
                try:
                    min_gap = float(m[0])
                except ValueError:
                    min_gap = -1.0
                body = m[1]
            if verb == "webhook-test":
                if not body:
                    return "usage: /monitor webhook-test <ref>"
                res = agent.webhook_test(body)
                if res is None:
                    return (f"no monitor {body!r} — "
                    "/monitor list to see the live ones")
                if not res.get("ok"):
                    err = res.get("error") or (res.get("result") or {}).get(
                        "error", "")
                    return f"webhook test FAILED: {err or 'no response'}"
                r = res["result"]
                return (f"webhook test OK — HTTP {r.get('status')} "
                        f"({r.get('attempts', 1)} attempt(s)) → "
                        f"{res['target']}")
            if verb == "add":
                interval = 300.0
                m = re.search(r"\bevery\s+(\d+(?:\.\d+)?)\s*s?\b", body)
                if m:
                    interval = float(m.group(1))
                    body = (body[:m.start()] + body[m.end():]).strip()
                target = body
                if not target:
                    return ("usage: /monitor add <target> [every 300s] "
                            "[--webhook URL] [--min-gap 60]")
                gap = 60.0 if min_gap < 0 else max(0.0, min_gap)
                info = agent.add(target, interval=interval,
                                 webhook=webhook, secret=secret,
                                 min_gap=gap, auto_decode=not no_decode)
                return (f"watching {target} ({info['kind']}, every "
                        f"{info['interval_s']:.0f}s) — I'll alert you on change"
                        + (f" · webhook → {info['webhook_url']}"
                           if info.get("webhook_url") else "")
                        + (f" · at most one alert per "
                           f"{info['min_alert_gap_s']:.0f}s"
                           if info.get("min_alert_gap_s") else ""))
            # alert
            if not body:
                return ("usage: /monitor alert <ref> [--webhook URL] "
                        "[--secret S] [--min-gap 60] [--no-decode]")
            row = agent.set_alerting(
                body,
                webhook=webhook or None,
                secret=secret or None,
                min_gap=min_gap if min_gap >= 0 else None,
                auto_decode=False if no_decode else None)
            if row is None:
                return (f"no monitor {body!r} — "
                    "/monitor list to see the live ones")
            return (f"alerting for {row['target']}:"
                    f" webhook={'on → ' + row['webhook_url'] if row['webhook_url'] else 'off'}"
                    f" · gap={row['min_alert_gap_s']:.0f}s"
                    + (" (every change)" if not row["min_alert_gap_s"] else ""))
        if tail == "tick":
            res = agent.tick()
            parts = [f"checked {res['checked']} due"]
            for c in res["changed"]:
                parts.append(f"changed: {c['target']}")
            for e in res["errors"]:
                parts.append(f"error: {e['target']} ({e['error'][:60]})")
            return "; ".join(parts)
        if tail.startswith("rm "):
            ref = tail[3:].strip()
            return ("removed" if agent.remove(ref)
                else f"no monitor {ref!r} — /monitor list to see the live ones")
        if tail == "status":
            st = agent.status()
            return f"{st['enabled']}/{st['total']} monitors active"
        # natural language fallback: "btc price movement in the next 2 minutes"
        nl = parse_monitor_nl(tail)
        if nl:
            info = agent.add(nl["target"], interval=nl["interval_s"])
            dur = nl["duration_s"]
            dur_note = ""
            if dur:
                if dur >= 3600:
                    dur_note = f" for the next {dur/3600:.0f}h"
                elif dur >= 60:
                    dur_note = f" for the next {dur/60:.0f}m"
                else:
                    dur_note = f" for the next {dur:.0f}s"
            return (f"watching {nl['target']} ({info['kind']}, every "
                    f"{info['interval_s']:.0f}s{dur_note}) — "
                    f"I'll alert you on change")
        return "usage: /monitor add <target> [every Ns] | list | tick | rm <ref>"

    def _control_cipher(self, tail: str) -> str:
        """/cipher enc <data> with <passphrase> | /cipher dec <blob> with <passphrase>
        | /cipher vault put <name> <secret> with <passphrase>
        | /cipher vault get <name> with <passphrase>
        | /cipher vault list | /cipher vault rm <name> with <passphrase>."""
        import re

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /cipher enc <data> with <passphrase>  |  "
                    "/cipher dec <blob> with <passphrase>  |  "
                    "/cipher vault put|get|list|rm …")
        from ..cipher import CipherAgent
        from ...core.cipher import CipherError

        agent = CipherAgent(context=self.context, name="chat-cipher")

        # ── named secrets vault ──
        if tail.startswith("vault "):
            body = tail[6:].strip()
            parts = body.split(" with ", 1)
            head, passphrase = (parts[0].strip(), parts[1].strip()) \
                if len(parts) == 2 else (body, "")
            tokens = head.split()
            verb = tokens[0].lower() if tokens else ""
            rest = tokens[1:]
            try:
                if verb == "list":
                    r = agent.run({"action": "vault_list"})
                    if not r.ok:
                        return f"vault failed: {r.error}"
                    entries = r.output.get("entries") or [
                        {"name": n, "key_scheme": "pass"}
                        for n in r.output.get("names", [])]
                    return ("🗝 vault:\n" +
                            ("\n".join(f"  {e['name']}  [{e['key_scheme']}]"
                                       for e in entries)
                             if entries else "  (empty)"))
                if verb == "export" and rest:
                    # /cipher vault export <file> with <key> [entry <epass>]
                    m = re.match(
                        r"^(?P<file>\S+)(?:\s+entry\s+(?P<epass>\S+))?$",
                        " ".join(rest))
                    if not m:
                        return ("usage: /cipher vault export <file> "
                                "with <export-key> [entry <epass>]")
                    r = agent.run({"action": "vault_export",
                                   "path": m.group("file"),
                                   "passphrase": passphrase,
                                   "entry_pass": m.group("epass") or ""})
                    if not r.ok:
                        return f"vault export failed: {r.error}"
                    skipped = f"  (skipped: {', '.join(r.output['skipped'])})" \
                        if r.output.get("skipped") else ""
                    return (f"📦 exported {r.output['exported']} entries "
                            f"→ {r.output['path']}{skipped}")
                if verb == "import" and rest:
                    r = agent.run({"action": "vault_import",
                                   "path": rest[0],
                                   "passphrase": passphrase})
                    if not r.ok:
                        return f"vault import failed: {r.error}"
                    names = ", ".join(r.output["imported"][:8])
                    return (f"📥 imported {r.output['count']} entries "
                            f"({names})")
                if verb in ("put", "get", "rm") and rest:
                    name = rest[0]
                    if verb == "put":
                        if len(rest) < 2:
                            return ("usage: /cipher vault put <name> "
                                    "<secret> with <passphrase>  (omit "
                                    "'with …' to seal under NM_VAULT_KEY)")
                        secret = " ".join(rest[1:])
                        r = agent.run({"action": "vault_put", "name": name,
                                       "data": secret,
                                       "passphrase": passphrase})
                        if not r.ok:
                            return f"vault failed: {r.error}"
                        return (f"🗝 stored {name} "
                                f"({r.output['bytes']} B sealed, AES-256, "
                                f"key: {r.output.get('key_scheme', 'pass')})")
                    if verb == "rm":
                        r = agent.run({"action": "vault_rm", "name": name})
                        if not r.ok:
                            return f"vault failed: {r.error}"
                        return (f"🗝 removed {name}"
                                if r.output["removed"]
                                else f"no entry {name!r}")
                    if not passphrase:
                        return f"usage: /cipher vault {verb} <name> with <passphrase>"
                    r = agent.run({"action": f"vault_{verb}", "name": name,
                                   "passphrase": passphrase})
                    if not r.ok:
                        return f"vault failed: {r.error}"
                    if verb == "get":
                        return f"🔓 {name}: " + r.output.get(
                            "data", r.output.get("hex", ""))
                return ("usage: /cipher vault put <name> <secret> with <pass>  |  "
                        "/cipher vault get <name> with <pass>  |  "
                        "/cipher vault list  |  /cipher vault rm <name>  |  "
                        "/cipher vault export <file> with <key> [entry <epass>]  |  "
                        "/cipher vault import <file> with <pass>")
            except CipherError as exc:
                return f"vault error: {exc}"

        m = re.match(r"^(enc|decrypt|dec)\s+(.+?)\s+with\s+(\S+)\s*$",
                     tail, re.I)
        if not m:
            return ("usage: /cipher enc <data> with <passphrase>  |  "
                    "/cipher dec <blob> with <passphrase>  |  "
                    "/cipher vault put|get|list|rm …")
        verb, body, passphrase = m.group(1).lower(), m.group(2).strip(), m.group(3)
        try:
            if verb == "enc":
                r = agent.run({"action": "encrypt", "data": body,
                               "passphrase": passphrase})
                if not r.ok:
                    return f"cipher failed: {r.error}"
                return f"🔒 sealed ({r.output['bytes']} bytes):\n{r.output['blob']}"
            r = agent.run({"action": "decrypt", "blob": body,
                           "passphrase": passphrase})
            if not r.ok:
                return f"cipher failed: {r.error}"
            return f"🔓 {r.output.get('data', r.output.get('hex', ''))}"
        except CipherError as exc:
            return f"cipher error: {exc}"

    def _control_dns(self, tail: str, chat_key: str = "") -> str:
        parts = (tail or "").split()
        if not parts:
            return "usage: /dns <domain> [record type]"
        domain = parts[0]
        record = parts[1].upper() if len(parts) > 1 else "A"
        outcome = self.context.tools.call("dns_lookup", domain=domain, record=record)
        if not outcome.ok:
            return f"dns failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        answers = ", ".join(v["answers"][:16]) or "(none)"
        return f"{v['domain']} {v['record']}: {answers}  ({v['seconds']}s)"

    def _control_scan(self, tail: str, chat_key: str = "") -> str:
        parts = (tail or "").split()
        if not parts:
            return "usage: /scan <target> [ports] [banner] — own infrastructure only"
        target = parts[0]
        rest = parts[1:]
        banner = ""
        if rest and rest[-1].lower() in {"banner", "1", "true", "yes", "on"}:
            banner = "true"
            rest = rest[:-1]
        ports = rest[0] if rest else ""
        outcome = self.context.tools.call("port_scan", target=target, ports=ports, banner=banner)
        if not outcome.ok:
            return f"scan failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        open_ports = ", ".join(str(p) for p in v["open"]) or "none"
        return (f"scan {v['scanned']} ({v['seconds']}s): open = {open_ports} | "
                f"closed {v['closed']} | filtered {v['filtered']}  [scope: {v['scope']}]")

    def _control_whois(self, tail: str, chat_key: str = "") -> str:
        domain = (tail or "").strip()
        if not domain:
            return "usage: /whois <domain>"
        outcome = self.context.tools.call("whois_lookup", domain=domain)
        if not outcome.ok:
            return f"whois failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        lines = [f"whois {v.get('domain')}"]
        for key in ("registrar", "registration", "expiration", "last_changed"):
            if v.get(key):
                lines.append(f"  {key}: {v[key]}")
        if v.get("nameservers"):
            lines.append(f"  nameservers: {', '.join(v['nameservers'][:6])}")
        if v.get("status"):
            lines.append(f"  status: {', '.join(v['status'][:5])}")
        return "\n".join(lines)

    def _control_ports(self, tail: str, chat_key: str = "") -> str:
        outcome = self.context.tools.call("local_services")
        if not outcome.ok:
            return f"ports failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        if not v["listeners"]:
            return "nothing listening (or /proc not available here)"
        lines = [f"listening on this machine ({v['count']}):"]
        for item in v["listeners"][:30]:
            lines.append(f"  {item['port']:<6} {item['address']}")
        return "\n".join(lines)

    def _control_osint(self, tail: str, chat_key: str) -> str:
        target = (tail or "").strip()
        if not target:
            return ("usage: /osint <domain|ip|url|email> — public intel\n"
                    "  /osint campaign <seeds…> — automated investigation walk\n"
                    "  /osint graph clusters | node <ref> | merge <a> <b> | stats")
        # campaign: automated walk over the discovered web
        if target.startswith("campaign"):
            seeds = target[len("campaign"):].strip()
            if not seeds:
                return "usage: /osint campaign <phone|email|username|domain|ip>…"
            chat = self._ref_from_key(chat_key)
            started = time.time()
            outcome = self.context.tools.call("osint_campaign", seeds=seeds)
            if not outcome.ok:
                return f"campaign failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            lines = [f"🕸️ campaign: {v['steps']} investigations, "
                     f"{v['entities_investigated']} entities, "
                     f"{time.time() - started:.0f}s — {v['stopped']}"]
            for c in v.get("clusters", [])[:5]:
                ids = ", ".join(f"{e['kind']}:{e['value'][:28]}"
                                for e in c["entities"][:6])
                lines.append(f"  cluster [{c['size']}] conf {c['confidence']}: {ids}")
            for f in v.get("findings", [])[:8]:
                if f.get("error"):
                    lines.append(f"  ! {f['kind']}:{f['value'][:30]} — {f['error'][:60]}")
                else:
                    summary = f.get("summary") or {}
                    lines.append(f"  · {f['kind']}:{f['value'][:30]} — "
                                 f"{str(summary)[:90]}")
            return self._send_long_checked(chat.platform, chat, "\n".join(lines))  # report already delivered in chunks
        # graph: identity correlation queries
        if target.startswith("graph"):
            parts = target.split()
            sub = parts[1].lower() if len(parts) > 1 else "stats"
            if sub == "clusters":
                outcome = self.context.tools.call("osint_graph", action="clusters")
                if not outcome.ok:
                    return f"graph failed: {getattr(outcome.error, 'message', outcome.error)}"
                clusters = outcome.value["clusters"]
                if not clusters:
                    return "identity graph is empty — /osint campaign <seed> to build it"
                lines = ["🕸️ identity clusters:"]
                for c in clusters[:10]:
                    ids = ", ".join(f"{e['value'][:26]}" for e in c["entities"][:5])
                    lines.append(f"  [{c['size']}] conf {c['confidence']}: {ids}")
                return "\n".join(lines)
            if sub == "node" and len(parts) >= 3:
                outcome = self.context.tools.call(
                    "osint_graph", action="node", node=" ".join(parts[2:]))
                if not outcome.ok:
                    return f"graph failed: {getattr(outcome.error, 'message', outcome.error)}"
                n = outcome.value
                lines = [f"🧬 {n['kind']}:{n['value']} — conf {n['confidence']}, "
                         f"sources: {len(n['sources'])}"]
                for link in n["links"][:12]:
                    lines.append(f"  [{link['edge']}/{link['weight']}] "
                                 f"{link['kind']}:{link['value'][:36]}")
                return "\n".join(lines)
            if sub == "merge" and len(parts) >= 4:
                outcome = self.context.tools.call(
                    "osint_graph", action="merge",
                    a=" ".join(parts[2:3]), b=" ".join(parts[3:4]))
                if not outcome.ok:
                    return f"merge failed: {getattr(outcome.error, 'message', outcome.error)}"
                v = outcome.value
                return (f"merged {v.get('alias')} → {v.get('representative')}"
                        if v.get("merged") else v.get("note", "no-op"))
            if sub == "decoder" and len(parts) >= 3:
                # feed a Universal Decoder report (inline JSON or a
                # workspace file) into the identity graph
                payload = " ".join(parts[2:])
                try:
                    from ...tools.filesystem import safe_path

                    payload = safe_path(self.context, payload,
                                        must_exist=True).read_text()
                except Exception:  # noqa: BLE001
                    pass  # not a file — treat as inline JSON
                outcome = self.context.tools.call(
                    "osint_graph", action="ingest_decoder", report=payload)
                if not outcome.ok:
                    return ("decoder ingest failed: "
                            f"{getattr(outcome.error, 'message', outcome.error)}")
                v = outcome.value["ingest_decoder"]
                persons = ", ".join(v["persons"][:6]) or "—"
                domains = ", ".join(v["domains"][:6]) or "—"
                return (f"🕸️ decoder findings ingested ({v['source']}):\n"
                        f"  persons: {persons}\n  domains: {domains}")
            outcome = self.context.tools.call("osint_graph", action="stats")
            if not outcome.ok:
                return f"graph failed: {getattr(outcome.error, 'message', outcome.error)}"
            s = outcome.value
            return (f"identity graph: {s['nodes']} entities, {s['edges']} edges, "
                    f"{s['clusters']} clusters — {s['by_kind']}")
        chat = self._ref_from_key(chat_key)
        started = time.time()
        try:
            outcome = self.context.tools.call("osint_report", target=target)
        except Exception as exc:  # noqa: BLE001
            return f"osint failed: {exc}"
        if not outcome.ok:
            return f"osint failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        lines = [f"🔎 osint report: {v.get('target')} ({v.get('kind')}) — "
                 f"{time.time() - started:.1f}s"]
        for key, value in v.items():
            if key in {"target", "kind", "note"}:
                continue
            if isinstance(value, dict):
                lines.append(f"· {key}:\n" + json.dumps(value, default=str, indent=1)[:1200])
            elif isinstance(value, list):
                joined = ", ".join(map(str, value))[:400]
                lines.append(f"· {key}: {joined or '—'}")
            elif value:
                lines.append(f"· {key}: {str(value)[:300]}")
        return self._send_long_checked(chat.platform, chat, "\n".join(lines))
