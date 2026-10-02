"""``nm crack`` / ``decode`` / ``osint`` / ``cipher`` / ``monitor`` / ``watch``."""

from __future__ import annotations

from ..emit import _emit



def _cmd_crack(args, context):
    """Offline hash cracking at the shell — one digest or a batch."""
    digests = [d for d in (list(getattr(args, "hashes", None) or [])
                           + [getattr(args, "hash", "") or ""]) if d]
    if not digests:
        print("usage: nm crack <digest> [digest…] [--algo md5|sha1|…] "
              "[--mode hybrid|dictionary|brute] [--words FILE]")
        return 0
    kw: dict = {}
    if getattr(args, "algo", ""):
        kw["algo"] = args.algo
    if getattr(args, "mode", ""):
        kw["mode"] = args.mode
    if getattr(args, "words", ""):
        kw["wordlist"] = args.words
    outcome = context.tools.call("hash_crack", target=",".join(digests), **kw)
    if not outcome.ok:
        print(f"crack failed: {getattr(outcome.error, 'message', outcome.error)}")
        return 1
    value = outcome.value or {}
    found = value.get("found") or {}
    _emit(args, value, "\n".join(f"{k} = {v}" for k, v in found.items())
          or f"no plaintext found for {len(digests)} digest(s)")
    return 0


def _cmd_decode(args, context):
    """Decode/identify: encodings, nested chains, known hashes (wave 71)."""
    from ...core import decoder as D

    if getattr(args, "history", False):
        rows = D.decode_history(context.db, limit=15)
        if not rows:
            print("no decode history yet")
            return 0
        for row in rows:
            print(f"{row['id']}  {row.get('created_at', '')}  "
                  f"{row.get('source', '')}/{row.get('kind', '')}  "
                  f"{str(row.get('input_preview', row.get('input_text', '')))[:60]}")
        return 0
    if getattr(args, "show", ""):
        row = D.get_report(context.db, args.show)
        if not row:
            print(f"no report {args.show!r}")
            return 1
        print(row.get("report_json", ""))
        return 0
    if getattr(args, "mode", "") == "decoders":
        names = [d.name for d in D.DECODERS]
        _emit(args, {"count": len(names), "names": names},
              f"decoders ({len(names)}): " + ", ".join(names))
        return 0

    digest = getattr(args, "hash", "") or ""
    if digest:
        info = D.analyze(digest).hash or {}
        known = info.get("known")
        if known:
            _emit(args, info, f"known plaintext: {known.get('plaintext')}"
                  f"  (algorithm {known.get('algorithm')})")
        else:
            cands = ", ".join(info.get("algorithms") or []) or "no hash candidates"
            _emit(args, info, f"no known plaintext; candidates: {cands}")
        return 0

    text = getattr(args, "text", "") or ""
    if not text:
        print("usage: nm decode <text|file:path>  |  nm decode --hash <digest>  |  "
              "nm decode --mode decoders")
        return 2

    from ...agents.decoder import DecoderAgent

    agent = DecoderAgent(context=context, name="cli")
    result = agent.run({"data": text, "explain": True})
    if not getattr(result, "ok", False):
        print(f"decode failed: {getattr(result, 'error', 'unknown')}")
        return 1
    out = dict(result.output)
    report = out.get("report") or {}
    best = report.get("best") or {}
    chain = "+".join(str(x) for x in (best.get("chain") or [])) \
        or str(best.get("decoder") or "?")
    best_out = str(best.get("output", ""))
    if len(best_out) > 300:
        best_out = best_out[:300] + " …"
    lines = [f"best: {chain} -> {best_out}"]
    for h in (report.get("hits") or [])[:6]:
        note = f"  {str(h.get('note'))[:70]}" if h.get("note") else ""
        lines.append(f"  {h.get('decoder')}: conf {h.get('confidence')}{note}")
    if out.get("explanation"):
        lines.append(f"explanation: {out['explanation']}")
    if out.get("saved_to"):
        lines.append(f"saved: {out['saved_to']}")
    _emit(args, out, "\n".join(lines))
    return 0


def _cmd_osint(args, context):
    """OSINT reports and the persistent identity graph, from the shell."""
    sub = getattr(args, "subcommand", "") or ""
    if sub in ("report", "domain", "ip", "url", "email"):
        tool = {"report": "osint_report", "domain": "osint_domain",
                "ip": "osint_ip", "url": "osint_url",
                "email": "osint_email"}[sub]
        outcome = context.tools.call(tool, target=args.target)
    elif sub == "campaign":
        outcome = context.tools.call("osint_campaign",
                                      seeds=",".join(args.seeds or []))
    elif sub == "decoder":
        from pathlib import Path as _P

        raw = _P(args.report).read_text()
        outcome = context.tools.call("osint_graph", action="ingest_decoder",
                                     report=raw, source=args.source)
    elif sub in ("stats", "clusters", "timeline", "clear", "node", "graph"):
        action = "stats" if sub == "graph" else sub
        extra: dict = {}
        if sub == "graph" and args.action:
            action = args.action
            if action == "node" and args.args:
                extra["node"] = args.args[0]
            if action == "merge" and len(args.args) >= 2:
                extra["a"], extra["b"] = args.args[0], args.args[1]
        if sub == "node":
            extra["node"] = args.ref
        outcome = context.tools.call("osint_graph", action=action, **extra)
    else:
        print("usage: nm osint report <target> | stats | clusters | node <ref>"
              " | decoder <report.json> [--source s] | campaign <seeds…>")
        return 0
    if not outcome.ok:
        print(f"error: {getattr(outcome.error, 'message', outcome.error)}")
        return 1
    import json as _json

    print(_json.dumps(outcome.value, indent=2, default=str)
          if getattr(args, "json", False) else str(outcome.value)[:3500])
    return 0


def _cmd_cipher(args, context):
    """nmc1 encryption at the shell: authenticated AES, classic ciphers, HMAC."""
    import base64 as _b64

    from ...core import cipher as core
    from ...tools.cipher import _key_bytes

    action = getattr(args, "action", "") or ""
    data = getattr(args, "data", "") or ""
    passphrase = getattr(args, "passphrase", "") or ""
    key_bytes = _key_bytes(getattr(args, "key", "") or "")

    if action in ("vault_export", "vault_import"):
        outcome = context.tools.call(
            "cipher", action=action, path=data, passphrase=passphrase,
            entry_pass=getattr(args, "entry_pass", "") or "")
        if not outcome.ok:
            print(f"error: {getattr(outcome.error, 'message', outcome.error)}")
            return 1
        result = outcome.value or {}
        if action == "vault_export":
            print(f"exported {result.get('count', '?')} entries to {data}")
        else:
            print(f"imported {result.get('count', 0)} entries from {data}")
        return 0

    if action in ("vault_put", "vault_get", "vault_list", "vault_rm"):
        # the sealed named-secrets vault lives in the cipher tool (agent
        # side); the CLI is a thin window onto it
        vault = getattr(args, "secret", "") or ""
        outcome = context.tools.call(
            "cipher", action=action, name=data, data=vault,
            passphrase=passphrase)
        if not outcome.ok:
            print(f"error: {getattr(outcome.error, 'message', outcome.error)}")
            return 1
        result = outcome.value or {}
        if action == "vault_put":
            print(f"stored {data} in the vault")
        elif action == "vault_get":
            print(result.get("data", ""))
        elif action == "vault_list":
            names = result.get("names", [])
            print("vault entries: " + (", ".join(names) if names else "none"))
        else:
            print(f"removed {data}" if result.get("removed", True)
                  else f"no vault entry {data}")
        return 0
    try:
        if action == "encrypt":
            if not data:
                print("usage: nm cipher encrypt <text> --passphrase <pw>")
                return 2
            if not passphrase and not key_bytes:
                print("error: encrypt needs --passphrase (or --key)")
                return 2
            print(core.aes_encrypt(
                data, passphrase=passphrase, key=key_bytes,
                mode=getattr(args, "mode", "") or "ctr"))
            return 0
        if action == "decrypt":
            blob = getattr(args, "blob", "") or data
            if not blob:
                print("usage: nm cipher decrypt --blob <nmc1:...> --passphrase <pw>")
                return 2
            plain = core.aes_decrypt(blob, passphrase=passphrase, key=key_bytes)
            try:
                print(plain.decode("utf-8"))
            except UnicodeDecodeError:
                print(f"<binary: {_b64.b64encode(plain).decode()}>")
            return 0
        if action == "classic":
            alg = (getattr(args, "algorithm", "") or "caesar").lower()
            do_dec = bool(getattr(args, "decrypt", False))
            if alg == "caesar":
                print(core.caesar(data, int(getattr(args, "shift", 1) or 1),
                                   decrypt=do_dec))
            elif alg == "vigenere":
                print(core.vigenere(data, getattr(args, "keyword", "") or "",
                                    decrypt=do_dec))
            elif alg == "atbash":
                print(core.atbash(data))
            elif alg == "b64":
                if do_dec:
                    print(core.b64_decode(data).decode("utf-8", "replace"))
                else:
                    print(core.b64_encode(data))
            else:
                print(f"unknown algorithm {alg!r} (caesar|vigenere|atbash|b64)")
                return 2
            return 0
        if action == "hmac":
            print(core.hmac_hex(getattr(args, "key", "") or "", data))
            return 0
        if action == "formats":
            _emit(args, {"algorithms": ["aes-ctr", "aes-cbc", "caesar",
                                        "vigenere", "atbash", "b64", "hmac-sha256"]},
                  "nmc1:v1 blobs - aes-ctr (default), aes-cbc; classic: caesar, "
                  "vigenere, atbash, b64; hmac-sha256")
            return 0
    except core.CipherError as exc:
        print(f"cipher error: {exc}")
        return 1
    print(f"unknown action: {action!r} (encrypt|decrypt|classic|hmac|formats)")
    return 2


def _cmd_monitor(args, context):
    """Watch files/URLs for changes — thin shell over agents.monitor."""
    from ...agents.monitor import MonitorAgent

    agent = MonitorAgent(context)
    action = getattr(args, "action", "") or "status"
    ref = getattr(args, "ref", "") or ""

    if action == "add":
        if not ref:
            print("usage: nm monitor add <file-or-url> [--interval SECONDS] "
                  "[--webhook URL --secret S] [--watch content|size]")
            return 2
        try:
            _kw: dict = dict(
                interval=float(getattr(args, "interval", 300.0) or 300.0),
                webhook=getattr(args, "webhook", "") or "",
                secret=getattr(args, "secret", "") or "",
                watch=getattr(args, "watch", "") or "content")
            if getattr(args, "min_gap", None) is not None:
                _kw["min_gap"] = float(args.min_gap)
            row = agent.add(ref, **_kw)
        except ValueError as exc:
            print(f"monitor add failed: {exc}")
            return 1
        _emit(args, row,
              f"watching {ref} ({row.get('kind', 'file')}, "
              f"every {int(row.get('interval', 300) or 300)}s)")
        return 0
    if action == "alert":
        row = agent.set_alerting(
            ref,
            webhook=getattr(args, "webhook", None) or None,
            secret=getattr(args, "secret", None) or None,
            min_gap=(float(args.min_gap)
                     if getattr(args, "min_gap", None) is not None else None))
        if row is None:
            print(f"no such monitor: {ref}")
            return 1
        gap = int(row.get("min_alert_gap_s", 0) or 0)
        wh = row.get("webhook_url", "") or "off"
        _emit(args, {"monitor": row}, f"alerting updated: webhook={wh} gap={gap}s")
        return 0
    if action in ("webhook_test", "webhook-test"):
        res = agent.webhook_test(ref)
        if res is None:
            print(f"no such monitor (or no webhook set): {ref}")
            return 1
        ok = res.get("ok")
        _emit(args, res, f"webhook test: {'sent' if ok else 'failed'} "
                        f"({res.get('status', res.get('error', ''))})")
        return 0 if ok else 1
    if action == "tick":
        result = agent.tick()
        if isinstance(result, dict):
            text = "tick: " + (", ".join(f"{k}={v}" for k, v in result.items())
                               or "nothing due")
        else:
            text = f"tick: {result}"
        _emit(args, result if isinstance(result, dict) else {"result": result}, text)
        return 0
    if action == "list":
        rows = agent.list()
        if not rows:
            _emit(args, [], "no monitors - add one with `nm monitor add <file|url>`")
            return 0
        lines = []
        for r in rows:
            label = r.get("ref") or r.get("target") or ""
            lines.append(f" {label}  kind={r.get('kind', '')}  "
                         f"{'enabled' if r.get('enabled', True) else 'paused'}  "
                         f"every {int(r.get('interval', 300) or 300)}s")
        _emit(args, rows, "\n".join(lines))
        return 0
    if action == "status":
        st = agent.status()
        _emit(args, st,
              f"monitors: {st.get('enabled', 0)}/{st.get('total', 0)} enabled")
        return 0
    if action == "remove":
        if not ref:
            print("usage: nm monitor remove <ref>")
            return 2
        ok = bool(agent.remove(ref))
        print(f"removed {ref}" if ok else f"no monitor matching {ref}")
        return 0 if ok else 1
    if action in ("enable", "disable"):
        row = agent.set_enabled(ref, action == "enable")
        if row is None:
            print(f"no monitor matching {ref}")
            return 1
        _emit(args, row, f"{ref} {action}d")
        return 0
    print(f"unknown action: {action}")
    return 2


def _cmd_watch(args, context):
    """Background watchers with smart alerts — thin shell over agents.watchers."""
    from ...agents.watchers import WatcherAgent

    agent = WatcherAgent(context)
    action = getattr(args, "action", "") or "list"
    ref = getattr(args, "ref", "") or ""

    if action == "add":
        if not ref:
            print('usage: nm watch add "<plain language>" '
                  '[--severity info|important|urgent] [--interval SECONDS] '
                  '[--quiet "22:00-07:00"]')
            print('  e.g. nm watch add "tell me if BTC drops below $60000"')
            return 2
        overrides: dict = {}
        if getattr(args, "severity", ""):
            overrides["severity"] = args.severity
        if getattr(args, "interval", None):
            overrides["interval_s"] = float(args.interval)
        quiet = getattr(args, "quiet", "") or ""
        if quiet:
            m = quiet.replace(" ", "")
            if "-" in m:
                start, _, end = m.partition("-")
                overrides["quiet_hours"] = {"start": start, "end": end}
        channels = getattr(args, "channels", "") or ""
        if channels:
            overrides["channels"] = [c.strip().lower() for c in
                                     channels.replace(",", " ").split()
                                     if c.strip()]
        try:
            res = agent.add(ref, **overrides)
        except ValueError as exc:
            print(f"watch add failed: {exc}")
            return 1
        if not res.get("ok"):
            # ambiguity policy: one clarifying question, no guessed watcher
            print(res.get("question", "could not parse that"))
            return 1
        w = res["watcher"]
        _emit(args, res, f"{res.get('echo', '')}\nwatcher {w['id']} created")
        return 0
    if action == "list":
        rows = agent.list()
        if not rows:
            _emit(args, [], "no watchers — add one with `nm watch add \"...\"`")
            return 0
        lines = [f" {r['id']}  {r['name'][:45]:45}  {r['kind']:9}  "
                 f"{r['state']:7}  {r['severity']:9}  "
                 f"every {int(r['interval_s'])}s" for r in rows]
        _emit(args, rows, "\n".join(lines))
        return 0
    if action in ("pause", "resume"):
        fn = agent.pause if action == "pause" else agent.resume
        row = fn(ref)
        if row is None:
            print(f"no watcher matching {ref!r}")
            return 1
        _emit(args, row, f"{row['id']} {action}d")
        return 0
    if action == "rm":
        ok = agent.remove(ref)
        print(f"removed {ref}" if ok else f"no watcher matching {ref!r}")
        return 0 if ok else 1
    if action == "history":
        row = agent.history(ref, limit=int(getattr(args, "limit", 50) or 50))
        if row is None:
            print(f"no watcher matching {ref!r}")
            return 1
        checks = row["checks"]
        lines = [f" {c['checked_at']:.0f}  "
                 f"{'CHANGED' if c['changed'] else 'same':7}  "
                 f"{c['summary'][:70]}" for c in checks]
        _emit(args, row, f"history for {row['watcher']['name']} "
                         f"({len(checks)} checks):\n" + "\n".join(lines)
              if lines else "no checks recorded yet")
        return 0
    if action == "alerts":
        rows = agent.alert_log(ref, limit=int(getattr(args, "limit", 50) or 50))
        if not rows:
            _emit(args, [], "no alerts recorded")
            return 0
        lines = [f" {a['created_at']:.0f}  {a['status']:10} {a['severity']:9}  "
                 f"{a['title'][:70]}" for a in rows]
        _emit(args, rows, "\n".join(lines))
        return 0
    if action == "tick":
        result = agent.tick()
        text = (f"tick: checked={result.get('checked')} "
                f"changed={result.get('changed')} "
                f"errors={result.get('errors')}")
        _emit(args, result, text)
        return 0
    print(f"unknown action: {action}")
    return 2
