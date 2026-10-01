"""Script generator: render validated automation scripts into the workspace.

Each generator is a template bound to a schema of config values; the output
is written to ``workspace/generated/`` and VALIDATED before it is returned
(``bash -n`` for shell, ``py_compile`` for Python) — a generated script that
does not parse is a bug, so it is refused, not shipped.

Kinds ship as a table, so adding one is one entry:
    backup          — tar/rsync backup with retention
    webhook_notify  — push a message to telegram/discord/ntfy/generic webhook
    jsonl_to_csv    — flatten JSONL to CSV (column select, dedupe)
    dedupe_lines    — order-preserving line dedupe
    log_rotate      — size-based rotation, retention
    termux_service  — Termux boot service wrapper (run.sh + description)
    git_autopush    — rebase+push wrapper with failure log
    hf_download     — resumable Hugging Face download wrapper
    cron_sh         — cron line + logged wrapper script
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from .filesystem import safe_path

_log = get_logger(__name__)

__all__ = ["SCRIPT_KINDS", "generate", "register"]

Name = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _name(value: str, field: str = "name") -> str:
    value = (value or "").strip()
    if not Name.match(value):
        raise ToolError(f"bad {field} {value!r} — use letters, digits, . _ -")
    return value


# ── generators ───────────────────────────────────────────────────────────────


def _gen_backup(cfg: dict) -> tuple[str, str]:
    source = cfg.get("source") or "~/data"
    dest = cfg.get("dest") or "~/backups"
    keep = int(cfg.get("keep") or 7)
    if not source or not dest:
        raise ToolError("backup needs 'source' and 'dest'")
    if keep < 1:
        raise ToolError("keep must be >= 1")
    script = f"""#!/bin/sh
# backup: {source} -> {dest}  (keep {keep})
set -eu
SOURCE={source!r}
DEST={dest!r}
KEEP={keep}
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p "$DEST"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "$SOURCE" "$DEST/backup-$STAMP/"
  TAG="$DEST/backup-$STAMP"
else
  mkdir -p "$DEST/backup-$STAMP"
  tar -C "$(dirname "$SOURCE")" -cf "$DEST/backup-$STAMP/tar.tgz" "$(basename "$SOURCE")"
  TAG="$DEST/backup-$STAMP"
fi
ln -sfn "$(basename "$TAG")" "$DEST/latest"
# retention
ls -1dt "$DEST"/backup-* 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r OLD; do rm -rf "$OLD"; done
echo "backup done: $TAG"
"""
    return script, "sh"


def _gen_webhook(cfg: dict) -> tuple[str, str]:
    kind = (cfg.get("service") or "ntfy").lower()
    target = cfg.get("target") or ""
    if not target:
        raise ToolError(f"{kind} webhook needs 'target' (chat id / token / host/topic)")
    if kind == "telegram":
        token, _, chat = target.partition(":")
        if not (token and chat):
            raise ToolError("telegram target must be 'token:chat_id'")
        body = (
            f"API='https://api.telegram.org/bot{token}/sendMessage'\n"
            f"curl -fsS -X POST -d chat_id={chat!r} -d text=\"$MSG\" \"$API\" >/dev/null\n"
        )
    elif kind == "discord":
        token, _, channel = target.partition(":")
        if not (token and channel):
            raise ToolError("discord target must be 'token:channel_id'")
        body = (
            f"API='https://discord.com/api/webhooks/{token}/{channel}'\n"
            f"curl -fsS -X POST -H 'Content-Type: application/json' "
            f"-d '{{\"content\":{json.dumps(json.dumps('$MSG'))}}}' \"$API\" >/dev/null\n"
        )
    elif kind == "ntfy":
        host, _, topic = target.partition("/")
        if not topic:
            raise ToolError("ntfy target must be 'host/topic' (or use a topic name)")
        body = f"curl -fsS {host or 'https://ntfy.sh'}/{topic} -d \"$MSG\" >/dev/null\n"
    else:
        if not target.startswith(("http://", "https://")):
            raise ToolError("generic webhook target must be a URL")
        body = f"curl -fsS -X POST -H 'Content-Type: text/plain' --data \"$MSG\" {target!r} >/dev/null\n"
    script = f"""#!/bin/sh
# webhook_notify: {kind}  (message via $1 or stdin)
MSG=${{1:-$(cat)}}
[ -n "$MSG" ] || {{ echo "no message" >&2; exit 1; }}
{body}
echo "notified via {kind}"
"""
    return script, "sh"


def _gen_jsonl_to_csv(cfg: dict) -> tuple[str, str]:
    src = cfg.get("src") or "in.jsonl"
    dst = cfg.get("dst") or "out.csv"
    columns = cfg.get("columns") or []
    dedupe = bool(cfg.get("dedupe") or False)
    code = f'''"""jsonl_to_csv: flatten {src} -> {dst}"""
import csv, json, sys

SRC, DST = {src!r}, {dst!r}
COLUMNS = {columns!r}
DEDUPE = {dedupe!r}

def main():
    rows, seen = [], set()
    with open(SRC, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"line {{lineno}}: skipping bad json ({{exc}})", file=sys.stderr)
                continue
            if not isinstance(obj, dict):
                continue
            if DEDUPE:
                key = tuple(obj.get(c) for c in (COLUMNS or sorted(obj)))
                if key in seen:
                    continue
                seen.add(key)
            rows.append(obj)
    cols = COLUMNS or (sorted({{k for r in rows for k in r}}) if rows else [])
    with open(DST, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {{len(rows)}} rows -> {{DST}}")

if __name__ == "__main__":
    main()
'''
    return code, "py"


def _gen_dedupe(cfg: dict) -> tuple[str, str]:
    src = cfg.get("src") or ""
    dst = cfg.get("dst") or ""
    if not src:
        raise ToolError("dedupe_lines needs 'src'")
    script = f"""#!/bin/sh
# dedupe_lines: {src} -> {dst or 'stdout'} (order-preserving)
if [ -n "{dst}" ]; then
  awk '{{ if (!seen[$0]++) print }}' {src!r} > {dst!r}
  echo "deduped -> {dst}"
else
  awk '{{ if (!seen[$0]++) print }}' {src!r}
fi
"""
    return script, "sh"


def _gen_log_rotate(cfg: dict) -> tuple[str, str]:
    log = cfg.get("log") or "/var/tmp/app.log"
    max_kb = int(cfg.get("max_kb") or 10240)
    keep = int(cfg.get("keep") or 10)
    script = f"""#!/bin/sh
# log_rotate: {log}  (max {max_kb} KB, keep {keep})
LOG={log!r}
MAX={max_kb}
KEEP={keep}
[ -f "$LOG" ] || exit 0
SIZE=$(wc -c < "$LOG" | tr -d ' ')
if [ "$SIZE" -ge $((MAX * 1024)) ]; then
  TS=$(date +%Y%m%d-%H%M%S)
  mv "$LOG" "$LOG.$TS"
  : > "$LOG"
fi
ls -1t "$LOG".* 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r OLD; do rm -f "$OLD"; done
"""
    return script, "sh"


def _gen_termux_service(cfg: dict) -> tuple[str, str]:
    cmd = cfg.get("command") or "python3 /data/data/com.termux/files/home/app.py"
    name = cfg.get("service") or "app"
    script = f"""#!/data/data/com.termux/files/usr/bin/sh
# termux service: {name} — runs at Termux boot, restarts on crash
while true; do
  echo "[{name}] starting: {cmd}"
  {cmd}
  CODE=$?
  echo "[{name}] exited {{CODE}} — restarting in 10s"
  sleep 10
done
"""
    return script, "sh"


def _gen_git_autopush(cfg: dict) -> tuple[str, str]:
    repo = cfg.get("repo") or "."
    branch = cfg.get("branch") or ""
    script = f"""#!/bin/sh
# git_autopush: {repo}  (rebase, push, log failures)
set -u
cd {repo!r} || exit 1
LOG=/tmp/git-autopush.log
{f'git checkout {branch!r} >>"$LOG" 2>&1' if branch else '# branch: current'}
if git pull --rebase >>"$LOG" 2>&1; then
  if git push >>"$LOG" 2>&1; then
    echo "pushed ok"
  else
    echo "push failed — see $LOG"
  fi
else
  git rebase --abort >/dev/null 2>&1
  echo "pull --rebase failed (divergence?) — see $LOG"
fi
"""
    return script, "sh"


def _gen_hf_download(cfg: dict) -> tuple[str, str]:
    repo = cfg.get("repo") or ""
    if not repo:
        raise ToolError("hf_download needs 'repo' (owner/name)")
    files = cfg.get("files") or ""
    dest = cfg.get("dest") or f"~/{repo.split('/')[-1]}"
    extra = f' --include {files!r}' if files else ""
    script = f"""#!/bin/sh
# hf_download: {repo} -> {dest}  (resumable)
set -eu
DEST={dest!r}
mkdir -p "$DEST"
if command -v huggingface-cli >/dev/null 2>&1; then
  TOKEN=${{HF_TOKEN:-$(grep -E '^HF_TOKEN=' ~/.nomorals/.env 2>/dev/null | cut -d= -f2-)}}
  huggingface-cli download {repo!r}{extra} --local-dir "$DEST" \\'
    --token "$TOKEN"
else
  echo "huggingface-cli not found: pip install -U 'huggingface_hub[cli]'"
  exit 1
fi
echo "downloaded {repo} -> $DEST"
"""
    return script, "sh"


def _gen_cron(cfg: dict) -> tuple[str, str]:
    schedule = cfg.get("schedule") or "0 6 * * *"
    command = cfg.get("command") or "true"
    name = cfg.get("name") or "job"
    if not command:
        raise ToolError("cron_sh needs 'command'")
    script = f"""#!/bin/sh
# cron wrapper: {name}  ({schedule})
# add to crontab:  {schedule!r} $0
LOG=/tmp/cron-{name}.log
echo "=== $(date '+%F %T') ===" >> "$LOG"
{command} >> "$LOG" 2>&1
RC=$?
# keep the log small
if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt 1048576 ]; then
  tail -n 500 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi
exit $RC
"""
    return script, "sh"


SCRIPT_KINDS: dict[str, dict[str, Any]] = {
    "backup": {"gen": _gen_backup,
               "config": {"source": "dir to back up", "dest": "backup dir", "keep": "int (optional, 7)"}},
    "webhook_notify": {"gen": _gen_webhook,
                       "config": {"service": "telegram|discord|ntfy|generic",
                                  "target": "telegram 'token:chat_id' | discord 'token:channel' | ntfy 'host/topic' | url"}},
    "jsonl_to_csv": {"gen": _gen_jsonl_to_csv,
                     "config": {"src": "in.jsonl path", "dst": "out.csv path",
                                "columns": "list (optional)", "dedupe": "bool (optional)"}},
    "dedupe_lines": {"gen": _gen_dedupe,
                     "config": {"src": "file to dedupe", "dst": "output file (optional = stdout)"}},
    "log_rotate": {"gen": _gen_log_rotate,
                   "config": {"log": "log file", "max_kb": "int (optional, 10240)", "keep": "int (optional, 10)"}},
    "termux_service": {"gen": _gen_termux_service,
                       "config": {"command": "the command to keep running", "service": "name (optional)"}},
    "git_autopush": {"gen": _gen_git_autopush,
                     "config": {"repo": "path (optional, .)", "branch": "str (optional)"}},
    "hf_download": {"gen": _gen_hf_download,
                    "config": {"repo": "owner/name", "files": "include pattern (optional)",
                               "dest": "dir (optional)"}},
    "cron_sh": {"gen": _gen_cron,
                "config": {"schedule": "cron expression", "command": "the command", "name": "job name (optional)"}},
}


# ── render + validate ────────────────────────────────────────────────────────


def _validate(text: str, ext: str) -> None:
    if ext == "sh":
        proc = subprocess.run(["bash", "-n"], input=text.encode(),
                              capture_output=True, timeout=10, check=False)
        if proc.returncode != 0:
            raise ToolError(
                f"generated shell failed bash -n: "
                f"{(proc.stderr or b'').decode('utf-8', 'replace')[:300]}"
            )
    elif ext == "py":
        import os
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(text)
            path = fh.name
        try:
            proc = subprocess.run([sys.executable, "-m", "py_compile", path],
                                  capture_output=True, timeout=10, check=False)
            if proc.returncode != 0:
                raise ToolError(
                    "generated python failed py_compile: "
                    f"{(proc.stderr or b'').decode('utf-8', 'replace')[:300]}"
                )
        finally:
            os.unlink(path)


def generate(context: Any, kind: str, name: str, config: dict[str, Any],
             out_root: str = "generated") -> dict[str, Any]:
    entry = SCRIPT_KINDS.get((kind or "").strip())
    if entry is None:
        raise ToolError(f"unknown script kind {kind!r}; available: {', '.join(sorted(SCRIPT_KINDS))}")
    name = _name(name)
    text, ext = entry["gen"](config or {})
    _validate(text, ext)
    base = safe_path(context, out_root)
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"{name}.{ext}"
    target.write_text(text, encoding="utf-8")
    try:
        target.chmod(0o755 if ext == "sh" else 0o644)
    except OSError as e:
        _log.debug("chmod failed for %s: %s", target, e)
    return {
        "path": str(target), "kind": kind, "name": name, "ext": ext,
        "bytes": len(text), "validated": True,
        "preview": text[:600],
    }


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "script_kinds",
        description="List the available script generator kinds with their config.",
        capability=Capability.FS_READ,
    )
    def script_kinds() -> dict[str, Any]:
        return {"kinds": {k: v["config"] for k, v in sorted(SCRIPT_KINDS.items())}}

    @registry.register(
        "script_gen",
        description=(
            "Generate a validated automation script (backup, webhook, jsonl→csv, "
            "log rotation, termux service, git autopush, hf download, cron wrapper) "
            "into the workspace."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "kind": "str — one of: " + ", ".join(sorted(SCRIPT_KINDS)),
            "name": "str — output base name",
            "config": "str — JSON object of the kind's config values",
        },
    )
    def script_gen(kind: str, name: str, *, config: str = "") -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        if config and config.strip():
            try:
                parsed = json.loads(config)
            except (ValueError, TypeError) as exc:
                raise ToolError(f"config must be a JSON object: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ToolError("config must be a JSON object")
            cfg = parsed
        return generate(context, kind, name, cfg)
