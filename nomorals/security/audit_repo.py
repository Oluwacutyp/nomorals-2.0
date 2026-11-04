"""RepoAuditor: review a repository like a security engineer.

Given a local path (or a git URL, cloned first), one pass:

- **secrets in the tree** — the shared pattern battery plus a
  high-entropy assignment scan over every text file;
- **secrets in history** — the battery over added lines of
  every commit (bounded): a key deleted from the tree but
  alive in history is still a leaked key, and it is reported
  as exactly that;
- **dependency audit** — the lockfiles present are parsed and
  checked against an advisory feed.  The feed is a SEED list
  bundled here, refreshed from the public OSV database when
  it answers (health-checked per query run); offline, the
  seed answers alone and the report says so;
- **insecure patterns** — string-built SQL, eval/exec sinks,
  ``shell=True``, unsafe yaml/pickle loads, TLS verification
  off, wildcard CORS, debug flags, password-grade weak hashes.

Secrets are redacted in the findings themselves — the report
never carries a usable secret, even the repo owner's.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from .audit_patterns import (
    entropy_hits, redact_secret, scan_text_for_secrets)

__all__ = [
    "RepoAuditor", "audit_repo", "SEED_ADVISORIES",
    "INSECURE_RULES", "parse_lockfiles", "compare_versions",
]

_SKIP_DIRS = {".git", "node_modules", ".venv", "venv",
              "__pycache__", "dist", "build", "vendor", ".next",
              ".tox", ".mypy_cache"}
_BINARY_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf",
                ".zip", ".gz", ".tgz", ".woff", ".woff2", ".mp4",
                ".mp3", ".pyc", ".so", ".dll", ".exe", ".bin",
                ".class", ".jar", ".wasm"}
_MAX_FILE_BYTES = 1_000_000
_MAX_FILES = 4000
_MAX_HISTORY_COMMITS = 200
_MAX_HISTORY_BYTES_PER_COMMIT = 400_000


def _finding(check: str, kind: str, title: str, location: str,
             detail: str, evidence: dict[str, Any] | None = None,
             **meta: Any) -> dict[str, Any]:
    return {"source": "repo", "check": check, "kind": kind,
            "title": title, "location": location,
            "detail": detail, "evidence": evidence, "meta": meta}


# ── versions ─────────────────────────────────────────────────
def _vkey(version: str) -> tuple[int, ...]:
    parts = re.split(r"[.\-+_]", str(version).lstrip("v"))[:4]
    out = []
    for part in parts:
        m = re.match(r"\d+", part)
        out.append(int(m.group(0)) if m else 0)
    return tuple(out)


def compare_versions(version: str, spec: str) -> bool:
    """True when ``version`` satisfies a simple spec: comma-
    joined ``<x`` / ``<=x`` / ``==x`` / ``>=x`` / ``>x`` terms."""
    v = _vkey(version)
    for term in str(spec).split(","):
        term = term.strip()
        if not term:
            continue
        m = re.match(r"(<=|>=|==|<|>|=)?\s*(.+)", term)
        if not m:
            return False
        op = m.group(1) or "=="
        t = _vkey(m.group(2))
        if op in ("==", "=") and v != t:
            return False
        if op == "<" and not v < t:
            return False
        if op == "<=" and not v <= t:
            return False
        if op == ">" and not v > t:
            return False
        if op == ">=" and not v >= t:
            return False
    return True


# ── advisory feed: seed + OSV refresh ────────────────────────
#: (ecosystem, package, affected-spec, advisory-id, severity,
#: summary).  Famous, settled cases only — the seed is a floor,
#: never the whole feed (the open-sources law: OSV refresh is
#: the growth path, health-checked per run).
SEED_ADVISORIES: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("npm", "lodash", "<4.17.21", "CVE-2020-8203", "high",
     "prototype pollution in lodash before 4.17.21"),
    ("npm", "minimist", "<1.2.6", "CVE-2021-44906", "critical",
     "prototype pollution in minimist before 1.2.6"),
    ("npm", "axios", "<0.21.2", "CVE-2020-28168", "high",
     "SSRF in axios before 0.21.2"),
    ("npm", "event-stream", "==3.3.6", "MAL-2018-event-stream",
     "critical",
     "event-stream 3.3.6 pulled a malicious dependency "
     "(the 2018 flatmap-stream incident)"),
    ("PyPI", "pyyaml", "<5.4", "CVE-2020-1747", "critical",
     "arbitrary code execution in PyYAML before 5.4"),
    ("PyPI", "urllib3", "<1.26.5", "CVE-2021-33503", "high",
     "denial of service in urllib3 before 1.26.5"),
    ("PyPI", "django", "<3.2.13", "CVE-2022-28346", "high",
     "SQL injection in Django before 3.2.13"),
)

OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
_OSV_ECOSYSTEMS = {"npm": "npm", "PyPI": "PyPI",
                   "Packagist": "Packagist", "RubyGems": "RubyGems"}


def _osv_query(deps: list[tuple[str, str, str]], *,
               timeout_s: float = 6.0) -> dict[tuple[str, str, str],
                                               list[dict[str, str]]] | None:
    """OSV querybatch for the parsed deps.  None = OSV did not
    answer (offline/blocked): the caller falls back to the seed
    and the report states which feed answered."""
    queries = []
    for eco, name, version in deps[:100]:
        queries.append({"package": {
            "name": name,
            "ecosystem": _OSV_ECOSYSTEMS.get(eco, eco)},
            "version": version})
    if not queries:
        return {}
    try:
        req = urllib.request.Request(
            OSV_BATCH_URL,
            data=json.dumps({"queries": queries}).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": "DevonSecurityAudit/1.0"},
            method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8",
                                                 errors="replace"))
    except Exception:  # noqa: BLE001 - offline is a stated
        return None         # outcome, not an exception
    out: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    results = data.get("results") or []
    for dep, result in zip(deps, results):
        vulns = (result or {}).get("vulns") or []
        if vulns:
            out[dep] = [{"id": str(v.get("id") or "?"),
                         "summary": str(v.get("summary") or "")}
                        for v in vulns]
    return out


# ── lockfiles ────────────────────────────────────────────────
def parse_lockfiles(root: Path) -> tuple[list[tuple[str, str, str]],
                                         list[str]]:
    """[(ecosystem, name, version)] + notes (unpinned counts)."""
    deps: list[tuple[str, str, str]] = []
    notes: list[str] = []

    lock = root / "package-lock.json"
    if lock.exists():
        try:
            data = json.loads(lock.read_text(encoding="utf-8",
                                             errors="replace"))
            packages = data.get("packages") or {}
            for key, meta in packages.items():
                if key.startswith("node_modules/") \
                        and isinstance(meta, dict) \
                        and meta.get("version"):
                    deps.append(("npm",
                                 key.rsplit("node_modules/", 1)[-1],
                                 str(meta["version"])))
            for name, meta in (data.get("dependencies")
                               or {}).items():
                if isinstance(meta, dict) and meta.get("version"):
                    deps.append(("npm", name, str(meta["version"])))
        except (OSError, ValueError):
            notes.append("package-lock.json unreadable")

    req = root / "requirements.txt"
    if req.exists():
        unpinned = 0
        for line in req.read_text(encoding="utf-8",
                                  errors="replace").splitlines():
            line = line.strip()
            m = re.match(r"([A-Za-z0-9_.-]+)\s*==\s*([^\s;#]+)",
                         line)
            if m:
                deps.append(("PyPI", m.group(1).lower(),
                             m.group(2)))
            elif line and not line.startswith(("#", "-")):
                unpinned += 1
        if unpinned:
            notes.append(f"requirements.txt: {unpinned} unpinned "
                         "requirement(s) not advisory-checked")

    pipfile = root / "Pipfile.lock"
    if pipfile.exists():
        try:
            data = json.loads(pipfile.read_text(
                encoding="utf-8", errors="replace"))
            for section in ("default", "develop"):
                for name, meta in (data.get(section)
                                   or {}).items():
                    ver = str((meta or {}).get("version") or "")
                    if ver.startswith("=="):
                        deps.append(("PyPI", name.lower(),
                                     ver[2:]))
        except (OSError, ValueError):
            notes.append("Pipfile.lock unreadable")

    poetry = root / "poetry.lock"
    if poetry.exists():
        text = poetry.read_text(encoding="utf-8",
                                errors="replace")
        for m in re.finditer(
                r'\[\[package\]\]\s*\nname\s*=\s*"([^"]+)"'
                r'\s*\nversion\s*=\s*"([^"]+)"', text):
            deps.append(("PyPI", m.group(1).lower(), m.group(2)))

    composer = root / "composer.lock"
    if composer.exists():
        try:
            data = json.loads(composer.read_text(
                encoding="utf-8", errors="replace"))
            for pkg in data.get("packages") or []:
                if pkg.get("name") and pkg.get("version"):
                    deps.append(("Packagist", pkg["name"],
                                 str(pkg["version"]).lstrip("v")))
        except (OSError, ValueError):
            notes.append("composer.lock unreadable")

    gemfile = root / "Gemfile.lock"
    if gemfile.exists():
        in_specs = False
        for line in gemfile.read_text(encoding="utf-8",
                                      errors="replace").splitlines():
            if line.strip() == "specs:":
                in_specs = True
                continue
            if in_specs:
                m = re.match(r"^    ([A-Za-z0-9_.-]+) \(([^)]+)\)",
                             line)
                if m:
                    deps.append(("RubyGems", m.group(1),
                                 m.group(2)))
                elif line and not line.startswith(" "):
                    in_specs = False
    # dedupe, stable order
    seen: set[tuple[str, str, str]] = set()
    out = []
    for dep in deps:
        if dep not in seen:
            seen.add(dep)
            out.append(dep)
    return out, notes


# ── insecure-pattern rules ───────────────────────────────────
#: (id, title, severity, extensions|None, regex)
INSECURE_RULES: tuple[tuple[str, str, str, frozenset[str] | None,
                            str], ...] = (
    ("sql_string_build", "SQL built by string composition",
     "high", frozenset({".py"}),
     r"\.execute(?:many)?\s*\(\s*(?:f[\"']|[\"'][^\"'\n]*[\"']\s*"
     r"(?:\+|%)|[^)]*\.format\s*\()"),
    ("sql_string_build_js", "SQL built by string composition",
     "high", frozenset({".js", ".ts", ".tsx", ".jsx", ".mjs"}),
     r"\.query\s*\(\s*(?:`[^`]*\$\{|[\"'][^\"']*[\"']\s*\+)"),
    ("eval_exec_sink", "eval/exec on a call path", "high",
     frozenset({".py", ".js", ".ts", ".tsx", ".jsx", ".mjs"}),
     r"\b(?:eval|exec)\s*\("),
    ("shell_true", "subprocess with shell=True", "medium",
     frozenset({".py"}), r"shell\s*=\s*True"),
    ("os_system", "os.system command execution", "medium",
     frozenset({".py"}), r"\bos\.system\s*\("),
    ("pickle_load", "pickle deserialisation of untrusted data",
     "medium", frozenset({".py"}), r"\bpickle\.loads?\s*\("),
    ("yaml_unsafe_load", "yaml.load without a SafeLoader",
     "medium", frozenset({".py"}),
     r"\byaml\.load\s*\((?![^)]*SafeLoader)"),
    ("tls_verify_false", "TLS verification disabled", "medium",
     frozenset({".py", ".js", ".ts"}), r"verify\s*=\s*False"),
    ("cors_wildcard", "CORS wildcard origin", "medium", None,
     r"Access-Control-Allow-Origin[\"']?\s*[:=]\s*[\"']\*[\"']|"
     r"allow_origins\s*=\s*\[\s*[\"']\*[\"']\s*\]|"
     r"origin\s*:\s*[\"']\*[\"']"),
    ("debug_enabled", "debug mode enabled in shipped config",
     "medium", None,
     r"(?i)\bdebug\b\s*[:=]\s*(?:true|1)\b"),
    ("weak_hash_password", "password hashed with md5/sha1",
     "medium", None,
     r"(?i)(?:md5|sha1)\s*\([^)\n]*(?:password|passwd|pwd)|"
     r"(?:password|passwd|pwd)[^\n=]*=\s*(?:hashlib\.)?"
     r"(?:md5|sha1)\s*\("),
    ("child_process_exec", "child_process.exec command string",
     "medium", frozenset({".js", ".ts", ".mjs"}),
     r"child_process\.exec\s*\("),
)

_COMPILED_RULES = tuple(
    (rid, title, sev, exts, re.compile(rx))
    for rid, title, sev, exts, rx in INSECURE_RULES)


def _iter_text_files(root: Path):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if count >= _MAX_FILES:
                return
            path = Path(dirpath) / name
            if path.suffix.lower() in _BINARY_EXTS:
                continue
            try:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8",
                                      errors="strict")
            except (OSError, UnicodeDecodeError):
                continue
            count += 1
            yield path, text


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _redact_text(text: str) -> str:
    """Replace every secret-shaped value in a text with its
    redacted form.  Snippets are evidence, but a snippet of the
    line that holds the secret must not republish it."""
    for hit in scan_text_for_secrets(text):
        text = text.replace(hit.value, hit.redacted)
    return text


def _snippet(text: str, offset: int, *, width: int = 160) -> str:
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    if end == -1:
        end = len(text)
    return _redact_text(text[start:end].strip()[:width])


class RepoAuditor:
    def __init__(self, *, osv_enabled: bool = True,
                 history_commits: int = _MAX_HISTORY_COMMITS
                 ) -> None:
        self.osv_enabled = osv_enabled
        self.history_commits = history_commits

    # ── secrets: tree ────────────────────────────────────────
    def scan_tree(self, root: Path, findings: list[dict[str, Any]]
                  ) -> set[str]:
        """Append tree findings; return the set of secret
        values found (for history dedupe)."""
        values: set[str] = set()
        for path, text in _iter_text_files(root):
            rel = str(path.relative_to(root))
            hits = scan_text_for_secrets(text) + entropy_hits(text)
            for hit in hits:
                values.add(hit.value)
                line = _line_of(text, hit.offset)
                findings.append(_finding(
                    "repo_secrets", "repo_secret",
                    f"{hit.pattern} in the working tree",
                    f"{rel}:{line}",
                    f"{hit.pattern}: {hit.redacted} at {rel}:{line}",
                    {"snippet": _snippet(text, hit.offset),
                     "pattern": hit.pattern,
                     "redacted": hit.redacted},
                    pattern=hit.pattern, in_history=False))
        return values

    # ── secrets: history ─────────────────────────────────────
    def scan_history(self, root: Path, findings: list[dict],
                     tree_values: set[str]) -> int:
        """Battery over added diff lines, newest first.  Only
        values NOT in the current tree are findings — the
        residue that rotation must kill.  Returns commits
        scanned."""
        try:
            log = subprocess.run(
                ["git", "-C", str(root), "log", "--all",
                 "--format=%H"],
                capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return 0
        if log.returncode != 0:
            return 0
        commits = [c for c in log.stdout.split()
                   if c][:self.history_commits]
        reported: set[str] = set()
        scanned = 0
        for commit in commits:
            try:
                show = subprocess.run(
                    ["git", "-C", str(root), "show", "--format=",
                     "-U0", "--no-color", commit],
                    capture_output=True, text=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                continue
            if show.returncode != 0:
                continue
            scanned += 1
            diff = show.stdout[:_MAX_HISTORY_BYTES_PER_COMMIT]
            current_file = ""
            new_line = 0
            for line in diff.splitlines():
                if line.startswith("+++ b/"):
                    current_file = line[6:]
                    continue
                m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", line)
                if m:
                    new_line = int(m.group(1))
                    continue
                if line.startswith("+") and not line.startswith("+++"):
                    content = line[1:]
                    for hit in scan_text_for_secrets(content):
                        if hit.value in tree_values \
                                or hit.value in reported:
                            continue
                        reported.add(hit.value)
                        findings.append(_finding(
                            "repo_secrets_history", "repo_secret",
                            f"{hit.pattern} in git history "
                            "(removed from the tree, still leaked)",
                            f"{current_file}:{new_line} "
                            f"({commit[:8]})",
                            f"{hit.pattern}: {hit.redacted} — "
                            f"present in commit {commit[:8]} at "
                            f"{current_file}:{new_line}; deleting "
                            "it from the tree did not un-leak it",
                            {"pattern": hit.pattern,
                             "redacted": hit.redacted,
                             "commit": commit[:8]},
                            pattern=hit.pattern, in_history=True))
                    new_line += 1
                elif not line.startswith("-"):
                    new_line += 1
        return scanned

    # ── insecure patterns ────────────────────────────────────
    def scan_patterns(self, root: Path,
                      findings: list[dict[str, Any]]) -> None:
        for path, text in _iter_text_files(root):
            rel = str(path.relative_to(root))
            if "test" in path.name.lower():
                continue  # fixtures/asserts are not shipped paths
            ext = path.suffix.lower()
            for rid, title, sev, exts, rx in _COMPILED_RULES:
                if exts is not None and ext not in exts:
                    continue
                for m in rx.finditer(text):
                    line = _line_of(text, m.start())
                    findings.append(_finding(
                        "insecure_patterns", rid, title,
                        f"{rel}:{line}",
                        f"{title} at {rel}:{line}: "
                        f"{_snippet(text, m.start())}",
                        {"snippet": _snippet(text, m.start()),
                         "rule": rid},
                        rule=rid, rule_severity=sev))

    # ── dependencies ─────────────────────────────────────────
    def audit_deps(self, root: Path, findings: list[dict],
                   notes: list[str]) -> tuple[int, str]:
        deps, parse_notes = parse_lockfiles(root)
        notes += parse_notes
        source = "seed"
        osv = None
        if self.osv_enabled and deps:
            osv = _osv_query(deps)
            if osv is not None:
                source = "OSV (refreshed) + seed"
        for eco, name, version in deps:
            hits: list[tuple[str, str, str]] = []
            for s_eco, s_name, spec, aid, sev, summary \
                    in SEED_ADVISORIES:
                if s_eco == eco and s_name == name.lower() \
                        and compare_versions(version, spec):
                    hits.append((aid, sev, summary))
            if osv:
                for vuln in osv.get((eco, name, version), []):
                    hits.append((vuln["id"], "high",
                                 vuln["summary"]
                                 or "per OSV advisory"))
            for aid, sev, summary in hits:
                findings.append(_finding(
                    "dependency_audit", "vulnerable_dependency",
                    f"{name} {version} — {aid}",
                    f"lockfile ({eco})",
                    f"{eco} dependency {name}=={version} matches "
                    f"{aid}: {summary}",
                    {"advisory": aid, "feed_severity": sev},
                    package=name, version=version, eco=eco,
                    advisory=aid, feed_severity=sev))
        return len(deps), source

    # ── the pass ─────────────────────────────────────────────
    def audit(self, root: Path, *, target_label: str,
              own_asset: bool = False) -> dict[str, Any]:
        started = time.time()
        findings: list[dict[str, Any]] = []
        notes: list[str] = []
        checks: list[str] = []
        checks.append("repo_secrets")
        tree_values = self.scan_tree(root, findings)
        checks.append("repo_secrets_history")
        commits = self.scan_history(root, findings, tree_values)
        if commits == 0:
            notes.append("no git history available (not a repo, "
                         "or git unreadable) — history scan skipped")
        else:
            notes.append(f"history scan covered {commits} commit(s)")
        checks.append("insecure_patterns")
        self.scan_patterns(root, findings)
        checks.append("dependency_audit")
        dep_count, source = self.audit_deps(root, findings, notes)
        notes.append(f"advisory feed: {source}; "
                     f"{dep_count} locked dependenc(ies) checked")
        return {"target": target_label, "own_asset": own_asset,
                "checks_run": checks, "findings": findings,
                "notes": notes, "deps_checked": dep_count,
                "advisory_source": source,
                "commits_scanned": commits,
                "started_at": started, "finished_at": time.time()}


def _clone(url: str, dest: Path) -> str:
    """git clone (bounded depth).  '' on success, else the
    failure text."""
    try:
        proc = subprocess.run(
            ["git", "clone", "--quiet", "--depth", "300", url,
             str(dest)],
            capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"git clone could not run: {exc}"
    if proc.returncode != 0:
        return (proc.stderr or "git clone failed").strip()[:300]
    return ""


def audit_repo(target: str, *, own_asset: bool = False,
               osv_enabled: bool = True,
               clone_root: Path | None = None) -> dict[str, Any]:
    """Audit a local path or a git URL.  URLs clone into a
    scratch dir that is removed after the pass."""
    auditor = RepoAuditor(osv_enabled=osv_enabled)
    path = Path(target).expanduser()
    if path.is_dir():
        return auditor.audit(path, target_label=target,
                             own_asset=own_asset)
    if re.match(r"^(https?://|git@|ssh://)", target):
        scratch = Path(tempfile.mkdtemp(prefix="secaudit-",
                                        dir=clone_root))
        dest = scratch / "repo"
        try:
            error = _clone(target, dest)
            if error:
                return {"target": target, "own_asset": own_asset,
                        "checks_run": [], "findings": [],
                        "notes": [f"clone failed: {error}"],
                        "deps_checked": 0, "advisory_source": "—",
                        "commits_scanned": 0,
                        "started_at": time.time(),
                        "finished_at": time.time()}
            return auditor.audit(dest, target_label=target,
                                 own_asset=own_asset)
        finally:
            import shutil
            shutil.rmtree(scratch, ignore_errors=True)
    return {"target": target, "own_asset": own_asset,
            "checks_run": [], "findings": [],
            "notes": [f"target is neither a local directory nor "
                      f"a git URL: {target}"],
            "deps_checked": 0, "advisory_source": "—",
            "commits_scanned": 0,
            "started_at": time.time(), "finished_at": time.time()}
