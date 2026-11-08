"""DisclosureDesk + the audit book.

The desk turns raw findings into the three things the owner
acts on:

- a **severity grade** per finding, from a defined rubric —
  data sensitivity × reachability × scale, never a vibe;
- a **report**: the full markdown file (findings, evidence-pack
  summaries, a remediation per finding) and the summary the
  chat card renders;
- a **disclosure draft**: contact discovery (security.txt,
  SECURITY.md, package metadata) + a factual draft.  Drafts
  only — sending a disclosure is the owner's act (the
  agent-drafts pattern).

The book is the durable store under the social data dir: the
audit journal (evidence packs at rest — MASKED ONLY, always),
the background job queue, and the owner's own-asset registry.
A corrupt store raises after quarantining, like every other
store in this system; it is never silently reset.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .audit_evidence import pack_for_storage

_log = get_logger(__name__)

__all__ = [
    "SEVERITY_ORDER", "grade_finding", "grade_findings",
    "REMEDIATION", "render_report_md", "summarize",
    "discover_contacts_web", "discover_contacts_repo",
    "draft_disclosure",
    "AuditBook", "AuditStoreError", "default_book",
]

SEVERITY_ORDER: tuple[str, ...] = (
    "info", "low", "medium", "high", "critical")


def _rank(sev: str) -> int:
    return SEVERITY_ORDER.index(sev) if sev in SEVERITY_ORDER else 0


def _bump(sev: str, steps: int = 1) -> str:
    return SEVERITY_ORDER[min(_rank(sev) + steps,
                              len(SEVERITY_ORDER) - 1)]


# ── the rubric ───────────────────────────────────────────────
_MASTER_PATTERNS = {
    "aws_secret_key", "stripe_live_key", "stripe_restricted_key",
    "private_key_block", "gcp_service_account",
}
_STRONG_PATTERNS = _MASTER_PATTERNS | {
    "aws_access_key", "github_token", "github_fine_grained",
    "github_oauth", "slack_token", "sendgrid_key",
}


def grade_finding(finding: dict[str, Any]) -> tuple[str, str]:
    """(severity, reason) for one finding.  The rubric:

    * data exposure: the schema's sensitivity tier sets the
      base (tier 3 identity/financial → critical, 2 personal →
      high, 1 contact → medium, 0 → low); a source-stated count
      ≥ 10k escalates one step; an empty collection de-escalates
      to low (an open door to an empty room);
    * secrets: master-key shapes (cloud secret, live payment
      key, private key) are critical in client code / high in a
      repo (a repo may be private; a shipped bundle is public
      by construction); strong tokens one step down; generic
      assignments medium; a firebase web key is low (public by
      design); a service_role JWT in client code is critical —
      it is a master key with an audience;
    * exposed files: .env & dumps are high, critical when real
      secret-shaped values are inside; .git is high (full
      source disclosure);
    * dependencies carry their feed severity; insecure-pattern
      rules carry their rule severity.
    """
    kind = str(finding.get("kind") or "")
    meta = finding.get("meta") or {}
    pattern = str(meta.get("pattern")
                  or (finding.get("evidence") or {}).get("pattern")
                  or "")

    if kind == "client_master_key":
        return "critical", "service_role (master) key shipped in client code"
    if kind == "client_config_key":
        return "low", "client config key — public by design, noted"
    if kind == "client_secret":
        if pattern in _MASTER_PATTERNS:
            return "critical", f"{pattern} in code every visitor downloads"
        if pattern in _STRONG_PATTERNS:
            return "high", f"{pattern} in code every visitor downloads"
        if pattern == "jwt":
            return "medium", "JWT shipped in client code (no privileged role)"
        return "medium", f"{pattern or 'secret-shaped value'} in client code"
    if kind in ("unauthenticated_api", "public_backend"):
        tier = int(meta.get("sensitivity_tier") or 0)
        count = meta.get("total_count")
        base = {3: "critical", 2: "high", 1: "medium",
                0: "low"}[tier]
        reason = f"sensitivity tier {tier}, unauthenticated"
        if isinstance(count, int):
            if count == 0:
                return "low", reason + "; collection is empty"
            if count >= 10_000 and tier >= 2:
                base = _bump(base)
                reason += f"; scale {count} records escalates"
            else:
                reason += f"; {count} records"
        return base, reason
    if kind == "public_bucket":
        keys = str(finding.get("detail") or "").lower()
        if any(w in keys for w in ("backup", "dump", "user",
                                   "export", ".sql", ".zip")):
            return "high", "public listing names data-shaped objects"
        return "medium", "public bucket listing"
    if kind == "exposed_git":
        return "high", "repository contents downloadable — full source disclosure"
    if kind == "exposed_file":
        if meta.get("secret_patterns"):
            return "critical", "exposed file contains secret-shaped values"
        return "high", "sensitive file served publicly"
    if kind == "repo_secret":
        where = "history" if meta.get("in_history") else "tree"
        if pattern in _MASTER_PATTERNS:
            return "critical", f"{pattern} in the repo {where}"
        if pattern in _STRONG_PATTERNS:
            return "high", f"{pattern} in the repo {where}"
        return "medium", f"{pattern or 'secret-shaped value'} in the repo {where}"
    if kind == "vulnerable_dependency":
        sev = str(meta.get("feed_severity") or "medium")
        return (sev if sev in SEVERITY_ORDER else "medium"), \
            f"advisory {meta.get('advisory') or ''} severity"
    # insecure-pattern rules carry their own severity in meta
    sev = str(meta.get("rule_severity") or "")
    if sev in SEVERITY_ORDER:
        return sev, "rule severity"
    return "low", "unclassified finding kind"


def grade_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copies of the findings with severity + grade_reason,
    sorted most-severe first."""
    graded = []
    for finding in findings:
        sev, reason = grade_finding(finding)
        graded.append({**finding, "severity": sev,
                       "grade_reason": reason})
    graded.sort(key=lambda f: -_rank(str(f.get("severity"))))
    return graded


# ── remediation ──────────────────────────────────────────────
REMEDIATION: dict[str, str] = {
    "exposed_file": "Remove the file from the web root, block dotfile/backup paths at the server, and rotate every secret it contained — assume it was read.",
    "exposed_git": "Remove /.git from the deployed tree (deploy artifacts, not the repo), block dotfile paths at the server, and treat the source as public.",
    "unauthenticated_api": "Require authentication on the route and authorize per-record on the server; a hidden URL is not a control.",
    "public_backend": "Turn on row-level security / database rules so unauthenticated reads are denied; the anon key is public — policy is the only wall.",
    "public_bucket": "Make the bucket private and serve files through signed URLs or the app; audit access logs for reads.",
    "client_secret": "Move the secret server-side behind a proxy endpoint; rotate it — it has been public since the deploy that shipped it.",
    "client_master_key": "Rotate the key NOW and never ship service_role/master keys to clients; use the anon key + row-level policies in the browser.",
    "client_config_key": "No action needed for the key itself; make sure database rules (not the key) are what protect the data.",
    "repo_secret": "Rotate the secret, then scrub history (filter-repo) — deleting the file does not un-leak it; audit the secret's usage logs.",
    "vulnerable_dependency": "Upgrade past the fixed version in the advisory and re-lock; if the package is abandoned, replace it.",
    "sql_string_build": "Use parameterized queries / bound parameters everywhere; string-built SQL is injectable by construction.",
    "sql_string_build_js": "Use parameterized queries / bound parameters everywhere; string-built SQL is injectable by construction.",
    "eval_exec_sink": "Remove eval/exec from data paths; where dynamic behaviour is real, use a parser or an allow-list dispatch.",
    "shell_true": "Call subprocess with an argv list and shell=False; a shell string built from input is a command-injection surface.",
    "os_system": "Replace os.system with subprocess argv calls (shell=False).",
    "pickle_load": "Never unpickle untrusted data; use JSON or a signed format.",
    "yaml_unsafe_load": "Use yaml.safe_load (or SafeLoader) for anything not hand-written.",
    "tls_verify_false": "Re-enable certificate verification; fix the underlying cert problem instead.",
    "cors_wildcard": "Reflect only allow-listed origins, especially on credentialed routes.",
    "debug_enabled": "Ship with debug off; debug consoles and verbose errors hand attackers the map.",
    "weak_hash_password": "Hash passwords with bcrypt/argon2 (salted, slow); md5/sha1 are brute-force speed records.",
    "child_process_exec": "Use execFile/spawn with an argv array; never a composed shell string.",
}


def _remediation_for(finding: dict[str, Any]) -> str:
    kind = str(finding.get("kind") or "")
    rule = str((finding.get("meta") or {}).get("rule") or "")
    return (REMEDIATION.get(rule) or REMEDIATION.get(kind)
            or "Review the finding and close the exposure.")


# ── the report ───────────────────────────────────────────────
def summarize(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts + light top-lines.  Deliberately evidence-free:
    the summary is stored in the journal and job results, and
    evidence packs (which can carry raw samples in the
    delivered form) must never ride along into a store."""
    counts = {sev: 0 for sev in SEVERITY_ORDER}
    for f in findings:
        sev = str(f.get("severity") or "low")
        counts[sev] = counts.get(sev, 0) + 1
    top = [{"severity": f.get("severity"), "title": f.get("title"),
            "location": f.get("location"), "kind": f.get("kind")}
           for f in findings[:5]]
    return {"total": len(findings), "counts": counts, "top": top}


def render_report_md(result: dict[str, Any],
                     findings: list[dict[str, Any]], *,
                     own_asset: bool = False) -> str:
    """The full markdown report.  Samples render from the
    pack's masked form for third-party targets and raw for the
    owner's own assets — the report file is the owner's copy,
    and the journal keeps the masked twin either way."""
    when = time.strftime("%Y-%m-%d %H:%M UTC",
                         time.gmtime(result.get("started_at")
                                     or time.time()))
    summary = summarize(findings)
    lines = [f"# Security audit — {result.get('target')}",
             "",
             f"- when: {when}",
             f"- checks run: {', '.join(result.get('checks_run') or []) or '—'}",
             f"- findings: {summary['total']} "
             f"(critical {summary['counts']['critical']}, "
             f"high {summary['counts']['high']}, "
             f"medium {summary['counts']['medium']}, "
             f"low {summary['counts']['low']}, "
             f"info {summary['counts']['info']})"]
    if result.get("advisory_source"):
        lines.append(f"- advisory feed: {result['advisory_source']}")
    if result.get("requests_made") is not None:
        lines.append(f"- requests made: {result['requests_made']}")
    for note in result.get("notes") or []:
        lines.append(f"- note: {note}")
    lines.append("")
    if not findings:
        lines += ["No findings. Every check in the battery came "
                  "back clean or unreachable — stated plainly, "
                  "not dressed up.", ""]
    for i, f in enumerate(findings, 1):
        lines += [f"## {i}. [{str(f.get('severity')).upper()}] "
                  f"{f.get('title')}",
                  "",
                  f"- where: {f.get('location')}",
                  f"- check: {f.get('check')} · kind: {f.get('kind')}",
                  f"- graded: {f.get('grade_reason')}",
                  f"- detail: {f.get('detail')}",
                  f"- fix: {_remediation_for(f)}"]
        pack = f.get("evidence")
        if isinstance(pack, dict) and pack.get("request"):
            req = pack["request"]
            resp = pack.get("response") or {}
            lines.append(f"- evidence: {req.get('method')} "
                         f"{req.get('url')} → "
                         f"{resp.get('status')}")
            if pack.get("total_count") is not None:
                lines.append(f"- source-stated record count: "
                             f"{pack['total_count']}")
            schema = pack.get("schema") or {}
            if schema:
                lines.append("- schema: " + ", ".join(
                    f"{k} ({v})" for k, v in list(schema.items())))
            sample = pack.get("sample") if own_asset else \
                pack.get("sample_masked")
            if sample:
                lines += ["- sample ("
                          + ("owner's own asset — unmasked"
                             if own_asset else
                             "masked — shape is the proof")
                          + "):",
                          "",
                          "```json",
                          json.dumps(sample, indent=1,
                                     default=str)[:4000],
                          "```"]
            if pack.get("notes"):
                lines.append(f"- pack note: {pack['notes']}")
        elif isinstance(pack, dict) and pack.get("snippet"):
            lines.append(f"- snippet: `{pack['snippet']}`")
            if pack.get("redacted"):
                lines.append(f"- value (redacted): {pack['redacted']}")
        lines.append("")
    return "\n".join(lines)


# ── disclosure desk ──────────────────────────────────────────
_EMAIL_RX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\."
                       r"[A-Za-z]{2,}")
_JUNK_EMAIL = ("example.", "noreply", "no-reply", "sentry.",
               "wixpress", "wordpress.com")


def _emails(text: str) -> list[str]:
    out: list[str] = []
    for m in _EMAIL_RX.finditer(text or ""):
        addr = m.group(0)
        if any(j in addr.lower() for j in _JUNK_EMAIL):
            continue
        if addr not in out:
            out.append(addr)
    return out[:5]


def discover_contacts_web(get: Callable[[str], Any],
                          base: str) -> list[str]:
    """security.txt first (the standard), then mailtos on the
    homepage.  ``get`` is the scanner's fetcher method."""
    contacts: list[str] = []
    base = base.rstrip("/")
    for path in ("/.well-known/security.txt", "/security.txt"):
        res = get(base + path)
        if getattr(res, "status", None) == 200:
            for line in res.text.splitlines():
                if line.lower().startswith("contact:"):
                    value = line.split(":", 1)[1].strip()
                    if value:
                        contacts.append(value)
            if contacts:
                return contacts
    res = get(base)
    if getattr(res, "status", None) == 200:
        contacts = _emails(res.text)
    return contacts


def discover_contacts_repo(root: Path, url: str = "") -> list[str]:
    contacts: list[str] = []
    for name in ("SECURITY.md", "security.md", ".github/SECURITY.md"):
        path = root / name
        if path.exists():
            contacts += _emails(path.read_text(
                encoding="utf-8", errors="replace"))
    pkg = root / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(
                encoding="utf-8", errors="replace"))
            author = data.get("author")
            text = author if isinstance(author, str) else \
                json.dumps(author or {})
            contacts += _emails(text)
        except (OSError, ValueError) as exc:
            _log.debug("audit_report.discover_contacts_repo failed", exc_info=True)
    if not contacts and url:
        m = re.search(r"github\.com[:/]([^/]+)/([^/.]+)", url)
        if m:
            contacts.append(
                f"GitHub: @{m.group(1)}/{m.group(2)} "
                "(via the repository's security advisory page)")
    return contacts[:5]


def draft_disclosure(*, target: str,
                     findings: list[dict[str, Any]],
                     contacts: list[str],
                     report_name: str = "") -> str:
    """A factual, sendable draft.  Never carries a secret value
    or an unmasked sample — the draft proves the finding by
    naming kinds, counts and field names only."""
    summary = summarize(findings)
    worst = findings[0] if findings else None
    lines = [
        f"To: {', '.join(contacts) if contacts else '(site owner)'}",
        f"Subject: Security disclosure — exposed data at {target}",
        "",
        "Hello,",
        "",
        f"I'm a security researcher. During a routine audit of "
        f"publicly reachable surfaces I found data exposure "
        f"affecting {target}. I am writing to disclose it "
        f"responsibly before any public mention.",
        "",
        f"What I found ({summary['total']} finding(s); "
        f"{summary['counts']['critical']} critical, "
        f"{summary['counts']['high']} high):",
    ]
    for f in findings[:8]:
        lines.append(f"- [{str(f.get('severity')).upper()}] "
                     f"{f.get('title')} — {f.get('location')}")
    lines += [
        "",
        "How verified: the requests above were made with no "
        "credentials of any kind. The evidence I hold is the "
        "request/response record, the exposed schema (field "
        "names), the record count your own backend states, and "
        "a small sample held in masked form — I did not "
        "download your database.",
        "",
        "What I ask: please confirm receipt, and let me know "
        "your remediation timeline. "
        + (f"The most urgent item is: {worst.get('title')}."
           if worst else ""),
        "I am happy to share the full technical report"
        + (f" ({report_name})" if report_name else "")
        + " and to re-check after a fix. I will not publish "
          "anything before we have spoken, and any eventual "
          "write-up will credit the fix.",
        "",
        "— Devon (security audit)",
    ]
    return "\n".join(lines)


# ── the book ─────────────────────────────────────────────────
class AuditStoreError(Exception):
    """The audit store is unreadable (quarantined, never reset)."""


@dataclass
class AuditBook:
    root: Path
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.root / "secaudit.json"

    @property
    def out_dir(self) -> Path:
        return self.root / "secaudit_out"

    def load(self) -> "AuditBook":
        if not self.path.exists():
            self.data = {"journal": [], "jobs": [], "own_assets": [],
                         "seq": 0, "job_seq": 0}
            return self
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("secaudit store root is not an object")
        except (OSError, ValueError) as exc:
            quarantine = self.path.with_suffix(
                f".corrupt-{int(time.time())}")
            try:
                self.path.replace(quarantine)
            except OSError as _exc:
                _log.debug("audit_report.load failed", exc_info=True)
            raise AuditStoreError(
                f"secaudit store at {self.path} is unreadable "
                f"({type(exc).__name__}: {exc}); quarantined to "
                f"{quarantine}") from exc
        for key, default in (("journal", []), ("jobs", []),
                             ("own_assets", []), ("seq", 0),
                             ("job_seq", 0)):
            raw.setdefault(key, default)
        self.data = raw
        return self

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)

    # ── journal ──────────────────────────────────────────────
    def record(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Journal one finished audit.  Evidence packs are
        stored via pack_for_storage — masked at rest, always."""
        self.data["seq"] = int(self.data.get("seq") or 0) + 1
        stored = dict(entry)
        stored["id"] = f"sa{self.data['seq']}"
        findings = []
        for f in entry.get("findings") or []:
            f = dict(f)
            pack = f.get("evidence")
            if isinstance(pack, dict) and pack.get("request"):
                f["evidence"] = pack_for_storage(pack)
            findings.append(f)
        stored["findings"] = findings
        self.data.setdefault("journal", []).append(stored)
        self.data["journal"] = self.data["journal"][-200:]
        self.save()
        return stored

    # ── jobs ─────────────────────────────────────────────────
    def enqueue(self, *, kind: str, target: str,
                chat_platform: str, chat_key: str,
                requester: str) -> dict[str, Any]:
        self.data["job_seq"] = int(self.data.get("job_seq") or 0) + 1
        job = {"id": f"j{self.data['job_seq']}", "kind": kind,
               "target": target, "status": "queued",
               "chat_platform": chat_platform,
               "chat_key": chat_key, "requester": requester,
               "queued_at": time.time(), "result": None}
        self.data.setdefault("jobs", []).append(job)
        self.save()
        return job

    def next_job(self) -> dict[str, Any] | None:
        for job in self.data.get("jobs") or []:
            if job.get("status") == "queued":
                job["status"] = "running"
                self.save()
                return job
        return None

    def finish_job(self, job_id: str,
                   result: dict[str, Any]) -> None:
        for job in self.data.get("jobs") or []:
            if job.get("id") == job_id:
                job["status"] = "done"
                job["finished_at"] = time.time()
                job["result"] = result
                break
        self.save()

    # ── own-asset registry ───────────────────────────────────
    def own_assets(self) -> list[str]:
        return list(self.data.get("own_assets") or [])

    def add_own_asset(self, value: str) -> bool:
        value = str(value or "").strip()
        if not value:
            return False
        assets = self.data.setdefault("own_assets", [])
        if value in assets:
            return False
        assets.append(value)
        self.save()
        return True

    def remove_own_asset(self, value: str) -> bool:
        value = str(value or "").strip()
        assets = self.data.get("own_assets") or []
        if value not in assets:
            return False
        assets.remove(value)
        self.save()
        return True

    def is_own_asset(self, target: str) -> bool:
        """Host match for URLs (the registered host or any of
        its subdomains), path match for local repos, host+path
        match for repo URLs."""
        import urllib.parse

        target = str(target or "").strip()
        if not target:
            return False
        t_parts = urllib.parse.urlsplit(
            target if "://" in target else "")
        t_host = (t_parts.hostname or "").lower()
        t_path = target.replace("\\", "/").rstrip("/")
        for asset in self.own_assets():
            a = asset.strip()
            if not a:
                continue
            if "://" in a:
                a_parts = urllib.parse.urlsplit(a)
                a_host = (a_parts.hostname or "").lower()
                a_path = a_parts.path.rstrip("/")
                if a_host and t_host and (
                        t_host == a_host
                        or t_host.endswith("." + a_host)) \
                        and (not a_path
                             or t_parts.path.startswith(a_path)):
                    return True
                continue
            a_low = a.lower()
            if t_host and (t_host == a_low
                           or t_host.endswith("." + a_low)):
                return True
            a_path = a.replace("\\", "/").rstrip("/")
            if t_path == a_path or t_path.startswith(a_path + "/"):
                return True
        return False


def default_book(workspace_dir: str | Path) -> AuditBook:
    """The runtime's book: beside the group store."""
    root = Path(workspace_dir) / "data" / "social"
    return AuditBook(root).load()
