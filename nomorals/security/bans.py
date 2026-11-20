"""Report / ban / appeal across platforms.

Three separate things, kept separate:

* REPORT — fire a platform's report endpoint from N authenticated
  sessions (from `session_forge`). Endpoints are versioned per platform
  and carry a `verified` flag: an unverified shape is attempted and
  reported as such, never presented as working.
* TRACK — a durable state machine per subject/platform
  (reported -> pending -> banned | survived -> appealing -> unbanned |
  upheld), so a campaign has a record.
* APPEAL — a draft per platform, plus a mass-appeal bundle. Drafts only.

Mass reporting is not one request: each platform needs its own session,
and a rate limit per session. `report_all` spreads the load and reports
per-platform outcome.
"""
from __future__ import annotations

import json
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.http import HttpClient
from ..core.policy import Capability
from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from .session_forge import _client_from, load as load_session, _root

_log = get_logger(__name__)
__all__ = ["PLATFORMS", "STATES", "report_one", "report_all", "set_state",
           "appeal_text", "mass_appeal", "register"]

STATES = ("reported", "pending", "banned", "survived", "appealing",
          "unbanned", "upheld")

#: platform -> (report URL template, method, auth cookie names, verified)
#: `{target}` is the handle. `verified: False` means the shape is a guess
#: that needs a live check before you trust a `sent` verdict.
PLATFORMS: dict[str, dict[str, Any]] = {
    "twitter": {
        "url": "https://api.twitter.com/1.1/users/report_spam.json"
               "?screen_name={target}&spam=true",
        "method": "POST", "domain": "x.com",
        "cookies": ("auth_token", "ct0"), "verified": True,
        "reason": "spam / fake engagement",
    },
    "instagram": {
        "url": "https://www.instagram.com/api/v1/web/reports/report/",
        "method": "POST", "domain": "instagram.com",
        "cookies": ("sessionid", "csrftoken"), "verified": False,
        "reason": "spam",
    },
    "reddit": {
        "url": "https://www.reddit.com/api/report",
        "method": "POST", "domain": "reddit.com",
        "cookies": ("reddit_session",), "verified": False,
        "reason": "spam",
    },
    "tiktok": {
        "url": "https://www.tiktok.com/api/report/",
        "method": "POST", "domain": "tiktok.com",
        "cookies": ("sessionid",), "verified": False,
        "reason": "spam",
    },
    "linkedin": {
        "url": "https://www.linkedin.com/li/track",
        "method": "POST", "domain": "linkedin.com",
        "cookies": ("li_at",), "verified": False,
        "reason": "spam",
    },
}


def report_one(platform: str, target: str, *, home: str | Path,
               username: str, reason: str = "",
               timeout: float = 25.0) -> dict[str, Any]:
    spec = PLATFORMS.get(platform)
    if spec is None:
        raise ToolError(f"unknown platform {platform!r}")
    stored = load_session(_root(home), spec["domain"], username)
    if not stored or not stored.get("session"):
        return {"platform": platform, "ok": False,
                "reason": f"no stored session for {username} on {spec['domain']}"}
    client = _client_from(stored, timeout, None)
    url = spec["url"].format(target=urllib.parse.quote(target))
    form = {"reason": reason or spec["reason"], "target": target}
    try:
        if spec["method"] == "POST":
            resp = client.post(url, data=form)
        else:
            resp = client.get(url)
    except Exception as exc:  # noqa: BLE001
        return {"platform": platform, "ok": False,
                "reason": f"{type(exc).__name__}: {exc}"[:120]}
    return {"platform": platform, "ok": resp.ok, "status": resp.status,
            "verified_shape": spec["verified"],
            "reason": "" if resp.ok else (resp.text or "")[:120]}


def report_all(target: str, *, home: str | Path, sessions: dict[str, str],
               platforms: Optional[list[str]] = None,
               reason: str = "") -> dict[str, Any]:
    """`sessions` maps platform -> the account that files the report."""
    wanted = platforms or list(sessions)
    out = []
    for platform in wanted:
        username = sessions.get(platform)
        if not username:
            continue
        out.append(report_one(platform, target, home=home, username=username,
                              reason=reason))
        time.sleep(1.0)
    return {"target": target, "sent": sum(1 for r in out if r.get("ok")),
            "attempted": len(out), "results": out}


# ── state tracking ──────────────────────────────────────────────────────────
def _state_path(home: str | Path) -> Path:
    path = Path(home) / "data" / "bans.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def set_state(home: str | Path, subject: str, platform: str, state: str,
              note: str = "") -> dict[str, Any]:
    if state not in STATES:
        raise ToolError(f"unknown state {state!r}")
    row = {"at": time.time(), "subject": subject.lower(), "platform": platform,
           "state": state, "note": note[:200]}
    with open(_state_path(home), "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def state_of(home: str | Path, subject: str) -> dict[str, str]:
    """Latest state per platform for a subject."""
    path = _state_path(home)
    latest: dict[str, str] = {}
    if not path.is_file():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("subject") == subject.lower():
            latest[row["platform"]] = row["state"]
    return latest


# ── appeals ────────────────────────────────────────────────────────────────
_APPEAL = """Subject: Appeal — @{handle} on {platform}

Hello,

My account @{handle} on {platform} was actioned on {date} for
{reason}. I believe this was {stance}.

{detail}

The account is {age} and holds work I rely on. I have re-read the
{platform} rules and {compliance}.

Could you re-review the decision, or tell me the specific rule that was
broken so I can fix it?

Thank you,
{name}
"""


def appeal_text(platform: str, handle: str, *, stance: str = "mistaken",
                detail: str = "", age: str = "several years old",
                compliance: str = "believe it is fully compliant",
                name: str = "", reason: str = "spam") -> str:
    return _APPEAL.format(handle=handle.lstrip("@"), platform=platform,
                          date=time.strftime("%Y-%m-%d"), reason=reason,
                          stance=stance, detail=detail or
                          "Nothing in my recent activity was automated and "
                          "nothing was posted in bulk.",
                          age=age, compliance=compliance, name=name or "")


def mass_appeal(home: str | Path, handle: str, *,
                platforms: Optional[list[str]] = None,
                out_dir: str | Path = "") -> dict[str, Any]:
    """One draft per platform, written to disk. Sending stays manual —
    each platform's appeal goes through a different form."""
    wanted = platforms or [p for p, s in state_of(home, handle).items()
                           if s in ("banned", "upheld")]
    target_dir = Path(out_dir) if out_dir else Path(home) / "data" / "appeals"
    target_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for platform in wanted:
        path = target_dir / f"{handle.lstrip('@')}-{platform}.txt"
        path.write_text(appeal_text(platform, handle), encoding="utf-8")
        written.append(str(path))
    return {"handle": handle, "platforms": wanted, "files": written}


def register(registry: Any) -> None:
    from .case import home_of

    context = registry.context
    home = Path(home_of(context))

    @registry.register("ban_report",
                       description="File a report against a handle on one "
                                   "platform using a stored session.",
                       capability=Capability.NET_OUT,
                       parameters={"platform": "str", "target": "str",
                                   "session_account": "str",
                                   "reason": "str (optional)"})
    def ban_report(platform: str, target: str, session_account: str,
                   *, reason: str = "") -> dict[str, Any]:
        result = report_one(platform, target, home=home,
                            username=session_account, reason=reason)
        set_state(home, target, platform,
                  "reported" if result.get("ok") else "pending",
                  result.get("reason", ""))
        return result

    @registry.register("ban_state",
                       description="Latest ban state per platform for a subject.",
                       capability=Capability.DB_READ,
                       parameters={"subject": "str"})
    def ban_state(subject: str) -> dict[str, Any]:
        return {"subject": subject, "states": state_of(home, subject)}

    @registry.register("ban_set_state",
                       description="Record a ban state transition.",
                       capability=Capability.DB_WRITE,
                       parameters={"subject": "str", "platform": "str",
                                   "state": "str", "note": "str (optional)"})
    def ban_set_state(subject: str, platform: str, state: str,
                      *, note: str = "") -> dict[str, Any]:
        return set_state(home, subject, platform, state, note)

    @registry.register("ban_appeal",
                       description="Generate appeal drafts for every banned "
                                   "platform of a handle.",
                       capability=Capability.DB_READ,
                       parameters={"handle": "str",
                                   "platforms": "str (optional) — comma list"})
    def ban_appeal(handle: str, *, platforms: str = "") -> dict[str, Any]:
        return mass_appeal(home, handle,
                           platforms=[p.strip() for p in platforms.split(",")
                                      if p.strip()] or None)
