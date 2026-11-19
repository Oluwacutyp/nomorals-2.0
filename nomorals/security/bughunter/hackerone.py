"""HackerOne read-only client (public GraphQL + gated Hacker API).

Port of ``mcp/hackerone/server.py`` from
elementalsouls/Claude-BugHunter (MIT, Copyright (c) 2026 Sachin
Sharma) — the client logic only, not the MCP-server surface:
Devon has no MCP-server surface for this.

Public tools need no key (disclosed-report duplicate search,
program stats and scope via public GraphQL). Logged-in tools
(``list_my_programs`` / ``get_my_program`` / ``list_my_reports``)
need the owner's HackerOne API credentials, which is his hand:
they are passed explicitly per call, never read from the
environment, never stored, never requested in chat. Without
them the gated functions refuse honestly and name the wall.

All HTTP goes through Devon's HttpClient (single TLS policy).
"""

from __future__ import annotations

import base64
import urllib.parse
from typing import Any

from ...core.errors import ToolError
from ...core.http import HttpClient
from ...core.logging_setup import get_logger

__all__ = [
    "search_disclosed_reports",
    "get_program",
    "get_program_scope",
    "list_my_programs",
    "get_my_program",
    "list_my_reports",
]

_log = get_logger(__name__)

GRAPHQL_URL = "https://hackerone.com/graphql"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Origin": "https://hackerone.com",
    "Referer": "https://hackerone.com/hacktivity",
}
REPORT_FIELDS = """... on HacktivityDocument {
  id
  severity_rating
  total_awarded_amount
  currency
  cwe
  cve_ids
  votes
  team { handle name }
  reporter { username }
  report { _id title url }
}"""
HACKER_API = "https://api.hackerone.com/v1/hackers"
TIMEOUT = 25
MAX_LIMIT = 50
SEVERITIES = ["none", "low", "medium", "high", "critical"]


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _graphql(query: str, client: HttpClient | None = None) -> dict[str, Any]:
    client = client or HttpClient(timeout=TIMEOUT)
    try:
        resp = client.post_json(GRAPHQL_URL, {"query": query},
                                headers=HEADERS, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - network failure is a ToolError
        raise ToolError(f"HackerOne GraphQL request failed: {exc}") from exc
    if resp.status >= 400:
        raise ToolError(f"HackerOne GraphQL returned HTTP {resp.status}")
    try:
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 - bad JSON is a ToolError
        raise ToolError(f"HackerOne GraphQL request failed: {exc}") from exc
    if not isinstance(data, dict):
        raise ToolError("HackerOne GraphQL request failed: bad response")
    if data.get("errors"):
        raise ToolError("HackerOne GraphQL error: " + "; ".join(
            str(err.get("message")) for err in data["errors"]))
    return data.get("data") or {}


def _hacker_api(path: str, params: dict[str, Any] | None = None, *,
                client: HttpClient | None = None,
                username: str = "", token: str = "") -> dict[str, Any]:
    """Read-only Hacker API call. Credentials are explicit per call."""
    if not username.strip() or not token.strip():
        raise ToolError(
            "HackerOne logged-in tools need the owner's HackerOne API "
            "credentials (HACKERONE_API_USERNAME + HACKERONE_API_TOKEN) — "
            "that is his hand. Pass username= and token= explicitly; "
            "Devon never stores them.")
    url = HACKER_API + path + (
        "?" + urllib.parse.urlencode(params) if params else "")
    auth = base64.b64encode(
        f"{username}:{token}".encode()).decode()
    client = client or HttpClient(timeout=TIMEOUT)
    try:
        resp = client.get(url, headers={
            "Authorization": "Basic " + auth,
            "Accept": "application/json"}, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - network failure is a ToolError
        raise ToolError(f"HackerOne Hacker API request failed: {exc}") from exc
    if resp.status in (401, 403):
        raise ToolError(
            f"HackerOne Hacker API refused the credentials "
            f"(HTTP {resp.status}).")
    if resp.status == 404:
        raise ToolError("Not found, or not visible to this account.")
    if resp.status >= 400:
        raise ToolError(
            f"HackerOne Hacker API returned HTTP {resp.status}")
    try:
        return resp.json()
    except Exception as exc:  # noqa: BLE001 - bad JSON is a ToolError
        raise ToolError(f"HackerOne Hacker API request failed: {exc}") from exc


_MAX_PAGES = 50  # hard cap: at page[size]=100 this is 5000 records


def _hacker_pages(path: str, limit: int, *,
                  client: HttpClient | None = None,
                  username: str = "", token: str = "") -> list[Any]:
    """Follow the Hacker API's ``links.next`` until ``limit`` records are collected.

    Bounded two ways so a misbehaving/MITM'd upstream cannot hang
    the caller: a hard page cap, and a stop as soon as a page yields
    zero new records.
    """
    out: list[Any] = []
    url_path: str = path
    params: dict[str, Any] = {"page[size]": 100}
    for _ in range(_MAX_PAGES):
        if not (url_path and len(out) < limit):
            break
        page = _hacker_api(url_path, params, client=client,
                           username=username, token=token)
        rows = page.get("data") or []
        if not rows:
            break
        out.extend(rows)
        nxt = (page.get("links") or {}).get("next")
        if not nxt:
            break
        parsed = urllib.parse.urlparse(nxt)
        url_path = parsed.path.replace("/v1/hackers", "", 1)
        params = dict(urllib.parse.parse_qsl(parsed.query))
    return out[:limit]


def _limit(args: dict[str, Any], default: int = 10,
           cap: int = MAX_LIMIT) -> int:
    try:
        n = int(args.get("limit", default))
    except (TypeError, ValueError):
        raise ToolError("limit must be a whole number") from None
    return max(1, min(n, cap))


def _handle(args: dict[str, Any]) -> str:
    handle = str(args.get("program") or "").strip()
    if not handle:
        raise ToolError(
            "program (the HackerOne handle, e.g. 'gitlab') is required")
    return handle


# ─── Public tools (no key) ────────────────────────────────────
def search_disclosed_reports(args: dict[str, Any],
                             client: HttpClient | None = None
                             ) -> dict[str, Any]:
    """Hacktivity search for duplicate checks and technique research."""
    query = str(args.get("query") or "").strip()
    program = str(args.get("program") or "").strip()
    if not query and not program:
        raise ToolError("Give a query, a program, or both.")
    terms = [f"({query})"] if query else []
    if program:
        terms.append(f"team_handle:{program}")
    sort = args.get("sort", "relevance")
    sort_str = {"votes": ', sort: {field: "votes", direction: DESC}',
                "bounty": ', sort: {field: "total_awarded_amount", '
                          'direction: DESC}'}.get(sort, "")
    severity = {s.lower() for s in (args.get("severity") or [])}
    limit = _limit(args)

    gql = (f'{{ search(index: CompleteHacktivityReportIndex, '
           f'query_string: "{_escape(" AND ".join(terms))}", '
           f'first: {MAX_LIMIT}{sort_str}) {{ total_count nodes {{ '
           f'{REPORT_FIELDS} }} }} }}')
    search = _graphql(gql, client).get("search") or {}

    reports = []
    for node in search.get("nodes") or []:
        report = node.get("report") or {}
        if not report:
            continue
        rating = (node.get("severity_rating") or "none").lower()
        if severity and rating not in severity:
            continue
        reports.append({
            "title": report.get("title"),
            "url": report.get("url"),
            "program": (node.get("team") or {}).get("handle"),
            "severity": rating,
            "cwe": node.get("cwe"),
            "cve_ids": node.get("cve_ids") or [],
            "bounty": node.get("total_awarded_amount"),
            "currency": (node.get("currency") or "usd").upper(),
            "votes": node.get("votes") or 0,
            "reporter": (node.get("reporter") or {}).get("username"),
        })
        if len(reports) >= limit:
            break
    return {"total_matches": search.get("total_count"),
            "returned": len(reports), "reports": reports}


def get_program(args: dict[str, Any],
                client: HttpClient | None = None) -> dict[str, Any]:
    """Bounty ranges, resolved count, launch date, state."""
    handle = _handle(args)
    team = _graphql(
        f'{{ team(handle: "{_escape(handle)}") {{ handle name url state '
        f'offers_bounties currency base_bounty resolved_report_count '
        f'average_bounty_lower_amount average_bounty_upper_amount '
        f'launched_at }} }}', client).get("team")
    if not team:
        raise ToolError(
            f"No public program '{handle}'. Private programs need the "
            "logged-in tools.")
    return {
        "program": team.get("handle"), "name": team.get("name"),
        "url": team.get("url"), "state": team.get("state"),
        "offers_bounties": team.get("offers_bounties"),
        "currency": (team.get("currency") or "usd").upper(),
        "base_bounty": team.get("base_bounty"),
        "average_bounty_range": [team.get("average_bounty_lower_amount"),
                                 team.get("average_bounty_upper_amount")],
        "resolved_reports": team.get("resolved_report_count"),
        "launched_at": (team.get("launched_at") or "")[:10],
    }


def get_program_scope(args: dict[str, Any],
                      client: HttpClient | None = None
                      ) -> dict[str, Any]:
    """Policy text and structured scope (in-scope assets, severity caps)."""
    handle = _handle(args)
    include_policy = bool(args.get("include_policy", True))
    team = _graphql(
        f'{{ team(handle: "{_escape(handle)}") {{ handle name '
        f'{"policy" if include_policy else ""} '
        f'structured_scopes(first: 100, archived: false) {{ total_count '
        f'nodes {{ asset_type asset_identifier eligible_for_bounty '
        f'eligible_for_submission max_severity instruction }} }} }} }}',
        client).get("team")
    if not team:
        raise ToolError(
            f"No public program '{handle}'. Private programs need the "
            "logged-in tools.")
    scopes = team.get("structured_scopes") or {}
    result: dict[str, Any] = {
        "program": team.get("handle"), "name": team.get("name"),
        "scope_total": scopes.get("total_count"),
        "in_scope": [s for s in (scopes.get("nodes") or [])
                     if s.get("eligible_for_submission")],
        "out_of_scope": [s for s in (scopes.get("nodes") or [])
                         if not s.get("eligible_for_submission")],
    }
    if include_policy:
        result["policy"] = team.get("policy")
    return result


# ─── Logged-in tools (owner's HackerOne credentials, read-only) ──
def list_my_programs(args: dict[str, Any],
                     client: HttpClient | None = None, *,
                     username: str = "", token: str = "") -> dict[str, Any]:
    """Programs your account can see, private invitations included."""
    limit = _limit(args, 100, 500)
    programs = []
    for item in _hacker_pages("/programs", limit, client=client,
                              username=username, token=token):
        a = item.get("attributes") or {}
        programs.append({"program": a.get("handle"), "name": a.get("name"),
                         "state": a.get("state"),
                         "offers_bounties": a.get("offers_bounties"),
                         "submission_state": a.get("submission_state"),
                         "started_accepting_at":
                             a.get("started_accepting_at")})
    return {"returned": len(programs), "programs": programs}


def get_my_program(args: dict[str, Any],
                   client: HttpClient | None = None, *,
                   username: str = "", token: str = "") -> dict[str, Any]:
    """One program's policy and full scope as your account sees it."""
    handle = _handle(args)
    item = _hacker_api(f"/programs/{urllib.parse.quote(handle)}",
                       client=client, username=username, token=token)
    a = item.get("attributes") or {}
    scopes = [s.get("attributes") or {} for s in
              ((item.get("relationships") or {})
               .get("structured_scopes") or {}).get("data") or []]
    keep = ("asset_type", "asset_identifier", "eligible_for_bounty",
            "eligible_for_submission", "max_severity", "instruction")
    scopes = [{k: s.get(k) for k in keep} for s in scopes]
    return {
        "program": a.get("handle"), "name": a.get("name"),
        "state": a.get("state"),
        "offers_bounties": a.get("offers_bounties"),
        "submission_state": a.get("submission_state"),
        "in_scope": [s for s in scopes if s.get("eligible_for_submission")],
        "out_of_scope": [s for s in scopes
                         if not s.get("eligible_for_submission")],
        "policy": a.get("policy"),
    }


def list_my_reports(args: dict[str, Any],
                    client: HttpClient | None = None, *,
                    username: str = "", token: str = "") -> dict[str, Any]:
    """Your own reports, optionally for one program."""
    program = str(args.get("program") or "").strip().lower()
    limit = _limit(args, 50, 500)
    reports = []
    for item in _hacker_pages("/me/reports",
                              1000 if program else limit, client=client,
                              username=username, token=token):
        a = item.get("attributes") or {}
        prog = (((item.get("relationships") or {}).get("program") or {})
                .get("data") or {})
        handle = (prog.get("attributes") or {}).get("handle")
        if program and (handle or "").lower() != program:
            continue
        reports.append({"id": item.get("id"), "title": a.get("title"),
                        "state": a.get("state"), "program": handle,
                        "created_at": a.get("created_at"),
                        "disclosed_at": a.get("disclosed_at")})
        if len(reports) >= limit:
            break
    return {"returned": len(reports), "reports": reports}
