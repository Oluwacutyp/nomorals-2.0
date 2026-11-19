"""Report drafts: platform templates + the engagement report.

Port of the ``report`` half of ``cbh/cli.py`` plus the ``report``
phase of ``engine/engine.py`` from elementalsouls/Claude-BugHunter
(MIT, Copyright (c) 2026 Sachin Sharma).

``parse_finding_metadata`` / ``render_report`` are verbatim:
H1 / Bugcrowd (VRT-aware severity-request paragraphs) /
Intigriti / Immunefi templates, built with plain string
concatenation to avoid textwrap.dedent issues with multi-line
interpolated content. Drafts only — nothing is ever sent from here.

Devon difference: the engagement report (``render_engagement_report``)
feeds Devon's audit book instead of writing loose files — the tool
surface journals it and saves the draft beside the engagement state.
The coverage caveats come from the closure ledger (``closure.py``),
never hand-written.
"""

from __future__ import annotations

import datetime
import os
import re
from typing import Any

__all__ = [
    "parse_finding_metadata",
    "render_report",
    "render_engagement_report",
]


def _mask_report_fences(body: str) -> str:
    """Hide fenced code from heading matching while retaining source offsets."""
    masked = []
    opener = None
    for line in body.splitlines(keepends=True):
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})([^\n]*)", line)
        if opener is not None:
            masked.append(re.sub(r"[^\n]", " ", line))
            if fence:
                marker, rest = fence.groups()
                if marker.startswith(opener) and not rest.strip():
                    opener = None
            continue
        if fence:
            marker, rest = fence.groups()
            if marker.startswith("~") or "`" not in rest:
                opener = marker
                masked.append(re.sub(r"[^\n]", " ", line))
                continue
        masked.append(line)
    return "".join(masked)


def parse_finding_metadata(text: str) -> dict[str, Any]:
    """Best-effort parse of YAML-ish frontmatter + section content."""
    md: dict[str, Any] = {"title": "", "severity": "Medium", "asset": "",
                          "endpoint": "", "summary": "", "steps": "",
                          "impact": "", "remediation": ""}
    # YAML frontmatter
    m = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.S)
    body = text
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                md[k.strip().lower()] = v.strip().strip('"').strip("'")
        body = m.group(2)
    # Match headings outside fenced code, then copy the original section text.
    headings_only = _mask_report_fences(body)
    sections = {
        "summary": ("summary", "description"),
        "steps": ("steps", "reproduction", "reproduce", "poc"),
        "impact": ("impact",),
        "remediation": ("remediation", "fix", "mitigation"),
    }
    for key, headings in sections.items():
        pattern = (r"^##[ \t]*(?:" + "|".join(headings) + r")[ \t]*\n"
                   r"(.*?)(?=^#{1,2}(?!#)|\Z)")
        match = re.search(pattern, headings_only, re.I | re.M | re.S)
        if match and not md.get(key):
            start, end = match.span(1)
            md[key] = body[start:end].strip()
    # First line as title if missing
    if not md["title"]:
        for line in headings_only.splitlines():
            line = line.strip()
            if line.startswith("# "):
                md["title"] = line[2:].strip()
                break
    return md


def render_report(md: dict[str, Any], platform: str) -> str:
    """Render a report draft using the report-writing + bugcrowd-reporting style."""
    title = md.get("title") or "Untitled finding"
    severity = md.get("severity") or "Medium"
    summary = md.get("summary") or "(fill in)"
    steps = md.get("steps") or "(fill in — curl commands per step)"
    impact = md.get("impact") or "(fill in — concrete dollar / PII / state impact)"
    remediation = md.get("remediation") or "(fill in)"
    asset = md.get("asset") or md.get("endpoint") or "(fill in)"
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "researcher"
    today = datetime.date.today().isoformat()

    if platform == "bugcrowd":
        return f"""# {title}

**Bug type (VRT):** _to be filled in — pick the closest match from VRT 1.x and include the manual override paragraph below if defaults underrate impact._
**Severity:** {severity}
**Asset:** {asset}
**Date:** {today}

## Severity request

The closest VRT category for this finding is _<VRT-path>_, which Bugcrowd defaults to **<default-severity>**. **I am requesting evaluation at {severity}** for the following reasons:

1. _<concrete impact reason>_
2. _<exploit complexity reason>_
3. _<chained-finding cross-reference if applicable>_

## Summary
{summary}

## Steps to reproduce
{steps}

## Impact
{impact}

## Suggested remediation
{remediation}

## Researcher account
- Bugcrowd handle: _<your-handle>_
- Test account email: _<your-alias>@bugcrowdninja.com_
"""

    if platform == "immunefi":
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower())
        return f"""# {title}

**Severity:** {severity}
**Chain ID / Contract:** {asset}
**Date:** {today}

## Summary
{summary}

## Vulnerability Details
{impact}

## Steps to reproduce (Foundry PoC required)
```bash
forge test --match-test test_{slug} -vvv
```
{steps}

## Proof of Concept
_Attach the Foundry test file producing the exploit._

## Suggested remediation
{remediation}
"""

    common = f"""# {title}

**Severity:** {severity}
**Asset:** {asset}
**Reporter:** {user}
**Date:** {today}

## Summary
{summary}

## Steps to reproduce
{steps}

## Impact
{impact}

## Suggested remediation
{remediation}
"""

    if platform == "intigriti":
        return (common + "\n## CVSS 3.1 vector\n"
                "`AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N` (fill in)\n")

    # Default: H1
    return common


def render_engagement_report(
        scope_name: str, in_scope: Any, summary: dict[str, Any],
        confirmed: list[dict[str, Any]],
        closure_decision: dict[str, Any],
        state: dict[str, Any]) -> str:
    """The engagement report (engine `report` phase, verbatim shape).

    Confirmed findings + the coverage verdict from the closure
    ledger. Coverage caveats are generated from the ledger, never
    hand-written.
    """
    from .closure import render_caveats

    lines = [f"# Engagement report — {scope_name}", "",
             f"In scope: `{'`, `'.join(in_scope)}`  ",
             f"Surface mapped: {summary['surface']} · "
             f"tested: {summary['tested']} · "
             f"candidates: {summary['candidates']} · "
             f"**confirmed: {summary['confirmed']}**  ",
             f"Coverage: **{'complete' if closure_decision['clean'] else 'INCOMPLETE'}**"
             f" — {closure_decision['reason']}", ""]
    if not confirmed:
        lines += ["No confirmed findings.", ""]
    flagged = [f for f in confirmed if f.get("scope_warning")]
    if flagged:
        lines += [f"> ⚠️ **{len(flagged)} finding(s) flagged: PoC referenced "
                  f"an out-of-scope host — review before reporting to the "
                  f"client.**", ""]
    for i, f in enumerate(confirmed, 1):
        block = [f"## {i}. {f.get('vuln_class', '?').upper()} — "
                 f"{f.get('severity', '?')}",
                 f"- **URL:** `{f['url']}`"
                 + (f" (param `{f['param']}`)" if f.get("param") else "")]
        if f.get("scope_warning"):
            block.append(f"- ⚠️ **SCOPE WARNING:** PoC referenced "
                         f"out-of-scope host(s): "
                         f"`{'`, `'.join(f['scope_warning'])}` — verify the "
                         f"in-scope finding holds without them.")
        if f.get("proof_note"):
            block.append(f"- **Proof note:** {f['proof_note']}")
        block += [f"- **Evidence:** {f.get('evidence', '')}",
                  f"- **Request:** `{f.get('request', '')}`",
                  f"- **Verifier:** {f.get('verdict', {}).get('reason', '')}",
                  ""]
        lines += block
    lines += ["", render_caveats(state, closure_decision)]
    return "\n".join(lines)
