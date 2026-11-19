"""BugHunter brain tools — the 8-tool surface over the bughunter organ.

Follows the security organ's exact pattern: the organ lives in
``nomorals/security/bughunter/``, the registry entries live here,
and ``nomorals/tools/security.py`` calls ``register(registry)``.
The 27 existing security tools are separate registry entries and
there is no single action-dispatched "security" tool to extend,
so bughunter registers its own 8, named like the organ's:

- ``bughunter_scope_check`` — deny-wins scope check for a URL.
- ``bughunter_classify`` — URL -> hunt skills + pattern pointers.
- ``bughunter_recon`` — deterministic recon (platform / OpenAPI /
  SPA mining) for a scope, scope-filtered.
- ``bughunter_surface`` — the recon manifest: ranked worklist +
  arsenal map for a scope.
- ``bughunter_triage`` — the 7-Question Gate over a finding.
- ``bughunter_report`` — render a finding / engagement report.
- ``bughunter_engage`` — run the full engine
  (recon,rank,map,hunt,validate,report); hunt/validate are opt-in
  and read-only by default, the model judges only through the
  injected ``suggest`` callable.
- ``bughunter_hackerone`` — public HackerOne reads (disclosed
  reports, program, scope); logged-in actions refuse honestly —
  they are the owner's hand.

The model callable follows the triage organ's exact pattern:
``context._bughunter_suggest`` wins (the test seam), else
Devon's brain via ``brain_for``. With no model, hunt/validate
record items as blocked, never as clean.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["register"]


def _suggest(context: Any):
    """A ``(prompt) -> text`` model callable, or None.

    A suggest passed in on the context (``context._bughunter_suggest``)
    wins — the test seam. Otherwise the brain, wrapped the way the
    triage ladder wraps it: a failed model call returns "" and the
    caller falls through to the code rung.
    """
    injected = getattr(context, "_bughunter_suggest", None)
    if injected is not None:
        return injected
    try:
        from ...llm.base import Message
        from ...llm.brain import brain_for
    except Exception:  # noqa: BLE001 - no brain, no model rung
        return None
    try:
        brain = brain_for(context)
    except Exception:  # noqa: BLE001 - no brain, no model rung
        return None

    def _call(prompt: str) -> str:
        try:
            resp = brain.chat(
                [Message(role="user", content=prompt)],
                task_kind="chat", timeout_s=30.0)
            return getattr(resp, "text", "") or ""
        except Exception:  # noqa: BLE001 - the model must never kill a job
            return ""

    return _call


def _home(context: Any) -> Path:
    """Devon's home from the tool context (disclose idiom)."""
    return Path(getattr(context, "home", ".") or ".")


def _book(context: Any):
    """The audit book, or None when it cannot be opened."""
    try:
        from ..audit_report import default_book
        settings = getattr(context, "settings", None)
        ws = getattr(settings, "workspace_dir", None) \
            or getattr(context, "home", ".")
        return default_book(ws)
    except Exception:  # noqa: BLE001 - journal is best-effort
        return None


def _split(value: str) -> list[str]:
    return [p.strip() for p in (value or "").split(",") if p.strip()]


def _load_scope(scope: str, name: str, in_scope: str, out_of_scope: str,
                seeds: str):
    """A Scope from a file path, an inline host list, or the params.

    ``scope`` may be a scope.md / scope.json file path (relative to
    the Devon home) or a comma-separated in-scope host list. The
    explicit ``in_scope`` / ``out_of_scope`` / ``seeds`` params win
    over the file when given.
    """
    from .scope import Scope, scope_from_md

    if scope:
        path = Path(scope)
        if not path.is_absolute():
            path = _home_current / path
        if path.is_file():
            loaded = scope_from_md(path) if path.suffix == ".md" \
                else Scope.load(path)
            if in_scope or out_of_scope or seeds:
                return Scope(
                    _split(in_scope) or loaded.in_scope,
                    _split(out_of_scope) or loaded.out_of_scope,
                    _split(seeds) or loaded.seeds,
                    name or loaded.name)
            return loaded
        return Scope(_split(scope), _split(out_of_scope),
                     _split(seeds), name or "engagement")
    return Scope(_split(in_scope), _split(out_of_scope), _split(seeds),
                 name or "engagement")


# Set at register() time so _load_scope can resolve relative paths.
_home_current: Path = Path(".")


def register(registry: Any) -> None:
    """Attach the 8 bughunter tools."""
    from ...core.policy import Capability

    global _home_current
    context = registry.context
    _home_current = _home(context)

    @registry.register(
        "bughunter_scope_check",
        description=(
            "Check whether a URL is in scope for a bug-bounty "
            "engagement (deny-wins, default-deny). scope= is a "
            "scope.md / scope.json path or a comma-separated host "
            "list; in_scope=/out_of_scope=/seeds= are comma-separated "
            "overrides. Returns in_scope + the reason when refused."
        ),
        capability=Capability.DB_READ,
        parameters={"scope": "str (optional) — file path or host list",
                    "url": "str — URL to check",
                    "name": "str (optional) — engagement name",
                    "in_scope": "str (optional) — comma-separated hosts",
                    "out_of_scope": "str (optional)",
                    "seeds": "str (optional) — comma-separated URLs"},
    )
    def bughunter_scope_check(url: str, *, scope: str = "",
                              name: str = "", in_scope: str = "",
                              out_of_scope: str = "",
                              seeds: str = "") -> dict[str, Any]:
        try:
            sc = _load_scope(scope, name, in_scope, out_of_scope, seeds)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        reason = sc.reject_reason(url)
        return {"ok": True, "url": url, "engagement": sc.name,
                "in_scope": reason is None, "reason": reason or ""}

    @registry.register(
        "bughunter_classify",
        description=(
            "Classify a URL into hunt skills: URL-pattern triggers "
            "plus description keywords over the 83 vendored skill "
            "packs, with pattern-library pointers. audit_boost= is a "
            "comma-separated list of vuln classes Devon's own audit "
            "already established for this URL — those rank first."
        ),
        capability=Capability.DB_READ,
        parameters={"url": "str",
                    "audit_boost": "str (optional) — comma-separated classes"},
    )
    def bughunter_classify(url: str, *,
                           audit_boost: str = "") -> dict[str, Any]:
        from .classify import classify_url
        try:
            result = classify_url(url, audit_boost=_split(audit_boost))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, **result}

    @registry.register(
        "bughunter_recon",
        description=(
            "Deterministic recon for an engagement scope: platform "
            "fingerprinting (well-known paths), OpenAPI discovery + "
            "auth-enforcement sweep, SPA mining (JS bundles, forms, "
            "secrets via Devon's audit patterns). Read-only by "
            "default; every mined URL is scope-filtered before it "
            "is returned. scope= is a file path or host list."
        ),
        capability=Capability.NET_OUT,
        parameters={"scope": "str (optional) — file path or host list",
                    "name": "str (optional) — engagement name",
                    "in_scope": "str (optional) — comma-separated hosts",
                    "out_of_scope": "str (optional)",
                    "seeds": "str (optional) — comma-separated URLs",
                    "allow_intrusive": "bool (optional) — default read-only"},
    )
    def bughunter_recon(*, scope: str = "", name: str = "",
                        in_scope: str = "", out_of_scope: str = "",
                        seeds: str = "",
                        allow_intrusive: bool = False) -> dict[str, Any]:
        from ...core.http import HttpClient
        from . import recon_map as _recon
        from .state import Engagement, engagements_root
        try:
            sc = _load_scope(scope, name, in_scope, out_of_scope, seeds)
            if not sc.seeds:
                return {"ok": False,
                        "error": "no seeds: give seeds= or a scope file "
                                 "with seeds"}
            http = HttpClient(timeout=20.0,
                              allow_private_ips=sc.allows_private())
            eng = Engagement(engagements_root(_home(context)), sc.name)
            eng.state["scope"] = sc.to_dict()
            eng.save()
            plat = _recon.platform_recon(sc, http, log=eng.log)
            api = _recon.openapi_recon(sc, http, log=eng.log,
                                       allow_intrusive=allow_intrusive)
            endpoints, oos = _recon.spa_recon(sc, http, log=eng.log)
            items = list(api) + list(plat)
            for it in endpoints:
                for cls in _recon.classify_all(it):
                    e = dict(it)
                    e["vuln_class"] = cls
                    e.pop("vuln_classes", None)
                    items.append(e)
            kept = [it for it in items if sc.in_scope_host(it["url"])]
            eng.add_surface(kept)
            return {"ok": True, "engagement": sc.name,
                    "dir": str(eng.dir),
                    "platform": len(plat), "openapi": len(api),
                    "endpoints": len(endpoints),
                    "surface": len(kept),
                    "out_of_scope_seen": sorted(oos),
                    "items": kept}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    @registry.register(
        "bughunter_surface",
        description=(
            "The recon manifest for a scope: run recon, rank the "
            "surface, and show the arsenal map (per target -> hunt "
            "skills + first curl). The default deliverable — the "
            "engine STOPS here unless bughunter_engage is asked to "
            "hunt. scope= is a file path or host list."
        ),
        capability=Capability.NET_OUT,
        parameters={"scope": "str (optional) — file path or host list",
                    "name": "str (optional) — engagement name",
                    "in_scope": "str (optional) — comma-separated hosts",
                    "out_of_scope": "str (optional)",
                    "seeds": "str (optional) — comma-separated URLs"},
    )
    def bughunter_surface(*, scope: str = "", name: str = "",
                          in_scope: str = "", out_of_scope: str = "",
                          seeds: str = "") -> dict[str, Any]:
        from ...core.http import HttpClient
        from .engine import Engine
        try:
            sc = _load_scope(scope, name, in_scope, out_of_scope, seeds)
            if not sc.seeds:
                return {"ok": False,
                        "error": "no seeds: give seeds= or a scope file "
                                 "with seeds"}
            eng = Engine(sc, home=_home(context),
                         http_client=HttpClient(
                             timeout=20.0,
                             allow_private_ips=sc.allows_private()))
            eng.recon()
            eng.rank()
            arsenal = eng.map()
            return {"ok": True, "engagement": sc.name,
                    "dir": str(eng.eng.dir),
                    "summary": eng.eng.summary(),
                    "arsenal": arsenal}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    @registry.register(
        "bughunter_triage",
        description=(
            "Run the 7-Question Gate over a finding before any "
            "report is drafted: real HTTP proof, accepted impact, "
            "in scope, no privileged access, not known/duplicate, "
            "provable impact, not on the never-submit list. "
            "finding= is a JSON finding dict (url, vuln_class, "
            "severity, evidence, request). Verdict PASS / DOWNGRADE "
            " / KILL. Q2 additionally passes when Devon's severity "
            "rubric scores the finding medium-or-better."
        ),
        capability=Capability.DB_READ,
        parameters={"finding": "str — JSON finding dict"},
    )
    def bughunter_triage(finding: str) -> dict[str, Any]:
        from .triage import gate_finding
        try:
            data = json.loads(finding)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"finding is not JSON: {exc}"}
        if not isinstance(data, dict):
            return {"ok": False, "error": "finding must be a JSON object"}
        return {"ok": True, **gate_finding(data)}

    @registry.register(
        "bughunter_report",
        description=(
            "Render a report draft for a finding "
            "(platform=hackerone/bugcrowd/intigriti/immunefi) or, "
            "with engagement=, the full engagement report for a "
            "finished run — confirmed findings plus the coverage "
            "verdict from the closure ledger. finding= is a JSON "
            "finding dict."
        ),
        capability=Capability.DB_READ,
        parameters={"finding": "str (optional) — JSON finding dict",
                    "platform": "str (optional) — default hackerone",
                    "engagement": "str (optional) — engagement name"},
    )
    def bughunter_report(*, finding: str = "", platform: str = "hackerone",
                         engagement: str = "") -> dict[str, Any]:
        from . import closure
        from .report import render_engagement_report, render_report
        from .state import Engagement, engagements_root
        try:
            if engagement:
                eng = Engagement(engagements_root(_home(context)),
                                 engagement)
                decision = closure.evaluate(eng.state)
                scope = eng.state.get("scope") or {}
                text = render_engagement_report(
                    scope.get("name") or engagement,
                    scope.get("in_scope") or [], eng.summary(),
                    eng.state["confirmed"], decision, eng.state)
                return {"ok": True, "engagement": engagement,
                        "dir": str(eng.dir), "report": text,
                        "coverage_clean": decision["clean"]}
            if not finding:
                return {"ok": False,
                        "error": "give finding= (JSON) or engagement="}
            data = json.loads(finding)
            if not isinstance(data, dict):
                return {"ok": False, "error": "finding must be a JSON object"}
            return {"ok": True,
                    "report": render_report(data, platform or "hackerone")}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    @registry.register(
        "bughunter_engage",
        description=(
            "Run a full engagement: recon, rank, map, then — only "
            "with hunt=true — hunt, validate and report. Read-only "
            "by default (allow_intrusive=true opts into intrusive "
            "PoCs). The model judges only in hunt/validate through "
            "Devon's brain; with no model those phases record items "
            "as blocked, never as clean. scope= is a file path or "
            "host list; seeds= are the starting URLs. Confirmed "
            "findings go into a hash-chained evidence pack and the "
            "audit book."
        ),
        capability=Capability.NET_OUT,
        parameters={"scope": "str (optional) — file path or host list",
                    "name": "str (optional) — engagement name",
                    "in_scope": "str (optional) — comma-separated hosts",
                    "out_of_scope": "str (optional)",
                    "seeds": "str (optional) — comma-separated URLs",
                    "hunt": "bool (optional) — default false (map only)",
                    "allow_intrusive": "bool (optional) — default read-only",
                    "max_hunts": "int (optional) — default 10"},
    )
    def bughunter_engage(*, scope: str = "", name: str = "",
                         in_scope: str = "", out_of_scope: str = "",
                         seeds: str = "", hunt: bool = False,
                         allow_intrusive: bool = False,
                         max_hunts: int = 10) -> dict[str, Any]:
        from .engine import run_engagement
        try:
            sc = _load_scope(scope, name, in_scope, out_of_scope, seeds)
            if not sc.seeds:
                return {"ok": False,
                        "error": "no seeds: give seeds= or a scope file "
                                 "with seeds"}
            out = run_engagement(sc, hunt=bool(hunt),
                                 suggest=_suggest(context),
                                 max_hunts=int(max_hunts),
                                 allow_intrusive=bool(allow_intrusive),
                                 book=_book(context),
                                 home=_home(context))
            return {"ok": True, **out}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    @registry.register(
        "bughunter_hackerone",
        description=(
            "Public HackerOne reads (no key): action=search finds "
            "disclosed reports (query=, program=, severity=, "
            "cwe=, limit=); action=program shows a program's bounty "
            "ranges and resolved count (program=); action=scope shows "
            "its policy and structured scope (program=). Logged-in "
            "actions (my_programs, my_program, my_reports) need a "
            "HackerOne login — the owner's hand — and refuse honestly "
            "here; pass username= and token= per call, never stored."
        ),
        capability=Capability.NET_OUT,
        parameters={"action": "str — search/program/scope/my_programs/my_program/my_reports",
                    "query": "str (optional)",
                    "program": "str (optional) — program handle",
                    "severity": "str (optional)",
                    "cwe": "str (optional)",
                    "limit": "int (optional) — default 10",
                    "username": "str (optional) — per-call only",
                    "token": "str (optional) — per-call only, never stored"},
    )
    def bughunter_hackerone(action: str = "search", *, query: str = "",
                            program: str = "", severity: str = "",
                            cwe: str = "", limit: int = 10,
                            username: str = "",
                            token: str = "") -> dict[str, Any]:
        from . import hackerone as _h1
        try:
            act = (action or "search").strip().lower()
            if act == "search":
                return {"ok": True, **_h1.search_disclosed_reports(
                    {"query": query, "program": program,
                     "severity": severity, "cwe": cwe, "limit": limit})}
            if act == "program":
                return {"ok": True, **_h1.get_program(
                    {"handle": program})}
            if act == "scope":
                return {"ok": True, **_h1.get_program_scope(
                    {"handle": program})}
            if act in ("my_programs", "my_program", "my_reports"):
                if not username or not token:
                    return {"ok": False,
                            "error": "logged-in HackerOne tools need a "
                                     "login — the owner's hand. Pass "
                                     "username= and token= per call; "
                                     "Devon never stores them."}
                fn = {"my_programs": _h1.list_my_programs,
                      "my_program": _h1.get_my_program,
                      "my_reports": _h1.list_my_reports}[act]
                return {"ok": True, **fn(
                    {"handle": program, "program": program,
                     "limit": limit},
                    username=username, token=token)}
            return {"ok": False,
                    "error": f"unknown action {act!r}: search, program, "
                             "scope, my_programs, my_program, my_reports"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
