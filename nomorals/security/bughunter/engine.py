"""Autonomous engagement orchestrator.

Port of ``engine/engine.py`` from elementalsouls/Claude-BugHunter
(MIT, Copyright (c) 2026 Sachin Sharma), same phases and rules:
scope -> recon -> rank -> map -> hunt -> validate -> report.
Scope is enforced in code at every boundary; state is persisted
after every step, so a run is auditable and resumable.

Devon differences from upstream:

- The judge is Devon's brain through an injected ``suggest``
  callable (``(prompt) -> text``, the triage-organ pattern) —
  no ``claude -p`` subprocess, no model name, no
  ``bypassPermissions``. With no model the hunt/validate phases
  record items as ``blocked`` ("no-model"), never as clean.
- State lives under Devon's home (``state.py``), not
  ``~/.bughunter-engagements``; the run is journaled into the
  audit book (``AuditBook``) when one is passed, and every
  confirmed finding is routed through a hash-chained evidence
  pack (``security/evidence.py``) with secrets masked at rest.
- Recon deduplicates against Devon's own passive recon
  (``security/recon.py``: crt.sh, takeover) instead of
  re-implementing it, and all HTTP goes through Devon's
  HttpClient. ``gau`` / ``katana`` stay optional.
- There is no separate ``osint`` phase: subdomain enumeration is
  Devon's recon organ's job and runs inside ``recon``.
- ``worklist_from_audit`` is new: it converts Devon's defensive
  audit findings (``audit_web`` / ``audit_repo``) into the same
  (target, class) work units the engine ranks — defensive audit
  and offensive hunting share one ranked worklist.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ...core.http import HttpClient
from ...core.logging_setup import get_logger
from . import agent as _agent
from . import closure, memory
from . import recon_map as _recon
from . import skillmap as _skillmap
from .report import render_engagement_report
from .scope import Scope
from .state import Engagement, engagements_root

__all__ = [
    "CLASS_WEIGHT",
    "ALL_PHASES",
    "DEFAULT_PHASES",
    "Engine",
    "worklist_from_audit",
    "run_engagement",
]

_log = get_logger(__name__)

#: Deterministic priority by class — kept in skillmap with the
#: other routing tables; re-exported here (upstream kept it here).
CLASS_WEIGHT = _skillmap.CLASS_WEIGHT

ALL_PHASES = ["recon", "rank", "map", "hunt", "validate", "report"]
DEFAULT_PHASES = ["recon", "rank", "map"]


def _host_of_url(url: str) -> str:
    u = url or ""
    if "://" not in u:
        u = "//" + u
    return (urlparse(u).hostname or "").lower().rstrip(".")


#: Devon's audit finding kind -> hunt class, for the shared worklist.
#: ``audit_web`` kinds are check names; ``audit_repo`` kinds are rule
#: names. Only the mappings with a real class land here — an unknown
#: kind becomes ``info-leak`` through the classifier, never dropped.
KIND_CLASS = {
    "sqli": "sqli", "sql_injection": "sqli", "string_sql": "sqli",
    "ssrf": "ssrf", "server_side_request_forgery": "ssrf",
    "idor": "idor", "broken_access_control": "auth-bypass",
    "auth_bypass": "auth-bypass", "auth-bypass": "auth-bypass",
    "xss": "xss", "xss_stored": "xss", "xss_reflected": "xss",
    "rce": "rce", "command_injection": "rce",
    "deserialization": "deserialization", "unsafe_deserialization": "deserialization",
    "xxe": "xxe", "xml_external_entity": "xxe",
    "ssti": "ssti", "template_injection": "ssti",
    "lfi": "lfi", "path_traversal": "lfi",
    "secret_leak": "secret", "exposed_secret": "secret",
    "jwt": "jwt-crypto", "weak_jwt": "jwt-crypto",
    "open_redirect": "open-redirect", "open-redirect": "open-redirect",
    "cors_wildcard": "cors", "csrf": "csrf",
    "file_upload": "file-upload",
    "exposed_data": "info-leak", "info": "info-leak",
    "unauthenticated_api": "auth-bypass",
    "unauthenticated_backend": "auth-bypass",
}


def worklist_from_audit(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Devon audit findings into (target, class) work units.

    A web finding with an unauthenticated endpoint keeps its URL
    and gains the classes the deterministic classifier assigns it;
    a repo finding with a sink kind maps through KIND_CLASS. A
    secret finding becomes a ``secret`` unit (the +55 rank boost
    applies in ``rank``). Findings without a usable target are
    skipped and counted in the returned ``skipped`` note — never
    silently treated as covered.
    """
    units: list[dict[str, Any]] = []
    skipped = 0
    for f in findings or []:
        target = str(f.get("location") or f.get("url")
                     or f.get("target") or "").strip()
        if not target:
            skipped += 1
            continue
        kind = str(f.get("kind") or f.get("check") or "").lower()
        cls = KIND_CLASS.get(kind)
        if cls is None:
            classes = _recon.classify_all({"url": target, "param": ""})
            cls = classes[0] if classes else "info-leak"
        unit: dict[str, Any] = {
            "url": target, "param": "", "vuln_class": cls,
            "source": f"audit:{f.get('source') or 'devon'}",
            "note": str(f.get("title") or f.get("detail") or "")[:200],
        }
        if kind in ("secret_leak", "exposed_secret"):
            unit["vuln_class"] = "secret"
            unit["source"] = "secret"
        units.append(unit)
    return units


class Engine:
    """One engagement run: scope -> recon -> rank -> map -> hunt -> validate -> report."""

    def __init__(self, scope: Scope | dict[str, Any],
                 base: str | Path | None = None, *,
                 suggest: Callable[[str], str] | None = None,
                 max_hunts: int = 10, allow_intrusive: bool = False,
                 parallel: int = 3, expand: bool = False,
                 use_memory: bool = False, mock: bool = False,
                 book: Any = None, home: str | Path | None = None,
                 http_client: HttpClient | None = None) -> None:
        self.scope = scope if isinstance(scope, Scope) \
            else Scope.from_dict(scope)
        self.home = Path(home) if home is not None else Path.home()
        self.base = Path(base) if base is not None \
            else engagements_root(self.home)
        self.eng = Engagement(self.base, self.scope.name)
        self.eng.state["scope"] = self.scope.to_dict()
        self.eng.save()
        self.suggest = suggest
        self.max_hunts = max_hunts
        self.allow_intrusive = allow_intrusive
        self.parallel = max(1, parallel)
        self.expand = expand
        self.use_memory = use_memory and not mock
        self.mock = mock
        self.book = book
        self.memory_root = memory.memory_root(self.home)
        self.http = http_client or HttpClient(
            timeout=20.0, allow_private_ips=self.scope.allows_private())
        self.eng.log(
            f"engine start | scope={self.scope.name} "
            f"in={self.scope.in_scope} out={self.scope.out_of_scope} "
            f"seeds={self.scope.seeds} mock={mock} "
            f"intrusive={'ALLOWED' if allow_intrusive else 'READ-ONLY (default)'}")

    # ── rules of engagement (injected into every agent) ──
    def _roe(self) -> str:
        hosts = ", ".join(self.scope.in_scope)
        roe = ("RULES OF ENGAGEMENT (mandatory — authorized SOW):\n"
               f"- SCOPE: ONLY these hosts are in scope: {hosts}. NEVER send a request to ANY other host "
               "(no third-party APIs / CDNs / SaaS such as emailjs, salesforce, google, etc.), even to 'prove' a "
               "finding. If proof would require touching an out-of-scope host/service, DO NOT do it — report the "
               "finding but set proof_note='would require out-of-scope action (NOT performed)'.\n")
        if not self.allow_intrusive:
            roe += ("- READ-ONLY / NON-DESTRUCTIVE: do NOT perform any state-changing or intrusive action — no sending "
                    "emails, no writes/updates/deletes, no cache purge/revalidate, no account creation, no spam, no "
                    "flooding/DoS, and do NOT exercise exposed credentials. Demonstrate a finding with the minimum safe, "
                    "read-only evidence (SHOW that something is exposed; do NOT use it). A finding you cannot demonstrate "
                    "safely read-only -> report it with proof_note='requires intrusive validation (NOT performed)'.\n")
        else:
            roe += "- Intrusive actions are AUTHORIZED for this run; still prefer the least-impactful PoC and log what you did.\n"
        return roe

    def _scope_audit(self, f: dict[str, Any]) -> list[str]:
        """Deterministic safety net: which out-of-scope hosts did the agent's PoC touch?"""
        blob = " ".join([str(f.get("request", "")), str(f.get("evidence", "")),
                         " ".join(f.get("hosts_contacted") or [])])
        hosts = set(re.findall(r'https?://([A-Za-z0-9.\-]+)', blob)) \
            | set(f.get("hosts_contacted") or [])
        return sorted({h for h in hosts
                       if h and not self.scope.in_scope_host(h)})

    # ── recon (deterministic, SPA-aware, per target) ──
    def recon(self) -> None:
        self.eng.set_phase("recon")
        if self.mock:
            items = []
            for seed in self.scope.seeds:
                items += [it for it in MOCK_RECON.get(seed, [])
                          if self.scope.in_scope_host(it.get("url", ""))]
            self.eng.add_surface(items)
            self.eng.log(f"recon(mock): {len(items)} item(s)")
            return
        # Devon's own passive recon first (crt.sh, takeover) — the
        # organ already owns it; bughunter does not re-implement it.
        try:
            from ..recon import recon as _devon_recon
            for seed in self.scope.seeds:
                apex = _host_of_url(seed)
                if not apex:
                    continue
                passive = _devon_recon(apex, include_ct=True, do_axfr=False)
                for asset in passive.get("assets", []):
                    host = asset.get("host", "")
                    if host and asset.get("takeover"):
                        self.eng.log(
                            f"recon: takeover lead {host} — "
                            f"{asset['takeover']}")
                self.eng.state.setdefault("targets", []).extend(
                    [{"host": a.get("host"), "tech": []}
                     for a in passive.get("assets", [])
                     if a.get("host")])
        except Exception as exc:  # noqa: BLE001 - passive recon is best-effort
            self.eng.log(f"recon: Devon passive recon skipped ({exc})")
        self.eng.save()
        plat = _recon.platform_recon(self.scope, self.http,
                                     log=self.eng.log)
        api = _recon.openapi_recon(self.scope, self.http,
                                   log=self.eng.log,
                                   allow_intrusive=self.allow_intrusive)
        endpoints, oos = _recon.spa_recon(self.scope, self.http,
                                          log=self.eng.log)
        params: list[dict[str, Any]] = []
        if self.expand:
            params = _recon.gather_params(self.scope, self.http,
                                          log=self.eng.log)
        # Categorize every (endpoint | param) into its applicable
        # attack classes, expand to work units.
        expanded = list(api) + list(plat)  # already categorized
        for it in endpoints:
            for cls in _recon.classify_all(it):
                e = dict(it)
                e["vuln_class"] = cls
                e.pop("vuln_classes", None)
                expanded.append(e)
        for it in params:
            for cls in _recon.classes_for_param(it.get("param", ""),
                                                it.get("url", "")):
                e = dict(it)
                e["vuln_class"] = cls
                expanded.append(e)
        # Scope-filter every item: mining can surface absolute URLs
        # to hosts outside the engagement.
        in_scope, dropped = [], 0
        for it in expanded:
            if self.scope.in_scope_host(it.get("url", "")):
                in_scope.append(it)
            else:
                dropped += 1
        added = self.eng.add_surface(in_scope)
        if oos:
            self.eng.log(f"recon: {len(oos)} out-of-scope host(s) seen, "
                         f"NOT tested: {', '.join(oos[:8])}")
        if dropped:
            self.eng.log(f"recon: {dropped} mined item(s) dropped as "
                         "out-of-scope")
        self.eng.log(f"recon: {len(api)} api + {len(plat)} platform + "
                     f"{len(endpoints)} endpoint + {len(params)} param "
                     f"-> {added} categorized (target,class) work unit(s)")

    # ── rank ──
    def rank(self) -> None:
        self.eng.set_phase("rank")
        for s in self.eng.state["surface"]:
            s["priority"] = _skillmap.rank_priority(s)
        self.eng.save()
        wl = self.eng.worklist()
        self.eng.log(f"rank: {len(wl)} item(s) prioritized; top: "
                     + ", ".join(f"{s['vuln_class']}@{s['url']}"
                                 for s in wl[:3]))

    # ── map (arsenal categorization — the default deliverable) ──
    def map(self) -> str:
        """Categorize the surface against the skill arsenal and SHOW it.

        Per target -> applicable hunt-* skill(s) + first curl. This
        is where the operator focuses; the engine does NOT auto-test
        by default (run with hunt=True to spend agent calls).
        """
        self.eng.set_phase("map")
        from collections import OrderedDict
        tech = sorted({t for tg in self.eng.state.get("targets", [])
                       for t in tg.get("tech", [])})
        groups: dict = OrderedDict()
        for s in sorted(self.eng.state["surface"],
                        key=lambda z: -z.get("priority", 0)):
            key = (s["url"], s.get("param", ""))
            g = groups.setdefault(key, {"classes": [],
                                        "src": s.get("source", "")})
            if s["vuln_class"] not in g["classes"]:
                g["classes"].append(s["vuln_class"])
        tech_sk = _skillmap.tech_skills(tech)
        lines = [f"# Arsenal map — {self.scope.name}", "",
                 f"**Service / tech:** {', '.join(tech) or '?'}  ",
                 f"**Tech-wide skills:** "
                 f"{', '.join('`' + s + '`' for s in tech_sk) or 'none'}  ",
                 f"**Attack surface:** {len(groups)} target(s). For each: "
                 "applicable skill(s) + the first curl.",
                 "",
                 "> Active testing is curl-first; Burp MCP only for "
                 "OOB/blind/fuzzing. Run the engine with hunt=True to "
                 "auto-test these with agents.", ""]
        self.eng.log(f"map: ARSENAL — tech-skills: "
                     f"{', '.join(tech_sk) or 'none'} | "
                     f"{len(groups)} target(s):")
        for (url, param), g in groups.items():
            label = param if param else "(endpoint)"
            per = "  ".join(
                f"{c}→{','.join(_skillmap.skills_for(c))}"
                for c in g["classes"])
            self.eng.log(f"map:   {label:16.16s} {url[:42]:42.42s}  {per}")
            lines.append(f"## `{label}`  —  {', '.join(g['classes'])}")
            lines.append(f"- **URL:** `{url}`"
                         + (f"  ·  source: {g['src']}" if g["src"] else ""))
            for c in g["classes"]:
                lines.append(
                    f"- **{c}** → "
                    f"{', '.join('`' + s + '`' for s in _skillmap.skills_for(c))}  ·  "
                    f"`{_skillmap.probe_for(c, url, param)}`")
            lines.append("")
        path = self.eng.dir / "arsenal.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        self.eng.log(f"map: wrote arsenal map -> {path}  ← focus your 20% "
                     "here (or run with hunt=True to auto-test)")
        return "\n".join(lines)

    # ── parallel runner ──
    def _run_parallel(self, fn: Callable, items: list[dict[str, Any]]):
        """Run fn(item) across self.parallel workers; yield (item, result) as each finishes.

        Workers do only the slow agent call (NO state mutation);
        the caller serializes all state writes, so there's no race
        on state.json.
        """
        if self.parallel <= 1 or len(items) <= 1:
            for it in items:
                try:
                    result = fn(it)
                except Exception as exc:  # noqa: BLE001 - worker error is data
                    self.eng.log(f"  worker error on {it.get('url')}: {exc}")
                    result = {}
                yield it, result
            return
        with ThreadPoolExecutor(max_workers=self.parallel) as ex:
            futs = {ex.submit(fn, it): it for it in items}
            for fut in as_completed(futs):
                it = futs[fut]
                try:
                    yield it, fut.result()
                except Exception as exc:  # noqa: BLE001 - worker error is data
                    self.eng.log(f"  worker error on {it.get('url')}: {exc}")
                    yield it, {}

    def _tech(self) -> list[str]:
        """Sorted union of tech fingerprints across recon targets."""
        return sorted({t for tg in self.eng.state.get("targets", [])
                       for t in tg.get("tech", [])})

    # ── hunt ──
    def hunt(self) -> None:
        self.eng.set_phase("hunt")
        wl = self.eng.worklist()[: self.max_hunts]
        todo = []
        tech = self._tech()
        for it in wl:
            if not self.scope.in_scope_host(it["url"]):
                # Belt-and-suspenders scope gate.
                self.eng.log(f"hunt: REFUSING out-of-scope {it['url']} "
                             f"({self.scope.reject_reason(it['url'])})")
                self.eng.mark_tested(it, coverage="out_of_scope",
                                     note=self.scope.reject_reason(it["url"]))
                continue
            if self.use_memory:
                d = memory.skip_decision(_host_of_url(it["url"]), tech, it,
                                         root=self.memory_root)
                if d["skip"]:
                    self.eng.log(f"hunt: skipped via ledger ({d['reason']}): "
                                 f"{it.get('vuln_class')}@{it['url']}")
                    if d["carry"]:
                        self.eng.confirm(d["carry"], d["carry"]["verdict"])
                    self.eng.mark_tested(
                        it, coverage="finding" if d["carry"] else "clean")
                    continue
            todo.append(it)
        self.eng.log(f"hunt: testing {len(todo)} item(s), "
                     f"{self.parallel} at a time")
        for it, f in self._run_parallel(self._hunt_agent, todo):
            if f and f.get("rate_limited"):
                self.eng.log("hunt: claude usage limit hit — remaining "
                             "results may be partial")
            if (not f) or f.get("errored"):
                # Not actually tested (agent hard error / worker
                # exception): do NOT mark tested, so the item stays
                # on the worklist and is retried on the next run.
                reason = (f or {}).get("error", "worker-exception")
                self.eng.log(f"hunt: NOT tested ({reason}) — left on "
                             f"worklist: {it.get('vuln_class')}@{it.get('url')}")
                # Blocked is NOT covered — it ships as a caveat until retried.
                self.eng.record_coverage(it, "blocked", reason)
                continue
            if f.get("vulnerable"):
                oos = self._scope_audit(f)
                cand = {"url": it["url"], "param": it.get("param", ""),
                        "vuln_class": it.get("vuln_class"),
                        "severity": f.get("severity", "unknown"),
                        "evidence": f.get("evidence", ""),
                        "request": f.get("request", ""),
                        "proof_note": f.get("proof_note", ""),
                        "scope_warning": oos or None}
                self.eng.mark_tested(it, candidate=cand, coverage="finding")
                if oos:
                    self.eng.log(
                        f"hunt: CANDIDATE {cand['vuln_class']} @ {it['url']} "
                        f"— PoC touched OUT-OF-SCOPE host(s): "
                        f"{', '.join(oos)} (flagged)")
                else:
                    self.eng.log(f"hunt: CANDIDATE {cand['vuln_class']} "
                                 f"@ {it['url']}")
            else:
                self.eng.mark_tested(it, coverage="clean")
                self.eng.log(f"hunt: nothing on {it.get('vuln_class')}"
                             f"@{it['url']}")

    def _hunt_agent(self, it: dict[str, Any]) -> dict[str, Any]:
        if self.mock:
            return MOCK_HUNT.get((it["url"], it.get("vuln_class")),
                                 {"vulnerable": False})
        tech = self._tech()
        skills = _skillmap.skills_for(it.get("vuln_class", "")) \
            + _skillmap.tech_skills(tech)
        probe = _skillmap.probe_for(it.get("vuln_class", ""), it["url"],
                                    it.get("param"))
        task = (self._roe() + "\n"
                f"TASK: Test {it['url']} (parameter `{it.get('param', '')}`) for {it.get('vuln_class')}, strictly "
                f"within the rules of engagement above. Apply the methodology of: {', '.join(skills)}. "
                f"curl-FIRST: use curl for your requests; use the Burp MCP tools ONLY if you genuinely need "
                f"OOB/blind/fuzzing (e.g. Collaborator for blind SSRF/XXE). Starter probe: {probe} . "
                f"Only claim a vulnerability if you can DEMONSTRATE real, exploitable impact (not mere reflection, "
                f"an error, or a permissive header) using ONLY in-scope, read-only actions. "
                f"End with a fenced ```json``` object: "
                f'{{"vulnerable":true|false,"severity":"low|medium|high|critical","evidence":"<what proves it>",'
                f'"request":"<the winning request>","hosts_contacted":["every host you sent a request to"],'
                f'"proof_note":"<empty, or why proof was limited by scope/read-only>"}}.')
        r = _agent.run_agent(task, suggest=self.suggest)
        if r.get("error"):
            self.eng.log(f"hunt agent error ({it['url']}): {r['error']}")
            # Hard error (no model / timeout / parse): the item was
            # NOT actually tested. Flag it so the caller leaves it
            # on the worklist for retry.
            return {"vulnerable": False, "errored": True,
                    "error": r["error"],
                    "rate_limited": r["error"] == "rate-limited"}
        verdict = _agent.extract_json(r["result"])
        if not isinstance(verdict, dict) \
                or not isinstance(verdict.get("vulnerable"), bool):
            # Agent ran but produced no usable verdict — inconclusive,
            # not a real negative.
            self.eng.log(f"hunt: no usable verdict for {it['url']} — "
                         "inconclusive, left for retry")
            return {"vulnerable": False, "errored": True,
                    "error": "no-verdict"}
        return verdict

    # ── validate ──
    def validate(self) -> None:
        self.eng.set_phase("validate")
        confirmed_keys = {self.eng._key(finding)
                          for finding in self.eng.state["confirmed"]}
        finished_keys = confirmed_keys | set(self.eng.state["rejected"])
        pending = [candidate for candidate in self.eng.state["candidates"]
                   if self.eng._key(candidate) not in finished_keys]
        in_scope = []
        for candidate in pending:
            reason = self.scope.reject_reason(candidate["url"])
            if reason is not None:
                self.eng.log(f"validate: REFUSING {candidate['url']} "
                             f"({reason})")
                continue
            in_scope.append(candidate)
        self.eng.log(f"validate: {len(in_scope)} candidate(s) to "
                     f"adversarially verify, {self.parallel} at a time")
        for c, v in self._run_parallel(self._validate_agent, in_scope):
            if not v or v.get("errored"):
                self.eng.log(f"validate: inconclusive {c['vuln_class']} "
                             f"@ {c['url']} — "
                             f"{(v or {}).get('reason', 'worker error')}")
            elif v.get("real"):
                if v.get("severity"):
                    # Use the validator's CALIBRATED severity, not the hunt's.
                    c["severity"] = v["severity"]
                self.eng.confirm(c, v)
                if not self.mock:
                    try:
                        memory.record_finding(
                            self.scope.name, _host_of_url(c["url"]),
                            self._tech(), c, v, root=self.memory_root)
                    except Exception as exc:  # noqa: BLE001 - ledger is best-effort
                        self.eng.log(f"memory: capture skipped ({exc})")
                self.eng.log(f"validate: CONFIRMED {c['vuln_class']} "
                             f"@ {c['url']} ({v.get('severity', '')})"
                             + (f" scope-flagged: {', '.join(c['scope_warning'])}"
                                if c.get('scope_warning') else ''))
            else:
                self.eng.state["rejected"].append(self.eng._key(c))
                self.eng.save()
                self.eng.log(f"validate: rejected (false positive) "
                             f"{c['vuln_class']} @ {c['url']} — "
                             f"{(v or {}).get('reason', 'no verdict')}")

    def _validate_agent(self, c: dict[str, Any]) -> dict[str, Any]:
        if self.mock:
            return MOCK_VALIDATE.get((c["url"], c["vuln_class"]),
                                     {"real": False,
                                      "reason": "mock-default"})
        task = (self._roe() + "\n"
                f"Adversarially verify a claimed finding (be skeptical — default to false positive if unproven). "
                f"Claim: {c['vuln_class']} at {c['url']} (param `{c.get('param', '')}`). "
                f"Reported evidence: {c.get('evidence', '')[:400]}. "
                f"Independently re-test it WITHIN the rules of engagement above (in-scope + read-only only). If the only "
                f"proof would require an out-of-scope host or an intrusive action, treat the impact as UNPROVEN and lower "
                f"the severity. Is it a REAL, exploitable vulnerability or a false positive? "
                f"End with a fenced ```json``` object: "
                f'{{"real":true|false,"severity":"low|medium|high|critical","reason":"<why, incl. any scope/intrusive caveat>"}}.')
        r = _agent.run_agent(task, suggest=self.suggest)
        if r.get("error"):
            self.eng.log(f"validate agent error: {r['error']}")
            return {"real": False, "reason": f"verify error: {r['error']}",
                    "errored": True}
        verdict = _agent.extract_json(r["result"])
        if not isinstance(verdict, dict) \
                or not isinstance(verdict.get("real"), bool):
            return {"real": False, "reason": "no usable verdict",
                    "errored": True}
        return verdict

    # ── report ──
    def report(self) -> Path:
        self.eng.set_phase("report")
        c = self.eng.state["confirmed"]
        s = self.eng.summary()
        cov = closure.evaluate(self.eng.state)
        text = render_engagement_report(
            self.scope.name, self.scope.in_scope, s, c, cov,
            self.eng.state)
        path = self.eng.dir / "report.md"
        path.write_text(text, encoding="utf-8")
        self.eng.log(f"report: wrote {len(c)} confirmed finding(s) -> {path}")
        # Every confirmed finding goes through a hash-chained
        # evidence pack (secrets masked at rest), and the run is
        # journaled into the audit book when one is available.
        if c:
            try:
                from ..evidence import EvidenceChain, Finding
                chain = EvidenceChain(self.home, self.scope.name)
                for f in c:
                    chain.add(Finding(
                        module="bughunter",
                        target=str(f.get("url", "")),
                        verdict=str(f.get("severity", "")),
                        request={"text": str(f.get("request", ""))},
                        response={"body": str(f.get("evidence", ""))},
                        detail={"vuln_class": f.get("vuln_class", ""),
                                "param": f.get("param", ""),
                                "proof_note": f.get("proof_note", ""),
                                "scope_warning": f.get("scope_warning")},
                    ))
                self.eng.log(f"report: {len(c)} finding(s) into evidence "
                             f"chain {chain.case} (head {chain.head()[:12]})")
            except Exception as exc:  # noqa: BLE001 - evidence is best-effort
                self.eng.log(f"report: evidence chain skipped ({exc})")
        if self.book is not None:
            try:
                self.book.record({
                    "kind": "bughunter", "target": self.scope.name,
                    "own_asset": False, "requester": "bughunter",
                    "checks_run": ["bughunter-engagement"],
                    "notes": [cov["reason"]],
                    "findings": [{"source": "bughunter",
                                  "kind": f.get("vuln_class", ""),
                                  "title": f"{f.get('vuln_class', '')} at "
                                           f"{f.get('url', '')}",
                                  "location": f.get("url", ""),
                                  "detail": f.get("evidence", ""),
                                  "meta": {"severity": f.get("severity", ""),
                                           "param": f.get("param", "")}}
                                 for f in c],
                    "summary": {"confirmed": len(c),
                                "coverage_clean": cov["clean"],
                                "gaps": len(cov["gaps"])},
                    "report_path": str(path),
                    "draft_path": "",
                    "contacts": [],
                    "started_at": None, "finished_at": None,
                })
            except Exception as exc:  # noqa: BLE001 - journal is best-effort
                self.eng.log(f"report: audit-book journal skipped ({exc})")
        if not self.mock:
            try:
                memory.record_run(self.scope.name, self._tech(),
                                  self.eng.state["tested"],
                                  self.eng.state["confirmed"],
                                  root=self.memory_root)
            except Exception as exc:  # noqa: BLE001 - ledger is best-effort
                self.eng.log(f"memory: rollup skipped ({exc})")
        return path

    def run(self, phases: list[str]) -> dict[str, Any]:
        for ph in ALL_PHASES:
            if ph in phases:
                getattr(self, ph)()
        self.eng.set_phase("done")
        self.eng.log(f"engine done | {self.eng.summary()}")
        return self.eng.summary()


# ---- mock fixtures for --mock flow validation (no agents) ----
MOCK_RECON = {
    "http://localhost:3002": [
        {"url": "http://localhost:3002/api/account?id=5", "param": "id",
         "vuln_class": "idor", "note": "id param"},
        {"url": "http://localhost:3002/redirect?next=x", "param": "next",
         "vuln_class": "open-redirect", "note": "redirect"},
        {"url": "http://localhost:3002/search?q=x", "param": "q",
         "vuln_class": "xss", "note": "reflected"},
        {"url": "http://evil.example/x", "param": "",
         "vuln_class": "sqli",
         "note": "OUT OF SCOPE — should be dropped"},
    ]
}
MOCK_HUNT = {
    ("http://localhost:3002/api/account?id=5", "idor"): {
        "vulnerable": True, "severity": "high",
        "evidence": "id=N returns other users' PII",
        "request": "GET /api/account?id=6"},
    ("http://localhost:3002/redirect?next=x", "open-redirect"): {
        "vulnerable": True, "severity": "medium",
        "evidence": "302 to external",
        "request": "GET /redirect?next=//evil.com"},
    ("http://localhost:3002/search?q=x", "xss"): {"vulnerable": False},
}
MOCK_VALIDATE = {
    ("http://localhost:3002/api/account?id=5", "idor"): {
        "real": True, "severity": "high",
        "reason": "confirmed PII exposure for arbitrary id"},
    ("http://localhost:3002/redirect?next=x", "open-redirect"): {
        "real": False,
        "reason": "only redirects to relative paths on re-test "
                  "(false positive)"},
}


def run_engagement(scope: Scope | dict[str, Any], *,
                   phases: list[str] | None = None,
                   hunt: bool = False,
                   suggest: Callable[[str], str] | None = None,
                   max_hunts: int = 10,
                   allow_intrusive: bool = False,
                   mock: bool = False,
                   book: Any = None, home: str | Path | None = None,
                   base: str | Path | None = None,
                   audit_findings: list[dict[str, Any]] | None = None,
                   use_memory: bool = False) -> dict[str, Any]:
    """Run an engagement end to end; return its summary + paths.

    Default phases are recon -> rank -> map (the engine STOPS for
    the operator); ``hunt=True`` opts into the full
    recon,rank,map,hunt,validate,report loop. ``audit_findings``
    (Devon audit output) are converted into work units and merged
    into the surface before ranking.
    """
    if phases is not None:
        unknown = sorted(set(phases) - set(ALL_PHASES))
        if unknown or not phases:
            raise ValueError(
                f"unknown phase(s): {', '.join(unknown)}; choose from "
                f"{', '.join(ALL_PHASES)}")
    elif hunt:
        phases = ALL_PHASES
    else:
        phases = DEFAULT_PHASES
    eng = Engine(scope, base, suggest=suggest, max_hunts=max_hunts,
                 allow_intrusive=allow_intrusive, mock=mock, book=book,
                 home=home, use_memory=use_memory)
    if audit_findings:
        units = worklist_from_audit(audit_findings)
        added = eng.eng.add_surface(units)
        eng.eng.log(f"audit worklist: {added} unit(s) from "
                    f"{len(audit_findings)} Devon finding(s)")
    summary = eng.run(phases)
    return {"summary": summary, "dir": str(eng.eng.dir),
            "phases": phases,
            "coverage": closure.evaluate(eng.eng.state),
            "report_path": str(eng.eng.dir / "report.md")
                         if "report" in phases else ""}
