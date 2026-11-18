"""nm engage — run the engagement chain against one case.

Stages run in order and each writes its artifacts into the case dir:
seed -> recon -> subbrute -> surface probes -> evidence -> report.

Two rules hold the shape:

* a stage that fails is recorded and the run continues; a chain that stops
  at the first 429 is useless;
* every stage can be skipped (`--from`/`--until`) and every network stage
  has a `--dry-run` that plans without fetching.

The case dir is the unit of work. Resume means re-running with `--from`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ...security import disclose, evidence, recon, report, seed_sources, subbrute
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["_cmd_engage"]

STAGES = ("seed", "recon", "brute", "probe", "evidence", "report")


def _home_of(context: Any) -> str:
    # The CLI context is an AgentContext: it has .settings, never .home
    # — the old getattr(context, "home", ".") always fell back to ".",
    # so case dirs landed in the caller's working directory.
    settings = getattr(context, "settings", None) if context else None
    if settings is not None:
        home = (getattr(settings, "home_path", None)
                or getattr(settings, "home", None))
        if home:
            return str(home)
    return str(getattr(context, "home", None) or ".")


def _case_root(context: Any, case: str) -> Path:
    import re
    slug = re.sub(r"[^a-z0-9_\-]+", "-", case.strip().lower()).strip("-")
    if not slug:
        raise ValueError("engage needs a case name")
    return Path(_home_of(context)) / "data" / "evidence" / slug


def _write(case_dir: Path, name: str, data: dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / f"{name}.json").write_text(json.dumps(data, indent=1,
                                                      default=str),
                                           encoding="utf-8")


def _record(chain: evidence.EvidenceChain, module: str, target: str,
            verdict: str, *, request: dict, response: dict,
            note: str = "", severity: str = "") -> None:
    chain.add(evidence.Finding(
        module=module, target=target, verdict=verdict,
        request=request, response=response,
        detail={"note": note, **({"severity": severity} if severity else {})}))


def run_engage(context: Any, domain: str, *, case: str = "",
               until: str = "report", start_from: str = "seed",
               dry_run: bool = False, workers: int = 20,
               brute_rounds: int = 2, proxy_pool: Any = None,
               client: str = "", scope: str = "") -> dict[str, Any]:
    case = case or domain.replace(".", "-")
    case_dir = _case_root(context, case)
    # EvidenceChain(root, case) appends "evidence/<case>" to root — so
    # root is the DATA dir, not the case dir's parent: the old call
    # passed case_dir.parent (…/data/evidence) and the chain landed at
    # …/data/evidence/evidence/<case>, invisible to /chain verify and
    # to report.build(case_dir), which both read the canonical
    # …/data/evidence/<case>/chain.jsonl. The report stage therefore
    # errored on every single run.
    chain = evidence.EvidenceChain(case_dir.parent.parent, case_dir.name)
    order = STAGES[STAGES.index(start_from):STAGES.index(until) + 1]
    proxy = (proxy_pool.acquire if proxy_pool is not None else None)
    results: dict[str, Any] = {"case": case, "domain": domain,
                               "stages": {}, "started": time.time()}
    hosts: list[str] = []

    for stage in order:
        began = time.monotonic()
        try:
            if stage == "seed":
                if dry_run:
                    results["stages"][stage] = {"skipped": "dry run"}
                else:
                    data = seed_sources.harvest(domain, proxy_factory=proxy)
                    _write(case_dir, "seed", data)
                    hosts += [h["value"] for h in data.get("host", [])]
                    results["stages"][stage] = {"hosts": len(data.get("host", [])),
                                                "urls": len(data.get("url", []))}
                    for host in data.get("host", [])[:40]:
                        _record(chain, "seed", host["value"], "info",
                                request={"method": "GET", "url": host["value"]},
                                response={"status": 0}, note=host["source"])

            elif stage == "recon":
                if dry_run:
                    results["stages"][stage] = {"skipped": "dry run"}
                else:
                    data = recon.recon(domain)
                    _write(case_dir, "recon", data)
                    hosts += [a["host"] for a in data.get("assets", [])]
                    results["stages"][stage] = {"hosts": data["host_count"],
                                                "axfr": list(data["axfr"])}
                    for cand in data.get("takeover_candidates", []):
                        _record(chain, "recon", cand["host"], "takeover",
                                request={"method": "GET",
                                         "url": f"http://{cand['host']}/"},
                                response={"status": 0},
                                note=f"{cand['provider']} — {cand['state']}",
                                severity="high" if cand["state"] == "likely" else "medium")

            elif stage == "brute":
                if dry_run:
                    results["stages"][stage] = {"skipped": "dry run"}
                else:
                    words: list[str] = []
                    seed_file = case_dir / "seed.json"
                    if seed_file.is_file():
                        seed_data = json.loads(seed_file.read_text(encoding="utf-8"))
                        words = seed_sources.wordlist(seed_data.get("url", []))
                    data = subbrute.SubBrute(
                        domain, rounds=brute_rounds, workers=workers,
                        extra_words=words, progress=False).run()
                    _write(case_dir, "subbrute", data)
                    hosts += [h["host"] for h in data.get("hits", [])
                              if not h.get("wildcard")]
                    results["stages"][stage] = {"found": data["found"],
                                                "queries": data["queries"]}

            elif stage == "probe":
                if dry_run:
                    results["stages"][stage] = {"skipped": "dry run"}
                else:
                    from ...security.audit_web import ExposureScanner
                    # No argument: ExposureScanner takes a Fetcher, not a
                    # context — the old call passed the CLI context as the
                    # fetcher, so every scan() raised AttributeError and
                    # the probe stage silently probed zero hosts.
                    scanner = ExposureScanner()
                    probed = 0
                    # list() first: dict.fromkeys(hosts)[:60] raises
                    # (a dict is not subscriptable), so the old probe
                    # loop errored before scanning a single host.
                    for host in list(dict.fromkeys(hosts))[:60]:
                        try:
                            scan = scanner.scan(f"https://{host}")
                        except Exception as exc:  # noqa: BLE001
                            _log.info("probe %s failed: %s", host, exc)
                            continue
                        probed += 1
                        for finding in (scan.get("findings") or [])[:10]:
                            # evidence is None when a check carries no
                            # pack — .get() on None killed the whole host.
                            # Findings carry no severity key at all, so
                            # none is recorded (never invented).
                            pack = finding.get("evidence") or {}
                            _record(chain, "probe", host,
                                    str(finding.get("kind") or "info"),
                                    request=pack.get("request") or {},
                                    response=pack.get("response") or {},
                                    note=str(finding.get("detail") or "")[:200])
                    results["stages"][stage] = {"hosts_probed": probed}

            elif stage == "evidence":
                results["stages"][stage] = chain.verify()

            elif stage == "report":
                built = report.build(case_dir, domain=domain, client=client,
                                     scope_note=scope)
                results["stages"][stage] = built

        except Exception as exc:  # noqa: BLE001 - a stage failure is data
            results["stages"][stage] = {"error": f"{type(exc).__name__}: {exc}"}
            _log.info("engage stage %s failed: %s", stage, exc)
        results["stages"][stage]["seconds"] = round(time.monotonic() - began, 1)

    results["finished"] = time.time()
    results["integrity"] = chain.verify()
    results["contacts"] = (disclose.find_contacts(domain).get("contacts", [])
                           if not dry_run else [])
    _write(case_dir, "engage", results)
    return results


def _cmd_engage(args: Any, context: Any) -> int:
    pool = None
    try:
        # proxy_rotation, not proxy_rotate: that module does not exist,
        # so the old import always failed and the pool was always None.
        from ...tools.proxy_rotation import pool_from_context
        pool = pool_from_context(context)
        pool.probe_all()
    except Exception:  # noqa: BLE001 - no proxies is a valid state
        pool = None
    result = run_engage(context, args.domain, case=args.case,
                        until=args.until, start_from=args.start_from,
                        dry_run=args.dry_run, workers=args.workers,
                        brute_rounds=args.rounds, proxy_pool=pool,
                        client=args.client, scope=args.scope)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        for stage, data in result["stages"].items():
            state = "error" if "error" in data else \
                "skip" if "skipped" in data else "ok"
            print(f"  {state:<5} {stage:<9} {data.get('seconds', 0):>5}s  "
                  f"{ {k: v for k, v in data.items() if k not in ('seconds',)} }")
        print(f"\n  integrity: {result['integrity']['state']}  "
              f"records: {result['integrity']['records']}")
        if result["contacts"]:
            print(f"  contact: {result['contacts'][0]['email']}")
    return 0 if result["integrity"]["state"] == "intact" else 1
