"""CI thin-caller contract checker (census §3B.4, craft-common-actions).

The pattern, mined from storytold/craft-common-actions (no licence —
pattern only, no code copied): an app repository's ``.github/workflows/``
holds ONLY thin callers — a workflow file contains ``name``, ``on``,
``permissions``, ``concurrency`` and jobs that are ``uses:`` calls into
shared reusable workflows.  No caller job carries its own ``runs-on`` /
``steps``; the job bodies live in the shared workflows, the step bodies
in composite actions.  A caller that embeds a job body is refused, not
tolerated.

The contract this module enforces, with the evidence for each rule in
the upstream README:

* Thin caller — every job in a caller workflow is a ``uses:`` call.
  (README: "An app's ``.github/workflows/`` only calls this
  repository's workflows ..., with inputs and secrets: no caller jobs
  with their own ``runs-on``/``steps``.")
* Permissions belong to the caller — a caller declares ``permissions``
  (top level or per job); a called workflow cannot raise them.
  (README: "A called workflow cannot increase the token permissions
  granted by its caller.")
* Concurrency belongs to the caller — a reusable workflow cannot set
  it for the run, so the caller must. (README, gate section.)
* Every ``uses:`` names a ref — a bare ``owner/repo/path`` with no
  ``@ref`` is refused; a call that floats is not a pin.
* Third-party *actions* are SHA-pinned — a full 40-hex commit SHA,
  the upstream's own practice for ``actions/checkout`` etc.
  (``actions/checkout@3d3c42e5... # v7.0.1`` in gate.yml.)  Shared
  *workflows* are called at a named ref (``@main`` upstream, a local
  path here) — tracking by design, so suite-wide fixes reach every
  caller without a PR in each repository.
* Secrets are listed by name — ``secrets: inherit`` is refused, and
  a secret may never appear inside a ``run:`` step (it would be
  expanded into the log stream); secrets flow only through
  ``secrets:`` / ``env:``.  (README: "List every signing secret
  explicitly ... an unlisted name is empty.")
* Inputs flow through — a caller's ``with:`` keys must be declared by
  the called workflow's ``workflow_call.inputs`` (checked for local
  reusable workflows, where the declaration is on disk), and required
  inputs must be supplied.

PyYAML is an optional dependency, imported locally exactly like the
other tools modules do it: when it is absent the checker reports
``yaml_unavailable`` and refuses to pass — an explicit "cannot
measure", never a silent pass.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "Violation",
    "check_workflow",
    "check_directory",
    "resolve_pin",
    "load_workflow",
]

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SECRET_REF_RE = re.compile(r"\$\{\{\s*secrets\.[^}]+\}\}")

# Codes the checker can emit.
CODES = (
    "yaml_unavailable",
    "unparsable",
    "caller_has_steps",
    "caller_missing_permissions",
    "caller_missing_concurrency",
    "uses_missing_ref",
    "action_not_sha_pinned",
    "secrets_inherit",
    "secret_in_run",
    "unknown_input",
    "missing_required_input",
    "missing_called_workflow",
)


class Violation:
    """One contract breach, with the file, the job/step and the rule."""

    def __init__(self, code: str, path: str, where: str, detail: str) -> None:
        if code not in CODES:
            raise ValueError(f"unknown violation code: {code}")
        self.code = code
        self.path = path
        self.where = where
        self.detail = detail

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path,
                "where": self.where, "detail": self.detail}

    def __str__(self) -> str:
        return f"{self.path}:{self.where}: {self.code}: {self.detail}"


def load_workflow(path: str | Path) -> dict[str, Any]:
    """Parse one workflow file. Raises on missing PyYAML / bad YAML."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyyaml is not installed") from exc
    text = Path(path).read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: workflow is not a YAML mapping")
    return doc


def _on_block(doc: dict[str, Any]) -> dict[str, Any]:
    """The ``on:`` block of a workflow.

    PyYAML parses the YAML 1.1 key ``on`` as boolean ``True``, so the
    block lives under ``True`` in the parsed document, not ``"on"``.
    """
    on = doc.get("on")
    if on is None:
        on = doc.get(True)
    return on if isinstance(on, dict) else {}


def _is_reusable_call(uses: str) -> bool:
    """True when a ``uses:`` string calls a reusable workflow file."""
    target = uses.split("@", 1)[0]
    return target.endswith((".yml", ".yaml"))


def _local_target(uses: str, repo_root: Path) -> Path | None:
    """Resolve a local-path ``uses:`` (``./...``) to an on-disk path."""
    target = uses.split("@", 1)[0]
    if not target.startswith("./"):
        return None
    return (repo_root / target[2:]).resolve()


def resolve_pin(uses: str, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Resolve what a ``uses:`` string pins to.

    Returns ``kind`` (``local`` / ``action`` / ``reusable``), the
    ``ref`` as written (``""`` when absent), whether it counts as
    pinned, and ``resolved`` — the commit SHA the ref resolves to
    when that is knowable on disk, else ``None`` (a tracking ref
    resolves only at call time, on the runner).
    """
    root = Path(repo_root) if repo_root else None
    if "@" in uses:
        target, ref = uses.split("@", 1)
    else:
        target, ref = uses, ""
    if target.startswith("./"):
        resolved = None
        if root is not None:
            try:
                resolved = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root,
                    text=True, timeout=30).strip()
            except (subprocess.SubprocessError, OSError):
                resolved = None
        return {"kind": "local", "ref": ref or "HEAD",
                "pinned": True, "resolved": resolved}
    if _is_reusable_call(uses):
        return {"kind": "reusable", "ref": ref, "pinned": bool(ref),
                "resolved": ref if _SHA_RE.match(ref) else None}
    return {"kind": "action", "ref": ref,
            "pinned": bool(_SHA_RE.match(ref)),
            "resolved": ref if _SHA_RE.match(ref) else None}


def _check_job(job_name: str, job: Any, doc: dict[str, Any],
               path: str, repo_root: Path,
               out: list[Violation]) -> None:
    where = f"jobs.{job_name}"
    if not isinstance(job, dict):
        out.append(Violation("caller_has_steps", path, where,
                             "job is not a mapping"))
        return
    if "uses" not in job:
        out.append(Violation(
            "caller_has_steps", path, where,
            "job has no 'uses:' — a caller job must call a shared "
            "workflow, never carry its own runs-on/steps"))
        return
    if "runs-on" in job or "steps" in job:
        out.append(Violation(
            "caller_has_steps", path, where,
            "job both calls a shared workflow and carries its own "
            "runs-on/steps — the body belongs in the shared workflow"))
    uses = str(job["uses"])
    pin = resolve_pin(uses, repo_root)
    if not pin["ref"]:
        out.append(Violation("uses_missing_ref", path, where,
                             f"'{uses}' names no ref — pin the call"))
    elif pin["kind"] == "action" and not pin["pinned"]:
        out.append(Violation(
            "action_not_sha_pinned", path, where,
            f"'{uses}' is not a full 40-hex SHA pin — third-party "
            "actions are pinned by commit, never by tag"))
    if job.get("secrets") == "inherit":
        out.append(Violation(
            "secrets_inherit", path, where,
            "'secrets: inherit' passes every secret — list each "
            "secret by name instead"))
    for step in job.get("steps") or []:
        if isinstance(step, dict) and isinstance(step.get("run"), str) \
                and _SECRET_REF_RE.search(step["run"]):
            out.append(Violation(
                "secret_in_run", path, f"{where}.steps",
                "a secret is referenced inside a run: step — "
                "secrets flow only through secrets:/env:, never "
                "through run text"))
    # Inputs flow through: check `with:` against the called workflow's
    # declared inputs when that declaration is on disk (local calls).
    target = _local_target(uses, repo_root)
    if target is None:
        return
    if not target.exists():
        out.append(Violation(
            "missing_called_workflow", path, where,
            f"called workflow '{uses}' does not exist on disk"))
        return
    try:
        called = load_workflow(target)
    except (RuntimeError, ValueError, OSError) as exc:
        out.append(Violation("unparsable", str(target), "top",
                             str(exc)))
        return
    declared = (_on_block(called).get("workflow_call") or {}
                ).get("inputs") or {}
    given = job.get("with") or {}
    for key in given:
        if key not in declared:
            out.append(Violation(
                "unknown_input", path, where,
                f"input '{key}' is not declared by '{uses}'"))
    for key, spec in declared.items():
        if isinstance(spec, dict) and spec.get("required") \
                and key not in given:
            out.append(Violation(
                "missing_required_input", path, where,
                f"required input '{key}' of '{uses}' is not passed"))


def check_workflow(doc: dict[str, Any], path: str,
                   repo_root: str | Path | None = None) -> list[Violation]:
    """Check one parsed workflow document against the contract.

    A workflow that declares ``on.workflow_call`` IS a shared workflow
    (its jobs carry the bodies) and is exempt from the thin-caller
    rules, but its own ``uses:`` lines are still pin-checked.
    """
    root = Path(repo_root) if repo_root else Path.cwd()
    out: list[Violation] = []
    on = _on_block(doc)
    if isinstance(on, dict) and "workflow_call" in on:
        # Shared workflow: bodies live here.  Pin-check its steps, and
        # refuse secrets inside any run: step.
        for job_name, job in (doc.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            for step in job.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                where = f"jobs.{job_name}.steps"
                if isinstance(step.get("run"), str) \
                        and _SECRET_REF_RE.search(step["run"]):
                    out.append(Violation(
                        "secret_in_run", path, where,
                        "a secret is referenced inside a run: step — "
                        "secrets flow only through secrets:/env:, "
                        "never through run text"))
                if "uses" not in step:
                    continue
                uses = str(step["uses"])
                pin = resolve_pin(uses, root)
                if not pin["ref"]:
                    out.append(Violation(
                        "uses_missing_ref", path, where,
                        f"'{uses}' names no ref — pin the call"))
                elif pin["kind"] == "action" and not pin["pinned"]:
                    out.append(Violation(
                        "action_not_sha_pinned", path, where,
                        f"'{uses}' is not a full 40-hex SHA pin — "
                        "third-party actions are pinned by commit, "
                        "never by tag"))
        return out
    # Caller workflow: thin-caller rules.
    if "permissions" not in doc:
        out.append(Violation(
            "caller_missing_permissions", path, "top",
            "no permissions: declared — a called workflow cannot "
            "raise the token permissions the caller grants"))
    if "concurrency" not in doc:
        out.append(Violation(
            "caller_missing_concurrency", path, "top",
            "no concurrency: declared — concurrency belongs to the "
            "caller; a reusable workflow cannot set it"))
    for job_name, job in (doc.get("jobs") or {}).items():
        _check_job(job_name, job, doc, path, root, out)
    return out


def check_directory(workflows_dir: str | Path,
                    repo_root: str | Path | None = None) -> dict[str, list[Violation]]:
    """Check every workflow file in a directory. Returns path → violations."""
    directory = Path(workflows_dir)
    root = Path(repo_root) if repo_root else directory.parent.parent
    results: dict[str, list[Violation]] = {}
    for path in sorted(directory.glob("*.y*ml")):
        try:
            doc = load_workflow(path)
        except RuntimeError:
            results[str(path)] = [Violation(
                "yaml_unavailable", str(path), "top",
                "pyyaml is not installed — cannot parse workflows")]
            continue
        except (ValueError, OSError) as exc:
            results[str(path)] = [Violation(
                "unparsable", str(path), "top", str(exc))]
            continue
        results[str(path)] = check_workflow(doc, str(path), root)
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python3 -m nomorals.tools.ci_contract check [paths...]``."""
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    if not args or args[0] != "check":
        print("usage: python3 -m nomorals.tools.ci_contract check "
              "[--json] [.github/workflows [workflow.yml ...]]",
              file=sys.stderr)
        return 2
    targets = args[1:]
    if not targets:
        targets = [".github/workflows"]
    repo_root = Path.cwd()
    results: dict[str, list[Violation]] = {}
    for target in targets:
        target_path = Path(target)
        if target_path.is_dir():
            results.update(check_directory(target_path, repo_root))
        elif target_path.exists():
            try:
                doc = load_workflow(target_path)
            except RuntimeError:
                results[str(target_path)] = [Violation(
                    "yaml_unavailable", str(target_path), "top",
                    "pyyaml is not installed — cannot parse workflows")]
                continue
            except (ValueError, OSError) as exc:
                results[str(target_path)] = [Violation(
                    "unparsable", str(target_path), "top", str(exc))]
                continue
            results[str(target_path)] = check_workflow(
                doc, str(target_path), repo_root)
        else:
            print(f"no such file or directory: {target}", file=sys.stderr)
            return 2
    total = sum(len(v) for v in results.values())
    if as_json:
        import json
        print(json.dumps(
            {p: [v.as_dict() for v in vs] for p, vs in results.items()},
            indent=2))
    else:
        for path, violations in results.items():
            if violations:
                for violation in violations:
                    print(violation)
            else:
                print(f"{path}: OK")
        print(f"{total} violation(s) in {len(results)} workflow(s)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
