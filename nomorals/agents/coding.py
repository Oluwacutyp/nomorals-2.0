"""The coding agent: draft -> run in the sandbox -> read the error -> fix.

This is the "not just a class" version: every step uses the real system.

* **Drafting** goes through the router — whatever model is actually active
  (groq / hf / local / mock), with the previous attempt's code and its exact
  stderr fed back in, so the model fixes the real failure, not a guess.
* **Writing** goes through the filesystem tool's ``safe_path`` — files land
  inside the workspace, traversal is rejected.
* **Running** goes through the sandbox (``run_sandboxed``) — bwrap/unshare/
  rlimit isolation, network off by default, process-group kill on timeout.
* **Every iteration** is journaled to ``coding_log`` so a whole session can
  be replayed or audited later.

The loop stops on a green acceptance run, on a clean model refusal, or when
``max_iterations`` is spent.  With the mock provider the loop fails fast and
loud ("model returned no code") rather than pretending to work.
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from ..llm.base import LLMResponse, Message, SamplingParams
from ..tools.filesystem import safe_path
from .patch import apply_unified_diff, parse_unified_diff
from .repo_map import build_repo_map

_log = get_logger(__name__)

#: Markers of context that the project layer INJECTS into a build task
#: before the coding agent sees it.  The owner's actual request is what
#: remains after the last injected block — reasoning gates (briefing,
#: draft review) must judge THAT, not the inflated execution prompt.
_SKILL_BLOCK_MARKER = "Relevant proven skills (apply their lessons):"


def core_request(task: str) -> str:
    """The owner's actual request with injected prior-art stripped.

    Projects prepend recalled-skill blocks (and goal knowledge) to the
    build task.  Those make the string long, but they are not new
    complexity: structuring or reviewing them would just spend model
    calls re-wrapping what was already injected.
    """
    t = (task or "").strip()
    idx = t.find(_SKILL_BLOCK_MARKER)
    if idx != -1:
        lines = t[idx:].splitlines()
        end = 1  # skip the marker line itself
        while end < len(lines) and (
                lines[end].startswith("- ") or not lines[end].strip()):
            end += 1
        t = "\n".join(lines[end:]).strip()
    return t

__all__ = ["CodingAgent", "CodingResult", "extract_code_block"]

_CODE_BLOCK_NL = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_CODE_BLOCK_INLINE = re.compile(r"```(?:python|py)?[ \t]+(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str:
    """Pull the first fenced code block out of a model reply.

    Handles both the normal shape (code on the line after the fence) and the
    lazy one (code starting on the fence line).  Returns "" when there is no
    block — the caller treats that as a refusal, not as an empty file.
    """
    if not text:
        return ""
    for pattern in (_CODE_BLOCK_NL, _CODE_BLOCK_INLINE):
        match = pattern.search(text)
        if match:
            return match.group(1).strip() + "\n"
    return ""


_JSON_BLOCK = re.compile(r"```(?:json)?[ \t]*\n(.*?)```", re.DOTALL)


def _parse_json_block(text: str) -> Any:
    """Pull the first fenced (```json) block out of a model reply and parse
    it. Returns None when there is no block or it is not valid JSON."""
    if not text:
        return None
    match = _JSON_BLOCK.search(text)
    raw = match.group(1) if match else text
    try:
        return json.loads(raw.strip())
    except (ValueError, TypeError):
        return None


_PATCH_BLOCK = re.compile(r"```(?:diff|patch)[ \t]*\n(.*?)```", re.DOTALL)


def _parse_patch_block(text: str) -> str | None:
    """Pull the first fenced ```diff (or ```patch) block out of a model
    reply.  Returns the raw unified-diff text, or None when there is no
    usable diff block."""
    if not text:
        return None
    match = _PATCH_BLOCK.search(text)
    if not match:
        return None
    patch_text = match.group(1).strip()
    if not patch_text or "--- " not in patch_text:
        return None
    return patch_text


def _parse_edits_block(text: str) -> list[dict[str, str]] | None:
    """Parse the surgical-edit protocol: ``{"edits": [{old_text, new_text}]}``.

    Returns the edit list (possibly empty), or None when the model gave
    nothing usable.
    """
    data = _parse_json_block(text)
    if not isinstance(data, dict):
        return None
    edits = data.get("edits")
    if not isinstance(edits, list):
        return None
    out: list[dict[str, str]] = []
    for edit in edits:
        if (isinstance(edit, dict)
                and isinstance(edit.get("old_text"), str)
                and isinstance(edit.get("new_text"), str)):
            out.append({"old_text": edit["old_text"],
                        "new_text": edit["new_text"]})
        else:
            return None
    return out


def _bg_progress(status: dict[str, Any]) -> None:
    """Live progress from a background test run (Phase D)."""
    _log.info("background tests: %d passed, %d failed, %d errors "
              "(%.1fs elapsed)", status["passed"], status["failed"],
              status["errors"], status["seconds"])


def _tests_nothing_ran(tres: dict[str, Any]) -> bool:
    """Did this runner result execute ZERO tests? (pass #58)

    Prefers the runner's explicit ``nothing_ran``/``ran`` signal;
    derives the count from passed/failed/errors for result dicts that
    predate the signal (e.g. monkeypatched runners in older tests).
    A zero-ran result is an honest skip at the runner level — but it
    must never certify a coding run as verified.
    """
    if tres.get("nothing_ran") is not None:
        return bool(tres["nothing_ran"])
    ran = tres.get("ran")
    if ran is None:
        ran = (int(tres.get("passed") or 0)
               + len(tres.get("failed") or [])
               + int(tres.get("errors") or 0))
    return int(ran) == 0


def _workdir_has_tests(workdir: Any) -> bool:
    """Does this workdir actually contain tests?  A simple single-file
    script (no tests/ dir, no test_*.py) should just run — not go through
    the pytest runner and print "nothing ran"."""
    from pathlib import Path
    root = Path(str(workdir))
    if (root / "tests").is_dir() or (root / "test").is_dir():
        return True
    # test files anywhere in the tree (bounded depth)
    try:
        for pat in ("test_*.py", "*_test.py"):
            for _ in root.glob(pat):
                return True
            for _ in root.glob(f"*/{pat}"):
                return True
    except Exception:  # noqa: BLE001 — unreadable dir, treat as no tests
        pass
    return False


def _substantive_lines(path: Any) -> int:
    """Non-blank, non-comment, non-docstring lines in a file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return 0
    count = 0
    in_docstring = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith('"""') or s.startswith("'''"):
            # single-line docstring
            if len(s) > 6 and s.endswith(('"""', "'''")):
                continue
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if s.startswith("#"):
            continue
        count += 1
    return count


#: minimum substantive lines across changed files to count as an artifact
_MIN_ARTIFACT_LINES = 5


def _unified_diff(before: str, after: str, rel: str) -> str:
    """Unified diff of two file texts ("" when identical)."""
    if before == after:
        return ""
    return "\n".join(difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm=""))


# Phase C: focus line for the diff-review gate — the critic reads the
# full multi-file unified diff, not a single draft.
_DIFF_REVIEW_FOCUS = (
    "a unified diff of the complete multi-file change: does the diff "
    "implement the task, are there wrong operators, indices, or names, "
    "does any hunk break something the diff touches or contradict the "
    "task's acceptance criteria")


# ── Pass #63 (W5): diff review always on, depth scaled by the DIFF ──────
#
# Before this pass, _review_flaws returned [] unless reasoning mode was
# on AND the task text "looked complex" (core >= 80 chars or analytical
# markers): a short task — "fix the parser bug" — silently disabled the
# per-diff review AND the Phase C gate no matter how risky the applied
# diff was, and the result still claimed review passed=True.  Now every
# applied diff is reviewed; WHAT is reviewed decides HOW DEEPLY:
#
#   tier      rule (first match wins)                        review calls
#   --------  ---------------------------------------------  ----------------
#   deep      a hard risk signal (sensitive path, public-    1 per file
#             API change, whole-file deletion, heavy         (cap 6) + 1
#             deletion) OR >= 5 files OR >= 400 changed      combined
#             lines
#   standard  >= 2 files OR >= 80 changed lines OR            1 combined
#             >= 30 deleted lines
#   focused   anything smaller                               1, max_calls=1,
#                                                            text cap 4000
#
# Reasoning mode may only RAISE the floor: mode "always" or an active
# power mode floors the tier at standard.  It never lowers a tier and
# never switches review off — "off"/"auto" simply contribute no floor.
# Risk signals are diff-structure facts (paths in the diff headers,
# +/- line shapes), never prompt keywords.

#: top-level def/class lines added or removed by a diff = public surface
_API_DEF_RE = re.compile(r"^(?:async\s+def|def|class)\s+[A-Za-z_]\w*")

#: path segments / file stems that make a touched file credential- or
#: config-adjacent (a diff there deserves the deep read)
_SENSITIVE_SEGMENTS = {"auth", "credentials", "secrets", "vault", ".ssh"}
_SENSITIVE_STEMS = {"config", "settings", "secrets", "credentials",
                    "auth", "vault", "password", "passwords", "token",
                    "tokens", "api_key", "apikey"}
_SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".env")

#: extra reviewer checks appended to the focus when a risk signal fires
_RISK_CHECKS = {
    "sensitive-path": (
        "this diff touches credential/config-adjacent paths — check "
        "that no secret values are introduced, logged, written to "
        "insecure defaults, or exposed to callers that lacked them"),
    "public-api-change": (
        "the public API surface changes (top-level def/class added or "
        "removed, __all__ edited) — check existing callers still work "
        "or are updated inside this same diff"),
    "file-deletion": (
        "a whole file is deleted — check nothing still imports, "
        "references, or depends on it"),
    "heavy-deletion": (
        "deletions dominate this diff — check the removed behavior is "
        "not still referenced or required elsewhere"),
    "wide-change": (
        "this is a wide multi-file change — check cross-file "
        "consistency: names, signatures, and imports agree across "
        "every file the diff touches"),
}

#: per-file review-call cap at deep tier (plus one combined call)
_DEEP_FILE_CAP = 6


def _sensitive_path(path: str) -> bool:
    """True when a diff-touched path is credential/config-adjacent."""
    parts = [p for p in re.split(r"[/\\]+", path.lower()) if p]
    if not parts:
        return False
    name = parts[-1]
    if name == ".env" or name.startswith(".env."):
        return True
    if name.endswith(_SENSITIVE_SUFFIXES):
        return True
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return (stem in _SENSITIVE_STEMS
            or any(p in _SENSITIVE_SEGMENTS for p in parts))


def _diff_metrics(text: str) -> dict[str, Any]:
    """Structural facts about a diff (or a raw draft, treated as one
    new file): touched files, added/deleted line counts, risk signals.
    Pure function of the text — the prompt is never consulted."""
    files: list[str] = []
    added = deleted = 0
    api_change = file_deleted = False
    saw_structure = False
    pending_src = ""
    for line in (text or "").splitlines():
        if line.startswith("--- "):
            saw_structure = True
            src = line[4:].strip()
            pending_src = src[2:] if src.startswith("a/") else src
        elif line.startswith("+++ "):
            saw_structure = True
            target = line[4:].strip()
            if target == "/dev/null":
                file_deleted = True
                if pending_src and pending_src not in files:
                    files.append(pending_src)
            else:
                path = (target[2:] if target.startswith(("a/", "b/"))
                        else target)
                if path and path not in files:
                    files.append(path)
            pending_src = ""
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
            body = line[1:]
            if _API_DEF_RE.match(body) or "__all__" in body:
                api_change = True
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1
            body = line[1:]
            if _API_DEF_RE.match(body) or "__all__" in body:
                api_change = True
    risks: list[str] = []
    if any(_sensitive_path(p) for p in files):
        risks.append("sensitive-path")
    if api_change:
        risks.append("public-api-change")
    if file_deleted:
        risks.append("file-deletion")
    if deleted >= 50 and deleted >= added:
        risks.append("heavy-deletion")
    file_count = len(files)
    if not saw_structure and not files and (text or "").strip():
        # Not a diff at all (a raw code draft): one implicit new file,
        # sized by its line count; no +/- structure, so no risk signals.
        file_count = 1
        added = len((text or "").splitlines())
    if file_count >= 5 or (added + deleted) >= 400:
        risks.append("wide-change")
    return {"files": files, "file_count": file_count, "added": added,
            "deleted": deleted, "risks": risks}


def _review_tier(metrics: dict[str, Any]) -> str:
    """The scaling table from the pass-#63 block comment, as code."""
    hard = {"sensitive-path", "public-api-change", "file-deletion",
            "heavy-deletion"}
    if (hard & set(metrics["risks"]) or metrics["file_count"] >= 5
            or (metrics["added"] + metrics["deleted"]) >= 400):
        return "deep"
    if (metrics["file_count"] >= 2 or metrics["deleted"] >= 30
            or (metrics["added"] + metrics["deleted"]) >= 80):
        return "standard"
    return "focused"


def _split_unified_diff(text: str) -> list[tuple[str, str]]:
    """Split a multi-file unified diff into (path, chunk) per file.

    Chunks start at "diff --git" or "--- a/…" headers — the shapes
    _unified_diff and git both emit.  Returns [] when the text carries
    no per-file structure (single-file or non-diff text)."""
    chunks: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    path = ""
    for line in (text or "").splitlines():
        if line.startswith("diff --git "):
            if current is not None:
                chunks.append((path, current))
            current, path = [line], line.split(" b/", 1)[-1].strip()
        elif line.startswith("--- a/") or line.startswith("--- /dev"):
            if current is not None and any(
                    ln.startswith("+++ ") for ln in current):
                chunks.append((path, current))
                current = None
            if current is None:
                current, path = [line], line[6:].strip()
            else:
                current.append(line)
        elif line.startswith("+++ "):
            target = line[4:].strip()
            if target.startswith("b/"):
                path = target[2:]
            if current is not None:
                current.append(line)
        elif current is not None:
            current.append(line)
    if current is not None:
        chunks.append((path, current))
    return [(p, "\n".join(lines)) for p, lines in chunks if lines]


@dataclass
class DiffReview:
    """One diff review's full outcome (pass #63).

    ``available`` False means the review could NOT run (model error,
    unparseable verdict): ``flaws`` is then empty because there is NO
    verdict — never read it as a clean pass.  ``depth`` is the tier the
    diff's own size/risk selected; ``calls`` counts reviewer model
    calls actually spent."""

    flaws: list[str] = field(default_factory=list)
    available: bool = True
    error: str = ""
    depth: str = "focused"
    files: int = 0
    added: int = 0
    deleted: int = 0
    risks: list[str] = field(default_factory=list)
    calls: int = 0


def _format_lint_result(lint_res: dict[str, Any]) -> str:
    """Lint violations as fix-loop feedback."""
    lines = ["ruff lint failed:"]
    for viol in lint_res.get("violations", [])[:10]:
        lines.append(f"{viol.get('file')}:{viol.get('line')}:{viol.get('col')}: "
                     f"{viol.get('code')} {viol.get('message')}")
    return "\n".join(lines)


# ── pass #65 (W7): token metering + the optional spend cap ──────────
#
# Every model call a run makes funnels through CodingAgent.chat /
# _phase_chat, and its diff reviews through _call_reviewer — the three
# points where usage is recorded.  Metering is ALWAYS ON (a run with
# no cap is metered exactly like a capped one); the cap is the only
# part that can stop a run, and it stops BEFORE the next model call
# is issued, never mid-call and never by rewriting a verdict.


class _SpendCapExceeded(Exception):
    """The run's metered token total crossed its spend cap.  Raised
    by the chat funnel / reviewer seam BEFORE the next model call is
    issued; run() settles it (green stands when one was earned and no
    edit has landed since; otherwise an honest cap failure)."""


def _blank_usage() -> dict[str, int]:
    """The usage counters shape: tokens as the provider reported
    them, ``calls`` = model calls made, ``usage_unreported`` = calls
    whose response carried no usable usage (counted, never
    estimated)."""
    return {"prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "calls": 0, "usage_unreported": 0}


def _response_usage(response: Any) -> tuple[int, int, int] | None:
    """(prompt, completion, total) from a model response, or None
    when the response carries no usage at all.  Never raises."""
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens") or 0)
            completion = int(usage.get("completion_tokens") or 0)
            total = int(usage.get("total_tokens") or 0)
        else:
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
            total = int(getattr(usage, "total_tokens", 0) or 0)
        if total <= 0 and (prompt or completion):
            total = prompt + completion
        return prompt, completion, total
    except Exception:  # noqa: BLE001 — metering must never break a call
        return None


@dataclass
class CodingResult:
    ok: bool
    iterations: int
    files: list[str] = field(default_factory=list)
    output: str = ""
    error: str = ""
    seconds: float = 0.0
    # Phase C: diff-review gate verdict — {"passed": bool, "rounds": int,
    # "objections": [str]}.  Empty when the gate never ran.
    # Pass #63 (W5): plus "available" (bool — False means the gate review
    # could NOT run; "passed" is then False too, never a silent clean
    # pass), "depth" (the tier the diff selected: focused|standard|deep)
    # and "notes" (review-unavailability notes from the whole run).
    review: dict[str, Any] = field(default_factory=dict)
    # Plan mode (item #4): when True the task paused BEFORE any file was
    # touched — the owner must approve plan_text (plan_id) before
    # execute_plan() runs it.
    needs_approval: bool = False
    plan_id: str = ""
    plan_text: str = ""
    # Pass #58: how the run was actually verified — "tests" (the runner
    # executed ≥1 test), "accept" (an explicit accept command decided),
    # "smoke" (no tests exist in the workdir; the default accept merely
    # executed the file), "unverified" (nothing could verify the change
    # and the run failed honestly), "" (no verify round ran — plan
    # pause, no-code draft).  Never let a smoke/unverified run dress
    # itself up as tested.
    verification: str = ""
    # Pass #60 (W4): per-step outcomes — one dict per step EVER in the
    # run (steps superseded by a re-plan stay in the list, marked
    # "skipped", so the weak step is named in the result instead of
    # only post-mortem): {id, path, files, status, reason, attempts,
    # instructions, acceptance, depends_on} with status one of
    # "planned" / "done" / "failed" / "skipped".  Empty when no plan
    # ran (defensive default).
    steps: list[dict[str, Any]] = field(default_factory=list)
    # Pass #60 (W4): how many times the run re-planned its remaining
    # steps after a step failed repeatedly (0 or 1 — never more).
    replans: int = 0
    # Pass #62: the runner's full-suite final gate verdict, when the
    # caller opted in (``final_gate`` on run()) and a verify round went
    # green — {"ran", "verdict", "failed", ...}.  A DISTINCT verdict:
    # a gate failure (e.g. pre-existing breakage elsewhere in the
    # suite) is reported here and never rewrites ``ok``/``verification``,
    # which belong to the change's own selection run.  Empty when the
    # gate never ran.
    final_gate: dict[str, Any] = field(default_factory=dict)
    # Pass #64 (W6): one metadata dict per edit call — the context
    # pack that fed it (tools/context_pack.ContextPack.metadata):
    # mode ("pack" / "whole-file"), budget vs used chars, sections,
    # everything the budget trimmed, and fallback notes.  Trimming and
    # fallbacks are visible here, never silent.  Empty when no edit
    # call ran (plan pause, new-file-only runs).
    context_packs: list[dict[str, Any]] = field(default_factory=list)
    # Pass #65 (W7): the run's metered token usage — ALWAYS recorded
    # (metering is not the cap): {prompt_tokens, completion_tokens,
    # total_tokens, calls, usage_unreported, by_phase: {phase: same
    # counters}} with phases plan/edit/chat/review/revise.  Plus
    # "cap" (the max_tokens the caller set) and "capped": True when
    # the cap stopped the run.  usage_unreported counts calls whose
    # response carried no usable usage — their tokens are honestly
    # unknown (0 in the totals), never estimated.  Empty when the
    # result did not come from a run (defensive default).
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "iterations": self.iterations,
            "files": self.files,
            "output": self.output[-2000:],
            "error": self.error[-2000:],
            "seconds": round(self.seconds, 2),
            "review": self.review,
            "needs_approval": self.needs_approval,
            "plan_id": self.plan_id,
            "plan_text": self.plan_text,
            "verification": self.verification,
            "steps": self.steps,
            "replans": self.replans,
            "final_gate": self.final_gate,
            "context_packs": self.context_packs,
            "usage": self.usage,
        }


# Pass #60 (W4): a plan is a list of STEPS, not a bare file list.  A step
# spec extends the historical {path, why, new_file} dict additively:
#   id           stable step id ("s1", "s2", ...) — dependencies and
#                per-step status refer to steps by id
#   instructions concrete per-step direction: what to change and how
#   depends_on   ids of steps that must complete before this one
#   acceptance   the check this step must satisfy — it rides every draft
#                call for the step, so the model works to a criterion
#   files        every file the step claims; "path" is the primary
#                target the edit loop drafts (the loop's unit of work
#                has always been one primary file per plan entry)
def _normalize_steps(specs: list[dict[str, Any]], *,
                     used_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Assign ids/defaults to raw plan specs; order dependencies first.

    Ids the planner emitted are positional (``s1``..``sn`` in array
    order).  When ``used_ids`` is given (a re-plan joining a run that
    already has step history), colliding ids are renumbered to the next
    free ``sK`` and ``depends_on`` references are rewritten through the
    same map, so history ids stay unique.  A stable topological order
    puts dependencies before their dependents; a dependency cycle keeps
    the original relative order instead of raising.  Specs are deduped
    by primary path (first wins), matching the planner's own rule.
    """
    used = set(used_ids or ())
    raw: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for item in specs or []:
        if not isinstance(item, dict):
            continue
        rel = str(item.get("path", "")).strip()
        if not rel or rel in seen_paths:
            continue
        seen_paths.add(rel)
        raw.append(item)
    # pass 1: tentative positional ids -> final collision-free ids
    final_id: dict[int, str] = {}
    counter = 0
    for idx, item in enumerate(raw):
        tentative = str(item.get("id") or f"s{idx + 1}").strip() or f"s{idx + 1}"
        candidate = tentative
        while candidate in used:
            counter += 1
            candidate = f"s{counter}"
            while candidate in used:
                counter += 1
                candidate = f"s{counter}"
        if candidate == tentative and tentative.startswith("s") \
                and tentative[1:].isdigit():
            counter = max(counter, int(tentative[1:]))
        used.add(candidate)
        final_id[idx] = candidate
    tentative_to_final = {
        (str(item.get("id") or f"s{idx + 1}").strip() or f"s{idx + 1}"):
        final_id[idx]
        for idx, item in enumerate(raw)}
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        rel = str(item.get("path", "")).strip()
        deps: list[str] = []
        for dep in item.get("depends_on") or []:
            dep_id = tentative_to_final.get(str(dep).strip())
            if dep_id and dep_id != final_id[idx] and dep_id not in deps:
                deps.append(dep_id)
        files: list[str] = []
        for f in item.get("files") or []:
            f = str(f).strip()
            if f and f not in files:
                files.append(f)
        if rel not in files:
            files.insert(0, rel)
        out.append({
            "path": rel,
            "why": str(item.get("why", "")),
            "new_file": bool(item.get("new_file")),
            "id": final_id[idx],
            "instructions": str(item.get("instructions", "")),
            "depends_on": deps,
            "acceptance": str(item.get("acceptance", "")),
            "files": files,
        })
    # stable topological order: dependencies before dependents
    by_id = {s["id"]: s for s in out}
    ordered: list[dict[str, Any]] = []
    placed: set[str] = set()
    remaining = list(out)
    while remaining:
        progressed = False
        for spec in list(remaining):
            if all(d in placed or d not in by_id
                   for d in spec["depends_on"]):
                ordered.append(spec)
                placed.add(spec["id"])
                remaining.remove(spec)
                progressed = True
        if not progressed:  # dependency cycle — keep original order
            ordered.extend(remaining)
            break
    return ordered


def _step_task(task: str, spec: dict[str, Any]) -> str:
    """The draft prompt for one step: the run's task plus the step's own
    instructions and acceptance check, so each draft works to its step's
    criterion instead of re-deriving intent from the bare task."""
    extra: list[str] = []
    instr = str(spec.get("instructions") or "").strip()
    if instr:
        extra.append(f"Step {spec.get('id', '')} instructions: {instr}")
    acc = str(spec.get("acceptance") or "").strip()
    if acc:
        extra.append(f"Step {spec.get('id', '')} is accepted when: {acc}")
    if not extra:
        return task
    return task + "\n\n" + "\n".join(extra)


class CodingAgent:
    """One coding session against the real tools and the active model."""

    def __init__(self, context: Any, *, root: str | None = None,
                 pack_budget_chars: int | None = None) -> None:
        self.context = context
        self.db = context.db
        self.router = context.router
        # Pass #64 (W6): char budget for one edit call's context pack
        # (tools.context_pack).  None takes the module default.
        if pack_budget_chars is None:
            from ..tools.context_pack import DEFAULT_PACK_BUDGET_CHARS

            pack_budget_chars = DEFAULT_PACK_BUDGET_CHARS
        self._pack_budget_chars = int(pack_budget_chars)
        # Pack memo: (rel, current hash, step id, error hash) -> pack.
        # Repair rounds against an unchanged file reuse the identical
        # pack instead of re-parsing the repo every round.
        self._pack_cache: dict[tuple[str, int, str, int], Any] = {}
        # The most recent pack built, harvested by run() after each
        # _draft_change call (same pattern as _last_review, pass #63).
        self._last_context_pack: Any | None = None
        # Optional explicit project root (used by `nm code --root`). When set,
        # files resolve under it with a containment check instead of the
        # workspace sandbox — so the agent can work on a real repo checkout.
        self._root = Path(root).expanduser().resolve() if root else None
        # Override in tests to isolate the error-recall index.
        self._recall_path: str | Path | None = None
        self._recall_index: Any | None = None
        # Git mission snapshot (snapshot()/rollback()); None when the
        # workdir is not a git repo or no mission is in flight.
        self._mission_snapshot: dict[str, Any] | None = None
        # Plan-mode approved scope: when execute_plan() sets this, patch
        # targets outside the set are rejected (no silent scope creep).
        self._plan_scope: set[str] | None = None
        self._mission_touched: list[str] = []
        # Checkpoint conversation state (item #17): what the agent was
        # working on, serialized into checkpoints and restored by rewind().
        self._last_task: str = ""
        self._last_plan_id: str = ""
        self._last_spec_id: str = ""
        self._last_scope: list[str] = []
        self._last_iterations: int = 0
        # Pass #63: the most recent _review_diff outcome, harvested by
        # run() right after each _review_flaws call (notes/availability
        # for the result) — the loop is single-threaded per run.
        self._last_review: DiffReview | None = None
        # Pass #65 (W7): the active run's token meter (see _meter_begin)
        # — None outside run(), when nothing is metered or capped.
        self._meter: dict[str, Any] | None = None
        # Spec-first (#4 extension): generation prompts carry the
        # backend-included block by default; set False to opt out.
        self._backend_included: bool = True

    def _error_recall(self) -> Any:
        """The Phase C embedding recall index (lazy, cached, best-effort)."""
        if self._recall_index is None:
            from .error_recall import ErrorRecallIndex

            self._recall_index = ErrorRecallIndex(store_path=self._recall_path)
        return self._recall_index

    def _resolve(self, rel: str) -> Path:
        """Resolve a project-relative path, honoring an explicit root."""
        if self._root is not None:
            candidate = (self._root / rel).resolve()
            try:
                candidate.relative_to(self._root)
            except ValueError as exc:
                raise ValueError(f"path {rel!r} escapes project root {self._root}") from exc
            return candidate
        return safe_path(self.context, rel)

    def _substantive_artifact(self, rels: list[str], workdir: Any) -> bool:
        """Did the run leave a real artifact?  Counts non-blank,
        non-comment lines across the changed files — an empty main.py
        (or a few comment lines) is NOT an artifact."""
        from pathlib import Path
        root = Path(str(workdir))
        total = 0
        for rel in rels or []:
            try:
                p = (root / rel) if not str(rel).startswith("/") else Path(rel)
                if p.is_file():
                    total += _substantive_lines(p)
            except Exception:  # noqa: BLE001 - one bad path never hides
                continue
        return total >= _MIN_ARTIFACT_LINES

    def _git(self, *args: str, cwd: Path | None = None,
             timeout: int = 60) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=cwd or self._resolve("."),
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)

    def _is_git_repo(self, workdir: Path) -> bool:
        try:
            proc = self._git("rev-parse", "--is-inside-work-tree",
                             cwd=workdir)
        except (OSError, subprocess.SubprocessError) as exc:
            _log.debug("snapshot: git probe failed: %s", exc)
            return False
        return proc.returncode == 0 and proc.stdout.strip() == "true"

    # ── git mission rollback ──────────────────────────────────────────
    def snapshot(self) -> dict[str, Any] | None:
        """Snapshot the workdir before a mission starts.

        When the workdir is a git repo with a dirty tree, the dirty state
        is stashed under a mission id; otherwise the current HEAD is
        recorded.  Returns the snapshot dict, or None when the workdir is
        not a git repo (debug-logged, never a crash).
        """
        workdir = self._resolve(".")
        if not self._is_git_repo(workdir):
            _log.debug("snapshot: %s is not a git repo — skipping",
                       workdir)
            self._mission_snapshot = None
            return None
        mission_id = new_short_id("mission")
        head = self._git("rev-parse", "HEAD", cwd=workdir).stdout.strip()
        status = self._git("status", "--porcelain", cwd=workdir).stdout
        dirty = bool(status.strip())
        # Untracked files/dirs present BEFORE the mission: anything
        # untracked that appears later is a mission side effect and gets
        # removed by rollback().
        untracked_before = sorted(
            line[3:].strip().strip('"')
            for line in status.splitlines()
            if line.startswith("??"))
        stashed = False
        if dirty:
            proc = self._git("stash", "push", "-m",
                             f"coding-mission {mission_id}", cwd=workdir)
            if proc.returncode != 0:
                raise RuntimeError(
                    "mission snapshot failed: "
                    f"git stash push: {proc.stderr.strip()}")
            stashed = True
            _log.info("mission %s: stashed dirty workdir", mission_id)
        self._mission_snapshot = {
            "mission_id": mission_id, "head": head,
            "stashed": stashed, "repo": str(workdir),
            "untracked_before": untracked_before,
        }
        self._mission_touched = []
        return self._mission_snapshot

    def rollback(self) -> bool:
        """Restore the pre-mission state captured by snapshot().

        Reverts exactly the files the mission touched (tracked files via
        ``git checkout --``, mission-created new files by deletion), then
        pops the mission stash when one was taken.  Files the mission
        never touched are never modified.  Returns True on success, False
        when any step failed (the failure is logged loudly — callers
        surface it in the mission error).
        """
        snap = self._mission_snapshot
        if not snap:
            _log.debug("rollback: no snapshot — nothing to restore")
            return True
        workdir = Path(snap["repo"])
        ok = True
        for rel in self._mission_touched:
            target = workdir / rel
            try:
                target.resolve().relative_to(workdir.resolve())
            except ValueError:
                _log.error("rollback: %r escapes repo — skipping", rel)
                ok = False
                continue
            try:
                untracked = (self._git("ls-files", "--error-unmatch", rel,
                                      cwd=workdir).returncode != 0)
                if untracked:
                    # Mission-created file: checkout cannot restore it,
                    # so it must be removed for a true rollback.
                    if target.is_file():
                        target.unlink()
                        _log.info("rollback: removed mission-created %s",
                                  rel)
                else:
                    proc = self._git("checkout", "--", rel, cwd=workdir)
                    if proc.returncode != 0:
                        raise RuntimeError(proc.stderr.strip())
            except (OSError, RuntimeError,
                    subprocess.SubprocessError) as exc:
                _log.error("rollback: failed to revert %s: %s", rel, exc)
                ok = False
        if snap.get("stashed"):
            try:
                proc = self._git("stash", "pop", cwd=workdir)
            except (OSError, subprocess.SubprocessError) as exc:
                _log.error("rollback: git stash pop crashed: %s", exc)
                return False
            if proc.returncode != 0:
                _log.error("rollback: git stash pop failed: %s",
                           (proc.stderr.strip() or proc.stdout.strip()))
                return False
            _log.info("mission %s: stash popped", snap.get("mission_id"))
        # Remove untracked files/dirs the mission created as side effects
        # (.edit_backups, __pycache__, …).  Anything untracked BEFORE the
        # mission is in the snapshot set and is never touched.
        try:
            status = self._git("status", "--porcelain",
                               cwd=workdir).stdout
            current = {line[3:].strip().strip('"')
                       for line in status.splitlines()
                       if line.startswith("??")}
            before = set(snap.get("untracked_before", []))
            for rel in sorted(current - before):
                target = workdir / rel
                try:
                    target.resolve().relative_to(workdir.resolve())
                except ValueError:
                    _log.error("rollback: %r escapes repo — skipping", rel)
                    ok = False
                    continue
                try:
                    if target.is_dir() and not target.is_symlink():
                        shutil.rmtree(target)
                    elif target.is_file() or target.is_symlink():
                        target.unlink()
                    else:
                        continue
                    _log.info("rollback: removed mission side effect %s",
                              rel)
                except OSError as exc:
                    _log.error("rollback: failed to remove %s: %s",
                               rel, exc)
                    ok = False
        except (OSError, subprocess.SubprocessError) as exc:
            _log.error("rollback: untracked cleanup failed: %s", exc)
            ok = False
        self._mission_snapshot = None
        self._mission_touched = []
        return ok

    # ── checkpoints + rewind (item #17) ──────────────────────────────
    def _checkpoint_store(self) -> Any:
        """The disk-backed checkpoint store for this agent's workdir."""
        from .checkpoints import CheckpointStore
        return CheckpointStore(
            settings=getattr(self.context, "settings", None))

    def _convo_state(self) -> dict[str, Any]:
        """Serializable snapshot of what the agent is working on."""
        return {
            "task": self._last_task,
            "plan_id": self._last_plan_id,
            "scope": list(self._last_scope),
            "iterations": self._last_iterations,
            "captured_at": time.time(),
        }

    def _restore_convo(self, convo: dict[str, Any]) -> None:
        """Restore conversation state from a checkpoint snapshot."""
        if not isinstance(convo, dict):
            return
        self._last_task = str(convo.get("task", "") or "")
        self._last_plan_id = str(convo.get("plan_id", "") or "")
        scope = convo.get("scope") or []
        self._last_scope = [str(s) for s in scope] \
            if isinstance(scope, list) else []
        try:
            self._last_iterations = int(convo.get("iterations", 0) or 0)
        except (TypeError, ValueError):
            self._last_iterations = 0

    def checkpoint(self, label: str = "",
                   *, convo: dict[str, Any] | None = None) -> Any:
        """Save a recovery checkpoint of the workdir + conversation state.

        Code is captured with ``git stash create`` (a stash *commit* —
        the user's own ``git stash`` list is never touched); untracked
        file contents are copied into the checkpoint dir.  Never raises:
        when git is unavailable a convo-only checkpoint is stored instead.
        """
        store = self._checkpoint_store()
        workdir = self._resolve(".")
        try:
            ckpt = store.capture(
                workdir, label=label,
                convo=convo if convo is not None else self._convo_state())
            return store.save(ckpt)
        except Exception as exc:  # noqa: BLE001 — checkpoint never kills a run
            _log.warning("checkpoint failed, storing convo-only: %s", exc)
            fallback = store.capture(
                Path("__no_repo__"), label=label,
                convo=self._convo_state())
            fallback.code_captured = False
            fallback.note = f"capture failed ({exc})"
            return store.save(fallback)

    def rewind(self, n: int = 1) -> str:
        """Restore the nth-latest checkpoint (n=1 → latest).

        A ``pre-rewind`` safety checkpoint is saved first, so the rewind
        itself is reversible.  Only working-tree files inside the
        checkpoint's scope are rewritten — HEAD is never moved.  Returns
        a human summary (never raises).
        """
        store = self._checkpoint_store()
        try:
            n = max(1, int(n))
        except (TypeError, ValueError):
            return f"bad checkpoint number {n!r} — /checkpoints to list."
        try:
            target = store.get_nth(n)
        except Exception as exc:  # noqa: BLE001
            return f"could not read checkpoints: {exc}"
        if target is None:
            return "no checkpoints yet — /checkpoint to save one first."
        safety = self.checkpoint(label="pre-rewind")
        summary = store.rewind_to(target)
        self._restore_convo(target.convo)
        return (summary +
                f"\n🛟 safety checkpoint {safety.id} (label 'pre-rewind') "
                "saved — /rewind 1 undoes this rewind.")

    def list_checkpoints(self) -> list[Any]:
        """All checkpoints, newest first. Never raises."""
        try:
            return self._checkpoint_store().list()
        except Exception as exc:  # noqa: BLE001
            _log.debug("list_checkpoints: %s", exc)
            return []

    def _rollback_after_failure(self, changed: list[str]) -> str:
        """Roll back a failed mission that made changes.

        Returns a warning suffix for the mission error ("" when the
        rollback succeeded or there was nothing to roll back).
        """
        if not changed:
            return ""
        self._mission_touched = list(dict.fromkeys(changed))
        try:
            rolled_back = self.rollback()
        except Exception as exc:  # noqa: BLE001 — never mask the mission error
            _log.error("mission rollback crashed: %s", exc)
            return (" | WARNING: mission rollback crashed — "
                    "working tree may be dirty")
        if not rolled_back:
            return (" | WARNING: mission rollback failed — "
                    "working tree may be dirty")
        _log.info("failed mission rolled back %d file(s)", len(changed))
        return ""

    def _explore_reads(
        self, plan: list[dict[str, Any]],
    ) -> dict[str, str | None]:
        """Phase D explore phase: read every planned file in one parallel
        ``call_many`` block (bounded at 8 workers) instead of N sequential
        reads.

        Routes through the registry's ``fs_read`` so explore reads are
        audited like every other tool call.  Falls back to direct reads
        when no registry is available (minimal contexts) or when
        ``fs_read`` cannot resolve a path — e.g. the agent root sits
        outside the tool workspace, or the file exceeds ``fs_read``'s
        byte cap.
        """
        rels = [spec["path"] for spec in plan]
        tools = getattr(self.context, "tools", None)
        call_many = getattr(tools, "call_many", None)
        if call_many is not None:
            outcomes: Any = None
            try:
                outcomes = call_many(
                    [("fs_read", {"path": str(self._resolve(rel))})
                     for rel in rels],
                    max_workers=8,
                )
            except Exception as exc:  # noqa: BLE001 — fall back to direct
                _log.debug("parallel explore reads failed: %s", exc)
            if outcomes is not None:
                texts: dict[str, str | None] = {}
                for rel, outcome in zip(rels, outcomes):
                    content: str | None = None
                    if getattr(outcome, "ok", False):
                        value = outcome.value or {}
                        content = value.get("content")
                    if content is None:
                        p = self._resolve(rel)
                        content = (p.read_text(encoding="utf-8")
                                   if p.is_file() else None)
                    texts[rel] = content
                return texts
        # legacy sequential path (no registry on the context)
        texts = {}
        for spec in plan:
            p = self._resolve(spec["path"])
            texts[spec["path"]] = (
                p.read_text(encoding="utf-8") if p.is_file() else None
            )
        return texts

    # ── pass #65 (W7): the run's token meter ────────────────────────

    def _meter_begin(self, cap: int | None) -> None:
        """Start a fresh meter for a run.  ``cap`` is the caller's
        optional spend cap (None = metering only, never blocks)."""
        self._meter = {"totals": _blank_usage(), "by_phase": {},
                       "cap": cap, "capped": False}

    def _meter_fold(self, phase: str, prompt: int, completion: int,
                    total: int, *, reported: bool) -> None:
        meter = self._meter
        if meter is None:
            return
        for bucket in (meter["totals"],
                       meter["by_phase"].setdefault(phase,
                                                    _blank_usage())):
            bucket["calls"] += 1
            if reported:
                bucket["prompt_tokens"] += prompt
                bucket["completion_tokens"] += completion
                bucket["total_tokens"] += total
            else:
                bucket["usage_unreported"] += 1

    def _meter_response(self, response: Any, phase: str) -> None:
        """Record one funneled model response's usage.  A response
        with no usable usage (or all-zero usage — a provider that did
        not report) counts as one unreported call: its tokens stay
        honestly unknown in the totals, never estimated."""
        if self._meter is None:
            return
        extracted = _response_usage(response)
        if extracted is None or extracted == (0, 0, 0):
            self._meter_fold(phase, 0, 0, 0, reported=False)
        else:
            self._meter_fold(phase, *extracted, reported=True)

    def _meter_external(self, usage: Any, phase: str) -> None:
        """Record a model call made OUTSIDE the funnel whose usage
        arrived as a counters dict (the reasoning review/revise calls,
        via their status channel).  A missing/empty dict means the
        call's cost was not reported — count it unreported."""
        if self._meter is None:
            return
        if isinstance(usage, dict) and usage.get("calls"):
            for bucket in (self._meter["totals"],
                           self._meter["by_phase"].setdefault(
                               phase, _blank_usage())):
                bucket["calls"] += int(usage.get("calls") or 0)
                bucket["prompt_tokens"] += int(
                    usage.get("prompt_tokens") or 0)
                bucket["completion_tokens"] += int(
                    usage.get("completion_tokens") or 0)
                bucket["total_tokens"] += int(
                    usage.get("total_tokens") or 0)
                bucket["usage_unreported"] += int(
                    usage.get("usage_unreported") or 0)
        else:
            self._meter_fold(phase, 0, 0, 0, reported=False)

    def _check_spend_cap(self) -> None:
        """Raise _SpendCapExceeded when the metered total has
        crossed the run's cap — called BEFORE issuing a model call,
        so a capped run never pays for the call that would cross it
        further.  No cap (or no active run) never raises."""
        meter = self._meter
        if meter is None:
            return
        cap = meter.get("cap")
        if cap is None:
            return
        spent = meter["totals"]["total_tokens"]
        if spent >= cap:
            meter["capped"] = True
            raise _SpendCapExceeded(
                f"spend cap reached: {spent} tokens spent "
                f"(cap {cap}) — run stopped before the next model call")

    def _usage_snapshot(self) -> dict[str, Any]:
        """The result-shaped usage dict for the active meter ({} when
        no run metered).  by_phase is copied so later calls cannot
        mutate a settled result."""
        meter = self._meter
        if meter is None:
            return {}
        out: dict[str, Any] = dict(meter["totals"])
        out["by_phase"] = {phase: dict(counters)
                           for phase, counters
                           in meter["by_phase"].items()}
        if meter.get("cap") is not None:
            out["cap"] = meter["cap"]
        if meter.get("capped"):
            out["capped"] = True
        return out

    def chat(self, prompt: str) -> Any:
        """Simple chat interface for tools like EditLoop and CodeExecutor."""
        from ..llm.base import Message

        # (Fixed 2026-10-09: this used to pass raw dicts, but providers
        # read ``message.role`` / ``message.content`` as attributes — every
        # call fell into the router's error path.  Real Messages now.)
        self._check_spend_cap()
        response = self._brain().chat([Message.user(prompt)],
                                      task_kind="chat")
        self._meter_response(response, "chat")
        return response

    def _brain(self) -> Any:
        """Lazy Brain over this agent's router.  Never raises."""
        brain = getattr(self, "_coding_brain", None)
        try:
            from ..llm.brain import Brain

            if brain is None or getattr(brain, "_external_router", None) is not self.router:
                brain = Brain(router=self.router)
                self._coding_brain = brain
            return brain
        except Exception:  # noqa: BLE001
            return self.router

    # ── complexity-routed model selection (build-map #18) ────────────────────
    def _select_model(self, phase: str) -> Any:
        """Return the chat target for a coding phase.

        ``phase="plan"`` (Architect): hard complexity, quality objective —
        the strongest available model does the planning.  ``phase="edit"``
        (Editor): medium complexity, cost objective — a fast/cheap model
        applies the edits.

        When intelligent routing is off (the default) this returns
        ``self.router`` unchanged: today's failover chain, zero behavior
        change.  Never raises.
        """
        try:
            from .router_select import TaskRouter
            picker = TaskRouter(self.context)
            if not picker.enabled():
                return self.router
            if phase == "plan":
                choice = picker.select(
                    task_type="coding", objective="quality", complexity="hard")
            else:
                choice = picker.select(
                    task_type="coding", objective="cost", complexity="medium")
            if choice is not None:
                provider = self.router.get(choice.name)
                if provider is not None:
                    return provider
        except Exception:  # noqa: BLE001 — routing must never break coding
            _log.debug("complexity routing unavailable; using default chain",
                       exc_info=True)
        return self.router

    def _phase_chat(self, phase: str, messages: Any,
                    params: Any = None, **kw: Any) -> Any:
        """Phase-aware chat: routed provider first, failover chain on failure.

        When routing is off (or the routed pick fails), this is exactly
        ``self.router.chat(...)`` — the existing behavior, untouched.
        """
        self._check_spend_cap()
        target = self._select_model(phase)
        if target is self.router:
            # Unrouted: the brain owns the chain — task-kind threading,
            # timeouts, the failure taxonomy, context fitting.
            resp = self._brain().chat(messages, params,
                                      task_kind="coding", **kw)
            self._meter_response(resp, phase)
            return resp
        tier = "hard" if phase == "plan" else "medium"
        try:
            resp = target.chat(messages, params, **kw)
            if resp is not None and getattr(resp, "ok", False):
                self._meter_response(resp, phase)
                try:
                    from ..llm.router import log_llm_call
                    log_llm_call(
                        getattr(target, "name", "?"),
                        getattr(target, "model_id", "") or "",
                        resp, tier=tier, operation="chat")
                except Exception:  # noqa: BLE001
                    _log.debug("cost log failed", exc_info=True)
                return resp
            _log.warning(
                "coding %s phase: provider %s failed (%s); chain fallback",
                phase, getattr(target, "name", "?"),
                getattr(resp, "error", "?") if resp is not None else "?")
        except Exception as exc:  # noqa: BLE001 — fall through to the chain
            _log.warning("coding %s phase: provider %s raised (%s); "
                         "chain fallback",
                         phase, getattr(target, "name", "?"), exc)
        resp = self._brain().chat(messages, params, task_kind="coding",
                                  **kw)
        self._meter_response(resp, phase)
        return resp

    # ── public ──────────────────────────────────────────────────────────────
    def run(
        self,
        task: str,
        *,
        filename: str = "main.py",
        accept: str = "",
        max_iterations: int = 5,
        timeout: float = 60.0,
        seed_code: str = "",
        background_tests: bool = False,
        test_cap_seconds: float = 900.0,
        plan_mode: Any = "auto",
        full_suite_gate: bool = False,
        max_tokens: int | None = None,
        _plan: list[dict[str, Any]] | None = None,
    ) -> CodingResult:
        """Run the surgical multi-file edit loop (audit Phase B).

        Pass #65 (W7): every run is METERED — the token usage of
        every model call the run makes (plan / draft / review /
        revise) lands on ``CodingResult.usage`` with per-phase
        attribution; calls whose provider reported no usage count as
        ``usage_unreported``, never as invented numbers.  Metering
        never blocks by itself.  ``max_tokens`` (default None = no
        cap) optionally bounds the spend: once the metered total
        crosses it, the loop stops BEFORE issuing the next model
        call and settles honestly — a verify-green that already
        stands (with no edit landed since) keeps its verdict with
        the cap noted; anything else fails naming the cap and the
        tokens spent, per-step status preserved.  The full loop
        contract is documented on :meth:`_run_impl`.
        """
        if max_tokens is not None:
            max_tokens = int(max_tokens)
            if max_tokens < 0:
                raise ValueError("max_tokens must be >= 0 or None")
        self._meter_begin(max_tokens)
        try:
            result = self._run_impl(
                task, filename=filename, accept=accept,
                max_iterations=max_iterations, timeout=timeout,
                seed_code=seed_code, background_tests=background_tests,
                test_cap_seconds=test_cap_seconds, plan_mode=plan_mode,
                full_suite_gate=full_suite_gate, _plan=_plan)
        finally:
            snapshot = self._usage_snapshot()
            self._meter = None
        result.usage = snapshot
        return result

    def _run_impl(
        self,
        task: str,
        *,
        filename: str = "main.py",
        accept: str = "",
        max_iterations: int = 5,
        timeout: float = 60.0,
        seed_code: str = "",
        background_tests: bool = False,
        test_cap_seconds: float = 900.0,
        plan_mode: Any = "auto",
        full_suite_gate: bool = False,
        _plan: list[dict[str, Any]] | None = None,
    ) -> CodingResult:
        """Run the surgical multi-file edit loop (audit Phase B).

        Plan which files change, edit them with exact-text replacement via
        the ``edit_file`` tool surface (never whole-file rewrites of
        existing files), verify with the pytest-aware runner, and gate on
        lint.  An explicitly passed ``accept`` command overrides the test
        runner (the ``--accept`` escape hatch for non-Python / exotic
        cases).  A verify round in which zero tests executed is never
        green (pass #58): the loop makes one widened attempt (the repo's
        test directories wholesale), then fails the round honestly as
        UNVERIFIED.  ``CodingResult.verification`` records how the run
        was actually verified ("tests" / "accept" / "smoke" /
        "unverified").  ``seed_code`` pre-loads the default file on round 1.
        ``background_tests`` (Phase D) runs the suite in the background
        while the lint gate runs concurrently, instead of blocking on it;
        ``test_cap_seconds`` caps the background run (default 15 min).

        ``plan_mode`` gates non-trivial work on owner approval:
        ``"auto"`` (default) pauses when the file plan is complex
        (>4 files, >2 new files, or >2 modules); ``True``/``"always"``
        always pauses; ``False``/``"never"`` never pauses.  A paused run
        returns ``CodingResult(needs_approval=True, plan_id=...,
        plan_text=...)`` with NO files touched — approve with
        :meth:`approve_and_execute` or :meth:`execute_plan` after
        :meth:`PlanStore.approve`.

        The plan is a STEP plan (pass #60, W4): each step carries its
        target file(s), concrete instructions, dependencies on other
        steps, and an acceptance check that rides the step's draft
        calls.  ``CodingResult.steps`` reports every step's outcome
        (planned/done/failed/skipped + reason + attempts) — including
        steps superseded by the one mid-run re-plan the loop performs
        when a step fails two rounds running (never inside an approved
        plan's scope).  The iteration budget (W2) is a per-file minimum
        (a draft + at least one repair) plus a shared repair pool the
        whole run draws from — it is NOT ``max_iterations`` divided by
        the file count, so multi-file plans keep real repair rounds at
        the chat caller's budget.

        Verify selection (pass #62) is import-graph aware: the round's
        verify passes the run's accumulated changed files to the
        runner, which selects the tests that actually import them
        (unioned with the filename-stem picks).  ``full_suite_gate``
        (default off) opts the run into the runner's final gate: after
        a green selection run, the full suite runs once and its
        distinct verdict lands on ``CodingResult.final_gate`` — a gate
        failure never rewrites the run's own verdict.

        Edit calls (pass #64, W6) are fed a CONTEXT PACK, not the
        whole file: the target's relevant region (step-steered), the
        definitions it imports, and its covering tests, under a char
        budget with every trim recorded on
        ``CodingResult.context_packs``.  Small or unparseable files
        fall back to the whole file, with the reason in the pack's
        notes — never a silently emptier prompt.
        """
        from ..tools import lint as _lint_mod
        from ..tools import pytest_runner as _pytest_mod
        from ..tools.edit_loop import EditLoop

        started = time.perf_counter()
        explicit_accept = bool(accept)
        use_runner = not accept
        accept = accept or f'python3 "{filename}"'
        # wave 68: a long/multi-clause build ask is structured into a
        # spec first — unchanged from Phase A.
        draft_task = task
        try:
            from .brief import BriefAgent, should_brief

            # gate on the CORE request (see core_request above)
            if should_brief(core_request(task)):
                brief = BriefAgent(self.context).refine(task, kind="code")
                if brief.by == "model":
                    draft_task = brief.as_goal()
        except Exception:  # noqa: BLE001 — structuring is best-effort
            pass

        workdir = self._resolve(".")
        # Item #17: track conversation state for checkpoints.
        self._last_task = task
        self._last_plan_id = ""
        self._last_scope = []
        self._last_iterations = 0
        # Simple scripts (no tests anywhere) just run — the pytest runner's
        # "nothing ran" output confuses more than it helps.
        if use_runner and not _workdir_has_tests(workdir):
            use_runner = False
        # Pass #58: how rounds will be verified.  "runner" and "accept"
        # are real verification; "smoke" means the workdir has NO tests
        # and no accept command was given, so the default accept merely
        # executes the file — disclosed on the result, never dressed up
        # as tested (see CodingResult.verification).
        if explicit_accept:
            verify_mode = "accept"
        elif use_runner:
            verify_mode = "runner"
        else:
            verify_mode = "smoke"
        # Git mission rollback: snapshot the workdir before any change is
        # made.  A failed snapshot must never kill the mission — it just
        # means rollback() becomes a no-op (loudly logged).
        try:
            self.snapshot()
        except Exception as exc:  # noqa: BLE001
            _log.warning("mission snapshot failed — rollback disabled: %s",
                         exc)
            self._mission_snapshot = None
        editor = EditLoop(agent=None, project_root=str(workdir))

        # ── plan step: which files change and why ──
        # Pass #65: the planner call is metered like every other call;
        # a cap crossed before it (only possible at cap 0) settles the
        # run before any step exists.
        try:
            plan = _plan if _plan is not None else self._plan_files(
                draft_task, filename, workdir)
        except _SpendCapExceeded as exc:
            return CodingResult(
                ok=False,
                iterations=0,
                error=str(exc),
                seconds=time.perf_counter() - started,
                steps=[],
                replans=0,
                context_packs=[],
            )
        # Pass #60 (W4): normalize whatever plan arrived (planner output,
        # the fallback, or an approved plan replayed by execute_plan)
        # into STEP specs — ids, instructions, dependencies, acceptance.
        plan = _normalize_steps(plan)
        # ── plan-mode gate (item #4): non-trivial plans pause for owner
        # approval BEFORE any file is touched.  _plan is set only by
        # execute_plan() on an already-approved plan, so it skips the gate.
        if _plan is None and plan_mode not in (False, "never", "off", None):
            from .plan_mode import PlanStore, is_complex, render_plan
            mode = str(plan_mode).lower() if not isinstance(
                plan_mode, bool) else ("always" if plan_mode else "never")
            if mode == "always" or (mode == "auto" and is_complex(plan)):
                code_plan = PlanStore.new(draft_task, plan,
                                          approach="", risks=[])
                self._last_plan_id = code_plan.id
                self._last_scope = [str(s.get("path", ""))
                                    for s in plan if s.get("path")]
                return CodingResult(
                    ok=False,
                    iterations=0,
                    needs_approval=True,
                    plan_id=code_plan.id,
                    plan_text=render_plan(code_plan),
                    seconds=round(time.perf_counter() - started, 2),
                    steps=[{
                        "id": s["id"], "path": s["path"],
                        "files": list(s["files"]),
                        "status": "planned",
                        "reason": "plan awaiting approval",
                        "attempts": 0,
                        "instructions": s.get("instructions", ""),
                        "acceptance": s.get("acceptance", ""),
                        "depends_on": list(s.get("depends_on") or []),
                    } for s in plan],
                )
        self._last_scope = [str(s.get("path", "")) if isinstance(s, dict)
                            else str(s["path"])
                            for s in plan if s.get("path")]
        # Pass #60 (W2): the budget is a per-file MINIMUM plus a shared
        # repair pool — never max_iterations sliced by file count.  The
        # old per_file = max_iterations // len(plan) gave a 5-file plan
        # at the chat budget (5) ONE draft per file and zero repair
        # rounds: the bigger (realer) the job, the less iteration it
        # got.  Now every step is guaranteed min_budget attempts (a
        # draft plus at least one repair), and attempts past the
        # minimum draw from a pool the whole run shares — spent only by
        # steps that keep failing, never pre-sliced by file count.
        # Rounds (verify cycles) still cap at max_iterations and a step
        # drafts at most once per round, so a small plan cannot inflate
        # the caller's budget either: the pool only ever funds repairs
        # for steps that are actually failing.
        min_budget = 2 if max_iterations >= 2 else 1
        repair_pool = max(1, max_iterations)
        spent: dict[str, int] = {}
        # Pass #60 (W4): per-step run state.  step_history holds EVERY
        # step the run has had (a re-plan appends; superseded steps stay
        # for the report), step_state/by_id index the live records.
        step_state: dict[str, dict[str, Any]] = {}
        step_history: list[dict[str, Any]] = []
        by_id: dict[str, dict[str, Any]] = {}

        def _register_steps(specs: list[dict[str, Any]]) -> None:
            for spec in specs:
                rel = spec["path"]
                if rel in step_state:
                    continue
                rec: dict[str, Any] = {
                    "spec": spec, "status": "planned", "reason": "",
                    "fail_streak": 0}
                step_state[rel] = rec
                by_id[spec["id"]] = rec
                step_history.append(rec)
                spent.setdefault(rel, 0)

        _register_steps(plan)

        def _pool_left() -> int:
            used = sum(max(0, n - min_budget) for n in spent.values())
            return repair_pool - used

        def _can_draft(rel: str) -> bool:
            return spent.get(rel, 0) < min_budget or _pool_left() > 0

        def _spend(rel: str) -> None:
            spent[rel] = spent.get(rel, 0) + 1

        def _dep_dead(spec: dict[str, Any]) -> str:
            """Id of a dependency of ``spec`` that can never complete
            (itself failed/skipped, or out of drafting budget with a
            residual error), or "" when every dependency can still land."""
            for dep_id in spec.get("depends_on") or []:
                rec = by_id.get(dep_id)
                if rec is None:
                    continue
                if rec["status"] in ("failed", "skipped"):
                    return dep_id
                dep_rel = rec["spec"]["path"]
                if not _can_draft(dep_rel) and file_errors.get(dep_rel):
                    return dep_id
            return ""

        def _finalize_steps(green: bool) -> list[dict[str, Any]]:
            """Resolve every step record into the result's step report.

            Steps explicitly resolved mid-run (dependency-skipped,
            superseded by a re-plan, fatal draft failure) keep their
            status; the rest resolve from how the run ended: attempted
            + green = done; attempted with a residual error on a failed
            run = failed (the weak step, named); attempted cleanly but
            never verified = skipped (its changes were rolled back);
            never attempted = skipped with the reason."""
            out: list[dict[str, Any]] = []
            for rec in step_history:
                spec = rec["spec"]
                rel = spec["path"]
                status = rec["status"]
                reason = rec["reason"]
                if status == "planned":
                    if spent.get(rel, 0) == 0:
                        status = "skipped"
                        dead = _dep_dead(spec)
                        if dead and dead in cap_stop["steps"]:
                            # The dependency itself was only skipped
                            # because the cap stopped the run — the
                            # cap is this step's root cause too, not
                            # the dependency.  (A dependency that
                            # failed on its own merits keeps the
                            # dependency reason below.)
                            dead = ""
                        if dead:
                            reason = (f"dependency {dead} "
                                      f"({by_id[dead]['spec']['path']}) "
                                      "did not complete")
                        elif green:
                            reason = ("not attempted — verification "
                                      "passed without it")
                        elif cap_stop["hit"]:
                            reason = ("not attempted — spend cap "
                                      "reached "
                                      f"({cap_stop['message']})")
                            cap_stop["steps"].add(spec["id"])
                        else:
                            reason = ("run ended before this step "
                                      "was attempted")
                    elif green:
                        status = "done"
                        reason = ""
                    elif file_errors.get(rel):
                        status = "failed"
                        reason = str(file_errors[rel])[-300:]
                    else:
                        status = "skipped"
                        reason = ("drafted cleanly but the run failed "
                                  "before this step's acceptance could "
                                  "be verified (changes rolled back)")
                    # Write the resolution back: step_history is in
                    # dependency order, so a later step's _dep_dead check
                    # sees its dependency's FINAL state, not "planned".
                    rec["status"] = status
                    rec["reason"] = reason
                out.append({
                    "id": spec["id"], "path": rel,
                    "files": list(spec.get("files") or [rel]),
                    "status": status, "reason": reason,
                    "attempts": spent.get(rel, 0),
                    "instructions": spec.get("instructions", ""),
                    "acceptance": spec.get("acceptance", ""),
                    "depends_on": list(spec.get("depends_on") or []),
                })
            return out

        # Phase D: explore phase reads all planned files in one parallel
        # call_many block instead of N sequential reads.
        texts = self._explore_reads(plan)
        file_errors = {spec["path"]: "" for spec in plan}
        changed: list[str] = []
        last_error = ""
        verification = ""
        # Pass #62: captured from the green verify round's runner
        # result when the caller opted into the final gate.
        final_gate: dict[str, Any] = {}
        # Pass #64 (W6): context-pack metadata, harvested after every
        # _draft_change call from self._last_context_pack.
        context_packs: list[dict[str, Any]] = []
        rounds = max(1, max_iterations)
        # Pass #60 (W4): re-plan bookkeeping — steps attempted last round
        # (for fail streaks) and the one-shot re-plan flags.
        prev_attempted: set[str] = set()
        replans = 0
        replan_tried = False
        # Phase C: diff-review gate state.  all_diffs keeps the latest diff
        # per file across attempts so the critic always sees the full
        # change; review_rounds bounds the critic->rework loop at 2.
        all_diffs: dict[str, str] = {}
        review_rounds = 0
        review_objections: list[str] = []
        gate_exhausted = False
        # Pass #63: review honesty state — unavailability notes from
        # every review that could not run, plus the gate's own outcome.
        # gate_available stays True when no review was attempted
        # because there was no diff to review (nothing to report).
        review_notes: list[str] = []
        gate_available = True
        gate_depth = ""
        # Pass #65 (W7): cap-stop bookkeeping.  cap_stop is read by
        # _finalize_steps (defined above; closures resolve at call
        # time) so unattempted steps name the cap as their reason.
        # green_state snapshots the latest round that completed
        # verify-green + lint, taken BEFORE the Phase C gate review;
        # dirty_since_green tracks whether any edit has landed since
        # — together they decide whether a cap stop may keep the
        # earned verdict (see the loop's _SpendCapExceeded settle).
        cap_stop: dict[str, Any] = {"hit": False, "message": "",
                                    "steps": set()}
        green_state: dict[str, Any] | None = None
        dirty_since_green = False

        try:
            for attempt in range(1, rounds + 1):
                self._last_iterations = attempt  # item #17: checkpoint state
                if attempt > 1:
                    # Pass #60 (W4): fail streaks from last round's attempts
                    # — a step attempted and still carrying an error grows
                    # its streak; a clean attempt resets it.
                    for rel in prev_attempted:
                        rec = step_state.get(rel)
                        if rec is not None:
                            rec["fail_streak"] = (
                                rec["fail_streak"] + 1
                                if file_errors.get(rel) else 0)
                    # Pass #60 (W4): a step failing round after round means
                    # the PLAN is wrong, not just the code — re-plan the
                    # remaining steps ONCE instead of grinding the dead plan
                    # to exhaustion.  Never inside an approved plan's scope
                    # (execute_plan): that scope is the owner's, not the
                    # agent's to redraw.
                    if (not replan_tried and self._plan_scope is None
                            and attempt < rounds):
                        failing = [rec for rec in step_history
                                   if rec["status"] == "planned"
                                   and rec["fail_streak"] >= 2]
                        if failing:
                            replan_tried = True
                            worst = max(failing,
                                        key=lambda r: r["fail_streak"])
                            wspec = worst["spec"]
                            replan_task = (
                                draft_task
                                + "\n\nRE-PLAN: the previous step plan is "
                                "failing — step "
                                f"{wspec['id']} ({wspec['path']}) has failed "
                                f"{worst['fail_streak']} rounds in a row. "
                                "Latest error:\n"
                                + str(file_errors.get(wspec["path"])
                                      or last_error)[-800:]
                                + "\nFiles already changed on disk: "
                                + (", ".join(changed) or "none")
                                + "\nProduce a REVISED step plan for the "
                                "REMAINING work: keep steps whose files are "
                                "already changed and working; replace or "
                                "split the failing approach instead of "
                                "repeating it.")
                            new_specs = self._plan_steps_model(
                                replan_task, filename, workdir,
                                used_ids=set(by_id))
                            if new_specs:
                                new_paths = {s["path"] for s in new_specs}
                                for rec in step_history:
                                    if (rec["status"] == "planned"
                                            and rec["spec"]["path"]
                                            not in new_paths
                                            and rec["spec"]["path"]
                                            not in changed):
                                        rec["status"] = "skipped"
                                        rec["reason"] = ("superseded by "
                                                          "re-plan")
                                for spec in new_specs:
                                    rel = spec["path"]
                                    rec = step_state.get(rel)
                                    if rec is None:
                                        _register_steps([spec])
                                    else:
                                        by_id.pop(rec["spec"]["id"], None)
                                        rec["spec"] = spec
                                        rec["fail_streak"] = 0
                                        by_id[spec["id"]] = rec
                                    file_errors.setdefault(rel, "")
                                    if rel not in texts:
                                        texts.update(
                                            self._explore_reads([spec]))
                                plan = new_specs
                                self._last_scope = [s["path"] for s in plan]
                                replans = 1
                                _log.info(
                                    "coding agent re-planned after step %s "
                                    "failed %d rounds", wspec["id"],
                                    worst["fail_streak"])
                if not any(step_state[spec["path"]]["status"] == "planned"
                           and _can_draft(spec["path"]) and not _dep_dead(spec)
                           for spec in plan):
                    break  # every step spent, done, or dependency-blocked
                spent_before = dict(spent)
                diffs: dict[str, str] = {}
                touched: list[str] = []
                for spec in plan:
                    rel = spec["path"]
                    rec = step_state[rel]
                    if rec["status"] != "planned":
                        continue
                    dead = _dep_dead(spec)
                    if dead:
                        rec["status"] = "skipped"
                        rec["reason"] = (
                            f"dependency {dead} "
                            f"({by_id[dead]['spec']['path']}) did not complete")
                        continue
                    if not _can_draft(rel):
                        continue
                    path = self._resolve(rel)
                    before = path.read_text(encoding="utf-8") if path.is_file() else ""
                    is_new = texts[rel] is None or spec.get("new_file")
                    if is_new:
                        # New files may still be drafted whole.
                        seeded = bool(seed_code) and rel == filename and attempt == 1
                        code = self._draft(
                            _step_task(draft_task, spec), rel,
                            (seed_code if seeded else texts[rel] or ""),
                            file_errors[rel] or last_error, attempt, seeded=seeded)
                        if not code:
                            rec["status"] = "failed"
                            rec["reason"] = ("model returned no code for "
                                             "this step")
                            rb_note = self._rollback_after_failure(changed)
                            return CodingResult(
                                ok=False,
                                iterations=attempt - 1,
                                error=("model returned no code block (check the "
                                       "active provider in /status)") + rb_note,
                                seconds=time.perf_counter() - started,
                                steps=_finalize_steps(False),
                                replans=replans,
                                context_packs=context_packs,
                            )
                        if attempt == 1:
                            code = self._reason_review_code(draft_task, code)
                        path.write_text(code, encoding="utf-8")
                        texts[rel] = code
                    else:
                        change = self._draft_change(
                            _step_task(draft_task, spec), rel, texts[rel] or "",
                            file_errors[rel] or last_error, attempt, workdir,
                            step=spec)
                        # Pass #64: record the pack that fed this edit call
                        # on the run (trimming/fallbacks stay visible).
                        pack = self._last_context_pack
                        self._last_context_pack = None
                        if pack is not None:
                            context_packs.append(pack.metadata())
                        if change is None:
                            file_errors[rel] = "model returned no usable edits"
                            _spend(rel)
                            continue
                        kind, payload = change
                        if kind == "patch":
                            # Unified-diff protocol: the model shipped a whole
                            # diff (possibly multi-file); apply it directly.
                            patch_ok, patch_err = self._apply_model_patch(
                                payload, workdir, texts, diffs, touched, changed)
                            _spend(rel)
                            if not patch_ok:
                                file_errors[rel] = patch_err
                                continue
                            file_errors[rel] = ""
                            if green_state is not None:
                                dirty_since_green = True
                            # Phase B: the harsh reviewer runs over patch diffs
                            # too, not just surgical-edit diffs.  Pass #63: it
                            # runs ALWAYS; an unavailable reviewer is noted on
                            # the result, never mistaken for a clean review.
                            if diffs.get(rel):
                                flaws = self._review_flaws(
                                    draft_task, diffs[rel])
                                last = self._last_review
                                if last is not None and not last.available:
                                    note = (f"per-diff review unavailable "
                                            f"for {rel}: {last.error}")
                                    review_notes.append(note)
                                    _log.warning("coding agent: %s", note)
                                if flaws:
                                    file_errors[rel] = (
                                        "reviewer found flaws in the applied "
                                        "diff: " + "; ".join(flaws))
                            continue
                        edits = payload
                        if not edits:
                            continue  # model judges no change needed
                        apply_errors: list[str] = []
                        for edit in edits:
                            try:
                                editor.surgical_replace(
                                    rel, edit["old_text"], edit["new_text"])
                            except (ValueError, FileNotFoundError) as exc:
                                apply_errors.append(str(exc))
                        texts[rel] = (path.read_text(encoding="utf-8")
                                      if path.is_file() else "")
                        if apply_errors:
                            file_errors[rel] = ("edit application failed: "
                                               + "; ".join(apply_errors))
                            _spend(rel)
                            continue
                        file_errors[rel] = ""
                    _spend(rel)
                    after = path.read_text(encoding="utf-8") if path.is_file() else ""
                    diff_text = _unified_diff(before, after, rel)
                    if not diff_text.strip():
                        continue
                    if green_state is not None:
                        dirty_since_green = True
                    diffs[rel] = diff_text
                    touched.append(rel)
                    if rel not in changed:
                        changed.append(rel)
                    # Phase B: the harsh reviewer runs over EVERY applied diff,
                    # not just attempt-1 new-file drafts. Flaws become next
                    # round's fix prompt for that file.  Pass #63: an
                    # unavailable reviewer is noted, never a silent clean pass.
                    flaws = self._review_flaws(draft_task, diff_text)
                    last = self._last_review
                    if last is not None and not last.available:
                        note = (f"per-diff review unavailable for {rel}: "
                                f"{last.error}")
                        review_notes.append(note)
                        _log.warning("coding agent: %s", note)
                    if flaws:
                        file_errors[rel] = (
                            "reviewer found flaws in the applied diff: "
                            + "; ".join(flaws))

                # Pass #60: which steps actually drafted this round — the
                # next round's fail-streak update reads this.  Recorded
                # BEFORE verify so the lint/review continue paths below
                # cannot skip it.
                prev_attempted = {rel for rel, n in spent.items()
                                  if n > spent_before.get(rel, 0)}

                # ── verify ──
                bg_lint_res: Any = None
                if use_runner:
                    if background_tests:
                        # Phase D: the suite runs in the background while the
                        # agent does the lint gate concurrently (prep work
                        # instead of blocking); poll surfaces live progress.
                        from .bg_tests import background_run_tests

                        bg = background_run_tests(
                            str(workdir), cap_seconds=test_cap_seconds)
                        if isinstance(bg, dict):
                            tres = bg  # honest skip: nothing matched
                        else:
                            bg_lint_res = _lint_mod.lint(
                                changed or [filename], repo=str(workdir))
                            tres = bg.wait(on_progress=_bg_progress)
                    else:
                        # Pass #62: selection follows the work — the runner
                        # gets the run's accumulated changed files (this
                        # round's touches included) instead of re-deriving
                        # the set from git, so selection also works in
                        # non-git workdirs; its import-graph ∪ stem picks
                        # cover those files.  The background path above
                        # keeps targeting the whole tests directory by
                        # design and needs no scoping.
                        tres = _pytest_mod.run_tests(
                            repo=str(workdir),
                            timeout=max(timeout * 5.0, 60.0),
                            changed_files=list(changed) or None,
                            full_suite_gate=full_suite_gate)
                    # Pass #58: ZERO executed tests is never green.  The
                    # runner reports an honest skip (ok=True, passed=0) when
                    # its stem-heuristic selection matches nothing — the
                    # loop used to certify that as a pass.  An explicit
                    # accept command can't occur here (use_runner is False
                    # whenever one was given, so the accept branch below is
                    # already the (a) fallback for those runs).  Otherwise
                    # make ONE widened attempt: the repo's test directories
                    # wholesale instead of the targeted selection.  The
                    # background path needs no widening — it already targets
                    # the whole tests directory (or ".").  Still nothing →
                    # the round fails honestly as UNVERIFIED below.
                    if _tests_nothing_ran(tres) and not background_tests:
                        wide_paths = [d for d in ("tests", "test")
                                      if (workdir / d).is_dir()] or ["."]
                        wide = _pytest_mod.run_tests(
                            paths=wide_paths, repo=str(workdir),
                            timeout=max(timeout * 5.0, 60.0))
                        if not _tests_nothing_ran(wide):
                            tres = wide
                    if _tests_nothing_ran(tres):
                        green = False
                        verification = "unverified"
                        verify_err = (
                            "UNVERIFIED — no tests ran for this change: "
                            + str(tres.get("note")
                                  or "the runner matched no tests")
                            + ". Zero executed tests cannot certify a "
                            "change; this round is NOT green. Add tests "
                            "covering the changed files or pass an explicit "
                            "accept command.")
                        verify_out = (verify_err + "\n"
                                      + _pytest_mod.format_test_result(tres))
                    else:
                        green = (tres["ok"] and not tres["failed"]
                                 and not tres["errors"])
                        verification = "tests"
                        # Pass #62: keep the gate's distinct verdict for
                        # the result (format_test_result already renders
                        # its line into verify_out).  It never touches
                        # `green` — the selection verdict stands alone.
                        if isinstance(tres.get("final_gate"), dict) \
                                and tres["final_gate"]:
                            final_gate = tres["final_gate"]
                        verify_out = _pytest_mod.format_test_result(tres)
                        verify_err = "" if green else verify_out
                else:
                    raw = self._run(accept, workdir, timeout)
                    out_text = (raw.get("stdout") or "").strip()
                    # Real acceptance: exit 0 AND (meaningful stdout OR a
                    # substantive artifact).  An empty main.py that exits 0
                    # with no output is FAILURE, not green — never report
                    # "0 tests, empty file" as success.
                    ran_something = bool(out_text)
                    made_artifact = self._substantive_artifact(
                        changed or [filename], workdir)
                    green = (raw["exit_code"] == 0 and not raw["timed_out"]
                             and (ran_something or made_artifact))
                    verify_out = (raw.get("stdout") or "")[-4000:]
                    verify_err = ((raw.get("stderr") or raw.get("stdout")
                                   or "non-zero exit"))[-4000:]
                    if not green:
                        if raw["exit_code"] == 0 and not raw["timed_out"]:
                            verify_err = (
                                "accept command exited 0 but produced no "
                                "output and no substantive artifact "
                                "(empty/trivial files) — not accepted as "
                                "success")
                            verify_out = (f"empty success rejected:\n{verify_err}")
                        else:
                            verify_out = (f"accept command failed "
                                          f"(exit {raw.get('exit_code')}):\n{verify_err}")
                    verification = verify_mode
                    if verify_mode == "smoke":
                        # No tests exist here and nobody gave an accept
                        # command: the file was EXECUTED, not tested.  Say
                        # so on the result instead of letting a green read
                        # as test-verified (pass #58).
                        verify_out += (
                            "\n[verification: smoke run only — this workdir "
                            "has no tests and no accept command was given; "
                            "the file was executed, not tested]")
                vresult = {"exit_code": 0 if green else 1, "timed_out": False,
                           "stdout": verify_out, "stderr": verify_err}
                for rel in touched:
                    self._journal(task, rel, attempt,
                                  diffs.get(rel, "")[:60000], vresult)
                all_diffs.update({rel: d for rel, d in diffs.items() if d.strip()})
                _log.info("coding agent round %d: %s%s", attempt,
                          "green" if green else "failing",
                          "" if green else " — fixing")

                if green:
                    # ── lint gate: runs AFTER tests go green; lint failures
                    # become fix-iterations exactly like test failures.  In
                    # background mode it already ran concurrently with the
                    # suite — reuse it instead of running it twice. ──
                    lint_res = (bg_lint_res if bg_lint_res is not None
                                else _lint_mod.lint(changed or [filename],
                                                    repo=str(workdir)))
                    if not lint_res["ruff_installed"]:
                        _log.info("ruff not installed — lint gate skipped "
                                  "(honest skip, never a silent pass)")
                    elif not lint_res["ok"]:
                        last_error = _format_lint_result(lint_res)
                        for rel in changed:
                            file_errors[rel] = last_error
                        for rel in touched:
                            self._journal(task, rel, attempt,
                                          diffs.get(rel, "")[:60000],
                                          {"exit_code": 1, "timed_out": False,
                                           "stdout": "", "stderr": last_error})
                        continue
                    # Pass #65: a verify-green now STANDS — snapshot it
                    # before the gate review spends anything, so a cap
                    # stop during the gate (or during a later rework
                    # round) can settle on the earned verdict whenever no
                    # edit has landed since the snapshot.
                    green_state = {
                        "attempt": attempt,
                        "files": list(changed or [filename]),
                        "output": verify_out[-2000:],
                        "verification": verification,
                        "final_gate": dict(final_gate),
                    }
                    dirty_since_green = False
                    # ── Phase C: diff-review gate.  The critic reads the FULL
                    # multi-file diff; its objections become one more targeted
                    # fix iteration (max 2 review rounds — never infinite).
                    gate_flaws: list[str] = []
                    if changed:
                        full_diff = "\n".join(
                            all_diffs.get(rel, "") for rel in changed
                            if all_diffs.get(rel))
                        if full_diff.strip():
                            gate_flaws = self._review_flaws(
                                draft_task, full_diff, focus=_DIFF_REVIEW_FOCUS)
                            # Pass #63: the gate always RUNS on a real diff
                            # now — record whether it produced a verdict.  An
                            # unavailable gate ships the green diff WITHOUT a
                            # clean-review claim (see the result below).
                            last = self._last_review
                            if last is not None:
                                gate_available = last.available
                                gate_depth = last.depth
                                if not last.available:
                                    review_notes.append(
                                        "diff-review gate unavailable: "
                                        f"{last.error}")
                    if gate_flaws:
                        review_objections.extend(gate_flaws)
                        review_rounds += 1
                        if review_rounds < 2:
                            objection_block = (
                                "CODE REVIEW objections — address EACH with a "
                                "minimal edit:\n- " + "\n- ".join(gate_flaws))
                            for rel in changed:
                                file_errors[rel] = objection_block
                                self._journal(
                                    task, rel, attempt,
                                    all_diffs.get(rel, "")[:60000],
                                    {"exit_code": 1, "timed_out": False,
                                     "stdout": "",
                                     "stderr": (f"code-review round "
                                                f"{review_rounds} objections:\n"
                                                f"{objection_block}")})
                            _log.info("code-review round %d found %d flaw(s); "
                                      "reworking", review_rounds, len(gate_flaws))
                            continue
                        # critic never approved and the 2-round budget is spent:
                        # ship the green diff WITH the objections attached —
                        # never a third rework round, never a hang.
                        gate_exhausted = True
                        break
                    # closed loop (wave 66): distill per changed file.
                    for rel in changed:
                        self._distill_session(task, rel, attempt)
                    return CodingResult(
                        ok=True,
                        iterations=attempt,
                        files=changed or [filename],
                        output=verify_out[-2000:],
                        seconds=time.perf_counter() - started,
                        review={"passed": not gate_flaws and gate_available,
                                "rounds": review_rounds,
                                "objections": review_objections,
                                "available": gate_available,
                                "depth": gate_depth,
                                "notes": review_notes},
                        verification=verification,
                        steps=_finalize_steps(True),
                        replans=replans,
                        final_gate=final_gate,
                        context_packs=context_packs,
                    )
                last_error = verify_err
                for rel in touched:
                    if not file_errors[rel]:
                        file_errors[rel] = verify_err[-2000:]

        except _SpendCapExceeded as exc:
            # Pass #65 (W7): the cap stopped the run BEFORE the next
            # model call.  Settle honestly: when a verify-green stands
            # and no edit has landed since it was earned, the verdict
            # is intact — finish green with the cap noted and the gate
            # reported as not completed.  Otherwise fail naming the
            # cap and the spend, roll the unverified changes back
            # (the failed-run rule), and let _finalize_steps mark
            # every unattempted step skipped-with-cap-reason.
            cap_stop["hit"] = True
            cap_stop["message"] = str(exc)
            if green_state is not None and not dirty_since_green:
                review_notes.append(
                    "spend cap reached after a green verify — the "
                    "diff-review gate did not complete; the verified "
                    "verdict stands")
                return CodingResult(
                    ok=True,
                    iterations=green_state["attempt"],
                    files=green_state["files"],
                    output=green_state["output"],
                    seconds=time.perf_counter() - started,
                    review={"passed": False,
                            "rounds": review_rounds,
                            "objections": review_objections,
                            "available": False,
                            "depth": gate_depth,
                            "notes": review_notes},
                    verification=green_state["verification"],
                    steps=_finalize_steps(True),
                    replans=replans,
                    final_gate=green_state["final_gate"],
                    context_packs=context_packs,
                )
            error = str(exc) + self._rollback_after_failure(changed)
            return CodingResult(
                ok=False,
                iterations=attempt,
                error=error,
                seconds=time.perf_counter() - started,
                review={"passed": False,
                        "rounds": review_rounds,
                        "objections": review_objections,
                        "available": gate_available,
                        "depth": gate_depth,
                        "notes": review_notes},
                verification=verification,
                steps=_finalize_steps(False),
                replans=replans,
                final_gate=final_gate,
                context_packs=context_packs,
            )
        if gate_exhausted:
            # Green diff, but the critic never approved within its 2-round
            # budget: ship it WITH the objections attached.
            for rel in changed:
                self._distill_session(task, rel, attempt)
            return CodingResult(
                ok=True,
                iterations=attempt,
                files=changed or [filename],
                output=verify_out[-2000:],
                seconds=time.perf_counter() - started,
                review={"passed": False,
                        "rounds": review_rounds,
                        "objections": review_objections,
                        "available": gate_available,
                        "depth": gate_depth,
                        "notes": review_notes},
                verification=verification,
                steps=_finalize_steps(True),
                replans=replans,
                final_gate=final_gate,
                context_packs=context_packs,
            )
        final_steps = _finalize_steps(False)
        stuck = [s["path"] for s in final_steps if s["status"] == "failed"]
        error = f"still failing after {rounds} attempts: {last_error[-500:]}"
        if stuck:
            error += f" | files that did not converge: {', '.join(stuck)}"
        # Failed mission after making changes: restore the pre-mission state.
        error += self._rollback_after_failure(changed)
        # Phase C: report-only critic on exhausted runs — the diff is still
        # reviewed, but no rework is scheduled.  Pass #63: if that review
        # could not run, the result says so — "passed" never claims a
        # clean review that never happened.
        exhausted_objections: list[str] = []
        exhausted_available = True
        exhausted_depth = ""
        if changed:
            full_diff = "\n".join(all_diffs.get(rel, "") for rel in changed
                                  if all_diffs.get(rel))
            if full_diff.strip():
                # Pass #65: a cap stop here changes nothing about the
                # verdict (the run already failed) — the final review
                # is simply reported as not run, never as clean.
                try:
                    exhausted_objections = self._review_flaws(
                        draft_task, full_diff, focus=_DIFF_REVIEW_FOCUS)
                except _SpendCapExceeded:
                    exhausted_available = False
                    review_notes.append(
                        "final review skipped: spend cap reached")
                else:
                    last = self._last_review
                    if last is not None:
                        exhausted_available = last.available
                        exhausted_depth = last.depth
                        if not last.available:
                            review_notes.append(
                                "final review unavailable: "
                                f"{last.error}")
        return CodingResult(
            ok=False,
            iterations=rounds,
            error=error,
            seconds=time.perf_counter() - started,
            review={"passed": (exhausted_available
                               and not exhausted_objections),
                    "rounds": review_rounds,
                    "objections": review_objections + exhausted_objections,
                    "available": exhausted_available,
                    "depth": exhausted_depth,
                    "notes": review_notes},
            verification=verification,
            steps=final_steps,
            replans=replans,
            final_gate=final_gate,
            context_packs=context_packs,
        )

    # ── plan mode: approve → execute (item #4) ────────────────────────

    def execute_plan(self, plan_id: str) -> CodingResult:
        """Execute an approved plan from :class:`plan_mode.PlanStore`.

        Returns a failed CodingResult (never raises) when the plan is
        unknown or not yet approved.  Execution stays within the approved
        scope: a model patch touching an unplanned file is rejected with
        a re-plan request instead of being silently applied.
        """
        from .plan_mode import PlanStore
        plan = PlanStore.get(plan_id)
        if plan is None:
            return CodingResult(
                ok=False, iterations=0,
                error=f"unknown plan {plan_id!r} — ask for a fresh plan")
        if not plan.approved:
            return CodingResult(
                ok=False, iterations=0,
                error=f"plan {plan_id} not approved — reply 'approve' first")
        # Spec-first (#4 extension): an approved plan becomes a durable
        # design document.  Best-effort — a spec failure never kills the run.
        try:
            from .plan_spec import plan_to_spec as _promote
            spec = _promote(plan)
            if spec is not None:
                self._last_spec_id = spec.key
        except Exception:  # noqa: BLE001
            pass
        # Pass #60: an approved plan's STEP fields ride through to
        # execution (ids, instructions, dependencies, acceptance) —
        # approving a step plan and then executing a bare file list
        # would silently drop everything the owner approved beyond the
        # paths.  The scope covers each step's declared files too.
        specs = []
        for f in plan.files:
            if not f.get("path"):
                continue
            spec = {"path": str(f.get("path", "")),
                    "why": str(f.get("why", "")),
                    "new_file": bool(f.get("new_file"))}
            for key in ("id", "instructions", "acceptance"):
                if f.get(key):
                    spec[key] = str(f[key])
            if f.get("depends_on"):
                spec["depends_on"] = [str(d) for d in f["depends_on"]]
            if f.get("files"):
                spec["files"] = [str(x) for x in f["files"]]
            specs.append(spec)
        self._plan_scope = set()
        for s in specs:
            self._plan_scope.add(s["path"])
            self._plan_scope.update(s.get("files") or [])
        # Item #17: checkpoint the pre-execution state (after the approval
        # check) so a bad run rewinds cleanly.  Never kills the run.
        try:
            self.checkpoint(
                label=f"plan:{plan_id}",
                convo={"task": plan.task, "plan_id": plan_id,
                       "scope": sorted(self._plan_scope),
                       "iterations": 0, "captured_at": time.time()},
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("execute_plan: pre-run checkpoint failed: %s", exc)
        try:
            return self.run(plan.task, _plan=specs, plan_mode="never")
        finally:
            self._plan_scope = None

    def approve_and_execute(self, plan_id: str) -> CodingResult:
        """Approve ``plan_id`` and immediately execute it."""
        from .plan_mode import PlanStore
        plan = PlanStore.approve(plan_id)
        if plan is None:
            return CodingResult(
                ok=False, iterations=0,
                error=f"unknown plan {plan_id!r} — ask for a fresh plan")
        return self.execute_plan(plan_id)

    # ── spec-first (#4 extension): the plan IS a durable design doc ──────

    def plan_to_spec(self, plan_id: str, goals: str = ""):
        """Promote an approved plan to a versioned :class:`plan_spec.PlanSpec`.

        The spec becomes the durable design artifact — code can later be
        regenerated from it with :meth:`regenerate_from_spec`.  Returns
        the PlanSpec, or None for an unknown plan.  Never raises.
        """
        try:
            from .plan_mode import PlanStore
            from .plan_spec import plan_to_spec as _promote
            plan = PlanStore.get(plan_id)
            if plan is None:
                return None
            return _promote(plan, goals=goals)
        except Exception:  # noqa: BLE001 — never raises
            return None

    def regenerate_from_spec(self, spec_key: str) -> dict:
        """Regenerate code FROM the spec document — never from memory.

        For each spec task with a ``path``, re-drafts the file with the
        spec as context and returns a per-file ``{"before", "after",
        "diff", "changed"}`` map (nothing is written; review the diff
        first).  ``{"ok": False, "reason": ...}`` for unknown specs or
        generation failures.  Never raises.
        """
        try:
            from .plan_spec import SpecStore
            spec = SpecStore.get(spec_key)
            if spec is None:
                return {"ok": False,
                        "reason": f"unknown spec {spec_key!r}"}
            files: dict[str, dict] = {}
            context = (
                f"SPEC {spec.key}\n"
                f"Goals: {spec.goals}\n"
                f"Architecture: {spec.architecture}\n"
                f"Data model: {spec.data_model}\n"
                f"API: {spec.api}\n"
                "Regenerate the file below so it implements this spec. "
                "Keep the public surface compatible unless the spec says otherwise."
            )
            for t in spec.tasks or []:
                if not isinstance(t, dict) or not t.get("path"):
                    continue
                rel = str(t["path"])
                try:
                    path = self._resolve(rel)
                    before = path.read_text(encoding="utf-8") \
                        if path.is_file() else ""
                except Exception:  # noqa: BLE001 — one bad path, not the run
                    before = ""
                task = (f"{context}\n\nTask for this file: "
                        f"{t.get('what') or t.get('why') or rel}")
                try:
                    after = self._draft(task, rel, before, "", 1)
                except Exception:  # noqa: BLE001 — never raises
                    after = ""
                if not after:
                    files[rel] = {"ok": False, "reason": "generation failed"}
                    continue
                files[rel] = {
                    "ok": True,
                    "before": before,
                    "after": after,
                    "diff": _unified_diff(before, after, rel),
                    "changed": after != before,
                }
            return {"ok": True, "spec": spec.key, "version": spec.version,
                    "files": files}
        except Exception as exc:  # noqa: BLE001 — never raises
            return {"ok": False, "reason": f"regenerate failed: {exc}"}

    def _plan_files(self, task: str, default: str,
                    workdir: Path) -> list[dict[str, Any]]:
        """Ask the model for the step plan (Phase B plan step, pass #60).

        Returns normalized step dicts (see :func:`_normalize_steps`).
        Falls back to the single default file when the model gives
        nothing usable — the loop stays single-file then, exactly like
        Phase A.
        """
        specs = self._plan_steps_model(task, default, workdir)
        if specs is not None:
            return specs
        return self._default_plan(default, workdir)

    def _plan_steps_model(self, task: str, default: str,
                          workdir: Path,
                          used_ids: set[str] | None = None
                          ) -> list[dict[str, Any]] | None:
        """The planner model call only — normalized steps, or None when
        the model gave nothing parseable.  Split from :meth:`_plan_files`
        so the mid-run re-plan (pass #60) can tell a real revised plan
        from the single-file fallback and never silently shrink a
        multi-step plan to the default file."""
        system = (
            "You are a build planner. Break the task into STEPS — one "
            "step per file created or modified, in the order they should "
            "be done. Respond with EXACTLY ONE fenced ```json block "
            "shaped like "
            '{"steps": [{"path": "relative/path.py", "why": "one line", '
            '"new_file": false, '
            '"instructions": "concrete: what to change in this file and '
            'how", "depends_on": ["s1"], '
            '"acceptance": "the check this step must satisfy"}]} '
            "— paths are relative to the project root. Steps are numbered "
            "s1, s2, ... in array order; depends_on lists the ids of "
            "steps that must be completed before this one (omit or use "
            "[] when there are none). No prose outside the block."
        )
        # Repo map as mission context: the planner sees the real layout
        # (top-level dirs, file purposes, symbol counts) instead of
        # guessing paths.  Best-effort — the mission never breaks over it.
        user = f"Task: {task}"
        try:
            repo_map = build_repo_map(workdir)
        except Exception as exc:  # noqa: BLE001
            _log.debug("repo map failed: %s", exc)
            repo_map = ""
        if repo_map:
            user += ("\n\nRepository map — real layout, prefer these paths "
                     "when choosing files:\n" + repo_map)
        # Architect phase (build-map #18): hard complexity, quality objective.
        response = self._phase_chat(
            "plan",
            [Message.system(system), Message.user(user)],
            SamplingParams(temperature=0.2),
        )
        if not getattr(response, "ok", False):
            return None
        data = _parse_json_block(response.text)
        items = None
        if isinstance(data, dict):
            items = data.get("steps") or data.get("files")
        if not items:
            return None
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for item in items[:12]:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("path", "")).strip()
            if not rel or rel.startswith("/") or ".." in Path(rel).parts:
                continue
            try:
                self._resolve(rel)
            except ValueError:
                continue  # escapes the project root
            if rel in seen:
                continue
            seen.add(rel)
            out.append({
                "path": rel,
                "why": str(item.get("why", ""))[:200],
                "new_file": bool(item.get("new_file"))
                and not (workdir / rel).is_file(),
                "id": str(item.get("id", "")),
                "instructions": str(item.get("instructions", ""))[:1000],
                "depends_on": [str(d) for d in item.get("depends_on") or []],
                "acceptance": str(item.get("acceptance", ""))[:400],
                "files": [str(f) for f in item.get("files") or []],
            })
        if not out:
            return None
        return _normalize_steps(out, used_ids=used_ids)

    def _default_plan(self, default: str,
                      workdir: Path) -> list[dict[str, Any]]:
        return _normalize_steps([{
            "path": default,
            "why": "default target (plan step fallback)",
            "new_file": not (workdir / default).is_file(),
            "instructions": "",
            "acceptance": "the run's verification passes for this file",
        }])

    def _context_pack(self, task: str, rel: str, current: str,
                      last_error: str, workdir: Path,
                      step: dict[str, Any] | None) -> Any:
        """Pass #64 (W6): the context pack for one edit call.

        Built by tools.context_pack from the repo's real structure —
        the step steers it, the last error's identifiers steer repair
        rounds.  Memoized per (file content, step, error) so repeat
        rounds against an unchanged file don't re-parse the repo.
        The pack builder never raises; this wrapper is a second belt:
        any failure degrades to the whole-file pack with the reason in
        its notes, so the prompt is never silently emptier than the
        pre-#64 behavior."""
        from ..tools.context_pack import build_context_pack, whole_file_pack

        key = (rel, hash(current), str((step or {}).get("id", "")),
               hash(last_error or ""))
        cached = self._pack_cache.get(key)
        if cached is not None:
            return cached
        try:
            pack = build_context_pack(
                workdir, rel, current, task=task, step=step,
                last_error=last_error,
                budget_chars=self._pack_budget_chars)
        except Exception as exc:  # noqa: BLE001 — never break the loop
            _log.warning("context pack failed for %s: %s", rel, exc)
            pack = whole_file_pack(
                rel, current, self._pack_budget_chars,
                f"context pack build raised ({type(exc).__name__}: "
                f"{exc}); whole file sent, as before pass #64")
        if len(self._pack_cache) > 64:
            self._pack_cache.clear()
        self._pack_cache[key] = pack
        return pack

    def _draft_change(self, task: str, rel: str, current: str,
                      last_error: str, attempt: int,
                      workdir: Path,
                      step: dict[str, Any] | None = None
                      ) -> tuple[str, Any] | None:
        """One model call for an existing file; two answer protocols.

        Returns ``("edits", [{old_text, new_text}])`` for the surgical-edit
        protocol, ``("patch", patch_text)`` for a unified diff (the caller
        applies it via :meth:`_apply_model_patch`), or None when the model
        gave nothing usable.

        Pass #64 (W6): the model is shown a CONTEXT PACK (see
        :meth:`_context_pack`) — the target's relevant region, its
        import surface, its covering tests — instead of the whole
        file pasted blind.  When the pack falls back to whole-file
        mode the prompt below is byte-for-byte the pre-#64 prompt.
        """
        system = (
            "You are a surgical code editor. Fix the file below with minimal "
            "changes. Answer with EXACTLY ONE fenced block, either:\n"
            "(a) a ```json block shaped like "
            '{"edits": [{"old_text": "<exact text copied verbatim from the file>", '
            '"new_text": "<replacement>"}]} — old_text must appear EXACTLY as '
            "written in the file (copy it verbatim, including whitespace) and "
            "should be unique — include surrounding context lines; or\n"
            "(b) a ```diff block holding a standard unified diff — use this "
            "for multi-file or large changes; paths are relative to the "
            "project root.\n"
            'For (a), if no change is needed, return {"edits": []}. '
            "No prose outside the block."
        )
        pack = self._context_pack(task, rel, current, last_error,
                                  workdir, step)
        self._last_context_pack = pack
        if pack.mode == "pack":
            system += (
                "\nThe file view below is a CONTEXT PACK, not the "
                "whole file: the relevant region (omitted regions are "
                "marked, with the definitions they contain named), "
                "the definitions this file imports from the repo, and "
                "its covering tests. For protocol (a), copy old_text "
                "verbatim from the shown region. If the code you must "
                "change sits in an omitted region, use protocol (b) — "
                "a unified diff against the real file."
            )
            user = (f"Task: {task}\n\nFile: {rel}\n\nContext pack:\n```\n"
                    f"{pack.text}\n```\n\nAttempt {attempt}.")
            _log.info(
                "context pack for %s: %d chars vs %d whole-file "
                "(trimmed %d item(s), %d note(s))",
                rel, len(pack.text), len(current), len(pack.trimmed),
                len(pack.notes))
        else:
            user = (f"Task: {task}\n\nFile: {rel}\n\nCurrent content:\n```\n"
                    f"{current}\n```\n\nAttempt {attempt}.")
            if pack.notes:
                _log.info("context pack for %s: whole file — %s",
                          rel, "; ".join(pack.notes))
        if last_error:
            user += (f"\n\nThe last round FAILED with this exact output:\n```\n"
                     f"{last_error}\n```\nFix it with minimal edits.")
            # hard mid-loop recall (wave 67), same as _draft
            fixes = self._recall_error_fixes(last_error)
            if fixes:
                user += "\n" + fixes
        elif attempt == 1:
            traps = self._trap_warning()
            if traps:
                user += "\n\n" + traps
        # Editor phase (build-map #18): medium complexity, cost objective.
        response = self._phase_chat(
            "edit",
            [Message.system(system), Message.user(user)],
            SamplingParams(temperature=0.2),
        )
        if not getattr(response, "ok", False):
            _log.warning("coding agent: edit-model call failed: %s",
                         getattr(response, "error", "?"))
            return None
        edits = _parse_edits_block(response.text)
        if edits is not None:
            return ("edits", edits)
        patch_text = _parse_patch_block(response.text)
        if patch_text is not None:
            return ("patch", patch_text)
        return None

    def _apply_model_patch(self, patch_text: str, workdir: Path,
                           texts: dict[str, str | None],
                           diffs: dict[str, str], touched: list[str],
                           changed: list[str]) -> tuple[bool, str]:
        """Apply a model-supplied unified diff; refresh texts/diffs/touched.

        Returns ``(True, "")`` on full success, ``(False, reason)`` when the
        patch is unparseable, rejected, or any hunk failed (per-file
        all-or-nothing is enforced by apply_unified_diff).
        """
        try:
            file_patches = parse_unified_diff(patch_text)
        except ValueError as exc:
            return False, f"unparseable unified diff: {exc}"
        if not file_patches:
            return False, "unified diff contained no file patches"
        before: dict[str, str] = {}
        for fp in file_patches:
            try:
                p = self._resolve(fp.target_rel)
            except ValueError as exc:
                return False, f"patch target rejected: {exc}"
            # Plan-mode scope enforcement: an approved plan lists the
            # files the mission may touch. A patch reaching outside that
            # set is not silently applied — the caller re-plans instead.
            if (self._plan_scope is not None
                    and fp.target_rel not in self._plan_scope):
                return False, (
                    f"patch target {fp.target_rel!r} is outside the "
                    "approved plan scope — re-plan instead of silent "
                    "scope creep")
            before[fp.target_rel] = (p.read_text(encoding="utf-8")
                                     if p.is_file() else "")
        try:
            result = apply_unified_diff(patch_text, workdir)
        except ValueError as exc:
            return False, f"patch rejected: {exc}"
        failed = result["failed_hunks"]
        if failed:
            detail = "; ".join(
                f"{h['file']} hunk {h['hunk']}: {h['reason']}"
                for h in failed[:5])
            return False, f"patch application failed: {detail}"
        for target_rel in result["applied_files"]:
            p = self._resolve(target_rel)
            after = p.read_text(encoding="utf-8") if p.is_file() else ""
            texts[target_rel] = after
            diff_text = _unified_diff(before.get(target_rel, ""), after,
                                      target_rel)
            if diff_text.strip():
                diffs[target_rel] = diff_text
                if target_rel not in touched:
                    touched.append(target_rel)
                if target_rel not in changed:
                    changed.append(target_rel)
        return True, ""

    def _review_floor(self) -> bool:
        """Pass #63: should reasoning mode RAISE this review's floor?

        Mode "always" or an active power mode floors the depth tier at
        standard.  That is all mode does now — it never lowers a tier
        the diff earned and never switches review off."""
        try:
            mode = str(getattr(self.context.settings, "reasoning_mode",
                               "auto") or "auto").strip().lower()
        except Exception:  # noqa: BLE001 — bad config must never kill agents
            mode = "auto"
        if mode == "always":
            return True
        try:
            from .power import power_mode_for
            return bool(power_mode_for(self.context).active)
        except Exception:  # noqa: BLE001
            return False

    def _call_reviewer(self, text: str, focus: str,
                       max_calls: int | None) -> tuple[list[str], bool, str]:
        """One reviewer call → (flaws, available, error).

        Uses review_text's pass-#63 status channel.  A reviewer double
        written against the pre-#63 signature (test fixtures patch
        review_text with (context, text, *, focus, max_calls)) rejects
        the extra kwarg — retry without it and take its verdict at face
        value, exactly as the loop did before this pass.

        Pass #65: the spend cap is checked BEFORE the call (raising
        _SpendCapExceeded out of the reviewer seam — run() settles
        it), and the call's usage is metered from the status channel's
        additive "usage" key; a real review that reported no usage
        counts as one unreported call."""
        # Outside the try below on purpose: the cap raise must reach
        # run(), never be swallowed as "reviewer unavailable".
        self._check_spend_cap()
        from .reasoning import review_text

        status: dict[str, Any] = {}
        try:
            flaws = review_text(self.context, text, focus=focus,
                                max_calls=max_calls, status=status)
        except TypeError as exc:
            if "unexpected keyword argument 'status'" not in str(exc):
                return [], False, f"reviewer failed: {exc}"[:200]
            try:
                flaws = review_text(self.context, text, focus=focus,
                                    max_calls=max_calls)
            except Exception as exc2:  # noqa: BLE001
                return [], False, f"reviewer failed: {exc2}"[:200]
            return list(flaws or []), True, ""
        except Exception as exc:  # noqa: BLE001 — never break the loop
            return [], False, f"reviewer failed: {exc}"[:200]
        if not status:
            # The call returned without reporting status (a stand-in
            # reviewer): its verdict stands at face value.  Nothing is
            # metered — the stand-in's machinery is not a model call
            # this loop can account for.
            return list(flaws or []), True, ""
        self._meter_external(status.get("usage"), "review")
        return (list(flaws or []), bool(status.get("available")),
                str(status.get("error") or ""))

    def _review_diff(self, task: str, text: str,
                     focus: str | None = None) -> DiffReview:
        """The harsh reviewer over an applied diff — ALWAYS runs
        (pass #63, W5).  Depth is a function of the diff itself; see
        the scaling table at the DiffReview definition.  ``task`` is
        accepted for signature stability and reviewer context only —
        it no longer gates or sizes anything."""
        base_focus = focus or (
            "the applied diff below: undefined names, wrong indices, "
            "unhandled edge cases, or changes that break the task's "
            "acceptance criteria")
        metrics = _diff_metrics(text)
        tier = _review_tier(metrics)
        if tier == "focused" and self._review_floor():
            tier = "standard"
        outcome = DiffReview(
            depth=tier, files=metrics["file_count"],
            added=metrics["added"], deleted=metrics["deleted"],
            risks=list(metrics["risks"]))
        checks = "; ".join(_RISK_CHECKS[r] for r in metrics["risks"]
                           if r in _RISK_CHECKS)
        deep_focus = base_focus + (f" Also check: {checks}" if checks
                                   else "")

        failed_calls = 0

        def absorb(flaws: list[str], available: bool, error: str,
                   prefix: str = "") -> None:
            nonlocal failed_calls
            outcome.calls += 1
            for f in flaws:
                f = f"{prefix}{f}" if prefix else f
                if f not in outcome.flaws:
                    outcome.flaws.append(f)
            if not available:
                failed_calls += 1
                if not outcome.error:
                    outcome.error = error

        if tier == "focused":
            absorb(*self._call_reviewer(text[:4000], base_focus,
                                        max_calls=1))
        elif tier == "standard":
            absorb(*self._call_reviewer(text, deep_focus, max_calls=None))
        else:  # deep
            chunks = _split_unified_diff(text)
            if len(chunks) >= 2:
                per_file_focus = (
                    "the applied diff for ONE file below: undefined "
                    "names, wrong indices, unhandled edge cases, broken "
                    "internal consistency")
                for path, chunk in chunks[:_DEEP_FILE_CAP]:
                    absorb(*self._call_reviewer(
                        chunk, per_file_focus, max_calls=None),
                        prefix=f"[{path}] " if path else "")
                if len(chunks) > _DEEP_FILE_CAP:
                    rest = "\n".join(c for _, c in
                                     chunks[_DEEP_FILE_CAP:])
                    absorb(*self._call_reviewer(rest, deep_focus,
                                                max_calls=None))
                absorb(*self._call_reviewer(text, deep_focus,
                                            max_calls=None))
            else:
                absorb(*self._call_reviewer(text, deep_focus,
                                            max_calls=None))
        if failed_calls and failed_calls >= outcome.calls:
            outcome.available = False  # no call produced a verdict
        elif failed_calls:
            note = (f"{failed_calls} of {outcome.calls} review calls "
                    "unavailable")
            outcome.error = (f"{note}: {outcome.error}" if outcome.error
                             else note)
            outcome.available = True  # the rest DID produce a verdict
        return outcome

    def _review_flaws(self, task: str, text: str,
                      focus: str | None = None) -> list[str]:
        """The harsh reviewer's flaw list for a diff (pass #63: always
        runs — depth scaled by the diff, never gated by prompt shape or
        reasoning mode).  The full outcome (availability, depth, risks)
        lands on ``self._last_review`` for callers that report it."""
        outcome = self._review_diff(task, text, focus=focus)
        self._last_review = outcome
        return outcome.flaws

    def sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Recent iterations, newest first (for inspection after the fact)."""
        try:
            return list(self.db.query(
                "SELECT id, task, filename, attempt, exit_code, timed_out, created_at "
                "FROM coding_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ))
        except Exception:  # noqa: BLE001 - journal is best-effort
            return []

    def _distill_session(self, task: str, filename: str,
                         iterations: int) -> None:
        """Turn a self-corrected build session into a reusable code skill
        (wave 66).

        Only sessions that FAILED at least once and then recovered carry
        a lesson.  The skill captures the exact error trail from the
        coding journal (error signature as the skill name, each failed
        attempt's stderr as the body, the final working code referenced
        from the journal), so the NEXT build that hits a similar error
        recalls it and applies the proven fix pattern instead of
        re-deriving it.  Best-effort: learning must never break a build.
        """
        if iterations < 2:
            return  # one-shot success: nothing to learn
        try:
            from .skills import SkillLibrary

            rows = list(self.db.query(
                "SELECT attempt, exit_code, stderr FROM coding_log "
                "WHERE task=? AND filename=? ORDER BY attempt DESC LIMIT 5",
                (task[:500], filename)))  # journal caps task at 500
            rows = list(reversed(rows))
            failed = [r for r in rows if int(r.get("exit_code", 0)) != 0
                      or (r.get("stderr") or "").strip()]
            if not failed:
                return
            # signature = the LAST non-empty line of the first failure —
            # for tracebacks that's the actual error ("ModuleNotFoundError:
            # ..."), not the "Traceback (most recent call last):" header
            lines = [ln.strip() for ln in
                     str(failed[0].get("stderr") or "").splitlines()
                     if ln.strip()]
            sig = " ".join((lines[-1] if lines else "").split())[:60]
            slug = re.sub(r"[^a-z0-9]+", "-", sig.lower()).strip("-")[:28]
            if not slug:
                slug = "runtime-error"
            trail = []
            for r in failed[:4]:
                err = " ".join(str(r.get("stderr") or "").split())[:200]
                trail.append(f"- attempt {r['attempt']} (exit "
                             f"{r['exit_code']}): {err or '(no stderr)'}")
            body = (
                f"Self-corrected build session ({iterations} iterations) "
                f"for task: {task[:200]}\n\nSandbox error trail "
                "(oldest first):\n" + "\n".join(trail) +
                f"\n\nResolving pattern: the final code for {filename} "
                "in the coding_log journal makes these errors go away. "
                "Before re-implementing from scratch on a similar task, "
                "recall this skill and apply the same fix pattern.")
            SkillLibrary(self.db).save(
                f"fix-{slug}", kind="code", body=body[:2000],
                description=f"Error -> fix trail for: {task[:120]}",
                tags=["coding", filename.replace(".py", "").replace(".", "-"),
                      slug.split("-")[0] if slug.split("-") else "error"],
                source="coding_session")
            _log.info("coding session distilled into skill fix-%s", slug)
            # Phase C: index the error signature + trail for embedding
            # recall — a renamed-variable variant of this error should
            # still find the fix.
            try:
                skill = SkillLibrary(self.db).get_by_name(f"fix-{slug}")
                if skill is not None:
                    self._error_recall().index(
                        skill.id, sig + " " + " ".join(trail))
            except Exception as exc:  # noqa: BLE001 — recall is best-effort
                _log.debug("error-recall indexing failed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            _log.debug("session distillation failed: %s", exc)

    # ── internals ───────────────────────────────────────────────────────────
    def _draft(
        self, task: str, filename: str, current: str, last_error: str, attempt: int,
        seeded: bool = False, backend_included: bool | None = None,
    ) -> str:
        system = (
            f"You are a coding agent. Write complete, runnable Python for the file "
            f"'{filename}'. Respond with EXACTLY ONE fenced ```python code block "
            "containing the whole file and nothing else — no prose outside the block."
        )
        # Spec-first (#4 extension): backend-included by default — apps/tools
        # ship with their data layer (SQLite schema, storage, scheduled jobs).
        include = (self._backend_included
                   if backend_included is None else backend_included)
        if include:
            from .plan_spec import inject_backend_included, should_backend_include
            if should_backend_include(filename):
                system = inject_backend_included(system)
        user = f"Task: {task}\n\nAttempt {attempt}."
        if current:
            if seeded:
                user += (
                    "\n\nExisting code from the previous step (EXTEND it — keep "
                    "what already works, do not start over):\n```\n"
                    f"{current}\n```"
                )
            else:
                user += f"\n\nPrevious code:\n```\n{current}\n```"
        if last_error:
            user += (
                f"\n\nIt was run and FAILED with this exact output:\n```\n{last_error}\n```\n"
                "Fix the code so the run succeeds."
            )
            # hard mid-loop recall (wave 67): a past session already fixed
            # THIS exact error family — inject the proven fix pattern.
            fixes = self._recall_error_fixes(last_error)
            if fixes:
                user += "\n" + fixes
        elif attempt == 1:
            # up-front systemic warning (wave 67): error families the
            # system keeps hitting across builds — avoid them before
            # they happen.
            traps = self._trap_warning()
            if traps:
                user += "\n\n" + traps
        # Editor phase (build-map #18): medium complexity, cost objective.
        response = self._phase_chat(
            "edit",
            [Message.system(system), Message.user(user)],
            SamplingParams(temperature=0.2),
        )
        if not getattr(response, "ok", False):
            _log.warning("coding agent: model call failed: %s", getattr(response, "error", "?"))
            return ""
        return extract_code_block(response.text)

    def _recall_error_fixes(self, last_error: str) -> str:
        """Phase C recall: embedding similarity search over distilled
        session skills, with the old deterministic string matcher as a
        fallback.  Returns a prompt block with the proven fix trail(s)
        and similarity scores, or '' when nothing matches.  Best-effort —
        memory must never break a build."""
        try:
            from .skills import SkillLibrary

            library = SkillLibrary(self.db)
            hits = self._error_recall().recall(last_error, top_k=3)
            skills: list[tuple[Any, float]] = []
            for hit in hits:
                skill = library.get(hit["skill_id"])
                if skill is not None and not skill.pruned:
                    skills.append((skill, hit["score"]))
            if not skills:
                # fallback: the wave-67 deterministic matcher
                for s in library.match_errors(last_error, limit=3):
                    skills.append((s, 0.0))
            if not skills:
                return ""
            lines = ["\nKNOWN FIX from a previous self-corrected session "
                     "for a SIMILAR error (apply the same pattern):\n"]
            for s, score in skills:
                body = s.body[:500].replace("\n", "\n    ")
                score_bit = f"similarity {score:.2f}, " if score else ""
                lines.append(f"  skill '{s.name}' ({score_bit}tried {s.uses}x, "
                             f"success {s.success_rate:.0%}): {body}")
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            _log.debug("error-fix recall failed: %s", exc)
            return ""

    def _trap_warning(self) -> str:
        """Wave 67 up-front systemic warning: 'known traps' accumulated
        across past builds.  Best-effort."""
        try:
            from .skills import SkillLibrary

            return SkillLibrary(self.db).trap_block(limit=4)
        except Exception as exc:  # noqa: BLE001
            _log.debug("trap warning failed: %s", exc)
            return ""

    def _reason_review_code(self, task: str, code: str) -> str:
        """Review the draft before it runs: a harsh reviewer finds real
        bugs (wrong indices, undefined names, missing edge cases); if it
        finds any, the code is revised once.  Pass #63: always runs —
        depth scales with the draft's size; reasoning mode can only
        deepen it.  An unavailable reviewer leaves the draft unrevised
        (the applied-diff reviews and the gate still cover the run)."""
        from .reasoning import revise_text

        flaws = self._review_flaws(
            task, code,
            focus="python code that must run without error: undefined "
                  "names, wrong indices, unhandled edge cases the task "
                  "implies")
        last = self._last_review
        if last is not None and not last.available:
            _log.info("coding draft review unavailable (%s); draft "
                      "stands unrevised", last.error)
        if not flaws:
            return code
        _log.info("coding draft review found %d flaw(s); revising", len(flaws))
        # Pass #65: the revision is a model call like any other —
        # cap-checked before it is issued, metered after it returns
        # (revise_text reports its usage on the status channel).  A
        # pre-#65 revise double rejects the status kwarg — retry
        # without it, unmetered, exactly the #63 compatibility seam.
        self._check_spend_cap()
        revise_status: dict[str, Any] = {}
        try:
            revised = revise_text(
                self.context, code, flaws,
                focus="the SAME format: exactly one fenced ```python block "
                      "containing the whole file",
                status=revise_status)
        except TypeError as exc:
            if "unexpected keyword argument 'status'" not in str(exc):
                raise
            revised = revise_text(
                self.context, code, flaws,
                focus="the SAME format: exactly one fenced ```python "
                      "block containing the whole file")
            revise_status = {}
        if revise_status:
            self._meter_external(revise_status.get("usage"), "revise")
        if revised == code:
            return code
        block = extract_code_block(revised)
        return block if block.strip() else code

    def _run(self, command: str, workdir: Path, timeout: float) -> dict[str, Any]:
        from ..tools.shell import SandboxLimits, run_sandboxed

        # The sandbox strips PATH down to system dirs; make sure the interpreter
        # that runs this process is still reachable (Termux keeps python under
        # its own prefix, not /usr/bin).
        python_dir = str(Path(sys.executable).parent)
        env = {"PATH": f"{python_dir}:/usr/local/bin:/usr/bin:/bin"}
        return run_sandboxed(
            command,
            cwd=workdir,
            timeout=timeout,
            env=env,
            limits=SandboxLimits(cpu_seconds=int(max(timeout, 5.0))),
        )

    def _journal(
        self, task: str, filename: str, attempt: int, code: str, result: dict[str, Any]
    ) -> None:
        try:
            self.db.execute(
                "INSERT INTO coding_log "
                "(id, task, filename, attempt, exit_code, timed_out, stdout, stderr, code, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_short_id("code"),
                    task[:500],
                    filename,
                    attempt,
                    result["exit_code"] if isinstance(result.get("exit_code"), int) else -1,
                    int(bool(result.get("timed_out"))),
                    (result.get("stdout") or "")[:8000],
                    (result.get("stderr") or "")[:8000],
                    code[:60000],
                    time.time(),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - journaling must never break the loop
            _log.warning("coding_log insert failed: %s", exc)
