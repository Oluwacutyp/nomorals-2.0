"""Deterministic checkers for the Capability suite.

Every checker returns (score, passed, detail): score is 0..1 (fractional
for multi-part checkers), passed is the strict verdict, detail explains
the outcome for the scorecard's failure list.

Checker types (the case's "checker" object):
  exact       {"type":"exact","answers":["..."]}       normalized equality
  numeric     {"type":"numeric","value":42,"tol":0.01} number within tol
  contains    {"type":"contains","all":[...],"any":[...]} substrings
  json        {"type":"json","equals":...}|{"subset":{...}} response is JSON
  code_run    {"type":"code_run","mode":"function","tests":["assert ..."]}
              {"type":"code_run","mode":"program","tests":[{"stdin":..,"stdout":..}]}
  constraints {"type":"constraints","rules":[{kind:...}, ...]} instruction following

Scoring semantics are the recorded pack's (2026-10-22), unchanged: in
particular the numeric checker reads the ANSWER POSITION only — the
final number stated, or an explicitly marked answer — never a number
that merely appears in mid-working scratch (that leak was caught in the
original proofing and fixed there).
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any

# ---------------------------------------------------------------- helpers

_PUNCT_EDGE = " \t\n\r\"'`.,;:!?()[]{}"


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower()).strip(_PUNCT_EDGE)


def _parse_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _numbers_in(text: str) -> list[float]:
    out: list[float] = []
    for m in re.finditer(r"-?\d[\d,]*(?:\.\d+)?", text or ""):
        value = _parse_float(m.group(0))
        if value is not None:
            out.append(value)
    return out


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    m = re.search(r"```(?:json|python|py)?\s*\n(.*?)```", t, re.DOTALL)
    if m:
        return m.group(1).strip()
    return t


# ---------------------------------------------------------------- checkers

def check_exact(resp: str, spec: dict[str, Any]) -> tuple[float, bool, str]:
    answers = spec.get("answers") or [spec.get("answer", "")]
    wanted = {_norm_text(a) for a in answers if a}
    candidates = [_norm_text(resp)]
    lines = [ln for ln in (resp or "").splitlines() if ln.strip()]
    if lines:
        candidates.append(_norm_text(lines[-1]))
        candidates.append(_norm_text(lines[0]))
    # Also try a "final answer:" style tail.
    m = re.search(r"(?:final answer|answer)\s*[:\-]\s*(.+)", resp or "",
                  re.IGNORECASE)
    if m:
        candidates.append(_norm_text(m.group(1)))
    for cand in candidates:
        if cand in wanted:
            return 1.0, True, f"matched {cand!r}"
    got = candidates[1] if len(candidates) > 1 else candidates[0]
    return 0.0, False, f"expected one of {sorted(wanted)}, got {got!r}"


def check_numeric(resp: str, spec: dict[str, Any]) -> tuple[float, bool, str]:
    value = float(spec["value"])
    tol = float(spec.get("tol", 0.0))
    nums = _numbers_in(resp)
    if not nums:
        return 0.0, False, "no number found in response"
    # Convention (the prompts demand it): the FINAL number stated is the
    # answer. An explicitly marked answer ("the answer is 42") also
    # counts. Numbers appearing mid-working do NOT — otherwise a wrong
    # answer whose scratch work happens to contain the right value would
    # pass, which is exactly how trap questions leak.
    candidates = [nums[-1]]
    m = re.search(r"(?:final answer|the answer is|answer)[:\s]*\$?"
                  r"(-?\d[\d,]*(?:\.\d+)?)", resp or "", re.IGNORECASE)
    if m:
        marked = _parse_float(m.group(1))
        if marked is not None:
            candidates.insert(0, marked)
    for n in candidates:
        if abs(n - value) <= tol:
            return 1.0, True, f"answer number {n} within tol {tol} of {value}"
    return 0.0, False, (f"expected {value}±{tol}, answer-position numbers: "
                        f"{candidates} (all numbers seen: {nums})")


def check_contains(resp: str, spec: dict[str, Any]) -> tuple[float, bool, str]:
    low = (resp or "").lower()
    all_items = [s.lower() for s in spec.get("all", [])]
    any_items = [s.lower() for s in spec.get("any", [])]
    total = len(all_items) + (1 if any_items else 0)
    if total == 0:
        return 0.0, False, "empty contains spec"
    hit = sum(1 for s in all_items if s in low)
    missing = [s for s in all_items if s not in low]
    any_hit = any(s in low for s in any_items) if any_items else True
    score = (hit + (1 if any_hit else 0)) / total if any_items else hit / total
    passed = hit == len(all_items) and any_hit
    detail = "all substrings present" if passed else f"missing: {missing}" + (
        "; none of the any-of substrings present" if any_items and not any_hit else "")
    return score, passed, detail


def _subset_score(expected: Any, actual: Any, path: str = "$") -> tuple[int, int, list[str]]:
    """(matched leaves, total leaves, mismatch notes) for subset matching."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return 0, max(1, len(expected)), [f"{path}: expected object"]
        matched = total = 0
        notes: list[str] = []
        for key, sub in expected.items():
            if key not in actual:
                total += 1
                notes.append(f"{path}.{key}: missing")
                continue
            m, t, n = _subset_score(sub, actual[key], f"{path}.{key}")
            matched += m
            total += t
            notes += n
        return matched, total, notes
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return 0, max(1, len(expected)), [f"{path}: list mismatch"]
        matched = total = 0
        notes = []
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            m, t, n = _subset_score(e, a, f"{path}[{i}]")
            matched += m
            total += t
            notes += n
        return matched, total, notes
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            ok = abs(float(expected) - float(actual)) < 1e-9
        except (TypeError, ValueError):
            ok = False
    else:
        ok = expected == actual
    return (1 if ok else 0), 1, ([] if ok else [f"{path}: {actual!r} != {expected!r}"])


def check_json(resp: str, spec: dict[str, Any]) -> tuple[float, bool, str]:
    raw = _strip_fences(resp)
    try:
        actual = json.loads(raw)
    except json.JSONDecodeError as exc:
        return 0.0, False, f"response is not valid JSON: {exc}"
    if "equals" in spec:
        expected = spec["equals"]
        m, t, notes = _subset_score(expected, actual)
        exact = actual == expected
        return (1.0 if exact else m / t), exact, (
            "exact JSON match" if exact else "; ".join(notes) or "value mismatch")
    subset = spec.get("subset", {})
    m, t, notes = _subset_score(subset, actual)
    score = m / t if t else 1.0
    return score, score == 1.0, "subset matched" if score == 1.0 else "; ".join(notes)


# ------------------------------------------------------------- code running

def _extract_code(resp: str) -> str:
    t = resp or ""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", t, re.DOTALL)
    if m:
        return m.group(1)
    # No fence: accept the whole response if it parses as Python.
    try:
        ast.parse(t)
        return t
    except SyntaxError:
        return ""


def _run_python(source: str, stdin: str = "", timeout: float = 10.0) -> tuple[int, str, str]:
    """Run candidate code in a scrubbed subprocess. Returns (rc, out, err).

    Isolation: python -I (isolated mode), minimal env, temp cwd, hard
    timeout. This is a local eval sandbox for the owner's own models, not
    a hostile-code jail — don't point it at untrusted third-party code.
    """
    with tempfile.TemporaryDirectory(prefix="braineval-") as td:
        script = os.path.join(td, "candidate.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(source)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
               "PYTHONIOENCODING": "utf-8", "LC_ALL": "C.UTF-8"}
        try:
            proc = subprocess.run(
                [sys.executable, "-I", script], input=stdin,
                capture_output=True, text=True, timeout=timeout, cwd=td,
                env=env)
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"timeout after {timeout}s"


def check_code_run(resp: str, spec: dict[str, Any]) -> tuple[float, bool, str]:
    code = _extract_code(resp)
    if not code.strip():
        return 0.0, False, "no runnable Python code found in response"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return 0.0, False, f"code does not parse: {exc}"
    mode = spec.get("mode", "function")
    tests = spec.get("tests", [])
    if not tests:
        return 0.0, False, "case has no tests (authoring bug)"
    passed_count = 0
    notes: list[str] = []
    if mode == "function":
        for i, test in enumerate(tests):
            harness = code + f"\n\n{test}\nprint('__TEST_OK__')\n"
            rc, out, err = _run_python(harness)
            if rc == 0 and "__TEST_OK__" in out:
                passed_count += 1
            else:
                tail = (err.strip().splitlines() or ["no output"])[-1]
                notes.append(f"test {i + 1} failed: {tail[:120]}")
    elif mode == "program":
        for i, test in enumerate(tests):
            rc, out, err = _run_python(code, stdin=test.get("stdin", ""))
            got = re.sub(r"\s+", " ", out.strip())
            want = re.sub(r"\s+", " ", str(test.get("stdout", "")).strip())
            if rc == 0 and got == want:
                passed_count += 1
            else:
                notes.append(
                    f"case {i + 1}: expected stdout {want!r}, got {got!r} "
                    f"(rc={rc})")
    else:
        return 0.0, False, f"unknown code_run mode {mode!r}"
    score = passed_count / len(tests)
    detail = f"{passed_count}/{len(tests)} tests passed"
    if notes:
        detail += "; " + "; ".join(notes[:3])
    return score, passed_count == len(tests), detail


# -------------------------------------------------------------- constraints

def _constraint_ok(resp: str, rule: dict[str, Any]) -> tuple[bool, str]:
    kind = rule.get("kind", "")
    lines = [ln for ln in (resp or "").splitlines() if ln.strip()]
    text = resp or ""
    if kind == "exact_lines":
        ok = len(lines) == int(rule["n"])
        return ok, f"{len(lines)} non-empty lines, want {rule['n']}"
    if kind == "max_words":
        n = len(text.split())
        return n <= int(rule["n"]), f"{n} words, max {rule['n']}"
    if kind == "min_words":
        n = len(text.split())
        return n >= int(rule["n"]), f"{n} words, min {rule['n']}"
    if kind == "exact_words":
        n = len(text.split())
        return n == int(rule["n"]), f"{n} words, want exactly {rule['n']}"
    if kind == "starts_with":
        ok = text.lstrip().startswith(rule["text"])
        return ok, f"starts with {rule['text']!r}: {ok}"
    if kind == "lines_start_with":
        ok = all(ln.lstrip().startswith(rule["text"]) for ln in lines) and bool(lines)
        return ok, f"all lines start with {rule['text']!r}: {ok}"
    if kind == "contains":
        ok = rule["text"].lower() in text.lower()
        return ok, f"contains {rule['text']!r}: {ok}"
    if kind == "not_contains":
        ok = rule["text"].lower() not in text.lower()
        return ok, f"must not contain {rule['text']!r}: {ok}"
    if kind == "max_sentences":
        n = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
        return n <= int(rule["n"]), f"~{n} sentences, max {rule['n']}"
    if kind == "uppercase":
        letters = [c for c in text if c.isalpha()]
        ok = bool(letters) and all(c.isupper() for c in letters)
        return ok, f"all-caps: {ok}"
    if kind == "json_only":
        raw = _strip_fences(text)
        try:
            json.loads(raw)
            return True, "parses as JSON"
        except json.JSONDecodeError:
            return False, "not parseable as JSON"
    return False, f"unknown constraint kind {kind!r}"


def check_constraints(resp: str, spec: dict[str, Any]) -> tuple[float, bool, str]:
    rules = spec.get("rules", [])
    if not rules:
        return 0.0, False, "case has no constraint rules (authoring bug)"
    results = [_constraint_ok(resp, r) for r in rules]
    ok_count = sum(1 for ok, _ in results if ok)
    notes = [d for ok, d in results if not ok]
    score = ok_count / len(rules)
    return score, ok_count == len(rules), (
        f"{ok_count}/{len(rules)} constraints satisfied"
        + ("; failed: " + "; ".join(notes) if notes else ""))


CHECKERS = {
    "exact": check_exact,
    "numeric": check_numeric,
    "contains": check_contains,
    "json": check_json,
    "code_run": check_code_run,
    "constraints": check_constraints,
}


def run_checker(resp: str, spec: dict[str, Any]) -> tuple[float, bool, str]:
    fn = CHECKERS.get(spec.get("type", ""))
    if fn is None:
        return 0.0, False, f"unknown checker type {spec.get('type')!r}"
    return fn(resp, spec)
