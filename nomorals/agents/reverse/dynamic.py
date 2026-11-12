"""The dynamic rung — Frida + GDB under the namespace sandbox.

The gate is dissolved (owner order 2026-11-12): the dynamic rung
is built, and its sandbox is part of the build, not a question.
What runs here runs ONLY through :func:`nomorals.tools.shell.run_sandboxed`
with the network namespace dropped, in a per-run workdir, as a
COPY of the target (the original is never executed in place), under
a hard timeout — and only when a namespace backend (unshare on
Linux) actually works. When no namespace backend works the rung
refuses outright: an rlimit fallback is not a sandbox, and this
module never pretends it is.

Two engines, one evidence shape:

* **Frida** (:mod:`.frida_driver`) — spawn the target suspended,
  intercept the main module's exported + imported functions, resume,
  and report the calls, the exit code, and whether the run timed
  out. Frida is self-provisioned (see :mod:`.provision`: pinned
  17.26.2, pip --require-hashes venv).
* **GDB** — drive ``gdb --batch`` with a generated command script:
  break at main (or the ELF entry point when main has no symbol),
  snapshot registers / arguments / locals / backtrace at each stop,
  continue to exit, and report the stops, the exit code, and any
  signal. GDB is a system debugger provisioned through apt.

Both fold into the Evidence core with the rung named on the
record (``dynamic.frida.trace`` / ``dynamic.gdb.session``),
confidence observed, authority controlled-replay. A run that
observes nothing opens a residual unknown instead of a dressed-up
empty.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import time
from pathlib import Path
from typing import Any

from ...tools.shell import (
    SandboxLimits,
    detect_backend,
    proc_confined,
    run_sandboxed,
)
from . import provision
from .failures import record_failure
from .inspectors import inspect_bytes

__all__ = [
    "PROVIDER_ID", "PROVIDER_NAME", "DynamicError",
    "RUN_TIMEOUT_S", "LIMITATIONS_FRIDA", "LIMITATIONS_GDB",
    "PROC_LIMITATION",
    "sandbox_state", "dynamic_state", "run_frida", "run_gdb",
    "run_dynamic", "fold_evidence",
]

PROVIDER_ID = "devon.re-dynamic"
PROVIDER_NAME = "Devon RE dynamic rung (Frida/GDB, sandboxed)"

#: Analysis budget for one dynamic run, seconds. Bounded by
#: design: a target that cannot finish inside it yields partial
#: evidence + a timeout flag, not an unbounded run.
RUN_TIMEOUT_S = 120.0
#: The sandbox gets a margin over the engine's own timeout so the
#: engine's kill lands first and its partial report survives.
_SANDBOX_MARGIN_S = 10.0
#: Refuse to stage an unbounded file for a dynamic run (same cap
#: as the static spine's inspect budget).
_MAX_DYNAMIC_BYTES = 256 * 1024 * 1024
#: Evidence result bodies are capped — the record is an index into
#: the truth, not a second copy of the trace.
_MAX_FOLD_EVENTS = 200
_MAX_FOLD_STOPS = 20

#: The limits carried on every Frida Evidence record.
LIMITATIONS_FRIDA = (
    "the trace covers only the functions the tracer intercepted "
    "(at most 200, allocator and thread-creation functions "
    "excluded) — unintercepted code paths are not observed",
    "the target ran as a copy in an isolated workdir with the "
    "network namespace dropped; under the unshare backend the "
    "filesystem is not jailed, so a target that reads outside "
    "the workdir can still read it — the run is network-isolated "
    "and time-bounded, not filesystem-jailed",
    "arguments and return values are the tracer's rendering of "
    "the call site (first four arguments only); they describe "
    "what was passed, not what the callee did with it",
    "a timed-out run records what was observed before the kill, "
    "not a complete trace",
)

#: Appended to a run's limitations when the sandbox ran unshare's
#: no-proc variant: the container refuses the proc mount, so the
#: target could see host process IDs and command lines in /proc.
#: It could not signal them (the PID namespace and the user
#: mapping hold), and the run's network isolation is unaffected.
#: The flag is carried on the run's sandbox summary, so the
#: limitation is a measured fact, not a posture.
PROC_LIMITATION = (
    "this container refuses the proc mount, so the run used "
    "unshare without a fresh /proc: the target could see host "
    "process IDs and command lines in /proc — it could not "
    "signal them (the PID namespace and the user mapping hold), "
    "and the run's network isolation is unaffected"
)

#: The limits carried on every GDB Evidence record.
LIMITATIONS_GDB = (
    "breakpoints cover main (or the ELF entry point when main "
    "has no symbol) — code after the last stop is not inspected",
    "registers, arguments and locals are snapshots at each stop, "
    "not a continuous record",
    "the target ran as a copy in an isolated workdir with the "
    "network namespace dropped; under the unshare backend the "
    "filesystem is not jailed (see the Frida limitations)",
    "a timed-out run records what was observed before the kill",
)

_DRIVER = Path(__file__).resolve().parent / "frida_driver.py"


class DynamicError(RuntimeError):
    """The dynamic rung failed; the message names where (engine,
    sandbox, target, parse) rather than swallowing the cause."""


# ── availability ───────────────────────────────────────────────────

def _engine_visible(path: str, backend: str) -> bool:
    """Can the engine at ``path`` be seen inside ``backend``?

    Under unshare the filesystem stays fully visible, so any path
    works. Under bwrap only /usr, /bin, /lib and /lib64 are
    bound in — a provisioned venv under ~/workspace is invisible
    there, while a system gdb or a system python is visible.
    Anything else (rlimit, none) sees nothing it can trust.
    """
    if not path:
        return False
    if backend == "unshare":
        return True
    if backend == "bwrap":
        try:
            resolved = str(Path(path).expanduser().resolve())
        except OSError:
            return False
        return any(resolved == base or resolved.startswith(base + "/")
                   for base in ("/usr", "/bin", "/lib", "/lib64"))
    return False


def sandbox_state(root: Any = None) -> dict[str, Any]:
    """The sandbox the dynamic rung stands on: the namespace
    backend that actually works on this device, probe-executed
    (see :mod:`nomorals.tools.shell`)."""
    backend = detect_backend()
    available = backend in ("bwrap", "unshare")
    confined = proc_confined(backend) if available else False
    if backend == "unshare":
        detail = ("unshare namespaces work on this device "
                  "(pid + mount + net dropped for dynamic runs)")
        if not confined:
            detail += (" — without a fresh /proc (this container "
                       "refuses the proc mount), so the target can "
                       "see host PIDs in /proc; it still cannot "
                       "signal them, and the network namespace "
                       "still isolates the run")
    elif backend == "bwrap":
        detail = ("bwrap works on this device; dynamic runs use "
                  "it only for engines visible under its binds "
                  "(/usr, /bin, /lib)")
    else:
        detail = ("no namespace sandbox works on this device "
                  f"(backend {backend!r}) — dynamic runs refuse "
                  "rather than run under rlimits alone")
    return {"engine": "dynamic-sandbox",
            "state": "ready" if available else "unavailable",
            "available": available, "backend": backend,
            "proc_confined": confined,
            "detail": detail}


def dynamic_state(root: Any = None) -> dict[str, Any]:
    """Per-engine availability for the dynamic rung, probe-true:
    an engine counts only when it is ready AND visible inside
    the sandbox backend that will run it."""
    fr = provision.frida_state(root)
    gd = provision.gdb_state(root)
    sb = sandbox_state(root)
    fr_ok = bool(fr["available"]
                 and _engine_visible(str(fr.get("path") or ""),
                                     sb["backend"]))
    fr_detail = fr["detail"]
    if fr_ok and not sb["proc_confined"]:
        # Frida resolves the target pid through /proc when it
        # attaches — with the host's /proc the namespace pid is
        # unresolvable, so attach cannot work here even though
        # the engine itself is ready. The rung is honestly
        # unavailable on this device, not silently broken.
        fr_ok = False
        fr_detail = (fr_detail + " — but this container refuses "
                     "the proc mount, so Frida cannot attach here "
                     "(no fresh /proc inside the sandbox)")
    gd_ok = bool(gd["available"]
                 and _engine_visible(str(gd.get("path") or ""),
                                     sb["backend"]))
    gd_detail = gd["detail"]
    if gd_ok and not sb["proc_confined"]:
        # GDB still runs without a fresh /proc, but honestly
        # reduced: starti + entry registers, no breakpoints.
        gd_detail = (gd_detail + " — on this device the sandbox "
                     "has no fresh /proc, so GDB sessions run "
                     "starti + entry registers, not breakpoints")
    if fr_ok:
        rung = "frida"
    elif gd_ok:
        rung = "gdb"
    else:
        rung = ""
    fr_row = {**fr, "available": fr_ok, "detail": fr_detail}
    gd_row = {**gd, "available": gd_ok, "detail": gd_detail}
    if fr["available"] and not fr_ok:
        # Provisioned but cannot run on this device — the state
        # names that, so probe-true readers see unavailable,
        # not a ready engine that silently fails at attach.
        fr_row["state"] = "unavailable"
    if gd["available"] and not gd_ok:
        gd_row["state"] = "unavailable"
    return {"frida": fr_row,
            "gdb": gd_row,
            "sandbox": sb,
            "available": bool(sb["available"] and rung),
            "rung": rung}


def _require_sandbox(root: Any = None) -> dict[str, Any]:
    """The sandbox gate: raise unless a namespace backend works."""
    state = sandbox_state(root)
    if not state["available"]:
        raise DynamicError(
            "dynamic runs need a namespace sandbox: " + state["detail"])
    return state


# ── staging: a copy in an isolated workdir, never the original ───

def _stage_run(target: Path, *, root: Any,
               rung: str) -> tuple[Path, Path, str, int]:
    """Copy the target into its per-run workdir.

    Returns (workdir, run_path, sha256, size). The copy is what
    gets executed; the original is never run in place, and the
    workdir is the sandbox cwd, so the target's writes land
    beside the copy, never beside the original.
    """
    if not target.is_file():
        raise DynamicError(f"no readable file at {target}")
    size = target.stat().st_size
    if size > _MAX_DYNAMIC_BYTES:
        raise DynamicError(
            f"{target.name} is {size} bytes — the dynamic rung "
            f"stages at most {_MAX_DYNAMIC_BYTES} bytes")
    digest = hashlib.sha256()
    with open(target, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    sha = digest.hexdigest()
    work = provision.re_root(root) / "dynamic-runs" / sha[:16]
    work.mkdir(parents=True, exist_ok=True)
    run_path = work / target.name
    shutil.copyfile(target, run_path)
    try:
        run_path.chmod(0o755)
    except OSError as exc:
        record_failure("reverse.dynamic.stage_chmod", exc)
    return work, run_path, sha, size


def _sandbox_result(work: Path, command: str, *,
                    timeout: float) -> dict[str, Any]:
    """Run one command in the workdir sandbox, network dropped."""
    return run_sandboxed(
        command, cwd=str(work), timeout=timeout + _SANDBOX_MARGIN_S,
        network=False, backend="auto",
        limits=SandboxLimits(cpu_seconds=int(timeout) + 60,
                             address_space_mb=8192),
        max_output=1_000_000)


def _sandbox_summary(res: dict[str, Any]) -> dict[str, Any]:
    return {"backend": res.get("backend"),
            "network": False,
            "network_enforced": bool(res.get("network_enforced")),
            "fs_confined": bool(res.get("fs_confined")),
            "proc_confined": bool(res.get("proc_confined")),
            "timed_out": bool(res.get("timed_out")),
            "seconds": res.get("seconds")}


# ── Frida ──────────────────────────────────────────────────────────

def run_frida(target: str | Path, *, root: Any = None,
              args: tuple[str, ...] | list[str] = (),
              script: str | Path | None = None,
              timeout: float = RUN_TIMEOUT_S) -> dict[str, Any]:
    """Trace one target with Frida → the trace dict.

    Raises :class:`DynamicError` when the rung is unavailable,
    the target cannot be staged, or the driver produced no
    report — a failed trace is never dressed up as an empty one.
    """
    state = provision.frida_state(root)
    if not state["available"]:
        raise DynamicError("frida rung unavailable: " + state["detail"])
    sb = _require_sandbox(root)
    if not sb["proc_confined"]:
        raise DynamicError(
            "frida rung unavailable: Frida resolves the target pid "
            "through /proc when it attaches, and this container "
            "refuses the proc mount, so there is no fresh /proc "
            "inside the sandbox — the rung needs a device whose "
            "sandbox mounts one")
    engine_python = state["path"]
    if not _engine_visible(engine_python, sb["backend"]):
        raise DynamicError(
            "frida rung unavailable: the provisioned interpreter "
            f"at {engine_python} is not visible inside the "
            "sandbox backend on this device")
    target = Path(target).expanduser()
    work, run_path, sha, size = _stage_run(target, root=root,
                                           rung="frida")
    driver = work / "frida_driver.py"
    shutil.copyfile(_DRIVER, driver)
    script_arg = ""
    if script:
        script_path = Path(script).expanduser()
        if not script_path.is_file():
            raise DynamicError(f"no readable script at {script}")
        staged = work / "trace.js"
        shutil.copyfile(script_path, staged)
        script_arg = " --script " + shlex.quote("trace.js")
    argv = [str(a) for a in (args or ())]
    command = (
        shlex.quote(engine_python) + " "
        + shlex.quote("frida_driver.py")
        + " --target " + shlex.quote(run_path.name)
        + " --args " + shlex.quote(json.dumps(argv))
        + " --timeout " + shlex.quote(str(timeout))
        + script_arg
        + " --out " + shlex.quote("frida-report.json"))
    started = time.monotonic()
    res = _sandbox_result(work, command, timeout=timeout)
    report_path = work / "frida-report.json"
    if not report_path.is_file():
        tail = (res.get("stderr") or "").strip().splitlines()[-3:]
        raise DynamicError(
            "frida driver produced no report (exit "
            f"{res.get('exit_code')}) — " + " · ".join(tail))
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DynamicError(
            f"frida driver report unreadable: {exc}") from exc
    return {
        "rung": "frida",
        "engine_version": state["version"],
        "target": str(target),
        "run_path": str(run_path),
        "sha256": sha,
        "size": size,
        "args": argv,
        "hooked": report.get("hooked") or [],
        "events": report.get("events") or [],
        "event_count": report.get("event_count", 0),
        "events_truncated": bool(report.get("events_truncated")),
        "errors": report.get("errors") or [],
        "exit_code": report.get("exit_code"),
        "timed_out": bool(report.get("timed_out"))
                     or bool(res.get("timed_out")),
        "duration_s": round(time.monotonic() - started, 3),
        "stdout": (res.get("stdout") or "")[:20000],
        "stderr": (res.get("stderr") or "")[:5000],
        "sandbox": _sandbox_summary(res),
    }


# ── GDB ────────────────────────────────────────────────────────────

_BREAK_RE = re.compile(r"^Breakpoint \d+, .*? in ([A-Za-z_][\w:.]*)")
_BREAK_AT_RE = re.compile(r"^Breakpoint \d+, (0x[0-9a-fA-F]+) in "
                          r"([A-Za-z_][\w:.]*)")
_BREAK_FN_RE = re.compile(r"^Breakpoint \d+, ([A-Za-z_][\w:.]*)")
_START_RE = re.compile(r"^0x[0-9a-fA-F]+ in ([A-Za-z_][\w:.]*)")
_FRAME_RE = re.compile(r"^#(\d+)\s+(?:(0x[0-9a-fA-F]+) in )?"
                       r"([A-Za-z_][\w:.]*)")
_REG_RE = re.compile(r"^([a-z0-9]+)\s+(0x[0-9a-fA-F]+)")
_VAR_RE = re.compile(r"^([A-Za-z_]\w*)\s*=\s*(.+)$")
_EXIT_RE = re.compile(r"exited with code (\d+)")
_EXITCODE_RE = re.compile(r"\$\d+\s*=\s*(-?\d+)")
_SIGNAL_RE = re.compile(r"received signal (SIG[A-Z]+)")


def _breakpoint_fn(line: str) -> str | None:
    """The function a ``Breakpoint N, ...`` line stopped in.

    GDB prints three shapes: ``Breakpoint 1, 0xADDR in main``,
    ``Breakpoint 1, main (argc=1, ...) at file.c:23``, and the
    plain ``Breakpoint 1, main`` — all three are a stop in main,
    and the parser must not miss the second shape (the one a
    debug build actually prints).
    """
    m = _BREAK_RE.search(line)
    if m:
        return m.group(1)
    m = _BREAK_AT_RE.search(line)
    if m:
        return m.group(2)
    m = _BREAK_FN_RE.search(line)
    if m:
        return m.group(1)
    # ``starti`` stops the inferior at the entry point and GDB prints
    # ``0x7f... in _start () from /lib64/ld-linux-x86-64.so.2``.
    m = _START_RE.search(line)
    if m:
        return m.group(1)
    return None


def _gdb_script(entry: int | None,
                args: list[str], *, proc_confined: bool) -> str:
    """The batch script: break at main (and the ELF entry point
    when it is known, so a stripped target still stops), snapshot
    the stop, run to exit, and print the exit code.

    Without a fresh ``/proc`` breakpoints cannot be inserted and
    GDB cannot read the inferior's memory at all, so the session is
    honestly reduced to ``starti`` plus the entry registers — a
    real stop, with the wall named in the fold, never a faked
    breakpoint.
    """
    if not proc_confined:
        return "\n".join((
            "set pagination off",
            "set confirm off",
            "starti",
            "info registers",
            "continue",
            "print $_exitcode",
            "quit",
        )) + "\n"
    lines = ["set pagination off", "set confirm off",
             "break main"]
    if entry is not None:
        lines.append(f"break *0x{entry:x}")
    arg_text = " ".join(shlex.quote(a) for a in args)
    lines.append(f"run {arg_text}".rstrip())
    lines.append("info args")
    lines.append("info locals")
    lines.append("info registers")
    lines.append("backtrace full")
    lines.append("continue")
    lines.append("info registers")
    lines.append("backtrace")
    lines.append("print $_exitcode")
    lines.append("quit")
    return "\n".join(lines) + "\n"


def _parse_gdb_log(text: str) -> dict[str, Any]:
    """Parse a gdb --batch log into stops / exit / signal.

    The log is a flat stream (batch mode echoes no commands), so
    parsing is positional: a ``Breakpoint`` line opens a stop,
    ``#N ... in fn`` lines and register lines belong to the open
    stop, and the exit / signal lines close the session. Anything
    unparsed rides in ``raw_tail`` (capped) — the parse never
    invents a frame it did not see.
    """
    stops: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    exit_code: int | None = None
    signal: str | None = None
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        sig = _SIGNAL_RE.search(line)
        if sig:
            signal = sig.group(1)
            continue
        m = _EXIT_RE.search(line)
        if m:
            exit_code = int(m.group(1))
            continue
        m = _EXITCODE_RE.search(line)
        if m and exit_code is None:
            exit_code = int(m.group(1))
            continue
        br = _breakpoint_fn(line)
        if br:
            current = {"function": br, "frames": [],
                       "registers": {}, "variables": {}, "raw": []}
            stops.append(current)
            continue
        if current is not None:
            fr = _FRAME_RE.match(line)
            if fr:
                current["frames"].append({
                    "level": int(fr.group(1)),
                    "address": fr.group(2) or "",
                    "function": fr.group(3)})
                continue
            rg = _REG_RE.match(line)
            if rg:
                current["registers"][rg.group(1)] = rg.group(2)
                continue
            vr = _VAR_RE.match(line)
            if vr:
                current["variables"][vr.group(1)] = vr.group(2)[:200]
                continue
            if len(current["raw"]) < 40:
                current["raw"].append(line[:300])
    return {"stops": stops, "exit_code": exit_code, "signal": signal}


def _gdb_run(work: Path, run_path: Path, gdb: str, *,
             entry: int | None, args: list[str],
             timeout: float, proc_confined: bool) -> dict[str, Any]:
    """One sandboxed GDB session: generate the command script,
    run ``gdb --batch`` on the staged copy, parse the log.

    ``entry`` is the ELF entry point, or None for a main-only
    session — the caller tries main first and only falls back to
    the entry break (a stripped target) when main never stopped.
    """
    script = _gdb_script(entry, args, proc_confined=proc_confined)
    script_path = work / "gdb-commands.txt"
    script_path.write_text(script, encoding="utf-8")
    command = (
        shlex.quote(gdb) + " --batch -x "
        + shlex.quote("gdb-commands.txt") + " --args "
        + shlex.quote(run_path.name)
        + "".join(" " + shlex.quote(a) for a in args))
    started = time.monotonic()
    res = _sandbox_result(work, command, timeout=timeout)
    log = (res.get("stdout") or "") + (res.get("stderr") or "")
    parsed = _parse_gdb_log(log)
    return {
        "stops": parsed["stops"][:_MAX_FOLD_STOPS],
        "stop_count": len(parsed["stops"]),
        "exit_code": parsed["exit_code"],
        "signal": parsed["signal"],
        "timed_out": bool(res.get("timed_out")),
        "duration_s": round(time.monotonic() - started, 3),
        "stdout": (res.get("stdout") or "")[:20000],
        "stderr": (res.get("stderr") or "")[:5000],
        "sandbox": _sandbox_summary(res),
    }


def run_gdb(target: str | Path, *, root: Any = None,
            args: tuple[str, ...] | list[str] = (),
            timeout: float = RUN_TIMEOUT_S) -> dict[str, Any]:
    """Run one target under GDB → the session dict.

    The session breaks at main first — a stripped target whose
    main has no symbol falls back to one second session breaking
    at the ELF entry point, and the result says so
    (``entry_fallback``). Raises :class:`DynamicError` when the
    rung is unavailable or the session produced no parseable log
    — a failed session is never dressed up as an empty one.
    """
    state = provision.gdb_state(root)
    if not state["available"]:
        raise DynamicError("gdb rung unavailable: " + state["detail"])
    _require_sandbox(root)
    sb = sandbox_state(root)
    if not _engine_visible(state["path"], sb["backend"]):
        raise DynamicError(
            "gdb rung unavailable: gdb at " + state["path"]
            + " is not visible inside the sandbox backend on "
              "this device")
    target = Path(target).expanduser()
    work, run_path, sha, size = _stage_run(target, root=root, rung="gdb")
    entry: int | None = None
    try:
        data = run_path.read_bytes()
        raw_entry = inspect_bytes(data).facts.get("entry_point")
        if isinstance(raw_entry, int):
            entry = raw_entry
        elif isinstance(raw_entry, str):
            try:
                entry = int(raw_entry, 16)
            except ValueError:
                entry = None
    except OSError as exc:
        record_failure("reverse.dynamic.gdb_entry", exc)
    argv = [str(a) for a in (args or ())]
    session = _gdb_run(work, run_path, state["path"], entry=None,
                       args=argv, timeout=timeout,
                       proc_confined=sb["proc_confined"])
    entry_fallback = False
    if (not session["stops"] and entry is not None
            and session["exit_code"] is not None
            and not session["timed_out"]):
        # main has no symbol and the target ran straight through
        # — one retry breaking at the entry point, not a guess.
        session = _gdb_run(work, run_path, state["path"], entry=entry,
                           args=argv, timeout=timeout,
                           proc_confined=sb["proc_confined"])
        entry_fallback = True
    if (not session["stops"] and session["exit_code"] is None
            and not session["signal"] and not session["timed_out"]):
        tail = (session["stderr"] or "").strip().splitlines()[-3:]
        raise DynamicError(
            "gdb session produced nothing parseable (no stops, "
            "no exit code) — " + " · ".join(tail))
    return {
        "rung": "gdb",
        "engine_version": state["version"],
        "target": str(target),
        "run_path": str(run_path),
        "sha256": sha,
        "size": size,
        "args": argv,
        "entry_fallback": entry_fallback,
        **session,
    }


# ── one entry point ────────────────────────────────────────────────

def run_dynamic(target: str | Path, *, root: Any = None,
                rung: str = "auto",
                args: tuple[str, ...] | list[str] = (),
                script: str | Path | None = None,
                timeout: float = RUN_TIMEOUT_S,
                provision_missing: bool = False) -> dict[str, Any]:
    """Run one target under the dynamic rung.

    ``rung`` is ``auto`` (Frida when ready, else GDB), ``frida``
    or ``gdb``. With ``provision_missing`` the drive provisions a
    missing engine first, then runs; without it a missing engine
    is a :class:`DynamicError` naming what is missing.
    """
    wanted = (rung or "auto").strip().lower()
    if wanted not in ("auto", "frida", "gdb"):
        raise DynamicError(
            f"unknown dynamic rung {rung!r} — auto|frida|gdb")
    target = Path(target).expanduser()
    if not target.is_file():
        raise DynamicError(f"no readable file at {target}")
    if wanted == "auto":
        state = dynamic_state(root)
        if not state["available"]:
            if provision_missing:
                state = _provision_dynamic(root)
            if not state["available"]:
                raise DynamicError(
                    "no dynamic rung can run on this device: "
                    "frida — " + state["frida"]["detail"]
                    + "; gdb — " + state["gdb"]["detail"]
                    + "; sandbox — " + state["sandbox"]["detail"])
        wanted = state["rung"]
    elif wanted == "frida":
        if not provision.frida_state(root)["available"]:
            if provision_missing:
                provision.provision_engine("frida", root=root)
            if not provision.frida_state(root)["available"]:
                raise DynamicError(
                    "frida rung unavailable: "
                    + provision.frida_state(root)["detail"])
        _require_sandbox(root)
    else:
        if not provision.gdb_state(root)["available"]:
            if provision_missing:
                provision.provision_engine("gdb", root=root)
            if not provision.gdb_state(root)["available"]:
                raise DynamicError(
                    "gdb rung unavailable: "
                    + provision.gdb_state(root)["detail"])
        _require_sandbox(root)
    if wanted == "frida":
        return run_frida(target, root=root, args=args, script=script,
                         timeout=timeout)
    return run_gdb(target, root=root, args=args, timeout=timeout)


def _provision_dynamic(root: Any) -> dict[str, Any]:
    """Provision what is missing for the dynamic rung, then
    re-probe. One engine's failure never masquerades as the
    other's — each row carries its own attempts."""
    for engine in ("frida", "gdb"):
        try:
            provision.provision_engine(engine, root=root)
        except provision.ProvisionError as exc:
            record_failure(f"reverse.dynamic.provision_{engine}", exc)
    return dynamic_state(root)


# ── evidence folding ───────────────────────────────────────────────

def fold_evidence(store: Any, *, subject_sha256: str,
                  subject_name: str, subject_size: int,
                  subject_path: str, result: dict[str, Any],
                  link_to: str = "") -> dict[str, Any]:
    """Fold one dynamic run into Evidence records on the store.

    Returns {"evidence_ids": [...], "unknowns": [...]} — the
    records are linked to the inspection's identify record when
    ``link_to`` names it, and each record carries the rung's
    limitations verbatim. A run that observed nothing opens a
    residual unknown instead of a dressed-up empty.
    """
    rung = result.get("rung") or "dynamic"
    version = str(result.get("engine_version") or "")
    common = {
        "subject_sha256": subject_sha256,
        "subject_name": subject_name,
        "subject_size": subject_size,
        "provider_id": PROVIDER_ID,
        "provider_name": PROVIDER_NAME,
        "provider_version": version,
        "locations": [{"type": "artifact-path",
                       "value": subject_path}],
    }
    links = [link_to] if link_to else []
    evidence_ids: list[str] = []
    unknowns: list[dict[str, Any]] = []
    limitations_frida = list(LIMITATIONS_FRIDA)
    limitations_gdb = list(LIMITATIONS_GDB)
    if not (result.get("sandbox") or {}).get("proc_confined", True):
        # the run's own sandbox summary says the proc mount was
        # refused — the limitation is a measured fact of this run
        limitations_frida.append(PROC_LIMITATION)
        limitations_gdb.append(PROC_LIMITATION)
    if rung == "frida":
        events = result.get("events") or []
        unique = sorted({str(e.get("function") or "")
                         for e in events if e.get("function")})
        rec = store.record(
            predicate_type="dynamic.frida.trace",
            operation="frida.trace",
            parameters={"engine": "frida",
                        "functions_intercepted":
                            len(result.get("hooked") or []),
                        "args": list(result.get("args") or [])},
            result={
                "call_count": result.get("event_count", len(events)),
                "unique_functions": unique,
                "calls": events[:_MAX_FOLD_EVENTS],
                "events_truncated":
                    bool(result.get("events_truncated")),
                "exit_code": result.get("exit_code"),
                "timed_out": bool(result.get("timed_out")),
                "duration_s": result.get("duration_s"),
                "sandbox": result.get("sandbox") or {},
            },
            confidence="observed", authority="controlled-replay",
            limitations=limitations_frida, links=links,
            **common)
        evidence_ids.append(rec.evidence_id)
        if not events:
            unknowns.append(store.open_unknown(
                f"no function calls were observed when "
                f"{subject_name} ran under Frida — the tracer "
                "intercepted no calls (no exports/imports found, "
                "or the target exited before any call); a trace "
                "with an explicit script, or a GDB session, "
                "closes this",
                subject_sha256=subject_sha256,
                raised_by=rec.evidence_id,
                supporting=[rec.evidence_id],
                required_authority="controlled-replay",
                required_confidence="observed").to_dict())
    else:
        stops = result.get("stops") or []
        rec = store.record(
            predicate_type="dynamic.gdb.session",
            operation="gdb.session",
            parameters={"engine": "gdb",
                        "args": list(result.get("args") or [])},
            result={
                "stop_count": result.get("stop_count", len(stops)),
                "stops": stops[:_MAX_FOLD_STOPS],
                "exit_code": result.get("exit_code"),
                "signal": result.get("signal"),
                "timed_out": bool(result.get("timed_out")),
                "duration_s": result.get("duration_s"),
                "sandbox": result.get("sandbox") or {},
            },
            confidence="observed", authority="controlled-replay",
            limitations=limitations_gdb, links=links,
            **common)
        evidence_ids.append(rec.evidence_id)
        if not stops:
            unknowns.append(store.open_unknown(
                f"no breakpoint was hit when {subject_name} ran "
                "under GDB — main has no symbol and the entry "
                "point break missed, or the target exited before "
                "any stop; a Frida trace, or a GDB session with "
                "an explicit address, closes this",
                subject_sha256=subject_sha256,
                raised_by=rec.evidence_id,
                supporting=[rec.evidence_id],
                required_authority="controlled-replay",
                required_confidence="observed").to_dict())
    return {"evidence_ids": evidence_ids, "unknowns": unknowns}
