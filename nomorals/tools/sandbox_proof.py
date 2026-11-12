"""Live proof that the sandbox backend enforces what it claims.

``run_sandboxed`` (tools/shell.py) picks the strongest backend that
actually works — bwrap, then unshare, then the rlimit floor — and its
results carry ``network_enforced`` / ``fs_confined`` flags naming which
guarantees the backend that ran could provide. This module PROVES
those claims instead of trusting them: hostile probes run through the
real ``run_sandboxed`` (never a mock of the sandbox itself), each with
an outside-sandbox control that must succeed, so a pass means
enforcement rather than a broken probe.

The proof is backend-relative on purpose. A namespace backend must
actually block egress, jail the filesystem and hide host PIDs; the
rlimit floor provides none of those, so on it the proof asserts the
flags honestly read False and the rlimit backstops (CPU ceiling,
file-size cap, env scrub) still hold. A flag that disagrees with its
observation is a LIE and fails the proof, even if the probe itself
"passed" — the verdict is about the correspondence, not the tier.

Run it anywhere with ``nm sandbox proof``. On a host where no
namespace backend works (this container: unshare runs its no-proc
variant — the ``/proc`` mount is refused, so host PIDs stay visible
and ``proc_confined`` reads False; EC2/Windows: no POSIX ``/bin/sh``
at all) the proof reports the tier as it is, and proves whatever
the tier below it does provide.
"""

from __future__ import annotations

import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .shell import (
    _PROBE_ARGV,
    _PROBE_ARGV_NOPROC,
    SandboxLimits,
    _run_probe_argv,
    detect_backend,
    proc_confined,
    run_sandboxed,
)

__all__ = ["render_proof", "run_proof", "verdict_for"]

# Wall-clock ceilings per probe. The CPU probe is expected to die at
# its 2s CPU ceiling; the ceilings only bound a misbehaving child.
_PROBE_TIMEOUT_S = 30.0
_NETWORK_TIMEOUT_S = 15.0
_SECRET_NAME = "NM_SANDBOX_PROOF_SECRET"
_TOKEN = "sandbox-proof-secret-token"


def _probe_result(
    name: str,
    status: str,
    evidence: str,
    *,
    observed: str = "",
    flag: bool | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "evidence": evidence,
        "observed": observed,
        "flag": flag,
    }


def backend_evidence(requested: str = "auto") -> dict[str, Any]:
    """Which namespace backends exist here, and why each probe passes/fails.

    Re-runs the exact probe argv ``run_sandboxed`` uses (imported from
    tools/shell, not duplicated), capturing stderr so the report can
    name the reason a tier is unavailable instead of just saying so.
    """
    candidates: dict[str, Any] = {}
    for name in ("bwrap", "unshare"):
        installed = shutil.which(name) is not None
        ok = False
        detail = "not installed"
        if installed:
            ok, raw = _run_probe_argv(_PROBE_ARGV[name])
            if ok:
                detail = "probe passed"
            else:
                err = raw
                # The tool prefixes its own name ("unshare: ...");
                # drop it so the reason reads once, not twice.
                prefix = name + ": "
                if err.startswith(prefix):
                    err = err[len(prefix):]
                detail = "probe failed: " + err
                if name == "unshare":
                    # the proc mount is the only failing piece on
                    # some containers — the no-proc variant is the
                    # same backend, one weaker tier
                    ok2, _raw2 = _run_probe_argv(
                        _PROBE_ARGV_NOPROC[name])
                    if ok2:
                        ok = True
                        detail += ("; no-proc variant passed — runs "
                                   "without a fresh /proc (host "
                                   "/proc stays visible)")
                    else:
                        detail += "; no-proc variant also failed"
        candidates[name] = {
            "installed": installed,
            "probe_ok": ok,
            "detail": detail,
        }
    selected = detect_backend(requested)
    if requested == "auto":
        expected = "rlimit"
        if candidates["bwrap"]["probe_ok"]:
            expected = "bwrap"
        elif candidates["unshare"]["probe_ok"]:
            expected = "unshare"
    elif requested in ("bwrap", "unshare"):
        expected = requested if candidates[requested]["probe_ok"] else "rlimit"
    else:
        expected = requested
    reasons = []
    for name in ("bwrap", "unshare"):
        reasons.append(name + ": " + candidates[name]["detail"])
    return {
        "candidates": candidates,
        "selected": selected,
        "expected": expected,
        "reason": "; ".join(reasons),
    }


def probe_backend_selection(requested: str = "auto") -> dict[str, Any]:
    """``detect_backend`` picks the strongest WORKING backend, not a wish."""
    evidence = backend_evidence(requested)
    selected = evidence["selected"]
    expected = evidence["expected"]
    if selected == expected:
        status = "pass"
        text = (
            "backend " + selected + " is the strongest working backend here"
            " (" + evidence["reason"] + ")"
        )
    else:
        status = "fail"
        text = (
            "backend " + selected + " selected but strongest working is "
            + expected + " (" + evidence["reason"] + ")"
        )
    return _probe_result("backend-selection", status, text)


def probe_network_egress(workdir: str | os.PathLike[str]) -> dict[str, Any]:
    """A connect attempt must be blocked iff ``network_enforced`` says so.

    The target is a listener in this process on 127.0.0.1: the control
    connect from the parent must succeed (proving the listener is
    real), and the child connect inside the sandbox is the probe.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            control = socket.create_connection(("127.0.0.1", port), timeout=3)
            control.close()
        except OSError as exc:
            return _probe_result(
                "network-egress", "fail",
                "control connect to the proof listener failed: " + str(exc)
                + " — the listener is not real, so the probe is invalid",
            )
        code = (
            "import socket\n"
            "s = socket.socket()\n"
            "s.settimeout(3)\n"
            "try:\n"
            f"    s.connect(('127.0.0.1', {port}))\n"
            "    print('CONNECTED')\n"
            "except OSError:\n"
            "    print('BLOCKED')\n"
        )
        try:
            result = run_sandboxed(
                "python3 -c " + shlex.quote(code),
                cwd=workdir,
                timeout=_NETWORK_TIMEOUT_S,
                network=False,
            )
        except Exception as exc:  # noqa: BLE001 - sandbox start failure is evidence
            return _probe_result(
                "network-egress", "fail",
                "sandbox would not start: " + str(exc),
            )
        stdout = result["stdout"]
        if "CONNECTED" in stdout:
            observed = "open"
        elif "BLOCKED" in stdout:
            observed = "blocked"
        else:
            return _probe_result(
                "network-egress", "fail",
                "probe vehicle failed inside the sandbox (no CONNECTED/"
                "BLOCKED in output; exit "
                + str(result["exit_code"]) + "; stderr: "
                + result["stderr"][:200] + ")",
            )
        flag = bool(result["network_enforced"])
        honest = flag == (observed == "blocked")
        if honest:
            if flag:
                text = (
                    "egress BLOCKED inside the sandbox; network_enforced=True"
                    " (control connect outside succeeded)"
                )
            else:
                text = (
                    "egress OPEN inside the sandbox; network_enforced=False"
                    " — honest: this tier has no net namespace (control"
                    " connect outside succeeded)"
                )
            return _probe_result(
                "network-egress", "pass", text,
                observed=observed, flag=flag,
            )
        return _probe_result(
            "network-egress", "fail",
            "LIE: network_enforced=" + str(flag) + " but egress observed "
            + observed + " inside the sandbox",
            observed=observed, flag=flag,
        )
    finally:
        listener.close()


def probe_fs_escape(
    workdir: str | os.PathLike[str],
    secret_path: str | os.PathLike[str],
    token: str,
) -> dict[str, Any]:
    """A file outside the workdir must be unreadable iff ``fs_confined``."""
    try:
        control = Path(secret_path).read_text(encoding="utf-8")
    except OSError as exc:
        return _probe_result(
            "fs-escape", "fail",
            "control read of the planted secret failed: " + str(exc)
            + " — the secret is not real, so the probe is invalid",
        )
    if token not in control:
        return _probe_result(
            "fs-escape", "fail",
            "control read of the planted secret did not contain the token"
            " — the secret is not real, so the probe is invalid",
        )
    try:
        result = run_sandboxed(
            "cat " + shlex.quote(str(secret_path)),
            cwd=workdir,
            timeout=_NETWORK_TIMEOUT_S,
            network=False,
        )
    except Exception as exc:  # noqa: BLE001 - sandbox start failure is evidence
        return _probe_result(
            "fs-escape", "fail", "sandbox would not start: " + str(exc),
        )
    observed = "blocked" if token not in result["stdout"] else "readable"
    flag = bool(result["fs_confined"])
    honest = flag == (observed == "blocked")
    if honest:
        if flag:
            text = (
                "secret outside the workdir UNREADABLE inside the sandbox;"
                " fs_confined=True (control read outside succeeded)"
            )
        else:
            text = (
                "secret outside the workdir READABLE inside the sandbox;"
                " fs_confined=False — honest: this tier does not jail the"
                " filesystem (control read outside succeeded)"
            )
        return _probe_result(
            "fs-escape", "pass", text, observed=observed, flag=flag,
        )
    return _probe_result(
        "fs-escape", "fail",
        "LIE: fs_confined=" + str(flag) + " but the secret outside the"
        " workdir was " + observed + " inside the sandbox",
        observed=observed, flag=flag,
    )


def probe_pid_namespace(workdir: str | os.PathLike[str]) -> dict[str, Any]:
    """The child must see only its own PIDs under a namespace backend
    with a fresh /proc — and the observation must match the flags
    that ran, not the tier wished for.

    Under bwrap / unshare-with-proc the child is PID 1 in a fresh
    PID namespace with a fresh /proc, so the parent's PID is
    invisible. Under unshare's no-proc variant (the container
    refuses the proc mount) the child is still PID 1 in its own
    PID namespace — it cannot signal host PIDs — but the host's
    /proc stays mounted, so the parent stays VISIBLE and
    ``proc_confined`` reads False. Under the rlimit floor the
    child shares the host PID namespace, so its PID is a host PID
    and the parent stays visible. Either way the observation must
    match the backend and the flag that ran.
    """
    parent_pid = os.getpid()
    command = (
        "echo PID=$$; ls /proc | grep -cE '^[0-9]+$'; "
        f"test -r /proc/{parent_pid}/cmdline && echo PARENT-VISIBLE || echo PARENT-HIDDEN"
    )
    try:
        result = run_sandboxed(
            command, cwd=workdir, timeout=_NETWORK_TIMEOUT_S, network=False,
        )
    except Exception as exc:  # noqa: BLE001 - sandbox start failure is evidence
        return _probe_result(
            "pid-namespace", "fail", "sandbox would not start: " + str(exc),
        )
    lines = result["stdout"].strip().splitlines()
    if len(lines) < 3 or not lines[0].startswith("PID="):
        return _probe_result(
            "pid-namespace", "fail",
            "probe vehicle failed inside the sandbox (unexpected output: "
            + result["stdout"][:200] + ")",
        )
    try:
        child_pid = int(lines[0][4:])
        proc_count = int(lines[1].strip())
    except ValueError:
        return _probe_result(
            "pid-namespace", "fail",
            "probe vehicle failed inside the sandbox (unparsable output: "
            + result["stdout"][:200] + ")",
        )
    parent_visible = lines[2].strip() == "PARENT-VISIBLE"
    backend = result["backend"]
    confined = bool(result.get("proc_confined"))
    evidence = (
        "child PID=" + str(child_pid) + ", /proc shows " + str(proc_count)
        + " processes, parent PID " + str(parent_pid) + " is "
        + ("VISIBLE" if parent_visible else "HIDDEN")
        + " inside (backend " + backend
        + ", proc_confined=" + str(confined) + ")"
    )
    pid_ok = (child_pid == 1) == (backend in ("bwrap", "unshare"))
    proc_ok = (not parent_visible) == confined
    if pid_ok and proc_ok:
        return _probe_result("pid-namespace", "pass", evidence)
    return _probe_result(
        "pid-namespace", "fail",
        "isolation does not match the flags that ran: " + evidence,
    )


def probe_cpu_ceiling(workdir: str | os.PathLike[str]) -> dict[str, Any]:
    """A CPU-spin child must die at its RLIMIT_CPU ceiling, on every backend."""
    limits = SandboxLimits(
        cpu_seconds=2, address_space_mb=2048, file_size_mb=256,
        max_processes=64,
    )
    try:
        control = run_sandboxed(
            "sleep 1", cwd=workdir, timeout=_PROBE_TIMEOUT_S,
            limits=limits, network=False,
        )
    except Exception as exc:  # noqa: BLE001 - sandbox start failure is evidence
        return _probe_result(
            "cpu-ceiling", "fail", "sandbox would not start: " + str(exc),
        )
    if control["exit_code"] != 0:
        return _probe_result(
            "cpu-ceiling", "fail",
            "control `sleep 1` under the same limits failed (exit "
            + str(control["exit_code"]) + ") — the limits themselves are"
            " broken, so the probe is invalid",
        )
    try:
        result = run_sandboxed(
            "while :; do :; done", cwd=workdir,
            timeout=_PROBE_TIMEOUT_S, limits=limits, network=False,
        )
    except Exception as exc:  # noqa: BLE001 - sandbox start failure is evidence
        return _probe_result(
            "cpu-ceiling", "fail", "sandbox would not start: " + str(exc),
        )
    if result["timed_out"]:
        return _probe_result(
            "cpu-ceiling", "fail",
            "spinner survived past the wall-clock timeout — the CPU"
            " ceiling did not kill it (seconds: "
            + str(result["seconds"]) + ")",
        )
    if result["exit_code"] == 0:
        return _probe_result(
            "cpu-ceiling", "fail",
            "spinner exited 0 — the CPU ceiling did not kill it",
        )
    if result["seconds"] > 20:
        return _probe_result(
            "cpu-ceiling", "fail",
            "spinner died but only after " + str(result["seconds"])
            + "s — past the ceiling plus its hard-limit grace",
        )
    return _probe_result(
        "cpu-ceiling", "pass",
        "spinner killed by the CPU ceiling (exit "
        + str(result["exit_code"]) + " after " + str(result["seconds"])
        + "s; control `sleep 1` under the same limits completed)",
    )


def probe_file_size_cap(workdir: str | os.PathLike[str]) -> dict[str, Any]:
    """A write past RLIMIT_FSIZE must fail and stop at the cap."""
    workdir_path = Path(workdir)
    capped = workdir_path / "big-capped.bin"
    control_file = workdir_path / "big-control.bin"
    limits = SandboxLimits(
        cpu_seconds=30, address_space_mb=2048, file_size_mb=1,
        max_processes=64,
    )
    try:
        control = run_sandboxed(
            "head -c 3000000 /dev/zero > big-control.bin",
            cwd=workdir, timeout=_PROBE_TIMEOUT_S, network=False,
        )
    except Exception as exc:  # noqa: BLE001 - sandbox start failure is evidence
        return _probe_result(
            "file-size-cap", "fail", "sandbox would not start: " + str(exc),
        )
    if control["exit_code"] != 0 or not control_file.exists():
        return _probe_result(
            "file-size-cap", "fail",
            "control 3MB write under default limits failed (exit "
            + str(control["exit_code"]) + ") — the write vehicle is"
            " broken, so the probe is invalid",
        )
    if control_file.stat().st_size != 3000000:
        return _probe_result(
            "file-size-cap", "fail",
            "control 3MB write wrote "
            + str(control_file.stat().st_size) + " bytes — the write"
            " vehicle is broken, so the probe is invalid",
        )
    try:
        result = run_sandboxed(
            "head -c 3000000 /dev/zero > big-capped.bin",
            cwd=workdir, timeout=_PROBE_TIMEOUT_S, limits=limits,
            network=False,
        )
    except Exception as exc:  # noqa: BLE001 - sandbox start failure is evidence
        return _probe_result(
            "file-size-cap", "fail", "sandbox would not start: " + str(exc),
        )
    if not capped.exists():
        return _probe_result(
            "file-size-cap", "fail",
            "capped write left no file at all — the write vehicle is"
            " broken, so the probe is invalid",
        )
    size = capped.stat().st_size
    if result["exit_code"] == 0:
        return _probe_result(
            "file-size-cap", "fail",
            "3MB write under a 1MB cap exited 0 (wrote " + str(size)
            + " bytes) — the cap did not bite",
        )
    if size > 1048576:
        return _probe_result(
            "file-size-cap", "fail",
            "3MB write under a 1MB cap wrote " + str(size) + " bytes —"
            " past the cap",
        )
    return _probe_result(
        "file-size-cap", "pass",
        "3MB write under a 1MB cap stopped at " + str(size) + " bytes"
        " (exit " + str(result["exit_code"]) + "; control 3MB write"
        " under default limits completed)",
    )


def probe_env_scrub(workdir: str | os.PathLike[str]) -> dict[str, Any]:
    """A secret in the parent's environment must not reach the child."""
    token = _TOKEN + "-" + str(time.time_ns())
    old = os.environ.get(_SECRET_NAME)
    os.environ[_SECRET_NAME] = token
    try:
        control = subprocess.run(
            ["/bin/sh", "-c", 'echo "secret=[$' + _SECRET_NAME + ']"'],
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if token not in control.stdout:
            return _probe_result(
                "env-scrub", "fail",
                "control echo outside the sandbox did not show the"
                " secret — the variable was not really set, so the"
                " probe is invalid",
            )
        try:
            result = run_sandboxed(
                'echo "secret=[$' + _SECRET_NAME + ']"',
                cwd=workdir, timeout=_NETWORK_TIMEOUT_S, network=False,
            )
        except Exception as exc:  # noqa: BLE001 - sandbox start failure is evidence
            return _probe_result(
                "env-scrub", "fail", "sandbox would not start: " + str(exc),
            )
        if token in result["stdout"]:
            return _probe_result(
                "env-scrub", "fail",
                "parent environment secret leaked into the sandboxed"
                " child (the B3 hole: full parent env in a 'sandboxed'"
                " run)",
            )
        return _probe_result(
            "env-scrub", "pass",
            "parent environment secret absent inside the sandbox"
            " (control echo outside showed it)",
        )
    finally:
        if old is None:
            os.environ.pop(_SECRET_NAME, None)
        else:
            os.environ[_SECRET_NAME] = old


def verdict_for(probes: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """The verdict over finished probes: pass iff none failed or lied."""
    failures = [
        p["name"] + ": " + p["evidence"]
        for p in probes
        if p["status"] == "fail"
    ]
    return ("pass" if not failures else "fail", failures)


def run_proof(
    *,
    workdir: str | os.PathLike[str] | None = None,
    backend: str = "auto",
) -> dict[str, Any]:
    """Run the whole hostile-probe battery against the real sandbox.

    Returns a report dict: backend + the reason it was selected, the
    claimed flags, one entry per probe, and the verdict. A report is
    data, not a claim — the caller renders it with ``render_proof``.
    """
    if os.name != "posix":
        return {
            "backend": "none",
            "backend_reason": (
                "sandboxed probes need a POSIX /bin/sh host; this"
                " platform (" + sys.platform + ") cannot run them"
            ),
            "network_enforced": False,
            "fs_confined": False,
            "proc_confined": False,
            "probes": [],
            "verdict": "not-applicable",
            "failures": ["platform cannot run sandboxed probes"],
        }
    try:
        evidence = backend_evidence(backend)
        selected = evidence["selected"]
    except Exception as exc:  # noqa: BLE001 - explicit-backend refusal is evidence
        return {
            "backend": "unknown",
            "backend_reason": "backend selection failed: " + str(exc),
            "network_enforced": False,
            "fs_confined": False,
            "proc_confined": False,
            "probes": [],
            "verdict": "fail",
            "failures": ["backend selection failed: " + str(exc)],
        }
    owned_tmp: tempfile.TemporaryDirectory | None = None
    if workdir is None:
        owned_tmp = tempfile.TemporaryDirectory(prefix="sandbox-proof-")
        base = Path(owned_tmp.name)
    else:
        base = Path(workdir)
        base.mkdir(parents=True, exist_ok=True)
    secret_dir = base / "outside"
    secret_dir.mkdir(exist_ok=True)
    secret_path = secret_dir / "secret.txt"
    secret_path.write_text(_TOKEN + "\n", encoding="utf-8")
    probes: list[dict[str, Any]] = []
    try:
        probes.append(probe_backend_selection(backend))
        probes.append(probe_network_egress(base))
        probes.append(probe_fs_escape(base, secret_path, _TOKEN))
        probes.append(probe_pid_namespace(base))
        probes.append(probe_cpu_ceiling(base))
        probes.append(probe_file_size_cap(base))
        probes.append(probe_env_scrub(base))
    finally:
        if owned_tmp is not None:
            owned_tmp.cleanup()
    verdict, failures = verdict_for(probes)
    return {
        "backend": selected,
        "backend_reason": evidence["reason"],
        "network_enforced": selected in ("bwrap", "unshare"),
        "fs_confined": selected == "bwrap",
        "proc_confined": proc_confined(selected),
        "probes": probes,
        "verdict": verdict,
        "failures": failures,
    }


def render_proof(report: dict[str, Any]) -> str:
    """One line per probe plus a final verdict line naming the backend."""
    lines = [
        "sandbox proof — backend: " + str(report["backend"]),
        "  reason: " + str(report["backend_reason"]),
    ]
    for probe in report["probes"]:
        status = str(probe["status"]).upper()
        lines.append("  " + status + " " + probe["name"] + "  " + probe["evidence"])
    verdict = str(report["verdict"]).upper()
    lines.append(
        "VERDICT: " + verdict + " — backend " + str(report["backend"])
        + ", network_enforced=" + str(report["network_enforced"])
        + ", fs_confined=" + str(report["fs_confined"])
        + ", proc_confined=" + str(report.get("proc_confined"))
    )
    return "\n".join(lines)
