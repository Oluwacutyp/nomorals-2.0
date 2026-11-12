"""Frida trace driver — run by the provisioned Frida venv python.

Stdlib + frida only: it must run inside the sandbox workdir under
the venv interpreter, with no repo on the path. It spawns the
target suspended, attaches, intercepts the main module's exported
and imported functions with the default tracer (or a caller
supplied JS file), resumes, and reports JSON: the intercepted
calls, the target's stdout/stderr, its exit code, and whether the
run timed out.

The driver never imports nomorals and never touches the network:
the trace is local process instrumentation, and the sandbox that
invokes it has already dropped the network namespace.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import warnings

# The driver forks once to start the target suspended (see
# _start_suspended); the child execs immediately, so the
# multi-threaded-fork warning is noise here, not a hazard.
warnings.filterwarnings(
    "ignore", message=".*multi-threaded, use of fork.*",
    category=DeprecationWarning)

#: Functions never intercepted: the allocator family recurses
#: through the interceptor itself, and thread creation inside the
#: agent deadlocks the session. Excluded by name, not by hope.
_EXCLUDED = frozenset({
    "malloc", "free", "calloc", "realloc", "reallocarray",
    "memalign", "posix_memalign", "aligned_alloc", "valloc",
    "mmap", "munmap", "brk", "sbrk", "__libc_malloc",
    "__libc_free", "__libc_calloc", "__libc_realloc",
    "pthread_create", "pthread_exit",
})

#: Bounded output: a chatty target must not grow the report without
#: limit — the trace is evidence, not a packet capture.
_MAX_EVENTS = 5000
_MAX_INTERCEPT = 200

_DEFAULT_JS = r"""
"use strict";
var MAX_INTERCEPT = 200;
var EXCLUDED = {
  "malloc": 1, "free": 1, "calloc": 1, "realloc": 1,
  "reallocarray": 1, "memalign": 1, "posix_memalign": 1,
  "aligned_alloc": 1, "valloc": 1, "mmap": 1, "munmap": 1,
  "brk": 1, "sbrk": 1, "__libc_malloc": 1, "__libc_free": 1,
  "__libc_calloc": 1, "__libc_realloc": 1,
  "pthread_create": 1, "pthread_exit": 1
};
function argStr(a) {
  try { return a.toString(); } catch (e) { return "?"; }
}
function hook(name, addr) {
  try {
    Interceptor.attach(addr, {
      onEnter: function (args) {
        var vals = [];
        for (var i = 0; i < 4; i++) {
          try { vals.push(argStr(args[i])); } catch (e) { vals.push("?"); }
        }
        send({kind: "enter", function: name, args: vals});
      },
      onLeave: function (retval) {
        send({kind: "return", function: name, retval: argStr(retval)});
      }
    });
    return true;
  } catch (e) { return false; }
}
var hooked = [];
try {
  var mod = Process.mainModule;
  var seen = {};
  // the target's own functions live in the symbol table
  // (.symtab on a debug build — a PIE executable does not
  // export them in .dynsym), so enumerateSymbols is the
  // source; imports cover the libc calls it makes.
  var syms = mod.enumerateSymbols();
  for (var i = 0; i < syms.length; i++) {
    if (hooked.length >= MAX_INTERCEPT) break;
    var s = syms[i];
    var sname = s.name.split("@")[0];
    if (s.type !== "function" || seen[sname] || EXCLUDED[sname]) continue;
    seen[sname] = true;
    if (hook(sname, s.address)) hooked.push(sname);
  }
  var imps = mod.enumerateImports();
  for (var j = 0; j < imps.length; j++) {
    if (hooked.length >= MAX_INTERCEPT) break;
    var imp = imps[j];
    var iname = imp.name.split("@")[0];
    if (imp.type !== "function" || seen[iname] || EXCLUDED[iname]) continue;
    seen[iname] = true;
    if (hook(iname, imp.address)) hooked.push(iname);
  }
} catch (e) {
  send({kind: "error", message: "enumerate failed: " + e});
}
send({kind: "hooked", functions: hooked});
"""


def _start_suspended(argv: list[str]) -> int:
    """Start the target suspended, as a real child of this driver.

    Frida's ``device.spawn`` double-forks, so the spawned target
    is not this process's child and ``waitpid`` cannot reap it
    (``ChildProcessError``) — and a shell that stops itself before
    ``exec`` leaves the process as the SHELL at attach time, so
    the tracer hooks the shell and loses the agent on exec. The
    suspend that works is PTRACE_TRACEME: fork, the child marks
    itself traced and execs the target (it stops with SIGTRAP
    after exec, before any target code runs, as the TARGET image),
    the parent detaches it with a SIGSTOP pending (so it stays
    stopped with no tracer), and Frida attaches to that stopped
    target. The child is a real child, so ``waitpid`` reaps the
    true exit code.
    """
    import ctypes

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    PTRACE_TRACEME = 0
    PTRACE_DETACH = 17
    pid = os.fork()
    if pid == 0:
        # child: traced, then exec the target (stops after exec)
        try:
            libc.ptrace(PTRACE_TRACEME, 0, 0, 0)
        except Exception:  # noqa: BLE001 - ptrace is best-effort
            pass
        try:
            os.execv(argv[0], argv)
        except OSError:
            pass
        os._exit(127)
    # parent: wait for the SIGTRAP stop after exec
    deadline = time.monotonic() + 10.0
    while True:
        try:
            done, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
        except ChildProcessError:
            raise RuntimeError(
                "target vanished before it could be traced") from None
        if done == pid:
            if os.WIFSTOPPED(status):
                break
            raise RuntimeError(
                "target exited before it could be traced "
                f"(status {status})")
        if time.monotonic() >= deadline:
            raise RuntimeError("target did not stop after exec")
        time.sleep(0.02)
    # detach with SIGSTOP pending: stopped, tracer-free, so
    # Frida can attach to it
    libc.ptrace(PTRACE_DETACH, pid, 0, signal.SIGSTOP)
    return pid


def _wait_exit(pid: int, timeout: float) -> tuple[bool, int | None]:
    """Wait for pid up to timeout seconds → (timed_out, exit_code).

    The only waitpid on this pid (besides the WUNTRACED stop
    wait above, which never reaps) — so the exit status lands
    here, whole.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            done, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return False, None
        if done == pid:
            try:
                code = os.waitstatus_to_exitcode(status)
            except AttributeError:  # pragma: no cover - old pythons
                code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
            return False, code
        if time.monotonic() >= deadline:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                done, status = os.waitpid(pid, 0)
            except ChildProcessError:
                return True, None
            try:
                code = os.waitstatus_to_exitcode(status)
            except AttributeError:  # pragma: no cover - old pythons
                code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
            return True, code
        time.sleep(0.05)


def _kill(pid: int) -> None:
    """SIGKILL the target and reap it (best effort)."""
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="frida trace driver")
    parser.add_argument("--target", required=True)
    parser.add_argument("--args", default="[]",
                        help="JSON list of target argv (after the path)")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--script", default="",
                        help="JS file to load instead of the default tracer")
    parser.add_argument("--out", required=True)
    opts = parser.parse_args(argv)
    try:
        import frida
    except ImportError as exc:
        print(f"frida is not importable in this interpreter: {exc}",
              file=sys.stderr)
        return 2
    try:
        target_args = json.loads(opts.args)
        if not isinstance(target_args, list):
            raise ValueError("args must be a JSON list")
    except ValueError as exc:
        print(f"bad --args: {exc}", file=sys.stderr)
        return 2
    if opts.script:
        try:
            with open(opts.script, encoding="utf-8") as fh:
                js = fh.read()
        except OSError as exc:
            print(f"cannot read --script: {exc}", file=sys.stderr)
            return 2
    else:
        js = _DEFAULT_JS
    target = os.path.abspath(opts.target)
    if not os.path.isfile(target):
        print(f"no such target: {target}", file=sys.stderr)
        return 2
    events: list[dict] = []
    hooked: list[str] = []
    errors: list[str] = []

    def on_message(message, _data):
        payload = message.get("payload")
        if message["type"] == "error":
            errors.append(str(message.get("description") or message))
            return
        if not isinstance(payload, dict):
            return
        kind = payload.get("kind")
        if kind == "hooked":
            hooked.extend(payload.get("functions") or [])
        elif kind == "error":
            errors.append(str(payload.get("message") or payload))
        elif kind in ("enter", "return"):
            if len(events) < _MAX_EVENTS:
                events.append(payload)

    started = time.monotonic()
    device = frida.get_local_device()
    try:
        pid = _start_suspended([target, *[str(a) for a in target_args]])
    except RuntimeError as exc:
        print(f"could not start target suspended: {exc}",
              file=sys.stderr)
        return 2
    try:
        session = device.attach(pid)
    except Exception as exc:  # noqa: BLE001 - attach is the failure
        # Attach resolves the target pid through /proc — on a
        # device whose sandbox has no fresh /proc this raises
        # ProcessNotFoundError for the namespace pid. Report it
        # plainly and do not leave the target stopped.
        print(f"frida attach failed: {exc}", file=sys.stderr)
        _kill(pid)
        return 3
    timed_out = False
    exit_code: int | None = None
    try:
        script = session.create_script(js)
        script.on("message", on_message)
        script.load()
        # Frida's own resume — not a raw SIGCONT, which cannot
        # clear the tracing stop the attach leaves behind. This
        # is what lets the target run after the tracer lands.
        device.resume(pid)
        # The target's stdout is inherited from this driver, whose
        # own stdout is captured by the sandbox — so the target's
        # prints ride along in the sandbox result, not here.
        timed_out, exit_code = _wait_exit(pid, opts.timeout)
    except Exception as exc:  # noqa: BLE001 - load/resume failure
        print(f"frida trace failed: {exc}", file=sys.stderr)
        _kill(pid)
        return 3
    finally:
        try:
            session.detach()
        except Exception:  # noqa: BLE001 - detach is best-effort
            pass
    report = {
        "rung": "frida",
        "target": target,
        "args": [str(a) for a in target_args],
        "hooked": hooked,
        "events": events,
        "event_count": len(events),
        "events_truncated": len(events) >= _MAX_EVENTS,
        "errors": errors,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_s": round(time.monotonic() - started, 3),
    }
    with open(opts.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
