"""Local GGUF model manager: find, download, and *run* llama.cpp on this machine.

This is the piece that makes "downloaded an 8B GGUF" actually work — especially
on Termux, where the failure modes are unglamorous and silent:

* ``llama-server`` is not on ``PATH`` (Termux installs it under ``$PREFIX/bin``,
  and builds vary between ``llama-server`` and the old ``llama-cli --server``).
* The model path points at a directory, a partial download, or the HF *repo*
  instead of a ``.gguf`` file.
* The port is already taken by a stale server from a crashed session.
* 8B q4 on a phone needs a few minutes to load — callers that wait 10 seconds
  conclude "it failed" and give up.

So this module:

* locates the binary across PATH, ``$PREFIX/bin``, and common build dirs;
* resolves a model *name* to a real ``.gguf`` file (local search first, then a
  token-authed resumable HF download of a GGUF from the catalog);
* spawns ``llama-server`` with mobile-sane defaults (single slot, sized ctx,
  physical-core threads, no flash-attn on ARM);
* waits a **configurable, generous** boot window, polling ``/health`` and
  ``/v1/models``;
* on failure, returns *actionable* diagnostics instead of a stack trace.

The manager owns the subprocess: ``stop()`` kills it cleanly, and a stale
process on the same port is detected and reported.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.http import HttpClient
from ..core.logging_setup import get_logger

__all__ = [
    "GGUFDiagnosis",
    "GGUFServerManager",
    "find_llama_binary",
    "find_gguf",
    "health_status",
    "port_in_use",
    "resolve_gguf_repo",
]

_log = get_logger(__name__)

#: Search order for the server binary. Termux first, because that's the phone.
_BINARY_CANDIDATES = (
    "llama-server",
    "llama-cli",
)


@dataclass
class GGUFDiagnosis:
    """A readable, actionable verdict about local model loading."""

    ok: bool
    binary: str = ""
    model_path: str = ""
    url: str = ""
    pid: int = 0
    problems: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "binary": self.binary,
            "model_path": self.model_path,
            "url": self.url,
            "pid": self.pid,
            "problems": self.problems,
            "hints": self.hints,
        }


def find_llama_binary() -> str:
    """Locate a usable llama.cpp server binary; '' when none is found."""
    seen: set[str] = set()
    for name in _BINARY_CANDIDATES:
        found = shutil.which(name)
        if found and found not in seen:
            return found
        # Termux and manual builds.
        prefix = os.environ.get("PREFIX")
        if prefix:
            candidate = Path(prefix) / "bin" / name
            if candidate.is_file():
                return str(candidate)
        for base in (
            str(Path.home() / "llama.cpp" / "build" / "bin"),  # standard cmake output
            str(Path.home() / "llama.cpp" / "build"),  # older layouts
            "/usr/local/bin",
        ):
            candidate = Path(base) / name
            if candidate.is_file():
                return str(candidate)
    return ""


def find_gguf(name: str, cache_dir: str | os.PathLike[str]) -> str:
    """Resolve a model *name* to a real ``.gguf`` file on disk.

    Accepts, in order: an absolute/relative path that exists; a basename or
    substring match inside the cache dir (preferring larger files — the full
    quant, not a shard); a single ``.gguf`` in the cache dir. Returns '' when
    nothing matches.
    """
    name = (name or "").strip()
    if not name:
        return ""
    direct = Path(os.path.expanduser(name))
    if direct.is_file():
        return str(direct)
    if direct.is_dir():
        # A directory was given: look inside it (a common mistake: pointing at
        # the HF repo folder instead of the weight file).
        inner = sorted(direct.glob("*.gguf"), key=lambda p: -p.stat().st_size)
        if inner:
            return str(inner[0])
        return ""
    cache = Path(os.path.expanduser(cache_dir))
    if not cache.exists():
        return ""
    matches = sorted(cache.rglob("*.gguf"), key=lambda p: -p.stat().st_size)
    if not matches:
        return ""
    lowered = name.lower()
    for path in matches:
        if lowered in path.name.lower():
            return str(path)
    # No name match: if there's exactly one GGUF, that's obviously it.
    if len(matches) == 1:
        return str(matches[0])
    return ""


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def health_status(host: str, port: int, timeout: float = 3.0) -> str:
    """The local model server's own word on what it is doing.

    llama-server /health answers one of:
      ``{"status":"ok"}``       loaded and free
      ``{"status":"busy"}``     loaded, serving a request
      ``{"status":"loading"}``  THE PHONE STATE — alive, burning CPU,
                                reading the 4.7 GB model into RAM.  Not
                                ready yet, but KILLING IT WASTES THE
                                WHOLE LOAD.  "Port open" and "ready" are
                                NOT the same thing — that is what kept a
                                phone hot and silent for an hour.
      ``{"status":"error"}``    the server itself hit a fault
    Returns those words, or ``"dead"`` when nothing answers at all.
    """
    try:
        client = HttpClient(timeout=timeout)
        resp = client.get(f"http://{host}:{port}/health", timeout=timeout)
        if not resp.ok:
            return "dead"
        body = (resp.text or "").strip()
        try:
            status = json.loads(body).get("status")
        except Exception:  # noqa: BLE001 - non-JSON 200 (test doubles) = alive
            return "ok" if body else "dead"
        return str(status) if status else "ok"
    except Exception:  # noqa: BLE001 - no answer of any kind
        return "dead"


def find_llama_server_pid(port: int, proc_dir: str = "/proc") -> int | None:
    """The pid of a llama-server holding ``--port <port>`` (or None).

    Scans /proc command lines — no root, no lsof, works on Termux.  Only a
    process whose own command line names our port is ever matched, so a
    foreign process squatting on the port is never touched.
    """
    try:
        pids = [int(p) for p in os.listdir(proc_dir) if p.isdigit()]
    except OSError:
        return None
    needles = (f"--port {port}", f"--port={port}", f":{port}")
    for pid in sorted(pids):
        try:
            with open(f"{proc_dir}/{pid}/cmdline", "rb") as fh:
                cmdline = fh.read().replace(b"\0", b" ").decode(
                    "utf-8", "replace")
        except OSError:
            continue
        if ("llama-server" in cmdline or "llama-cli" in cmdline) \
                and any(n in cmdline for n in needles):
            return pid
    return None


def _is_zombie(pid: int) -> bool:
    """A zombie is dead — it just has not been reaped by its parent yet,
    and os.kill(pid, 0) would still 'succeed' against it."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            return fh.read().rsplit(b")", 1)[-1].split()[:1] == [b"Z"]
    except OSError:
        return False


def kill_pid(pid: int, timeout: float = 10.0) -> bool:
    """SIGTERM, wait, SIGKILL.  True when the process is gone (zombie counts)."""
    import signal

    def alive() -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return not _is_zombie(pid)

    if not alive():
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return not alive()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not alive():
            return True
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return not alive()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not alive():
            return True
        time.sleep(0.2)
    return not alive()


def gguf_check(path: str | os.PathLike[str]) -> tuple[bool, str]:
    """Integrity sniff for a GGUF file: magic + a plausible size.

    A 310 MB 'downloaded' 4.7 GB model passed every check that only looked
    at the filename — the file on disk never had to prove anything.  This is
    the cheapest proof: the GGUF magic at offset 0 and a non-trivial size.
    """
    p = Path(path)
    if not p.is_file():
        return False, "file not found"
    size = p.stat().st_size
    if size < 10 * 1024 * 1024:
        return False, f"only {size // (1024 * 1024)} MB — far too small for a model"
    with p.open("rb") as fh:
        magic = fh.read(4)
    if magic != b"GGUF":
        return False, (f"bad magic {magic!r} (expected b'GGUF') — "
                       "the file is not a valid GGUF (partial or wrong download)")
    return True, f"{size // (1024 * 1024)} MB, GGUF magic OK"



def _list_failure_hints(error_text: str) -> list[str]:
    """Turn a list-files failure into actionable hints.

    A 401/403 on an *anonymous* request usually means the repo is private,
    gated, or deleted — it does NOT mean the user needs a token (public repos
    list fine without one). A 404 means the id is gone or renamed. Only an
    unclassified failure points at network/token generally.
    """
    text = (error_text or "").lower()
    if "401" in text or "403" in text or "unauthorized" in text:
        return [
            "repo is not anonymously readable — it is private, gated, or was deleted",
            "public repos need NO HF_TOKEN: if the id was deleted/renamed, pass an "
            "exact one (nm models --fetch 'org/repo') after checking huggingface.co",
            "if it is genuinely a gated model, set HF_TOKEN in ~/.nomorals/.env and retry",
        ]
    if "404" in text or "not found" in text:
        return [
            "no such repo on Hugging Face — the id was renamed or deleted; verify it "
            "on huggingface.co or pass an exact 'org/repo'",
        ]
    return ["check network / HF_TOKEN; repo may be gated"]


def resolve_gguf_repo(query: str) -> str:
    """Resolve a model *name* to a Hugging Face repo that serves GGUF files.

    ``"org/repo"`` passes through unchanged. A family name (``dolphin-8b``,
    ``dolphin``, ``mistral``) is matched against the catalog: entries marked
    ``kind="gguf"`` win, with a fallback to any entry whose repo id or tags
    say GGUF. Returns ``""`` when nothing matches — callers then ask for an
    explicit repo id.
    """
    query = (query or "").strip()
    if not query:
        return ""
    if "/" in query:
        return query
    from .registry import search_catalog

    entries = search_catalog(query, kind="gguf")
    if not entries:
        entries = [
            e for e in search_catalog(query)
            if "gguf" in e.repo_id.lower() or "gguf" in {t.lower() for t in e.tags}
        ]
    return entries[0].repo_id if entries else ""


class GGUFServerManager:
    """Spawn, wait for, and stop a local ``llama-server`` on a GGUF file."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        cache_dir: str = "models",
        ctx_size: int = 2048,
        threads: int = 0,
        boot_timeout: float = 1800.0,
        extra_args: str = "",
        token: str = "",
        lora_files: str = "",
        process_env: dict[str, str] | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.cache_dir = os.path.expanduser(cache_dir)
        self.ctx_size = int(ctx_size)
        self.threads = min(int(threads) or _physical_cores(), 4)
        self.boot_timeout = max(30.0, float(boot_timeout))
        self.extra_args = [a for a in (extra_args or "").split() if a]
        self.token = token
        # Comma-separated LoRA GGUF files loaded ON TOP of the base model
        # (the fine-tune artifacts from the Colab notebook).  Each one
        # becomes a repeated ``--lora`` flag on llama-server.
        self.lora_files = [
            p.strip() for p in (lora_files or "").split(",") if p.strip()
        ]
        self._env = {**os.environ, **(process_env or {})}
        self._process: subprocess.Popen[bytes] | None = None
        self.model_path = ""
        self.binary = ""
        self.problems: list[str] = []

    # ── server argument builder (unit-testable, no process needed) ──────────
    def build_args(self, model_path: str, binary: str) -> list[str]:
        """The exact argv llama-server would be spawned with."""
        args = [
            binary,
            "-m", model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--ctx-size", str(self.ctx_size),
            "-t", str(self.threads),
            "--parallel", "1",
        ]
        for lora in self.lora_files:
            args += ["--lora", lora]
        # ARM/Termux: flash-attn is x86-avx heavy; keep it off by default.
        args += self.extra_args
        if Path(binary).name == "llama-cli":
            args = [binary, "--server"] + args[1:]
        return args

    # ── property: the public URL for the router ─────────────────────────────
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # ── download path ────────────────────────────────────────────────────────
    def fetch_gguf(self, repo_id: str, *, prefer_quant: str = "Q4_K_M") -> GGUFDiagnosis:
        """Download a GGUF from Hugging Face (token-authed, resumable)."""
        from .download import HuggingFaceDownloader

        problems: list[str] = []
        hints: list[str] = []
        # Prefer a catalog-known GGUF repo when the caller gave a family name.
        repo = resolve_gguf_repo(repo_id)
        if not repo:
            problems.append(f"no GGUF repo found for {repo_id!r}")
            hints.append("use a family name that exists in the catalog "
                         "(try `nm models --catalog --kind gguf`): dolphin-8b, mistral, qwen, "
                         "phi-3.5, llama — or an exact id like 'org/repo'")
            return GGUFDiagnosis(ok=False, problems=problems, hints=hints)
        if repo != repo_id:
            hints.append(f"{repo_id!r} → {repo}")

        downloader = HuggingFaceDownloader(token=self.token, cache_dir=self.cache_dir)
        try:
            files = downloader.list_files(repo, patterns=("gguf",))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"cannot list {repo}: {exc}")
            hints.extend(_list_failure_hints(str(exc)))
            return GGUFDiagnosis(ok=False, problems=problems, hints=hints)
        candidates = [f for f in files if f.path.endswith(".gguf") and "split" not in f.path.lower()
                      and "imatrix" not in f.path.lower()]
        if prefer_quant:
            candidates = [f for f in candidates if prefer_quant.lower() in f.path.lower()] or candidates
        if not candidates:
            problems.append(f"no GGUF files in {repo}")
            return GGUFDiagnosis(ok=False, problems=problems, hints=hints)
        pick = min(candidates, key=lambda f: f.size)  # smallest usable quant
        try:
            result = downloader.download_file(
                repo, pick.path, expected_sha256=pick.sha256
            )
        except Exception as exc:  # noqa: BLE001
            problems.append(f"download failed: {exc}")
            hints.append("downloads are resumable — re-run the same command "
                         "(it continues from where it stopped, it does not start over)")
            return GGUFDiagnosis(ok=False, problems=problems, hints=hints)
        if not result.verified:
            problems.append("file is not verified after download "
                            f"({result.size} bytes expected {pick.size}) — "
                            "re-run the same command; it resumes, it does not start over")
        self.model_path = result.path
        hints.append(f"downloaded {pick.path} ({result.size // (1024 * 1024)} MB)")
        return GGUFDiagnosis(ok=not problems, model_path=result.path, problems=problems, hints=hints)

    # ── diagnostics without starting ─────────────────────────────────────────
    def doctor(self, model: str) -> GGUFDiagnosis:
        problems: list[str] = []
        hints: list[str] = []
        binary = find_llama_binary()
        model_path = find_gguf(model, self.cache_dir)
        if not binary:
            problems.append("no llama.cpp binary found (llama-server / llama-cli)")
            hints.append(
                "Termux: pkg install clang cmake && cmake -B build -DGGML_CPU_ALL_INSTRUCTIONS=ON llama.cpp "
                "&& cmake --build build --config Release -j4 (builds 'llama-server'), "
                "or use a prebuilt from a Termux package mirror"
            )
        if not model_path:
            problems.append(f"no .gguf found for {model!r} under {self.cache_dir}")
            hints.append(
                "download one: nm models --fetch 'dolphin' (public repos need no "
                "HF_TOKEN; gated ones via HF_TOKEN) or put a .gguf file under the "
                "model cache dir"
            )
        for lora in self.lora_files:
            if not Path(os.path.expanduser(lora)).exists():
                problems.append(f"LoRA file not found: {lora}")
                hints.append(
                    "the LoRA GGUF from the Colab notebook must be on disk "
                    "before the local brain can load it (NM_LLM_LOCAL_LORA)")
        if port_in_use(self.host, self.port) and self._process is None:
            state = self.health_status()
            if state in self.READY_STATES:
                hints.append(
                    f"a model server is already serving at {self.base_url} "
                    "(started by another session) — no action needed"
                )
            elif state == "loading":
                hints.append(
                    f"the model server is STILL LOADING the model at {self.base_url} "
                    "— alive and working, not frozen. On a phone this can take "
                    "several minutes; do NOT kill it, it just restarts the load"
                )
            else:
                stale = find_llama_server_pid(self.port)
                if stale:
                    problems.append(
                        f"FROZEN — llama-server pid {stale} holds port {self.port} "
                        "and does not answer"
                    )
                    hints.append(
                        "repair: `nm models --start-local` (kills it and restarts "
                        "automatically), or `kill -9 "
                        f"{stale}` by hand"
                    )
                else:
                    problems.append(
                        f"port {self.port} is held by a process that is not a "
                        "llama-server and does not answer"
                    )
                    hints.append("find and kill that process, or pick another port")
        if model_path:
            magic_ok, magic_detail = gguf_check(model_path)
            if not magic_ok:
                problems.append(f"model file is NOT a valid GGUF: {magic_detail}")
                hints.append("re-download it: `nm models --fetch <model>`")
        size = Path(model_path).stat().st_size if model_path else 0
        if model_path and size and size < 400 * 1024 * 1024:
            hints.append(f"model is only {size // (1024 * 1024)} MB — may be a partial download")
        return GGUFDiagnosis(
            ok=not problems, binary=binary, model_path=model_path,
            url=self.base_url if model_path and binary else "",
            problems=problems, hints=hints,
        )

    # ── start / stop ─────────────────────────────────────────────────────────
    def start(self, model: str) -> GGUFDiagnosis:
        """Spawn llama-server and wait for it to serve. Actionable on failure."""
        problems: list[str] = []
        hints: list[str] = []
        self.problems = problems

        # A healthy model server already owns the port (a previous CLI call,
        # or the chat's auto-start) → nothing to do, whatever the model arg.
        if port_in_use(self.host, self.port) and self._process is None and self._healthy():
            return GGUFDiagnosis(
                ok=True, binary="", model_path="", problems=[],
                hints=[f"already serving at {self.base_url} — nothing to do"],
            )

        # BEFORE resolving binary/model: a FROZEN server holding the port must
        # be repaired regardless of model args.  The port is held by something
        # that does not answer — if it is OUR llama-server (frozen — OOM on a
        # phone is the classic), kill + restart.  A foreign process we leave
        # alone.  A server that is still LOADING also does not answer — and
        # killing it throws away the entire load, so it is never touched.
        if port_in_use(self.host, self.port) and self._process is None:
            state = self.health_status()
            if state == "loading":
                return GGUFDiagnosis(
                    ok=True, binary="", model_path="",
                    problems=[],
                    hints=["model still loading — left the running server "
                           "alone (killing it would restart the whole load)"],
                )
            stale = find_llama_server_pid(self.port)
            if stale is None:
                problems.append(
                    f"port {self.port} is held by a FOREIGN process that does not "
                    "answer and is not a recognizable llama-server — not touching it")
                hints.append("find and kill it (e.g. `fuser -k "
                             f"{self.port}/tcp` on a machine that has fuser), "
                             "or change llm.local_port")
                return GGUFDiagnosis(ok=False, binary="", model_path="",
                                     problems=problems, hints=hints)
            _log.warning("stale llama-server (pid %s) holds port %s and is not "
                         "answering — killing and restarting", stale, self.port)
            if not kill_pid(stale):
                problems.append(
                    f"could not kill the stale llama-server (pid {stale})")
                hints.append("kill it manually: `kill -9 "
                             f"{stale}`, then re-run")
                return GGUFDiagnosis(ok=False, binary="", model_path="",
                                     problems=problems, hints=hints)
            deadline = time.time() + 15.0
            while port_in_use(self.host, self.port) and time.time() < deadline:
                time.sleep(0.3)
            if port_in_use(self.host, self.port):
                problems.append(
                    f"stale server killed but port {self.port} is still held")
                hints.append("wait a few seconds and re-run")
                return GGUFDiagnosis(ok=False, binary="", model_path="",
                                     problems=problems, hints=hints)
            hints.insert(0, f"repaired: killed frozen server (pid {stale}), "
                            "starting a fresh one")

        binary = find_llama_binary()
        model_path = find_gguf(model, self.cache_dir)
        if not model_path and "/" in model and not Path(os.path.expanduser(model)).exists():
            # Looks like a repo id: fetch it, then retry resolution.
            fetch = self.fetch_gguf(model)
            if fetch.ok:
                model_path = fetch.model_path
                hints.extend(fetch.hints)
            else:
                problems.extend(fetch.problems)
                hints.extend(fetch.hints)
        if not binary:
            problems.append("no llama.cpp binary found")
            hints.append("Termux: build llama.cpp (see `nm models --local-doctor` output)")
        if not model_path:
            problems.append(f"no .gguf found for {model!r} under {self.cache_dir}")
            hints.append("run `nm models --fetch <model>` or set llm.local_model to a file path")
        if not binary or not model_path:
            return GGUFDiagnosis(ok=False, binary=binary, model_path=model_path,
                                 problems=problems, hints=hints)

        for lora in self.lora_files:
            lora_path = os.path.expanduser(lora)
            if not Path(lora_path).exists():
                problems.append(f"LoRA file not found: {lora}")
                hints.append("check NM_LLM_LOCAL_LORA — the LoRA GGUF from "
                             "the Colab notebook must be on disk first")
                return GGUFDiagnosis(ok=False, binary=binary, model_path=model_path,
                                     problems=problems, hints=hints)

        if port_in_use(self.host, self.port):
            # The frozen/foreign cases were handled up front, before
            # binary/model resolution.  Whatever still holds the port here
            # either answers (a healthy external server) or appeared mid-run.
            if self._healthy():
                return GGUFDiagnosis(
                    ok=True, binary=binary, model_path=model_path,
                    problems=[],
                    hints=[f"already serving at {self.base_url} — nothing to do"],
                )
            problems.append(
                f"port {self.port} is still held by a process that does not answer")
            hints.append("kill it, or change llm.local_port")
            return GGUFDiagnosis(ok=False, binary=binary, model_path=model_path,
                                 problems=problems, hints=hints)

        args = self.build_args(model_path, binary)
        _log.info("starting llama.cpp: %s", " ".join(args[:6]) + " …")
        try:
            self._process = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=self._env,
            )
        except FileNotFoundError as exc:
            problems.append(f"binary not executable: {exc}")
            hints.append("check the llama.cpp build; `llama-server --help` should work in a shell")
            return GGUFDiagnosis(ok=False, binary=binary, model_path=model_path,
                                 problems=problems, hints=hints)

        # Mobile loads are slow: an 8B q4 can take several minutes on a phone.
        deadline = time.time() + self.boot_timeout
        last_err = ""
        while time.time() < deadline:
            if self._process.poll() is not None:
                tail = _drain_tail(self._process)
                last_err = tail[-400:] if tail else f"exited rc={self._process.returncode}"
                break
            if self._healthy():
                self.binary, self.model_path = binary, model_path
                return GGUFDiagnosis(
                    ok=True, binary=binary, model_path=model_path, url=self.base_url,
                    pid=self._process.pid,
                    hints=hints + [f"ready in {time.time() - (deadline - self.boot_timeout):.0f}s"],
                )
            time.sleep(2.0)
        else:
            problems.append(f"model was still loading after {self.boot_timeout:.0f}s "
                            "(the server is alive and working, just slow)")
            hints.append("on a phone this is NORMAL under heat/RAM pressure — "
                         "raise llm.local_boot_timeout, "
                         "or lower llm.local_ctx / use a smaller quant "
                         "(Q4_K_S / Q3_K_M load faster)")
            self.stop()
            return GGUFDiagnosis(ok=False, binary=binary, model_path=model_path,
                                 problems=problems, hints=hints)
        problems.append(f"server process died during boot: {last_err}")
        hints.append(
            "usually OOM on mobile: lower --ctx-size (e.g. 2048), use Q4_K_S/Q4_0, "
            "close other apps, or drop to a 1.5–4B model"
        )
        return GGUFDiagnosis(ok=False, binary=binary, model_path=model_path,
                             problems=problems, hints=hints)

    def heal(self) -> GGUFDiagnosis:
        """Bring the local model to serving, whatever is blocking it.

        The phone incident in one method: a llama-server frozen by RAM
        pressure still holds its port, so every naive 'is anything
        listening?' check sees a healthy-looking setup while every request
        hangs until timeout.  heal() distinguishes the states and repairs
        only the frozen one:

        * port free              → start
        * answering (ok/busy)    → nothing to do
        * still loading          → LEAVE IT ALONE.  Killing a server that is
                                   mid-load just restarts the whole load —
                                   an infinite hot-but-silent loop.
        * silent                 → if it's our llama-server: kill + restart;
                                   if it's a foreign process: say so, don't touch
        """
        if self._process is not None and self._process.poll() is None:
            if self._healthy():
                return GGUFDiagnosis(ok=True, binary=self.binary,
                                     model_path=self.model_path, problems=[],
                                     hints=["already serving (this session)"])
            if self.health_status() == "loading":
                return GGUFDiagnosis(ok=True, binary=self.binary,
                                     model_path=self.model_path, problems=[],
                                     hints=["model still loading (this session) "
                                            "— left it to finish"])
            # our own child is wedged — stop it and fall through to restart
            _log.warning("this session's llama-server (pid %s) is wedged — "
                         "restarting", self._process.pid)
            self.stop()
        if port_in_use(self.host, self.port):
            state = self.health_status()
            if state in self.READY_STATES:
                return GGUFDiagnosis(ok=True, binary=self.binary,
                                     model_path=self.model_path, problems=[],
                                     hints=[f"already serving at {self.base_url}"])
            if state == "loading":
                return GGUFDiagnosis(ok=True, binary=self.binary,
                                     model_path=self.model_path, problems=[],
                                     hints=["model still loading — left the "
                                            "running server alone"])
            stale = find_llama_server_pid(self.port)
            if stale is None:
                return GGUFDiagnosis(
                    ok=False, binary="", model_path="",
                    problems=[f"port {self.port} is held by a foreign process "
                              "that does not answer — not touching it"],
                    hints=["find and kill it, or change llm.local_port"])
            _log.warning("heal: frozen llama-server (pid %s) on port %s — "
                         "killing and restarting", stale, self.port)
            if not kill_pid(stale):
                return GGUFDiagnosis(
                    ok=False, binary="", model_path="",
                    problems=[f"could not kill frozen llama-server (pid {stale})"],
                    hints=[f"kill it manually: `kill -9 {stale}`"])
            deadline = time.time() + 15.0
            while port_in_use(self.host, self.port) and time.time() < deadline:
                time.sleep(0.3)
        if not self.model_path:
            return GGUFDiagnosis(
                ok=False, binary="", model_path="",
                problems=["no model known to this manager — set llm.local_model"],
                hints=["nm models --promote-local <path to .gguf>"])
        diagnosis = self.start(self.model_path)
        if diagnosis.ok:
            diagnosis.hints = ["repaired: frozen server replaced with a fresh one"] \
                + [h for h in diagnosis.hints if "repaired" not in h]
        return diagnosis

    def health_status(self) -> str:
        """``ok`` | ``busy`` | ``loading`` | ``error`` | ``dead`` — see
        module-level :func:`health_status`."""
        return health_status(self.host, self.port)

    #: States in which the model can actually answer a message.
    READY_STATES = ("ok", "busy")

    def _healthy(self) -> bool:
        # "loading" is NOT healthy: the model cannot answer yet, and a
        # request sent now would sit in the queue for the whole load.
        return self.health_status() in self.READY_STATES

    def stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            process.terminate()
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        except Exception:  # noqa: BLE001
            pass

    def status(self) -> dict[str, Any]:
        """running / not_running / FROZEN / foreign_holder — with pids.

        'Port open' is NOT 'running': a llama-server frozen by RAM pressure
        holds its port while answering to nobody.  That is the state a
        phone gets stuck in, so status must say it by name.
        """
        process = self._process
        own = process is not None and process.poll() is None
        running = own
        external_pid = 0
        state = "not_running"
        if not running and port_in_use(self.host, self.port):
            state = self.health_status()
            if state in self.READY_STATES:
                # Started by ANOTHER process (a previous CLI call, or the
                # chat's auto-start) but genuinely serving → running.
                running = True
                state = "running"
            elif state == "loading":
                external_pid = find_llama_server_pid(self.port) or 0
                # stays "loading" — alive, hot, not ready. NOT frozen.
            else:
                external_pid = find_llama_server_pid(self.port) or 0
                state = "frozen" if external_pid else "foreign_holder"
        return {
            "running": running,
            "state": state,
            "pid": process.pid if own else external_pid,
            "external_pid": external_pid,
            "url": self.base_url if running else "",
            "healthy": self._healthy() if running else False,
            "model": self.model_path,
            "binary": self.binary,
        }


def _physical_cores() -> int:
    try:
        return max(1, os.process_cpu_count() or os.cpu_count() or 4)
    except Exception:  # noqa: BLE001
        return 4


def _drain_tail(process: subprocess.Popen[bytes], limit: int = 4000) -> str:
    try:
        if process.stdout is None:
            return ""
        data = process.stdout.read()
        return data.decode("utf-8", "replace")[-limit:] if data else ""
    except Exception:  # noqa: BLE001
        return ""
