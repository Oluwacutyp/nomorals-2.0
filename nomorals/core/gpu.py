"""GPU capability detection — the single detection point (register
§11(2), owner-flagged IMPORTANT).

Every organ used to probe the GPU its own way (trellis's ``probe_gpu``,
videogen's ``torch.cuda`` import, the edit engine's ``nvidia-smi``,
lipsync's per-backend checks, the models provisioner's subprocess
gate) — same question, five answers, none of them visible to the
profile runtime. This module answers it once, and the profile runtime
carries the answer:

* ``path`` — ``"local"`` (an NVIDIA GPU on this box), ``"forwarded"``
  (a GPU reached over the network, declared by configuration), or
  ``"cpu"`` (no GPU path at all).
* ``detect_gpu()`` — precedence: explicit ``pin`` argument →
  ``NM_GPU`` env var → ``NM_GPU_URL`` (a configured endpoint is itself
  the declaration) → local probing (``nvidia-smi``, then ``torch``) →
  CPU fallback. An explicit pin always wins, like ``NM_PROFILE`` and
  ``NM_CLOUD``. Detection never raises and never pretends: a forwarded
  GPU is attested before it is believed (see below), and a forwarded
  pin with no endpoint fails closed (``has_gpu`` False — nothing routes
  to a GPU with no address).
* ``gpu_up(family)`` — the activation seam: ``seeing`` / ``speaking`` /
  ``generating`` / ``training`` run on the GPU when one is present;
  ``text`` never does. Per-organ VRAM floors (LatentSync 18 GB, …)
  stay with the organs — this module answers "is there a GPU path",
  not "does this model fit".

Attestation (forwarded path): a bare ``NM_GPU_URL`` string is a claim,
not proof — an endpoint that only says "I have 24 GB" is not believed.
``detect_gpu()`` challenges the endpoint (``GET {url}/gpu/attest``)
and believes it only when the endpoint measures its own GPU *and*
allocates the challenged bytes on CUDA, returning a receipt derived
from the allocated memory (a nonce pattern read back from the buffer,
plus ``mem_get_info`` before/after showing the allocation landing).
The trust model is plain: the endpoint is owner-configured, so
attestation distinguishes a real Devon GPU endpoint from a dead URL,
a wrong service, a canned responder, or a GPU too small or too busy
— it does not defend against a deliberately malicious endpoint that
implements the protocol while lying (a remote client cannot
cryptographically verify remote memory allocation without trusted
hardware; the fresh nonce per attestation is what defeats canned and
replayed receipts). A failed, timed-out, or refused attestation
degrades to the declared-but-not-probed answer, honestly — the floors
keep refusing it. Attestation is cached per process with a TTL
(``ATTEST_TTL_S``) and re-attempted on refresh only once the TTL has
expired, so a dead endpoint is not hammered on every gate call.

The endpoint side lives here too: ``probe_attestation()`` is what a
Devon API server on a GPU host runs for ``GET /gpu/attest`` — it
measures its own GPU with the local probe and performs the allocation
in a torch subprocess (torch stays out of the server process, exactly
as the detector keeps it out of this one).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "GPU_PATHS",
    "GPU_FAMILIES",
    "GPU_PIN_ENV",
    "GPU_URL_ENV",
    "ATTEST_PROOF_BYTES",
    "ATTEST_MIN_BYTES",
    "ATTEST_MAX_BYTES",
    "ATTEST_TIMEOUT_S",
    "ATTEST_TTL_S",
    "ATTEST_THROTTLE_S",
    "GpuInfo",
    "detect_gpu",
    "gpu_up",
    "attest_forwarded",
    "probe_attestation",
]

#: GPU path values: local GPU / network-forwarded GPU / CPU only.
GPU_PATHS: tuple[str, ...] = ("local", "forwarded", "cpu")

#: Capability families that run on the GPU when one is present.
#: Text is deliberately absent — text-only work never needs a GPU.
GPU_FAMILIES: frozenset[str] = frozenset(
    {"seeing", "speaking", "generating", "training"})

#: Env vars: NM_GPU pins the path (auto|local|forwarded|cpu);
#: NM_GPU_URL names the forwarded endpoint (implies forwarded).
GPU_PIN_ENV = "NM_GPU"
GPU_URL_ENV = "NM_GPU_URL"

#: Attestation: the proof allocation the endpoint must make on CUDA
#: (1 GiB — enough to prove a live device that can allocate, small
#: enough to be cheap; the *total* VRAM number comes from the
#: endpoint's own local probe, not from the proof size). The server
#: refuses proofs outside [ATTEST_MIN_BYTES, ATTEST_MAX_BYTES].
ATTEST_PROOF_BYTES = 1 << 30
ATTEST_MIN_BYTES = 1 << 20
ATTEST_MAX_BYTES = 2 << 30

#: How long the client waits for an attestation answer. A cold torch
#: import on the endpoint can take seconds; past this the client
#: degrades to unattested rather than hanging the caller.
ATTEST_TIMEOUT_S = 30.0

#: How long an attestation (or a failed attempt) is trusted before a
#: refresh re-challenges the endpoint.
ATTEST_TTL_S = 3600.0

#: Server-side: at most one proof allocation per interval, so a public
#: /gpu/attest endpoint cannot be hammered into repeated allocations.
ATTEST_THROTTLE_S = 60.0

#: The receipt is a sha256 over this many samples of this many bytes,
#: read back from the allocated buffer at eighths of its length.
_ATTEST_SAMPLE_BYTES = 8192
_ATTEST_NSAMPLES = 8


def _run(args: list[str], timeout: int | None = None,
         cwd: str | None = None) -> Any:
    """Subprocess seam (same shape as media.trellis._run)."""
    return subprocess.run(args, capture_output=True, text=True,
                          timeout=timeout, cwd=cwd)


@dataclass(frozen=True)
class GpuInfo:
    """One GPU-capability answer: which path, and what it offers."""

    path: str = "cpu"          # local | forwarded | cpu
    has_gpu: bool = False
    name: str = ""             # device name, "" when unknown/unprobed
    vram_gb: float = 0.0       # 0.0 when unknown/unprobed
    source: str = ""           # nvidia-smi | torch.cuda | env | attested | ""
    detail: str = ""           # human-readable: what was found, and how
    forward_url: str = ""      # forwarded endpoint, "" when not forwarded
    attested: bool = False     # forwarded only: the endpoint proved its VRAM

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "has_gpu": self.has_gpu,
            "name": self.name,
            "vram_gb": self.vram_gb,
            "source": self.source,
            "detail": self.detail,
            "forward_url": self.forward_url,
            "attested": self.attested,
        }


def _probe_nvidia_smi() -> dict[str, Any] | None:
    """VRAM via nvidia-smi. None when it cannot answer."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        proc = _run(["nvidia-smi", "--query-gpu=name,memory.total",
                     "--format=csv,noheader,nounits"], timeout=60)
    except Exception:  # noqa: BLE001 - a failed probe is "no GPU"
        return None
    if proc.returncode != 0:
        return None
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return None
    parts = [p.strip() for p in line[0].split(",")]
    if len(parts) < 2:
        return None
    try:
        vram_mib = float(parts[1])
    except ValueError:
        return None
    return {"has_gpu": True, "name": parts[0],
            "vram_gb": round(vram_mib / 1024.0, 2),
            "source": "nvidia-smi"}


def _probe_torch() -> dict[str, Any] | None:
    """VRAM via torch.cuda (subprocess — torch stays out of this
    process). None when torch/CUDA cannot answer."""
    snippet = ("import torch; "
               "p = torch.cuda.get_device_properties(0); "
               "print(p.name); print(p.total_memory)")
    try:
        proc = _run([sys.executable, "-c", snippet], timeout=120)
    except Exception:  # noqa: BLE001 - a failed probe is "no GPU"
        return None
    if proc.returncode != 0:
        return None
    lines = (proc.stdout or "").strip().splitlines()
    if len(lines) < 2:
        return None
    try:
        total_bytes = float(lines[1])
    except ValueError:
        return None
    return {"has_gpu": True, "name": lines[0].strip(),
            "vram_gb": round(total_bytes / 1e9, 2),
            "source": "torch.cuda"}


def _cpu_gpu(detail: str) -> GpuInfo:
    return GpuInfo(path="cpu", has_gpu=False, vram_gb=0.0, source="",
                   detail=detail)


def _local_gpu() -> GpuInfo:
    """Local probing: nvidia-smi first, torch second, CPU fallback."""
    for probe in (_probe_nvidia_smi, _probe_torch):
        try:
            found = probe()
        except Exception:  # noqa: BLE001 - a probe never kills detection
            found = None
        if found is not None:
            return GpuInfo(
                path="local", has_gpu=True, name=found["name"],
                vram_gb=found["vram_gb"], source=found["source"],
                detail=f"{found['name']} with {found['vram_gb']} GB VRAM "
                       f"({found['source']})")
    return _cpu_gpu("no NVIDIA GPU detected")


def _forwarded_gpu(url: str) -> GpuInfo:
    """A declared remote GPU, unattested. has_gpu is True — the owner's
    endpoint configuration IS the claim, exactly like an NM_PROFILE pin
    — but no VRAM or device name is ever reported here: the endpoint
    was never challenged, so the floors cannot verify against it."""
    return GpuInfo(path="forwarded", has_gpu=True, source="env",
                   forward_url=url,
                   detail=f"forwarded GPU declared at {url} "
                          "(not probed — attestation failed or timed out)")


# ── attestation ───────────────────────────────────────────────────────
# The endpoint proves its VRAM in two parts: it measures its own GPU
# (its local probe — a real measurement made on its side), and it
# allocates the challenged bytes on CUDA and returns a receipt derived
# from the allocated memory. The receipt is a sha256 over samples read
# back from the buffer; the samples are the challenge nonce repeated,
# so the client recomputes the expected receipt from the nonce alone
# without holding the memory. A fresh nonce per attestation defeats
# canned and replayed receipts; the mem_get_info before/after proves
# the allocation actually landed (free VRAM drops by the proof size).

def _attest_sample_offsets(n_bytes: int) -> list[int]:
    """Sample offsets: eighths of the buffer, 8 samples."""
    return [i * n_bytes // _ATTEST_NSAMPLES
            for i in range(_ATTEST_NSAMPLES)]


def _attest_pattern(challenge: str, offset: int, length: int) -> bytes:
    """The nonce pattern at ``offset``: the nonce bytes repeated."""
    raw = bytes.fromhex(challenge)
    return bytes(raw[(offset + i) % len(raw)] for i in range(length))


def _attest_expected_receipt(challenge: str, n_bytes: int) -> str:
    """The receipt a correct endpoint returns for this challenge."""
    samples = [_attest_pattern(challenge, off, _ATTEST_SAMPLE_BYTES)
               for off in _attest_sample_offsets(n_bytes)]
    return hashlib.sha256(b"".join(samples)).hexdigest()


def _http_get_json(url: str, timeout: float) -> dict[str, Any]:
    """GET ``url`` as JSON. The seam tests patch (``gpu_mod``).

    ``allow_private_ips`` is on deliberately: the URL is the owner's
    own configuration (``NM_GPU_URL``), and a GPU host on his LAN is a
    private address by definition — the SSRF guard would otherwise
    refuse exactly the endpoints this exists for.
    """
    from .http import HttpClient

    resp = HttpClient(timeout=timeout, allow_private_ips=True).get(url)
    if not resp.ok:
        raise ValueError(
            f"attestation endpoint answered HTTP {resp.status}")
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("attestation endpoint answered non-JSON-object")
    return data


def attest_forwarded(url: str, *, timeout: float = ATTEST_TIMEOUT_S,
                     proof_bytes: int = ATTEST_PROOF_BYTES,
                     http_get: Any = None) -> GpuInfo | None:
    """Challenge a forwarded endpoint and verify its VRAM proof.

    Returns the attested ``GpuInfo`` (name + VRAM, ``source="attested"``)
    or None when the endpoint does not answer, refuses, times out, or
    its receipt does not verify. Never raises.
    """
    url = (url or "").strip()
    if not url:
        return None
    proof_bytes = int(proof_bytes)
    if proof_bytes < ATTEST_MIN_BYTES or proof_bytes > ATTEST_MAX_BYTES:
        return None
    challenge = secrets.token_hex(16)
    target = (f"{url.rstrip('/')}/gpu/attest?challenge={challenge}"
              f"&bytes={proof_bytes}")
    try:
        data = (http_get or _http_get_json)(target, timeout)
    except Exception:  # noqa: BLE001 - any transport failure is unattested
        return None
    if not isinstance(data, dict) or data.get("ok") is not True:
        return None
    try:
        name = str(data.get("name") or "").strip()
        vram_gb = float(data.get("vram_gb") or 0.0)
        total_bytes = int(data.get("total_bytes") or 0)
        free_before = int(data.get("free_before_bytes") or 0)
        free_after = int(data.get("free_after_bytes") or 0)
        allocated = int(data.get("allocated_bytes") or 0)
        receipt = str(data.get("receipt") or "")
    except (TypeError, ValueError):
        return None
    if not name or vram_gb <= 0 or total_bytes <= 0:
        return None
    if allocated != proof_bytes:
        return None
    # The allocation must have landed: free VRAM drops by the proof
    # size (half is the tolerance — the caching allocator holds a
    # little extra, and another process may allocate concurrently).
    if free_before <= 0 or free_after < 0 or free_before > total_bytes:
        return None
    if free_before - free_after < proof_bytes // 2:
        return None
    # The reported total must be coherent with the reported VRAM
    # (nvidia-smi MiB/1024 vs torch bytes/1e9 differ by ~7%).
    if vram_gb > (total_bytes / 1e9) * 1.1 + 0.5:
        return None
    if receipt != _attest_expected_receipt(challenge, proof_bytes):
        return None
    vram_gb = round(vram_gb, 2)
    return GpuInfo(path="forwarded", has_gpu=True, name=name,
                   vram_gb=vram_gb, source="attested", attested=True,
                   forward_url=url,
                   detail=f"forwarded GPU attested at {url}: {name} "
                          f"with {vram_gb} GB VRAM (proof allocation "
                          f"{proof_bytes} bytes)")


def _attest_allocate_in_process(challenge: str,
                                n_bytes: int) -> dict[str, Any]:
    """Server-side proof, run in a torch subprocess (never in the
    server process itself): allocate ``n_bytes`` on CUDA, fill it with
    the nonce pattern, read the samples back. Never raises — a failure
    is an honest ``{"ok": False}`` answer, never a crash.

    The fill broadcasts the 16-byte nonce pattern across the buffer,
    so byte ``p`` of the buffer is ``nonce[p % 16]`` — exactly what
    ``_attest_pattern`` predicts, and what the client verifies.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - torch absent is a fact
        return {"ok": False,
                "reason": f"torch is not installed on this host ({exc})"}
    try:
        if not torch.cuda.is_available():
            return {"ok": False,
                    "reason": "torch is installed but no CUDA device "
                              "is available"}
        free_before, total = torch.cuda.mem_get_info()
        raw = bytes.fromhex(challenge)
        pattern = torch.tensor(list(raw), dtype=torch.uint8)
        buf = torch.empty(int(n_bytes), dtype=torch.uint8, device="cuda")
        try:
            buf[:] = pattern
            free_after, _ = torch.cuda.mem_get_info()
            samples = [bytes(buf[o:o + _ATTEST_SAMPLE_BYTES].tolist())
                       for o in _attest_sample_offsets(int(n_bytes))]
        finally:
            del buf
            # Freeing is best-effort — a failed empty_cache must not
            # turn a good proof into a failure.
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
        receipt = hashlib.sha256(b"".join(samples)).hexdigest()
        return {"ok": True, "allocated_bytes": int(n_bytes),
                "receipt": receipt,
                "free_before_bytes": int(free_before),
                "free_after_bytes": int(free_after),
                "total_bytes": int(total)}
    except Exception as exc:  # noqa: BLE001 - a failed proof is "no"
        return {"ok": False,
                "reason": f"allocation probe failed: {exc}"}


#: The allocation runs in a subprocess so torch never enters the
#: server process (same rule as the detector's torch probe).
_ATTEST_SNIPPET = (
    "import json, sys; "
    "from nomorals.core.gpu import _attest_allocate_in_process; "
    "print(json.dumps(_attest_allocate_in_process("
    "sys.argv[1], int(sys.argv[2]))))")


def probe_attestation(challenge: str, n_bytes: int) -> dict[str, Any]:
    """Endpoint side of the protocol (``GET /gpu/attest``): measure
    this host's GPU with the local probe, then prove it by allocating
    ``n_bytes`` on CUDA. Never raises."""
    info = _local_gpu()
    if not info.has_gpu:
        return {"ok": False, "reason": "no CUDA device on this host"}
    repo_root = Path(__file__).resolve().parents[2]
    try:
        proc = _run([sys.executable, "-c", _ATTEST_SNIPPET, challenge,
                     str(int(n_bytes))], timeout=120, cwd=str(repo_root))
    except Exception as exc:  # noqa: BLE001 - a failed probe is "no"
        return {"ok": False,
                "reason": f"attestation probe did not run: {exc}"}
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        tail = err[-1] if err else "unknown error"
        if "No module named 'torch'" in tail:
            return {"ok": False,
                    "reason": "torch is not installed on this host"}
        return {"ok": False,
                "reason": f"attestation probe failed: {tail}"}
    try:
        alloc = json.loads(proc.stdout or "{}")
    except ValueError:
        return {"ok": False,
                "reason": "attestation probe answered unparseable output"}
    if not isinstance(alloc, dict) or alloc.get("ok") is not True:
        reason = (str(alloc.get("reason") or "allocation probe failed")
                  if isinstance(alloc, dict) else "allocation probe failed")
        return {"ok": False, "reason": reason}
    try:
        total_bytes = int(alloc["total_bytes"])
        free_before = int(alloc["free_before_bytes"])
        free_after = int(alloc["free_after_bytes"])
        allocated = int(alloc["allocated_bytes"])
        receipt = str(alloc["receipt"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False,
                "reason": "attestation probe answered an incomplete proof"}
    return {
        "ok": True, "name": info.name, "vram_gb": info.vram_gb,
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / 1e9, 2),
        "free_before_bytes": free_before,
        "free_after_bytes": free_after,
        "free_before_gb": round(free_before / 1e9, 2),
        "free_after_gb": round(free_after / 1e9, 2),
        "allocated_bytes": allocated,
        "receipt": receipt,
    }


#: Process-wide cache: probing spawns subprocesses, so the first
#: detection pays and the rest reuse it. refresh=True re-probes.
_GPU_CACHE: GpuInfo | None = None

#: When the cached answer was written (monotonic) — a cached attested
#: answer older than ATTEST_TTL_S is re-challenged.
_GPU_CACHE_AT: float = 0.0

#: When the last attestation attempt ran (monotonic) — a failed or
#: fresh attempt is not retried inside the TTL, so a dead endpoint is
#: not hammered on every gate call.
_ATTEST_ATTEMPT_AT: float = 0.0


def _forwarded_attested_or_declared(url: str, now: float) -> GpuInfo:
    """The forwarded branch: challenge the endpoint unless a fresh
    attestation (or a fresh failed attempt) is already cached."""
    global _ATTEST_ATTEMPT_AT
    cached = _GPU_CACHE
    if (cached is not None and cached.path == "forwarded"
            and cached.forward_url == url
            and now - _ATTEST_ATTEMPT_AT < ATTEST_TTL_S):
        return cached
    _ATTEST_ATTEMPT_AT = now
    attested = attest_forwarded(url)
    if attested is not None:
        return attested
    return _forwarded_gpu(url)


def detect_gpu(*, pin: str | None = None,
               forward_url: str | None = None,
               refresh: bool = False) -> GpuInfo:
    """Detect this box's GPU path. Never raises.

    Precedence: explicit ``pin`` → ``NM_GPU`` → ``NM_GPU_URL`` (a
    configured endpoint implies forwarded) → local probing → CPU.
    ``"auto"``, ``""`` and unknown pins all fall through to local
    probing; ``"cpu"`` disables the GPU outright; ``"forwarded"``
    without an endpoint fails closed (no address to route to). A
    forwarded endpoint is challenged (``attest_forwarded``) unless a
    fresh attestation — or a fresh failed attempt — is cached inside
    ``ATTEST_TTL_S``; a stale attested answer is re-challenged even
    without ``refresh=True``.
    """
    global _GPU_CACHE, _GPU_CACHE_AT
    now = time.monotonic()
    if not refresh and _GPU_CACHE is not None:
        if (_GPU_CACHE.path == "forwarded" and _GPU_CACHE.attested
                and now - _ATTEST_ATTEMPT_AT > ATTEST_TTL_S):
            pass  # stale attestation: fall through and re-challenge
        else:
            return _GPU_CACHE
    pin_value = (pin if pin is not None
                 else os.environ.get(GPU_PIN_ENV, "") or "")
    pin_value = str(pin_value).strip().lower()
    url = (forward_url if forward_url is not None
           else os.environ.get(GPU_URL_ENV, "") or "")
    url = str(url).strip()
    if pin_value == "cpu":
        info = _cpu_gpu("GPU disabled by NM_GPU=cpu")
    elif pin_value == "forwarded":
        if url:
            info = _forwarded_attested_or_declared(url, now)
        else:
            info = GpuInfo(path="forwarded", has_gpu=False, source="env",
                           detail="forwarded GPU declared "
                                  "(NM_GPU=forwarded) but no endpoint "
                                  "configured — set NM_GPU_URL")
    elif pin_value == "local":
        info = _local_gpu()
    elif url:
        info = _forwarded_attested_or_declared(url, now)
    else:
        info = _local_gpu()
    _GPU_CACHE = info
    _GPU_CACHE_AT = now
    return info


def gpu_up(family: str, gpu: GpuInfo | None = None) -> bool:
    """True when ``family`` should run on the GPU on this box.

    ``seeing`` / ``speaking`` / ``generating`` / ``training`` are GPU
    work when a GPU path exists (local or forwarded); ``text`` is
    never GPU work; anything else is not a known family. ``gpu``
    overrides the detected box (tests and embedders).
    """
    fam = (family or "").strip().lower()
    if fam == "text" or fam not in GPU_FAMILIES:
        return False
    info = gpu if gpu is not None else detect_gpu()
    return bool(info.has_gpu)
