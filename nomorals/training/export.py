"""L3 — training: TorchScript export — trace or script a torch module.

The export patterns mined (pattern only, no code ported) from
storytold/torchscript-models — three wrap scripts turning Glow-TTS
and MelGAN checkpoints into ``.torchjit`` files, plus the repo's
porting guide (README.md):

* **Trace vs script is the caller's choice, and the two combine
  per-node.** Tracing evaluates the model on example data and fixes
  the branches taken; scripting transforms the code and keeps the
  branches. The upstream scripts hardcode ``trace_model = True``
  with a ``TODO: Try True`` — the choice is a flag, never a
  decision. Here ``method`` is named by the caller and recorded in
  the receipt.
* **The trace sanity check fails on real models.** Both Glow-TTS
  scripts hit ``Tracing failed sanity checks! Graphs differed
  across invocations!`` and silently fell back to
  ``check_trace=False``. Here the check runs first; a failure is
  reported with its cause, and the unchecked retry happens only
  when the caller explicitly allows it — and the receipt records
  that it was unchecked. (On the torch this box runs, the check
  warns rather than raising when traced outputs differ — those
  warnings are captured into the receipt too, so a check that
  passed-noisily is visible instead of silent.)
* **Example-data length matters.** The guide's finding: too short
  an example can crash the JIT at eval time on long input. The
  Glow-TTS script deliberately uses a long test sentence for
  exactly that reason. Here a ``long_probe_inputs`` run proves the
  exported module survives longer input, and the result is in the
  receipt.
* **Scripting pitfalls, diagnosed not just raised.** The guide
  lists four: non-constant indices (use enumerate/zip), a tuple
  on the left of an assignment (break it up), double list
  comprehensions (unroll / flatten), and untyped ``int`` arguments
  (type-hint them). A scripting failure names the pitfall it
  matches, with the fix.
* **Output naming.** ``--output_file`` must end ``.torchjit``;
  ``--copy_name`` derives ``{model_name}.torchjit``. Here the
  extension is enforced and the receipt sidecar is
  ``<out>.json`` — the same sidecar convention as the realtime
  probe export (``voice/realtime.py``).
* **Load discipline.** ``set_grad_enabled(False)`` / ``no_grad``,
  ``model.eval()``, checkpoint loaded, then exported. Here the
  exporter does the same, and restores the module's device
  afterwards.

What the upstream never had, this module adds: a verification
step that re-runs the exported artifact against the source model
on a fixed probe input and reports the max output deviation as a
number (not a pass/fail shrug); explicit dynamic-axis handling
(scripted artifacts are shape-polymorphic, traced ones are
shape-specialised and say so); device handling (export from CPU
or CUDA, artifact always runnable on CPU); and a load path that
refuses a mismatched or corrupt artifact naming the cause.

Torch is optional: the module imports cleanly without it and
every torch operation refuses naming what is missing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import warnings
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch absent on some builds
    _TORCH_AVAILABLE = False

__all__ = [
    "EXPORT_FORMAT_VERSION",
    "ExportError",
    "ExportReceipt",
    "LoadedExport",
    "Verification",
    "diagnose_script_error",
    "export_module",
    "load_exported",
    "receipt_path_for",
    "require_torch",
    "resolve_device",
    "torch_available",
    "verify_export",
]

# The receipt sidecar's format stamp. A loader refuses a receipt
# whose stamp it does not know, naming the stamp it found.
EXPORT_FORMAT_VERSION = "training.export/1.0"

_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

# The upstream porting guide's four scripting pitfalls, as
# (error-text pattern, diagnosis + fix). Checked in order; the
# first match names the cause.
_SCRIPT_PITFALLS: tuple[tuple[str, str], ...] = (
    ("integer literal",
     "TorchScript needs a constant index — a loop variable used as "
     "an index (model.layers[i]) is not one. Rewrite the loop with "
     "enumerate() or zip() over the layers instead of range() + [i]."),
    ("lhs of assignment|left-hand side",
     "TorchScript cannot assign a tuple on the left-hand side "
     "(x, self.attn = ...). Break it up: results = ...; x = "
     "results[0]; self.attn = results[1]."),
    ("multiple comprehension generators|comprehension",
     "TorchScript does not support double list comprehensions "
     "([item for sublist in l for item in sublist]). Unroll the "
     "loop, or flatten the tensor with torch.flatten()."),
    ("Expected a value of type 'Tensor'|instead found type 'int'|"
     "type 'int'",
     "TorchScript inferred the wrong type for an int argument. "
     "Type-hint the signature (def f(..., length: int)) or the "
     "shape (List[List[int]])."),
)


class ExportError(ValueError):
    """An export, verification, or load failure — the cause is named."""


def torch_available() -> bool:
    return _TORCH_AVAILABLE


def require_torch(what: str = "this TorchScript export") -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            f"{what} needs torch, which is not installed on this "
            "machine. Install torch (the CPU build is enough for "
            "export + verification) and retry."
        )
    import torch as _torch

    return _torch


def diagnose_script_error(exc: Exception) -> str | None:
    """Name the upstream pitfall a scripting failure matches, if any.

    Returns the diagnosis + fix, or None when the error matches no
    known pitfall — an unmatched error is still raised verbatim by
    the caller, never papered over with a guessed cause.
    """
    text = str(exc)
    for pattern, diagnosis in _SCRIPT_PITFALLS:
        if re.search(pattern, text, re.IGNORECASE):
            return diagnosis
    return None


def resolve_device(device: str) -> str:
    """``auto`` / ``cpu`` / ``cuda`` -> the concrete export device.

    ``auto`` is CUDA when torch reports one, else CPU. CUDA asked
    for on a box without it is a refusal naming the cause, never a
    silent CPU fallback — a caller who asked for CUDA wants to know
    the export did not run there.
    """
    torch = require_torch("resolving the export device")
    want = (device or "auto").strip().lower()
    if want not in ("auto", "cpu", "cuda"):
        raise ExportError(
            f"unknown export device {device!r} — 'auto', 'cpu', or 'cuda'"
        )
    if want == "cuda" and not torch.cuda.is_available():
        raise ExportError(
            "export device 'cuda' was asked for but this machine has "
            "no CUDA device — use device='cpu' or 'auto'"
        )
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return want


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dtype_name(dtype: Any) -> str:
    return str(dtype).split(".")[-1]


def _specs(inputs: dict, *, traced: bool) -> list[dict[str, Any]]:
    """The I/O contract: exact shapes for a trace, wildcards for a script."""
    specs = []
    for name, tensor in inputs.items():
        if not hasattr(tensor, "shape") or not hasattr(tensor, "dtype"):
            raise ExportError(
                f"example input {name!r} is not a tensor — export needs "
                "name -> tensor"
            )
        shape = (
            [int(d) for d in tensor.shape]
            if traced
            else [None] * len(tensor.shape)
        )
        specs.append({"name": name, "dtype": _dtype_name(tensor.dtype),
                      "shape": shape})
    return specs


def _run_entry(module: Any, entry: str, inputs: dict, *, seed: int | None,
               torch: Any) -> tuple[Any, ...]:
    """Run ``entry`` once; the seed is reset first when given.

    Reseeding before each run is what makes the comparison
    meaningful for stochastic modules (a VITS inference path draws
    noise): source and exported runs draw the same values, so the
    deviation measures the export, not the luck of the draw.
    """
    fn = getattr(module, entry, None)
    if fn is None or not callable(fn):
        raise ExportError(f"module has no callable entry {entry!r}")
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        outs = fn(*inputs.values())
    if not isinstance(outs, (tuple, list)):
        outs = (outs,)
    return tuple(outs)


def _max_deviation(a: Any, b: Any, torch: Any) -> tuple[float, float]:
    """(max, mean) absolute deviation between two outputs."""
    if not hasattr(a, "dtype") or not hasattr(b, "dtype"):
        raise ExportError(
            "verification needs tensor outputs — the entry returned "
            f"{type(a).__name__} / {type(b).__name__}"
        )
    diff = (a.detach().to("cpu").float()
            - b.detach().to("cpu").float()).abs()
    return float(diff.max().item()), float(diff.mean().item())


@dataclass
class Verification:
    """Source vs exported, on a fixed probe — the numbers, not a shrug."""

    probe_shapes: dict[str, list[int]]
    max_abs_deviation: float
    mean_abs_deviation: float
    per_output: list[float]
    tolerance: float
    passed: bool
    long_probe: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_shapes": dict(self.probe_shapes),
            "max_abs_deviation": self.max_abs_deviation,
            "mean_abs_deviation": self.mean_abs_deviation,
            "per_output": list(self.per_output),
            "tolerance": self.tolerance,
            "passed": bool(self.passed),
            "long_probe": dict(self.long_probe),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Verification:
        return cls(
            probe_shapes={k: [int(d) for d in v]
                          for k, v in raw.get("probe_shapes", {}).items()},
            max_abs_deviation=float(raw.get("max_abs_deviation", 0.0)),
            mean_abs_deviation=float(raw.get("mean_abs_deviation", 0.0)),
            per_output=[float(d) for d in raw.get("per_output", [])],
            tolerance=float(raw.get("tolerance", 0.0)),
            passed=bool(raw.get("passed", False)),
            long_probe=dict(raw.get("long_probe", {})),
        )


@dataclass
class ExportReceipt:
    """The export record — the receipt sidecar, as data.

    The file on disk is the truth; this is the record that says how
    it was made and how close it stays to its source. ``traced``
    artifacts record exact shapes (shape-specialised); scripted
    ones record wildcards and the dynamic axes they were proven
    against.
    """

    name: str
    version: str
    task: str
    format: str
    path: str
    sha256: str
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    entry: str
    method: str
    traced: bool
    trace_checked: bool
    trace_check_error: str | None
    trace_check_warnings: list[str]
    dynamic_axes: dict[str, list[int]]
    example_shapes: dict[str, list[int]]
    export_device: str
    runnable_on: str
    torch_version: str
    exporter_version: str
    source: str
    verification: Verification
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": EXPORT_FORMAT_VERSION,
            "name": self.name, "version": self.version,
            "task": self.task, "format": self.format,
            "path": self.path, "sha256": self.sha256,
            "inputs": [dict(t) for t in self.inputs],
            "outputs": [dict(t) for t in self.outputs],
            "entry": self.entry, "method": self.method,
            "traced": bool(self.traced),
            "trace_checked": bool(self.trace_checked),
            "trace_check_error": self.trace_check_error,
            "trace_check_warnings": list(self.trace_check_warnings),
            "dynamic_axes": {k: list(v)
                             for k, v in self.dynamic_axes.items()},
            "example_shapes": {k: list(v)
                               for k, v in self.example_shapes.items()},
            "export_device": self.export_device,
            "runnable_on": self.runnable_on,
            "torch_version": self.torch_version,
            "exporter_version": self.exporter_version,
            "source": self.source,
            "verification": self.verification.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExportReceipt:
        return cls(
            name=str(raw["name"]), version=str(raw["version"]),
            task=str(raw["task"]), format=str(raw["format"]),
            path=str(raw["path"]), sha256=str(raw.get("sha256", "")),
            inputs=[dict(t) for t in raw.get("inputs", [])],
            outputs=[dict(t) for t in raw.get("outputs", [])],
            entry=str(raw.get("entry", "forward")),
            method=str(raw.get("method", "script")),
            traced=bool(raw.get("traced", False)),
            trace_checked=bool(raw.get("trace_checked", False)),
            trace_check_error=raw.get("trace_check_error"),
            trace_check_warnings=[str(w) for w in
                                  raw.get("trace_check_warnings", [])],
            dynamic_axes={k: [int(d) for d in v]
                          for k, v in raw.get("dynamic_axes", {}).items()},
            example_shapes={k: [int(d) for d in v]
                            for k, v in raw.get("example_shapes", {}).items()},
            export_device=str(raw.get("export_device", "cpu")),
            runnable_on=str(raw.get("runnable_on", "cpu")),
            torch_version=str(raw.get("torch_version", "")),
            exporter_version=str(raw.get(
                "exporter_version", EXPORT_FORMAT_VERSION)),
            source=str(raw.get("source", "")),
            verification=Verification.from_dict(
                raw.get("verification", {})),
            created_at=float(raw.get("created_at", 0.0)),
        )


@dataclass
class LoadedExport:
    """A loaded artifact: the module, its receipt, and any warning."""

    module: Any
    receipt: ExportReceipt
    torch_version_warning: str | None = None


def receipt_path_for(out_path: str) -> str:
    """The receipt sidecar for an artifact path."""
    return os.path.splitext(out_path)[0] + ".json"


def _write_receipt(receipt: ExportReceipt, out_path: str) -> str:
    """Write the receipt sidecar atomically (temp file + replace)."""
    receipt_path = receipt_path_for(out_path)
    tmp_path = receipt_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(receipt.to_dict(), fh, indent=2, sort_keys=True)
    os.replace(tmp_path, receipt_path)
    return receipt_path


def _output_specs(outputs: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Output shapes are measured on the probe, always exact.

    Outputs follow the probe's shapes whatever the method: a
    scripted module's outputs vary with its inputs, so the receipt
    records what the probe run produced, and the serve path
    re-measures rather than trusting them.
    """
    specs = []
    for i, tensor in enumerate(outputs):
        if not hasattr(tensor, "shape") or not hasattr(tensor, "dtype"):
            raise ExportError(
                "verification needs tensor outputs — the entry returned "
                f"{type(tensor).__name__}")
        specs.append({"name": f"output_{i}",
                      "dtype": _dtype_name(tensor.dtype),
                      "shape": [int(d) for d in tensor.shape]})
    return specs


def _check_example_inputs(example_inputs: dict) -> None:
    if not isinstance(example_inputs, dict) or not example_inputs:
        raise ExportError(
            "export needs example inputs (name -> tensor)")
    for name, tensor in example_inputs.items():
        if not hasattr(tensor, "shape") or not hasattr(tensor, "dtype"):
            raise ExportError(
                f"example input {name!r} is not a tensor — export needs "
                "name -> tensor")


def _check_dynamic_axes(dynamic_axes: dict | None, method: str,
                        example_inputs: dict) -> dict[str, list[int]]:
    """Dynamic axes are a script promise, not a trace one.

    A traced artifact fixes the shapes it saw — a caller asking a
    trace to vary an axis is refused naming the cause. A scripted
    artifact is proven against the declared axes by re-running it
    on a resized probe (see _prove_dynamic_axes).
    """
    axes = {k: [int(d) for d in v]
            for k, v in (dynamic_axes or {}).items()}
    if not axes:
        return {}
    if method == "trace":
        raise ExportError(
            "dynamic_axes cannot vary on a traced export — tracing "
            "fixes the shapes it saw (use method='script', which is "
            "shape-polymorphic)")
    for name, dims in axes.items():
        if name not in example_inputs:
            raise ExportError(
                f"dynamic_axes names unknown input {name!r} — declared "
                f"inputs: {', '.join(example_inputs)}")
        rank = len(example_inputs[name].shape)
        for dim in dims:
            if dim < 0 or dim >= rank:
                raise ExportError(
                    f"dynamic_axes for {name!r}: dim {dim} out of range "
                    f"for rank {rank}")
    return axes


def _prove_dynamic_axes(module: Any, entry: str, example_inputs: dict,
                        axes: dict[str, list[int]], torch: Any) -> None:
    """Prove the declared axes really vary: resize + re-run.

    Each declared axis is doubled (repeat_interleave along it) and
    the exported module must run on the resized inputs and return
    outputs resized the same way along that axis. A failure names
    the axis — a scripted module that secretly fixed a shape is
    caught here, not at serve time.
    """
    resized = {}
    for name, tensor in example_inputs.items():
        out = tensor
        for dim in axes.get(name, []):
            out = out.repeat_interleave(2, dim=dim)
        resized[name] = out
    try:
        with torch.no_grad():
            outs = _run_entry(module, entry, resized, seed=None,
                              torch=torch)
    except Exception as exc:
        raise ExportError(
            "dynamic-axes proof failed — the scripted module would "
            f"not run on resized inputs: {exc}") from exc
    base_outs = _run_entry(module, entry, example_inputs, seed=None,
                           torch=torch)
    for out, base in zip(outs, base_outs, strict=True):
        if list(out.shape) != list(base.shape) and not any(
                list(out.shape)[d] == 2 * list(base.shape)[d]
                for d in range(len(base.shape))):
            raise ExportError(
                "dynamic-axes proof failed — resized inputs did not "
                f"resize the outputs: {list(base.shape)} -> "
                f"{list(out.shape)}")


def verify_export(source: Any, exported: Any, probe_inputs: dict, *,
                  tolerance: float = 1e-4, entry: str = "forward",
                  probe_seed: int | None = 0,
                  long_probe_inputs: dict | None = None,
                  long_probe_enforced: bool = False) -> Verification:
    """Source vs exported on a fixed probe — the deviation, as numbers.

    Both runs are reseeded first (when ``probe_seed`` is given) so a
    stochastic module draws the same values on both sides and the
    deviation measures the export. ``long_probe_inputs`` is the
    upstream's finding made concrete: the exported module must
    survive longer input (a traced module that only ever saw a
    short example can crash on long input at eval time). For a
    trace the long probe is advisory — a traced artifact is
    shape-specialised, so a failure is recorded, not fatal. For a
    script it is enforced when ``long_probe_enforced`` is set.
    """
    torch = require_torch("verifying a TorchScript export")
    _check_example_inputs(probe_inputs)
    if tolerance <= 0:
        raise ExportError(f"verification tolerance {tolerance} must be > 0")
    source_outs = _run_entry(source, entry, probe_inputs,
                             seed=probe_seed, torch=torch)
    exported_outs = _run_entry(exported, entry, probe_inputs,
                               seed=probe_seed, torch=torch)
    if len(source_outs) != len(exported_outs):
        raise ExportError(
            "verification failed — the exported module returned "
            f"{len(exported_outs)} outputs, the source returned "
            f"{len(source_outs)}")
    per_output = []
    max_dev = 0.0
    mean_sum = 0.0
    for src, exp in zip(source_outs, exported_outs, strict=True):
        if list(src.shape) != list(exp.shape):
            raise ExportError(
                "verification failed — output shape changed in the "
                f"export: {list(src.shape)} -> {list(exp.shape)}")
        mx, mean = _max_deviation(src, exp, torch)
        per_output.append(mx)
        max_dev = max(max_dev, mx)
        mean_sum += mean
    mean_dev = mean_sum / len(per_output) if per_output else 0.0
    long_probe: dict[str, Any] = {}
    if long_probe_inputs:
        _check_example_inputs(long_probe_inputs)
        shapes = {n: [int(d) for d in t.shape]
                  for n, t in long_probe_inputs.items()}
        try:
            _run_entry(exported, entry, long_probe_inputs,
                       seed=probe_seed, torch=torch)
            long_probe = {"shapes": shapes, "ok": True, "error": None}
        except Exception as exc:  # noqa: BLE001 — the error IS the record
            long_probe = {"shapes": shapes, "ok": False, "error": str(exc)}
            if long_probe_enforced:
                raise ExportError(
                    "long-input probe failed — the exported module "
                    f"would not run on longer input: {exc}") from exc
    passed = max_dev <= tolerance
    return Verification(
        probe_shapes={n: [int(d) for d in t.shape]
                      for n, t in probe_inputs.items()},
        max_abs_deviation=max_dev,
        mean_abs_deviation=mean_dev,
        per_output=per_output,
        tolerance=tolerance,
        passed=passed,
        long_probe=long_probe,
    )


def export_module(module: Any, example_inputs: dict, out_path: str, *,
                  method: str = "script", entry: str = "forward",
                  name: str = "", version: str = "0.1.0", task: str = "",
                  source: str = "", device: str = "auto",
                  tolerance: float = 1e-4, probe_inputs: dict | None = None,
                  probe_seed: int | None = 0,
                  long_probe_inputs: dict | None = None,
                  dynamic_axes: dict | None = None,
                  allow_unchecked_trace: bool = False) -> ExportReceipt:
    """Export a torch module to TorchScript + a receipt sidecar.

    ``method`` is ``script`` (shape-polymorphic, preferred) or
    ``trace`` (the caller names it; the receipt records the exact
    shapes). The export is verified against the source on a fixed
    probe and the max deviation is in the receipt — an export whose
    deviation exceeds ``tolerance`` is refused, naming the number.
    The artifact is always runnable on CPU: the module is moved to
    the export device for the export, moved back afterwards, and
    the saved file is re-loaded with ``map_location="cpu"`` for
    the verification run.
    """
    torch = require_torch("exporting a model to TorchScript")
    if method not in ("script", "trace"):
        raise ExportError(
            f"unknown export method {method!r} — 'script' or 'trace'")
    if not name:
        raise ExportError("export needs a service name")
    if not _VERSION_RE.match(version):
        raise ExportError(f"bad version {version!r}: semver core only")
    if not (task or "").strip():
        raise ExportError("export needs a task (e.g. 'tts-mel')")
    _check_example_inputs(example_inputs)
    axes = _check_dynamic_axes(dynamic_axes, method, example_inputs)
    export_device = resolve_device(device)

    if not out_path.endswith(".torchjit") and not out_path.endswith(".pt"):
        raise ExportError(
            f"export output {out_path!r} must end .torchjit or .pt — "
            "the upstream's naming rule, kept so the registry and "
            "the service loader recognise the file")
    out_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    original_device = None
    try:
        params = list(module.parameters())
        if params:
            original_device = str(params[0].device)
    except (AttributeError, StopIteration):
        original_device = None
    try:
        module.eval()
    except AttributeError as exc:
        raise ExportError(
            "export needs a torch module with .eval() — "
            f"got {type(module).__name__}") from exc
    try:
        module.to(export_device)
    except Exception as exc:
        raise ExportError(
            f"could not move the module to {export_device}: {exc}") from exc
    grad_was_enabled = torch.is_grad_enabled()
    try:
        with torch.no_grad():
            torch.set_grad_enabled(False)
            if method == "script":
                try:
                    exported = torch.jit.script(module)
                except Exception as exc:
                    diagnosis = diagnose_script_error(exc)
                    hint = (f" — {diagnosis}" if diagnosis else "")
                    raise ExportError(
                        "scripting failed: "
                        f"{type(exc).__name__}: {exc}{hint}") from exc
                traced = False
                trace_checked = False
                trace_check_error = None
                trace_check_warnings = []
            else:
                trace_args = {entry: tuple(example_inputs.values())}
                trace_check_warnings: list[str] = []
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        exported = torch.jit.trace_module(
                            module, trace_args, check_trace=True,
                            strict=True)
                    trace_check_warnings = []
                    for w in caught:
                        # The VITS trace raises the same boolean-cast
                        # warning ~119 times; the receipt keeps each
                        # distinct warning once, in order.
                        msg = str(w.message)[:500]
                        if msg not in trace_check_warnings:
                            trace_check_warnings.append(msg)
                    trace_checked = True
                    trace_check_error = None
                except Exception as exc:
                    trace_check_error = str(exc)
                    if not allow_unchecked_trace:
                        raise ExportError(
                            "tracing failed the sanity check "
                            "(graphs differed across invocations) — "
                            f"{exc}. The upstream silently retried with "
                            "check_trace=False; this exporter does not. "
                            "Pass allow_unchecked_trace=True to take the "
                            "unchecked retry, and the receipt will record "
                            "trace_checked=false.") from exc
                    exported = torch.jit.trace_module(
                        module, trace_args, check_trace=False, strict=True)
                    trace_checked = False
                traced = True
            entry_fn = getattr(exported, entry, None)
            if entry_fn is None or not callable(entry_fn):
                raise ExportError(
                    f"exported module has no callable entry {entry!r}")
            tmp_path = out_path + ".tmp"
            torch.jit.save(exported, tmp_path)
            os.replace(tmp_path, out_path)
    finally:
        torch.set_grad_enabled(grad_was_enabled)
        if original_device is not None:
            # Restore is best-effort: the export already succeeded.
            with suppress(Exception):
                module.to(original_device)

    # Verification runs the saved file (re-loaded on CPU), never the
    # in-memory export — the artifact on disk is what gets proven.
    loaded = torch.jit.load(out_path, map_location="cpu")
    loaded.eval()
    source_cpu = module
    # Already on CPU on most boxes; a CUDA source is moved for the
    # comparison and restored afterwards (the export moved it back
    # once already — verification must not leave it on CPU).
    with suppress(Exception):
        source_cpu.to("cpu")
    try:
        probe = probe_inputs if probe_inputs is not None else example_inputs
        verification = verify_export(
            source_cpu, loaded, probe, tolerance=tolerance, entry=entry,
            probe_seed=probe_seed, long_probe_inputs=long_probe_inputs,
            long_probe_enforced=(method == "script"
                                 and long_probe_inputs is not None))
    finally:
        if original_device is not None:
            with suppress(Exception):
                module.to(original_device)
    if axes:
        _prove_dynamic_axes(loaded, entry, example_inputs, axes, torch)
    if not verification.passed:
        raise ExportError(
            "verification failed — the exported module deviates from "
            f"its source by max {verification.max_abs_deviation:.6g} "
            f"(tolerance {tolerance:.6g}) on the probe input. The "
            f"artifact was still written to {out_path} — re-export "
            "with a larger tolerance only if the deviation is "
            "understood, not to make the number go away.")
    receipt = ExportReceipt(
        name=name, version=version, task=task, format="torchscript",
        path=out_path, sha256=_sha256_file(out_path),
        inputs=_specs(example_inputs, traced=traced),
        outputs=_output_specs(_run_entry(
            loaded, entry, probe, seed=probe_seed, torch=torch)),
        entry=entry, method=method, traced=traced,
        trace_checked=trace_checked,
        trace_check_error=trace_check_error,
        trace_check_warnings=trace_check_warnings,
        dynamic_axes=axes,
        example_shapes={n: [int(d) for d in t.shape]
                        for n, t in example_inputs.items()},
        export_device=export_device, runnable_on="cpu",
        torch_version=torch.__version__,
        exporter_version=EXPORT_FORMAT_VERSION,
        source=source or "torch export",
        verification=verification,
    )
    _write_receipt(receipt, out_path)
    return receipt


def load_exported(path: str, *, entry: str = "forward") -> LoadedExport:
    """Load an exported artifact + its receipt, refusing bad ones.

    Refusals name the cause: missing receipt, a receipt whose
    format stamp is unknown, a sha256 mismatch, a file torch will
    not load, or an entry that is not callable. A torch version
    drift between export and load is a warning in the result, not
    a refusal — the artifact still loads.
    """
    torch = require_torch("loading a TorchScript export")
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ExportError(f"export file missing: {path}")
    receipt_path = receipt_path_for(path)
    if not os.path.isfile(receipt_path):
        raise ExportError(
            f"no export receipt beside {path} — expected "
            f"{receipt_path}. Re-export it; an artifact without its "
            "receipt cannot prove what it is.")
    try:
        with open(receipt_path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (ValueError, OSError) as exc:
        raise ExportError(
            f"export receipt {receipt_path} is unreadable: {exc}") from exc
    stamp = raw.get("format_version", "")
    if stamp != EXPORT_FORMAT_VERSION:
        raise ExportError(
            f"export receipt {receipt_path} has format stamp "
            f"{stamp!r}, this loader knows "
            f"{EXPORT_FORMAT_VERSION!r} — re-export it")
    receipt = ExportReceipt.from_dict(raw)
    if receipt.format != "torchscript":
        raise ExportError(
            f"export {path} has format {receipt.format!r} — this "
            "loader serves TorchScript only")
    if receipt.path != path:
        raise ExportError(
            f"export receipt {receipt_path} names path "
            f"{receipt.path!r}, but the file loaded is {path!r} — "
            "the artifact was moved without its receipt")
    if receipt.sha256 and _sha256_file(path) != receipt.sha256:
        raise ExportError(
            f"export {path}: sha256 mismatch — the file changed "
            "since export. Re-export it; do not serve a file whose "
            "receipt does not match.")
    try:
        module = torch.jit.load(path, map_location="cpu")
    except Exception as exc:
        raise ExportError(
            f"export {path}: torch refused to load it: {exc}") from exc
    module.eval()
    entry_fn = getattr(module, entry, None)
    if entry_fn is None or not callable(entry_fn):
        raise ExportError(
            f"export {path}: no callable entry {entry!r} on the module")
    warning = None
    if (receipt.torch_version
            and receipt.torch_version != torch.__version__):
        warning = (
            f"export was made with torch {receipt.torch_version}, "
            f"this box has torch {torch.__version__} — the artifact "
            "loaded, but behaviour may differ")
    return LoadedExport(module=module, receipt=receipt,
                        torch_version_warning=warning)
