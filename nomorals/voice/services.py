"""Model services: train-in-Python -> export -> register -> serve.

The storyteller-ml pattern (census §3B.11), mined, not ported — no
upstream code or text lives here. The upstream monorepo is ~16
containerised model services behind a Rust control plane: worker pods
("jobs") pull inference jobs from queues and run them in an in-container
Python server that LRU-caches models in memory, and the native side loads
a trained model file by path behind a fixed load/execute/unload seam with
an explicit tensor contract (named input/output tensors + element counts).

Devon's realisation, inside the voice organ:

- :class:`ModelArtifact` — a versioned, servable artifact (TorchScript
  first): path, sha256, explicit input/output tensor spec, source.
- :class:`ServiceDeclaration` + :class:`ContainerSpec` — the service and
  its container shape AS DATA (ghcr.io refs, version + short-SHA tags,
  build context, run-number label), the docker-publish workflow shape
  without requiring Docker here.
- :class:`ServiceRegistry` — the JSON store (``services.json``, atomic
  write), ``slug@version`` refs, ``latest`` resolution.
- :func:`export_torchscript_service` — the export rung: a torch module +
  example inputs -> TorchScript file + artifact record.
- :class:`ModelServer` — the serving rung: the NAL shape
  (load/execute/unload) over the existing
  :class:`~nomorals.voice.lifecycle.ModelLifecycleManager` LRU cache.
- :func:`service_executor` — the worker-pod shape: a callable for
  :class:`nomorals.core.inference_jobs.InferenceJobStore.drive`.
- :func:`export_vits_probe_service` — one real end-to-end service: the
  real VITS architecture with deterministic random weights, traced,
  registered, and served through the registry + server.

Stdlib only at import time; torch is lazy (``require_torch`` names it
when a rung needs it). No new brain tool: the CLI (``nm voice service``)
and chat (``/voice service``) verbs reach the whole surface.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ContainerSpec",
    "ModelArtifact",
    "ModelServer",
    "ServiceDeclaration",
    "ServiceRegistry",
    "TensorSpec",
    "control_service",
    "default_services_dir",
    "export_torchscript_service",
    "export_vits_probe_service",
    "health",
    "service_executor",
]

#: Slug rule: lowercase, hyphen-separated, no leading/trailing hyphen
#: (the upstream service slugs: gpt-sovits, so-vits-svc-v2, mocapnet).
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
#: Versions are semver core only: 0.1.0, 1.2.3. No pre-release tags.
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

#: Torch dtypes the serve rung can build from a JSON payload.
_TORCH_DTYPES = ("float32", "float64", "int64", "int32", "int16",
                 "uint8", "bool")


def _semver_key(version: str) -> tuple[int, int, int]:
    return tuple(int(p) for p in version.split("."))


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dtype_name(dtype: Any) -> str:
    text = str(dtype)
    return text[6:] if text.startswith("torch.") else text


def _coerce_shape(shape: Any) -> list:
    if shape is None:
        return []
    return [None if d is None else int(d) for d in shape]


@dataclass(frozen=True)
class TensorSpec:
    """One named tensor in an artifact's explicit I/O contract.

    ``shape`` dims are exact sizes, except ``None`` which is a wildcard
    (scripted artifacts are shape-polymorphic; traced artifacts record
    exact shapes and serve refuses anything else).
    """

    name: str
    dtype: str
    shape: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype,
                "shape": list(self.shape)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TensorSpec:
        return cls(name=str(raw["name"]), dtype=str(raw["dtype"]),
                   shape=_coerce_shape(raw.get("shape")))

    def matches(self, dtype: str, shape: list) -> str | None:
        """None when a (dtype, shape) fits; else the mismatch, named."""
        if dtype != self.dtype:
            return (f"tensor {self.name!r}: dtype {dtype!r} != "
                    f"declared {self.dtype!r}")
        if len(shape) != len(self.shape):
            return (f"tensor {self.name!r}: rank {len(shape)} != "
                    f"declared {len(self.shape)}")
        for got, want in zip(shape, self.shape, strict=True):
            if want is not None and got != want:
                return (f"tensor {self.name!r}: shape {list(shape)} != "
                        f"declared {list(self.shape)}")
        return None


@dataclass
class ModelArtifact:
    """A versioned, servable model artifact.

    The file on disk is the truth; the record is the receipt. ``source``
    names where the weights came from (a train run, a checkpoint, or —
    for the probe service — random weights with the seed).
    """

    name: str
    version: str
    task: str
    format: str  # "torchscript" today; "state_dict" / "onnx" later
    path: str
    sha256: str = ""
    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)
    entry: str = "forward"
    traced: bool = False
    source: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "version": self.version, "task": self.task,
            "format": self.format, "path": self.path,
            "sha256": self.sha256,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "entry": self.entry, "traced": bool(self.traced),
            "source": self.source, "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelArtifact:
        return cls(
            name=str(raw["name"]), version=str(raw["version"]),
            task=str(raw["task"]), format=str(raw["format"]),
            path=str(raw["path"]), sha256=str(raw.get("sha256", "")),
            inputs=[TensorSpec.from_dict(t) for t in raw.get("inputs", [])],
            outputs=[TensorSpec.from_dict(t) for t in raw.get("outputs", [])],
            entry=str(raw.get("entry", "forward")),
            traced=bool(raw.get("traced", False)),
            source=str(raw.get("source", "")),
            created_at=float(raw.get("created_at", 0.0)),
        )

    def validate(self) -> None:
        """Prove the artifact on disk matches its receipt.

        Existence + sha256 always; format sniff always; a TorchScript
        artifact also loads under torch when torch is present (a file
        that will not load is refused at registration, not at serve
        time). Raises the first failure, naming it.
        """
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"artifact file missing: {self.path}")
        if not self.sha256:
            raise ValueError(f"artifact {self.name}@{self.version} "
                             "carries no sha256 — re-export it")
        if _sha256_file(self.path) != self.sha256:
            raise ValueError(f"artifact {self.name}@{self.version}: "
                             f"sha256 mismatch at {self.path}")
        if self.format == "torchscript":
            if not zipfile.is_zipfile(self.path):
                raise ValueError(
                    f"artifact {self.name}@{self.version}: not a "
                    "TorchScript zip file")
            from .vits._torch import require_torch, torch_available
            if torch_available():
                torch = require_torch("validating a TorchScript artifact")
                try:
                    module = torch.jit.load(self.path, map_location="cpu")
                except Exception as exc:
                    raise ValueError(
                        f"artifact {self.name}@{self.version}: torch "
                        f"refused to load it: {exc}") from exc
                entry = getattr(module, self.entry, None)
                if entry is None or not callable(entry):
                    raise ValueError(
                        f"artifact {self.name}@{self.version}: no "
                        f"callable entry {self.entry!r} on the module")
        elif self.format not in ("state_dict", "onnx"):
            raise ValueError(f"artifact {self.name}@{self.version}: "
                             f"unknown format {self.format!r}")


@dataclass(frozen=True)
class ContainerSpec:
    """The container shape AS DATA — the docker-publish workflow shape.

    No image is built here and no Docker daemon is needed: this is the
    declaration a build would consume. ``image_refs`` mirrors the
    upstream workflow exactly: the version tag plus the short-SHA tag.
    """

    base_image: str = "python:3.11-slim"
    owner: str = ""
    build_context: str = ""
    run_label: str = ""

    def image_name(self, slug: str) -> str:
        if self.owner:
            return f"ghcr.io/{self.owner}/{slug}"
        return f"local/{slug}"

    def image_refs(self, slug: str, version: str,
                   short_sha: str) -> list[str]:
        """[version tag, short-SHA tag], lowercased, like the workflow."""
        image = self.image_name(slug).lower()
        return [f"{image}:{version.lower()}",
                f"{image}:{short_sha.lower()}"]

    def to_dict(self) -> dict[str, Any]:
        return {"base_image": self.base_image, "owner": self.owner,
                "build_context": self.build_context,
                "run_label": self.run_label}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ContainerSpec:
        return cls(base_image=str(raw.get("base_image", "python:3.11-slim")),
                   owner=str(raw.get("owner", "")),
                   build_context=str(raw.get("build_context", "")),
                   run_label=str(raw.get("run_label", "")))


@dataclass
class ServiceDeclaration:
    """One servable model service: slug + version + task + artifact."""

    slug: str
    version: str
    task: str
    artifact: ModelArtifact
    runtime: str = "python-torch"
    container: ContainerSpec = field(default_factory=ContainerSpec)
    health_probe: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def validate(self) -> None:
        if not _SLUG_RE.match(self.slug):
            raise ValueError(
                f"bad service slug {self.slug!r}: lowercase, "
                "hyphen-separated, e.g. 'vits-probe'")
        if not _VERSION_RE.match(self.version):
            raise ValueError(
                f"bad service version {self.version!r}: semver core "
                "only, e.g. '0.1.0'")
        if not (self.task or "").strip():
            raise ValueError("a service needs a task (e.g. 'tts-mel')")
        if self.artifact.name != self.slug:
            raise ValueError(
                f"artifact name {self.artifact.name!r} != service slug "
                f"{self.slug!r}")
        if self.artifact.version != self.version:
            raise ValueError(
                f"artifact version {self.artifact.version!r} != service "
                f"version {self.version!r}")
        if self.artifact.task != self.task:
            raise ValueError(
                f"artifact task {self.artifact.task!r} != service task "
                f"{self.task!r}")
        self.artifact.validate()

    @property
    def ref(self) -> str:
        return f"{self.slug}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug, "version": self.version, "task": self.task,
            "artifact": self.artifact.to_dict(), "runtime": self.runtime,
            "container": self.container.to_dict(),
            "health_probe": self.health_probe,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ServiceDeclaration:
        return cls(
            slug=str(raw["slug"]), version=str(raw["version"]),
            task=str(raw["task"]),
            artifact=ModelArtifact.from_dict(raw["artifact"]),
            runtime=str(raw.get("runtime", "python-torch")),
            container=ContainerSpec.from_dict(raw.get("container", {})),
            health_probe=dict(raw.get("health_probe", {})),
            created_at=float(raw.get("created_at", 0.0)),
        )


def default_services_dir() -> str:
    """Canonical service store: env override, else user-level default."""
    override = os.environ.get("NOMORALS_VOICE_SERVICES_DIR", "").strip()
    if override:
        return os.path.expanduser(override)
    return os.path.join(os.path.expanduser("~"), ".config",
                        "nomorals", "voice", "services")


class ServiceRegistry:
    """The service store — ``services.json`` next to the artifacts.

    Same persistence pattern as the voice catalogue: one JSON file,
    atomic tmp+replace writes. A corrupt store raises naming the file
    instead of silently dropping services.
    """

    def __init__(self, services_dir: str = "") -> None:
        self.services_dir = services_dir or default_services_dir()
        self._path = os.path.join(self.services_dir, "services.json")
        self._services: dict[str, ServiceDeclaration] = {}
        self._load()

    # -- persistence --------------------------------------------------

    def _load(self) -> None:
        if not os.path.isfile(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as fh:
                raw = json.load(fh) or {}
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"service store corrupt at {self._path}: {exc}") from exc
        for item in raw.get("services", []):
            decl = ServiceDeclaration.from_dict(item)
            self._services[decl.ref] = decl

    def _save(self) -> None:
        os.makedirs(self.services_dir, exist_ok=True)
        payload = {"services": [d.to_dict()
                                for d in self._services.values()]}
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, self._path)

    # -- registry -----------------------------------------------------

    def register(self, declaration: ServiceDeclaration) -> ServiceDeclaration:
        """Register a service; the artifact must validate first."""
        declaration.validate()
        if declaration.ref in self._services:
            raise ValueError(
                f"service {declaration.ref} already registered — "
                "bump the version for a new artifact")
        self._services[declaration.ref] = declaration
        self._save()
        return declaration

    def resolve(self, ref: str) -> ServiceDeclaration:
        """``slug`` (latest) or ``slug@version`` — else KeyError, named."""
        ref = (ref or "").strip()
        if "@" in ref:
            slug, version = ref.split("@", 1)
            key = f"{slug}@{version}"
            try:
                return self._services[key]
            except KeyError:
                raise KeyError(f"no service {key!r} registered") from None
        slug = ref
        matches = [d for d in self._services.values() if d.slug == slug]
        if not matches:
            raise KeyError(f"no service {slug!r} registered")
        return max(matches, key=lambda d: _semver_key(d.version))

    def get(self, slug: str, version: str = "") -> ServiceDeclaration:
        if version:
            return self.resolve(f"{slug}@{version}")
        return self.resolve(slug)

    def list(self, task: str = "") -> list[ServiceDeclaration]:
        rows = list(self._services.values())
        if task:
            rows = [d for d in rows if d.task == task]
        rows.sort(key=lambda d: (d.slug, _semver_key(d.version)))
        return rows

    def remove(self, slug: str, version: str) -> ServiceDeclaration:
        """Drop the declaration. The artifact file stays — removal drops
        the registration, never the evidence."""
        key = f"{slug}@{version}"
        try:
            decl = self._services.pop(key)
        except KeyError:
            raise KeyError(f"no service {key!r} registered") from None
        self._save()
        return decl


# -- export rung -------------------------------------------------------

def _tensor_payload(value: Any) -> dict[str, Any]:
    """A tensor as a JSON-able payload: dtype + shape + nested values."""
    return {"dtype": _dtype_name(value.dtype),
            "shape": [int(d) for d in value.shape],
            "values": value.detach().cpu().tolist()}


def _payload_to_tensor(payload: Any) -> Any:
    """A JSON payload (or a bare tensor) back into a torch tensor."""
    from .vits._torch import require_torch
    torch = require_torch("serving a model service")
    if hasattr(payload, "tolist") and hasattr(payload, "dtype"):
        return payload
    if not isinstance(payload, dict) or "values" not in payload:
        raise ValueError("each input must be a tensor or a "
                         "{dtype, shape, values} payload")
    dtype = str(payload.get("dtype", "float32"))
    if dtype not in _TORCH_DTYPES:
        raise ValueError(f"unsupported input dtype {dtype!r} — "
                         f"one of {', '.join(_TORCH_DTYPES)}")
    torch_dtype = getattr(torch, dtype)
    values = payload["values"]
    tensor = torch.tensor(values, dtype=torch_dtype)
    want = payload.get("shape")
    if want is not None and list(tensor.shape) != [int(d) for d in want]:
        raise ValueError(f"input shape {list(tensor.shape)} != declared "
                         f"shape {list(want)}")
    return tensor


def export_torchscript_service(module: Any, example_inputs: dict,
                               out_path: str, *, method: str = "script",
                               entry: str = "forward", task: str = "",
                               name: str = "", version: str = "0.1.0",
                               source: str = "") -> ModelArtifact:
    """Export a torch module to a TorchScript artifact + record.

    ``example_inputs`` is an ordered dict of name -> tensor. ``method``
    is ``script`` (shape-polymorphic, preferred) or ``trace`` (the
    caller names it; traced artifacts are shape-specialised and the
    artifact records the exact shapes). The exported module is run once
    on the example inputs so the output spec is measured, not guessed.
    """
    from .vits._torch import require_torch
    torch = require_torch("exporting a model service to TorchScript")
    if method not in ("script", "trace"):
        raise ValueError(f"unknown export method {method!r} — "
                         "'script' or 'trace'")
    if not example_inputs:
        raise ValueError("export needs example inputs (name -> tensor)")
    if not name:
        raise ValueError("export needs a service name")
    if not _VERSION_RE.match(version):
        raise ValueError(f"bad version {version!r}: semver core only")
    if not (task or "").strip():
        raise ValueError("export needs a task (e.g. 'tts-mel')")

    if method == "script":
        scripted = torch.jit.script(module)
        entry_fn = getattr(scripted, entry, None)
        if entry_fn is None or not callable(entry_fn):
            raise ValueError(f"scripted module has no entry {entry!r}")
    else:
        traced = torch.jit.trace_module(
            module, {entry: tuple(example_inputs.values())},
            check_trace=False, strict=True)
        scripted = traced
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    torch.jit.save(scripted, out_path)

    with torch.no_grad():
        outs = getattr(scripted, entry)(*example_inputs.values())
    if not isinstance(outs, (tuple, list)):
        outs = (outs,)
    inputs = [TensorSpec(name=n, dtype=_dtype_name(t.dtype),
                         shape=([None] * len(t.shape)
                                if method == "script"
                                else [int(d) for d in t.shape]))
              for n, t in example_inputs.items()]
    outputs = [TensorSpec(name=f"output_{i}",
                          dtype=_dtype_name(t.dtype),
                          shape=[int(d) for d in t.shape])
               for i, t in enumerate(outs)]
    return ModelArtifact(
        name=name, version=version, task=task, format="torchscript",
        path=os.path.abspath(out_path), sha256=_sha256_file(out_path),
        inputs=inputs, outputs=outputs, entry=entry,
        traced=(method == "trace"), source=source or "torch export",
    )


# -- serving rung ------------------------------------------------------

class ModelServer:
    """The serving rung: load / execute / unload over the LRU cache.

    The NAL shape from storyteller-ml — a model is an opaque handle
    loaded by path, executed against an explicit tensor contract, and
    evicted when the policy says so. The cache IS the existing
    :class:`~nomorals.voice.lifecycle.ModelLifecycleManager`: bounded,
    LRU, failure-as-state with retry on the next acquire.
    """

    def __init__(self, registry: ServiceRegistry, *,
                 policy: Any = None) -> None:
        from .lifecycle import LifecyclePolicy, ModelLifecycleManager
        self.registry = registry
        self._manager = ModelLifecycleManager(
            policy or LifecyclePolicy())

    # -- NAL seam ------------------------------------------------------

    def load(self, ref: str) -> Any:
        """Load (or return the cached handle for) ``slug[@version]``."""
        decl = self.registry.resolve(ref)
        artifact = decl.artifact
        if artifact.format != "torchscript":
            raise ValueError(
                f"service {decl.ref}: cannot serve format "
                f"{artifact.format!r} — TorchScript only")

        def _loader() -> Any:
            # Validate inside the loader: the LRU cache calls it only on
            # a miss, so a cache hit never re-hashes a 240MB artifact.
            artifact.validate()
            from .vits._torch import require_torch
            torch = require_torch(
                f"serving model service {decl.ref}")
            module = torch.jit.load(artifact.path, map_location="cpu")
            module.eval()
            return module

        return self._manager.acquire(
            decl.ref, _loader,
            size_bytes=os.path.getsize(artifact.path))

    def execute(self, ref: str, inputs: dict) -> dict[str, Any]:
        """Run ``ref`` on named inputs; returns outputs + shapes."""
        decl = self.registry.resolve(ref)
        artifact = decl.artifact
        if not isinstance(inputs, dict) or not inputs:
            raise ValueError(f"service {decl.ref}: execute needs named "
                             "inputs (name -> tensor or payload)")
        declared = {t.name: t for t in artifact.inputs}
        unknown = sorted(set(inputs) - set(declared))
        if unknown:
            raise ValueError(
                f"service {decl.ref}: unknown input(s) "
                f"{', '.join(unknown)} — declared: "
                f"{', '.join(declared)}")
        missing = sorted(set(declared) - set(inputs))
        if missing:
            raise ValueError(
                f"service {decl.ref}: missing input(s) "
                f"{', '.join(missing)}")
        tensors = {}
        for spec in artifact.inputs:
            tensor = _payload_to_tensor(inputs[spec.name])
            bad = spec.matches(_dtype_name(tensor.dtype),
                               [int(d) for d in tensor.shape])
            if bad is not None:
                raise ValueError(f"service {decl.ref}: {bad}")
            tensors[spec.name] = tensor
        module = self.load(ref)
        entry = getattr(module, artifact.entry, None)
        if entry is None or not callable(entry):
            raise ValueError(
                f"service {decl.ref}: no callable entry "
                f"{artifact.entry!r} on the loaded module")
        from .vits._torch import require_torch
        torch = require_torch(f"serving model service {decl.ref}")
        with torch.no_grad():
            outs = entry(*[tensors[s.name] for s in artifact.inputs])
        if not isinstance(outs, (tuple, list)):
            outs = (outs,)
        if len(outs) != len(artifact.outputs):
            raise ValueError(
                f"service {decl.ref}: entry {artifact.entry!r} returned "
                f"{len(outs)} outputs, artifact declares "
                f"{len(artifact.outputs)}")
        payloads = {}
        for spec, out in zip(artifact.outputs, outs, strict=True):
            payloads[spec.name] = _tensor_payload(out)
        return {"service": decl.slug, "version": decl.version,
                "outputs": payloads,
                "output_shapes": {n: p["shape"]
                                  for n, p in payloads.items()}}

    def unload(self, ref: str) -> bool:
        """Evict a loaded service. True when something was evicted."""
        decl = self.registry.resolve(ref)
        return self._manager.evict(decl.ref)

    def status(self) -> dict[str, Any]:
        """The whole picture: registry + cache, as data."""
        return {"services": [d.to_dict() for d in self.registry.list()],
                "cache": self._manager.status()}


def health(ref: str, server: ModelServer | None = None,
           registry: ServiceRegistry | None = None) -> dict[str, Any]:
    """Run a service's declared probe and report ok + timing.

    Never raises: health is an observation, and a failure is a result
    (ok False, error named) — a health check that crashes is worse
    than one that reports.
    """
    started = time.monotonic()
    try:
        registry = registry or ServiceRegistry()
        server = server or ModelServer(registry)
        decl = registry.resolve(ref)
        probe = decl.health_probe or {}
        inputs = probe.get("inputs")
        if not inputs:
            return {"ok": False, "service": decl.slug,
                    "version": decl.version,
                    "error": f"service {decl.ref} declares no "
                             "health probe inputs"}
        result = server.execute(ref, inputs)
        got = set(result["outputs"])
        want = {t.name for t in decl.artifact.outputs}
        if want - got:
            return {"ok": False, "service": decl.slug,
                    "version": decl.version,
                    "error": "health probe outputs missing: "
                             + ", ".join(sorted(want - got))}
        return {"ok": True, "service": decl.slug,
                "version": decl.version,
                "ms": round((time.monotonic() - started) * 1000, 1),
                "output_shapes": result["output_shapes"]}
    except Exception as exc:  # noqa: BLE001 — health reports, never raises
        return {"ok": False, "service": (ref or "").split("@")[0],
                "version": (ref or "").split("@")[1]
                if "@" in (ref or "") else "",
                "error": f"{type(exc).__name__}: {exc}"}


# -- chat control ------------------------------------------------------

def control_service(tail: str, *, services_dir: str = "") -> str:
    """/voice service — the chat surface for model services.

    list / health <ref> / status. Serve takes tensor JSON, which no one
    types in chat — ``health`` is the serve demonstration here (it runs
    the service's declared probe through the server).
    """
    registry = ServiceRegistry(services_dir)
    parts = (tail or "").strip().split(None, 1)
    verb = parts[0].lower() if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""
    if verb in ("list", "ls"):
        rows = registry.list()
        if not rows:
            return ("no model services registered yet — run "
                    "`nm voice service export-vits-probe`")
        lines = ["model services:"]
        for d in rows:
            lines.append(f"  {d.slug}@{d.version} [{d.task}] "
                         f"{d.artifact.format}")
        return "\n".join(lines)
    if verb == "health":
        if not rest:
            return "usage: /voice service health <slug[@version]>"
        result = health(rest, registry=registry)
        if result.get("ok"):
            shapes = ", ".join(
                f"{n} {s}" for n, s in
                result.get("output_shapes", {}).items())
            return (f"✅ {result['service']}@{result['version']} healthy "
                    f"— {result.get('ms', '?')}ms, outputs: {shapes}")
        return (f"❌ {result.get('service', rest)} unhealthy: "
                f"{result.get('error', 'unknown failure')}")
    if verb == "status":
        server = ModelServer(registry)
        state = server.status()
        cache = state.get("cache", {})
        return (f"{len(state['services'])} service(s) registered, "
                f"{cache.get('loaded', 0)} loaded in the model cache")
    return ("usage: /voice service list | health <slug[@version]> | status")


# -- job wiring (the worker-pod shape) ---------------------------------

def service_executor(registry: ServiceRegistry,
                     server: ModelServer | None = None) -> Callable:
    """A callable for ``InferenceJobStore.drive`` — one service per job.

    ``job.model`` is ``slug@version`` (or a bare slug = latest);
    ``job.params["inputs"]`` is the named-tensor payload. The result
    carries service, version, outputs, and shapes. A bad job raises
    naming the problem; the job store records it as ``executor_error``.
    """
    server = server or ModelServer(registry)

    def execute(job: Any) -> dict[str, Any]:
        model = getattr(job, "model", "") or ""
        if not model.strip():
            raise ValueError("service job needs job.model = "
                             "'slug@version' (or a bare slug)")
        params = getattr(job, "params", {}) or {}
        inputs = params.get("inputs")
        if not inputs:
            raise ValueError(
                f"service job {model!r}: params carry no 'inputs'")
        decl = registry.resolve(model)
        result = server.execute(decl.ref, inputs)
        return {"service": decl.slug, "version": decl.version,
                "outputs": result["outputs"],
                "output_shapes": result["output_shapes"]}

    return execute


# -- the one real end-to-end service -----------------------------------

def export_vits_probe_service(services_dir: str = "", *, seed: int = 0,
                              owner: str = "") -> ServiceDeclaration:
    """Build, export, and register the VITS probe service.

    The REAL VITS architecture (``SynthesizerTrn`` from the bundled
    config, ``manual_seed(0)``) traced through ``infer_ts_noistft`` —
    the same method the existing ``export_torchscript`` traces. The
    emotion/BERT tensors are random dummies of the right shapes: the
    moji/BERT weights are absent on this box, and this service is a
    PIPELINE probe, not a trained voice — its weights are random, so
    its output is the iSTFT magnitude + phase of noise (spec/phase),
    not audio, and the survey says so.
    A real trained voice comes from ``export_torchscript(checkpoint,
    config)`` and registers into this same registry.
    """
    from .vits._torch import require_torch
    torch = require_torch("exporting the VITS probe service")
    from .vits.configs import load_config
    from .vits.models import SynthesizerTrn
    from .vits.text.symbols import symbols

    services_dir = services_dir or default_services_dir()
    os.makedirs(services_dir, exist_ok=True)
    hps = load_config(config_name="ljs_li44_tmbert_ft_s1_arpa.json")
    torch.manual_seed(seed)
    net = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    net.eval()
    torch.manual_seed(seed)
    example = {
        "x": torch.randint(0, len(symbols), (1, 12)),
        "x_lengths": torch.LongTensor([12]),
        "tm_hidden": torch.randn(1, 2304),
        "bert": torch.randn(1, 10, hps.model.bert_size),
        "bert_lens": torch.LongTensor([10]),
        "length_scale": torch.FloatTensor([1.0]),
    }
    out_path = os.path.join(services_dir, "vits-probe-0.1.0.pt")
    artifact = export_torchscript_service(
        net, example, out_path, method="trace",
        entry="infer_ts_noistft", task="tts-spectral", name="vits-probe",
        version="0.1.0",
        source=f"SynthesizerTrn random weights, seed {seed} — probe, "
               "not a trained voice")
    # Name the traced outputs after what infer_ts_noistft returns.
    # (TensorSpec is frozen — rename by replacement, not assignment.)
    from dataclasses import replace as _replace
    artifact.outputs = [_replace(spec, name=name) for spec, name in
                        zip(artifact.outputs,
                            ("spec", "phase", "x1", "attn"), strict=True)]
    artifact.sha256 = _sha256_file(out_path)
    probe_inputs = {n: _tensor_payload(t)
                    for n, t in example.items()}
    decl = ServiceDeclaration(
        slug="vits-probe", version="0.1.0", task="tts-spectral",
        artifact=artifact, runtime="python-torch",
        container=ContainerSpec(
            base_image="pytorch/pytorch:2.5.1-cuda12.1-cudnn8-runtime",
            owner=owner, build_context="voice/vits",
            run_label="storyteller-ml probe"),
        health_probe={"inputs": probe_inputs},
    )
    ServiceRegistry(services_dir).register(decl)
    return decl
