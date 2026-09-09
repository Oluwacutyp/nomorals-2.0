"""Model registry and catalog.

Two jobs:

1. **Catalog** — a curated list of good open-weights starting points, so "give me
   a capable uncensored instruct model" is a lookup rather than a research task.
2. **Registry** — persistent records of every model this system knows about:
   foundation checkpoints it downloaded, and personal fine-tunes it produced, with
   their provenance, checksums, and eval scores.

The registry is what makes promotion safe: a fine-tune only becomes the active
model after its eval row says it beat the incumbent.

A note on what "uncensored" means here: these are models published without an
vendor-side refusal layer, chosen because you are running them on your own
hardware and want the model to follow *your* instructions rather than a vendor's.
They are ordinary open-weight checkpoints from Hugging Face; the capability
policy in :mod:`nomorals.core.policy` still governs what the *agent* may do with
them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..core.errors import NotFound, ValidationError
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.db import Database
from ..storage.repository import Repository

__all__ = ["CatalogEntry", "MODEL_CATALOG", "ModelRecord", "ModelRegistry", "search_catalog"]

_log = get_logger(__name__)


@dataclass(frozen=True)
class CatalogEntry:
    """A known-good starting point."""

    repo_id: str
    family: str
    params: int
    context_length: int
    kind: str  # instruct | base | vlm | embedding
    license: str
    notes: str
    tags: tuple[str, ...] = ()

    @property
    def size_hint_gb(self) -> float:
        """Rough fp16 weight size, for capacity planning before a download."""
        return round(self.params * 2 / 1e9, 1)


#: Curated catalog. Params are approximate; sizes are for planning only.
MODEL_CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        "cognitivecomputations/dolphin-2.9.1-llama-3-8b",
        "llama3", 8_000_000_000, 8192, "instruct", "llama3",
        "Dolphin on Llama-3 8B. Strong general instruct baseline.",
        ("dolphin", "8b", "general"),
    ),
    CatalogEntry(
        "cognitivecomputations/dolphin-2.9.2-qwen2-72b",
        "qwen2", 72_000_000_000, 32768, "instruct", "qwen2",
        "Dolphin on Qwen2 72B. Highest capability, needs real hardware.",
        ("dolphin", "72b", "general"),
    ),
    CatalogEntry(
        "cognitivecomputations/dolphin-2.9.4-llama3.1-8b",
        "llama3.1", 8_000_000_000, 131072, "instruct", "llama3.1",
        "Dolphin on Llama-3.1 8B with a long context window.",
        ("dolphin", "8b", "long-context"),
    ),
    CatalogEntry(
        "cognitivecomputations/dolphin-2.9.3-qwen2.5-1.5b",
        "qwen2.5", 1_500_000_000, 32768, "instruct", "qwen2",
        "Small Dolphin. Runs on a phone; the Termux default.",
        ("dolphin", "1.5b", "mobile"),
    ),
    CatalogEntry(
        "cognitivecomputations/dolphin-2.9.4-phi-4",
        "phi4", 14_000_000_000, 16384, "instruct", "mit",
        "Dolphin on Phi-4. Good reasoning per parameter.",
        ("dolphin", "14b", "reasoning"),
    ),
    CatalogEntry(
        "NousResearch/Hermes-3-Llama-3.1-8B",
        "llama3.1", 8_000_000_000, 131072, "instruct", "llama3.1",
        "Hermes 3. Strong tool use and structured output.",
        ("hermes", "8b", "tool-use"),
    ),
    CatalogEntry(
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistral", 7_000_000_000, 32768, "instruct", "apache-2.0",
        "Apache-licensed baseline for commercial-friendly fine-tunes.",
        ("mistral", "7b", "apache"),
    ),
    CatalogEntry(
        "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5", 7_000_000_000, 131072, "instruct", "apache-2.0",
        "Strong multilingual instruct model, permissive license.",
        ("qwen", "7b", "multilingual"),
    ),
    CatalogEntry(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "qwen2.5", 7_000_000_000, 131072, "instruct", "apache-2.0",
        "Code-specialized. The coding agent's default.",
        ("qwen", "7b", "code"),
    ),
    CatalogEntry(
        "microsoft/Phi-3.5-mini-instruct",
        "phi3.5", 3_800_000_000, 128000, "instruct", "mit",
        "Very small, long context, MIT licensed.",
        ("phi", "3.8b", "mobile"),
    ),
    CatalogEntry(
        "llava-hf/llava-1.5-7b-hf",
        "llava", 7_000_000_000, 4096, "vlm", "llama2",
        "Vision-language model for image understanding.",
        ("llava", "7b", "vision"),
    ),
    CatalogEntry(
        "Qwen/Qwen2-VL-7B-Instruct",
        "qwen2vl", 7_000_000_000, 32768, "vlm", "apache-2.0",
        "Strong open VLM with long-context image handling.",
        ("qwen", "7b", "vision"),
    ),
    CatalogEntry(
        "sentence-transformers/all-MiniLM-L6-v2",
        "minilm", 22_000_000, 256, "embedding", "apache-2.0",
        "Fast 384-dim embeddings. The default retrieval embedder.",
        ("embedding", "384d"),
    ),
    CatalogEntry(
        "BAAI/bge-m3",
        "bge", 568_000_000, 8192, "embedding", "mit",
        "Multilingual, long-context embeddings.",
        ("embedding", "multilingual"),
    ),
    CatalogEntry(
        "TheBloke/dolphin-2.9.1-llama-3-8b-GGUF",
        "llama3", 8_000_000_000, 8192, "instruct", "llama3",
        "GGUF quants of Dolphin 8B for llama.cpp.",
        ("dolphin", "gguf", "quantized"),
    ),
)


def search_catalog(
    query: str = "",
    *,
    kind: str = "",
    max_params: int | None = None,
    min_context: int = 0,
    tag: str = "",
    limit: int = 10,
) -> list[CatalogEntry]:
    """Filter the catalog. Empty query returns everything up to ``limit``."""
    needle = query.lower()
    out: list[CatalogEntry] = []
    for entry in MODEL_CATALOG:
        if kind and entry.kind != kind:
            continue
        if max_params is not None and entry.params > max_params:
            continue
        if entry.context_length < min_context:
            continue
        if tag and tag.lower() not in {t.lower() for t in entry.tags}:
            continue
        if needle:
            haystack = " ".join(
                [entry.repo_id.lower(), entry.family.lower(), entry.notes.lower(), *entry.tags]
            )
            if needle not in haystack:
                continue
        out.append(entry)
    out.sort(key=lambda e: -e.params)
    return out[:limit]


def _scores_of(record: Any) -> dict[str, Any]:
    """Read eval_scores from either a ModelRecord or a raw row.

    ``by_name()`` returns a dataclass while ``repo.get()`` returns a dict, and
    both can reach the promotion gate. Calling ``.get()`` on the dataclass was an
    AttributeError waiting to happen on every gated promotion.
    """
    raw = getattr(record, "eval_scores", None)
    if raw is None and isinstance(record, dict):
        raw = record.get("eval_scores")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "{}")
        except json.JSONDecodeError:
            raw = {}
    return raw if isinstance(raw, dict) else {}


@dataclass
class ModelRecord:
    """A model this system knows about."""

    id: str
    name: str
    family: str = ""
    kind: str = "foundation"
    source: str = ""
    revision: str = ""
    params: int = 0
    context_length: int = 0
    quantization: str = ""
    license: str = ""
    sha256: str = ""
    path: str = ""
    size_bytes: int = 0
    active: bool = False
    base_model: str = ""
    eval_scores: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ModelRecord":
        return cls(
            id=row["id"],
            name=row["name"],
            family=row.get("family", "") or "",
            kind=row.get("kind", "foundation") or "foundation",
            source=row.get("source", "") or "",
            revision=row.get("revision", "") or "",
            params=int(row.get("params") or 0),
            context_length=int(row.get("context_length") or 0),
            quantization=row.get("quantization", "") or "",
            license=row.get("license", "") or "",
            sha256=row.get("sha256", "") or "",
            path=row.get("path", "") or "",
            size_bytes=int(row.get("size_bytes") or 0),
            active=bool(row.get("active")),
            base_model=row.get("base_model", "") or "",
            eval_scores=row.get("eval_scores") or {},
            created_at=float(row.get("created_at") or 0.0),
            metadata=row.get("metadata") or {},
        )

    @property
    def is_local(self) -> bool:
        return bool(self.path)


class ModelRegistry:
    """Persistent model registry backed by the ``models`` table."""

    def __init__(self, db: Database) -> None:
        self.repo = Repository(
            db, "models", json_columns=("eval_scores", "metadata"), timestamp_columns=("created_at",)
        )
        self.db = db

    # ── registration ─────────────────────────────────────────────────────────
    def register(
        self,
        name: str,
        *,
        source: str = "",
        kind: str = "foundation",
        path: str = "",
        sha256: str = "",
        size_bytes: int = 0,
        base_model: str = "",
        revision: str = "",
        metadata: dict[str, Any] | None = None,
        activate: bool = False,
    ) -> ModelRecord:
        if not name:
            raise ValidationError("model name is required")
        catalog = next((e for e in MODEL_CATALOG if e.repo_id == name or e.repo_id == source), None)
        values: dict[str, Any] = {
            "name": name,
            "source": source or name,
            "kind": kind,
            "path": path,
            "sha256": sha256,
            "size_bytes": size_bytes,
            "base_model": base_model,
            "revision": revision,
            "family": catalog.family if catalog else "",
            "params": catalog.params if catalog else 0,
            "context_length": catalog.context_length if catalog else 0,
            "license": catalog.license if catalog else "",
            "metadata": metadata or {},
            "eval_scores": {},
            "created_at": time.time(),
        }
        existing = self.repo.find_one(name=name, revision=revision)
        if existing is not None:
            self.repo.update(existing["id"], values)
            record_id = existing["id"]
        else:
            record_id = self.repo.create(values)["id"]
        if activate:
            self.activate(record_id)
        return self.get(record_id)

    def register_finetune(
        self,
        name: str,
        *,
        base_model: str,
        output_path: str,
        run_id: str = "",
        eval_scores: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelRecord:
        """Record a model this system trained itself."""
        record_id = self.repo.create(
            {
                "name": name,
                "source": "self-trained",
                "kind": "finetune",
                "path": output_path,
                "base_model": base_model,
                "family": base_model.split("/")[0] if "/" in base_model else "",
                "eval_scores": eval_scores or {},
                "metadata": {**(metadata or {}), "training_run": run_id},
                "created_at": time.time(),
            }
        )["id"]
        return self.get(record_id)

    # ── reads ────────────────────────────────────────────────────────────────
    def get(self, record_id: str) -> ModelRecord:
        return ModelRecord.from_row(self.repo.require(record_id))

    def by_name(self, name: str) -> ModelRecord | None:
        row = self.repo.find_one(name=name)
        return ModelRecord.from_row(row) if row else None

    def list(self, *, kind: str = "", active_only: bool = False, limit: int = 100) -> list[ModelRecord]:
        filters: dict[str, Any] = {}
        if kind:
            filters["kind"] = kind
        if active_only:
            filters["active"] = 1
        rows = self.repo.all(order_by="created_at DESC", limit=limit)
        records = [ModelRecord.from_row(r) for r in rows]
        for key, value in filters.items():
            records = [r for r in records if getattr(r, key) == value or (key == "active" and r.active == bool(value))]
        return records

    def active(self) -> ModelRecord | None:
        row = self.repo.find_one(active=1)
        return ModelRecord.from_row(row) if row else None

    # ── activation ───────────────────────────────────────────────────────────
    def activate(self, name_or_id: str) -> ModelRecord:
        """Make one model active, deactivating all others. Atomic."""
        row = self.repo.get(name_or_id) or self.repo.find_one(name=name_or_id)
        if row is None:
            raise NotFound(f"model {name_or_id!r} is not registered")
        with self.db.transaction():
            self.db.execute("UPDATE models SET active = 0 WHERE active = 1")
            self.db.execute("UPDATE models SET active = 1 WHERE id = ?", (row["id"],))
        record = self.get(row["id"])
        _log.info("activated model %s (%s)", record.name, record.kind)
        return record

    def deactivate(self, name_or_id: str) -> int:
        row = self.repo.get(name_or_id) or self.repo.find_one(name=name_or_id)
        if row is None:
            return 0
        return self.db.execute("UPDATE models SET active = 0 WHERE id = ?", (row["id"],)).rowcount

    # ── evaluation ───────────────────────────────────────────────────────────
    def record_eval(self, name_or_id: str, scores: dict[str, Any]) -> ModelRecord:
        row = self.repo.get(name_or_id) or self.repo.find_one(name=name_or_id)
        if row is None:
            raise NotFound(f"model {name_or_id!r} is not registered")
        merged = {**(row.get("eval_scores") or {}), **scores}
        if isinstance(merged, str):
            merged = json.loads(merged or "{}")
        self.repo.update(row["id"], {"eval_scores": merged})
        return self.get(row["id"])

    def beats_incumbent(
        self, candidate: str, metric: str = "score", *, tolerance: float = 0.0
    ) -> bool:
        """The promotion gate: is the candidate at least as good as what is live?

        Without this check, an auto-finetune loop monotonically degrades the
        system: every run that "trains successfully" gets promoted regardless of
        whether it actually improved anything.
        """
        current = self.active()
        if current is None:
            return True
        candidate_record = self.by_name(candidate) or self.repo.get(candidate)
        if candidate_record is None:
            raise NotFound(f"model {candidate!r} is not registered")
        candidate_scores = _scores_of(candidate_record)
        current_scores = _scores_of(current)
        if metric not in candidate_scores:
            return False
        if metric not in current_scores:
            return True
        return float(candidate_scores[metric]) >= float(current_scores[metric]) - tolerance

    # ── maintenance ──────────────────────────────────────────────────────────
    def delete(self, name_or_id: str) -> int:
        row = self.repo.get(name_or_id) or self.repo.find_one(name=name_or_id)
        if row is None:
            return 0
        return self.repo.delete(row["id"])

    def local_models(self) -> list[ModelRecord]:
        return [m for m in self.list(limit=500) if m.is_local]

    def stats(self) -> dict[str, Any]:
        active = self.active()
        return {
            "total": self.repo.count(),
            "finetunes": len(self.list(kind="finetune")),
            "local": len(self.local_models()),
            "active": active.name if active else None,
            "bytes": sum(m.size_bytes for m in self.list(limit=1000)),
        }
