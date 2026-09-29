"""Dataset codecs and the dataset registry.

Four formats matter in practice, and they are all trivially convertible once you
notice that every one of them is just "a list of (role, content) turns":

- **ChatML** — ``<|im_start|>role … <|im_end|>``, what llama.cpp and most
  uncensored finetunes expect.
- **Alpaca** — ``instruction`` / ``input`` / ``output``.
- **ShareGPT** — ``conversations`` with ``from``/``value``.
- **Messages** — the OpenAI ``role``/``content`` shape.

The registry stores only metadata in SQLite. The actual bytes live on disk:
putting a 4 GB corpus in a BLOB column is how you make a database unusable.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..core.errors import NotFound, ParseError, ValidationError
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.repository import Repository

__all__ = ["Turn", "Example", "Dataset", "DatasetRegistry", "to_chatml", "read_jsonl", "write_jsonl"]

_log = get_logger(__name__)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class Example:
    """One training example: a list of turns, plus optional weighting."""

    turns: list[Turn]
    weight: float = 1.0
    source: str = ""

    @property
    def prompt(self) -> str:
        return "\n".join(t.content for t in self.turns if t.role != "assistant")

    @property
    def completion(self) -> str:
        return "\n".join(t.content for t in self.turns if t.role == "assistant")

    def to_chatml(self, *, add_generation_prompt: bool = False) -> str:
        return to_chatml(self.turns, add_generation_prompt=add_generation_prompt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [{"role": t.role, "content": t.content} for t in self.turns],
            "weight": self.weight,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Example":
        messages = data.get("messages") or data.get("conversation") or []
        turns = [
            Turn(str(m.get("role") or "user"), str(m.get("content") or ""))
            for m in messages
            if isinstance(m, dict)
        ]
        if not turns:
            raise ParseError("example has no messages")
        return cls(
            turns=turns,
            weight=float(data.get("weight", 1.0) or 1.0),
            source=str(data.get("source", "") or ""),
        )


def to_chatml(turns: Iterable[Turn], *, add_generation_prompt: bool = False) -> str:
    """Render turns in ChatML, the format llama.cpp and most finetunes expect."""
    parts = [f"{IM_START}{t.role}\n{t.content}{IM_END}" for t in turns]
    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")
    return "\n".join(parts)


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Stream JSONL, skipping blank lines and reporting the bad ones."""
    target = Path(path).expanduser()
    with target.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                _log.warning("%s:%d skipped, invalid JSON: %s", target.name, lineno, exc)


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def decode_example(row: dict[str, Any], *, kind: str = "") -> Example:
    """Coerce any supported shape into an Example.

    Format sniffing is ordered by specificity: the OpenAI shape is checked last
    because ShareGPT also uses a list, and Alpaca is identified by its keys.
    """
    if kind == "" :
        if "conversations" in row:
            kind = "sharegpt"
        elif "instruction" in row:
            kind = "alpaca"
        elif "text" in row and "messages" not in row:
            kind = "pretrain"
        else:
            kind = "messages"

    if kind == "sharegpt":
        turns = [
            Turn("assistant" if str(m.get("from")) in {"gpt", "assistant", "bot"} else "user",
                 str(m.get("value") or ""))
            for m in row.get("conversations") or []
            if isinstance(m, dict)
        ]
        if not turns:
            raise ParseError("sharegpt example has no conversations")
        return Example(turns=turns, source=str(row.get("source", "")))

    if kind == "alpaca":
        instruction = str(row.get("instruction") or "")
        extra = str(row.get("input") or "")
        prompt = f"{instruction}\n{extra}".strip() if extra else instruction
        output = str(row.get("output") or "")
        if not prompt:
            raise ParseError("alpaca example has no instruction")
        turns = [Turn("user", prompt)]
        if row.get("system"):
            turns.insert(0, Turn("system", str(row["system"])))
        if output:
            turns.append(Turn("assistant", output))
        return Example(turns=turns, source=str(row.get("source", "")))

    if kind == "pretrain":
        text = str(row.get("text") or "")
        if not text.strip():
            raise ParseError("pretrain example has no text")
        return Example(turns=[Turn("text", text)], source=str(row.get("source", "")))

    return Example.from_dict(row)


@dataclass
class Dataset:
    """A named corpus on disk, with metadata in SQLite."""

    name: str
    path: str
    kind: str = "chat"
    id: str = field(default_factory=new_id)
    rows: int = 0
    bytes: int = 0
    tokens: int = 0
    checksum: str = ""
    schema_: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "kind": self.kind, "path": self.path,
            "rows": self.rows, "bytes": self.bytes, "tokens": self.tokens,
            "checksum": self.checksum, "schema_": self.schema_,
            "created_at": self.created_at, "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Dataset":
        raw_schema = row.get("schema_") or "{}"
        raw_meta = row.get("metadata") or "{}"
        return cls(
            id=row["id"], name=row["name"], kind=row.get("kind") or "chat",
            path=row.get("path") or "", rows=int(row.get("rows") or 0),
            bytes=int(row.get("bytes") or 0), tokens=int(row.get("tokens") or 0),
            checksum=row.get("checksum") or "",
            schema_=raw_schema if isinstance(raw_schema, dict) else json.loads(raw_schema or "{}"),
            created_at=float(row.get("created_at") or 0.0),
            metadata=raw_meta if isinstance(raw_meta, dict) else json.loads(raw_meta or "{}"),
        )

    def examples(self, *, limit: int = 0) -> Iterator[Example]:
        """Stream examples, decoding each row lazily."""
        if not self.path or not Path(self.path).is_file():
            raise NotFound(f"dataset file missing: {self.path}")
        for index, row in enumerate(read_jsonl(self.path)):
            if limit and index >= limit:
                return
            try:
                yield decode_example(row, kind=self.kind)
            except ParseError as exc:
                _log.debug("skipping malformed row: %s", exc)

    def chatml_corpus(self, *, limit: int = 0) -> Iterator[str]:
        for example in self.examples(limit=limit):
            yield example.to_chatml()


def file_checksum(path: str | os.PathLike[str], *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


class DatasetRegistry:
    """Metadata for datasets; the bytes stay on disk."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self.repo = Repository(
            db, "datasets", json_columns=("schema_", "metadata"),
            timestamp_columns=("created_at",),
        )

    def register(
        self,
        name: str,
        path: str | os.PathLike[str],
        *,
        kind: str = "chat",
        tokens: int = 0,
        checksum: str = "",
        schema_: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        count_rows: bool = True,
    ) -> Dataset:
        target = Path(path).expanduser()
        if not target.is_file():
            raise ValidationError(f"dataset file does not exist: {target}", field="path")
        if kind not in {"chat", "alpaca", "sharegpt", "pretrain", "eval", "messages"}:
            raise ValidationError(f"unknown dataset kind {kind!r}", field="kind")

        rows = 0
        if count_rows:
            with target.open("r", encoding="utf-8", errors="replace") as handle:
                rows = sum(1 for line in handle if line.strip())

        dataset = Dataset(
            name=name,
            path=str(target),
            kind=kind,
            rows=rows,
            bytes=target.stat().st_size,
            tokens=tokens,
            checksum=checksum or file_checksum(target),
            schema_=schema_ or {},
            created_at=__import__("time").time(),
            metadata=metadata or {},
        )
        self.repo.create(dataset.to_row())
        _log.info("registered dataset %s: %d rows, %d bytes", name, rows, dataset.bytes)
        return dataset

    def register_examples(
        self,
        name: str,
        examples: Iterable[Example],
        directory: str | os.PathLike[str],
        *,
        kind: str = "chat",
        metadata: dict[str, Any] | None = None,
    ) -> Dataset:
        """Materialize examples to JSONL and register the file."""
        target_dir = Path(directory).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{name}.jsonl"
        count = write_jsonl(target, (e.to_dict() for e in examples))
        if count == 0:
            raise ValidationError("refusing to register an empty dataset", field="examples")
        return self.register(name, target, kind=kind, metadata=metadata)

    def get(self, dataset_id: str) -> Dataset:
        row = self.repo.get(dataset_id) or self.repo.find_one(name=dataset_id)
        if row is None:
            raise NotFound(f"no dataset {dataset_id!r}")
        return Dataset.from_row(row)

    def list(self, *, kind: str = "", limit: int = 50) -> list[Dataset]:
        # Same trap as elsewhere: find() has no limit parameter, so passing one
        # filters on a column called "limit" and matches nothing.
        query = self.repo.query()
        if kind:
            query.where("kind = ?", kind)
        rows = self.db.query(*query.order_by("created_at DESC").limit(limit).build())
        return [Dataset.from_row(dict(r)) for r in rows]

    def delete(self, dataset_id: str, *, remove_file: bool = False) -> int:
        dataset = self.get(dataset_id)
        removed = self.repo.delete(dataset_id)
        if remove_file and dataset.path:
            Path(dataset.path).unlink(missing_ok=True)
        return removed

    def stats(self) -> dict[str, Any]:
        return {
            "datasets": int(self.db.scalar("SELECT COUNT(*) FROM datasets", default=0) or 0),
            "rows": int(self.db.scalar("SELECT COALESCE(SUM(rows),0) FROM datasets", default=0) or 0),
            "bytes": int(self.db.scalar("SELECT COALESCE(SUM(bytes),0) FROM datasets", default=0) or 0),
            "tokens": int(self.db.scalar("SELECT COALESCE(SUM(tokens),0) FROM datasets", default=0) or 0),
        }


def to_alpaca(example: Any) -> dict[str, str]:
    """Convert example to Alpaca format.

    Handles dict input (standard) and list input (messages format).
    """
    if isinstance(example, list):
        # Messages format: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
        instruction = ""
        output = ""
        for msg in example:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user" and not instruction:
                    instruction = content
                elif role == "assistant" and not output:
                    output = content
        return {"instruction": instruction, "input": "", "output": output}

    if not isinstance(example, dict):
        return {"instruction": str(example), "input": "", "output": ""}

    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    # Fallback: if no instruction but has messages
    if not instruction and "messages" in example:
        return to_alpaca(example["messages"])

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


def to_sharegpt(example: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Stub: convert example to ShareGPT format."""
    instruction = example.get("instruction", "")
    output = example.get("output", "")
    conversations = [
        {"from": "human", "value": instruction},
        {"from": "gpt", "value": output},
    ]
    return {"conversations": conversations}


def write_format_bundles(examples: list[dict[str, Any]], output_dir: str) -> dict[str, int]:
    """Stub: write examples in multiple formats to output_dir."""
    return {"alpaca": len(examples), "sharegpt": len(examples)}
