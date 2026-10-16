"""LLM cassettes: record / replay provider responses for deterministic tests.

Storage is the generic mechanism in :mod:`nomorals.core.cassette` — the
same store L2 network I/O records to. This module keeps the LLM key
semantics and record shape on top of it.

A cassette is one JSON line per request in ``<dir>/cassettes.jsonl``
(default ``~/.nomorals/cassettes``). The router consults it in
:meth:`nomorals.llm.router.LLMRouter.chat` / ``complete`` /
``describe_image`` — never in ``_dispatch``, which only sees a lambda.

Key semantics (deliberate, read before changing):

* The lookup key is ``sha256`` of canonical JSON over
  ``(operation, task_kind, payload, params)``. It EXCLUDES provider
  and model: the serving provider is chosen at dispatch with failover
  and is nondeterministic, so keying on it would make replays flake.
  Provider + model are RECORDED for audit, and a replayed response
  reports the recorded provider/model. Tradeoff, stated plainly: a
  cassette recorded under model A serves model B's identical request —
  that is the intended test-determinism semantic, not a bug.
  The key format is FROZEN: on-disk cassettes recorded by earlier
  versions must keep matching, so this key is NOT routed through the
  generic ``canonical_key`` kind-prefix — it hashes the payload dict
  directly, exactly as before.
* Vision keys use the image ``sha256``, never the bytes.
* ``task_kind`` is part of the key: the same prompt means something
  different to a code task than to a generic chat.
* Recording is default-on (the suite needs cassettes to exist before
  it can replay them); ``NM_LLM_CASSETTE=0`` disables recording — the
  off-switch for a feature that writes prompts to disk. Recording
  happens only on success, never for replays, and never while
  replay is on — replay mode is read-only (otherwise a replay run
  would serve cassettes recorded earlier in the same run).
* Replay is opt-in: ``NM_LLM_REPLAY=1`` (read at call time) or the
  router's ``replay=True`` ctor param. A replayed response carries
  ``raw["replayed_from_cassette"] = True`` and a fresh, empty
  ``route_trace`` — it was served from disk, not from any provider.
* The store dir is the ctor ``cassette_dir``, else
  ``NM_LLM_CASSETTE_DIR``, else ``~/.nomorals/cassettes``. ISOLATE IT
  for replay runs: a replay serves the recorded response INSTEAD of
  the live provider — by design it bypasses failover, degradation
  bookkeeping, and repair hooks. Replaying a shared dir across
  behavior tests (e.g. the failover suite) makes those tests assert
  against recorded responses instead of live logic. Replay is for
  deterministic-response tests with an isolated dir, not for the
  whole behavior suite.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path
from typing import Any, Sequence

from ..core.cassette import GenericCassetteStore, canonical_json
from .base import LLMResponse, Message, SamplingParams, Usage

__all__ = [
    "CassetteStore",
    "canonical_key",
    "params_key",
    "recording_enabled",
    "replay_enabled",
    "response_from_record",
]

#: Set to "0" to stop writing cassettes (prompts go to disk otherwise).
CASSETTE_ENV = "NM_LLM_CASSETTE"
#: Override the store dir (isolation for replay runs / tests).
CASSETTE_DIR_ENV = "NM_LLM_CASSETTE_DIR"
#: Set to "1" to serve recorded responses instead of calling providers.
REPLAY_ENV = "NM_LLM_REPLAY"


def recording_enabled() -> bool:
    """Recording is default-on; ``NM_LLM_CASSETTE=0`` disables it."""
    return os.environ.get(CASSETTE_ENV, "1") != "0"


def replay_enabled() -> bool:
    """Replay is opt-in via ``NM_LLM_REPLAY=1``."""
    return os.environ.get(REPLAY_ENV, "") == "1"


def params_key(params: SamplingParams | None) -> Any:
    """JSON-able params for the key. None stays None (provider defaults)."""
    if params is None:
        return None
    data = dataclasses.asdict(params)
    # tuples are not JSON values — normalise them so the key is stable.
    return {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in data.items()}


def message_key(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Serialise messages for the key (images are paths/ids, not bytes)."""
    return [dataclasses.asdict(m) for m in messages]


def canonical_key(operation: str, task_kind: str, payload: Any,
                  params: SamplingParams | None) -> str:
    """sha256 of the canonical request (provider/model excluded).

    FROZEN format — see the module docstring. Uses the generic
    ``canonical_json`` primitive so the hashing is one mechanism, but
    the hashed dict is exactly the historical one (no kind prefix).
    """
    return hashlib.sha256(canonical_json(
        {"operation": operation, "task_kind": task_kind or "",
         "payload": payload, "params": params_key(params)},
    ).encode("utf-8")).hexdigest()


def response_from_record(record: dict[str, Any]) -> LLMResponse:
    """Rebuild a recorded response, stamped as a replay."""
    data = record["response"]
    raw = dict(data.get("raw") or {})
    raw["replayed_from_cassette"] = True
    return LLMResponse(
        text=data.get("text", ""),
        model=data.get("model", ""),
        provider=data.get("provider", ""),
        usage=Usage(**(data.get("usage") or {})),
        finish_reason=data.get("finish_reason", "stop"),
        latency_ms=float(data.get("latency_ms", 0.0)),
        raw=raw,
        error=data.get("error", ""),
        degraded=False,
        failed_providers=[],
        fallback_note="",
        failure_class=data.get("failure_class", ""),
        route_trace=[],
        cost_usd=float(data.get("cost_usd", 0.0)),
    )


class CassetteStore:
    """LLM cassette store: :class:`GenericCassetteStore` + LLM record shape.

    Same ``cassettes.jsonl`` file, same first-write-wins semantics — the
    generic store is the mechanism, this class is the LLM view over it.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        if directory is None:
            directory = os.environ.get(CASSETTE_DIR_ENV) or None
        resolved = Path(directory) if directory is not None \
            else Path.home() / ".nomorals" / "cassettes"
        self._store = GenericCassetteStore(resolved, filename="cassettes.jsonl")
        self.directory = self._store.directory
        self.path = self._store.path

    def lookup(self, key: str) -> dict[str, Any] | None:
        """The stored record for ``key``, or None on a miss."""
        return self._store.lookup(key)

    def record(self, key: str, response: LLMResponse) -> bool:
        """Persist ``response`` under ``key``. False when already stored."""
        return self._store.record(key, {
            "key": key,
            "provider": response.provider,
            "model": response.model,
            "response": {**response.to_dict(), "raw": dict(response.raw)},
        })

    def __len__(self) -> int:
        return len(self._store)
