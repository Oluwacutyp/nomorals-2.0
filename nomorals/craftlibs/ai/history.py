"""History (Craft-Libs port of ai-core ``history.rs``).

One line per run start and one per run end, joined by run id into
entries, newest first. A start with no end stays unfinished; an
end whose start was rotated away is dropped; the first end wins
and a repeated start is ignored. Cost is only what the provider
reported, and only when it named a currency.
"""

from __future__ import annotations

from .error import ErrorKind

__all__ = [
    "EndRecord",
    "HISTORY_SCHEMA",
    "HistoryEntry",
    "HistoryLog",
    "HistoryRecord",
    "MAX_PROMPT_CHARS",
    "Outcome",
    "StartRecord",
    "Usage",
    "clip_prompt",
    "merge",
]

HISTORY_SCHEMA = 1
MAX_PROMPT_CHARS = 2000


class Outcome:
    """How a run ended: done, cancelled, or failed with a kind."""

    def __init__(self, kind: str, error_kind: ErrorKind | None = None
                 ) -> None:
        self.kind = kind
        self.error_kind = error_kind

    @classmethod
    def failed(cls, error_kind: ErrorKind) -> Outcome:
        return cls("failed", error_kind)

    def to_dict(self) -> dict:
        if self.kind == "failed":
            return {"kind": "failed",
                    "error_kind": self.error_kind.value}
        return {"kind": self.kind}

    @classmethod
    def from_dict(cls, data) -> Outcome:
        if isinstance(data, dict):
            if data.get("kind") == "failed":
                return cls.failed(ErrorKind(data["error_kind"]))
            return cls(data.get("kind", "done"))
        return cls(str(data))

    def __eq__(self, other) -> bool:
        return (isinstance(other, Outcome)
                and self.kind == other.kind
                and self.error_kind == other.error_kind)

    def __repr__(self) -> str:
        if self.kind == "failed":
            return f"Outcome.failed({self.error_kind!r})"
        return f"Outcome.{self.kind.upper()}"


Outcome.DONE = Outcome("done")
Outcome.CANCELLED = Outcome("cancelled")


class Usage:
    """What the provider reported a run cost."""

    def __init__(self, cost_micros: int | None = None,
                 currency: str | None = None) -> None:
        self.cost_micros = cost_micros
        self.currency = currency

    def to_dict(self) -> dict:
        data: dict = {}
        if self.cost_micros is not None:
            data["cost_micros"] = self.cost_micros
        if self.currency is not None:
            data["currency"] = self.currency
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Usage:
        return cls(data.get("cost_micros"), data.get("currency"))

    def __eq__(self, other) -> bool:
        return (isinstance(other, Usage)
                and self.to_dict() == other.to_dict())


class StartRecord:
    """One run's start line."""

    def __init__(self, run: str, at_ms: int, app: str, role_id: str,
                 role_label: str, capability: str, connection: str,
                 connection_label: str, model: str, locality,
                 destination: str, sent: list | None = None,
                 params: list | None = None,
                 provider_job: str | None = None,
                 estimate=None, prompt: str | None = None,
                 v: int = HISTORY_SCHEMA) -> None:
        self.v = v
        self.run = run
        self.at_ms = at_ms
        self.app = app
        self.role_id = role_id
        self.role_label = role_label
        self.capability = capability
        self.connection = connection
        self.connection_label = connection_label
        self.model = model
        self.locality = locality
        self.destination = destination
        self.sent = list(sent or [])
        self.params = list(params or [])
        self.provider_job = provider_job
        self.estimate = estimate
        self.prompt = prompt

    def to_dict(self) -> dict:
        from .connection import Locality
        data: dict = {"phase": "start", "v": self.v, "run": self.run,
                      "at_ms": self.at_ms, "app": self.app,
                      "role_id": self.role_id,
                      "role_label": self.role_label,
                      "capability": self.capability,
                      "connection": self.connection,
                      "connection_label": self.connection_label,
                      "model": self.model,
                      "locality": self.locality.value
                      if isinstance(self.locality, Locality)
                      else str(self.locality),
                      "destination": self.destination,
                      "sent": self.sent, "params": self.params}
        if self.provider_job is not None:
            data["provider_job"] = self.provider_job
        if self.estimate is not None:
            data["estimate"] = self.estimate
        if self.prompt is not None:
            data["prompt"] = self.prompt
        return data

    @classmethod
    def from_dict(cls, data: dict) -> StartRecord:
        from .connection import Locality
        return cls(data["run"], int(data["at_ms"]), data["app"],
                   data["role_id"], data["role_label"],
                   data["capability"], data["connection"],
                   data["connection_label"], data["model"],
                   Locality(data["locality"]), data["destination"],
                   list(data.get("sent", [])),
                   list(data.get("params", [])),
                   data.get("provider_job"), data.get("estimate"),
                   data.get("prompt"), int(data.get("v", HISTORY_SCHEMA)))

    def __eq__(self, other) -> bool:
        return (isinstance(other, StartRecord)
                and self.to_dict() == other.to_dict())


class EndRecord:
    """One run's end line."""

    def __init__(self, run: str, at_ms: int, outcome: Outcome,
                 usage: Usage | None = None,
                 v: int = HISTORY_SCHEMA) -> None:
        self.v = v
        self.run = run
        self.at_ms = at_ms
        self.outcome = outcome
        self.usage = usage

    def to_dict(self) -> dict:
        data: dict = {"phase": "end", "v": self.v, "run": self.run,
                      "at_ms": self.at_ms,
                      "outcome": self.outcome.to_dict()}
        if self.usage is not None:
            data["usage"] = self.usage.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> EndRecord:
        usage = data.get("usage")
        return cls(data["run"], int(data["at_ms"]),
                   Outcome.from_dict(data["outcome"]),
                   Usage.from_dict(usage) if usage is not None else None,
                   int(data.get("v", HISTORY_SCHEMA)))

    def __eq__(self, other) -> bool:
        return (isinstance(other, EndRecord)
                and self.to_dict() == other.to_dict())


class HistoryRecord:
    """One line: a start or an end."""

    def __init__(self, phase: str, record) -> None:
        self.phase = phase
        self.record = record

    @classmethod
    def start(cls, record: StartRecord) -> HistoryRecord:
        return cls("start", record)

    @classmethod
    def end(cls, record: EndRecord) -> HistoryRecord:
        return cls("end", record)

    def to_dict(self) -> dict:
        return self.record.to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> HistoryRecord:
        phase = data.get("phase")
        if phase == "start":
            return cls.start(StartRecord.from_dict(data))
        if phase == "end":
            return cls.end(EndRecord.from_dict(data))
        raise ValueError(f"unknown history phase {phase!r}")

    def __eq__(self, other) -> bool:
        return (isinstance(other, HistoryRecord)
                and self.to_dict() == other.to_dict())


class HistoryEntry:
    """One run: its start, and its end if it has one."""

    def __init__(self, start: StartRecord, end: EndRecord | None = None
                 ) -> None:
        self.start = start
        self.end = end

    def duration_ms(self) -> int | None:
        if self.end is None:
            return None
        if self.end.at_ms < self.start.at_ms:
            return None
        return self.end.at_ms - self.start.at_ms

    def outcome(self):
        return self.end.outcome if self.end is not None else None

    def reported_cost(self):
        """(micros, currency) — only when the provider reported
        both. No currency, no cost."""
        if self.end is None or self.end.usage is None:
            return None
        usage = self.end.usage
        if usage.cost_micros is None or usage.currency is None:
            return None
        return (usage.cost_micros, usage.currency)


class HistoryLog:
    """Merged entries, newest first, plus how many lines were
    skipped on read."""

    def __init__(self, entries: list, skipped: int = 0) -> None:
        self.entries = entries
        self.skipped = skipped


def merge(records, skipped: int = 0) -> HistoryLog:
    """Join start and end lines by run id. The first end wins, a
    repeated start is ignored, an end with no start is dropped."""
    starts: dict[str, StartRecord] = {}
    ends: dict[str, EndRecord] = {}
    for record in records:
        if record.phase == "start":
            starts.setdefault(record.record.run, record.record)
        elif record.phase == "end":
            ends.setdefault(record.record.run, record.record)
    entries = [HistoryEntry(start, ends.get(run))
               for run, start in starts.items()]
    entries.sort(key=lambda e: e.start.at_ms, reverse=True)
    return HistoryLog(entries, skipped)


def clip_prompt(prompt: str) -> str:
    """Clip a stored prompt to :data:`MAX_PROMPT_CHARS`
    characters, on a character boundary."""
    return prompt[:MAX_PROMPT_CHARS]
