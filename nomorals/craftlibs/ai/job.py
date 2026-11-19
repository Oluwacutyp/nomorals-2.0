"""Jobs (Craft-Libs port of ai-core ``job.rs``).

A job moves ``Queued -> Running -> Succeeded | Failed |
Cancelled``. Progress never goes backwards, and a terminal state
is final. :class:`JobSnapshot` is what a poll returns: the state
plus the events so far (text deltas and outputs).
"""

from __future__ import annotations

from .error import AiError, ErrorKind

__all__ = ["JobSnapshot", "JobState"]


class JobState:
    """Where a job is. ``Running`` carries a percent (``None``
    when the provider reports none)."""

    def __init__(self, kind: str, percent: int | None = None,
                 error: AiError | None = None) -> None:
        self.kind = kind
        self.percent = percent
        self.error = error

    @classmethod
    def running(cls, percent: int | None = None) -> JobState:
        return cls("running", percent=percent)

    @classmethod
    def failed(cls, error: AiError) -> JobState:
        return cls("failed", error=error)

    def is_terminal(self) -> bool:
        return self.kind in ("succeeded", "failed", "cancelled")

    def can_become(self, other: JobState) -> bool:
        """The transition law: forward only, progress never
        backwards, terminal states final."""
        if self.kind == "queued":
            if other.kind == "queued":
                return True
            if other.kind == "running":
                return True
            return other.kind in ("succeeded", "failed", "cancelled")
        if self.kind == "running":
            if other.kind == "running":
                mine = self.percent if self.percent is not None else 0
                theirs = other.percent if other.percent is not None else 0
                return theirs >= mine
            return other.kind in ("succeeded", "failed", "cancelled")
        return self == other

    def to_dict(self) -> dict:
        if self.kind == "running":
            return {"kind": "running", "percent": self.percent}
        if self.kind == "failed":
            return {"kind": "failed",
                    "error": self.error.to_dict() if self.error else None}
        return {"kind": self.kind}

    @classmethod
    def from_dict(cls, data: dict) -> JobState:
        kind = data["kind"]
        if kind == "running":
            return cls.running(data.get("percent"))
        if kind == "failed":
            err = data.get("error")
            return cls.failed(
                AiError.from_dict(err) if err
                else AiError.new(ErrorKind.PROVIDER, "failed"))
        return cls(kind)

    def __eq__(self, other) -> bool:
        return (isinstance(other, JobState)
                and self.kind == other.kind
                and self.percent == other.percent
                and self.error == other.error)

    def __hash__(self):
        return hash((self.kind, self.percent, self.error))

    def __repr__(self) -> str:
        if self.kind == "running":
            return f"JobState.running({self.percent})"
        if self.kind == "failed":
            return f"JobState.failed({self.error!r})"
        return f"JobState.{self.kind.upper()}"


JobState.QUEUED = JobState("queued")
JobState.SUCCEEDED = JobState("succeeded")
JobState.CANCELLED = JobState("cancelled")


class JobSnapshot:
    """What a poll returns: the state plus the events so far."""

    def __init__(self, state: JobState, events: list | None = None,
                 usage=None) -> None:
        self.state = state
        self.events: list[dict] = list(events or [])
        self.usage = usage

    @classmethod
    def new(cls, state: JobState) -> JobSnapshot:
        return cls(state)

    def outputs(self) -> list[dict]:
        """The output events only (text deltas are not outputs)."""
        return [e for e in self.events if e.get("type") == "output"]

    def to_dict(self) -> dict:
        data: dict = {"state": self.state.to_dict(),
                      "events": self.events}
        if self.usage is not None:
            data["usage"] = self.usage
        return data

    @classmethod
    def from_dict(cls, data: dict) -> JobSnapshot:
        return cls(JobState.from_dict(data["state"]),
                   list(data.get("events", [])), data.get("usage"))

    def __eq__(self, other) -> bool:
        return (isinstance(other, JobSnapshot)
                and self.to_dict() == other.to_dict())
