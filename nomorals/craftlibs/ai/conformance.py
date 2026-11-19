"""Conformance (Craft-Libs port of ai-core ``conformance.rs``).

The checks every provider must pass: a job runs its lifecycle,
cancelling stops it, a bad request is refused before a job is
made, and an unknown job is an error, not a silent empty. The
mock provider is the reference; a real provider is checked the
same way.
"""

from __future__ import annotations

from .error import AiError
from .job import JobState

__all__ = [
    "Completed",
    "ConformanceError",
    "MAX_POLLS",
    "run_cancel",
    "run_lifecycle",
    "run_rejects",
    "run_unknown_job",
]

MAX_POLLS = 1000


class ConformanceError(AssertionError):
    """What a provider got wrong, in the words of the check."""


class Completed:
    """A finished lifecycle run: its terminal state and every
    event it produced, each exactly once."""

    def __init__(self, state: JobState, events: list) -> None:
        self.state = state
        self.events = list(events)


def _as_snapshot(poll):
    if isinstance(poll, dict):
        return poll
    return poll.to_dict()


def run_lifecycle(provider, model, request) -> Completed:
    """Submit, poll to the end, and check the whole life cycle:
    every observed transition is legal, the terminal state is
    stable on further polls, and later polls repeat no events.
    ``model`` is the model the request runs against. Returns a
    :class:`Completed` with the terminal state and every event,
    each exactly once."""
    try:
        job_id = provider.submit(model, request)
    except AiError as exc:
        raise ConformanceError(f"submit failed: {exc}") from exc
    prev = JobState.QUEUED
    events: list = []
    for _ in range(MAX_POLLS):
        try:
            snapshot = _as_snapshot(provider.poll(job_id))
        except AiError as exc:
            raise ConformanceError(f"poll failed: {exc}") from exc
        state = JobState.from_dict(snapshot.get("state"))
        if not prev.can_become(state):
            raise ConformanceError(
                f"illegal transition {prev.kind} -> {state.kind}")
        for event in snapshot.get("events", []):
            if event in events:
                raise ConformanceError(f"an event repeated: {event!r}")
            events.append(event)
        prev = state
        if prev.is_terminal():
            for _ in range(2):
                try:
                    again = _as_snapshot(provider.poll(job_id))
                except AiError as exc:
                    raise ConformanceError(
                        f"poll after finish failed: {exc}") from exc
                if again.get("state") != snapshot.get("state"):
                    raise ConformanceError(
                        "terminal state changed after the job finished")
                if again.get("events"):
                    raise ConformanceError(
                        "events were repeated after the job finished")
            return Completed(prev, events)
    raise ConformanceError(
        f"a job did not finish in {MAX_POLLS} polls")


def run_cancel(provider, model, request) -> None:
    """Cancelling a running job ends in ``Cancelled``; cancelling
    twice is not an error. A job that already finished before
    the cancel also passes."""
    try:
        job_id = provider.submit(model, request)
    except AiError as exc:
        raise ConformanceError(f"submit failed: {exc}") from exc
    try:
        before = _as_snapshot(provider.poll(job_id))
    except AiError as exc:
        raise ConformanceError(f"poll failed: {exc}") from exc
    try:
        provider.cancel(job_id)
        provider.cancel(job_id)
    except AiError as exc:
        raise ConformanceError(f"cancel failed: {exc}") from exc
    last = JobState.from_dict(before.get("state"))
    for _ in range(MAX_POLLS):
        try:
            snapshot = _as_snapshot(provider.poll(job_id))
        except AiError as exc:
            raise ConformanceError(f"poll failed: {exc}") from exc
        state = JobState.from_dict(snapshot.get("state"))
        if not last.can_become(state):
            raise ConformanceError(
                f"illegal transition {last.kind} -> {state.kind}")
        last = state
        if last.is_terminal():
            break
    before_state = JobState.from_dict(before.get("state"))
    if before_state.is_terminal() or last == JobState.CANCELLED:
        return
    raise ConformanceError(f"a job ended as {last.kind} after cancel")


def run_rejects(provider, model, request, kind) -> None:
    """A request the model cannot handle must be refused at
    submit with the given kind."""
    try:
        provider.submit(model, request)
    except AiError as exc:
        if str(exc.kind) != str(kind):
            raise ConformanceError(
                f"a refused request came back {exc.kind}, not {kind}"
            ) from exc
        return
    raise ConformanceError(
        "a request no model can serve was accepted")


def run_unknown_job(provider) -> None:
    """Polling and cancelling a job that was never made is an
    error."""
    try:
        provider.poll("no-such-job")
    except AiError:
        pass
    else:
        raise ConformanceError("polling an unknown job did not fail")
    try:
        provider.cancel("no-such-job")
    except AiError:
        pass
    else:
        raise ConformanceError(
            "cancelling an unknown job did not fail")
