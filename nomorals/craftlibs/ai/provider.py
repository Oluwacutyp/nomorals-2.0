"""Providers (Craft-Libs port of ai-core ``provider.rs``).

What an engine behind a model looks like to the rest of the
layer: which models it offers, what a job should cost, and the
submit / poll / cancel cycle every provider keeps. A provider
answers synchronously when it can, and returns a job otherwise.

Every call names the model it runs against: submit validates
the request against that model's spec and refuses an unknown
model before anything is made.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["Provider"]


class Provider(ABC):
    """One engine behind one or more models."""

    @abstractmethod
    def models(self) -> list:
        """The models this provider offers."""

    @abstractmethod
    def estimate(self, model, request) -> dict:
        """What a job should cost, before it is submitted."""

    @abstractmethod
    def submit(self, model, request) -> str:
        """Start a job for ``model``. Returns the job id. Refuses
        an unknown model, and a request the model cannot serve,
        before anything is made."""

    @abstractmethod
    def poll(self, job_id: str) -> dict:
        """One status check. Returns a snapshot dict with
        ``state`` and ``events`` — the events since the last
        poll, never repeated."""

    @abstractmethod
    def cancel(self, job_id: str) -> None:
        """Ask for a job to stop. Idempotent."""
