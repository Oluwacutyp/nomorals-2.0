"""Inference-job system — async generation jobs with tokens, polling,
provider webhooks and wallet gating.

Pattern mined from storytold/artcraft-services (fair-source: pattern
only, no code copied — see ~/workspace/jobsystem-verify/
JOBSYSTEM_SURVEY.md). The mined design, in Devon's own words:

* A generation request is validated, priced, billed against the
  caller's wallet, recorded as an inference job, and answered with a
  job token (``jinf_…``). The caller polls a job-status endpoint until
  the job reaches a terminal state, then reads the result URL.
* The job state machine has eight states. Three are live —
  ``pending``, ``started``, ``attempt_failed`` — and five are terminal:
  ``complete_success``, ``complete_failure``, ``dead``,
  ``cancelled_by_user``, ``cancelled_by_system``. An attempt that
  fails goes to ``attempt_failed`` and is retried; a job whose
  attempts are exhausted goes to ``dead``.
* Completion arrives two ways: provider webhooks (the provider calls
  back into the API when its work finishes) or polling workers (the
  API asks the provider). Clients learn the outcome by polling.
* Every request carries a fresh idempotency token (UUID). Reusing one
  is rejected as a duplicate.
* A wallet that cannot cover the price answers ``402 Payment
  Required`` — insufficient credits/balance.
* The job-status answer carries the job token, the original request
  (category, prompt token, model), the status (state, first-started
  time, failure category, progress 0–100), the result (null until
  success — then the finished media with its CDN URL), and the
  created/updated times. A batch status call takes many tokens and
  returns one state per token.

Devon's own additions on top of the mined pattern (marked as such in
the survey):

* The wallet is a real local credit ledger (:class:`WalletStore`,
  JSON, atomically written) — there is no credits organ in the repo
  today, so the store ships one instead of a fake balance. A job
  system with no wallet attached runs unbilled and says so on the
  record (``wallet: "unavailable"``); once a wallet is attached and
  funded, the 402 gate is live.
* The cost table is Devon's own pricing per generation category.
* Outbound client webhooks: a job may carry a ``webhook_url``. On a
  terminal state the system POSTs the signed job state to it
  (HMAC-SHA256 over the raw body, stable delivery id across retries,
  exponential backoff, per-attempt delivery log). Upstream's clients
  poll only; the push half is Devon's own, so a client that cannot
  poll still learns the outcome.
* The provider that does the work is injected: :meth:`run_local`
  drives a job through a caller-supplied executor, and the API's
  provider-webhook route drives jobs from real provider callbacks.
  Nothing here pretends to generate.

Layer: L1 (core). Imports stdlib + :mod:`nomorals.core.errors` +
:mod:`nomorals.core.http` only. The wallet is an injected protocol,
never an import of the commerce organ (L4).
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .errors import NoMoralsError, PaymentRequired, ValidationError

__all__ = [
    "CATEGORIES",
    "COSTS",
    "DuplicateIdempotencyToken",
    "IllegalTransition",
    "InferenceJob",
    "InferenceJobStatus",
    "InferenceJobStore",
    "InsufficientBalance",
    "TERMINAL_STATUSES",
    "TRANSITIONS",
    "WalletStore",
    "estimate_cost",
    "new_job_token",
    "utcnow_iso",
    "webhook_retry_delay",
]

#: Generation categories the job system prices. The names mirror the
#: upstream inference-category vocabulary (the pattern); the models
#: inside each category are Devon's own, supplied per request.
CATEGORIES: frozenset[str] = frozenset({
    "image_generation",
    "video_generation",
    "audio_generation",
    "text_generation",
    "mesh_generation",
    "splat_generation",
})

#: Devon's own credit prices (abstract credit units). ``base`` is
#: charged per job; ``per_second`` is charged per second of timed
#: output (video/audio ``duration_seconds``). These are Devon's
#: numbers, not the upstream's — the upstream's prices live in its
#: own billing service and were not mined.
COSTS: dict[str, dict[str, int]] = {
    "image_generation": {"base": 10},
    "video_generation": {"base": 60, "per_second": 10},
    "audio_generation": {"base": 20, "per_second": 2},
    "text_generation": {"base": 5},
    "mesh_generation": {"base": 40},
    "splat_generation": {"base": 40},
}

#: Retries after a failed webhook delivery: 60s, then 120s, then
#: 240s. After the third retry the delivery is dead-lettered — the
#: log keeps every attempt and the job's state stays authoritative,
#: so a missed delivery is always recoverable by polling.
WEBHOOK_RETRY_DELAYS_S: tuple[float, ...] = (60.0, 120.0, 240.0)

#: Attempts a job gets before it goes ``dead``.
DEFAULT_MAX_ATTEMPTS = 3

#: Terminal states older than this are swept from an in-memory-only
#: store's registry (debugging visibility without unbounded growth).
#: A persisted store keeps its rows — the file is the record.
TERMINAL_JOB_TTL_S = 3600.0

#: Header carrying the delivery id (stable across retries of one event).
DELIVERY_ID_HEADER = "X-Devon-Delivery-Id"
#: Header carrying the HMAC-SHA256 signature over the raw body.
SIGNATURE_HEADER = "X-Devon-Signature"
#: Header naming the event (``job.completed`` / ``job.failed`` /
#: ``job.cancelled``).
EVENT_HEADER = "X-Devon-Event"

#: Crockford base32 without the ambiguous letters (i, l, o, u) —
#: the same alphabet the upstream's ``artcraft_api_`` keys use.
_TOKEN_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"


def utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def new_job_token() -> str:
    """A fresh inference-job token: ``jinf_`` + 25 Crockford chars."""
    return "jinf_" + "".join(
        secrets.choice(_TOKEN_ALPHABET) for _ in range(25))


def webhook_retry_delay(attempt: int) -> float | None:
    """Seconds to wait before webhook-delivery retry ``attempt`` (1-based).

    Returns ``None`` when no retries remain (the delivery is
    dead-lettered after the third retry).
    """
    if attempt < 1 or attempt > len(WEBHOOK_RETRY_DELAYS_S):
        return None
    return WEBHOOK_RETRY_DELAYS_S[attempt - 1]


def estimate_cost(category: str, params: Mapping[str, Any] | None = None) -> int:
    """Credit price of a job before it is submitted.

    ``duration_seconds`` in ``params`` scales timed categories
    (video/audio). Raises :class:`ValidationError` for an unknown
    category or a negative duration.
    """
    if category not in CATEGORIES:
        raise ValidationError(
            f"unknown inference category {category!r} — expected one of "
            + ", ".join(sorted(CATEGORIES)))
    spec = COSTS[category]
    duration = 0.0
    if params:
        raw = params.get("duration_seconds", 0)
        try:
            duration = float(raw or 0)
        except (TypeError, ValueError):
            raise ValidationError(
                "duration_seconds must be a number") from None
        if duration < 0:
            raise ValidationError("duration_seconds must not be negative")
    return int(spec["base"] + spec.get("per_second", 0) * duration)


class InferenceJobStatus(str):
    """The eight job states (``JobStatusPlus`` in the mined design).

    Wire values are snake_case, exactly as the upstream exposes them
    — the vocabulary is the pattern. Three states are live
    (``pending``, ``started``, ``attempt_failed``); five are terminal.
    """

    PENDING = "pending"
    STARTED = "started"
    COMPLETE_SUCCESS = "complete_success"
    COMPLETE_FAILURE = "complete_failure"
    ATTEMPT_FAILED = "attempt_failed"
    DEAD = "dead"
    CANCELLED_BY_USER = "cancelled_by_user"
    CANCELLED_BY_SYSTEM = "cancelled_by_system"

    ALL: tuple[str, ...] = (
        PENDING, STARTED, COMPLETE_SUCCESS, COMPLETE_FAILURE,
        ATTEMPT_FAILED, DEAD, CANCELLED_BY_USER, CANCELLED_BY_SYSTEM,
    )

    @classmethod
    def from_str(cls, value: str) -> str:
        """Parse a wire value, or raise :class:`ValidationError`."""
        if value in cls.ALL:
            return value
        raise ValidationError(
            f"invalid job status {value!r} — expected one of "
            + ", ".join(cls.ALL))

    @classmethod
    def is_terminal(cls, value: str) -> bool:
        """True when no further transition may leave this state."""
        return value in TERMINAL_STATUSES


#: Terminal states — a job that reaches one never moves again.
TERMINAL_STATUSES: frozenset[str] = frozenset({
    InferenceJobStatus.COMPLETE_SUCCESS,
    InferenceJobStatus.COMPLETE_FAILURE,
    InferenceJobStatus.DEAD,
    InferenceJobStatus.CANCELLED_BY_USER,
    InferenceJobStatus.CANCELLED_BY_SYSTEM,
})

#: Legal transitions. Terminal states have no outgoing edge.
#: ``attempt_failed -> started`` is the retry; ``started -> dead``
#: is the exhaustion edge — when the failing attempt was the last
#: one allowed, the job goes straight to ``dead`` instead of
#: lingering in ``attempt_failed`` with no retries left (see
#: :meth:`InferenceJobStore.record_attempt_failure`).
TRANSITIONS: dict[str, frozenset[str]] = {
    InferenceJobStatus.PENDING: frozenset({
        InferenceJobStatus.STARTED,
        InferenceJobStatus.CANCELLED_BY_USER,
        InferenceJobStatus.CANCELLED_BY_SYSTEM,
    }),
    InferenceJobStatus.STARTED: frozenset({
        InferenceJobStatus.COMPLETE_SUCCESS,
        InferenceJobStatus.COMPLETE_FAILURE,
        InferenceJobStatus.ATTEMPT_FAILED,
        InferenceJobStatus.DEAD,
        InferenceJobStatus.CANCELLED_BY_USER,
        InferenceJobStatus.CANCELLED_BY_SYSTEM,
    }),
    InferenceJobStatus.ATTEMPT_FAILED: frozenset({
        InferenceJobStatus.STARTED,
        InferenceJobStatus.DEAD,
        InferenceJobStatus.CANCELLED_BY_USER,
        InferenceJobStatus.CANCELLED_BY_SYSTEM,
    }),
}


class IllegalTransition(NoMoralsError):
    """A transition the state machine does not allow."""

    code = "job.illegal_transition"


class DuplicateIdempotencyToken(NoMoralsError):
    """A submit reused an idempotency token — rejected as a duplicate.

    The existing job's token rides in ``details`` so the caller can
    poll the original instead of resubmitting.
    """

    code = "job.duplicate_idempotency_token"


class InsufficientBalance(NoMoralsError):
    """A wallet charge would overdraw — the 402's cause, pre-mapping."""

    code = "wallet.insufficient_balance"


class WalletLike(Protocol):
    """What the job system needs from a wallet. Anything that
    implements ``balance`` + ``charge`` plugs in — the local
    :class:`WalletStore`, or a future commerce wallet."""

    def balance(self) -> int: ...

    def charge(self, amount: int, *, job_token: str) -> dict[str, Any]: ...

    def bills(self) -> list[dict[str, Any]]: ...


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically: temp file in the same directory + fsync
    + rename — the same discipline as the unified scheduler's store,
    so a crash mid-write never leaves a torn record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


@dataclass
class WalletStore:
    """A real local credit ledger, persisted as JSON.

    Credits are integer units. :meth:`charge` deducts and returns the
    bill (amount, balances before/after, job token, time); it raises
    :class:`InsufficientBalance` instead of going negative. There is
    no credit creation here — :meth:`credit` is the only way money
    enters, and it is the owner's explicit act.
    """

    path: Path | str | None = None

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._balance = 0
        self._bills: list[dict[str, Any]] = []
        if self.path is not None:
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        p = Path(self.path)
        if not p.is_file():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict):
            try:
                self._balance = int(data.get("balance", 0))
            except (TypeError, ValueError):
                self._balance = 0
            bills = data.get("bills", [])
            if isinstance(bills, list):
                self._bills = [b for b in bills if isinstance(b, dict)]

    def _save(self) -> None:
        if self.path is None:
            return
        _atomic_write_json(Path(self.path), {
            "balance": self._balance,
            "bills": self._bills[-500:],
        })

    def balance(self) -> int:
        """Current credit balance."""
        with self._lock:
            return self._balance

    def credit(self, amount: int, *, note: str = "") -> dict[str, Any]:
        """Add credits. Only the owner funds a wallet — never negative,
        never zero."""
        if not isinstance(amount, int) or isinstance(amount, bool) \
                or amount <= 0:
            raise ValidationError("credit amount must be a positive integer")
        with self._lock:
            before = self._balance
            self._balance += amount
            entry = {
                "kind": "credit", "amount": amount,
                "balance_before": before, "balance_after": self._balance,
                "note": str(note or ""), "at": utcnow_iso(),
            }
            self._bills.append(entry)
            self._save()
            return dict(entry)

    def charge(self, amount: int, *, job_token: str) -> dict[str, Any]:
        """Deduct ``amount`` for ``job_token``. Raises
        :class:`InsufficientBalance` (with required/available/
        shortfall) instead of going negative."""
        if not isinstance(amount, int) or isinstance(amount, bool) \
                or amount < 0:
            raise ValidationError("charge amount must be a non-negative integer")
        with self._lock:
            if amount > self._balance:
                raise InsufficientBalance(
                    f"insufficient credits: need {amount}, "
                    f"have {self._balance}",
                    details={
                        "required": amount,
                        "available": self._balance,
                        "shortfall": amount - self._balance,
                    })
            before = self._balance
            self._balance -= amount
            bill = {
                "kind": "charge", "amount": amount,
                "balance_before": before, "balance_after": self._balance,
                "job_token": job_token, "at": utcnow_iso(),
            }
            self._bills.append(bill)
            self._save()
            return dict(bill)

    def bills(self) -> list[dict[str, Any]]:
        """The ledger, oldest first (credits and charges)."""
        with self._lock:
            return [dict(b) for b in self._bills]


@dataclass
class InferenceJob:
    """One inference job — the record the store keeps and the poll
    endpoint answers with."""

    token: str
    idempotency_token: str
    category: str
    model: str
    prompt: str
    params: dict[str, Any] = field(default_factory=dict)
    status: str = InferenceJobStatus.PENDING
    progress_percentage: int = 0
    failure_category: str | None = None
    first_started_at: str | None = None
    result: dict[str, Any] | None = None
    attempts: int = 0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    cost_credits: int = 0
    billed: bool = False
    bill: dict[str, Any] | None = None
    wallet: str = "unavailable"
    webhook_url: str | None = None
    provider: str | None = None
    provider_job_id: str | None = None
    last_progress_note: str = ""
    deliveries: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)

    def to_state(self) -> dict[str, Any]:
        """The job-status answer — the mined Omni shape: token,
        request, status block, result, created/updated. Extra Devon
        fields (cost, wallet, deliveries) ride alongside, never
        instead of the mined shape."""
        return {
            "job_token": self.token,
            "request": {
                "inference_category": self.category,
                "maybe_prompt_token": None,
                "maybe_model_type": self.model,
                "maybe_model_token": None,
                "prompt": self.prompt,
                "params": dict(self.params),
            },
            "status": {
                "status": self.status,
                "maybe_first_started_at": self.first_started_at,
                "maybe_failure_category": self.failure_category,
                "progress_percentage": self.progress_percentage,
            },
            "maybe_result": self.result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "cost_credits": self.cost_credits,
            "billed": self.billed,
            "wallet": self.wallet,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "last_progress_note": self.last_progress_note,
        }

    def to_row(self) -> dict[str, Any]:
        """The persisted row — everything, except the webhook secret
        (secrets never hit disk; they live in the store's in-memory
        map keyed by token)."""
        row = {
            "token": self.token,
            "idempotency_token": self.idempotency_token,
            "category": self.category,
            "model": self.model,
            "prompt": self.prompt,
            "params": dict(self.params),
            "status": self.status,
            "progress_percentage": self.progress_percentage,
            "failure_category": self.failure_category,
            "first_started_at": self.first_started_at,
            "result": self.result,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "cost_credits": self.cost_credits,
            "billed": self.billed,
            "bill": self.bill,
            "wallet": self.wallet,
            "webhook_url": self.webhook_url,
            "provider": self.provider,
            "provider_job_id": self.provider_job_id,
            "last_progress_note": self.last_progress_note,
            "deliveries": [dict(d) for d in self.deliveries],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        return row

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> InferenceJob:
        """Rebuild a job from a persisted row. Unknown fields are
        ignored so an older file still loads."""
        job = cls(
            token=str(row["token"]),
            idempotency_token=str(row.get("idempotency_token", "")),
            category=str(row.get("category", "")),
            model=str(row.get("model", "")),
            prompt=str(row.get("prompt", "")),
            params=dict(row.get("params") or {}),
            status=str(row.get("status", InferenceJobStatus.PENDING)),
            progress_percentage=int(row.get("progress_percentage", 0) or 0),
            failure_category=row.get("failure_category"),
            first_started_at=row.get("first_started_at"),
            result=row.get("result"),
            attempts=int(row.get("attempts", 0) or 0),
            max_attempts=int(row.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
                             or DEFAULT_MAX_ATTEMPTS),
            cost_credits=int(row.get("cost_credits", 0) or 0),
            billed=bool(row.get("billed", False)),
            bill=row.get("bill"),
            wallet=str(row.get("wallet", "unavailable")),
            webhook_url=row.get("webhook_url"),
            provider=row.get("provider"),
            provider_job_id=row.get("provider_job_id"),
            last_progress_note=str(row.get("last_progress_note", "")),
            deliveries=[dict(d) for d in (row.get("deliveries") or [])
                        if isinstance(d, dict)],
            created_at=str(row.get("created_at", utcnow_iso())),
            updated_at=str(row.get("updated_at", utcnow_iso())),
        )
        return job


def _verify_signature(secret: str, raw_body: bytes,
                      signature: str) -> bool:
    """HMAC-SHA256 over the exact bytes the sender signed, constant-
    time compare — the same convention as the commerce gateway's
    ``webhook_verify``. Never raises: a verification failure is
    ``False``."""
    try:
        if not secret or not signature:
            return False
        expected = hmac.new(secret.encode("utf-8"), raw_body,
                            hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature.strip())
    except Exception:  # noqa: BLE001 - verification failure is False
        return False


def _http_post_status(url: str, body: bytes, headers: dict[str, str],
                      *, allow_private_ips: bool = False) -> int:
    """POST and return the HTTP status, through Devon's HttpClient.

    HttpClient raises on a non-2xx status, but the status is
    delivery DATA (a 500 means "retry", a 400 means "dead-letter"),
    so it is pulled back out of the error's ``details`` and returned
    instead of letting the exception swallow it. A transport failure
    with no status still raises.
    """
    from nomorals.core.http import HttpClient

    client = HttpClient(allow_private_ips=allow_private_ips)
    try:
        resp = client.post(url, data=body, headers=headers, timeout=10.0)
        return int(resp.status)
    except Exception as exc:  # noqa: BLE001 - delivery failure is data
        details = getattr(exc, "details", None) or {}
        status = details.get("http_status")
        if isinstance(status, int):
            return status
        raise


class InferenceJobStore:
    """The inference-job registry: state machine, billing gate,
    polling answers, provider webhooks in, client webhooks out.

    Thread-safe. ``path`` persists every job as JSON (atomic
    writes); ``None`` keeps the registry in memory only. ``wallet``
    is optional — ``None`` means no wallet is attached and jobs run
    unbilled, honestly marked ``wallet: "unavailable"``; attach one
    (``attach_wallet``) and the 402 gate goes live. ``post`` is the
    outbound-webhook sender, ``(url, body, headers) -> status`` —
    tests inject a fake; the default posts through Devon's
    :class:`nomorals.core.http.HttpClient` with its SSRF guard on.
    """

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        wallet: WalletLike | None = None,
        clock: Callable[[], str] | None = None,
        post: Callable[[str, bytes, dict[str, str]], int] | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self._wallet = wallet
        self._clock = clock or utcnow_iso
        self._post = post or self._default_post
        self._lock = threading.RLock()
        self._jobs: dict[str, InferenceJob] = {}
        self._by_idempotency: dict[str, str] = {}
        self._by_provider: dict[tuple[str, str], str] = {}
        self._seen_provider_events: dict[str, dict[str, Any]] = {}
        self._webhook_secrets: dict[str, str] = {}
        self._provider_secrets: dict[str, str] = {}
        if self.path is not None:
            self._load()

    # ── persistence ───────────────────────────────────────────────

    def _load(self) -> None:
        assert self.path is not None
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        for row in data.get("jobs", []):
            if not isinstance(row, dict) or "token" not in row:
                continue
            try:
                job = InferenceJob.from_row(row)
            except (KeyError, TypeError, ValueError):
                continue
            self._jobs[job.token] = job
            if job.idempotency_token:
                self._by_idempotency[job.idempotency_token] = job.token
            if job.provider and job.provider_job_id:
                self._by_provider[(job.provider, job.provider_job_id)] = \
                    job.token

    def _save(self) -> None:
        if self.path is None:
            return
        _atomic_write_json(self.path, {
            "jobs": [j.to_row() for j in self._jobs.values()],
        })

    def _touch(self, job: InferenceJob) -> None:
        job.updated_at = self._clock()

    # ── wallet ────────────────────────────────────────────────────

    def attach_wallet(self, wallet: WalletLike | None) -> None:
        """Attach (or detach, with ``None``) the wallet. Attaching a
        funded wallet turns the 402 gate on; detaching turns it off
        and later submits run unbilled again."""
        with self._lock:
            self._wallet = wallet

    def wallet(self) -> WalletLike | None:
        """The attached wallet, or ``None`` when jobs run unbilled."""
        with self._lock:
            return self._wallet

    # ── submission ────────────────────────────────────────────────

    def submit(
        self,
        category: str,
        model: str,
        prompt: str = "",
        *,
        params: Mapping[str, Any] | None = None,
        idempotency_token: str | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> InferenceJob:
        """Validate, price, bill, record — and answer with a job token.

        Raises :class:`ValidationError` on a bad category/model/
        idempotency token, :class:`DuplicateIdempotencyToken` when the
        token was already used, and :class:`PaymentRequired` (402)
        when an attached wallet cannot cover the price.
        """
        if category not in CATEGORIES:
            raise ValidationError(
                f"unknown inference category {category!r} — expected one "
                f"of {', '.join(sorted(CATEGORIES))}")
        if not str(model or "").strip():
            raise ValidationError("model is required")
        token = str(idempotency_token or "").strip()
        if not token:
            raise ValidationError(
                "idempotency_token is required — send a fresh UUID per "
                "request")
        try:
            uuid.UUID(token)
        except ValueError:
            raise ValidationError(
                "idempotency_token must be a UUID") from None
        params = dict(params or {})
        cost = estimate_cost(category, params)
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) \
                or max_attempts < 1:
            raise ValidationError("max_attempts must be a positive integer")
        with self._lock:
            if token in self._by_idempotency:
                existing = self._by_idempotency[token]
                raise DuplicateIdempotencyToken(
                    f"idempotency_token was already used — poll the "
                    f"original job {existing} instead of resubmitting",
                    details={"existing_job_token": existing})
            wallet = self._wallet
            bill = None
            billed = False
            wallet_name = "unavailable"
            job_token = new_job_token()
            while job_token in self._jobs:  # vanishingly unlikely; be exact
                job_token = new_job_token()
            if wallet is not None:
                try:
                    bill = wallet.charge(cost, job_token=job_token)
                except InsufficientBalance as exc:
                    raise PaymentRequired(
                        f"insufficient credits: need {cost}, have "
                        f"{exc.details.get('available', 0)}",
                        details={
                            "required": cost,
                            "available": exc.details.get("available", 0),
                            "shortfall": exc.details.get("shortfall", cost),
                        }) from exc
                billed = True
                wallet_name = "local"
            job = InferenceJob(
                token=job_token,
                idempotency_token=token,
                category=category,
                model=str(model).strip(),
                prompt=str(prompt or ""),
                params=params,
                max_attempts=max_attempts,
                cost_credits=cost,
                billed=billed,
                bill=bill,
                wallet=wallet_name,
                webhook_url=(str(webhook_url).strip() or None
                             if webhook_url else None),
                created_at=self._clock(),
                updated_at=self._clock(),
            )
            self._jobs[job.token] = job
            self._by_idempotency[token] = job.token
            if webhook_secret:
                self._webhook_secrets[job.token] = str(webhook_secret)
            self._save()
            return job

    def bind_provider(self, token: str, provider: str,
                      provider_job_id: str) -> InferenceJob:
        """Record the provider-side id for a job — the router calls
        this when the provider answers a submit with its own job id,
        so later provider callbacks can find Devon's job. Only a
        live job can be bound; a terminal job never gains a provider
        after the fact."""
        with self._lock:
            job = self._jobs.get(str(token or ""))
            if job is None:
                raise ValidationError(f"unknown job token {token!r}")
            if InferenceJobStatus.is_terminal(job.status):
                raise IllegalTransition(
                    f"cannot bind a provider to a {job.status} job")
            provider = str(provider or "").strip()
            provider_job_id = str(provider_job_id or "").strip()
            if not provider or not provider_job_id:
                raise ValidationError(
                    "provider and provider_job_id are required")
            job.provider = provider
            job.provider_job_id = provider_job_id
            self._by_provider[(provider, provider_job_id)] = job.token
            self._touch(job)
            self._save()
            return job

    def get(self, token: str) -> InferenceJob | None:
        """The job for ``token``, or ``None``."""
        with self._lock:
            return self._jobs.get(str(token or ""))

    def job_status(self, token: str) -> dict[str, Any]:
        """The single-job poll answer: ``{success, state}``."""
        job = self.get(token)
        if job is None:
            raise ValidationError(f"unknown job token {token!r}")
        return {"success": True, "state": job.to_state()}

    def job_status_batch(self, tokens: list[str]) -> dict[str, Any]:
        """The batch poll answer: ``{success, job_states}`` — one
        state per token, in the order asked. An unknown token yields
        an error entry, never a silent skip."""
        states: list[dict[str, Any]] = []
        for token in tokens:
            job = self.get(token)
            if job is None:
                states.append({
                    "job_token": str(token),
                    "error": f"unknown job token {token!r}",
                })
            else:
                states.append(job.to_state())
        return {"success": True, "job_states": states}

    def list_jobs(self, *, status: str | None = None,
                  limit: int = 50) -> list[InferenceJob]:
        """Jobs newest first, optionally filtered by status."""
        with self._lock:
            jobs = sorted(self._jobs.values(),
                          key=lambda j: j.created_at, reverse=True)
        if status is not None:
            InferenceJobStatus.from_str(status)
            jobs = [j for j in jobs if j.status == status]
        return jobs[:max(0, int(limit))]

    # ── the state machine ─────────────────────────────────────────

    def transition(self, token: str, target: str, *,
                   failure_category: str | None = None,
                   result: dict[str, Any] | None = None) -> InferenceJob:
        """Move a job to ``target`` if the machine allows it. Terminal
        states never move again; an illegal edge raises
        :class:`IllegalTransition` naming both ends. Reaching a
        terminal state with a ``webhook_url`` queues the outbound
        delivery (attempted by :meth:`deliver_webhooks`)."""
        target = InferenceJobStatus.from_str(target)
        with self._lock:
            job = self._jobs.get(str(token or ""))
            if job is None:
                raise ValidationError(f"unknown job token {token!r}")
            allowed = TRANSITIONS.get(job.status, frozenset())
            if target not in allowed:
                raise IllegalTransition(
                    f"cannot move job {job.token} from {job.status} to "
                    f"{target}")
            if target == InferenceJobStatus.STARTED \
                    and job.first_started_at is None:
                job.first_started_at = self._clock()
            job.status = target
            if failure_category is not None:
                job.failure_category = str(failure_category)
            if result is not None:
                job.result = dict(result)
            if target == InferenceJobStatus.COMPLETE_SUCCESS:
                job.progress_percentage = 100
            self._touch(job)
            if InferenceJobStatus.is_terminal(target) and job.webhook_url:
                self._queue_delivery(job)
            self._save()
            return job

    def start(self, token: str) -> InferenceJob:
        """``pending``/``attempt_failed`` → ``started`` (the retry
        re-enters here)."""
        return self.transition(token, InferenceJobStatus.STARTED)

    def report_progress(self, token: str, percentage: int,
                        note: str | None = None) -> InferenceJob:
        """Set progress 0–100 on a ``started`` job. Progress is a
        report, not a transition — a job that is not started cannot
        report it."""
        with self._lock:
            job = self._jobs.get(str(token or ""))
            if job is None:
                raise ValidationError(f"unknown job token {token!r}")
            if job.status != InferenceJobStatus.STARTED:
                raise IllegalTransition(
                    f"cannot report progress on a {job.status} job")
            job.progress_percentage = max(0, min(100, int(percentage)))
            if note:
                job.last_progress_note = str(note)[:300]
            self._touch(job)
            self._save()
            return job

    def complete(self, token: str,
                 result: dict[str, Any]) -> InferenceJob:
        """``started`` → ``complete_success`` with the finished media.
        ``result`` must carry the media links — a success without a
        result URL is a lie the poller cannot use."""
        if not isinstance(result, dict) or not result:
            raise ValidationError(
                "a completed job needs a result (media links)")
        return self.transition(token, InferenceJobStatus.COMPLETE_SUCCESS,
                               result=result)

    def fail(self, token: str, failure_category: str | None = None,
             message: str | None = None) -> InferenceJob:
        """``started`` → ``complete_failure``. The category is the
        machine-readable reason; the message (when given) is recorded
        on the result for humans."""
        result = {"error": str(message or "")[:500]} if message else None
        return self.transition(token, InferenceJobStatus.COMPLETE_FAILURE,
                               failure_category=failure_category,
                               result=result)

    def record_attempt_failure(
            self, token: str, failure_category: str | None = None) -> InferenceJob:
        """One attempt failed. Attempts remain → ``attempt_failed``
        (retryable); attempts exhausted → ``dead``."""
        with self._lock:
            job = self._jobs.get(str(token or ""))
            if job is None:
                raise ValidationError(f"unknown job token {token!r}")
            if job.status != InferenceJobStatus.STARTED:
                raise IllegalTransition(
                    "record_attempt_failure needs a started job, got "
                    f"{job.status}")
            job.attempts += 1
            target = (InferenceJobStatus.DEAD
                      if job.attempts >= job.max_attempts
                      else InferenceJobStatus.ATTEMPT_FAILED)
            # transition() re-locks (RLock) and re-reads the job.
            return self.transition(token, target,
                                   failure_category=failure_category)

    def cancel(self, token: str, *, by: str = "user") -> InferenceJob:
        """Cancel a live job: ``by="user"`` → ``cancelled_by_user``,
        ``by="system"`` → ``cancelled_by_system``."""
        if by not in ("user", "system"):
            raise ValidationError("cancel by must be 'user' or 'system'")
        target = (InferenceJobStatus.CANCELLED_BY_USER if by == "user"
                  else InferenceJobStatus.CANCELLED_BY_SYSTEM)
        return self.transition(token, target)

    # ── running a job locally ─────────────────────────────────────

    def run_local(
        self, token: str,
        executor: Callable[[InferenceJob], dict[str, Any]],
    ) -> InferenceJob:
        """Drive a ``pending`` job through a caller-supplied executor.

        The executor receives the job and returns the result dict
        (media links); an exception from it records an attempt
        failure instead of crashing the store. This is the seam the
        media organs will plug real generation into — the store never
        pretends to generate on its own.
        """
        job = self.get(token)
        if job is None:
            raise ValidationError(f"unknown job token {token!r}")
        if job.status != InferenceJobStatus.PENDING:
            raise IllegalTransition(
                f"run_local needs a pending job, got {job.status}")
        self.start(token)
        try:
            result = executor(job)
        except Exception:  # noqa: BLE001 - failure is a job outcome
            return self.record_attempt_failure(
                token, failure_category="executor_error")
        if not isinstance(result, dict) or not result:
            return self.record_attempt_failure(
                token, failure_category="empty_result")
        return self.complete(token, result)

    # ── provider webhooks (inbound) ───────────────────────────────

    def register_provider_secret(self, provider: str, secret: str) -> None:
        """Register the shared secret a provider signs its callbacks
        with. Secrets live in memory only — never in the job file."""
        with self._lock:
            self._provider_secrets[str(provider)] = str(secret)

    def receive_provider_webhook(
        self,
        provider: str,
        event: Mapping[str, Any],
        *,
        signature: str | None = None,
        raw_body: bytes | None = None,
    ) -> dict[str, Any]:
        """Handle one provider callback and drive the job it names.

        ``event`` carries ``provider_job_id`` and ``event`` — one of
        ``started`` / ``progress`` / ``succeeded`` / ``failed`` —
        plus ``progress`` / ``result`` / ``failure_category`` as the
        event needs. When a secret is registered for ``provider`` the
        signature (HMAC-SHA256 over ``raw_body``) is REQUIRED and a
        bad one is rejected; with no secret registered the callback
        is accepted and marked ``signature_verified: false`` — an
        explicit opt-in, never silent. Delivery is idempotent by
        ``event_id`` (or ``provider_job_id`` + event): a replay
        returns the first outcome without moving the job twice.
        """
        provider = str(provider or "").strip()
        if not provider:
            raise ValidationError("provider is required")
        event = dict(event or {})
        provider_job_id = str(event.get("provider_job_id") or "").strip()
        if not provider_job_id:
            raise ValidationError("provider_job_id is required")
        kind = str(event.get("event") or "").strip()
        if kind not in ("started", "progress", "succeeded", "failed"):
            raise ValidationError(
                f"unknown provider event {kind!r} — expected started, "
                "progress, succeeded or failed")
        event_id = str(event.get("event_id") or
                       f"{provider}:{provider_job_id}:{kind}")
        with self._lock:
            # Verification comes BEFORE the replay check: a replayed
            # event with a forged signature must be rejected, not
            # answered from the cache.
            secret = self._provider_secrets.get(provider)
            verified = False
            if secret is not None:
                if raw_body is None:
                    raise ValidationError(
                        "raw_body is required to verify a signed "
                        "provider webhook")
                verified = _verify_signature(secret, raw_body,
                                             signature or "")
                if not verified:
                    raise ValidationError(
                        "provider webhook signature did not verify")
            if event_id in self._seen_provider_events:
                return dict(self._seen_provider_events[event_id])
            token = self._by_provider.get((provider, provider_job_id))
            if token is None:
                raise ValidationError(
                    f"no job for provider {provider} job "
                    f"{provider_job_id!r}")
            job = self._jobs[token]
            if kind == "started" and job.status in (
                    InferenceJobStatus.PENDING,
                    InferenceJobStatus.ATTEMPT_FAILED):
                self.transition(token, InferenceJobStatus.STARTED)
            elif kind == "progress":
                if job.status == InferenceJobStatus.STARTED:
                    try:
                        pct = int(event.get("progress", 0))
                    except (TypeError, ValueError):
                        pct = job.progress_percentage
                    self.report_progress(token, pct)
            elif kind == "succeeded":
                result = event.get("result")
                if not isinstance(result, dict) or not result:
                    raise ValidationError(
                        "a succeeded provider event needs a result")
                if job.status == InferenceJobStatus.STARTED:
                    self.complete(token, dict(result))
            elif kind == "failed" \
                    and job.status == InferenceJobStatus.STARTED:
                self.record_attempt_failure(
                    token,
                    failure_category=str(
                        event.get("failure_category") or "provider_error"))
            outcome = {
                "success": True,
                "job_token": token,
                "status": self._jobs[token].status,
                "signature_verified": verified,
                "duplicate": False,
            }
            self._seen_provider_events[event_id] = dict(outcome)
            return dict(outcome)

    # ── client webhooks (outbound) ────────────────────────────────

    def _queue_delivery(self, job: InferenceJob) -> None:
        """Record a pending delivery for a terminal job. The delivery
        id is stable across retries of the same terminal event, so a
        receiver can dedupe."""
        event = {
            InferenceJobStatus.COMPLETE_SUCCESS: "job.completed",
            InferenceJobStatus.COMPLETE_FAILURE: "job.failed",
            InferenceJobStatus.DEAD: "job.failed",
            InferenceJobStatus.CANCELLED_BY_USER: "job.cancelled",
            InferenceJobStatus.CANCELLED_BY_SYSTEM: "job.cancelled",
        }[job.status]
        job.deliveries.append({
            "delivery_id": f"dlv_{job.token[5:]}_{job.status}",
            "event": event,
            "url": job.webhook_url,
            "status": "pending",
            "attempts": 0,
            "next_attempt_at": None,
            "attempt_log": [],
        })

    @staticmethod
    def _delivery_event(status: str) -> str:
        return {
            InferenceJobStatus.COMPLETE_SUCCESS: "job.completed",
            InferenceJobStatus.COMPLETE_FAILURE: "job.failed",
            InferenceJobStatus.DEAD: "job.failed",
            InferenceJobStatus.CANCELLED_BY_USER: "job.cancelled",
            InferenceJobStatus.CANCELLED_BY_SYSTEM: "job.cancelled",
        }[status]

    def _default_post(self, url: str, body: bytes,
                      headers: dict[str, str]) -> int:
        """POST through Devon's HttpClient (SSRF guard on — a webhook
        URL that points at a private address is refused, honestly).
        Returns the HTTP status; a transport failure with no status
        raises and the delivery is retried on schedule."""
        return _http_post_status(url, body, headers)

    def deliver_webhooks(self, now: float | None = None) -> list[dict[str, Any]]:
        """Attempt every due delivery: pending ones, plus failed ones
        whose backoff has elapsed. Returns one outcome per delivery
        attempted. A delivery that exhausts its retries is
        dead-lettered (``status: "failed"``) — the log keeps every
        attempt and the job's state stays authoritative, so polling
        always recovers a missed push."""
        now = time.time() if now is None else float(now)
        outcomes: list[dict[str, Any]] = []
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            with self._lock:
                pending = [d for d in job.deliveries
                           if d["status"] in ("pending", "retrying")
                           and (d.get("next_attempt_at") is None
                                or d["next_attempt_at"] <= now)]
                deliveries = list(pending)
            for delivery in deliveries:
                outcomes.append(self._attempt_delivery(job, delivery, now))
        with self._lock:
            self._save()
        return outcomes

    def _attempt_delivery(self, job: InferenceJob,
                          delivery: dict[str, Any],
                          now: float) -> dict[str, Any]:
        payload = json.dumps(
            {"success": True, "state": job.to_state()},
            ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            DELIVERY_ID_HEADER: str(delivery["delivery_id"]),
            EVENT_HEADER: str(delivery["event"]),
        }
        secret = self._webhook_secrets.get(job.token)
        if secret:
            headers[SIGNATURE_HEADER] = hmac.new(
                secret.encode("utf-8"), payload,
                hashlib.sha256).hexdigest()
        status_code: int | None = None
        error = ""
        try:
            status_code = int(self._post(str(delivery["url"]), payload,
                                         headers))
        except Exception as exc:  # noqa: BLE001 - delivery failure is data
            error = f"{type(exc).__name__}: {exc}"[:300]
        with self._lock:
            delivery["attempts"] = int(delivery.get("attempts", 0)) + 1
            attempt_no = delivery["attempts"]
            delivery["attempt_log"].append({
                "attempt": attempt_no,
                "status_code": status_code,
                "error": error,
                "at": utcnow_iso(),
            })
            if status_code is not None and 200 <= status_code < 300:
                delivery["status"] = "delivered"
                delivery["next_attempt_at"] = None
            else:
                delay = webhook_retry_delay(attempt_no)
                if delay is None:
                    delivery["status"] = "failed"
                    delivery["next_attempt_at"] = None
                else:
                    delivery["status"] = "retrying"
                    delivery["next_attempt_at"] = now + delay
            self._save()
        return {
            "job_token": job.token,
            "delivery_id": delivery["delivery_id"],
            "event": delivery["event"],
            "attempt": attempt_no,
            "status_code": status_code,
            "status": delivery["status"],
            "error": error,
        }
