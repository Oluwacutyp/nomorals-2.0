"""Typed error hierarchy.

Design rule: exceptions signal *programmer error* or *unrecoverable state*.
Expected failure modes (network down, model overloaded, file missing) travel as
:class:`~nomorals.core.result.Err` values so callers cannot accidentally ignore them.

Every error carries:
  * ``code``    — stable machine-readable string, safe to match on across versions
  * ``message`` — human-readable, safe to log
  * ``details`` — structured payload
  * ``retryable`` — whether a retry with backoff could plausibly succeed
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any


# ── HTTP status helpers ──────────────────────────────────────────────────────
#: Status codes worth matching out of a free-text error message, for errors
#: that did not come through :func:`nomorals.core.http.http_error` (which
#: sets ``details["http_status"]`` directly).
_STATUS_RE = re.compile(r"\b(400|401|403|404|408|425|429|5\d\d)\b")


def http_status_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status for an error, or ``None`` when unknown.

    Prefers ``details["http_status"]`` (set by the HTTP layer); falls back
    to the first status-like code in the message, which covers errors
    built from raw exception strings (tests, third-party clients).
    """
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        status = details.get("http_status")
        if isinstance(status, bool):
            pass  # a bool is not a status; fall through to the message
        elif isinstance(status, int):
            return status
        elif isinstance(status, str) and status.isdigit():
            return int(status)
    text = str(getattr(exc, "message", "") or exc)
    match = _STATUS_RE.search(text)
    return int(match.group(1)) if match else None


def is_auth_error(exc: BaseException) -> bool:
    """401/403 — the key is bad or lacks access.  Terminal: retrying with
    the same key, or swapping models under it, can never succeed."""
    status = http_status_of(exc)
    if status is not None:
        return status in (401, 403)
    lowered = str(getattr(exc, "message", "") or exc).lower()
    return "unauthorized" in lowered or "forbidden" in lowered


def is_not_found_error(exc: BaseException) -> bool:
    """404 — the URL or model id does not exist.  Terminal for this id:
    retrying the same id is pointless, but a *different* model id may work.

    Also matches a dead model id reported with a non-404 status: Groq
    answers an unknown model with **400** + ``model_not_found`` in the
    body, and OpenRouter with "no endpoints found".  Status alone would
    miss those, so the model-not-found body text counts too.  Auth
    errors (401/403) never count — a bad key cannot be healed by
    swapping the model.
    """
    status = http_status_of(exc)
    if status == 404:
        return True
    if status in (401, 403):
        return False
    text = str(getattr(exc, "message", "") or exc).lower()
    if "model_not_found" in text or "model not found" in text:
        return True
    if "model" in text and ("does not exist" in text or "unknown model" in text):
        return True
    if "no endpoints found" in text:
        return True
    return False


def is_rate_limited_error(exc: BaseException) -> bool:
    """429 — back off (honoring ``retry_after``) and retry."""
    return isinstance(exc, RateLimited) or http_status_of(exc) == 429


def is_server_error(exc: BaseException) -> bool:
    """5xx / 408 / 425 — the server stumbled; back off and retry."""
    status = http_status_of(exc)
    if status is None:
        return isinstance(exc, (ProviderUnavailable, ProviderError)) and getattr(
            exc, "retryable", False
        )
    return status in (408, 425) or 500 <= status <= 599


def retry_after_of(exc: BaseException) -> float | None:
    """Best-effort ``Retry-After`` seconds for an error, or ``None``.

    Single shared helper so the retry loop, the rate limiter, and the error
    intelligence all honor the same signal instead of each inventing their
    own extraction. Prefers an explicit ``retry_after`` attribute, then a
    ``Retry-After`` response header, then ``details["retry_after"]``.
    """
    direct = getattr(exc, "retry_after", None)
    if isinstance(direct, (int, float)) and direct >= 0:
        return float(direct)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        try:
            raw = headers.get("Retry-After") or headers.get("retry-after")
        except Exception:  # noqa: BLE001 - exotic header mappings
            raw = None
        if raw is not None:
            try:
                return max(0.0, float(raw))
            except (TypeError, ValueError):
                pass
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        raw = details.get("retry_after")
        if isinstance(raw, (int, float)) and raw >= 0:
            return float(raw)
    return None


class NoMoralsError(Exception):
    """Base class for all framework errors."""

    code: str = "error"
    retryable: bool = False

    #: code → error class, populated automatically by __init_subclass__.
    _registry: dict[str, type["NoMoralsError"]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        code = getattr(cls, "code", "")
        if code and code != "error":
            NoMoralsError._registry[code] = cls

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message or self.__class__.__name__)
        self.message = message or self.__class__.__name__
        if code is not None:
            self.code = code
        if retryable is not None:
            self.retryable = retryable
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }

    # -- anyhow-style context chaining -------------------------------------
    def with_context(self, note: str, **fields: Any) -> "NoMoralsError":
        """Attach a breadcrumb as the error propagates (anyhow's ``.context()``).

        Each layer adds *what it was doing* — the chain reads top-down like
        a stack trace in prose. Returns ``self`` so it chains::

            raise classify(exc).with_context("syncing memory", user=user_id)
        """
        trail = self.details.setdefault("context", [])
        entry: dict[str, Any] = {"note": str(note)}
        if fields:
            entry["fields"] = dict(fields)
        trail.append(entry)
        return self

    def note(self, note: str, **fields: Any) -> "NoMoralsError":
        """Alias for :meth:`with_context`."""
        return self.with_context(note, **fields)

    @property
    def context_trail(self) -> list[dict[str, Any]]:
        """Breadcrumbs added by :meth:`with_context`, oldest first."""
        trail = self.details.get("context")
        return list(trail) if isinstance(trail, list) else []

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NoMoralsError":
        """Rebuild an error from :meth:`to_dict` output. Never raises.

        Resolves the concrete class by ``code`` (then by ``type`` name);
        falls back to a plain :class:`NoMoralsError` when the class has a
        custom constructor that the generic path cannot satisfy.
        """
        try:
            code = data.get("code", "") or ""
            type_name = data.get("type", "") or ""
            klass: type[NoMoralsError] = NoMoralsError._registry.get(code) or NoMoralsError
            if klass is NoMoralsError and type_name:
                candidate = globals().get(type_name)
                if isinstance(candidate, type) and issubclass(candidate, NoMoralsError):
                    klass = candidate
            message = data.get("message", "") or klass.__name__
            details = data.get("details") or {}
            retryable = data.get("retryable")
            # Forward structured extras (retry_after, field, role, ...) that the
            # concrete __init__ actually accepts — otherwise reconstruction
            # silently drops them (e.g. RateLimited.retry_after).
            fwd: dict[str, Any] = {}
            try:
                params = set(inspect.signature(klass.__init__).parameters)
            except (TypeError, ValueError):
                params = set()
            for attr in ("field", "capability", "actor", "role", "tool",
                         "reason", "kind", "retry_after", "dependency"):
                if attr in details and attr in params:
                    fwd[attr] = details[attr]
            try:
                err = klass(message, code=code or None,
                            details=dict(details),
                            retryable=bool(retryable) if retryable is not None else None,
                            **fwd)
            except TypeError:
                # custom __init__ (e.g. AmbiguousRef) — generic reconstruction
                err = NoMoralsError(message, code=code or "error",
                                    details=dict(details),
                                    retryable=bool(retryable) if retryable is not None else None)
            # restore structured extras that subclasses setdefault into details
            for attr in ("field", "capability", "actor", "role", "tool",
                         "reason", "kind", "retry_after", "breaker"):
                if attr in details and not hasattr(err, attr):
                    try:
                        setattr(err, attr, details[attr])
                    except Exception:  # noqa: BLE001 - best effort
                        pass
            return err
        except Exception:  # noqa: BLE001 - from_dict never raises
            return NoMoralsError(str(data)[:200], code="error.deser")

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


def resolve_error(code: str) -> type[NoMoralsError] | None:
    """Look up the error class registered for ``code`` (or ``None``)."""
    return NoMoralsError._registry.get(code)


def error_codes() -> dict[str, str]:
    """All registered codes → class names. Useful for docs and dashboards."""
    return {code: klass.__name__ for code, klass in sorted(NoMoralsError._registry.items())}


def cause_chain(exc: BaseException) -> list[BaseException]:
    """The ``__cause__``/``__context__`` chain, outermost first."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        nxt = current.__cause__ or current.__context__
        current = nxt if nxt is not current else None
    return chain


def format_chain(exc: BaseException, *, theme: Any = None) -> str:
    """Render the full error chain: breadcrumbs + causes, human-readable.

    Reads like anyhow's ``{:?}``/``{:#}`` output — each layer's context
    trail, then the causal chain underneath.
    """
    from .style import paint

    lines: list[str] = []
    for depth, err in enumerate(cause_chain(exc)):
        prefix = "  " * depth + ("↳ " if depth else "")
        code = getattr(err, "code", type(err).__name__)
        lines.append(f"{prefix}{paint(type(err).__name__, 'error', theme)} "
                     f"[{paint(str(code), 'muted', theme)}] {err}")
        trail = getattr(err, "context_trail", [])
        if isinstance(trail, list):
            for crumb in trail:
                note = crumb.get("note", "") if isinstance(crumb, dict) else str(crumb)
                fields = crumb.get("fields", {}) if isinstance(crumb, dict) else {}
                extra = (" " + " ".join(f"{k}={v}" for k, v in fields.items())
                         if fields else "")
                lines.append(f"{'  ' * (depth + 1)}· {paint(str(note), 'label', theme)}{extra}")
    return "\n".join(lines)


# ── Configuration ──────────────────────────────────────────────────────────────
class ConfigError(NoMoralsError):
    code = "config.invalid"


class ValidationError(NoMoralsError):
    code = "validation.failed"

    def __init__(self, message: str = "", *, field: str | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.field = field
        if field:
            self.details.setdefault("field", field)


# ── Storage ────────────────────────────────────────────────────────────────────
class StorageError(NoMoralsError):
    code = "storage.error"
    retryable = True


class MigrationError(StorageError):
    code = "storage.migration"
    retryable = False


class NotFound(NoMoralsError):
    code = "storage.not_found"
    retryable = False


class AmbiguousRef(NoMoralsError):
    """A short id/name reference matched more than one entity.

    Raised by id-prefix resolvers instead of guessing: the chat handler
    catches it and asks the user to disambiguate. ``candidates`` is a list
    of ``(id, human label)`` pairs; ``min_prefix_len`` is the smallest id
    prefix length that identifies every candidate uniquely.
    """

    code = "resolve.ambiguous"
    retryable = False

    def __init__(
        self,
        ref: str,
        candidates: list[tuple[str, str]],
        min_prefix_len: int,
        *,
        entity: str = "entity",
        hint: str = "",
    ) -> None:
        self.ref = ref
        self.candidates = list(candidates)
        self.min_prefix_len = min_prefix_len
        self.entity = entity
        self.hint = hint
        super().__init__(self._message())

    def _message(self) -> str:
        lines = [f"{self.ref!r} is ambiguous — matches "
                 f"{len(self.candidates)} {self.entity}s:"]
        for cid, label in self.candidates[:8]:
            lines.append(f"  · {cid} — {label}"[:120])
        if len(self.candidates) > 8:
            lines.append(f"  … +{len(self.candidates) - 8} more")
        lines.append(
            f"use a longer id prefix (at least {self.min_prefix_len} "
            f"characters) to pick one.")
        if self.hint:
            lines.append(self.hint)
        return "\n".join(lines)


class ConstraintViolation(StorageError):
    code = "storage.constraint"
    retryable = False


# ── Models / providers ─────────────────────────────────────────────────────────
class ModelError(NoMoralsError):
    code = "model.error"


class ProviderError(ModelError):
    code = "model.provider"
    retryable = True


class ProviderUnavailable(ProviderError):
    code = "model.provider.unavailable"
    retryable = True


class RateLimited(ProviderError):
    code = "model.provider.rate_limited"
    retryable = True

    def __init__(self, message: str = "rate limited", *, retry_after: float = 1.0, **kw: Any):
        super().__init__(message, **kw)
        self.retry_after = retry_after
        self.details.setdefault("retry_after", retry_after)


class ContextOverflow(ModelError):
    code = "model.context_overflow"
    retryable = False


class DownloadError(ModelError):
    code = "model.download"
    retryable = True


# ── Network / auth / quotas ────────────────────────────────────────────────────
class NetworkError(NoMoralsError):
    """Generic transport failure not tied to a model provider.

    DNS failures, refused connections, TLS errors on non-model traffic.
    Distinct from :class:`ProviderUnavailable`, which is model-provider scoped.
    """

    code = "network.error"
    retryable = True


class AuthError(NoMoralsError):
    """Credentials are bad, expired, or missing.

    Terminal for these credentials: retrying the same secret can never
    succeed — refresh it first, then retry.
    """

    code = "auth.failed"
    retryable = False


class QuotaExceeded(NoMoralsError):
    """A quota (not a rate window) is exhausted — monthly tokens, seats, storage.

    Unlike :class:`RateLimited`, waiting a few seconds will not help.
    ``retry_after`` names the reset time when it is known.
    """

    code = "quota.exceeded"
    retryable = True

    def __init__(self, message: str = "quota exceeded", *, retry_after: float | None = None,
                 **kw: Any) -> None:
        super().__init__(message, **kw)
        self.retry_after = retry_after
        if retry_after is not None:
            self.details.setdefault("retry_after", retry_after)


class DependencyError(NoMoralsError):
    """A downstream service this operation depends on failed."""

    code = "dependency.failed"
    retryable = True

    def __init__(self, message: str = "", *, dependency: str | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.dependency = dependency
        if dependency:
            self.details.setdefault("dependency", dependency)


# ── Tools / capabilities ───────────────────────────────────────────────────────
class ToolError(NoMoralsError):
    code = "tool.error"


class ToolNotFound(ToolError):
    code = "tool.not_found"
    retryable = False


class CapabilityDenied(NoMoralsError):
    """A tool call was blocked by the capability policy."""

    code = "policy.denied"
    retryable = False

    def __init__(
        self,
        message: str = "capability denied",
        *,
        capability: str | None = None,
        actor: str | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(message, **kw)
        self.capability = capability
        self.actor = actor
        if capability:
            self.details.setdefault("capability", capability)
        if actor:
            self.details.setdefault("actor", actor)


class ToolDenied(ToolError):
    """A tool call was blocked by a role's allowlist.

    Structured so role agents can adapt (the denial names the role, the
    tool, and the reason) and so the failure ledger can categorize it
    distinctly from capability-policy denials.
    """

    code = "tool.denied"
    retryable = False

    def __init__(
        self,
        message: str = "tool denied by role allowlist",
        *,
        role: str | None = None,
        tool: str | None = None,
        reason: str = "",
        **kw: Any,
    ) -> None:
        super().__init__(message, **kw)
        self.role = role
        self.tool = tool
        self.reason = reason
        if role:
            self.details.setdefault("role", role)
        if tool:
            self.details.setdefault("tool", tool)
        if reason:
            self.details.setdefault("reason", reason)


class SandboxError(ToolError):
    code = "tool.sandbox"


class TimeoutError_(ToolError):
    """Named with a trailing underscore to avoid shadowing the builtin."""

    code = "tool.timeout"
    retryable = True


# ── Agents / tasks ─────────────────────────────────────────────────────────────
class TaskCancelled(NoMoralsError):
    code = "task.cancelled"
    retryable = False


class TaskFailed(NoMoralsError):
    code = "task.failed"
    retryable = False


class BudgetExceeded(NoMoralsError):
    code = "budget.exceeded"
    retryable = False

    def __init__(self, message: str = "budget exceeded", *, kind: str = "", **kw: Any) -> None:
        super().__init__(message, **kw)
        self.kind = kind
        if kind:
            self.details.setdefault("kind", kind)


class DeadlineExceeded(TaskCancelled):
    code = "task.deadline"
    retryable = False


class StateError(NoMoralsError):
    """Invalid internal state — a state machine received an event it cannot
    handle in its current state, or an invariant was violated."""

    code = "state.invalid"
    retryable = False


# ── Media / parsing ────────────────────────────────────────────────────────────
class ParseError(NoMoralsError):
    code = "parse.error"
    retryable = False


class UnsupportedFormat(ParseError):
    code = "parse.unsupported"
    retryable = False


class MediaError(NoMoralsError):
    code = "media.error"
    retryable = True


def classify(exc: BaseException) -> NoMoralsError:
    """Wrap an arbitrary exception in the framework hierarchy.

    Idempotent: a :class:`NoMoralsError` passes through unchanged.
    """
    if isinstance(exc, NoMoralsError):
        return exc
    text = str(exc) or type(exc).__name__
    lowered = text.lower()
    if isinstance(exc, (TimeoutError,)):
        return TimeoutError_(text, retryable=True)
    if isinstance(exc, (ConnectionError,)):
        return ProviderUnavailable(text, retryable=True)
    if isinstance(exc, (PermissionError,)):
        return CapabilityDenied(text)
    if isinstance(exc, (FileNotFoundError,)):
        return NotFound(text)
    if isinstance(exc, json.JSONDecodeError):
        return ParseError(text)
    if isinstance(exc, (ValueError, TypeError)):
        return ValidationError(text)
    if "rate limit" in lowered or "429" in lowered:
        return RateLimited(text)
    if "timed out" in lowered or "timeout" in lowered:
        return TimeoutError_(text, retryable=True)
    return NoMoralsError(text, code=f"unhandled.{type(exc).__name__.lower()}")


# ── User-facing formatting (infra spec §4.3) ───────────────────────────────────
# One place, never empty, secrets redacted. Every formatter that turns an
# exception into a user-facing string goes through here.

_SECRET_PATTERNS = (
    (re.compile(r"Bearer\s+[A-Za-z0-9._\-~+/=]+", re.IGNORECASE), "Bearer ***"),
    (re.compile(r"(api[_-]?key\s*[:=]\s*)[^\s\"',;]+", re.IGNORECASE), r"\1***"),
    (re.compile(r"(Authorization\s*[:=]\s*)[^\s\"',;]+", re.IGNORECASE), r"\1***"),
    (re.compile(r"\b(sk-[A-Za-z0-9_-]{8,})\b"), "sk-***"),
    (re.compile(r"\b(ghp_[A-Za-z0-9]{8,})\b"), "ghp-***"),
    (re.compile(r"\b(gsk_[A-Za-z0-9_-]{8,})\b"), "gsk-***"),
    (re.compile(r"\b(xox[baprs]-[A-Za-z0-9-]{8,})\b"), "xox-***"),
)


def redact_secrets(text: str) -> str:
    """Redact credential-looking values from ``text`` (idempotent)."""
    out = text or ""
    for pattern, replacement in _SECRET_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def exc_str(exc: BaseException | None, *, context: str = "") -> str:
    """Exception -> never-empty message for user-facing reasons.

    Some exceptions carry an empty message (bare ``EOFError`` from
    ``wave.open`` on empty bytes surfaced as "TTS vocal failed ()").
    Fall back to the exception class name (+ context when given) so a
    failure reason is never ``()``, never ``""``, never whitespace.
    """
    if exc is None:
        return f"unknown error{(' (' + context + ')') if context else ''}"
    msg = str(exc).strip()
    if msg:
        return msg
    name = type(exc).__name__ or "unknown error"
    if context:
        return f"{name} ({context})"
    return name


def classify_user_error(exc: BaseException | Any) -> str:
    """Deterministic user-facing error class.

    auth / format / network / provider / config / bug. From the
    exception's status/code attributes and stable message markers
    (plus :func:`http_status_of` for framework errors) — never guessed.
    """
    status = http_status_of(exc) if isinstance(exc, BaseException) else None
    if status is None:
        raw = getattr(exc, "status", None) or getattr(exc, "status_code", None) or getattr(exc, "code", None)
        try:
            status = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            status = None
    text = (str(exc) or "").lower()
    if status in (401, 403) or "unauthorized" in text or "invalid api key" in text or "authentication" in text:
        return "auth"
    if status in (400, 422) or "unsupported_audio_format" in text or ("invalid" in text and ("format" in text or "extension" in text or "mime" in text)):
        return "format"
    if status == 404 or "not found" in text:
        return "config"
    if status == 429 or "rate limit" in text or "too many requests" in text:
        return "provider"
    if status is not None and status >= 500:
        return "provider"
    if any(marker in text for marker in ("connection", "timeout", "timed out", "network", "dns", "certificate", "ssl", "unreachable", "refused")):
        return "network"
    if any(marker in text for marker in ("not configured", "no backend", "not installed", "missing", "unmapped", "no stt", "no tts")):
        return "config"
    if status is not None:
        return "provider"
    return "bug"


_USER_ERROR_HINTS = {
    "auth": "check the key for that provider",
    "format": "the file may be misnamed or in an unsupported format",
    "network": "check connectivity and retry",
    "provider": "the provider failed — retry, or try another provider",
    "config": "check the setting / key mapping",
    "bug": "this looks like a Devon bug — the full cause is in the log",
}


def format_error(what: str, exc: BaseException | Any, *, hint: str | None = None) -> str:
    """Full user-facing error shape (spec §4.3).

    ``I couldn't {what} (<class>: <real cause>). <hint>`` — single line,
    secrets redacted, never empty, never generic.
    """
    if isinstance(exc, BaseException):
        cause = redact_secrets(exc_str(exc))
    else:
        cause = redact_secrets(str(exc or "").strip() or "unknown error")
    kind = classify_user_error(exc)
    suffix = hint or _USER_ERROR_HINTS.get(kind, "")
    out = f"I couldn't {what} ({kind} — {cause})"
    if suffix:
        out += f". {suffix}"
    return out


def short_stt_reason(error: Any) -> str:
    """Classify an STT failure into the short reason shown to the user.

    THE single implementation (infra spec §4.2, one path per job).
    ``nomorals.agents.partner.brain._short_stt_reason`` and
    ``nomorals.voice.pingpong._short_stt_reason`` are aliases to this —
    the two copies drifted before (one collapsed empty errors to a bare
    generic) and must never diverge again.

    "no STT backend" when nothing can transcribe; anything else keeps the
    REAL underlying error (e.g. "STT API 401: {...}" /
    "CERTIFICATE_VERIFY_FAILED") instead of collapsing to a bare
    "STT API error". Secrets are redacted; the detail is single-line
    and truncated.
    """
    raw = getattr(error, "message", None)
    text = str(raw if raw else (error or "")).strip()
    lowered = text.lower()
    if ("no stt backend" in lowered or "no backend" in lowered
            or "not installed" in lowered or "no stt" in lowered):
        return "no STT backend"
    if not text:
        # Never a bare generic: an empty-message exception still names
        # its class (spec §4.3 — never-empty reasons).
        return f"STT API error: {exc_str(error) if isinstance(error, BaseException) else type(error).__name__}"
    detail = " ".join(text.split())
    detail = redact_secrets(detail)
    if len(detail) > 300:
        detail = detail[:300].rstrip() + "…"
    if detail.lower().startswith("stt api"):
        return detail
    return f"STT API error: {detail}"
