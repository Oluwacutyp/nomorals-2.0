"""LinkedIn adapter — :class:`SocialManager` over the native connector.

The connector (``nomorals/connectors/linkedin.py``, L5) already owns the
real LinkedIn surface: OAuth token in the vault, ``POST /rest/posts``
with the stored ``person_urn``, plain-word API errors. This adapter (L4)
may not import it — the layering law keeps ``social`` below
``connectors`` — so the composition root injects a *connector builder*
(:func:`set_connector_builder`, wired in ``connectors/wiring.py`` from
``agents/context.build_context``, the proxylab/vision factory pattern).
The builder opens the vault and returns a fresh connector per call, so
the adapter never touches a credential itself.

Authorization (#43): the adapter calls ``connector.post(...,
confirmed=True)``. That flag is the connector's attestation seam — "the
caller attests the owner already approved this exact payload" — and
every production path into :meth:`SocialManager.publish` is
owner-authorized upstream (an approved draft's single-use confirmation
token, a review approval, or a schedule the owner set). The adapter
adds no new path to the connector and weakens no gate; it changes
WHERE an authorized post can go, never WHO authorizes it.

Metrics: the connector exposes NO metrics operation — there is no
engagement/socialActions call on it and its OAuth scopes carry no
``r_member_social``. This adapter therefore deliberately does NOT
override :meth:`PlatformAdapter.metrics`: the WS3 P1 measurement spine
checks capability structurally, so LinkedIn rows are named
``unavailable`` there and never fed fabricated numbers.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from ..base import (
    ERROR_AUTH,
    ERROR_UNKNOWN,
    Account,
    PlatformAdapter,
    PostResult,
    classify_http_error,
)

__all__ = [
    "LinkedInAdapter",
    "set_connector_builder",
    "wired_connector_builder",
]

#: Process-wide connector builder, installed by the composition root
#: (``connectors/wiring.wire_linkedin_social_adapter``). Zero-arg,
#: returns a connected-or-not ``LinkedInConnector``; may raise when the
#: vault cannot open — the adapter turns that into words, not a crash.
_BUILDER: Callable[[], Any] | None = None


def set_connector_builder(builder: Callable[[], Any] | None) -> None:
    """Install (or clear, with ``None``) the connector builder."""
    global _BUILDER
    _BUILDER = builder


def wired_connector_builder() -> Callable[[], Any] | None:
    return _BUILDER


class LinkedInAdapter(PlatformAdapter):
    """Posts text to LinkedIn through the native connector."""

    name = "linkedin"
    max_chars = 3000  # the connector's own POST /rest/posts limit
    supports_media = False  # the connector's post() is text-only
    supports_thread = False

    def __init__(
        self,
        connector_builder: Callable[[], Any] | None = None,
    ) -> None:
        # An explicit builder wins (tests, embedding); otherwise the
        # composition-root builder installed at startup is used.
        self._builder = connector_builder

    def _connector(self) -> Any | None:
        builder = self._builder or _BUILDER
        if builder is None:
            return None
        return builder()

    def post(self, account: Account, content: str, **kwargs: Any) -> PostResult:
        started = time.perf_counter()
        try:
            connector = self._connector()
        except Exception as exc:  # noqa: BLE001 - vault/registry failure → words
            return PostResult(
                platform=self.name, ok=False,
                error=f"linkedin connector could not be opened: {exc}",
                error_code=ERROR_AUTH,
                seconds=time.perf_counter() - started)
        if connector is None:
            return PostResult(
                platform=self.name, ok=False,
                error=("linkedin is not wired in this process — the "
                       "composition root installs its connector builder "
                       "at startup, and draft posting runs where a "
                       "context was built (the partner runtime)"),
                error_code=ERROR_AUTH,
                seconds=time.perf_counter() - started)
        try:
            data = connector.post((content or "").strip(), confirmed=True)
        except Exception as exc:  # noqa: BLE001 - connector errors are worded
            status = 0
            try:
                status = int(getattr(exc, "status_code", 0) or 0)
            except (TypeError, ValueError):
                status = 0
            message = str(exc) or "linkedin post failed"
            code = (classify_http_error(status, message) if status
                    else ERROR_AUTH if "not connected" in message.lower()
                    else ERROR_UNKNOWN)
            return PostResult(
                platform=self.name, ok=False, error=message,
                error_code=code, status_code=status,
                seconds=time.perf_counter() - started)
        external_id = ""
        if isinstance(data, dict):
            external_id = str(data.get("id") or "")
        url = ""
        if external_id.startswith(("urn:li:share:", "urn:li:activity:")):
            url = f"https://www.linkedin.com/feed/update/{external_id}/"
        return PostResult(
            platform=self.name, ok=True, external_id=external_id, url=url,
            seconds=time.perf_counter() - started)

    def health(self, account: Account) -> bool:
        try:
            connector = self._connector()
        except Exception:  # noqa: BLE001 - health is a probe, never a crash
            return False
        if connector is None:
            return False
        try:
            return bool(connector.test_connection())
        except Exception:  # noqa: BLE001
            return False
