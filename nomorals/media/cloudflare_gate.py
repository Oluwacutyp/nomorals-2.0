"""Cloudflare gate for the media downloader: the kit's second home.

:mod:`nomorals.media.download_kit` built the Cloudflare state
machine and owner-cookie vault flow without a consumer; Pass #79's
browser solver (:mod:`nomorals.browser.cloudflare_solver`) is the
machine's first driver. The DOWNLOADER is the flow the kit was
designed around — its docstring: "a real browser run resolves the
challenge (or carries the owner's stored clearance)". Downloads
fetch with plain HTTP first, so when a body comes back as a
Cloudflare interstitial this gate decides what the downloader may
do about it:

* **stored clearance** — a live ``cf_clearance`` in the vault store
  is handed to the fetch layer as a cookie and the page is retried
  once;
* **expired** — the machine's terminal truth, reported verbatim;
* **ask** — the kit's owner-ask text, carried into the download
  report so the next rung (a browser run / the owner) can act on
  it. The machine asks exactly once per instance.

The store is the kit's duck-typed mapping (``get``/``set``); with
no store wired the gate only ever *detects* — it never pretends a
cookie exists.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from .download_kit import (
    CloudflareStateMachine,
    get_or_request_cookie,
    is_cloudflare_challenge,
)

__all__ = ["CloudflareGate"]


class CloudflareGate:
    """One download flow's Cloudflare decision point."""

    def __init__(self, store: Any = None) -> None:
        self._store = store
        #: the state machine driving this flow's decisions
        self.machine = CloudflareStateMachine()

    @property
    def state(self) -> str:
        return self.machine.state.value

    def inspect(self, body_text: str) -> bool:
        """True when a fetched body is a Cloudflare interstitial.

        Feeds the machine's ``challenge`` edge on first sight.
        """
        if not is_cloudflare_challenge(body_text or ""):
            return False
        self.machine.feed("challenge")
        return True

    def cookie_decision(self, url: str) -> dict[str, Any]:
        """The kit's cookie verdict for this URL's domain.

        Returns ``{"action", "cookie", "message"}`` — action is
        ``"inject"`` (a live clearance: retry with it),
        ``"expired"`` or ``"ask"`` (no cookie to inject; the
        message says why / what the owner is being asked).
        """
        if self._store is None:
            ask = self.machine.ask_owner()
            return {"action": "ask", "cookie": "",
                    "message": ask or "Cloudflare challenge page; no "
                                       "cookie vault wired for "
                                       "downloads"}
        domain = (urlparse(url or "").hostname or "").lower()
        status, payload = get_or_request_cookie(self._store, domain)
        if status == "have":
            self.machine.feed("cookie_provided")
            return {"action": "inject",
                    "cookie": f"cf_clearance={payload}",
                    "message": "stored cf_clearance applied"}
        if status == "expired":
            return {"action": "expired", "cookie": "",
                    "message": payload}
        ask = self.machine.ask_owner()
        return {"action": "ask", "cookie": "",
                "message": ask or payload}

    def resolve(self, cleared: bool) -> None:
        """Record the retry's outcome on the machine."""
        self.machine.feed("success" if cleared else "challenge")
