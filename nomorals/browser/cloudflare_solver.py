"""Cloudflare solver: the download kit's state machine, driven live.

The Cloudflare state machine in :mod:`nomorals.media.download_kit`
(``CloudflareStateMachine`` + the owner-cookie vault flow) was
built in the Pass 2 era and never had a single production consumer
— it models a flow nobody drove. This module is that driver, and
the machine's first consumer:

1. **Managed wait.** A Cloudflare managed challenge usually
   completes by itself in a real browser — the machine is fed
   ``challenge`` and the tab is polled until the interstitial is
   gone (``success`` → ``PASSED``).
2. **Vault cookie.** If the wait exhausts, the kit's own cookie
   flow runs: a live vault ``cf_clearance`` for the domain is
   injected into the session and the page reloaded
   (``cookie_provided`` → ``PASSED``, or a fresh challenge after a
   cookie → ``COOKIE_EXPIRED``, terminal by the machine's own law —
   never a second ask, never a retry loop). With no live cookie,
   ``ask_owner()`` produces the kit's ask text exactly once and
   the solver returns it as its failure detail — the routing
   layer's takeover rung is where the owner acts on it.
3. **Clearance capture.** On a clear, the session's fresh
   ``cf_clearance`` is stored back through the kit's vault flow
   (``store_owner_cookie``), so the downloader-side consumer and
   the next visit start from the stored clearance. Solved-session
   persistence itself belongs to the ladder
   (:class:`nomorals.browser.challenge.ChallengeSolver` saves the
   tab's cookies on any verified clear).

The vault store is the kit's duck-typed mapping (``get``/``set``) —
injected via the constructor or the challenge metadata; no
production vault adapter existed, so an unwired store simply means
the cookie rung reports itself unwired, honestly.

Kind strings, and one deliberate non-registration: this solver
registers under ``cloudflare`` (the kind name flows use) and as
the ``local_cloudflare`` backend (which also accepts provider
metadata). It does NOT register under the browser ladder's signal
name ``cloudflare-challenge``: the ladder already owns that wall's
UX — its wait rung is the managed wait, its takeover manager is
the owner ask (watched live, auto-detected), and it saves the
solved session on any clear. A second, long-polling solver inside
the ladder's local rung would preempt the owner's takeover and
fight the manager's own polling (two pre-existing suites pin
exactly that behavior). The ladder's wall is served by the ladder;
the state machine's consumers are this module's flow/backend
registrations and the downloader gate.
"""
from __future__ import annotations

import time
from typing import Any, Callable
from urllib.parse import urlparse

from ..core.logging_setup import get_logger
from ..media.download_kit import (
    CloudflareStateMachine,
    get_or_request_cookie,
    is_cloudflare_challenge,
    store_owner_cookie,
)
from ..tools.captcha import CaptchaBackend, SolveResult, register_backend
from .kind_solvers import register_kind_solver

_log = get_logger(__name__)

__all__ = [
    "CLOUDFLARE_KIND",
    "CloudflareSolver",
    "solve_cloudflare_kind",
    "LocalCloudflareBackend",
    "register_local_backend",
]

#: The estate's canonical kind name for Cloudflare interstitials.
CLOUDFLARE_KIND = "cloudflare"

#: The browser ladder's detection string for the same wall.
CLOUDFLARE_LADDER_KIND = "cloudflare-challenge"


def _domain_of(url: str) -> str:
    host = urlparse(url or "").hostname
    return (host or "").lower()


def _clearance_from(tab: Any) -> str:
    """The session's cf_clearance value, "" when absent/unreadable."""
    cookies_fn = getattr(tab, "cookies", None)
    if not callable(cookies_fn):
        return ""
    try:
        for cookie in cookies_fn() or []:
            if isinstance(cookie, dict) and \
                    cookie.get("name") == "cf_clearance":
                return str(cookie.get("value") or "")
    except Exception:  # noqa: BLE001 - capture is best-effort
        return ""
    return ""


class CloudflareSolver:
    """Drive the kit's state machine against a live tab."""

    def __init__(self, *, cookie_store: Any = None,
                 wait_s: float = 20.0, poll_s: float = 1.0,
                 sleeper: Callable[[float], None] | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        self._store = cookie_store
        self._wait_s = max(0.0, float(wait_s))
        self._poll_s = max(0.05, float(poll_s))
        self._sleep = sleeper or time.sleep
        self._clock = clock or time.time
        #: the machine of the most recent solve (introspection/tests)
        self.machine: CloudflareStateMachine | None = None

    # -- helpers -------------------------------------------------------------
    def _store_for(self, challenge: dict | None) -> Any:
        if self._store is not None:
            return self._store
        if challenge:
            metadata = challenge.get("metadata") or {}
            return metadata.get("cookie_store")
        return None

    def _is_challenged(self, tab: Any) -> bool:
        try:
            return is_cloudflare_challenge(str(tab.html() or ""))
        except Exception:  # noqa: BLE001 - unreadable page: not a clear
            return True

    def _wait_out(self, tab: Any, deadline: float) -> bool:
        """Poll until the interstitial is gone. True = cleared."""
        while True:
            if not self._is_challenged(tab):
                return True
            if self._clock() >= deadline:
                return False
            self._sleep(self._poll_s)

    def _capture_clearance(self, tab: Any, store: Any,
                           domain: str) -> str:
        value = _clearance_from(tab)
        if value and store is not None and domain:
            if store_owner_cookie(store, domain, value):
                return "fresh cf_clearance stored in the vault"
        return ""

    # -- main entry ------------------------------------------------------------
    def solve_tab(self, tab: Any, challenge: dict | None = None
                  ) -> dict[str, Any]:
        machine = CloudflareStateMachine()
        self.machine = machine
        url = str(getattr(tab, "url", "") or
                  (challenge or {}).get("page_url") or "")
        domain = _domain_of(url)
        store = self._store_for(challenge)

        def result(ok: bool, detail: str) -> dict[str, Any]:
            return {"ok": ok, "backend": "cloudflare_local",
                    "detail": detail,
                    "state": machine.state.value,
                    "transitions": list(machine.transitions)}

        if not self._is_challenged(tab):
            return result(True, "no Cloudflare challenge on this page")
        machine.feed("challenge")

        # 1 — managed wait: the challenge completes itself, or not.
        deadline = self._clock() + self._wait_s
        if self._wait_out(tab, deadline):
            machine.feed("success")
            note = self._capture_clearance(tab, store, domain)
            return result(True, "Cloudflare managed challenge "
                                "completed in-session"
                          + (f"; {note}" if note else ""))

        # 2 — the kit's cookie flow.
        if store is None:
            ask = machine.ask_owner()
            return result(False, (ask + " No cookie vault is wired "
                                       "for this flow — owner "
                                       "takeover is next.")
                          if ask else
                          "Cloudflare challenge did not complete "
                          "in-session, and no cookie vault is wired "
                          "for this flow — owner takeover is next")
        status, payload = get_or_request_cookie(store, domain)
        if status == "have":
            machine.feed("cookie_provided")
            add = getattr(tab, "add_cookies", None)
            if not callable(add):
                machine.feed("failed")
                return result(False, "a vault cf_clearance exists "
                                     "but the tab cannot take "
                                     "cookies")
            try:
                add([{"name": "cf_clearance", "value": payload,
                      "domain": f".{domain}" if domain else "",
                      "path": "/"}])
            except Exception as exc:  # noqa: BLE001
                machine.feed("failed")
                return result(False, f"cookie injection failed: {exc}")
            reload_fn = getattr(tab, "reload", None)
            if callable(reload_fn):
                try:
                    reload_fn()
                except Exception:  # noqa: BLE001 - poll decides
                    pass
            deadline = self._clock() + min(10.0, self._wait_s)
            if self._wait_out(tab, deadline):
                machine.feed("success")
                note = self._capture_clearance(tab, store, domain)
                return result(True, "Cloudflare cleared with the "
                                    "vault cf_clearance cookie"
                              + (f"; {note}" if note else ""))
            machine.feed("challenge")  # → COOKIE_EXPIRED (terminal)
            return result(False, "the stored cf_clearance did not "
                                 "clear the challenge — the machine "
                                 "marks it expired; a fresh owner "
                                 "cookie is needed")
        if status == "expired":
            return result(False, payload)
        # status == "ask": the kit asks exactly once; the message IS
        # the detail the owner sees on the escalation path.
        ask = machine.ask_owner()
        return result(False, ask or payload)


def solve_cloudflare_kind(tab: Any, challenge: dict) -> dict[str, Any]:
    """Kind-router entry point for ``cloudflare`` kinds."""
    if tab is None:
        return {"ok": False, "backend": "cloudflare_local",
                "detail": "Cloudflare solving needs a live tab"}
    return CloudflareSolver().solve_tab(tab, challenge)


# ── captcha backend registration (tools/captcha.py untouched) ────────────

class LocalCloudflareBackend(CaptchaBackend):
    """The state-machine driver as a captcha backend.

    Applicable to challenges captcha.py names ``unknown`` with
    ``metadata["provider"] == "cloudflare"`` (its interstitial
    shape), or an explicit ``cloudflare`` kind. Needs the live tab
    in ``metadata["session"]``.
    """

    name = "local_cloudflare"

    def __init__(self, settings: Any = None, *,
                 solver: CloudflareSolver | None = None,
                 **_: Any) -> None:
        self._settings = settings
        self._solver = solver

    def available(self) -> bool:
        return True  # the managed wait needs nothing but a live tab

    def solve(self, challenge: Any) -> SolveResult:
        started = time.time()
        kind = str(getattr(challenge, "kind", "") or "")
        metadata = getattr(challenge, "metadata", None) or {}
        provider = str(metadata.get("provider") or "")
        if kind not in (CLOUDFLARE_KIND, CLOUDFLARE_LADDER_KIND) \
                and provider != "cloudflare":
            return SolveResult(
                ok=False, kind=kind, backend=self.name,
                detail=f"local Cloudflare backend solves Cloudflare "
                       f"interstitials, not {kind!r}")
        session = metadata.get("session") or metadata.get("tab")
        if session is None:
            return SolveResult(
                ok=False, kind=kind, backend=self.name,
                detail="Cloudflare solving needs the live browser "
                       "session in challenge.metadata['session']")
        solver = self._solver or CloudflareSolver()
        out = solver.solve_tab(session, {"kind": kind,
                                         "metadata": metadata})
        return SolveResult(
            ok=bool(out.get("ok")), kind=kind, backend=self.name,
            elapsed_ms=int((time.time() - started) * 1000),
            detail=str(out.get("detail") or ""))


_REGISTERED = False


def register_local_backend() -> bool:
    """Register ``local_cloudflare`` in the captcha backend registry."""
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        register_backend("local_cloudflare", LocalCloudflareBackend)
        _REGISTERED = True
        return True
    except Exception as exc:  # noqa: BLE001 - registration is additive
        _log.warning("local Cloudflare backend registration failed: "
                     "%s", exc)
        return False


register_local_backend()
register_kind_solver(CLOUDFLARE_KIND, solve_cloudflare_kind)
# NOT CLOUDFLARE_LADDER_KIND — see the module docstring: the ladder
# owns the cloudflare-challenge wall (wait → takeover → session
# save), and a registry entry here would preempt it.
