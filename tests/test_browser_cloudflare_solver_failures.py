"""Tail-109 pins for nomorals/browser/cloudflare_solver.py — 3 exclusions,
0 converts.

Each test pins one census site's fallback behavior plus a control half
proving the except path was really taken. The module imports no
failures-ledger record function (grep-proven); its degradation paths
either log at debug (the cookie-store factory, not a census site) or
are pinned silent here by the verdict assertions — nothing on these
paths reports a success the code did not earn.

Sites:
* 136 E104 — _clearance_from(): the tab's cookie jar raises -> "" —
  byte-identical to the no-clearance-cookie answer. Capture is
  best-effort annotation on an already-decided solve; a stored-cookie
  note is simply not added.
* 170 E104 — _is_challenged(): tab.html() raises -> True. FAIL-CLOSED:
  an unreadable page counts as still challenged ("unreadable page: not
  a clear"), so the solver never reports a clearance it could not see.
* 249 E103 — solve_tab(): the post-injection reload raises -> pass,
  because the poll decides: _wait_out re-reads the page and its verdict
  (success or expired) is computed from the page, never from the
  reload. The success result is identical with and without the crash.
"""

from __future__ import annotations

import time

from nomorals.browser.cloudflare_solver import CloudflareSolver, _clearance_from


class _Tab:
    def __init__(self, *, cookies=None, cookies_exc=None, html="",
                 html_exc=None, reload_exc=None):
        self.url = "https://example.com/page"
        self._cookies = cookies or []
        self._cookies_exc = cookies_exc
        self._html = html
        self._html_exc = html_exc
        self._reload_exc = reload_exc
        self.added = None
        self.reload_calls = 0

    def cookies(self):
        if self._cookies_exc is not None:
            raise self._cookies_exc
        return self._cookies

    def html(self):
        if self._html_exc is not None:
            raise self._html_exc
        return self._html

    def add_cookies(self, items):
        self.added = items

    def reload(self):
        self.reload_calls += 1
        if self._reload_exc is not None:
            raise self._reload_exc


# ── 136: clearance capture is best-effort ────────────────────────────────────

def test_136_clearance_read_crash_returns_empty() -> None:
    # Control: a present cf_clearance cookie is captured verbatim.
    tab = _Tab(cookies=[{"name": "cf_clearance", "value": "V123"}])
    assert _clearance_from(tab) == "V123"
    # Control: no clearance cookie -> "".
    assert _clearance_from(_Tab(cookies=[{"name": "other", "value": "x"}])) == ""
    # Trigger: the jar read crashes -> the same "" as absence.
    tab = _Tab(cookies_exc=RuntimeError("jar boom"))
    assert _clearance_from(tab) == ""


# ── 170: unreadable page counts as challenged (fail-closed) ──────────────────

def test_170_unreadable_page_counts_as_challenged() -> None:
    solver = CloudflareSolver(wait_s=1.0, poll_s=0.05)
    # Trigger: html() raises -> challenged (never a fabricated clear).
    assert solver._is_challenged(_Tab(html_exc=RuntimeError("dom gone"))) is True
    # Controls: real challenge markup -> True; a clean page -> False.
    assert solver._is_challenged(_Tab(html="<html>just a moment…</html>")) is True
    assert solver._is_challenged(_Tab(html="<html>welcome to the shop</html>")) is False


# ── 249: reload crash after injection — the poll decides ─────────────────────

class _Vault:
    def get(self, key):
        return {"value": "CLEAR-9", "expires_at": time.time() + 3600}


def _solve(reload_exc):
    solver = CloudflareSolver(cookie_store=_Vault(), wait_s=1.0, poll_s=0.05)
    tab = _Tab(html="<html>checking your browser</html>", reload_exc=reload_exc)
    waits = iter([False, True])  # managed wait fails; post-injection poll clears
    solver._wait_out = lambda tab_, deadline: next(waits)
    return solver.solve_tab(tab), tab


def test_249_reload_crash_poll_decides() -> None:
    # Trigger: reload raises after the vault cookie was injected -> the
    # poll's verdict stands and the result reports success.
    result, tab = _solve(RuntimeError("reload boom"))
    assert result["ok"] is True
    assert tab.reload_calls == 1
    assert tab.added and tab.added[0]["value"] == "CLEAR-9"
    # Control: a clean reload yields the identical result, proving the
    # verdict came from the poll and not from the reload either way.
    clean, _ = _solve(None)
    assert clean["ok"] is True and clean["detail"] == result["detail"]
