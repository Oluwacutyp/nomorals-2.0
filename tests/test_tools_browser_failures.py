"""Tail-61 pins for nomorals/tools/browser.py —
1 convert / 6 exclusions (1386 converted by parent amendment).

8 tests for 7 sites (site 1386's convert carries a guard test
plus a benign-case test after the parent amendment). Each
test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED, asserting via the double's own call log. The file
contains no exception handlers, so it adds zero census sites.

browser.py calls no failures-ledger record function anywhere
(grep-proven), so NOT-recording is pinned with the tail-8
snapshot method against the agency ledger
(nomorals/agency/failures.py): recent_failures() before/after
every trigger (tail-34 sandbox_l2 / tail-44 tab_tools
precedent).

Verdict notes (full grounds in the tail-61 SURVEY.md):
- 160: parser choke -> debug log + partial DOM, the
  documented degrade ("malformed HTML should degrade, not
  raise"); the DOM describes exactly what was parsed.
- 398: charset parse failure -> "utf-8", the same default as
  a missing charset; decoding is errors="replace" throughout.
- 1386 CONVERTED (parent amendment overruling the flag):
  scroll's guarded evaluate IS the operation — on evaluate
  failure the result now carries ok False + the classified
  error instead of the success-shaped dict. The benign
  sub-case (evaluate ok, settle wait timed out) keeps the
  exact success dict: the try is split, not raised.
- 1479: close_tab failure -> pass; the session teardown the
  receipt describes completes unconditionally in the same
  breath (documented best-effort; subordinate cleanup, not
  the operation itself — distinct from 1386).
- 1766: robots check failure -> True, the written fail-open
  policy at the site ("robots should never hard-block").
- 1817: one corrupt persisted file -> skipped alone
  (documented); the count counts adopted cookies only.
- 1823: whole-seeding failure -> 0 ("Never raises" /
  best-effort contract). Pin includes the observed corner:
  a mid-loop failure returns 0 even with a cookie already
  stored — the count's only consumer is a debug log line.
"""

import json
from unittest import mock

from nomorals.agents import failures as ag_fail
from nomorals.tools import browser as B
from nomorals.tools import web as web_mod


def _snap():
    return list(ag_fail.recent_failures(200))


def test_160_parse_html_failure_yields_partial_dom():
    before = _snap()
    log = []

    def bad_feed(self, markup):
        log.append("feed")
        raise RuntimeError("parser boom")

    with mock.patch.object(B._DOMBuilder, "feed", bad_feed):
        root = B.parse_html("<p>hello</p>")
    assert log == ["feed"]  # the parse was actually attempted
    assert root.tag == "#root" and root.children == []
    good = B.parse_html("<p>hello</p>")
    assert good.inner_text() == "hello"  # control
    assert _snap() == before


class _Headers(dict):
    def __init__(self, value, log):
        super().__init__()
        self._value = value
        self._log = log

    def get(self, key, default=None):
        self._log.append(key)
        return self._value


class _Resp:
    def __init__(self, body, headers):
        self._body = body
        self.headers = headers
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        return self._body

    def geturl(self):
        return "https://x.test/"


class _Opener:
    def __init__(self, resp):
        self._resp = resp
        self.calls = 0

    def open(self, request, timeout=None):
        self.calls += 1
        return self._resp


def test_398_charset_failure_defaults_utf8():
    before = _snap()
    sess = B.BrowserSession(respect_robots=False, retries=0)
    hlog = []
    opener = _Opener(_Resp(b"caf\xe9", _Headers(None, hlog)))
    sess._opener = opener
    out = sess._fetch("https://x.test/")
    assert opener.calls == 1 and hlog == ["Content-Type"]
    assert out["text"] == "caf\ufffd"  # utf-8/replace fallback fired
    hlog2 = []
    sess._opener = _Opener(_Resp(
        b"caf\xe9", _Headers("text/html; charset=iso-8859-1", hlog2)))
    out2 = sess._fetch("https://x.test/")
    assert out2["text"] == "café"  # control: charset honored
    assert _snap() == before


class _Tab:
    def __init__(self, fail, wait_fail=False):
        self.url = "https://x.test/"
        self._fail = fail
        self._wait_fail = wait_fail
        self.eval_log = []
        self.wait_log = []

    def evaluate(self, js):
        self.eval_log.append(js)
        if self._fail:
            raise RuntimeError("evaluate boom")
        return None

    def wait_for_load_state(self, state, timeout=None):
        self.wait_log.append(state)
        if self._wait_fail:
            raise RuntimeError("wait boom")
        return None


def test_1386_scroll_evaluate_failure_reports_not_ok():
    # CONVERTED site (parent amendment): the guarded evaluate IS
    # the scroll, so its failure must not read as a success.
    # (a) evaluate raises -> ok False + the classified error, and
    # the call log proves the scroll was actually attempted.
    before = _snap()
    sess = B.RenderedBrowserSession(name="probe")
    bad = _Tab(fail=True)
    sess._tab = bad
    out = sess.scroll("down")
    assert len(bad.eval_log) == 1  # the scroll was actually attempted
    assert out == {"engine": "rendered", "url": "https://x.test/",
                   "scrolled": "down", "ok": False,
                   "error": "evaluate boom"}
    assert bad.wait_log == []  # no settle wait after a failed scroll
    assert _snap() == before


def test_1386_scroll_wait_timeout_keeps_success_dict():
    # (b) benign control: evaluate succeeds, the settle wait times
    # out -> the result is byte-identical to the plain success
    # dict (the page DID scroll; no ok/error keys appear).
    before = _snap()
    sess2 = B.RenderedBrowserSession(name="probe")
    slow = _Tab(fail=False, wait_fail=True)
    sess2._tab = slow
    benign = sess2.scroll("down")
    assert slow.eval_log and slow.wait_log == ["load"]
    # (c) plain success control, unchanged.
    sess3 = B.RenderedBrowserSession(name="probe")
    sess3._tab = _Tab(fail=False)
    control = sess3.scroll("down")
    assert benign == control == {"engine": "rendered",
                                 "url": "https://x.test/",
                                 "scrolled": "down"}
    assert _snap() == before


class _Svc:
    def __init__(self, fail):
        self._fail = fail
        self.log = []

    def close_tab(self, tab_id):
        self.log.append(tab_id)
        if self._fail:
            raise RuntimeError("close boom")


def test_1479_close_failure_still_tears_down_session():
    before = _snap()
    sess = B.RenderedBrowserSession(name="probe")
    svc = _Svc(fail=True)
    sess._svc = svc
    sess._tab = object()
    sess._tab_id = "tab-9"
    out = sess.close()
    assert svc.log == ["tab-9"]  # the service close was attempted
    assert out == {"ok": True, "closed": "probe", "engine": "rendered"}
    assert sess._tab is None and sess._tab_id == ""
    sess2 = B.RenderedBrowserSession(name="probe")
    svc2 = _Svc(fail=False)
    sess2._svc = svc2
    sess2._tab = object()
    sess2._tab_id = "tab-9"
    assert sess2.close() == out  # control: identical receipt
    assert _snap() == before


class _Robots:
    def __init__(self, mode):
        self._mode = mode
        self.log = []

    def allowed(self, url, user_agent):
        self.log.append(url)
        if self._mode == "raise":
            raise RuntimeError("robots boom")
        return self._mode


def test_1766_robots_failure_fails_open():
    before = _snap()
    bad = _Robots("raise")
    with mock.patch.object(web_mod, "_robots", bad):
        assert B._robots_allowed("https://x.test/a", "UA") is True
    assert bad.log == ["https://x.test/a"]  # the check was attempted
    deny = _Robots(False)
    with mock.patch.object(web_mod, "_robots", deny):
        assert B._robots_allowed("https://x.test/a", "UA") is False
    assert _snap() == before  # control: gate value passes through


_COOKIE = [{"name": "sid", "value": "1", "domain": ".example.com",
            "path": "/"}]


class _Jar:
    def __init__(self, fail_on=None):
        self.stored = []
        self.calls = 0
        self._fail_on = fail_on

    def set_cookie(self, cookie):
        self.calls += 1
        if self._fail_on is not None and self.calls == self._fail_on:
            raise RuntimeError("jar boom")
        self.stored.append(cookie)


class _Sess:
    def __init__(self, jar):
        self.cookie_jar = jar


def test_1817_seed_skips_one_bad_file(tmp_path):
    before = _snap()
    (tmp_path / "a_bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "b_good.json").write_text(json.dumps(_COOKIE),
                                          encoding="utf-8")
    jar = _Jar()
    count = B.seed_session_from_persisted(_Sess(jar), str(tmp_path))
    assert jar.calls == 1  # set_cookie attempted for the good file
    assert count == 1  # corrupt file skipped, iteration continued
    jar2 = _Jar()
    assert B.seed_session_from_persisted(
        _Sess(jar2), str(tmp_path)) == 1  # control: same count
    assert _snap() == before


def test_1823_seed_outer_failure_returns_zero(tmp_path):
    before = _snap()
    two = _COOKIE + [{"name": "two", "value": "2",
                      "domain": ".example.com", "path": "/"}]
    (tmp_path / "s.json").write_text(json.dumps(two), encoding="utf-8")
    jar = _Jar(fail_on=2)  # stores cookie 1, raises on cookie 2
    count = B.seed_session_from_persisted(_Sess(jar), str(tmp_path))
    assert jar.calls == 2  # both adoptions were actually attempted
    # OBSERVED corner: mid-loop failure -> 0 despite 1 stored cookie
    assert count == 0 and len(jar.stored) == 1
    jar2 = _Jar()
    assert B.seed_session_from_persisted(
        _Sess(jar2), str(tmp_path)) == 2
    assert len(jar2.stored) == 2  # control
    assert _snap() == before
