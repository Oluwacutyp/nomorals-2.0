"""Pass #155 pins: the browser failure ledger (NEW) + service conversions.

No browser ledger existed before this pass: ``nomorals/browser/failures.py``
(namespace ``browser``) is created here, durability-wired through the shared
``nomorals/core/failure_store.py`` exactly like the #141–#154 ledgers. The
17 durability tests below mirror #154's shape exactly; the 25 site pins
prove each conversion records without changing flow (fallback value
unchanged AND a ledger entry under its site name); the 2 exclusion pins
prove the element_image_bytes ladder rungs record nothing (a succeeding
fallback yields identical bytes; total failure raises, never swallows).
Every test redirects stores with the app's own ``NM_HOME`` override (never
the real home, never the repo tree). Restart tests use real subprocesses.
Every trigger was probed live before its pin was written.
"""
import base64
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.browser import failures as bf
from nomorals.browser import service as S


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    bf.clear_failures()
    yield
    bf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in bf.recent_failures()]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


def _tab(tmp_path):
    tab = S.RenderedTab("t1", "s", tmp_path / "state.json", object())
    tab.url = "http://x/"
    return tab


# ---- Store-level (durability mirror of #154) -----------------------------

def test_store_round_trip():
    store = fs.Store("browser")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("browser").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("browser")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("browser").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("browser")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("browser").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("browser").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("browser").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("browser")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("browser").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("browser").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("browser").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("browser").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("browser")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("browser").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        bf.record_failure("no-home", ValueError("boom"))
        assert bf.failure_count() == 1
        assert [f.site for f in bf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert bf.failure_count() == 0
    assert bf.recent_failures() == []
    assert bf.format_failures() == "no browser failures recorded."
    bf.record_failure("browser.site-a", ValueError("boom"))
    bf.record_failure("browser.site-b", RuntimeError("bang"))
    assert bf.failure_count() == 2
    assert [f.site for f in bf.recent_failures()] == ["browser.site-b", "browser.site-a"]
    assert [f.site for f in bf.recent_failures(1)] == ["browser.site-b"]
    text = bf.format_failures()
    assert "2 total" in text and "browser.site-a" in text and "browser.site-b" in text
    bf.clear_failures()
    assert bf.failure_count() == 0
    assert bf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    bf.record_failure("browser.disk", ValueError("boom"))
    entries, total = fs.Store("browser").load()
    assert total == 1 and entries[0]["site"] == "browser.disk"
    bf.clear_failures()
    assert fs.Store("browser").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    bf.record_failure("browser.mem", ValueError("boom"))
    assert bf.failure_count() == 1
    assert [f.site for f in bf.recent_failures()] == ["browser.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                bf.record_failure(f"browser.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert bf.failure_count() == 400
    assert len(bf.recent_failures(1000)) == 200
    entries, total = bf._store.load()
    assert len(entries) == 200
    assert total == 400


# ---- Restart (subprocess) ------------------------------------------------

def _run_subprocess(home: Path, code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["NM_HOME"] = str(home)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), env=env,
        capture_output=True, text=True, timeout=120)


def _record_code(n: int, clear_first: bool = True) -> str:
    clear = "f.clear_failures()\n" if clear_first else ""
    return (
        "from nomorals.browser import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.browser import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.browser import failures as f\n"
               "f.clear_failures()\nprint('ok')\n")


def test_restart_record_survives(tmp_path):
    home = tmp_path / "home"
    proc = _run_subprocess(home, _record_code(3))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 3
    assert data["sites"] == ["site2", "site1", "site0"]


def test_restart_clear_is_gone(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    assert _run_subprocess(home, _CLEAR_CODE).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []


def test_restart_total_preserved_and_cap_holds(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(250)).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 250
    assert len(data["sites"]) == 200
    assert data["sites"][0] == "site249"


def test_restart_corrupt_store_recovers(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    store_path = fs.Store("browser").path()
    store_path = home / "failures" / store_path.name
    assert store_path.exists()
    store_path.write_text("{not json", encoding="utf-8")
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []
    proc = _run_subprocess(home, _record_code(1, clear_first=False))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 1


# ---- Converted-site behavior: fallback unchanged + recorded -------------

def test_converted_mask_proxy_records_and_returns_input_unmasked():
    bad = "http://user:pass@host:99999/"
    assert S._mask_proxy(bad) == bad
    assert "service._mask_proxy" in _sites()
    # Happy path untouched: a good proxy still masks, records nothing new.
    assert S._mask_proxy("http://user:pass@host:8080/p") == "http://user:***@host:8080/p"
    assert _sites() == ["service._mask_proxy"]


def test_converted_console_msg_text_records_and_empties(tmp_path):
    tab = _tab(tmp_path)

    class Msg:
        type = "log"
        location = {}

        @property
        def text(self):
            raise RuntimeError("text boom")

    tab._on_console_msg(Msg())
    assert tab._console_log[-1]["text"] == ""
    assert "service._on_console_msg.text" in _sites()


def test_converted_console_msg_kind_records_and_defaults(tmp_path):
    tab = _tab(tmp_path)

    class Msg:
        text = "hello"
        location = {}

        @property
        def type(self):
            raise RuntimeError("type boom")

    tab._on_console_msg(Msg())
    assert tab._console_log[-1]["type"] == "log"
    assert "service._on_console_msg.kind" in _sites()


def test_converted_console_msg_loc_records_and_empties(tmp_path):
    tab = _tab(tmp_path)

    class Msg:
        text = "hello"
        type = "log"

        @property
        def location(self):
            raise RuntimeError("loc boom")

    tab._on_console_msg(Msg())
    assert tab._console_log[-1]["location"] == {
        "url": None, "lineNumber": None, "columnNumber": None}
    assert "service._on_console_msg.loc" in _sites()


def test_converted_on_request_records_and_drops_entry(tmp_path):
    tab = _tab(tmp_path)

    class Req:
        @property
        def url(self):
            raise RuntimeError("url boom")

        method = "GET"

    tab._on_request(Req())
    assert tab._network_log == []
    assert "service._on_request" in _sites()


def test_converted_on_response_records_and_drops_entry(tmp_path):
    tab = _tab(tmp_path)

    class Resp:
        @property
        def url(self):
            raise RuntimeError("url boom")

        status = 200

    tab._on_response(Resp())
    assert tab._network_log == []
    assert "service._on_response" in _sites()


def test_converted_on_response_method_records_and_empties(tmp_path):
    tab = _tab(tmp_path)

    class Resp:
        url = "http://x/"
        status = 200

        @property
        def request(self):
            raise RuntimeError("request boom")

    tab._on_response(Resp())
    assert tab._network_log[-1]["method"] == ""
    assert "service._on_response.method" in _sites()


def test_converted_on_dialog_records_and_empties_fields(tmp_path):
    tab = _tab(tmp_path)

    class Dialog:
        message = "m"

        @property
        def type(self):
            raise RuntimeError("type boom")

        def accept(self):
            pass

    tab._on_dialog(Dialog())
    assert tab._dialog_log[-1]["type"] == ""
    assert tab._dialog_log[-1]["message"] == ""
    assert "service._on_dialog" in _sites()


def test_converted_teardown_quiet_records_and_clears_attrs(tmp_path):
    tab = _tab(tmp_path)
    tab._page = SimpleNamespace(close=_boom)
    tab._context = None
    tab._browser = None
    tab._teardown_quiet()
    assert tab._page is None and tab._context is None and tab._browser is None
    assert "service._teardown_quiet.close" in _sites()


def test_converted_navigate_title_records_and_empties(tmp_path, monkeypatch):
    tab = _tab(tmp_path)

    class Page:
        url = "http://x/"

        def title(self):
            raise RuntimeError("title boom")

    page = Page()
    monkeypatch.setattr(tab, "_ensure_page", lambda: page)
    monkeypatch.setattr(tab, "_goto_once", lambda *a, **k: None)
    out = tab.navigate("http://x/")
    assert out["title"] == ""
    assert tab.history[-1]["title"] == ""
    assert "service.navigate.title" in _sites()


def test_converted_content_sample_records_and_returns_empty(tmp_path):
    tab = _tab(tmp_path)
    tab._page = SimpleNamespace(content=_boom)
    assert tab._content_sample() == ""
    assert "service._content_sample" in _sites()


def test_converted_goto_once_response_records_and_skips_400(tmp_path):
    tab = _tab(tmp_path)

    class Resp:
        @property
        def status(self):
            raise RuntimeError("status boom")

        headers = {}

    class Page:
        def goto(self, *a, **k):
            return Resp()

        def title(self):
            return ""

    page = Page()
    tab._page = page
    assert tab._goto_once(page, "http://x/", "domcontentloaded") is None
    assert "service._goto_once.response" in _sites()


def test_converted_page_title_records_and_returns_empty(tmp_path):
    tab = _tab(tmp_path)
    tab._page = SimpleNamespace(title=_boom)
    assert tab._page_title() == ""
    assert "service._page_title" in _sites()


def test_converted_set_date_enter_records_and_returns_ok(tmp_path, monkeypatch):
    tab = _tab(tmp_path)

    class Page:
        def fill(self, *a, **k):
            pass

        def press(self, *a, **k):
            raise RuntimeError("press boom")

    tab._page = Page()
    monkeypatch.setattr(tab, "_resolve_field", lambda name: ("#d", None))
    monkeypatch.setattr(tab, "_kind_of",
                        lambda name, info: {"tag": "input", "type": "text"})
    out = tab.set_date("dob", "2026-10-04")
    assert out == {"ok": True, "field": "dob", "date": "2026-10-04",
                   "tab_id": "t1"}
    assert "service.set_date.enter" in _sites()


def test_converted_click_url_records_and_keeps_before(tmp_path):
    tab = _tab(tmp_path)
    tab.url = "http://before/"

    class Page:
        def __init__(self):
            self._n = 0

        @property
        def url(self):
            self._n += 1
            if self._n == 1:
                return "http://before/"
            raise RuntimeError("url boom")

        def click(self, *a, **k):
            pass

    tab._page = Page()
    out = tab.click("#btn")
    assert out["url"] == "http://before/"
    assert out["navigated"] is False
    assert "service.click.url" in _sites()


def test_converted_click_title_records_and_keeps_stale(tmp_path):
    tab = _tab(tmp_path)
    tab.url = "http://before/"
    tab.title = "Old"

    class Page:
        def __init__(self):
            self._n = 0

        @property
        def url(self):
            self._n += 1
            return "http://before/" if self._n == 1 else "http://after/"

        def title(self):
            raise RuntimeError("title boom")

        def click(self, *a, **k):
            pass

    tab._page = Page()
    out = tab.click("#btn")
    assert out["url"] == "http://after/"
    assert tab.history[-1]["title"] == "Old"
    assert "service.click.title" in _sites()


def test_converted_submit_refresh_records_and_keeps_stale(tmp_path):
    tab = _tab(tmp_path)
    tab.url = "http://old/"
    tab.title = "OldT"

    class Page:
        def eval_on_selector(self, *a, **k):
            pass

        @property
        def url(self):
            raise RuntimeError("url boom")

        def title(self):
            return "NewT"

    tab._page = Page()
    out = tab.submit()
    assert out["url"] == "http://old/"
    assert tab.history[-1]["url"] == "http://old/"
    assert "service.submit.refresh" in _sites()


def test_converted_wait_for_url_refresh_records_and_keeps_stale(tmp_path):
    tab = _tab(tmp_path)
    tab.url = "http://old/"

    class Page:
        def wait_for_url(self, *a, **k):
            pass

        @property
        def url(self):
            raise RuntimeError("url boom")

        def title(self):
            return "NewT"

    tab._page = Page()
    out = tab.wait_for_url("x")
    assert out["url"] == "http://old/"
    assert "service.wait_for_url.refresh" in _sites()


def test_converted_type_text_verify_records_and_reports_unverified(tmp_path):
    tab = _tab(tmp_path)

    class Keyboard:
        def type(self, *a, **k):
            pass

    class Locator:
        def click(self, *a, **k):
            pass

    class Page:
        keyboard = Keyboard()

        def locator(self, selector):
            return Locator()

        def evaluate(self, *a, **k):
            raise RuntimeError("eval boom")

    tab._page = Page()
    out = tab.type_text("#f", "hi", delay_ms=0)
    assert out["verified"] is False
    assert "service.type_text.verify" in _sites()


def test_converted_mouse_drag_path_release_records_and_keeps_cause(tmp_path):
    tab = _tab(tmp_path)

    class Mouse:
        def move(self, *a, **k):
            raise RuntimeError("move boom")

        def down(self):
            pass

        def up(self):
            raise RuntimeError("up boom")

    class Page:
        def __init__(self):
            self.mouse = Mouse()
            self.url = "http://x/"

    tab._page = Page()
    with pytest.raises(S.BrowserError) as excinfo:
        tab.mouse_drag_path([(0, 0, 0), (1, 1, 0)])
    assert "move boom" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "move boom"
    assert "service.mouse_drag_path.release" in _sites()


def test_converted_frames_html_records_and_skips_bad_frame(tmp_path):
    tab = _tab(tmp_path)
    good = SimpleNamespace(url="http://a/", content=lambda: "<html>")
    bad = SimpleNamespace(url="http://b/", content=_boom)
    tab._page = SimpleNamespace(frames=[good, bad])
    assert tab.frames_html() == [{"url": "http://a/", "html": "<html>"}]
    assert "service.frames_html" in _sites()


def test_converted_after_navigation_url_records_and_reports_stale(tmp_path):
    tab = _tab(tmp_path)
    tab.url = "http://before/"

    class Page:
        @property
        def url(self):
            raise RuntimeError("url boom")

    out = tab._after_navigation(Page(), "http://before/", "reloaded")
    assert out["navigated"] is False
    assert tab.history == []
    assert "service._after_navigation.url" in _sites()


def test_converted_after_navigation_title_records_and_keeps_stale(tmp_path):
    tab = _tab(tmp_path)
    tab.url = "http://before/"
    tab.title = "OldT"

    class Page:
        url = "http://after/"

        def title(self):
            raise RuntimeError("title boom")

    out = tab._after_navigation(Page(), "http://before/", "reloaded")
    assert out["navigated"] is True
    assert tab.history[-1]["title"] == "OldT"
    assert "service._after_navigation.title" in _sites()


def test_converted_cookies_records_and_returns_empty(tmp_path):
    adapter = S.RenderedTabTakeoverAdapter(SimpleNamespace(cookies=_boom))
    assert adapter.cookies() == []
    assert "service.cookies" in _sites()


def test_converted_storage_state_records_and_returns_empty(tmp_path):
    adapter = S.RenderedTabTakeoverAdapter(SimpleNamespace(storage_state=_boom))
    assert adapter.storage_state() == {}
    assert "service.storage_state" in _sites()


# ---- Excluded handlers: proven NOT recording -----------------------------

def test_excluded_canvas_rung_records_nothing_on_fallback_success(tmp_path):
    tab = _tab(tmp_path)
    b64 = base64.b64encode(b"img").decode()

    def evaluate(js, *a, **k):
        if "CANVAS" in js:
            raise RuntimeError("canvas boom")
        return b64

    tab._page = SimpleNamespace(evaluate=evaluate)
    assert tab.element_image_bytes("#c") == b"img"
    assert bf.recent_failures() == []


def test_excluded_img_rung_records_nothing_on_screenshot_fallback(tmp_path):
    tab = _tab(tmp_path)
    calls = {"n": 0}

    def evaluate(js, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return ""  # no canvas element: rung yields nothing, no failure
        raise RuntimeError("fetch boom")

    class Locator:
        def screenshot(self):
            return b"shot"

    tab._page = SimpleNamespace(
        evaluate=evaluate, locator=lambda selector: Locator())
    assert tab.element_image_bytes("#i") == b"shot"
    assert bf.recent_failures() == []
