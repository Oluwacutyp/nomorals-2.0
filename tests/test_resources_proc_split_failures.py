"""Pins for Pass #136: /proc-reader tuple splits in nomorals/os/resources.py.

Follow-up to #123's named remainder. #123 excluded the four /proc
readers (_read_cpu_stat, _read_swap, _read_swap_io, _self_rss_mb) plus
effective_memory_mb's inner /proc/meminfo handler because each caught
(OSError, ValueError, IndexError) in ONE tuple handler: the OSError
branch is designed absence (/proc absent on a host is not a failure;
the contract is "None when unavailable"), while the ValueError /
IndexError branch is malformed content on present-but-broken input.
Splitting the tuple would leave a still-flagging OSError pass behind,
so #123 named the malformed corner as a remainder instead of
restructuring. This pass does the split:

- OSError branch: bare ``pass``, still silent BY DESIGN, still flagged
  by the scanner (one finding per site, exactly as before). No record.
- ValueError / IndexError branch: ``record_failure`` with the site
  name, then the byte-identical fall-through (same return, same
  partial state, same cgroup fall-through for the meminfo site).

COUNT HONESTY: the suppression-off count does NOT drop (8 -> 8) and
the flagged count does NOT drop (6 -> 6 for resources.py) — that is
CORRECT, not a failed conversion. Each partially-converted site keeps
its silent OSError branch, which the scanner still flags. The measure
of this pass is branch-level behaviour, pinned below, not the count.
New remainder accounting, with arithmetic: 8 findings = 5
partially-converted sites (the 4 readers + the meminfo inner handler:
OSError branch still silent, malformed branch now recorded) + 3
fully-excluded sites (_cgroup_mount's single OSError handler,
_sample_network's sysfs pass, sample()'s environment fallback — all
untouched by this pass).

Out of scope and untouched: _cgroup_mount, _sample_network, sample(),
_sample_mem, and the budgets/sampling/analysis regions closed by
#125/#127/#128.
"""

import ast
import builtins
import io
import os

import pytest

import nomorals.os.resources as res
from nomorals.os.resources_failures import (
    clear_failures,
    failure_count,
    recent_failures,
)


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


def _sites():
    return [f.site for f in recent_failures(500)]


def _entries():
    return recent_failures(500)


def _bare_manager(**attrs):
    m = res.ResourceManager.__new__(res.ResourceManager)
    for key, value in attrs.items():
        setattr(m, key, value)
    return m


def _raise_open(exc):
    def _open(*args, **kwargs):
        raise exc
    return _open


def _string_open(content):
    def _open(*args, **kwargs):
        return io.StringIO(content)
    return _open


class _RaiseOnIter:
    """Fake /proc file whose iteration raises ``exc`` on first read."""

    def __init__(self, exc):
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        raise self._exc


class _YieldThenRaise:
    """Fake /proc file yielding ``lines`` then raising ``exc``."""

    def __init__(self, lines, exc):
        self._lines = lines
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        yield from self._lines
        raise self._exc


# ---------------------------------------------------------------------------
# _read_cpu_stat: OSError silent, malformed recorded, return None either way
# ---------------------------------------------------------------------------

def test_cpu_stat_oserror_is_silent_none(monkeypatch):
    monkeypatch.setattr(builtins, "open", _raise_open(OSError("no /proc")))
    assert res._read_cpu_stat() is None
    assert _sites() == []


def test_cpu_stat_value_error_records(monkeypatch):
    monkeypatch.setattr(builtins, "open",
                        _string_open("cpu a b c d e f g h\n"))
    assert res._read_cpu_stat() is None
    assert failure_count() == 1
    assert _sites() == ["_read_cpu_stat"]
    assert _entries()[0].exc_type == "ValueError"


def test_cpu_stat_index_error_records(monkeypatch):
    # No natural IndexError path: the reader pads its number list to 8
    # before unpacking, so the branch is exercised with a /proc file
    # whose iteration itself raises IndexError — the branch's trigger
    # is a broken read, not a particular line shape.
    monkeypatch.setattr(builtins, "open",
                        lambda *a, **k: _RaiseOnIter(IndexError("short")))
    assert res._read_cpu_stat() is None
    assert failure_count() == 1
    assert _sites() == ["_read_cpu_stat"]
    assert _entries()[0].exc_type == "IndexError"


# ---------------------------------------------------------------------------
# _read_swap: OSError silent, malformed recorded, partial state preserved
# ---------------------------------------------------------------------------

def test_swap_oserror_is_silent_empty(monkeypatch):
    monkeypatch.setattr(builtins, "open", _raise_open(OSError("no /proc")))
    assert res._read_swap() == (None, None)
    assert _sites() == []


def test_swap_oserror_after_partial_total_still_returned_silently(monkeypatch):
    monkeypatch.setattr(
        builtins, "open",
        lambda *a, **k: _YieldThenRaise(["SwapTotal: 1000 kB\n"],
                                        OSError("truncated")))
    assert res._read_swap() == (1000, None)
    assert _sites() == []


def test_swap_value_error_records_with_partial_total(monkeypatch):
    monkeypatch.setattr(builtins, "open",
                        _string_open("SwapTotal: 1000 kB\nSwapFree: xx\n"))
    assert res._read_swap() == (1000, None)
    assert failure_count() == 1
    assert _sites() == ["_read_swap"]
    assert _entries()[0].exc_type == "ValueError"


def test_swap_index_error_records(monkeypatch):
    monkeypatch.setattr(builtins, "open", _string_open("SwapTotal:\n"))
    assert res._read_swap() == (None, None)
    assert failure_count() == 1
    assert _sites() == ["_read_swap"]
    assert _entries()[0].exc_type == "IndexError"


# ---------------------------------------------------------------------------
# _read_swap_io: OSError silent, malformed recorded, return None either way
# ---------------------------------------------------------------------------

def test_swap_io_oserror_is_silent_none(monkeypatch):
    monkeypatch.setattr(builtins, "open", _raise_open(OSError("no /proc")))
    assert res._read_swap_io() is None
    assert _sites() == []


def test_swap_io_value_error_records(monkeypatch):
    monkeypatch.setattr(builtins, "open", _string_open("pswpin xx\n"))
    assert res._read_swap_io() is None
    assert failure_count() == 1
    assert _sites() == ["_read_swap_io"]
    assert _entries()[0].exc_type == "ValueError"


def test_swap_io_index_error_records(monkeypatch):
    monkeypatch.setattr(builtins, "open", _string_open("pswpin\n"))
    assert res._read_swap_io() is None
    assert failure_count() == 1
    assert _sites() == ["_read_swap_io"]
    assert _entries()[0].exc_type == "IndexError"


# ---------------------------------------------------------------------------
# _self_rss_mb: OSError silent, malformed recorded, return None either way
# ---------------------------------------------------------------------------

def test_self_rss_oserror_is_silent_none(monkeypatch):
    monkeypatch.setattr(builtins, "open", _raise_open(OSError("no /proc")))
    assert res._self_rss_mb() is None
    assert _sites() == []


def test_self_rss_value_error_records(monkeypatch):
    monkeypatch.setattr(builtins, "open", _string_open("VmRSS: xx kB\n"))
    assert res._self_rss_mb() is None
    assert failure_count() == 1
    assert _sites() == ["_self_rss_mb"]
    assert _entries()[0].exc_type == "ValueError"


def test_self_rss_index_error_records(monkeypatch):
    monkeypatch.setattr(builtins, "open", _string_open("VmRSS:\n"))
    assert res._self_rss_mb() is None
    assert failure_count() == 1
    assert _sites() == ["_self_rss_mb"]
    assert _entries()[0].exc_type == "IndexError"


# ---------------------------------------------------------------------------
# effective_memory_mb inner meminfo handler: OSError silent fall-through,
# malformed recorded fall-through, same value either way, no double record
# ---------------------------------------------------------------------------

def _meminfo_open(monkeypatch, content=None, exc=None):
    orig_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if str(path).startswith("/proc/meminfo"):
            if exc is not None:
                raise exc
            return io.StringIO(content)
        return orig_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)


def test_meminfo_oserror_falls_through_silently(monkeypatch):
    monkeypatch.setattr(res, "cgroup_memory_limit_bytes", lambda: None)
    _meminfo_open(monkeypatch, exc=OSError("no meminfo here"))
    m = _bare_manager(_eff_cache={})
    assert m.effective_memory_mb() is None
    assert _sites() == []


def test_meminfo_oserror_falls_through_to_cgroup_limit_silently(monkeypatch):
    monkeypatch.setattr(res, "cgroup_memory_limit_bytes", lambda: 2 ** 30)
    _meminfo_open(monkeypatch, exc=OSError("no meminfo here"))
    m = _bare_manager(_eff_cache={})
    assert m.effective_memory_mb() == 1024.0
    assert _sites() == []


def test_meminfo_value_error_records_and_falls_through(monkeypatch):
    monkeypatch.setattr(res, "cgroup_memory_limit_bytes", lambda: None)
    _meminfo_open(monkeypatch, content="MemTotal: xx kB\n")
    m = _bare_manager(_eff_cache={})
    assert m.effective_memory_mb() is None
    assert failure_count() == 1
    assert _sites() == ["ResourceManager.effective_memory_mb.meminfo"]
    assert _entries()[0].exc_type == "ValueError"


def test_meminfo_value_error_falls_through_to_cgroup_limit(monkeypatch):
    monkeypatch.setattr(res, "cgroup_memory_limit_bytes", lambda: 2 ** 30)
    _meminfo_open(monkeypatch, content="MemTotal: xx kB\n")
    m = _bare_manager(_eff_cache={})
    assert m.effective_memory_mb() == 1024.0
    assert _sites() == ["ResourceManager.effective_memory_mb.meminfo"]


def test_meminfo_index_error_records_and_falls_through(monkeypatch):
    monkeypatch.setattr(res, "cgroup_memory_limit_bytes", lambda: None)
    _meminfo_open(monkeypatch, content="MemTotal:\n")
    m = _bare_manager(_eff_cache={})
    assert m.effective_memory_mb() is None
    assert failure_count() == 1
    assert _sites() == ["ResourceManager.effective_memory_mb.meminfo"]
    assert _entries()[0].exc_type == "IndexError"


# ---------------------------------------------------------------------------
# structural pins — the split shape itself, not just behaviour
# ---------------------------------------------------------------------------

def _record_sites_in(body):
    sites = []
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "record_failure" and node.args \
                and isinstance(node.args[0], ast.Constant):
            sites.append(node.args[0].value)
    return sites


def _is_split_pair(try_node, site):
    """True iff try_node is exactly [OSError-pass, (ValueError, IndexError)-record]."""
    handlers = try_node.handlers
    if len(handlers) != 2:
        return False
    first, second = handlers
    if not (isinstance(first.type, ast.Name) and first.type.id == "OSError"
            and first.name is None
            and len(first.body) == 1 and isinstance(first.body[0], ast.Pass)):
        return False
    if not (isinstance(second.type, ast.Tuple)
            and [e.id for e in second.type.elts if isinstance(e, ast.Name)]
            == ["ValueError", "IndexError"]
            and second.name == "e"
            and _record_sites_in(second.body) == [site]):
        return False
    return True


def _function_node(tree, name, class_name=None):
    if class_name is None:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
    else:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name == name:
                        return sub
    raise AssertionError(f"function {name} not found")


def test_split_shape_for_all_five_sites():
    tree = ast.parse(open(res.__file__).read())
    for name, site in (("_read_cpu_stat", "_read_cpu_stat"),
                       ("_read_swap", "_read_swap"),
                       ("_read_swap_io", "_read_swap_io"),
                       ("_self_rss_mb", "_self_rss_mb")):
        fn = _function_node(tree, name)
        tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
        assert len(tries) == 1
        assert _is_split_pair(tries[0], site), name
    fn = _function_node(tree, "effective_memory_mb", "ResourceManager")
    tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
    split = [t for t in tries if _is_split_pair(
        t, "ResourceManager.effective_memory_mb.meminfo")]
    assert len(split) == 1
    # The outer guard is unchanged: it still records the plain site name.
    outer = [t for t in tries if t is not split[0]]
    assert len(outer) == 1
    assert _record_sites_in(outer[0].handlers[0].body) == \
        ["ResourceManager.effective_memory_mb"]


# ---------------------------------------------------------------------------
# scan pins — branch-level remainder accounting (count unchanged by design)
# ---------------------------------------------------------------------------

def _enclosing_function(path, line):
    tree = ast.parse(open(path).read())
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= line <= node.end_lineno:
                if best is None or (node.end_lineno - node.lineno) < (
                        best.end_lineno - best.lineno):
                    best = node
    return best.name if best else None


def test_scan_remainder_is_8_with_branch_level_accounting():
    from collections import Counter

    from nomorals.tools import error_scan as es

    original = es._noqa_suppresses
    es._noqa_suppresses = lambda line, rule: False
    try:
        report = es.scan([os.path.join(os.path.dirname(res.__file__),
                                       "resources.py")])
        # 8 -> 8, NOT a failed conversion: each partially-converted site
        # keeps its silent OSError branch, which the scanner still flags.
        # 5 partially-converted (OSError branch silent, malformed branch
        # records): _read_cpu_stat, _read_swap, _read_swap_io,
        # _self_rss_mb, effective_memory_mb. 3 fully-excluded:
        # _cgroup_mount, _sample_network, sample.
        assert len(report.findings) == 8
        fns = Counter(_enclosing_function(f.file, f.line)
                      for f in report.findings)
        assert fns == Counter({
            "_cgroup_mount": 1, "_read_cpu_stat": 1, "_read_swap": 1,
            "_read_swap_io": 1, "_self_rss_mb": 1,
            "effective_memory_mb": 1, "_sample_network": 1, "sample": 1,
        })
    finally:
        es._noqa_suppresses = original


def test_scan_flagged_is_6_oserror_branches_only():
    from collections import Counter

    from nomorals.tools import error_scan as es

    report = es.scan([os.path.join(os.path.dirname(res.__file__),
                                   "resources.py")])
    # Default mode: the OSError branches carry no noqa, so they flag;
    # the recording branches use the exception, so they do not.
    # _sample_network and sample stay noqa-suppressed here, as before.
    fns = Counter(_enclosing_function(f.file, f.line)
                  for f in report.findings)
    assert fns == Counter({
        "_cgroup_mount": 1, "_read_cpu_stat": 1, "_read_swap": 1,
        "_read_swap_io": 1, "_self_rss_mb": 1, "effective_memory_mb": 1,
    })
