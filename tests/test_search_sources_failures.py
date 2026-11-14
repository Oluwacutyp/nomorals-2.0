"""Pass #216 pins: search/sources.py (BooksAdapter cache) — 3 exclusions, 0 converts.

The three sites are the context Library cache, which is best-effort
by design: ``BooksAdapter._lib`` documents that "a context that
cannot hold attributes falls back to per-adapter construction". Each
excluded handler returns the documented absence (None) or skips the
cache store, and the caller constructs a fresh Library — no data is
lost and no error is hidden:

- 135: ``_context_books_library`` getattr failure -> None. The
  caller constructs fresh, exactly as the docstring says ("Anything
  unexpected (a Mock context, a context that cannot hold attributes)
  returns None — the caller constructs fresh, exactly as before").
- 149: ``_context_books_library`` stamp validation failure -> None.
  Same fresh-construction fallback; a cache entry whose DB path or
  ambient store no longer matches is not used.
- 162: ``_cache_books_library`` setattr failure -> pass. The Library
  itself is still returned to the caller (and stored on the adapter);
  only the context cache entry is missing, so the next adapter
  constructs fresh.

search/sources.py calls no failures-ledger record function anywhere,
so NOT-recording is pinned with the agency snapshot method: agency
recent_failures() before/after each trigger must be unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Triggers use contexts that raise on
attribute access and a mocked Library (no real I/O, no real home),
instead of swallowing.
"""

from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents import failures as ag_fail
from nomorals.search import sources as src


def _snap():
    return list(ag_fail.recent_failures(limit=200))


class _RaisingAttrs:
    """A context whose attribute access raises a non-AttributeError."""

    def __getattr__(self, name):
        raise RuntimeError("context is broken")


class _NoSetAttrs:
    """A context that reads fine but cannot hold attributes."""

    def __setattr__(self, name, value):
        raise AttributeError(name)


# ── L135: getattr failure returns None, caller constructs fresh ────

def test_135_context_getattr_failure_returns_none():
    before = _snap()
    # getattr(context, attr, None) only swallows AttributeError; a
    # context raising anything else trips the handler -> None.
    assert src._context_books_library(_RaisingAttrs()) is None
    assert _snap() == before


def test_135_bad_cache_shape_returns_none():
    # Control: a cache entry that is not a 3-tuple is also None.
    before = _snap()
    ctx = SimpleNamespace()
    setattr(ctx, src._BOOKS_LIBRARY_CACHE_ATTR, "not-a-tuple")
    assert src._context_books_library(ctx) is None
    assert _snap() == before


# ── L149: stamp validation failure returns None ────────────────────

def test_149_stamp_mismatch_returns_none():
    # A cached Library whose DB path no longer matches the live one
    # is not used; the caller constructs fresh.
    before = _snap()
    ctx = SimpleNamespace()
    setattr(ctx, src._BOOKS_LIBRARY_CACHE_ATTR,
            (object(), "/nonexistent/library.db", object()))
    assert src._context_books_library(ctx) is None
    assert _snap() == before


def test_149_validation_exception_returns_none():
    # Control: a validation step that raises (here _workspace_dir)
    # also yields None via the handler.
    before = _snap()
    ctx = SimpleNamespace()
    setattr(ctx, src._BOOKS_LIBRARY_CACHE_ATTR,
            (object(), "/x/library.db", object()))
    with patch.object(src, "_workspace_dir", side_effect=RuntimeError("boom")):
        assert src._context_books_library(ctx) is None
    assert _snap() == before


# ── L162: cache-store failure is a silent no-op ────────────────────

def test_162_cache_store_failure_does_not_raise():
    before = _snap()
    # _cache_books_library must not raise on a context that cannot
    # hold attributes; the Library is still usable by the caller.
    src._cache_books_library(_NoSetAttrs(), object())
    assert _snap() == before


def test_162_cache_store_failure_next_lookup_constructs_fresh():
    # End-to-end: a context that cannot cache still yields a working
    # Library from _lib() — it just constructs one per adapter.
    # Library is mocked (no real I/O, no real home).
    before = _snap()
    sentinel = object()
    with patch.object(src, "Library", return_value=sentinel):
        adapter = src.BooksAdapter(_NoSetAttrs())
        assert adapter._lib() is sentinel
        adapter2 = src.BooksAdapter(_NoSetAttrs())
        assert adapter2._lib() is sentinel
    assert _snap() == before
