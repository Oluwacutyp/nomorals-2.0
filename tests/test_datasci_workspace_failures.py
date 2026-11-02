"""Tail-102 pins: nomorals/datasci/workspace.py census sites —
all best-effort helper fallbacks, all held as exclusions.

Every site degrades to the module's own documented default on a
diagnostic/provenance helper — the docstrings state the fallback
("best-effort", "never to a constant") — and none of the four
feeds a decision gate: provenance here is traceability (source,
sha, rows, columns printed beside an analysis), not a dedupe or
admission check. All four triggers were probed live before
pinning.

124 — ``Dataset.memory_bytes``: ``frame.memory_usage`` raises
-> 0. Zero is the property's unknown value for a display stat;
the tension is stated plainly: 0 also reads as a measurement.
Held: the property is a diagnostic beside rows/columns, no
consumer sizes or admits anything by it, and a frame whose
memory_usage raises is exotic (duck-typed frames).
163 — ``_sha256_frame``: the pandas content hash raises
(exotic dtypes) -> hash of shape+dtypes instead. The fallback
is still input-derived, never a constant, exactly as the
docstring contracts. The tension: two distinct frames sharing
shape+dtypes collide under the fallback, so the provenance sha
can over-claim identity — held because the sha's in-module job
is traceability stamping at load (line 314), not equality
gating, and the degraded inputs are the exotic-dtype minority.
185 — ``_cat_label``: ``pd.isna`` on an array-like returns an
array, ``bool()`` raises -> fall through to ``str(value)``,
which is the designed label path for non-scalars. Missing
values still render as the ∅ bucket (control below).
884 — ``_is_numeric_dtype``: any probe failure (e.g. an object
with no ``.dtype``) -> False, routing the column to categorical
handling — the safe direction for charting, and the same
answer a genuinely non-numeric dtype earns.

A bind + ``_log.debug`` sketch for 124/163 (the module HAS a
logger; both sites are bare) is shelved as a follow-up —
observability only, the returned values are already the
documented defaults.

NOT-recording is pinned with the snapshot method: the module
calls no failures-ledger record function anywhere, so both
general ledgers must be unchanged across each trigger.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.datasci import workspace as ws


def _snapshot():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def test_124_memory_bytes_unknown_is_zero():
    before = _snapshot()

    class ExplodingFrame:
        shape = (3, 2)
        columns = ["a", "b"]

        def memory_usage(self, index=True, deep=True):
            raise RuntimeError("no memory")

    ds = ws.Dataset(name="d", frame=ExplodingFrame(), source="s", sha256="x")
    assert ds.memory_bytes == 0
    # control: a real frame measures positively
    real = ws.Dataset(name="r", frame=pd.DataFrame({"a": [1, 2]}),
                      source="s", sha256="y")
    assert real.memory_bytes > 0
    assert _snapshot() == before


def test_163_frame_hash_fallback_is_shape_and_dtypes(monkeypatch):
    before = _snapshot()
    real_hash = ws._sha256_frame(pd.DataFrame({"a": [1, 2]}))

    def _exotic(*_args, **_kwargs):
        raise RuntimeError("exotic dtype")

    monkeypatch.setattr(pd.util, "hash_pandas_object", _exotic)
    fallback = ws._sha256_frame(pd.DataFrame({"a": [1, 2]}))
    expected = hashlib.sha256(repr(((2, 1), ["int64"])).encode()).hexdigest()
    assert fallback == expected  # input-derived, never a constant
    assert fallback != real_hash
    assert _snapshot() == before


def test_185_cat_label_array_value_falls_to_str():
    before = _snapshot()
    assert ws._cat_label(np.array([1, 2])) == "[1 2]"
    # controls: missing renders as the ∅ bucket, scalars as themselves
    assert ws._cat_label(None) == "∅"
    assert ws._cat_label("x") == "x"
    assert _snapshot() == before


def test_884_is_numeric_dtype_probe_failure_is_false():
    before = _snapshot()
    assert ws._is_numeric_dtype(SimpleNamespace()) is False
    # controls: genuine dtypes answer for themselves
    assert ws._is_numeric_dtype(pd.Series([1.5])) is True
    assert ws._is_numeric_dtype(pd.Series(["a"])) is False
    assert _snapshot() == before
