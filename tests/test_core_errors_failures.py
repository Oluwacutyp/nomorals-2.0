"""Tail-104 pins for nomorals/core/errors.py — 0 converts / 4 exclusions.

Non-recording is structural: the module has no logger and no failure
ledger (grep-proven); its swallows feed return values only.

Sites:
117 — ``retry_after_of``: an exotic header mapping whose ``.get``
raises Exception -> ``raw = None``; extraction continues to the
``details`` source. The function's contract is best-effort-or-None;
None is an honest "no signal", never a fabricated delay.
122 — ``retry_after_of``: ``float(raw)`` TypeError/ValueError ->
``pass``; an unparseable ``Retry-After`` header falls through to the
details source / None. Try-parse idiom.
245 — ``NoMoralsError.from_dict``: the resolved class's custom
``__init__`` rejects the generic kwargs (TypeError) -> reconstruct as
a plain ``NoMoralsError``. This IS the docstring's stated contract
("falls back to a plain NoMoralsError when the class has a custom
constructor"), and the fallback preserves message, code, details and
retryable verbatim — nothing is claimed that was not in the payload.
248 — ``from_dict`` restore loop: ``setattr`` of a structured extra
raises Exception -> ``pass``. The value is NOT lost: it already lives
in ``err.details`` (the payload copy); the attribute is a convenience
mirror only.
"""

from __future__ import annotations

from types import SimpleNamespace

from nomorals.core.errors import NoMoralsError, retry_after_of


# ── 117: header .get crash -> extraction continues ───────────────

class _BadHeaders:
    def get(self, key, default=None):
        raise RuntimeError("exotic mapping boom")


def test_117_bad_header_mapping_falls_through_to_details():
    exc = SimpleNamespace(
        response=SimpleNamespace(headers=_BadHeaders()),
        details={"retry_after": 7},
    )
    assert retry_after_of(exc) == 7.0
    # and with no later source, the honest answer is None, not a guess
    exc2 = SimpleNamespace(response=SimpleNamespace(headers=_BadHeaders()))
    assert retry_after_of(exc2) is None


# ── 122: unparseable Retry-After header -> falls through ─────────

def test_122_unparseable_retry_after_header():
    exc = SimpleNamespace(
        response=SimpleNamespace(headers={"Retry-After": "soon"}),
    )
    assert retry_after_of(exc) is None
    # control: a parseable header is honored
    exc2 = SimpleNamespace(
        response=SimpleNamespace(headers={"Retry-After": "30"}),
    )
    assert retry_after_of(exc2) == 30.0


# ── 245: custom __init__ -> documented plain-class fallback ──────

def test_245_from_dict_custom_init_falls_back_with_payload_intact():
    # AmbiguousRef is the real in-module case: its __init__ takes
    # (ref, candidates, min_prefix_len), so the generic reconstruction
    # call raises TypeError and from_dict must degrade per contract.
    err = NoMoralsError.from_dict({
        "code": "resolve.ambiguous",
        "type": "AmbiguousRef",
        "message": "'ab' is ambiguous",
        "details": {"ref": "ab"},
        "retryable": False,
    })
    assert type(err) is NoMoralsError
    assert err.message == "'ab' is ambiguous"
    assert err.code == "resolve.ambiguous"
    assert err.details == {"ref": "ab"}


# ── 248: setattr of a structured extra fails -> value stays in details

class _PropErr(NoMoralsError):
    code = "tail104.prop"

    @property
    def field(self):  # type: ignore[override]
        raise AttributeError("field is not readable")


def test_248_from_dict_setattr_failure_keeps_value_in_details():
    try:
        err = NoMoralsError.from_dict({
            "code": "tail104.prop",
            "message": "prop boom",
            "details": {"field": "email"},
            "retryable": None,
        })
    finally:
        NoMoralsError._registry.pop("tail104.prop", None)
    # hasattr() is False (getter raises) so the restore loop attempts
    # setattr, which the setter-less property refuses — swallowed —
    # yet the value survives in the details payload copy
    assert type(err) is _PropErr
    assert err.details.get("field") == "email"
    # registry was cleaned back up for other tests
    assert "tail104.prop" not in NoMoralsError._registry
