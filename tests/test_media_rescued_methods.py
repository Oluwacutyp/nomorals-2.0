"""Regression test: the 8 media control methods rescued from inside
``_resolve_play_title`` must be real methods of ``RuntimeMediaMixin``
so dispatch (``self._control_*``) can reach them.
"""
import ast
import inspect

import pytest

from nomorals.agents.partner.runtime_media import (
    RuntimeMediaMixin,
    _resolve_play_title,
)

RESCUED = [
    "_control_video",
    "_control_image",
    "_control_lens",
    "_control_gen",
    "_control_apps",
    "_control_podcast",
    "_control_record",
    "_control_publish",
]


def test_all_eight_are_class_methods():
    for name in RESCUED:
        fn = getattr(RuntimeMediaMixin, name, None)
        assert callable(fn), f"{name} missing from RuntimeMediaMixin"
        assert "self" in inspect.signature(fn).parameters, f"{name} has no self"


def test_dispatch_signatures_compatible():
    """Every rescued method must accept the call shape dispatch uses."""
    for name in RESCUED:
        sig = inspect.signature(getattr(RuntimeMediaMixin, name))
        params = list(sig.parameters.values())
        # (self, tail, [chat_key]) — tail required, chat_key optional
        assert params[1].name == "tail"
        assert params[1].default is inspect.Parameter.empty
        for p in params[2:]:
            assert p.default != inspect.Parameter.empty, (
                f"{name}: param {p.name} has no default — dispatch may omit it")


def test_no_nested_defs_in_resolve_play_title():
    src = inspect.getsource(_resolve_play_title)
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    nested = [n.name for n in ast.walk(fn)
              if isinstance(n, ast.FunctionDef) and n is not fn]
    assert nested == [], f"still nested: {nested}"


def test_dispatch_call_shapes():
    """Simulate the exact dispatch call shapes from runtime.py.

    Proves each method is *reachable* via dispatch (the original bug was
    AttributeError on the method itself). Deep execution may need a real
    context — an AttributeError naming anything *other* than the method
    is fine here.
    """
    class FakeSelf:
        context = None
        gateway = None

        def _control_hub(self, tail, chat_key=""):
            return f"hub:{tail}"

    inst = FakeSelf()
    calls = [
        ("_control_video", ("x",), {}),
        ("_control_apps", ("x",), {}),
        ("_control_podcast", ("x",), {"chat_key": "k"}),
        ("_control_gen", ("x",), {}),
        ("_control_record", ("x",), {}),
        ("_control_publish", ("x",), {}),
        ("_control_image", ("x",), {}),
        ("_control_lens", ("x",), {}),
    ]
    for name, args, kwargs in calls:
        fn = getattr(RuntimeMediaMixin, name, None)
        assert callable(fn), f"{name} not reachable via dispatch"
        try:
            result = fn.__get__(inst)(*args, **kwargs)
            assert isinstance(result, str), f"{name} returned {type(result)}"
        except AttributeError as exc:
            # reachable but needs real context — only fail if the *method*
            # itself is what's missing (the original bug)
            assert name not in str(exc), f"{name} still unreachable: {exc}"
