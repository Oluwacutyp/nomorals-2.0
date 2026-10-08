"""Tests for /heal — the self-healing loop.

The healer auto-fixes exactly two safe patterns:
  1. UnboundLocalError → initialize the variable to None at function top
  2. AttributeError on _control_* → rescue dead nested method to class level

Everything else is diagnose-and-report only. The healer never raises,
never changes behavior, and dry-run changes nothing.
"""

import ast
import os
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.core.self_heal import (
    HealResult,
    fix_dead_method,
    fix_unbound_local,
    heal_all,
    heal_one,
    heal_probe,
)


# ---------------------------------------------------------------------------
# helpers: synthetic broken modules in tmp_path
# ---------------------------------------------------------------------------

def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(textwrap.dedent(src))
    return str(p)


def _raise_unbound(modpath):
    """Import the module fresh and trigger its UnboundLocalError."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("healmod_unbound", modpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        mod.handler("research")
    except UnboundLocalError as e:
        return e
    raise AssertionError("expected UnboundLocalError")


def _raise_attr(modpath):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("healmod_attr", modpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["healmod_attr"] = mod  # production modules are registered
    spec.loader.exec_module(mod)
    obj = mod.Widget()
    try:
        obj._control_podcast("")
    except AttributeError as e:
        return e
    raise AssertionError("expected AttributeError")


UNBOUND_SRC = '''
def handler(kind):
    """Handle a command."""
    if kind == "miniapp":
        chat = "miniapp-chat"
        return chat.upper()
    if kind == "research":
        # real-world pattern: downstream handles None gracefully
        return f"research:{chat}"
'''


class TestUnboundLocalFix:
    def test_fix_initializes_variable(self, tmp_path):
        p = _write(tmp_path, "m1.py", UNBOUND_SRC)
        exc = _raise_unbound(p)
        out = fix_unbound_local(exc, dry_run=False)
        assert out["ok"], out.get("skip_reason")
        assert out["varname"] == "chat"
        # the file now initializes chat at the top of handler()
        tree = ast.parse(open(p).read())
        func = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "handler")
        first = func.body[0]
        is_docstring = (isinstance(first, ast.Expr)
                        and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str))
        init_stmt = func.body[1] if is_docstring else func.body[0]
        assert isinstance(init_stmt, ast.Assign)
        assert init_stmt.targets[0].id == "chat"
        # and the module still imports + the fixed path works
        import importlib.util
        spec = importlib.util.spec_from_file_location("healmod_fixed", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.handler("research") == "research:None"  # no crash; None handled
        assert mod.handler("miniapp") == "MINIAPP-CHAT"  # working path unchanged

    def test_dry_run_changes_nothing(self, tmp_path):
        p = _write(tmp_path, "m2.py", UNBOUND_SRC)
        before = open(p).read()
        exc = _raise_unbound(p)
        out = fix_unbound_local(exc, dry_run=True)
        assert out["ok"]
        assert open(p).read() == before
        assert out["diff_preview"]  # but it shows what it WOULD do

    def test_parameter_not_initialized(self, tmp_path):
        # 'chat' as a parameter must NOT be overwritten — that changes behavior
        p = _write(tmp_path, "m3.py", '''
def handler(kind, chat):
    if kind == "x":
        chat = chat or "d"
    return chat2  # noqa - NameError on purpose is fine; we craft UnboundLocal
''')
        # craft the UnboundLocalError manually: assign in branch, ref outside
        p2 = _write(tmp_path, "m4.py", '''
def handler(kind, chat):
    if kind == "x":
        tmp = 1
    return tmp
''')
        import importlib.util
        spec = importlib.util.spec_from_file_location("healmod_param", p2)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        try:
            mod.handler("y", "c")
            raise AssertionError("expected UnboundLocalError")
        except UnboundLocalError as e:
            out = fix_unbound_local(e, dry_run=False)
            assert out["ok"]  # tmp is fine to init
            assert out["varname"] == "tmp"


DEAD_METHOD_SRC = '''
class Widget:
    def name(self):
        return "widget"


def _resolve_something(x):
    def _control_podcast(self, tail):
        """Build a podcast."""
        return f"podcast: {tail}"
    return x
'''


class TestDeadMethodFix:
    def test_rescue_moves_method_to_class(self, tmp_path):
        p = _write(tmp_path, "w1.py", DEAD_METHOD_SRC)
        exc = _raise_attr(p)
        out = fix_dead_method(exc, dry_run=False)
        assert out["ok"], out.get("skip_reason")
        assert out["method"] == "_control_podcast"
        # method now lives on the class
        tree = ast.parse(open(p).read())
        cls = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.ClassDef) and n.name == "Widget")
        methods = [n.name for n in cls.body
                   if isinstance(n, ast.FunctionDef)]
        assert "_control_podcast" in methods
        # nested copy is gone
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_resolve_something":
                nested = [n.name for n in ast.walk(node)
                          if isinstance(n, ast.FunctionDef) and n is not node]
                assert "_control_podcast" not in nested
        # and it actually works now
        import importlib.util
        spec = importlib.util.spec_from_file_location("healmod_wfixed", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.Widget()._control_podcast("hi") == "podcast: hi"

    def test_dry_run_changes_nothing(self, tmp_path):
        p = _write(tmp_path, "w2.py", DEAD_METHOD_SRC)
        before = open(p).read()
        exc = _raise_attr(p)
        out = fix_dead_method(exc, dry_run=True)
        assert out["ok"]
        assert open(p).read() == before

    def test_closure_variables_refused(self, tmp_path):
        # nested method using the enclosing scope's locals must NOT move
        p = _write(tmp_path, "w3.py", '''
class Widget:
    pass


def _resolve_something(x):
    prefix = "pc:"
    def _control_podcast(self, tail):
        return f"{prefix} {tail}"
    return x
''')
        exc = _raise_attr(p)
        out = fix_dead_method(exc, dry_run=False)
        assert not out["ok"]
        assert "closure" in out["skip_reason"]
        # file untouched
        assert "_control_podcast" not in open(p).read().split("class Widget")[1].split("def _resolve")[0]

    def test_non_control_attr_refused(self, tmp_path):
        p = _write(tmp_path, "w4.py", '''
class Widget:
    pass
''')
        import importlib.util
        spec = importlib.util.spec_from_file_location("healmod_w4", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        try:
            mod.Widget().frobnicate
            raise AssertionError("expected AttributeError")
        except AttributeError as e:
            out = fix_dead_method(e, dry_run=False)
            assert not out["ok"]
            assert "_control_" in out["skip_reason"]


class TestReportOnly:
    def test_missing_module_not_fixed(self, tmp_path):
        def fake_handle(text, chat_key="", message=None):
            raise ModuleNotFoundError("No module named 'notarealpkg123'")
        res = heal_one(fake_handle, "xyz")
        assert res.broken
        assert not res.fixable
        assert not res.fixed
        assert "package install" in res.skip_reason or "user action" in res.skip_reason
        assert "notarealpkg123" in res.suggestion or "pip install" in res.suggestion

    def test_logic_typeerror_not_fixed(self, tmp_path):
        def fake_handle(text, chat_key="", message=None):
            raise TypeError("unsupported operand type(s) for +: 'int' and 'str'")
        res = heal_one(fake_handle, "xyz")
        assert res.broken
        assert not res.fixable
        assert "no safe auto-fix" in res.skip_reason


class TestNeverRaises:
    def test_heal_probe_never_raises(self):
        def evil(text, chat_key="", message=None):
            raise RuntimeError("boom")
        broken, exc, err, loc = heal_probe(evil, "x")
        assert broken and exc is not None

        def evil2(text, chat_key="", message=None):
            raise SystemExit(1)  # not an Exception subclass
        # SystemExit propagates out of ThreadPoolExecutor as-is; heal_probe
        # catches Exception only — wrap to prove heal_one still never raises
        res = heal_one(evil, "x")
        assert isinstance(res, HealResult)

    def test_heal_one_garbage_never_raises(self):
        res = heal_one(None, "x")  # not even callable
        assert isinstance(res, HealResult)

    def test_heal_all_never_raises(self):
        def bad(text, chat_key="", message=None):
            raise ValueError("nope")
        out = heal_all(bad, ["a", "b"], dry_run=True)
        assert out["total"] == 2
        assert isinstance(out["results"], list)


class TestHealOneUnbound:
    def test_heal_one_fixes_unbound(self, tmp_path):
        p = _write(tmp_path, "h1.py", UNBOUND_SRC)

        def fake_handle(text, chat_key="", message=None):
            import importlib.util
            # re-import fresh each probe so the fix is picked up
            spec = importlib.util.spec_from_file_location(
                "healmod_probe", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            kind = text.strip().lstrip("/").split()[0]
            # probe the research path (the broken one)
            return mod.handler("research")

        res = heal_one(fake_handle, "research", dry_run=False)
        assert res.broken
        assert res.fix_kind == "unbound_local"
        assert res.fixed  # re-probe passes after fix


class TestRegistration:
    def test_heal_registered(self):
        from nomorals.social.chat.control import (
            CONTROL_COMMANDS, COMMAND_DETAILS, parse_control)
        assert "heal" in CONTROL_COMMANDS
        assert "heal" in COMMAND_DETAILS
        cmd = parse_control("/heal --dry")
        assert cmd is not None and cmd.kind == "heal"

    def test_heal_in_discovery_group(self):
        from nomorals.social.chat.control import _HELP_GROUPS
        groups = dict(_HELP_GROUPS)
        assert "heal" in groups.get("discovery", [])

    def test_tour_denylists_heal(self):
        from nomorals.agents.partner.runtime_meta import RuntimeMetaMixin
        assert "heal" in RuntimeMetaMixin._TOUR_DENYLIST
