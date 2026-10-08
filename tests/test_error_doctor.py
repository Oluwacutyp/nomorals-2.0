"""Tests for nomorals.core.error_doctor — the dynamic error diagnostician.

All offline. Each test constructs a real failure, catches it, and checks
the diagnosis is correct and actionable. The diagnostician must never raise.
"""

import unittest

from nomorals.core.error_doctor import diagnose, diagnosis_to_text


def _capture(fn, *args, **kwargs):
    """Run fn, return the caught exception."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return exc
    raise AssertionError(f"{fn.__name__} did not raise")


class UnboundLocalTests(unittest.TestCase):
    def test_branch_assignment_diagnosed(self):
        def buggy(flag):
            if flag:
                chat = {"key": "abc"}
            return chat["key"].upper()  # noqa: F821 - the point of the test

        exc = _capture(buggy, False)
        self.assertIsInstance(exc, UnboundLocalError)
        d = diagnose(exc)
        self.assertEqual(d["error_type"], "UnboundLocalError")
        self.assertIn("chat", d["root_cause"])
        # the assignment line must be identified
        self.assertTrue(d["evidence"]["assigned_at_lines"],
                        "should find where chat is assigned")
        # condition values: flag=False at crash time
        self.assertIn("flag=False", d["root_cause"])
        # actionable fix
        self.assertIn("chat", d["suggested_fix"])
        self.assertIn("location", d)
        self.assertIn(":", d["location"])

    def test_never_assigned(self):
        def buggy2():
            return ghost_var + 1  # noqa: F821

        exc = _capture(buggy2)
        d = diagnose(exc)
        # NameError (module-level style) or UnboundLocalError depending on path
        self.assertIn(d["error_type"], ("NameError", "UnboundLocalError"))
        self.assertIn("ghost_var", d["root_cause"])

    def test_frames_have_source(self):
        def buggy(flag):
            if flag:
                x = 1
            return x + 1  # noqa: F821

        exc = _capture(buggy, False)
        d = diagnose(exc)
        self.assertTrue(d["frames"])
        last = d["frames"][-1]
        self.assertTrue(last["source"], "frames should carry source lines")
        self.assertTrue(any(">>>" in line for line in last["source"]))


class ImportErrorTests(unittest.TestCase):
    def test_missing_module_gives_pip_command(self):
        try:
            import nonexistent_module_xyz_abc  # noqa: F401
        except ImportError as exc:
            d = diagnose(exc)
        self.assertEqual(d["error_type"], "ModuleNotFoundError")
        self.assertIn("pip install nonexistent_module_xyz_abc",
                      d["suggested_fix"])
        self.assertIn("not installed", d["root_cause"])

    def test_stdlib_broken_import(self):
        # simulate: stdlib module that fails -> should NOT suggest pip install
        exc = ModuleNotFoundError("No module named 'json'; 'json' is not a package")
        exc.name = "json"
        d = diagnose(exc)
        self.assertNotIn("pip install json", d["suggested_fix"])
        self.assertIn("standard-library", d["root_cause"])


class AttributeErrorTests(unittest.TestCase):
    def test_close_match_suggested(self):
        class Widget:
            def __init__(self):
                self.send_message = lambda: None

        def buggy():
            w = Widget()
            return w.send_mesage  # typo

        exc = _capture(buggy)
        d = diagnose(exc)
        self.assertEqual(d["error_type"], "AttributeError")
        self.assertIn("send_message", d["root_cause"])
        self.assertIn("send_message", d["suggested_fix"])

    def test_no_close_match_lists_available(self):
        class Empty:
            pass

        def buggy():
            return Empty().zzz_nope

        exc = _capture(buggy)
        d = diagnose(exc)
        self.assertIn("zzz_nope", d["root_cause"])
        self.assertTrue(d["suggested_fix"])


class KeyErrorTests(unittest.TestCase):
    def test_available_keys_shown(self):
        def buggy():
            cfg = {"host": "x", "port": 1, "token": "t"}
            return cfg["hot"]  # typo for host

        exc = _capture(buggy)
        d = diagnose(exc)
        self.assertEqual(d["error_type"], "KeyError")
        self.assertIn("host", d["root_cause"])
        self.assertIn("port", d["root_cause"])

    def test_keyerror_no_args(self):
        d = diagnose(KeyError())
        self.assertEqual(d["error_type"], "KeyError")
        self.assertTrue(d["summary"])


class ConnectionErrorTests(unittest.TestCase):
    def test_host_extracted(self):
        exc = ConnectionError("failed to reach https://api.example.com:8443/v1")
        d = diagnose(exc)
        self.assertIn("api.example.com", d["root_cause"])
        self.assertEqual(d["evidence"]["host"], "https://api.example.com:8443/v1")

    def test_timeout_error(self):
        d = diagnose(TimeoutError("timed out after 30s"))
        self.assertEqual(d["error_type"], "TimeoutError")
        self.assertTrue(d["suggested_fix"])


class TypeErrorTests(unittest.TestCase):
    def test_operand_types(self):
        def buggy():
            return "a" + 1

        exc = _capture(buggy)
        d = diagnose(exc)
        self.assertEqual(d["error_type"], "TypeError")
        # operands are literals: the AST pass should spot str and int
        self.assertIn("str", d["evidence"].get("operand_types_on_line", []))
        self.assertIn("int", d["evidence"].get("operand_types_on_line", []))
        self.assertIn("str", d["root_cause"])


class NameErrorTests(unittest.TestCase):
    def test_typo_suggested(self):
        def buggy():
            chat_key = "a:b"
            return chaat_key  # noqa: F821 - typo on purpose

        exc = _capture(buggy)
        d = diagnose(exc)
        self.assertEqual(d["error_type"], "NameError")
        self.assertIn("chat_key", d["root_cause"])


class NeverRaisesTests(unittest.TestCase):
    def test_garbage_input(self):
        for bad in (None, "not an exception", 42, object()):
            d = diagnose(bad)
            self.assertIsInstance(d, dict)
            self.assertIn("summary", d)

    def test_exception_without_traceback(self):
        exc = ValueError("boom")
        exc.__traceback__ = None
        d = diagnose(exc)
        self.assertEqual(d["error_type"], "ValueError")
        self.assertEqual(d["location"], "?")
        self.assertTrue(d["frames"] == [])

    def test_broken_repr_locals(self):
        class Evil:
            def __repr__(self):
                raise RuntimeError("repr is evil")

        def buggy():
            e = Evil()
            raise ValueError("with evil local")

        exc = _capture(buggy)
        d = diagnose(exc)  # must not raise despite evil repr
        self.assertEqual(d["error_type"], "ValueError")

    def test_diagnosis_to_text_never_raises(self):
        self.assertIsInstance(diagnosis_to_text({}), str)
        self.assertIsInstance(diagnosis_to_text(None), str)

    def test_chained_exception(self):
        def buggy():
            try:
                {}["k"]
            except KeyError as e:
                raise RuntimeError("wrapping") from e

        exc = _capture(buggy)
        d = diagnose(exc)
        self.assertEqual(d["error_type"], "RuntimeError")
        self.assertTrue(d["frames"])


class StructuredOutputTests(unittest.TestCase):
    def test_all_keys_present(self):
        def buggy():
            return 1 / 0

        exc = _capture(buggy)
        d = diagnose(exc)
        for key in ("error_type", "location", "root_cause", "evidence",
                    "suggested_fix", "frames", "summary"):
            self.assertIn(key, d, f"missing key: {key}")
        self.assertEqual(d["error_type"], "ZeroDivisionError")
        # summary is a non-empty paragraph
        self.assertTrue(len(d["summary"]) > 20)
        self.assertIn("ZeroDivisionError", d["summary"])

    def test_context_command_carried(self):
        d = diagnose(ValueError("x"), context={"command": "/research"})
        self.assertEqual(d["evidence"]["command"], "/research")


if __name__ == "__main__":
    unittest.main()
