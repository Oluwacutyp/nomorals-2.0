"""Tests for the error-handling scanner."""

import textwrap
import unittest

from nomorals.tools.error_scan import Finding, scan


def run_scan(source: str) -> list[Finding]:
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(textwrap.dedent(source))
        path = fh.name
    try:
        return scan([path]).findings
    finally:
        os.unlink(path)


def rules(findings):
    return sorted(f.rule for f in findings)


class TestErrorScan(unittest.TestCase):
    def test_bare_except_is_error(self):
        f = run_scan("""
            try:
                x()
            except:
                pass
        """)
        self.assertIn("E101", rules(f))

    def test_except_base_exception_is_error(self):
        f = run_scan("""
            try:
                x()
            except BaseException as e:
                log(e)
        """)
        self.assertIn("E102", rules(f))

    def test_swallowed_pass_is_error(self):
        f = run_scan("""
            try:
                x()
            except ValueError:
                pass
        """)
        self.assertIn("E103", rules(f))

    def test_broad_except_ignoring_value_is_warning(self):
        f = run_scan("""
            try:
                x()
            except Exception:
                print("oops")
        """)
        self.assertIn("E104", rules(f))

    def test_broad_except_that_logs_is_clean(self):
        f = run_scan("""
            try:
                x()
            except Exception as e:
                log.warning("failed: %s", e)
        """)
        self.assertEqual(rules(f), [])

    def test_broad_except_that_reraises_is_clean(self):
        f = run_scan("""
            try:
                x()
            except Exception:
                cleanup()
                raise
        """)
        self.assertEqual(rules(f), [])

    def test_broad_except_that_classifies_is_clean(self):
        f = run_scan("""
            try:
                x()
            except Exception as e:
                err = classify(e)
                task.mark_failed(err.message)
        """)
        self.assertEqual(rules(f), [])

    def test_narrow_except_ignoring_value_is_clean(self):
        # specific catches are deliberate; the scanner only polices broad ones
        f = run_scan("""
            try:
                x()
            except ValueError:
                use_default()
        """)
        self.assertEqual(rules(f), [])

    def test_redundant_tuple_is_info(self):
        f = run_scan("""
            try:
                x()
            except (ValueError, Exception) as e:
                log(e)
        """)
        self.assertIn("E105", rules(f))

    def test_keyboard_interrupt_is_warning(self):
        f = run_scan("""
            try:
                x()
            except KeyboardInterrupt:
                shutdown()
        """)
        self.assertIn("E106", rules(f))

    def test_clean_code_has_no_findings(self):
        f = run_scan("""
            try:
                x()
            except ValueError as e:
                raise ConfigError(str(e)) from e
            except OSError as e:
                log.error("io: %s", e)
                return None
        """)
        self.assertEqual(rules(f), [])

    def test_report_ok_reflects_errors_only(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write("try:\n    x()\nexcept KeyboardInterrupt:\n    shutdown()\n")
            path = fh.name
        try:
            report = scan([path])
        finally:
            os.unlink(path)
        # warnings alone do not fail the scan
        self.assertTrue(report.ok)
        self.assertEqual(len(report.findings), 1)

    def test_noqa_suppresses_reviewed_catches(self):
        f = run_scan("""
            try:
                x()
            except Exception:  # noqa: BLE001 - deliberately resilient
                pass
        """)
        self.assertEqual(rules(f), [])

    def test_bare_noqa_suppresses_all(self):
        f = run_scan("""
            try:
                x()
            except:  # noqa
                pass
        """)
        self.assertEqual(rules(f), [])

    def test_noqa_after_pragma_is_honored(self):
        f = run_scan("""
            try:
                x()
            except ValueError:  # pragma: no cover, noqa: E103 - deliberate
                pass
        """)
        self.assertEqual(rules(f), [])

    def test_unparseable_file_does_not_crash_scan(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write("def broken(:\n")
            path = fh.name
        try:
            report = scan([path])
        finally:
            os.unlink(path)
        self.assertEqual(report.files_failed, 1)
        self.assertEqual(report.findings, [])


if __name__ == "__main__":
    unittest.main()
