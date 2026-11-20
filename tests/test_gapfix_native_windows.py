"""Gapfix — Windows-native self-provision for nomorals.native.

The owner reported the startup log on EC2: "native mem not built ...
Run `nm native --build` to compile for this machine" on every process
start.  The build path was Unix-shaped (``.so`` names, ``g++``-only
flags, no MSVC), nothing ever built the libraries automatically, and
the log handed the owner a command.  This pins the fix: Windows-correct
names and flags, Devon builds the libraries herself at boot (installing
a compiler first on Windows when none exists), a failed attempt is
recorded on disk and not retried on every boot, and the startup log
never tells the owner to run a command.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals import native

_FAKE_GCC = native.Compiler(path="/usr/bin/g++", kind="gcc", vcvarsall=None)
_FAKE_CL = native.Compiler(path="cl.exe", kind="msvc", vcvarsall=None)
_FAKE_CL_VS = native.Compiler(
    path=r"C:\VS\VC\Tools\MSVC\14.44\bin\Hostx64\x64\cl.exe",
    kind="msvc",
    vcvarsall=r"C:\VS\VC\Auxiliary\Build\vcvarsall.bat",
)


def _fake_run_factory(tmp: Path, failures: dict[str, int] | None = None):
    """A subprocess.run stand-in that 'compiles' by touching targets."""
    calls: list[list[str]] = []
    failures = failures or {}

    def _run(argv, **_kwargs):
        calls.append(list(argv))
        target: str | None = None
        if "-o" in argv:
            target = argv[argv.index("-o") + 1]
        else:
            for arg in argv:
                if isinstance(arg, str) and arg.startswith("/OUT:"):
                    target = arg[5:]
        if target is None:
            for arg in argv:
                if isinstance(arg, str) and "/OUT:" in arg:
                    # MSVC via vcvarsall: the whole command is one string.
                    target = arg[arg.index("/OUT:") + 5:].strip()
                    break
        if target:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            Path(target).touch()
        key = " ".join(str(a) for a in argv)
        rc = 0
        for needle, code in failures.items():
            if needle in key:
                rc = code
        return subprocess.CompletedProcess(
            args=argv, returncode=rc, stdout="", stderr="boom" if rc else "")

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


class LibNameTest(unittest.TestCase):
    def test_windows_uses_dll(self):
        with mock.patch.object(platform, "system", return_value="Windows"):
            self.assertEqual(native._lib_name("libvecsim"), "libvecsim.dll")

    def test_darwin_uses_dylib(self):
        with mock.patch.object(platform, "system", return_value="Darwin"):
            self.assertEqual(native._lib_name("libvecsim"), "libvecsim.dylib")

    def test_linux_uses_so(self):
        with mock.patch.object(platform, "system", return_value="Linux"):
            self.assertEqual(native._lib_name("libvecsim"), "libvecsim.so")


class CompilerDetectionTest(unittest.TestCase):
    def test_windows_finds_cl_exe_on_path(self):
        with mock.patch.object(platform, "system", return_value="Windows"), \
                mock.patch.object(
                    shutil, "which",
                    lambda name, **_kw: "cl.exe" if name == "cl.exe" else None), \
                mock.patch.object(native, "_find_vswhere", return_value=None), \
                mock.patch.object(native, "_find_msvc", return_value=None):
            found = native.detect_compiler()
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.kind, "msvc")
        self.assertEqual(found.path, "cl.exe")

    def test_windows_prefers_mingw_gcc_over_msvc(self):
        def _which(name, **_kw):
            return {"g++": r"C:\msys64\ucrt64\bin\g++.exe",
                    "cl": "cl.exe", "cl.exe": "cl.exe"}.get(name)
        with mock.patch.object(platform, "system", return_value="Windows"), \
                mock.patch.object(shutil, "which", _which):
            found = native.detect_compiler()
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.kind, "gcc")
        self.assertEqual(found.path, r"C:\msys64\ucrt64\bin\g++.exe")

    def test_windows_falls_back_to_vswhere_install(self):
        with mock.patch.object(platform, "system", return_value="Windows"), \
                mock.patch.object(shutil, "which", lambda name, **_kw: None), \
                mock.patch.object(native, "_find_vswhere", return_value=None), \
                mock.patch.object(native, "_find_msvc",
                                  return_value=_FAKE_CL_VS):
            found = native.detect_compiler()
        self.assertEqual(found, _FAKE_CL_VS)

    def test_find_vswhere_reads_install_path(self):
        tmp = Path(tempfile.mkdtemp(prefix="nm-vswhere-"))
        installer = tmp / "Microsoft Visual Studio" / "Installer"
        installer.mkdir(parents=True)
        (installer / "vswhere.exe").touch()
        # _find_vswhere reads the capitalized names (SIM112); on
        # Windows os.environ lookups are case-insensitive, so the
        # real "ProgramFiles(x86)" still resolves.
        with mock.patch.object(platform, "system", return_value="Windows"), \
                mock.patch.dict("os.environ",
                                {"PROGRAMFILES(X86)": str(tmp)}):
            found = native._find_vswhere()
        self.assertEqual(found, str(installer / "vswhere.exe"))
        shutil.rmtree(tmp)

    def test_find_msvc_picks_newest_toolset(self):
        tmp = Path(tempfile.mkdtemp(prefix="nm-msvc-"))
        installer = tmp / "Microsoft Visual Studio" / "Installer"
        installer.mkdir(parents=True)
        (installer / "vswhere.exe").touch()
        for version in ("14.38", "14.44"):
            bindir = (tmp / "VC" / "Tools" / "MSVC" / version
                      / "bin" / "Hostx64" / "x64")
            bindir.mkdir(parents=True)
            (bindir / "cl.exe").touch()
        (tmp / "VC" / "Auxiliary" / "Build").mkdir(parents=True)
        (tmp / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat").touch()
        fake = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=str(tmp) + "\r\n", stderr="")
        with mock.patch.object(platform, "system", return_value="Windows"), \
                mock.patch.dict("os.environ",
                                {"PROGRAMFILES(X86)": str(tmp)}), \
                mock.patch.object(subprocess, "run", return_value=fake):
            found = native._find_msvc()
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.kind, "msvc")
        self.assertIn("14.44", found.path)
        self.assertTrue(found.vcvarsall.endswith("vcvarsall.bat"))
        shutil.rmtree(tmp)

    def test_posix_keeps_historical_order(self):
        def _which(name, **_kw):
            return {"c++": "/usr/bin/c++", "g++": "/usr/bin/g++"}.get(name)
        with mock.patch.object(platform, "system", return_value="Linux"), \
                mock.patch.object(shutil, "which", _which):
            found = native.detect_compiler()
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.path, "/usr/bin/c++")
        self.assertEqual(found.kind, "gcc")

    def test_find_compiler_returns_path_or_none(self):
        with mock.patch.object(native, "detect_compiler",
                               return_value=_FAKE_GCC):
            self.assertEqual(native.find_compiler(), "/usr/bin/g++")
        with mock.patch.object(native, "detect_compiler", return_value=None):
            self.assertIsNone(native.find_compiler())


class BuildCommandTest(unittest.TestCase):
    """build() argv shape per compiler, with subprocess.run mocked."""

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="nm-build-"))
        self._old_here = native._HERE
        native._HERE = self._tmp
        self._old_targets = native.BUILD_TARGETS
        sources = []
        for stem in ("libvecsim", "libmlptrain", "libbpe", "libmemextract"):
            src = self._tmp / f"{stem}.cpp"
            src.touch()
            sources.append((src, stem))
        native.BUILD_TARGETS = tuple(sources)
        self._saved = dict(native._LIBS)
        for slot in native._LIBS:
            native._LIBS[slot] = native._NOT_GIVEN
        self.addCleanup(self._restore)

    def _restore(self):
        native._HERE = self._old_here
        native.BUILD_TARGETS = self._old_targets
        native._LIBS.update(self._saved)
        shutil.rmtree(self._tmp)

    def test_gcc_argv_shape(self):
        run = _fake_run_factory(self._tmp)
        with mock.patch.object(platform, "system", return_value="Linux"), \
                mock.patch.object(native, "detect_compiler",
                                  return_value=_FAKE_GCC), \
                mock.patch.object(subprocess, "run", run):
            ok, detail = native.build()
        self.assertTrue(ok, detail)
        self.assertEqual(len(run.calls), 4)
        # The four targets compile in parallel: compare as a set.
        expected = [[
            "/usr/bin/g++", "-O3", "-fPIC", "-shared", "-std=c++17",
            "-fno-semantic-interposition", str(src), "-o",
            str(self._tmp / f"{stem}.so")]
            for src, stem in native.BUILD_TARGETS]
        self.assertEqual(sorted(map(tuple, run.calls)),
                         sorted(map(tuple, expected)))
        for _src, stem in native.BUILD_TARGETS:
            self.assertTrue((self._tmp / f"{stem}.so").exists())

    def test_msvc_argv_shape_via_path(self):
        run = _fake_run_factory(self._tmp)
        with mock.patch.object(platform, "system", return_value="Windows"), \
                mock.patch.object(native, "detect_compiler",
                                  return_value=_FAKE_CL), \
                mock.patch.object(subprocess, "run", run):
            ok, detail = native.build()
        self.assertTrue(ok, detail)
        self.assertEqual(len(run.calls), 4)
        # _msvc_command always invokes bare "cl": a cl.exe on PATH is
        # assumed to be the same compiler detect_compiler found.
        expected = [[
            "cl", "/O2", "/LD", "/std:c++17", "/EHsc", "/nologo",
            str(src), "/link",
            "/OUT:" + str(self._tmp / f"{stem}.dll")]
            for src, stem in native.BUILD_TARGETS]
        self.assertEqual(sorted(map(tuple, run.calls)),
                         sorted(map(tuple, expected)))

    def test_msvc_argv_shape_via_vcvarsall(self):
        run = _fake_run_factory(self._tmp)
        with mock.patch.object(platform, "system", return_value="Windows"), \
                mock.patch.object(native, "detect_compiler",
                                  return_value=_FAKE_CL_VS), \
                mock.patch.object(subprocess, "run", run):
            ok, detail = native.build()
        self.assertTrue(ok, detail)
        self.assertEqual(len(run.calls), 4)
        expected = [[
            "cmd", "/c",
            'call "C:\\VS\\VC\\Auxiliary\\Build\\vcvarsall.bat" x64 '
            "&& cl /O2 /LD /std:c++17 /EHsc /nologo " + str(src) +
            " /link /OUT:" + str(self._tmp / f"{stem}.dll")]
            for src, stem in native.BUILD_TARGETS]
        self.assertEqual(sorted(map(tuple, run.calls)),
                         sorted(map(tuple, expected)))

    def test_build_failure_names_the_error(self):
        run = _fake_run_factory(self._tmp, failures={"vecsim": 1})
        with mock.patch.object(platform, "system", return_value="Linux"), \
                mock.patch.object(native, "detect_compiler",
                                  return_value=_FAKE_GCC), \
                mock.patch.object(subprocess, "run", run):
            ok, detail = native.build()
        self.assertFalse(ok)
        self.assertIn("boom", detail)

    def test_build_without_compiler(self):
        with mock.patch.object(native, "detect_compiler", return_value=None):
            ok, detail = native.build()
        self.assertFalse(ok)
        self.assertIn("no C++ compiler found", detail)


class ProvisionCompilerTest(unittest.TestCase):
    def test_non_windows_returns_honest_reason(self):
        with mock.patch.object(platform, "system", return_value="Linux"):
            path, attempts = native.provision_compiler()
        self.assertIsNone(path)
        self.assertTrue(any("Windows-only" in a for a in attempts))

    def test_windows_without_winget_records_both_routes_dead(self):
        def _run(argv, timeout):
            return 1, "nope"
        with mock.patch.object(platform, "system", return_value="Windows"), \
                mock.patch.object(shutil, "which",
                                  lambda name, **_kw: None), \
                mock.patch.object(native, "_find_msvc", return_value=None):
            path, attempts = native.provision_compiler(runner=_run)
        self.assertIsNone(path)
        self.assertTrue(any("winget is not available" in a
                            for a in attempts))

    def test_windows_msys2_route_returns_mingw_gcc(self):
        tmp = Path(tempfile.mkdtemp(prefix="nm-msys2-"))
        gcc = tmp / "ucrt64" / "bin" / "g++.exe"
        gcc.parent.mkdir(parents=True)
        gcc.touch()
        (tmp / "usr" / "bin").mkdir(parents=True)
        (tmp / "usr" / "bin" / "bash.exe").touch()

        def _run(argv, timeout):
            return 0, ""
        with mock.patch.object(platform, "system", return_value="Windows"), \
                mock.patch.object(shutil, "which",
                                  lambda name, **_kw: "winget"
                                  if name == "winget" else None), \
                mock.patch.object(native, "_msys2_roots",
                                  return_value=(tmp,)), \
                mock.patch.object(native, "_find_msvc", return_value=None):
            path, attempts = native.provision_compiler(runner=_run)
        self.assertEqual(path, str(gcc))
        shutil.rmtree(tmp)

    def test_windows_falls_through_to_build_tools(self):
        def _run(argv, timeout):
            # MSYS2 install fails; the Build Tools install succeeds.
            rc = 0 if "Microsoft.VisualStudio" in " ".join(argv) else 1
            return rc, "" if rc == 0 else "nope"
        with mock.patch.object(platform, "system", return_value="Windows"), \
                mock.patch.object(shutil, "which",
                                  lambda name, **_kw: "winget"
                                  if name == "winget" else None), \
                mock.patch.object(native, "_msys2_roots", return_value=()), \
                mock.patch.object(native, "_find_msvc",
                                  return_value=_FAKE_CL_VS):
            path, attempts = native.provision_compiler(runner=_run)
        # provision_compiler returns vcvarsall when one exists: that is
        # what a later build shells out through.
        self.assertEqual(path, _FAKE_CL_VS.vcvarsall)
        self.assertTrue(any("MSYS2" in a and "failed" in a
                            for a in attempts))


class EnsureBuiltTest(unittest.TestCase):
    """The self-provision contract: attempt once, record, never nag."""

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="nm-ensure-"))
        self._old_here = native._HERE
        native._HERE = self._tmp
        self._old_state = native._PROVISION_STATE_FILE
        native._PROVISION_STATE_FILE = self._tmp / "state.json"
        self._saved = dict(native._LIBS)
        for slot in native._LIBS:
            native._LIBS[slot] = native._NOT_GIVEN
        self._build_calls = 0
        self._build_result: tuple[bool, str] = (
            True, "built libvecsim.so with /usr/bin/g++ in 1.0s; "
            "built libmlptrain.so with /usr/bin/g++ in 1.0s; "
            "built libbpe.so with /usr/bin/g++ in 1.0s; "
            "built libmemextract.so with /usr/bin/g++ in 1.0s")
        self.addCleanup(self._restore)

    def _restore(self):
        native._HERE = self._old_here
        native._PROVISION_STATE_FILE = self._old_state
        native._LIBS.update(self._saved)
        shutil.rmtree(self._tmp)

    def _fake_build(self, *args, **kwargs):
        self._build_calls += 1
        if self._build_result[0]:
            # A successful build leaves the libraries behind, exactly
            # as the real build() does (ensure_built re-checks the disk).
            for stem in ("libvecsim", "libmlptrain", "libbpe",
                         "libmemextract"):
                (self._tmp / native._lib_name(stem)).touch()
        return self._build_result

    def test_missing_libs_plus_compiler_builds_once_and_records(self):
        with mock.patch.object(native, "detect_compiler",
                               return_value=_FAKE_GCC), \
                mock.patch.object(native, "build", self._fake_build):
            result = native.ensure_built()
        self.assertEqual(result["status"], "built")
        self.assertEqual(result["missing"], [])
        self.assertEqual(self._build_calls, 1)
        state = json.loads((self._tmp / "state.json").read_text())
        self.assertEqual(state["outcome"], "built")

    def test_failed_build_is_recorded_and_not_retried(self):
        self._build_result = (False, "boom: no such header")
        with mock.patch.object(native, "detect_compiler",
                               return_value=_FAKE_GCC), \
                mock.patch.object(native, "build", self._fake_build):
            first = native.ensure_built()
            second = native.ensure_built()
        self.assertEqual(first["status"], "build_failed")
        self.assertEqual(second["status"], "build_failed")
        self.assertEqual(self._build_calls, 1)
        self.assertIn("boom", first["detail"])

    def test_no_compiler_is_an_honest_record(self):
        with mock.patch.object(platform, "system", return_value="Linux"), \
                mock.patch.object(native, "detect_compiler",
                                  return_value=None), \
                mock.patch.object(native, "build", self._fake_build):
            first = native.ensure_built()
            second = native.ensure_built()
        self.assertEqual(first["status"], "no_compiler")
        self.assertEqual(self._build_calls, 0)
        self.assertEqual(second["status"], "no_compiler")
        state = json.loads((self._tmp / "state.json").read_text())
        self.assertEqual(state["outcome"], "no_compiler")

    def test_changed_sources_retry_a_failed_build(self):
        self._build_result = (False, "boom")
        with mock.patch.object(native, "detect_compiler",
                               return_value=_FAKE_GCC), \
                mock.patch.object(native, "build", self._fake_build):
            native.ensure_built()
            with mock.patch.object(native, "_source_hash",
                                   return_value="changed-sources"):
                native.ensure_built()
        self.assertEqual(self._build_calls, 2)

    def test_stale_record_retries_after_a_day(self):
        self._build_result = (False, "boom")
        with mock.patch.object(native, "detect_compiler",
                               return_value=_FAKE_GCC), \
                mock.patch.object(native, "build", self._fake_build):
            native.ensure_built()
            state = json.loads((self._tmp / "state.json").read_text())
            state["attempted_at"] = time.time() - 25 * 3600
            (self._tmp / "state.json").write_text(json.dumps(state))
            native.ensure_built()
        self.assertEqual(self._build_calls, 2)

    def test_already_built_does_not_call_build(self):
        for stem in ("libvecsim", "libmlptrain", "libbpe", "libmemextract"):
            (self._tmp / native._lib_name(stem)).touch()
        with mock.patch.object(native, "detect_compiler",
                               return_value=_FAKE_GCC), \
                mock.patch.object(native, "build", self._fake_build):
            result = native.ensure_built()
        self.assertEqual(result["status"], "built")
        self.assertEqual(self._build_calls, 0)


class LogLineTest(unittest.TestCase):
    """The startup log never hands the owner a command."""

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="nm-logline-"))
        self._old_here = native._HERE
        native._HERE = self._tmp
        self._old_state = native._PROVISION_STATE_FILE
        native._PROVISION_STATE_FILE = self._tmp / "state.json"
        self._saved = dict(native._LIBS)
        for slot in native._LIBS:
            native._LIBS[slot] = native._NOT_GIVEN
        self.addCleanup(self._restore)

    def _restore(self):
        native._HERE = self._old_here
        native._PROVISION_STATE_FILE = self._old_state
        native._LIBS.update(self._saved)
        shutil.rmtree(self._tmp)

    def _probe_messages(self):
        with self.assertLogs("nomorals.native", level=logging.INFO) as logs:
            native.available()
        return logs.output

    def test_no_record_says_devon_builds_at_startup(self):
        messages = self._probe_messages()
        self.assertTrue(any("not built yet — Devon builds it at startup" in m
                            for m in messages))
        self.assertFalse(any("nm native --build" in m for m in messages))

    def test_failed_build_names_the_reason(self):
        (self._tmp / "state.json").write_text(json.dumps({
            "source_hash": native._source_hash(),
            "outcome": "build_failed",
            "detail": "boom: no such header",
            "attempted_at": time.time()}))
        messages = self._probe_messages()
        self.assertTrue(any("the last build attempt failed: boom" in m
                            for m in messages))
        self.assertFalse(any("nm native --build" in m for m in messages))

    def test_no_compiler_names_the_gap(self):
        (self._tmp / "state.json").write_text(json.dumps({
            "source_hash": native._source_hash(),
            "outcome": "no_compiler",
            "detail": "no compiler found on PATH",
            "attempted_at": time.time()}))
        messages = self._probe_messages()
        self.assertTrue(any("no C++ compiler on this machine" in m
                            for m in messages))
        self.assertFalse(any("nm native --build" in m for m in messages))

    def test_built_record_but_lib_missing_says_so(self):
        (self._tmp / "state.json").write_text(json.dumps({
            "source_hash": native._source_hash(),
            "outcome": "built",
            "detail": "built all four",
            "attempted_at": time.time()}))
        messages = self._probe_messages()
        self.assertTrue(any("built at the last attempt but this library "
                            "is still missing" in m for m in messages))
        self.assertFalse(any("nm native --build" in m for m in messages))

    def test_info_carries_the_provision_block(self):
        info = native.info()
        self.assertEqual(info["provision"], native.provision_status())

    def test_doctor_text_never_names_the_manual_command(self):
        from nomorals import compat
        from nomorals.compat import native_section
        # feature_report() leaves "native" None by design (compat is
        # L0 and must not import the L2 native package); the `nm
        # doctor` command fills it in from native.info(), as here.
        report = compat.feature_report()
        report["native"] = native_section(native.info())
        text = compat.report_as_text(report)
        self.assertNotIn("nm native --build", text)
        self.assertIn("Devon builds this at startup", text)


class RuntimeHookTest(unittest.TestCase):
    """The boot hook: best-effort, never bricks the boot."""

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="nm-hook-"))
        self.addCleanup(lambda: shutil.rmtree(self._tmp))

    def _mixin(self):
        from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin
        return RuntimeProvisionMixin()

    def test_hook_returns_early_when_libs_exist(self):
        paths = {}
        for stem in ("vecsim", "mlptrain", "bpe", "memextract"):
            path = self._tmp / f"lib{stem}.so"
            path.touch()
            paths[stem] = path
        calls = []
        with mock.patch.object(native, "lib_path",
                               return_value=paths["vecsim"]), \
                mock.patch.object(native, "mlp_lib_path",
                                  return_value=paths["mlptrain"]), \
                mock.patch.object(native, "bpe_lib_path",
                                  return_value=paths["bpe"]), \
                mock.patch.object(native, "mem_lib_path",
                                  return_value=paths["memextract"]), \
                mock.patch.object(native, "ensure_built",
                                  side_effect=lambda **_kw: calls.append(1)):
            result = self._mixin()._ensure_native_built()
        self.assertEqual(result["status"], "built")
        self.assertEqual(calls, [])

    def test_hook_builds_when_libs_missing(self):
        with mock.patch.object(native, "lib_path",
                               return_value=self._tmp / "libvecsim.so"), \
                mock.patch.object(native, "mlp_lib_path",
                                  return_value=self._tmp / "libmlptrain.so"), \
                mock.patch.object(native, "bpe_lib_path",
                                  return_value=self._tmp / "libbpe.so"), \
                mock.patch.object(native, "mem_lib_path",
                                  return_value=self._tmp / "libmemextract.so"), \
                mock.patch.object(
                    native, "ensure_built",
                    return_value={"status": "built", "detail": "ok",
                                  "missing": []}):
            result = self._mixin()._ensure_native_built()
        self.assertEqual(result["status"], "built")

    def test_hook_failure_is_a_status_not_an_exception(self):
        with mock.patch.object(native, "lib_path",
                               return_value=self._tmp / "libvecsim.so"), \
                mock.patch.object(native, "mlp_lib_path",
                                  return_value=self._tmp / "libmlptrain.so"), \
                mock.patch.object(native, "bpe_lib_path",
                                  return_value=self._tmp / "libbpe.so"), \
                mock.patch.object(native, "mem_lib_path",
                                  return_value=self._tmp / "libmemextract.so"), \
                mock.patch.object(native, "ensure_built",
                                  side_effect=RuntimeError("boom")):
            result = self._mixin()._ensure_native_built()
        self.assertEqual(result["status"], "skipped")

    def test_start_native_build_spawns_a_daemon_thread(self):
        from nomorals.agents.partner.runtime import PartnerRuntime
        done = threading.Event()
        calls: list[int] = []

        def _ensure():
            calls.append(1)
            done.set()
            return {"status": "built", "detail": "ok", "missing": []}

        obj = SimpleNamespace(_ensure_native_built=_ensure)
        PartnerRuntime._start_native_build(obj)
        self.assertTrue(done.wait(timeout=5))
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()
