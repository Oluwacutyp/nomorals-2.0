"""Pass tail-7 pins: security/dnsleak failures stay silent, honest, and
unrecorded.

Target: nomorals/security/dnsleak.py — 0 converts / 11 exclusions.
Every site below maps a failure to a genuine degraded value (the
documented miss/unknown value of that function) and the stack proceeds
honestly — no leak verdict is ever fabricated from a failure, and no
failure claims a success it did not have.

NOT-recording: the security package has NO failures ledger and the
target contains no record call at all, so NOT-recording is pinned
against the module's only observability surface, ``_log``: each trigger
below runs with ``_log`` replaced by a Mock and asserts it received no
calls. None of the 11 counted sites logs.

No bare ``except: pass`` anywhere in this file — the scanner counts
test files too; the tests below need no except blocks at all (stubs
raise, the code under test catches).
"""

import sys
from unittest import mock

from nomorals.security import dnsleak as dl


def _silent(log):
    assert log.method_calls == []


def _with_silent_log(fn):
    with mock.patch.object(dl, "_log") as log:
        out = fn()
    _silent(log)
    return out


def test_081_public_ip_failure_is_empty_string():
    calls = []

    def boom_text(url, timeout=10):
        calls.append(url)
        raise RuntimeError("net down")

    def run():
        with mock.patch.object(dl, "_http_text", boom_text):
            return dl._public_ip()

    assert _with_silent_log(run) == ""
    # both ipify URLs tried (failover, not first-failure give-up)
    assert len(calls) == 2


def test_091_public_ipv6_failure_is_empty_string():
    def run():
        with mock.patch.object(dl, "_http_text",
                               side_effect=RuntimeError("net down")):
            return dl._public_ipv6()

    assert _with_silent_log(run) == ""


def test_119_resolvectl_failure_falls_through():
    def which(name):
        return "/usr/bin/resolvectl" if name == "resolvectl" else None

    def run():
        with mock.patch.object(dl.shutil, "which", which), \
             mock.patch.object(dl.subprocess, "run",
                               side_effect=RuntimeError("resolvectl boom")), \
             mock.patch.object(dl, "_resolvers_resolv_conf",
                               return_value=["9.9.9.9"]):
            return dl._resolvers_linux()

    assert _with_silent_log(run) == ["9.9.9.9"]


def test_130_nmcli_failure_falls_through():
    def which(name):
        return "/usr/bin/nmcli" if name == "nmcli" else None

    def run():
        with mock.patch.object(dl.shutil, "which", which), \
             mock.patch.object(dl.subprocess, "run",
                               side_effect=RuntimeError("nmcli boom")), \
             mock.patch.object(dl, "_resolvers_resolv_conf",
                               return_value=["1.1.1.1"]):
            return dl._resolvers_linux()

    assert _with_silent_log(run) == ["1.1.1.1"]


def test_149_resolv_conf_oserror_is_empty_list():
    def run():
        with mock.patch("builtins.open", side_effect=OSError("no file")):
            return dl._resolvers_resolv_conf()

    assert _with_silent_log(run) == []


def test_162_macos_scutil_failure_is_empty_list():
    def run():
        with mock.patch.object(dl.subprocess, "run",
                               side_effect=RuntimeError("scutil boom")):
            return dl._resolvers_macos()

    assert _with_silent_log(run) == []


class _FakeKey:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeWinregValuesRaise:
    """One interface; every value read raises OSError (site 183)."""

    HKEY_LOCAL_MACHINE = "HKLM"

    def __init__(self):
        self.qve_calls = []

    def OpenKey(self, root, path):
        return _FakeKey()

    def QueryInfoKey(self, key):
        return (1, 0, 0)

    def EnumKey(self, key, i):
        return "iface0"

    def QueryValueEx(self, sub, val):
        self.qve_calls.append(val)
        raise OSError("no such value")


def test_183_winreg_value_failure_skips_value():
    fake = _FakeWinregValuesRaise()

    def run():
        with mock.patch.dict(sys.modules, {"winreg": fake}), \
             mock.patch.object(dl.subprocess, "run",
                               side_effect=RuntimeError("ipconfig boom")):
            return dl._resolvers_windows()

    assert _with_silent_log(run) == []
    assert fake.qve_calls == ["NameServer", "DhcpNameServer"]


class _BoomOpenKey:
    """OpenKey itself raises (site 187's outer handler)."""

    HKEY_LOCAL_MACHINE = "HKLM"

    def OpenKey(self, root, path):
        raise OSError("cannot open key")

    def QueryInfoKey(self, key):
        return (0, 0, 0)

    def EnumKey(self, key, i):
        raise OSError("none")

    def QueryValueEx(self, sub, val):
        raise OSError("none")


def test_187_winreg_open_failure_falls_to_ipconfig():
    def run():
        with mock.patch.dict(sys.modules, {"winreg": _BoomOpenKey()}), \
             mock.patch.object(dl.subprocess, "run",
                               side_effect=RuntimeError("ipconfig boom")):
            return dl._resolvers_windows()

    assert _with_silent_log(run) == []


class _EmptyWinreg:
    """Zero interfaces (site 197's ipconfig branch)."""

    HKEY_LOCAL_MACHINE = "HKLM"

    def OpenKey(self, root, path):
        return _FakeKey()

    def QueryInfoKey(self, key):
        return (0, 0, 0)

    def EnumKey(self, key, i):
        raise OSError("none")

    def QueryValueEx(self, sub, val):
        raise OSError("none")


def test_197_ipconfig_failure_is_empty_list():
    def run():
        with mock.patch.dict(sys.modules, {"winreg": _EmptyWinreg()}), \
             mock.patch.object(dl.subprocess, "run",
                               side_effect=RuntimeError("ipconfig boom")):
            return dl._resolvers_windows()

    assert _with_silent_log(run) == []


def test_212_configured_resolvers_failure_is_empty_list():
    def run():
        with mock.patch.object(dl.platform, "system", return_value="Linux"), \
             mock.patch.object(dl, "_resolvers_linux",
                               side_effect=RuntimeError("boom")):
            return dl.configured_resolvers()

    assert _with_silent_log(run) == []


def test_370_resolve_one_oserror_is_none():
    def run():
        with mock.patch.object(dl.socket, "getaddrinfo",
                               side_effect=OSError("nxdomain")):
            return dl._resolve_one("1.test.bash.ws")

    assert _with_silent_log(run) is None
