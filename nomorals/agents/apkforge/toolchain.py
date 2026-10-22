"""ApkForge toolchain probe — the honest host capability matrix.

A1 is pure Python (androguard + APKiD), so dissect/diff/teach run at
full capability on every Devon profile including Termux.  Edit /
build / sign arrive in A2 behind the Java toolchain (apktool, jadx,
bundletool, build-tools); the probe reports those as **A2 — not yet
built**, never as fake-available, and reports the Java binaries it can
already see so A2 provisioning starts from facts.
"""

from __future__ import annotations

import platform
import shutil
import sys
from typing import Any

__all__ = ["probe_capabilities", "androguard_version", "apkid_version"]

_BINARIES = ("java", "apktool", "jadx", "bundletool", "zipalign",
             "apksigner", "keytool", "adb")


def androguard_version() -> str | None:
    try:
        import androguard

        return str(getattr(androguard, "__version__", "unknown"))
    except Exception:  # noqa: BLE001 — absence is a reportable fact
        return None


def apkid_version() -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version("apkid")
    except Exception:  # noqa: BLE001
        return None


def probe_capabilities() -> dict[str, Any]:
    """What this host can actually do for the APK class, right now."""
    ag = androguard_version()
    ak = apkid_version()
    binaries = {name: (shutil.which(name) or "") for name in _BINARIES}
    dissect = (
        "full" if ag else
        "unavailable — install the apk extra: pip install 'nomorals[apk]' "
        "(androguard is the dissect engine)")
    caps: dict[str, Any] = {
        "intake": "full",
        "dissect": dissect,
        "teach": "full" if ag else "unavailable — dissect engine missing",
        "diff": "full" if ag else "unavailable — dissect engine missing",
        "edit": "A2 — not yet built",
        "build": "A2 — not yet built",
        "sign": "A2 — not yet built",
        "splits_merge": "A2 — not yet built",
        "install_test": "out of scope — no device install-test lab organ (design §5)",
    }
    return {
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
        },
        "python_engines": {
            "androguard": ag,
            "apkid": ak,
        },
        "binaries": binaries,
        "capabilities": caps,
        "notes": [
            "dissect/diff/teach are pure Python (androguard) — full on "
            "every profile, Termux included; no Java needed in A1",
            "edit/build/sign land in A2 with toolchain provisioning; "
            "this probe never marks them available before they exist",
        ],
    }
