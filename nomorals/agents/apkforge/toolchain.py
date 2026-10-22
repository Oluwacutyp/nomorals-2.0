"""ApkForge toolchain — detect, self-provision, and honestly report
the Java-side build chain (A2).

A1 is pure Python (androguard + APKiD) and runs everywhere.  A2's
edit / build / sign / splits ops need real binaries: a JDK, apktool,
jadx, bundletool, APKEditor and the Android build-tools (zipalign,
apksigner, aapt2).  Following Devon's self-provisioning pattern
(desk Pass 8), what's missing is fetched — pinned version, pinned
sha256, from the publisher's own release channel — into the class
tool dir (``<apkforge root>/tools/``), verified byte-for-byte, and
then *executed*: a tool that cannot run on this host (the classic
Termux aapt2 exec-format pinch) is reported broken in the capability
matrix with the probe error attached, never as available.

Every invocation goes through :func:`run_tool`, which pins
JAVA_HOME/PATH to the provisioned JDK, enforces a hard timeout with
a process-group kill, and never puts a secret in argv or in its
error text (signing passwords travel by ``env:`` reference only).

Paired-host routing: this build has no host-routing organ to reuse,
so the matrix states this host's capabilities and says routing is
unavailable rather than pretending an op will run elsewhere.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

__all__ = [
    "ToolchainError",
    "PINS",
    "androguard_version",
    "apkid_version",
    "tools_dir",
    "tool_paths",
    "run_tool",
    "provision",
    "probe_capabilities",
    "toolchain_state",
]

# ── pinned artifacts (publisher release channels) ────────────────────
# sha256 provenance: JDK + jadx + APKEditor digests are the
# publishers' own published checksums; apktool / bundletool /
# build-tools digests were computed from the artifacts served by the
# publishers' release channels at pin time (2026-10-22) and are
# re-verified on every provision — a mismatch refuses the install.
_JDK_VER = "21.0.5+11"
_JDK_FILE = "OpenJDK21U-jdk_{arch}_linux_hotspot_21.0.5_11.tar.gz"
_JDK_URL = ("https://github.com/adoptium/temurin21-binaries/releases/"
            "download/jdk-21.0.5%2B11/" + _JDK_FILE)
_JDK_SHA = {
    "x64": "3c654d98404c073b8a7e66bffb27f4ae3e7ede47d13284c132d40a83144bfd8c",
    "aarch64": "6482639ed9fd22aa2e704cc366848b1b3e1586d2bf1213869c43e80bca58fe5c",
}

PINS: dict[str, dict[str, Any]] = {
    "jdk": {"version": _JDK_VER, "kind": "jdk"},
    "apktool": {
        "version": "2.10.0", "kind": "jar",
        "url": ("https://github.com/iBotPeaches/Apktool/releases/"
                "download/v2.10.0/apktool_2.10.0.jar"),
        "sha256": "c0350abbab5314248dfe2ee0c907def4edd14f6faef1f5d372d3d4abd28f0431",
        "file": "apktool.jar",
    },
    "jadx": {
        "version": "1.5.3", "kind": "zip",
        "url": ("https://github.com/skylot/jadx/releases/download/"
                "v1.5.3/jadx-1.5.3.zip"),
        "sha256": "8280f3799c0273fe797a2bcd90258c943e451fd195f13d05400de5e6451d15ec",
        "dir": "jadx",
    },
    "bundletool": {
        "version": "1.18.1", "kind": "jar",
        "url": ("https://github.com/google/bundletool/releases/"
                "download/1.18.1/bundletool-all-1.18.1.jar"),
        "sha256": "675786493983787ffa11550bdb7c0715679a44e1643f3ff980a529e9c822595c",
        "file": "bundletool.jar",
    },
    "apkeditor": {
        "version": "1.4.9", "kind": "jar",
        "url": ("https://github.com/REAndroid/APKEditor/releases/"
                "download/V1.4.9/APKEditor-1.4.9.jar"),
        "sha256": "a9cd40df818845456be6d696de6110c89edf4b0a0580cb83438ed6b25a366e67",
        "file": "apkeditor.jar",
    },
    "build-tools": {
        "version": "35.0.0", "kind": "zip",
        "url": ("https://dl.google.com/android/repository/"
                "build-tools_r35_linux.zip"),
        "sha256": "bd3a4966912eb8b30ed0d00b0cda6b6543b949d5ffe00bea54c04c81e1561d88",
        "dir": "build-tools",
    },
}

_STATE_FILE = "toolchain.json"
_MARKER = ".pinned-sha256"


class ToolchainError(RuntimeError):
    """A designed toolchain failure — the message names the step."""


# ── python engines (A1) ──────────────────────────────────────────────

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


# ── paths ────────────────────────────────────────────────────────────

def _root(base: str = "") -> Path:
    from .project import apkforge_dir

    return apkforge_dir(None, base)


def tools_dir(base: str = "", *, root: Any = None) -> Path:
    tdir = (Path(root) if root is not None else _root(base)) / "tools"
    tdir.mkdir(parents=True, exist_ok=True)
    return tdir


def _arch() -> str:
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "x64"
    if m in ("aarch64", "arm64"):
        return "aarch64"
    return m


def _jdk_artifact() -> tuple[str, str]:
    arch = _arch()
    sha = _JDK_SHA.get(arch)
    if not sha:
        raise ToolchainError(
            f"no pinned Temurin JDK for host arch {platform.machine()!r} "
            f"(pinned: {sorted(_JDK_SHA)}) — install a system JDK or "
            "extend the pin table")
    return _JDK_URL.format(arch=arch), sha


def tool_paths(base: str = "", *, root: Any = None) -> dict[str, Any]:
    """Resolve every binary: system PATH first, provisioned second.

    Returns {name: {"path": str, "source": "system"|"provisioned"|""}}.
    Jar tools resolve to their jar path (run via the resolved java)."""
    tdir = tools_dir(base, root=root)
    prov = {
        "java": tdir / "jdk" / "bin" / "java",
        "keytool": tdir / "jdk" / "bin" / "keytool",
        "apktool": tdir / "apktool.jar",
        "jadx": tdir / "jadx" / "bin" / "jadx",
        "bundletool": tdir / "bundletool.jar",
        "apkeditor": tdir / "apkeditor.jar",
        "zipalign": tdir / "build-tools" / "zipalign",
        "apksigner": tdir / "build-tools" / "apksigner",
        "aapt2": tdir / "build-tools" / "aapt2",
    }
    out: dict[str, Any] = {}
    for name, ppath in prov.items():
        sys_path = shutil.which(name)
        if sys_path:
            out[name] = {"path": sys_path, "source": "system"}
        elif ppath.exists():
            out[name] = {"path": str(ppath), "source": "provisioned"}
        else:
            out[name] = {"path": "", "source": ""}
    return out


def _argv_for(paths: dict[str, Any], name: str,
              args: list[str]) -> list[str]:
    """Build the argv for a tool, wrapping jars in the resolved java."""
    info = paths[name]
    if not info["path"]:
        raise ToolchainError(f"{name} is not available on this host "
                             "(not on PATH, not provisioned)")
    if name in ("apktool", "bundletool", "apkeditor"):
        java = paths["java"]["path"]
        if not java:
            raise ToolchainError(f"{name} needs a java runtime — none "
                                 "on PATH and no JDK provisioned")
        return [java, "-jar", info["path"], *args]
    return [info["path"], *args]


def _tool_env(paths: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    java = paths.get("java", {}).get("path", "")
    if java:
        jdk_home = str(Path(java).resolve().parent.parent)
        env["JAVA_HOME"] = jdk_home
        env["PATH"] = (str(Path(java).parent) + os.pathsep
                       + env.get("PATH", ""))
    bt = paths.get("zipalign", {}).get("path", "")
    if bt:
        env["PATH"] = (str(Path(bt).parent) + os.pathsep
                       + env.get("PATH", ""))
    return env


def run_tool(base: str, name: str, args: list[str], *,
             timeout: int = 300, extra_env: dict[str, str] | None = None,
             cwd: str = "", root: Any = None) -> dict[str, Any]:
    """Run one toolchain binary with a hard timeout + group kill.

    Returns {rc, stdout, stderr}.  Raises ToolchainError on timeout;
    a nonzero rc is data, not an exception — callers name the
    failing step."""
    paths = tool_paths(base, root=root)
    argv = _argv_for(paths, name, args)
    env = _tool_env(paths)
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=cwd or None, env=env, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, err = proc.communicate()
        raise ToolchainError(
            f"{name} {' '.join(args[:3])} timed out after {timeout}s "
            "and was killed") from None
    return {"rc": proc.returncode, "stdout": out or "",
            "stderr": err or ""}


def _version_of(base: str, name: str, args: list[str], root: Any = None,
                timeout: int = 60) -> dict[str, Any]:
    """Execute a binary's version flag — the probe is the run itself."""
    try:
        res = run_tool(base, name, args, timeout=timeout, root=root)
    except ToolchainError as exc:
        return {"ok": False, "version": "", "error": str(exc)}
    text = (res["stdout"] + res["stderr"]).strip()
    first = text.splitlines()[0].strip() if text else ""
    return {"ok": res["rc"] == 0, "version": first[:120],
            "error": "" if res["rc"] == 0 else
            (text.splitlines()[-1][:200] if text else f"rc={res['rc']}")}


# ── provisioning ─────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for info in zf.infolist():
        target = (dest / info.filename).resolve()
        if target != dest and dest not in target.parents:
            raise ToolchainError(
                f"archive entry {info.filename!r} escapes its target "
                "dir — refusing the install (zip-slip guard)")
        if info.filename.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)


def _install_zip(data_path: Path, dest: Path) -> None:
    """Extract a zip; a single top-level dir is stripped into dest."""
    tmp = dest.parent / (dest.name + ".extracting")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(data_path) as zf:
        _safe_extract_zip(zf, tmp)
    children = [c for c in tmp.iterdir()]
    src = children[0] if len(children) == 1 and children[0].is_dir() \
        else tmp
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(str(src), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)


def _install_tar_gz(data_path: Path, dest: Path) -> None:
    tmp = dest.parent / (dest.name + ".extracting")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    root = tmp.resolve()
    with tarfile.open(data_path, "r:gz") as tf:
        for member in tf.getmembers():
            target = (tmp / member.name).resolve()
            if target != root and root not in target.parents:
                raise ToolchainError(
                    f"tar entry {member.name!r} escapes its target "
                    "dir — refusing the install")
        tf.extractall(tmp)  # noqa: S202 — members path-checked above
    children = [c for c in tmp.iterdir()]
    src = children[0] if len(children) == 1 and children[0].is_dir() \
        else tmp
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(str(src), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)


def _installed_sha(dest: Path) -> str:
    marker = dest / _MARKER if dest.is_dir() else dest.parent / (
        dest.name + _MARKER)
    try:
        return marker.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _mark_installed(dest: Path, sha: str) -> None:
    marker = dest / _MARKER if dest.is_dir() else dest.parent / (
        dest.name + _MARKER)
    marker.write_text(sha + "\n", encoding="utf-8")


def toolchain_state(base: str = "", *, root: Any = None) -> dict[str, Any]:
    path = tools_dir(base, root=root) / _STATE_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(base: str, state: dict[str, Any], root: Any = None) -> None:
    path = tools_dir(base, root=root) / _STATE_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(path)


_SMOKE_MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="devon.apkforge.smoke" android:versionCode="1"
    android:versionName="1.0">
    <uses-sdk android:minSdkVersion="24" android:targetSdkVersion="34"/>
    <application android:label="Smoke"/>
</manifest>
"""

_SMOKE_YML = """version: 2.10.0
apkFileName: smoke.apk
isFrameworkApk: false
usesFramework:
  ids:
  - 1
sdkInfo:
  minSdkVersion: 24
  targetSdkVersion: 34
packageInfo:
  forcedPackageId: 127
versionInfo:
  versionCode: 1
  versionName: '1.0'
"""


def smoke_test(base: str, *, root: Any = None) -> dict[str, Any]:
    """The ground-truth probe: apktool builds a minimal project.

    This executes apktool's bundled aapt2 for real — the exact
    operation that fails on hosts where aapt2 cannot run — so the
    capability matrix rests on an observed build, not a guess."""
    tdir = tools_dir(base, root=root)
    work = tdir / ".smoke"
    shutil.rmtree(work, ignore_errors=True)
    proj = work / "proj"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "AndroidManifest.xml").write_text(_SMOKE_MANIFEST,
                                              encoding="utf-8")
    (proj / "apktool.yml").write_text(_SMOKE_YML, encoding="utf-8")
    try:
        res = run_tool(base, "apktool",
                       ["b", str(proj), "-o", str(work / "smoke.apk")],
                       timeout=300, root=root)
        ok = res["rc"] == 0 and (work / "smoke.apk").exists()
        detail = (res["stdout"] + res["stderr"]).strip().splitlines()
        return {"ok": ok, "at": time.time(),
                "detail": detail[-1][:200] if detail else ""}
    except ToolchainError as exc:
        return {"ok": False, "at": time.time(), "detail": str(exc)[:200]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def provision(base: str = "", *, force: bool = False, root: Any = None) -> dict[str, Any]:
    """Provision every missing pinned artifact, verify, probe.

    Idempotent: an artifact whose installed marker matches the pin
    is left alone.  A download whose sha256 does not match the pin
    is deleted and the provision fails closed, naming the artifact.
    A pre-downloaded artifact cache (``APKFORGE_TOOL_CACHE`` dir,
    files named per the pin table) may seed the downloads — the
    sha256 check runs either way."""
    from ...core.http import download

    tdir = tools_dir(base, root=root)
    dl_dir = tdir / ".downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    cache = os.environ.get("APKFORGE_TOOL_CACHE", "").strip()
    state = toolchain_state(base, root=root)
    installed: dict[str, Any] = dict(state.get("artifacts") or {})
    report: dict[str, Any] = {"installed": [], "present": [],
                              "failed": []}

    def _obtain(url: str, fname: str) -> Path:
        dest = dl_dir / fname
        if cache:
            cached = Path(cache) / fname
            if cached.exists():
                shutil.copyfile(cached, dest)
                return dest
        return Path(download(url, dest))

    plan: list[tuple[str, str, str, str, Path, str]] = []
    jdk_url, jdk_sha = _jdk_artifact()
    plan.append(("jdk", jdk_url, jdk_sha, "jdk.tar.gz",
                 tdir / "jdk", "tar.gz"))
    for name, pin in PINS.items():
        if name == "jdk":
            continue
        dest = tdir / (pin.get("file") or pin["dir"])
        plan.append((name, pin["url"], pin["sha256"],
                     pin.get("file") or (name + ".zip"), dest,
                     pin["kind"]))

    for name, url, sha, fname, dest, kind in plan:
        if not force and dest.exists() and \
                _installed_sha(dest) == sha:
            report["present"].append(name)
            installed.setdefault(name, {
                "version": PINS[name]["version"], "sha256": sha,
                "source": "provisioned", "path": str(dest)})
            continue
        raw = _obtain(url, fname)
        got = _sha256(raw)
        if got != sha:
            raw.unlink(missing_ok=True)
            raise ToolchainError(
                f"{name}: downloaded artifact sha256 {got[:16]}… does "
                f"not match the pin {sha[:16]}… — install refused, "
                "file deleted")
        if kind == "jar":
            shutil.copyfile(raw, dest)
        elif kind == "zip":
            _install_zip(raw, dest)
        elif kind == "tar.gz":
            _install_tar_gz(raw, dest)
        if dest.is_dir():
            for exe in dest.rglob("*"):
                if exe.is_file() and (exe.suffix in ("", ".sh")
                                      or exe.parent.name == "bin"):
                    try:
                        exe.chmod(exe.stat().st_mode | 0o111)
                    except OSError:
                        pass
        _mark_installed(dest, sha)
        installed[name] = {"version": PINS[name]["version"],
                           "sha256": sha, "source": "provisioned",
                           "path": str(dest)}
        report["installed"].append(name)

    state.update({"artifacts": installed, "arch": _arch(),
                  "provisioned_at": time.time()})
    state["smoke"] = smoke_test(base, root=root)
    _save_state(base, state, root=root)
    report["smoke"] = state["smoke"]
    report["state"] = state
    return report


# ── capability probe ─────────────────────────────────────────────────

_VERSION_ARGS = {
    "java": ["-version"],
    "keytool": ["-help"],
    "apktool": ["--version"],
    "jadx": ["--version"],
    "bundletool": ["version"],
    "apkeditor": ["-version"],
    "zipalign": [],
    "apksigner": ["version"],
    "aapt2": ["version"],
}

_PROBE_CACHE: dict[str, dict[str, Any]] = {}


def probe_capabilities(base: str = "", *, root: Any = None) -> dict[str, Any]:
    """What this host can actually do for the APK class, right now.

    Same shape as A1 (host / python_engines / binaries /
    capabilities / notes); A2 fills the edit/build/sign/splits rows
    from executed version probes + the recorded apktool smoke
    build, and marks unprovisioned rows with the exact remedy."""
    ag = androguard_version()
    ak = apkid_version()
    paths = tool_paths(base, root=root)
    binaries = {name: info["path"] for name, info in paths.items()}
    dissect = (
        "full" if ag else
        "unavailable — install the apk extra: pip install "
        "'nomorals[apk]' (androguard is the dissect engine)")

    cache_key = str(tools_dir(base, root=root))
    probes = _PROBE_CACHE.get(cache_key)
    if probes is None:
        probes = {}
        for name, args in _VERSION_ARGS.items():
            if not paths[name]["path"]:
                probes[name] = {"ok": False, "version": "",
                                "error": "not installed"}
            elif name == "zipalign":
                try:
                    res = run_tool(base, name, [], timeout=30, root=root)
                    probes[name] = {
                        "ok": res["rc"] in (0, 1, 2),
                        "version": "present (usage banner on no-args)",
                        "error": ""}
                except ToolchainError as exc:
                    probes[name] = {"ok": False, "version": "",
                                    "error": str(exc)}
            elif name == "apkeditor":
                try:
                    res = run_tool(base, name, args, timeout=60, root=root)
                    text = (res["stdout"] + res["stderr"]).strip()
                    probes[name] = {
                        "ok": bool(text),
                        "version": text.splitlines()[0][:120]
                        if text else "", "error": ""}
                except ToolchainError as exc:
                    probes[name] = {"ok": False, "version": "",
                                    "error": str(exc)}
            else:
                probes[name] = _version_of(base, name, args, root)
        _PROBE_CACHE[cache_key] = probes

    state = toolchain_state(base, root=root)
    smoke = state.get("smoke") or {}
    java_ok = probes["java"]["ok"]
    apktool_ok = probes["apktool"]["ok"] and java_ok
    provisioned = bool(state.get("artifacts"))
    not_prov = ("unavailable — Java toolchain not provisioned on "
                "this host (run: apkforge toolchain provision)")

    caps: dict[str, Any] = {
        "intake": "full",
        "dissect": dissect,
        "teach": "full" if ag else "unavailable — dissect engine missing",
        "diff": "full" if ag else "unavailable — dissect engine missing",
        "edit": "full" if apktool_ok else (
            not_prov if not provisioned else
            "unavailable — apktool/java probe failed: "
            + (probes["apktool"]["error"] or probes["java"]["error"])),
        "build": "full" if smoke.get("ok") else (
            not_prov if not provisioned else
            "unavailable — apktool smoke build failed on this host: "
            + str(smoke.get("detail", ""))),
        "sign": "full" if (probes["zipalign"]["ok"]
                           and probes["apksigner"]["ok"]
                           and probes["keytool"]["ok"]) else (
            not_prov if not provisioned else
            "unavailable — zipalign/apksigner/keytool probe failed"),
        "splits_merge": "full" if (probes["apkeditor"]["ok"]
                                   or probes["bundletool"]["ok"]) else (
            not_prov if not provisioned else
            "unavailable — APKEditor/bundletool probe failed"),
        "jadx": "full" if probes["jadx"]["ok"] else (
            "unavailable — jadx not provisioned"),
        "install_test": "out of scope — no device install-test lab "
                        "organ (design §5)",
    }
    return {
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
            "arch": _arch(),
        },
        "python_engines": {"androguard": ag, "apkid": ak},
        "binaries": binaries,
        "binary_sources": {n: i["source"] for n, i in paths.items()},
        "probes": probes,
        "toolchain": {
            "provisioned": provisioned,
            "artifacts": state.get("artifacts") or {},
            "provisioned_at": state.get("provisioned_at"),
            "smoke_apktool_build": smoke,
            "routing": "unavailable — no paired-host execution organ "
                       "in this build; capabilities stated are this "
                       "host's own (design §3.2 routing is honest "
                       "reporting until such an organ exists)",
        },
        "capabilities": caps,
        "notes": [
            "dissect/diff/teach are pure Python (androguard) — full "
            "on every profile, Termux included; no Java needed",
            "edit/build/sign/splits rows are EXECUTED probes (version "
            "runs + the recorded apktool smoke build), never claims",
        ],
    }
