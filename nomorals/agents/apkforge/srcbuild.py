"""ApkForge build-from-source (A4) — the real builder.

A2's builder rebuilds an *existing* APK from its apktool decoding.
This module builds an APK from a **source project**, the way the
design's pipeline §1.4 lays it out — Gradle/AGP is a wrapper over
six standalone steps, and every step here is driven directly:

    aapt2 compile → aapt2 link → javac → d8 (or R8) → assemble
    → zipalign → apksigner sign → verify

Two routes, auto-detected from the project's own files:

* **manual** — a simple layout (AndroidManifest.xml + res/ +
  java/ [+ assets/, lib/<abi>/]): the six steps above, hand-driven
  against the provisioned platform's android.jar.  No Gradle, no
  network, deterministic.
* **gradle** — a project carrying gradle build files: the SDK is
  provisioned + licensed, ``sdk.dir`` is wired, and the project's
  own wrapper (or the provisioned Gradle) runs assembleDebug /
  assembleRelease / bundleRelease with an honest timeout budget.

This lives in its own module rather than build.py because build.py's
contract is the apktool round-trip of a landed project (decode →
rebuild → sign); a source build has a different input (a source
tree), different provenance (the tree's hash), and two routes.
Both builders emit the same build-card idiom and sign through the
one vault-keystore path in signing.py.

Every step fails fast: a nonzero exit raises SrcBuildError naming
the step and carrying the tool's real stderr tail.  There are no
silent fallbacks — a project that does not fit a route says why.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

from . import signing as signing_mod
from .project import sha256_file
from .toolchain import (ToolchainError, android_jar_path,
                        assemble_sdk_root, jvm_truststore, run_tool,
                        strip_proxy_env, tool_paths, tools_dir)

__all__ = [
    "SrcBuildError",
    "GRADLE_TIMEOUT",
    "detect_route",
    "scaffold_project",
    "build_source",
    "render_srcbuild_card",
    "proguard_to_tiny",
    "tree_hash",
]

GRADLE_TIMEOUT = 1800
_A = "{http://schemas.android.com/apk/res/android}"


class SrcBuildError(RuntimeError):
    """A designed build-from-source failure — names the step."""


# ── small helpers ────────────────────────────────────────────────────

def _tail(res: dict[str, Any], limit: int = 1200) -> str:
    text = ((res.get("stdout") or "") + (res.get("stderr") or "")).strip()
    return text[-limit:] if text else f"rc={res.get('rc')}"


def _run_step(root: Path, steps: list[dict[str, Any]], name: str,
              tool: str, args: list[str], *, timeout: int = 900,
              cwd: str = "",
              extra_env: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one toolchain step; record it; fail fast naming it."""
    try:
        res = run_tool("", tool, args, timeout=timeout, root=root,
                       cwd=cwd, extra_env=extra_env or {})
    except ToolchainError as exc:
        steps.append({"name": name, "ok": False, "detail": str(exc)})
        raise SrcBuildError(f"{name} step: {exc}") from exc
    ok = res["rc"] == 0
    steps.append({"name": name, "ok": ok,
                  "detail": _tail(res, 300) if not ok else ""})
    if not ok:
        raise SrcBuildError(f"{name} step failed: {_tail(res)}")
    return res


def tree_hash(src: Path) -> str:
    """Content hash of a source tree (paths + bytes), skipping
    build outputs and VCS noise."""
    skip = {".git", ".gradle", "build", "__pycache__"}
    h = hashlib.sha256()
    for path in sorted(src.rglob("*")):
        if any(part in skip for part in path.parts):
            continue
        if path.is_file():
            h.update(str(path.relative_to(src)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def _manifest_info(src: Path) -> dict[str, Any]:
    mpath = src / "AndroidManifest.xml"
    if not mpath.exists():
        mpath = src / "app" / "src" / "main" / "AndroidManifest.xml"
    if not mpath.exists():
        raise SrcBuildError(
            f"no AndroidManifest.xml in {src} — neither the manual "
            "layout (manifest at project root) nor a gradle app "
            "module manifest was found")
    try:
        root_el = ET.parse(mpath).getroot()
    except ET.ParseError as exc:
        raise SrcBuildError(
            f"manifest parse step: {mpath} is not well-formed XML: "
            f"{exc}") from exc
    package = root_el.get("package") or ""
    sdk_el = root_el.find("uses-sdk")
    def _sdk(attr: str, default: int) -> int:
        if sdk_el is not None and sdk_el.get(_A + attr):
            try:
                return int(sdk_el.get(_A + attr))  # type: ignore[arg-type]
            except ValueError:
                pass
        return default
    app_el = root_el.find("application")
    main_activity = ""
    if app_el is not None:
        for act in list(app_el.iter("activity")) + \
                list(app_el.iter("activity-alias")):
            name = act.get(_A + "name") or ""
            filters = act.findall("intent-filter")
            is_main = any(
                a.get(_A + "name") == "android.intent.action.MAIN"
                for f in filters for a in f.findall("action"))
            if is_main or not main_activity:
                if name.startswith("."):
                    name = package + name
                elif "." not in name:
                    name = f"{package}.{name}"
                main_activity = name
                if is_main:
                    break
    return {"path": mpath, "package": package,
            "version_code": root_el.get(_A + "versionCode"),
            "version_name": root_el.get(_A + "versionName") or "",
            "min_sdk": _sdk("minSdkVersion", 24),
            "target_sdk": _sdk("targetSdkVersion", 35),
            "main_activity": main_activity}


def _java_sources(src: Path) -> list[Path]:
    roots = [src / "java", src / "src" / "main" / "java",
             src / "app" / "src" / "main" / "java", src / "src"]
    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        if r.exists():
            for f in sorted(r.rglob("*.java")):
                if f not in seen and "build" not in f.parts:
                    seen.add(f)
                    out.append(f)
    return out


def detect_route(src: Path) -> str:
    """gradle when the project carries gradle build files, manual
    for the simple manifest-at-root layout; anything else refuses
    by name."""
    src = Path(src)
    gradle_markers = ("build.gradle", "build.gradle.kts",
                      "settings.gradle", "settings.gradle.kts",
                      "gradlew", "gradlew.bat")
    if any((src / m).exists() for m in gradle_markers) or \
            any((src / "app" / m).exists()
                for m in ("build.gradle", "build.gradle.kts")):
        return "gradle"
    if (src / "AndroidManifest.xml").exists():
        return "manual"
    raise SrcBuildError(
        f"{src} is not a buildable source project: no gradle build "
        "files (build.gradle[.kts] / settings.gradle / gradlew) and "
        "no AndroidManifest.xml at its root for the manual pipeline")


# ── scaffold ─────────────────────────────────────────────────────────

_SCAFFOLD_MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="{package}" android:versionCode="1" android:versionName="1.0">
    <uses-sdk android:minSdkVersion="24" android:targetSdkVersion="35"/>
    <uses-permission android:name="android.permission.INTERNET"/>
    <application android:label="@string/app_name">
        <activity android:name=".MainActivity" android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN"/>
                <category android:name="android.intent.category.LAUNCHER"/>
            </intent-filter>
        </activity>
    </application>
</manifest>
"""

_SCAFFOLD_ACTIVITY = """package __PACKAGE__;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

/** The scaffold's one screen: it renders the app_name string
 * resource and counts button taps on screen — real behavior the
 * owner edits from, not a placeholder. */
public class MainActivity extends Activity {
    private int taps = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        final TextView counter = new TextView(this);
        counter.setText(getString(R.string.app_name) + " — taps: 0");
        Button button = new Button(this);
        button.setText(getString(R.string.tap_button));
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                taps++;
                counter.setText(getString(R.string.app_name)
                        + " — taps: " + taps);
            }
        });
        layout.addView(counter);
        layout.addView(button);
        setContentView(layout);
    }
}
"""

_SCAFFOLD_STRINGS = """<?xml version="1.0" encoding="utf-8"?>
<resources>
    <string name="app_name">{title}</string>
    <string name="tap_button">Tap me</string>
</resources>
"""


def scaffold_project(parent: Path, name: str, *,
                     package: str = "") -> dict[str, Any]:
    """Create a minimal but REAL, complete app project: manifest,
    a MainActivity with actual behavior, string resources.  It
    builds through the manual pipeline the moment it exists."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise SrcBuildError(f"project name {name!r} has no usable "
                            "characters for a project slug")
    package = package or f"com.devon.{slug.replace('-', '')}"
    if not re.fullmatch(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+",
                        package):
        raise SrcBuildError(
            f"package {package!r} is not a valid application id "
            "(dot-separated lowercase segments)")
    dest = Path(parent) / slug
    if dest.exists() and any(dest.iterdir()):
        raise SrcBuildError(f"source project {slug!r} already "
                            f"exists at {dest}")
    title = name.strip() or slug
    files = {
        "AndroidManifest.xml": _SCAFFOLD_MANIFEST.format(
            package=package),
        "res/values/strings.xml": _SCAFFOLD_STRINGS.format(
            title=title.replace("&", "&amp;").replace("<", "&lt;")),
        f"java/{package.replace('.', '/')}/MainActivity.java":
            _SCAFFOLD_ACTIVITY.replace("__PACKAGE__", package),
    }
    for rel, text in files.items():
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return {"name": slug, "package": package, "dir": str(dest),
            "route": "manual",
            "files": sorted(files)}


# ── manual pipeline ──────────────────────────────────────────────────

def _manual_build(context: Any, root: Path, src: Path,
                  steps: list[dict[str, Any]], *,
                  alias: str, use_r8: bool) -> dict[str, Any]:
    jar = android_jar_path(root=root)
    if jar is None:
        raise SrcBuildError(
            "android.jar step: SDK platform 35 is not provisioned "
            "on this host (run: apkforge toolchain "
            "provision-srcbuild)")
    info = _manifest_info(src)
    if not info["package"]:
        raise SrcBuildError(
            "manifest step: AndroidManifest.xml declares no package "
            "attribute — the manual pipeline needs it (a gradle "
            "project may carry it as namespace instead; this tree "
            "was routed manual)")
    work = src / "build" / "apkforge-manual"
    shutil.rmtree(work, ignore_errors=True)
    (work / "gen").mkdir(parents=True, exist_ok=True)
    (work / "classes").mkdir(parents=True, exist_ok=True)
    (work / "dex").mkdir(parents=True, exist_ok=True)

    # 1 — aapt2 compile (resources → .flat bundle)
    link_inputs: list[str] = []
    res_dir = src / "res"
    if res_dir.exists() and any(res_dir.rglob("*")):
        _run_step(root, steps, "aapt2 compile", "aapt2",
                  ["compile", "--dir", str(res_dir),
                   "-o", str(work / "res.zip")], timeout=600)
        link_inputs.append(str(work / "res.zip"))
    else:
        steps.append({"name": "aapt2 compile", "ok": True,
                      "detail": "no res/ directory — skipped"})

    # 2 — aapt2 link (merged resources + manifest → resource APK
    #     + generated R.java), against the platform android.jar
    link_args = ["link", "-o", str(work / "res.apk"),
                 "-I", str(jar),
                 "--manifest", str(info["path"]),
                 "--java", str(work / "gen"),
                 "--min-sdk-version", str(info["min_sdk"]),
                 "--target-sdk-version", str(info["target_sdk"])]
    if (src / "assets").exists():
        link_args += ["-A", str(src / "assets")]
    link_args += link_inputs
    _run_step(root, steps, "aapt2 link", "aapt2", link_args,
              timeout=600)

    # 3 — javac (sources + generated R.java → .class), pinned by
    #     experiment on JDK 21: -source/-target 8 with android.jar
    #     as the boot classpath is the combination d8 accepts.
    sources = _java_sources(src)
    gen_java = sorted((work / "gen").rglob("*.java"))
    if not sources and not gen_java:
        raise SrcBuildError(
            "javac step: no .java sources found under java/, "
            "src/main/java or src/ — a source project for the "
            "manual pipeline carries its sources in one of those")
    argfile = work / "sources.txt"
    argfile.write_text(
        "\n".join(f'"{p}"' for p in [*sources, *gen_java]),
        encoding="utf-8")
    _run_step(root, steps, "javac", "javac",
              ["-encoding", "UTF-8", "-source", "8", "-target", "8",
               "-bootclasspath", str(jar),
               "-d", str(work / "classes"), f"@{argfile}"],
              timeout=1200)
    _run_step(root, steps, "jar classes", "jar",
              ["--create", "--file", str(work / "classes.jar"),
               "-C", str(work / "classes"), "."], timeout=300)

    # 4 — dex: d8, or R8 when shrinking was requested (R8 emits
    #     mapping.txt; the launcher activity is kept by rule so the
    #     app still starts, everything else may be renamed)
    mapping_path: Path | None = None
    if use_r8:
        keep = work / "keep-rules.pro"
        keep_lines = []
        if info["main_activity"]:
            keep_lines.append(
                f"-keep class {info['main_activity']} {{ *; }}")
        keep_lines.append("-keep class **.R { *; }")
        keep_lines.append("-keep class **.R$* { *; }")
        keep.write_text("\n".join(keep_lines) + "\n",
                        encoding="utf-8")
        mapping_path = work / "mapping.txt"
        _run_step(root, steps, "r8", "r8",
                  ["--release", "--output", str(work / "dex"),
                   "--min-api", str(info["min_sdk"]),
                   "--pg-map-output", str(mapping_path),
                   "--lib", str(jar),
                   "--pg-conf", str(keep),
                   str(work / "classes.jar")], timeout=1200)
    else:
        _run_step(root, steps, "d8", "d8",
                  ["--output", str(work / "dex"),
                   "--min-api", str(info["min_sdk"]),
                   str(work / "classes.jar")], timeout=1200)
    dex_files = sorted((work / "dex").glob("classes*.dex"))
    if not dex_files:
        raise SrcBuildError("dex step produced no classes*.dex")

    # 5 — assemble: the linked resource APK + every dex + native
    #     libs (.so stored uncompressed, as Android requires for
    #     directly-mmap'd libraries)
    unsigned = work / "unsigned.apk"
    shutil.copyfile(work / "res.apk", unsigned)
    with zipfile.ZipFile(unsigned, "a") as zf:
        for dex in dex_files:
            zf.write(dex, dex.name, zipfile.ZIP_DEFLATED)
        lib_dir = src / "lib"
        if lib_dir.exists():
            for so in sorted(lib_dir.rglob("*.so")):
                arc = str(so.relative_to(src))
                zf.write(so, arc, zipfile.ZIP_STORED)
    steps.append({"name": "assemble", "ok": True,
                  "detail": f"{len(dex_files)} dex file(s)"})

    # 6 — zipalign → sign → verify, through the one signing path
    out_dir = src / "build" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{info['package'].replace('.', '_')}-manual"
    signed = out_dir / f"{stem}-signed.apk"
    try:
        sig = signing_mod.sign_apk(context, root, unsigned, signed,
                                   alias=alias)
    except signing_mod.SigningError as exc:
        raise SrcBuildError(f"sign step: {exc}") from exc
    steps.append({"name": "zipalign+sign+verify", "ok": True,
                  "detail": f"alias {sig['alias']}"})
    return {"artifacts": [{
        "kind": "apk", "path": str(signed),
        "sha256": sha256_file(signed), "size": signed.stat().st_size,
        "signed_by": "apkforge keystore (class signing key)",
        "alias": sig["alias"], "cert_sha256": sig["cert_sha256"],
        "verify": sig["verify"]}],
        "mapping": str(mapping_path) if mapping_path else "",
        "dex_files": [d.name for d in dex_files],
        "info": info}


# ── gradle route ─────────────────────────────────────────────────────

def _gradle_env(root: Path, sdk: Path,
                truststore: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for var in strip_proxy_env():
        env.pop(var, None)
    paths = tool_paths(root=root)
    java = paths.get("java", {}).get("path", "")
    if java:
        env["JAVA_HOME"] = str(Path(java).resolve().parent.parent)
        env["PATH"] = (str(Path(java).parent) + os.pathsep
                       + env.get("PATH", ""))
    env["ANDROID_HOME"] = str(sdk)
    env["ANDROID_SDK_ROOT"] = str(sdk)
    env["GRADLE_USER_HOME"] = str(tools_dir(root=root) / ".gradle")
    # The Gradle client talks to its daemon over a loopback TCP
    # socket.  On hosts where dual-stack (AF_INET6) sockets are
    # intercepted or NAT'd away from loopback — observed on the
    # owner's VM, where the client's local address came back as a
    # routed interface address and the daemon never saw a byte —
    # the build dies with "Could not receive a message from the
    # daemon".  Pinning the JVM to the IPv4 stack keeps the
    # client↔daemon channel on true loopback; JAVA_TOOL_OPTIONS
    # reaches both the client JVM and the forked daemon JVM.  The
    # merged truststore rides the same channel so the WRAPPER's
    # own distribution download (client JVM, before any daemon or
    # gradle.properties applies) validates proxied TLS too.
    props = ["-Djava.net.preferIPv4Stack=true"]
    if truststore is not None:
        props.append(f"-Djavax.net.ssl.trustStore={truststore}")
        props.append("-Djavax.net.ssl.trustStorePassword=changeit")
    existing = env.get("JAVA_TOOL_OPTIONS", "")
    for prop in props:
        if prop not in existing:
            existing = (existing + " " + prop).strip()
    env["JAVA_TOOL_OPTIONS"] = existing
    return env


def _proxy_property_lines() -> list[str]:
    """gradle.properties systemProp.* proxy lines translated from
    the process's proxy env vars (http_proxy/https_proxy), when
    set.  Gradle's HTTP layer (Apache HttpClient) honors these —
    including the credentials — where raw JVM tooling cannot use
    the env form at all.  Values come only from the live
    environment; nothing is hardcoded, and the file lives in the
    class's own GRADLE_USER_HOME (chmod 600), never in a project."""
    import urllib.parse

    lines: list[str] = []
    for scheme in ("http", "https"):
        raw = os.environ.get(f"{scheme}_proxy") or \
            os.environ.get(f"{scheme.upper()}_PROXY") or ""
        if not raw:
            continue
        u = urllib.parse.urlparse(raw)
        if not u.hostname:
            continue
        lines.append(f"systemProp.{scheme}.proxyHost={u.hostname}")
        if u.port:
            lines.append(f"systemProp.{scheme}.proxyPort={u.port}")
        if u.username:
            lines.append(f"systemProp.{scheme}.proxyUser="
                         f"{urllib.parse.unquote(u.username)}")
        if u.password:
            lines.append(f"systemProp.{scheme}.proxyPassword="
                         f"{urllib.parse.unquote(u.password)}")
    if lines:
        lines.append("systemProp.http.nonProxyHosts="
                     "localhost|127.0.0.1")
        lines.append("systemProp.https.nonProxyHosts="
                     "localhost|127.0.0.1")
    return lines


def _ensure_gradle_truststore(root: Path) -> Path | None:
    """The merged JDK+system truststore (toolchain.jvm_truststore)
    that Gradle's JVMs need on TLS-intercepted hosts — without it
    every dependency download dies in certificate validation and
    surfaces as an opaque "plugin not found"."""
    return jvm_truststore(root=root)


def _ensure_gradle_properties(root: Path,
                              truststore: Path | None = None) -> None:
    """GRADLE_USER_HOME/gradle.properties carrying (a) the
    daemon-side half of the loopback fix (systemProp.* reaches the
    daemon JVM even when it is already running), (b) the proxy
    translation above, without which Gradle cannot reach Google
    Maven / the plugin portal on proxied hosts, and (c) the
    merged truststore, when one was built.  An existing file is
    only appended to for properties it lacks — never rewritten."""
    guh = tools_dir(root=root) / ".gradle"
    guh.mkdir(parents=True, exist_ok=True)
    props = guh / "gradle.properties"
    wanted = ["systemProp.java.net.preferIPv4Stack=true",
              *_proxy_property_lines()]
    if truststore is not None:
        wanted += [
            f"systemProp.javax.net.ssl.trustStore={truststore}",
            "systemProp.javax.net.ssl.trustStorePassword=changeit",
        ]
    existing = props.read_text(encoding="utf-8",
                               errors="replace") if props.exists() \
        else ""
    missing = [line for line in wanted
               if line.split("=", 1)[0] not in existing]
    if missing:
        with open(props, "a", encoding="utf-8") as fh:
            for line in missing:
                fh.write(line + "\n")
    try:
        props.chmod(0o600)
    except OSError:
        pass


def _run_gradle(root: Path, src: Path, tasks: list[str],
                steps: list[dict[str, Any]]) -> None:
    sdk = assemble_sdk_root(root=root)
    truststore = _ensure_gradle_truststore(root)
    _ensure_gradle_properties(root, truststore=truststore)
    (src / "local.properties").write_text(
        f"sdk.dir={sdk.as_posix()}\n", encoding="utf-8")
    wrapper = src / "gradlew"
    if wrapper.exists():
        try:
            wrapper.chmod(wrapper.stat().st_mode | 0o111)
        except OSError:
            pass
        argv = [str(wrapper), *tasks, "--console=plain"]
        label = "gradlew " + " ".join(tasks)
    else:
        paths = tool_paths(root=root)
        gradle = paths.get("gradle", {}).get("path", "")
        if not gradle:
            raise SrcBuildError(
                "gradle step: this project has no gradlew wrapper "
                "and no Gradle is provisioned on this host (run: "
                "apkforge toolchain provision-srcbuild)")
        argv = [gradle, *tasks, "--console=plain"]
        label = "gradle " + " ".join(tasks)
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=str(src),
        env=_gradle_env(root, sdk, truststore=truststore),
        start_new_session=True)
    try:
        out, err = proc.communicate(timeout=GRADLE_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, err = proc.communicate()
        steps.append({"name": label, "ok": False,
                      "detail": f"timed out after {GRADLE_TIMEOUT}s"})
        raise SrcBuildError(
            f"gradle step timed out after {GRADLE_TIMEOUT}s and "
            f"was killed — tail: {(out + err)[-600:]}") from None
    ok = proc.returncode == 0
    steps.append({"name": label, "ok": ok,
                  "detail": "" if ok else (out + err)[-300:]})
    if not ok:
        raise SrcBuildError(
            f"gradle step ({' '.join(tasks)}) failed: "
            f"{(out + err)[-1200:]}")


def _gradle_build(context: Any, root: Path, src: Path,
                  steps: list[dict[str, Any]], *,
                  alias: str, variant: str,
                  bundle: bool) -> dict[str, Any]:
    tasks = ["assembleDebug"]
    if variant in ("release", "all"):
        tasks.append("assembleRelease")
    if bundle:
        tasks.append("bundleRelease")
    steps.append({"name": "sdk assemble+licenses", "ok": True,
                  "detail": "sdk layout assembled by toolchain; "
                            "licenses accepted at provision time "
                            "via sdkmanager"})
    _run_gradle(root, src, tasks, steps)
    artifacts: list[dict[str, Any]] = []
    apk_roots = [src / "app" / "build" / "outputs" / "apk",
                 src / "build" / "outputs" / "apk"]
    for apk_root in apk_roots:
        if not apk_root.exists():
            continue
        for apk in sorted(apk_root.rglob("*.apk")):
            if apk.name.endswith("-signed.apk"):
                continue
            verdict = signing_mod.verify_apk(root, apk)
            entry: dict[str, Any] = {
                "kind": "apk", "path": str(apk),
                "sha256": sha256_file(apk),
                "size": apk.stat().st_size, "verify": verdict}
            if verdict.get("verifies"):
                entry["signed_by"] = "project (gradle signing)"
                entry["cert_sha256"] = verdict.get("cert_sha256", "")
            else:
                signed = apk.with_name(
                    apk.name.replace(".apk", "-signed.apk"))
                try:
                    sig = signing_mod.sign_apk(context, root, apk,
                                               signed, alias=alias)
                except signing_mod.SigningError as exc:
                    raise SrcBuildError(
                        f"sign step for gradle output "
                        f"{apk.name}: {exc}") from exc
                entry = {"kind": "apk", "path": str(signed),
                         "sha256": sha256_file(signed),
                         "size": signed.stat().st_size,
                         "signed_by": "apkforge keystore (class signing key)",
                         "alias": sig["alias"],
                         "cert_sha256": sig["cert_sha256"],
                         "verify": sig["verify"],
                         "unsigned_source": str(apk)}
            artifacts.append(entry)
    bundle_roots = [src / "app" / "build" / "outputs" / "bundle",
                    src / "build" / "outputs" / "bundle"]
    for bundle_root in bundle_roots:
        if not bundle_root.exists():
            continue
        for aab in sorted(bundle_root.rglob("*.aab")):
            artifacts.append({
                "kind": "aab", "path": str(aab),
                "sha256": sha256_file(aab),
                "size": aab.stat().st_size,
                "signed_by": "unsigned — an .aab is signed by "
                             "bundletool at splits-generate time "
                             "with the class keystore",
                "verify": {}})
    if not artifacts:
        raise SrcBuildError(
            "gradle step produced no APK/AAB outputs under "
            "build/outputs — the build reported success but shipped "
            "nothing; refusing to card an empty build")
    return {"artifacts": artifacts, "mapping": "",
            "dex_files": [], "info": {}}


# ── top-level build ──────────────────────────────────────────────────

def build_source(context: Any, root: Path, src: Path, *,
                 alias: str = "", route: str = "",
                 variant: str = "debug", bundle: bool = False,
                 use_r8: bool = False) -> dict[str, Any]:
    """Build one landed source project → the build card.

    The card follows build.py's idiom: source hash, the route that
    ran, every step's verdict, the artifacts with their hashes and
    signing facts, and the standing re-sign facts on every card
    where the class keystore signed."""
    src = Path(src)
    if not src.is_dir():
        raise SrcBuildError(f"source project {src} is not a "
                            "directory")
    route = (route or detect_route(src)).strip().lower()
    if route not in ("manual", "gradle"):
        raise SrcBuildError(f"unknown route {route!r} — "
                            "manual|gradle (or omit to auto-detect)")
    steps: list[dict[str, Any]] = []
    started = time.time()
    if route == "manual":
        result = _manual_build(context, root, src, steps,
                               alias=alias, use_r8=use_r8)
    else:
        result = _gradle_build(context, root, src, steps,
                               alias=alias, variant=variant,
                               bundle=bundle)
    info = result.get("info") or {}
    if not info:
        try:
            info = _manifest_info(src)
        except SrcBuildError:
            info = {}
    artifacts = result["artifacts"]
    primary = artifacts[0]
    card: dict[str, Any] = {
        "route": route,
        "source_dir": str(src),
        "source_tree_sha256": tree_hash(src),
        "package": info.get("package", ""),
        "version_name": info.get("version_name", ""),
        "main_activity": info.get("main_activity", ""),
        "built_at": time.time(),
        "duration_s": round(time.time() - started, 2),
        "steps": steps,
        "artifacts": artifacts,
        "output": primary["path"],
        "output_sha256": primary["sha256"],
        "size": primary["size"],
        "dex_files": result.get("dex_files") or [],
        "mapping": result.get("mapping", ""),
        "signing": {
            "signed_by": primary.get("signed_by", ""),
            "alias": primary.get("alias", ""),
            "cert_sha256": primary.get("cert_sha256", ""),
            "schemes": (primary.get("verify") or {}).get(
                "schemes", {})},
        "resign_facts": signing_mod.RESIGN_FACTS,
        "museum_card": {
            "id": f"srcbuild-{tree_hash(src)[:12]}",
            "title": f"{info.get('package') or src.name} "
                     f"({route} build)",
            "kind": "apk-build",
            "shelf": "apk-builds",
            "path": primary["path"],
            "hash": primary["sha256"],
            "size": primary["size"],
            "provenance": {"type": "buildsrc", "ref": str(src),
                           "tool": "apkforge"},
            "tags": ["apk", "build", "srcbuild", route,
                     info.get("package", "")],
        },
    }
    out_dir = src / "build" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    card_path = out_dir / "srcbuild-card.json"
    card_path.write_text(json.dumps(card, indent=1, sort_keys=True),
                         encoding="utf-8")
    card["card_path"] = str(card_path)
    return card


def render_srcbuild_card(card: dict[str, Any]) -> str:
    steps = ", ".join(
        f"{s['name']}={'ok' if s['ok'] else 'FAILED'}"
        for s in card.get("steps") or [])
    lines = [
        f"🪓 **built from source** — {card.get('package') or '?'} "
        f"via the {card.get('route')} route",
        f"source: {card.get('source_dir')} "
        f"(tree {str(card.get('source_tree_sha256'))[:16]}…)",
        f"steps: {steps}",
    ]
    for art in card.get("artifacts") or []:
        lines.append(
            f"{art['kind']}: {art['path']} — "
            f"{art.get('size', 0):,} bytes · sha256 "
            f"{str(art.get('sha256'))[:16]}… · signed by "
            f"{art.get('signed_by')}")
    if card.get("mapping"):
        lines.append(f"R8 mapping: {card['mapping']}")
    lines.append("re-sign facts:")
    lines += [f"  • {fact}" for fact in card.get("resign_facts") or []]
    return "\n".join(lines)


# ── ProGuard/R8 mapping → jadx Tiny v2 ───────────────────────────────

_PRIMITIVES = {"void": "V", "boolean": "Z", "byte": "B", "char": "C",
               "short": "S", "int": "I", "long": "J", "float": "F",
               "double": "D"}


def _descriptor(java_type: str) -> str:
    t = java_type.strip()
    dims = ""
    while t.endswith("[]"):
        dims += "["
        t = t[:-2].strip()
    if t in _PRIMITIVES:
        return dims + _PRIMITIVES[t]
    return f"{dims}L{t.replace('.', '/')};"


_CLASS_LINE = re.compile(r"^(\S+) -> (\S+):$")
_MEMBER_METHOD = re.compile(
    r"^(?:\d+:\d+:)?(\S+) (\S+)\((.*)\)(?::\d+(?::\d+)?)? -> (\S+)$")
_MEMBER_FIELD = re.compile(r"^(\S+) (\S+) -> (\S+)$")


def proguard_to_tiny(mapping_text: str) -> str:
    """Convert a ProGuard/R8 mapping.txt to a Tiny v2 mapping that
    jadx's --mappings-path surface consumes (namespaces: official =
    the obfuscated names in the APK, named = the originals).

    Class entries always convert; method/field entries convert
    when their signatures parse — jadx ignores what it cannot
    match, and the class restoration does not depend on them."""
    lines = ["tiny\t2\t0\tofficial\tnamed"]
    owner = ""
    for raw in mapping_text.splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        m = _CLASS_LINE.match(raw.strip()) if not raw[0].isspace() \
            else None
        if m:
            original, obfuscated = m.group(1), m.group(2)
            owner = obfuscated.replace(".", "/")
            lines.append(
                f"c\t{owner}\t{original.replace('.', '/')}")
            continue
        if not owner:
            continue
        body = raw.strip()
        mm = _MEMBER_METHOD.match(body)
        if mm and "(" in body:
            ret, name, args, obf = (mm.group(1), mm.group(2),
                                    mm.group(3), mm.group(4))
            arg_desc = "".join(
                _descriptor(a) for a in args.split(",") if a.strip())
            desc = f"({arg_desc}){_descriptor(ret)}"
            lines.append(f"m\t{desc}\t{owner}\t{obf}\t{name}")
            continue
        mf = _MEMBER_FIELD.match(body)
        if mf:
            ftype, name, obf = mf.group(1), mf.group(2), mf.group(3)
            lines.append(
                f"f\t{_descriptor(ftype)}\t{owner}\t{obf}\t{name}")
    return "\n".join(lines) + "\n"
