"""ApkForge A4 — the real builder: build-from-source.

Everything here is proven against REAL builds produced by the
provisioned toolchain itself:

* the manual pipeline (aapt2 compile/link → javac → d8 → assemble
  → zipalign → vault-keystore sign → apksigner verify) builds the
  scaffolded app, and A1's dissect reads the product back: real
  package, INTERNET permission, launcher activity, arsc-resolved
  app label, v2+v3 signature schemes;
* a generated 3000-class fixture compiles through the same
  pipeline so d8 emits classes.dex + classes2.dex FOR REAL, and
  the dissect dex census is pinned to cover both;
* the R8 variant shrinks with a keep rule, produces mapping.txt,
  the dossier shows the obfuscated name, and jadx WITH the
  mapping restores the original class name;
* a Gradle fixture project (AGP 8.7.3) builds assembleDebug +
  assembleRelease + bundleRelease through the Gradle route — a
  REAL .aab, which then runs through A2's splits generation AND
  universal merge for real (A2 could only error-path these: no
  valid .aab fixture existed);
* the editor's general text ops flip a const in a decoded smali
  file, and the A2 rebuild carries the change;
* failure honesty: broken Java fails at the javac step with the
  real compiler error, no card, no artifact.

The toolchain provisions once into a shared root under the
workspace (APKFORGE_A4_TEST_ROOT overrides it); the tool cache at
~/workspace/a4-verify/tool-cache makes re-provisioning offline.

Environment notes (this VM): Gradle JVMs need the class's own
host fixes — IPv4-stack pinning for the daemon's loopback socket,
gradle.properties proxy translation, and a truststore merged with
the host CA bundle — all shipped in srcbuild/toolchain and
exercised by the Gradle test below, not mocked around.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

_WS = Path(os.path.expanduser("~/workspace"))
SESSION_ROOT = Path(os.environ.get(
    "APKFORGE_A4_TEST_ROOT", str(_WS / "a4-verify" / "a4-test-root")))
TOOL_CACHE = _WS / "a4-verify" / "tool-cache"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_GRADLE_SETTINGS = """pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}
rootProject.name = "gfix"
include ':app'
"""

_GRADLE_ROOT = """plugins {
    id 'com.android.application' version '8.7.3' apply false
}
"""

_GRADLE_APP = """plugins {
    id 'com.android.application'
}

android {
    buildToolsVersion = "35.0.0"
    namespace 'com.example.gfix'
    compileSdk 35

    defaultConfig {
        applicationId "com.example.gfix"
        minSdk 24
        targetSdk 35
        versionCode 1
        versionName "1.0"
    }
}

dependencies {
}
"""

_GFIX_MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
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

_GFIX_ACTIVITY = """package com.example.gfix;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

public class MainActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        TextView tv = new TextView(this);
        tv.setText(getString(R.string.app_name));
        setContentView(tv);
    }
}
"""

_GFIX_STRINGS = """<?xml version="1.0" encoding="utf-8"?>
<resources>
    <string name="app_name">Gradle Fixture</string>
</resources>
"""

_MULTI_MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="dev.apkforge.multi">
    <uses-sdk android:minSdkVersion="24" android:targetSdkVersion="35"/>
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

_MULTI_STRINGS = """<?xml version="1.0" encoding="utf-8"?>
<resources>
    <string name="app_name">Multidex Fixture</string>
</resources>
"""

_MULTI_MAIN = """package dev.apkforge.multi;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

public class MainActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        TextView tv = new TextView(this);
        tv.setText("classes: " + Gen0000.marker());
        setContentView(tv);
    }
}
"""

_BROKEN_MANIFEST = _MULTI_MANIFEST.replace("dev.apkforge.multi",
                                           "dev.apkforge.broken")
_BROKEN_MAIN = """package dev.apkforge.broken;

import android.app.Activity;
import android.os.Bundle;

public class MainActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState)
        int deliberately = broken syntax here;
    }
}
"""


def _write_gradle_fixture(dest: Path) -> Path:
    (dest / "app" / "src" / "main" / "java" / "com" / "example"
     / "gfix").mkdir(parents=True, exist_ok=True)
    (dest / "app" / "src" / "main" / "res" / "values").mkdir(
        parents=True, exist_ok=True)
    (dest / "settings.gradle").write_text(_GRADLE_SETTINGS)
    (dest / "build.gradle").write_text(_GRADLE_ROOT)
    (dest / "app" / "build.gradle").write_text(_GRADLE_APP)
    (dest / "app" / "src" / "main" / "AndroidManifest.xml").write_text(
        _GFIX_MANIFEST)
    (dest / "app" / "src" / "main" / "java" / "com" / "example"
     / "gfix" / "MainActivity.java").write_text(_GFIX_ACTIVITY)
    (dest / "app" / "src" / "main" / "res" / "values"
     / "strings.xml").write_text(_GFIX_STRINGS)
    return dest


def _write_multidex_fixture(dest: Path, n_classes: int = 3000,
                            n_methods: int = 25) -> Path:
    pkg_dir = dest / "java" / "dev" / "apkforge" / "multi"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (dest / "res" / "values").mkdir(parents=True, exist_ok=True)
    (dest / "AndroidManifest.xml").write_text(_MULTI_MANIFEST)
    (dest / "res" / "values" / "strings.xml").write_text(
        _MULTI_STRINGS)
    (pkg_dir / "MainActivity.java").write_text(_MULTI_MAIN)
    for i in range(n_classes):
        methods = []
        for j in range(n_methods):
            methods.append(
                f"    public static int m{j}(int x) "
                f"{{ return x + {j + 1}; }}")
        if i == 0:
            methods.append(
                "    public static int marker() { return 3000; }")
        body = "\n".join(methods)
        (pkg_dir / f"Gen{i:04d}.java").write_text(
            f"package dev.apkforge.multi;\n\n"
            f"public class Gen{i:04d} {{\n{body}\n}}\n")
    return dest


class ApkForgeA4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved_env = {k: os.environ.get(k) for k in
                          ("NM_VAULT_PASSPHRASE",
                           "APKFORGE_TOOL_CACHE")}
        os.environ["NM_VAULT_PASSPHRASE"] = "apkforge-a4-test-seal"
        if TOOL_CACHE.exists():
            os.environ["APKFORGE_TOOL_CACHE"] = str(TOOL_CACHE)
        SESSION_ROOT.mkdir(parents=True, exist_ok=True)
        cls.root = SESSION_ROOT / "apkforge"
        shutil.rmtree(cls.root, ignore_errors=True)
        cls.root.mkdir(parents=True)

        from nomorals.agents.apkforge import ApkForge
        from nomorals.storage.db import Database

        cls.db = Database(str(cls.root / "vault.db"))
        cls.ctx = SimpleNamespace(
            db=cls.db,
            settings=SimpleNamespace(workspace_dir=str(SESSION_ROOT)),
            extras={})
        cls.forge = ApkForge(cls.ctx, base_dir=str(cls.root))
        cls.provision = cls.forge.toolchain_provision()
        assert not cls.provision.get("error"), cls.provision
        cls.provision_src = cls.forge.toolchain_provision_srcbuild()
        assert not cls.provision_src.get("error"), cls.provision_src

        # ── shared build 1: scaffold → manual pipeline ──────────
        new = cls.forge.new_project("A4 Counter",
                                    package="dev.apkforge.counter")
        assert not new.get("error"), new
        cls.scaffold_dir = Path(new["project"]["dir"])
        built = cls.forge.buildsrc(str(cls.scaffold_dir))
        assert not built.get("error"), built
        cls.scaffold_build = built["build"]
        cls.scaffold_apk = Path(
            cls.scaffold_build["artifacts"][0]["path"])
        add = cls.forge.add(str(cls.scaffold_apk))
        cls.scaffold_pid = add["project"]["id"]
        diss = cls.forge.dissect(cls.scaffold_pid)
        assert not diss.get("error"), diss
        cls.scaffold_dossier = diss["dossier"]

        # ── shared build 2: R8 variant of a second scaffold ─────
        from nomorals.agents.apkforge import srcbuild as srcbuild_mod
        r8_src = SESSION_ROOT / "r8-src"
        shutil.rmtree(r8_src, ignore_errors=True)
        srcbuild_mod.scaffold_project(
            r8_src, "A4 Shrink", package="dev.apkforge.shrink")
        r8_built = srcbuild_mod.build_source(
            cls.ctx, cls.root, r8_src / "a4-shrink", use_r8=True)
        cls.r8_build = r8_built
        cls.r8_apk = Path(r8_built["artifacts"][0]["path"])
        add = cls.forge.add(str(cls.r8_apk))
        cls.r8_pid = add["project"]["id"]
        diss = cls.forge.dissect(cls.r8_pid)
        assert not diss.get("error"), diss
        cls.r8_dossier = diss["dossier"]

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # ── toolchain ────────────────────────────────────────────────
    def test_srcbuild_provision_and_capabilities(self):
        res = self.provision_src
        for name in ("platform-35", "cmdline-tools", "r8", "gradle"):
            self.assertIn(name, res["already_present"]
                          + res["provisioned_now"])
        self.assertIn("android-sdk-license",
                      res["sdk_licenses"]["accepted"])
        status = self.forge.toolchain_status()
        caps = status["capabilities"]
        for key in ("aapt2", "d8", "r8", "gradle",
                    "buildsrc_manual", "buildsrc_gradle"):
            self.assertEqual(caps[key], "full", f"{key}: {caps[key]}")
        self.assertTrue(caps["android_jar"].startswith("full"))
        self.assertIn("35", caps["android_jar"])
        probes = status["probes"]
        for tool in ("aapt2", "d8", "javac", "r8", "gradle",
                     "sdkmanager"):
            self.assertTrue(probes[tool]["ok"],
                            f"{tool}: {probes[tool]}")
        self.assertIn("8.5.35", probes["r8"]["version"])
        from nomorals.agents.apkforge.toolchain import toolchain_state
        state = toolchain_state(root=self.forge.root)
        self.assertEqual(state["artifacts"]["gradle"]["version"],
                         "8.9")
        self.assertEqual(state["artifacts"]["r8"]["version"],
                         "8.5.35")
        self.assertEqual(
            state["artifacts"]["platform-35"]["version"], "35_r02")
        self.assertEqual(
            state["artifacts"]["cmdline-tools"]["version"], "12.0")

    # ── manual pipeline + scaffold round trip ────────────────────
    def test_manual_pipeline_card_and_verify(self):
        card = self.scaffold_build
        self.assertEqual(card["route"], "manual")
        names = [s["name"] for s in card["steps"]]
        for step in ("aapt2 compile", "aapt2 link", "javac", "d8",
                     "assemble", "zipalign+sign+verify"):
            self.assertIn(step, names)
        self.assertTrue(all(s["ok"] for s in card["steps"]))
        art = card["artifacts"][0]
        self.assertTrue(art["verify"]["verifies"])
        self.assertTrue(art["verify"]["schemes"]["v2"])
        self.assertTrue(art["verify"]["schemes"]["v3"])
        self.assertEqual(art["signed_by"],
                         "apkforge keystore (class signing key)")
        self.assertIn("resign_facts", card)

    def test_scaffold_dissect_round_trip(self):
        d = self.scaffold_dossier
        self.assertEqual(d["identity"]["package"],
                         "dev.apkforge.counter")
        # the label resolves through the REAL resources.arsc
        self.assertEqual(d["identity"]["app_name"], "A4 Counter")
        self.assertEqual(d["identity"]["main_activity"],
                         "dev.apkforge.counter.MainActivity")
        perms = [p["name"] for p in d["permissions"]]
        self.assertIn("android.permission.INTERNET", perms)
        acts = d["components"]["activities"]
        self.assertEqual(len(acts), 1)
        self.assertTrue(acts[0]["exported"])
        filters = acts[0]["intent_filters"]
        self.assertTrue(any(
            "android.intent.category.LAUNCHER" in f["categories"]
            for f in filters))
        self.assertTrue(d["resources"]["arsc"]["present"])
        self.assertTrue(d["resources"]["arsc"]["parsed"])
        self.assertTrue(d["signing"]["v2"])
        self.assertTrue(d["signing"]["v3"])
        self.assertFalse(d["identity"]["multidex"])

    def test_new_project_refuses_bad_package(self):
        res = self.forge.new_project("Bad Pkg", package="Not A Pkg")
        self.assertIn("error", res)
        self.assertIn("package", res["error"])

    def test_buildsrc_from_zip(self):
        zip_path = SESSION_ROOT / "scaffold-src.zip"
        with zipfile.ZipFile(zip_path, "w",
                             zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(self.scaffold_dir.rglob("*")):
                if p.is_file() and "build" not in p.parts:
                    zf.write(p, p.relative_to(self.scaffold_dir))
        res = self.forge.buildsrc(str(zip_path))
        self.assertNotIn("error", res, res)
        self.assertEqual(res["build"]["route"], "manual")
        self.assertTrue(
            res["build"]["artifacts"][0]["verify"]["verifies"])

    # ── multidex ─────────────────────────────────────────────────
    def test_multidex_census_covers_both_dexes(self):
        from nomorals.agents.apkforge import srcbuild as srcbuild_mod
        src = SESSION_ROOT / "multidex-src"
        if not (src / "build" / "outputs" / "srcbuild-card.json") \
                .exists():
            shutil.rmtree(src, ignore_errors=True)
            _write_multidex_fixture(src)
            built = srcbuild_mod.build_source(
                self.ctx, self.root, src)
            self.assertEqual(built["route"], "manual")
            names = [s["name"] for s in built["steps"]]
            self.assertIn("d8", names)
        else:
            built = json.loads((src / "build" / "outputs"
                                / "srcbuild-card.json")
                               .read_text())
        apk = Path(built["artifacts"][0]["path"])
        with zipfile.ZipFile(apk) as zf:
            dex_names = sorted(n for n in zf.namelist()
                               if n.endswith(".dex"))
        self.assertGreaterEqual(len(dex_names), 2, dex_names)
        self.assertIn("classes2.dex", dex_names)
        add = self.forge.add(str(apk))
        diss = self.forge.dissect(add["project"]["id"])
        self.assertNotIn("error", diss, diss)
        dex = diss["dossier"]["dex"]
        self.assertGreaterEqual(len(dex["per_dex"]), 2)
        self.assertGreaterEqual(dex["totals"]["classes"], 3000)
        self.assertTrue(diss["dossier"]["identity"]["multidex"])

    # ── R8 + mapping ─────────────────────────────────────────────
    def test_r8_mapping_and_obfuscated_dossier(self):
        card = self.r8_build
        self.assertTrue(card["mapping"])
        mapping = Path(card["mapping"])
        self.assertTrue(mapping.exists())
        text = mapping.read_text()
        self.assertIn("dev.apkforge.shrink.MainActivity$1 -> a.a",
                      text)
        self.assertTrue(card["artifacts"][0]["verify"]["verifies"])
        rollup = {r["package"]
                  for r in self.r8_dossier["dex"]["package_rollup"]}
        self.assertIn("a", rollup)

    def test_jadx_mapping_restores_names(self):
        plain = self.forge.jadx(self.r8_pid)
        self.assertNotIn("error", plain, plain)
        plain_tree = Path(plain["jadx"])
        self.assertTrue(any(
            p.name == "a.java" for p in plain_tree.rglob("*.java")))
        mapped = self.forge.jadx(
            self.r8_pid, mapping=self.r8_build["mapping"])
        self.assertNotIn("error", mapped, mapped)
        tree = Path(mapped["jadx"])
        self.assertTrue(any(
            p.name == "MainActivity$1.java"
            for p in tree.rglob("*.java")),
            "mapping did not restore MainActivity$1")

    # ── gradle route + real AAB ──────────────────────────────────
    def test_gradle_route_and_real_aab_splits(self):
        src = SESSION_ROOT / "gfix-src"
        if not (src / "settings.gradle").exists():
            shutil.rmtree(src, ignore_errors=True)
            _write_gradle_fixture(src)
        res = self.forge.buildsrc(str(src), variant="all",
                                  bundle=True)
        self.assertNotIn("error", res, res)
        card = res["build"]
        self.assertEqual(card["route"], "gradle")
        kinds = {a["kind"] for a in card["artifacts"]}
        self.assertIn("apk", kinds)
        self.assertIn("aab", kinds)
        apks = [a for a in card["artifacts"] if a["kind"] == "apk"]
        self.assertGreaterEqual(len(apks), 2)  # debug + release
        for a in apks:
            self.assertTrue(a["verify"]["verifies"], a)
        release = [a for a in apks
                   if "release" in a["path"]][0]
        self.assertEqual(release["signed_by"],
                         "apkforge keystore (class signing key)")
        aab = [a for a in card["artifacts"]
               if a["kind"] == "aab"][0]
        # splits on the REAL bundle: generate + universal merge
        add = self.forge.add(aab["path"])
        self.assertNotIn("error", add, add)
        pid = add["project"]["id"]
        gen = self.forge.splits(pid, "generate")
        self.assertNotIn("error", gen, gen)
        self.assertTrue(any(n.endswith(".apk")
                            for n in gen["generate"]["entries"]))
        merged = self.forge.splits(pid, "merge")
        self.assertNotIn("error", merged, merged)
        out = Path(merged["merge"]["output"])
        self.assertTrue(out.exists())
        diss = self.forge.dissect(
            self.forge.add(str(out))["project"]["id"])
        self.assertNotIn("error", diss, diss)
        self.assertEqual(diss["dossier"]["identity"]["package"],
                         "com.example.gfix")

    # ── editor completion: smali write → rebuild ─────────────────
    def test_smali_write_rebuild_round_trip(self):
        pid = self.scaffold_pid
        dec = self.forge.decode(pid)
        self.assertNotIn("error", dec, dec)
        rel = "smali/dev/apkforge/counter/MainActivity.smali"
        read = self.forge.read_text(pid, rel)
        self.assertNotIn("error", read, read)
        text = read["file"]["text"]
        anchor = "add-int/lit8"
        self.assertIn(anchor, text)
        i = text.index(anchor)
        j = text.index("0x1", i)
        flipped = text[:j] + "0x7" + text[j + 3:]
        ed = self.forge.edit(pid, "write", target=rel, value=flipped)
        self.assertNotIn("error", ed, ed)
        self.assertEqual(ed["edit"]["op"], "edit_text")
        built = self.forge.build(pid)
        self.assertNotIn("error", built, built)
        out = built["build"]["output"]
        add = self.forge.add(out)
        jx = self.forge.jadx(add["project"]["id"])
        self.assertNotIn("error", jx, jx)
        tree = Path(jx["jadx"])
        joined = "\n".join(
            p.read_text(errors="replace")
            for p in tree.rglob("MainActivity.java"))
        self.assertTrue("+ 7" in joined or "+= 7" in joined,
                        "smali const flip did not reach the rebuild")

    def test_text_ops_refuse_escape_and_binary(self):
        pid = self.scaffold_pid
        res = self.forge.edit(pid, "write", target="../escape.smali",
                              value=".class public LEvil;")
        self.assertIn("error", res)
        res = self.forge.edit(pid, "write",
                              target="lib/arm64-v8a/libx.so",
                              value="text")
        self.assertIn("error", res)
        self.assertIn("binary", res["error"])
        res = self.forge.read_text(pid, "original.apk")
        self.assertIn("error", res)

    # ── failure honesty ──────────────────────────────────────────
    def test_broken_java_fails_at_javac_with_real_error(self):
        from nomorals.agents.apkforge import srcbuild as srcbuild_mod
        src = SESSION_ROOT / "broken-src"
        shutil.rmtree(src, ignore_errors=True)
        pkg = src / "java" / "dev" / "apkforge" / "broken"
        pkg.mkdir(parents=True)
        (src / "res" / "values").mkdir(parents=True)
        (src / "AndroidManifest.xml").write_text(_BROKEN_MANIFEST)
        (src / "res" / "values" / "strings.xml").write_text(
            _MULTI_STRINGS)
        (pkg / "MainActivity.java").write_text(_BROKEN_MAIN)
        with self.assertRaises(srcbuild_mod.SrcBuildError) as cm:
            srcbuild_mod.build_source(self.ctx, self.root, src)
        msg = str(cm.exception)
        self.assertIn("javac", msg)
        self.assertIn("error", msg)
        self.assertFalse(
            (src / "build" / "outputs" / "srcbuild-card.json")
            .exists())


if __name__ == "__main__":
    unittest.main()
