"""ApkForge A5 — the weakness burn-down.

Every item is pinned against REAL evidence, in the A3/A4 idiom:

* signature discovery reads the dossier's REFERENCED-type census:
  synthetic dossiers whose only evidence for a prefix lives in
  references become proposals with the defined/referenced split
  intact, and a real apktool fixture whose smali merely *calls*
  two classes of an unbundled SDK (it defines neither) grows the
  table the same way.  The recurrence threshold is config
  (NOMORALS_APKFORGE_SIG_THRESHOLD), proven by lowering it.
* the exported: bisect predicate is three-valued: component_state
  unit pins + a real series crossing absent → present-not-exported
  → present-exported, both crossings named; a direct
  absent → present-exported series names that too.
* the native fixture is a REAL compiled library (host cc for
  x86_64, the aarch64 cross cc for arm64-v8a): exported functions
  calling each other, an imported strlen, a .rodata version
  string, dynamic relocations asserted present so the fixture can
  never degrade to a flat blob.  It rides the A4 manual pipeline
  into an APK; dissect's ELF census reads it; a wildcard hex
  pattern spanning the magic constant patches it; the patched
  bytes land in a rebuilt signed APK; undo is byte-exact.
* the bisect skip is proven against a REAL unparseable APK: a
  four-version series built by the A4 pipeline, the second
  version's classes.dex truncated in the landed file — androguard
  records the real parse error, the bisect skips that version
  with the error journaled, brackets the permission flip from
  the surviving neighbors, and resumes with the reasons intact.
* teach v2 walks one method in smali AND jadx Java on the A4 R8
  fixture: mapping-restored name, the real smali op and the real
  Java line cited, the no-brain side-by-side fallback, and the
  brain path held to the citation check.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_WS = Path(os.path.expanduser("~/workspace"))
SESSION_ROOT = Path(os.environ.get(
    "APKFORGE_A5_TEST_ROOT", str(_WS / "a5-verify" / "a5-test-root")))
TOOL_CACHE = _WS / "a4-verify" / "tool-cache"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── pure-logic contract tests (no toolchain) ─────────────────────────

def _unit_root() -> Path:
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="a5unit-", dir=str(SESSION_ROOT)))


def _ref_dossier(refs: dict[str, int], package: str = "com.example.host",
                 defined: list[str] | None = None) -> dict:
    return {"identity": {"package": package},
            "dex": {"class_names": defined or [],
                    "referenced_packages": refs}}


class SignatureReferencedUnit(unittest.TestCase):
    def test_recurrence_threshold_config(self):
        from nomorals.agents.apkforge import signatures as sig

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOMORALS_APKFORGE_SIG_THRESHOLD", None)
            self.assertEqual(sig.recurrence_threshold(), 3)
        with mock.patch.dict(
                os.environ, {"NOMORALS_APKFORGE_SIG_THRESHOLD": "2"}):
            self.assertEqual(sig.recurrence_threshold(), 2)
        with mock.patch.dict(
                os.environ,
                {"NOMORALS_APKFORGE_SIG_THRESHOLD": "junk"}):
            self.assertEqual(sig.recurrence_threshold(), 3)
        with mock.patch.dict(
                os.environ, {"NOMORALS_APKFORGE_SIG_THRESHOLD": "0"}):
            self.assertEqual(sig.recurrence_threshold(), 3)

    def test_referenced_only_prefix_proposed_with_split(self):
        from nomorals.agents.apkforge import signatures as sig

        root = _unit_root()
        dossier = _ref_dossier({"com.acme.refs": 3})
        for i in range(3):
            sig.observe_dossier(root, f"proj{i}", dossier)
        props = {p["prefix"]: p for p in sig.proposals(root)}
        self.assertIn("com.acme.refs", props)
        row = props["com.acme.refs"]
        self.assertEqual(row["occurrences"], 3)
        self.assertEqual(row["dossiers"], ["proj0", "proj1", "proj2"])
        self.assertEqual(row["defined_classes_total"], 0)
        self.assertEqual(row["referenced_types_total"], 9)
        self.assertEqual(row["classes_total"], 9)
        # promotion carries the split as provenance
        entry = sig.promote(root, "com.acme.refs", label="Acme Refs",
                            kind="analytics")
        self.assertEqual(entry["provenance"]["defined_classes_total"],
                         0)
        self.assertEqual(entry["provenance"]["referenced_types_total"],
                         9)

    def test_threshold_config_governs_discovery_and_promotion(self):
        from nomorals.agents.apkforge import signatures as sig

        root = _unit_root()
        dossier = _ref_dossier({"com.acme.refs": 2})
        sig.observe_dossier(root, "p1", dossier)
        sig.observe_dossier(root, "p2", dossier)
        # default threshold 3: two dossiers is not a proposal yet
        self.assertEqual(sig.proposals(root), [])
        with self.assertRaises(sig.SignatureError) as cm:
            sig.promote(root, "com.acme.refs")
        self.assertIn("threshold 3", str(cm.exception))
        # config lowers it to 2: the same census is now a proposal
        # and the promotion gate honors the same value
        with mock.patch.dict(
                os.environ, {"NOMORALS_APKFORGE_SIG_THRESHOLD": "2"}):
            props = sig.proposals(root)
            self.assertEqual([p["prefix"] for p in props
                              if p["prefix"] == "com.acme.refs"],
                             ["com.acme.refs"])
            entry = sig.promote(root, "com.acme.refs")
            self.assertEqual(entry["origin"], "grown")

    def test_referenced_noise_floor(self):
        from nomorals.agents.apkforge import signatures as sig

        root = _unit_root()
        # a single referenced type per dossier is noise, not an SDK
        for i in range(4):
            sig.observe_dossier(root, f"p{i}",
                                _ref_dossier({"com.acme.thin": 1}))
        st = sig.signatures_status(root)
        prefixes = {c["prefix"] for c in st["candidates"]}
        self.assertNotIn("com.acme.thin", prefixes)
        self.assertNotIn("com.acme", prefixes)

    def test_mixed_defined_and_referenced_accumulate(self):
        from nomorals.agents.apkforge import signatures as sig

        root = _unit_root()
        defined = ["Lcom/acme/mix/Alpha;", "Lcom/acme/mix/Beta;"]
        d1 = {"identity": {"package": "com.example.host"},
              "dex": {"class_names": defined,
                      "referenced_packages": {}}}
        d2 = _ref_dossier({"com.acme.mix": 4})
        sig.observe_dossier(root, "p1", d1)
        sig.observe_dossier(root, "p2", d2)
        sig.observe_dossier(root, "p3", d2)
        props = {p["prefix"]: p for p in sig.proposals(root)}
        row = props["com.acme.mix"]
        self.assertEqual(row["defined_classes_total"], 2)
        self.assertEqual(row["referenced_types_total"], 8)
        self.assertEqual(row["classes_total"], 10)


class ComponentStateUnit(unittest.TestCase):
    def _dossier(self, rows, target=35):
        return {"identity": {"target_sdk": target},
                "components": {"activities": rows, "services": [],
                               "receivers": [], "providers": []}}

    def test_three_states_and_platform_default(self):
        from nomorals.agents.apkforge.bisect import component_state

        d = self._dossier([
            {"name": "com.x.On", "exported": True,
             "intent_filters": []},
            {"name": "com.x.Off", "exported": False,
             "intent_filters": []},
            {"name": "com.x.FilterDefault", "exported": None,
             "intent_filters": [{"actions": ["a"],
                                 "categories": []}]},
            {"name": "com.x.PlainDefault", "exported": None,
             "intent_filters": []}])
        self.assertEqual(component_state(d, "com.x.On"),
                         "present-exported")
        self.assertEqual(component_state(d, "com.x.Off"),
                         "present-not-exported")
        self.assertEqual(component_state(d, "com.x.FilterDefault"),
                         "present-exported")
        self.assertEqual(component_state(d, "com.x.PlainDefault"),
                         "present-not-exported")
        self.assertEqual(component_state(d, "com.x.Missing"),
                         "absent")
        # pre-31 target: a filter-bearing component with no
        # exported attribute defaults to NOT exported
        d30 = self._dossier([{"name": "com.x.F", "exported": None,
                              "intent_filters": [{"actions": ["a"],
                                                  "categories": []}]}],
                            target=30)
        self.assertEqual(component_state(d30, "com.x.F"),
                         "present-not-exported")
        # no component census at all → cannot judge
        self.assertIsNone(component_state({"components": None}, "x"))
        self.assertIsNone(component_state(None, "x"))

    def test_binary_predicate_derives_from_state(self):
        from nomorals.agents.apkforge.bisect import (
            ABSENT, FOUND, evaluate_predicate, parse_predicate)

        d = self._dossier([
            {"name": "com.x.On", "exported": True,
             "intent_filters": []},
            {"name": "com.x.Off", "exported": False,
             "intent_filters": []}])
        self.assertEqual(evaluate_predicate(
            d, parse_predicate("exported: com.x.On")), FOUND)
        self.assertEqual(evaluate_predicate(
            d, parse_predicate("exported: com.x.Off")), ABSENT)
        self.assertEqual(evaluate_predicate(
            d, parse_predicate("exported: com.x.Missing")), ABSENT)

    def test_damaged_dossier_skips_everything(self):
        from nomorals.agents.apkforge.bisect import (
            SKIP, dossier_damage, evaluate_predicate, parse_predicate)

        dmg = {"dex": {"errors": ["classes.dex: ValueError: "
                                  "Not a DEX file, Header too "
                                  "small."]},
               "permissions": [{"name": "android.permission.CAMERA"}],
               "components": {"activities": [
                   {"name": "com.x.On", "exported": True}],
                   "services": [], "receivers": [], "providers": []}}
        self.assertIn("Not a DEX file", dossier_damage(dmg))
        self.assertEqual(evaluate_predicate(
            dmg, parse_predicate(
                "permission: android.permission.CAMERA")), SKIP)
        self.assertEqual(evaluate_predicate(
            dmg, parse_predicate("exported: com.x.On")), SKIP)


class HalveStatesUnit(unittest.TestCase):
    def _run(self, truth, known=None):
        from nomorals.agents.apkforge.bisect import halve_states

        return halve_states(list(truth), lambda it: truth[it],
                            known=known)

    def test_absent_to_exported(self):
        res = self._run({"a": "absent", "b": "absent",
                         "c": "present-exported",
                         "d": "present-exported"})
        self.assertEqual(res["status"], "flip")
        self.assertEqual(len(res["transitions"]), 1)
        t = res["transitions"][0]
        self.assertEqual((t["older_state"], t["newer_state"]),
                         ("absent", "present-exported"))
        self.assertEqual((t["older"], t["newer"]), ("b", "c"))
        self.assertTrue(t["exact"])

    def test_absent_to_not_exported_is_a_transition(self):
        # the A3 conflation, pinned shut: the component APPEARING
        # unexported is a finding, not "absent throughout"
        res = self._run({"a": "absent", "b": "present-not-exported",
                         "c": "present-not-exported"})
        self.assertEqual(res["status"], "flip")
        t = res["transitions"][0]
        self.assertEqual((t["older_state"], t["newer_state"]),
                         ("absent", "present-not-exported"))

    def test_not_exported_to_exported(self):
        res = self._run({"a": "present-not-exported",
                         "b": "present-not-exported",
                         "c": "present-exported"})
        self.assertEqual(res["status"], "flip")
        t = res["transitions"][0]
        self.assertEqual((t["older_state"], t["newer_state"]),
                         ("present-not-exported", "present-exported"))
        self.assertEqual((t["older"], t["newer"]), ("b", "c"))

    def test_two_step_series_names_both_crossings(self):
        res = self._run({"a": "absent", "b": "present-not-exported",
                         "c": "present-exported",
                         "d": "present-exported"})
        self.assertEqual(res["status"], "flip")
        self.assertEqual(
            [(t["older_state"], t["newer_state"])
             for t in res["transitions"]],
            [("absent", "present-not-exported"),
             ("present-not-exported", "present-exported")])
        self.assertTrue(all(t["exact"] for t in res["transitions"]))

    def test_skip_brackets_state_crossing(self):
        res = self._run({"a": "absent", "b": "skip",
                         "c": "present-exported"})
        self.assertEqual(res["status"], "flip_bracketed")
        t = res["transitions"][0]
        self.assertEqual((t["older"], t["newer"]), ("a", "c"))
        self.assertFalse(t["exact"])
        self.assertEqual(res["skipped"], ["b"])

    def test_resume_from_known_states(self):
        known = {"a": "absent", "b": "present-not-exported",
                 "c": "present-exported"}

        def judge(it):  # pragma: no cover — must never be called
            raise AssertionError(f"re-judged {it} on resume")

        from nomorals.agents.apkforge.bisect import halve_states

        res = halve_states(list(known), judge, known=known)
        self.assertEqual(res["probes"], 0)
        self.assertEqual(len(res["transitions"]), 2)

    def test_state_throughout(self):
        res = self._run({"a": "present-not-exported",
                         "b": "present-not-exported"})
        self.assertEqual(res["status"], "not_exported_throughout")
        self.assertEqual(res["transitions"], [])


class ElfInfoUnit(unittest.TestCase):
    def test_non_elf_and_truncated(self):
        from nomorals.agents.apkforge.dissect import elf_info

        self.assertIsNone(elf_info(b""))
        self.assertIsNone(elf_info(b"just text, not an elf"))
        self.assertIsNone(elf_info(b"\x7fELF" + b"\x00" * 8))


if __name__ == "__main__":
    unittest.main()


# ── integration fixtures ─────────────────────────────────────────────

_FIXTURE_C = r"""
/* ApkForge A5 fixture library — a REAL native library: exported
 * functions that call each other (PLT relocations), an imported
 * libc call, a .rodata version string, and a magic constant baked
 * into one function's instruction stream. */
#include <stddef.h>
#include <string.h>

const char fixture_version_string[] =
    "apkforge-fixture-lib v5.0 - real ELF, no synthetic bytes";
static const char fixture_rodata_marker[] = "fixture-rodata-marker";

extern unsigned fixture_external_counter;

static unsigned helper_triple(unsigned x)
{
    return x * 3u + (unsigned)fixture_rodata_marker[0];
}

__attribute__((noinline))
unsigned fixture_magic(void)
{
    return 0xC0FFEE11u;
}

unsigned fixture_compute(unsigned seed)
{
    unsigned v = helper_triple(seed) + fixture_magic();
    v += (unsigned)strlen(fixture_version_string);
    v += fixture_external_counter;
    return v;
}

const char *fixture_version(void)
{
    return fixture_version_string;
}
"""

_APKTOOL_YML = """version: 2.10.0
apkFileName: fixture.apk
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
  versionCode: {vcode}
  versionName: '{vname}'
doNotCompress:
- arsc
"""

_MAIN_SMALI_BASIC = """.class public Lcom/example/fixture/MainActivity;
.super Landroid/app/Activity;
.source "MainActivity.java"

.method public constructor <init>()V
    .locals 0
    invoke-direct {p0}, Landroid/app/Activity;-><init>()V
    return-void
.end method
"""

# MainActivity that CALLS two classes of an SDK it does not define
_REFS_SMALI = """.class public Lcom/example/fixture/MainActivity;
.super Landroid/app/Activity;
.source "MainActivity.java"

.method public constructor <init>()V
    .locals 0
    invoke-direct {p0}, Landroid/app/Activity;-><init>()V
    return-void
.end method

.method public static pingRefs()V
    .locals 0
    invoke-static {}, Lcom/acme/refs/SdkAlpha;->ping()V
    invoke-static {}, Lcom/acme/refs/SdkBeta;->ping()V
    return-void
.end method
"""

_SECRET_SMALI = """.class public Lcom/example/fixture/SecretActivity;
.super Landroid/app/Activity;
.source "SecretActivity.java"

.method public constructor <init>()V
    .locals 0
    invoke-direct {p0}, Landroid/app/Activity;-><init>()V
    return-void
.end method
"""

_STRINGS_XML = """<?xml version="1.0" encoding="utf-8"?>
<resources>
    <string name="app_name">Fixture App</string>
</resources>
"""

_SERIES_MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="dev.apkforge.series" android:versionCode="{vcode}" android:versionName="{vcode}.0">
    <uses-sdk android:minSdkVersion="24" android:targetSdkVersion="35"/>
{perms}    <application android:label="@string/app_name">
        <activity android:name=".MainActivity" android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN"/>
                <category android:name="android.intent.category.LAUNCHER"/>
            </intent-filter>
        </activity>
    </application>
</manifest>
"""

_SERIES_ACTIVITY = """package dev.apkforge.series;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

public class MainActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        TextView tv = new TextView(this);
        tv.setText("series");
        setContentView(tv);
    }
}
"""

_NATIVE_MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="dev.apkforge.nativelib" android:versionCode="1" android:versionName="1.0">
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

_NATIVE_ACTIVITY = _SERIES_ACTIVITY.replace("dev.apkforge.series",
                                            "dev.apkforge.nativelib")


class ApkForgeA5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved_env = {k: os.environ.get(k) for k in
                          ("NM_VAULT_PASSPHRASE",
                           "APKFORGE_TOOL_CACHE")}
        os.environ["NM_VAULT_PASSPHRASE"] = "apkforge-a5-test-seal"
        if TOOL_CACHE.exists():
            os.environ["APKFORGE_TOOL_CACHE"] = str(TOOL_CACHE)
        SESSION_ROOT.mkdir(parents=True, exist_ok=True)
        cls.root = SESSION_ROOT / "apkforge"
        shutil.rmtree(cls.root, ignore_errors=True)
        cls.root.mkdir(parents=True)

        from nomorals.agents.apkforge import ApkForge
        from nomorals.storage.db import Database

        cls.db = Database(str(SESSION_ROOT / "vault.db"))
        cls.ctx = SimpleNamespace(
            db=cls.db,
            settings=SimpleNamespace(workspace_dir=str(SESSION_ROOT)),
            extras={})
        cls.forge = ApkForge(cls.ctx, base_dir=str(cls.root))
        prov = cls.forge.toolchain_provision()
        assert not prov.get("error"), prov
        prov_src = cls.forge.toolchain_provision_srcbuild()
        assert not prov_src.get("error"), prov_src
        cls._seq = 0

        # ── the REAL compiled native libraries ──────────────────
        assert shutil.which("gcc"), "host cc required for A5"
        assert shutil.which("aarch64-linux-gnu-gcc"), \
            "aarch64 cross cc required for A5 (gcc-aarch64-linux-gnu)"
        lib_src = SESSION_ROOT / "lib-src"
        shutil.rmtree(lib_src, ignore_errors=True)
        lib_src.mkdir(parents=True)
        c_path = lib_src / "fixture_lib.c"
        c_path.write_text(_FIXTURE_C)
        cls.so_bytes = {}
        for cc, abi in (("gcc", "x86_64"),
                        ("aarch64-linux-gnu-gcc", "arm64-v8a")):
            out = lib_src / abi / "libfixture.so"
            out.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                [cc, "-shared", "-fPIC", "-O2", "-o", str(out),
                 str(c_path)], capture_output=True, text=True)
            assert proc.returncode == 0, proc.stderr
            cls.so_bytes[abi] = out.read_bytes()

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # ── helpers ──────────────────────────────────────────────────
    @classmethod
    def _apktool_build(cls, dest: Path, *, manifest: str,
                       smali_files: dict[str, str]) -> str:
        from nomorals.agents.apkforge import signing as signing_mod
        from nomorals.agents.apkforge.toolchain import run_tool

        cls._seq += 1
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "AndroidManifest.xml").write_text(manifest)
        import re as _re

        vm = _re.search(r'android:versionCode="(\d+)"', manifest)
        vn = _re.search(r'android:versionName="([^"]+)"', manifest)
        (dest / "apktool.yml").write_text(
            _APKTOOL_YML.format(
                vcode=vm.group(1) if vm else "1",
                vname=vn.group(1) if vn else "1.0"))
        for rel, text in smali_files.items():
            p = dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        values = dest / "res" / "values"
        values.mkdir(parents=True, exist_ok=True)
        (values / "strings.xml").write_text(_STRINGS_XML)
        unsigned = dest / "unsigned.apk"
        res = run_tool("", "apktool",
                       ["b", str(dest), "-o", str(unsigned)],
                       root=cls.forge.root)
        assert res["rc"] == 0, res["stderr"][-400:]
        signed = dest / "signed.apk"
        signing_mod.sign_apk(cls.ctx, cls.forge.root, unsigned,
                             signed)
        return str(signed)

    @classmethod
    def _fixture_manifest(cls, *, vcode: int, vname: str,
                          extra_activity: str = "") -> str:
        head = f"""<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="com.example.fixture" android:versionCode="{vcode}" android:versionName="{vname}">
    <uses-sdk android:minSdkVersion="24" android:targetSdkVersion="34"/>
    <application android:label="@string/app_name">
        <activity android:name=".MainActivity" android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN"/>
                <category android:name="android.intent.category.LAUNCHER"/>
            </intent-filter>
        </activity>
{extra_activity}    </application>
</manifest>
"""
        return head

    # ── item 3: real ELF fixtures ────────────────────────────────
    def test_compiled_fixture_is_a_real_linked_elf(self):
        from nomorals.agents.apkforge.dissect import elf_info

        for abi, machine in (("x86_64", "x86-64"),
                             ("arm64-v8a", "AArch64")):
            info = elf_info(self.so_bytes[abi])
            self.assertIsNotNone(info, abi)
            self.assertEqual(info["machine"], machine)
            self.assertEqual(info["elf_class"], 64)
            self.assertEqual(info["object_type"], "shared object")
            for section in (".text", ".rodata", ".dynsym"):
                self.assertIn(section, info["sections"], abi)
            self.assertTrue(info["has_relocations"], abi)
            self.assertGreater(info["relocations"]["count"], 0)
            for sym in ("fixture_magic", "fixture_compute",
                        "fixture_version_string"):
                self.assertIn(sym, info["dynamic_symbols"], abi)
            self.assertIn(b"apkforge-fixture-lib v5.0",
                          self.so_bytes[abi])

    def test_native_census_and_hexpatch_round_trip(self):
        from nomorals.agents.apkforge import srcbuild as srcbuild_mod
        from nomorals.agents.apkforge.dissect import elf_info

        src = SESSION_ROOT / "native-src"
        shutil.rmtree(src, ignore_errors=True)
        pkg = src / "java" / "dev" / "apkforge" / "nativelib"
        pkg.mkdir(parents=True)
        (src / "res" / "values").mkdir(parents=True)
        (src / "AndroidManifest.xml").write_text(_NATIVE_MANIFEST)
        (src / "res" / "values" / "strings.xml").write_text(
            _STRINGS_XML)
        (pkg / "MainActivity.java").write_text(_NATIVE_ACTIVITY)
        for abi, data in self.so_bytes.items():
            lib = src / "lib" / abi / "libfixture.so"
            lib.parent.mkdir(parents=True, exist_ok=True)
            lib.write_bytes(data)
        built = srcbuild_mod.build_source(self.ctx, self.root, src)
        self.assertEqual(built["route"], "manual")
        apk = Path(built["artifacts"][0]["path"])
        with zipfile.ZipFile(apk) as zf:
            names = set(zf.namelist())
        self.assertIn("lib/x86_64/libfixture.so", names)
        self.assertIn("lib/arm64-v8a/libfixture.so", names)

        pid = self.forge.add(str(apk))["project"]["id"]
        diss = self.forge.dissect(pid)
        self.assertNotIn("error", diss, diss)
        native = {n["abi"]: n for n in diss["dossier"]["native"]}
        self.assertEqual(set(native), {"x86_64", "arm64-v8a"})
        self.assertEqual(native["x86_64"]["elf"]["machine"],
                         "x86-64")
        self.assertEqual(native["arm64-v8a"]["elf"]["machine"],
                         "AArch64")
        self.assertIn("R_X86_64_JUMP_SLOT",
                      native["x86_64"]["elf"]["relocations"]
                      ["types"])
        self.assertIn("fixture_magic",
                      native["x86_64"]["elf"]["dynamic_symbols"])

        # hexpatch: wildcard pattern spanning the magic constant
        dec = self.forge.decode(pid)
        self.assertIn("decoded", dec, dec)
        so = (self.root / "projects" / pid / "decoded" / "lib"
              / "x86_64" / "libfixture.so")
        before = so.read_bytes()
        self.assertEqual(before, self.so_bytes["x86_64"])
        doc = {"name": "retune-magic",
               "target": "lib/x86_64/libfixture.so",
               "match": {"pattern": "b8 ?? ee ff c0 c3",
                         "expect_matches": 1},
               "replace": "b8 22 ee ff c0 c3"}
        res = self.forge.hexpatch(pid, json.dumps(doc))
        self.assertIn("hexpatch_result", res, res)
        patched = so.read_bytes()
        self.assertEqual(elf_info(patched)["dynamic_symbol_count"],
                         elf_info(before)["dynamic_symbol_count"])
        self.assertIn(bytes.fromhex("22eeffc0"), patched)
        self.assertNotIn(bytes.fromhex("11eeffc0"), patched)
        # the patched lib lands in a rebuilt, signed APK
        rebuilt = self.forge.build(pid)
        self.assertIn("build", rebuilt, rebuilt)
        self.assertTrue(rebuilt["build"]["verify"]["verifies"])
        with zipfile.ZipFile(rebuilt["build"]["output"]) as zf:
            data = zf.read("lib/x86_64/libfixture.so")
        self.assertIn(bytes.fromhex("22eeffc0"), data)
        add2 = self.forge.add(rebuilt["build"]["output"])
        diss2 = self.forge.dissect(add2["project"]["id"])
        row = {n["abi"]: n
               for n in diss2["dossier"]["native"]}["x86_64"]
        self.assertEqual(row["elf"]["machine"], "x86-64")
        self.assertTrue(row["elf"]["has_relocations"])
        # undo restores the compiled bytes exactly
        un = self.forge.undo(
            pid, res["hexpatch_result"]["edit_id"])
        self.assertIn("undo", un, un)
        self.assertEqual(so.read_bytes(), before)

    # ── item 1 (integration): referenced-only SDK grows the table ─
    def test_referenced_sdk_discovery_promotion(self):
        sig_root = SESSION_ROOT / "sig5-apkforge"
        shutil.rmtree(sig_root, ignore_errors=True)
        from nomorals.agents.apkforge import ApkForge

        forge = ApkForge(self.ctx, base_dir=str(sig_root))
        work = SESSION_ROOT / "refs-fixtures"
        shutil.rmtree(work, ignore_errors=True)
        ids = []
        for i in range(1, 5):
            path = self._apktool_build(
                work / f"v{i}",
                manifest=self._fixture_manifest(vcode=i,
                                                vname=f"{i}.0"),
                smali_files={
                    "smali/com/example/fixture/MainActivity.smali":
                        _REFS_SMALI})
            ids.append(forge.add(path)["project"]["id"])
        # the fixture defines NO com.acme.refs classes — dissect
        # proves the census shape before discovery reads it
        diss = forge.dissect(ids[0])
        dex = diss["dossier"]["dex"]
        self.assertFalse(any("acme" in c
                             for c in dex["class_names"]))
        self.assertEqual(
            dex["referenced_packages"].get("com.acme.refs"), 2)
        forge.dissect(ids[1])
        forge.dissect(ids[2])
        st = forge.signatures("list")["signatures"]
        props = {p["prefix"]: p for p in st["proposals"]}
        self.assertIn("com.acme.refs", props)
        self.assertEqual(props["com.acme.refs"]["occurrences"], 3)
        self.assertEqual(
            props["com.acme.refs"]["defined_classes_total"], 0)
        self.assertGreaterEqual(
            props["com.acme.refs"]["referenced_types_total"], 6)
        res = forge.signatures("promote", prefix="com.acme.refs",
                               label="Acme Refs SDK", kind="analytics")
        self.assertIn("promoted", res, res)
        self.assertEqual(
            res["promoted"]["provenance"]["referenced_types_total"],
            props["com.acme.refs"]["referenced_types_total"])
        # the grown entry now MATCHES on references alone
        diss = forge.dissect(ids[3])
        hits = {s["name"]: s
                for s in diss["dossier"]["sdk_signatures"]}
        self.assertIn("Acme Refs SDK", hits)
        self.assertEqual(hits["Acme Refs SDK"]["origin"], "grown")
        self.assertGreaterEqual(hits["Acme Refs SDK"]["class_count"],
                                2)

    # ── item 2 (integration): exported state series ──────────────
    def test_exported_state_series_bisect(self):
        work = SESSION_ROOT / "exported-fixtures"
        shutil.rmtree(work, ignore_errors=True)
        variants = [
            (1, "", {}),
            (2, '        <activity android:name=".SecretActivity" '
                'android:exported="false"/>\n',
             {"smali/com/example/fixture/SecretActivity.smali":
              _SECRET_SMALI}),
            (3, '        <activity android:name=".SecretActivity" '
                'android:exported="true"/>\n',
             {"smali/com/example/fixture/SecretActivity.smali":
              _SECRET_SMALI}),
        ]
        ids = []
        for vcode, extra, extra_smali in variants:
            smali = {"smali/com/example/fixture/MainActivity.smali":
                     _MAIN_SMALI_BASIC}
            smali.update(extra_smali)
            path = self._apktool_build(
                work / f"v{vcode}",
                manifest=self._fixture_manifest(
                    vcode=vcode, vname=f"{vcode}.0",
                    extra_activity=extra),
                smali_files=smali)
            ids.append(self.forge.add(path)["project"]["id"])
        res = self.forge.bisect(
            ids, "exported: com.example.fixture.SecretActivity")
        self.assertNotIn("error", res, res)
        self.assertEqual(res["status"], "flip")
        self.assertEqual(
            [(t["older_state"], t["newer_state"])
             for t in res["transitions"]],
            [("absent", "present-not-exported"),
             ("present-not-exported", "present-exported")])
        self.assertIn("absent → present-not-exported",
                      res["card_text"])
        self.assertIn("present-not-exported → present-exported",
                      res["card_text"])
        # the direct jump, as its own two-version series
        res2 = self.forge.bisect(
            [ids[0], ids[2]],
            "exported: com.example.fixture.SecretActivity")
        self.assertNotIn("error", res2, res2)
        self.assertEqual(
            [(t["older_state"], t["newer_state"])
             for t in res2["transitions"]],
            [("absent", "present-exported")])

    # ── item 4: bisect skip against a real unparseable APK ───────
    def test_bisect_skip_real_corrupt_dex(self):
        from nomorals.agents.apkforge import srcbuild as srcbuild_mod

        work = SESSION_ROOT / "series-src"
        shutil.rmtree(work, ignore_errors=True)
        ids = []
        for vcode in (1, 2, 3, 4):
            src = work / f"v{vcode}"
            pkg = src / "java" / "dev" / "apkforge" / "series"
            pkg.mkdir(parents=True)
            (src / "res" / "values").mkdir(parents=True)
            perms = ('    <uses-permission '
                     'android:name="android.permission.CAMERA"/>\n'
                     if vcode >= 3 else "")
            (src / "AndroidManifest.xml").write_text(
                _SERIES_MANIFEST.format(vcode=vcode, perms=perms))
            (src / "res" / "values" / "strings.xml").write_text(
                _STRINGS_XML)
            (pkg / "MainActivity.java").write_text(
                _SERIES_ACTIVITY)
            built = srcbuild_mod.build_source(self.ctx, self.root,
                                              src)
            apk = Path(built["artifacts"][0]["path"])
            if vcode == 2:
                # corrupt the dex FOR REAL: truncate classes.dex
                # inside the built APK — androguard's parser
                # records the failure in the dossier's dex census
                corrupt = work / "v2-corrupt.apk"
                with zipfile.ZipFile(apk) as zin, \
                        zipfile.ZipFile(corrupt, "w",
                                         zipfile.ZIP_DEFLATED) as zout:
                    for info in zin.infolist():
                        data = zin.read(info.filename)
                        if info.filename == "classes.dex":
                            data = data[:100]
                        zout.writestr(info, data)
                apk = corrupt
            ids.append(self.forge.add(str(apk))["project"]["id"])
        # the damaged specimen's dossier carries the real error
        dmg = self.forge.dissect(ids[1])
        self.assertNotIn("error", dmg, dmg)
        self.assertTrue(dmg["dossier"]["dex"]["errors"])
        self.assertIn("Not a DEX file",
                      dmg["dossier"]["dex"]["errors"][0])

        spec = "permission: android.permission.CAMERA"
        res = self.forge.bisect(ids, spec)
        self.assertNotIn("error", res, res)
        self.assertEqual(res["status"], "flip_bracketed")
        self.assertEqual(res["skipped"], [ids[1]])
        t = res["transition"]
        self.assertEqual(res["versions"][t["older"]]["version_code"],
                         1)
        self.assertEqual(res["versions"][t["newer"]]["version_code"],
                         3)
        self.assertIn(ids[1], res["skip_reasons"])
        self.assertIn("Not a DEX file", res["skip_reasons"][ids[1]])
        self.assertIn("Not a DEX file", res["card_text"])
        # the session journal shows the skip WITH its reason
        log = (self.root / "bisects" / res["session"]
               / "log.jsonl").read_text(encoding="utf-8")
        probes = [json.loads(ln) for ln in log.splitlines()
                  if ln.strip()
                  and json.loads(ln)["op"] == "probe"]
        skip_rows = [r for r in probes
                     if r["project"] == ids[1]]
        self.assertEqual(len(skip_rows), 1)
        self.assertEqual(skip_rows[0]["verdict"], "skip")
        self.assertIn("Not a DEX file", skip_rows[0]["detail"])
        # resume: nothing re-judged, the skip and its reason stand
        res2 = self.forge.bisect(ids, spec)
        self.assertTrue(res2["resumed"])
        self.assertEqual(res2["probes"], 0)
        self.assertEqual(res2["skipped"], [ids[1]])
        self.assertIn("Not a DEX file",
                      res2["skip_reasons"][ids[1]])
        self.assertEqual(res2["transition"], res["transition"])

    # ── item 5: teach v2 ─────────────────────────────────────────
    @classmethod
    def _r8_project(cls):
        from nomorals.agents.apkforge import srcbuild as srcbuild_mod

        if hasattr(cls, "_r8_pid"):
            return cls._r8_pid, cls._r8_mapping
        src = SESSION_ROOT / "r8-src"
        shutil.rmtree(src, ignore_errors=True)
        srcbuild_mod.scaffold_project(
            src, "A5 Shrink", package="dev.apkforge.shrink")
        built = srcbuild_mod.build_source(
            cls.ctx, cls.root, src / "a5-shrink", use_r8=True)
        cls._r8_mapping = built["mapping"]
        add = cls.forge.add(built["artifacts"][0]["path"])
        cls._r8_pid = add["project"]["id"]
        return cls._r8_pid, cls._r8_mapping

    def test_teach_v2_r8_side_by_side(self):
        import nomorals.llm.brain as brain_mod
        from nomorals.agents.apkforge import teach as teach_mod

        pid, mapping = self._r8_project()
        names = teach_mod.parse_mapping(
            Path(mapping).read_text())
        obf = next(o for o, orig in names.items()
                   if orig.endswith("MainActivity$1"))
        with mock.patch.object(brain_mod, "brain_for",
                               return_value=None):
            res = self.forge.teach(pid, target_class=obf,
                                   method="onClick",
                                   mapping=mapping)
        self.assertNotIn("error", res, res)
        self.assertEqual(res["version"], 2)
        self.assertEqual(res["source"], "side-by-side")
        self.assertEqual(res["original_class"],
                         "dev.apkforge.shrink.MainActivity$1")
        self.assertEqual(res["method"], "onClick")
        text = res["text"]
        # the mapping-restored name, the real smali, the real Java
        self.assertIn("dev.apkforge.shrink.MainActivity$1", text)
        self.assertIn("invoke", text)
        self.assertIn("onClick", text)
        self.assertIn("```smali", text)
        self.assertIn("```java", text)

    def test_teach_v2_brain_must_cite_real_ops(self):
        import nomorals.llm.brain as brain_mod

        pid, mapping = self._r8_project()
        from nomorals.agents.apkforge import teach as teach_mod

        names = teach_mod.parse_mapping(
            Path(mapping).read_text())
        obf = next(o for o, orig in names.items()
                   if orig.endswith("MainActivity$1"))
        good = SimpleNamespace(chat=lambda msgs: SimpleNamespace(
            text=("In onClick the smali does the real work: iget "
                  "reads the counter field, iput writes the "
                  "incremented value back, invoke-virtual "
                  "dispatches the setText call on the label, and "
                  "the Java merely narrates the same steps in "
                  "friendlier clothes. " * 10)
            + "onClick iget iput invoke-virtual"))
        with mock.patch.object(brain_mod, "brain_for",
                               return_value=good):
            res = self.forge.teach(pid, target_class=obf,
                                   method="onClick",
                                   mapping=mapping)
        self.assertEqual(res["source"], "brain", res)
        # a brain answer that cites nothing real is rejected to
        # the grounded fallback
        vague = SimpleNamespace(chat=lambda msgs: SimpleNamespace(
            text="word " * 150))
        with mock.patch.object(brain_mod, "brain_for",
                               return_value=vague):
            res = self.forge.teach(pid, target_class=obf,
                                   method="onClick",
                                   mapping=mapping)
        self.assertEqual(res["source"], "side-by-side", res)

    def test_teach_v2_plain_project_and_v1_intact(self):
        import nomorals.llm.brain as brain_mod
        from nomorals.agents.apkforge import srcbuild as srcbuild_mod

        src = SESSION_ROOT / "teach-src"
        shutil.rmtree(src, ignore_errors=True)
        srcbuild_mod.scaffold_project(
            src, "A5 Counter", package="dev.apkforge.counter")
        built = srcbuild_mod.build_source(
            self.ctx, self.root, src / "a5-counter")
        pid = self.forge.add(
            built["artifacts"][0]["path"])["project"]["id"]
        with mock.patch.object(brain_mod, "brain_for",
                               return_value=None):
            res = self.forge.teach(
                pid, target_class="dev.apkforge.counter.MainActivity",
                method="onCreate")
            v1 = self.forge.teach(pid, "overview")
        self.assertNotIn("error", res, res)
        self.assertEqual(res["version"], 2)
        self.assertEqual(res["source"], "side-by-side")
        self.assertIn("onCreate", res["text"])
        self.assertIn("invoke-super", res["text"])
        # v1 lessons are byte-for-byte the old behavior's shape
        self.assertEqual(v1["source"], "dossier")
        self.assertNotIn("version", v1)
        # honest refusals
        res = self.forge.teach(
            pid, target_class="dev.apkforge.counter.MainActivity",
            method="noSuchMethod")
        self.assertIn("error", res)
        self.assertIn("noSuchMethod", res["error"])
        res = self.forge.teach(pid,
                               target_class="dev.apkforge.Nope")
        self.assertIn("error", res)

    def test_teach_v2_surfaces(self):
        from nomorals.agents.apkforge import register

        pid, mapping = self._r8_project()
        from nomorals.agents.apkforge import teach as teach_mod

        names = teach_mod.parse_mapping(
            Path(mapping).read_text())
        obf = next(o for o, orig in names.items()
                   if orig.endswith("MainActivity$1"))
        captured = {}

        class _Reg:
            context = self.ctx

            def register(self, name, **kw):
                def deco(fn):
                    captured[name] = (kw, fn)
                    return fn
                return deco

        register(_Reg())
        read_fn = captured["apkforge"][1]
        import nomorals.llm.brain as brain_mod

        with mock.patch.object(brain_mod, "brain_for",
                               return_value=None):
            out = read_fn(action="teach", project=pid,
                          target_class=obf, method="onClick",
                          mapping=mapping)
        self.assertEqual(out.get("version"), 2, out)
        self.assertEqual(out.get("method"), "onClick")
        # nm apk CLI carries the v2 form too
        from nomorals.cmdline.commands.apk import _cmd_apk

        args = SimpleNamespace(
            apk_action="teach", project=pid, topic=None,
            target_class=obf, method="onClick", mapping=mapping)
        buf = io.StringIO()
        with mock.patch.object(brain_mod, "brain_for",
                               return_value=None):
            with redirect_stdout(buf):
                rc = _cmd_apk(args, self.ctx)
        self.assertEqual(rc, 0, buf.getvalue())
        self.assertIn("MainActivity$1", buf.getvalue())
