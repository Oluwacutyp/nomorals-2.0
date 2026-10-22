"""ApkForge A2 — editor + builder + signing + splits.

Fixtures are REAL toolchain products: a minimal apktool source
project (manifest + smali + strings) is built by the provisioned
apktool itself and signed through the class's own vault keystore,
so the "original" the tests land is a genuine signed APK.  Every
claim is then checked against the A1 dissection of the outputs —
the permission an edit added must appear in the rebuilt APK's
dossier, the version diff must show it, undo must restore exact
bytes, and the keystore must provably never touch the project tree.

The toolchain provisions once into a shared root under the
workspace (APKFORGE_A2_TEST_ROOT overrides it); per-test projects
are built as unique variants so tests never share mutable state.

What this does NOT prove: real-world APKs (multidex, R8-obfuscated,
framework-dependent), a valid .aab fixture (none could be
fabricated without Gradle/AGP — the AAB path is error-path tested
against real bundletool), and install behavior on a device (out of
scope by design).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

_WS = Path(os.path.expanduser("~/workspace"))
SESSION_ROOT = Path(os.environ.get(
    "APKFORGE_A2_TEST_ROOT", str(_WS / "a2-verify" / "a2-test-root")))
TOOL_CACHE = _WS / "a2-verify" / "tool-cache"

_MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="com.example.fixture" android:versionCode="1" android:versionName="{vname}">
    <uses-sdk android:minSdkVersion="24" android:targetSdkVersion="34"/>
    <uses-permission android:name="android.permission.INTERNET"/>
    <application android:label="@string/app_name" android:debuggable="true">
        <activity android:name=".MainActivity" android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN"/>
                <category android:name="android.intent.category.LAUNCHER"/>
            </intent-filter>
        </activity>
    </application>
</manifest>
"""

_YML = """version: 2.10.0
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
  versionCode: 1
  versionName: '{vname}'
doNotCompress:
- arsc
"""

_SMALI = """.class public Lcom/example/fixture/MainActivity;
.super Landroid/app/Activity;
.source "MainActivity.java"

.method public constructor <init>()V
    .locals 0
    invoke-direct {p0}, Landroid/app/Activity;-><init>()V
    return-void
.end method

.method public checkLicense()Z
    .locals 1
    const-string v0, "license"
    invoke-virtual {v0}, Ljava/lang/String;->length()I
    move-result v0
    if-eqz v0, :cond_0
    const/4 v0, 0x1
    return v0
    :cond_0
    const/4 v0, 0x0
    return v0
.end method
"""

_STRINGS = """<?xml version="1.0" encoding="utf-8"?>
<resources>
    <string name="app_name">Fixture App</string>
</resources>
"""

_PATCH = {
    "name": "license-always-true",
    "match": {"class": "com/example/fixture/MainActivity",
              "method": "checkLicense", "prototype": "()Z",
              "fingerprint": {"strings": ["license"],
                              "opcodes": ["invoke-virtual",
                                          "move-result"]}},
    "op": "return_const", "value": "1", "expect_matches": 1,
}


def _write_skeleton(dest: Path, vname: str, *,
                    split: str = "") -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    manifest = _MANIFEST.format(vname=vname)
    if split:
        manifest = manifest.replace(
            'package="com.example.fixture"',
            f'package="com.example.fixture" split="{split}"')
    (dest / "AndroidManifest.xml").write_text(manifest)
    (dest / "apktool.yml").write_text(_YML.format(vname=vname))
    smali_dir = dest / "smali" / "com" / "example" / "fixture"
    smali_dir.mkdir(parents=True, exist_ok=True)
    (smali_dir / "MainActivity.smali").write_text(_SMALI)
    values = dest / "res" / "values"
    values.mkdir(parents=True, exist_ok=True)
    (values / "strings.xml").write_text(_STRINGS)
    return dest


class ApkForgeA2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved_env = {k: os.environ.get(k) for k in
                          ("NM_VAULT_PASSPHRASE", "APKFORGE_TOOL_CACHE")}
        os.environ["NM_VAULT_PASSPHRASE"] = "apkforge-a2-test-seal"
        if TOOL_CACHE.exists():
            os.environ["APKFORGE_TOOL_CACHE"] = str(TOOL_CACHE)
        SESSION_ROOT.mkdir(parents=True, exist_ok=True)
        for stale in SESSION_ROOT.glob("a2fix-*"):
            shutil.rmtree(stale, ignore_errors=True)
        # the forge root is SESSION_ROOT/"apkforge" so the base-dir
        # construction (tests) and the context-workspace construction
        # (chat/CLI surfaces) resolve to the SAME root
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
        cls._variant_seq = 0

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # ── fixture factory ──────────────────────────────────────────
    def _make_project(self, *, split: str = "") -> str:
        """Build + sign a fresh variant APK with the real chain and
        land it; returns the new project id."""
        from nomorals.agents.apkforge import signing as signing_mod
        from nomorals.agents.apkforge.toolchain import run_tool

        type(self)._variant_seq += 1
        tag = f"t{type(self)._variant_seq}"
        work = Path(tempfile.mkdtemp(prefix=f"a2fix-{tag}-",
                                     dir=str(SESSION_ROOT)))
        skel = _write_skeleton(work / "skel", f"1.0-{tag}",
                               split=split)
        unsigned = work / "unsigned.apk"
        res = run_tool("", "apktool",
                       ["b", str(skel), "-o", str(unsigned)],
                       root=self.forge.root)
        self.assertEqual(res["rc"], 0, res["stderr"][-400:])
        signed = work / "signed.apk"
        signing_mod.sign_apk(self.ctx, self.forge.root, unsigned,
                             signed)
        add = self.forge.add(str(signed))
        self.assertIn("project", add, add)
        return add["project"]["id"]

    def _smali_path(self, pid: str) -> Path:
        return (self.forge.root / "projects" / pid / "decoded"
                / "smali" / "com" / "example" / "fixture"
                / "MainActivity.smali")

    # ── toolchain ────────────────────────────────────────────────
    def test_provision_pins_and_capabilities(self):
        from nomorals.agents.apkforge.toolchain import PINS, \
            toolchain_state

        state = toolchain_state(root=self.forge.root)
        self.assertTrue(state.get("provisioned_at"))
        for name, pin in PINS.items():
            self.assertIn(name, state["artifacts"])
            self.assertEqual(state["artifacts"][name]["version"],
                             pin["version"])
            self.assertEqual(state["artifacts"][name]["sha256"],
                             pin["sha256"] if name != "jdk"
                             else state["artifacts"][name]["sha256"])
        self.assertTrue(state["smoke"]["ok"])
        caps = self.forge.toolchain_status()["capabilities"]
        for cap in ("edit", "build", "sign", "splits_merge", "jadx"):
            self.assertEqual(caps[cap], "full", f"{cap}: {caps[cap]}")

    def test_provision_is_idempotent_detection(self):
        res = self.forge.toolchain_provision()
        self.assertEqual(res["provisioned_now"], [])
        self.assertEqual(sorted(res["already_present"]),
                         ["apkeditor", "apktool", "build-tools",
                          "bundletool", "jadx", "jdk"])

    # ── round trip ───────────────────────────────────────────────
    def test_decode_edit_build_roundtrip(self):
        pid = self._make_project()
        dec = self.forge.decode(pid)
        self.assertIn("decoded", dec, dec)
        ed = self.forge.edit(
            pid, "manifest",
            add_permission="android.permission.CAMERA",
            version_name="9.9")
        self.assertIn("edit", ed, ed)
        built = self.forge.build(pid)
        self.assertIn("build", built, built)
        card = built["build"]
        # the card tells the truth, completely
        self.assertTrue(Path(card["output"]).exists())
        self.assertEqual(card["signing"]["alias"], "devon")
        self.assertFalse(card["signing"]["debug_key"])
        self.assertTrue(card["verify"]["verifies"])
        self.assertTrue(card["verify"]["schemes"]["v2"])
        self.assertEqual(len(card["resign_facts"]), 3)
        self.assertEqual(card["live_edit_ids"],
                         [ed["edit"]["edit_id"]])
        # the edit LANDED in the rebuilt APK's own dossier
        add2 = self.forge.add(card["output"])
        diss = self.forge.dissect(add2["project"]["id"])
        perms = [p["name"] for p in diss["dossier"]["permissions"]]
        self.assertIn("android.permission.CAMERA", perms)
        self.assertEqual(diss["dossier"]["identity"]["version_name"],
                         "9.9")
        # and the A1 diff sees exactly that delta
        delta = self.forge.diff(pid, add2["project"]["id"])
        added = delta["diff"]["permissions"]["added"]
        self.assertIn("android.permission.CAMERA", added)

    # ── patch engine ─────────────────────────────────────────────
    def test_patch_applies_and_refusals_are_fail_closed(self):
        pid = self._make_project()
        self.forge.decode(pid)
        before = self._smali_path(pid).read_bytes()
        res = self.forge.patch(pid, json.dumps(_PATCH))
        self.assertIn("patch_result", res, res)
        body = self._smali_path(pid).read_text()
        start = body.index("checkLicense")
        method_text = body[start:body.index(".end method", start)]
        self.assertIn("const/4 v0, 0x1", method_text)
        self.assertNotIn("invoke-virtual", method_text)
        # fingerprint mismatch → refused, bytes untouched, journaled
        after_patch = self._smali_path(pid).read_bytes()
        self.assertNotEqual(before, after_patch)
        bad = dict(_PATCH, name="bad-fingerprint",
                   match={**_PATCH["match"],
                          "fingerprint": {"strings": ["nope-not-here"]}})
        res = self.forge.patch(pid, json.dumps(bad))
        self.assertIn("error", res)
        self.assertIn("fail-closed", res["error"])
        self.assertEqual(self._smali_path(pid).read_bytes(),
                         after_patch)
        proj = self.forge.resolve(pid)
        ops = [r["op"] for r in proj.journal_rows(limit=100)]
        self.assertIn("patch_refused", ops)
        # expect_matches=2 → refused
        two = dict(_PATCH, name="two", expect_matches=2)
        res = self.forge.patch(pid, json.dumps(two))
        self.assertIn("error", res)
        # target sha mismatch → refused
        wrong = dict(_PATCH, name="wrong-sha",
                     target_apk_sha256="0" * 64)
        res = self.forge.patch(pid, json.dumps(wrong))
        self.assertIn("error", res)
        self.assertIn("sha256", res["error"])

    # ── undo ─────────────────────────────────────────────────────
    def test_undo_restores_exact_bytes(self):
        pid = self._make_project()
        self.forge.decode(pid)
        manifest_before = (self.forge.root / "projects" / pid
                           / "decoded" / "AndroidManifest.xml"
                           ).read_bytes()
        smali_before = self._smali_path(pid).read_bytes()
        ed = self.forge.edit(
            pid, "manifest",
            add_permission="android.permission.RECORD_AUDIO")
        pr = self.forge.patch(pid, json.dumps(_PATCH))
        self.assertNotEqual(self._smali_path(pid).read_bytes(),
                            smali_before)
        un = self.forge.undo(pid, pr["patch_result"]["edit_id"])
        self.assertIn("undo", un, un)
        self.assertEqual(self._smali_path(pid).read_bytes(),
                         smali_before)
        un2 = self.forge.undo(pid, ed["edit"]["edit_id"])
        self.assertIn("undo", un2, un2)
        manifest_after = (self.forge.root / "projects" / pid
                          / "decoded" / "AndroidManifest.xml"
                          ).read_bytes()
        self.assertEqual(manifest_after, manifest_before)
        edits = self.forge.edits(pid)["edits"]
        self.assertTrue(all(e["undone"] for e in edits))
        # nothing left to undo by default
        res = self.forge.undo(pid)
        self.assertIn("error", res)

    # ── strings + resource edits ─────────────────────────────────
    def test_strings_and_resource_edits(self):
        pid = self._make_project()
        self.forge.decode(pid)
        res = self.forge.edit(pid, "strings", name="app_name",
                              value="Renamed App")
        self.assertIn("edit", res, res)
        got = self.forge.get_string(pid, "app_name")
        self.assertEqual(got["value"], "Renamed App")
        src = SESSION_ROOT / "payload.txt"
        src.write_text("hello-assets", encoding="utf-8")
        res = self.forge.edit(pid, "add", target="assets/payload.txt",
                              source=str(src))
        self.assertIn("edit", res, res)
        landed = (self.forge.root / "projects" / pid / "decoded"
                  / "assets" / "payload.txt")
        self.assertEqual(landed.read_text(), "hello-assets")
        res = self.forge.undo(pid, res["edit"]["edit_id"])
        self.assertIn("undo", res, res)
        self.assertFalse(landed.exists())
        # path guards refuse by name
        res = self.forge.edit(pid, "add", target="../escape.txt",
                              source=str(src))
        self.assertIn("error", res)
        res = self.forge.edit(pid, "add",
                              target="smali/evil.smali",
                              source=str(src))
        self.assertIn("error", res)

    # ── keystore hygiene ─────────────────────────────────────────
    def test_keystore_never_touches_project_or_journal(self):
        pid = self._make_project()
        built = self.forge.build(pid)
        self.assertIn("build", built, built)
        root = self.forge.root
        offenders = [str(p) for p in root.rglob("*")
                     if p.is_file() and p.suffix
                     in (".jks", ".keystore", ".p12")]
        self.assertEqual(offenders, [])
        keys = self.forge.keys()
        self.assertIn("keystores", keys, keys)
        devon = [k for k in keys["keystores"]
                 if k["alias"] == "devon"]
        self.assertEqual(len(devon), 1)
        self.assertEqual(devon[0]["cert_sha256"],
                         built["build"]["signing"]["cert_sha256"])
        # the vault record's secrets appear NOWHERE on disk under
        # the class root (journal, project.json, build cards, state)
        from nomorals.agents.apkforge.signing import open_class_vault

        vault = open_class_vault(self.ctx)
        record = json.loads(
            vault.get("apkforge-keystore", "devon",
                      mark_used=False).password)
        secrets = [record["store_password"], record["keystore_b64"]]
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".jsonl",
                                                 ".md", ".txt"):
                text = path.read_text(encoding="utf-8",
                                      errors="ignore")
                for secret in secrets:
                    self.assertNotIn(secret, text,
                                     f"secret leaked into {path}")
        # no leftover materialization dirs
        leftovers = [p.name for p in root.iterdir()
                     if p.name.startswith(".sign-")]
        self.assertEqual(leftovers, [])

    # ── splits ───────────────────────────────────────────────────
    def test_splits_merge_installset_extract(self):
        base_pid = self._make_project()
        proj = self.forge.resolve(base_pid)
        split_pid = self._make_project(split="config.en")
        split_proj = self.forge.resolve(split_pid)
        bundle_path = SESSION_ROOT / f"bundle-{base_pid}.apks"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.write(proj.original_path, "base.apk")
            zf.write(split_proj.original_path, "config.en.apk")
        add = self.forge.add(str(bundle_path))
        bid = add["project"]["id"]
        lst = self.forge.splits(bid, "list")
        self.assertEqual(lst["count"], 2, lst)
        ext = self.forge.splits(bid, "extract", part="config.en.apk")
        self.assertIn("extract", ext, ext)
        self.assertTrue(Path(ext["extract"]["path"]).exists())
        merged = self.forge.splits(bid, "merge")
        self.assertIn("merge", merged, merged)
        from nomorals.agents.apkforge.signing import verify_apk

        verdict = verify_apk(self.forge.root,
                             Path(merged["merge"]["output"]))
        self.assertTrue(verdict["verifies"])
        from nomorals.agents.apkforge.dissect import light_identity

        ident = light_identity(merged["merge"]["output"])
        self.assertEqual(ident["package"], "com.example.fixture")
        spec = {"supportedAbis": ["arm64-v8a"],
                "supportedLocales": ["en-US"], "screenDensity": 420,
                "sdkVersion": 34}
        iset = self.forge.splits(bid, "installset", spec=spec)
        names = [p["name"] for p in iset["installset"]["parts"]]
        self.assertIn("base.apk", names)
        self.assertIn("config.en.apk", names)
        spec_fr = {**spec, "supportedLocales": ["fr-FR"]}
        iset2 = self.forge.splits(bid, "installset", spec=spec_fr)
        names2 = [p["name"] for p in iset2["installset"]["parts"]]
        self.assertIn("base.apk", names2)
        self.assertNotIn("config.en.apk", names2)

    def test_aab_paths_are_honest(self):
        # a structural .aab stand-in: bundletool must fail on it by
        # name (its manifest is not a proto manifest) — never a
        # fake success
        fake = SESSION_ROOT / "fake.aab"
        with zipfile.ZipFile(fake, "w") as zf:
            zf.writestr("BundleConfig.pb", b"\x00")
            zf.writestr("base/manifest/AndroidManifest.xml",
                        b"not-a-proto-manifest")
            zf.writestr("base/dex/classes.dex", b"dex\n035\x00")
        add = self.forge.add(str(fake))
        self.assertIn("project", add, add)
        self.assertEqual(add["project"]["kind_detail"], "aab")
        res = self.forge.splits(add["project"]["id"], "generate")
        self.assertIn("error", res)
        self.assertIn("bundletool", res["error"])
        # generate from a plain APK refuses with the reason
        pid = self._make_project()
        res = self.forge.splits(pid, "generate")
        self.assertIn("error", res)
        self.assertIn(".aab", res["error"])

    # ── jadx ─────────────────────────────────────────────────────
    def test_jadx_reading_surface(self):
        pid = self._make_project()
        res = self.forge.jadx(pid)
        self.assertIn("jadx", res, res)
        self.assertGreaterEqual(res["java_files"], 1)
        main_java = list(Path(res["jadx"]).rglob("MainActivity.java"))
        self.assertTrue(main_java)
        self.assertIn("checkLicense",
                      main_java[0].read_text(encoding="utf-8",
                                             errors="replace"))

    # ── surfaces ─────────────────────────────────────────────────
    def test_tool_registration_and_actions(self):
        from nomorals.agents.apkforge import register

        captured = {}

        class _Reg:
            context = self.ctx

            def register(self, name, **kw):
                def deco(fn):
                    captured[name] = (kw, fn)
                    return fn
                return deco

        register(_Reg())
        self.assertIn("apkforge", captured)
        self.assertIn("apkforge_build", captured)
        self.assertEqual(captured["apkforge"][0]["capability"],
                         "db.read")
        self.assertEqual(captured["apkforge_build"][0]["capability"],
                         "exec.shell")
        build_fn = captured["apkforge_build"][1]
        out = build_fn(action="toolchain")
        self.assertIn("capabilities", out)
        pid = self._make_project()
        out = build_fn(action="edits", project=pid)
        self.assertIn("edits", out, out)

    def test_cli_surface(self):
        from nomorals.cmdline.commands.apk import _cmd_apk

        def run(**kw):
            args = SimpleNamespace(
                apk_action=kw.pop("action"), **kw)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_apk(args, self.ctx)
            return rc, buf.getvalue()

        rc, out = run(action="status")
        self.assertEqual(rc, 0, out)
        rc, out = run(action="toolchain", op="status")
        self.assertEqual(rc, 0, out)
        self.assertIn("toolchain", out.lower())
        pid = self._make_project()
        rc, out = run(action="edits", project=pid)
        self.assertEqual(rc, 0, out)


if __name__ == "__main__":
    unittest.main()
