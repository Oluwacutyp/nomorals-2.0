"""ApkForge A3 — bisect, signature growth, hex patching, Museum.

Fixtures are REAL toolchain products, like A2's: a minimal apktool
source project is built by the provisioned apktool itself and
signed through the class's own vault keystore.  The version series
(five builds, versionCode 1-5) plants three flips at known versions
— CAMERA permission appears in v3, a NewFeature class appears in
v4 — and every version carries an unknown "tracker" package
(com.acme.track, 3 classes) that no seed signature covers, so the
signature-growth flow has real discovery evidence to work on.  One
fixture carries a fake-but-ELF-headed .so with planted byte
patterns for the hex-patch tests.

What this does NOT prove: real-world APKs (multidex, R8, packed),
bisect predicates that need runtime behavior (static only, by
design), and delivery through a live chat gateway (a fake gateway
stands in for the transport; the send path itself is the repo's
real filesend surface).
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
    "APKFORGE_A3_TEST_ROOT", str(_WS / "a3-verify" / "a3-test-root")))
TOOL_CACHE = _WS / "a2-verify" / "tool-cache"

_MANIFEST_HEAD = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="{package}" android:versionCode="{vcode}" android:versionName="{vname}">
    <uses-sdk android:minSdkVersion="24" android:targetSdkVersion="34"/>
    <uses-permission android:name="android.permission.INTERNET"/>
{extra_perms}"""

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
  versionCode: {vcode}
  versionName: '{vname}'
doNotCompress:
- arsc
"""

_MAIN_SMALI = """.class public Lcom/example/fixture/MainActivity;
.super Landroid/app/Activity;
.source "MainActivity.java"

.method public constructor <init>()V
    .locals 0
    invoke-direct {p0}, Landroid/app/Activity;-><init>()V
    return-void
.end method
"""

_NEWFEATURE_SMALI = """.class public Lcom/example/fixture/NewFeature;
.super Ljava/lang/Object;
.source "NewFeature.java"

.method public constructor <init>()V
    .locals 0
    invoke-direct {p0}, Ljava/lang/Object;-><init>()V
    return-void
.end method
"""

_TRACKER_SMALI = """.class public Lcom/acme/track/{name};
.super Ljava/lang/Object;
.source "{name}.java"

.method public constructor <init>()V
    .locals 0
    invoke-direct {p0}, Ljava/lang/Object;-><init>()V
    return-void
.end method

.method public static ping()V
    .locals 0
    return-void
.end method
"""

_STRINGS = """<?xml version="1.0" encoding="utf-8"?>
<resources>
    <string name="app_name">Fixture App</string>
</resources>
"""

# fake native lib: ELF magic + header pad + one unique pattern +
# one pattern that occurs exactly twice
_SO_BYTES = (
    b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 24
    + b"\xde\xad\xbe\xef" + b"\x90" * 8
    + b"\x01\x02\x03\x04" + b"\x90" * 8
    + b"\x01\x02\x03\x04" + b"\x90" * 8
)


def _write_skeleton(dest: Path, *, package: str = "com.example.fixture",
                    vcode: int = 1, vname: str = "1.0",
                    camera: bool = False, new_feature: bool = False,
                    native: bool = False) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    extra_perms = ""
    if camera:
        extra_perms = ('    <uses-permission '
                       'android:name="android.permission.CAMERA"/>\n')
    manifest_body = (
        '    <application android:label="@string/app_name" '
        'android:debuggable="true">\n'
        '        <activity android:name=".MainActivity" '
        'android:exported="true">\n'
        '            <intent-filter>\n'
        '                <action android:name="android.intent.action.MAIN"/>\n'
        '                <category android:name="android.intent.category.LAUNCHER"/>\n'
        '            </intent-filter>\n'
        '        </activity>\n'
        '    </application>\n')
    (dest / "AndroidManifest.xml").write_text(
        _MANIFEST_HEAD.format(package=package, vcode=vcode,
                              vname=vname, extra_perms=extra_perms)
        + manifest_body + "</manifest>\n")
    (dest / "apktool.yml").write_text(
        _YML.format(vcode=vcode, vname=vname))
    smali_dir = dest / "smali" / "com" / "example" / "fixture"
    smali_dir.mkdir(parents=True, exist_ok=True)
    (smali_dir / "MainActivity.smali").write_text(_MAIN_SMALI)
    if new_feature:
        (smali_dir / "NewFeature.smali").write_text(_NEWFEATURE_SMALI)
    track_dir = dest / "smali" / "com" / "acme" / "track"
    track_dir.mkdir(parents=True, exist_ok=True)
    for name in ("Alpha", "Beta", "Gamma"):
        (track_dir / f"{name}.smali").write_text(
            _TRACKER_SMALI.replace("{name}", name))
    values = dest / "res" / "values"
    values.mkdir(parents=True, exist_ok=True)
    (values / "strings.xml").write_text(_STRINGS)
    if native:
        lib_dir = dest / "lib" / "arm64-v8a"
        lib_dir.mkdir(parents=True, exist_ok=True)
        (lib_dir / "libfixture.so").write_bytes(_SO_BYTES)
    return dest


# ── pure-logic contract tests (no toolchain) ─────────────────────────

class BisectContract(unittest.TestCase):
    def test_halve_finds_exact_flip(self):
        from nomorals.agents.apkforge.bisect import halve

        items = [f"v{i}" for i in range(8)]
        truth = {f"v{i}": ("found" if i >= 5 else "absent")
                 for i in range(8)}
        res = halve(items, lambda it: truth[it])
        self.assertEqual(res["status"], "flip")
        self.assertEqual(res["transition"]["older"], "v4")
        self.assertEqual(res["transition"]["newer"], "v5")
        self.assertEqual(res["direction"], "appears")
        # halving, not scanning: endpoints + ~log2 probes
        self.assertLessEqual(res["probes"], 5)

    def test_halve_removal_direction(self):
        from nomorals.agents.apkforge.bisect import halve

        items = ["a", "b", "c", "d"]
        truth = {"a": "found", "b": "found", "c": "absent",
                 "d": "absent"}
        res = halve(items, lambda it: truth[it])
        self.assertEqual(res["status"], "flip")
        self.assertEqual(res["direction"], "disappears")
        self.assertEqual(res["transition"]["newer"], "c")

    def test_halve_skips_bracket_not_guess(self):
        from nomorals.agents.apkforge.bisect import halve

        items = [f"v{i}" for i in range(6)]
        truth = {"v0": "absent", "v1": "absent", "v2": "skip",
                 "v3": "skip", "v4": "found", "v5": "found"}
        res = halve(items, lambda it: truth[it])
        self.assertEqual(res["status"], "flip_bracketed")
        self.assertEqual(res["transition"]["older"], "v1")
        self.assertEqual(res["transition"]["newer"], "v4")
        self.assertEqual(sorted(res["skipped"]), ["v2", "v3"])

    def test_halve_resume_uses_known_verdicts(self):
        from nomorals.agents.apkforge.bisect import halve

        items = [f"v{i}" for i in range(5)]
        known = {"v0": "absent", "v2": "absent", "v3": "found",
                 "v4": "found"}

        def judge(it):  # pragma: no cover — must never be called
            raise AssertionError(f"re-judged {it} on resume")

        res = halve(items, judge, known=known)
        self.assertEqual(res["status"], "flip")
        self.assertEqual(res["probes"], 0)
        self.assertEqual(res["transition"]["older"], "v2")
        self.assertEqual(res["transition"]["newer"], "v3")

    def test_halve_throughout_and_inconclusive(self):
        from nomorals.agents.apkforge.bisect import halve

        res = halve(["a", "b", "c"], lambda it: "found")
        self.assertEqual(res["status"], "present_throughout")
        res = halve(["a", "b", "c"], lambda it: "skip")
        self.assertEqual(res["status"], "inconclusive")

    def test_predicate_kinds_and_skip(self):
        from nomorals.agents.apkforge.bisect import (
            ABSENT, FOUND, SKIP, evaluate_predicate, parse_predicate)

        dossier = {
            "permissions": [{"name": "android.permission.CAMERA"}],
            "components": {"activities": [
                {"name": ".Main", "exported": True}], "services": [],
                "receivers": [], "providers": []},
            "signing": {"certificates": [{"sha256": "ab12cd"}]},
            "dex": {"class_names": ["Lcom/x/Y;"],
                    "class_hashes": {"Lcom/x/Y;": "h"},
                    "urls": ["https://api.example.com/v1"],
                    "domains": ["example.com"],
                    "_all_strings": ["hello tracker"]},
            "sdk_signatures": [{"name": "Acme",
                                 "package_prefix": "com.acme.track"}],
            "native": [{"path": "lib/arm64-v8a/libx.so",
                        "name": "libx.so"}],
        }
        cases = [
            ("permission: android.permission.CAMERA", FOUND),
            ("permission: android.permission.SMS", ABSENT),
            ("class: com.x.Y", FOUND),
            ("class: Lcom/x/Y;", FOUND),
            ("tracker: acme", FOUND),
            ("sdk: com.acme", FOUND),
            ("exported: .Main", FOUND),
            ("exported: .Nope", ABSENT),
            ("endpoint: api.example", FOUND),
            ("string: tracker", FOUND),
            ("cert: AB12", FOUND),
            ("cert: ff00", ABSENT),
            ("native: libx.so", FOUND),
        ]
        for spec, want in cases:
            got = evaluate_predicate(dossier, parse_predicate(spec))
            self.assertEqual(got, want, spec)
        # a dossier that cannot judge → SKIP, never a guess
        self.assertEqual(
            evaluate_predicate({}, parse_predicate(
                "permission: android.permission.CAMERA")), SKIP)
        self.assertEqual(
            evaluate_predicate(None, parse_predicate(
                "class: com.x.Y")), SKIP)
        with self.assertRaises(Exception):
            parse_predicate("")


# ── integration over real toolchain-built fixtures ───────────────────

class ApkForgeA3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved_env = {k: os.environ.get(k) for k in
                          ("NM_VAULT_PASSPHRASE",
                           "APKFORGE_TOOL_CACHE")}
        os.environ["NM_VAULT_PASSPHRASE"] = "apkforge-a3-test-seal"
        if TOOL_CACHE.exists():
            os.environ["APKFORGE_TOOL_CACHE"] = str(TOOL_CACHE)
        SESSION_ROOT.mkdir(parents=True, exist_ok=True)
        for stale in SESSION_ROOT.glob("a3fix-*"):
            shutil.rmtree(stale, ignore_errors=True)
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
        cls.provision = cls.forge.toolchain_provision()
        assert not cls.provision.get("error"), cls.provision
        cls._seq = 0
        # the version series: CAMERA lands in v3, NewFeature in v4
        cls.series = []
        for i in range(1, 6):
            path = cls._build_apk(vcode=i, vname=f"{i}.0",
                                  camera=i >= 3,
                                  new_feature=i >= 4)
            add = cls.forge.add(path)
            assert "project" in add, add
            cls.series.append(add["project"]["id"])

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    @classmethod
    def _build_apk(cls, **kw) -> str:
        """Build + sign a fixture APK with the real chain; returns
        the signed file path (landing is the caller's choice)."""
        from nomorals.agents.apkforge import signing as signing_mod
        from nomorals.agents.apkforge.toolchain import run_tool

        cls._seq += 1
        work = Path(tempfile.mkdtemp(prefix=f"a3fix-{cls._seq}-",
                                     dir=str(SESSION_ROOT)))
        skel = _write_skeleton(work / "skel", **kw)
        unsigned = work / "unsigned.apk"
        res = run_tool("", "apktool",
                       ["b", str(skel), "-o", str(unsigned)],
                       root=cls.forge.root)
        assert res["rc"] == 0, res["stderr"][-400:]
        signed = work / "signed.apk"
        signing_mod.sign_apk(cls.ctx, cls.forge.root, unsigned,
                             signed)
        return str(signed)

    # ── bisect ───────────────────────────────────────────────────
    def test_bisect_finds_permission_flip_with_diff(self):
        ids = list(reversed(self.series))  # scrambled input order
        res = self.forge.bisect(
            ids, "permission: android.permission.CAMERA")
        self.assertNotIn("error", res, res)
        self.assertEqual(res["status"], "flip")
        self.assertEqual(res["direction"], "appears")
        t = res["transition"]
        self.assertEqual(res["versions"][t["older"]]["version_code"],
                         2)
        self.assertEqual(res["versions"][t["newer"]]["version_code"],
                         3)
        # the transition pair's diff is the evidence
        self.assertIn("diff", res)
        self.assertIn("android.permission.CAMERA",
                      res["diff"]["permissions"]["added"])
        self.assertTrue(Path(res["diff_path"]).exists())
        # journaled on the projects themselves
        proj = self.forge.resolve(t["newer"])
        ops = [r["op"] for r in proj.journal_rows(limit=100)]
        self.assertIn("bisect", ops)

    def test_bisect_resumes_from_journaled_state(self):
        spec = "class: com.example.fixture.NewFeature"
        first = self.forge.bisect(self.series, spec)
        self.assertNotIn("error", first, first)
        self.assertEqual(first["status"], "flip")
        self.assertEqual(
            first["versions"][first["transition"]["newer"]]
            ["version_code"], 4)
        self.assertGreater(first["probes"], 0)
        second = self.forge.bisect(self.series, spec)
        self.assertTrue(second["resumed"])
        self.assertEqual(second["probes"], 0)  # nothing re-judged
        self.assertEqual(second["transition"],
                         first["transition"])
        # the session log tells the story: start, probes, completes
        log = (self.root / "bisects" / second["session"]
               / "log.jsonl").read_text(encoding="utf-8")
        ops = [json.loads(ln)["op"] for ln in log.splitlines()
               if ln.strip()]
        self.assertIn("bisect_start", ops)
        self.assertIn("probe", ops)
        self.assertEqual(ops.count("bisect_complete"), 2)

    def test_bisect_refusals_are_honest(self):
        res = self.forge.bisect(self.series[:1],
                                "permission: android.permission.CAMERA")
        self.assertIn("error", res)
        res = self.forge.bisect(self.series, "")
        self.assertIn("error", res)
        # a series spanning two packages refuses by name
        other = self._build_apk(package="com.other.app", vcode=9,
                                vname="9.0")
        add = self.forge.add(other)
        res = self.forge.bisect(
            [self.series[0], add["project"]["id"]],
            "permission: android.permission.INTERNET")
        self.assertIn("error", res)
        self.assertIn("more than one app", res["error"])
        # property absent everywhere → a verdict, not an error
        res = self.forge.bisect(
            self.series, "permission: android.permission.SEND_SMS")
        self.assertEqual(res["status"], "absent_throughout")

    # ── signature growth ─────────────────────────────────────────
    def test_signature_discovery_promotion_demotion(self):
        sig_root = SESSION_ROOT / "sig-apkforge"
        shutil.rmtree(sig_root, ignore_errors=True)
        from nomorals.agents.apkforge import ApkForge

        forge = ApkForge(self.ctx, base_dir=str(sig_root))
        ids = []
        for i in range(1, 4):
            path = self._build_apk(vcode=i, vname=f"{i}.0")
            add = forge.add(path)
            ids.append(add["project"]["id"])
        # one dissection: candidate noted, promotion refused as
        # below threshold — no silent table change
        forge.dissect(ids[0])
        res = forge.signatures("promote", prefix="com.acme.track")
        self.assertIn("error", res)
        self.assertIn("threshold", res["error"])
        # three dissections: the unknown tracker is proposed with
        # its evidence
        forge.dissect(ids[1])
        forge.dissect(ids[2])
        st = forge.signatures("list")["signatures"]
        props = {p["prefix"]: p for p in st["proposals"]}
        self.assertIn("com.acme.track", props)
        self.assertEqual(props["com.acme.track"]["occurrences"], 3)
        self.assertEqual(props["com.acme.track"]["dossiers"], ids)
        self.assertEqual(st["grown_count"], 0)
        # seeds are untouched and cannot be demoted
        from nomorals.agents.apkforge.dissect import SDK_SIGNATURES

        self.assertEqual(st["seed_count"], len(SDK_SIGNATURES))
        res = forge.signatures("demote", prefix="com.flurry")
        self.assertIn("error", res)
        self.assertIn("seed", res["error"])
        # promotion: journaled, with evidence; the entry then
        # matches in a fresh dissection, marked grown
        res = forge.signatures("promote", prefix="com.acme.track",
                               label="Acme Tracker", kind="analytics")
        self.assertIn("promoted", res, res)
        self.assertEqual(res["promoted"]["origin"], "grown")
        self.assertEqual(res["promoted"]["provenance"]["dossiers"],
                         ids)
        path = self._build_apk(vcode=4, vname="4.0")
        add = forge.add(path)
        diss = forge.dissect(add["project"]["id"])
        hits = {s["name"]: s
                for s in diss["dossier"]["sdk_signatures"]}
        self.assertIn("Acme Tracker", hits)
        self.assertEqual(hits["Acme Tracker"]["origin"], "grown")
        self.assertIn("grown", diss["dossier"]["sdk_seed_note"])
        events = (sig_root / "signatures.jsonl").read_text(
            encoding="utf-8")
        ops = [json.loads(ln)["op"] for ln in events.splitlines()
               if ln.strip()]
        self.assertIn("signature_proposed", ops)
        self.assertIn("signature_promoted", ops)
        # demotion: journaled, reversible — the hit disappears,
        # and the entry can be promoted again from its kept census
        res = forge.signatures("demote", prefix="com.acme.track",
                               reason="test demotion")
        self.assertIn("demoted", res, res)
        diss2 = forge.dissect(add["project"]["id"])
        names = {s["name"] for s in diss2["dossier"]["sdk_signatures"]}
        self.assertNotIn("Acme Tracker", names)
        res = forge.signatures("promote", prefix="com.acme.track",
                               label="Acme Tracker", kind="analytics")
        self.assertIn("promoted", res, res)

    # ── hex patching ─────────────────────────────────────────────
    def test_hexpatch_apply_refuse_undo_rebuild(self):
        path = self._build_apk(vcode=1, vname="1.0", native=True)
        add = self.forge.add(path)
        pid = add["project"]["id"]
        dec = self.forge.decode(pid)
        self.assertIn("decoded", dec, dec)
        so = (self.root / "projects" / pid / "decoded" / "lib"
              / "arm64-v8a" / "libfixture.so")
        self.assertTrue(so.exists())
        before = so.read_bytes()
        self.assertEqual(before, _SO_BYTES)
        doc = {"name": "flip-pattern",
               "target": "lib/arm64-v8a/libfixture.so",
               "match": {"pattern": "de ad be ef",
                         "expect_matches": 1},
               "replace": "c0 03 5f d6"}
        res = self.forge.hexpatch(pid, json.dumps(doc))
        self.assertIn("hexpatch_result", res, res)
        hr = res["hexpatch_result"]
        off = before.index(b"\xde\xad\xbe\xef")
        self.assertEqual(hr["offsets"], [off])
        after = so.read_bytes()
        self.assertEqual(len(after), len(before))  # length preserved
        self.assertEqual(after[off:off + 4], b"\xc0\x03\x5f\xd6")
        # count mismatch → refused whole, bytes untouched, journaled
        doc2 = {"name": "two-sites",
                "target": "lib/arm64-v8a/libfixture.so",
                "match": {"pattern": "01 02 03 04",
                          "expect_matches": 1},
                "replace": "aa bb cc dd"}
        res = self.forge.hexpatch(pid, json.dumps(doc2))
        self.assertIn("error", res)
        self.assertIn("fail-closed", res["error"])
        self.assertEqual(so.read_bytes(), after)
        proj = self.forge.resolve(pid)
        ops = [r["op"] for r in proj.journal_rows(limit=100)]
        self.assertIn("hexpatch_refused", ops)
        # the honest count succeeds; undo restores exact bytes
        doc2["match"]["expect_matches"] = 2
        res = self.forge.hexpatch(pid, json.dumps(doc2))
        self.assertIn("hexpatch_result", res, res)
        self.assertEqual(len(res["hexpatch_result"]["offsets"]), 2)
        un = self.forge.undo(
            pid, res["hexpatch_result"]["edit_id"])
        self.assertIn("undo", un, un)
        self.assertEqual(so.read_bytes(), after)
        un = self.forge.undo(pid, hr["edit_id"])
        self.assertEqual(so.read_bytes(), before)
        # length-changing replacement refuses at parse
        bad = dict(doc, replace="c0 03")
        res = self.forge.hexpatch(pid, json.dumps(bad))
        self.assertIn("error", res)
        self.assertIn("length", res["error"])
        # non-ELF target refuses by name
        src = SESSION_ROOT / "notelf.so"
        src.write_bytes(b"just text, not an elf")
        res = self.forge.edit(pid, "add",
                              target="lib/arm64-v8a/libfake.so",
                              source=str(src))
        self.assertIn("edit", res, res)
        fake = dict(doc, target="lib/arm64-v8a/libfake.so")
        res = self.forge.hexpatch(pid, json.dumps(fake))
        self.assertIn("error", res)
        self.assertIn("ELF", res["error"])

    def test_hexpatch_survives_rebuild_and_resign(self):
        path = self._build_apk(vcode=1, vname="1.0", native=True)
        pid = self.forge.add(path)["project"]["id"]
        self.forge.decode(pid)
        doc = {"name": "flip-pattern",
               "target": "lib/arm64-v8a/libfixture.so",
               "match": {"pattern": "de ad be ef",
                         "expect_matches": 1},
               "replace": "c0 03 5f d6"}
        res = self.forge.hexpatch(pid, json.dumps(doc))
        self.assertIn("hexpatch_result", res, res)
        built = self.forge.build(pid)
        self.assertIn("build", built, built)
        self.assertTrue(built["build"]["verify"]["verifies"])
        with zipfile.ZipFile(built["build"]["output"]) as zf:
            data = zf.read("lib/arm64-v8a/libfixture.so")
        self.assertIn(b"\xc0\x03\x5f\xd6", data)
        self.assertNotIn(b"\xde\xad\xbe\xef", data)
        # and the rebuilt APK dissects with its native census intact
        add2 = self.forge.add(built["build"]["output"])
        diss = self.forge.dissect(add2["project"]["id"])
        libs = {n["name"] for n in diss["dossier"]["native"]}
        self.assertIn("libfixture.so", libs)

    # ── Museum wiring ────────────────────────────────────────────
    def test_museum_census_and_send(self):
        pid = self.series[0]
        diss = self.forge.dissect(pid)
        self.assertIn("dossier", diss, diss)
        built = self.forge.build(pid)
        self.assertIn("build", built, built)
        res = self.forge.artifacts()
        shelves = res["by_shelf"]
        self.assertGreaterEqual(shelves.get("apk-projects", 0), 5)
        self.assertGreaterEqual(shelves.get("apk-dossiers", 0), 1)
        self.assertGreaterEqual(shelves.get("apk-builds", 0), 1)
        dossier_cards = [c for c in res["artifacts"]
                         if c["id"] == f"{pid}-dossier"]
        self.assertEqual(len(dossier_cards), 1)
        card = dossier_cards[0]
        self.assertTrue(Path(card["path"]).exists())
        self.assertEqual(card["path"],
                         str(self.root / "projects" / pid
                             / "dossier.json"))
        # send through the repo's real file surface, fake gateway
        sent = {}

        class _Result:
            ok = True
            message_id = "m-1"
            error = ""

        class _Gateway:
            adapters = {"telegram": object()}

            def send_file(self, platform, chat_id, path,
                          caption=""):
                sent.update(platform=platform, chat_id=chat_id,
                            path=path, caption=caption)
                return _Result()

        self.ctx.extras["gateway"] = _Gateway()
        try:
            out = self.forge.send_artifact(card["id"],
                                           "telegram", "4242")
            self.assertIn("sent", out, out)
            self.assertEqual(sent["path"], card["path"])
            self.assertEqual(sent["chat_id"], "telegram:4242")
        finally:
            self.ctx.extras.pop("gateway", None)
        # no gateway attached → the real surface's error, in-band
        out = self.forge.send_artifact(card["id"], "telegram", "4242")
        self.assertIn("error", out)
        self.assertIn("gateway", out["error"])
        out = self.forge.send_artifact("apk_nope", "telegram", "1")
        self.assertIn("error", out)

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
        self.assertEqual(captured["apkforge"][0]["capability"],
                         "db.read")
        self.assertEqual(captured["apkforge_build"][0]["capability"],
                         "exec.shell")
        read_fn = captured["apkforge"][1]
        out = read_fn(action="bisect",
                      series=",".join(self.series),
                      check="permission: android.permission.CAMERA")
        self.assertEqual(out["status"], "flip", out)
        out = read_fn(action="signatures", op="list")
        self.assertIn("signatures", out, out)
        out = read_fn(action="artifacts")
        self.assertIn("artifacts", out, out)
        build_fn = captured["apkforge_build"][1]
        out = build_fn(action="signatures", op="promote",
                       prefix="com.acme.track", label="Acme",
                       value="analytics")
        # the main root's census already saw the tracker 5 times
        self.assertIn("promoted", out, out)
        out = build_fn(action="signatures", op="demote",
                       prefix="com.acme.track", reason="surface test")
        self.assertIn("demoted", out, out)
        pid = self.series[0]
        self.forge.decode(pid)
        doc = {"name": "surface-noop-check",
               "target": "lib/arm64-v8a/libfixture.so",
               "match": {"pattern": "de ad be ef",
                         "expect_matches": 1},
               "replace": "c0 03 5f d6"}
        # this project has no .so → honest error, not a crash
        out = build_fn(action="hexpatch", project=pid,
                       patch=json.dumps(doc))
        self.assertIn("error", out, out)

    def test_cli_surface(self):
        from nomorals.cmdline.commands.apk import _cmd_apk

        def run(**kw):
            args = SimpleNamespace(
                apk_action=kw.pop("action"), **kw)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_apk(args, self.ctx)
            return rc, buf.getvalue()

        rc, out = run(action="bisect", series=self.series,
                      check="permission: android.permission.CAMERA")
        self.assertEqual(rc, 0, out)
        self.assertIn("first appears in", out)
        rc, out = run(action="signatures", op="list", prefix="",
                      label="", kind="", reason="")
        self.assertEqual(rc, 0, out)
        self.assertIn("signatures", out.lower())
        rc, out = run(action="artifacts")
        self.assertEqual(rc, 0, out)
        self.assertIn("apk-dossiers", out)


if __name__ == "__main__":
    unittest.main()
