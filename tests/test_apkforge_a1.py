"""ApkForge A1 — intake + dissect + teach + diff.

Fixtures are built programmatically in this file: a minimal but REAL
APK (hand-built binary AXML manifest + hand-built minimal DEX +
a genuine v1 PKCS#7 signature from `cryptography`), in two versions
that differ by a permission added, a permission removed, a class
added, a class removed, a method added, and a DIFFERENT signing key.

What the fixtures prove: every dissect claim is checked against the
ground truth baked into the fixture (package, versions, permissions
+ AOSP protection levels, components + exported flags, dex class /
method census, the fixture URL in strings, native inventory, the
certificate fingerprint computed independently via `cryptography`).

What they do NOT prove: real-world APK scale (multidex, real
resources.arsc string tables, v2/v3 signing blocks, R8-obfuscated
dex) — the fixture arsc is a stub and androguard reports it as
unparsed; the fixture dex is a valid parseable Dalvik file with
method/class tables but no code bodies, so call-graph claims are
not exercised.  APKiD runs for real on the fixtures; on this
hand-made dex it honestly matches only the ELF marker on the .so
entries, and the tests pin exactly that.
"""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
import struct
import tempfile
import unittest
import zipfile
import zlib
from types import SimpleNamespace

# ── fixture builder (binary AXML + minimal DEX + PKCS#7) ────────────


class _Pool:
    def __init__(self):
        self.strings: list[str] = []
        self.index: dict[str, int] = {}

    def add(self, s: str) -> int:
        if s not in self.index:
            self.index[s] = len(self.strings)
            self.strings.append(s)
        return self.index[s]

    def chunk(self) -> bytes:
        def uleb(n: int) -> bytes:
            out = b""
            while True:
                c = n & 0x7F
                n >>= 7
                if n:
                    out += bytes([c | 0x80])
                else:
                    out += bytes([c])
                    break
            return out

        data = b""
        offsets = []
        for s in self.strings:
            b = s.encode("utf-8")
            offsets.append(len(data))
            data += uleb(len(s)) + uleb(len(b)) + b + b"\x00"
        while len(data) % 4:
            data += b"\x00"
        header_size = 28 + 4 * len(self.strings)
        size = header_size + len(data)
        out = struct.pack("<HHIIIIII", 0x0001, 28, size,
                          len(self.strings), 0, 0x100, header_size, 0)
        out += b"".join(struct.pack("<I", o) for o in offsets) + data
        return out


_STR, _INT, _BOOL = 0x03, 0x10, 0x12
_NS_URI = "http://schemas.android.com/apk/res/android"


def _start_ns(pool, prefix, uri):
    return struct.pack("<HHIiiii", 0x0100, 16, 24, 0, -1,
                       pool.index[prefix], pool.index[uri])


def _end_ns(pool, prefix, uri):
    return struct.pack("<HHIiiii", 0x0101, 16, 24, 0, -1,
                       pool.index[prefix], pool.index[uri])


def _start_el(pool, name, attrs):
    body = struct.pack("<HHIiiii", 0x0102, 16, 36 + 20 * len(attrs),
                       0, -1, -1, pool.index[name])
    body += struct.pack("<HHHHHH", 20, 20, len(attrs), 0, 0, 0)
    for ns, nm, raw, typ, data in attrs:
        body += struct.pack("<iiiHBBI", ns, pool.index[nm], raw,
                            8, 0, typ, data & 0xFFFFFFFF)
    return body


def _end_el(pool, name):
    return struct.pack("<HHIiiii", 0x0103, 16, 24, 0, -1, -1,
                       pool.index[name])


def _build_axml(m: dict) -> bytes:
    pool = _Pool()
    strings = ["manifest", "package", "versionCode", "versionName",
               "uses-sdk", "minSdkVersion", "targetSdkVersion",
               "uses-permission", "name", "application", "label",
               "debuggable", "activity", "exported", "service",
               "receiver", "provider", "intent-filter", "action",
               "category", "authorities", "android", _NS_URI,
               m["package"], m["version_name"], m["label"],
               m["package"] + ".data", ".MainActivity",
               "android.intent.action.MAIN",
               "android.intent.category.LAUNCHER", ".SyncService",
               ".BootReceiver", ".DataProvider"] + list(m["permissions"])
    for s in strings:
        pool.add(s)
    nsidx = pool.index[_NS_URI]
    chunks = [pool.chunk(), _start_ns(pool, "android", _NS_URI)]

    def ai(name, val):
        return (nsidx, name, -1, _INT, val)

    def as_(name, s):
        return (nsidx, name, pool.index[s], _STR, pool.index[s])

    chunks.append(_start_el(pool, "manifest", [
        (-1, "package", pool.index[m["package"]], _STR,
         pool.index[m["package"]]),
        ai("versionCode", m["version_code"]),
        as_("versionName", m["version_name"])]))
    chunks.append(_start_el(pool, "uses-sdk", [
        ai("minSdkVersion", m["min_sdk"]),
        ai("targetSdkVersion", m["target_sdk"])]))
    chunks.append(_end_el(pool, "uses-sdk"))
    for p in m["permissions"]:
        chunks.append(_start_el(pool, "uses-permission", [as_("name", p)]))
        chunks.append(_end_el(pool, "uses-permission"))
    chunks.append(_start_el(pool, "application", [
        as_("label", m["label"]),
        (nsidx, "debuggable", -1, _BOOL, -1 if m["debuggable"] else 0)]))
    chunks.append(_start_el(pool, "activity", [
        as_("name", ".MainActivity"), (nsidx, "exported", -1, _BOOL, -1)]))
    chunks.append(_start_el(pool, "intent-filter", []))
    chunks.append(_start_el(pool, "action",
                            [as_("name", "android.intent.action.MAIN")]))
    chunks.append(_end_el(pool, "action"))
    chunks.append(_start_el(
        pool, "category",
        [as_("name", "android.intent.category.LAUNCHER")]))
    chunks.append(_end_el(pool, "category"))
    chunks.append(_end_el(pool, "intent-filter"))
    chunks.append(_end_el(pool, "activity"))
    chunks.append(_start_el(pool, "service", [
        as_("name", ".SyncService"), (nsidx, "exported", -1, _BOOL, 0)]))
    chunks.append(_end_el(pool, "service"))
    chunks.append(_start_el(pool, "receiver", [
        as_("name", ".BootReceiver"), (nsidx, "exported", -1, _BOOL, -1)]))
    chunks.append(_end_el(pool, "receiver"))
    chunks.append(_start_el(pool, "provider", [
        as_("name", ".DataProvider"),
        as_("authorities", m["package"] + ".data"),
        (nsidx, "exported", -1, _BOOL, 0)]))
    chunks.append(_end_el(pool, "provider"))
    chunks.append(_end_el(pool, "application"))
    chunks.append(_end_el(pool, "manifest"))
    chunks.append(_end_ns(pool, "android", _NS_URI))
    body = b"".join(chunks)
    return struct.pack("<HHI", 0x0003, 8, 8 + len(body)) + body


def _uleb(n: int) -> bytes:
    out = b""
    while True:
        c = n & 0x7F
        n >>= 7
        if n:
            out += bytes([c | 0x80])
        else:
            out += bytes([c])
            break
    return out


def _build_dex(strings: list[str],
               classes: list[tuple[str, list[str]]],
               extra_refs: tuple = ()) -> bytes:
    strs = sorted(set(strings) | {"V"}
                  | {d for d, _m in extra_refs}
                  | {m for _d, m in extra_refs})
    sidx = {s: i for i, s in enumerate(strs)}
    types = sorted({s for s in strs
                    if (s.startswith("L") and s.endswith(";"))
                    or len(s) == 1})
    tidx = {t: i for i, t in enumerate(types)}
    methods = sorted({(d, m) for d, ms in classes for m in ms}
                     | set(extra_refs))
    midx = {m: i for i, m in enumerate(methods)}
    n_str, n_type, n_proto = len(strs), len(types), 1
    n_meth, n_cls = len(methods), len(classes)
    str_ids_off = 0x70
    type_ids_off = str_ids_off + 4 * n_str
    proto_ids_off = type_ids_off + 4 * n_type
    meth_ids_off = proto_ids_off + 12 * n_proto
    class_defs_off = meth_ids_off + 8 * n_meth
    data_off = class_defs_off + 32 * n_cls
    blob = b""
    str_offs = []
    cur = data_off
    for s in strs:
        b = s.encode("utf-8")
        str_offs.append(cur)
        chunk = _uleb(len(s)) + b + b"\x00"
        blob += chunk
        cur += len(chunk)
    class_data = {}
    for desc, ms in classes:
        cd = _uleb(0) + _uleb(0) + _uleb(len(ms)) + _uleb(0)
        prev = 0
        for m in ms:
            mi = midx[(desc, m)]
            cd += _uleb(mi - prev) + _uleb(
                0x10001 if m == "<init>" else 0x1) + _uleb(0)
            prev = mi
        class_data[desc] = (cur, cd)
        blob += cd
        cur += len(cd)
    if len(blob) % 4:
        blob += b"\x00" * (4 - len(blob) % 4)
    map_off = data_off + len(blob)
    entries = [(0x0000, 1, 0), (0x0001, n_str, str_ids_off),
               (0x0002, n_type, type_ids_off),
               (0x0003, n_proto, proto_ids_off),
               (0x0005, n_meth, meth_ids_off),
               (0x0006, n_cls, class_defs_off), (0x1000, 1, map_off),
               (0x2002, n_str, str_offs[0]),
               (0x2000, n_cls, class_data[classes[0][0]][0])]
    mapb = struct.pack("<I", len(entries)) + b"".join(
        struct.pack("<HHII", t, 0, sz, off) for t, sz, off in entries)
    file_size = map_off + len(mapb)
    out = bytearray(b"\x00" * file_size)
    out[0:8] = b"dex\n035\x00"
    struct.pack_into("<I", out, 32, file_size)
    struct.pack_into("<I", out, 36, 0x70)
    struct.pack_into("<I", out, 40, 0x12345678)
    struct.pack_into("<I", out, 52, map_off)
    struct.pack_into("<I", out, 56, n_str)
    struct.pack_into("<I", out, 60, str_ids_off)
    struct.pack_into("<I", out, 64, n_type)
    struct.pack_into("<I", out, 68, type_ids_off)
    struct.pack_into("<I", out, 72, n_proto)
    struct.pack_into("<I", out, 76, proto_ids_off)
    struct.pack_into("<I", out, 80, 0)
    struct.pack_into("<I", out, 84, meth_ids_off)
    struct.pack_into("<I", out, 88, n_meth)
    struct.pack_into("<I", out, 92, meth_ids_off)
    struct.pack_into("<I", out, 96, n_cls)
    struct.pack_into("<I", out, 100, class_defs_off)
    struct.pack_into("<I", out, 104, file_size - data_off)
    struct.pack_into("<I", out, 108, data_off)
    for i, o in enumerate(str_offs):
        struct.pack_into("<I", out, str_ids_off + 4 * i, o)
    for i, t in enumerate(types):
        struct.pack_into("<I", out, type_ids_off + 4 * i, sidx[t])
    struct.pack_into("<III", out, proto_ids_off, sidx["V"], tidx["V"], 0)
    for i, (desc, m) in enumerate(methods):
        struct.pack_into("<HHI", out, meth_ids_off + 8 * i,
                         tidx[desc], 0, sidx[m])
    for i, (desc, _ms) in enumerate(classes):
        off, _cd = class_data[desc]
        struct.pack_into("<IIIIIIII", out, class_defs_off + 32 * i,
                         tidx[desc], 0x1, tidx["Ljava/lang/Object;"],
                         0, 0xFFFFFFFF, 0, off, 0)
    out[data_off:data_off + len(blob)] = blob
    out[map_off:map_off + len(mapb)] = mapb
    out[12:32] = hashlib.sha1(bytes(out[32:])).digest()
    struct.pack_into("<I", out, 8, zlib.adler32(bytes(out[12:])))
    return bytes(out)


def _make_signer():
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Fixture Dev"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Fixture Org")])
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime(2024, 1, 1))
            .not_valid_after(datetime.datetime(2034, 1, 1))
            .sign(key, hashes.SHA256()))
    return key, cert


def _sign_v1(axml: bytes, key, cert) -> tuple[bytes, bytes, bytes]:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.serialization import pkcs7

    mf = (b"Manifest-Version: 1.0\r\nCreated-By: ApkForge fixture\r\n\r\n"
          b"Name: AndroidManifest.xml\r\nSHA-256-Digest: "
          + base64.b64encode(hashlib.sha256(axml).digest()) + b"\r\n\r\n")
    sf = (b"Signature-Version: 1.0\r\nCreated-By: ApkForge fixture\r\n"
          b"SHA-256-Digest-Manifest: "
          + base64.b64encode(hashlib.sha256(mf).digest()) + b"\r\n\r\n")
    p7 = (pkcs7.PKCS7SignatureBuilder().set_data(sf)
          .add_signer(cert, key, hashes.SHA256())
          .sign(serialization.Encoding.DER,
                [pkcs7.PKCS7Options.Binary]))
    return mf, sf, p7


V1 = {"package": "com.example.fixture", "version_code": 1,
      "version_name": "1.0", "min_sdk": 24, "target_sdk": 34,
      "label": "Fixture App", "debuggable": True,
      "permissions": ["android.permission.INTERNET",
                      "android.permission.CAMERA",
                      "android.permission.ACCESS_FINE_LOCATION"]}
V1_STRINGS = ["Lcom/example/fixture/MainActivity;",
              "Lcom/example/fixture/Tracker;",
              "Lcom/google/android/gms/analytics/Tracker;",
              "Ljava/lang/Object;", "<init>", "onCreate", "trackEvent",
              "https://api.example.com/v1/events"]
V1_CLASSES = [("Lcom/example/fixture/MainActivity;",
               ["<init>", "onCreate"]),
              ("Lcom/example/fixture/Tracker;", ["<init>", "trackEvent"])]

V2 = dict(V1, version_code=2, version_name="1.1",
          permissions=["android.permission.INTERNET",
                       "android.permission.ACCESS_FINE_LOCATION",
                       "android.permission.RECORD_AUDIO"])
V2_STRINGS = ["Lcom/example/fixture/MainActivity;",
              "Lcom/example/fixture/Settings;",
              "Ljava/lang/Object;", "<init>", "onCreate", "onResume",
              "https://api.example.com/v1/events",
              "https://api.example.com/v2/metrics"]
V2_CLASSES = [("Lcom/example/fixture/MainActivity;",
               ["<init>", "onCreate", "onResume"]),
              ("Lcom/example/fixture/Settings;", ["<init>"])]


# a genuine referenced call into the GMS analytics SDK — the SDK
# census must find it in the method table, not the string pool
GMS_REF = (("Lcom/google/android/gms/analytics/Tracker;",
            "trackEvent"),)


def build_apk(path: str, manifest: dict, strings: list[str],
              classes: list, signer) -> dict:
    key, cert = signer
    axml = _build_axml(manifest)
    dex = _build_dex(strings, classes, GMS_REF)
    mf, sf, p7 = _sign_v1(axml, key, cert)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("AndroidManifest.xml", axml)
        z.writestr("classes.dex", dex)
        z.writestr("resources.arsc", b"\x02\x00\x0c\x00" + b"\x00" * 60)
        z.writestr("res/drawable/icon.png", b"\x89PNG\r\n\x1a\n")
        z.writestr("assets/config.json",
                   b'{"endpoint":"https://api.example.com"}')
        z.writestr("lib/arm64-v8a/libfixture.so",
                   b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 64)
        z.writestr("lib/armeabi-v7a/libfixture.so",
                   b"\x7fELF\x01\x01\x01\x00" + b"\x00" * 64)
        z.writestr("META-INF/MANIFEST.MF", mf)
        z.writestr("META-INF/CERT.SF", sf)
        z.writestr("META-INF/CERT.RSA", p7)
    return {"cert_sha256": hashlib.sha256(cert.public_bytes(
        __import__("cryptography.hazmat.primitives.serialization",
                   fromlist=["Encoding"]).Encoding.DER)).hexdigest()}


# ── the suite ────────────────────────────────────────────────────────


class ApkForgeA1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="apkforge-a1-")
        cls.signer1 = _make_signer()
        cls.signer2 = _make_signer()
        cls.v1_path = os.path.join(cls.tmp, "fixture-v1.apk")
        cls.v1_truth = build_apk(cls.v1_path, V1, V1_STRINGS,
                                 V1_CLASSES, cls.signer1)
        cls.v2_path = os.path.join(cls.tmp, "fixture-v2.apk")
        cls.v2_truth = build_apk(cls.v2_path, V2, V2_STRINGS,
                                 V2_CLASSES, cls.signer2)
        # APKiD rules compile once per process; warm the dossier here
        from nomorals.agents.apkforge import ApkForge

        ctx = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=os.path.join(cls.tmp, "ws")))
        cls.forge = ApkForge(ctx)
        cls.add1 = cls.forge.add(cls.v1_path)
        assert "project" in cls.add1, cls.add1
        cls.diss1 = cls.forge.dissect(cls.add1["project"]["id"])
        assert "dossier" in cls.diss1, cls.diss1

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _forge(self, sub: str = "ws2") -> object:
        from nomorals.agents.apkforge import ApkForge

        ctx = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=os.path.join(self.tmp, sub)))
        return ApkForge(ctx)

    # ── intake ───────────────────────────────────────────────────────
    def test_intake_provenance_and_hash(self):
        card = self.add1["project"]
        raw = open(self.v1_path, "rb").read()
        self.assertTrue(self.add1["created"])
        self.assertEqual(card["hash"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(card["size"], len(raw))
        self.assertEqual(card["package"], "com.example.fixture")
        proj = self.forge.resolve(card["id"])
        prov = proj.record["provenance"]
        self.assertEqual(prov["type"], "upload")
        self.assertEqual(prov["sha256"], card["hash"])
        self.assertTrue(prov["fetched_at"])
        # immutable original + idempotent re-landing
        again = self.forge.add(self.v1_path)
        self.assertFalse(again["created"])
        self.assertEqual(again["project"]["id"], card["id"])

    def test_intake_missing_file_is_a_designed_error(self):
        res = self.forge.add(os.path.join(self.tmp, "nope.apk"))
        self.assertIn("no such file", res["error"])

    def test_zip_slip_bundle_refused_clean(self):
        evil = os.path.join(self.tmp, "evil.apkm")
        with zipfile.ZipFile(evil, "w") as z:
            z.writestr("info.json", "{}")
            z.writestr("../escape.apk", b"PK\x03\x04")
            z.writestr("base.apk", open(self.v1_path, "rb").read())
        forge = self._forge("ws-slip")
        with self.assertRaises(Exception) as cm:
            from nomorals.agents.apkforge.intake import land_file

            land_file(forge.root, evil)
        self.assertIn("zip-slip", str(cm.exception))
        self.assertFalse(os.path.exists(
            os.path.join(self.tmp, "escape.apk")))
        self.assertEqual(forge.list_projects()["projects"], [])

    def test_bundle_inventory_honest(self):
        bundle = os.path.join(self.tmp, "app.apks")
        raw = open(self.v1_path, "rb").read()
        with zipfile.ZipFile(bundle, "w") as z:
            z.writestr("base.apk", raw)
            z.writestr("splits/config.arm64_v8a.apk", raw)
            z.writestr("toc.pb", b"\x00")
        forge = self._forge("ws-bundle")
        res = forge.add(bundle)
        self.assertIn("project", res)
        self.assertEqual(res["project"]["kind_detail"], "bundle-apks")
        proj = forge.resolve(res["project"]["id"])
        inv = proj.record["inventory"]
        names = [p["name"] for p in inv["parts"]]
        self.assertEqual(len(names), 2)
        part = [p for p in inv["parts"] if p["name"] == "base.apk"][0]
        self.assertEqual(part["package"], "com.example.fixture")
        self.assertIn("A2", inv["note"])
        # the bundle dissects through its base part, and says so
        diss = forge.dissect(res["project"]["id"], include_apkid=False)
        self.assertIn("dossier", diss)
        self.assertEqual(diss["dossier"]["bundle_part"]["part"],
                         "base.apk")

    # ── dissect ground truth ─────────────────────────────────────────
    def test_dissect_identity_and_manifest(self):
        d = self.diss1["dossier"]
        ident = d["identity"]
        self.assertEqual(ident["package"], "com.example.fixture")
        self.assertEqual(ident["app_name"], "Fixture App")
        self.assertEqual(ident["version_name"], "1.0")
        self.assertEqual(ident["version_code"], 1)
        self.assertEqual(ident["min_sdk"], 24)
        self.assertEqual(ident["target_sdk"], 34)
        self.assertEqual(ident["main_activity"],
                         "com.example.fixture.MainActivity")
        self.assertTrue(d["application"]["debuggable"])

    def test_dissect_permissions_with_aosp_weight(self):
        d = self.diss1["dossier"]
        perms = {p["name"]: p for p in d["permissions"]}
        self.assertEqual(set(perms), set(V1["permissions"]))
        self.assertEqual(perms["android.permission.CAMERA"]["weight"],
                         "high")
        self.assertEqual(
            perms["android.permission.ACCESS_FINE_LOCATION"]["weight"],
            "high")
        self.assertEqual(perms["android.permission.INTERNET"]["weight"],
                         "low")
        self.assertIn("take pictures",
                      perms["android.permission.CAMERA"]["label"])

    def test_dissect_components_and_exported(self):
        d = self.diss1["dossier"]
        comps = d["components"]
        acts = {c["name"]: c for c in comps["activities"]}
        self.assertIn("com.example.fixture.MainActivity", acts)
        self.assertTrue(acts["com.example.fixture.MainActivity"]
                        ["exported"])
        filters = acts["com.example.fixture.MainActivity"][
            "intent_filters"]
        self.assertEqual(filters[0]["actions"],
                         ["android.intent.action.MAIN"])
        svcs = {c["name"]: c for c in comps["services"]}
        self.assertFalse(svcs["com.example.fixture.SyncService"]
                         ["exported"])
        provs = {c["name"]: c for c in comps["providers"]}
        self.assertEqual(
            provs["com.example.fixture.DataProvider"]["authorities"],
            "com.example.fixture.data")

    def test_dissect_certificate(self):
        d = self.diss1["dossier"]
        self.assertTrue(d["signing"]["v1"])
        certs = d["signing"]["certificates"]
        self.assertEqual(len(certs), 1)
        self.assertEqual(certs[0]["sha256"], self.v1_truth["cert_sha256"])
        self.assertTrue(certs[0]["self_signed"])
        self.assertIn("Fixture Dev", certs[0]["subject"])

    def test_dissect_dex_census(self):
        d = self.diss1["dossier"]
        totals = d["dex"]["totals"]
        self.assertEqual(totals["classes"], 2)
        # 4 defined methods + the referenced GMS trackEvent call
        self.assertEqual(totals["methods"], 5)
        names = set(d["dex"]["class_names"])
        self.assertIn("Lcom/example/fixture/Tracker;", names)
        self.assertIn("https://api.example.com/v1/events",
                      d["dex"]["urls"])
        sdk = {s["name"] for s in d["sdk_signatures"]}
        self.assertIn("Google Analytics (GMS)", sdk)
        abis = {n["abi"] for n in d["native"]}
        self.assertEqual(abis, {"arm64-v8a", "armeabi-v7a"})

    def test_dissect_apkid_verdict_honest(self):
        d = self.diss1["dossier"]
        self.assertTrue(d["apkid"]["available"])
        descs = [f["description"]
                 for rows in d["apkid"]["matches"].values()
                 for f in rows]
        self.assertIn("ELF", descs)  # the .so magic — all APKiD claims
        self.assertFalse((d["apkid"]["verdicts"] or {}).get("packer"))

    def test_dossier_json_shape_and_feeds(self):
        proj = self.forge.resolve(self.add1["project"]["id"])
        on_disk = proj.read_dossier()
        self.assertIsNotNone(on_disk)
        for key in ("identity", "signing", "permissions", "components",
                    "dex", "native", "apkid", "findings",
                    "source_sha256", "static_note"):
            self.assertIn(key, on_disk)
        self.assertTrue((proj.dir / "dossier.md").exists())
        feed = self.diss1["techforge_feed"]
        self.assertEqual(feed["kind"], "apk-dossier")
        self.assertEqual(feed["package"], "com.example.fixture")
        card = proj.record
        self.assertEqual(card["dossier_sha256"], proj.sha256)

    # ── teach ────────────────────────────────────────────────────────
    def test_teach_factual_fallback_offline(self):
        from unittest import mock

        import nomorals.llm.brain as brain_mod

        with mock.patch.object(brain_mod, "brain_for",
                               return_value=None):
            res = self.forge.teach(self.add1["project"]["id"],
                                   "permissions")
            comp = self.forge.teach(self.add1["project"]["id"],
                                    "components")
        self.assertEqual(res["source"], "dossier")  # no brain → factual
        self.assertIn("android.permission.CAMERA", res["text"])
        self.assertIn("EXPORTED", comp["text"])

    def test_teach_brain_path_when_brain_answers(self):
        from unittest import mock

        import nomorals.llm.brain as brain_mod

        fake = SimpleNamespace(
            chat=lambda msgs: SimpleNamespace(
                text="word " * 120 + "grounded lesson"))
        with mock.patch.object(brain_mod, "brain_for",
                               return_value=fake):
            res = self.forge.teach(self.add1["project"]["id"],
                                   "overview")
        self.assertEqual(res["source"], "brain")
        self.assertIn("grounded lesson", res["text"])

    # ── diff ─────────────────────────────────────────────────────────
    def test_diff_v1_to_v2(self):
        add2 = self.forge.add(self.v2_path)
        res = self.forge.diff(self.add1["project"]["id"],
                              add2["project"]["id"])
        self.assertIn("diff", res)
        delta = res["diff"]
        self.assertEqual(delta["permissions"]["added"],
                         ["android.permission.RECORD_AUDIO"])
        self.assertEqual(delta["permissions"]["removed"],
                         ["android.permission.CAMERA"])
        self.assertIn("Lcom/example/fixture/Tracker;",
                      delta["dex"]["classes"]["removed"])
        self.assertIn("Lcom/example/fixture/Settings;",
                      delta["dex"]["classes"]["added"])
        self.assertIn("Lcom/example/fixture/MainActivity;",
                      delta["dex"]["classes"]["changed"])
        self.assertFalse(delta["certificate"]["same_signer"])
        self.assertEqual(delta["identity"]["version_name"]["to"], "1.1")
        self.assertIn("RECORD_AUDIO", res["md"])
        # journaled to both projects
        pa = self.forge.resolve(self.add1["project"]["id"])
        pb = self.forge.resolve(add2["project"]["id"])
        self.assertTrue(any(r["op"] == "diff"
                            for r in pa.journal_rows()))
        self.assertTrue(any(r["op"] == "diff"
                            for r in pb.journal_rows()))

    # ── capabilities ─────────────────────────────────────────────────
    def test_status_capability_honesty(self):
        st = self.forge.status()
        self.assertEqual(st["capabilities"]["dissect"], "full")
        # A2: edit/build/sign are real ops now — the matrix must
        # report their probed state (full once provisioned, an
        # honest "not provisioned" before), never the A1-era
        # "not yet built" placeholder and never a fake "full".
        for cap in ("edit", "build", "sign", "splits_merge"):
            state = st["capabilities"][cap]
            self.assertNotIn("not yet built", state)
            self.assertTrue(
                state == "full" or "unavailable" in state,
                f"{cap} reported an unverifiable state: {state!r}")
        self.assertIn("install", st["capabilities"]["install_test"])


if __name__ == "__main__":
    unittest.main()
