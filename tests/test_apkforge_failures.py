"""Pass #108 pins: apkforge failures stop vanishing.

Every pin drives a real converted site (30 sites across 7 files) and
asserts the fallback still holds while the failure is now recorded.
The ledger itself is pinned first (basics, bound, unprintable
exceptions, empty render), then the package ``__all__`` (unchanged —
the ledger is a direct import, not a widened export), then the sites,
then a scanner pin asserting the whole apkforge package is clean.
"""

import base64
import hashlib
import io
import json
import sys
import tarfile
import types
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents.apkforge import failures as ledger
from nomorals.agents.apkforge.failures import (
    clear_failures,
    failure_count,
    format_failures,
    recent_failures,
    record_failure,
)


@contextmanager
def _sys_modules(names):
    """Temporarily install stub modules, restoring sys.modules after."""
    saved = {n: sys.modules.get(n, None) for n in names}
    try:
        yield
    finally:
        for n, mod in saved.items():
            if mod is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod


def _sites():
    return [f.site for f in recent_failures(1000)]


def _recorded(site):
    return site in _sites()


# ── ledger basics ──────────────────────────────────────────────────


def test_ledger_basics():
    clear_failures()
    assert failure_count() == 0
    assert format_failures() == "no apkforge failures recorded."
    record_failure("apkforge.test", ValueError("bad value"))
    assert failure_count() == 1
    (entry,) = recent_failures()
    assert entry.site == "apkforge.test"
    assert entry.exc_type == "ValueError"
    assert entry.message == "bad value"
    assert "apkforge failures (1 total" in format_failures()
    assert "apkforge.test: ValueError: bad value" in format_failures()
    clear_failures()
    assert failure_count() == 0
    assert recent_failures() == []


def test_ledger_bound():
    clear_failures()
    for i in range(250):
        record_failure(f"apkforge.bound#{i}", RuntimeError(str(i)))
    assert failure_count() == 250
    assert len(recent_failures(1000)) == 200
    assert recent_failures(1)[0].site == "apkforge.bound#249"
    clear_failures()


def test_ledger_unprintable_exc():
    clear_failures()

    class _BadStr(Exception):
        def __str__(self):
            raise RuntimeError("cannot print")

    record_failure("apkforge.unprintable", _BadStr())
    (entry,) = recent_failures()
    assert entry.exc_type == "_BadStr"
    assert "unprintable" in entry.message
    clear_failures()


def test_package_all_unchanged():
    import nomorals.agents.apkforge as pkg

    assert pkg.__all__ == [
        "ApkForge",
        "apkforge_dir",
        "museum_card",
        "techforge_feed",
        "render_list",
        "render_project",
        "render_status",
        "render_toolchain",
        "render_build_card",
        "render_srcbuild_card",
        "render_dossier_card",
        "render_dossier_md",
        "render_diff_card",
        "render_diff_md",
        "render_bisect_card",
        "render_signatures",
        "render_artifacts",
        "register",
    ]


# ── bisect ─────────────────────────────────────────────────────────


def test_bisect_load_session_corrupt(tmp_path):
    clear_failures()
    from nomorals.agents.apkforge import bisect as bisect_mod

    key = bisect_mod.session_key(["proj-a", "proj-b"], "permission:X")
    sdir = bisect_mod._session_dir(tmp_path, key)
    (sdir / "state.json").write_text("not json{", encoding="utf-8")
    state = bisect_mod.load_session(tmp_path, ["proj-a", "proj-b"],
                                    "permission:X")
    assert state["resumed"] is False
    assert state["verdicts"] == {}
    assert _recorded("bisect.load_session")
    clear_failures()


# ── dissect: probes ────────────────────────────────────────────────


def test_dissect_androguard_available(monkeypatch):
    clear_failures()
    from nomorals.agents.apkforge import dissect as dissect_mod

    with _sys_modules(["androguard"]):
        sys.modules["androguard"] = None
        assert dissect_mod.androguard_available() is False
        assert _recorded("dissect.androguard_available")
        clear_failures()
        fake = types.ModuleType("androguard")
        fake.__version__ = "9.9"
        sys.modules["androguard"] = fake
        assert dissect_mod.androguard_available() is True
        assert not _recorded("dissect.androguard_available")
    clear_failures()


def test_dissect_components_from_dom_raising():
    clear_failures()
    from nomorals.agents.apkforge import dissect as dissect_mod

    class _Raising:
        def get_android_manifest_xml(self):
            raise RuntimeError("dom boom")

    components, app_flags = dissect_mod._components_from_dom(
        _Raising(), "com.example.test")
    assert components == {"activities": [], "services": [],
                          "receivers": [], "providers": []}
    assert app_flags == {}
    assert _recorded("dissect._components_from_dom")
    clear_failures()


def test_dissect_dex_census_raising_getters():
    clear_failures()
    from nomorals.agents.apkforge import dissect as dissect_mod

    class _Raising:
        def get_dex_names(self):
            raise RuntimeError("no dex names")

        def get_all_dex(self):
            raise RuntimeError("no dex blobs")

    dex_mod = types.ModuleType("androguard.core.dex")
    dex_mod.DEX = object
    with _sys_modules(["androguard", "androguard.core",
                       "androguard.core.dex"]):
        sys.modules["androguard"] = types.ModuleType("androguard")
        sys.modules["androguard.core"] = types.ModuleType(
            "androguard.core")
        sys.modules["androguard.core.dex"] = dex_mod
        census = dissect_mod._dex_census(_Raising())
    assert census["per_dex"] == []
    assert census["totals"]["classes"] == 0
    assert _recorded("dissect._dex_census")
    assert _recorded("dissect._dex_census#2")
    clear_failures()


def test_dissect_dex_census_fake_dex():
    clear_failures()
    from nomorals.agents.apkforge import dissect as dissect_mod

    class _CDef:
        def get_name(self):
            return "Lcom/example/Foo;"

        def get_superclassname(self):
            raise RuntimeError("no super")

        def get_class_data(self):
            raise RuntimeError("no class data")

    class _Method:
        def get_class_name(self):
            raise RuntimeError("no method class")

    class _FakeDEX:
        def __init__(self, blob):
            pass

        def get_classes(self):
            return [_CDef()]

        def get_methods(self):
            return [_Method()]

        def get_fields(self):
            raise RuntimeError("no fields")

        def get_strings(self):
            raise RuntimeError("no strings")

    class _FakeAPK:
        def get_dex_names(self):
            return ["classes.dex"]

        def get_all_dex(self):
            return [b"fake-dex"]

    dex_mod = types.ModuleType("androguard.core.dex")
    dex_mod.DEX = _FakeDEX
    with _sys_modules(["androguard", "androguard.core",
                       "androguard.core.dex"]):
        sys.modules["androguard"] = types.ModuleType("androguard")
        sys.modules["androguard.core"] = types.ModuleType(
            "androguard.core")
        sys.modules["androguard.core.dex"] = dex_mod
        census = dissect_mod._dex_census(_FakeAPK())
    assert census["per_dex"][0]["classes"] == 1
    for site in ("dissect._dex_census#3", "dissect._dex_census#4",
                 "dissect._dex_census#7", "dissect._dex_census#8",
                 "dissect._dex_census#9"):
        assert _recorded(site), site
    # The complementary sites provably did NOT fire: methods listed,
    # and the per-field loop never ran because get_fields raised.
    assert not _recorded("dissect._dex_census#5")
    assert not _recorded("dissect._dex_census#6")
    clear_failures()


def test_dissect_apk_full_fake(tmp_path, monkeypatch):
    clear_failures()
    from nomorals.agents.apkforge import dissect as dissect_mod

    class _FakeAPK:
        def __init__(self, path):
            pass

        def get_package(self):
            return "com.example.test"

        def get_app_name(self):
            return "Test"

        def get_androidversion_name(self):
            return "1.0"

        def get_androidversion_code(self):
            return "1"

        def get_min_sdk_version(self):
            return "21"

        def get_target_sdk_version(self):
            return "33"

        def get_max_sdk_version(self):
            return None

        def get_main_activity(self):
            return "com.example.test.Main"

        def is_multidex(self):
            return False

        def get_certificates(self):
            return [object()]  # .subject raises -> cert skipped

        def is_signed(self):
            return False

        def is_signed_v1(self):
            return False

        def is_signed_v2(self):
            return False

        def is_signed_v3(self):
            return False

        def get_signature_names(self):
            return []

        def get_details_permissions(self):
            raise RuntimeError("no details")

        def get_permissions(self):
            return []

        def get_declared_permissions(self):
            raise RuntimeError("no declared")

        def get_android_manifest_xml(self):
            return None

        def get_activities(self):
            return []

        def get_services(self):
            return []

        def get_receivers(self):
            return []

        def get_providers(self):
            return []

        def get_dex_names(self):
            return []

        def get_all_dex(self):
            return []

        def get_files(self):
            return ["lib/arm64-v8a/libx.so", "resources.arsc"]

        def get_android_resources(self):
            raise RuntimeError("no arsc")

    apk_path = tmp_path / "test.apk"
    with zipfile.ZipFile(apk_path, "w") as zf:
        zf.writestr("lib/arm64-v8a/libx.so", b"not-an-elf")
    apk_mod = types.ModuleType("androguard.core.apk")
    apk_mod.APK = _FakeAPK
    dex_mod = types.ModuleType("androguard.core.dex")
    dex_mod.DEX = object
    ag = types.ModuleType("androguard")
    ag.__version__ = "9.9"
    monkeypatch.setattr(dissect_mod, "elf_info",
                        lambda data: (_ for _ in ()).throw(
                            RuntimeError("elf boom")))
    with _sys_modules(["androguard", "androguard.core",
                       "androguard.core.apk", "androguard.core.dex"]):
        sys.modules["androguard"] = ag
        sys.modules["androguard.core"] = types.ModuleType(
            "androguard.core")
        sys.modules["androguard.core.apk"] = apk_mod
        sys.modules["androguard.core.dex"] = dex_mod
        dossier = dissect_mod.dissect_apk(apk_path, include_apkid=False)
    assert dossier["identity"]["package"] == "com.example.test"
    native = [r for r in dossier["native"]
              if r["path"] == "lib/arm64-v8a/libx.so"]
    assert len(native) == 1
    assert "elf" not in native[0]
    assert dossier["resources"]["arsc"]["parsed"] is False
    for site in ("dissect.dissect_apk", "dissect.dissect_apk#2",
                 "dissect.dissect_apk#3", "dissect.dissect_apk#4",
                 "dissect.dissect_apk#5"):
        assert _recorded(site), site
    # The dex census ran clean on empty dex: none of its sites fired.
    assert not any(s.startswith("dissect._dex_census") for s in _sites())
    assert not _recorded("dissect._components_from_dom")
    clear_failures()


# ── intake ─────────────────────────────────────────────────────────


def test_intake_land_url_unlink_failure(tmp_path, monkeypatch):
    clear_failures()
    from nomorals.agents.apkforge import intake as intake_mod
    import nomorals.core.http as http_mod

    dest_file = tmp_path / "inbox" / "dl.apk"
    monkeypatch.setattr(
        http_mod, "download",
        lambda url, dest: (Path(dest).mkdir(parents=True, exist_ok=True),
                           (Path(dest) / "dl.apk").write_bytes(b"x"),
                           str(Path(dest) / "dl.apk"))[2])
    monkeypatch.setattr(intake_mod, "land_file",
                        lambda *a, **k: {"project": "landed"})
    real_unlink = Path.unlink

    def _raising_unlink(self, *a, **k):
        raise OSError("unlink blocked in test")

    monkeypatch.setattr(Path, "unlink", _raising_unlink)
    out = intake_mod.land_url(tmp_path, "https://example.invalid/x.apk")
    assert out == {"project": "landed"}
    assert _recorded("intake.land_url")
    monkeypatch.setattr(Path, "unlink", real_unlink)
    clear_failures()


# ── signing ────────────────────────────────────────────────────────


def test_signing_ensure_keystore_absent(tmp_path, monkeypatch):
    clear_failures()
    from nomorals.agents.apkforge import signing as signing_mod

    class _Vault:
        def get(self, *a, **k):
            raise KeyError("no such alias")

    monkeypatch.setattr(signing_mod, "open_class_vault",
                        lambda ctx: _Vault())
    with pytest.raises(signing_mod.SigningError):
        signing_mod.ensure_keystore(None, tmp_path, "ghost",
                                    create=False)
    assert _recorded("signing.ensure_keystore")
    clear_failures()


def test_signing_materialize_shred_failure(tmp_path, monkeypatch):
    clear_failures()
    from nomorals.agents.apkforge import signing as signing_mod

    record = {"keystore_b64": base64.b64encode(b"fake-ks").decode(),
              "store_password": "store-pw", "key_password": "key-pw",
              "cert_sha256": "00" * 32, "debug": False}

    class _Vault:
        def get(self, *a, **k):
            return SimpleNamespace(password=json.dumps(record))

    monkeypatch.setattr(signing_mod, "open_class_vault",
                        lambda ctx: _Vault())
    ctx = signing_mod.materialize_keystore(None, tmp_path, "alias")
    yielded = ctx.__enter__()
    assert yielded["path"].exists()
    real_write = Path.write_bytes

    def _raising_write(self, *a, **k):
        raise OSError("shred blocked in test")

    monkeypatch.setattr(Path, "write_bytes", _raising_write)
    ctx.__exit__(None, None, None)
    monkeypatch.setattr(Path, "write_bytes", real_write)
    assert _recorded("signing.materialize_keystore")
    clear_failures()


# ── srcbuild ───────────────────────────────────────────────────────


def test_srcbuild_sdk_bad_value(tmp_path):
    clear_failures()
    from nomorals.agents.apkforge import srcbuild as srcbuild_mod

    src = tmp_path / "proj"
    src.mkdir()
    (src / "AndroidManifest.xml").write_text(
        '<manifest xmlns:android='
        '"http://schemas.android.com/apk/res/android" '
        'package="com.example.test">'
        '<uses-sdk android:minSdkVersion="abc" '
        'android:targetSdkVersion="33"/>'
        '<application><activity android:name=".Main"/>'
        '</application></manifest>', encoding="utf-8")
    info = srcbuild_mod._manifest_info(src)
    assert info["min_sdk"] == 24  # default held on the bad value
    assert info["target_sdk"] == 33
    assert info["main_activity"] == "com.example.test.Main"
    assert _recorded("srcbuild._manifest_info._sdk")
    clear_failures()


def test_srcbuild_gradle_properties_chmod(tmp_path, monkeypatch):
    clear_failures()
    from nomorals.agents.apkforge import srcbuild as srcbuild_mod

    def _raising_chmod(self, *a, **k):
        raise OSError("chmod blocked in test")

    monkeypatch.setattr(Path, "chmod", _raising_chmod)
    srcbuild_mod._ensure_gradle_properties(tmp_path, truststore=None)
    props = (tmp_path / "tools" / ".gradle" / "gradle.properties")
    assert props.exists()
    assert _recorded("srcbuild._ensure_gradle_properties")
    clear_failures()


def test_srcbuild_run_gradle_chmod(tmp_path, monkeypatch):
    clear_failures()
    from nomorals.agents.apkforge import srcbuild as srcbuild_mod

    src = tmp_path / "src"
    src.mkdir()
    (src / "gradlew").touch()

    class _Proc:
        returncode = 0

        def communicate(self, timeout=None):
            return "", ""

    class _Subprocess:
        PIPE = -1
        TimeoutExpired = type("TimeoutExpired", (Exception,), {})
        Popen = staticmethod(lambda *a, **k: _Proc())

    monkeypatch.setattr(srcbuild_mod, "assemble_sdk_root",
                        lambda **k: tmp_path)
    monkeypatch.setattr(srcbuild_mod, "_ensure_gradle_truststore",
                        lambda root: None)
    monkeypatch.setattr(srcbuild_mod, "_ensure_gradle_properties",
                        lambda *a, **k: None)
    monkeypatch.setattr(srcbuild_mod, "subprocess", _Subprocess())

    def _raising_chmod(self, *a, **k):
        raise OSError("chmod blocked in test")

    monkeypatch.setattr(Path, "chmod", _raising_chmod)
    steps: list = []
    srcbuild_mod._run_gradle(tmp_path, src, ["assembleDebug"], steps)
    assert steps and steps[0]["ok"] is True
    assert _recorded("srcbuild._run_gradle")
    clear_failures()


# ── teach ──────────────────────────────────────────────────────────


def test_teach_lesson_brain_failure(monkeypatch):
    clear_failures()
    from nomorals.agents.apkforge import teach as teach_mod
    import nomorals.llm.brain as brain_mod

    class _RaisingBrain:
        def chat(self, *a, **k):
            raise RuntimeError("brain down")

    monkeypatch.setattr(brain_mod, "brain_for",
                        lambda ctx: _RaisingBrain())
    out = teach_mod.teach_lesson(None, {}, topic="overview")
    assert out["source"] == "dossier"
    assert out["topic"] == "overview"
    assert _recorded("teach.teach_lesson")
    clear_failures()


def test_teach_code_lesson_brain_failure(monkeypatch):
    clear_failures()
    from nomorals.agents.apkforge import teach as teach_mod
    import nomorals.llm.brain as brain_mod

    class _RaisingBrain:
        def chat(self, *a, **k):
            raise RuntimeError("brain down")

    monkeypatch.setattr(brain_mod, "brain_for",
                        lambda ctx: _RaisingBrain())
    smali = (".method public foo()V\n"
             "    .locals 1\n"
             "    const-string v0, \"hi\"\n"
             "    return-void\n"
             ".end method\n")
    out = teach_mod.teach_code_lesson(None, class_name="Lcom/x/Foo;",
                                      smali_text=smali)
    assert out["source"] == "side-by-side"
    assert out["method"] == "foo"
    assert _recorded("teach.teach_code_lesson")
    clear_failures()


# ── toolchain ──────────────────────────────────────────────────────


def test_toolchain_versions_absent(monkeypatch):
    clear_failures()
    from nomorals.agents.apkforge import toolchain as toolchain_mod
    import importlib.metadata as metadata_mod

    with _sys_modules(["androguard"]):
        sys.modules["androguard"] = None
        assert toolchain_mod.androguard_version() is None
        assert _recorded("toolchain.androguard_version")
    monkeypatch.setattr(metadata_mod, "version",
                        lambda name: (_ for _ in ()).throw(
                            metadata_mod.PackageNotFoundError(name)))
    assert toolchain_mod.apkid_version() is None
    assert _recorded("toolchain.apkid_version")
    clear_failures()


def test_toolchain_provision_chmod(tmp_path, monkeypatch):
    clear_failures()
    from nomorals.agents.apkforge import toolchain as toolchain_mod

    # One crafted zip (single top dir + .sh exe) serves every zip pin;
    # one crafted tar serves jdk; one blob serves every jar pin.
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("tool/run.sh", "#!/bin/sh\necho fake-1.0\n")
    zip_bytes = zip_buf.getvalue()
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tf:
        data = b"#!/bin/sh\necho fake-1.0\n"
        info = tarfile.TarInfo("jdk/bin/java")
        info.size = len(data)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(data))
    tar_bytes = tar_buf.getvalue()
    jar_bytes = b"fake-jar-bytes"
    zip_sha = hashlib.sha256(zip_bytes).hexdigest()
    tar_sha = hashlib.sha256(tar_bytes).hexdigest()
    jar_sha = hashlib.sha256(jar_bytes).hexdigest()

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "jadx.zip").write_bytes(zip_bytes)
    (cache / "build-tools.zip").write_bytes(zip_bytes)
    for jar in ("apktool.jar", "bundletool.jar", "apkeditor.jar"):
        (cache / jar).write_bytes(jar_bytes)
    (cache / "jdk.tar.gz").write_bytes(tar_bytes)

    saved_shas = {}
    for name, pin in toolchain_mod.PINS.items():
        saved_shas[name] = pin.get("sha256")
        if pin["kind"] == "zip":
            pin["sha256"] = zip_sha
        elif pin["kind"] == "jar":
            pin["sha256"] = jar_sha
    monkeypatch.setattr(toolchain_mod, "_jdk_artifact",
                        lambda: ("https://fake.invalid/jdk.tar.gz",
                                 tar_sha))
    monkeypatch.setenv("APKFORGE_TOOL_CACHE", str(cache))

    def _raising_chmod(self, *a, **k):
        raise OSError("chmod blocked in test")

    monkeypatch.setattr(Path, "chmod", _raising_chmod)
    try:
        report = toolchain_mod.provision(base="", root=tmp_path)
    finally:
        for name, sha in saved_shas.items():
            toolchain_mod.PINS[name]["sha256"] = sha
    assert "jadx" in report["installed"]
    assert "build-tools" in report["installed"]
    assert "jdk" in report["installed"]
    assert _recorded("toolchain.provision")
    clear_failures()


# ── subsystem scan ─────────────────────────────────────────────────


def test_subsystem_scan_clean():
    from nomorals.tools.error_scan import scan

    report = scan(["nomorals/agents/apkforge"])
    offenders = [f for f in report.findings
                 if f.file.startswith("nomorals/agents/apkforge")]
    assert offenders == []
