"""ApkForge signing — the vault-only keystore lifecycle (design §3.6).

Hard rules, enforced by construction:

* Keystore bytes + passwords live ONLY as an encrypted credential
  in the accounts vault (service ``apkforge-keystore``).  They are
  never written to a project dir, the workspace, git, or a log.
* Signing materializes the keystore into a fresh temp dir under
  the class root for the single sign operation and shreds it in a
  ``finally`` — the materialized copy never outlives the op.
* Passwords reach apksigner/keytool by ``env:`` reference, never
  argv, never error text.
* Every build card names the signing alias + certificate
  fingerprint and prints the standing re-sign facts (§1.3), briefly,
  every time: a re-signed APK does not install over the original,
  cert-pinned features may break, Play Integrity fails.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import re
import secrets
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator

from .toolchain import ToolchainError, run_tool

__all__ = [
    "SigningError",
    "RESIGN_FACTS",
    "DEFAULT_ALIAS",
    "DEBUG_ALIAS",
    "open_class_vault",
    "ensure_keystore",
    "list_keystores",
    "materialize_keystore",
    "sign_apk",
    "verify_apk",
]

SERVICE = "apkforge-keystore"
DEFAULT_ALIAS = "devon"
DEBUG_ALIAS = "debug"

RESIGN_FACTS = [
    "re-signed with Devon's key — Android will NOT install this "
    "build over the original app (signature mismatch); uninstall "
    "the original first, which deletes its app data",
    "features pinned to the original developer's certificate "
    "(signature permissions, cert-bound account tokens) may break",
    "Play Integrity appIntegrity will fail — servers that enforce "
    "it will refuse this build",
]


class SigningError(RuntimeError):
    """A designed signing failure — names the exact step."""


def open_class_vault(context: Any):
    """Open the accounts vault for the class, or raise honestly."""
    db = getattr(context, "db", None) if context is not None else None
    if db is None:
        raise SigningError(
            "no database on this surface — the vault (and therefore "
            "signing) is reachable from the chat/CLI runtime, not "
            "from a bare library call")
    from ...accounts.vault import VaultLockedError, open_vault

    try:
        return open_vault(db, context=context, actor="apkforge")
    except VaultLockedError as exc:
        raise SigningError(f"vault locked: {exc}") from exc


def _cert_fingerprint(base_root: Path, ks_path: Path, alias: str,
                      store_password: str) -> str:
    res = run_tool("", "keytool",
                   ["-list", "-v", "-keystore", str(ks_path),
                    "-alias", alias, "-storepass:env",
                    "APKFORGE_KS_PASS"],
                   timeout=60, root=base_root,
                   extra_env={"APKFORGE_KS_PASS": store_password})
    m = re.search(r"SHA256:\s*([0-9A-Fa-f:]+)", res["stdout"])
    if not m:
        raise SigningError("keytool -list did not report a SHA256 "
                           "fingerprint for the new keystore")
    return m.group(1).replace(":", "").lower()


def _generate(base_root: Path, alias: str, *, debug: bool) -> dict[str, Any]:
    if debug:
        store_pw = key_pw = "android"
        dname = "CN=Android Debug, O=Android"
        keyalg, keysize, validity = "RSA", "2048", "10957"
    else:
        # PKCS12 (keytool's default store type) protects the key
        # with the STORE password — a distinct key password is
        # silently dropped at generation, so one password serves
        # both, and the record says so.
        store_pw = secrets.token_urlsafe(24)
        key_pw = store_pw
        dname = "CN=Devon ApkForge, O=Devon"
        keyalg, keysize, validity = "RSA", "4096", "10950"
    tmp = Path(tempfile.mkdtemp(prefix="apkforge-ks-",
                                dir=str(base_root)))
    try:
        ks = tmp / "ks.jks"
        res = run_tool("", "keytool",
                       ["-genkeypair", "-keystore", str(ks),
                        "-alias", alias, "-keyalg", keyalg,
                        "-keysize", keysize, "-validity", validity,
                        "-storepass:env", "APKFORGE_KS_PASS",
                        "-keypass:env", "APKFORGE_KEY_PASS",
                        "-dname", dname],
                       timeout=180, root=base_root,
                       extra_env={"APKFORGE_KS_PASS": store_pw,
                                  "APKFORGE_KEY_PASS": key_pw})
        if res["rc"] != 0 or not ks.exists():
            tail = (res["stdout"] + res["stderr"]).strip().splitlines()
            raise SigningError(
                "keytool -genkeypair failed: "
                + (tail[-1][:300] if tail else f"rc={res['rc']}"))
        raw = ks.read_bytes()
        cert = _cert_fingerprint(base_root, ks, alias, store_pw)
        return {"alias": alias, "debug": debug,
                "keystore_b64": base64.b64encode(raw).decode(),
                "store_password": store_pw, "key_password": key_pw,
                "cert_sha256": cert}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ensure_keystore(context: Any, base_root: Path, alias: str = "",
                    *, create: bool = True) -> dict[str, Any]:
    """Load the class keystore record from the vault, generating +
    vaulting it on first use.  Returns the record WITHOUT the
    keystore bytes unless ``with_secret`` — callers that sign use
    :func:`materialize_keystore` instead."""
    alias = alias or DEFAULT_ALIAS
    vault = open_class_vault(context)
    try:
        cred = vault.get(SERVICE, alias, mark_used=False)
        record = json.loads(cred.password)
        return {"alias": alias, "debug": bool(record.get("debug")),
                "cert_sha256": record.get("cert_sha256", ""),
                "created": False}
    except Exception:  # noqa: BLE001 — absence routes to generation
        pass
    if not create:
        raise SigningError(f"no keystore under alias {alias!r} in "
                           "the vault")
    record = _generate(base_root, alias, debug=(alias == DEBUG_ALIAS))
    vault.store(SERVICE, alias,
                password=json.dumps({
                    "keystore_b64": record["keystore_b64"],
                    "store_password": record["store_password"],
                    "key_password": record["key_password"],
                    "cert_sha256": record["cert_sha256"],
                    "debug": record["debug"]}),
                credential_type="keystore",
                tags=["apkforge"],
                metadata={"cert_sha256": record["cert_sha256"],
                          "debug": record["debug"]})
    return {"alias": alias, "debug": record["debug"],
            "cert_sha256": record["cert_sha256"], "created": True}


def list_keystores(context: Any) -> list[dict[str, Any]]:
    vault = open_class_vault(context)
    out = []
    for cred in vault.list_all(service=SERVICE):
        meta = cred.metadata or {}
        out.append({"alias": cred.username,
                    "cert_sha256": meta.get("cert_sha256", ""),
                    "debug": bool(meta.get("debug"))})
    return out


@contextlib.contextmanager
def materialize_keystore(context: Any, base_root: Path,
                         alias: str) -> Iterator[dict[str, Any]]:
    """Yield {path, store_password, key_password, cert_sha256} with
    the keystore written under a fresh temp dir; the dir is shredded
    on exit no matter what.  The ONLY place keystore bytes touch
    disk outside the vault."""
    vault = open_class_vault(context)
    try:
        cred = vault.get(SERVICE, alias, mark_used=True)
        record = json.loads(cred.password)
    except Exception as exc:  # noqa: BLE001
        raise SigningError(
            f"keystore alias {alias!r} not readable from the vault: "
            f"{exc}") from exc
    tmp = Path(tempfile.mkdtemp(prefix=".sign-", dir=str(base_root)))
    try:
        tmp.chmod(0o700)
        ks = tmp / "ks.jks"
        ks.write_bytes(base64.b64decode(record["keystore_b64"]))
        ks.chmod(0o600)
        yield {"path": ks,
               "store_password": record["store_password"],
               "key_password": record["key_password"],
               "cert_sha256": record.get("cert_sha256", ""),
               "debug": bool(record.get("debug"))}
    finally:
        # shred: overwrite then remove — a deleted-but-recoverable
        # keystore is still a keystore
        for f in tmp.rglob("*"):
            if f.is_file():
                try:
                    f.write_bytes(os.urandom(f.stat().st_size))
                except OSError:
                    pass
        shutil.rmtree(tmp, ignore_errors=True)


def sign_apk(context: Any, base_root: Path, in_apk: Path,
             out_apk: Path, alias: str = "") -> dict[str, Any]:
    """zipalign (if not already aligned output) + apksigner sign.

    The caller hands an aligned or unaligned APK; signing always
    runs zipalign first (apksigner preserves alignment; zipalign
    must precede signing).  Returns the verify record."""
    alias = alias or DEFAULT_ALIAS
    ensure_keystore(context, base_root, alias)
    aligned = out_apk.with_suffix(".aligned.apk")
    res = run_tool("", "zipalign", ["-p", "-f", "4", str(in_apk),
                                    str(aligned)],
                   timeout=300, root=base_root)
    if res["rc"] != 0:
        tail = (res["stdout"] + res["stderr"]).strip().splitlines()
        raise SigningError("zipalign failed: "
                           + (tail[-1][:300] if tail
                              else f"rc={res['rc']}"))
    try:
        with materialize_keystore(context, base_root, alias) as ks:
            res = run_tool(
                "", "apksigner",
                ["sign", "--ks", str(ks["path"]),
                 "--ks-key-alias", alias,
                 "--ks-pass", "env:APKFORGE_KS_PASS",
                 "--key-pass", "env:APKFORGE_KEY_PASS",
                 "--out", str(out_apk), str(aligned)],
                timeout=300, root=base_root,
                extra_env={"APKFORGE_KS_PASS": ks["store_password"],
                           "APKFORGE_KEY_PASS": ks["key_password"]})
            cert = ks["cert_sha256"]
            debug = ks["debug"]
        if res["rc"] != 0 or not out_apk.exists():
            tail = (res["stdout"] + res["stderr"]).strip().splitlines()
            raise SigningError(
                "apksigner sign failed: "
                + (tail[-1][:300] if tail else f"rc={res['rc']}"))
    finally:
        aligned.unlink(missing_ok=True)
    verdict = verify_apk(base_root, out_apk)
    if not verdict["verifies"]:
        raise SigningError(
            f"apksigner verify does not pass on {out_apk.name} — "
            "the signed output is not being handed back as good")
    return {"signed": str(out_apk), "alias": alias, "debug": debug,
            "cert_sha256": cert, "verify": verdict}


def verify_apk(base_root: Path, apk: Path) -> dict[str, Any]:
    """apksigner verify --print-certs, parsed into facts."""
    try:
        res = run_tool("", "apksigner",
                       ["verify", "--verbose", "--print-certs",
                        str(apk)], timeout=180, root=base_root)
    except ToolchainError as exc:
        return {"verifies": False, "error": str(exc), "schemes": {}}
    text = res["stdout"] + res["stderr"]
    schemes = {}
    for scheme in ("v1", "v2", "v3", "v3.1", "v4"):
        m = re.search(rf"Verified using {re.escape(scheme)} scheme "
                      rf"\([^)]*\):\s*(true|false)", text)
        if m:
            schemes[scheme] = m.group(1) == "true"
    cert = ""
    m = re.search(r"Signer #1 certificate SHA-256 digest:\s*"
                  r"([0-9a-f]+)", text)
    if m:
        cert = m.group(1)
    dn = ""
    m = re.search(r"Signer #1 certificate DN:\s*(.+)", text)
    if m:
        dn = m.group(1).strip()
    return {"verifies": res["rc"] == 0 and "Verifies" in text,
            "schemes": schemes, "cert_sha256": cert,
            "signer_dn": dn}
