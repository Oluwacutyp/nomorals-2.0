"""CipherAgent — real encryption at the tip of the tool system.

Wraps :mod:`nomorals.core.cipher` (hermetic pure-Python AES + classic
ciphers) as an agent and a tool:

* ``cipher``  — encrypt/decrypt with AES (passphrase or raw key),
  classic ciphers (caesar/vigenere/atbash/xor), base64, HMAC.
* The agent form persists a *cipher vault*: sealed blobs are stored in
  the workspace so secrets can be recovered later without re-deriving.

    from nomorals.agents.cipher import CipherAgent
    agent = CipherAgent(context)
    blob = agent.work({"action": "encrypt", "data": "db url: postgres://…",
                       "passphrase": "hunter2"})
    back = agent.work({"action": "decrypt", "blob": blob,
                       "passphrase": "hunter2"})

Registered as the ``cipher`` tool.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from ..core.cipher import (CipherError, aes_decrypt, aes_decrypt_cbc,
                           aes_encrypt, aes_encrypt_cbc, atbash, b64_decode,
                           b64_encode, caesar, derive_key, hmac_hex, vigenere,
                           xor_cipher)
from ..core.errors import ToolError
from ..core.policy import Capability
from .base import Agent

_log = logging.getLogger("nomorals.agents.cipher")

__all__ = ["CipherAgent", "register"]


class CipherAgent(Agent):
    """Encrypt/decrypt anything, classically or with real AES."""

    role = "cipher"
    required_capabilities = (Capability.FS_READ, Capability.FS_WRITE)

    def work(self, task_input: Any) -> Any:
        spec = self._normalize(task_input)
        action = spec.get("action", "encrypt")
        data = spec.get("data", "")
        blob = spec.get("blob", "")
        passphrase = spec.get("passphrase", "")
        key = spec.get("key", "")
        mode = spec.get("mode", "ctr")
        algorithm = spec.get("algorithm", "")
        shift = int(spec.get("shift", 3) or 3)
        keyword = spec.get("keyword", "")
        decrypt_flag = bool(spec.get("decrypt", False))

        if action == "encrypt":
            if not data:
                raise ToolError("cipher encrypt needs data")
            out = aes_encrypt(data, passphrase=passphrase, key=key,
                              mode=mode)
            return {"action": "encrypt", "blob": out, "mode": mode,
                    "bytes": len(out)}
        if action == "decrypt":
            if not blob:
                raise ToolError("cipher decrypt needs blob")
            out = aes_decrypt(blob, passphrase=passphrase, key=key)
            try:
                text = out.decode("utf-8")
            except UnicodeDecodeError:
                return {"action": "decrypt", "bytes": len(out),
                        "hex": out.hex()}
            return {"action": "decrypt", "data": text, "bytes": len(out)}
        if action == "classic":
            if not data:
                raise ToolError("cipher classic needs data")
            alg = (algorithm or "caesar").lower()
            if alg == "caesar":
                out = caesar(data, shift, decrypt=decrypt_flag)
            elif alg == "vigenere":
                if not keyword:
                    raise ToolError("vigenere needs keyword")
                out = vigenere(data, keyword, decrypt=decrypt_flag)
            elif alg == "atbash":
                out = atbash(data)
            elif alg == "xor":
                key = key or "codebeast"
                raw = xor_cipher(data if isinstance(data, bytes)
                                 else data.encode(), key.encode())
                out = raw.decode("utf-8", "ignore")
            elif alg in ("b64", "base64"):
                out = (b64_encode(data) if not decrypt_flag
                       else b64_decode(data).decode("utf-8", "ignore"))
            else:
                raise ToolError(
                    f"unknown classic algorithm {alg!r} "
                    "(caesar|vigenere|atbash|xor|b64)")
            return {"action": "classic", "algorithm": alg, "data": out}
        if action == "derive":
            if not passphrase:
                raise ToolError("cipher derive needs passphrase")
            salt = spec.get("salt", b"") or b"nomorals"
            iters = int(spec.get("iterations", 100_000) or 100_000)
            key = derive_key(passphrase, salt if isinstance(salt, bytes)
                             else salt.encode(), iterations=iters,
                             key_len=32)
            return {"action": "derive", "key_hex": key.hex(),
                    "iterations": iters}
        if action == "hmac":
            if not key:
                raise ToolError("cipher hmac needs key")
            return {"action": "hmac", "digest": hmac_hex(key, data)}
        if action == "formats":
            return {"action": "formats",
                    "algorithms": ["aes-ctr", "aes-cbc", "caesar",
                                   "vigenere", "atbash", "xor", "b64",
                                   "hmac-sha256", "pbkdf2"]}
        if action in ("vault_put", "vault_get", "vault_list", "vault_rm"):
            return self._vault(action, spec.get("name", ""),
                               data, passphrase)
        if action in ("vault_export", "vault_import"):
            return self._vault_export_import(
                action, spec.get("path", ""), passphrase,
                spec.get("entry_pass", ""))
        raise ToolError(f"unknown cipher action {action!r}")

    # ── named secrets vault (AES-256 per entry, key-hierarchy sealed) ─────
    _VAULT_DDL = ("CREATE TABLE IF NOT EXISTS cipher_vault "
                 "(name TEXT PRIMARY KEY, blob TEXT NOT NULL, "
                 "key_scheme TEXT NOT NULL DEFAULT 'pass', "
                 "created_at REAL NOT NULL, updated_at REAL NOT NULL)")

    @staticmethod
    def _master_key() -> str:
        """The optional master key (NM_VAULT_KEY env / ~/.nomorals/.env).

        Entries put under it (key_scheme='master') need no per-entry
        passphrase — but only the master key can open them.  Entries put
        with a passphrase (key_scheme='pass') are unaffected."""
        return os.environ.get("NM_VAULT_KEY", "").strip()

    def _vault(self, action: str, name: str, data: str,
               passphrase: str) -> dict[str, Any]:
        """Named secrets: AES-256-CBC sealed per entry.  The key material
        never touches disk — only its sealed output does."""
        db = getattr(self.context, "db", None)
        if db is None:
            raise ToolError("cipher vault needs a context with a database")
        name = (name or "").strip()
        if action in ("vault_put", "vault_get", "vault_rm") and not name:
            raise ToolError(f"{action} needs name")
        db.execute(self._VAULT_DDL)
        now = time.time()
        if action == "vault_put":
            if not data:
                raise ToolError("vault put needs data (the secret)")
            master = self._master_key()
            if passphrase:
                scheme, key = "pass", passphrase
            elif master:
                scheme, key = "master", master
            else:
                raise ToolError(
                    "vault put needs a passphrase (or set NM_VAULT_KEY "
                    "for master-key entries — secrets are never stored "
                    "in plaintext)")
            blob = aes_encrypt_cbc(data, passphrase=key)
            row = db.query_one("SELECT 1 FROM cipher_vault WHERE name=?",
                               (name,))
            if row is not None:
                db.execute("UPDATE cipher_vault SET blob=?, key_scheme=?, "
                           "updated_at=? WHERE name=?",
                           (blob, scheme, now, name))
            else:
                db.execute(
                    "INSERT INTO cipher_vault "
                    "(name, blob, key_scheme, created_at, updated_at) "
                    "VALUES (?,?,?,?,?)",
                    (name, blob, scheme, now, now))
            return {"action": "vault_put", "name": name, "stored": True,
                    "key_scheme": scheme, "bytes": len(blob)}
        if action == "vault_get":
            row = db.query_one(
                "SELECT blob, key_scheme FROM cipher_vault WHERE name=?",
                (name,))
            if row is None:
                raise ToolError(f"no vault entry {name!r}")
            scheme = row.get("key_scheme") or "pass"
            if scheme == "master":
                if not self._master_key():
                    raise ToolError(
                        f"{name!r} is sealed with the master key — set "
                        "NM_VAULT_KEY to open it")
                key = self._master_key()
            else:
                if not passphrase:
                    raise ToolError(f"{action} needs passphrase")
                key = passphrase
            try:
                raw = aes_decrypt_cbc(row["blob"], passphrase=key)
            except CipherError:
                if scheme == "master":
                    raise ToolError(
                        "wrong key (or the entry is corrupted) — the "
                        "master key is set via NM_VAULT_KEY") from None
                raise ToolError("wrong passphrase (or the entry is "
                                "corrupted)") from None
            out: dict[str, Any] = {"action": "vault_get", "name": name,
                                   "key_scheme": scheme}
            try:
                out["data"] = raw.decode("utf-8")
            except UnicodeDecodeError:
                out["hex"] = raw.hex()
            return out
        if action == "vault_rm":
            cur = db.execute("DELETE FROM cipher_vault WHERE name=?", (name,))
            return {"action": "vault_rm", "name": name,
                    "removed": bool(cur.rowcount)}
        rows = db.query("SELECT name, key_scheme, created_at, updated_at "
                        "FROM cipher_vault ORDER BY name")
        return {"action": "vault_list", "count": len(rows),
                "names": [r["name"] for r in rows],
                "entries": [{
                    "name": r["name"],
                    "key_scheme": r.get("key_scheme") or "pass",
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                } for r in rows]}

    def _vault_export_import(self, action: str, path: str, passphrase: str,
                             entry_pass: str) -> dict[str, Any]:
        """Encrypted vault export/import — move the whole vault between
        machines under ONE new key.

        * ``vault_export`` — read every entry (master entries need
          NM_VAULT_KEY set; passphrase entries are opened with
          ``entry_pass`` or the export passphrase), re-seal everything
          under the export passphrase, write a single JSON file.
        * ``vault_import`` — read the file, re-seal every entry under the
          import passphrase, store it (overwriting same-named entries).
        """
        db = getattr(self.context, "db", None)
        if db is None:
            raise ToolError("cipher vault needs a context with a database")
        if not passphrase:
            raise ToolError(f"{action} needs passphrase (the file key)")
        db.execute(self._VAULT_DDL)
        if action == "vault_export":
            if not path:
                raise ToolError("vault export needs path= (the file)")
            rows = db.query("SELECT name, blob, key_scheme FROM "
                            "cipher_vault ORDER BY name")
            entries: dict[str, str] = {}
            skipped: list[str] = []
            for r in rows:
                scheme = r.get("key_scheme") or "pass"
                if scheme == "master":
                    key = self._master_key()
                    if not key:
                        skipped.append(r["name"])
                        continue
                else:
                    key = entry_pass or passphrase
                    if not key:
                        skipped.append(r["name"])
                        continue
                try:
                    raw = aes_decrypt_cbc(r["blob"], passphrase=key)
                except CipherError:
                    skipped.append(r["name"])
                    continue
                entries[r["name"]] = aes_encrypt_cbc(
                    raw.decode("utf-8", "ignore"), passphrase=passphrase)
            bundle = {"format": "nomorals-vault", "version": 1,
                      "created": time.time(), "entries": entries}
            target = path
            if not os.path.isabs(target):
                ws = getattr(self.context, "settings", None)
                wsdir = getattr(ws, "workspace_dir", "") if ws else ""
                if wsdir:
                    target = os.path.join(wsdir, target)
            parent = os.path.dirname(os.path.abspath(target))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(target, "w", encoding="utf-8") as fh:
                json.dump(bundle, fh)
            return {"action": "vault_export", "path": target,
                    "exported": len(entries), "skipped": skipped,
                    "note": ("entries need NM_VAULT_KEY / a valid "
                             "entry-pass to be read" if skipped else "")}
        # vault_import
        if not path:
            raise ToolError("vault import needs path= (the file)")
        if not os.path.isabs(path) and not os.path.exists(path):
            ws = getattr(self.context, "settings", None)
            wsdir = getattr(ws, "workspace_dir", "") if ws else ""
            if wsdir and os.path.exists(os.path.join(wsdir, path)):
                path = os.path.join(wsdir, path)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                bundle = json.load(fh)
        except (OSError, ValueError) as exc:
            raise ToolError(f"cannot read vault file: {exc}") from None
        if not isinstance(bundle, dict) or bundle.get("format")                 != "nomorals-vault" or "entries" not in bundle:
            raise ToolError("not a nomorals vault export file")
        now = time.time()
        imported: list[str] = []
        failed: list[str] = []
        for name, blob in (bundle.get("entries") or {}).items():
            try:
                raw = aes_decrypt_cbc(blob, passphrase=passphrase)
            except CipherError:
                failed.append(name)
                continue
            new_blob = aes_encrypt_cbc(
                raw.decode("utf-8", "ignore"), passphrase=passphrase)
            db.execute("INSERT INTO cipher_vault "
                       "(name, blob, key_scheme, created_at, updated_at) "
                       "VALUES (?,?, 'pass', ?, ?) "
                       "ON CONFLICT(name) DO UPDATE SET blob=excluded.blob, "
                       "key_scheme='pass', updated_at=excluded.updated_at",
                       (name, new_blob, now, now))
            imported.append(name)
        if failed:
            raise ToolError(
                f"wrong passphrase for {len(failed)} entries "
                f"(e.g. {failed[:3]}) — {len(imported)} imported before "
                "the failure")
        return {"action": "vault_import", "imported": imported,
                "count": len(imported)}

    @staticmethod
    def _normalize(task_input: Any) -> dict[str, Any]:
        if isinstance(task_input, dict):
            return dict(task_input)
        if isinstance(task_input, str):
            s = task_input.strip()
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, TypeError):
                pass
            return {"action": "decrypt" if s.startswith("nmc1:") else "encrypt",
                    "data": s, "blob": s}
        raise ToolError("cipher task must be a dict or JSON string")


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "cipher",
        description=(
            "Real cryptography: AES-256 (CTR/CBC) encrypt/decrypt from a "
            "passphrase (PBKDF2) or raw key, plus classic ciphers "
            "(caesar/vigenere/atbash/xor), base64, and HMAC-SHA256. "
            "action=encrypt (data, passphrase|key, mode) | decrypt (blob, "
            "passphrase|key) | classic (data, algorithm, shift, keyword, "
            "decrypt) | derive (passphrase) | hmac (key, data) | formats | "
            "vault_put (name, data, passphrase) | vault_get (name, "
            "passphrase) | vault_list | vault_rm (name) | vault_export "
            "(path, passphrase, entry_pass) | vault_import (path, "
            "passphrase) — the named secrets vault (each entry "
            "AES-256 sealed; NM_VAULT_KEY master-key entries need no "
            "per-entry passphrase)."
        ),
        capability=Capability.FS_READ,
    )
    def cipher(
        action: str = "encrypt", data: str = "", blob: str = "",
        passphrase: str = "", key: str = "", mode: str = "ctr",
        algorithm: str = "", shift: int = 3, keyword: str = "",
        decrypt: bool = False, iterations: int = 100_000,
        name: str = "", path: str = "", entry_pass: str = "",
    ) -> dict[str, Any]:
        agent = CipherAgent(context=context, name="cipher-tool")
        try:
            return agent.work({
                "action": action, "data": data, "blob": blob,
                "passphrase": passphrase, "key": key, "mode": mode,
                "algorithm": algorithm, "shift": shift, "keyword": keyword,
                "decrypt": decrypt, "iterations": iterations, "name": name,
                "path": path, "entry_pass": entry_pass,
            })
        except CipherError as exc:
            raise ToolError(str(exc)) from exc
