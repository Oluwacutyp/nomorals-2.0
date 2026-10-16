"""Ship: GitHub Releases transport + the local outbox.

Pipeline: ``ensure_repo`` -> ``create_release(tag)`` -> upload assets
(encrypted blobs, split into 100MB parts). The backup repo itself only
ever receives manifests (text) via a normal git push — blobs NEVER go
into commit history.

Token scope truth (checked against ``GitHubConnector.create_repo``,
which calls ``POST /user/repos``):

- Classic PAT: needs the ``repo`` scope (creation + contents read/write).
- Fine-grained PAT: repo-scoped permissions alone do NOT allow creating
  a repo. Creating a repo needs **Administration: read + write on the
  user account**. Contents read/write (repo permission) covers pushing
  manifests + releases once the repo exists.
- Minimum for steady-state shipping (repo already exists): Contents
  read/write on ``nomorals-backup``. Nothing else.

The token lives in the vault (via the connector's vault-backed
credential), NOT in ``.env``. The token reaches git through the
connector's ``GIT_ASKPASS`` plumbing — never on a command line.

Failure modes: network down -> snapshot is queued in the local outbox
and retried with backoff (never a silent skip); partial upload (size
mismatch) -> :class:`ChecksumMismatchError`, never "partial success";
token expired (401) -> typed error + notify, never a silent skip;
rate limit (429/403) -> typed error carrying ``retry_after``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .errors import (
    BackupError,
    ChecksumMismatchError,
    RateLimitedBackupError,
    TokenExpiredError,
    TokenScopeError,
)
from .plan import REPO_NAME

__all__ = [
    "BACKOFF_SECONDS",
    "Outbox",
    "ensure_repo",
    "check_token_scopes",
    "create_release_for_backup",
    "upload_release_asset",
    "list_releases",
    "delete_release",
    "download_release_asset",
    "ship_snapshot",
    "ship_manifest_to_repo",
    "ship_to_blob_store",
    "release_tag_for",
]

_log = get_logger(__name__)

UPLOAD_BASE = "https://uploads.github.com"
BACKOFF_SECONDS: tuple[float, ...] = (60.0, 300.0, 900.0, 3600.0, 14400.0)
MIN_SCOPES_NOTE = (
    "minimum scopes: repo creation + Contents read/write; nothing else"
)


def release_tag_for(backup_id: str) -> str:
    return backup_id


def _status_of(exc: Exception) -> int:
    return int(getattr(exc, "status_code", 0) or 0)


def _token(connector: Any) -> str:
    cred = connector._require_credential()  # noqa: SLF001 - token via vault
    return str(cred.password)


def _headers(connector: Any) -> dict[str, str]:
    return connector._headers(_token(connector))


def ensure_repo(
    connector: Any,
    name: str = REPO_NAME,
    *,
    private: bool = True,
    description: str = "Encrypted offsite backups: manifests only, blobs in Releases.",
) -> dict[str, Any]:
    """Idempotent: use the repo if it exists, create it (private) if not.

    Never pushes code into it. A 401 becomes :class:`TokenExpiredError`;
    a 403 on create becomes :class:`TokenScopeError` naming the exact
    scope/permission required.
    """
    # Owner login: prefer the live API, fall back to the stored credential.
    try:
        me = connector._api("GET", "/user")  # noqa: SLF001
        login = str(me.get("login", ""))
    except Exception:  # noqa: BLE001 - fall back to stored credential
        login = ""
    if not login:
        cred = connector._load_credential()  # noqa: SLF001
        login = str(getattr(cred, "username", "") or "")
    if not login:
        raise TokenExpiredError(
            "cannot determine the GitHub account: token rejected or "
            "missing — reconnect with a fresh token"
        )
    full_name = f"{login}/{name}"
    try:
        repo = connector.get_repo(full_name)
    except BackupError:
        raise
    except Exception as exc:  # noqa: BLE001 - 404 vs auth mapped below
        status = _status_of(exc)
        if status == 401:
            raise TokenExpiredError(
                "github rejected the token (401): invalid, expired, or "
                "revoked — reconnect with a fresh token"
            ) from exc
        if status not in (0, 404):
            # A 403 here can mean "no access" (fine-grained PAT without
            # the repo in scope) — surfacing it beats guessing "missing".
            if status == 403:
                raise TokenScopeError(
                    f"cannot read repo {full_name} (403): the token lacks "
                    "access — grant Contents read/write on the repo (or "
                    "Administration read/write to create it).",
                    required="Contents: read/write (existing) / "
                    "Administration: read+write (create)",
                ) from exc
            raise
        repo = None
    if repo is not None:
        if not bool(repo.get("private", True)) and private:
            _log.warning(
                "backup repo %s exists but is PUBLIC — refusing to ship "
                "into a public repo; make it private first", full_name
            )
            raise BackupError(
                f"backup repo {full_name} exists but is public — "
                "backups must be private"
            )
        return {"repo": repo, "full_name": full_name, "created": False}
    try:
        repo = connector.create_repo(
            name, private=private, description=description, auto_init=True
        )
    except Exception as exc:  # noqa: BLE001 - map to typed errors
        status = _status_of(exc)
        message = str(exc)
        if status == 401:
            raise TokenExpiredError(
                "github rejected the token (401) while creating the repo"
            ) from exc
        if status == 403 or "403" in message:
            raise TokenScopeError(
                "creating a repo needs more than repo-scoped permissions: "
                "classic PAT needs the `repo` scope; fine-grained PAT "
                "needs Administration read+write on the user account "
                "(repo-scoped Contents alone cannot create a repo).",
                required="classic: repo scope; fine-grained: "
                "Administration read+write (account)",
            ) from exc
        if status == 422 or "already exists" in message.lower():
            # Race: repo appeared between get and create — re-read it.
            return ensure_repo(connector, name, private=private,
                                description=description)
        raise
    _log.info("github backup repo created: %s (private=%s)",
              repo.get("full_name"), private)
    return {"repo": repo, "full_name": full_name, "created": True}


def check_token_scopes(connector: Any) -> dict[str, Any]:
    """Report what the stored token can do — honestly, incl. unknowns.

    Fine-grained PATs do not report classic scopes via the API, so
    ``can_create_repo`` is ``None`` (unknown) for them unless the stored
    credential metadata says otherwise. Callers must not treat unknown
    as granted.
    """
    cred = connector._load_credential()  # noqa: SLF001
    scopes: list[str] = []
    if cred is not None:
        scopes = list((getattr(cred, "metadata", None) or {}).get("scopes", []))
    fine_grained = scopes == ["fine-grained (see token settings)"] or not scopes
    return {
        "scopes": scopes,
        "fine_grained": fine_grained,
        "can_create_repo": None if fine_grained else ("repo" in scopes),
        "min_scopes": MIN_SCOPES_NOTE,
        "create_repo_needs": (
            "classic PAT: `repo` scope; fine-grained PAT: Administration "
            "read+write on the user account (POST /user/repos)"
        ),
    }


def create_release_for_backup(
    connector: Any, full_name: str, tag: str, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Create the Release that will hold one generation's blobs."""
    from .manifest import render_restore_recipe

    body = (
        f"Encrypted backup {manifest.get('backup_id')} — "
        f"{manifest.get('file_count')} files, "
        f"{manifest.get('total_encrypted_bytes')} encrypted bytes.\n\n"
        "Blobs are AES-256 encrypted; decrypt with the backup key "
        "(vault github-backup/backup-key + offline copy). "
        "Verify ciphertext SHA-256 against manifest.json before decrypting.\n\n"
        "Restore recipe:\n\n" + render_restore_recipe(manifest)
    )
    try:
        return connector.create_release(
            full_name, tag, name=tag, body=body
        )
    except Exception as exc:  # noqa: BLE001
        if _status_of(exc) == 401:
            raise TokenExpiredError("token rejected creating release") from exc
        if _status_of(exc) == 403:
            raise TokenScopeError(
                "release creation denied (403): token needs Contents "
                "read/write on the backup repo.",
                required="Contents: read/write",
            ) from exc
        raise


def upload_release_asset(
    connector: Any, release: dict[str, Any], path: str | Path, *,
    label: str = "",
) -> dict[str, Any]:
    """Upload one file as a Release asset. Size-verified, never partial.

    Uses the release ``upload_url`` template against uploads.github.com.
    A size mismatch after upload is :class:`ChecksumMismatchError`.
    """
    path = Path(path)
    if hasattr(connector, "upload_release_asset"):
        return connector.upload_release_asset(release, path, label=label)
    upload_url = str(release.get("upload_url", ""))
    if not upload_url:
        raise BackupError("release has no upload_url — cannot upload assets")
    base = upload_url.split("{")[0]
    url = f"{base}?name={path.name}"
    headers = dict(_headers(connector))
    headers["Content-Type"] = "application/octet-stream"
    data = path.read_bytes()
    try:
        resp = connector.http.request(
            "POST", url, data=data, headers=headers, timeout=600.0
        )
    except Exception as exc:  # noqa: BLE001 - network layer is opaque
        raise BackupError(f"asset upload failed for {path.name}: {exc}") from exc
    if resp.status == 401:
        raise TokenExpiredError("token rejected uploading release asset")
    if resp.status == 429:
        retry = resp.headers.get("retry-after", "60")
        try:
            wait = float(retry)
        except (TypeError, ValueError):
            wait = 60.0
        raise RateLimitedBackupError(
            "github rate limit hit uploading asset — back off",
            retry_after=wait,
        )
    if resp.status == 403 and "rate limit" in resp.text.lower():
        raise RateLimitedBackupError(
            "github rate limit hit uploading asset",
            retry_after=60.0,
        )
    if not 200 <= resp.status < 300:
        raise BackupError(
            f"asset upload failed for {path.name} ({resp.status}): "
            f"{resp.text[:300]}"
        )
    try:
        asset = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise BackupError(f"asset upload returned bad JSON: {exc}") from exc
    if int(asset.get("size", -1)) != path.stat().st_size:
        raise ChecksumMismatchError(
            f"uploaded asset {path.name} size {asset.get('size')} != local "
            f"{path.stat().st_size} — FAILED, not partial; re-upload"
        )
    _log.info("uploaded release asset %s (%d bytes)", path.name,
              path.stat().st_size)
    return asset


def list_releases(connector: Any, full_name: str) -> list[dict[str, Any]]:
    try:
        return list(connector._api(  # noqa: SLF001
            "GET", f"/repos/{full_name}/releases",
            params={"per_page": 100},
        ))
    except Exception as exc:  # noqa: BLE001
        if _status_of(exc) == 401:
            raise TokenExpiredError("token rejected listing releases") from exc
        raise


def delete_release(
    connector: Any, full_name: str, release_id: int
) -> None:
    connector._api("DELETE",  # noqa: SLF001
                   f"/repos/{full_name}/releases/{release_id}")


def download_release_asset(
    connector: Any, asset: dict[str, Any], dest: Path
) -> Path:
    """Download one Release asset to ``dest`` (streamed, octet-stream)."""
    url = str(asset.get("url", ""))
    if not url:
        raise BackupError("asset has no download url")
    headers = dict(_headers(connector))
    headers["Accept"] = "application/octet-stream"
    resp = connector.http.request("GET", url, headers=headers, timeout=600.0)
    if resp.status == 401:
        raise TokenExpiredError("token rejected downloading asset")
    if not 200 <= resp.status < 300:
        raise BackupError(
            f"asset download failed ({resp.status}): {resp.text[:300]}"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    # HttpResponse carries the body; write it atomically via tmp + rename.
    body = getattr(resp, "body", None)
    if body is None:
        body = getattr(resp, "content", b"")
        if isinstance(body, str):
            body = body.encode("utf-8")
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(bytes(body))
    tmp.rename(dest)
    return dest


class Outbox:
    """Durable local queue: network down -> snapshot waits here, retried.

    A queued snapshot is NEVER a silent skip: the queue file on disk is
    the evidence, and :meth:`pending` is what the next run drains first.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._file = self.directory / "outbox.json"

    def _load(self) -> list[dict[str, Any]]:
        if not self._file.is_file():
            return []
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            return list(data) if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, items: list[dict[str, Any]]) -> None:
        tmp = self._file.with_name(self._file.name + ".tmp")
        tmp.write_text(json.dumps(items, indent=2), encoding="utf-8")
        tmp.rename(self._file)

    def enqueue(self, entry: dict[str, Any]) -> dict[str, Any]:
        items = self._load()
        entry = {**entry, "queued_at": time.time(), "attempts": 0,
                 "next_retry_at": time.time() + BACKOFF_SECONDS[0]}
        items.append(entry)
        self._save(items)
        _log.warning("backup queued in outbox: %s (retry in %ss)",
                     entry.get("backup_id"), BACKOFF_SECONDS[0])
        return entry

    def pending(self, *, now: float | None = None) -> list[dict[str, Any]]:
        now = time.time() if now is None else now
        return [i for i in self._load() if i.get("next_retry_at", 0) <= now]

    def mark_sent(self, backup_id: str) -> None:
        self._save([i for i in self._load()
                    if i.get("backup_id") != backup_id])

    def mark_failed(self, backup_id: str) -> dict[str, Any] | None:
        items = self._load()
        for item in items:
            if item.get("backup_id") == backup_id:
                item["attempts"] = int(item.get("attempts", 0)) + 1
                delay = BACKOFF_SECONDS[min(
                    item["attempts"] - 1, len(BACKOFF_SECONDS) - 1
                )]
                item["next_retry_at"] = time.time() + delay
                item["last_error_at"] = time.time()
                self._save(items)
                return item
        return None

    def depth(self) -> int:
        return len(self._load())


def _is_network_failure(exc: Exception) -> bool:
    name = type(exc).__name__
    text = str(exc).lower()
    return (
        "network" in name.lower()
        or "connection" in text
        or "timed out" in text
        or "timeout" in text
        or "name resolution" in text
        or "dns" in text
        or "unreachable" in text
    )


def ship_snapshot(
    connector: Any,
    full_name: str,
    manifest: dict[str, Any],
    blob_files: list[str | Path],
    *,
    outbox_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create the Release and upload every blob part for one generation.

    On network failure the generation is queued in the outbox and the
    result says ``queued: True`` — never a silent skip. On any other
    failure the exception propagates (typed) and nothing is marked sent.
    """
    tag = release_tag_for(str(manifest.get("backup_id", "")))
    try:
        release = create_release_for_backup(connector, full_name, tag, manifest)
    except Exception as exc:  # noqa: BLE001 - queue only network failures
        if _is_network_failure(exc) and outbox_dir is not None:
            entry = Outbox(outbox_dir).enqueue({
                "backup_id": manifest.get("backup_id"),
                "manifest": manifest,
                "blob_files": [str(p) for p in blob_files],
                "error": str(exc),
            })
            return {"shipped": False, "queued": True, "release": None,
                    "assets": [], "outbox_entry": entry,
                    "reason": f"network down — queued for retry: {exc}"}
        # 422 "already_exists": a previous attempt created the release but
        # died before uploading — reuse it, never duplicate the tag.
        if _status_of(exc) == 422 or "already_exists" in str(exc):
            try:
                release = connector._api(  # noqa: SLF001
                    "GET", f"/repos/{full_name}/releases/tags/{tag}"
                )
            except Exception:
                raise exc from None
        else:
            raise
    assets: list[dict[str, Any]] = []
    for blob in blob_files:
        assets.append(upload_release_asset(connector, release, blob))
    return {"shipped": True, "queued": False, "release": release,
            "assets": assets,
            "release_id": release.get("id"), "tag": tag}


def ship_manifest_to_repo(
    connector: Any,
    full_name: str,
    manifest: dict[str, Any],
    *,
    work_dir: str | Path,
) -> dict[str, Any]:
    """Push ONLY the manifest (+ restore recipe + .gitignore) into the repo.

    Uses the connector's :meth:`push_code` (GIT_ASKPASS, token never on
    argv). Blobs never enter this path — a manifest-only commit keeps
    the repo at KB scale and binary snapshots out of history.
    """
    import subprocess
    import tempfile

    from .manifest import (
        BACKUP_REPO_GITIGNORE,
        BACKUP_REPO_README,
        MANIFEST_NAME,
        render_restore_recipe,
        write_manifest,
    )

    tmp = Path(tempfile.mkdtemp(prefix="nm-backup-manifest-"))
    try:
        backup_id = str(manifest.get("backup_id", "unknown"))
        # Clone the existing repo (shallow) so the push fast-forwards;
        # a fresh remote (no branch yet) falls back to a local init.
        # The token reaches git through the connector's GIT_ASKPASS
        # plumbing — never on argv, never in config.
        repo_dir = tmp / "repo"
        cloned = False
        try:
            with connector._askpass_env(_token(connector)) as env:  # noqa: SLF001
                proc = subprocess.run(
                    ["git", "clone", "--depth", "1", "--branch", "main",
                     f"https://github.com/{full_name}.git", str(repo_dir)],
                    capture_output=True, text=True, timeout=120, env=env,
                )
            cloned = proc.returncode == 0
        except Exception:  # noqa: BLE001 - fall through to fresh init
            cloned = False
        if not cloned:
            repo_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=str(repo_dir),
                           capture_output=True, check=True, timeout=60)
        target = repo_dir / "manifests" / backup_id
        write_manifest(target, manifest)
        (target / "RESTORE.md").write_text(
            render_restore_recipe(manifest), encoding="utf-8"
        )
        (repo_dir / ".gitignore").write_text(BACKUP_REPO_GITIGNORE,
                                              encoding="utf-8")
        (repo_dir / "README.md").write_text(BACKUP_REPO_README,
                                            encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=str(repo_dir),
                       capture_output=True, check=True, timeout=60)
        commit = subprocess.run(
            ["git", "-c", "user.name=NoMorals Backup",
             "-c", "user.email=backup@localhost",
             "commit", "-m", f"backup manifest {backup_id}"],
            cwd=str(repo_dir), capture_output=True, text=True, timeout=60,
        )
        if commit.returncode != 0 and "nothing to commit" in (
            commit.stdout + commit.stderr
        ):
            return {"pushed": False, "reason": "nothing changed",
                    "backup": backup_id}
        if commit.returncode != 0:
            raise BackupError(
                f"manifest commit failed: "
                f"{(commit.stdout + commit.stderr)[-500:]}"
            )
        return connector.push_code(repo_dir, full_name, branch="main")
    except subprocess.CalledProcessError as exc:
        raise BackupError(f"manifest commit failed: {exc}") from exc
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def ship_to_blob_store(store: Any, files: list[str | Path]) -> dict[str, Any]:
    """Optional 4th copy: S3-compatible store (B2/R2/Wasabi via s3blob.py).

    ``store`` is anything with the blob API —
    :func:`nomorals.storage.s3blob.open_blob_store`. Uploads are
    content-addressed; the store's SHA-256 is checked against the local
    file's, exactly like :meth:`BackupManager.push_to_s3`.
    """
    import hashlib

    objects: list[dict[str, Any]] = []
    for path in files:
        path = Path(path)
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        info = store.put_file(path, mime="application/octet-stream")
        if info.sha256 != h.hexdigest():
            raise ChecksumMismatchError(
                f"blob-store upload corrupted {path.name}: store sha256 "
                f"{info.sha256} != local {h.hexdigest()}"
            )
        objects.append({"file": path.name, "sha256": info.sha256,
                        "size": info.size})
    return {"shipped": True, "objects": objects}
