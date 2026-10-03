"""GitHub connector — the reference implementation.

Auth: fine-grained personal access token (classic PATs also work). The token
is created by the owner in the GitHub web UI — the API cannot mint PATs —
so :meth:`connect` guides them there, then validates and vault-stores it.

Capabilities:
* repos: create, list, fetch, branch
* code push from a local checkout (token via GIT_ASKPASS, never argv/config)
* releases, webhooks, deploy keys (provisioning)
* repo backup: ``git clone --mirror`` + ``fsck`` verify + manifest
"""

from __future__ import annotations

import contextlib
import json
import os
import stat
import subprocess
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .auth import prompt_secret
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = ["GitHubConnector", "GitHubError"]

_log = get_logger(__name__)

API_BASE = "https://api.github.com"
API_VERSION = "2022-11-28"
TOKEN_URL = "https://github.com/settings/tokens?type=beta"

#: Suggested fine-grained PAT repository permissions for Devon's usual work
#: (write code, create repos, back them up). Metadata read is automatic.
SUGGESTED_PERMISSIONS = [
    "Contents: read and write",
    "Pull requests: read and write (optional)",
    "Webhooks: read and write (only if Devon manages webhooks)",
]


class GitHubError(ConnectorError):
    """A GitHub API call failed."""

    def __init__(self, message: str, *, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


@register_connector
class GitHubConnector(Connector):
    """Devon's GitHub adapter: repos, code push, releases, backups."""

    id = "github"
    name = "GitHub"
    description = (
        "Create repos, push code, manage releases/webhooks/deploy keys, "
        "and back up repositories. Authenticates with a fine-grained "
        "personal access token."
    )
    auth_methods = (AuthMethod.PAT,)
    PROVISIONABLE = (
        "repo",
        "branch",
        "release",
        "webhook",
        "deploy_key",
        "repo_backup",
        "github_account",
    )

    #: GitHub username rules: 1-39 chars, alphanumerics and hyphens,
    #: no leading/trailing hyphen.
    _USERNAME_RE = None  # compiled lazily to keep import light

    # ── lifecycle ────────────────────────────────────────────────

    def connect(self, *, token: str | None = None) -> ConnectResult:
        """Guide the owner through PAT creation, validate, vault-store."""
        pat = (token or "").strip()
        if not pat and not os.environ.get("GITHUB_TOKEN", "").strip():
            # Interactive path: show the human steps first, then prompt.
            print(self.connect_instructions())
        pat = pat or prompt_secret(
            "GitHub fine-grained PAT", env_var="GITHUB_TOKEN"
        )
        if not pat:
            raise ConnectorError("empty token: nothing to connect with")
        me = self._api("GET", "/user", token=pat)
        login = str(me.get("login", ""))
        if not login:
            raise GitHubError("GitHub did not return a login for this token")
        scopes = self._scopes_from_headers(me)
        cred = self._store_credential(
            login,
            pat,
            credential_type="pat",
            scopes=scopes,
            metadata={
                "user_id": me.get("id"),
                "auth": "fine-grained-pat",
            },
        )
        _log.info("github connected as %s", login)
        return ConnectResult(
            ok=True,
            account=login,
            scopes=scopes,
            message=(
                f"connected to GitHub as {login}. The token is in the "
                "encrypted vault; revoke it any time at "
                "https://github.com/settings/tokens."
            ),
            credential_id=cred.id,
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect --name github`",
            )
        try:
            me = self._api("GET", "/user", token=cred.password)
        except GitHubError as exc:
            return ConnectorStatus(
                connected=False,
                account=cred.username,
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"token rejected ({exc}): reconnect with a fresh token",
            )
        return ConnectorStatus(
            connected=True,
            account=str(me.get("login", cred.username)),
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail="token valid",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api("GET", "/user", token=cred.password)
            return True
        except ConnectorError:
            return False

    # ── provisioning ───────────────────────────────────────────

    def provision(self, kind: str, **kwargs: Any) -> dict[str, Any]:
        db = kwargs.pop("db", None)
        context = kwargs.pop("context", None)
        if kind == "github_account":
            return self._provision_github_account(
                db=db, context=context, **kwargs
            )
        handlers = {
            "repo": self.create_repo,
            "branch": self.create_branch,
            "release": self.create_release,
            "webhook": self.create_webhook,
            "deploy_key": self.add_deploy_key,
            "repo_backup": self.backup_repo,
        }
        try:
            handler = handlers[kind]
        except KeyError:
            raise ConnectorError(
                f"github cannot provision {kind!r} "
                f"(provisionable: {', '.join(self.PROVISIONABLE)})"
            ) from None
        return handler(**kwargs)

    # ── guided account creation (human-in-the-loop) ──────────────

    @staticmethod
    def _valid_username(username: str) -> bool:
        import re

        return bool(
            re.fullmatch(r"[a-zA-Z0-9]([a-zA-Z0-9-]{0,37}[a-zA-Z0-9])?",
                         username or "")
        )

    def _public_api(self, method: str, path: str) -> Any:
        """Unauthenticated API call (public endpoints, e.g. username check)."""
        url = f"{API_BASE}{path}"
        try:
            resp = self.http.get(url, headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": API_VERSION,
            })
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise GitHubError(f"github request failed: {exc}") from exc
        if resp.status == 404:
            raise GitHubError("not found", status_code=404)
        if not resp.ok:
            raise GitHubError(
                f"github {method} {path} failed ({resp.status})",
                status_code=resp.status,
            )
        return resp.json()

    def _username_available(self, username: str) -> bool:
        try:
            self._public_api("GET", f"/users/{username}")
            return False
        except GitHubError as exc:
            if exc.status_code == 404:
                return True
            raise

    def _provision_github_account(
        self,
        username: str = "",
        *,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Guide the owner through creating their own GitHub account.

        Devon prepares everything it can — validates the name, checks
        availability — then pauses at the human-only steps (signup form +
        CAPTCHA, email verification). The owner solves those personally;
        the checks are thereby satisfied, not bypassed. One account per
        service: refuses while a credential is stored.
        """
        from .checkpoints import CheckpointKind

        if db is None:
            raise ConnectorError(
                "github_account provisioning needs a database for "
                "checkpoints (pass db=)"
            )
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                f"github is already connected as {existing.username} — "
                "one account per service. Disconnect first to switch."
            )
        username = (username or "").strip()
        if not self._valid_username(username):
            raise ConnectorError(
                f"{username!r} is not a valid GitHub username "
                "(1-39 chars, letters/digits/hyphens, no leading/trailing "
                "hyphen)"
            )
        if not self._username_available(username):
            raise ConnectorError(
                f"github username {username!r} is taken — pick another"
            )
        cp = self.request_human(
            CheckpointKind.MANUAL_STEP,
            f"Create your GitHub account @{username}",
            "\n".join([
                "Devon prepared everything it can — now the human part:",
                "1. Open https://github.com/signup",
                f"2. Choose username: {username}",
                "3. Use YOUR email and a strong password you save.",
                "4. Solve the CAPTCHA yourself (this proves a human is "
                "involved — Devon never touches it).",
                "5. Accept the Terms of Service.",
            ]),
            db=db,
            context=context,
            resume_state={"stage": "signup", "username": username},
        )
        # Interactive: the checkpoint resolved already — keep going.
        return self.resume_checkpoint(cp, db=db, context=context)

    def resume_checkpoint(
        self,
        checkpoint: Any,
        *,
        db: Any,
        context: Any = None,
    ) -> dict[str, Any]:
        """Continue the guided account flow after the owner acts."""
        from .checkpoints import CheckpointKind, CheckpointState

        if checkpoint.state != CheckpointState.RESOLVED:
            raise ConnectorError(
                f"checkpoint {checkpoint.id} is {checkpoint.state.value}, "
                "not resolved — the owner must complete the human step first"
            )
        stage = (checkpoint.resume_state or {}).get("stage", "")
        username = (checkpoint.resume_state or {}).get("username", "")
        if stage == "signup":
            if not username or self._username_available(username):
                raise ConnectorError(
                    f"@{username or '?'} does not exist on GitHub yet — "
                    "finish the signup first, then resolve the checkpoint "
                    "again"
                )
            cp = self.request_human(
                CheckpointKind.EMAIL_VERIFY,
                f"Verify the email for @{username}",
                "\n".join([
                    "GitHub sent a verification email to the address you "
                    "signed up with.",
                    "1. Open it and click the verification link yourself.",
                    "2. Come back here when done.",
                ]),
                db=db,
                context=context,
                resume_state={"stage": "verify_email", "username": username},
            )
            return self.resume_checkpoint(cp, db=db, context=context)
        if stage == "verify_email":
            # The owner's resolve IS the attestation: the click happened.
            return {
                "account": username,
                "done": True,
                "identity": "owner",
                "message": (
                    f"GitHub account @{username} is ready. Next step: "
                    "`nm connectors connect --name github` and paste a "
                    "fine-grained PAT — Devon will validate it, vault-store "
                    "it, and can then create repos, push code, and back "
                    "them up."
                ),
            }
        raise ConnectorError(
            f"github cannot resume checkpoint stage {stage!r}"
        )

    # ── repos ──────────────────────────────────────────────────

    def create_repo(
        self,
        name: str,
        *,
        private: bool = True,
        description: str = "",
        auto_init: bool = False,
    ) -> dict[str, Any]:
        """Create a repository under the connected account."""
        repo = self._api(
            "POST",
            "/user/repos",
            {
                "name": name,
                "private": private,
                "description": description,
                "auto_init": auto_init,
            },
        )
        _log.info("github repo created: %s", repo.get("full_name"))
        return repo

    def list_repos(self, *, limit: int = 30) -> list[dict[str, Any]]:
        """Repositories the token can see (newest first)."""
        repos = self._api(
            "GET",
            "/user/repos",
            params={"per_page": max(1, min(limit, 100)), "sort": "created",
                    "direction": "desc"},
        )
        return repos if isinstance(repos, list) else []

    def get_repo(self, full_name: str) -> dict[str, Any]:
        """One repository by ``owner/name``."""
        return self._api("GET", f"/repos/{full_name}")

    def create_branch(
        self, full_name: str, branch: str, *, from_branch: str = "main"
    ) -> dict[str, Any]:
        """Create ``branch`` from ``from_branch``'s HEAD."""
        base = self._api(
            "GET", f"/repos/{full_name}/git/ref/heads/{from_branch}"
        )
        sha = ((base.get("object") or {}).get("sha") or "")
        if not sha:
            raise GitHubError(
                f"could not resolve {from_branch} in {full_name}"
            )
        return self._api(
            "POST",
            f"/repos/{full_name}/git/refs",
            {"ref": f"refs/heads/{branch}", "sha": sha},
        )

    # ── releases / webhooks / deploy keys ──────────────────────

    def create_release(
        self,
        full_name: str,
        tag: str,
        *,
        name: str = "",
        body: str = "",
        draft: bool = False,
        prerelease: bool = False,
    ) -> dict[str, Any]:
        """Create a release for ``tag`` on ``owner/name``."""
        return self._api(
            "POST",
            f"/repos/{full_name}/releases",
            {
                "tag_name": tag,
                "name": name or tag,
                "body": body,
                "draft": draft,
                "prerelease": prerelease,
            },
        )

    def create_webhook(
        self,
        full_name: str,
        url: str,
        *,
        events: list[str] | None = None,
        secret: str | None = None,
        active: bool = True,
    ) -> dict[str, Any]:
        """Register a webhook on ``owner/name``."""
        config: dict[str, Any] = {"url": url, "content_type": "json"}
        if secret:
            # The secret itself is never returned by the API; the caller
            # hands it to the owner and vault-stores it.
            config["secret"] = secret
        return self._api(
            "POST",
            f"/repos/{full_name}/hooks",
            {
                "name": "web",
                "active": active,
                "events": events or ["push"],
                "config": config,
            },
        )

    def add_deploy_key(
        self,
        full_name: str,
        title: str,
        key: str,
        *,
        read_only: bool = True,
    ) -> dict[str, Any]:
        """Add a deploy key to ``owner/name``."""
        return self._api(
            "POST",
            f"/repos/{full_name}/keys",
            {"title": title, "key": key, "read_only": read_only},
        )

    # ── code push ──────────────────────────────────────────────

    def push_code(
        self,
        local_dir: str | Path,
        full_name: str,
        *,
        branch: str = "main",
        remote_name: str = "origin",
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Push a local checkout to ``owner/name`` using the stored PAT.

        The token reaches git through a one-shot GIT_ASKPASS script fed from
        an environment variable — never on the command line, never written
        to git config or disk.

        FAIL-FAST: a failed push raises :class:`GitHubError` with git's
        output — it never returns a soft ``{"pushed": False}`` that a
        caller could silently ignore.
        """
        cred = self._require_credential()
        repo = Path(local_dir)
        self._git(repo, "rev-parse", "--git-dir", timeout=30.0)
        try:
            self._git(repo, "remote", "get-url", remote_name, timeout=30.0)
        except GitHubError:
            self._git(
                repo, "remote", "add", remote_name,
                f"https://github.com/{full_name}.git", timeout=30.0,
            )
        with self._askpass_env(cred.password) as env:
            proc = self._git_raw(
                repo, "push", remote_name, branch,
                timeout=timeout, env=env,
            )
        out = (proc.stdout + proc.stderr)[-2000:]
        if proc.returncode != 0:
            raise GitHubError(
                f"git push {remote_name} {branch} to {full_name} failed: "
                f"{out.strip() or '(no output)'}"
            )
        _log.info("github push %s %s: ok", full_name, branch)
        return {"pushed": True, "branch": branch, "output": out}

    # ── repo backup ────────────────────────────────────────────

    def backup_repo(
        self,
        repo_ref: str,
        dest_dir: str | Path,
        *,
        timeout: float = 900.0,
    ) -> dict[str, Any]:
        """Mirror-clone a repo to ``dest_dir`` and verify the mirror.

        ``repo_ref`` is ``owner/name`` or any git URL (``file://`` URLs work
        offline, which is how the tests exercise this). Produces
        ``<owner>__<repo>.git`` (a bare mirror: every branch, tag, and ref)
        plus a ``manifest.json``. Returns the manifest fields.
        """
        cred = self._load_credential()
        token = cred.password if cred else None
        url = (
            repo_ref
            if "://" in repo_ref
            else f"https://github.com/{repo_ref}.git"
        )
        label = repo_ref.replace("://", "_").replace("/", "__")
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / f"{label}.git"
        if target.exists():
            with self._askpass_env(token) as env:
                self._git(
                    dest, "--git-dir", str(target), "remote", "update",
                    "--prune", timeout=timeout, env=env,
                )
        else:
            with self._askpass_env(token) as env:
                proc = self._git_raw(
                    dest, "clone", "--mirror", url, str(target),
                    timeout=timeout, env=env,
                )
            if proc.returncode != 0:
                raise GitHubError(
                    f"backup clone failed for {repo_ref}: "
                    f"{(proc.stderr or proc.stdout)[-500:]}"
                )
        fsck = self._git_raw(
            dest, "--git-dir", str(target), "fsck", "--full", timeout=timeout
        )
        verified = fsck.returncode == 0
        head = self._git(
            dest, "--git-dir", str(target), "rev-parse", "HEAD",
            timeout=60.0,
        ).stdout.strip()
        manifest = {
            "repo": repo_ref,
            "path": str(target),
            "cloned_at": time.time(),
            "head": head,
            "verified": verified,
        }
        (dest / f"{label}.manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        if not verified:
            raise GitHubError(
                f"backup of {repo_ref} failed integrity check: "
                f"{(fsck.stderr or fsck.stdout)[-500:]}"
            )
        _log.info("github backup ok: %s -> %s", repo_ref, target)
        return manifest

    # ── HTTP plumbing ──────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "github is not connected — run "
                "`nm connectors connect --name github` first"
            )
        return cred

    def _headers(self, token: str) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "Authorization": f"Bearer {token}",
        }

    def _api(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> Any:
        """One GitHub API call; errors become GitHubError with status."""
        pat = token or self._require_credential().password
        url = f"{API_BASE}{path}"
        try:
            if method == "GET":
                resp = self.http.get(
                    url, headers=self._headers(pat), params=params
                )
            elif method == "POST":
                resp = self.http.post_json(
                    url, payload or {}, headers=self._headers(pat)
                )
            elif method == "DELETE":
                resp = self.http.request(
                    "DELETE", url, headers=self._headers(pat)
                )
            else:
                raise ConnectorError(f"unsupported method {method}")
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise GitHubError(f"github request failed: {exc}") from exc
        if resp.status == 401:
            raise GitHubError(
                "github rejected the token (401): it is invalid, expired, "
                "or revoked — reconnect with a fresh token",
                status_code=401,
            )
        if resp.status == 403 and "rate limit" in resp.text.lower():
            reset = resp.headers.get("x-ratelimit-reset", "")
            raise GitHubError(
                "github rate limit exceeded"
                + (f" (resets at unix ts {reset})" if reset else "")
                + " — wait or use a token with more quota",
                status_code=403,
            )
        if not resp.ok:
            try:
                detail = resp.json().get("message", "")
            except Exception:  # noqa: BLE001 - fall back to raw text
                detail = resp.text[:200]
            raise GitHubError(
                f"github {method} {path} failed ({resp.status}): {detail}",
                status_code=resp.status,
            )
        # stash the response for scope extraction on connect()
        self._last_response = resp
        return resp.json()

    def _scopes_from_headers(self, _me: dict[str, Any]) -> list[str]:
        resp = getattr(self, "_last_response", None)
        raw = ""
        if resp is not None:
            raw = resp.headers.get("x-oauth-scopes", "")
        scopes = [s.strip() for s in raw.split(",") if s.strip()]
        if scopes:
            return scopes
        # Fine-grained PATs don't report classic scopes; record that honestly.
        return ["fine-grained (see token settings)"]

    # ── git plumbing ───────────────────────────────────────────

    def _git(
        self,
        cwd: Path,
        *args: str,
        timeout: float,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        proc = self._git_raw(cwd, *args, timeout=timeout, env=env)
        if proc.returncode != 0:
            raise GitHubError(
                f"git {' '.join(args)} failed: "
                f"{(proc.stderr or proc.stdout)[-500:]}"
            )
        return proc

    @staticmethod
    def _git_raw(
        cwd: Path,
        *args: str,
        timeout: float,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise GitHubError(f"git {' '.join(args)} timed out") from exc
        except OSError as exc:
            raise GitHubError(f"git not available: {exc}") from exc

    @staticmethod
    @contextlib.contextmanager
    def _askpass_env(token: str | None) -> Iterator[dict[str, str] | None]:
        """GIT_ASKPASS plumbing that never puts the token on the command line.

        A one-shot script echoes the token from an environment variable; the
        script is removed afterwards. ``None`` token yields ``None`` env
        (public/file URLs need no auth).
        """
        if not token:
            yield None
            return
        fd, path = tempfile.mkstemp(prefix="nm-gh-askpass-", suffix=".sh")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write('#!/bin/sh\necho "$NM_GH_ASKPASS_TOKEN"\n')
            os.chmod(path, stat.S_IRWXU)
            env = {
                **os.environ,
                "GIT_ASKPASS": path,
                "NM_GH_ASKPASS_TOKEN": token,
                "GIT_TERMINAL_PROMPT": "0",
            }
            yield env
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)

    def connect_instructions(self) -> str:
        """The human steps for the one part the API cannot do: mint the PAT."""
        lines = [
            "1. Open " + TOKEN_URL,
            "2. 'Generate new token' -> fine-grained, give it a name + expiry.",
            "3. Repository access: 'All repositories' or select ones.",
            "4. Permissions (suggested):",
        ]
        lines.extend(f"   - {p}" for p in SUGGESTED_PERMISSIONS)
        lines.append("5. Generate, copy the token, and paste it below")
        lines.append("   (or set the GITHUB_TOKEN environment variable).")
        return "\n".join(lines)
