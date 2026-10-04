"""Single-account trial flow: research a platform, save ONE account's
credentials, and deliver them to the owner on whatever channels are live.

This is the *single-account, owner-driven* shape the owner asked for —
"create a single account to try it out, save the credentials, and send
it to me on WhatsApp if linked or Telegram, or both if both are active".

Two paths:
- ``start``: research what a signup needs (no automation, just intel).
- ``assist``: browser-assisted signup via AccountCreator — one account
  per service, the owner's own identity (from the identity bank),
  human-in-the-loop checkpoints for CAPTCHA/verification. The owner
  stays in control; the bot drives the form-filling.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable

from ...core.errors import ToolError
from ...core.ids import new_short_id
from ...core.logging_setup import get_logger
from .vault import TrialVault

_log = get_logger(__name__)

__all__ = ["TrialFlow", "active_delivery_platforms", "new_id"]

#: delivery order — the owner said "WhatsApp if linked or Telegram, or both".
_DELIVERY_ORDER = ("whatsapp", "telegram")

#: max concurrent background assist runs — browser signups are heavy.
_ASSIST_MAX_INFLIGHT = 2

#: kv_store key where the latest grabbed temp number is stashed so
#: ``/trial sms code`` can poll it without the owner pasting JSON around.
TEMP_SMS_KV_KEY = "trial.temp_sms"


def active_delivery_platforms(gateway: Any) -> list[str]:
    """Platforms that are running in this session, in delivery order.

    ``local`` is excluded (the owner is already there); unknown keys are
    skipped. Returns e.g. ``["whatsapp", "telegram"]``.
    """
    if gateway is None:
        return []
    try:
        status = gateway.status()
    except Exception:  # noqa: BLE001
        return []
    out = []
    for name in _DELIVERY_ORDER:
        info = status.get(name)
        if isinstance(info, dict) and info.get("running_in_session"):
            out.append(name)
    return out


class TrialFlow:
    """Owns the vault and the delivery of one trial account's credentials."""

    def __init__(self, context: Any, *, sender: Callable[[str, str, str], Any] | None = None,
                 gateway: Any = None) -> None:
        self.context = context
        self.settings = getattr(context, "settings", None)
        self.db = getattr(context, "db", None)
        home = getattr(self.settings, "home", "~/.nomorals") if self.settings else "~/.nomorals"
        self.vault = TrialVault(home)
        # sender(platform, chat_key, text) -> SendResult-like; injected for tests.
        self._sender = sender
        #: live chat gateway for background-run reports; resolved from the
        #: context when not passed explicitly.
        self.gateway = gateway
        # bounded background assist runs (browser signups are heavy)
        self._assist_sem = threading.Semaphore(_ASSIST_MAX_INFLIGHT)
        self._assist_lock = threading.Lock()
        #: run_id -> {platform, started, state, note}
        self._assist_runs: dict[str, dict[str, Any]] = {}

    # ── research (what a single signup needs) ────────────────────────────────
    def start(self, platform: str) -> str:
        """Research a platform so the owner can complete ONE real signup."""
        platform = (platform or "").strip()
        if not platform:
            raise ToolError("usage: /trial start <platform>")
        engine = self._search_engine()
        lines = [f"trial account plan for: {platform}"]
        lines.append("one account, your details, you finish the signup in the browser.")
        try:
            report = engine.run(f"{platform} create account sign up required details",
                                mode="quick", pages=2)
            summary = (report.get("summary") or "").strip()
            if summary:
                lines.append("")
                lines.append("what they usually ask for:")
                lines.append(summary[:1200])
        except Exception as exc:  # noqa: BLE001 - research is best-effort
            lines.append(f"(research note: {exc})")
        lines.append("")
        lines.append("when you've signed up, store it with:")
        lines.append(f"  /trial save {platform} <login> <password>")
        lines.append("then I'll send it to you on WhatsApp and/or Telegram.")
        lines.append("")
        lines.append("want me to drive the signup instead? use:")
        lines.append(f"  /trial assist {platform}")
        return "\n".join(lines)

    def assist(self, platform: str, *, chat_key: str = "") -> str:
        """Browser-assisted signup via AccountCreator — actually launched.

        One account per service, owner's identity from the identity bank,
        human-in-the-loop checkpoints for CAPTCHA/verification.  The signup
        runs on a bounded background thread (at most
        ``_ASSIST_MAX_INFLIGHT`` at once) and the outcome — finished,
        paused-for-human, or failed — is reported back to the owner's chat.
        Returns immediately with a truthful status.
        """
        platform = (platform or "").strip()
        if not platform:
            raise ToolError("usage: /trial assist <platform>")
        try:
            from ...accounts.creator import AccountCreator  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return f"account automation unavailable: {exc}"
        # Identity bank check — the owner must have set their details first.
        identity = self._owner_identity()
        if not identity.get("name") or not identity.get("email"):
            return (
                "set your identity first so I can fill forms:\n"
                "  /identity set name <your name>\n"
                "  /identity set email <your email>\n"
                f"then: /trial assist {platform}"
            )
        if not os.environ.get("NM_VAULT_PASSPHRASE", ""):
            return (
                "vault is locked: set the NM_VAULT_PASSPHRASE environment "
                "variable so I can store the new credentials, then ask me again."
            )
        if not self._assist_sem.acquire(blocking=False):
            return (
                f"already driving {_ASSIST_MAX_INFLIGHT} assisted signups — "
                "wait for one to finish, then try again. "
                "check status: /trial status"
            )
        run_id = new_id()
        with self._assist_lock:
            self._assist_runs[run_id] = {
                "platform": platform.lower(),
                "started": time.time(),
                "state": "starting",
                "note": "",
            }
        thread = threading.Thread(
            target=self._assist_run,
            args=(run_id, platform, chat_key, dict(identity)),
            name=f"trial-assist-{platform.lower()[:20]}",
            daemon=True,
        )
        thread.start()
        return (
            f"assisted signup for {platform} — started.\n"
            f"identity: {identity.get('name')} <{identity.get('email')}>\n"
            "I'm driving the signup in the background; I'll report back here "
            "when it pauses for you (CAPTCHA/verification) or finishes.\n"
            "check status: /trial status"
        )

    def _assist_run(self, run_id: str, platform: str, chat_key: str,
                    identity: dict[str, str]) -> None:
        """Background body of :meth:`assist` — runs the real creator."""
        with self._assist_lock:
            self._assist_runs[run_id]["state"] = "running"
        note = ""
        ok = False
        try:
            import asyncio

            from ...accounts.creator import (
                AccountCheckpointPending,
                AccountCreator,
                AccountExistsError,
            )
            from ...accounts.vault import CredentialVault
            from ...tools.captcha import creator_solver_adapter

            vault = CredentialVault(
                self.db,
                master_passphrase=os.environ.get("NM_VAULT_PASSPHRASE", ""),
            )
            creator = AccountCreator(
                vault,
                db=self.db,
                captcha_solver=creator_solver_adapter(
                    settings=getattr(self.context, "settings", None)),
            )
            account = asyncio.run(
                creator.create_account(platform, email=identity.get("email")))
            note = (
                f"✅ {account.service} account ready — "
                f"username: {account.username}, email: {account.email}. "
                "Credentials are stored in the vault."
            )
            ok = True
        except AccountExistsError as exc:
            note = f"ℹ️ {exc}"
            ok = True
        except AccountCheckpointPending as pending:
            cp = pending.checkpoint
            note = (
                "⏸️ account creation paused — I need your help:\n\n"
                f"**{cp.title}**\n{cp.instructions}\n\n"
                f"When you're done: `nm account resume --id {cp.id}`"
            )
            ok = True
        except Exception as exc:  # noqa: BLE001 - the report must go out
            _log.exception("trial assist run %s failed", run_id)
            note = f"❌ assisted signup for {platform} failed: {exc}"
        finally:
            self._assist_sem.release()
            try:
                from ...storage.db import release_thread_connection

                release_thread_connection(self.db)
            except Exception:  # noqa: BLE001 - cleanup is best-effort
                pass
        with self._assist_lock:
            self._assist_runs[run_id].update(
                state="done" if ok else "failed", note=note)
        self._notify_owner(f"trial assist — {platform}", note, chat_key)

    def assist_status(self) -> str:
        """Status of background assist runs (``/trial status``)."""
        with self._assist_lock:
            runs = list(self._assist_runs.items())
        if not runs:
            return "no assisted signups yet — /trial assist <platform> to start one."
        lines = ["assisted signups:"]
        for run_id, run in sorted(runs, key=lambda kv: kv[1].get("started", 0)):
            when = time.strftime("%H:%M", time.localtime(run.get("started", 0)))
            lines.append(
                f"  {run.get('platform')} [{run.get('state')}] started {when}"
            )
            note = str(run.get("note") or "")
            if note:
                lines.append(f"    {note[:220]}")
        return "\n".join(lines)

    def _notify_owner(self, title: str, body: str, chat_key: str = "") -> None:
        """Deliver a background-run report to the owner. Never raises."""
        try:
            from ..notifier import Notifier

            gateway = self.gateway
            if gateway is None:
                gateway = getattr(self.context, "gateway", None)
            notifier = Notifier(self.context, gateway=gateway)
            notifier.publish("trial", title, body, force=True)
        except Exception:  # noqa: BLE001 - reporting must not crash the run
            _log.exception("trial owner notification failed")

    # ── temp SMS numbers (verification codes) ────────────────────────────

    def temp_number(self, country: str = "us",
                    provider: str = "simcodes") -> str:
        """Grab a free temp number for SMS verification (``/trial sms``).

        The number is stashed in kv_store so ``/trial sms code`` can poll
        it without the owner pasting JSON around.  Free providers, no API
        key — the number is public, so it is only good for throwaway
        verification codes, never for anything sensitive.
        """
        from ...accounts.temp_sms import grab_number

        info = grab_number(country=(country or "us").strip() or "us",
                           provider=(provider or "simcodes").strip()
                           or "simcodes")
        if info.get("status") != "ok":
            return f"❌ {info.get('notes', 'could not grab a number')}"
        self._stash_temp_number(info)
        return (
            "📱 temp number (free, public inbox — verification codes only):\n"
            f"  number: {info['number']}\n"
            f"  country: {info.get('country_name') or info.get('country')}\n"
            f"  provider: {info['provider']}\n"
            "hand it to the signup form, then:\n"
            "  /trial sms code"
        )

    def temp_sms_code(self, *, sender_hint: str = "",
                      timeout: float = 120) -> str:
        """Poll the stashed temp number for a verification code.

        Blocks up to ``timeout`` seconds polling the public inbox, then
        returns the code or a timeout note.  Blocking by nature — chat
        handlers should prefer :meth:`temp_sms_code_async`.
        """
        info = self._load_temp_number()
        if not info:
            return "no temp number stashed — grab one first: /trial sms [country]"
        from ...accounts.temp_sms import wait_code

        code = wait_code(info, sender_hint=sender_hint, timeout=timeout)
        if code:
            return f"📩 verification code: {code}"
        return (
            f"no code arrived on {info.get('masked') or info.get('number')} "
            f"within {timeout:.0f}s — the site may not have sent one yet. "
            "try again: /trial sms code"
        )

    def temp_sms_code_async(self, chat_key: str = "",
                            timeout: float = 180) -> str:
        """Watch the stashed temp number in the background (``/trial sms code``).

        Returns immediately; the code (or a timeout note) is reported to
        the owner's chat when it lands.  The chat thread never blocks.
        """
        info = self._load_temp_number()
        if not info:
            return "no temp number stashed — grab one first: /trial sms [country]"
        number = info.get("masked") or info.get("number") or "?"

        def _watch() -> None:
            try:
                result = self.temp_sms_code(timeout=timeout)
            finally:
                try:
                    from ...storage.db import release_thread_connection

                    release_thread_connection(self.db)
                except Exception:  # noqa: BLE001 - cleanup is best-effort
                    pass
            self._notify_owner(f"sms code watch — {number}", result,
                               chat_key)

        thread = threading.Thread(target=_watch, name="trial-sms-watch",
                                  daemon=True)
        thread.start()
        return (
            f"👀 watching {number} for a verification code "
            f"(up to {timeout:.0f}s) — I'll ping you here the moment one lands."
        )

    def _stash_temp_number(self, info: dict[str, Any]) -> None:
        if self.db is None:
            return
        try:
            self.db.execute(
                "INSERT OR REPLACE INTO kv_store (key, value, kind, updated_at)"
                " VALUES (?, ?, 'json', ?)",
                (TEMP_SMS_KV_KEY, json.dumps(info), time.time()),
            )
        except Exception:  # noqa: BLE001 - stash is best-effort
            _log.debug("temp number stash failed")

    def _load_temp_number(self) -> dict[str, Any]:
        if self.db is None:
            return {}
        try:
            row = self.db.query_one(
                "SELECT value FROM kv_store WHERE key = ?",
                (TEMP_SMS_KV_KEY,),
            )
            if row and row.get("value"):
                data = json.loads(row["value"])
                return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            pass
        return {}

    # ── disposable email inbox ───────────────────────────────────────────

    def disposable_inbox(self, service: str, limit: int = 10) -> str:
        """Poll the inbox of a stored disposable-email credential.

        ``service`` is the disposable provider name used at creation
        (``email_mailtm``, ``email_1secmail``, ``email_guerrilla``, ...).
        Needs the vault passphrase — the credential lives in the vault.
        """
        service = (service or "").strip().lower()
        if not service:
            raise ToolError("usage: /trial inbox <service>")
        username, msgs, error = self.disposable_inbox_messages(
            service, limit=limit)
        if error:
            return error
        if not msgs:
            return f"inbox for {username} is empty — no messages yet."
        lines = [f"📧 inbox: {username}"]
        for m in msgs[:limit]:
            frm = str(m.get("from", ""))[:60]
            subj = str(m.get("subject", ""))[:80]
            date = str(m.get("date", ""))[:24]
            lines.append(f"  • {subj or '(no subject)'} — {frm} [{date}]")
        return "\n".join(lines)

    def disposable_inbox_messages(
        self, service: str, limit: int = 10
    ) -> tuple[str, list[dict[str, Any]], str]:
        """Raw disposable-inbox poll: (address, messages, error).

        ``error`` is "" on success.  Used by the CLI's ``--json`` path.
        """
        service = (service or "").strip().lower()
        if not service:
            return "", [], "usage: /trial inbox <service>"
        passphrase = os.environ.get("NM_VAULT_PASSPHRASE", "")
        if not passphrase:
            return "", [], (
                "vault is locked: set the NM_VAULT_PASSPHRASE environment "
                "variable so I can read the disposable credential.")
        try:
            from ...accounts.creator import AccountCreator
            from ...accounts.vault import CredentialVault

            vault = CredentialVault(self.db, master_passphrase=passphrase)
            creator = AccountCreator(vault, db=self.db)
            creds = vault.list_all(service=service, active_only=True)
            targets = [c for c in creds
                       if (c.credential_type or "") == "disposable_email"]
            if not targets:
                names = sorted({c.service for c in creds})
                hint = (f" (vault has for {service!r}: {', '.join(names)})"
                        if names else "")
                return "", [], (f"no disposable-email credential stored for "
                                 f"{service!r}{hint}")
            cred = targets[0]
            msgs = creator.check_disposable_inbox(cred, limit=limit)
            return cred.username, list(msgs or []), ""
        except Exception as exc:  # noqa: BLE001
            return "", [], f"inbox check failed: {exc}"

    def _owner_identity(self) -> dict[str, str]:
        """Read the owner's identity bank (set via /identity)."""
        try:
            from ...accounts.creator import AccountCreator
            # AccountCreator persists identity to kv_store; read it directly
            # without needing a full creator instance.
            db = self.db
            if db is None:
                return {}
            row = db.query_one(
                "SELECT value FROM kv_store WHERE key = ?",
                (AccountCreator.IDENTITY_KV_KEY,),
            )
            if not row or not row.get("value"):
                return {}
            import json
            return json.loads(row["value"])
        except Exception:  # noqa: BLE001
            return {}

    # ── save + deliver ───────────────────────────────────────────────────────
    def save(self, platform: str, login: str, secret: str, note: str = "") -> dict:
        if not (platform and login and secret):
            raise ToolError("usage: /trial save <platform> <login> <password>")
        self.vault.store(platform, login, secret, note)
        return {"platform": platform.lower(), "login": login}

    def deliver(self, platform: str, gateway: Any = None, via: str = "") -> str:
        """Send a stored credential pair to the owner on live channels.

        ``via`` is the chat the command came from (e.g. ``telegram:123``) —
        used as a fallback destination per platform.
        """
        entry = self.vault.get(platform)
        if entry is None:
            raise ToolError(f"no stored trial account for {platform!r} (try /trial list)")
        if entry.get("unreadable"):
            raise ToolError(f"{platform}: stored credential is unreadable (key/tamper)")
        secret = entry["secret"]
        body = (
            f"trial account — {entry['platform']}\n"
            f"login: {entry['login']}\n"
            f"password: {secret}"
            + (f"\nnote: {entry['note']}" if entry.get("note") else "")
        )
        targets = self._delivery_targets(gateway, via)
        sent = []
        for plat, key in targets:
            ok = self._send(plat, key, body, gateway)
            if ok:
                sent.append(plat)
        if not sent:
            # No live channel (or dry run): hand the text back so the caller
            # can show it — the credential still exists, just undelivered.
            return "(no live WhatsApp/Telegram in this session — shown here)\n" + body
        return "sent on: " + ", ".join(sent)

    def _delivery_targets(self, gateway: Any, via: str) -> list[tuple[str, str]]:
        """(platform, chat_key) pairs to try, deduped, in delivery order."""
        targets: list[tuple[str, str]] = []
        for plat in active_delivery_platforms(gateway):
            key = self._owner_chat_key(plat) or self._fallback_key(plat, via)
            if key:
                targets.append((plat, key))
        # If the command came from a delivery platform, make sure it's covered
        # even if the gateway report is stale.
        if via and ":" in via:
            plat, _, cid = via.partition(":")
            if plat in _DELIVERY_ORDER and not any(t[0] == plat for t in targets):
                targets.append((plat, f"{plat}:{cid}"))
        return targets

    def _owner_chat_key(self, platform: str) -> str:
        partner = getattr(self.settings, "partner", None) if self.settings else None
        raw = getattr(partner, "owner_chats", "") or ""
        for key in raw.split(","):
            key = key.strip()
            if key.startswith(f"{platform}:"):
                return key
        return ""

    def _fallback_key(self, platform: str, via: str) -> str:
        if via and via.startswith(f"{platform}:"):
            return via
        return ""

    def _send(self, platform: str, chat_key: str, text: str, gateway: Any = None) -> bool:
        if self._sender is not None:
            result = self._sender(platform, chat_key, text)
            return bool(getattr(result, "ok", result))
        if gateway is None:
            gateway = getattr(self.context, "gateway", None)
        if gateway is None:
            return False
        from ...social.chat.base import ChatRef

        plat, _, cid = chat_key.partition(":")
        try:
            result = gateway.send(plat, ChatRef(platform=plat, chat_id=cid), text)
            return bool(getattr(result, "ok", False))
        except Exception:  # noqa: BLE001 - delivery is best-effort
            return False

    # ── listing / removal ────────────────────────────────────────────────────
    def list(self) -> str:
        rows = self.vault.list()
        if not rows:
            return "no stored trial accounts yet."
        lines = ["stored trial accounts:"]
        for row in rows:
            when = time.strftime("%Y-%m-%d", time.localtime(row.get("saved_at", 0)))
            lines.append(f"  {row['platform']}: {row['login']}  (saved {when})")
        return "\n".join(lines)

    def remove(self, platform: str) -> str:
        if self.vault.delete(platform):
            return f"deleted trial account for {platform.lower()}."
        return f"no stored trial account for {platform.lower()}."

    def _search_engine(self) -> Any:
        from ..search.engine import SearchEngine

        return SearchEngine(self.context)


def new_id() -> str:
    return new_short_id(length=12)
