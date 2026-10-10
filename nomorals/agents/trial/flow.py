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
from ...storage.kv import KVStore
from .vault import TrialVault

_log = get_logger(__name__)

__all__ = ["TrialFlow", "active_delivery_platforms", "new_id"]

#: delivery order — the owner said "WhatsApp if linked or Telegram, or both".
_DELIVERY_ORDER = ("whatsapp", "telegram")

#: max concurrent background assist runs — browser signups are heavy.
#: Profile-gated: use profile_value("assist_max_inflight") at runtime.
#: Kept as a fallback default for import-time references.
_ASSIST_MAX_INFLIGHT = 2

def _assist_max_inflight() -> int:
    from ...core.profiles import profile_value
    return int(profile_value("assist_max_inflight", _ASSIST_MAX_INFLIGHT))

#: terminal assist-run states.  Any row in ``trial_assist_runs`` whose
#: state is *not* in this set was in flight when the process died and
#: gets marked ``interrupted`` + reported on the next boot.
_ASSIST_TERMINAL_STATES = frozenset({"done", "failed", "interrupted"})

#: durable assist-run state — the table is created here (IF NOT EXISTS)
#: and again in the migrations so fresh and upgraded DBs both have it.
_ASSIST_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS trial_assist_runs (
    run_id     TEXT PRIMARY KEY,
    platform   TEXT NOT NULL DEFAULT '',
    chat_key   TEXT NOT NULL DEFAULT '',
    started    REAL NOT NULL DEFAULT 0,
    state      TEXT NOT NULL DEFAULT '',
    note       TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_trial_assist_runs_state
    ON trial_assist_runs(state);
"""

#: terminal runs older than this are pruned on load — status history,
#: not an archive.
_ASSIST_RUN_TTL = 7 * 24 * 3600

#: terminal sms-watch states.  Any row in ``trial_sms_watches`` whose
#: state is *not* in this set was watching when the process died and
#: gets resumed (deadline still ahead) or closed as timed-out on the
#: next boot — never silently dropped.
_SMS_WATCH_TERMINAL_STATES = frozenset({"done", "timeout", "interrupted"})

#: durable sms-watch state — the table is created here (IF NOT EXISTS)
#: and again in the migrations so fresh and upgraded DBs both have it.
_SMS_WATCHES_DDL = """
CREATE TABLE IF NOT EXISTS trial_sms_watches (
    watch_id    TEXT PRIMARY KEY,
    number      TEXT NOT NULL DEFAULT '',
    number_info TEXT NOT NULL DEFAULT '',
    chat_key    TEXT NOT NULL DEFAULT '',
    started     REAL NOT NULL DEFAULT 0,
    deadline    REAL NOT NULL DEFAULT 0,
    timeout     REAL NOT NULL DEFAULT 0,
    state       TEXT NOT NULL DEFAULT '',
    code        TEXT NOT NULL DEFAULT '',
    note        TEXT NOT NULL DEFAULT '',
    updated_at  REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_trial_sms_watches_state
    ON trial_sms_watches(state);
"""

#: terminal watches older than this are pruned on load — watch history,
#: not an archive.
_SMS_WATCH_TTL = 7 * 24 * 3600

#: kv_store key where the latest grabbed temp number is stashed so
#: ``/trial sms code`` can poll it without the owner pasting JSON around.
TEMP_SMS_KV_KEY = "trial.temp_sms"

#: process-local confirmation gate for disposable-persona signups (see
#: TrialFlow._confirmation_gate).  A restart drops pending
#: confirmations; ``/trial confirm`` then honestly reports the token as
#: unknown instead of misfiring.
_CONFIRM_GATE = None

#: secret-free trial action audit log, under the trial home dir.  Every
#: credential-touching action (assist launch/complete, save, deliver,
#: delete, sms watch start/finish) appends one JSON line: timestamp,
#: action, platform, outcome — never the secret itself.
_TRIAL_AUDIT_LOG = "trial_audit.jsonl"

#: kv_store key prefix where ``start`` stashes its research plan so a
#: later ``assist`` can pick up the discovered signup URL instead of
#: re-researching from zero.
_PLAN_KV_PREFIX = "trial.plan."


def _format_sms_poll_result(info: dict[str, Any], code: str,
                            timeout: float) -> str:
    """One phrasing for a finished SMS poll, blocking or background."""
    number = info.get("masked") or info.get("number") or "?"
    if code:
        return f"📩 verification code: {code}"
    return (
        f"no code arrived on {number} within {timeout:.0f}s — "
        "the site may not have sent one yet. "
        "try again: /trial sms code"
    )


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
        self._assist_sem = threading.Semaphore(_assist_max_inflight())
        self._assist_lock = threading.Lock()
        #: run_id -> {platform, chat_key, started, state, note}
        self._assist_runs: dict[str, dict[str, Any]] = {}
        # durable assist-run state: runs survive a bot restart so the
        # owner can still see what happened and get the report.
        self._ensure_assist_runs_table()
        self._load_assist_runs()
        self._maybe_recover_assist_runs()
        # durable sms-watch state: same treatment — a restart must not
        # silently swallow a background verification-code watch.
        self._sms_lock = threading.Lock()
        #: watch_id -> {number, number_info, chat_key, started,
        #:              deadline, timeout, state, code, note}
        self._sms_watches: dict[str, dict[str, Any]] = {}
        self._ensure_sms_watches_table()
        self._load_sms_watches()
        self._maybe_recover_sms_watches()

    # one recovery pass per process — the first TrialFlow use after boot
    # marks in-flight runs interrupted and reports them; later instances
    # just read the (now terminal) rows.
    _assist_recovery_done: bool = False

    # one sms-watch recovery pass per process — the first TrialFlow use
    # after boot resumes live watches / closes expired ones; later
    # instances just read the rows.
    _sms_recovery_done: bool = False

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
        text = "\n".join(lines)
        # Stash the plan so a later /trial assist can pick up the
        # discovered signup URL instead of re-researching from zero.
        self._stash_plan(platform, text)
        return text

    def _stash_plan(self, platform: str, text: str) -> None:
        """Persist a research plan for the research→execute handoff."""
        if self.db is None:
            return
        try:
            KVStore(self.db).set(
                f"{_PLAN_KV_PREFIX}{platform.strip().lower()}",
                {"platform": platform.strip().lower(),
                 "plan": text, "saved_at": time.time()},
            )
        except Exception:  # noqa: BLE001 - stash is best-effort
            _log.debug("trial plan stash failed for %s", platform)

    def _load_plan(self, platform: str) -> dict[str, Any]:
        if self.db is None:
            return {}
        try:
            data = KVStore(self.db).get(
                f"{_PLAN_KV_PREFIX}{platform.strip().lower()}")
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            pass
        return {}

    @staticmethod
    def _plan_signup_url(plan: dict[str, Any]) -> str:
        """Extract a likely signup URL from a stashed research plan."""
        import re

        text = str(plan.get("plan") or "")
        for match in re.finditer(r"https?://[^\s)\"'<>]+", text):
            url = match.group(0).rstrip(".,;")
            low = url.lower()
            if any(k in low for k in ("signup", "sign-up", "register",
                                      "join", "create-account")):
                return url
        return ""

    def assist(self, platform: str, *, chat_key: str = "", auto_yes: bool = False) -> str:
        """Browser-assisted signup via AccountCreator — actually launched.

        One account per service, owner's identity from the identity bank,
        human-in-the-loop checkpoints for CAPTCHA/verification.  The signup
        runs on a bounded background thread (at most
        ``_ASSIST_MAX_INFLIGHT`` at once) and the outcome — finished,
        paused-for-human, or failed — is reported back to the owner's chat.
        Returns immediately with a truthful status.

        If no identity is set, a disposable persona is drafted from the
        identity bank and shown with a warning — the run proceeds ONLY on
        explicit confirmation (``/trial confirm <token>``) or immediately
        with auto_yes=True (--yes flag, the standing go-ahead).
        """
        platform = (platform or "").strip()
        if not platform:
            raise ToolError("usage: /trial assist <platform> [--yes]")
        try:
            from ...accounts.creator import AccountCreator  # noqa: F401
            from ...accounts.identity_bank import (  # noqa: F401
                IdentityBank, render_persona_card)
        except Exception as exc:  # noqa: BLE001
            return f"account automation unavailable: {exc}"
        # One account per service — refuse before drafting any identity.
        existing = self._existing_account(platform)
        if existing:
            source, login = existing
            self._audit("assist_refused", platform,
                        f"already have account ({source} vault)")
            return (
                f"already have an account for {platform} "
                f"({source} vault, login: {login}) — one account per "
                "service.\n"
                f"  /trial send {platform}   — redeliver the credentials\n"
                f"  /trial rm {platform}     — remove it, then retry assist"
            )
        # Owner's own identity wins when set (services tied to them).
        # Otherwise draft a disposable persona — confirmation-gated.
        identity = self._owner_identity()
        persona_id = ""
        if not identity.get("name") or not identity.get("email"):
            persona, gen = self._generate_disposable_identity(platform)
            persona_id = gen["persona_id"]
            identity = {
                "name": gen["name"],
                "email": gen["email"],
                "disposable": "true",
                "persona_id": persona_id,
            }
            if not auto_yes:
                gate = self._confirmation_gate()
                token = gate.request(
                    subject=f"trial signup: {platform}",
                    card=render_persona_card(persona, platform),
                    payload={"platform": platform,
                             "persona_id": persona_id,
                             "chat_key": chat_key or ""},
                )
                return (
                    f"{render_persona_card(persona, platform)}\n\n"
                    f"to proceed: `/trial confirm {token}`\n"
                    f"or rerun: `/trial assist {platform} --yes`\n"
                    "to use your own details instead:\n"
                    "  /identity set name <your name>\n"
                    "  /identity set email <your email>"
                )
            # auto_yes: the --yes flag IS the explicit confirmation.
        return self._launch_assist(platform, chat_key=chat_key,
                                   identity=identity, persona_id=persona_id)

    def confirm_signup(self, token: str, *, chat_key: str = "") -> str:
        """Redeem a ``/trial confirm <token>`` — the explicit go-ahead for
        a drafted disposable persona.  One-shot: the token dies here."""
        token = (token or "").strip()
        if not token:
            return "usage: /trial confirm <token>"
        gate = self._confirmation_gate()
        # The `/trial confirm <token>` command itself is the owner's
        # explicit yes — record it on the gate, then one-shot redeem.
        if not gate.confirm(token, "yes"):
            return (
                "unknown or expired confirmation token — nothing was "
                "started. Rerun `/trial assist <platform>` for a fresh "
                "identity draft.")
        payload = gate.consume(token)
        if not payload:
            return (
                "unknown or expired confirmation token — nothing was "
                "started. Rerun `/trial assist <platform>` for a fresh "
                "identity draft.")
        platform = payload.get("platform", "")
        # Re-fetch the exact persona the owner confirmed (same name the
        # warning card showed) instead of a "(disposable persona)"
        # placeholder.
        _, gen = self._generate_disposable_identity(
            platform, persona_id=payload.get("persona_id", ""))
        return self._launch_assist(
            platform,
            chat_key=chat_key or payload.get("chat_key", ""),
            identity={"name": gen["name"],
                      "email": gen["email"],
                      "disposable": "true",
                      "persona_id": gen["persona_id"]},
            persona_id=gen["persona_id"],
        )

    def _existing_account(self, platform: str) -> tuple[str, str] | None:
        """(vault_source, login) if an account for ``platform`` exists.

        Checks the trial vault (always readable) and the accounts vault
        (when the passphrase is set — the passphrase gate in
        ``_launch_assist`` refuses the run anyway when it isn't).  Used
        to enforce one-account-per-service in code before a new signup
        is launched, instead of relying on prose.
        """
        key = (platform or "").strip().lower()
        if not key:
            return None
        try:
            entry = self.vault.get(key)
        except Exception:  # noqa: BLE001 - treat as unknown, keep going
            entry = None
        if entry is not None and not entry.get("unreadable"):
            return ("trial", str(entry.get("login") or "?"))
        passphrase = os.environ.get("NM_VAULT_PASSPHRASE", "")
        if passphrase and self.db is not None:
            try:
                from ...accounts.vault import CredentialVault

                vault = CredentialVault(self.db,
                                        master_passphrase=passphrase)
                for cred in vault.list_all(service=key, active_only=True):
                    return ("accounts",
                            str(getattr(cred, "username", "?") or "?"))
            except Exception:  # noqa: BLE001 - absence of evidence
                _log.debug("trial existing-account check failed for %s",
                           key)
        return None

    def _launch_assist(self, platform: str, *, chat_key: str,
                       identity: dict[str, str], persona_id: str = "",
                       supersedes: str = "") -> str:
        """Start the background assist run. The confirmation gate has
        already been passed by the caller (card + /trial confirm, or
        --yes); a resume re-drive carries the owner's action as the
        go-ahead."""
        if not os.environ.get("NM_VAULT_PASSPHRASE", ""):
            return (
                "vault is locked: set the NM_VAULT_PASSPHRASE environment "
                "variable so I can store the new credentials, then ask me again."
            )
        # One account per service, enforced in code: refuse to drive a
        # second signup when one already exists in either vault.  The
        # owner redelivers with /trial send or removes it first.
        existing = self._existing_account(platform)
        if existing:
            source, login = existing
            self._audit("assist_refused", platform,
                        f"already have account ({source} vault)")
            return (
                f"already have an account for {platform} "
                f"({source} vault, login: {login}) — one account per "
                "service.\n"
                f"  /trial send {platform}   — redeliver the credentials\n"
                f"  /trial rm {platform}     — remove it, then retry assist"
            )
        if not self._assist_sem.acquire(blocking=False):
            return (
                f"already driving {_assist_max_inflight()} assisted signups — "
                "wait for one to finish, then try again. "
                "check status: /trial status"
            )
        run_id = new_id()
        with self._assist_lock:
            self._assist_runs[run_id] = {
                "platform": platform.lower(),
                "chat_key": chat_key or "",
                "started": time.time(),
                "state": "starting",
                "note": "",
            }
        # durable before the thread starts — a restart from here on
        # still reports the run instead of swallowing it.
        self._persist_assist_run(run_id)
        thread = threading.Thread(
            target=self._assist_run,
            args=(run_id, platform, chat_key, dict(identity), persona_id,
                  supersedes),
            name=f"trial-assist-{platform.lower()[:20]}",
            daemon=True,
        )
        thread.start()
        self._audit("assist_start", platform, f"run {run_id}")
        return (
            f"assisted signup for {platform} — started.\n"
            f"identity: {identity.get('name')} <{identity.get('email')}>\n"
            "I'm driving the whole signup myself — forms, CAPTCHA solver, "
            "email + SMS verification. I'll ping you only if I get genuinely "
            "stuck.\n"
            "check status: /trial status"
        )

    def resume(self, checkpoint_id: str, *, note: str = "",
               chat_key: str = "") -> str:
        """Continue a paused account flow from chat: ``/trial resume <id>``.

        The phone-side counterpart of ``nm account resume --id``: after the
        owner completes the human step (solves the CAPTCHA, finishes the
        manual signup), one chat message continues the flow — no SSH, no
        CLI. Returns a truthful status either way.
        """
        checkpoint_id = (checkpoint_id or "").strip()
        if not checkpoint_id:
            return "usage: /trial resume <checkpoint-id>"
        if not os.environ.get("NM_VAULT_PASSPHRASE", ""):
            return (
                "vault is locked: set the NM_VAULT_PASSPHRASE environment "
                "variable so I can store the credentials, then ask me again."
            )
        try:
            from ...accounts.creator import (
                AccountCheckpoint,
                AccountCheckpointPending,
                AccountCreator,
                CreatedAccount,
            )
            from ...accounts.vault import CredentialVault
            from ...core.errors import NotFound
        except Exception as exc:  # noqa: BLE001
            return f"account automation unavailable: {exc}"

        vault = CredentialVault(
            self.db,
            master_passphrase=os.environ.get("NM_VAULT_PASSPHRASE", ""),
        )
        creator = AccountCreator(vault, db=self.db)
        try:
            cp = creator.checkpoints.get(checkpoint_id)
        except NotFound:
            return f"no checkpoint {checkpoint_id!r} — check /trial status"
        if cp.state.value != "pending":
            return (f"checkpoint {checkpoint_id} is already {cp.state.value} "
                    f"({cp.title}) — nothing to resume")
        identity = self._owner_identity()
        try:
            result = creator.resume_checkpoint(
                checkpoint_id, note,
                owner_name=identity.get("name") or None,
                owner_email=identity.get("email") or None)
        except AccountCheckpointPending as pending:
            nxt = pending.checkpoint
            return (
                "⏸️ still paused — one more human step:\n\n"
                f"**{nxt.title}**\n{nxt.instructions}\n\n"
                f"When you're done: `/trial resume {nxt.id}`"
            )
        except Exception as exc:  # noqa: BLE001 - truthful report
            _log.exception("trial resume %s failed", checkpoint_id)
            return f"❌ resume failed: {exc}"

        if isinstance(result, CreatedAccount):
            return (
                f"✅ account ready — {result.service}, username: "
                f"{result.username}"
                + (f", email: {result.email}" if result.email else "")
                + ". Credentials are stored in the vault."
            )
        if isinstance(result, AccountCheckpoint):
            flow = (result.resume_state or {}).get("flow", "")
            service = result.service or (result.resume_state or {}).get(
                "service", "")
            if flow in ("captcha_takeover",) and service:
                # The human step is done; the dead browser session is not
                # coming back — re-drive the signup automatically instead
                # of asking the owner to type another command.
                retry = self.assist(service, chat_key=chat_key)
                return (
                    f"✅ human step recorded ({result.title}).\n\n"
                    f"Re-driving the signup automatically:\n{retry}"
                )
            if flow == "signup_wall":
                # Owner acted on a genuinely-stuck signup — re-drive it
                # with the same persona, linked to the previous attempt.
                resume_state = result.resume_state or {}
                attempt_id = resume_state.get("attempt_id", "")
                persona_id = ""
                if attempt_id and self.db is not None:
                    try:
                        from ...accounts.signup_driver import (
                            SignupAttemptStore)
                        from ...accounts.identity_bank import IdentityBank

                        prev = SignupAttemptStore(self.db).get(attempt_id)
                        persona_id = prev.persona_id
                        service = service or prev.service
                    except Exception:  # noqa: BLE001
                        pass
                retry = self._launch_assist(
                    service or "unknown", chat_key=chat_key,
                    identity={"name": "(disposable persona)",
                              "email": "(temp address minted at signup)",
                              "disposable": "true",
                              "persona_id": persona_id},
                    persona_id=persona_id, supersedes=attempt_id)
                return (
                    f"✅ human step recorded ({result.title}).\n\n"
                    f"Re-driving the signup automatically:\n{retry}"
                )
            if flow == "need_identity":
                return (
                    "✅ identity recorded. Re-run the signup to continue:\n"
                    f"  /trial assist {service}" if service else
                    "✅ identity recorded.")
            return f"✅ checkpoint resolved: {result.title}"
        return f"✅ resumed {checkpoint_id}"

    def _assist_run_persona(self, run_id: str, platform: str,
                              chat_key: str, creator, vault,
                              persona_id: str, supersedes: str = "") -> str:
        """Drive the whole signup for a disposable persona.

        Confirmation was already collected (card + ``/trial confirm`` or
        ``--yes``), so ``confirmed=True`` here is the code-level gate.
        The owner's real contacts are passed as the invariant set — the
        driver fails closed if the flow would touch them.
        """
        import asyncio

        from ...accounts.identity_bank import IdentityBank
        from ...accounts.signup_driver import (
            SignupAttemptStore,
            SignupDriver,
            render_attempt_summary,
        )

        bank = IdentityBank(db=self.db, vault=vault)
        persona = bank.get(persona_id) or bank.get_or_mint(platform)
        attempts = SignupAttemptStore(self.db)
        driver = SignupDriver(
            creator,
            attempts,
            notify=lambda t, b, aid: self._notify_owner(t, b, chat_key),
            page_factory=self._trial_page_factory,
        )
        # Research→execute handoff: if /trial start already found the
        # signup URL, hand it to the driver instead of guessing.
        signup_url = self._plan_signup_url(self._load_plan(platform))
        attempt = asyncio.run(driver.adrive(
            service=platform,
            persona=persona,
            confirmed=True,
            owner_contacts=self._owner_contacts(),
            supersedes=supersedes,
            **({"signup_url": signup_url} if signup_url else {}),
        ))
        return (
            f"✅ {attempt.service} account ready — credentials are in the "
            f"vault.\n{render_attempt_summary(attempt)}"
        )

    def _assist_run(self, run_id: str, platform: str, chat_key: str,
                    identity: dict[str, str], persona_id: str = "",
                    supersedes: str = "") -> None:
        """Background body of :meth:`assist`.

        Disposable-persona path (``persona_id`` set): drives the WHOLE
        signup via :class:`SignupDriver` — form fill, CAPTCHA solver,
        temp-mail + temp-SMS verification — pinging the owner only when
        genuinely stuck.  Owner-identity path: the legacy
        ``AccountCreator`` flow with their own details.
        """
        self._set_assist_run_state(run_id, "running")
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
            if persona_id:
                note = self._assist_run_persona(
                    run_id, platform, chat_key, creator, vault, persona_id,
                    supersedes=supersedes)
            else:
                account = asyncio.run(
                    creator.create_account(
                        platform, email=identity.get("email")))
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
                "⏸️ signup needs you — I drove it as far as I could:\n\n"
                f"**{cp.title}**\n{cp.instructions}\n\n"
                f"When you're done, just say: `/trial resume {cp.id}`\n"
                f"(or on the machine: `nm account resume --id {cp.id}`)"
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
        self._persist_assist_run(run_id)
        if ok and "⏸️" in note:
            outcome = "paused_for_human"
        else:
            outcome = "done" if ok else "failed"
        self._audit("assist_finish", platform, f"run {run_id}: {outcome}")
        self._notify_owner(f"trial assist — {platform}", note, chat_key)

    # ── durable assist-run state ──────────────────────────────────────────
    #
    # Background signups outlive the chat turn that started them; the
    # rows in ``trial_assist_runs`` outlive the process.  A restart can
    # never silently swallow a run: anything not in a terminal state is
    # marked ``interrupted`` and reported once via the durable notifier.

    def _ensure_assist_runs_table(self) -> None:
        if self.db is None:
            return
        try:
            self.db.executescript(_ASSIST_RUNS_DDL)
        except Exception:  # noqa: BLE001 - table is best-effort
            _log.debug("trial assist runs table unavailable")

    def _set_assist_run_state(self, run_id: str, state: str,
                              note: str = "") -> None:
        with self._assist_lock:
            run = self._assist_runs.get(run_id)
            if run is None:
                return
            run["state"] = state
            if note:
                run["note"] = note
        self._persist_assist_run(run_id)

    def _persist_assist_run(self, run_id: str) -> None:
        """Write one run row. Never holds the lock while touching the DB."""
        if self.db is None:
            return
        with self._assist_lock:
            run = dict(self._assist_runs.get(run_id, {}))
        if not run:
            return
        try:
            self.db.execute(
                "INSERT OR REPLACE INTO trial_assist_runs"
                " (run_id, platform, chat_key, started, state, note,"
                " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, str(run.get("platform", "")),
                 str(run.get("chat_key", "")),
                 float(run.get("started", 0) or 0),
                 str(run.get("state", "")),
                 str(run.get("note", ""))[:2000], time.time()),
            )
        except Exception:  # noqa: BLE001 - persistence is best-effort
            _log.debug("trial assist run persist failed for %s", run_id)

    def _load_assist_runs(self) -> None:
        """Reload persisted runs; prune terminal runs older than the TTL."""
        if self.db is None:
            return
        try:
            rows = self.db.query(
                "SELECT run_id, platform, chat_key, started, state, note"
                " FROM trial_assist_runs ORDER BY started")
        except Exception:  # noqa: BLE001
            return
        cutoff = time.time() - _ASSIST_RUN_TTL
        terminal = ", ".join(f"'{s}'"
                             for s in sorted(_ASSIST_TERMINAL_STATES))
        with self._assist_lock:
            for row in rows or []:
                rid = str(row.get("run_id") or "")
                if not rid or rid in self._assist_runs:
                    continue
                self._assist_runs[rid] = {
                    "platform": str(row.get("platform") or ""),
                    "chat_key": str(row.get("chat_key") or ""),
                    "started": float(row.get("started") or 0),
                    "state": str(row.get("state") or ""),
                    "note": str(row.get("note") or ""),
                }
        try:
            self.db.execute(
                "DELETE FROM trial_assist_runs WHERE started < ?"
                f" AND state IN ({terminal})",
                (cutoff,),
            )
        except Exception:  # noqa: BLE001 - pruning is best-effort
            _log.debug("trial assist run prune failed")

    def _maybe_recover_assist_runs(self) -> None:
        """One recovery pass per process, on first use after boot."""
        cls = type(self)
        if cls._assist_recovery_done:
            return
        cls._assist_recovery_done = True
        try:
            recovered = cls.recover_interrupted_runs(self.context,
                                                     self.gateway)
        except Exception:  # noqa: BLE001 - recovery must not break init
            _log.exception("trial assist recovery failed")
            return
        if recovered:
            # keep the in-memory view consistent with the rows just
            # marked interrupted in the DB.
            with self._assist_lock:
                for item in recovered:
                    run = self._assist_runs.get(item.get("run_id", ""))
                    if run is not None and run.get("state") not in (
                            _ASSIST_TERMINAL_STATES):
                        run["state"] = "interrupted"
                        run["note"] = (
                            "⚠️ interrupted by a bot restart — rerun: "
                            f"/trial assist {run.get('platform') or '?'}")

    @classmethod
    def recover_interrupted_runs(
        cls, context: Any, gateway: Any = None
    ) -> list[dict[str, Any]]:
        """Mark in-flight runs as interrupted and report them. Never raises.

        Called once per process (first TrialFlow use, plus the runtime
        startup hook) so a restart never silently swallows a background
        signup.  Each interrupted run gets a terminal ``interrupted``
        state and one honest report through the durable notifier — even
        with no live gateway the row persists for later redelivery.
        Returns the recovered runs (``run_id``/``platform``).
        """
        cls._assist_recovery_done = True
        recovered: list[dict[str, Any]] = []
        db = getattr(context, "db", None)
        if db is None:
            return recovered
        try:
            db.executescript(_ASSIST_RUNS_DDL)
            terminal = ", ".join(f"'{s}'"
                                 for s in sorted(_ASSIST_TERMINAL_STATES))
            rows = db.query(
                "SELECT run_id, platform, chat_key, started"
                " FROM trial_assist_runs WHERE state NOT IN"
                f" ({terminal}) ORDER BY started")
        except Exception:  # noqa: BLE001
            return recovered
        notifier = None
        for row in rows or []:
            rid = str(row.get("run_id") or "")
            platform = str(row.get("platform") or "?")
            note = (
                "⚠️ the assisted signup for "
                f"{platform} was interrupted by a bot restart — I don't "
                "know how far the signup got before the process died. "
                "Check the site directly, then rerun if needed:\n"
                f"  /trial assist {platform}"
            )
            try:
                db.execute(
                    "UPDATE trial_assist_runs SET state='interrupted',"
                    " note=?, updated_at=? WHERE run_id=?",
                    (note, time.time(), rid),
                )
            except Exception:  # noqa: BLE001
                _log.debug("trial run interrupt-mark failed for %s", rid)
            recovered.append({"run_id": rid, "platform": platform})
            try:
                if notifier is None:
                    from ..notifier import Notifier

                    gw = gateway
                    if gw is None:
                        gw = getattr(context, "gateway", None)
                    notifier = Notifier(context, gateway=gw)
                notifier.publish("trial",
                                 f"trial assist interrupted — {platform}",
                                 note, force=True)
            except Exception:  # noqa: BLE001 - reporting must not crash
                _log.exception("trial interrupt report failed for %s", rid)
        return recovered

    def assist_status(self) -> str:
        """Status of background assist runs (``/trial status``).

        Reads the durable run table, so runs from before a restart show
        up too — interrupted ones say so honestly instead of vanishing.
        """
        with self._assist_lock:
            runs = list(self._assist_runs.items())
        watch_lines = self._sms_watch_status_lines()
        if not runs and not watch_lines:
            return "no assisted signups yet — /trial assist <platform> to start one."
        lines = []
        if runs:
            lines.append("assisted signups:")
            ordered = sorted(runs, key=lambda kv: kv[1].get("started", 0))
            if len(ordered) > 10:
                lines.append(f"  (showing latest 10 of {len(ordered)})")
                ordered = ordered[-10:]
            for run_id, run in ordered:
                when = time.strftime("%H:%M",
                                     time.localtime(run.get("started", 0)))
                state = run.get("state") or "?"
                marker = " ⚠️" if state == "interrupted" else ""
                lines.append(
                    f"  {run.get('platform')} [{state}]{marker} started {when}"
                )
                note = str(run.get("note") or "")
                if note:
                    lines.append(f"    {note[:220]}")
        lines.extend(watch_lines)
        lines.extend(self._signup_attempt_lines())
        return "\n".join(lines)

    def _signup_attempt_lines(self) -> list[str]:
        """Per-attempt signup records for ``/trial status``."""
        if self.db is None:
            return []
        try:
            from ...accounts.signup_driver import SignupAttemptStore
        except Exception:  # noqa: BLE001
            return []
        try:
            attempts = SignupAttemptStore(self.db).list(limit=10)
        except Exception:  # noqa: BLE001
            return []
        if not attempts:
            return []
        lines = ["signup attempts:"]
        for attempt in attempts:
            wall = (f" — wall: {attempt.wall_kind.value}"
                    if attempt.wall_kind.value != "none" else "")
            lines.append(
                f"  {attempt.service} [{attempt.stage.value}]{wall} "
                f"persona: {attempt.persona_name or attempt.persona_id}"
            )
            if attempt.note:
                lines.append(f"    {attempt.note[:200]}")
        return lines

    # ── durable sms-watch state ─────────────────────────────────────────
    #
    # A verification-code watch outlives the chat turn that started it;
    # the rows in ``trial_sms_watches`` outlive the process.  A restart
    # can never silently swallow a watch: anything still ``watching``
    # past its deadline is closed as ``timeout`` and reported once via
    # the durable notifier; anything still inside its deadline is
    # re-armed for the remaining time, because the pinned number info
    # is all a fresh poll thread needs.

    def _ensure_sms_watches_table(self) -> None:
        if self.db is None:
            return
        try:
            self.db.executescript(_SMS_WATCHES_DDL)
        except Exception:  # noqa: BLE001 - table is best-effort
            _log.debug("trial sms watches table unavailable")

    def _register_sms_watch(self, watch_id: str, info: dict[str, Any],
                            chat_key: str, started: float,
                            timeout: float) -> None:
        number = info.get("masked") or info.get("number") or "?"
        with self._sms_lock:
            self._sms_watches[watch_id] = {
                "number": str(number),
                "number_info": dict(info),
                "chat_key": str(chat_key or ""),
                "started": started,
                "deadline": started + timeout,
                "timeout": timeout,
                "state": "watching",
                "code": "",
                "note": "",
            }
        self._persist_sms_watch(watch_id)

    def _set_sms_watch_state(self, watch_id: str, state: str, *,
                             code: str = "", note: str = "") -> None:
        with self._sms_lock:
            watch = self._sms_watches.get(watch_id)
            if watch is None:
                return
            watch["state"] = state
            if code:
                watch["code"] = code
            if note:
                watch["note"] = note
        self._persist_sms_watch(watch_id)

    def _persist_sms_watch(self, watch_id: str) -> None:
        """Write one watch row. Never holds the lock while touching the DB."""
        if self.db is None:
            return
        with self._sms_lock:
            watch = dict(self._sms_watches.get(watch_id, {}))
        if not watch:
            return
        try:
            self.db.execute(
                "INSERT OR REPLACE INTO trial_sms_watches"
                " (watch_id, number, number_info, chat_key, started,"
                "  deadline, timeout, state, code, note, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (watch_id, str(watch.get("number", "")),
                 json.dumps(watch.get("number_info") or {}),
                 str(watch.get("chat_key", "")),
                 float(watch.get("started", 0) or 0),
                 float(watch.get("deadline", 0) or 0),
                 float(watch.get("timeout", 0) or 0),
                 str(watch.get("state", "")),
                 str(watch.get("code", "")),
                 str(watch.get("note", ""))[:2000], time.time()),
            )
        except Exception:  # noqa: BLE001 - persistence is best-effort
            _log.debug("trial sms watch persist failed for %s", watch_id)

    def _load_sms_watches(self) -> None:
        """Reload persisted watches; prune terminal ones older than the TTL."""
        if self.db is None:
            return
        try:
            rows = self.db.query(
                "SELECT watch_id, number, number_info, chat_key, started,"
                " deadline, timeout, state, code, note"
                " FROM trial_sms_watches ORDER BY started")
        except Exception:  # noqa: BLE001
            return
        cutoff = time.time() - _SMS_WATCH_TTL
        terminal = ", ".join(f"'{s}'"
                             for s in sorted(_SMS_WATCH_TERMINAL_STATES))
        with self._sms_lock:
            for row in rows or []:
                wid = str(row.get("watch_id") or "")
                if not wid or wid in self._sms_watches:
                    continue
                try:
                    info = json.loads(row.get("number_info") or "{}")
                except Exception:  # noqa: BLE001 - corrupt row, keep going
                    info = {}
                self._sms_watches[wid] = {
                    "number": str(row.get("number") or ""),
                    "number_info": info if isinstance(info, dict) else {},
                    "chat_key": str(row.get("chat_key") or ""),
                    "started": float(row.get("started") or 0),
                    "deadline": float(row.get("deadline") or 0),
                    "timeout": float(row.get("timeout") or 0),
                    "state": str(row.get("state") or ""),
                    "code": str(row.get("code") or ""),
                    "note": str(row.get("note") or ""),
                }
        try:
            self.db.execute(
                "DELETE FROM trial_sms_watches WHERE started < ?"
                f" AND state IN ({terminal})",
                (cutoff,),
            )
        except Exception:  # noqa: BLE001 - pruning is best-effort
            _log.debug("trial sms watch prune failed")

    def _maybe_recover_sms_watches(self) -> None:
        """One recovery pass per process, on first use after boot."""
        cls = type(self)
        if cls._sms_recovery_done:
            return
        cls._sms_recovery_done = True
        try:
            resumed = cls.recover_interrupted_watches(self.context,
                                                      self.gateway)
        except Exception:  # noqa: BLE001 - recovery must not break init
            _log.exception("trial sms watch recovery failed")
            return
        if resumed:
            # keep the in-memory view consistent with the rows the
            # recovery pass just re-armed or closed.
            with self._sms_lock:
                for item in resumed:
                    watch = self._sms_watches.get(item.get("watch_id", ""))
                    if watch is None:
                        continue
                    watch["state"] = item.get("state", watch.get("state"))
                    if item.get("note"):
                        watch["note"] = item["note"]

    @classmethod
    def recover_interrupted_watches(
        cls, context: Any, gateway: Any = None
    ) -> list[dict[str, Any]]:
        """Resume or close watches killed by a restart. Never raises.

        Called once per process (first TrialFlow use, plus the runtime
        startup hook).  Watches still inside their deadline are re-armed
        for the remaining time on a fresh daemon thread — the pinned
        number info is everything the poll needs.  Watches past their
        deadline are closed as ``timeout`` with an honest note.  Both
        paths report once through the durable notifier, so even with no
        live gateway the row persists for later redelivery.
        Returns the recovered watches (``watch_id``/``state``/``note``).
        """
        cls._sms_recovery_done = True
        recovered: list[dict[str, Any]] = []
        db = getattr(context, "db", None)
        if db is None:
            return recovered
        try:
            db.executescript(_SMS_WATCHES_DDL)
            rows = db.query(
                "SELECT watch_id, number, number_info, chat_key, started,"
                " deadline, timeout FROM trial_sms_watches"
                " WHERE state='watching' ORDER BY started")
        except Exception:  # noqa: BLE001
            return recovered
        now = time.time()
        notifier = None
        # (number, note) per watch — reported as ONE message after the loop
        # so a restart with many live watches doesn't spam one ping each.
        resumed_items: list[tuple[str, str]] = []
        expired_items: list[tuple[str, str]] = []

        def _notifier() -> Any:
            nonlocal notifier
            if notifier is None:
                from ..notifier import Notifier

                gw = gateway
                if gw is None:
                    gw = getattr(context, "gateway", None)
                notifier = Notifier(context, gateway=gw)
            return notifier

        def _finish(watch_id: str, state: str, code: str,
                    note: str) -> None:
            try:
                db.execute(
                    "UPDATE trial_sms_watches SET state=?, code=?, note=?,"
                    " updated_at=? WHERE watch_id=?",
                    (state, code, note[:2000], time.time(), watch_id),
                )
            except Exception:  # noqa: BLE001
                _log.debug("trial sms watch close failed for %s", watch_id)

        def _report_watch_recovery(
            resumed: list[tuple[str, str]],
            expired: list[tuple[str, str]],
            notify: Callable[[], Any],
        ) -> None:
            """Report the recovery pass in ONE owner message. Never raises.

            A single watch keeps the exact per-watch message it always had
            (same title/body, so nothing downstream changes).  Two or more
            collapse into a digest — one line per watch — so a restart with
            a busy watch table doesn't deliver N separate pings.
            """
            total = len(resumed) + len(expired)
            if total == 0:
                return
            if total == 1:
                if resumed:
                    number, note = resumed[0]
                    title = f"sms code watch resumed — {number}"
                else:
                    number, note = expired[0]
                    title = f"sms code watch expired — {number}"
            else:
                lines: list[str] = []
                if resumed:
                    lines.append(f"🔄 resumed ({len(resumed)}):")
                    lines.extend(
                        f"  • {number} — {note[:220]}"
                        for number, note in resumed
                    )
                if expired:
                    lines.append(f"⌛ expired ({len(expired)}):")
                    lines.extend(
                        f"  • {number} — {note[:220]}"
                        for number, note in expired
                    )
                title = (f"sms code watches recovered — "
                         f"{len(resumed)} resumed, {len(expired)} expired")
                note = "\n".join(lines)
            try:
                notify().publish("trial", title, note, force=True)
            except Exception:  # noqa: BLE001 - reporting never crashes
                _log.exception("sms watch recovery report failed")

        for row in rows or []:
            wid = str(row.get("watch_id") or "")
            number = str(row.get("number") or "?")
            chat_key = str(row.get("chat_key") or "")
            try:
                info = json.loads(row.get("number_info") or "{}")
            except Exception:  # noqa: BLE001
                info = {}
            if not isinstance(info, dict):
                info = {}
            deadline = float(row.get("deadline") or 0)
            timeout = float(row.get("timeout") or 0)
            if deadline <= now:
                note = (
                    f"⌛ the sms code watch on {number} expired while the "
                    "bot was restarting — no code arrived in time. "
                    "The number may still work; start a fresh watch:\n"
                    "  /trial sms code"
                )
                _finish(wid, "timeout", "", note)
                recovered.append({"watch_id": wid, "state": "timeout",
                                  "note": note})
                expired_items.append((number, note))
                continue
            # still inside the deadline — re-arm for the remaining time.
            remaining = max(1.0, deadline - now)
            note = (
                f"🔄 resumed after a bot restart — still watching {number} "
                f"for another {remaining:.0f}s."
            )
            try:
                db.execute(
                    "UPDATE trial_sms_watches SET note=?, updated_at=?"
                    " WHERE watch_id=?",
                    (note[:2000], time.time(), wid),
                )
            except Exception:  # noqa: BLE001
                _log.debug("trial sms watch resume-mark failed: %s", wid)
            recovered.append({"watch_id": wid, "state": "watching",
                              "note": note})
            resumed_items.append((number, note))

            def _resumed(info: dict = dict(info), wid: str = wid,
                         number: str = number, chat_key: str = chat_key,
                         remaining: float = remaining,
                         note: str = note) -> None:
                from ...accounts.temp_sms import wait_code

                code = ""
                error = ""
                try:
                    code = wait_code(info, timeout=remaining)
                except Exception as exc:  # noqa: BLE001 - close honestly
                    error = str(exc)
                    _log.exception("resumed sms watch poll failed: %s",
                                   wid)
                finally:
                    try:
                        from ...storage.db import release_thread_connection

                        release_thread_connection(db)
                    except Exception:  # noqa: BLE001 - best-effort
                        pass
                if code:
                    result = f"📩 verification code: {code}"
                    _finish(wid, "done", code, result)
                elif error:
                    result = (f"⚠️ the resumed sms code watch on {number} "
                              f"errored: {error}. try again: /trial sms code")
                    _finish(wid, "timeout", "", result)
                else:
                    result = (
                        f"no code arrived on {number} within "
                        f"{remaining:.0f}s of the resumed watch — the site "
                        "may not have sent one yet. try again: "
                        "/trial sms code"
                    )
                    _finish(wid, "timeout", "", result)
                try:
                    _notifier().publish(
                        "trial", f"sms code watch — {number}",
                        result, force=True)
                except Exception:  # noqa: BLE001 - reporting never crashes
                    _log.exception("resumed sms watch report failed: %s",
                                   wid)

            thread = threading.Thread(target=_resumed,
                                      name="trial-sms-watch-resumed",
                                      daemon=True)
            thread.start()
        # one owner message for the whole recovery pass — digest when
        # several watches were live, the same single message as before
        # when only one was.
        _report_watch_recovery(resumed_items, expired_items, _notifier)
        return recovered

    def _sms_watch_status_lines(self) -> list[str]:
        """Watch section for ``/trial status`` — reads the durable table."""
        with self._sms_lock:
            watches = list(self._sms_watches.items())
        if not watches:
            return []
        lines = ["sms code watches:"]
        ordered = sorted(watches, key=lambda kv: kv[1].get("started", 0))
        if len(ordered) > 10:
            lines.append(f"  (showing latest 10 of {len(ordered)})")
            ordered = ordered[-10:]
        for _watch_id, watch in ordered:
            when = time.strftime("%H:%M",
                                 time.localtime(watch.get("started", 0)))
            state = watch.get("state") or "?"
            lines.append(f"  {watch.get('number')} [{state}] started {when}")
            note = str(watch.get("note") or "")
            if note:
                lines.append(f"    {note[:220]}")
            elif watch.get("code"):
                lines.append(f"    code: {watch.get('code')}")
        return lines

    def _audit(self, action: str, platform: str = "",
               outcome: str = "") -> None:
        """Append one secret-free row to the trial action audit log.

        Never carries credentials, codes, or secrets — action/platform/
        outcome only.  The log lives next to the trial vault data so a
        later review can answer "what did the trial flow touch, when".
        """
        try:
            home = self.vault.home
            home.mkdir(parents=True, exist_ok=True)
            row = {
                "ts": time.time(),
                "action": action,
                "platform": (platform or "").lower(),
                "outcome": (outcome or "")[:200],
            }
            with open(home / _TRIAL_AUDIT_LOG, "a",
                       encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except Exception:  # noqa: BLE001 - audit never breaks the action
            _log.debug("trial action audit write failed")

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

        Tries providers in cascade order (then fallback countries), so
        one provider outage doesn't hard-fail the grab — the ``provider``
        argument just picks which provider is tried first.  The number
        is stashed in kv_store so ``/trial sms code`` can poll it without
        the owner pasting JSON around.  Free providers, no API key — the
        number is public, so it is only good for throwaway verification
        codes, never for anything sensitive.
        """
        from ...accounts.temp_sms import (
            CASCADE_PROVIDERS,
            grab_number_cascade,
        )

        preferred = (provider or "simcodes").strip() or "simcodes"
        providers = tuple([preferred]
                          + [p for p in CASCADE_PROVIDERS if p != preferred])
        info = grab_number_cascade(
            country=(country or "us").strip() or "us",
            providers=providers,
        )
        if info.get("status") != "ok":
            return f"❌ {info.get('notes', 'could not grab a number')}"
        self._stash_temp_number(info)
        self._audit("temp_number", "", info.get("provider", ""))
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
        return _format_sms_poll_result(info, code, timeout)

    def temp_sms_code_async(self, chat_key: str = "",
                            timeout: float = 180) -> str:
        """Watch the stashed temp number in the background (``/trial sms code``).

        Returns immediately; the code (or a timeout note) is reported to
        the owner's chat when it lands.  The chat thread never blocks.

        The watch is durable: its row in ``trial_sms_watches`` pins the
        exact temp number grabbed (a newer ``/trial sms`` can't hijack
        it) and a restart resumes live watches or closes expired ones
        instead of silently dropping them.
        """
        info = self._load_temp_number()
        if not info:
            return "no temp number stashed — grab one first: /trial sms [country]"
        number = info.get("masked") or info.get("number") or "?"
        watch_id = new_id()
        started = time.time()
        self._register_sms_watch(watch_id, info, chat_key, started, timeout)

        thread = threading.Thread(
            target=self._run_sms_watch,
            args=(watch_id, dict(info), chat_key, str(number), timeout),
            name="trial-sms-watch",
            daemon=True,
        )
        thread.start()
        self._audit("sms_watch_start", "", str(number))
        return (
            f"👀 watching {number} for a verification code "
            f"(up to {timeout:.0f}s) — I'll ping you here the moment one lands."
        )

    def _run_sms_watch(self, watch_id: str, info: dict[str, Any],
                       chat_key: str, number: str, timeout: float) -> None:
        """Background poll for one watch. Pins ``info`` — never re-reads
        the kv_store, so a newer grabbed number can't hijack the watch."""
        from ...accounts.temp_sms import wait_code

        code = ""
        try:
            code = wait_code(info, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - the watch must close honestly
            _log.exception("sms watch poll failed for %s", watch_id)
            result = (f"⚠️ the sms code watch on {number} errored: {exc}. "
                      "try again: /trial sms code")
            self._set_sms_watch_state(watch_id, "timeout", note=result)
            self._audit("sms_watch", "", "error")
            self._notify_owner(f"sms code watch — {number}", result,
                               chat_key)
            return
        finally:
            try:
                from ...storage.db import release_thread_connection

                release_thread_connection(self.db)
            except Exception:  # noqa: BLE001 - cleanup is best-effort
                pass
        result = _format_sms_poll_result(info, code, timeout)
        self._set_sms_watch_state(
            watch_id, "done" if code else "timeout",
            code=code or "", note=result)
        self._audit("sms_watch", "",
                    "code received" if code else "timeout, no code")
        self._notify_owner(f"sms code watch — {number}", result,
                           chat_key)

    def _stash_temp_number(self, info: dict[str, Any]) -> None:
        if self.db is None:
            return
        try:
            KVStore(self.db).set(TEMP_SMS_KV_KEY, info)
        except Exception:  # noqa: BLE001 - stash is best-effort
            _log.debug("temp number stash failed")

    def _load_temp_number(self) -> dict[str, Any]:
        if self.db is None:
            return {}
        try:
            data = KVStore(self.db).get(TEMP_SMS_KV_KEY)
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
            data = KVStore(db).get(AccountCreator.IDENTITY_KV_KEY)
            if not data:
                return {}
            return data
        except Exception:  # noqa: BLE001
            return {}

    def _generate_disposable_identity(
        self, platform: str, persona_id: str = ""
    ) -> tuple[Any, dict[str, str]]:
        """Mint (or re-fetch) the disposable persona for a trial signup.

        The single construction path for disposable identities, used by
        :meth:`assist` and :meth:`confirm_signup`.  Delegates to the
        vault-side :class:`IdentityBank` — the same persona is reused for
        the service's retries (same name on the form, same username
        attempts, same temp email).  Uses temp/disposable contacts —
        NEVER the owner's real details.  Returns ``(persona,
        identity_dict)`` with name, email slot, persona_id, and the
        generated flag.
        """
        bank = self._identity_bank()
        persona = bank.get(persona_id) if persona_id else None
        if persona is None:
            persona = bank.get_or_mint(platform)
        identity = {
            "name": persona.name,
            "email": persona.email or "(temp address minted at signup)",
            "disposable": "true",
            "platform": platform,
            "persona_id": persona.id,
        }
        return persona, identity

    def _identity_bank(self):
        """Vault-side identity bank (vault when the passphrase is set)."""
        from ...accounts.identity_bank import IdentityBank
        from ...accounts.vault import CredentialVault

        passphrase = os.environ.get("NM_VAULT_PASSPHRASE", "")
        vault = None
        if passphrase and self.db is not None:
            try:
                vault = CredentialVault(self.db,
                                        master_passphrase=passphrase)
            except Exception:  # noqa: BLE001 - fall back to table backend
                vault = None
        return IdentityBank(db=self.db, vault=vault)

    def _confirmation_gate(self):
        """Process-local confirmation gate for persona use.

        Process-local by design: a restart drops pending confirmations
        and ``/trial confirm`` then honestly reports an unknown token
        (the owner reruns ``/trial assist`` for a fresh card).
        """
        global _CONFIRM_GATE
        if _CONFIRM_GATE is None:
            from ...accounts.identity_bank import ConfirmationGate

            _CONFIRM_GATE = ConfirmationGate()
        return _CONFIRM_GATE

    def _owner_contacts(self) -> dict[str, list[str]]:
        """The owner's real contact details — the signup driver checks
        every contact it touches against these and fails closed."""
        identity = self._owner_identity()
        emails = [identity.get("email", "")] if identity.get("email") else []
        phones = [identity.get("phone", "")] if identity.get("phone") else []
        return {"emails": [e for e in emails if e],
                "phones": [p for p in phones if p]}

    @staticmethod
    def _trial_page_factory():
        """Open a real rendered browser tab for form driving.

        Raises when no browser/playwright is available — the driver
        treats that as "no page" and hands to the owner honestly.
        """
        from ...browser.service import BrowserService

        svc = BrowserService()
        return svc.open_rendered_tab("trial-signup")

    @staticmethod
    def run_browser_task(steps: list[dict[str, Any]],
                         session: str = "trial-signup") -> dict[str, Any]:
        """Run a declarative browser task program for a signup scenario.

        ``steps`` is a list of ``{act, ...}`` — open/fill/click/submit/
        wait/screenshot/extract/observe/scroll/press — executed on a real
        Chromium tab via the spine ``browser`` tool's rendered engine.
        Lets signup scenarios be expressed as data (brain-generatable)
        instead of hardcoded Python. Raises honestly when Playwright is
        missing.
        """
        from ...tools.browser import get_rendered_session
        sess = get_rendered_session(session)
        try:
            return sess.task(steps=steps)
        finally:
            try:
                sess.close()
            except Exception:  # noqa: BLE001 - close is best-effort
                pass

    # ── save + deliver ───────────────────────────────────────────────────────
    def save(self, platform: str, login: str, secret: str, note: str = "") -> dict:
        if not (platform and login and secret):
            raise ToolError("usage: /trial save <platform> <login> <password>")
        stored = self.vault.store(platform, login, secret, note)
        overwrote = bool(stored.get("overwrote"))
        self._audit("save", platform,
                    "overwrote existing" if overwrote else "stored")
        return {"platform": platform.lower(), "login": login,
                "overwrote": overwrote}

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
            self._audit("deliver", platform, "no live channel — shown inline")
            return "(no live WhatsApp/Telegram in this session — shown here)\n" + body
        self._audit("deliver", platform, "sent on " + ",".join(sent))
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
            stale = ("  ⚠️ saved over 90 days ago — consider rotating"
                     if row.get("stale") else "")
            lines.append(f"  {row['platform']}: {row['login']}  (saved {when}){stale}")
        return "\n".join(lines)

    def remove(self, platform: str) -> str:
        if self.vault.delete(platform):
            self._audit("delete", platform, "deleted")
            return f"deleted trial account for {platform.lower()}."
        self._audit("delete", platform, "not found")
        return f"no stored trial account for {platform.lower()}."

    def audit_log(self, platform: str = "", limit: int = 20) -> str:
        """Recent secret-free trial action audit rows (``/trial audit``).

        Answers "what did the trial flow touch, when" — actions and
        platforms only, never secrets or codes.
        """
        home = self.vault.home
        path = home / _TRIAL_AUDIT_LOG
        if not path.exists():
            return "no trial audit rows yet."
        rows: list[dict[str, Any]] = []
        try:
            for line in path.read_text("utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001 - skip corrupt lines
                    continue
                if isinstance(row, dict) and (
                        not platform
                        or str(row.get("platform") or "") == platform.lower()):
                    rows.append(row)
        except Exception:  # noqa: BLE001 - best-effort read
            return "could not read the trial audit log."
        rows.sort(key=lambda r: r.get("ts", 0), reverse=True)
        rows = rows[:max(1, limit)]
        if not rows:
            return "no trial audit rows yet."
        lines = ["trial audit (newest first):"]
        for row in rows:
            when = time.strftime("%m-%d %H:%M",
                                 time.localtime(row.get("ts", 0)))
            plat = row.get("platform") or "—"
            outcome = row.get("outcome") or ""
            lines.append(f"  {when}  {row.get('action')}  {plat}"
                         + (f"  ({outcome})" if outcome else ""))
        lines.extend(self._vault_audit_lines(platform, limit))
        return "\n".join(lines)

    def _vault_audit_lines(self, platform: str, limit: int) -> list[str]:
        """Vault-level access rows (store/get/delete/list) for the audit."""
        try:
            rows = self.vault.audit_trail(platform, limit=limit)
        except Exception:  # noqa: BLE001 - best-effort
            return []
        if not rows:
            return []
        lines = ["vault access (newest first):"]
        for row in rows:
            when = time.strftime("%m-%d %H:%M",
                                 time.localtime(row.get("ts", 0)))
            plat = row.get("platform") or "—"
            ok = "" if row.get("ok", True) else "  (not found/unreadable)"
            lines.append(f"  {when}  {row.get('action')}  {plat}{ok}")
        return lines

    def _search_engine(self) -> Any:
        from ..search.engine import SearchEngine

        return SearchEngine(self.context)


def new_id() -> str:
    return new_short_id(length=12)
