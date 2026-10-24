"""RuntimeProvisionMixin: self-provisioning credentials in chat.

The chat face of :mod:`nomorals.accounts.provisioning`:

* ``/provision [service]`` — status board, or run the drive: Devon
  logs in on the owner's own account (or creates it — owner identity,
  one per service), creates the API key, validates it, vaults it,
  bridges it into the environment, and retries the request that
  originally failed for the missing key.
* ``/provision login <service> <username>`` — arms a ONE-TIME login
  intake: the owner's next message in that chat is the password. It is
  consumed at the very top of the inbound pipeline — before the
  autonomy interest recorder and before the brain — so the secret
  never reaches history, logs, or interest models, and goes straight
  into the CredentialVault. After the handoff the drive proceeds on
  its own. (The host-side equivalent is the hidden-prompt CLI:
  ``nm account store-login``.)
* The missing-credential offer: when a feature raises
  :class:`~nomorals.core.errors.MissingCredentialError` on an owner
  surface, the failure reply offers the drive and remembers the
  request for the post-provision retry. Offers NEVER appear on the
  gated public surface (pass #55: surface gating is law — on
  ``telegram-bot`` nobody is an operator, so no offer is made).
* Drives are watchable: every tab the drive opens gets a LiveView
  attached (default ON; ``NM_PROVISION_LIVE_VIEW=0`` opts out) and is
  registered in the watch registry that ``/browser watch`` attaches
  to on demand.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ...core.logging_setup import get_logger
from .failures import record_failure

_log = get_logger(__name__)

#: How long an armed login intake / pending retry stays valid.
_INTAKE_TTL_S = 300.0


class RuntimeProvisionMixin:
    """RuntimeProvisionMixin for :class:`PartnerRuntime`."""

    # ── shared state (lazily built; the runtime is one instance) ─────

    def _provision_state(self) -> dict[str, Any]:
        state = getattr(self, "_provision_mixin_state", None)
        if state is None:
            state = {
                "intakes": {},        # chat_key -> intake dict
                "consumed_ids": set(),  # message ids of consumed secrets
                "retries": {},        # chat_key -> retry dict
                "watch_tabs": {},     # session name -> live tab
                "watch_views": {},    # chat_key -> LiveView
            }
            self._provision_mixin_state = state
        return state

    # ── vault / organs ───────────────────────────────────────────────

    def _provision_vault(self) -> Any:
        """The accounts vault (same one /connectors uses). Raises the
        connectors chat error (user-facing text) when locked/absent."""
        return self._connectors_vault()

    def _provision_solver(self) -> Any:
        try:
            from ...tools.captcha import creator_solver_adapter

            return creator_solver_adapter(settings=self.settings)
        except Exception:  # noqa: BLE001 - solver absence = human steps
            _log.debug("provisioning captcha solver unavailable",
                       exc_info=True)
            return None

    def _adapter_for(self, platform: str) -> Any | None:
        gateway = getattr(self, "gateway", None)
        if gateway is None:
            return None
        adapters = getattr(gateway, "adapters", None)
        if isinstance(adapters, dict):
            return adapters.get(platform)
        lookup = getattr(gateway, "_adapter_for", None)
        return lookup(platform) if callable(lookup) else None

    # ── startup bridge ───────────────────────────────────────────────

    def _bridge_provisioned_credentials(self) -> list[str]:
        """Populate registry env keys from the vault at startup.

        Best-effort by design: a locked vault or missing DB must never
        brick boot — features then fail honestly (and offer the drive)
        when used. Returns the env keys populated."""
        try:
            # Pass 8: the betting desk's keyed services register from
            # the agents layer (betdesk.sources), which boot never
            # imported — a vaulted desk key would never bridge at
            # startup.  Ensure them first (idempotent, best-effort).
            try:
                from ..betdesk.sources import ensure_provisioning

                ensure_provisioning()
            except Exception:  # noqa: BLE001 - desk registration is
                _log.debug("desk provisioning registration skipped",
                           exc_info=True)
            from ...accounts.provisioning import bridge_vault_credentials

            vault = self._provision_vault()
            applied = bridge_vault_credentials(vault)
            if applied:
                _log.info("provisioning bridge at startup: %s",
                          ", ".join(applied))
            return applied
        except Exception:  # noqa: BLE001 - boot must not brick
            _log.debug("provisioning startup bridge skipped",
                       exc_info=True)
            return []

    # ── the drive ────────────────────────────────────────────────────

    def _provision_open_tab(self, service: str, *, chat: Any = None,
                            label: str = "") -> Any:
        """Production tab factory (mirrors `nm account login`): a
        rendered tab in the service's persistent cookie session, with
        the plain-HTTP tab as fallback. Every tab is registered for
        /browser watch and, when a chat is attached, gets the drive's
        LiveView (default ON)."""
        from ...browser.service import BrowserService

        svc = BrowserService(
            data_dir=self.settings.resolve("data/browser-service"))
        try:
            tab = svc.open_rendered_tab(service)
        except Exception:  # noqa: BLE001 - rendered needs playwright
            handle = svc.open_session(service)
            tab = handle.open_tab()
        state = self._provision_state()
        state["watch_tabs"][service] = tab
        if chat is not None:
            self._drive_live_view(tab, chat, label=label or service)
        return tab

    def _drive_live_view(self, tab: Any, chat: Any, *, label: str) -> Any:
        """Attach (or retarget) this chat's provisioning LiveView.

        One window per drive: the first tab starts the view, later
        tabs retarget it. Default ON — the owner watches credential
        drives happen. Never raises (LiveView's own contract)."""
        if os.environ.get("NM_PROVISION_LIVE_VIEW", "1") == "0":
            return None
        state = self._provision_state()
        chat_key = str(getattr(chat, "key", "") or "")
        view = state["watch_views"].get(chat_key)
        if view is not None and getattr(view, "_active", False):
            view.retarget(tab)
            return view
        adapter = self._adapter_for(str(getattr(chat, "platform", "") or ""))
        if adapter is None:
            return None
        from ...browser.liveview import LiveView

        view = LiveView.attach(
            tab, chat, adapter,
            task_label=f"provisioning: {label}")
        if view is not None:
            state["watch_views"][chat_key] = view
        return view

    def _run_provision_drive(self, service_id: str, *, chat_key: str = "",
                             message: Any = None) -> str:
        """Run one provisioning drive and render the outcome. On
        success, a pending retry for this chat+service runs too."""
        from ...accounts.provisioning import ProvisionDriver

        try:
            vault = self._provision_vault()
        except Exception as exc:  # noqa: BLE001 - user-facing text
            return str(exc)
        chat = getattr(message, "chat", None) if message is not None \
            else None
        solver = self._provision_solver()

        def _open_tab(service: str) -> Any:
            return self._provision_open_tab(service, chat=chat,
                                            label=service_id)

        driver = ProvisionDriver(
            vault, open_tab=_open_tab, captcha_solver=solver)
        try:
            result = driver.provision(service_id)
        except Exception as exc:  # noqa: BLE001 - honest, value-free
            _log.exception("provisioning drive crashed for %s", service_id)
            return (f"provisioning {service_id} crashed before it could "
                    f"finish: {exc} — nothing was stored.")
        lines = [result.message]
        for warning in result.warnings:
            lines.append(f"note: {warning}")
        view = self._provision_state()["watch_views"].get(chat_key)
        if view is not None:
            try:
                if result.ok:
                    view.finish(result.message)
                else:
                    view.fail(result.message)
            except Exception as exc:  # noqa: BLE001 - cosmetic
                record_failure("runtime_provision._run_provision_drive", exc)
        if result.ok and chat_key:
            retry_out = self._run_pending_provision_retry(
                chat_key, result.service_id)
            if retry_out:
                lines.append("")
                lines.append("retrying what you asked for:")
                lines.append(retry_out)
        return "\n".join(lines)

    # ── /provision ───────────────────────────────────────────────────

    def _control_provision(self, tail: str, *, chat_key: str = "",
                           message: Any = None) -> str:
        """Self-provisioning from chat (owner only — the grant already
        denied non-owners and the gated surface before dispatch)."""
        from ...accounts.provisioning import (
            get_provision_service,
            provisioned_status,
            resolve_provision_services,
        )

        if message is not None and not self._is_operator(message):
            return "that one's just for the owner."
        # The desk's services register from the agents layer; ensure
        # them here too (idempotent) so /provision desk and
        # /provision odds_api resolve even on a runtime whose
        # startup bridge was skipped.
        try:
            from ..betdesk.sources import ensure_provisioning

            ensure_provisioning()
        except Exception:  # noqa: BLE001 - registration best-effort
            _log.debug("desk provisioning registration skipped",
                       exc_info=True)
        text = (tail or "").strip()
        if not text or text.lower() in ("list", "status"):
            try:
                vault = self._provision_vault()
            except Exception as exc:  # noqa: BLE001 - user-facing text
                return str(exc)
            rows = provisioned_status(vault)
            if not rows:
                return "no provisionable services are registered."
            lines = ["provisioning — what I can fetch myself:"]
            for row in rows:
                if row["env_set"]:
                    state = f"live ({', '.join(row['env_set'])} set)"
                elif row["has_key"]:
                    state = "key in the vault (bridged at startup)"
                elif row["has_login"]:
                    state = (f"login saved ({', '.join(row['login_usernames'])})"
                             " — key not fetched yet")
                else:
                    state = "not provisioned"
                lines.append(
                    f"  {row['service']} ({row['display']}): {state}")
            lines.append("run /provision <service> and I'll go get the key.")
            return "\n".join(lines)

        parts = text.split()
        if parts[0].lower() == "login":
            if len(parts) < 3:
                return ("usage: /provision login <service> <username> — "
                        "then your NEXT message is the password (consumed "
                        "silently, straight into the vault, never stored "
                        "or logged).")
            svc = get_provision_service(parts[1])
            if svc is None:
                return (f"no provisioning registry entry for {parts[1]!r}.")
            return self._arm_login_intake(chat_key, svc, parts[2])

        targets = resolve_provision_services(parts[0])
        if not targets:
            from ...accounts.provisioning import PROVISION_REGISTRY

            known = ", ".join(sorted(PROVISION_REGISTRY)) or "none"
            return (f"I don't know how to provision {parts[0]!r} — "
                    f"registered services: {known}.")
        if len(targets) == 1:
            return self._run_provision_drive(targets[0].id,
                                             chat_key=chat_key,
                                             message=message)
        # A bundle word (e.g. "desk"): drive each member in turn and
        # report per-service outcomes — never one blob verdict.
        sections: list[str] = []
        for svc in targets:
            sections.append(f"── {svc.display} ──")
            sections.append(self._run_provision_drive(
                svc.id, chat_key=chat_key, message=message))
        return "\n".join(sections)

    # ── the missing-credential offer + pending retry ─────────────────

    def _provision_offer_for_missing(self, missing: Any, *,
                                     retry: tuple[str, str],
                                     chat_key: str = "",
                                     message: Any = None) -> str:
        """The offer appended to a missing-credential failure, or ""
        when no offer may be made (non-owner, gated surface — where
        nobody is an operator — or an unregistered service). Arms the
        pending retry so the drive can re-run the original request."""
        if message is None or not chat_key:
            return ""
        try:
            if not self._is_operator(message):
                return ""
        except Exception as exc:  # noqa: BLE001 - no offer beats a crash
            record_failure("runtime_provision._provision_offer_for_missing", exc)
            return ""
        from ...accounts.provisioning import get_provision_service

        svc = get_provision_service(getattr(missing, "service", "") or "")
        if svc is None:
            return ""
        state = self._provision_state()
        state["retries"][chat_key] = {
            "service": svc.id,
            "retry": retry,
            "expires": time.time() + _INTAKE_TTL_S,
        }
        env_key = getattr(missing, "env_key", "") or (
            svc.env_keys[0] if svc.env_keys else "")
        return (
            f"that's a missing credential, not a dead end: I can go get "
            f"a {svc.display} key myself — say /provision {svc.id} and "
            f"I'll fetch one on your account, vault it, wire it into "
            f"{env_key} and re-run this. (The offer only exists here in "
            f"your own chats — never on the public bot.)")

    def _run_pending_provision_retry(self, chat_key: str,
                                     service_id: str) -> str:
        state = self._provision_state()
        pending = state["retries"].get(chat_key)
        if not pending:
            return ""
        if pending.get("service") != service_id:
            return ""
        if float(pending.get("expires", 0)) < time.time():
            state["retries"].pop(chat_key, None)
            return ""
        state["retries"].pop(chat_key, None)
        kind, payload = pending.get("retry") or ("", "")
        if kind == "image_generate":
            return self._image_generate(payload)
        return ""

    # ── secure login handoff (one-time armed intake) ─────────────────

    def _arm_login_intake(self, chat_key: str, svc: Any,
                          username: str) -> str:
        if not chat_key:
            return ("I can only take a login handoff inside a chat — "
                    "or use the host CLI's hidden prompt: "
                    f"nm account store-login --service {svc.vault_service} "
                    f"--username {username}")
        state = self._provision_state()
        state["intakes"][chat_key] = {
            "service": svc.id,
            "vault_service": svc.vault_service,
            "display": svc.display,
            "login_url": svc.login_url,
            "username": username,
            "expires": time.time() + _INTAKE_TTL_S,
        }
        return (
            f"armed: your NEXT message in this chat is the {svc.display} "
            f"password for {username}. It goes straight into the "
            "encrypted vault — it is consumed before the brain or any "
            "history/log sees it — and then I'll fetch the key without "
            "further input. (Delete the password message from the chat "
            "afterwards for hygiene; Devon never stores it.) "
            "Prefer not to type it here? Use the host's hidden prompt "
            f"instead: nm account store-login --service "
            f"{svc.vault_service} --username {username}")

    def _try_consume_login_intake(self, message: Any) -> bool:
        """Consume an armed login intake. Runs at the TOP of the
        inbound pipeline (before autonomy recorders and the brain);
        returns True when the message was consumed (the caller must
        not process it further). A redelivery of an already-consumed
        secret (same platform message id) is dropped silently."""
        if not getattr(message, "incoming", False):
            return False
        text = getattr(message, "text", "") or ""
        chat = getattr(message, "chat", None)
        chat_key = str(getattr(chat, "key", "") or "")
        if not chat_key:
            return False
        state = self._provision_state()
        message_id = str(getattr(message, "message_id", "") or "")
        if message_id and message_id in state["consumed_ids"]:
            return True  # replay of a consumed secret: never reprocess
        intake = state["intakes"].get(chat_key)
        if intake is None:
            return False
        if float(intake.get("expires", 0)) < time.time():
            state["intakes"].pop(chat_key, None)
            return False
        if not text or text.strip().startswith("/"):
            return False  # commands are never passwords; stays armed
        try:
            if not self._is_operator(message):
                state["intakes"].pop(chat_key, None)
                return False
        except Exception:  # noqa: BLE001 - fail closed
            state["intakes"].pop(chat_key, None)
            return False
        # Consume. From here the secret value lives ONLY in this frame
        # and the vault write — it is never logged or rendered.
        state["intakes"].pop(chat_key, None)
        if message_id:
            state["consumed_ids"].add(message_id)
        password = text.strip()
        try:
            vault = self._provision_vault()
            vault.store(
                intake["vault_service"], intake["username"], password,
                credential_type="password",
                tags=["login", "handoff"],
                metadata={"login_url": intake.get("login_url", ""),
                          "source": "chat_handoff",
                          "handed_over_at": time.time()},
            )
        except Exception as exc:  # noqa: BLE001 - user-facing text
            reply = (f"the handoff didn't land: {exc} — nothing was "
                     "stored. The intake is disarmed; arm it again with "
                     f"/provision login {intake['service']} "
                     f"{intake['username']} when ready.")
            self._send_intake_reply(message, reply)
            return True
        _log.info("provisioning: login handed over for service %s "
                  "(value never logged)", intake["service"])
        reply = (
            f"login saved to the vault for {intake['display']} "
            f"({intake['username']}). Fetching the key now…")
        self._send_intake_reply(message, reply)
        drive = self._run_provision_drive(
            intake["service"], chat_key=chat_key, message=message)
        self._send_intake_reply(message, drive)
        return True

    def _send_intake_reply(self, message: Any, reply: str) -> None:
        try:
            self._typing_for(message.chat, reply)
            self.gateway.send(message.chat.platform, message.chat, reply)
        except Exception:  # noqa: BLE001 - reply is best-effort
            _log.debug("intake reply send failed", exc_info=True)

    # ── /browser watch (on-demand LiveView attach) ───────────────────

    def _watchable_sessions(self) -> dict[str, Any]:
        """Sessions a LiveView can attach to RIGHT NOW: tabs this
        runtime is driving (provisioning registry) plus the takeover
        browser service's open sessions. Tabs inside per-job browser
        services are separate objects and are NOT visible here — the
        /browser watch reply says so rather than pretending."""
        out: dict[str, Any] = dict(self._provision_state()["watch_tabs"])
        try:
            svc = self._browser_service_for_takeover()
            for name, handle in (getattr(svc, "_sessions", {}) or {}).items():
                if name not in out:
                    tab = handle.active_tab()
                    if tab is not None:
                        out[name] = tab
        except Exception:  # noqa: BLE001 - probe is best-effort
            _log.debug("watch: takeover session probe failed",
                       exc_info=True)
        return out

    def _control_browser_watch(self, tail: str, *, chat_key: str = "",
                               message: Any = None) -> str:
        """``/browser watch [session]`` — attach a LiveView to a live
        browser session so the owner can watch Devon's work on demand,
        not only when a task starts. Owner-only."""
        if message is None or not self._is_operator(message):
            return "watch is owner-only — run it from the owner's chat."
        sessions = self._watchable_sessions()
        if not sessions:
            return ("no live browser session to watch right now — "
                    "sessions exist while a drive is running "
                    "(provisioning, takeover, browser runs on the "
                    "runtime's own service). Per-job browser sessions "
                    "live in their own service objects and can't be "
                    "attached from here.")
        want = (tail or "").strip()
        name = want if want in sessions else (
            next(iter(sessions)) if len(sessions) == 1 else "")
        if not name:
            return ("which session? live right now: "
                    + ", ".join(sorted(sessions))
                    + " — say /browser watch <name>.")
        chat = getattr(message, "chat", None)
        adapter = self._adapter_for(str(getattr(chat, "platform", "") or ""))
        if adapter is None or chat is None:
            return "this chat has no adapter to stream the live view to."
        from ...browser.liveview import LiveView

        state = self._provision_state()
        old = state["watch_views"].get(chat_key)
        if old is not None:
            try:
                old.detach()
            except Exception as exc:  # noqa: BLE001
                record_failure("runtime_provision._control_browser_watch", exc)
        view = LiveView.attach(
            sessions[name], chat, adapter,
            task_label=f"watching: {name}")
        if view is None:
            return (f"couldn't attach a live view to session {name!r} "
                    "(the tab can't take screenshots).")
        state["watch_views"][chat_key] = view
        return (f"live view attached to session {name!r} — I'll keep "
                "posting frames to this chat as she works. It detaches "
                "when the session's task finishes.")
