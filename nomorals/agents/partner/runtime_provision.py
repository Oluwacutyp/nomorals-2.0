"""RuntimeProvisionMixin: self-provisioning credentials in chat.

The chat face of :mod:`nomorals.accounts.provisioning`:

* ``/provision [service]`` — status board, or run the drive: Devon
  logs in on the owner's own account (or creates it — owner identity,
  one per service), creates the API key, validates it, vaults it,
  bridges it into the environment, and retries the request that
  originally failed for the missing key.
* ``/provision login service username`` — arms a ONE-TIME login
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


def _provision_failure_head(result: Any) -> str:
    """The one-line failure head for the live view's ❌ frame.

    The frame is the visual surface (screenshot + this head); the
    drive's returned reply carries the full failure text exactly
    once. The head names the failed step when the result carries
    one, else the outcome — never the whole message (feeding the
    frame the whole message put the same text in front of the owner
    twice: once inside the ❌ frame, once as the standalone reply —
    the 2026-10-26 double-send).
    """
    step = (getattr(result, "step", "") or "").strip()
    service = (getattr(result, "service_id", "") or "").strip() \
        or "provisioning"
    if step:
        return f"{service} failed at step {step}"
    outcome = getattr(result, "outcome", "")
    label = getattr(outcome, "value", outcome)
    return f"{service} failed ({label})"


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
                    # Short head only — the full failure text is the
                    # reply below, exactly once (see
                    # _provision_failure_head).
                    view.fail(_provision_failure_head(result))
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
            lines.append("run /provision with a service name and I'll go get the key.")
            return "\n".join(lines)

        parts = text.split()
        if parts[0].lower() == "login":
            if len(parts) < 3:
                return ("usage: /provision login service username — "
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

    # ── pairing-with-code account pool (§15) ────────────────────────
    def _pairing_supervisor(self) -> Any:
        sup = getattr(self, "_pair_supervisor", None)
        if sup is None:
            from ...social.pairing import default_supervisor

            ws = getattr(self.context.settings, "workspace_dir", ".")
            sup = default_supervisor(ws)
            self._pair_supervisor = sup
        return sup

    @staticmethod
    def _pair_find(sup: Any, ref: str) -> Any | None:
        ref = (ref or "").strip()
        if not ref:
            return None
        rec = sup.registry.get(ref)
        if rec is not None:
            return rec
        digits = "".join(c for c in ref if c.isdigit())
        matches = [r for r in sup.registry.all()
                   if r.identity == digits or r.id.endswith(ref)]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _pair_launch_note(sup: Any, rec: Any) -> str | None:
        """The beat after a bridge (re)start: if the process died
        on the spot (missing packages, a crash on launch), the
        bridge's own last line is the reply — not a cheerful
        "started".  A healthy bridge says nothing inside the
        beat and the caller answers as before; a death that takes
        longer belongs to the summary / supervision, not here."""
        import time as _time

        for _ in range(6):
            reason = sup.pairing_error(rec.id)
            if reason:
                return f"⚠️ {reason}"
            _time.sleep(0.1)
        return None

    def _pairing_personal(self) -> dict[str, Any]:
        """The §15 slices 2–3 service bundle: registry + bridge
        supervisor (slice 1), the Telegram pairer, the personal
        surfaces organ, and the supervision sweep — built once,
        lazily, over the runtime workspace."""
        svc = getattr(self, "_pair_personal_svc", None)
        if svc is not None:
            return svc
        from pathlib import Path

        from ...social.pairing import PoolSupervision
        from ...social.personal import PersonalSurfaces, PersonalVault
        from ...social.telegram_pairing import TelegramPairer

        sup = self._pairing_supervisor()
        ws = Path(getattr(self.context.settings, "workspace_dir", "."))
        chat = getattr(self.context.settings, "chat", None)
        pairer = TelegramPairer(
            sup.registry,
            sessions_root=ws / "data" / "telegram_sessions",
            api_id=getattr(chat, "telegram_api_id", ""),
            api_hash=getattr(chat, "telegram_api_hash", ""))

        def _story_fetcher(rec: Any) -> list[dict[str, Any]]:
            adapter = self._adapter_for("telegram")
            fetch = getattr(adapter, "fetch_active_stories", None)
            if not callable(fetch):
                return []  # adapter exposes no story feed — absence
            return list(fetch() or [])

        def _presence_setter(platform: str, rec: Any,
                             online: bool) -> bool:
            """The always-online verb for one platform (§15
            presence pack): the live adapter's set_presence —
            bridge presence on WhatsApp, Telethon account status
            on Telegram.  No adapter / no verb / a refused verdict
            all read as not-landed, never as success."""
            adapter = self._adapter_for(platform)
            fn = getattr(adapter, "set_presence", None)
            if not callable(fn):
                return False
            try:
                res = fn(online)
            except Exception as exc:  # noqa: BLE001
                record_failure(
                    "runtime_provision.presence_setter", exc)
                return False
            if isinstance(res, dict):
                if not res.get("ok"):
                    # A refused verb used to collapse to a bare
                    # False and vanish (truth pass, 2026-11-07):
                    # the bridge's own reason goes to the failure
                    # ledger so a broken surface is diagnosable.
                    record_failure(
                        "runtime_provision.presence_setter",
                        RuntimeError(str(
                            res.get("error")
                            or "presence verb refused")))
                return bool(res.get("ok"))
            return bool(res)

        def _status_viewer(rec: Any, item: dict[str, Any]) -> bool:
            """Mark one WhatsApp status viewed as the owner: the
            adapter's view_status (bridge status_view verb)."""
            adapter = self._adapter_for("whatsapp")
            fn = getattr(adapter, "view_status", None)
            if not callable(fn):
                return False
            try:
                res = fn(str(item.get("poster_id") or ""),
                         str(item.get("source_id") or ""),
                         float(item.get("ts") or 0.0))
            except Exception as exc:  # noqa: BLE001
                record_failure(
                    "runtime_provision.status_viewer", exc)
                return False
            if isinstance(res, dict):
                if not res.get("ok"):
                    record_failure(
                        "runtime_provision.status_viewer",
                        RuntimeError(str(
                            res.get("error")
                            or "status_view verb refused")))
                return bool(res.get("ok"))
            return bool(res)

        def _story_viewer(rec: Any, poster_id: str,
                          max_story_id: int) -> bool:
            """Mark a Telegram peer's stories viewed: the
            adapter's view_stories (ReadStories)."""
            adapter = self._adapter_for("telegram")
            fn = getattr(adapter, "view_stories", None)
            if not callable(fn):
                return False
            try:
                return bool(fn(poster_id, max_story_id))
            except Exception as exc:  # noqa: BLE001
                record_failure(
                    "runtime_provision.story_viewer", exc)
                return False

        def _channel_fetcher(rec: Any) -> list[dict[str, Any]]:
            """The followed channels' recent posts for one
            WhatsApp account (channel stories build): the
            adapter's fetch_channel_posts (bridge channel_fetch
            — the Go engine enumerates the followed channels
            itself; the Baileys rung fetches the channels this
            surface knows).  No verb / a refused verdict reads
            as an empty feed — absence, never invented posts."""
            adapter = self._adapter_for("whatsapp")
            fn = getattr(adapter, "fetch_channel_posts", None)
            if not callable(fn):
                return []
            try:
                return list(fn() or [])
            except Exception as exc:  # noqa: BLE001
                record_failure(
                    "runtime_provision.channel_fetcher", exc)
                return []

        def _channel_viewer(rec: Any, item: dict[str, Any]) -> bool:
            """Mark one channel post viewed as the owner (channel
            stories build): the WhatsApp adapter's
            channel_view_posts (bridge channel_view — whatsmeow
            NewsletterMarkViewed on the Go engine; the Baileys
            rung's library has no mark-viewed verb and its
            honest ok:false lands here as not-landed, with the
            bridge's own reason recorded — never a bare False
            that reads like "toggle off", truth-pass law)."""
            adapter = self._adapter_for("whatsapp")
            fn = getattr(adapter, "channel_view_posts", None)
            if not callable(fn):
                return False
            try:
                res = fn(str(item.get("poster_id") or ""),
                         [str(item.get("source_id") or "")])
            except Exception as exc:  # noqa: BLE001
                record_failure(
                    "runtime_provision.channel_viewer", exc)
                return False
            if isinstance(res, dict):
                if not res.get("ok"):
                    record_failure(
                        "runtime_provision.channel_viewer",
                        RuntimeError(str(
                            res.get("error")
                            or "channel_view verb refused")))
                return bool(res.get("ok"))
            return bool(res)

        def _status_reactor(rec: Any, item: dict[str, Any]) -> bool:
            """React to one contact's status post as the owner
            (chat & privacy wave — the status_react toggle's
            verb): the WhatsApp adapter's react_status (bridge
            status_react).  Telegram grows no seam here: MTProto
            exposes no status-reaction verb on this surface, so
            the toggle simply never fires there — absence, not
            a fake."""
            adapter = self._adapter_for("whatsapp")
            fn = getattr(adapter, "react_status", None)
            if not callable(fn):
                return False
            try:
                res = fn(str(item.get("poster_id") or ""),
                         str(item.get("source_id") or ""),
                         str(item.get("emoji") or "❤️"))
            except Exception as exc:  # noqa: BLE001
                record_failure(
                    "runtime_provision.status_reactor", exc)
                return False
            if isinstance(res, dict):
                if not res.get("ok"):
                    record_failure(
                        "runtime_provision.status_reactor",
                        RuntimeError(str(
                            res.get("error")
                            or "status_react verb refused")))
                return bool(res.get("ok"))
            return bool(res)

        def _status_react_pool(rec: Any) -> list[str]:
            """The rotation's custom half for one account: the
            unicode faces of HER custom emoji set (the
            /customemoji registry).  Read live from the
            expression library so a set she grows mid-run joins
            the rotation without a restart; any failure leaves
            the built-in + account pool, never a dead chain."""
            try:
                registry = self._expression_library().custom_emoji
                return [str(entry.get("unicode") or "")
                        for _name, entry in
                        sorted(registry.to_dict().items())
                        if str(entry.get("unicode") or "")]
            except Exception as exc:  # noqa: BLE001 - garnish seam
                record_failure(
                    "runtime_provision.status_react_pool", exc)
                return []

        def _call_rejector(rec: Any, item: dict[str, Any]) -> bool:
            """Reject one ringing call as the owner (chat &
            privacy wave — the anticall toggle's verb): the
            WhatsApp adapter's reject_call (bridge reject_call)."""
            adapter = self._adapter_for("whatsapp")
            fn = getattr(adapter, "reject_call", None)
            if not callable(fn):
                return False
            try:
                res = fn(str(item.get("call_id") or ""),
                         str(item.get("from_id") or ""))
            except Exception as exc:  # noqa: BLE001
                record_failure(
                    "runtime_provision.call_rejector", exc)
                return False
            if isinstance(res, dict):
                if not res.get("ok"):
                    record_failure(
                        "runtime_provision.call_rejector",
                        RuntimeError(str(
                            res.get("error")
                            or "reject_call verb refused")))
                return bool(res.get("ok"))
            return bool(res)

        personal = PersonalSurfaces(
            sup.registry,
            PersonalVault(ws / "data" / "social" / "personal"),
            story_fetcher=_story_fetcher,
            presence_setter=_presence_setter,
            status_viewer=_status_viewer,
            story_viewer=_story_viewer,
            channel_fetcher=_channel_fetcher,
            channel_viewer=_channel_viewer,
            status_reactor=_status_reactor,
            call_rejector=_call_rejector,
            status_react_pool=_status_react_pool)
        svc = {
            "supervisor": sup, "registry": sup.registry,
            "pairer": pairer, "personal": personal,
            "supervision": PoolSupervision(sup.registry, sup, pairer),
            "last_cycle": 0.0,
        }
        self._pair_personal_svc = svc
        return svc

    #: Tag on last_error values THIS reconcile wrote, so a later
    #: successful attach clears exactly its own notes and never a
    #: bridge error the supervisor recorded.
    _WA_SURFACE_NOTE_PREFIX = "chat surface: "

    def _wa_surface_note(self, sup: Any, rec_id: str, note: str) -> str:
        """Record a surface failure on the account (visible in
        /pair) and return it for the caller's reply. Never raises."""
        try:
            if rec_id:
                sup.registry.set_error(
                    rec_id, self._WA_SURFACE_NOTE_PREFIX + note)
        except Exception as exc:  # noqa: BLE001 - noting never dies
            record_failure("runtime_provision.wa_surface_note", exc)
        return note

    def _wa_surface_clear_note(self, sup: Any, rec: Any) -> None:
        """A successful attach retires this reconcile's own earlier
        failure note on the account. Never raises."""
        try:
            if rec is not None and str(
                    getattr(rec, "last_error", "") or "").startswith(
                    self._WA_SURFACE_NOTE_PREFIX):
                sup.registry.set_error(rec.id, "")
        except Exception as exc:  # noqa: BLE001
            record_failure("runtime_provision.wa_surface_clear", exc)

    def _wa_bridge_down(self, sup: Any, rec: Any, reason: str) -> str:
        """The bridge-down verdict for one reconcile pass: the
        note the caller answers with (also recorded on the
        account via :meth:`_wa_surface_note`, so /pair shows it),
        plus ONE failure-ledger entry per EPISODE.  An episode is
        one (account, reason) signature — beats that keep finding
        the same dead bridge do not spam the ledger; a recovery
        (the reconcile clears the signature when the bridge
        serves again) or a new reason re-arms the recording."""
        note = f"bridge down — {reason}"
        sig = f"{rec.id}|{reason}"
        if getattr(self, "_wa_bridge_down_sig", "") != sig:
            self._wa_bridge_down_sig = sig
            record_failure(
                "runtime_provision.wa_bridge_down",
                RuntimeError(
                    f"whatsapp bridge down for {rec.id}: {reason}"))
            _log.warning("whatsapp surface not attached for %s: %s",
                         rec.id, note)
        return self._wa_surface_note(sup, rec.id, note)

    def _presence_suffix(self, platform: str) -> str:
        """The /platforms suffix for always-online (§15 presence
        pack): when the platform's resolved account carries the
        owner's always_online flag, the line says the presence
        truth the last beat computed — "online", or why not.
        No flag → no suffix.  Never raises: a listing must not
        die on a presence read."""
        try:
            from ...social.pairing import resolve_account

            svc = self._pairing_personal()
            rec = resolve_account(svc["registry"], platform)
            if rec is None or not rec.toggle("always_online"):
                return ""
            state = svc["personal"].presence_state(rec.id)
            return f" — always-online: {state or 'on'}"
        except Exception:  # noqa: BLE001 - listing never dies
            return ""

    def _set_platform_owner_stopped(self, platform: str,
                                    stopped: bool) -> None:
        """Record the owner's start/stop order for a platform in
        the durable pairing registry (see
        :meth:`PairingRegistry.set_owner_stopped`) — the input the
        live-attach reconcile was missing when an owner's
        /stop whatsapp resurrected within a beat (2026-11-04):
        the pool said "linked, bridge live", nothing anywhere said
        "the owner told it to stay down".  Never raises: recording
        intent must not break the stop/start it rides on."""
        try:
            sup = self._pairing_supervisor()
            sup.registry.set_owner_stopped(platform, stopped)
        except Exception as exc:  # noqa: BLE001 - noting never dies
            record_failure("runtime_provision.owner_stopped", exc)

    def _sync_whatsapp_surface(self) -> str:
        """Live-attach the WhatsApp conversational surface to the pool.

        The gateway's whatsapp adapter is built ONCE at boot against
        the static config endpoint, and gateway start drops it when
        no bridge is listening yet (the adapter's preflight) — so an
        account paired mid-session used to stay conversationally
        dark until a restart that happened to coincide with a live
        bridge.  This reconcile runs on the pairing lifecycle points
        (/pair whatsapp|start|stop|restart|remove|use) and on the
        supervision beat, and keeps exactly one truth: the resolved
        WhatsApp account (the §15 routing primitive: primary → first
        linked → first registered) whose bridge the supervisor can
        serve gets an adapter pointed at ITS bridge port, started
        in the running process via the gateway's own ``start_one``;
        when no account qualifies any more,
        an adapter this reconcile attached (or adopted — one already
        serving that very bridge) is stopped again.  An adapter
        running for a bridge outside the pool is left alone whenever
        the pool has no qualifying account: the reconcile only ever
        stops what it owns (``_wa_surface_account``).

        One hard exception outranks the pool: an owner stop order
        (``registry.owner_stopped("whatsapp")``, recorded by /stop
        and /pair stop, cleared by /start, /pair start and a fresh
        pair).  While it stands, the desired state is DOWN no
        matter how live the account looks — any WhatsApp adapter
        found up is stopped and nothing is ever attached (and no
        bridge is ever launched for it).  Without
        this term a manual /stop lasted only until the next beat.

        And the bridge is part of the truth, not an assumption:
        before any attach, the supervisor ENSURES the account's
        bridge process (``BridgeSupervisor.ensure_running`` —
        running, listening on its port, or (re)launched, with a
        throttled retry for a bridge that dies on launch).  When
        no bridge can serve, the surface does not attach and does
        not claim to: an owned adapter is put down and the answer
        is a bridge-down note (see :meth:`_wa_bridge_down`).

        Returns "" on success/no-op, else a plain-words note naming
        why the surface did not come up; the note is ALSO recorded
        on the account's ``last_error`` (tagged, see
        ``_WA_SURFACE_NOTE_PREFIX``) so the supervision-beat caller —
        which has no reply to append to — still surfaces it in
        /pair, never silently.  Never raises."""
        desired_id = ""
        try:
            gateway = getattr(self, "gateway", None)
            if gateway is None:
                return ""
            from ...social.pairing import resolve_account

            sup = self._pairing_supervisor()
            if sup.registry.owner_stopped("whatsapp"):
                # The owner ordered this surface down; the order
                # stands until an owner start path clears it.  Keep
                # it down — stop any adapter that is somehow up
                # (a beat that raced the stop, a boot-time builder)
                # — and never attach.
                if gateway.adapters.get("whatsapp") is not None:
                    gateway.stop_one("whatsapp")
                self._wa_surface_account = ""
                self._wa_bridge_down_sig = ""
                return ""
            rec = resolve_account(sup.registry, "whatsapp")
            desired_port = 0
            if rec is not None and rec.port and (
                    rec.status in ("linked", "down")
                    or sup.is_running(rec.id)):
                desired_id = rec.id
                desired_port = int(rec.port)
            attached = getattr(self, "_wa_surface_account", "") or ""
            running = gateway.adapters.get("whatsapp")
            running_live = bool(
                running is not None
                and getattr(running, "_thread", None) is not None
                and running._thread.is_alive())
            if not desired_id:
                if attached:
                    gateway.stop_one("whatsapp")
                    self._wa_surface_account = ""
                self._wa_bridge_down_sig = ""
                return ""
            # The bridge itself, before any adapter: a registry
            # status is a MEMORY of liveness, not liveness — a
            # persisted "linked" used to qualify an account whose
            # bridge process did not exist in this session, and
            # the adapter then sat "live-attached" to a dead port
            # in an endless reconnect loop (the owner's EC2 boot,
            # 2026-11-04: attached at 8787, nothing listening, no
            # launch ever attempted).  The supervisor ENSURES the
            # process now — running, listening (a survivor is
            # adopted, never double-launched), or (re)launched
            # through the same start() /pair uses, throttled
            # against fork-looping a bridge that dies on launch.
            serving, bridge_reason = sup.ensure_running(desired_id)
            if not serving:
                # Never sit attached to a dead bridge: an adapter
                # this reconcile owns comes down, the episode is
                # recorded once, and the answer is the bridge-down
                # note — on the reply AND on the account record.
                if attached == desired_id and \
                        gateway.adapters.get("whatsapp") is not None:
                    gateway.stop_one("whatsapp")
                    self._wa_surface_account = ""
                reason = bridge_reason or sup.pairing_error(
                    desired_id) or "the bridge process is not running"
                return self._wa_bridge_down(sup, rec, reason)
            # A serving bridge ends any bridge-down episode: the
            # next distinct failure records afresh.
            self._wa_bridge_down_sig = ""
            if running_live and int(
                    getattr(running, "port", 0) or 0) == desired_port:
                # Already serving this account's bridge (boot start
                # or an earlier attach): adopt it so a later removal
                # of the account still tears the surface down.
                self._wa_surface_account = desired_id
                self._wa_surface_clear_note(sup, rec)
                return ""
            if running is not None:
                gateway.stop_one("whatsapp")
            from ...social.chat import build_adapter

            adapter = build_adapter(self.settings, "whatsapp",
                                    whatsapp_port=desired_port)
            if adapter is None:
                # chat.whatsapp_enabled is off — paired but
                # surfaceless; say so on the record AND the reply.
                self._wa_surface_account = ""
                return self._wa_surface_note(
                    sup, desired_id,
                    "the WhatsApp platform is disabled in config "
                    "(chat.whatsapp_enabled=false) — this account "
                    "has no live chat surface until it's enabled")
            result = gateway.start_one("whatsapp", self.on_message,
                                       adapter=adapter)
            if result.get("ok") or "already running" in str(
                    result.get("error") or ""):
                self._wa_surface_account = desired_id
                self._wa_surface_clear_note(sup, rec)
                _log.info("whatsapp surface live-attached to %s "
                          "(bridge port %s)", desired_id, desired_port)
                return ""
            _log.warning("whatsapp surface attach for %s failed: %s",
                         desired_id, result.get("error"))
            return self._wa_surface_note(
                sup, desired_id,
                "the WhatsApp chat surface didn't start: "
                f"{result.get('error') or 'unknown error'}")
        except Exception as e:  # noqa: BLE001 - reconcile never
            record_failure(       # breaks intake or a /pair reply
                "runtime_provision.wa_surface_sync", e)
            if desired_id:
                try:
                    sup = self._pairing_supervisor()
                    return self._wa_surface_note(
                        sup, desired_id,
                        f"the WhatsApp chat surface didn't start: {e}")
                except Exception:  # noqa: BLE001 - noting is best-effort
                    pass
            return ""

    def _start_whatsapp_surface(self) -> dict[str, Any] | None:
        """The /start whatsapp route (see ``PartnerRuntime.
        start_platform``): the surface's start is the reconcile's
        start — the resolved account's bridge is part of the
        surface.  An account the owner stopped at the pool level
        (registry status "stopped") is first revived through the
        supervisor — the /pair start act — then the reconcile
        ensures the bridge and attaches the adapter to it.  The
        generic gateway start knows none of this: it would start
        an adapter against the static config endpoint with no
        bridge behind it.

        Returns the start verdict, or None when no WhatsApp
        account is paired (the caller's generic start owns that
        case, exactly as before).  Never raises."""
        try:
            from ...social.pairing import PairingUnavailable, \
                resolve_account

            gateway = getattr(self, "gateway", None)
            if gateway is None:
                return None
            sup = self._pairing_supervisor()
            rec = resolve_account(sup.registry, "whatsapp")
            if rec is None:
                return None
            if rec.status == "stopped" \
                    and not sup.is_running(rec.id):
                try:
                    sup.start(rec.id)
                except PairingUnavailable as exc:
                    return {"ok": False, "error": str(exc)}
            note = self._sync_whatsapp_surface()
            if gateway.adapters.get("whatsapp") is not None:
                return {"ok": True, "platform": "whatsapp"}
            if note:
                return {"ok": False, "error": note}
            return {"ok": False,
                    "error": "the WhatsApp surface did not start"}
        except Exception as exc:  # noqa: BLE001 - a broken route
            record_failure(       # falls back, never breaks /start
                "runtime_provision.start_whatsapp", exc)
            return None

    def _housekeeping_beat(self) -> None:
        """The throttled §15 housekeeping cycle, drivable by ANY
        inbound message.

        One beat, three jobs: the supervision sweep (dropped
        sessions flip honestly, dead bridges relaunch, stale
        pairing flows expire), the WhatsApp live-attach reconcile
        (a bridge the sweep just relaunched, or a probe-earned
        "linked", raises the conversational surface in this
        process — no restart), and the Telegram story sweep.
        Message-driven, like every other cycle in this runtime —
        no new daemon.

        The beat used to live inside ``_personal_surface_turn``,
        behind its whatsapp/telegram platform gate: in a bot-only
        deployment no inbound message ever passes that gate
        (WhatsApp cannot message without an adapter, and bot
        traffic carries a bot platform), so the reconcile never
        ran and a paired account stayed conversationally dark
        until a /pair or /start happened to call it directly.
        ``PartnerRuntime._process_inner`` now drives the beat for
        every platform, and the personal turn still calls it so
        direct callers keep the cadence — the 60s throttle state
        lives on the shared service bundle (``last_cycle``), so
        the two call sites can never double-sweep: whichever
        fires first in a window does the work.  Each job is
        failure-isolated; housekeeping never breaks intake."""
        svc = self._pairing_personal()
        now = time.time()
        if now - float(svc.get("last_cycle") or 0.0) <= 60.0:
            return
        svc["last_cycle"] = now
        try:
            svc["supervision"].sweep()
        except Exception as e:  # noqa: BLE001 - housekeeping
            record_failure(           # never breaks intake
                "runtime_provision.supervision_sweep", e)
        # Same beat: a bridge the sweep just relaunched, or a
        # probe-earned "linked", raises the conversational
        # surface in this process — no restart (live-attach).
        self._sync_whatsapp_surface()
        try:
            svc["personal"].cycle_stories("telegram")
        except Exception as e:  # noqa: BLE001
            record_failure(
                "runtime_provision.story_cycle", e)
        # Same beat: the WhatsApp channel sweep (channel
        # stories build) — followed channels' new posts saved /
        # marked viewed per the account's status toggles.
        try:
            svc["personal"].cycle_channel_posts("whatsapp")
        except Exception as e:  # noqa: BLE001
            record_failure(
                "runtime_provision.channel_cycle", e)
        # Same beat: always-online keepalive (§15 presence pack) —
        # re-asserts presence for the accounts the owner flagged,
        # releases it where the flag/stop/liveness says otherwise.
        try:
            svc["personal"].cycle_presence()
        except Exception as e:  # noqa: BLE001
            record_failure(
                "runtime_provision.presence_cycle", e)

    def _personal_surface_turn(self, message: Any) -> bool:
        """Observe one inbound message for the §15 personal surfaces.

        Returns True only when the message is FULLY consumed (a
        status post, a DM delete service event — neither is
        conversation).  Ordinary DMs are observed (DM cache,
        view-once preservation) and flow on to the conversation
        pipeline untouched.  Only the owner's personal-account
        platforms feed this: whatsapp + the telegram userbot —
        never the gated bot surface, never groups (group antidelete
        evidence is the defense organ's, and this never duplicates
        it), never channels."""
        platform = str(getattr(message.chat, "platform", "") or "")
        if platform not in ("whatsapp", "telegram"):
            return False
        meta = message.meta or {}
        service = str(meta.get("service") or "")
        kind = str(getattr(message.chat, "kind", "") or "")
        svc = self._pairing_personal()
        personal = svc["personal"]
        # The throttled housekeeping beat (supervision sweep,
        # WhatsApp live-attach, story sweep) is shared now — see
        # _housekeeping_beat.  It used to live ONLY here, behind
        # this turn's personal-platform gate, where bot-only
        # deployments never reached it; the generic inbound path
        # drives it for every platform, and this call keeps the
        # cadence for direct callers (the shared throttle makes
        # the second call in a window a no-op).  It still runs
        # BEFORE the observation logic below, as it always did.
        self._housekeeping_beat()
        now = time.time()
        chat_key = str(getattr(message.chat, "key", "") or "")
        counterpart = str(getattr(message.chat, "peer", "")
                          or getattr(message, "sender_id", "") or "")
        media = [{"path": str(m.path or ""), "kind": str(m.kind or "")}
                 for m in (getattr(message, "media", None) or [])
                 if getattr(m, "path", "")]
        ts = float(getattr(message, "ts", 0.0) or now)
        if service == "call_offer":
            # Chat & privacy wave: a call ringing his WhatsApp
            # account.  The personal organ rejects it when his
            # anticall toggle is on (and stays completely silent
            # under a standing owner stop).  Consumed either way —
            # a call offer is never conversation.
            personal.observe_call(
                platform,
                from_id=str(getattr(message, "sender_id", "")
                            or counterpart),
                from_name=str(getattr(message, "sender", "") or ""),
                call_id=str(meta.get("call_id") or ""),
                ts=ts)
            return True
        if service == "status_post":
            personal.observe_status_post(
                platform, poster_id=str(
                    getattr(message, "sender_id", "") or counterpart),
                poster_name=str(getattr(message, "sender", "") or ""),
                text=str(getattr(message, "text", "") or ""),
                media=media,
                source_id=str(getattr(message, "message_id", "")
                              or ""),
                media_missing=bool(
                    meta.get("status_media_missing")),
                ts=ts)
            return True
        if service == "delete":
            if kind != "dm":
                return False  # group deletes belong to the surface
            personal.observe_delete(
                platform, chat_key=chat_key,
                counterpart_id=counterpart,
                deleted_ids=[str(d) for d in
                             (meta.get("deleted_ids") or [])],
                ts=ts)
            return True
        if kind == "group":
            # Chat & privacy wave — the antitag sensor: a group
            # message whose mention payload mass-tags the owner.
            # Detection only: the organ journals it (toggle on)
            # and hands back a flag entry at most once an hour
            # per chat, which lands in his DM — never a callout
            # in the room.  The message itself flows on to the
            # conversation pipeline untouched.
            self._antitag_observe(message, personal, platform,
                                  chat_key=chat_key, ts=ts)
            return False
        if kind != "dm":
            return False
        if meta.get("edited"):
            personal.observe_edit(
                platform, chat_key=chat_key,
                counterpart_id=counterpart,
                message_id=str(getattr(message, "message_id", "")
                               or ""),
                new_text=str(getattr(message, "text", "") or ""),
                sender_id=str(getattr(message, "sender_id", "")
                              or ""),
                sender_name=str(getattr(message, "sender", "") or ""),
                ts=ts)
            return False
        if getattr(message, "incoming", False) \
                or meta.get("own_message"):
            personal.observe_message(
                platform, chat_key=chat_key,
                counterpart_id=counterpart,
                sender_id=str(getattr(message, "sender_id", "")
                              or ""),
                sender_name=str(getattr(message, "sender", "") or ""),
                message_id=str(getattr(message, "message_id", "")
                               or ""),
                text=str(getattr(message, "text", "") or ""),
                media=media,
                view_once=bool(meta.get("view_once")),
                ttl_seconds=int(meta.get("ttl_seconds") or 0),
                media_missing=bool(
                    meta.get("view_once_media_missing")),
                incoming=bool(getattr(message, "incoming", False)),
                ts=ts)
        return False

    def _antitag_observe(self, message: Any, personal: Any,
                         platform: str, *, chat_key: str,
                         ts: float) -> None:
        """The antitag sensor's live half (chat & privacy wave):
        count one group message's mention payload from its
        defense envelope, confirm the owner's own mention is in
        it (the adapters' ``mentioned`` flag), and hand the
        detection to the personal organ.  A returned flag entry
        is delivered to the owner's DM — his own chat with
        himself on the same surface — best-effort: a failed
        delivery is ledgered, never retried into a storm, and
        the organ's cooldown has already consumed the slot.
        Never raises: a sensor must not break intake."""
        try:
            if not bool(getattr(message, "mentioned", False)):
                return
            meta = getattr(message, "meta", None) or {}
            envelope = meta.get("envelope") or {}
            mentions = envelope.get("mentions") or []
            count = len([m for m in mentions if m])
            if not count:
                count = len([
                    e for e in (envelope.get("entities") or [])
                    if str(e.get("type") or "") in (
                        "mention", "text_mention")])
            from ...social.chat.chatops import is_mass_tag

            if not is_mass_tag(count):
                return
            entry = personal.observe_mass_tag(
                platform, chat_key=chat_key,
                sender_id=str(getattr(message, "sender_id", "")
                              or ""),
                sender_name=str(getattr(message, "sender", "")
                                or ""),
                mention_count=count, ts=ts)
            if not entry:
                return
            title = str(getattr(message.chat, "title", "")
                        or chat_key)
            who = str(entry.get("poster_name") or "someone")
            line = (f"🏷️ mass tag: {who} mentioned "
                    f"{int(entry.get('mention_count') or 0)} "
                    f"people including you in “{title}”.")
            suppressed = int(entry.get("suppressed_before") or 0)
            if suppressed:
                line += (f" ({suppressed} more mass tag"
                         f"{'' if suppressed == 1 else 's'} "
                         f"there since the last flag.)")
            self._antitag_deliver(platform, line)
        except Exception as exc:  # noqa: BLE001 - sensor discipline
            record_failure("runtime_provision.antitag_observe",
                           exc)

    def _antitag_deliver(self, platform: str, line: str) -> None:
        """Send one antitag flag to the owner's self-chat on the
        surface it happened on (WhatsApp: his own JID chat;
        Telegram: Saved Messages).  No self-chat address on the
        adapter → no delivery; the journal entry already
        recorded the detection."""
        gateway = getattr(self, "gateway", None)
        if gateway is None:
            return
        ref = None
        adapter = self._adapter_for(platform)
        if platform == "whatsapp":
            jid = str(getattr(adapter, "user_jid", "") or "")
            if jid:
                from ...social.chat.base import ChatKind, ChatRef

                ref = ChatRef(platform="whatsapp", chat_id=jid,
                              kind=ChatKind.DM, title="you",
                              peer=jid)
        elif platform == "telegram":
            saved = getattr(adapter, "saved_messages_ref", None)
            if callable(saved):
                ref = saved()
        if ref is None:
            return
        result = gateway.send(platform, ref, line)
        if not getattr(result, "ok", False):
            record_failure(
                "runtime_provision.antitag_deliver",
                RuntimeError(str(getattr(result, "error", "")
                                 or "self-chat send refused")))

    def _pair_toggle_live(self, rec: Any, *, svc: Any = None,
                          sup: Any = None) -> dict[str, str]:
        """The /pair live verdicts for one account: per ON toggle,
        whether the running process is actually honoring it —
        computed from the legs this runtime can see (registry stop
        intent, supervisor process, the gateway's attached adapter
        for THIS account and its connection, the organ's presence
        truth).  Set-state alone cannot tell "off" from "broken";
        this is the display half of that fix.  Never raises — a
        verdict helper must not kill the /pair surface."""
        from ...social.pairing import toggle_live_verdicts

        try:
            if sup is None:
                sup = self._pairing_supervisor()
            stopped = bool(sup.registry.owner_stopped(rec.platform))
            running = False
            try:
                running = bool(sup.is_running(rec.id))
            except Exception:  # noqa: BLE001 - one leg, not the call
                running = False
            attached: bool | None = None
            connected: bool | None = None
            gateway = getattr(self, "gateway", None)
            if gateway is not None:
                if rec.platform == "whatsapp":
                    adapter = gateway.adapters.get("whatsapp")
                    attached = bool(
                        adapter is not None
                        and int(getattr(adapter, "port", 0) or 0)
                        == int(rec.port or 0))
                    if attached:
                        event = getattr(adapter, "connected", None)
                        connected = bool(
                            event is not None and event.is_set())
                elif rec.platform == "telegram":
                    attached = gateway.adapters.get(
                        "telegram") is not None
                    connected = attached
            presence: str | None = None
            if svc is not None:
                try:
                    presence = svc["personal"].presence_state(
                        rec.id)
                except Exception:  # noqa: BLE001 - one leg only
                    presence = None
            return toggle_live_verdicts(
                rec, stopped=stopped, running=running,
                attached=attached, connected=connected,
                presence_state=presence)
        except Exception as exc:  # noqa: BLE001 - display helper
            record_failure(
                "runtime_provision.pair_toggle_live", exc)
            return {}

    def _control_pair(self, tail: str, *, chat_key: str = "",
                      message: Any = None) -> str:
        """/pair — the pairing-with-code account pool (§15).

        /pair                              pool summary + accounts
        /pair whatsapp number [label]    pair a WhatsApp account:
                                           returns the 8-char code to
                                           type into Linked Devices
        /pair telegram number [label]    pair a Telegram account:
                                           Telegram sends IT a login
                                           code; answer with tgcode
        /pair tgcode id code           complete a Telegram pairing
        /pair tgpass id password       its 2FA password, when asked
        /pair code id                    re-show a live code
        /pair start|stop|restart id      supervise that session
        /pair remove id                  forget the account
        /pair use id                     name a platform's default
                                           account (routing, §15)
        /pair route [platform]             show routing resolution
        /pair toggle id name <on|off>  personal surfaces per
                                           account: status_saving |
                                           view_once | antidelete |
                                           always_online |
                                           status_viewing |
                                           status_react | anticall |
                                           antitag | antiedit
        /pair saved [id]                   saved statuses/view-once
        /pair recovered [id]               DM delete/edit recoveries
        /pair page                         the pairing page (same
                                           state, in a browser)

        Owner-only (not a public command): pairing an account in is
        the owner's explicit act, and the pool is his accounts.
        """
        from ...social.pairing import PairingStoreError, \
            PairingUnavailable

        try:
            sup = self._pairing_supervisor()
        except PairingStoreError as exc:
            return f"⚠️ {exc}"
        parts = (tail or "").split()
        if not parts or parts[0].lower() in ("list", "status"):
            counts = sup.counts()
            lines = [f"📟 paired accounts — Registered: "
                     f"{counts['registered']} · Connected: "
                     f"{counts['connected']} · Dropped: "
                     f"{counts.get('dropped', 0)}"]
            try:
                pair_bundle = self._pairing_personal()
                pairer = pair_bundle["pairer"]
            except Exception:  # noqa: BLE001 - summary never dies
                pair_bundle = None
                pairer = None       # on the Telegram bundle
            for row in sup.status_all():
                label = f" “{row['label']}”" if row.get("label") else ""
                extra = ""
                rec = sup.registry.get(row["id"])
                if rec is not None:
                    if rec.primary:
                        extra += " · primary"
                    from ...social.pairing import PERSONAL_TOGGLES

                    on = [name for name in PERSONAL_TOGGLES
                          if rec.toggle(name)]
                    if on:
                        extra += f" · surfaces: {', '.join(on)}"
                if pairer is not None and row["platform"] == "telegram":
                    pending = pairer.pending(row["id"])
                    if pending:
                        extra += (f" · ⏳ login code sent "
                                  f"({pending['state']}, "
                                  f"{int(pending['expires_in_s'])}s "
                                  "left)")
                if row.get("last_error"):
                    extra += f" · ⚠️ {row['last_error']}"
                lines.append(
                    f"  {row['id']}{label} — {row['status']}"
                    f" (bridge {row['process']}){extra}")
                if rec is not None:
                    # Set-on-but-not-live must be VISIBLE, with
                    # its reason — the owner's 2026-11-07 trap was
                    # toggles on, surface dead, display silent.
                    verdicts = self._pair_toggle_live(
                        rec, svc=pair_bundle, sup=sup)
                    if verdicts:
                        from ...social.pairing import PERSONAL_TOGGLES as _TOGGLES

                        lines.append("    live: " + " · ".join(
                            f"{name} {verdicts[name]}"
                            for name in _TOGGLES
                            if name in verdicts))
            # Paired-and-linked but surfaceless is a state the owner
            # must be able to SEE: the platform config gates the
            # conversational adapter, never the pairing itself.
            _ctx = getattr(self, "context", None)
            chat_cfg = getattr(
                getattr(_ctx, "settings", None), "chat", None)
            if chat_cfg is not None and not getattr(
                    chat_cfg, "whatsapp_enabled", True) and any(
                    r.platform == "whatsapp" and r.status == "linked"
                    for r in sup.registry.all()):
                lines.append(
                    "⚠️ WhatsApp is paired and linked, but the "
                    "WhatsApp platform is disabled in config "
                    "(chat.whatsapp_enabled=false) — paired accounts "
                    "have no live chat surface until it's enabled.")
            lines.append("pair one: /pair whatsapp number [label] "
                         "· /pair telegram number [label]")
            lines.append(
                "personal surfaces on these accounts: "
                "/statusreact [on|off|emoji] sets the status "
                "reaction · /react and /forward act on messages "
                "as you · /viewonce (or /vv) shows a captured "
                "view-once · /poststatus posts as you · "
                "/chanstory posts to a channel you admin · "
                "/joinlink joins a WhatsApp group from its "
                "invite link · /newgroup and /newchannel "
                "create Telegram chats as you · /leave leaves "
                "a group or channel · "
                "/pair saved and /pair recovered list what the "
                "surfaces kept — all of them answer in your "
                "private chats (personal bot, userbot DM, "
                "WhatsApp self-chat), never on the public bot.")
            return "\n".join(lines)
        sub = parts[0].lower()
        if sub == "whatsapp":
            if len(parts) < 2:
                return ("usage: /pair whatsapp number [label] — the "
                        "number in international format, e.g. "
                        "2348012345678")
            label = " ".join(parts[2:])
            try:
                rec = sup.pair_whatsapp(parts[1], label)
            except (PairingUnavailable, ValueError) as exc:
                # PairingUnavailable's text is plain guidance
                # (restore the bridge program, re-run) — a normal
                # reply, never the runtime's "blew up" traceback
                # digest.
                return f"⚠️ {exc}"
            # A fresh pair is an owner start: any earlier stop
            # order for the platform is revoked.
            self._set_platform_owner_stopped("whatsapp", False)
            # The bridge is launched: attach the conversational
            # surface to it NOW, so the moment the link completes
            # the account is live in this process (live-attach).
            # A surface failure rides THIS reply — never silent.
            surface_note = self._sync_whatsapp_surface()
            import time as _time

            for _ in range(50):  # up to ~25s for WhatsApp to issue it
                got = sup.pairing_code(rec.id)
                if got:
                    code, _age = got
                    reply = (
                        f"📟 pairing code for +{rec.identity}: "
                        f"**{code}**\nOn that phone: WhatsApp → "
                        "Linked Devices → Link a device → “Link with "
                        "phone number instead” → enter the code. It "
                        "expires in a few minutes — once linked, the "
                        "account is a live Devon session.")
                    if surface_note:
                        reply += f"\n⚠️ {surface_note}"
                    return reply
                reason = sup.pairing_error(rec.id)
                if reason:
                    # The bridge died or WhatsApp refused during
                    # the wait — that IS the answer, not "hasn't
                    # issued the code yet".
                    return f"⚠️ {reason}"
                _time.sleep(0.5)
            reply = (f"bridge started for +{rec.identity} "
                     f"({rec.id}) — WhatsApp hasn't issued the code "
                     f"yet. /pair code {rec.id} in a few seconds.")
            if surface_note:
                reply += f"\n⚠️ {surface_note}"
            return reply
        if sub == "code":
            rec = self._pair_find(sup, parts[1] if len(parts) > 1 else "")
            if rec is None:
                return "which account? /pair lists them."
            got = sup.pairing_code(rec.id)
            if not got:
                reason = sup.pairing_error(rec.id)
                if reason:
                    return (f"⚠️ {reason}\n/pair whatsapp "
                            f"{rec.identity} issues a fresh one.")
                return (f"no live code for {rec.id} — codes expire in "
                        f"minutes. /pair whatsapp {rec.identity} "
                        "issues a fresh one.")
            code, age = got
            if age >= 300:
                # Codes die in minutes — a five-minute-old code is
                # a corpse, and saying "type it in" would send the
                # owner to a guaranteed "couldn't link" wall.
                return (f"📟 {rec.id}: **{code}** — issued "
                        f"{int(age // 60)} minutes ago. WhatsApp "
                        "codes die within a few minutes, so this "
                        "one is almost certainly dead. /pair "
                        f"whatsapp {rec.identity} issues a fresh "
                        "one — type THAT one in right away.")
            if age >= 120:
                return (f"📟 {rec.id}: **{code}** "
                        f"({int(age)}s old — codes die within a "
                        "few minutes, so type it in NOW, or /pair "
                        f"whatsapp {rec.identity} for a fresh one)")
            return (f"📟 {rec.id}: **{code}** "
                    f"({int(age)}s old — type it into Linked Devices)")
        if sub == "telegram":
            if len(parts) < 2:
                return ("usage: /pair telegram number [label] — the "
                        "number in international format; Telegram "
                        "sends that account a login code, and you "
                        "answer it here with /pair tgcode")
            label = " ".join(parts[2:])
            try:
                svc = self._pairing_personal()
                rec = svc["pairer"].start(parts[1], label)
            except Exception as exc:  # noqa: BLE001 - the flow's own
                return f"⚠️ {exc}"     # words are the honest answer
            return (f"📟 login code sent to the Telegram account "
                    f"+{rec.identity} ({rec.id}). When the code "
                    f"arrives (Telegram service message or SMS), "
                    f"send it back like this:\n/pair tgcode {rec.id} the-code-from-telegram\n"
                    "The flow lives ten minutes; the code itself is "
                    "never stored anywhere.")
        if sub in ("tgcode", "tgpass"):
            rec = self._pair_find(sup, parts[1] if len(parts) > 1
                                  else "")
            if rec is None:
                return "which account? /pair lists them."
            if len(parts) < 3:
                return (f"usage: /pair {sub} {rec.id} "
                        f"{'the code' if sub == 'tgcode' else 'the password'}")
            try:
                svc = self._pairing_personal()
                if sub == "tgcode":
                    out = svc["pairer"].confirm_code(
                        rec.id, parts[2])
                else:
                    # The 2FA password rides this one message into
                    # Telethon's sign-in and nowhere else — never
                    # stored, never logged, never journaled.
                    out = svc["pairer"].confirm_password(
                        rec.id, " ".join(parts[2:]))
            except Exception as exc:  # noqa: BLE001
                return f"⚠️ {exc}"
            if out.get("status") == "password_needed":
                return (f"🔐 {rec.id} has two-factor auth on. Send "
                        f"the password:\n/pair tgpass {rec.id} "
                        "your-password\nIt goes straight to the sign-in "
                        "and is not kept.")
            return (f"✅ {rec.id} is linked — its session file is "
                    "saved under its own name and the account is a "
                    "live member of the pool.")
        if sub in ("start", "stop", "restart", "remove"):
            rec = self._pair_find(sup, parts[1] if len(parts) > 1
                                  else "")
            if rec is None:
                return "which account? /pair lists them."
            if rec.platform == "telegram":
                if sub == "remove":
                    self._set_platform_owner_stopped(
                        rec.platform, False)
                    sup.remove(rec.id)
                    return (f"🗑️ {rec.id} removed from the pool. "
                            "Its session file stays on disk; the "
                            "number can be paired again any time.")
                if sub == "stop":
                    sup.registry.set_status(rec.id, "stopped")
                    return (f"⏸️ {rec.id} marked stopped — Devon "
                            "will not treat it as connected. Its "
                            "session file is untouched.")
                # start/restart on Telegram = an authorization probe:
                # there is no process to launch; the session file
                # either still authorizes or it does not.
                try:
                    state = self._pairing_personal()["pairer"].probe(
                        rec)
                except Exception as exc:  # noqa: BLE001
                    return f"⚠️ probe failed for {rec.id}: {exc}"
                if state == "linked":
                    sup.registry.set_status(rec.id, "linked")
                    return f"▶️ {rec.id} — session authorized, live."
                sup.registry.set_status(rec.id, "down")
                return (f"⚠️ {rec.id} — the saved session no longer "
                        "authorizes. Re-pair it: /pair telegram "
                        f"{rec.identity}")
            if sub == "start":
                self._set_platform_owner_stopped("whatsapp", False)
                try:
                    sup.start(rec.id)
                except PairingUnavailable as exc:
                    return f"⚠️ {exc}"
                surface_note = self._sync_whatsapp_surface()
                note = self._pair_launch_note(sup, rec)
                if note:
                    return note
                if surface_note:
                    return (f"▶️ bridge starting for {rec.id}."
                            f"\n⚠️ {surface_note}")
                return f"▶️ bridge starting for {rec.id}."
            if sub == "stop":
                # An account-level stop is an owner stop of the
                # surface too: record the intent so the reconcile
                # never resurrects it behind the owner's back.
                self._set_platform_owner_stopped("whatsapp", True)
                sup.stop(rec.id)
                self._sync_whatsapp_surface()
                return f"⏸️ bridge stopped for {rec.id}."
            if sub == "restart":
                self._set_platform_owner_stopped("whatsapp", False)
                sup.stop(rec.id)
                try:
                    sup.start(rec.id)
                except PairingUnavailable as exc:
                    return f"⚠️ {exc}"
                surface_note = self._sync_whatsapp_surface()
                note = self._pair_launch_note(sup, rec)
                if note:
                    return note
                if surface_note:
                    return (f"🔁 bridge restarted for {rec.id}."
                            f"\n⚠️ {surface_note}")
                return f"🔁 bridge restarted for {rec.id}."
            # Removal retires the account AND any stop order that
            # named its platform — a future pair starts clean.
            self._set_platform_owner_stopped(rec.platform, False)
            sup.remove(rec.id)
            self._sync_whatsapp_surface()
            return (f"🗑️ {rec.id} removed from the pool. Its saved "
                    "session stays on disk; the number can be paired "
                    "again any time.")
        if sub == "use":
            rec = self._pair_find(sup, parts[1] if len(parts) > 1
                                  else "")
            if rec is None:
                return "which account? /pair lists them."
            sup.registry.set_primary(rec.id)
            surface_note = ""
            if rec.platform == "whatsapp":
                # The conversational surface serves the resolved
                # (primary-first) account — follow the switch.
                surface_note = self._sync_whatsapp_surface()
            reply = (f"🎯 {rec.id} is now the default {rec.platform} "
                     "account — surfaces that act as you on "
                     f"{rec.platform} use it unless they name another.")
            if surface_note:
                reply += f"\n⚠️ {surface_note}"
            return reply
        if sub == "route":
            from ...social.pairing import resolve_account

            platforms = [parts[1].lower()] if len(parts) > 1 else \
                ["whatsapp", "telegram"]
            lines = []
            for platform in platforms:
                rec = resolve_account(sup.registry, platform)
                if rec is None:
                    lines.append(f"{platform}: no paired account — "
                                 "surfaces must say so, never borrow "
                                 "another platform's session")
                else:
                    why = ("named primary" if rec.primary else
                           "first connected" if rec.status == "linked"
                           else "first registered")
                    lines.append(f"{platform}: {rec.id} ({why}, "
                                 f"status {rec.status})")
            return "\n".join(lines)
        if sub == "toggle":
            if len(parts) < 4:
                # All nine toggles named, bracket-free: the
                # WhatsApp presentation strips tag-shaped text,
                # so a <a|b|c> list would reach the owner's
                # phone with the names eaten (truth pass,
                # 2026-11-07 — the old line also named only
                # five of the nine, hiding status_react,
                # anticall, antitag and antiedit outright).
                return ("usage: /pair toggle ACCOUNT TOGGLE "
                        "on|off — TOGGLE is one of: "
                        "status_saving, view_once, antidelete, "
                        "always_online, status_viewing, "
                        "status_react, anticall, antitag, "
                        "antiedit")
            name = parts[2].lower()
            on = parts[3].lower() in ("on", "true", "yes", "1")
            try:
                svc = self._pairing_personal()
                rec = self._pair_find(sup, parts[1])
                if rec is None and parts[1].lower().endswith(
                        ":primary"):
                    rec = svc["personal"].ensure_primary(
                        parts[1].split(":")[0])
                if rec is None:
                    return "which account? /pair lists them."
                svc["personal"].set_toggle(rec.id, name, on)
            except ValueError as exc:
                return f"⚠️ {exc}"
            reply = (f"{'✅' if on else '⏸️'} {rec.id}: {name} "
                     f"{'on' if on else 'off'}.")
            if on:
                # "On" is a setting, not a fact about the world:
                # the reply names whether the toggle is LIVE right
                # now, with the reason when it is not — the owner
                # must never have to guess which one he got.
                verdicts = self._pair_toggle_live(
                    rec, svc=svc, sup=sup)
                if name in verdicts:
                    reply += f"\nlive: {verdicts[name]}"
            return reply
        if sub in ("saved", "recovered"):
            from datetime import datetime

            svc = self._pairing_personal()
            vault = svc["personal"].vault
            rec = self._pair_find(sup, parts[1]) if len(parts) > 1 \
                else None
            if sub == "saved":
                rows = vault.saved_entries(rec.id if rec else "")
            else:
                rows = vault.recoveries(rec.id if rec else "")
            if not rows:
                what = "saved" if sub == "saved" else "recovered"
                scope = f" for {rec.id}" if rec else ""
                return (f"nothing {what}{scope} yet — saves land "
                        "when a per-account toggle is on (/pair "
                        "toggle) and the surface sees traffic.")
            lines = []
            for row in rows[-15:]:
                when = datetime.fromtimestamp(
                    float(row.get("ts") or 0)).strftime("%d %b %H:%M")
                if sub == "saved":
                    who = row.get("poster_name") or \
                        row.get("counterpart_id") or "?"
                    lines.append(
                        f"  {when} · {row.get('kind')} · {who} · "
                        f"{row.get('media_kind') or 'text'}"
                        f"{' · ' + str(row.get('path')).split('/')[-1] if row.get('path') else ''}")
                else:
                    mark = "🎯" if row.get("recovered") else "∅"
                    orig = str(row.get("original_text") or "")
                    if len(orig) > 120:
                        orig = orig[:117] + "…"
                    lines.append(
                        f"  {when} · {mark} {row.get('kind')} · "
                        f"{row.get('chat_key', '')} · “{orig}”")
            head = (f"{'💾 saved' if sub == 'saved' else '🕵️ recovered'}"
                    f"{f' — {rec.id}' if rec else ''} "
                    f"(latest {len(lines)} of {len(rows)}):")
            return "\n".join([head] + lines)
        if sub == "page":
            # The web rungs, best first: Devon's own pairing page
            # in PHP (served by Devon on loopback when a PHP
            # runtime resolves or provisions), then the read-only
            # render on the API.  The reply names which rung the
            # owner actually got — never a page that isn't there.
            from ...social import pairing_web as _pairing_web

            try:
                out = _pairing_web.launch_pairing_page(self.context)
            except PairingUnavailable as exc:
                return (f"📟 the PHP pairing page could not be "
                        f"served here: {exc}\nThe read-only render "
                        "is at GET /pair on Devon's API, with your "
                        "owner token — it shows live pairing codes, "
                        "so nothing less than the owner token opens "
                        "it. Approval still only ever happens on "
                        "the phone.")
            verb = "already serving" if out.get("already") \
                else "now serving"
            return (
                f"📟 Devon's pairing page (PHP) is {verb} on "
                f"this machine: {out['url']}\nIt is gated by its "
                f"own page token — yours is: {out['token']}\n"
                "The page drives the same pool as this chat "
                "(pair, start, stop, restart, remove, toggles) "
                "through Devon's API with your owner token. "
                "Approval still only ever happens on the phone.")
        return ("usage: /pair whatsapp number [label] · "
                "/pair telegram number [label] · /pair tgcode "
                "id code · /pair tgpass id password · /pair "
                "code id · /pair start, stop, restart or remove "
                "id · /pair use id · /pair route [platform] · "
                "/pair toggle id name on|off · /pair saved [id] "
                "· /pair recovered [id] · /pair page")

    # ── mass actions (§14: massreact / pollvote / chfollow) ─────────
    def _mass_service(self) -> dict[str, Any]:
        """The §14 mass-action bundle over the §15 pool: arms
        store + run journal + runner, built once, lazily.  The
        handle builder reads the LIVE pool each run (registry
        accounts + gateway sessions), so an account paired
        mid-session joins the next run without a restart."""
        svc = getattr(self, "_mass_svc", None)
        if svc is not None:
            return svc
        from pathlib import Path

        from ...social.mass_actions import (MassArms, MassJournal,
                                            MassRunner,
                                            default_mass_root)

        sup = self._pairing_supervisor()
        ws = Path(getattr(self.context.settings, "workspace_dir", "."))
        root = default_mass_root(ws)
        arms = MassArms(root / "arms.json")
        journal = MassJournal(root / "journal.jsonl")
        svc = {"supervisor": sup, "registry": sup.registry,
               "arms": arms, "journal": journal,
               "runner": MassRunner(arms, journal)}
        self._mass_svc = svc
        return svc

    def _mass_handles(self, platform: str) -> list[Any]:
        """The pool for one platform as mass handles: every
        LINKED registry account (WhatsApp → its own bridge
        actor; Telegram → its own session actor), plus the
        gateway's live session as the platform primary when the
        registry doesn't already carry that same account.
        Bot adapters are never handles — mass tools act as him,
        never as the bot."""
        from ...social.mass_actions import (TG_CAPABILITIES,
                                            WA_CAPABILITIES,
                                            GatewayAdapterActor,
                                            MassAccount,
                                            TelegramSessionActor,
                                            WhatsAppBridgeActor)
        from ...social.pairing import AccountRecord

        svc = self._mass_service()
        registry = svc["registry"]
        chat = getattr(self.context.settings, "chat", None)
        api_id = getattr(chat, "telegram_api_id", "")
        api_hash = getattr(chat, "telegram_api_hash", "")
        adapter = self._adapter_for(platform)
        handles: list[Any] = []
        gateway_covered = False
        me_id = str(getattr(adapter, "_me_id", "") or "") \
            if adapter is not None else ""
        for rec in registry.accounts_for(platform):
            if platform == "whatsapp":
                if rec.port:
                    handles.append(MassAccount(
                        record=rec, capabilities=WA_CAPABILITIES,
                        factory=lambda r=rec: WhatsAppBridgeActor(r)))
                elif adapter is not None:
                    gateway_covered = True
                    handles.append(MassAccount(
                        record=rec, capabilities=WA_CAPABILITIES,
                        factory=lambda a=adapter:
                        GatewayAdapterActor(a, "whatsapp")))
            elif platform == "telegram":
                if rec.session_path:
                    if me_id and rec.remote_id == me_id \
                            and adapter is not None:
                        # This record IS the gateway's own
                        # account, paired twice over — the
                        # gateway session speaks for it; a
                        # second session of the same account
                        # would double-vote.
                        gateway_covered = True
                        handles.append(MassAccount(
                            record=rec,
                            capabilities=TG_CAPABILITIES,
                            factory=lambda a=adapter:
                            GatewayAdapterActor(a, "telegram")))
                        continue
                    handles.append(MassAccount(
                        record=rec, capabilities=TG_CAPABILITIES,
                        factory=lambda r=rec: TelegramSessionActor(
                            r, api_id=api_id, api_hash=api_hash)))
                elif adapter is not None:
                    gateway_covered = True
                    handles.append(MassAccount(
                        record=rec, capabilities=TG_CAPABILITIES,
                        factory=lambda a=adapter:
                        GatewayAdapterActor(a, "telegram")))
        if adapter is not None and not gateway_covered:
            caps = WA_CAPABILITIES if platform == "whatsapp" \
                else TG_CAPABILITIES
            handles.append(MassAccount(
                record=AccountRecord(
                    id=f"{platform}:primary", platform=platform,
                    identity="primary",
                    label="live session (gateway)", status="linked"),
                capabilities=caps,
                factory=lambda a=adapter: GatewayAdapterActor(
                    a, platform)))
        handles.sort(key=lambda h: (
            0 if (h.record.primary
                  or h.record.id.endswith(":primary")) else 1,
            h.record.created_at))
        return handles

    def _control_mass(self, tool: str, tail: str) -> str:
        """/massreact · /pollvote · /chfollow — the §14 mass
        tools over the owner's own paired accounts.

        /massreact <post link> | emoji [| count] [| live|dry]
        /pollvote <poll link> | <option> [| count] [| live|dry]
        /chfollow <channel link> [| count] [| live|dry]
        /<tool> arm|disarm|status|runs|pacing <seconds>

        Owner-only (control dispatch).  A live run BLOCKS this
        chat's turn for its paced duration — the pacing is the
        safety control, and the reply when it lands IS the run
        report (the design's deliverable shape)."""
        from ...social.mass_actions import (WARN_TEXT, MassRefusal,
                                            parse_target, render_run)

        svc = self._mass_service()
        arms, journal, runner = svc["arms"], svc["journal"], \
            svc["runner"]
        raw = str(tail or "").strip()
        segments = [s.strip() for s in raw.split("|")]
        head = segments[0] if segments else ""
        words = head.split()
        meta = words[0].lower() if words else ""

        if meta == "arm":
            arms.arm(tool)
            return (f"{WARN_TEXT}\n\n✅ /{tool} is ARMED. "
                    "Live runs are possible now; the tool's "
                    "first run still dry-runs unless you say "
                    "“| live”. Disarm any time: "
                    f"/{tool} disarm")
        if meta == "disarm":
            arms.disarm(tool)
            return (f"⏸️ /{tool} disarmed — live runs refuse "
                    "until you arm it again.")
        if meta == "pacing":
            if len(words) < 2:
                return (f"/{tool} pacing is "
                        f"{arms.pacing(tool):.0f}s between "
                        "accounts. Raise it: "
                        f"/{tool} pacing <seconds>")
            try:
                value = arms.set_pacing(tool, float(words[1]))
            except (MassRefusal, ValueError) as exc:
                return f"⚠️ {exc}"
            return f"⏱️ /{tool} pacing set to {value:.0f}s."
        if meta == "status":
            st = arms.state(tool)
            rows = svc["registry"].all()
            linked = sum(1 for r in rows if r.status == "linked")
            return "\n".join([
                f"🎛️ /{tool}: "
                f"{'ARMED' if st.get('armed') else 'disarmed'} · "
                f"pacing {arms.pacing(tool):.0f}s · "
                f"{int(st.get('live_runs') or 0)} live run(s) so "
                "far",
                f"pool: {len(rows)} registered · {linked} "
                "connected (mass reach = connected accounts on "
                "the target's platform)",
                WARN_TEXT if not st.get("armed") else
                "armed — the platform-risk warning was shown "
                "at arm time and stands."])
        if meta == "runs":
            rows = journal.tail(5, tool=tool)
            if not rows:
                return f"no /{tool} runs journaled yet."
            lines = [f"📓 recent /{tool} runs:"]
            for row in rows:
                accounts = row.get("accounts") or []
                ok = sum(1 for a in accounts if a.get("ok"))
                kind = row.get("kind", "run")
                mode = "dry" if row.get("dry_run") else "live"
                lines.append(
                    f"  {row.get('run_id', '?')} · {kind}/{mode} "
                    f"· {ok}/{len(accounts)} ok · "
                    f"{row.get('target', '')}")
            return "\n".join(lines)
        if not head:
            return (f"usage: /{tool} <link>"
                    + (" | emoji" if tool == "massreact"
                       else " | <option>" if tool == "pollvote"
                       else "")
                    + " [| count] [| live|dry] — or "
                    f"/{tool} arm|status|runs|pacing <s>")

        # ── a run ──
        try:
            target, riding_count = parse_target(words[0])
        except MassRefusal as exc:
            return f"⚠️ {exc}"
        payload = ""
        count: int | None = riding_count or None
        dry_run: bool | None = None
        for seg in segments[1:]:
            low = seg.lower()
            if low in ("live",):
                dry_run = False
            elif low in ("dry", "dry-run", "dryrun"):
                dry_run = True
            elif seg.isdigit():
                count = int(seg)
            elif seg and not payload:
                payload = seg
            elif seg:
                payload = f"{payload} {seg}"
        if tool == "massreact" and not payload:
            return ("⚠️ massreact needs its emoji: "
                    "/massreact <post link> | emoji [| count] "
                    "[| live]")
        if tool == "pollvote" and not payload:
            return ("⚠️ pollvote needs its option: "
                    "/pollvote <poll link> | <option text or "
                    "number> [| count] [| live]")
        try:
            run = runner.run(
                tool, target,
                accounts=self._mass_handles(target.platform),
                emoji=payload if tool == "massreact" else "",
                option_spec=payload if tool == "pollvote" else "",
                count=count, dry_run=dry_run)
        except MassRefusal as exc:
            return f"⚠️ {exc}"
        return render_run(run, tool_label=f"/{tool}")

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
                    + " — say /browser watch name.")
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
