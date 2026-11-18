"""RuntimeReachMixin: PartnerRuntime command group (reachability wave).

Commands added here expose capabilities that already existed in code but
had no chat surface: temp mail / temp SMS (per-user inboxes), the
browser tool, paper trading, the people spine, the commitment ledger,
and focus sessions.

Per-user scoping (temp mail / temp SMS): an inbox is stored under the
requester's identity (``chat_key`` + platform ``sender_id``), and a
read from any other identity is refused. A temp inbox created in one
chat can never be read from another chat or by another user.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ...core.logging_setup import get_logger
from .failures import record_failure

_log = get_logger(__name__)

#: Temp-mail inboxes live at most a day; temp numbers a couple of hours.
#: KV TTL expiry makes a stale inbox read as "none" — the honest answer
#: is "run /tempmail new", never a resurrected address.
_TEMP_MAIL_TTL_S = 86400.0
_TEMP_SMS_TTL_S = 7200.0
#: Chat replies stay bounded — a full email body is truncated, never dumped.
_MAX_BODY_CHARS = 2000


def _tempmail_dies_in(expires_at: float) -> str:
    """Human countdown to an expires_at: 'in 9m 12s' / '9m ago'."""
    delta = expires_at - time.time()
    past = delta < 0
    total = max(0, int(abs(delta)))
    mins, secs = divmod(total, 60)
    if mins >= 60:
        hours, mins = divmod(mins, 60)
        text = f"{hours}h {mins}m"
    elif mins:
        text = f"{mins}m {secs}s"
    else:
        text = f"{secs}s"
    return f"{text} ago" if past else f"in {text}"


def _own_domain_live_expires(address: str) -> float | None:
    """Live expires_at for an own-domain address from the store, or
    None when the mailbox is gone or the rung is unconfigured. The
    store is authoritative — the per-user record can outlive the
    mailbox (its KV TTL is 24h), so a reaper-deleted inbox must read
    as gone here, never as alive. Never raises."""
    try:
        from ...accounts import temp_mail as _tm

        box = _tm.get_provider("owndomain")._store_for().mailbox(address)
    except Exception:  # noqa: BLE001 - store failure reads as gone
        return None
    return float(box["expires_at"]) if box else None


def _attach_approval_receipt(result: dict[str, Any],
                             approver: Any) -> None:
    """Attach a run's consequential verdicts to its result IN PLACE.

    The receipt names every consequential verdict, and a stop at an
    approval says WHY — owner denied / no answer / no channel —
    never a bare denial. Shared by the model-driven runner and the
    recipe-replay runner: both doors report the same way.
    """
    outcomes = getattr(approver, "outcomes", None) or []
    if not outcomes:
        return
    result["consequential_approvals"] = list(outcomes)
    last = outcomes[-1]
    if result.get("stopped_reason") == "approval denied / no approver":
        verdict_lines = {
            "denied": "approval denied by owner",
            "timeout": "approval timed out (no owner answer)",
            "no_channel": "approval impossible (no owner channel)",
        }
        line = verdict_lines.get(str(last.get("verdict", "")))
        if line:
            result["stopped_reason"] = (
                f"{line}: {last.get('kind', '')} on "
                f"{last.get('url', '')} — run stopped")


class RuntimeReachMixin:
    """RuntimeReachMixin for :class:`PartnerRuntime`."""

    # ── per-user temp-inbox store ──────────────────────────────────────
    def _temp_kv(self) -> Any:
        """KVStore namespaced to temp inboxes, or None when no DB."""
        db = getattr(self.context, "db", None)
        if db is None:
            return None
        from ...storage.kv import KVStore

        return KVStore(db).namespaced("temp_inbox")

    @staticmethod
    def _temp_owner(message: Any, chat_key: str) -> str:
        """Identity a temp inbox belongs to.

        ``message=None`` means an internal call, which runs as the
        owner. Otherwise the platform sender id disambiguates users
        sharing one group chat; the chat key is the fallback when the
        adapter did not supply a sender id.
        """
        if message is None:
            return "owner:internal"
        sid = (getattr(message, "sender_id", "") or "").strip()
        if sid:
            return f"{chat_key}:{sid}"
        return chat_key

    def _temp_load(self, kind: str, message: Any, chat_key: str) -> Any:
        """Stored temp record for this identity, or None."""
        kv = self._temp_kv()
        if kv is None:
            return None
        rec = kv.get(f"{kind}:{self._temp_owner(message, chat_key)}")
        if not isinstance(rec, dict):
            return None
        return rec

    def _temp_save(self, kind: str, message: Any, chat_key: str,
                   record: dict[str, Any], ttl: float) -> bool:
        kv = self._temp_kv()
        if kv is None:
            return False
        record = dict(record)
        record["owner"] = self._temp_owner(message, chat_key)
        kv.set(f"{kind}:{record['owner']}", record, ttl=ttl)
        return True

    def _temp_forget(self, kind: str, message: Any, chat_key: str) -> bool:
        kv = self._temp_kv()
        if kv is None:
            return False
        try:
            kv.delete(f"{kind}:{self._temp_owner(message, chat_key)}")
        except AttributeError:
            return False
        return True

    # ── temp mail: /tempmail /checkmail ─────────────────────────────────
    def _control_tempmail(self, tail: str, *, chat_key: str = "",
                          message: Any = None) -> str:
        """Temp inbox: /tempmail new | status | extend | forget (bare shows it)."""
        from ...accounts import temp_mail as _tm

        verb = (tail or "").strip().split(None, 1)[0].lower() \
            if (tail or "").strip() else ""
        if verb in ("new", "generate", "create"):
            try:
                addr = _tm.grab_address_cascade()
            except Exception as exc:  # noqa: BLE001 - providers are network
                return f"couldn't get a temp address — all providers failed ({exc})"
            if not self._temp_save("mail", message, chat_key, {
                    "address": addr.address, "login": addr.login,
                    "domain": addr.domain, "provider": addr.provider,
                    "token": addr.token,
                    "expires_at": addr.expires_at}, _TEMP_MAIL_TTL_S):
                return "temp-mail storage is off in this session."
            if addr.provider == "owndomain" and addr.expires_at:
                return (f"temp inbox: {addr.address} (via owndomain — "
                        "your own domain)\nsend mail to it, then "
                        "/checkmail to read it. "
                        f"It dies {_tempmail_dies_in(addr.expires_at)}; "
                        "/tempmail extend adds 10 more · /tempmail "
                        "forget drops it now.")
            return (f"temp inbox: {addr.address} (via {addr.provider})\n"
                    "send mail to it, then /checkmail to read it. "
                    "It expires in 24h; /tempmail forget drops it now.")
        if verb == "forget":
            self._temp_forget("mail", message, chat_key)
            return "temp inbox forgotten."
        if verb == "extend":
            rec = self._temp_load("mail", message, chat_key)
            if rec is None:
                return "no temp inbox here — /tempmail new makes one."
            if str(rec.get("provider", "")) != "owndomain":
                return ("only own-domain inboxes extend — this one is "
                        f"via {rec.get('provider')}.")
            addr = _tm.TempAddress(
                address=str(rec.get("address", "")),
                login=str(rec.get("login", "")),
                domain=str(rec.get("domain", "")),
                provider="owndomain",
                token=str(rec.get("token", "")),
                expires_at=float(rec.get("expires_at") or 0))
            try:
                new_expires = _tm.get_provider("owndomain") \
                    .extend_address(addr)
            except Exception as exc:  # noqa: BLE001
                return f"couldn't extend that inbox ({exc})"
            if not new_expires:
                return ("that inbox is already gone — /tempmail new "
                        "makes a fresh one.")
            rec["expires_at"] = new_expires
            self._temp_save("mail", message, chat_key, rec,
                            _TEMP_MAIL_TTL_S)
            return (f"extended — {addr.address} now dies "
                    f"{_tempmail_dies_in(new_expires)}.")
        if verb in ("status", ""):
            rec = self._temp_load("mail", message, chat_key)
            if rec is None:
                if verb == "status":
                    return "no temp inbox here — /tempmail new makes one."
                return ("usage: /tempmail new — get a temp inbox\n"
                        "       /checkmail — read it\n"
                        "       /tempmail status | extend | forget")
            expires_at = float(rec.get("expires_at") or 0)
            if str(rec.get("provider", "")) == "owndomain":
                live = _own_domain_live_expires(
                    str(rec.get("address", "")))
                if live is None:
                    if expires_at:
                        return ("that inbox is gone — it died "
                                f"{_tempmail_dies_in(expires_at)} · "
                                "/tempmail new makes a fresh one.")
                    return ("that inbox is gone — /tempmail new makes "
                            "a fresh one.")
                expires_at = live
            if expires_at:
                if expires_at <= time.time():
                    return ("that inbox is gone — it died "
                            f"{_tempmail_dies_in(expires_at)} · "
                            "/tempmail new makes a fresh one.")
                return (f"temp inbox: {rec.get('address')} "
                        f"(via {rec.get('provider')})\n"
                        f"dies {_tempmail_dies_in(expires_at)} · "
                        "/checkmail to read it · /tempmail extend adds "
                        "10 minutes · /tempmail new for a fresh one.")
            return (f"temp inbox: {rec.get('address')} "
                    f"(via {rec.get('provider')})\n"
                    "/checkmail to read it · /tempmail new for a fresh one.")
        return ("usage: /tempmail new — get a temp inbox\n"
                "       /checkmail — read it\n"
                "       /tempmail status | extend | forget")

    def _control_checkmail(self, tail: str, *, chat_key: str = "",
                           message: Any = None) -> str:
        """Read the caller's temp inbox: /checkmail [read <n> | code | forget]."""
        from ...accounts import temp_mail as _tm

        rec = self._temp_load("mail", message, chat_key)
        if rec is None:
            return "no temp inbox here (it may have expired) — /tempmail new makes one."
        expires_at = float(rec.get("expires_at") or 0)
        if str(rec.get("provider", "")) == "owndomain":
            live = _own_domain_live_expires(str(rec.get("address", "")))
            if live is None:
                if expires_at:
                    return ("that inbox died "
                            f"{_tempmail_dies_in(expires_at)} — "
                            "/tempmail new makes a fresh one.")
                return ("that inbox is gone — /tempmail new makes a "
                        "fresh one.")
            expires_at = live
        if expires_at and expires_at <= time.time():
            return ("that inbox died "
                    f"{_tempmail_dies_in(expires_at)} — /tempmail new "
                    "makes a fresh one.")
        addr = _tm.TempAddress(
            address=str(rec.get("address", "")),
            login=str(rec.get("login", "")),
            domain=str(rec.get("domain", "")),
            provider=str(rec.get("provider", "")),
            token=str(rec.get("token", "")))
        try:
            prov = _tm.get_provider(addr.provider or "1secmail")
        except ValueError as exc:
            return str(exc)
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""
        if verb == "forget":
            self._temp_forget("mail", message, chat_key)
            return "temp inbox forgotten."
        try:
            msgs = prov.get_messages(addr)
        except Exception as exc:  # noqa: BLE001 - providers are network
            return f"couldn't read the inbox ({exc})"
        if verb == "code":
            for m in msgs[:5]:
                try:
                    full = prov.read_message(addr, m)
                except Exception as exc:  # noqa: BLE001 - one bad read skips
                    record_failure("runtime_reach._control_checkmail", exc)
                    continue
                if full.code:
                    return (f"code: {full.code}\nfrom: {full.sender}\n"
                            f"subject: {full.subject}")
            return "no verification code in the inbox yet."
        if verb in ("read", "show"):
            try:
                idx = int((rest or "1").strip()) - 1
            except ValueError:
                return "usage: /checkmail read <n> — e.g. /checkmail read 1"
            if idx < 0 or idx >= len(msgs):
                return f"only {len(msgs)} message(s) in the inbox."
            try:
                full = prov.read_message(addr, msgs[idx])
            except Exception as exc:  # noqa: BLE001
                return f"couldn't read that message ({exc})"
            body = (full.body_text or "").strip()
            if len(body) > _MAX_BODY_CHARS:
                body = body[:_MAX_BODY_CHARS] + "… [truncated]"
            code_line = f"\ncode: {full.code}" if full.code else ""
            return (f"from: {full.sender}\nsubject: {full.subject}\n"
                    f"date: {full.date}{code_line}\n\n{body or '(empty body)'}")
        if verb in ("", "list"):
            if not msgs:
                return (f"inbox for {addr.address} is empty — "
                        "nothing arrived yet.")
            lines = [f"inbox for {addr.address} ({len(msgs)}):"]
            for i, m in enumerate(msgs, 1):
                lines.append(f"  {i}. {m.sender} — {m.subject} ({m.date})")
            lines.append("read one: /checkmail read <n> · code: /checkmail code")
            return "\n".join(lines)
        return "usage: /checkmail [read <n> | code | forget]"

    # ── temp SMS: /vnumber /checksms ────────────────────────────────────
    def _control_vnumber(self, tail: str, *, chat_key: str = "",
                         message: Any = None) -> str:
        """Temp number: /vnumber [country] | new [country] | status | forget."""
        from ...accounts import temp_sms as _ts

        parts = (tail or "").strip().split()
        verb = parts[0].lower() if parts else ""
        if verb in ("status",):
            rec = self._temp_load("sms", message, chat_key)
            if rec is None:
                return "no temp number here — /vnumber new [country] gets one."
            shown = rec.get("number") or rec.get("masked") or "?"
            return (f"temp number: {shown} "
                    f"(via {rec.get('provider')}, {rec.get('country')})\n"
                    "/checksms to read it · /vnumber new for a fresh one.")
        if verb == "forget":
            self._temp_forget("sms", message, chat_key)
            return "temp number forgotten."
        country = "us"
        if verb in ("new", "get"):
            if len(parts) > 1:
                country = parts[1].lower()
        elif verb:
            country = verb
        else:
            rec = self._temp_load("sms", message, chat_key)
            if rec is not None:
                shown = rec.get("number") or rec.get("masked") or "?"
                return (f"temp number: {shown} "
                        f"(via {rec.get('provider')}, {rec.get('country')})\n"
                        "/checksms to read it · /vnumber new for a fresh one.")
            return ("usage: /vnumber new [country] — get a temp number\n"
                    "       /checksms — read SMS on it\n"
                    "       /vnumber status | forget")
        info = _ts.grab_number_cascade(country=country)
        if info.get("status") != "ok":
            return (f"couldn't get a temp number for {country} "
                    f"({info.get('notes', 'all sources exhausted')})")
        if not self._temp_save("sms", message, chat_key, dict(info),
                               _TEMP_SMS_TTL_S):
            return "temp-number storage is off in this session."
        shown = info.get("number") or info.get("masked") or "?"
        return (f"temp number: {shown} "
                f"(via {info.get('provider')}, {info.get('country')})\n"
                "send SMS to it, then /checksms to read it. "
                "It expires in 2h; /vnumber forget drops it now.")

    def _control_checksms(self, tail: str, *, chat_key: str = "",
                          message: Any = None) -> str:
        """Read SMS on the caller's temp number: /checksms [code | forget]."""
        from ...accounts import temp_sms as _ts

        rec = self._temp_load("sms", message, chat_key)
        if rec is None:
            return ("no temp number here (it may have expired) — "
                    "/vnumber new [country] gets one.")
        try:
            prov = _ts.get_provider(str(rec.get("provider", "simcodes")))
        except ValueError as exc:
            return str(exc)
        num = _ts.TempNumber(
            number=str(rec.get("number", "")),
            masked=str(rec.get("masked", "")),
            country=str(rec.get("country", "us")),
            country_name=str(rec.get("country_name", "")),
            provider=str(rec.get("provider", "simcodes")),
            inbox_id=str(rec.get("inbox_id", "")))
        verb = (tail or "").strip().split(None, 1)[0].lower() \
            if (tail or "").strip() else ""
        if verb == "forget":
            self._temp_forget("sms", message, chat_key)
            return "temp number forgotten."
        try:
            msgs = prov.get_messages(num)
        except Exception as exc:  # noqa: BLE001 - providers are network
            return f"couldn't read the SMS inbox ({exc})"
        if verb == "code":
            for m in msgs:
                if m.code:
                    return f"code: {m.code}\nfrom: {m.sender}\n{m.body[:300]}"
            return "no verification code in the SMS inbox yet."
        if verb in ("", "list"):
            if not msgs:
                return "SMS inbox is empty — nothing arrived yet."
            lines = [f"SMS for {num.number or num.masked} ({len(msgs)}):"]
            for m in msgs[:20]:
                code = f" [code {m.code}]" if m.code else ""
                lines.append(f"  {m.sender}: {m.body[:160]}{code}")
            lines.append("latest code: /checksms code")
            return "\n".join(lines)
        return "usage: /checksms [code | forget]"

    # ── browser: /browser ──────────────────────────────────────────────
    def _control_browser(self, tail: str, *, chat_key: str = "",
                         message: Any = None) -> str:
        """The browser tool in chat: /browser open|text|links|forms <url>.

        ``takeover`` is the owner-only takeover verb:
        ``/browser takeover`` = active takeover sessions;
        ``/browser takeover <url>`` = open a live tab on the URL and run
        the challenge ladder (self-solve first, takeover last resort).
        """
        parts = (tail or "").strip().split(None, 1)
        if parts and parts[0].lower() == "takeover":
            rest = parts[1].strip() if len(parts) > 1 else ""
            return self._control_browser_takeover(
                rest, chat_key=chat_key, message=message)
        if parts and parts[0].lower() == "watch":
            rest = parts[1].strip() if len(parts) > 1 else ""
            return self._control_browser_watch(
                rest, chat_key=chat_key, message=message)
        if parts and parts[0].lower() == "run":
            rest = parts[1].strip() if len(parts) > 1 else ""
            return self._control_browser_run(
                rest, chat_key=chat_key, message=message)
        if parts and parts[0].lower() == "recipe":
            rest = parts[1].strip() if len(parts) > 1 else ""
            return self._control_browser_recipe(
                rest, chat_key=chat_key, message=message)
        if parts and parts[0].lower() in ("approve", "deny"):
            rest = parts[1].strip() if len(parts) > 1 else ""
            return self._control_browser_approval(
                rest, approved=parts[0].lower() == "approve",
                message=message)
        if len(parts) < 2:
            return ("usage: /browser open <url> — fetch + show the text\n"
                    "       /browser text <url> — page text only\n"
                    "       /browser links <url> — outgoing links\n"
                    "       /browser forms <url> — forms + fields\n"
                    "       /browser takeover [url] — takeover sessions "
                    "(owner-only)\n"
                    "       /browser run <url> [goal] — drive a live tab "
                    "with the agent loop (owner-only, background job)\n"
                    "       /browser run status [job] — job progress\n"
                    "       /browser recipe list|show|replay|delete — "
                    "compiled drive recipes (owner-only)\n"
                    "       /browser approve <id> — let a paused run do "
                    "its consequential step (owner-only)\n"
                    "       /browser deny <id> — stop that run at the "
                    "step (owner-only)\n"
                    "       /browser watch [session] — attach a live view "
                    "to a running browser session (owner-only)")
        verb, url = parts[0].lower(), parts[1].strip()
        if verb not in ("open", "text", "links", "forms", "headings"):
            return ("usage: /browser open|text|links|forms|headings <url>\n"
                    "       /browser takeover [url] — takeover sessions "
                    "(owner-only)\n"
                    "       /browser run <url> [goal] — agent loop on a "
                    "live tab (owner-only, background job)\n"
                    "       /browser run status [job] — job progress\n"
                    "       /browser approve|deny <id> — answer a paused "
                    "run's consequential step (owner-only)")
        if not url.startswith(("http://", "https://")):
            return "give a full URL starting with http:// or https://"
        from ...tools.browser import BrowserSession
        try:
            # A command argument is the owner directing the
            # open himself — robots.txt governs autonomous
            # crawling, never this.
            sess = BrowserSession(origin="owner")
            opened = sess.open(url)
            if verb == "links":
                res = sess.links()
                out = [f"links on {opened.get('url')} ({len(res.get('links', []))}):"]
                for link in res.get("links", [])[:30]:
                    out.append(f"  {link.get('text', '')} → {link.get('url', '')}")
                return "\n".join(out) or "no links on that page."
            if verb == "forms":
                res = sess.extract(kind="forms")
                forms = res.get("forms", [])
                if not forms:
                    return "no forms on that page."
                out = [f"forms on {opened.get('url')} ({len(forms)}):"]
                for form in forms:
                    fields = ", ".join(
                        f"{f.get('name', '?')}({f.get('type', '?')})"
                        for f in form.get("fields", []))
                    out.append(f"  {form.get('method', '?')} "
                               f"{form.get('action', '')} [{fields}]")
                return "\n".join(out)
            if verb == "headings":
                res = sess.extract(kind="headings")
                heads = res.get("items", [])
                if not heads:
                    return "no headings on that page."
                out = [f"outline of {opened.get('url')}:"]
                for h in heads[:40]:
                    out.append(f"  {'#' * int(h.get('level', 1))} {h.get('text', '')}")
                return "\n".join(out)
            res = sess.text(max_chars=_MAX_BODY_CHARS)
            body = (res.get("text") or "").strip()
            title = res.get("title") or opened.get("title") or url
            if verb == "open":
                head = (f"{title}\n{opened.get('url')} "
                        f"(HTTP {opened.get('status')}, "
                        f"{opened.get('links', 0)} links, "
                        f"{opened.get('forms', 0)} forms)")
                return head + "\n\n" + (body or "(page has no readable text)")
            return body or "(page has no readable text)"
        except Exception as exc:  # noqa: BLE001 - network + parse errors
            return f"browser failed: {exc}"[:500]

    # ── browser takeover: the Telegram seam ─────────────────────────────
    # The takeover library (browser/takeover.py + challenge.py) can issue
    # grants and format escalation messages, but it cannot reach the
    # owner by itself — its notifier is injected. These three helpers
    # close that seam: one persistent manager, one browser service, and
    # a notifier bound to the real partner-runtime Telegram sender.
    def _takeover_notifier(self) -> Any:
        """``(message, url) -> None`` sending escalations to the owner.

        Owner-only by construction: :class:`Notifier` resolves delivery
        targets from ``partner.owner_chats`` alone — a non-owner chat
        can never be a target. ``force=True`` bypasses dedupe because
        every wall escalation carries a FRESH one-time link; deduping a
        second escalation would strand the owner with the first grant's
        (expired) link. Proactive gates still apply for non-proactive
        kinds (none here), and per-chat proactive-off is honored.
        Never raises — a failed send must not break solving.
        """
        def _notify(message: str, url: str) -> None:
            try:
                from ..notifier import Notifier

                Notifier(self.context, self.gateway).publish(
                    kind="browser-takeover",
                    title="🧱 Browser wall — your turn",
                    body=message,
                    force=True,
                    channels=["telegram"],
                    alert_hints={"actionable": True,
                                 "subject": "browser wall"},
                )
            except Exception:  # noqa: BLE001 - notify failure is reported
                _log.warning("takeover escalation notify failed",
                             exc_info=True)

        return _notify

    def _takeover_manager(self) -> Any:
        """The ONE persistent takeover manager for this runtime.

        Grants must be visible to both the challenge ladder (which
        issues them) and ``/browser takeover`` (which lists them), and
        the stream server must stay up — so the manager is built once
        and cached, never fresh per command.
        """
        mgr = getattr(self, "_takeover_manager_inst", None)
        if mgr is None:
            from ...browser.takeover import TakeoverConfig, TakeoverManager

            cfg = TakeoverConfig.from_settings(self.settings)
            mgr = TakeoverManager(
                cfg,
                data_dir=self.settings.resolve("data/takeover"),
                notifier=self._takeover_notifier())
            self._takeover_manager_inst = mgr
        return mgr

    def _browser_service_for_takeover(self) -> Any:
        """The ONE browser service for takeover tabs (cached)."""
        svc = getattr(self, "_takeover_browser_service", None)
        if svc is None:
            from ...browser.service import BrowserService

            svc = BrowserService(
                data_dir=self.settings.resolve("data/browser-service"))
            self._takeover_browser_service = svc
        return svc

    def _control_browser_takeover(self, tail: str, *, chat_key: str = "",
                                  message: Any = None) -> str:
        """Owner-only: ``/browser takeover [url]``.

        No URL → status of active takeover sessions (metadata only —
        never the token-bearing link). With a URL → open a live tab on
        it and run the challenge ladder: self-solve first, takeover
        LAST. On a wall the escalation (attempt receipt + link) is sent
        to the owner's Telegram AND returned here. The verb never
        blocks waiting for the owner — the grant stays open until the
        owner taps Done (then the wall is verified before resume) or
        it expires.
        """
        if message is None or not self._is_operator(message):
            return "takeover is owner-only — run it from the owner's chat."
        mgr = self._takeover_manager()
        rest = (tail or "").strip()
        if not rest:
            grants = mgr.active_grants()
            if not grants:
                return "no active takeover sessions."
            lines = ["active takeover sessions:"]
            for g in grants:
                exp = f"expires in {g['expires_in_s']}s"
                warn = (" — ⚠️ host-network-only link "
                        "(set NM_TAKEOVER_BASE_URL for a public link)"
                        if g["needs_public_base"] else "")
                lines.append(
                    f"  {g['session_id']} — {g['task_label']} "
                    f"({g['signal_kind']}, {exp}){warn}")
            return "\n".join(lines)
        url = rest.split()[0]
        if not url.startswith(("http://", "https://")):
            return "give a full URL starting with http:// or https://"
        try:
            service = self._browser_service_for_takeover()
            tab = service.open_rendered_tab("takeover", url)
        except Exception as exc:  # noqa: BLE001 - playwright/config errors
            return f"takeover failed: could not open a live tab: {exc}"[:500]
        try:
            from ...browser.challenge import ChallengeSolver
            from ...browser.challenge_memory import ChallengeMemory
            from ...browser.service import RenderedTabTakeoverAdapter

            memory = ChallengeMemory(
                self.settings.resolve("data/takeover")
                / "challenge-memory.sqlite3")
            solver = ChallengeSolver(
                takeover_manager=mgr, notifier=self._takeover_notifier(),
                memory=memory,
                session_store=service.solved_session_store())
            outcome = solver.solve(
                tab=RenderedTabTakeoverAdapter(tab),
                resume_state={"step": 0, "url": url},
                owner_chat=chat_key or "",
                task_label=f"takeover {url}",
                session_id=f"takeover-{int(time.time() * 1000)}")
        except Exception as exc:  # noqa: BLE001 - solver errors are data
            return f"takeover failed: {exc}"[:500]
        if outcome.cleared:
            if outcome.via == "no_challenge":
                return f"no wall on {url} — page is clean."
            return f"wall cleared via {outcome.via} — no takeover needed."
        if outcome.via == "takeover_required":
            return ("wall persists and takeover is unavailable "
                    "(no takeover manager).")
        if outcome.takeover_url:
            lines = [f"takeover open: {outcome.takeover_url}",
                     "I also sent this link to your Telegram — solve the "
                     "challenge in the live view, then tap "
                     "\"Done, take over\"."]
            grant = outcome.takeover_grant
            if grant is not None and getattr(
                    grant, "needs_public_base", False):
                lines.append("⚠️ No public takeover URL is configured "
                             "(NM_TAKEOVER_BASE_URL) — this link works on "
                             "the host network only until a tunnel / "
                             "port-forward is set up.")
            return "\n".join(lines)
        return f"takeover failed: {outcome.message}"[:500]

    def _browser_run_approver(self, *, task_label: str,
                              owner_chat: str) -> Any:
        """The approver a browser run hands its loop.

        Default (owner's standing order, 2026-11-11): she acts on
        consequential steps herself — the ActingApprover approves
        and records each one for the run's receipt. With
        ``partner.browser_require_approval`` set, the approval
        broker parks the run and asks the owner instead.
        """
        partner = getattr(self.settings, "partner", None)
        if bool(getattr(partner, "browser_require_approval", False)):
            return self._browser_approval_broker().approver(
                task_label=task_label, owner_chat=owner_chat)
        from ...browser.approvals import ActingApprover

        return ActingApprover(task_label=task_label,
                              owner_chat=owner_chat)

    def _browser_approval_broker(self) -> Any:
        """The ONE approval broker for browser runs (cached).

        Used only in the opt-in ask-first mode
        (``partner.browser_require_approval``): consequential steps
        ask through this broker — the ask goes to the owner's chat
        via the takeover notifier seam, the run parks, and
        ``/browser approve|deny <id>`` answers it. The park timeout
        matches the takeover wait — silence is a no.
        """
        broker = getattr(self, "_browser_approval_broker_inst", None)
        if broker is None:
            from ...browser.approvals import ApprovalBroker
            from ...browser.takeover import TakeoverConfig

            timeout_s = float(
                TakeoverConfig.from_settings(self.settings)
                .wait_timeout_s)
            broker = ApprovalBroker(
                notifier=self._takeover_notifier(), timeout_s=timeout_s)
            self._browser_approval_broker_inst = broker
        return broker

    def _control_browser_approval(self, tail: str, *, approved: bool,
                                  message: Any = None) -> str:
        """Owner-only: ``/browser approve <id>`` / ``/browser deny <id>``.

        Answers a consequential-step approval a browser run is parked
        on: approve resumes the run and performs the step; deny stops
        it at that step. With no id, lists what is waiting.
        """
        if message is None or not self._is_operator(message):
            return ("browser approvals are owner-only — run it from "
                    "the owner's chat.")
        broker = self._browser_approval_broker()
        rest = (tail or "").strip()
        if not rest:
            pending = broker.pending()
            if not pending:
                return "no browser approvals are waiting on you."
            lines = ["browser approvals waiting on your word:"]
            for req in pending:
                lines.append(
                    f"  {req['request_id']} — {req['kind']} "
                    f"{req['label'] or req['target']} on {req['url']} "
                    f"({req['task_label']})")
            lines.append("/browser approve <id> · /browser deny <id>")
            return "\n".join(lines)
        request_id = rest.split()[0]
        if not broker.resolve(request_id, approved):
            return (f"no waiting approval {request_id} — it may have "
                    "timed out or already been answered.")
        if approved:
            return (f"approved {request_id} — the paused run will do "
                    "that step now.")
        return (f"denied {request_id} — the run stops at that step, "
                "and the step will not happen.")

    def _control_browser_run(self, tail: str, *, chat_key: str = "",
                              message: Any = None) -> str:
        """Owner-only: ``/browser run <url> [goal]`` / ``run status [job]``.

        The production entry point for the wired agent loop: a live
        rendered tab driven by an ``AgentLoop`` whose ``on_blocked``
        is the fully-wired ``ChallengeSolver`` handler (takeover
        manager + owner notifier + challenge memory + solved-session
        store) and whose decider is the real brain (LLM decider,
        heuristic fallback). The brain's vision rung backs the
        perception escalator, so a page the text snapshot cannot
        read is looked at, not given up on. A consequential step
        (submit/post/pay/…) PAUSES the run and asks the owner via
        the approval broker — ``/browser approve|deny <id>`` —
        instead of being denied and killing the run.

        NON-BLOCKING: the command starts the loop as a background job
        and returns immediately with the job id. The loop runs
        detached; the owner gets Telegram updates on wall hit ->
        escalation, cleared -> resumed, blocked -> stopped, and done;
        ``/browser run status <job>`` reports steps, stopped-reason,
        final URL, and the wall outcome (e.g. "cleared via takeover").
        Takeover escalation still pauses the loop for resolution and
        resumes where it left off. A second run for the same URL+goal
        returns the live job instead of a duplicate loop, unless
        ``--new`` starts a fresh one explicitly.
        """
        if message is None or not self._is_operator(message):
            return "browser run is owner-only — run it from the owner's chat."
        rest = (tail or "").strip()
        bits = rest.split(None, 1)
        if bits and bits[0].lower() == "status":
            return self._browser_run_status(
                bits[1].strip() if len(bits) > 1 else "")
        if not rest:
            return ("usage: /browser run <url> [goal] — e.g. /browser run "
                    "https://example.test/ browse\n"
                    "       /browser run status [job] — job progress")
        tokens = rest.split()
        force = False
        if tokens and tokens[-1] in ("--new", "--force"):
            force = True
            tokens.pop()
        if not tokens:
            return "usage: /browser run <url> [goal]"
        url = tokens[0]
        goal = " ".join(tokens[1:]) or "browse"
        if not url.startswith(("http://", "https://")):
            return "give a full URL starting with http:// or https://"

        try:
            job, duplicate = self.start_browser_run(
                url, goal, chat_key=chat_key, force=force)
        except ValueError as exc:
            return str(exc)
        if duplicate:
            return (f"browser run already running: job {job.job_id} — "
                    f"{url} (goal: {goal}). "
                    f"/browser run status {job.job_id} for progress. "
                    "Add --new to start a fresh one.")
        lines = [f"browser run started: job {job.job_id} — {url} "
                 f"(goal: {goal}). "
                 f"/browser run status {job.job_id} for progress. "
                 "I'll ping you on wall / cleared / blocked / done."]
        if getattr(job, "recipe_name", ""):
            lines.append(f"📋 replaying recipe {job.recipe_name} — "
                         "compiled from an earlier drive on this "
                         "site, no model time spent.")
        if getattr(job, "recipe_offer", ""):
            lines.append(str(job.recipe_offer))
        return "\n".join(lines)

    def _control_browser_recipe(self, tail: str, *, chat_key: str = "",
                                message: Any = None) -> str:
        """Owner-only: ``/browser recipe`` — the compiled-drive
        catalogue (navigator A5).

        Recipes are compiled automatically from successful drives.
        ``replay`` starts a job that replays the recipe on a live
        tab: consequential steps still go through the run's
        approver, a stale recipe refuses loudly, and a parameter
        the recipe needs must be supplied as ``key=value`` — it is
        never guessed.
        """
        if message is None or not self._is_operator(message):
            return ("browser recipes are owner-only — run it from "
                    "the owner's chat.")
        store = self._browser_recipe_store()
        if store is None:
            return ("browser recipes need the home database — "
                    "storage is off in this session.")
        parts = (tail or "").strip().split()
        if not parts:
            return ("usage: /browser recipe list — saved recipes\n"
                    "       /browser recipe show <name>\n"
                    "       /browser recipe replay <name> "
                    "[key=value ...]\n"
                    "       /browser recipe delete <name>")
        verb = parts[0].lower()
        if verb == "list":
            recipes = store.list_all()
            if not recipes:
                return ("no browser recipes yet — a successful drive "
                        "compiles one automatically.")
            lines = ["browser recipes:"]
            for rec in recipes:
                flag = (" (stale — " + (rec.last_error or "page changed")
                        [:80] + ")") if rec.stale else ""
                trust = "trusted" if rec.trusted else "untrusted"
                lines.append(f"  {rec.name} v{rec.version} — {rec.goal} "
                             f"({rec.site}, {rec.successes} ok / "
                             f"{rec.failures} fail, {trust}){flag}")
            return "\n".join(lines)
        if verb == "show":
            if len(parts) < 2:
                return "usage: /browser recipe show <name>"
            rec = store.get(parts[1])
            if rec is None:
                return f"no browser recipe {parts[1]!r}."
            lines = [f"recipe {rec.name} v{rec.version} — {rec.goal}",
                     f"site: {rec.site} · origin: {rec.origin}",
                     f"steps: {len(rec.steps)} · successes: "
                     f"{rec.successes} · failures: {rec.failures}",
                     "stale: " + ("yes — " + rec.last_error
                                   if rec.stale else "no"),
                     "trusted: " + ("yes" if rec.trusted else "no")]
            if rec.slots:
                lines.append("parameters: "
                             + ", ".join(sorted(rec.slots)))
            for i, step in enumerate(rec.steps):
                mark = " ⚠️ consequential" if step.consequential else ""
                lines.append(f"  {i + 1}. {step.kind} "
                             f"{step.label or step.target}{mark}")
            return "\n".join(lines)
        if verb == "delete":
            if len(parts) < 2:
                return "usage: /browser recipe delete <name>"
            if store.delete(parts[1]):
                return f"deleted browser recipe {parts[1]!r}."
            return f"no browser recipe {parts[1]!r}."
        if verb == "replay":
            if len(parts) < 2:
                return ("usage: /browser recipe replay <name> "
                        "[key=value ...]")
            name = parts[1]
            rec = store.get(name)
            if rec is None:
                return f"no browser recipe {name!r}."
            params: dict[str, str] = {}
            for token in parts[2:]:
                if "=" not in token:
                    return (f"bad parameter {token!r} — use key=value, "
                            "e.g. /browser recipe replay "
                            f"{name} search=shoes")
                key, value = token.split("=", 1)
                params[key.strip()] = value
            first = rec.steps[0] if rec.steps else None
            url = (first.target if first is not None
                   and first.kind == "goto" and first.target
                   else rec.url_pattern or rec.site)
            if not str(url).startswith(("http://", "https://")):
                return (f"recipe {name!r} names no start URL — it "
                        "cannot replay.")
            try:
                job, duplicate = self.start_browser_run(
                    str(url), rec.goal, chat_key=chat_key, recipe=name,
                    recipe_params=params)
            except ValueError as exc:
                return str(exc)
            if duplicate:
                return (f"browser run already running: job {job.job_id} "
                        f"— {url} (goal: {rec.goal}).")
            return (f"📋 replaying recipe {name} v{rec.version} — job "
                    f"{job.job_id} ({url}, goal: {rec.goal}). "
                    f"/browser run status {job.job_id} for progress.")
        return ("usage: /browser recipe list|show <name>|replay <name> "
                "[key=value ...]|delete <name>")

    def start_browser_run(self, url: str, goal: str, *,
                          chat_key: str = "",
                          force: bool = False,
                          recipe: str | None = None,
                          recipe_params: dict[str, str] | None = None,
                          use_recipe: bool = True
                          ) -> tuple[Any, bool]:
        """Start ONE wired browser-run job — the single entry every
        door shares: the ``/browser run`` verb and the conversational
        drive tool (coremind's ``drive`` intent) both start runs
        here, so the decider, the vision escalator, the challenge
        ladder and the approver selection (ActingApprover by
        default, the approval broker under
        ``partner.browser_require_approval``) are identical however
        the run was asked for. Everything the loop touches is built
        inside the runner, on the job's own thread.

        Navigator A5 — recipes: ``recipe=<name>`` replays that
        compiled recipe on a live tab instead of running the model
        loop (a stale recipe refuses with ValueError, never a
        silent model-driven fall-through). Otherwise, when a
        matching recipe has earned trust (``TRUSTED_SUCCESSES``
        consecutive replays) the run replays it automatically; a
        matching recipe that has NOT earned trust is OFFERED on the
        job (``job.recipe_offer``) while the run proceeds
        model-driven. ``use_recipe=False`` skips the lookup
        entirely. ``recipe_params`` supplies the recipe's slots.

        Returns ``(job, duplicate)`` — ``duplicate`` is True when a
        live job for the same URL+goal was returned instead.
        """
        store = self._browser_recipe_store()
        chosen: Any = None
        offer = ""
        if recipe:
            if store is None:
                raise ValueError(
                    "recipes need the home database — storage is off "
                    "in this session.")
            chosen = store.get(recipe)
            if chosen is None:
                raise ValueError(
                    f"no browser recipe {recipe!r} — /browser recipe "
                    "list shows what exists.")
            if chosen.stale:
                raise ValueError(
                    f"recipe {chosen.name!r} is stale "
                    f"({chosen.last_error or 'page changed'}) — it "
                    "will not replay. Drive the site model-driven "
                    "and a fresh run will re-compile it, or delete it.")
        elif use_recipe and store is not None:
            candidate = store.find_for(url=url, labels=[], goal=goal)
            if candidate is not None:
                if candidate.trusted:
                    chosen = candidate
                else:
                    offer = (f"📋 recipe {candidate.name!r} "
                             f"v{candidate.version} matches this "
                             "site/goal "
                             f"({candidate.successes} successes) but "
                             "has not earned trust yet — /browser "
                             f"recipe replay {candidate.name} to use "
                             "it.")

        def _run_brain() -> Any:
            # The real brain, shared by the decider and the vision
            # escalator; None when unavailable (the heuristic and
            # the text path then cover the run, honestly degraded).
            try:
                from ...llm.brain import brain_for

                return brain_for(self.context)
            except Exception:  # noqa: BLE001 - heuristic covers us
                _log.debug("browser run: brain unavailable; "
                           "heuristic fallback", exc_info=True)
                return None

        def _decider() -> Any:
            # The real brain, up front; the heuristic is the fallback
            # when no LLM is available or it returns nothing usable.
            # The decider is wrapped so a consequential proposal it
            # drops is surfaced to the loop as a Decision: the run's
            # approver then rules on it (see _browser_run_approver)
            # instead of the proposal dying at the step.
            brain = _run_brain()
            if brain is None:
                return None
            try:
                from ...browser.llm_decider import (
                    ApprovalSurfacingDecider,
                    build_llm_decider,
                )

                return ApprovalSurfacingDecider(
                    build_llm_decider(brain, goal=goal))
            except Exception:  # noqa: BLE001 - heuristic covers us
                _log.debug("browser run: LLM decider unavailable; "
                           "heuristic fallback", exc_info=True)
                return None

        def _runner(job: Any) -> dict[str, Any]:
            # Everything the loop touches is built HERE, on the job's
            # own thread: the Playwright tab and the sqlite-backed
            # challenge memory are both thread-bound. Building them
            # on the command thread and using them here would break.
            from ...browser.challenge_memory import ChallengeMemory
            from ...browser.loop_wiring import build_wired_agent_loop

            service = self._browser_service_for_run_job(job.job_id)
            tab = service.open_rendered_tab(
                f"browser-run-{job.job_id}", url)
            memory = ChallengeMemory(
                self.settings.resolve("data/takeover")
                / "challenge-memory.sqlite3")
            approver = self._browser_run_approver(
                task_label=f"browser run {url}",
                owner_chat=chat_key or "")
            loop = build_wired_agent_loop(
                tab=tab, takeover_manager=self._takeover_manager(),
                notifier=self._takeover_notifier(), memory=memory,
                session_store=service.solved_session_store(),
                owner_chat=chat_key or "",
                task_label=f"browser run {url}",
                resume_state={"url": url, "goal": goal},
                max_steps=30, approve=approver,
                decide_llm=_decider(),
                vision_brain=_run_brain(),
                on_progress=job.note_progress,
                on_wall_outcome=job.wall_event,
                cancel_requested=job.cancel_requested)
            result = loop.run(url, goal=goal)
            _attach_approval_receipt(result, approver)
            return result

        if chosen is not None:
            runner = self._replay_runner_for(
                chosen, dict(recipe_params or {}), chat_key=chat_key,
                url=url, goal=goal)
        else:
            runner = _runner
        job, duplicate = self._browser_run_jobs().start(
            url, goal, runner, force=force)
        if not duplicate:
            if chosen is not None:
                job.recipe_name = chosen.name
            if offer:
                job.recipe_offer = offer
        return job, duplicate

    def _replay_runner_for(self, recipe: Any, params: dict[str, str],
                           *, chat_key: str, url: str, goal: str) -> Any:
        """A job runner that replays a compiled recipe on a live tab
        instead of running the model loop. Built on the job's thread
        like the model runner: the service, the tab and the approver
        are all constructed here, and a consequential recipe step
        goes through the SAME approver the loop would hand it to —
        a recipe never smuggles one past the gate."""

        def _replay(job: Any) -> dict[str, Any]:
            from ...browser.loop_wiring import RenderedTabDriver
            from ...browser.recipe_replay import replay_recipe

            service = self._browser_service_for_run_job(job.job_id)
            tab = service.open_rendered_tab(
                f"browser-run-{job.job_id}", url)
            approver = self._browser_run_approver(
                task_label=f"browser recipe {recipe.name} {url}",
                owner_chat=chat_key or "")
            result = replay_recipe(
                recipe, driver=RenderedTabDriver(tab),
                approve=approver, slots=params, goal=goal,
                on_progress=job.note_progress,
                cancel_requested=job.cancel_requested,
                store=self._browser_recipe_store(),
                cookie_estate=service.cookie_estate)
            _attach_approval_receipt(result, approver)
            return result

        return _replay

    def _browser_recipe_store(self) -> Any:
        """The durable recipe store for this runtime (cached; None
        when the home database is off — recipes simply stay off,
        and drives proceed model-driven exactly as before)."""
        if getattr(self, "_browser_recipe_store_tried", False):
            return getattr(self, "_browser_recipe_store_inst", None)
        self._browser_recipe_store_tried = True
        store: Any = None
        db = getattr(getattr(self, "context", None), "db", None)
        if db is not None:
            try:
                from ...browser.recipes import DurableRecipeStore

                store = DurableRecipeStore(db)
            except Exception as exc:  # noqa: BLE001 - recipes never gate
                record_failure("runtime_reach.recipe_store", exc)
                store = None
        self._browser_recipe_store_inst = store
        return store

    def _browser_run_status(self, arg: str) -> str:
        """``/browser run status [job]`` — job progress, no new loop."""
        arg = (arg or "").strip()
        if not arg:
            jobs = self._browser_run_jobs().list_recent(5)
            if not jobs:
                return ("no browser run jobs yet — "
                        "/browser run <url> [goal] starts one.")
            lines = ["browser run jobs (newest first):"]
            for job in jobs:
                lines.append(
                    f"  {job.job_id} [{job.status}] {job.url} — "
                    f"{job.steps} steps, wall: {job.wall_outcome}")
            broker = getattr(self, "_browser_approval_broker_inst",
                             None)
            if broker is not None:
                for req in broker.pending():
                    lines.append(
                        f"  ⏸ waiting on you: {req['request_id']} — "
                        f"{req['kind']} on {req['url']}")
            lines.append("detail: /browser run status <job>")
            return "\n".join(lines)
        return self._browser_run_jobs().status_text(arg)

    def _browser_run_jobs(self) -> Any:
        """The ONE background-job registry for this runtime (cached)."""
        mgr = getattr(self, "_browser_run_jobs_inst", None)
        if mgr is None:
            from ...browser.jobs import BrowserJobManager

            mgr = BrowserJobManager(
                notifier=self._takeover_notifier(),
                db=getattr(getattr(self, "context", None), "db", None),
                recipe_store=self._browser_recipe_store())
            self._browser_run_jobs_inst = mgr
        return mgr

    def _browser_service_for_run_job(self, job_id: str) -> Any:
        """A FRESH browser service per job, built on the job's thread.

        The cached takeover service is NOT reused here: its Playwright
        driver is thread-bound to whichever thread opened the first
        tab, and a background job runs on its own thread. Each job
        also gets its own session name (``browser-run-<job>``), so two
        jobs never share one tab's cookie space. (Navigator A6: the
        COOKIES are shared through the unified estate — domain-scoped,
        so a login earned by one job is live for another job on the
        same site, and never leaks to another site.)
        """
        from ...browser.service import BrowserService

        return BrowserService(
            data_dir=self.settings.resolve("data/browser-service"))

    # ── paper trading: /paper ───────────────────────────────────────────
    def _control_paper(self, tail: str, *, chat_key: str = "") -> str:
        """Paper trading (simulated, never real): /paper start|status|stop|list."""
        from ...tools import trading as _trading

        parts = (tail or "").strip().split()
        verb = parts[0].lower() if parts else ""
        args = parts[1:]
        if verb == "start":
            if not args:
                return "usage: /paper start <symbol> [market] [capital] — e.g. /paper start BTC crypto 10000"
            symbol = args[0].upper()
            market = args[1] if len(args) > 1 else "crypto"
            try:
                capital = float(args[2]) if len(args) > 2 else 10000.0
            except ValueError:
                return "capital must be a number — e.g. /paper start BTC crypto 10000"
            try:
                res = _trading.paper_start(self.context, symbol, market, capital)
            except Exception as exc:  # noqa: BLE001 - engine/data errors
                return f"paper start failed: {exc}"[:500]
            resumed = " (resumed the open session)" if res.get("resumed") else ""
            return (f"paper session {res.get('session_id')} open: "
                    f"{symbol}/{market} with {res.get('capital'):,.2f}{resumed}\n"
                    "/paper status to mark it to market.")
        if verb == "status":
            try:
                if args and args[0].startswith("paper"):
                    res = _trading.paper_status(self.context, session_id=args[0])
                elif args:
                    res = _trading.paper_status(self.context, symbol=args[0].upper())
                else:
                    res = _trading.paper_status(self.context)
            except Exception as exc:  # noqa: BLE001
                return f"paper status failed: {exc}"[:500]
            last = res.get("last_action") or {}
            return (f"paper {res.get('session_id')}: {res.get('symbol')}/"
                    f"{res.get('market')} — equity {res.get('equity'):,.2f} "
                    f"(PnL {res.get('pnl'):+,.2f}, {res.get('pnl_pct', 0) * 100:+.2f}%), "
                    f"cash {res.get('cash'):,.2f}, units {res.get('units')}, "
                    f"orders {res.get('orders')}, "
                    f"bias {last.get('bias')}/{last.get('regime')}")
        if verb == "stop":
            try:
                res = _trading.paper_stop(
                    self.context,
                    session_id=args[0] if args else "")
            except Exception as exc:  # noqa: BLE001
                return f"paper stop failed: {exc}"[:500]
            return f"paper session {res.get('session_id')} closed ({res.get('symbol')})."
        if verb == "list":
            db = getattr(self.context, "db", None)
            if db is None:
                return "paper trading storage is off in this session."
            try:
                rows = db.query(
                    "SELECT id, symbol, market, capital, status FROM "
                    "paper_sessions ORDER BY created_at DESC LIMIT 20")
            except Exception as exc:  # noqa: BLE001 - table missing etc.
                return f"paper list failed: {exc}"[:500]
            if not rows:
                return "no paper sessions yet — /paper start <symbol> opens one."
            lines = ["paper sessions:"]
            for r in rows:
                lines.append(f"  {r['id']} {r['symbol']}/{r['market']} "
                             f"{r['capital']:,.2f} [{r['status']}]")
            return "\n".join(lines)
        return ("usage: /paper start <symbol> [market] [capital] — simulated, never real money\n"
                "       /paper status [session|symbol] — mark to market\n"
                "       /paper stop [session] · /paper list")

    # ── people spine: /people ───────────────────────────────────────────
    def _people_store(self) -> Any:
        """PersonStore on the durable home path (created on demand)."""
        from ...people.store import PersonStore

        path = self.context.settings.home_path / "data" / "people.sqlite"
        return PersonStore(path)

    def _sync_people_from_tracker(self, store: Any = None) -> int:
        """Copy every JSON-tracker contact into the people spine.

        One-shot-able and idempotent: ``migrate_from_relationships``
        inserts only persons not already in the store and appends
        only notes not already present, so re-running (at boot, or
        via ``/people migrate``) is a cheap no-op when nothing
        changed. The tracker is still the live feed — the triage
        hook notes every inbound message into it — so this runs at
        every boot, not once behind a marker: a once-marker would
        freeze the spine at first boot while the tracker keeps
        growing. Returns the number of persons newly inserted.
        """
        from ...people.migrate import migrate_from_relationships
        from ...social.relationships import RelationshipTracker

        tracker = RelationshipTracker(
            path=str(self.context.settings.home_path / "data"
                     / "social_relationships.json"))
        if store is None:
            with self._people_store() as own:
                return migrate_from_relationships(tracker, own)
        return migrate_from_relationships(tracker, store)

    @staticmethod
    def _slug(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
        return slug or ""

    def _control_people(self, tail: str, *, chat_key: str = "") -> str:
        """The people spine: /people list | show <name> | add <name> | note <name> <text> | migrate."""
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""
        try:
            with self._people_store() as store:
                if verb == "list":
                    try:
                        limit = int(rest) if rest.strip() else 10
                    except ValueError:
                        return "usage: /people list [n]"
                    rows = store.top_contacts(limit=max(1, min(limit, 50)))
                    if not rows:
                        return "no people recorded yet — /people add <name> starts the spine."
                    lines = ["people (closest first):"]
                    for r in rows:
                        lines.append(
                            f"  {r.get('display_name') or r.get('person_id')} "
                            f"[closeness {float(r.get('closeness') or 0):.2f}, "
                            f"in {r.get('inbound_count', 0)}/out {r.get('outbound_count', 0)}]")
                    return "\n".join(lines)
                if verb == "add":
                    name = rest.strip()
                    if not name:
                        return "usage: /people add <name>"
                    slug = self._slug(name)
                    if not slug:
                        return "give a name with letters in it."
                    person = store.ensure(slug, name)
                    return (f"added {person.get('display_name')} "
                            f"(id {person.get('person_id')}). "
                            "/people note <name> <text> attaches a note.")
                if verb in ("show", "get"):
                    name = rest.strip()
                    if not name:
                        return "usage: /people show <name>"
                    found = store.search(name)
                    exact = [r for r in found
                             if (r.get("display_name") or "").lower() == name.lower()]
                    if len(exact) == 1:
                        found = exact
                    if not found:
                        return f"no one matching {name!r} — /people add {name} first."
                    if len(found) > 1:
                        lines = [f"{len(found)} matches for {name!r} — be more specific:"]
                        for r in found[:10]:
                            lines.append(f"  {r.get('display_name')} (id {r.get('person_id')})")
                        return "\n".join(lines)
                    person = store.get(found[0]["person_id"])
                    if person is None:
                        return f"no one matching {name!r}."
                    notes = person.get("notes") or []
                    lines = [
                        f"{person.get('display_name')} (id {person.get('person_id')})",
                        f"closeness {float(person.get('closeness') or 0):.2f} · "
                        f"inbound {person.get('inbound_count', 0)} · "
                        f"outbound {person.get('outbound_count', 0)}",
                    ]
                    if notes:
                        lines.append("notes:")
                        lines.extend(f"  - {n}" for n in notes[-10:])
                    return "\n".join(lines)
                if verb == "note":
                    bits = rest.strip().split(None, 1)
                    if len(bits) < 2:
                        return "usage: /people note <name> <text>"
                    name, text = bits
                    found = store.search(name)
                    exact = [r for r in found
                             if (r.get("display_name") or "").lower() == name.lower()]
                    if len(exact) == 1:
                        found = exact
                    if len(found) != 1:
                        if not found:
                            return f"no one matching {name!r} — /people add {name} first."
                        return f"{len(found)} matches for {name!r} — be more specific."
                    store.add_note(found[0]["person_id"], text)
                    return f"noted on {found[0].get('display_name')}: {text[:120]}"
                if verb == "migrate":
                    count = self._sync_people_from_tracker(store)
                    if count == 0:
                        return ("nothing new to migrate — the spine is "
                                "already caught up with the tracker.")
                    noun = "person" if count == 1 else "people"
                    return (f"migrated {count} {noun} from the old "
                            "tracker — /people list shows them now.")
                return ("usage: /people list [n] · /people show <name>\n"
                        "       /people add <name> · /people note <name> <text>\n"
                        "       /people migrate — copy the old tracker into the spine")
        except Exception as exc:  # noqa: BLE001 - sqlite errors surface honestly
            return f"people failed: {exc}"[:500]

    # ── commitment ledger: /commitments ────────────────────────────────
    def _commitments_ledger(self) -> Any:
        """CommitmentLedger on the durable home path."""
        from ...commitments.ledger import CommitmentLedger

        path = self.context.settings.home_path / "data" / "commitments.json"
        return CommitmentLedger(str(path))

    def _world_graph(self) -> Any:
        """The shared planning WorldGraph (same object /graph uses).

        Cached on ``_commitment_world_graph`` — NOT on the method's
        own name (runtime_memory's ``_control_graph`` convention).
        """
        g = getattr(self, "_commitment_world_graph", None)
        if g is None:
            from ...planning.graph import WorldGraph

            g = WorldGraph()
            self._commitment_world_graph = g
        return g

    def _sync_commitment_graph(self, ledger: Any) -> int:
        """Project the commitment ledger into the planning graph (AA-12).

        Called after every ledger mutation on this surface, so
        commitment / person / deadline nodes in the world graph track
        the ledger (open → kept / broken / cancelled) instead of the
        graph silently never syncing. Idempotent (identity is the
        commitment id). Best-effort: the ledger is the truth and the
        graph a projection — a sync failure is logged, never raised
        into the command. Returns the number of commitments projected.
        """
        try:
            from ...planning.commitment_sync import sync_ledger_to_graph

            return sync_ledger_to_graph(ledger, self._world_graph())
        except Exception as exc:  # noqa: BLE001 — projection, never fatal
            _log.warning("commitment graph sync failed: %s", exc)
            return 0

    def _resolve_commitment(self, ledger: Any, ref: str) -> Any:
        """Exact id, else unique id-prefix, else None."""
        ref = (ref or "").strip()
        if ref in ledger.commitments:
            return ledger.commitments[ref]
        matches = [c for cid, c in ledger.commitments.items()
                   if cid.startswith(ref)]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _fmt_commitment(c: Any) -> str:
        from ...commitments.ledger import format_deadline

        due = format_deadline(c.deadline) if c.deadline else "no deadline"
        if c.deadline and c.is_overdue():
            due += " (OVERDUE)"
        who = f" → {c.promisee}" if c.promisee else ""
        return f"  {c.id}: {c.what}{who} — {due} [{c.status}]"

    def _control_commitments(self, tail: str, *, chat_key: str = "") -> str:
        """The obligation layer: /commitments list|add|kept|broken|cancel|overdue|due|reliability|verify."""
        from ...commitments.ledger import parse_deadline

        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""
        ledger = self._commitments_ledger()
        if verb in ("list", "open", ""):
            rows = ledger.open_commitments()
            if not rows:
                return ("nothing open — /commitments add <what> [by <deadline>] "
                        "records a promise.")
            lines = [f"open commitments ({len(rows)}):"]
            lines.extend(self._fmt_commitment(c) for c in rows)
            return "\n".join(lines)
        if verb == "add":
            what = rest.strip()
            if not what:
                return "usage: /commitments add <what> [by <deadline>] — e.g. /commitments add call Ada back by Friday"
            deadline = None
            deadline_note = ""
            if " by " in f" {what.lower()} ":
                idx = what.lower().rfind(" by ")
                candidate = what[idx + 4:].strip()
                parsed = parse_deadline(candidate)
                if parsed is not None:
                    deadline = parsed
                    what = what[:idx].strip()
                else:
                    deadline_note = " (deadline not understood — recorded without one)"
            if not what:
                return "usage: /commitments add <what> [by <deadline>]"
            c = ledger.add(what=what, deadline=deadline,
                           source_text=f"/commitments add {rest}")
            self._sync_commitment_graph(ledger)
            return f"recorded: {self._fmt_commitment(c).strip()}{deadline_note}"
        if verb in ("kept", "broken", "cancel"):
            bits = rest.strip().split(None, 1)
            if not bits:
                return f"usage: /commitments {verb} <id> [evidence]"
            c = self._resolve_commitment(ledger, bits[0])
            if c is None:
                return f"no commitment matching {bits[0]!r} — /commitments list shows ids."
            evidence = bits[1] if len(bits) > 1 else ""
            try:
                if verb == "kept":
                    ledger.mark_kept(c.id, evidence)
                elif verb == "broken":
                    ledger.mark_broken(c.id, evidence)
                else:
                    ledger.cancel(c.id, evidence)
            except (KeyError, ValueError) as exc:
                return str(exc)
            self._sync_commitment_graph(ledger)
            return f"marked {verb}: {c.what} [{c.id}]"
        if verb == "overdue":
            rows = ledger.overdue()
            if not rows:
                return "nothing overdue."
            return "\n".join(["overdue:"] + [self._fmt_commitment(c) for c in rows])
        if verb == "due":
            rows = ledger.due_soon()
            if not rows:
                return "nothing due in the next 24h."
            return "\n".join(["due soon:"] + [self._fmt_commitment(c) for c in rows])
        if verb == "reliability":
            rel = ledger.reliability(rest.strip() or None)
            rate = rel.get("rate")
            rate_txt = f"{rate:.0%}" if rate is not None else "no resolved promises yet"
            return (f"reliability: {rel.get('kept', 0)} kept / "
                    f"{rel.get('broken', 0)} broken / "
                    f"{rel.get('open', 0)} open "
                    f"(kept rate {rate_txt})")
        if verb == "verify":
            v = ledger.verify()
            return (f"ledger verify: {v.get('entries')} entries, "
                    f"{v.get('commitments')} commitments, "
                    f"seq contiguous: {v.get('seq_contiguous')}, "
                    f"snapshot matches replay: {v.get('snapshot_matches_replay')}")
        return ("usage: /commitments list · add <what> [by <deadline>]\n"
                "       kept|broken|cancel <id> [evidence] · overdue · due\n"
                "       reliability [promisee] · verify")

    # ── focus sessions: /focus ─────────────────────────────────────────
    def _focus_timer(self) -> Any:
        """FocusTimer on the durable home path."""
        from ...focus.sessions import FocusTimer

        path = self.context.settings.home_path / "data" / "focus.json"
        return FocusTimer(str(path))

    def _control_focus(self, tail: str, *, chat_key: str = "") -> str:
        """Focus sessions + week review: /focus start|pause|resume|stop|status|today|week|month|mix."""
        from ...focus.sessions import format_duration

        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else "status"
        rest = parts[1] if len(parts) > 1 else ""
        timer = self._focus_timer()
        if verb == "mix":
            # MusicForProgramming (#31): /focus mix list|show|play|
            # assemble|refresh over the focus-mix store.
            from ...music_sources.focus_mixes import (
                MOOD_TAGS,
                FocusMixStore,
                assemble_mix,
                find_mix,
                format_mix_reply,
                refresh_from_series,
                render_mix,
                start_focus_session_with_mix,
            )
            from ...tools.focus_mix import dj_planner, read_owner_taste

            store = FocusMixStore(self.context.settings.home_path)
            sub = (rest or "").strip().split(None, 1)
            subverb = sub[0].lower() if sub else "list"
            subrest = sub[1] if len(sub) > 1 else ""
            if subverb in ("list", ""):
                mixes = store.list()
                if not mixes:
                    return ("no focus mixes yet — /focus mix assemble "
                            "builds one from your library, /focus mix "
                            "refresh pulls the Music For Programming "
                            "series.")
                lines = ["focus mixes:"]
                for mix in mixes:
                    lines.append(
                        f"  {mix.id} — {mix.title} "
                        f"({len(mix.tracks)} tracks, "
                        f"{format_duration(mix.duration_s)}, "
                        f"played in {mix.sessions_played} session(s))")
                return "\n".join(lines)
            if subverb == "show":
                mix = find_mix(store, subrest)
                if mix is None:
                    return ("no focus mix matches that id or title — "
                            "/focus mix list to see them.")
                return render_mix(mix)
            if subverb == "play":
                mix = find_mix(store, subrest)
                if mix is None:
                    return ("no focus mix matches that id or title — "
                            "/focus mix list to see them.")
                result = start_focus_session_with_mix(
                    timer, store, mix,
                    player=self._focus_mix_player(chat_key))
                return format_mix_reply(mix, result["session"],
                                        result["play"])
            if subverb == "assemble":
                mood = subrest.strip().lower()
                tags = dict(MOOD_TAGS[mood]) if mood in MOOD_TAGS else {}
                if mood and mood not in MOOD_TAGS:
                    return (f"unknown mood {mood!r} — darker | lighter | "
                            "deep | driving | calm | ambient")
                library = None
                db = getattr(self.context, "db", None)
                if db is not None:
                    try:
                        from ...media.library import MusicLibrary

                        library = MusicLibrary(self.context)
                    except Exception:  # noqa: BLE001 - no library, honest
                        library = None
                try:
                    mix = assemble_mix(
                        store, library=library,
                        feedback_rows=read_owner_taste(), tags=tags,
                        planner=dj_planner)
                except ValueError as exc:
                    return str(exc)
                return render_mix(mix)
            if subverb == "refresh":
                result = refresh_from_series(store)
                if result.get("error"):
                    return f"series refresh failed: {result['error']}"
                return (f"series refreshed: {result['added']} new "
                        f"episode(s) of {result['total']} in the feed.")
            return ("usage: /focus mix [list] · show <id> · play <id> · "
                    "assemble [mood] · refresh")
        if verb == "start":
            # /focus start <label> --mix <id|title|episode N>: start
            # the session with a mix attached, first track playing.
            label, _, mix_ref = rest.partition("--mix")
            label = label.strip()
            mix_ref = mix_ref.strip()
            if mix_ref:
                from ...music_sources.focus_mixes import (
                    FocusMixStore,
                    find_mix,
                    format_mix_reply,
                    start_focus_session_with_mix,
                )

                store = FocusMixStore(self.context.settings.home_path)
                mix = find_mix(store, mix_ref)
                if mix is None:
                    return (f"no focus mix matches {mix_ref!r} — "
                            "/focus mix list to see them.")
                result = start_focus_session_with_mix(
                    timer, store, mix,
                    player=self._focus_mix_player(chat_key), label=label)
                return format_mix_reply(mix, result["session"],
                                        result["play"])
            try:
                s = timer.start(label=label)
            except ValueError as exc:
                return str(exc)
            label = f" ({s.label})" if s.label else ""
            from ...social.chat.tgbot_buttons import actionable

            return actionable(
                f"focus started{label}.",
                [[("⏸️ Pause", "focus pause"),
                  ("⏹️ Stop", "focus stop")]])
        if verb == "pause":
            try:
                s = timer.pause()
            except (KeyError, ValueError) as exc:
                return str(exc)
            from ...social.chat.tgbot_buttons import actionable

            return actionable(
                f"paused ({s.label or 'untitled'}).",
                [[("▶️ Resume", "focus resume")]])
        if verb == "resume":
            try:
                s = timer.resume()
            except (KeyError, ValueError) as exc:
                return str(exc)
            return f"resumed ({s.label or 'untitled'})."
        if verb == "stop":
            try:
                s = timer.stop(note=rest.strip())
            except (KeyError, ValueError) as exc:
                return str(exc)
            total = format_duration(s.elapsed())
            reply = f"stopped ({s.label or 'untitled'}) — {total} focused."
            # R3-55 remainder: the hold lifts with the session — a
            # held alert arrives now, not at the next tick.  The
            # flush never blocks the stop reply (it never raises;
            # a failure just leaves the rows held for the tick).
            try:
                from ..notifier import Notifier

                flushed = Notifier(
                    self.context, self.gateway).flush_held_focus()
                if flushed.get("sent"):
                    reply += (f"\n{flushed['sent']} held alert(s) "
                              "delivered.")
            except Exception:  # noqa: BLE001 - stop reply stands alone
                pass
            return reply
        if verb == "today":
            import datetime as _dt

            now = time.time()
            start = _dt.datetime.fromtimestamp(now).replace(
                hour=0, minute=0, second=0, microsecond=0).timestamp()
            summary = timer.summary(start=start, end=now)
            lines = [f"today: {format_duration(summary['total_seconds'])} "
                     f"across {summary['sessions']} session(s)"]
            for row in summary["by_label"]:
                lines.append(f"  {row['label']}: {format_duration(row['seconds'])} "
                             f"({row['sessions']}×, {row['interruptions']} interruptions)")
            return "\n".join(lines)
        if verb == "week":
            # AA-02: the periodic-review composer (nomorals.reviews)
            # wraps the focus builder — same wins/totals/calibration,
            # plus slipped / blocked / next-week synthesis.
            from ...reviews.review import build_weekly_review
            ledger = self._commitments_ledger()
            review = build_weekly_review(
                commitments=ledger, decisions=self._decisions_journal(),
                focus=timer)
            return review.render_markdown()
        if verb == "month":
            # AA-02: monthly = aggregation of the last 4 weekly reviews,
            # each built from the real ledgers at its own week offset —
            # never fabricated weeks.
            from ...reviews.review import build_monthly_review, build_weekly_review
            ledger = self._commitments_ledger()
            journal = self._decisions_journal()
            now = time.time()
            weeks = []
            for back in range(4):
                wk_now = now - back * 7 * 86400.0
                weeks.append(build_weekly_review(
                    commitments=ledger, decisions=journal, focus=timer,
                    now=wk_now))
            review = build_monthly_review(weeks=weeks, now=now)
            return review.render_markdown()
        if verb in ("status", ""):
            active = timer.active()
            import datetime as _dt

            now = time.time()
            start = _dt.datetime.fromtimestamp(now).replace(
                hour=0, minute=0, second=0, microsecond=0).timestamp()
            today = timer.summary(start=start, end=now)
            today_line = (f"today: {format_duration(today['total_seconds'])} "
                          f"across {today['sessions']} session(s)")
            if active is None:
                return f"no focus session running.\n{today_line}\n/focus start [label] to begin."
            state = "paused" if active.is_paused() else "running"
            label = f" ({active.label})" if active.label else ""
            return (f"focus {state}{label} — {format_duration(active.elapsed())} so far.\n"
                    f"{today_line}")
        return ("usage: /focus start [label] [--mix <id>] · pause · resume · stop [note]\n"
                "       /focus status · today · week · month · mix [list|show|play|assemble|refresh]")

    # ── decision journal: /decisions ──────────────────────────────────
    def _decisions_journal(self) -> Any:
        """DecisionJournal on the durable home path (AA-24)."""
        from ...decisions.journal import DecisionJournal

        path = self.context.settings.home_path / "data" / "decisions.json"
        return DecisionJournal(str(path))

    def _resolve_decision(self, journal: Any, ref: str) -> Any:
        """Exact id, else unique id-prefix, else None."""
        ref = (ref or "").strip()
        if ref in journal.decisions:
            return journal.decisions[ref]
        matches = [d for did, d in journal.decisions.items()
                   if did.startswith(ref)]
        return matches[0] if len(matches) == 1 else None

    def _assets_ledger(self) -> Any:
        """AssetLedger on the durable home path (E-6 vehicle wiring)."""
        from ...assets.maintenance import AssetLedger

        path = self.context.settings.home_path / "data" / "assets.json"
        return AssetLedger(str(path))

    def _resolve_asset(self, ledger: Any, ref: str) -> Any:
        """Exact id, else unique id-prefix, else None."""
        ref = (ref or "").strip()
        if ref in ledger.assets:
            return ledger.assets[ref]
        matches = [a for aid, a in ledger.assets.items()
                   if aid.startswith(ref)]
        return matches[0] if len(matches) == 1 else None

    def _control_assets(self, tail: str, *, chat_key: str = "") -> str:
        """Vehicle & asset maintenance: /assets list|add|service|due|costs|show.

        The MORE_TO_ADD gap 7 organ (nomorals/assets/maintenance.py) had
        no production importer at all — this is its surface. Costs are
        typed in major units and stored as integer minor units; a service
        defaults to today and takes an optional YYYY-MM-DD.
        """
        import datetime as _dt

        from ...assets.maintenance import Asset, AssetLedger, ServiceRecord

        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""
        ledger = self._assets_ledger()
        if verb in ("list", ""):
            if not ledger.assets:
                return ("no assets yet — /assets add <name> [| kind | "
                        "interval_km | interval_days | odometer_km] records one.")
            lines = [f"assets ({len(ledger.assets)}):"]
            for asset in ledger.assets.values():
                last = (asset.last_service_date.isoformat()
                        if asset.last_service_date else "never")
                lines.append(
                    f"  {asset.id}: {asset.name} ({asset.kind}) — "
                    f"{asset.odometer_km:,.0f} km, last service {last}")
            return "\n".join(lines)
        if verb == "add":
            fields = [f.strip() for f in rest.split("|")]
            name = fields[0] if fields else ""
            if not name:
                return ("usage: /assets add <name> [| kind | interval_km | "
                        "interval_days | odometer_km]")
            kind = fields[1] if len(fields) > 1 and fields[1] else "other"
            try:
                interval_km = (float(fields[2])
                               if len(fields) > 2 and fields[2] else None)
                interval_days = (int(fields[3])
                                 if len(fields) > 3 and fields[3] else None)
                odometer_km = (float(fields[4])
                               if len(fields) > 4 and fields[4] else 0.0)
            except ValueError:
                return ("intervals and odometer must be numbers — e.g. "
                        "/assets add Camry | vehicle | 5000 | 180 | 14000")
            try:
                asset = ledger.add_asset(Asset(
                    name=name, kind=kind, interval_km=interval_km,
                    interval_days=interval_days, odometer_km=odometer_km))
            except ValueError as exc:
                return str(exc)
            out = f"recorded: {asset.id}: {asset.name} ({asset.kind})"
            if interval_km:
                out += f" — service every {interval_km:,.0f} km"
            if interval_days:
                out += f" / {interval_days} days"
            return out + (" — log a service with "
                          f"/assets service {asset.id} <what> "
                          "[| cost | odometer_km | vendor]")
        if verb == "service":
            fields = [f.strip() for f in rest.split("|")]
            head = fields[0].split(None, 1) if fields else []
            if len(head) < 2:
                return ("usage: /assets service <id> <description> "
                        "[| cost | odometer_km | vendor | YYYY-MM-DD]")
            asset = self._resolve_asset(ledger, head[0])
            if asset is None:
                return (f"no asset matching {head[0]!r} — "
                        "/assets list shows ids.")
            try:
                cost_minor = 0
                if len(fields) > 1 and fields[1]:
                    cost_minor = int(round(float(fields[1]) * 100))
                odometer_km = (float(fields[2])
                               if len(fields) > 2 and fields[2] else None)
                vendor = fields[3] if len(fields) > 3 else ""
                service_date = _dt.date.today()
                if len(fields) > 4 and fields[4]:
                    service_date = _dt.date.fromisoformat(fields[4])
            except ValueError:
                return ("cost and odometer must be numbers and the date "
                        "YYYY-MM-DD — e.g. /assets service <id> oil change "
                        "| 45000 | 15000 | AutoServe")
            try:
                ledger.log_service(ServiceRecord(
                    asset_id=asset.id, service_date=service_date,
                    description=head[1], cost_minor=cost_minor,
                    odometer_km=odometer_km, vendor=vendor))
            except ValueError as exc:
                return str(exc)
            cost_text = AssetLedger.format_cost(cost_minor, asset.currency)
            return (f"logged: {asset.name} — {head[1]} on "
                    f"{service_date.isoformat()} ({cost_text})")
        if verb == "due":
            try:
                upcoming = int(rest.strip()) if rest.strip() else 14
            except ValueError:
                return "/assets due [days] — days must be a number."
            reminders = ledger.due_reminders(
                _dt.date.today(), upcoming_days=upcoming)
            if not reminders:
                return f"no services due in the next {upcoming} days."
            lines = [f"services due ({len(reminders)}):"]
            for reminder in reminders:
                lines.append(f"  {reminder.detail}")
            return "\n".join(lines)
        if verb == "costs":
            if rest.strip():
                asset = self._resolve_asset(ledger, rest.strip())
                if asset is None:
                    return (f"no asset matching {rest.strip()!r} — "
                            "/assets list shows ids.")
                total = ledger.total_cost(asset.id)
                count = sum(1 for r in ledger.records
                            if r.asset_id == asset.id)
                return (f"{asset.name}: "
                        f"{AssetLedger.format_cost(total, asset.currency)} "
                        f"across {count} service(s).")
            if not ledger.assets:
                return "no assets yet — /assets add <name> records one."
            lines = ["service costs:"]
            for asset in ledger.assets.values():
                total = ledger.total_cost(asset.id)
                lines.append(f"  {asset.name}: "
                             f"{AssetLedger.format_cost(total, asset.currency)}")
            return "\n".join(lines)
        if verb == "show":
            asset = self._resolve_asset(ledger, rest.strip())
            if asset is None:
                return (f"no asset matching {rest.strip()!r} — "
                        "/assets list shows ids.")
            last = (asset.last_service_date.isoformat()
                    if asset.last_service_date else "never")
            lines = [
                f"{asset.name} ({asset.kind}) — {asset.id}",
                f"  odometer: {asset.odometer_km:,.0f} km · last service: {last}",
            ]
            if asset.interval_km:
                every = f"  service every {asset.interval_km:,.0f} km"
                if asset.interval_days:
                    every += f" / {asset.interval_days} days"
                lines.append(every)
            elif asset.interval_days:
                lines.append(f"  service every {asset.interval_days} days")
            rows = [r for r in ledger.records if r.asset_id == asset.id]
            if not rows:
                lines.append("  no services logged yet.")
            else:
                lines.append(f"  services ({len(rows)}):")
                for record in rows:
                    odo = (f", {record.odometer_km:,.0f} km"
                           if record.odometer_km is not None else "")
                    vendor = f" — {record.vendor}" if record.vendor else ""
                    lines.append(
                        f"    {record.service_date.isoformat()}: "
                        f"{record.description} "
                        f"({AssetLedger.format_cost(record.cost_minor, asset.currency)}"
                        f"{odo}{vendor})")
            return "\n".join(lines)
        return ("usage: /assets list | add <name> [| kind | interval_km | "
                "interval_days | odometer_km] | service <id> <description> "
                "[| cost | odometer_km | vendor | YYYY-MM-DD] | due [days] | "
                "costs [id] | show <id>")

    def _legacy_store(self) -> Any:
        """WillStore on the durable home path (E-6 legacy wiring)."""
        from ...legacy.will import WillStore

        path = self.context.settings.home_path / "data" / "legacy.json"
        return WillStore(str(path))

    def _legacy_vault(self) -> Any:
        """Open the accounts vault for will sealing (actor "legacy").

        Lazy import, same shape as runtime_system/runtime_media: the
        vault opens with whatever owner credential is at hand; a locked
        vault raises VaultLockedError naming every source checked, so
        seal/reveal fail closed instead of storing anything unsealed.
        """
        from ...accounts.vault import open_vault

        return open_vault(getattr(self.context, "db", None),
                          context=self.context, actor="legacy")

    def _legacy_show(self, will: Any) -> str:
        """Metadata-only rendering — the sealed body surfaces as the marker, never content."""
        from ...legacy.will import SEALED_MARKER

        summary = will.summary(is_owner=True)
        lines = [
            f"will for {will.owner_name} — {will.id} "
            f"(created {summary['created']})",
            "  executor: " + will.executor.name
            + (f" (backup: {will.executor.backup_name})"
               if will.executor.backup_name else ""),
            "  beneficiaries (" + str(summary["beneficiaries"]) + "): "
            + (", ".join(b.name for b in will.beneficiaries) or "none"),
            "  digital assets (" + str(summary["digital_assets"]) + "): "
            + (", ".join(a.label for a in will.digital_assets) or "none"),
            "  instructions (" + str(summary["instructions"]) + "):",
        ]
        for text in will.instructions:
            lines.append(f"    - {text}")
        lines.append(
            "  witnesses (" + str(summary["witnesses"]) + "): "
            + (", ".join(will.witnesses) or "none"))
        if summary["has_sealed_content"]:
            lines.append("  sealed body: " + SEALED_MARKER)
        else:
            lines.append(
                "  no sealed body yet — /legacy seal <text> seals one.")
        return "\n".join(lines)

    def _control_legacy(self, tail: str, *, chat_key: str = "") -> str:
        """Digital will: /legacy show|create|executor|beneficiary|asset|instruction|witness|seal|reveal|reset.

        The MORE_TO_ADD gap 8 organ (nomorals/legacy/will.py) had no
        production importer at all — this is its surface. Owner-only:
        the kind is absent from PUBLIC_COMMANDS, so caller_may_run
        denies non-owners before this runs (is_owner=True below means
        the dispatch layer already established the owner). The sealed
        body is written ONLY through the accounts vault (encrypt_blob,
        purpose "legacy-will") and read back ONLY through it; this
        handler never sees a key, and a locked vault fails closed. A
        corrupt will file refuses to load instead of silently becoming
        an empty will.
        """
        from ...legacy.will import (
            Beneficiary,
            DigitalAsset,
            Executor,
            Will,
        )

        usage = ("usage: /legacy show | create <owner> | <executor> "
                 "[| contact | backup] | executor <name> [| contact | backup] "
                 "| beneficiary <name> [| relationship | share note] "
                 "| asset <label> [| kind | location hint | instruction] "
                 "| instruction <text> | witness <name> "
                 "| seal <text> | reveal | reset yes")
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else "show"
        rest = parts[1] if len(parts) > 1 else ""
        try:
            store = self._legacy_store()
        except ValueError as exc:
            return str(exc)
        will = store.get()
        if verb == "show":
            if will is None:
                return ("no will yet — /legacy create <owner> | <executor> "
                        "[| contact | backup] records one.")
            return self._legacy_show(will)
        if verb == "create":
            if will is not None:
                return ("a will already exists (" + will.id + ") — edit it "
                        "with /legacy executor | beneficiary | asset | "
                        "instruction | witness, or /legacy reset yes to "
                        "start over.")
            fields = [f.strip() for f in rest.split("|")]
            owner = fields[0] if fields else ""
            executor_name = fields[1] if len(fields) > 1 else ""
            if not owner or not executor_name:
                return ("usage: /legacy create <owner> | <executor> "
                        "[| contact | backup]")
            executor = Executor(
                name=executor_name,
                contact=fields[2] if len(fields) > 2 else "",
                backup_name=fields[3] if len(fields) > 3 else "")
            new_will = Will(owner_name=owner, executor=executor)
            store.put(new_will)
            return ("recorded: " + new_will.id + ": will for " + owner
                    + " — executor " + executor_name + ". Add beneficiaries "
                    "with /legacy beneficiary <name>, assets with /legacy "
                    "asset <label>, and seal the body with "
                    "/legacy seal <text>.")
        if will is None:
            return ("no will yet — /legacy create <owner> | <executor> "
                    "[| contact | backup] records one.")
        if verb == "executor":
            fields = [f.strip() for f in rest.split("|")]
            name = fields[0] if fields else ""
            if not name:
                return "usage: /legacy executor <name> [| contact | backup]"
            will.executor = Executor(
                name=name,
                contact=fields[1] if len(fields) > 1 else "",
                backup_name=fields[2] if len(fields) > 2 else "")
            store.put(will)
            return "executor set: " + name
        if verb == "beneficiary":
            fields = [f.strip() for f in rest.split("|")]
            name = fields[0] if fields else ""
            if not name:
                return ("usage: /legacy beneficiary <name> "
                        "[| relationship | share note]")
            will.beneficiaries.append(Beneficiary(
                name=name,
                relationship=fields[1] if len(fields) > 1 else "",
                share_note=fields[2] if len(fields) > 2 else ""))
            store.put(will)
            return "recorded beneficiary: " + name
        if verb == "asset":
            fields = [f.strip() for f in rest.split("|")]
            label = fields[0] if fields else ""
            if not label:
                return ("usage: /legacy asset <label> "
                        "[| kind | location hint | instruction]")
            kind = fields[1] if len(fields) > 1 and fields[1] else "account"
            will.digital_assets.append(DigitalAsset(
                label=label, kind=kind,
                location_hint=fields[2] if len(fields) > 2 else "",
                instruction=fields[3] if len(fields) > 3 else ""))
            store.put(will)
            return "recorded asset: " + label + " (" + kind + ")"
        if verb == "instruction":
            text = rest.strip()
            if not text:
                return "usage: /legacy instruction <text>"
            will.instructions.append(text)
            store.put(will)
            return "recorded instruction."
        if verb == "witness":
            name = rest.strip()
            if not name:
                return "usage: /legacy witness <name>"
            will.witnesses.append(name)
            store.put(will)
            return "recorded witness: " + name
        if verb == "seal":
            text = rest.strip()
            if not text:
                return "usage: /legacy seal <text>"
            from ...accounts.vault import VaultLockedError

            try:
                vault = self._legacy_vault()
            except VaultLockedError as exc:
                return "cannot seal — vault locked: " + str(exc)
            already = will.summary(is_owner=True)["has_sealed_content"]

            def seal_fn(plaintext: str) -> str:
                return vault.encrypt_blob(plaintext, purpose="legacy-will")

            will.seal_content(text, seal_fn, is_owner=True)
            store.put(will)
            note = "sealed — the body is stored only as a vault blob"
            if already:
                note += " (replaces the previous sealed body)"
            return (note + "; /legacy show never shows it, /legacy reveal "
                    "unlocks it with the vault.")
        if verb == "reveal":
            if not will.summary(is_owner=True)["has_sealed_content"]:
                return ("no sealed body on this will — "
                        "/legacy seal <text> seals one.")
            from ...accounts.vault import VaultLockedError
            from ...core.cipher import CipherError
            from ...legacy.will import SealedError

            try:
                vault = self._legacy_vault()
            except VaultLockedError as exc:
                return "cannot reveal — vault locked: " + str(exc)

            def unlock_fn(blob: str) -> str:
                return vault.decrypt_blob(blob, purpose="legacy-will")

            try:
                return will.reveal(unlock_fn, is_owner=True)
            except SealedError as exc:
                return "reveal failed: " + str(exc)
            except CipherError as exc:
                return ("reveal failed — the vault could not unlock this: "
                        + str(exc))
        if verb == "reset":
            if rest.strip() != "yes":
                return ("reset needs confirmation — /legacy reset yes "
                        "deletes the will. This cannot be undone.")
            store.clear()
            return "will deleted."
        return usage

    def _family_registry(self) -> Any:
        """FamilyRegistry on the durable home path (family-mode wiring)."""
        from ...family.registry import family_registry

        root = self.context.settings.home_path / "family"
        return family_registry(root)

    def _control_family(self, tail: str, *, chat_key: str = "") -> str:
        """Family mode: /family list|add|revoke|role|item|items|show|share|delete|audit|check|policy.

        The MORE_TO_ADD gap 4 organ (nomorals/family/) had no production
        importer at all — this is its surface. The household lives under
        <home>/family/ (members.json, audit.jsonl, one directory per
        person); the organ owns the layout, this handler only points it
        at Devon's home. Owner-only: the kind is absent from
        PUBLIC_COMMANDS, so caller_may_run denies non-owners before
        this runs. The chat surface speaks as the household's first
        active owner (resolved from the roster, never a claimed role —
        the roster may call the owner by their real name, not "owner").
        Item writes act as the item's owner instead, so the organ's
        per-role kid-safe rules apply to the member the item belongs to
        (a child's flagged write is refused, a teen's is masked). The
        audit read goes through the organ's claimed-role path on
        purpose: the dispatch layer already established the caller is
        the owner, and the roster-gated path would refuse a household
        whose owner record uses a different person_id (or none yet).
        The household shared layer (nomorals/household — commitments,
        calendar, assets) is NOT exposed here; it stays import-only
        (see household/library.json).
        """
        import datetime as _dt
        import uuid as _uuid

        from ...family.locales import get_locale_pack, list_locale_packs
        from ...family.policy import family_policy

        usage = ("usage: /family list | add <person_id> <name> "
                 "[| role | locale] | revoke <person_id> [reason] "
                 "| role <person_id> <role> | item <person_id> <kind> "
                 "<title> [| body | shared|private] | items <person_id> "
                 "| show <person_id> <item_id> | share <person_id> "
                 "<item_id> | delete <person_id> <item_id> | audit [n] "
                 "| check <role> <text> | policy")
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""
        registry = self._family_registry()
        household = registry.household

        def _owner_actor() -> str:
            members = household.list_members(actor_role="owner")
            for m in members:
                if (m.get("role") == "owner"
                        and m.get("status") == "active"):
                    return str(m["person_id"])
            return "owner"

        def _roster_lines(members: list) -> list[str]:
            lines = [f"household ({len(members)} member(s)):"]
            for m in members:
                line = (f"  {m['person_id']} — {m['display_name']} "
                        f"({m['role']}, {m.get('locale', 'ng')})")
                if m.get("status") == "revoked":
                    line += " [revoked]"
                lines.append(line)
            return lines

        try:
            if verb == "":
                members = household.list_members(actor_role="owner")
                if not members:
                    return ("no household yet — /family add owner "
                            "<your name> sets you up as the first owner.")
                lines = _roster_lines(members)
                lines.append("verbs: add | revoke | role | item | items | "
                             "show | share | delete | audit | check | policy")
                return "\n".join(lines)
            if verb == "list":
                members = household.list_members(actor_role="owner")
                if not members:
                    return ("no members yet — /family add owner <your name> "
                            "sets you up as the first owner.")
                return "\n".join(_roster_lines(members))
            if verb == "add":
                fields = [f.strip() for f in rest.split("|")]
                head = fields[0].split(None, 1) if fields else []
                if len(head) < 2:
                    return ("usage: /family add <person_id> <name> "
                            "[| role | locale]")
                person_id, display_name = head
                role = fields[1] if len(fields) > 1 and fields[1] else ""
                locale = fields[2] if len(fields) > 2 and fields[2] else "ng"
                if not role:
                    existing = household.list_members(actor_role="owner")
                    role = "owner" if not existing else "adult"
                added = registry.call(
                    "family_person_add", actor_id=_owner_actor(),
                    actor_role="owner", person_id=person_id,
                    display_name=display_name, role=role, locale=locale)
                return (f"added: {added['person_id']} — "
                        f"{added['display_name']} "
                        f"({added['role']}, {added.get('locale', 'ng')})")
            if verb == "revoke":
                head = rest.split(None, 1)
                if not head:
                    return "usage: /family revoke <person_id> [reason]"
                reason = head[1] if len(head) > 1 else ""
                revoked = registry.call(
                    "family_person_revoke", actor_id=_owner_actor(),
                    actor_role="owner", person_id=head[0], reason=reason)
                return (f"revoked: {revoked['person_id']} — "
                        f"{revoked['display_name']} ({revoked['role']})")
            if verb == "role":
                head = rest.split(None, 1)
                if len(head) < 2:
                    return "usage: /family role <person_id> <role>"
                before = household.get_member(head[0])
                old = before["role"] if before else ""
                changed = household.set_role(
                    actor_id=_owner_actor(), actor_role="owner",
                    person_id=head[0], role=head[1].strip())
                return (f"role: {changed['person_id']} is now "
                        f"{changed['role']} (was {old})")
            if verb == "item":
                fields = [f.strip() for f in rest.split("|")]
                head = fields[0].split(None, 2) if fields else []
                if len(head) < 3:
                    return ("usage: /family item <person_id> <kind> <title> "
                            "[| body | shared|private]")
                person_id, kind, title = head
                body = fields[1] if len(fields) > 1 else ""
                visibility = (fields[2].lower()
                              if len(fields) > 2 and fields[2] else "private")
                if visibility not in ("shared", "private"):
                    return ("visibility must be shared or private — e.g. "
                            "/family item ada note title | body | shared")
                item_id = _uuid.uuid4().hex[:12]
                stored = registry.call(
                    "family_item_put", actor_id=person_id,
                    actor_role="owner", person_id=person_id,
                    item_id=item_id, kind=kind, title=title, body=body,
                    visibility=visibility)
                out = (f"stored: {stored['id']} — {stored['title']} "
                       f"({stored['kind']}, {stored['visibility']})")
                if stored.get("flags"):
                    out += " — flagged: " + ", ".join(stored["flags"])
                return out
            if verb == "items":
                person_id = rest.strip()
                if not person_id:
                    return "usage: /family items <person_id>"
                items = registry.call(
                    "family_item_list", actor_id=_owner_actor(),
                    actor_role="owner", person_id=person_id)
                if not items:
                    return f"no items for {person_id}."
                lines = [f"items for {person_id} ({len(items)}):"]
                for it in items:
                    line = (f"  {it['id']}: {it['kind']} — {it['title']} "
                            f"({it['visibility']})")
                    if it.get("flags"):
                        line += " [flagged: " + ", ".join(it["flags"]) + "]"
                    lines.append(line)
                return "\n".join(lines)
            if verb == "show":
                head = rest.split(None, 1)
                if len(head) < 2:
                    return "usage: /family show <person_id> <item_id>"
                item = registry.call(
                    "family_item_get", actor_id=_owner_actor(),
                    actor_role="owner", person_id=head[0],
                    item_id=head[1].strip())
                lines = [item["title"],
                         f"  kind: {item['kind']} · visibility: "
                         f"{item['visibility']} · kid-safe: "
                         f"{'yes' if item.get('kid_safe') else 'no'}"]
                if item.get("body"):
                    lines.append(f"  {item['body']}")
                if item.get("flags"):
                    lines.append("  flagged: " + ", ".join(item["flags"]))
                return "\n".join(lines)
            if verb == "share":
                head = rest.split(None, 1)
                if len(head) < 2:
                    return "usage: /family share <person_id> <item_id>"
                shared = registry.call(
                    "family_item_share", actor_id=_owner_actor(),
                    actor_role="owner", person_id=head[0],
                    item_id=head[1].strip())
                return f"shared: {shared['id']} — {shared['title']}"
            if verb == "delete":
                head = rest.split(None, 1)
                if len(head) < 2:
                    return "usage: /family delete <person_id> <item_id>"
                result = registry.call(
                    "family_item_delete", actor_id=_owner_actor(),
                    actor_role="owner", person_id=head[0],
                    item_id=head[1].strip())
                if result.get("deleted"):
                    return f"deleted: {head[1].strip()}"
                return f"no item {head[1].strip()} for {head[0]}."
            if verb == "audit":
                try:
                    limit = int(rest.strip()) if rest.strip() else 10
                except ValueError:
                    return "/family audit [n] — n must be a number."
                entries = household.audit_log(actor_role="owner")
                if not entries:
                    return "audit log is empty."
                shown = entries[-limit:] if limit > 0 else entries
                lines = [f"audit ({len(entries)} entries):"]
                for e in shown:
                    when = _dt.datetime.fromtimestamp(
                        e["ts"]).strftime("%Y-%m-%d %H:%M")
                    detail = f" {e['detail']}" if e.get("detail") else ""
                    lines.append(
                        f"  {when} — {e['actor_id']} {e['action']} "
                        f"{e.get('target', '')}{detail}")
                return "\n".join(lines)
            if verb == "check":
                head = rest.split(None, 1)
                if len(head) < 2:
                    return "usage: /family check <role> <text>"
                result = registry.call(
                    "family_kid_check", text=head[1],
                    role=head[0].strip(), locale="ng")
                flags = result.get("flags") or []
                if not result.get("allowed"):
                    return (f"refused for {result['role']} — flagged: "
                            + ", ".join(flags))
                text = result.get("text", "")
                if text != head[1]:
                    return (f"masked for {result['role']}: {text} "
                            f"(flagged: " + ", ".join(flags) + ")")
                if flags:
                    return (f"passes for {result['role']} — flagged: "
                            + ", ".join(flags))
                return f"passes for {result['role']} — clean"
            if verb == "policy":
                policy = family_policy()
                packs = list_locale_packs()
                return ("family policy — allowlist: "
                        + ", ".join(policy["allowlist"])
                        + " (everything else denied).\nlocales: "
                        + ", ".join(packs)
                        + f" (default {get_locale_pack(None)['code']}).")
            return usage
        except (ValueError, KeyError, PermissionError, LookupError) as exc:
            return str(exc)

    def _show_store(self) -> Any:
        """ShowStore on the durable home path (shows wiring)."""
        from ...shows.store import ShowStore

        path = (self.context.settings.home_path / "data" / "shows.json")
        return ShowStore(path)

    def _resolve_show(self, store: Any, ref: str) -> Any:
        """Exact show id, else unique id-prefix, else None."""
        ref = (ref or "").strip()
        hit = store.get_show(ref)
        if hit is not None:
            return hit
        matches = [s for s in store.list_shows()
                   if str(s.get("id", "")).startswith(ref)]
        return matches[0] if len(matches) == 1 else None

    def _resolve_episode(self, store: Any, ref: str) -> Any:
        """Exact episode id, else unique id-prefix, else None."""
        ref = (ref or "").strip()
        hit = store.get_episode(ref)
        if hit is not None:
            return hit
        matches = []
        for show in store.list_shows():
            matches.extend(
                ep for ep in store.list_episodes(str(show.get("id", "")))
                if str(ep.get("id", "")).startswith(ref))
        return matches[0] if len(matches) == 1 else None

    def _control_show(self, tail: str, *, chat_key: str = "") -> str:
        """Shows: /show list|create|show|episode|episodes|rss|publish|delete|remove.

        The audit gap R1-4 organ (nomorals/shows/composer.py) had no
        production importer at all — no durable store, no chat
        command, no way to create, keep, or publish a show. This is
        its surface. Shows + composed episodes live in
        <home>/data/shows.json (atomic tmp+replace; a corrupt file is
        quarantined and the load raises instead of silently starting
        empty). Owner-only: the kind is absent from PUBLIC_COMMANDS,
        so caller_may_run denies non-owners before this runs.
        Composing never generates scripts — the caller supplies the
        segments (``kind:text`` fields, kind defaults to script);
        publishing delegates to the composer's publicface gate, so an
        episode stays a draft unless the owner publishes it AND it
        carries no private-data markers. The link a published episode
        gets (/p/show-<id>) is the descriptor the gate returns; no
        route serves that path yet, and the reply says so instead of
        pretending the episode is live.
        """
        from ...shows.composer import Episode, Show, compose_episode, publish_episode

        usage = ("usage: /show list | create <title> [| schedule | "
                 "voices,comma | bed_track] | show <id> | episode "
                 "<show_id> <title> [| kind:text | ...] | episodes "
                 "<show_id> | rss <episode_id> | publish <episode_id> "
                 "| delete <show_id> | remove <episode_id>")
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""
        try:
            store = self._show_store()
        except ValueError as exc:
            return str(exc)
        try:
            if verb in ("list", ""):
                shows = store.list_shows()
                if not shows:
                    return ("no shows yet — /show create <title> "
                            "[| schedule | voices,comma | bed_track] "
                            "starts one.")
                lines = [f"shows ({len(shows)}):"]
                for show in shows:
                    count = len(store.list_episodes(
                        str(show.get("id", ""))))
                    line = (f"  {show['id']}: {show['title']} "
                            f"({count} episode(s))")
                    if show.get("schedule"):
                        line += f" — {show['schedule']}"
                    lines.append(line)
                return "\n".join(lines)
            if verb == "create":
                fields = [f.strip() for f in rest.split("|")]
                title = fields[0] if fields else ""
                if not title:
                    return ("usage: /show create <title> [| schedule | "
                            "voices,comma | bed_track]")
                schedule = fields[1] if len(fields) > 1 else ""
                voices = ([v.strip() for v in fields[2].split(",")
                           if v.strip()]
                          if len(fields) > 2 and fields[2] else [])
                bed_track = fields[3] if len(fields) > 3 else ""
                show = Show(title=title, schedule=schedule,
                            voices=voices, bed_track=bed_track)
                stored = store.put_show(show.to_dict())
                return (f"show created: {stored['id']} — {stored['title']}"
                        + (f" — {stored['schedule']}"
                           if stored.get("schedule") else ""))
            if verb == "show":
                show = self._resolve_show(store, rest)
                if show is None:
                    return (f"no show matching {rest.strip()!r} — "
                            "/show list shows ids.")
                episodes = store.list_episodes(str(show.get("id", "")))
                lines = [f"{show['title']} — {show['id']}"]
                if show.get("schedule"):
                    lines.append(f"  schedule: {show['schedule']}")
                if show.get("voices"):
                    lines.append("  voices: "
                                 + ", ".join(show["voices"]))
                if show.get("bed_track"):
                    lines.append(f"  bed track: {show['bed_track']}")
                if show.get("sched_last_slot"):
                    lines.append("  last scheduled: "
                                 + str(show["sched_last_slot"]))
                if show.get("sched_note"):
                    lines.append("  schedule note: "
                                 + str(show["sched_note"]))
                if not episodes:
                    lines.append("  no episodes yet — /show episode "
                                 f"{show['id']} <title> | <script text> "
                                 "composes one.")
                else:
                    lines.append(f"  episodes ({len(episodes)}):")
                    for ep in episodes:
                        line = (f"    {ep['id']}: {ep['title']} — "
                                f"{ep.get('word_count', 0)} words, "
                                f"~{ep.get('duration_seconds', 0)}s")
                        if ep.get("published"):
                            line += " [published]"
                        lines.append(line)
                return "\n".join(lines)
            if verb == "episode":
                fields = [f.strip() for f in rest.split("|")]
                head = fields[0].split(None, 1) if fields else []
                if len(head) < 2:
                    return ("usage: /show episode <show_id> <title> "
                            "[| kind:text | ...] — kinds: script, voice, "
                            "bed, ad; a bare field is a script segment")
                show = self._resolve_show(store, head[0])
                if show is None:
                    return (f"no show matching {head[0]!r} — "
                            "/show list shows ids.")
                segments: list[dict[str, Any]] = []
                for field in fields[1:]:
                    if not field:
                        continue
                    kind, sep, value = field.partition(":")
                    head_word = kind.strip()
                    if sep and " " not in head_word and head_word not in (
                            "script", "voice", "bed", "ad"):
                        return (f"unknown segment kind {head_word!r} — "
                                "kinds: script, voice, bed, ad")
                    if not sep:
                        segments.append({"kind": "script", "text": field})
                    elif head_word in ("script", "voice", "ad"):
                        segments.append({"kind": head_word,
                                         "text": value.strip()})
                    elif head_word == "bed":
                        segments.append({"kind": "bed",
                                         "ref": value.strip()})
                    else:
                        segments.append({"kind": "script", "text": field})
                if not segments:
                    return ("an episode needs at least one segment — "
                            "/show episode <show_id> <title> | <script "
                            "text>")
                episode = compose_episode(Show(**show), segments,
                                          title=head[1])
                stored = store.put_episode(episode.to_dict())
                return (f"episode composed: {stored['id']} — "
                        f"{stored['title']} ({stored['word_count']} "
                        f"words, ~{stored['duration_seconds']}s, "
                        f"{show['title']})")
            if verb == "episodes":
                show = self._resolve_show(store, rest)
                if show is None:
                    return (f"no show matching {rest.strip()!r} — "
                            "/show list shows ids.")
                episodes = store.list_episodes(str(show.get("id", "")))
                if not episodes:
                    return (f"no episodes for {show['title']} yet — "
                            "/show episode "
                            f"{show['id']} <title> | <script text> "
                            "composes one.")
                lines = [f"episodes for {show['title']} "
                         f"({len(episodes)}):"]
                for ep in episodes:
                    line = (f"  {ep['id']}: {ep['title']} — "
                            f"{ep.get('word_count', 0)} words, "
                            f"~{ep.get('duration_seconds', 0)}s")
                    if ep.get("published"):
                        line += " [published]"
                    lines.append(line)
                return "\n".join(lines)
            if verb == "rss":
                episode = self._resolve_episode(store, rest)
                if episode is None:
                    return (f"no episode matching {rest.strip()!r} — "
                            "/show episodes <show_id> lists them.")
                return episode.get("rss_item") or "(no RSS item stored)"
            if verb == "publish":
                episode = self._resolve_episode(store, rest)
                if episode is None:
                    return (f"no episode matching {rest.strip()!r} — "
                            "/show episodes <show_id> lists them.")
                # The stored row carries publish state the dataclass
                # does not know — pass only the composer's own fields.
                kept = {k: episode[k] for k in (
                    "id", "show_id", "title", "script_text",
                    "duration_seconds", "word_count", "segments",
                    "rss_item", "created_ts") if k in episode}
                result = publish_episode(Episode(**kept),
                                         owner_approved=True)
                store.set_episode_published(
                    str(episode["id"]), published=result.published,
                    link=(result.link.url_path
                          if result.link is not None else None))
                if not result.published:
                    return (f"left as draft: {episode['title']} — "
                            f"{result.reason}")
                return (f"published: {episode['title']} — "
                        f"{result.link.url_path}. No route serves that "
                        "path yet — the link is the descriptor the gate "
                        "returns, not a live page.")
            if verb == "delete":
                show = self._resolve_show(store, rest)
                if show is None:
                    return (f"no show matching {rest.strip()!r} — "
                            "/show list shows ids.")
                store.delete_show(str(show["id"]))
                return f"deleted show: {show['title']}"
            if verb == "remove":
                episode = self._resolve_episode(store, rest)
                if episode is None:
                    return (f"no episode matching {rest.strip()!r} — "
                            "/show episodes <show_id> lists them.")
                store.delete_episode(str(episode["id"]))
                return f"removed episode: {episode['title']}"
            return usage
        except (ValueError, KeyError) as exc:
            return str(exc)

    def _life_archive(self) -> Any:
        """LifeArchive on the durable home path (life-archive wiring)."""
        from ...life.archive import LifeArchive

        data = self.context.settings.home_path / "data"
        data.mkdir(parents=True, exist_ok=True)
        return LifeArchive(str(data / "life.sqlite"))

    def _resolve_life_event(self, archive: Any, ref: str) -> Any:
        """Exact event id, else unique id-prefix, else None."""
        ref = (ref or "").strip()
        hit = archive.get(ref, owner=True)
        if hit is not None:
            return hit
        matches = [e for e in archive.timeline(limit=500, owner=True)
                   if str(e.get("id", "")).startswith(ref)]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _life_lines(events: list) -> list[str]:
        """One line per event: date, kind, text, source, id."""
        import datetime as _dt

        wat = _dt.timezone(_dt.timedelta(hours=1))
        lines = []
        for event in events:
            when = _dt.datetime.fromtimestamp(
                float(event.get("ts") or 0), tz=wat).strftime("%Y-%m-%d %H:%M")
            line = (f"  {when} — {event.get('kind', '')} — "
                    f"{event.get('text', '')}")
            if event.get("source"):
                line += f" (source: {event['source']}"
                if event.get("provenance"):
                    line += f", provenance: {event['provenance']}"
                line += ")"
            line += f" [{event.get('id', '')}]"
            lines.append(line)
        return lines

    def _control_life(self, tail: str, *, chat_key: str = "") -> str:
        """Life archive: /life list|add|show|search.

        The audit gap R1-17 organ (nomorals/life/archive.py) had no
        production importer at all — no way to log or recall a life
        event. This is its surface. Events live in
        <home>/data/life.sqlite (the organ's own durable store;
        the Mongo sidecar port is a cutover concern, not this
        wiring — the native SQLite path serves the default).
        Owner-only: the kind is absent from PUBLIC_COMMANDS, so
        caller_may_run denies non-owners before this runs, and
        every organ call passes owner=True. Append-only by design:
        the organ has no delete method and neither does this
        surface.
        """
        import datetime as _dt

        usage = ("usage: /life list [kind] [n] | add <kind> | <text> "
                 "[| source | provenance | YYYY-MM-DD] | show <id> "
                 "| search <query>")
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""
        try:
            archive = self._life_archive()
        except Exception as exc:  # noqa: BLE001 - archive never breaks chat
            return f"life archive unavailable: {exc}"
        try:
            if verb in ("list", ""):
                kind = None
                limit = 10
                for token in rest.split():
                    if token.isdigit():
                        limit = max(1, int(token))
                    elif kind is None:
                        kind = token
                rows = archive.timeline(limit=limit, kind=kind, owner=True)
                if not rows:
                    return ("no events yet — /life add <kind> <text> "
                            "records one.")
                lines = [f"life events ({len(rows)}"
                         + (f" — {kind}" if kind else "") + "):"]
                lines.extend(self._life_lines(rows))
                return "\n".join(lines)
            if verb == "add":
                fields = [f.strip() for f in rest.split("|")]
                kind = fields[0] if fields else ""
                text = fields[1] if len(fields) > 1 else ""
                if not kind or not text:
                    return ("usage: /life add <kind> | <text> [| source | "
                            "provenance | YYYY-MM-DD]")
                source = fields[2] if len(fields) > 2 and fields[2] else "chat"
                provenance = fields[3] if len(fields) > 3 else ""
                ts: float | None = None
                if len(fields) > 4 and fields[4]:
                    try:
                        day = _dt.datetime.strptime(
                            fields[4], "%Y-%m-%d")
                    except ValueError:
                        return ("bad date — use YYYY-MM-DD, e.g. "
                                "/life add milestone | won the case | "
                                "journal | court:2026-11-13 | 2026-11-13")
                    wat = _dt.timezone(_dt.timedelta(hours=1))
                    ts = day.replace(tzinfo=wat).timestamp()
                event_id = archive.record_event(
                    source, kind, text, ts=ts, provenance=provenance,
                    owner=True)
                return f"recorded: {event_id} — {kind}"
            if verb == "show":
                event = self._resolve_life_event(archive, rest)
                if event is None:
                    return (f"no event matching {rest.strip()!r} — "
                            "/life list shows ids.")
                return "\n".join(self._life_lines([event]))
            if verb == "search":
                if not rest.strip():
                    return "usage: /life search <query>"
                rows = archive.search(rest.strip(), owner=True)
                if not rows:
                    return (f"no events matching {rest.strip()!r}.")
                lines = [f"matches ({len(rows)}):"]
                lines.extend(self._life_lines(rows[:10]))
                return "\n".join(lines)
            return usage
        except (ValueError, KeyError, PermissionError) as exc:
            return str(exc)

    def _prefs_store(self) -> Any:
        """PreferenceStore on the durable home path (preferences wiring)."""
        from ...preferences.store import PreferenceStore

        data = self.context.settings.home_path / "data"
        data.mkdir(parents=True, exist_ok=True)
        return PreferenceStore(str(data / "preferences.sqlite"))

    @staticmethod
    def _prefs_parse_pairs(fields: list[str]) -> dict[str, Any] | None:
        """``axis:value`` fields → item dict (repeat axis → list)."""
        item: dict[str, Any] = {}
        for field in fields:
            if ":" not in field:
                return None
            axis, _, value = field.partition(":")
            axis = axis.strip().lower()
            value = value.strip()
            if not axis or not value:
                return None
            if axis in item:
                if isinstance(item[axis], list):
                    item[axis].append(value)
                else:
                    item[axis] = [item[axis], value]
            else:
                item[axis] = value
        return item

    def _control_prefs(self, tail: str, *, chat_key: str = "") -> str:
        """Preferences: /prefs list|show|like|dislike|rank|signals|learn.

        The audit gap R1-18 organ (nomorals/preferences/) had zero
        consumers — Devon never learned what the owner likes. This
        is its surface, and its first consumer: ``like``/``dislike``
        record explicit approve/disapprove signals through
        ``record_signal`` (approve +1.0, disapprove -1.5), and
        ``rank`` re-orders owner-supplied candidates with
        ``suggest`` — the feedback module's own contract (rank
        supplied candidates, never invent them). Weights start at
        zero and only move on real signals. ``learn`` pulls owner
        feedback out of the representation ledger through
        ``learn_from_representation``. State lives in
        <home>/data/preferences.sqlite (the organ's own durable
        store; the Mongo sidecar port is a cutover concern).
        Owner-only: the kind is absent from PUBLIC_COMMANDS, so
        caller_may_run denies non-owners before this runs.
        """
        from ...preferences.feedback import (
            learn_from_representation,
            record_signal,
            suggest,
        )
        from ...preferences.store import DOMAIN_DESCRIPTORS

        usage = ("usage: /prefs list | show <domain> | like <domain> "
                 "| axis:value ... | dislike <domain> | axis:value ... "
                 "| rank <domain> | <label>=axis:value,... | ... "
                 "| signals [domain] [n] | learn")
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""
        try:
            store = self._prefs_store()
        except Exception as exc:  # noqa: BLE001 - prefs never break chat
            return f"preferences unavailable: {exc}"
        try:
            if verb in ("list", ""):
                lines = ["preferences (learned weights):"]
                any_learned = False
                for domain in store.domains():
                    weights = store.get_weights(domain)
                    learned = sum(
                        1 for bucket in weights.values()
                        for w in bucket.values() if w != 0.0)
                    count = len(store.signals(domain))
                    if learned:
                        any_learned = True
                    lines.append(f"  {domain}: {learned} learned "
                                 f"weight(s), {count} signal(s)")
                if not any_learned:
                    lines.append("no learned weights yet — /prefs like "
                                 "<domain> | axis:value ... teaches it.")
                return "\n".join(lines)
            if verb == "show":
                domain = rest.strip().split()[0] if rest.strip() else ""
                if domain not in DOMAIN_DESCRIPTORS:
                    return ("unknown domain "
                            f"{domain!r} — known: "
                            + ", ".join(store.domains()))
                descriptor = store.describe(domain) or {}
                weights = store.get_weights(domain)
                lines = [f"{domain} — "
                         + ", ".join(descriptor.get("axes", []))]
                shown = False
                for axis in descriptor.get("axes", []):
                    for value, weight in sorted(
                            weights.get(axis, {}).items()):
                        if weight != 0.0:
                            lines.append(f"  {axis}:{value} "
                                         f"{weight:+.1f}")
                            shown = True
                if not shown:
                    lines.append("no learned weights yet — /prefs like "
                                 f"{domain} | axis:value ... teaches it.")
                return "\n".join(lines)
            if verb in ("like", "dislike"):
                fields = [f.strip() for f in rest.split("|")]
                domain = fields[0] if fields else ""
                if domain not in DOMAIN_DESCRIPTORS:
                    return ("unknown domain "
                            f"{domain!r} — known: "
                            + ", ".join(store.domains())
                            + "\n" + usage)
                item = self._prefs_parse_pairs(fields[1:])
                if not item:
                    return ("usage: /prefs " + verb + " <domain> "
                            "| axis:value [| axis:value ...]")
                descriptor = store.describe(domain) or {}
                unknown = [a for a in item
                           if a not in descriptor.get("axes", [])]
                if unknown:
                    return ("unknown axis for " + domain + ": "
                            + ", ".join(unknown) + " — known axes: "
                            + ", ".join(descriptor.get("axes", [])))
                result = record_signal(
                    store, domain, item, verb == "like", source="chat")
                lines = [f"recorded: {domain} — "
                         + ("approve (+1.0)" if verb == "like"
                            else "disapprove (-1.5)")]
                for axis, values in sorted(
                        result.get("updates", {}).items()):
                    for value, weight in sorted(values.items()):
                        lines.append(f"  {axis}:{value} now {weight:+.1f}")
                if not result.get("updates"):
                    lines.append("no known axes in the item — nothing "
                                 "learned (signal logged).")
                return "\n".join(lines)
            if verb == "rank":
                fields = [f.strip() for f in rest.split("|")]
                domain = fields[0] if fields else ""
                if domain not in DOMAIN_DESCRIPTORS:
                    return ("unknown domain "
                            f"{domain!r} — known: "
                            + ", ".join(store.domains())
                            + "\n" + usage)
                if len(fields) < 2:
                    return ("usage: /prefs rank <domain> "
                            "| <label>=axis:value,... | ...")
                candidates: list[dict[str, Any]] = []
                for index, field in enumerate(fields[1:], 1):
                    label, sep, axes_text = field.partition("=")
                    if not sep:
                        axes_text = label
                        label = f"candidate {index}"
                    pairs = [p.strip() for p in axes_text.split(",")
                             if p.strip()]
                    item = self._prefs_parse_pairs(pairs)
                    if item is None:
                        return ("bad candidate "
                                f"{field!r} — use "
                                "axis:value,axis:value")
                    item["title"] = label.strip()
                    candidates.append(item)
                ranked = suggest(store, domain, candidates)
                lines = [f"ranked ({domain}):"]
                for position, row in enumerate(ranked, 1):
                    item = row["item"]
                    lines.append(f"  {position}. {item.get('title', '?')} "
                                 f"— {row['score']:+.1f}")
                return "\n".join(lines)
            if verb == "signals":
                domain = None
                limit = 5
                for token in rest.split():
                    if token.isdigit():
                        limit = max(1, int(token))
                    elif domain is None:
                        domain = token
                if domain is not None and domain not in DOMAIN_DESCRIPTORS:
                    return ("unknown domain "
                            f"{domain!r} — known: "
                            + ", ".join(store.domains()))
                rows = store.signals(domain)
                if not rows:
                    return "no signals yet — /prefs like teaches it."
                import datetime as _dt
                import json as _json

                wat = _dt.timezone(_dt.timedelta(hours=1))
                lines = [f"signals ({len(rows)}"
                         + (f" — {domain}" if domain else "") + "):"]
                for row in rows[-limit:][::-1]:
                    when = _dt.datetime.fromtimestamp(
                        float(row.get("created_at") or 0),
                        tz=wat).strftime("%Y-%m-%d %H:%M")
                    verdict = ("approve" if row.get("approve")
                               else "disapprove")
                    item_text = _json.dumps(
                        row.get("item") or {}, ensure_ascii=False)[:120]
                    lines.append(f"  {when} {row.get('domain', '')} "
                                 f"{verdict} {row.get('delta', 0.0):+.1f} "
                                 f"[{row.get('source', '')}] {item_text}")
                return "\n".join(lines)
            if verb == "learn":
                from ...cognition.representation import _ledger_for

                settings = getattr(self.context, "settings", None)
                ledger = _ledger_for(settings)
                result = learn_from_representation(
                    store, ledger.recent(limit=200))
                out = (f"learned from the representation ledger: "
                       f"recorded {result.get('recorded', 0)}, "
                       f"skipped {result.get('skipped', 0)}.")
                if not result.get("recorded"):
                    out += ("\n  nothing carried a preference domain + "
                            "owner feedback yet — the ledger only learns "
                            "once actions carry metadata.domain and "
                            "feedback is recorded on them.")
                return out
            return usage
        except (ValueError, KeyError) as exc:
            return str(exc)

    def _presence_watch_store(self) -> Any:
        """PresenceWatchStore on the durable home path."""
        from ...social.presence_watch import PresenceWatchStore

        path = (self.context.settings.home_path / "data"
                / "presence_watch.json")
        return PresenceWatchStore(path)

    def _presence_watch_event(self, platform: str, signal: str, *,
                              user_id: str, username: str = "",
                              name: str = "", episode_id: str = "",
                              group: str = "", ts: float | None = None,
                              online: bool | None = None
                              ) -> list[dict[str, Any]]:
        """One observed presence event: check the watches, deliver
        any alert through the notifier, return the alerts delivered.

        A sensor, not a command: it never raises and never replies
        in the room — a failed delivery is ledgered by the
        notifier, never retried into a storm.  With no watch file
        at all it returns before touching the disk.  ``online``
        carries a Telegram presence transition (True = came online,
        False = went offline); for any other signal it is ignored.
        """
        try:
            from ...social.presence_watch import render_alert
            from ..notifier import notify
        except Exception:  # noqa: BLE001 - a sensor must not break intake
            return []
        try:
            store = self._presence_watch_store()
            if not store.path.exists():
                return []
            if signal == "online" and online is False:
                store.note_offline(platform, user_id=user_id,
                                   username=username, name=name)
                return []
            alert = store.check(
                platform, signal, user_id=user_id, username=username,
                name=name, episode_id=episode_id, group=group, ts=ts)
            if alert is None:
                return []
            title, body = render_alert(alert)
            notify(self.context, "presence_watch", title, body,
                   alert_subject=str(alert.get("label") or ""))
            return [alert]
        except Exception:  # noqa: BLE001 - a sensor must not break intake
            return []

    def _triage_store(self) -> Any:
        """TriageLogStore on the durable home path."""
        from ...social.triage import TriageLogStore

        path = (self.context.settings.home_path / "data"
                / "triage.json")
        return TriageLogStore(path)

    def _triage_tracker(self) -> Any:
        """RelationshipTracker on the durable home path (cached
        on the host: the tracker loads its JSON once and saves
        on every note, so one instance stays current)."""
        tracker = getattr(self, "_triage_tracker_obj", None)
        if tracker is None:
            from ...social.relationships import RelationshipTracker

            tracker = RelationshipTracker(
                path=str(self.context.settings.home_path / "data"
                         / "social_relationships.json"))
            self._triage_tracker_obj = tracker
        return tracker

    def _triage_inbound(self, message: Any) -> dict[str, Any] | None:
        """Score one inbound message, record it durably, buzz
        when the scorer says so (audit gap R1-3).

        A sensor, not a command: it never raises and never
        replies in the room — a failed delivery is ledgered by
        the notifier, never retried into a storm.  The owner's
        own messages are never triaged (his own words don't page
        him); neither are service events, which never reach this
        beat.  Closeness comes from the relationship tracker —
        and the hook feeds it: ``note_inbound`` has no other
        production caller, so a read-only lookup would score
        everyone 0 forever and the "who" signal would be dead.
        A burst refusal from the notifier is the notifier's own
        dedupe/gate, same as presence watch — never worked
        around here.
        """
        try:
            from ...social.triage import triage_message
            from ..notifier import notify
        except Exception:  # noqa: BLE001 - a sensor must not break intake
            return None
        try:
            platform = str(getattr(message.chat, "platform", "") or "")
            sender_id = str(getattr(message, "sender_id", "") or "")
            sender_name = str(getattr(message, "sender", "") or "")
            meta = getattr(message, "meta", None) or {}
            if meta.get("own_message"):
                return None
            if sender_id:
                try:
                    if self._owner_registry().is_owner(platform,
                                                       sender_id):
                        return None
                except Exception:  # noqa: BLE001 - registry fault:
                    # fall through; an unprovable sender is triaged
                    # like a stranger, never skipped silently
                    pass
            kind = str(getattr(message.chat, "kind", "") or "")
            chat_kind = {"group": "group",
                         "channel": "channel"}.get(kind, "dm")
            # Closeness from the relationship tracker (see the
            # docstring: the hook feeds it, or everyone scores 0).
            closeness = 0.0
            if sender_id:
                try:
                    tracker = self._triage_tracker()
                    person_id = (f"{platform}:{sender_id}"
                                 if platform else sender_id)
                    tracker.note_inbound(person_id,
                                         display_name=sender_name)
                    rel = tracker.get(person_id)
                    if rel is not None:
                        closeness = float(
                            getattr(rel, "closeness", 0.0) or 0.0)
                except Exception:  # noqa: BLE001 - tracker fault:
                    # triage degrades to the message alone; the
                    # record below still lands
                    closeness = 0.0
            score = triage_message(
                message, is_owner=False, closeness=closeness,
                kind=chat_kind,
                mentioned=bool(getattr(message, "mentioned", False)),
                # In a 1:1 DM a reply from the peer answers the
                # owner-or-Devon thread; in a group the replied-to
                # message is unprovable without a ledger lookup,
                # so only DMs claim it.
                reply_to_owner=bool(
                    getattr(message, "reply_to", "")
                    and chat_kind == "dm"),
            )
            store = self._triage_store()
            store.record(
                str(getattr(message.chat, "key", "") or ""),
                score.tier, score.score,
                sender=sender_name or sender_id)
            if score.buzz:
                who = sender_name or sender_id or "someone"
                snippet = " ".join(
                    (getattr(message, "text", "") or "").split())
                if len(snippet) > 60:
                    snippet = snippet[:57] + "..."
                title = f"{who}: {snippet}" if snippet else who
                body = (f"{score.tier} ({score.score:.0%}) in "
                        f"{chat_kind} — "
                        + "; ".join(score.reasons))
                notify(self.context, "triage", title, body,
                       alert_subject=who)
            return score.to_dict()
        except Exception:  # noqa: BLE001 - a sensor must not break intake
            return None

    def _control_presence(self, tail: str, *, chat_key: str = "") -> str:
        """Presence watch: /presence watch <contact> [signals] | unwatch
        <contact> [signals] | list | status.

        Census §6 superhuman #4: alert the owner when a named contact
        posts a status, comes online, or joins a channel she follows.
        Owner-only: the kind is absent from PUBLIC_COMMANDS, so
        caller_may_run denies non-owners before this runs.  The
        contact may carry an explicit platform prefix (wa: / tg:);
        without one it is the platform the command arrived on.
        Signals default to every signal the platform supports —
        never an unobservable one.  A WhatsApp online watch is
        refused at add time with the wall named: neither bridge
        forwards contact presence, so it could never fire.
        """
        from ...social.presence_watch import (
            PLATFORM_SIGNALS,
            SIGNALS,
            platform_label,
            unsupported_reason,
        )

        usage = ("usage: /presence watch <contact> [status|online|join ...]"
                 " | unwatch <contact> [status|online|join ...] | list"
                 " | status — watch a contact: wa:234... / tg:@name / a"
                 " name on this surface. Signals default to all the"
                 " platform supports.")
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""
        if verb not in ("watch", "unwatch", "list", "status"):
            return usage
        arrival = str(chat_key or "").split(":", 1)[0].lower() or "telegram"
        if verb == "list":
            store = self._presence_watch_store()
            watches = store.list_watches()
            if not watches:
                return ("no presence watches. /presence watch <contact>"
                        " [status|online|join ...] to add one.")
            lines = [f"watching {len(watches)} contact(s):"]
            for watch in watches:
                signals = ", ".join(watch.get("signals") or [])
                lines.append(
                    f"• {watch.get('contact')} "
                    f"({platform_label(str(watch.get('platform') or ''))})"
                    f": {signals}")
            if store.corrupted is not None:
                lines.append(
                    "note: the watch list was reset — a corrupt file "
                    f"was quarantined at {store.corrupted}.")
            lines.append(
                "Telegram story watches fire when status saving is on"
                " (the toggle that preserves stories).")
            return "\n".join(lines)
        if verb == "status":
            lines = ["presence watch — what can fire where:"]
            for platform in ("whatsapp", "telegram"):
                live = ", ".join(PLATFORM_SIGNALS[platform])
                lines.append(
                    f"• {platform_label(platform)}: {live}")
            lines.append(
                "WhatsApp online can't be watched: "
                + unsupported_reason("whatsapp", "online"))
            lines.append(
                "Telegram story watches fire when status saving is on.")
            store = self._presence_watch_store()
            lines.append(
                f"{len(store.list_watches())} watch(es) stored.")
            if store.corrupted is not None:
                lines.append(
                    "note: the watch list was reset — a corrupt file "
                    f"was quarantined at {store.corrupted}.")
            return "\n".join(lines)
        tokens = (rest or "").strip().split()
        if not tokens:
            return usage
        contact = tokens[0]
        platform = arrival
        for prefix, name in (("wa:", "whatsapp"), ("tg:", "telegram"),
                             ("whatsapp:", "whatsapp"),
                             ("telegram:", "telegram")):
            if contact.lower().startswith(prefix):
                platform = name
                contact = contact[len(prefix):]
                break
        if not contact:
            return usage
        signals = [str(t).lower() for t in tokens[1:]]
        unknown = [s for s in signals if s not in SIGNALS]
        if unknown:
            return ("unknown signal(s): " + ", ".join(unknown)
                    + " — signals are " + ", ".join(SIGNALS) + ".")
        store = self._presence_watch_store()
        if verb == "watch":
            watch, refused = store.add_watch(platform, contact, signals)
            if watch is None:
                reasons = "; ".join(
                    f"{sig}: {reason}" for sig, reason in refused)
                return f"can't watch {contact}: {reasons}"
            out = (f"watching {watch['contact']} on "
                   f"{platform_label(platform)} for "
                   f"{', '.join(watch['signals'])}.")
            for sig, reason in refused:
                out += f"\n{sig} not watched: {reason}"
            return out
        removed = store.remove_watch(platform, contact,
                                     signals or None)
        if not removed:
            return f"no watch for {contact} on {platform_label(platform)}."
        if signals:
            return (f"stopped watching {contact} for "
                    f"{', '.join(signals)} on {platform_label(platform)}.")
        return (f"stopped watching {contact} on "
                f"{platform_label(platform)}.")

    def _publicface_store(self) -> Any:
        """PublicPageStore on the durable home path (E-6 public-face wiring)."""
        from ...publicface.publish import PublicPageStore

        path = self.context.settings.home_path / "data" / "publicface.json"
        return PublicPageStore(str(path))

    def _resolve_publicface(self, store: Any, ref: str) -> Any:
        """Exact id, else unique id-prefix, else None."""
        ref = (ref or "").strip()
        if ref in store.pages:
            return store.pages[ref]
        matches = [p for pid, p in store.pages.items()
                   if pid.startswith(ref)]
        return matches[0] if len(matches) == 1 else None

    def _control_publicface(self, tail: str, *, chat_key: str = "") -> str:
        """Public face: /publicface list|draft|approve|unpublish|remove|show.

        The MORE_TO_ADD gap 10 organ (nomorals/publicface/publish.py)
        had no durable surface at all — its only production importer,
        publish_episode in shows/composer.py, itself has no production
        callers, and the gate keeps pages in memory only. This is its
        surface. Owner-only: the kind is absent from PUBLIC_COMMANDS,
        so caller_may_run denies non-owners before this runs.
        Approving IS the approval — drafting never publishes, and the
        gate still refuses content carrying private-data markers. The
        link a published page gets (/p/<slug>) is the descriptor the
        gate returns; no route serves that path yet, and the reply says
        so instead of pretending the page is live.
        """
        from ...publicface.publish import PublicPage, publish, unpublish

        usage = ("usage: /publicface list | draft <pack> | <title> | "
                 "<content> | approve <id> | unpublish <id> | remove <id> "
                 "| show <id>")
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""
        store = self._publicface_store()
        if verb in ("list", ""):
            if not store.pages:
                return ("no pages yet — /publicface draft <pack> | <title> "
                        "| <content> records one.")
            lines = [f"pages ({len(store.pages)}):"]
            for page in store.pages.values():
                line = (f"  {page.id}: {page.title} ({page.pack}) — "
                        f"{page.status}")
                if page.status == "public":
                    line += f" — /p/{page.slug}"
                lines.append(line)
            return "\n".join(lines)
        if verb == "draft":
            fields = [f.strip() for f in rest.split("|", 2)]
            pack = fields[0] if fields else ""
            title = fields[1] if len(fields) > 1 else ""
            content = fields[2] if len(fields) > 2 else ""
            if not pack or not title or not content:
                return ("usage: /publicface draft <pack> | <title> | "
                        "<content>")
            try:
                page = store.add(PublicPage(
                    title=title, content=content, pack=pack))
            except ValueError as exc:
                return str(exc)
            return (f"draft recorded: {page.id}: {page.title} "
                    f"({page.pack}) — approve it with "
                    f"/publicface approve {page.id}")
        if verb == "show":
            page = self._resolve_publicface(store, rest)
            if page is None:
                return (f"no page matching {rest.strip()!r} — "
                        "/publicface list shows ids.")
            body = page.content
            if len(body) > 1000:
                body = body[:1000] + f"\n… ({len(page.content)} chars total)"
            lines = [
                f"{page.title} ({page.pack}) — {page.id}",
                f"  status: {page.status} · slug: {page.slug} · "
                f"created: {page.created.isoformat()}",
            ]
            if page.status == "public":
                lines.append(f"  link: /p/{page.slug}")
            lines.append("  " + body.replace("\n", "\n  "))
            return "\n".join(lines)
        if verb == "approve":
            page = self._resolve_publicface(store, rest)
            if page is None:
                return (f"no page matching {rest.strip()!r} — "
                        "/publicface list shows ids.")
            result = publish(page, owner_approved=True)
            store.put(page)
            if not result.published:
                return (f"left as draft: {page.title} — {result.reason}")
            return (f"published: {page.title} — /p/{page.slug}. "
                    "No route serves that path yet — the link is the "
                    "descriptor the gate returns, not a live page.")
        if verb == "unpublish":
            page = self._resolve_publicface(store, rest)
            if page is None:
                return (f"no page matching {rest.strip()!r} — "
                        "/publicface list shows ids.")
            unpublish(page)
            store.put(page)
            return f"back to draft: {page.title}"
        if verb == "remove":
            page = self._resolve_publicface(store, rest)
            if page is None:
                return (f"no page matching {rest.strip()!r} — "
                        "/publicface list shows ids.")
            if page.status == "public":
                return (f"cannot remove {page.title} while it is public — "
                        f"/publicface unpublish {page.id} first.")
            store.remove(page.id)
            return f"removed draft: {page.title}"
        return usage

    def _decisions_whatif_block(self, decision: str, options: list[str]) -> str:
        """Render the what-if comparison for free-text options (read-only).

        Matches each option to a world-graph node, simulates the change
        against the live graph, and names the least-disruptive option.
        The simulation never writes: the journal is untouched and no
        graph node is marked disrupted.  A failure to simulate is
        reported, never raised into the command.
        """
        from ...planning.counterfactual import render_whatif_block, whatif_for_options

        try:
            graph = self._world_graph()
            result = whatif_for_options(graph, options)
        except Exception as exc:  # noqa: BLE001 — simulation never fails a command
            return f"what-if unavailable: {exc}"
        return render_whatif_block(decision, graph, result)

    def _control_decisions(self, tail: str, *, chat_key: str = "") -> str:
        """The decision journal: /decisions list|add|resolve|review|calibration|whatif.

        Records the OWNER's decisions with their predictions BEFORE
        outcomes, so the weekly review (and this surface) can show how
        their judgement actually calibrates (AA-24 — the journal had
        no production importer at all).
        """
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""
        journal = self._decisions_journal()
        if verb in ("list", "open", ""):
            rows = journal.open_decisions()
            lines = []
            sentence = journal.calibration_sentence()
            if sentence:
                lines.append(f"🧭 {sentence}")
            if not rows:
                lines.append(
                    "no open decisions — /decisions add <decision> "
                    "[| chosen | prediction | conf%] records one with "
                    "its prediction, before the outcome.")
                return "\n".join(lines)
            lines.append(f"open decisions ({len(rows)}):")
            for d in rows:
                conf = (f" · {d.confidence:.0%} sure"
                        if d.confidence is not None else "")
                chosen = f" → {d.chosen}" if d.chosen else ""
                lines.append(f"  {d.id}: {d.decision}{chosen}{conf}")
            return "\n".join(lines)
        if verb == "add":
            fields = [f.strip() for f in rest.split("|")]
            decision = fields[0] if fields else ""
            if not decision:
                return ("usage: /decisions add <decision> "
                        "[| chosen | prediction | conf% | opt1; opt2; ...]")
            chosen = fields[1] if len(fields) > 1 else ""
            prediction = fields[2] if len(fields) > 2 else ""
            options = ([o.strip() for o in fields[4].split(";") if o.strip()]
                       if len(fields) > 4 and fields[4] else [])
            confidence = None
            if len(fields) > 3 and fields[3]:
                raw_conf = fields[3].rstrip("%").strip()
                try:
                    confidence = float(raw_conf)
                except ValueError:
                    return (f"confidence {fields[3]!r} is not a number — "
                            "use a percent like 70 or a fraction like 0.7")
                if confidence > 1.0:
                    confidence = confidence / 100.0
            try:
                d = journal.record(
                    decision=decision, chosen=chosen,
                    prediction=prediction, confidence=confidence,
                    options=options)
            except ValueError as exc:
                return str(exc)
            reply = (f"recorded: {d.id}: {d.decision}"
                     + (f" → {d.chosen}" if d.chosen else "")
                     + " — resolve it when the outcome lands: "
                       f"/decisions resolve {d.id} [good|bad] <outcome>")
            if options:
                # The comparison runs at record time, before the decision
                # is settled: the owner sees what each option would disturb
                # while their pick is still just a pick.
                reply += "\n" + self._decisions_whatif_block(d.decision, options)
            return reply
        if verb == "resolve":
            bits = rest.strip().split(None, 1)
            if not bits:
                return "usage: /decisions resolve <id> [good|bad] <outcome>"
            d = self._resolve_decision(journal, bits[0])
            if d is None:
                return (f"no decision matching {bits[0]!r} — "
                        "/decisions list shows ids.")
            tail_text = bits[1].strip() if len(bits) > 1 else ""
            success: bool | None = None
            outcome = tail_text
            lowered = tail_text.lower()
            for word, verdict in (("good", True), ("bad", False)):
                if lowered == word or lowered.startswith(word + " "):
                    success = verdict
                    outcome = tail_text[len(word):].strip()
                    break
            try:
                d = journal.resolve(d.id, outcome=outcome, success=success)
            except KeyError as exc:
                return str(exc)
            lines = [f"resolved: {d.decision} → {d.outcome or 'no outcome noted'}"]
            sentence = journal.calibration_sentence()
            if sentence:
                lines.append(f"🧭 {sentence}")
            return "\n".join(lines)
        if verb == "whatif":
            if not rest.strip():
                return "usage: /decisions whatif <id>"
            d = self._resolve_decision(journal, rest.strip().split(None, 1)[0])
            if d is None:
                return (f"no decision matching {rest.strip()!r} — "
                        "/decisions list shows ids.")
            if not d.options:
                return (f"no options recorded for {d.id} — record them with "
                        "/decisions add <decision> | chosen | prediction "
                        "| conf% | opt1; opt2")
            return self._decisions_whatif_block(d.decision, list(d.options))
        if verb == "review":
            period = rest.strip().lower() or "weekly"
            if period not in ("weekly", "monthly"):
                return "usage: /decisions review [weekly|monthly]"
            return journal.render_review(period)
        if verb == "calibration":
            sentence = journal.calibration_sentence()
            if sentence is None:
                return ("no resolved decisions yet — nothing to "
                        "calibrate. Record one with /decisions add, "
                        "then resolve it when the outcome lands.")
            return f"🧭 {sentence}"
        return ("usage: /decisions list · add <decision> "
                "[| chosen | prediction | conf% | opt1; opt2; ...]\n"
                "       resolve <id> [good|bad] <outcome> · "
                "review [weekly|monthly] · calibration · whatif <id>")

    # ── backups: /backup ──────────────────────────────────────────────
    def _control_backup(self, tail: str, *, chat_key: str = "") -> str:
        """Backup health: /backup status | verify.

        Read-only. A full backup run needs a vault-configured
        :class:`BackupCoordinator` (the encryption key lives in the vault,
        never in chat), so runs stay with the scheduled backup job —
        chat reports health and verifies what exists. The full restore
        drill likewise needs the vault key and is not runnable from chat.
        """
        from ...backup.manifest import (
            MANIFEST_NAME,
            load_manifest,
            verify_manifest_files,
        )
        from ...backup.verify import verify_database_file

        verb = (tail or "").strip().split(None, 1)[0].lower() \
            if (tail or "").strip() else "status"
        if verb not in ("status", "verify"):
            return ("usage: /backup status — generations + latest manifest + DB integrity\n"
                    "       /backup verify — integrity check + manifest file check")
        backup_dir = self.context.settings.backup_dir
        gens: list[str] = []
        if backup_dir.is_dir():
            gens = sorted(p.name for p in backup_dir.iterdir()
                          if p.is_dir())
        latest_manifest: dict[str, Any] | None = None
        latest_error = ""
        if gens:
            try:
                latest_manifest = load_manifest(
                    backup_dir / gens[-1] / MANIFEST_NAME)
            except Exception as exc:  # noqa: BLE001 - bad manifest is a finding
                latest_error = str(exc)
        db_path = self.context.settings.db_path
        try:
            db = verify_database_file(db_path)
        except Exception as exc:  # noqa: BLE001 - unreadable DB is a finding
            db = {"integrity": f"UNREADABLE ({exc})", "tables": -1,
                  "schema_version": -1, "row_counts": {}}
        if verb == "status":
            lines = [f"backups in {backup_dir}: {len(gens)} generation(s)"]
            if gens:
                lines.append(f"  latest: {gens[-1]}")
            if latest_manifest is not None:
                lines.append(
                    f"  manifest: {latest_manifest.get('backup_id')} · "
                    f"{latest_manifest.get('file_count')} files · "
                    f"{latest_manifest.get('total_encrypted_bytes')} encrypted bytes · "
                    f"{latest_manifest.get('created_utc')}")
            elif latest_error:
                lines.append(f"  latest manifest UNREADABLE: {latest_error}")
            lines.append(
                f"database: integrity={db['integrity']} · "
                f"{db['tables']} tables · schema v{db['schema_version']}")
            return "\n".join(lines)
        # verify
        lines = [f"database {db_path}: integrity={db['integrity']} · "
                 f"{db['tables']} tables · schema v{db['schema_version']}"]
        counts = db.get("row_counts") or {}
        if counts:
            lines.append("row counts: " + ", ".join(
                f"{k}={v}" for k, v in list(counts.items())[:12]))
        if latest_manifest is not None and gens:
            problems = verify_manifest_files(
                backup_dir / gens[-1], latest_manifest)
            if problems:
                lines.append("manifest file problems:")
                lines.extend(f"  - {p}" for p in problems[:10])
            else:
                lines.append(
                    f"manifest files OK ({latest_manifest.get('file_count')} files)")
        elif latest_error:
            lines.append(f"latest manifest UNREADABLE: {latest_error}")
        else:
            lines.append("no backup generations to verify yet.")
        return "\n".join(lines)
