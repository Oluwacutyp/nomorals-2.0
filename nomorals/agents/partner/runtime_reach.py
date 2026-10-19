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

_log = get_logger(__name__)

#: Temp-mail inboxes live at most a day; temp numbers a couple of hours.
#: KV TTL expiry makes a stale inbox read as "none" — the honest answer
#: is "run /tempmail new", never a resurrected address.
_TEMP_MAIL_TTL_S = 86400.0
_TEMP_SMS_TTL_S = 7200.0
#: Chat replies stay bounded — a full email body is truncated, never dumped.
_MAX_BODY_CHARS = 2000


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
        """Temp inbox: /tempmail new | status | forget (bare shows it)."""
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
                    "token": addr.token}, _TEMP_MAIL_TTL_S):
                return "temp-mail storage is off in this session."
            return (f"temp inbox: {addr.address} (via {addr.provider})\n"
                    "send mail to it, then /checkmail to read it. "
                    "It expires in 24h; /tempmail forget drops it now.")
        if verb == "forget":
            self._temp_forget("mail", message, chat_key)
            return "temp inbox forgotten."
        if verb in ("status", ""):
            rec = self._temp_load("mail", message, chat_key)
            if rec is None:
                if verb == "status":
                    return "no temp inbox here — /tempmail new makes one."
                return ("usage: /tempmail new — get a temp inbox\n"
                        "       /checkmail — read it\n"
                        "       /tempmail status | forget")
            return (f"temp inbox: {rec.get('address')} "
                    f"(via {rec.get('provider')})\n"
                    "/checkmail to read it · /tempmail new for a fresh one.")
        return ("usage: /tempmail new — get a temp inbox\n"
                "       /checkmail — read it\n"
                "       /tempmail status | forget")

    def _control_checkmail(self, tail: str, *, chat_key: str = "",
                           message: Any = None) -> str:
        """Read the caller's temp inbox: /checkmail [read <n> | code | forget]."""
        from ...accounts import temp_mail as _tm

        rec = self._temp_load("mail", message, chat_key)
        if rec is None:
            return "no temp inbox here (it may have expired) — /tempmail new makes one."
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
                except Exception:  # noqa: BLE001 - one bad read skips
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
        if parts and parts[0].lower() == "run":
            rest = parts[1].strip() if len(parts) > 1 else ""
            return self._control_browser_run(
                rest, chat_key=chat_key, message=message)
        if len(parts) < 2:
            return ("usage: /browser open <url> — fetch + show the text\n"
                    "       /browser text <url> — page text only\n"
                    "       /browser links <url> — outgoing links\n"
                    "       /browser forms <url> — forms + fields\n"
                    "       /browser takeover [url] — takeover sessions "
                    "(owner-only)\n"
                    "       /browser run <url> [goal] — drive a live tab "
                    "with the agent loop (owner-only)")
        verb, url = parts[0].lower(), parts[1].strip()
        if verb not in ("open", "text", "links", "forms", "headings"):
            return ("usage: /browser open|text|links|forms|headings <url>\n"
                    "       /browser takeover [url] — takeover sessions "
                    "(owner-only)\n"
                    "       /browser run <url> [goal] — agent loop on a "
                    "live tab (owner-only)")
        if not url.startswith(("http://", "https://")):
            return "give a full URL starting with http:// or https://"
        from ...tools.browser import BrowserSession
        try:
            sess = BrowserSession()
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

    def _control_browser_run(self, tail: str, *, chat_key: str = "",
                              message: Any = None) -> str:
        """Owner-only: ``/browser run <url> [goal]``.

        The production entry point for the wired agent loop: open a
        live rendered tab and drive it with an ``AgentLoop`` whose
        ``on_blocked`` is the fully-wired ``ChallengeSolver`` handler
        (takeover manager + owner notifier + challenge memory +
        solved-session store). When the loop hits a wall it works the
        ladder, escalates to the owner's Telegram as a last resort,
        and PAUSES for resolution — this command blocks while waiting
        for the owner (bounded by the takeover wait timeout), then
        resumes where it left off on a verified clear, or stops
        honestly on still-blocked. Decisions are the loop's heuristic
        fallback (free-gain clicks) until an LLM decider is plugged
        in — the wall path, not the decider, is what this verb wires.
        """
        if message is None or not self._is_operator(message):
            return "browser run is owner-only — run it from the owner's chat."
        rest = (tail or "").strip()
        if not rest:
            return "usage: /browser run <url> [goal] — e.g. /browser run https://example.test/ browse"
        bits = rest.split(None, 1)
        url = bits[0]
        goal = bits[1].strip() if len(bits) > 1 else "browse"
        if not url.startswith(("http://", "https://")):
            return "give a full URL starting with http:// or https://"
        try:
            service = self._browser_service_for_takeover()
            tab = service.open_rendered_tab("browser-run", url)
        except Exception as exc:  # noqa: BLE001 - playwright/config errors
            return f"browser run failed: could not open a live tab: {exc}"[:500]
        try:
            from ...browser.agent import CONSEQUENTIAL_KINDS
            from ...browser.challenge_memory import ChallengeMemory
            from ...browser.loop_wiring import build_wired_agent_loop

            def _approve_benign(decision: Any, _obs: Any) -> bool:
                # Benign acts (navigate/click around) proceed;
                # consequential ones (spend/post/delete/irreversible)
                # are denied — no human approver is attached, so they
                # stop the loop instead of running.
                return (not decision.irreversible
                        and decision.action.kind not in CONSEQUENTIAL_KINDS)

            memory = ChallengeMemory(
                self.settings.resolve("data/takeover")
                / "challenge-memory.sqlite3")
            loop = build_wired_agent_loop(
                tab=tab, takeover_manager=self._takeover_manager(),
                notifier=self._takeover_notifier(), memory=memory,
                session_store=service.solved_session_store(),
                owner_chat=chat_key or "",
                task_label=f"browser run {url}",
                resume_state={"url": url, "goal": goal},
                max_steps=30, approve=_approve_benign)
            result = loop.run(url, goal=goal)
        except Exception as exc:  # noqa: BLE001 - loop errors are data
            return f"browser run failed: {exc}"[:500]
        return (f"browser run done: {result.get('steps', 0)} steps, "
                f"stopped: {result.get('stopped_reason', '')}, "
                f"final URL: {result.get('final_url', url)}")

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

    @staticmethod
    def _slug(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
        return slug or ""

    def _control_people(self, tail: str, *, chat_key: str = "") -> str:
        """The people spine: /people list | show <name> | add <name> | note <name> <text>."""
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
                return ("usage: /people list [n] · /people show <name>\n"
                        "       /people add <name> · /people note <name> <text>")
        except Exception as exc:  # noqa: BLE001 - sqlite errors surface honestly
            return f"people failed: {exc}"[:500]

    # ── commitment ledger: /commitments ────────────────────────────────
    def _commitments_ledger(self) -> Any:
        """CommitmentLedger on the durable home path."""
        from ...commitments.ledger import CommitmentLedger

        path = self.context.settings.home_path / "data" / "commitments.json"
        return CommitmentLedger(str(path))

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
        """Focus sessions + week review: /focus start|pause|resume|stop|status|today|week."""
        from ...focus.sessions import (
            build_weekly_review, format_duration)

        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else "status"
        rest = parts[1] if len(parts) > 1 else ""
        timer = self._focus_timer()
        if verb == "start":
            try:
                s = timer.start(label=rest.strip())
            except ValueError as exc:
                return str(exc)
            label = f" ({s.label})" if s.label else ""
            return f"focus started{label} — /focus stop when done, /focus pause to pause."
        if verb == "pause":
            try:
                s = timer.pause()
            except (KeyError, ValueError) as exc:
                return str(exc)
            return f"paused ({s.label or 'untitled'}) — /focus resume to continue."
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
            return f"stopped ({s.label or 'untitled'}) — {total} focused."
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
            ledger = self._commitments_ledger()
            review = build_weekly_review(commitments=ledger, focus=timer)
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
        return ("usage: /focus start [label] · pause · resume · stop [note]\n"
                "       /focus status · today · week")

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
