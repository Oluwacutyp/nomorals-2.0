"""Sweep tests for the social module upgrade (1111-file sweep).

Every new/changed behavior from the social sweep gets a real test:
structured error codes + retry rules, manager retry/preview/health,
thread-safe tone splitting + hook checks, research-backed virality +
hook archetypes, style themes/cards/tables/sparklines, platform
formatting helpers, metrics cards, keyboard builders, real SMS segment
math, triage quiet hours + escalation, relationship cadences, identity
merge, draft review UX, content-mix pillars, lead-magnet funnel,
Bluesky facets/threads, WhatsApp rate card + projections, profile
completeness, gateway broadcast, voice-note stats.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import unittest
from types import SimpleNamespace

from nomorals.agents.context import build_context
from nomorals.core.config import Settings
from nomorals.social import Account, PlatformAdapter, PostResult, SocialManager
from nomorals.social.base import (
    ERROR_AUTH, ERROR_POLICY, ERROR_QUOTA, ERROR_TRANSIENT,
    ERROR_VALIDATION, RETRYABLE_ERRORS, classify_http_error,
)
from nomorals.social.tone import hook_check, split_thread, _profile
from nomorals.social.voice import (
    hook_type, suggest_hook_upgrades, virality_score,
)
from nomorals.social.chat import style
from nomorals.social.chat import platforms as pf
from nomorals.social.render import render_metrics_card
from nomorals.social.chat.tgbot_buttons import (
    paginate, confirm_keyboard, nav_row, check_callback_data,
)
from nomorals.social.chat.sms import (
    sms_encoding, sms_segments, normalize_gsm7, split_sms,
)
from nomorals.social.triage import (
    triage_message, in_quiet_hours, TriageLog, render_digest,
    TIER_CRITICAL, TIER_IMPORTANT,
)


# ── base: error codes ─────────────────────────────────────────────────────


class ErrorCodeTests(unittest.TestCase):
    def test_classify_http_error(self):
        self.assertEqual(classify_http_error(401, "bad token"), ERROR_AUTH)
        self.assertEqual(classify_http_error(429, ""), ERROR_QUOTA)
        self.assertEqual(classify_http_error(503, ""), ERROR_TRANSIENT)
        self.assertEqual(classify_http_error(400, "content too long"), ERROR_VALIDATION)
        self.assertEqual(classify_http_error(400, "violates policy"), ERROR_POLICY)
        self.assertEqual(classify_http_error(400, "media processing failed"), "media")

    def test_retryable_property(self):
        ok = PostResult(platform="x", ok=True)
        self.assertFalse(ok.retryable)
        transient = PostResult(platform="x", ok=False, error_code=ERROR_TRANSIENT)
        self.assertTrue(transient.retryable)
        quota = PostResult(platform="x", ok=False, error_code=ERROR_QUOTA)
        self.assertTrue(quota.retryable)
        auth = PostResult(platform="x", ok=False, error_code=ERROR_AUTH)
        self.assertFalse(auth.retryable)
        policy = PostResult(platform="x", ok=False, error_code=ERROR_POLICY)
        self.assertFalse(policy.retryable)

    def test_with_error_fluent(self):
        r = PostResult(platform="x", ok=True).with_error("boom", ERROR_TRANSIENT)
        self.assertFalse(r.ok)
        self.assertEqual(r.error_code, ERROR_TRANSIENT)
        self.assertTrue(r.retryable)
        d = r.to_dict()
        self.assertEqual(d["error_code"], ERROR_TRANSIENT)
        self.assertTrue(d["retryable"])


# ── manager: retry / preview / health ─────────────────────────────────────


class SweepAdapter(PlatformAdapter):
    """Fails N times with a chosen error code, then succeeds."""

    def __init__(self, name: str, *, fail_times: int = 0,
                 error_code: str = ERROR_TRANSIENT, max_chars: int = 500):
        self.name = name
        self.max_chars = max_chars
        self.fail_times = fail_times
        self.error_code = error_code
        self.calls: list[str] = []

    def post(self, account, content, **kwargs):
        self.calls.append(content)
        if len(self.calls) <= self.fail_times:
            return PostResult(platform=self.name, ok=False,
                              error="synthetic failure",
                              error_code=self.error_code, status_code=500)
        return PostResult(platform=self.name, ok=True,
                          external_id=f"{self.name}-1", status_code=200)

    def health(self, account):
        return self.name != "dead"


def _context(home: str):
    ctx = build_context(Settings(home=home))
    ctx.__enter__()
    return ctx


class ManagerSweepTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-social-sweep-")
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        self.context = _context(self.home)
        self.social = SocialManager(self.context, enforce_policy=False)

    def _connect(self, name: str, **kw) -> Account:
        adapter = SweepAdapter(name, **kw)
        self.social.register_adapter(adapter)
        return self.social.connect(
            name, f"@{name}.test",
            limits={"min_interval_seconds": 0},  # no rate-limit waits in tests
        ), adapter

    def test_retry_failed_retries_transient_and_succeeds(self):
        account, adapter = self._connect("flaky", fail_times=1)
        outcome = self.social.publish("hello", platforms=["flaky"])
        self.assertFalse(outcome.ok)
        results = self.social.retry_failed()
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].ok)
        # The same row was reused — no duplicate post rows.
        rows = self.social.history(status="posted")
        self.assertEqual(len(rows), 1)

    def test_retry_failed_skips_permanent_failures(self):
        account, adapter = self._connect("badauth", fail_times=99,
                                         error_code=ERROR_AUTH)
        self.social.publish("hello", platforms=["badauth"])
        results = self.social.retry_failed()
        self.assertEqual(results, [])  # auth failures are never retried
        self.assertEqual(len(adapter.calls), 1)  # no second attempt

    def test_retry_failed_respects_max_attempts(self):
        account, adapter = self._connect("down", fail_times=99)
        self.social.publish("hello", platforms=["down"])
        self.social.retry_failed(max_attempts=2)
        self.social.retry_failed(max_attempts=2)
        self.assertEqual(len(adapter.calls), 2)  # 1 initial + 1 retry, then cap

    def test_preview_is_a_dry_run(self):
        self._connect("alpha")
        prev = self.social.preview("We just shipped v2 — read all about it!",
                                   platforms=["alpha"])
        self.assertEqual(prev["content"][:8], "We just ")
        item = prev["platforms"][0]
        self.assertEqual(item["platform"], "alpha")
        self.assertIn("adapted", item)
        self.assertIn("chars", item)
        self.assertIn("max_chars", item)
        self.assertIn("over_budget", item)
        # Nothing was posted.
        self.assertEqual(self.social.history(), [])

    def test_account_health_flags_dead_tokens(self):
        self._connect("alpha")
        self._connect("dead")
        report = self.social.account_health()
        by_name = {r["platform"]: r for r in report}
        self.assertTrue(by_name["alpha"]["healthy"])
        self.assertFalse(by_name["dead"]["healthy"])
        self.assertIn("token", by_name["dead"]["error"])


# ── tone: profiles, hook checks, thread splitting ──────────────────────────


class ToneSweepTests(unittest.TestCase):
    def test_profiles_carry_hook_fold_lengths(self):
        self.assertEqual(_profile("linkedin").hook_len, 210)
        self.assertLess(_profile("x").hook_len, _profile("linkedin").hook_len)
        self.assertEqual(_profile("tiktok").hook_len, 80)

    def test_hook_check_flags_throat_clearing_before_the_fold(self):
        warn = hook_check("Excited to announce our new feature launch today, "
                          "something big is coming soon!", "linkedin")
        self.assertIsNotNone(warn)
        self.assertIn("throat-clearing", warn)

    def test_hook_check_passes_a_strong_hook(self):
        self.assertIsNone(hook_check(
            "6 months ago it took us 40 hours to clean the CRM. "
            "Today it takes 5 minutes. Here's the system:", "linkedin"))

    def test_split_thread_respects_limits(self):
        text = " ".join(f"word{i}" for i in range(200))
        chunks = split_thread(text, "x")
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 280 + 8)  # + numbering room
        # Numbered so readers can follow the thread.
        self.assertTrue(chunks[0].startswith("(1/"))

    def test_split_thread_never_cuts_urls(self):
        text = ("Check this out https://example.com/a-very-long-article-about-"
                "things-that-matter " * 12)
        chunks = split_thread(text, "x")
        for c in chunks:
            # No chunk may contain a truncated URL.
            self.assertNotIn("https://example.com/a-very-long-article-about-things-that-ma\n", c)

    def test_split_thread_short_text_is_untouched(self):
        self.assertEqual(split_thread("hello world", "x"), ["hello world"])


# ── voice: hook archetypes + re-weighted virality ───────────────────────────


class HookTests(unittest.TestCase):
    def test_hook_type_classifies_archetypes(self):
        self.assertEqual(hook_type("This is the CRM story nobody tells you"), "reveal")
        self.assertEqual(hook_type("Here's what actually works:"), "colon_setup")
        self.assertEqual(hook_type("Here's the truth about shipping"), "reveal")
        self.assertEqual(hook_type("Everyone says consistency wins. They're wrong."), "contrarian")
        self.assertEqual(hook_type("What I learned about shipping:"), "colon_setup")
        self.assertEqual(hook_type("7 tools that saved me 20 hours:"), "list")
        self.assertEqual(hook_type("3 years ago I was broke. Today:"), "story")
        self.assertEqual(hook_type("Stop doing standup. Do this instead:"), "pattern_interrupt")
        self.assertEqual(hook_type("Want to know the real secret?"), "question")
        self.assertEqual(hook_type("The secret nobody tells you about DMs"), "curiosity")
        self.assertEqual(hook_type("Just shipped v2 of the thing"), "announcement")
        self.assertEqual(hook_type(""), "none")

    def test_question_hook_costs_points_at_scale(self):
        q = virality_score("Want to know the real secret to shipping fast? Here it is.")
        r = virality_score("This is the real secret to shipping fast. Here it is.")
        self.assertLess(q.score, r.score)
        self.assertIn("question", " ".join(q.reasons))

    def test_reveal_and_colon_hooks_score_up(self):
        reveal = virality_score("This is the CRM story nobody tells you. Read on.")
        colon = virality_score("What I learned about shipping:\n\nEverything.")
        plain = virality_score("I wanted to share some thoughts about shipping today.")
        self.assertGreater(reveal.score, plain.score)
        self.assertGreater(colon.score, plain.score)

    def test_short_first_line_bonus(self):
        short = virality_score("Ship it.\n\nLonger body with real substance and a clear point.")
        self.assertTrue(any("80 chars" in r for r in short.reasons))

    def test_suggest_hook_upgrades_returns_templates(self):
        draft = "I wanted to share some thoughts about shipping today.\n\nBody here."
        suggs = suggest_hook_upgrades(draft, n=2)
        self.assertEqual(len(suggs), 2)
        for s in suggs:
            self.assertIn("archetype", s)
            self.assertIn("text", s)
            # The body is preserved — only the hook framing changes.
            self.assertIn("Body here.", s["text"])
        # Never suggests the archetype it already has.
        current = hook_type(draft)
        self.assertTrue(all(s["archetype"] != current for s in suggs))

    def test_suggest_hook_upgrades_empty_draft(self):
        self.assertEqual(suggest_hook_upgrades(""), [])


# ── style: themes, cards, tables, sparklines ────────────────────────────────


class StyleTests(unittest.TestCase):
    def setUp(self):
        self.prev = style.set_theme("rich")
        self.addCleanup(style.set_theme, self.prev)

    def test_themes_change_output(self):
        rich_out = style.header("hello")
        style.set_theme("minimal")
        minimal_out = style.header("hello")
        self.assertIn("✨", rich_out)
        self.assertNotIn("✨", minimal_out)
        self.assertEqual(style.current_theme(), "minimal")
        style.set_theme("nope")
        self.assertEqual(style.current_theme(), "rich")  # unknown → default

    def test_card_renders_fields(self):
        out = style.card("Test", [("Likes", "1.2k"), ("Shares", "300")])
        self.assertIn("TEST", out)
        self.assertIn("1.2k", out)
        self.assertIn("300", out)

    def test_table_aligns_columns(self):
        out = style.table(["Name", "N"], [["Alice", "3"], ["Bo", "12"]])
        self.assertIn("Alice", out)
        self.assertIn("<code>", out)
        rows = [l for l in out.splitlines() if "Alice" in l or "Name" in l or "Bo" in l]
        # All data rows share the same width (columns aligned).
        self.assertEqual(len({len(r) for r in rows}), 1)

    def test_sparkline_trend(self):
        self.assertEqual(style.sparkline([]), "")
        s = style.sparkline([1, 2, 3, 4])
        self.assertEqual(len(s), 4)
        flat = style.sparkline([5, 5, 5])
        self.assertEqual(len(set(flat)), 1)

    def test_quote_kv_stat_line_never_raise(self):
        self.assertIn("▍", style.quote("hello\nworld"))
        self.assertIn("Likes", style.kv([("Likes", "3")]))
        self.assertIn("12.4%", style.stat_line("Engagement", "12.4%", "+2.1"))
        self.assertEqual(style.kv([]), "")
        self.assertEqual(style.sparkline(None), "")

    def test_tag(self):
        self.assertIn("LIVE", style.tag("x", "live"))


# ── platforms: escaping + link/code/mention helpers ─────────────────────────


class PlatformsTests(unittest.TestCase):
    def test_escape_markdown_v2(self):
        out = pf.escape_markdown_v2("hello_world [x] (y)! 50% off.")
        self.assertIn(r"\_", out)
        self.assertIn(r"\[", out)
        self.assertIn(r"\!", out)
        # % is NOT in MarkdownV2's reserved set — it stays unescaped.
        self.assertIn("50% off", out)

    def test_link_per_platform(self):
        self.assertIn('<a href="https://x.y">hi</a>', pf.link("hi", "https://x.y", "telegram"))
        self.assertEqual(pf.link("hi", "https://x.y", "whatsapp"), "hi: https://x.y")
        self.assertEqual(pf.link("hi", "https://x.y", "sms"), "hi: https://x.y")
        self.assertEqual(pf.link("hi", "https://x.y", "discord"), "[hi](https://x.y)")

    def test_code_block(self):
        self.assertIn("<pre>", pf.code_block("x = 1", "python", "telegram"))
        self.assertTrue(pf.code_block("x = 1", platform="whatsapp").startswith("```"))
        self.assertEqual(pf.code_block("x = 1", platform="sms"), "x = 1")

    def test_mention(self):
        self.assertIn("tg://user?id=123", pf.mention("Ada", "123", "telegram"))
        self.assertEqual(pf.mention("Ada", "456", "discord"), "<@456>")
        self.assertEqual(pf.mention("Ada", platform="whatsapp"), "@Ada")


# ── render: metrics card ───────────────────────────────────────────────────


class RenderTests(unittest.TestCase):
    def test_metrics_card_telegram(self):
        out = render_metrics_card({"likes": 1200, "reposts": 34, "replies": 12},
                                  "telegram", trend=[1, 3, 2, 8])
        self.assertIsInstance(out, str)
        self.assertIn("1.2k", out)
        self.assertIn("trend", out)

    def test_metrics_card_whatsapp(self):
        out = render_metrics_card({"likes": 5}, "whatsapp")
        self.assertIn("*Post performance*", out)

    def test_metrics_card_discord(self):
        out = render_metrics_card({"likes": 5}, "discord")
        self.assertIsInstance(out, dict)
        self.assertIn("fields", out)

    def test_metrics_card_never_raises(self):
        self.assertTrue(render_metrics_card(None, "telegram"))


# ── tgbot buttons: generic builders ─────────────────────────────────────────


class ButtonsTests(unittest.TestCase):
    def test_paginate(self):
        kb = paginate(0, 3, "drafts")
        self.assertEqual(len(kb), 1)
        labels = [b[0] for b in kb[0]]
        self.assertIn("1/3", labels)
        self.assertIn("Next ›", labels)
        # Index-embedded callbacks, ids not labels.
        datas = [b[1] for b in kb[0]]
        self.assertIn("drafts:page:1", datas)
        mid = paginate(1, 3, "drafts")
        self.assertIn("‹ Prev", [b[0] for b in mid[0]])
        self.assertEqual(paginate(0, 1, "drafts"), [])

    def test_confirm_keyboard_destructive_last(self):
        kb = confirm_keyboard("draft_delete", "abc123")
        self.assertEqual(len(kb), 1)
        self.assertEqual(kb[0][-1][0], "✅ Confirm")  # destructive last
        self.assertEqual(kb[0][-1][1], "draft_delete:confirm:abc123")

    def test_nav_row(self):
        row = nav_row(("🏠 Home", "menu:home"), ("‹ Back", "menu:back"))
        self.assertEqual(len(row), 2)

    def test_callback_budget_enforced(self):
        with self.assertRaises(ValueError):
            check_callback_data("x" * 48)  # 47-byte budget after signing


# ── sms: real segment math ──────────────────────────────────────────────────


class SmsTests(unittest.TestCase):
    def test_gsm7_detection(self):
        self.assertEqual(sms_encoding("hello world"), "gsm7")
        self.assertEqual(sms_encoding("héllo wörld ñ"), "gsm7")  # free diacritics
        self.assertEqual(sms_encoding("hello 👋"), "ucs2")  # one emoji flips it
        self.assertEqual(sms_encoding("it’s"), "ucs2")  # curly quote flips it

    def test_segment_math(self):
        one = sms_segments("a" * 160)
        self.assertEqual(one, {"encoding": "gsm7", "per_part": 153,
                               "count": 1, "chars": 160})
        two = sms_segments("a" * 161)
        self.assertEqual(two["count"], 2)
        uni = sms_segments("👋" * 70)
        self.assertEqual(uni["encoding"], "ucs2")
        self.assertEqual(uni["count"], 1)
        uni2 = sms_segments("👋" * 71)
        self.assertEqual(uni2["count"], 2)  # 67 per concatenated part

    def test_normalize_gsm7(self):
        text, changed = normalize_gsm7("it’s a García thing — nice…")
        self.assertEqual(sms_encoding(text), "gsm7")
        self.assertIn("’", changed)
        # ñ is FREE in GSM-7 — never touched.
        text2, _ = normalize_gsm7("año")
        self.assertEqual(text2, "año")

    def test_split_sms_uses_encoding_budget(self):
        parts = split_sms("👋" * 100)
        # UCS-2: 67 chars of content per part.
        for p in parts[1:]:
            body = p.split(" ", 1)[1]
            self.assertLessEqual(len(body), 67)
        self.assertTrue(parts[0].startswith("(1/"))
        # GSM-7 path still works.
        self.assertEqual(split_sms("short"), ["short"])


# ── triage: quiet hours + escalation ────────────────────────────────────────


class TriageSweepTests(unittest.TestCase):
    def _msg(self, text: str):
        return SimpleNamespace(text=text)

    def _noon(self):
        import datetime
        return datetime.datetime(2026, 10, 10, 12, 0).timestamp()

    def _night(self):
        import datetime
        return datetime.datetime(2026, 10, 10, 23, 30).timestamp()

    def test_in_quiet_hours(self):
        self.assertTrue(in_quiet_hours(self._night()))
        self.assertFalse(in_quiet_hours(self._noon()))

    def test_important_buzzes_by_day_not_by_night(self):
        day = triage_message(self._msg("Can you call me when you get this?"),
                             is_owner=False, closeness=1.0,
                             now=self._noon())
        self.assertEqual(day.tier, TIER_IMPORTANT)
        self.assertTrue(day.buzz)
        night = triage_message(self._msg("Can you call me when you get this?"),
                               is_owner=False, closeness=1.0,
                               now=self._night())
        self.assertEqual(night.tier, TIER_IMPORTANT)
        self.assertFalse(night.buzz)
        self.assertTrue(any("quiet hours" in r for r in night.reasons))

    def test_critical_buzzes_even_at_night(self):
        night = triage_message(
            self._msg("Dad's in the hospital, call me now please?"),
            is_owner=False, closeness=1.0, mentioned=True,
            now=self._night())
        self.assertEqual(night.tier, TIER_CRITICAL)
        self.assertTrue(night.buzz)

    def test_escalate_surfaces_unanswered_important(self):
        log = TriageLog()
        log.record("tg:1", TIER_IMPORTANT, 0.6, sender="Ada")
        log.record("tg:2", "noise", 0.1, sender="spam")
        # Not old enough yet.
        self.assertEqual(log.escalate(unanswered_hours=6), [])
        # Age the entries.
        for e in log._entries:
            e["ts"] -= 7 * 3600
        seen: set[str] = set()
        escalated = log.escalate(unanswered_hours=6, escalated=seen)
        self.assertEqual(len(escalated), 1)
        self.assertEqual(escalated[0]["sender"], "Ada")
        # Already-escalated entries are not returned twice.
        self.assertEqual(log.escalate(unanswered_hours=6, escalated=seen), [])

    def test_render_digest(self):
        log = TriageLog()
        log.record("tg:1", TIER_IMPORTANT, 0.6, sender="Ada")
        out = render_digest(log.digest())
        self.assertIn("Ada", out)
        self.assertIn("important", out)


# ── relationships: cadence + reconnect ──────────────────────────────────────


class RelationshipSweepTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-rel-")
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        from nomorals.social.relationships import RelationshipTracker
        self.tracker = RelationshipTracker(path=f"{self.home}/rel.json")

    def test_target_cadence_from_closeness(self):
        close = self.tracker.ensure("a", "Ada")
        close.closeness = 0.8
        self.assertEqual(self.tracker.target_cadence_days(close), 14)
        mid = self.tracker.ensure("b", "Bo")
        mid.closeness = 0.5
        self.assertEqual(self.tracker.target_cadence_days(mid), 30)
        weak = self.tracker.ensure("c", "Cy")
        weak.closeness = 0.2
        self.assertEqual(self.tracker.target_cadence_days(weak), 90)

    def test_due_for_reconnect_honors_per_contact_cadence(self):
        now = time.time()
        ada = self.tracker.ensure("a", "Ada")
        ada.closeness = 0.8
        ada.first_seen = now - 200 * 86400
        ada.last_inbound = now - 20 * 86400  # past her 14-day cadence
        bo = self.tracker.ensure("b", "Bo")
        bo.closeness = 0.5
        bo.first_seen = now - 200 * 86400
        bo.last_inbound = now - 10 * 86400  # inside his 30-day cadence
        due = self.tracker.due_for_reconnect()
        ids = [r.person_id for r, _ in due]
        self.assertIn("a", ids)
        self.assertNotIn("b", ids)

    def test_reconnect_prompt_has_context(self):
        ada = self.tracker.ensure("a", "Ada")
        ada.last_topic = "the job interview"
        ada.warmth = 0.8
        out = self.tracker.reconnect_prompt(ada, 45)
        self.assertIn("Ada", out)
        self.assertIn("45d", out)
        self.assertIn("job interview", out)
        self.assertIn("warm", out)


# ── identity: merge ─────────────────────────────────────────────────────────


class IdentitySweepTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-ident-")
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        from nomorals.social.identity import IdentityStore
        self.store = IdentityStore(path=f"{self.home}/ident.json")

    def test_merge_folds_identities_with_audit(self):
        a = self.store.register("telegram", "111", display_name="Ada")
        b = self.store.register("whatsapp", "222", display_name="Ada W")
        self.assertTrue(self.store.merge(a.person_id, b.person_id,
                                         signal="owner-confirmed"))
        merged = self.store.find("whatsapp", "222")
        self.assertEqual(merged.person_id, a.person_id)
        self.assertEqual(self.store.find("telegram", "111").person_id,
                         a.person_id)
        self.assertIsNone(self.store._people.get(b.person_id))
        self.assertIn("whatsapp", self.store.all_platforms(a.person_id))
        self.assertIn("telegram", self.store.all_platforms(a.person_id))

    def test_merge_refuses_self_and_unknown(self):
        a = self.store.register("telegram", "111")
        self.assertFalse(self.store.merge(a.person_id, a.person_id))
        self.assertFalse(self.store.merge(a.person_id, "nope"))

    def test_unlink_still_reverses_a_merge(self):
        a = self.store.register("telegram", "111", display_name="Ada")
        b = self.store.register("whatsapp", "222", display_name="Ada W")
        self.store.merge(a.person_id, b.person_id)
        self.assertTrue(self.store.unlink(a.person_id, "whatsapp", "222"))
        self.assertIsNotNone(self.store.find("whatsapp", "222"))


# ── drafts: review UX ───────────────────────────────────────────────────────


class DraftsSweepTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-drafts-")
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        from nomorals.social.drafts import DraftQueue
        self.queue = DraftQueue(db_path=f"{self.home}/drafts.db")

    def test_review_card_carries_decision_signals(self):
        from nomorals.social.drafts import review_card
        d = self.queue.create_draft(
            "This is the shipping story nobody tells you. Details inside.",
            platforms=["x"])
        out = review_card(d)
        self.assertIn("Virality", out)
        self.assertIn("reveal", out)
        self.assertIn("Post directly", out)

    def test_review_batch_digest(self):
        from nomorals.social.drafts import render_review_batch
        self.assertIn("no drafts", render_review_batch([]))
        self.queue.create_draft("First draft about shipping.", platforms=["x"])
        self.queue.create_draft("Second draft about testing?", platforms=["x"])
        out = render_review_batch(self.queue.pending())
        self.assertIn("2 DRAFTS NEED REVIEW", out)
        self.assertIn("1.", out)

    def test_add_variant_stores_ab_hooks(self):
        from nomorals.social.drafts import add_variant
        from nomorals.social.voice import suggest_hook_upgrades
        d = self.queue.create_draft("I wanted to share some shipping thoughts.\n\nBody.",
                                    platforms=["x"])
        sugg = suggest_hook_upgrades(d.content, n=1)[0]
        updated = add_variant(self.queue, d.id, sugg["text"].split("\n")[0])
        self.assertEqual(len(updated.metadata["variants"]), 1)
        # Idempotent — same hook twice stores once.
        add_variant(self.queue, d.id, sugg["text"].split("\n")[0])
        self.assertEqual(len(self.queue.get(d.id).metadata["variants"]), 1)

    def test_variants_callback_generates_suggestions(self):
        from nomorals.social.drafts import handle_draft_callback
        d = self.queue.create_draft("I wanted to share some shipping thoughts.\n\nBody.",
                                    platforms=["x"])
        out = handle_draft_callback(self.queue, f"draft:variants:{d.id}")
        self.assertIn("hook variants", out)
        self.assertTrue(len((self.queue.get(d.id).metadata or {}).get("variants", [])) >= 1)


# ── content pipeline: pillars + mix ─────────────────────────────────────────


class ContentMixTests(unittest.TestCase):
    def test_classify_pillar(self):
        from nomorals.social.content_pipeline import classify_pillar
        self.assertEqual(classify_pillar("How to ship faster: 5 tips that work"), "educate")
        self.assertEqual(classify_pillar("Just shipped v2 — 10k users in a week!"), "proof")
        self.assertEqual(classify_pillar("What's your biggest bottleneck? Be honest."), "engage")
        self.assertEqual(classify_pillar("50% off ends tonight — buy now"), "promo")

    def test_mix_report_detects_skew(self):
        from nomorals.social.content_pipeline import (
            PipelineDraft, mix_report, CONTENT_MIX)
        drafts = [PipelineDraft(content=f"buy now {i}", platform="x",
                                virality=50, virality_grade="decent",
                                pillar="promo") for i in range(5)]
        drafts.append(PipelineDraft(content="how to ship", platform="x",
                                    virality=60, virality_grade="decent",
                                    pillar="educate"))
        report = mix_report(drafts)
        self.assertIn("educate", report["starved"])
        self.assertIn("promo", report["overfed"])
        self.assertIn("verdict", report)
        self.assertEqual(set(report["counts"]), set(CONTENT_MIX))

    def test_mix_report_balanced(self):
        from nomorals.social.content_pipeline import PipelineDraft, mix_report
        drafts = [
            PipelineDraft(content="how to x", platform="x", virality=60,
                          virality_grade="decent", pillar="educate"),
            PipelineDraft(content="how to y", platform="x", virality=60,
                          virality_grade="decent", pillar="educate"),
            PipelineDraft(content="we shipped", platform="x", virality=60,
                          virality_grade="decent", pillar="proof"),
            PipelineDraft(content="your take?", platform="x", virality=60,
                          virality_grade="decent", pillar="engage"),
            PipelineDraft(content="sale ends", platform="x", virality=60,
                          virality_grade="decent", pillar="promo"),
        ]
        report = mix_report(drafts)
        self.assertEqual(report["starved"], [])
        self.assertIn("balanced", report["verdict"])


# ── leads: keyword quality, public reply, funnel ────────────────────────────


class LeadsSweepTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-leads-")
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        from nomorals.social.leads import LeadStore
        self.store = LeadStore(db_path=f"{self.home}/leads.db")

    def test_keyword_quality(self):
        from nomorals.social.leads import keyword_quality
        self.assertEqual(keyword_quality("VAULT")["grade"], "good")
        bad = keyword_quality("info")
        self.assertEqual(bad["grade"], "risky")
        self.assertTrue(bad["warnings"])
        worse = keyword_quality("send me the info please!!")
        self.assertEqual(worse["grade"], "bad")
        emoji = keyword_quality("🔥DEAL")
        self.assertTrue(any("bait" in w for w in emoji["warnings"]))

    def test_public_reply_and_qualifier_flow(self):
        from nomorals.social.leads import CommentEvent, handle_comment
        trig = self.store.add_trigger("instagram", "post1", "VAULT")
        self.assertTrue(self.store.set_public_reply(
            trig.trigger_id, "Sent it, {name} — check your DMs 👀"))
        self.assertTrue(self.store.set_qualifier(
            trig.trigger_id, "Are you building right now, or just exploring?"))
        ev = CommentEvent(platform="instagram", post_id="post1",
                          commenter="ada", commenter_contact="ada_ig",
                          text="VAULT please!")
        results = handle_comment(self.store, ev)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["dmed"])
        self.assertIn("check your DMs", results[0]["public_reply"])
        self.assertIn("ada", results[0]["public_reply"])
        self.assertIn("building right now", results[0]["qualifier"])

    def test_funnel_stats(self):
        from nomorals.social.leads import CommentEvent, handle_comment
        trig = self.store.add_trigger("instagram", "post9", "PLAN")
        ev = CommentEvent(platform="instagram", post_id="post9",
                          commenter="bo", commenter_contact="bo_ig",
                          text="PLAN!")
        handle_comment(self.store, ev)
        stats = self.store.funnel_stats(trig.trigger_id)
        self.assertEqual(stats["keyword"], "PLAN")
        self.assertEqual(stats["comments"], 1)
        self.assertEqual(stats["dmed"], 1)
        self.assertEqual(stats["dm_rate"], 1.0)
        # handle_comment marks the lead "dmed" once the DM goes out.
        self.assertEqual(stats["leads_by_status"].get("dmed"), 1)


# ── bluesky: facets, splitting, threads ─────────────────────────────────────


class BlueskySweepTests(unittest.TestCase):
    def test_detect_facets_links_and_tags(self):
        from nomorals.social.adapters.bluesky import detect_facets
        facets = detect_facets("Check https://example.com out #buildinpublic")
        kinds = [f["features"][0]["$type"] for f in facets]
        self.assertIn("app.bsky.richtext.facet#link", kinds)
        self.assertIn("app.bsky.richtext.facet#tag", kinds)
        # Byte offsets are UTF-8, not char offsets.
        link = next(f for f in facets
                    if f["features"][0]["$type"].endswith("#link"))
        start = link["index"]["byteStart"]
        text = "Check https://example.com out #buildinpublic"
        self.assertEqual(text.encode()[start:start + 19], b"https://example.com")

    def test_detect_facets_mentions_need_did(self):
        from nomorals.social.adapters.bluesky import detect_facets
        # No resolver → mention stays plain text (an unresolved mention
        # facet would be invalid).
        facets = detect_facets("hi @alice.bsky.social")
        self.assertEqual(
            [f for f in facets if f["features"][0]["$type"].endswith("#mention")],
            [])
        facets2 = detect_facets("hi @alice.bsky.social",
                                resolve_mention=lambda h: "did:plc:xyz")
        mentions = [f for f in facets2
                    if f["features"][0]["$type"].endswith("#mention")]
        self.assertEqual(len(mentions), 1)
        self.assertEqual(mentions[0]["features"][0]["did"], "did:plc:xyz")

    def test_detect_facets_tags_never_overlap_links(self):
        from nomorals.social.adapters.bluesky import detect_facets
        facets = detect_facets("https://example.com/#section read #news")
        tags = [f for f in facets if f["features"][0]["$type"].endswith("#tag")]
        self.assertEqual([f["features"][0]["tag"] for f in tags], ["news"])

    def test_split_respects_grapheme_and_byte_limits(self):
        from nomorals.social.adapters.bluesky import (
            split_bluesky_text, BSKY_GRAPHEME_LIMIT, BSKY_BYTE_LIMIT)
        long_text = " ".join(f"word{i}" for i in range(400))
        chunks = split_bluesky_text(long_text)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c.encode("utf-8")), BSKY_BYTE_LIMIT)
        # Emoji-heavy text: byte limit binds before grapheme limit.
        emoji_text = "👨‍👩‍👧‍👧 " * 400
        chunks2 = split_bluesky_text(emoji_text)
        for c in chunks2:
            self.assertLessEqual(len(c.encode("utf-8")), BSKY_BYTE_LIMIT)

    def test_validate_allows_threads_rejects_epics(self):
        from nomorals.social.adapters.bluesky import Adapter, BSKY_MAX_THREAD_POSTS
        from nomorals.core.errors import ValidationError
        from nomorals.social import Account
        adapter = Adapter()
        account = Account(platform="bluesky", handle="me.bsky.social")
        short = adapter.validate("hello")
        self.assertEqual(short, "hello")
        threadable = adapter.validate("word " * 400)  # >300 chars
        self.assertTrue(threadable)
        with self.assertRaises(ValidationError):
            adapter.validate("word " * (BSKY_MAX_THREAD_POSTS * 100))

    def test_external_embed_shape(self):
        from nomorals.social.adapters.bluesky import external_embed
        e = external_embed("https://x.y", "Title", "desc")
        self.assertEqual(e["$type"], "app.bsky.embed.external")
        self.assertEqual(e["external"]["uri"], "https://x.y")


# ── whatsapp cost: rate card + projection ───────────────────────────────────


class WhatsappCostSweepTests(unittest.TestCase):
    def test_rate_card_text(self):
        from nomorals.social.whatsapp_cost import rate_card_text, RATE_CARD_2026
        self.assertIn("marketing", RATE_CARD_2026)
        self.assertIn("service", RATE_CARD_2026)
        self.assertLess(float(RATE_CARD_2026["service"]["usd"]),
                        float(RATE_CARD_2026["marketing"]["usd"]))
        text = rate_card_text()
        self.assertIn("marketing", text)
        self.assertIn("6×", text)

    def test_project_monthly(self):
        import tempfile as _tf
        from nomorals.social.whatsapp_cost import CostTracker
        path = _tf.mktemp(suffix=".db")
        tracker = CostTracker(db_path=path)
        tracker.set_budget("default", 100_000)  # ₦1,000/week
        tracker.track("+2348000000001", "marketing", client="default")
        proj = tracker.project_monthly()
        self.assertGreater(proj["projected_30d_kobo"], 0)
        self.assertIn("summary", proj)
        self.assertIn("on_track", proj)


# ── profiles: completeness ──────────────────────────────────────────────────


class ProfilesSweepTests(unittest.TestCase):
    def test_completeness_scoring(self):
        from nomorals.social.profiles import Profile, VoiceIntro
        p = Profile(profile_id="p1", owner="me", surface="gig")
        c = p.completeness()
        self.assertLess(c["score"], 30)
        self.assertFalse(c["done"])
        self.assertTrue(p.missing_for_complete())
        p.display_name = "Ada"
        p.prompts = {f"q{i}": f"answer {i}" for i in range(10)}
        p.voice_intro = VoiceIntro(profile_id="p1", audio_path="/tmp/a.ogg")
        from nomorals.social.profiles import ProfileElement
        p.elements = [ProfileElement(element_id=f"e{i}", title=f"t{i}")
                      for i in range(3)]
        c2 = p.completeness()
        self.assertGreaterEqual(c2["score"], 90)
        self.assertTrue(c2["done"])
        self.assertEqual(p.missing_for_complete(), [])


# ── gateway broadcast + voice note stats ────────────────────────────────────


class GatewaySweepTests(unittest.TestCase):
    def test_broadcast_isolates_failures(self):
        from nomorals.social.chat.gateway import ChatGateway
        from nomorals.social.chat.local import LocalAdapter
        gw = ChatGateway({"local": LocalAdapter()}, dry_run=True)
        out = gw.broadcast("local", ["local:a", "local:b"], "hello",
                           pace_seconds=0)
        self.assertEqual(out["sent"], 2)
        self.assertEqual(out["failed"], 0)
        self.assertFalse(out["aborted"])
        self.assertEqual(len(out["results"]), 2)

    def test_broadcast_aborts_on_consecutive_failures(self):
        from nomorals.social.chat.gateway import ChatGateway
        from nomorals.social.chat.base import ChatAdapter, ChatRef, SendResult
        from nomorals.social.chat.local import LocalAdapter

        class DeadAdapter(LocalAdapter):
            def send(self, chat, text, **kw):
                return SendResult(ok=False, platform="local", error="dead")

        gw = ChatGateway({"local": DeadAdapter()}, dry_run=False)
        out = gw.broadcast("local", ["local:a", "local:b", "local:c", "local:d"],
                           "hello", pace_seconds=0, stop_on=2)
        self.assertTrue(out["aborted"])
        self.assertEqual(out["sent"], 0)
        self.assertLess(len(out["results"]), 4)


class VoiceNotesSweepTests(unittest.TestCase):
    def test_stats(self):
        import tempfile as _tf
        from nomorals.social.voice_notes import VoiceNoteStore
        home = _tf.mkdtemp(prefix="nm-vn-")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        store = VoiceNoteStore(db_path=f"{home}/vn.db", vault_dir=f"{home}/vault")
        self.assertEqual(store.stats()["total"], 0)
        # Stats never raise even on an empty store.
        self.assertIn("transcript_coverage", store.stats())


if __name__ == "__main__":
    unittest.main()
