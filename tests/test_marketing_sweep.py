"""Sweep tests: mined-then-built upgrades across nomorals/marketing.

Covers the new capability added in the marketing sweep:
- aeo: consideration layer, sentiment, share of voice, prompt variants,
  GEO readiness, trends/sparklines, citation quality
- send: spam score, warmup, A/B testing, suppression, scheduled campaigns,
  deliverability audit, seed tests
- briefs: composite scoring, term bands, question clusters, GEO score,
  brief quality, decay watchdog, platform variants, calendar
- guardrails: multi-condition rules, presets, alert action, dry-run preview,
  creative-fatigue detection, min-spend evidence gate
- competitor: per-pillar engagement, top/viral posts, best times, hashtags,
  key insights, benchmark, content gaps, brief feed

All offline; network-touching helpers degrade gracefully. Nothing raises.
"""

import time

import pytest

from nomorals.marketing import aeo as A
from nomorals.marketing import send as S
from nomorals.marketing import briefs as B
from nomorals.marketing import guardrails as G
from nomorals.marketing import competitor as C


# ══════════════════════════════════════════════════════════════════════
# aeo — consideration, sentiment, SOV, variants
# ══════════════════════════════════════════════════════════════════════

def test_classify_positioning_recommended():
    assert A.classify_positioning("Acme is the best choice, I recommend it.") == "recommended"


def test_classify_positioning_negative():
    assert A.classify_positioning("Avoid Acme, it's a total scam.") == "negative"


def test_classify_positioning_compared():
    assert A.classify_positioning("Acme vs Rival: Acme wins on price.") == "compared"


def test_classify_positioning_mentioned_default():
    assert A.classify_positioning("Acme was founded in 2020.") == "mentioned"
    assert A.classify_positioning("") == "mentioned"


def test_sentiment_score_polarity():
    assert A.sentiment_score("amazing excellent reliable product, love it") > 0.2
    assert A.sentiment_score("terrible awful scam, horrible experience") < -0.2
    assert A.sentiment_score("the product exists and ships") == 0.0


def test_find_mentions_sets_sentiment_and_positioning():
    ms = A.find_mentions("Acme is the best widget maker around.", "Acme", "chatgpt", "p")
    assert len(ms) == 1
    assert ms[0].positioning == "recommended"
    assert ms[0].sentiment > 0


def test_expand_prompts_variants_deterministic():
    out = A.expand_prompts(["What are the best CRM tools?"])
    assert out[0] == "What are the best CRM tools?"
    assert any("top" in p for p in out[1:]), out
    # idempotent shape: canonical first, no dupes
    assert len(out) == len(set(out))


def test_expand_prompts_never_raises():
    assert A.expand_prompts([]) == []
    assert isinstance(A.expand_prompts(None), list)


def _fake_caller_factory(text="Acme is the best widget maker. See https://acme.com. Rival Corp also exists."):
    def fake(engine, prompt):
        if prompt.startswith("Answer with exactly YES or NO"):
            return "YES — it names Acme with real context."
        return text
    return fake


def test_track_visibility_share_of_voice_and_consideration():
    tr = A.AEOTracker(":memory:")
    rep = tr.track_visibility(
        "Acme", prompts=["Tell me about Acme.", "Is Acme any good?"],
        engines=["chatgpt", "claude"],
        engine_caller=_fake_caller_factory(),
        competitors=["Rival Corp"],
    )
    assert rep.share > 0
    assert rep.competitor_hits.get("Rival Corp", 0) > 0
    assert 0.0 < rep.share_of_voice < 1.0
    cons = rep.consideration
    assert cons["recommended"] >= 1
    sent = rep.sentiment_breakdown
    assert sent["positive"] >= 1
    assert "share of voice" in rep.format()
    assert "positioning" in rep.format()


def test_track_visibility_expand_variants():
    tr = A.AEOTracker(":memory:")
    rep = tr.track_visibility(
        "Acme", prompts=["What are the best widgets?"],
        engines=["chatgpt"], engine_caller=_fake_caller_factory(),
        expand_variants=True)
    assert len(rep.prompts) > 1  # variants added


def test_sparkline_and_gauge():
    sp = A.sparkline([0.1, 0.5, 0.9])
    assert len(sp) == 3
    assert A.sparkline([]) == ""
    assert A.sparkline([5, 5, 5]) == "▄▄▄"


def test_trend_report_with_history():
    tr = A.AEOTracker(":memory:")
    tr.track_visibility("Acme", prompts=["Tell me about Acme."],
                        engines=["chatgpt"], engine_caller=_fake_caller_factory())
    tr.track_visibility("Acme", prompts=["Tell me about Acme."],
                        engines=["chatgpt"], engine_caller=_fake_caller_factory())
    out = tr.trend_report("Acme")
    assert "trend" in out and "verdict" in out


def test_citation_quality_classifies_sources():
    m = A.Mention(engine="e", prompt="p", brand="Acme",
                  citations=["https://acme.com/pricing", "https://trustpilot.com/x",
                             "https://nytimes.com/article"])
    q = A.citation_quality([m], brand_domain="acme.com")
    assert q == {"own": 1, "third_party": 1, "review_sites": 1, "total": 3}


def test_site_readiness_check_bad_domain_never_raises():
    res = A.site_readiness_check("not a domain at all!!!")
    assert res["domain"] == "" or res["score"] == 0
    res2 = A.site_readiness_check("this-domain-definitely-does-not-exist-xyz.com")
    assert res2["grade"] in ("A", "B", "C", "D", "F")
    assert isinstance(res2["checks"], list)


def test_control_aeo_track_with_vs_flag():
    # no API keys configured → engines honestly unavailable, flags must parse
    out = A.control_aeo("track Acme --vs Rival1,Rival2")
    assert isinstance(out, str) and "AEO visibility" in out


def test_control_aeo_readiness_and_trend():
    assert "usage" in A.control_aeo("readiness").lower()
    assert isinstance(A.control_aeo("trend"), str)


# ══════════════════════════════════════════════════════════════════════
# send — spam, warmup, A/B, suppression, scheduling, deliverability
# ══════════════════════════════════════════════════════════════════════

def _engine():
    eng = S.SendEngine(":memory:")
    eng.register_provider("mock1", kind="mock", rate_per_minute=600)
    eng.save_template("hello", "Hi {{name}}, welcome!", "Welcome {{name}}")
    return eng


def test_spam_score_clean_vs_spammy():
    clean = S.spam_score("Your receipt", "Hi Ada, here is your receipt for order 42.")
    assert clean["score"] < 30
    spammy = S.spam_score("FREE MONEY!!!", "CONGRATULATIONS!!! Click here now for FREE cash $$$!!!")
    assert spammy["score"] >= 60
    assert spammy["issues"]


def test_spam_score_never_raises():
    res = S.spam_score(None, None)
    assert isinstance(res["score"], int) and res["verdict"] and isinstance(res["issues"], list)


def test_warmup_plan_ramp_shape():
    plan = S.warmup_plan(100, 7)
    assert len(plan) == 7
    caps = [p["cap"] for p in plan]
    assert caps[0] < caps[-1]
    assert caps[-1] == 100
    assert all(caps[i] <= caps[i + 1] for i in range(len(caps) - 1))


def test_ab_campaign_split_and_report():
    eng = _engine()
    eng.save_variant("hello", "b", "Hi {{name}} — quick question…", "Quick one {{name}}")
    res = eng.create_ab_campaign("t", "hello", ["control", "b"],
                                 ["a@x.com", "b@x.com", "c@x.com", "d@x.com"],
                                 "mock1")
    assert res["campaign_id"].startswith("cmp_")
    assert res["per_variant"] == {"control": 2, "b": 2}
    stats = eng.process()
    assert stats["sent"] == 4
    rep = eng.ab_report(res["campaign_id"])
    assert set(rep["variants"]) == {"control", "b"}
    win = eng.ab_winner(res["campaign_id"])
    assert win["winner"] in ("control", "b")


def test_variant_body_overrides_in_attempt():
    eng = _engine()
    seen = []
    eng.register_provider("cap", kind="mock", rate_per_minute=600,
                          sender=lambda to, subj, body: seen.append(body) or True)
    eng.save_variant("hello", "b", "VARIANT BODY HERE")
    eng.enqueue("a@x.com", "hello", {}, "cap", variant="b")
    eng.process()
    assert seen and seen[0] == "VARIANT BODY HERE"


def test_unsubscribe_suppresses_future_sends():
    eng = _engine()
    assert eng.unsubscribe("gone@x.com")
    assert eng.is_suppressed("gone@x.com")
    assert eng.enqueue("gone@x.com", "hello", {}, "mock1") == ""
    assert eng.suppression_count() == 1


def test_schedule_campaign_future_not_processed_until_due():
    eng = _engine()
    res = eng.schedule_campaign("hello", ["f@x.com"], "mock1",
                                time.time() + 3600)
    assert res["queued"] == 1
    stats = eng.process()
    assert stats["sent"] == 0  # not due yet


def test_warmup_cap_enforced():
    eng = _engine()
    assert eng.set_warmup("mock1", 40)
    st = eng.warmup_status("mock1")
    assert st["cap"] <= 40 and st["remaining"] > 0
    # simulate a full day of sends
    for _ in range(st["cap"]):
        eng._db.execute(
            "INSERT INTO send_log (send_id, provider, event, at) VALUES (?,?,?,?)",
            ("x", "mock1", "sent", time.time()))
    eng._db.commit()
    assert eng._warmup_ok("mock1") is False
    eng.enqueue("w@x.com", "hello", {}, "mock1")
    stats = eng.process()
    assert stats["sent"] == 0 and stats["deferred"] >= 1


def test_deliverability_check_never_raises():
    res = S.deliverability_check("definitely-not-a-real-domain-xyz123.com")
    assert res["grade"] in ("A", "B", "C", "D", "F")
    assert any(c["name"] == "SPF" for c in res["checks"])
    assert S.deliverability_check("")["score"] == 0


def test_seed_test_and_spamcheck_template():
    eng = _engine()
    res = eng.send_seed("hello", ["seed@x.com"], "mock1")
    assert res["seeded"] == 1
    sc = eng.spam_check_template("hello")
    assert sc["score"] < 30
    assert eng.spam_check_template("nope")["verdict"] == "template not found"


def test_campaign_report_funnel():
    eng = _engine()
    res = eng.create_ab_campaign("t", "hello", ["control"],
                                 ["a@x.com", "b@x.com"], "mock1")
    eng.process()
    rep = eng.campaign_report(res["campaign_id"])
    assert rep["total"] == 2
    assert rep["by_status"].get("sent") == 2


def test_control_send_new_commands():
    eng = _engine()
    assert "saved" in S.control_send("variant add hello b | Hi there!", engine=eng)
    out = S.control_send("ab hello control,b to a@x.com,b@x.com via mock1", engine=eng)
    assert "A/B test cmp_" in out
    cid = out.split("A/B test ")[1].split(" started")[0]
    assert "winner" in S.control_send(f"abreport {cid}", engine=eng).lower() or \
           "fail rate" in S.control_send(f"abreport {cid}", engine=eng)
    assert "spam score" in S.control_send("spamcheck hello", engine=eng)
    assert "deliverability" in S.control_send("deliver example.com", engine=eng)
    assert "warmup started" in S.control_send("warmup mock1 100", engine=eng)
    assert "unsubscribed" in S.control_send("unsub z@x.com", engine=eng)
    assert "seed test" in S.control_send("seed hello to s@x.com via mock1", engine=eng)
    st = S.control_send("status", engine=eng)
    assert "suppressed" in st


# ══════════════════════════════════════════════════════════════════════
# briefs — composite scoring, bands, clusters, GEO, watchdog, variants
# ══════════════════════════════════════════════════════════════════════

def _search_fn(topic):
    return [
        {"title": f"{topic} guide", "snippet": f"{topic} pricing and {topic} features explained. How does {topic} work? What is {topic} pricing?"},
        {"title": f"{topic} review", "snippet": f"Best {topic} tools reviewed. {topic} costs compared. Why choose {topic}?"},
    ]


def _store():
    return B.BriefStore(":memory:")


def test_letter_grade_boundaries():
    assert B.letter_grade(98) == "A++"
    assert B.letter_grade(95) == "A+"
    assert B.letter_grade(85) == "B+"
    assert B.letter_grade(72) == "C"
    assert B.letter_grade(10) == "F"


def test_brief_quality_scores_depth():
    st = _store()
    brief = st.build_brief("email deliverability", search_fn=_search_fn)
    q = B.brief_quality(brief)
    assert q["score"] > 40
    assert q["grade"] != "F"
    thin = B.ContentBrief(topic="x")
    assert B.brief_quality(thin)["score"] == 0.0


def test_question_clusters_grouped():
    clusters = B._cluster_questions([
        "What is email warmup?", "How does warmup work?",
        "What is SPF authentication?",
    ])
    assert len(clusters) >= 2
    brief = B.ContentBrief(questions=["What is email warmup?", "How does warmup work?"])
    assert isinstance(brief.clusters, dict)


def test_geo_score_definition_first():
    g = B.geo_score("Email warmup is the practice of gradually increasing send volume. "
                    "It builds sender reputation. Steps:\n- start small\n- ramp daily\n"
                    "A 2024 study found warmup lifts inbox placement by 30%.",
                    topic="email warmup")
    assert g["score"] >= 60, g
    g2 = B.geo_score("Some vague thoughts about stuff.", topic="email warmup")
    assert g2["score"] < g["score"]


def test_term_bands_persisted():
    st = _store()
    brief = st.build_brief("email deliverability", search_fn=_search_fn)
    assert brief.term_bands  # bands computed from source texts
    again = st.get_brief(brief.brief_id)
    assert again.term_bands == {k: list(v) for k, v in brief.term_bands.items()}


def test_score_draft_composite_fields():
    st = _store()
    brief = st.build_brief("email deliverability", search_fn=_search_fn)
    draft = ("# Email deliverability guide\n\nEmail deliverability is the ability "
             "of an email to reach the inbox. Warmup builds reputation.\n\n"
             "## How does warmup work?\n\n- start with engaged contacts\n"
             "- ramp volume daily\n- monitor bounces\n\nA 2024 study found "
             "warmup lifts placement by 30%.")
    sc = st.score_draft(draft, brief)
    assert sc.structure > 0
    assert sc.geo > 0
    assert sc.grade in ("A++", "A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F")
    assert "grade" in sc.format() and "GEO" in sc.format()
    # missing terms carry frequency-band hints
    assert any("aim" in m for m in sc.missing_terms)


def test_stale_drafts_flags_aging_unscheduled():
    st = _store()
    brief = st.build_brief("widgets", search_fn=_search_fn)
    draft = st.draft_from_brief(brief)
    assert draft is not None
    st._db.execute("UPDATE drafts SET created_at = ? WHERE draft_id = ?",
                   (time.time() - 40 * 86400, draft.draft_id))
    st._db.commit()
    stale = st.stale_drafts(30)
    assert any(d["draft_id"] == draft.draft_id for d in stale)


def test_platform_variants_thread_and_newsletter():
    st = _store()
    brief = st.build_brief("widgets", search_fn=_search_fn)
    draft = st.draft_from_brief(brief)
    v = st.platform_variants(draft.draft_id)
    assert "[1/" in v["x"]  # thread numbering
    assert v["newsletter"].startswith("Subject:")
    assert "linkedin" in v


def test_calendar_lists_scheduled():
    st = _store()
    brief = st.build_brief("widgets", search_fn=_search_fn)
    draft = st.draft_from_brief(brief)
    res = st.schedule_draft(draft.draft_id, time.time() + 7200)
    assert res["ok"]
    cal = st.calendar()
    assert any(c["draft_id"] == draft.draft_id for c in cal)


def test_control_content_new_commands():
    st = _store()
    brief = st.build_brief("widgets", search_fn=_search_fn)
    draft = st.draft_from_brief(brief)
    assert "GEO score" in B.control_content("geo some draft text here", store=st)
    out = B.control_content(f"variants {draft.draft_id}", store=st)
    assert "── x ──" in out
    assert "nothing scheduled" in B.control_content("calendar", store=st)
    assert "stale" in B.control_content("stale", store=st).lower() or \
           "fresh" in B.control_content("stale", store=st).lower()
    assert "brief quality" in brief.summary()


# ══════════════════════════════════════════════════════════════════════
# guardrails — compound rules, presets, preview, fatigue, alert
# ══════════════════════════════════════════════════════════════════════

def _gstore():
    return G.GuardrailStore(":memory:")


def test_parse_rule_compound_and():
    r = G.parse_rule("pause if cpa > ₦50000 and frequency > 4 for 3 days on adset1")
    assert r is not None
    assert r.cond_op == "and"
    assert len(r.conditions) == 2
    assert r.conditions[0]["metric"] == "cpa"
    assert r.conditions[1]["metric"] == "frequency"
    assert r.days == 3


def test_parse_rule_compound_or():
    r = G.parse_rule("alert if ctr < 0.5 or cvr < 1 for 2 days on adset1")
    assert r is not None and r.cond_op == "or" and r.action == "alert"


def test_parse_rule_mixed_and_or_refused():
    assert G.parse_rule("pause if cpa > 10 and ctr < 1 or cvr < 2 on ad1") is None


def test_parse_rule_minspend_and_backwards_compat():
    r = G.parse_rule("pause if cpa > ₦50000 for 3 days on adset123 via meta")
    assert r is not None and r.platform == "meta" and len(r.conditions) == 1
    r2 = G.parse_rule("pause if cpa > ₦50000 on adset1 minspend ₦20000")
    assert r2 is not None and r2.min_spend_kobo == 2000000
    r3 = G.parse_rule("scale 20% if roas > 3 for 2 days on adset1")
    assert r3 is not None and r3.action_param == 20.0


def test_parse_rule_garbage_none():
    assert G.parse_rule("") is None
    assert G.parse_rule("do something magical") is None


def test_add_preset_cpa_kill():
    st = _gstore()
    r = G.add_preset(st, "cpa_kill", "adset9", "₦75000")
    assert r is not None
    assert r.metric == "cpa" and r.action == "pause"
    assert "75000" in r.threshold_raw or "₦" in r.threshold_raw
    assert G.add_preset(st, "nope", "adset9") is None
    assert set(G.RULE_PRESETS) >= {"cpa_kill", "scale_winners", "fatigue_watch",
                                   "frequency_cap", "budget_guard"}


def test_compound_rule_fires_through_mandate_gate():
    st = _gstore()
    r = G.parse_rule("pause if cpa > ₦10000 and frequency > 3 for 1 days on ad1")
    st.add_rule(r)
    metrics = {"cpa": 2_000_000, "frequency": 4.5, "spend": 5_000_000}  # kobo
    fired = G.evaluate(st, metrics_fn=lambda ad: metrics)
    assert len(fired) == 1
    # no mandate in test env → blocked, but the compound path fired
    assert "mandate" in (fired[0].reason or "").lower() or fired[0].ok


def test_compound_rule_and_semantics():
    st = _gstore()
    st.add_rule(G.parse_rule("alert if cpa > ₦10000 and frequency > 3 for 1 days on ad1"))
    # only one condition holds → no fire
    assert G.evaluate(st, metrics_fn=lambda ad: {"cpa": 2_000_000, "frequency": 1.0}) == []
    st2 = _gstore()
    st2.add_rule(G.parse_rule("alert if cpa > ₦10000 or frequency > 3 for 1 days on ad1"))
    fired = G.evaluate(st2, metrics_fn=lambda ad: {"cpa": 100, "frequency": 9.0})
    assert len(fired) == 1 and fired[0].action == "alert" and fired[0].ok


def test_min_spend_evidence_gate_holds_streak():
    st = _gstore()
    st.add_rule(G.parse_rule("alert if cpa > ₦100 on ad1 minspend ₦50000"))
    # spend below evidence → no streak advance, no fire
    assert G.evaluate(st, metrics_fn=lambda ad: {"cpa": 99_000_00, "spend": 100}) == []
    rule = st.list_rules()[0]
    assert rule.streak == 0


def test_preview_dry_run_touches_nothing():
    st = _gstore()
    st.add_rule(G.parse_rule("alert if ctr < 1 for 2 days on ad1"))
    rows = G.preview(st, metrics_fn=lambda ad: {"ctr": 0.5})
    assert len(rows) == 1
    assert rows[0]["would_fire"] is False  # streak 0/2 — holding
    assert "holding" in rows[0]["reason"]
    assert st.list_rules()[0].streak == 0  # untouched
    assert st.audit_log() == []  # nothing audited


def test_fatigue_signal_detects_creative_fatigue():
    st = _gstore()
    for _ in range(3):
        G.record_metrics(st, "ad9", {"ctr": 2.0, "frequency": 2.0})
    for _ in range(3):
        G.record_metrics(st, "ad9", {"ctr": 1.0, "frequency": 3.0})
    sig = G.fatigue_signal(st, "ad9")
    assert sig["fatigued"] is True
    assert sig["ctr_drop_pct"] >= 30
    assert "fatigue" in sig["reason"]
    sig2 = G.fatigue_signal(st, "unknown-ad")
    assert sig2["fatigued"] is False


def test_format_rule_shows_compound_and_streak():
    r = G.parse_rule("pause if cpa > ₦50000 and frequency > 4 for 3 days on adset1")
    r.streak = 2
    out = G.format_rule(r)
    assert "CPA" in out and "FREQUENCY" in out and "2/3d" in out


def test_control_guardrails_preset_preview_fatigue():
    st = _gstore()
    out = G.control_guardrails("preset cpa_kill on adset1 ₦60000", store=st)
    assert "preset armed" in out
    out = G.control_guardrails("preview", store=st)
    assert "dry run" in out
    assert "usage" in G.control_guardrails("preset", store=st).lower()
    for _ in range(4):
        G.record_metrics(st, "adset1", {"ctr": 2.0, "frequency": 2.0})
    out = G.control_guardrails("fatigue adset1", store=st)
    assert "fatigue check" in out


# ══════════════════════════════════════════════════════════════════════
# competitor — pillar engagement, top/viral, times, hashtags, insights
# ══════════════════════════════════════════════════════════════════════

def _posts():
    base = time.time()
    texts = [
        ("protein powder review: best protein powder for muscle growth #fitness", "video", 100, 20, 10),
        ("protein powder vs creatine — which protein powder wins? #fitness", "video", 120, 25, 12),
        ("protein powder pancakes recipe #food #fitness", "carousel", 200, 60, 40),
        ("my morning routine: wake up, stretch, journal #lifestyle", "image", 30, 5, 2),
        ("morning routine for deep work — no phone for 2 hours", "video", 10, 1, 0),
        ("Q&A: your morning routine questions answered #lifestyle", "text", 50, 10, 5),
    ]
    return [C.Post(platform="instagram", account="rival", post_id=f"p{i}",
                   text=t, format=f, posted_at=base - i * 86400,
                   likes=li, comments=co, shares=sh)
            for i, (t, f, li, co, sh) in enumerate(texts)]


def test_analyze_pillars_carries_engagement():
    pillars = C.analyze_pillars(_posts())
    assert len(pillars) >= 2
    best = max(pillars, key=lambda p: p.avg_engagement)
    assert best.avg_engagement > 0
    assert best.top_post_excerpt  # best post excerpt captured
    assert all(p.posts > 0 and 0 < p.share <= 1.0 for p in pillars)


def test_top_posts_and_viral():
    posts = _posts()
    tops = C.top_posts(posts, 2)
    assert len(tops) == 2
    assert C._engagement_of(tops[0]) >= C._engagement_of(tops[1])
    viral = C.viral_posts(posts, multiple=3.0)
    mean = sum(C._engagement_of(p) for p in posts) / len(posts)
    assert all(C._engagement_of(p) >= mean * 3.0 for p in viral)


def test_best_times_and_hashtags():
    posts = _posts()
    bt = C.best_times(posts)
    assert bt["best_hour"] is not None and bt["by_weekday"]
    tags = C.analyze_hashtags(posts)
    assert tags and tags[0]["tag"] == "#fitness"
    assert tags[0]["posts"] >= 3


def test_key_insights_narrative():
    posts = _posts()
    rep = C.CompetitorReport(report_id="r", account="rival", platform="instagram",
                             pillars=C.analyze_pillars(posts),
                             cadence=C.analyze_cadence(posts),
                             engagement=C.analyze_engagement(posts),
                             post_count=len(posts))
    out = C.key_insights(rep)
    assert "strongest pillar" in out
    assert "opportunity" in out or "watch" in out or "format mix" in out


def test_report_format_shows_pillar_bars_and_insight():
    posts = _posts()
    rep = C.CompetitorReport(report_id="r", account="rival", platform="instagram",
                             pillars=C.analyze_pillars(posts),
                             cadence=C.analyze_cadence(posts),
                             engagement=C.analyze_engagement(posts),
                             post_count=len(posts))
    out = rep.format()
    assert "by engagement" in out and "eng/post" in out


def test_benchmark_side_by_side():
    mine = [C.Post(platform="x", account="me", post_id="m1", text="hello world",
                   posted_at=time.time(), likes=500, comments=50, shares=20)]
    out = C.benchmark(mine, _posts(), own_name="me", rival_name="rival")
    assert "me vs rival" in out and "verdict" in out


def test_content_gaps_and_to_briefs():
    posts = _posts()
    rep = C.CompetitorReport(report_id="r", account="rival", platform="instagram",
                             pillars=C.analyze_pillars(posts),
                             cadence=C.analyze_cadence(posts),
                             engagement=C.analyze_engagement(posts),
                             post_count=len(posts))
    gaps = C.content_gaps(rep, own_keywords=["vlog"])
    assert gaps  # 'protein' pillar not covered by 'vlog'
    assert all(g["their_avg_eng"] > 0 for g in gaps)
    briefs = C.to_briefs(rep, n=2)
    assert len(briefs) == 2
    assert all("angle" in b and "why" in b for b in briefs)
    covered = ["protein", "powder", "fitness", "review", "best", "muscle",
               "growth", "creatine", "wins", "pancakes", "recipe", "food",
               "morning", "routine", "lifestyle", "wake", "stretch", "journal",
               "deep", "work", "phone", "hours", "questions", "answered"]
    assert C.content_gaps(rep, covered) == []


def test_control_competitor_new_commands():
    scrape = lambda account, platform: _posts()
    out = C.control_competitor("insights rival", scrape_fn=scrape)
    assert "key insights" in out
    out = C.control_competitor("top rival", scrape_fn=scrape)
    assert "top posts" in out and "eng" in out
    out = C.control_competitor("hashtags rival", scrape_fn=scrape)
    assert "#fitness" in out
    out = C.control_competitor("besttime rival", scrape_fn=scrape)
    assert "sweet spot" in out
    out = C.control_competitor("gaps rival vlog", scrape_fn=scrape)
    assert "content gaps" in out
    out = C.control_competitor("gaps rival", scrape_fn=scrape)
    assert "usage" in out.lower()


def test_control_competitor_legacy_still_works():
    st = C.CompetitorStore(":memory:")
    assert "tracking" in C.control_competitor("track rival", store=st)
    assert "rival" in C.control_competitor("list", store=st)
    assert "digest" in C.control_competitor("digest", store=st).lower() or \
           "history" in C.control_competitor("digest", store=st).lower()
