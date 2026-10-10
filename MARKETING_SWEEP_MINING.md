# MARKETING SWEEP — external mining report

Module: `nomorals/marketing/` (aeo.py, send.py, briefs.py, guardrails.py, competitor.py).
Date: 2026-10-10. Method: real web research, best AND trash implementations mined first.
Rule honored: every significant class compared against the best external implementation
with "How does the best implementation of X do it?" before any code was written.

---

## 1. AEOTracker (`aeo.py`) — AEO/GEO visibility tracking

**Current state:** fans prompts to 4 AI engines, parses citations, two-model
cross-checks brand mentions, reports "share of answer" (confirmed pairs / total).
Solid honest-by-default architecture (no key → unavailable, never fabricated).

### What the best do

**Somantra AEO Metrics Suite** (globenewswire.com, 2026-09-29 — newest industry
standard, launched ~11 days before this sweep):
- Three complementary scores, not one flat count:
  1. **Brand Mindshare Score** — share of voice AND share of citation across the
     brand's *full query landscape*, including **query neighbourhoods and
     word-shift variants** that standard tracking misses. A single-word change
     to a query can shift which brand an AI recommends — flat citation counts
     never surface this.
  2. **Brand Consideration Score** — strength and *favourability* of positioning
     relative to named competitors *within a response*: distinguishes a genuine
     **recommendation** from a passing mention. "A brand mentioned once in
     passing and a brand actively recommended look the same in a citation count.
     They are not the same."
  3. **Brand Engagement Score** — how deeply the assistant engages with the
     brand's *specific claims* across multi-turn conversations, not just naming.
- Verdict: our "share of answer" is only Somantra's Mindshare layer. The
  Consideration layer (recommendation vs mention) is the gap that decides
  whether AI is *closing the sale*.

**aeoranks.com — "How to Track AEO & GEO Success":**
- **AI Share of Voice (AI SOV)** = (brand mentions ÷ total brand mentions across
  a fixed prompt set) × 100 — the north-star metric, distinct from our
  share-of-answer. Ours asks "in how many pairs are you cited"; SOV asks "how
  much of the conversation do you own".
- Four metric clusters: Visibility, Authority & Trust, Technical (crawlability),
  Business Impact. We only cover Visibility.

**stonegiantstudio/skills (GitHub) — AI-visibility connector:**
- Manual citation protocol as honest fallback (cited y/n, position/order, URL
  cited, *which competitors were cited*); Otterly.AI API + MCP for automated
  brand citations + share-of-voice across ChatGPT/Gemini/AI Overviews/
  Perplexity/Copilot. Confirms our engine-fanout + injectable-caller design is
  the right shape; adds **position/order of citation** and **competitor-cited**
  as fields we don't capture.

**GEO readiness (what makes engines cite you) — dev.to GEO checklist,
creaitor.ai LLM SEO, semarkglobal.com, endertech.com:**
- robots.txt must allow AI search crawlers: GPTBot, OAI-SearchBot, ChatGPT-User,
  PerplexityBot, ClaudeBot, Google-Extended; Bing index matters (ChatGPT search
  leans on Bing). Check for 403/429 to unfamiliar user agents.
- llms.txt (llmstxt.org, v1.7.0): H1 site name, blockquote summary, H2 sections
  with `- [Title](URL): description` lists, `## Optional` section last; serve at
  /llms.txt as text/plain or text/markdown. Honest caveat: a *proposal*, no major
  vendor committed — weight accordingly, don't oversell.
- Citation magnets: definitions as the first sentence, numbered lists with
  context, statistics with source attribution, FAQ with H3 questions,
  answer-first content (answer in the first few sentences), specific declarative
  statements over vague marketing copy, "Last Updated" dates (RAG recency bias),
  consistent entity facts everywhere.
- Schema: Google says NO special AI schema exists; Ahrefs' 1,885-page test found
  adding JSON-LD barely moved AI citations. Cheap hygiene, not a lever. (Trash
  to avoid: vendors selling "AI schema" as a ranking lever.)

### Gold to take
1. **Consideration layer** — classify each mention: recommended / mentioned /
   compared / negative. Heuristic lexicon, never raises.
2. **AI Share of Voice** — competitor brand counting across the same responses;
   `competitors=[...]` param on track_visibility.
3. **Word-shift prompt variants** — `expand_prompts()` generating perturbation
   variants (best↔top, reviews↔ratings…) to surface query sensitivity.
4. **Sentiment per mention + report aggregate** — flat counts carry no sentiment
   signal (Somantra gap #1).
5. **GEO readiness check** — `site_readiness_check(url)`: robots.txt AI-crawler
   tokens, /llms.txt presence + structure validation, graded checklist. Honest
   about llms.txt's proposal status.
6. **Trend delta + sparkline** in report formatting; citation quality (own-domain
   vs third-party vs review-site citations).

### Trash to avoid
- Vendors selling llms.txt as "the key to ChatGPT ranking" (oversold; Google
  explicitly says it doesn't use it as a search signal).
- "AI schema" upsells (Ahrefs proved ~no citation lift).

---

## 2. SendEngine (`send.py`) — self-hosted send layer

**Current state:** SQLite queue, sliding-window rate limiting, exponential-backoff
retry, DSN bounce parsing (hard→blocklist), tiny template renderer with
conditionals, provider registry (smtp/ses/twilio/mock), cost gating (#68).
Clean-room architecture; honest about being pattern-level, not Listmonk.

### What the best do

**Listmonk** (per its docs/guides): campaign workflow with template variables
(`{{ .Subscriber.Attribs.x }}`), **"always send a test email first"** (seed
testing), bounce registration on campaign stats, REST API for lists/subscribers/
campaigns/templates, custom subscriber fields. Our vars-dict + blocklist covers
the shape; missing: seed/test send as a first-class action, list management.

**Mautic best-practices (hawkhost.com):**
- "Build for deliverability, not just automation": warm up new IPs/domains
  gradually (low daily volume → most engaged contacts first), never import
  stale lists, **make unsubscribing easy and immediate**, clean dead contacts
  before they hurt reputation, watch bounce/complaint rates like a hawk.
- **Fix DNS first**: SPF, DKIM, DMARC misconfigured → spam folder. Use a sending
  subdomain to isolate main-domain reputation.
- Scope creep warning: one clear goal per campaign, explainable customer journey.

**Email deliverability canon (revnew.com 27-step, stripo.email, medium.com):**
- Warmup: start 5–10 msgs/day to warm addresses, consistent daily volume, ramp
  over 2–4 weeks; pace sending speed (never blast all at once).
- Thresholds: bounce rate <2–5%, unsubscribe <5%, open >50%, reply >8%.
- Feedback loops (spam complaints), weekly blacklist checks (Spamhaus), double
  opt-in, role-based address pruning, spam-trap avoidance (never buy lists).

### Gold to take
1. **Deliverability audit** — `deliverability_check(domain)`: DNS TXT for SPF
   (v=spf1), `_dmarc` TXT, MX presence; graded checklist. Pure stdlib (socket).
2. **Warmup plan generator** — `warmup_plan(daily_target, days)`: exponential
   ramp schedule; `warmup_cap(day)` the engine can consult.
3. **Unsubscribe as first-class** — `unsubscribe(address, reason)` → suppression
   (distinct from bounce blocklist); `List-Unsubscribe` header on SMTP sends;
   `{{unsubscribe_url}}` template var.
4. **Spam-score heuristic** — trigger words, caps ratio, exclamation density,
   link count → 0–100 with fix suggestions (Mail-Tester pattern, offline).
5. **A/B testing** — variant templates, recipient split, per-variant stats,
   `ab_winner()` (fewest failures/bounces; engagement hook injectable).
6. **Scheduled campaigns** — `schedule_campaign()` (send_at per send; process()
   picks up due ones — the next_attempt_at column already supports it).
7. **Campaign analytics** — `campaign_report()`: funnel queued→sent→failed→
   dead→blocked per provider, cost, ASCII funnel bars.
8. **Seed test** — `send_seed(template, seed_addresses)`.

### Trash to avoid
- Buying lists / spam-trap harvesting (reputation suicide; we never go there).
- "Warmup services" that fake engagement — we do honest gradual ramping only.

---

## 3. BriefStore (`briefs.py`) — brief-first content pipeline

**Current state:** build_brief (keyword/question/angle/outline extraction) →
score_draft (term coverage + readability + brand voice via #44) →
schedule_draft (refuses without brief) → predict_performance (Anyword pattern) →
record_outcome (learned calibration). Frase/Surfer rule "no brief, no draft"
enforced. Honest about heuristic scoring.

### What the best do

**Surfer SEO** (medium.com comparison, 2026): Content Score 0–100 is a
*composite* — keyword density, NLP terms, heading count, paragraph structure,
content length, image usage. You can satisfy all NLP terms and still score low
if heading structure/word count are off. **Real-time score updates as you
write**; term suggestions. Lesson: our score is only term-coverage-shaped; the
structure layer is missing.

**Clearscope**: grades F → A++ on **concept coverage**; NLP-extracted concepts
carry **salience 0.0–1.0** (how central the concept is) which drives weighting;
term frequency benchmarks (how often each term should appear); Google Docs
integration; competitor content structure comparison. Lesson: salience-weighted
terms + frequency bands + letter grades (easier to communicate than raw %).

**Frase** (frase.io): brief generation strongest on **question research** —
People Also Ask + Reddit + Quora + related searches, **organized by topic** so
writers see which reader questions to answer. Separate **GEO score** alongside
SEO score ("how your page reads to AI search engines"). **Content Guard**:
watches for ranking decay, proposes a fix for approval before republishing.
Lesson: question clustering, a GEO-readiness score, and a decay watchdog.

**AI SEO workflow (digitaleyen.com)**: search intent first (guide vs list vs
comparison vs tutorial) → brief (headings, entities, FAQs) → draft in sections →
on-page (title, H1/H2, internal links, readability) → trust signals (author,
sources, unique screenshots) → publish + refresh decayed content.

### Gold to take
1. **Structure score** — heading count, word count vs brief target, paragraph
   structure → Surfer-style composite (coverage 45 / readability 20 / voice 20 /
   structure 15).
2. **Term frequency bands** — per-keyword min/max from source texts
   (Clearscope-style); missing_terms now carries recommended counts.
3. **Question clustering** — group questions by topic (Frase-style) in briefs.
4. **GEO score** — separate AI-citation-readiness score: definition-first,
   FAQ structure, stats with attribution, lists, answer-first opening.
5. **Brief quality score** — score the brief itself (keyword depth, question
   coverage, angle diversity) so bad briefs don't silently produce bad drafts.
6. **Decay watchdog** — `stale_drafts(days)`: drafts with poor outcomes or age →
   refresh suggestions (Frase Content Guard pattern, approval stays human).
7. **Letter grades** — Clearscope-style F…A++ alongside the 0–100 score.
8. **Multi-platform adaptation** — hook/style rewrite guidance per platform.

### Trash to avoid
- Score-chasing → keyword stuffing (Surfer's known failure mode; our
  suggestions must warn when coverage is gamed, and frequency bands cap it).
- Auto-republishing decay fixes without approval (Frase keeps humans in the
  loop — we do too: watchdog *proposes*).

---

## 4. GuardrailStore (`guardrails.py`) — ad-spend guardrails

**Current state:** natural-language rule parsing ("pause if cpa > ₦50000 for 3
days on adset123"), streak-based multi-day confirmation, cooldowns, audit log,
#69 mandate gate, explicit owner override. Bïrch/Madgicx pattern faithfully
re-implemented; honest about money-safety layering.

### What the best do

**Bïrch (Revealbot)** (startuptalky.com, lifestyle.pspl.com): rule-based
automation across Meta/Google/TikTok/Snapchat; **multi-condition AND/OR logic**
for ROAS, CPA, spend, fatigue signals; **ready-made automation strategies**
(presets) for common tasks; rules run every 15 min; custom metrics; cross-account
reporting; clear execution logs = audit trail. Philosophy: every action traces
to a human-set rule (auditable) — matches our design exactly.

**Madgicx**: automated rules + AI bidding + **daily account checks**; AI Marketer
flags wasted spend and growth opportunities; Ads Manager 2.0 bulk editing,
trend views.

**Diagnostic canon (adadvisor.ai)** — the patterns worth encoding:
- Creative fatigue: $840/3 days, 0 purchases, CTR down 38% from first week →
  pause, move ~15% budget to best new-customer CPA ad set.
- Audience fatigue: CPA up 60% WoW + frequency 4.2 rising → refresh creative or
  widen audience; hold budget flat.
- Budget misallocation: one ad set takes 70% of budget at 1.4 ROAS while a 2.6
  ROAS set is capped → shift toward the higher-contribution set.
- **Minimum-evidence window**: a two-day-old campaign judged on CPA is still in
  learning — leave it until it clears the window (don't reset learning).
- Tracking gap vs real problem: clicks steady, purchases down 40% → check
  Pixel/CAPI health before touching budgets.

### Gold to take
1. **Multi-condition rules** — AND/OR compound conditions
   ("pause if cpa > ₦50k for 3 days AND frequency > 4 on adset1").
   Backwards compatible: single-condition rules parse as before.
2. **Rule presets** — `RULE_PRESETS`: "cpa kill switch", "scale winners",
   "creative fatigue watch", "frequency cap", "budget misallocation";
   `/guardrails preset <name> on <adset>`.
3. **Creative-fatigue detection** — `record_metrics()` snapshots + fatigue
   signal: CTR decay ≥30% over window with frequency rising → alert/pause.
4. **`alert` action** — notify-only (no platform change); extends the action set.
5. **Dry-run preview** — `preview()`: "what would fire right now" without
   executing, marking, or auditing.
6. **Minimum-evidence gate** — `min_spend_kobo` per rule: no firing before the
   ad set has spent enough to judge (learning-window protection).
7. **Frequency metric** + richer formatting (deltas, streak progress "2/3 days").

### Trash to avoid
- Fully autonomous AI ad-buying with no human-set rules (un-auditable; our
  explicit-rule philosophy is the differentiator — keep it).
- Judging campaigns inside the learning window (encoded as min-evidence gate).

---

## 5. CompetitorStore (`competitor.py`) — no-access competitor intel

**Current state:** public-posts-only analysis (injectable scrape_fn, honestly
empty default), keyword-clustered content pillars, cadence (posts/week by
format), engagement averages + 1st/2nd-half trend, snapshot history, window-vs-
window digests, AEO face-off vs own brand. Socialinsider pattern.

### What the best do

**Socialinsider** (socialinsider.io, 2026 — best-in-class for competitor content
strategy):
- **Content pillar analysis with per-pillar engagement rates** — "a pillar with
  fewer posts often outperforms one with much heavier posting volume" (quality
  vs frequency insight). AI-generated pillars, no manual tagging.
- **Key Insights summary** — written summary of posting volume, engagement,
  follower growth, views, *closing with observations* (where frequency could
  increase, which platform carries engagement). "A brief you can forward, not
  an export you have to interpret."
- Side-by-side benchmarking (engagement, cadence, formats, growth), top-
  performing posts (clickable spike → the posts that caused it), content format
  mix, earned media value.

**Metricool**: competitor profiles (posting frequency, engagement rates, top
content, growth trends), **hashtag tracking** (what's trending in the niche),
**best times to post** per platform from audience data, peak posting times +
content type breakdowns.

**Socialinsider analytics guide** — the competitor metrics that matter:
follower growth, engagement rate, posting frequency, avg engagement/post,
top-performing content, content formats, content pillars/themes, **share of
voice**; paid: CPM/CPC/CPA/ROAS; commerce: AOV, social commerce ROAS.

### Gold to take
1. **Per-pillar engagement rates** — extend Pillar with avg engagement; pillars
   ranked by engagement, not just volume (the quality-vs-frequency insight).
2. **Top posts** — `top_posts(n)`: highest-engagement posts with excerpts
   (Socialinsider's "which posts caused the spike").
3. **Best posting times** — hour-of-day × day-of-week engagement heat
   (Metricool pattern, from public timestamps).
4. **Hashtag tracking** — `analyze_hashtags()`: top hashtags with avg
   engagement.
5. **Key Insights summary** — `key_insights(report)`: auto-written narrative +
   observations ("video pillar drives 2.3x avg engagement — consider more").
6. **Viral-post alerts** — posts >3x mean engagement flagged in digests.
7. **Share of voice** — own-account vs competitor engagement comparison
   (`benchmark()`).
8. **Content gaps** — pillars they cover that your keyword list doesn't →
   feed `to_briefs()` into the #45/#99 content pipeline.
9. Richer formatting: engagement bars per pillar, top-post list.

### Trash to avoid
- Credentialed scraping / ToS-violating data collection (our no-access boundary
  is the product's integrity — never crossed).
- Vanity follower-count leaderboards without engagement context.

---

## Cross-module style direction

All five modules report into chat. God-tier bar for this sweep:
- Sparklines for trends (share-of-answer history, engagement trend).
- ASCII bar gauges for scores (content score, pillar engagement, SOV).
- Tables for rules, providers, tracked accounts.
- One consistent voice: numbers first, verdict line last, next action named.
- New `format_*` richness must stay inside the modules' "never raises" contract.
