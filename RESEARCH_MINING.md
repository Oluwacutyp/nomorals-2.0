# Phase 7 Mining Report: Research/OSINT

## Internal survey (10,364 lines across 19 files)

**Research organ** (`nomorals/research/`):
- `pipeline.py` (1233 lines): ResearchBudget, decompose (LLM + template), run_job (concurrent search + depth fetch), assess_worth, deliver, synthesize with [S<n>] citations
- `grounded.py` (416): grounded synthesis discipline
- `citations.py` (343): citation handling
- `autonomy.py` (389): autonomous research
- `scheduler.py` (193): research scheduling
- `grounded_store.py` (288): grounded storage

**Search** (`nomorals/search/`):
- `federated.py` (276): federated search across adapters
- `web.py` (620): WebSearchSource, SearXNG, DDG, Tavily, Serper, Exa, Brave
- `sources.py` (620): Books, Docs, Memory, Wisdom, Code, Timeline adapters
- `osint.py` (339): UsernameSweep, EmailCheck (Hudson Rock), DomainRecon (crt.sh + RDAP), IpIntel (ipwho.is + Shodan)
- `osint_browser.py` (193): BrowserPeopleSearch, BrowserSiteSearch
- `adaptive.py`, `rerank.py` (BM25), `model.py`, `base.py`

**Agents**: `researcher.py` (740), `research_loop.py` (888), `research_swarm.py` (811), `osint_graph.py` (922), `research_digest.py`, `research_lexicon.py`

**Tools**: `osint.py` (728), `osint_people.py` (622), `research.py` (168)

## External mining: Deep Research systems

**Perplexity Deep Research** (turion.ai, ziptie.ai, jaxonparrott.com):
1. Query decomposition (3-5 sub-queries)
2. Parallel retrieval (BM25 + semantic hybrid)
3. Cross-encoder reranking (entity relevance, authority, freshness)
4. Source filtering (authority, recency, dedup)
5. Reasoning pass — reads sources, identifies contradictions/gaps, decides if more searches needed
6. Synthesis with inline citations
7. The loop is iterative, not single-pass

**OpenAI Deep Research**: multi-step, clarifies scope first, browses hundreds of sources, cross-validates, adapts search based on findings.

**Key open-source pattern** (sharma-anubhav/deep-research-agent):
- Claim–evidence separation: claims extracted, linked to evidence, fact-checked
- Reflection-based context: findings distilled into notepad (~66% token savings)
- Pluggable search (Tavily/DuckDuckGo)

**Provider variations** (flohiwg/ai-ml-knowledge-base):
- Claude: lead agent + parallel sub-agents, each explores one angle
- Perplexity: iterative retrieval loop, hybrid model routing
- Grok: credibility assessment per segment

## External mining: OSINT

**Five-phase process** (multiple sources):
1. Passive collection (WHOIS, DNS, crt.sh, search engines)
2. Email intelligence (theHarvester, h8mail, holehe, HIBP)
3. Username/social (sherlock 400+ sites, maigret 3000+ sites)
4. Domain/asset discovery (subfinder, amass, Shodan)
5. Credential/code leak (GitHub, pastebins, IntelligenceX)

**Key techniques**:
- Username > full name for person OSINT (7/11 vs 0/11 platforms)
- LeakCheck.net free API: `https://leakcheck.net/api/public?check=USER_OR_EMAIL`
- Gravatar for email-to-identity correlation
- crt.sh for certificate transparency
- Parallel investigation groups (infrastructure → deep recon → content)
- Confidence ratings: High (multiple sources), Medium (single reliable), Low (unverified)
- Entity graph (Maltego-style transforms): username → platform → URL, domain → email/host

## Gaps identified

**Research pipeline:**
1. NO iterative refinement — single pass: decompose → search → fetch → assess. Perplexity/OpenAI all do reasoning pass → gap analysis → more searches.
2. NO claim-evidence separation — synthesis cites findings but doesn't extract atomic claims and verify each against evidence.
3. NO contradiction detection — when sources disagree, no mechanism to flag it.
4. NO reflection notepad — raw findings go into context (token waste).
5. NO interactive clarification — doesn't ask follow-ups to refine scope before deep research.

**OSINT:**
6. Missing: phone OSINT (carrier, location), GitHub recon (repos, commits, gists), Wayback Machine, Gravatar, LeakCheck API, holehe-style registration checks
7. NO entity graph linking — osint_graph.py exists but findings aren't auto-linked into a graph
8. NO confidence ratings on OSINT findings
9. NO parallel investigation orchestration — adapters run but not in phased groups

## Implementation plan

1. **Iterative research loop** (`pipeline.py`): after initial findings, LLM reasoning pass identifies gaps/contradictions → generates follow-up queries → searches again → repeats until satisfied or budget exhausted (max 3 iterations)
2. **Claim-evidence extraction** (`grounded.py`): extract atomic claims from synthesis, link each to supporting findings, flag unsupported claims
3. **Contradiction detection** (`pipeline.py`): when findings disagree on factual points, surface the conflict explicitly
4. **OSINT expansion** (`osint.py`): add PhoneIntel (carrier via free APIs), GitHubRecon (profile/repos/commits), WaybackAdapter (historical snapshots), GravatarAdapter (email→avatar→identity), LeakCheckAdapter (breach data)
5. **Entity graph auto-linking** (`osint_graph.py`): auto-build graph from OSINT findings (username→platform, domain→email, email→breach)
6. **Confidence scoring** (`osint.py`): rate each finding High/Medium/Low based on source count and reliability
