# Legal Module — External Mining Report (sweep(legal), 2026-10-10)

Mined from outside the repo before writing any code. Every class in
`nomorals/legal/` gets a "how does the best implementation of X do it?"
comparison, plus gaps found. Sources cited from live web search; only
brief paraphrases recorded here (no copied text).

## 1. contracts.py — contract review / clause extraction

### CUAD — Contract Understanding Atticus Dataset (NeurIPS 2021)
- 510 real commercial contracts, ~13k expert annotations across **41 clause
  categories**, each mapping to 5 review tiers. The industry benchmark for
  clause identification.
- Clause taxonomy that matters: Basic Info (Document Name, Parties, Agreement
  Date, Effective Date, Expiration Date, Renewal Term, Notice Period,
  Governing Law), Restrictions (Non-Compete, Exclusivity, No-Solicit,
  Anti-Assignment), Financial (Cap/Uncapped Liability, Liquidated Damages,
  MFN), IP & Licensing (License Grant, IP Ownership, Joint IP), Termination
  (Termination for Convenience, Change of Control), Legal (Audit Rights,
  Covenant Not To Sue, Third Party Beneficiary).
- **Our gap:** we have 3 playbooks with ~24 rules total. CUAD says the real
  world has 41 categories — we miss governing law, confidentiality,
  indemnification, assignment, dispute resolution/venue, force majeure,
  audit rights, anti-assignment, warranty duration, change-of-control.
- Key architectural finding (anthonyrodrigues443/legal-contract-analyzer):
  full-document access beats chunk-level modeling. LightGBM reading 100% of
  the contract beat fine-tuned RoBERTa (0.716 vs 0.650 macro-F1). Lesson:
  our whole-document regex playbook architecture is sound; the gap is
  taxonomy breadth, not model choice.

### Legalquants lq-plugin-oss — playbook review skill
- **Two-tier output architecture** we should steal: Tier 1 = in-chat
  executive Issues Summary Table (rows sorted by severity, with Status,
  Deviation & Commercial Context, Action/External Comment columns); Tier 2 =
  Word export with an *internal cut* (candid rationale, privileged) and an
  *external cut* (diplomatic comment for the counterparty, drops playbook
  refs and risk ratings).
- **Our gap:** format_review is a single flat list. No triage-table mode,
  no proposed markup, no negotiation-facing vs internal distinction.

### chgreer1070/contract-risk-assessment — scoring architecture
- Pipeline: hierarchical clause segmentation with stable IDs → typed clause
  extraction (JSON schema) → playbook retrieval (RAG per clause type) →
  **deviation × likelihood × consequence scoring** → deterministic validation
  (schema + verbatim citation match) → threshold routing/abstain →
  audit log → self-improvement loop (golden dataset + CI regression gate).
- **Our gap:** our deductions are severity-flat (critical=25, high=15…); no
  likelihood×consequence nuance, no verbatim-citation validation, no golden
  regression set. We can add: verbatim-citation check on findings, and a
  golden fixture set in tests (which this sweep partially builds).

### DoNotPay "Do Not Sign"
- Reads license agreements, flags loopholes, **highlights opt-outs and
  rights the user is owed under the agreement**.
- **Our gap:** our reviews only flag risks. We never extract the *rights
  granted* to the user (right to repair, right to terminate with notice,
  refund rights). Positive-rights extraction is a named feature to add.

### Vedant legal-clause-llm (CUAD + QLoRA)
- Pipeline pattern: clause classification (41 types) → risk assessment →
  **plain-English rewriting** of clauses.
- **Our gap:** we explain clauses but never rewrite them in plain English.
  A rule-based legalese→plain rewrite map is a strong, offline-friendly
  addition.

### What's trash out there (avoid)
- Many "contract review MVPs" are LLM-prompt wrappers with no playbook,
  no deterministic validation, no abstention — confident errors on scanned
  PDFs. ContractSafe's own guidance: *have a reviewer accept/correct every
  AI-suggested value; check money amounts, renewal and notice terms by
  hand.* Lesson: keep our deterministic rules + human-checkpoint posture;
  never claim AI-verified anything.

## 2. aid.py — consumer legal aid

### DoNotPay (FTC settlement, $193k, order finalized Jan 2025)
- The cautionary tale: "world's first robot lawyer" claims, no attorney
  testing, no verified corpus → deceptive-practice order prohibiting claims
  of performing professional services without "competent and reliable
  evidence."
- **Confirms our architecture:** information-not-advice posture, disclaimer
  on every output, information_only_check, attorney-escalation. This is not
  boilerplate — it's the legal risk surface.
- Their strength we should mine: **guided digital pathways** — the user
  answers branching questions and the bot assembles the outcome (demand
  letters, contest forms). Our aid is Q&A only; no guided triage, no
  letter generation.

### African legal-aid bots (the real peer set)
- **MedLex AI** (Team UNILORIN, won ₦1m at National AI Justice Hackathon
  2026, Port Harcourt): WhatsApp-based AI chatbot for instant legal aid —
  our exact distribution surface. Validates WhatsApp-native, plain-language
  design.
- **Case Radar** (Nigeria): ChatGPT-for-lawyers trained on digitized
  Nigerian court documents that were never online. Lesson: *corpus is the
  moat* — partner/license legal publishers (our aid.py already says this).
- **Ask-Attorney (Uganda)**: legal guidance via **SMS** for low-internet
  populations — the fallback surface our WhatsApp-native design ignores.
  SMS-length formatting (160-char chunks) is a real feature gap.
- **LawPadi (Nigeria)**: lawyer matching for affordable consultations.
  **MyJustice (Kenya)**: free legal advice + document templates + referrals.
- Recurring pattern across all: **referrals to real lawyers** (Legal Aid
  Council of Nigeria, NBA branches) and **document templates** (complaint
  letters, demand letters). **Our gaps:** no lawyer-referral directory, no
  letter/document templates, no guided triage questions, only 4 scenarios
  (missing: bail-money extortion, inheritance/intestate, CAC registration,
  debt-collection harassment, police search).

### Trash out there (avoid)
- "AI lawyer" marketing without testing (the FTC case). LegalCheek's 2016
  test: DoNotPay failed basic legal questions. Lesson: scenario-matched,
  corpus-cited answers beat generative swagger; keep the honest fallback.

## 3. research.py — grounded legal RAG

### Harvey LAB-AA v1.1 (hallucination grading, 2026)
- Hallucination grading pipeline: **compare every deliverable claim against
  the source documents**; distinguish unsupported task-specific claims from
  general legal knowledge; conservative flagging of material errors.
- **Our gap:** we compose evidence-only summaries but never *verify* that
  each sentence in the output traces to a corpus span. Add verbatim-snippet
  verification (snippets must be substrings of the source section).

### Lexis+ AI / Westlaw AI (Contrary Research, 2026)
- Lexis+ AI claims "100% hallucination-free linked legal citations
  connected to source documents"; measured hallucination rates: Lexis+ AI
  0.17, Westlaw 0.33, GPT-4 0.43. Access to the verified corpus is what
  moves the needle.
- Study conclusion: **a response is hallucinated when it claims an authority
  backs a proposition it does not** — groundedness scored separately from
  correctness. **Our gap:** we don't score or display groundedness per
  claim; add a groundedness check (claim→citation link) to research output.

### Schwarcz et al. / Vincent AI (Harvard JOLT)
- RCT: RAG-powered tool gave statistically significant gains on legal
  tasks and "reduc[ed] hallucinations… to levels comparable to work
  completed without AI assistance." RAG works best on narrow issues.
- **Our gap:** our queries are one-shot. Best practice is **query
  expansion** (synonym/rewrite variants, merged + deduped) and abstention
  on uncovered topics — we have abstention, missing expansion.

### AC-RAG paper (ICLR 2026 submission)
- Learns the mapping between case text and legal article numbers, avoiding
  rigid decision thresholds — adaptive relevance beats fixed τ.
- Lesson: prefer adaptive coverage logic (we already gate on term coverage
  — keep and extend).

## 4. portfolio.py — contract repository / obligation tracking

### Ironclad (contract tracking software)
- AI does: automated metadata extraction (counterparty, effective dates,
  payment terms, termination clauses), **obligation monitoring** (flags
  approaching/overdue obligations), **anomaly detection** (clause deviates
  from standard language), **natural-language search** ("show me all NDAs
  with auto-renewal expiring this year").
- Feature checklist to mine: searchable repository (full-text + metadata
  filters), automated alerts (renewal dates, termination windows),
  **dynamic dashboards: renewal pipeline 30/60/90 days**, workflow
  automation, AI metadata extraction, audit trails.
- **Our gaps:** no full-text search over stored contracts; no renewal
  pipeline buckets; no auto-renewal roll-forward (dates keep rolling);
  no amendment/version chains; no compliance-status dashboard; no
  natural-language query over the repository.

### ContractSafe (AI data extraction)
- Pattern: **AI suggests values → human accepts, corrects, or skips each
  one before it's saved.** Check money amounts, renewal/notice terms,
  payment frequency, execution dates by hand.
- **Our gap:** extracted obligations go straight in as "extracted" with no
  confirm/flag state. Add an extraction-approval state machine.

### ContractSafe pricing page (market signal)
- Date alerts + renewal tracking keep renewal dates, notice deadlines,
  reminders current **as each term rolls forward** — auto-renew roll-forward
  is table stakes, not a nice-to-have.

## 5. regulatory.py — regulatory change monitoring

### Compliance.ai (RegTech reference)
- Pattern: automatically monitor the regulatory environment → organize +
  interpret content → **filter to what's relevant to the organization** →
  **ensure change tasks are completed** → real-time insight on compliance
  status. ML maps regulatory updates to **internal policies, procedures,
  and controls**.
- **Our gaps:** we have controls mapping but no *task lifecycle* for
  checklist items (track → in progress → done); no impact scoring; no
  change-detection (item updated vs new).

### Thomson Reuters Regulatory Intelligence (+ IBM Watson)
- NLP parses regulations, extracts obligations, **maps them to business
  processes**, detects changes over time. Visibility into *pending*
  regulatory changes, projecting business impact.
- **Our gaps:** change detection (diff), pending/effective-date awareness
  (our items have `published` but no `effective_date`), business-impact
  projection.

### RegTech 2026 trend (datafield.dev)
- NLP models parse dense regulation text, extract obligations, map to
  business processes, **detect changes over time**; ML reduces false
  positives from historical investigation outcomes.
- Lesson: add user feedback ("was this alert relevant?") to tune relevance
  filtering — simple thumbs mechanism.

## Cross-module findings

1. **Natural-language query** over both the contract repository and the
   regulatory store ("which of my contracts mention NDPA?") — Ironclad
   pattern; neither module has it.
2. **Style gap everywhere:** outputs are single-format plain text. Best
   practice from legalquants: executive triage table (chat) + full detail
   (briefing). Add style modes: `triage`/`compact`/`full`.
3. **Corpus is the moat** (Case Radar lesson): our 4-doc seed corpus is
   honest but thin. Add a corpus-breadth API (register publisher feeds,
   `describe_corpus()` coverage report) without building 70 volumes.
4. **Anti-hallucination** is the product message (Harvey/Contrary): add
   verbatim-citation verification + groundedness display to research.py.
