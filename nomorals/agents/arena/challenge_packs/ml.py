"""ML + security challenge pack for the self-improvement arena.

``chall_ml``: 8 categories x 12 challenges (3 easy / 5 medium / 4 deep):

* ``security_eng`` — hardened configs, secrets handling, sandboxing,
  input validation, crypto-misuse detection.
* ``threat_modeling`` — STRIDE per component, abuse cases, attack
  trees, mitigations backed by tests.
* ``secure_defaults`` — deny-by-default APIs, capability scoping,
  safe deserialization.
* ``ml_ops`` — inference serving, batching, quantization,
  monitoring/drift.
* ``evals`` — benchmark harnesses, pass@k, human-vs-auto agreement,
  regression gates.
* ``finetune`` — LoRA/QLoRA pipelines, dataset curation,
  eval-before/after, GGUF export checks.
* ``rag`` — chunking strategies, hybrid retrieval, reranking,
  citation faithfulness.
* ``agents`` — tool-use loops, planning, multi-agent orchestration,
  sandboxing, trajectory evals.

Every entry is a concrete task with a one-line checkable outcome
(``verify``) and a ``kind`` (``"code"`` | ``"research"`` | ``"build"``).
"""

from __future__ import annotations

from ..challenges import challenge
from ..topics import register_topic_pack

_PACK = {
    # ------------------------------------------------------------------
    # security_eng: hardened configs, secrets handling, sandboxing,
    # input validation, crypto-misuse detection
    # ------------------------------------------------------------------
    "security_eng": [
        challenge(
            "Harden a sample FastAPI app's config: env-only secrets, "
            "no debug in prod, secure headers middleware, strict CORS",
            1,
            "audit script reports 0 findings: debug off, no hardcoded "
            "secrets, CSP/HSTS headers present on every response",
            "code", "hardening", "config", "fastapi"),
        challenge(
            "Write a pre-commit hook that blocks common secret patterns "
            "(AWS keys, RSA private headers, bearer tokens) before commit",
            1,
            "hook rejects a fixture commit containing 5 planted secret "
            "formats and passes a clean fixture with no false positives",
            "code", "secrets", "git", "pre-commit"),
        challenge(
            "Build an input-validation layer for a JSON API using "
            "pydantic: length caps, allowlists, rejection of extra fields",
            1,
            "pytest passes: 12 malicious payloads rejected (oversize, "
            "nested-depth, extra fields) while 8 valid ones are accepted",
            "code", "validation", "pydantic", "api"),
        challenge(
            "Implement a crypto-misuse scanner that flags ECB mode, "
            "static IVs, md5/sha1 for signatures, and weak RSA key sizes",
            2,
            "scanner catches all 10 planted misuse cases in the fixture "
            "codebase with zero false positives on the clean corpus",
            "code", "crypto", "static-analysis", "misuse"),
        challenge(
            "Sandbox an untrusted Python plugin loader with "
            "resource limits, blocked imports, and a syscall allowlist",
            2,
            "pytest passes: malicious plugin samples (fork bomb, socket "
            "exfil, file overwrite) are contained; benign plugins run",
            "code", "sandboxing", "plugins", "containment"),
        challenge(
            "Design a secrets-rotation runbook plus code: versioned "
            "secrets, dual-active window, automated rollback on failure",
            2,
            "rotation script migrates a fixture service from key v1 to v2 "
            "with zero dropped requests, then rolls back cleanly",
            "build", "secrets", "rotation", "runbook"),
        challenge(
            "Build a dependency-vulnerability gate for CI that fails the "
            "build on critical CVEs and opens tracking issues per finding",
            2,
            "CI gate blocks a fixture lockfile with 3 critical CVEs and "
            "passes a patched one; issues contain CVE id + fix version",
            "build", "supply-chain", "ci", "cve"),
        challenge(
            "Implement mTLS between two microservices with automatic "
            "cert issuance, short-lived certs, and revocation checking",
            2,
            "integration test passes: revoked client cert is rejected, "
            "expired cert refused, valid pair completes handshake in <1s",
            "build", "mtls", "pki", "microservices"),
        challenge(
            "Write a JWT validation library that enforces alg allowlist, "
            "exp/nbf/iat, audience, and rejects alg=none confusion",
            3,
            "pytest passes: 15 attack tokens (alg-none, RS256/HS256 "
            "confusion, expired, wrong aud) all rejected; valid tokens OK",
            "code", "jwt", "auth", "validation"),
        challenge(
            "Build a WAF-style request inspector for SQLi/XSS/path-"
            "traversal with a tunable ruleset and a bypass regression set",
            3,
            "inspector blocks all 40 payloads in the bypass regression "
            "suite with <=2 false positives on 200 benign requests",
            "code", "waf", "sqli", "xss"),
        challenge(
            "Implement a secure file-upload pipeline: magic-byte "
            "verification, AV-scan hook, randomized storage names, "
            "content-disposition hardening",
            3,
            "pytest passes: polyglot GIF/PHP, double-extension, and zip-"
            "bomb uploads are rejected; 5 legit files stored and served",
            "build", "uploads", "malware", "pipeline"),
        challenge(
            "Design and implement a privilege-separated worker pool: "
            "broker drops privileges, workers run chrooted with seccomp",
            3,
            "escape-attempt fixture fails: worker cannot read broker "
            "files, open sockets, or exec; broker still routes 1k tasks",
            "build", "sandboxing", "seccomp", "privilege"),
    ],
    # ------------------------------------------------------------------
    # threat_modeling: STRIDE per component, abuse cases, attack trees,
    # mitigations backed by tests
    # ------------------------------------------------------------------
    "threat_modeling": [
        challenge(
            "Write a STRIDE threat model for a URL-shortener: 6 "
            "components, 2+ threats per component, each with a mitigation",
            1,
            "threat model document lists >=12 threats across all STRIDE "
            "categories, each mapped to a concrete mitigation",
            "research", "stride", "web", "modeling"),
        challenge(
            "Enumerate 10 abuse cases for a public file-sharing API and "
            "map each to a rate-limit or quota mitigation",
            1,
            "abuse-case document lists >=10 cases, each with a mapped "
            "mitigation and the config value that enforces it",
            "research", "abuse-cases", "api", "rate-limiting"),
        challenge(
            "Build an attack tree for credential stuffing against a login "
            "endpoint, with cost/effort annotations on each leaf",
            1,
            "attack tree has >=15 nodes to depth 3, every leaf annotated "
            "with cost/effort and a linked countermeasure",
            "research", "attack-trees", "auth", "credential-stuffing"),
        challenge(
            "Threat-model an MCP-style tool server: enumerate tool "
            "poisoning, prompt injection via tool output, and over-"
            "privileged tool scopes",
            2,
            "document lists >=10 tool-specific threats with mitigations; "
            "a test proves a poisoned tool description is flagged",
            "research", "mcp", "tool-poisoning", "llm"),
        challenge(
            "Model data-exfiltration paths out of a RAG system: chunk "
            "leakage via citations, embedding inversion, prompt-extraction",
            2,
            "document maps >=8 exfil paths to mitigations; a probe "
            "extracts 0 chunks after mitigations vs >=3 before",
            "research", "rag", "exfiltration", "privacy"),
        challenge(
            "Write a threat model for a CI/CD pipeline covering secret "
            "leakage, artifact tampering, and runner compromise",
            2,
            "document lists >=12 threats with mitigations; a tampered "
            "artifact is rejected by the pinned-hash check in a demo",
            "research", "cicd", "supply-chain", "artifacts"),
        challenge(
            "Threat-model a multi-agent system: agent impersonation, "
            "task-hijack via message injection, and runaway tool loops",
            2,
            "document lists >=10 agent-specific threats; a test shows the "
            "message-auth layer blocks a forged inter-agent message",
            "research", "agents", "impersonation", "injection"),
        challenge(
            "Build a STRIDE-per-component model of a fintech ledger "
            "service with formal abuse-case tests for each mitigation",
            3,
            "document lists >=18 threats; pytest passes: one failing-then-"
            "passing test per mitigation (>=12 tests green)",
            "research", "stride", "fintech", "ledger"),
        challenge(
            "Construct a full attack tree for model-extraction against a "
            "hosted LLM endpoint, with query-budget math per branch",
            3,
            "attack tree has >=20 nodes; a simulated extraction run "
            "validates the cheapest branch cost within 2x of the model",
            "research", "model-extraction", "llm", "attack-trees"),
        challenge(
            "Threat-model a browser extension with host permissions: "
            "XSS in content scripts, update hijack, overbroad matches",
            2,
            "document lists >=14 threats; a test proves the manifest v3 "
            "CSP blocks an injected inline script in the fixture",
            "research", "browser-extension", "xss", "permissions"),
        challenge(
            "Model supply-chain threats for a Python package you publish: "
            "typosquat defense, build provenance, maintainer takeover",
            3,
            "document lists >=10 threats with mitigations; SLSA-style "
            "provenance attestation verifies on a real test build",
            "research", "supply-chain", "packaging", "provenance"),
        challenge(
            "Write a threat model for an on-device voice assistant: mic "
            "always-on risks, ultrasonic injection, cloud transcript leak",
            3,
            "document lists >=12 threats; a test shows the on-device VAD "
            "gate blocks a replayed ultrasonic trigger fixture",
            "research", "voice", "iot", "sensors"),
    ],
    # ------------------------------------------------------------------
    # secure_defaults: deny-by-default APIs, capability scoping,
    # safe deserialization
    # ------------------------------------------------------------------
    "secure_defaults": [
        challenge(
            "Design a deny-by-default file API: reads require explicit "
            "path allowlist, writes disabled unless opted in",
            1,
            "pytest passes: default config rejects all 6 fixture paths; "
            "allowlisted reads succeed with no traversal possible",
            "code", "deny-by-default", "api", "filesystem"),
        challenge(
            "Replace pickle with a safe deserialization path for a job "
            "queue, rejecting non-allowlisted types",
            1,
            "pytest passes: malicious pickle payload raises on load; the "
            "same job round-trips fine through the safe serializer",
            "code", "deserialization", "pickle", "hardening"),
        challenge(
            "Build a capability-scoped HTTP client: per-host allowlist, "
            "no redirects to unlisted hosts, timeout + size caps",
            1,
            "pytest passes: SSRF fixtures (169.254.169.254, redirect hop, "
            "oversize body) all blocked; allowlisted fetch works",
            "code", "ssrf", "http", "capabilities"),
        challenge(
            "Write a YAML loader wrapper that forbids arbitrary object "
            "construction and validates against a schema",
            2,
            "pytest passes: !!python/object payload rejected; valid "
            "config loads and schema violations raise with field paths",
            "code", "yaml", "deserialization", "validation"),
        challenge(
            "Design secure defaults for a plugin system: plugins get no "
            "network/fs by default, capabilities granted per manifest",
            2,
            "pytest passes: default plugin cannot open sockets or files; "
            "manifest-granted capability works and is auditable",
            "code", "plugins", "capabilities", "least-privilege"),
        challenge(
            "Implement an HTML sanitizer with an allowlist of tags/"
            "attributes that strips event handlers and javascript: URLs",
            2,
            "pytest passes: 20 XSS fixtures sanitized to safe output; "
            "benign rich-text round-trips with formatting intact",
            "code", "xss", "sanitization", "html"),
        challenge(
            "Build a shell-command builder that forbids string "
            "concatenation and forces argv arrays with allowlisted bins",
            2,
            "pytest passes: injection fixtures ('; rm', $(), backticks) "
            "never reach exec; valid commands run and return output",
            "code", "command-injection", "shell", "api-design"),
        challenge(
            "Create a crypto-defaults module: AEAD only, random 96-bit "
            "nonces, Argon2id for passwords, no knobs to pick ECB",
            2,
            "module exposes no insecure mode at all; pytest passes: "
            "encrypt/decrypt round-trip and nonce-reuse test impossible",
            "code", "crypto", "defaults", "aead"),
        challenge(
            "Design a safe expression evaluator for user filters "
            "(no eval/exec): AST allowlist with timeout and depth limits",
            3,
            "pytest passes: 10 escape payloads (dunder, import, getattr "
            "chains) rejected; 15 legit filter expressions evaluate",
            "code", "sandboxing", "ast", "evaluator"),
        challenge(
            "Implement object-level authorization middleware: every "
            "resource access checks owner scope, default deny on miss",
            3,
            "pytest passes: cross-user IDOR fixtures all return 403/404; "
            "owner access works across 5 resource types",
            "code", "authorization", "idor", "middleware"),
        challenge(
            "Build a secure-by-default config loader: unknown keys "
            "rejected, secrets from env only, type-coerced, immutable",
            3,
            "pytest passes: typosquat key, inline secret, and type-"
            "confusion fixtures rejected; frozen config resists mutation",
            "code", "config", "defaults", "immutability"),
        challenge(
            "Design a capability-based subprocess API: no shell, fd "
            "inheritance off by default, cgroup memory/cpu caps",
            3,
            "pytest passes: child cannot inherit parent fds or exceed "
            "caps; a 1k-task batch completes under the limits",
            "code", "subprocess", "capabilities", "cgroups"),
    ],
    # ------------------------------------------------------------------
    # ml_ops: inference serving, batching, quantization, monitoring/drift
    # ------------------------------------------------------------------
    "ml_ops": [
        challenge(
            "Serve a small classifier behind FastAPI with request "
            "validation, /health, and structured JSON logging",
            1,
            "service starts, /health returns 200, 100 sequential "
            "requests complete with p99 latency logged per request",
            "build", "serving", "fastapi", "logging"),
        challenge(
            "Add Prometheus metrics (latency histogram, error counter, "
            "in-flight gauge) to an inference endpoint",
            1,
            "/metrics exposes all three series; a load run of 500 "
            "requests moves the counters and histogram buckets",
            "code", "monitoring", "prometheus", "metrics"),
        challenge(
            "Implement request batching for an embedding endpoint: "
            "dynamic batching with max wait 50ms and max batch 32",
            2,
            "load test shows throughput >=3x vs single-item and p95 "
            "added latency <=60ms; results map back to request ids",
            "code", "batching", "throughput", "embeddings"),
        challenge(
            "Quantize a transformer to int8 and measure accuracy vs "
            "latency tradeoff on a 1k-sample eval set",
            2,
            "report shows int8 model: size <=50% of fp32, latency drop "
            ">=30%, accuracy drop <=2pp on the eval set",
            "code", "quantization", "int8", "benchmark"),
        challenge(
            "Build a canary deployment harness: route 5% traffic to the "
            "new model, auto-rollback on error-rate delta",
            2,
            "harness shifts traffic, detects an injected 10% error "
            "regression, and rolls back to 0% within 60 seconds",
            "build", "canary", "deployment", "rollback"),
        challenge(
            "Implement input-drift detection on live traffic using PSI "
            "over feature histograms vs the training baseline",
            2,
            "drift job flags a synthetic shifted feature stream (PSI "
            ">0.2) and stays green on a clean replay stream",
            "code", "drift", "monitoring", "psi"),
        challenge(
            "Build a model registry with versioned artifacts, SHA256 "
            "pins, and promotion gates (eval score must improve)",
            2,
            "registry rejects promotion of a worse-scoring model and "
            "accepts a better one; pin mismatch fails the load",
            "build", "registry", "versioning", "gates"),
        challenge(
            "Implement output monitoring: toxicity/PII regex screens on "
            "LLM outputs with sampling and an alert threshold",
            3,
            "monitor flags >=90% of 50 planted PII/toxic outputs at <=5% "
            "false-positive rate on 500 clean outputs",
            "code", "monitoring", "pii", "llm"),
        challenge(
            "Build an autoscaling inference pool: scale workers on queue "
            "depth, drain gracefully, keep p99 under SLO during burst",
            3,
            "burst test 10x traffic: p99 stays under 2s SLO, workers "
            "scale up then drain to baseline with zero dropped jobs",
            "build", "autoscaling", "slo", "serving"),
        challenge(
            "Implement KV-cache-aware continuous batching for an LLM "
            "server and benchmark tokens/sec vs naive batching",
            3,
            "benchmark shows >=1.5x tokens/sec vs naive batching at "
            "equal quality on a 200-prompt fixture set",
            "code", "batching", "llm", "kv-cache"),
        challenge(
            "Build an offline eval pipeline that runs nightly: pulls the "
            "serving model, scores it on the golden set, pages on drop",
            3,
            "pipeline detects an injected 3pp accuracy regression on the "
            "golden set and emits an alert with the failing slice",
            "build", "regression", "nightly", "alerting"),
        challenge(
            "Implement shadow-mode evaluation: duplicate live traffic to "
            "a candidate model and diff outputs without user impact",
            1,
            "shadow run over 1k live requests records agreement rate and "
            "latency delta; live responses never touch the candidate",
            "build", "shadow", "evaluation", "diffing"),
    ],
    # ------------------------------------------------------------------
    # evals: benchmark harnesses, pass@k, human-vs-auto eval agreement,
    # regression gates
    # ------------------------------------------------------------------
    "evals": [
        challenge(
            "Build a tiny benchmark harness: YAML-defined tasks, model "
            "adapter interface, JSON results with per-task scores",
            1,
            "harness runs 5 fixture tasks end-to-end and writes valid "
            "JSON with per-task scores and total runtime",
            "build", "harness", "benchmark", "json"),
        challenge(
            "Implement pass@k scoring for code-gen evals with unbiased "
            "estimator over n samples",
            1,
            "pytest passes: pass@k matches hand-computed values on 4 "
            "fixture (n, c, k) triples within 1e-9",
            "code", "pass-at-k", "codegen", "metrics"),
        challenge(
            "Write an LLM-as-judge rubric for summarization quality with "
            "a 5-point scale and calibration examples",
            2,
            "judge scores a 20-item fixture set; Spearman rho >=0.7 vs "
            "the human labels shipped with the fixture",
            "code", "llm-judge", "rubric", "calibration"),
        challenge(
            "Measure human-vs-auto eval agreement: collect 100 paired "
            "labels and compute Cohen's kappa per dimension",
            2,
            "notebook reports kappa per dimension with confidence "
            "intervals; >=2 dimensions reach kappa >=0.6",
            "research", "agreement", "kappa", "human-eval"),
        challenge(
            "Build a regression-gate CI job: new model must beat baseline "
            "on the golden set within tolerance or the PR fails",
            2,
            "CI job fails a fixture PR with a 2pp regression and passes "
            "one with a 1pp gain; gate config is a single YAML",
            "build", "regression", "ci", "gates"),
        challenge(
            "Implement bootstrap confidence intervals for benchmark "
            "scores to decide if a 1pp gain is real",
            2,
            "script reports 95% CIs; on fixture data the 1pp gain is "
            "inside the CI (not significant) and 5pp is outside",
            "code", "statistics", "bootstrap", "significance"),
        challenge(
            "Build a contamination checker: n-gram overlap + canary "
            "strings between train data and the eval set",
            1,
            "checker flags a fixture eval set with 30% leaked n-grams "
            "and passes a clean one; canary string detected verbatim",
            "code", "contamination", "data-leak", "n-gram"),
        challenge(
            "Design an eval set for tool-use agents: 40 tasks with "
            "deterministic checkers, covering 6 tool categories",
            3,
            "eval set runs fully offline; reference solution scores "
            "100% and a broken-tool baseline scores <40%",
            "build", "tool-use", "eval-set", "agents"),
        challenge(
            "Implement Elo/Bradley-Terry ranking over pairwise model "
            "comparisons with tie handling and uncertainty",
            3,
            "ranker recovers the known ordering on 300 synthetic "
            "pairwise votes with 95% CIs excluding ties",
            "code", "elo", "ranking", "pairwise"),
        challenge(
            "Build a slice-based eval report: scores broken down by "
            "topic, difficulty, and input length with worst-slice alert",
            2,
            "report on fixture results names the worst slice correctly "
            "and the alert fires only when a slice drops >3pp",
            "build", "slicing", "reporting", "analysis"),
        challenge(
            "Implement cost-aware evals: track $/task and score-per-"
            "dollar to compare a big vs small model",
            3,
            "report shows score-per-dollar for both models on the same "
            "50-task set; cheaper model wins the metric by >=2x",
            "code", "cost", "efficiency", "metrics"),
        challenge(
            "Build an adversarial eval generator: mutate seed prompts "
            "with 6 perturbation types and keep only score-flipping cases",
            3,
            "generator produces >=30 kept cases from 100 seeds; kept "
            "cases flip the reference model score by >=20pp",
            "build", "adversarial", "robustness", "generation"),
    ],
    # ------------------------------------------------------------------
    # finetune: LoRA/QLoRA pipelines, dataset curation, eval-before/after,
    # GGUF export checks
    # ------------------------------------------------------------------
    "finetune": [
        challenge(
            "Write a dataset curation script: dedupe, length filter, "
            "language detect, and train/val split for JSONL instruction data",
            1,
            "script turns a 10k-row noisy fixture into a clean split: 0 "
            "exact dupes remain, val set is 10% with no leakage",
            "code", "dataset", "curation", "dedupe"),
        challenge(
            "Implement eval-before/after harness: score base vs finetuned "
            "model on the same 200-prompt set with a fixed judge",
            1,
            "harness outputs a before/after table with delta and a "
            "paired t-test p-value on the fixture runs",
            "build", "evaluation", "before-after", "harness"),
        challenge(
            "Build a LoRA training config generator: rank/alpha/dropout "
            "presets per model size with VRAM estimates",
            1,
            "generator emits valid configs for 1B/3B/8B presets; VRAM "
            "estimates match measured peaks within 20% on a smoke run",
            "code", "lora", "config", "vram"),
        challenge(
            "Run a QLoRA finetune on a 1B model (4-bit) and verify the "
            "adapter merges cleanly back to fp16",
            2,
            "merged weights load without error; tensor checksums match a "
            "reference merge and sample outputs are coherent",
            "build", "qlora", "merge", "4bit"),
        challenge(
            "Implement a data-mixture experiment: train 3 LoRA adapters "
            "on different domain mixes and compare eval deltas",
            2,
            "report shows per-domain deltas for all 3 mixes; the "
            "domain-heavy mix wins its domain by >=3pp",
            "research", "data-mixture", "lora", "experiments"),
        challenge(
            "Build a GGUF export check: convert, verify magic+version, "
            "quantize Q4_K_M, and smoke-test with llama.cpp perplexity",
            2,
            "exported GGUF passes header verification and perplexity on "
            "a 100-line fixture is within 5% of the HF model",
            "build", "gguf", "export", "quantization"),
        challenge(
            "Implement catastrophic-forgetting probes: 50 general-"
            "knowledge prompts scored before and after finetuning",
            2,
            "probe report shows retention >=95% of pre-finetune score "
            "on the general set while the target task improves",
            "code", "forgetting", "probes", "retention"),
        challenge(
            "Write a hyperparameter sweep runner for LoRA (lr, rank, "
            "epochs) with early stopping and best-checkpoint selection",
            2,
            "runner completes a 6-config sweep on fixture data; the "
            "selected checkpoint beats the median config by >=2pp",
            "build", "sweep", "hyperparameters", "lora"),
        challenge(
            "Build a preference-data (DPO) pipeline: format pairs, train, "
            "and verify win-rate lift vs the SFT baseline",
            3,
            "DPO model wins >=60% of 100 pairwise comparisons vs the "
            "SFT baseline under a fixed judge",
            "build", "dpo", "preference", "alignment"),
        challenge(
            "Implement gradient-checkpointing + 8-bit optimizer memory "
            "profiling to fit a 3B LoRA run on a 12GB GPU",
            3,
            "profiling report shows peak VRAM <=11GB for the 3B run; "
            "training completes 100 steps without OOM",
            "code", "memory", "vram", "optimization"),
        challenge(
            "Build a dataset contamination audit for finetuning: check "
            "train rows against the eval set with embedding similarity",
            3,
            "audit flags >=90% of 50 planted near-duplicate eval rows "
            "at <=5% false positives on 500 clean rows",
            "code", "contamination", "embeddings", "audit"),
        challenge(
            "Implement full-parameter finetune checkpoint resumption: "
            "kill mid-epoch, resume, and prove bitwise-identical loss "
            "trajectory vs uninterrupted run",
            3,
            "resumed run's loss curve matches the uninterrupted run "
            "within 1e-6 for 50 steps after resume",
            "code", "checkpointing", "resumption", "reproducibility"),
    ],
    # ------------------------------------------------------------------
    # rag: chunking strategies, hybrid retrieval, reranking,
    # citation faithfulness
    # ------------------------------------------------------------------
    "rag": [
        challenge(
            "Build a RAG pipeline over 200 docs with hybrid BM25+vector "
            "retrieval and citation-faithfulness scoring",
            2,
            "pytest passes: recall@5 >= 0.8 on the 50-question fixture "
            "set; every answer carries >=1 cited chunk id",
            "build", "hybrid", "retrieval", "citations"),
        challenge(
            "Compare 4 chunking strategies (fixed, sentence, semantic, "
            "recursive) on the same corpus and report recall@k",
            1,
            "report shows recall@5 for all 4 strategies on the fixture "
            "set; best strategy beats worst by a measured margin",
            "research", "chunking", "comparison", "recall"),
        challenge(
            "Implement a reranker stage: cross-encoder rescores top-50 "
            "to top-5 and measure nDCG lift",
            2,
            "reranked top-5 nDCG@5 >= 0.15 above the first-stage "
            "baseline on the 50-question fixture set",
            "code", "reranking", "ndcg", "retrieval"),
        challenge(
            "Build citation-faithfulness scoring: NLI model checks each "
            "claim against its cited chunk",
            2,
            "scorer flags >=85% of 40 planted unsupported claims and "
            "keeps false positives <=10% on 100 supported claims",
            "code", "faithfulness", "nli", "citations"),
        challenge(
            "Implement query rewriting (HyDE + multi-query) and measure "
            "recall lift vs the raw query",
            1,
            "report shows recall@10 lift >=10pp on the fixture set for "
            "at least one rewriting strategy",
            "code", "query-rewriting", "hyde", "recall"),
        challenge(
            "Build an eval fixture: 50 questions with gold chunk ids and "
            "answers over a synthetic 200-doc corpus",
            1,
            "fixture validates: gold answers are retrievable (recall@20 "
            "= 1.0) and a README documents the generation recipe",
            "build", "eval-fixture", "dataset", "ground-truth"),
        challenge(
            "Implement metadata filtering (date range, source type) "
            "combined with vector search",
            2,
            "pytest passes: filtered queries return only in-scope docs; "
            "recall@5 on scoped questions >= 0.85",
            "code", "filtering", "metadata", "vector-search"),
        challenge(
            "Build a RAG system that refuses when retrieval confidence "
            "is low instead of hallucinating",
            2,
            "on 30 unanswerable fixture questions the system abstains "
            ">=80% of the time; answerable accuracy stays >=85%",
            "build", "abstention", "hallucination", "confidence"),
        challenge(
            "Implement incremental indexing: add/remove docs without a "
            "full re-embed, with a tombstone + version scheme",
            3,
            "after 100 upserts the index serves correct results for new "
            "docs and returns nothing for deleted ones; no full rebuild",
            "code", "indexing", "incremental", "updates"),
        challenge(
            "Build a multi-hop RAG that decomposes questions, retrieves "
            "per hop, and stitches cited intermediate answers",
            3,
            "on 25 two-hop fixture questions, exact-match >= 0.6 with "
            "both hops cited; single-hop baseline scores <0.3",
            "build", "multi-hop", "reasoning", "citations"),
        challenge(
            "Implement embedding-model A/B testing: swap encoders and "
            "measure recall@k + latency per encoder",
            3,
            "report compares >=3 encoders on recall@5 and p50 latency; "
            "the swap is a one-line config change",
            "research", "embeddings", "ab-testing", "benchmark"),
        challenge(
            "Build a chunk-leakage guard: verify answers never reveal "
            "chunks outside the user's access scope",
            3,
            "pytest passes: 20 cross-scope probe questions leak 0 "
            "restricted chunks; in-scope recall@5 stays >= 0.8",
            "code", "access-control", "leakage", "security"),
    ],
    # ------------------------------------------------------------------
    # agents: tool-use loops, planning, multi-agent orchestration,
    # sandboxing, trajectory evals
    # ------------------------------------------------------------------
    "agents": [
        challenge(
            "Build a ReAct tool-use loop: thought/action/observation "
            "with 3 tools and a max-step guard",
            1,
            "agent solves 8/10 fixture tasks within 12 steps; the step "
            "guard fires cleanly on the 2 unsolvable ones",
            "build", "react", "tool-use", "loop"),
        challenge(
            "Implement a planner-executor split: planner writes a step "
            "plan, executor runs tools, replanner on failure",
            3,
            "on 15 fixture tasks the split beats a single-loop baseline "
            "by >=15pp success rate with a logged plan per task",
            "build", "planning", "executor", "replanner"),
        challenge(
            "Sandbox agent tool calls: filesystem jail, network deny, "
            "timeout per call, and an audit log of every invocation",
            2,
            "pytest passes: jailbreak fixtures (path escape, socket, "
            "sleep-bomb) are blocked and logged; legit calls succeed",
            "code", "sandboxing", "tools", "audit"),
        challenge(
            "Build trajectory evals: record full agent traces and score "
            "them for efficiency, tool misuse, and goal completion",
            2,
            "evaluator scores 30 fixture traces; agreement with human "
            "labels reaches Cohen's kappa >= 0.65",
            "build", "trajectory", "evaluation", "traces"),
        challenge(
            "Implement a multi-agent debate: 3 agents propose, critique, "
            "and vote on answers to 20 questions",
            2,
            "debate accuracy beats the best single agent by >=10pp on "
            "the fixture set; all votes are logged",
            "build", "multi-agent", "debate", "voting"),
        challenge(
            "Build a tool registry with JSON-schema validation, "
            "capability scopes, and per-tool rate limits",
            1,
            "pytest passes: schema-invalid args rejected, out-of-scope "
            "tool calls denied, rate limit trips at the configured N",
            "code", "tools", "registry", "schema"),
        challenge(
            "Implement agent memory: short-term scratchpad + long-term "
            "vector store with relevance-based recall",
            2,
            "agent recalls a fact stored 50 turns earlier in 4/5 probe "
            "trials; scratchpad never exceeds the token budget",
            "code", "memory", "vector-store", "recall"),
        challenge(
            "Build an orchestrator that fans out subtasks to worker "
            "agents and merges results with conflict resolution",
            3,
            "orchestrator completes a 5-subtask fixture job 3x faster "
            "than serial with all conflicts resolved per the policy",
            "build", "orchestration", "fan-out", "merge"),
        challenge(
            "Implement prompt-injection defenses for tool outputs: "
            "delimiters, output validation, and instruction-hierarchy",
            3,
            "pytest passes: 25 injected tool-output fixtures are "
            "neutralized; 15 benign outputs pass through unchanged",
            "code", "prompt-injection", "defense", "tools"),
        challenge(
            "Build a cost/latency budget enforcer: per-task token and "
            "tool-call caps with graceful degradation",
            1,
            "enforcer halts a runaway fixture agent at exactly the cap; "
            "degraded mode still returns a partial answer",
            "code", "budget", "cost", "guardrails"),
        challenge(
            "Implement self-correction: agent critiques its own draft "
            "answer with a checklist, then revises once",
            2,
            "on 20 fixture tasks self-correction lifts the judge score "
            "by >=10pp vs single-pass with the checklist logged",
            "build", "self-correction", "critique", "revision"),
        challenge(
            "Build a red-team harness for agents: 30 adversarial tasks "
            "(jailbreak, tool abuse, data exfil) with pass/fail checkers",
            3,
            "harness runs fully offline; hardened agent passes >=80% "
            "while the unhardened baseline passes <40%",
            "build", "red-team", "adversarial", "safety"),
    ],
}

register_topic_pack("chall_ml", _PACK)
