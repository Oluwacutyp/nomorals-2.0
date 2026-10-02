"""Challenge pack: product-domain challenges with verifiable outcomes.

Nine categories covering the product surface of an AI agent: chat UX,
latency, model routing, media pipelines, OCR, vision editing, research
methods, source trust, and distillation. Each entry is a concrete task
(built via ``challenge()``) with a one-line checkable acceptance
criterion — no essay prompts.

Self-registers as the ``"chall_product"`` topic pack on import.
"""

from __future__ import annotations

from ..challenges import challenge
from ..topics import register_topic_pack

_PACK = {
    "ux_chat": [
        # ---- easy (d=1) ----
        challenge(
            "Implement streaming token rendering with interruption handling "
            "and a 300ms first-token budget",
            1,
            "pytest passes: 50 simulated sessions, zero dropped tokens, "
            "p95 first-token < 300ms",
            "code", "streaming", "tokens", "ux",
        ),
        challenge(
            "Build a chat retry affordance that resumes a failed send exactly "
            "once without duplicating messages",
            1,
            "test passes: 200 flaky-network sims, zero duplicate message ids, "
            "resume state persisted",
            "code", "retry", "affordances", "idempotency",
        ),
        challenge(
            "Implement a typing indicator with graceful timeout: shows "
            "'typing', degrades to 'still working' after 8s",
            1,
            "test passes: indicator states transition on schedule across 20 "
            "timed scenarios",
            "code", "presence", "latency-perception", "ux",
        ),
        # ---- medium (d=2) ----
        challenge(
            "Build multi-turn conversation editing where editing the last "
            "message re-forks history without corrupting branches",
            2,
            "pytest passes: 100 fork/edit/delete sequences, branch histories "
            "diff-clean",
            "code", "editing", "history", "branching",
        ),
        challenge(
            "Implement stop/interrupt handling that cleanly cancels a "
            "streaming response and preserves the partial text",
            2,
            "test passes: 60 interrupt-at-random-offset runs, partial text "
            "intact, no orphaned tokens",
            "code", "interruption", "streaming", "cancellation",
        ),
        challenge(
            "Build an error-recovery flow for a dead LLM backend: user-visible "
            "status, queued retries with backoff, draft preservation",
            2,
            "demo passes: backend killed mid-stream, draft preserved, send "
            "resumes on recovery, backoff logged",
            "build", "error-recovery", "resilience", "queueing",
        ),
        challenge(
            "Implement adaptive latency perception: skeleton placeholders plus "
            "progressive disclosure tuned to measured TTFT bands",
            2,
            "test passes: placeholder strategy chosen from 4 TTFT bands, "
            "renders within 50ms of first byte",
            "code", "latency-perception", "skeleton", "progressive",
        ),
        challenge(
            "Build a suggestion-chip / quick-reply affordance system driven by "
            "conversation context with a max-3 display rule",
            2,
            "test passes: chips render from context on 30 fixtures, never more "
            "than 3, click maps to intent",
            "code", "affordances", "suggestions", "context",
        ),
        # ---- deep (d=3) ----
        challenge(
            "Design and implement a full conversation state machine covering "
            "streaming, interrupted, failed, retried, forked, and archived "
            "states with exhaustive transition coverage",
            3,
            "pytest passes: model-checker covers all 7 states x all "
            "transitions, zero illegal transitions",
            "code", "state-machine", "correctness", "streaming",
        ),
        challenge(
            "Build a chat accessibility harness: screen-reader live-region "
            "announcements for streamed tokens at throttled cadence, focus "
            "management, keyboard-only flows",
            3,
            "test passes: audit clean, announcement cadence <= 2/sec, full "
            "keyboard flow script green",
            "build", "accessibility", "aria", "keyboard",
        ),
        challenge(
            "Implement conversation compaction (context summarization) with "
            "user-visible boundary markers and on-demand expansion of "
            "summarized regions",
            3,
            "test passes: 20 long sessions compact without losing named "
            "entities, expansion restores verbatim text",
            "code", "compaction", "context", "transparency",
        ),
        challenge(
            "Build a multi-device sync conflict resolver for chat state "
            "(order, edits, deletions) using CRDT-style merge",
            3,
            "test passes: 500 randomized op interleavings converge to identical "
            "history on 3 replicas",
            "code", "sync", "crdt", "conflict-resolution",
        ),
    ],
    "latency": [
        # ---- easy (d=1) ----
        challenge(
            "Implement a p50/p95/p99 latency percentile calculator over a "
            "rolling window that alerts when p99 exceeds budget",
            1,
            "pytest passes: 10k synthetic samples, percentile error < 1%, "
            "alert fires on budget breach",
            "code", "percentiles", "monitoring", "slo",
        ),
        challenge(
            "Build a first-token (TTFT) timer measuring time-to-first-byte "
            "from request start across a chat client",
            1,
            "test passes: timer reports TTFT within 5ms of ground truth on "
            "100 requests",
            "code", "ttft", "measurement", "chat",
        ),
        challenge(
            "Implement an in-memory TTL cache for repeated LLM prompts with "
            "hit-rate reporting",
            1,
            "pytest passes: 1000 mixed queries, hit-rate >= 60% on 70% "
            "duplicate stream, TTL expiry honored",
            "code", "caching", "ttl", "hit-rate",
        ),
        # ---- medium (d=2) ----
        challenge(
            "Build a tail-latency hunter: correlate slow requests with request "
            "attributes (tokens, model, time-of-day) and report top-3 "
            "contributors",
            2,
            "script identifies injected slow-model contributor on 5k-request "
            "trace with 100% attribution",
            "code", "tail-latency", "correlation", "profiling",
        ),
        challenge(
            "Implement prompt-prefix caching (KV-cache reuse) benchmark "
            "comparing cached vs cold TTFT across prompt lengths",
            2,
            "benchmark shows >= 2x TTFT reduction at 4k prefix on 30 prompts, "
            "numbers reproducible across 3 runs",
            "code", "kv-cache", "prefix-caching", "benchmark",
        ),
        challenge(
            "Build a latency-budget allocator that splits an end-to-end SLO "
            "(e.g. 2s) across pipeline stages and flags over-budget stages",
            2,
            "test passes: allocator flags the right stage on 25 staged traces, "
            "budgets sum to <= SLO",
            "code", "budgets", "slo", "pipeline",
        ),
        challenge(
            "Implement adaptive request batching for an inference server: batch "
            "up to N or T ms, whichever hits first, and measure throughput vs "
            "added latency",
            2,
            "sim passes: batcher hits >= 80% of optimal throughput with <= "
            "50ms added p99",
            "code", "batching", "throughput", "inference",
        ),
        challenge(
            "Build a streaming-vs-batch latency comparator charting TTFT, "
            "inter-token latency, and total time across 50 prompts",
            2,
            "report shows streaming TTFT >= 3x faster than batch-wait on all "
            "50 prompts, chart artifact saved",
            "code", "streaming", "comparison", "tokens",
        ),
        # ---- deep (d=3) ----
        challenge(
            "Implement speculative decoding (draft + target model) for latency "
            "reduction and measure acceptance rate vs speedup",
            3,
            "pytest passes: acceptance >= 60%, end-to-end speedup >= 1.5x on "
            "100 prompts, outputs byte-identical",
            "code", "speculative-decoding", "speedup", "llm",
        ),
        challenge(
            "Build a distributed trace correlator stitching client->gateway->"
            "model->token latencies into one waterfall with per-span budgets",
            3,
            "demo traces 200 requests end-to-end, waterfall renders, budget "
            "violations highlighted per span",
            "build", "tracing", "waterfall", "observability",
        ),
        challenge(
            "Implement a cold-start eliminator for serverless inference: "
            "warm-pool sizing model derived from arrival-rate data",
            3,
            "sim on 24h arrival trace: cold starts < 2% with <= 30% "
            "over-provisioned warm capacity",
            "code", "cold-start", "serverless", "autoscaling",
        ),
        challenge(
            "Design and run a latency SLO experiment proving a caching layer "
            "moves p99 from >2s to <800ms with statistical significance on "
            "production-shaped load",
            3,
            "report: 10k-request A/B, p99 delta significant at p < 0.01 "
            "(Mann-Whitney), method documented",
            "research", "slo", "experiment", "significance",
        ),
    ],
    "routing": [
        # ---- easy (d=1) ----
        challenge(
            "Implement a keyword/heuristic router sending simple queries to a "
            "fast model and complex ones to a heavy model",
            1,
            "test passes: 200 labeled queries, >= 85% route agreement with "
            "oracle labels",
            "code", "heuristic", "fast-path", "routing",
        ),
        challenge(
            "Build a cost tracker logging per-request model, tokens, and "
            "estimated cost, then reporting daily spend by route",
            1,
            "test passes: 500 logged requests, cost math matches price table "
            "to the cent, daily rollup correct",
            "code", "cost", "tracking", "observability",
        ),
        challenge(
            "Implement round-robin plus least-latency fallback across two "
            "model endpoints",
            1,
            "test passes: primary failure simulated, 100% of traffic served by "
            "fallback, latency-aware picks faster endpoint",
            "code", "fallback", "load-balancing", "resilience",
        ),
        # ---- medium (d=2) ----
        challenge(
            "Build a learned router: train a small classifier on 1000 labeled "
            "queries to pick fast vs heavy path, with confidence threshold "
            "and abstain-to-heavy",
            2,
            "router hits >= 90% agreement with labels, abstention rate < 15%, "
            "calibration error < 0.1",
            "code", "classifier", "ml", "confidence",
        ),
        challenge(
            "Implement a cost/quality Pareto frontier: sweep router thresholds "
            "and plot cost vs quality score to pick the operating point",
            2,
            "sweep over 20 thresholds yields a frontier plot; chosen point "
            "saves >= 30% cost at <= 2% quality drop",
            "code", "pareto", "cost-quality", "tuning",
        ),
        challenge(
            "Build a circuit-breaker fallback chain: primary -> secondary -> "
            "cached answer, with breaker state transitions tested",
            2,
            "pytest passes: breaker opens after 5 failures, half-opens after "
            "30s, 100% requests eventually served",
            "code", "circuit-breaker", "fallback", "resilience",
        ),
        challenge(
            "Implement semantic caching in the router: embed the incoming "
            "query, serve cached response on cosine similarity >= 0.97",
            2,
            "test passes: 40 near-duplicate queries served from cache, "
            "precision 100%, no false hits on 40 distinct queries",
            "code", "semantic-cache", "embeddings", "deduplication",
        ),
        challenge(
            "Build a router benchmark harness scoring cheapest-capable-model "
            "selection on 100 labeled queries",
            2,
            "benchmark passes: router picks the cheapest capable model on 100 "
            "labeled queries with >= 90% agreement",
            "code", "benchmark", "evaluation", "cost",
        ),
        # ---- deep (d=3) ----
        challenge(
            "Implement a multi-armed-bandit router learning per-query-type "
            "model assignment online from reward (quality/cost) signals",
            3,
            "sim passes: bandit regret < 10% vs oracle after 5k queries, "
            "converges within 1k pulls",
            "code", "bandit", "online-learning", "routing",
        ),
        challenge(
            "Build a cascade router: try tiny model, escalate on low "
            "confidence, cap total latency budget; prove quality parity with "
            "always-heavy at lower cost",
            3,
            "eval on 500 queries: quality within 1% of always-heavy, cost <= "
            "50%, latency budget honored on p99",
            "code", "cascade", "escalation", "confidence",
        ),
        challenge(
            "Design a router chaos experiment: inject provider outages and "
            "latency spikes, measure recovery time and user-visible error rate",
            3,
            "report: 6 fault scenarios, recovery < 60s each, user error rate "
            "< 1%, runbook produced",
            "research", "chaos", "resilience", "experiment",
        ),
        challenge(
            "Implement a context-aware router routing by user tier, query "
            "domain, and privacy constraints (local-only for PII)",
            3,
            "pytest passes: 120 policy fixtures, zero PII routed to cloud, "
            "tier rules 100% honored",
            "code", "policy", "privacy", "tiers",
        ),
    ],
    "media_pipelines": [
        # ---- easy (d=1) ----
        challenge(
            "Implement a thumbnail generator: any image in -> 320px JPEG "
            "thumb with preserved aspect ratio",
            1,
            "test passes: 20 images incl. portrait/landscape/alpha, thumbs <= "
            "320px, correct ratio, < 200ms each",
            "code", "thumbnails", "images", "resize",
        ),
        challenge(
            "Build an EXIF reader extracting orientation, GPS, and capture "
            "date, normalizing orientation on load",
            1,
            "test passes: 12 EXIF fixtures, orientation normalized, GPS "
            "parsed to +-1e-6 degrees",
            "code", "exif", "metadata", "orientation",
        ),
        challenge(
            "Implement an image format converter (PNG/JPEG/WebP/AVIF) with "
            "quality parameter and output-size reporting",
            1,
            "test passes: 4x4 format matrix converts, WebP <= 60% of PNG size "
            "at quality 80 with PSNR >= 35dB",
            "code", "transcoding", "formats", "conversion",
        ),
        # ---- medium (d=2) ----
        challenge(
            "Build a transcode DAG: decode -> resize -> filter -> encode as a "
            "declarative graph with per-node timing and cacheable intermediate "
            "nodes",
            2,
            "test passes: graph executes 10-node pipeline, node cache hits on "
            "rerun, total time logged per node",
            "code", "dag", "transcoding", "pipeline",
        ),
        challenge(
            "Implement deterministic/reproducible media builds: same inputs + "
            "same pipeline version -> byte-identical outputs (strip "
            "timestamps, pin encoders)",
            2,
            "test passes: 3 runs of 5 assets, sha256 identical across runs "
            "and across two machines",
            "code", "reproducible", "determinism", "builds",
        ),
        challenge(
            "Build a video keyframe extractor plus contact-sheet generator: N "
            "evenly spaced frames -> grid montage with timestamps",
            2,
            "test passes: 5 videos, sheet has N frames +-1, timestamps match "
            "frame times within 100ms",
            "code", "video", "keyframes", "montage",
        ),
        challenge(
            "Implement perceptual-hash duplicate detection (pHash) across a "
            "media library with near-duplicate clustering",
            2,
            "test passes: 100 images with 15 near-dup pairs, all pairs "
            "clustered, zero false merges",
            "code", "phash", "dedup", "clustering",
        ),
        challenge(
            "Build an audio normalization pipeline: loudness to -16 LUFS, peak "
            "limit, silence trim, with before/after loudness report",
            2,
            "test passes: 10 clips, output loudness -16 +- 1 LUFS, no "
            "clipping, report lists deltas",
            "code", "audio", "loudness", "normalization",
        ),
        # ---- deep (d=3) ----
        challenge(
            "Implement a resumable chunked media upload/download pipeline with "
            "per-chunk checksums and parallel streams",
            3,
            "test passes: 2GB file over simulated flaky link, resume from 50% "
            "works, final sha256 matches",
            "code", "upload", "chunked", "resumable",
        ),
        challenge(
            "Build a content-aware adaptive bitrate ladder generator: analyze "
            "complexity, emit 4 renditions, verify VMAF ordering",
            3,
            "demo on 3 videos: ladder produced, VMAF monotonic with bitrate, "
            "1080p rendition VMAF >= 90",
            "build", "abr", "transcoding", "vmaf",
        ),
        challenge(
            "Implement a media provenance ledger: every pipeline run records "
            "inputs, params, tool versions, output hashes - queryable by "
            "output hash",
            3,
            "test passes: 20 runs logged, hash lookup returns full lineage, "
            "tampered log entry detected",
            "code", "provenance", "ledger", "audit",
        ),
        challenge(
            "Design a storage-tiering experiment for a 10TB media library: "
            "hot/warm/cold placement policy vs access-cost simulation",
            3,
            "report: sim on 90-day access log shows >= 40% cost saving vs "
            "all-hot, policy rules documented",
            "research", "storage", "tiering", "cost",
        ),
    ],
    "ocr": [
        # ---- easy (d=1) ----
        challenge(
            "Build an OCR runner extracting text from 40 scanned samples and "
            "reporting mean CER",
            1,
            "script scores 40 OCR samples with CER < 5% on clean scans, "
            "report lists per-sample CER",
            "code", "ocr", "cer", "scoring",
        ),
        challenge(
            "Implement a text-cleanup post-processor fixing common OCR "
            "confusions (0/O, 1/l) and normalizing whitespace",
            1,
            "test passes: 200 synthetic OCR errors, >= 80% corrected, zero "
            "introduced errors on clean text",
            "code", "postprocessing", "cleanup", "text",
        ),
        challenge(
            "Build a PDF text-layer vs OCR comparator flagging pages where "
            "the embedded text layer disagrees with OCR",
            1,
            "test passes: 15 PDFs, flags the 3 with broken text layers, "
            "agreement metric reported",
            "code", "pdf", "comparison", "validation",
        ),
        # ---- medium (d=2) ----
        challenge(
            "Implement layout analysis: detect text blocks, columns, and "
            "reading order on 20 multi-column documents",
            2,
            "test passes: reading order correct on 18/20 docs, block IoU >= "
            "0.8 vs ground truth",
            "code", "layout", "columns", "reading-order",
        ),
        challenge(
            "Build a handwriting OCR benchmark scoring a handwriting model on "
            "100 samples, reporting CER/WER by writer",
            2,
            "benchmark: 100 samples, CER/WER reported per writer, best-config "
            "CER < 15%",
            "code", "handwriting", "benchmark", "cer",
        ),
        challenge(
            "Implement table extraction: detect table structure from a scanned "
            "page and emit CSV with cell alignment verified",
            2,
            "test passes: 12 table images, >= 90% cell accuracy, CSV "
            "round-trips through parser",
            "code", "tables", "extraction", "csv",
        ),
        challenge(
            "Build a multi-language OCR router: detect script/language per "
            "region and dispatch to the right engine",
            2,
            "test passes: 30 mixed-script pages, correct engine chosen >= "
            "90%, CER within 2pp of monolingual",
            "code", "multilingual", "detection", "routing",
        ),
        challenge(
            "Implement OCR confidence calibration: map engine confidences to "
            "empirical accuracy bins and flag low-confidence spans",
            2,
            "test passes: calibration curve within 5pp of diagonal on 5k "
            "words, flags capture 90% of errors",
            "code", "confidence", "calibration", "quality",
        ),
        # ---- deep (d=3) ----
        challenge(
            "Build an end-to-end document understanding pipeline: OCR -> "
            "layout -> entities -> structured JSON with field-level accuracy "
            "report",
            3,
            "demo on 25 invoices: >= 90% field accuracy on total/date/vendor, "
            "JSON schema-valid",
            "build", "pipeline", "entities", "structured",
        ),
        challenge(
            "Implement a degraded-document OCR stress suite: blur, noise, "
            "skew, low-contrast variants with CER degradation curves",
            3,
            "suite: 200 degraded variants, CER-vs-degradation curves plotted, "
            "worst-case CER < 25%",
            "code", "robustness", "stress", "degradation",
        ),
        challenge(
            "Design a human-in-the-loop OCR correction experiment measuring "
            "correction throughput and residual error with/without confidence "
            "highlighting",
            3,
            "report: 5 annotators x 50 pages, highlighting cuts correction "
            "time >= 25%, residual CER < 1%",
            "research", "hitl", "experiment", "throughput",
        ),
        challenge(
            "Implement receipt/financial-document OCR with line-item parsing "
            "and totals reconciliation (sum of items == total)",
            3,
            "test passes: 30 receipts, >= 85% line-item accuracy, totals "
            "reconcile on all parseable receipts",
            "code", "receipts", "parsing", "reconciliation",
        ),
    ],
    "vision_edit": [
        # ---- easy (d=1) ----
        challenge(
            "Implement a rectangular region selector that crops/edits a masked "
            "region and composites it back seamlessly",
            1,
            "test passes: 10 images, masked region edited, border seam PSNR "
            ">= 40dB vs original surround",
            "code", "masks", "crop", "compositing",
        ),
        challenge(
            "Build an edit-history stack: every vision edit records params + "
            "mask, supporting undo/redo of 20 steps",
            1,
            "test passes: 20 random edits then full undo, pixel-identical to "
            "original, redo restores each step",
            "code", "history", "undo", "editing",
        ),
        challenge(
            "Implement a before/after comparison view generator: side-by-side "
            "+ slider HTML for an edit pair",
            1,
            "test passes: HTML renders for 5 edit pairs, slider position maps "
            "to blend ratio +-1%",
            "code", "comparison", "html", "preview",
        ),
        # ---- medium (d=2) ----
        challenge(
            "Build a mask-refinement pipeline: rough mask -> feather -> "
            "edge-aware refine, with IoU vs ground-truth masks",
            2,
            "test passes: 15 masks, refined IoU >= 0.92 vs ground truth, "
            "feather radius configurable",
            "code", "masks", "refinement", "segmentation",
        ),
        challenge(
            "Implement a reproducible edit graph: nodes = operations with "
            "seeds/params, re-running the graph yields byte-identical output",
            2,
            "test passes: 8-op graph re-run 3x, sha256 identical, JSON graph "
            "serializes/deserializes losslessly",
            "code", "edit-graph", "reproducibility", "seeds",
        ),
        challenge(
            "Build an inpainting pipeline: mask + prompt -> filled region, "
            "with a quality gate (LPIPS vs surrounding context threshold)",
            2,
            "demo: 10 masked images inpainted, LPIPS gate rejects the 2 "
            "worst, accepted edits LPIPS < 0.3",
            "code", "inpainting", "quality-gate", "lpips",
        ),
        challenge(
            "Implement object removal with background reconstruction and a "
            "no-ghosting check (template-match for remnants)",
            2,
            "test passes: 12 removals, template match finds zero remnants "
            "above 0.7 correlation",
            "code", "removal", "inpainting", "verification",
        ),
        challenge(
            "Build an edit-quality metric suite: PSNR, SSIM, LPIPS between "
            "edit and reference across a 50-pair benchmark",
            2,
            "suite runs on 50 pairs, all three metrics computed, ranking "
            "correlates with human labels (rho >= 0.7)",
            "code", "metrics", "psnr", "ssim",
        ),
        # ---- deep (d=3) ----
        challenge(
            "Implement instruction-based editing with region grounding: parse "
            "'remove the red car' -> detect box -> mask -> edit -> verify "
            "removal",
            3,
            "demo: 15 instructions, correct region located >= 80%, edit "
            "applied, detector confirms removal",
            "build", "grounding", "detection", "editing",
        ),
        challenge(
            "Build a multi-turn edit session: chain 5 dependent edits with "
            "intermediate previews and per-step rollback",
            3,
            "test passes: 5-step chains on 8 images, rollback to any step "
            "restores exact pixels, previews generated",
            "code", "multi-turn", "sessions", "rollback",
        ),
        challenge(
            "Design a blind A/B evaluation of two inpainting models: protocol, "
            "200 image pairs, rater agreement measured",
            3,
            "report: protocol executed on 200 pairs, inter-rater kappa >= "
            "0.6, winner significant at p < 0.05",
            "research", "evaluation", "ab-test", "protocol",
        ),
        challenge(
            "Implement style transfer with a content-preservation guard: style "
            "applied, content drift (LPIPS vs original) capped",
            3,
            "test passes: 10 styles, drift LPIPS < 0.45 while style classifier "
            "confidence >= 0.8",
            "code", "style-transfer", "guardrails", "lpips",
        ),
    ],
    "research_methods": [
        # ---- easy (d=1) ----
        challenge(
            "Implement a reproducible experiment scaffold: seeded RNG, config "
            "JSON, results CSV, one-command rerun",
            1,
            "test passes: two runs with same seed produce byte-identical "
            "results CSV, config logged",
            "code", "reproducibility", "scaffolding", "seeds",
        ),
        challenge(
            "Build a baseline runner: run 3 standard baselines on a toy task "
            "and tabulate mean +- std over 5 seeds",
            1,
            "script runs 3 baselines x 5 seeds, table shows mean +- std, "
            "completes in < 5 minutes",
            "code", "baselines", "seeds", "tabulation",
        ),
        challenge(
            "Implement a train/val/test splitter with stratification and a "
            "leakage check (no overlapping ids across splits)",
            1,
            "pytest passes: 10k-row dataset, stratified ratios within 1%, "
            "zero id overlap across splits",
            "code", "splits", "stratification", "leakage",
        ),
        # ---- medium (d=2) ----
        challenge(
            "Build an ablation harness: toggle N components, run all 2^N "
            "configs (N <= 5), attribute performance deltas",
            2,
            "harness on 4 components: all 16 configs run, deltas attributed, "
            "top component identified matches injected truth",
            "code", "ablation", "factorial", "attribution",
        ),
        challenge(
            "Implement statistical significance testing: paired t-test + "
            "bootstrap CIs for comparing two systems on 200 examples",
            2,
            "test passes: detects injected 3% lift at p < 0.05 with 90% "
            "power, CI covers true lift",
            "code", "significance", "t-test", "bootstrap",
        ),
        challenge(
            "Build a hyperparameter search with early stopping and a fixed "
            "compute budget, reporting the budget-vs-best curve",
            2,
            "search on 30 configs: stops losers early, finds config within 2% "
            "of full-budget best at <= 50% cost",
            "code", "hpo", "early-stopping", "budget",
        ),
        challenge(
            "Implement a dataset bias audit: slice metrics by subgroup, report "
            "worst-group vs average gap with CIs",
            2,
            "audit on labeled data: worst-group gap detected within 1pp of "
            "injected truth, slices all n >= 30",
            "code", "bias", "subgroups", "fairness",
        ),
        challenge(
            "Build a literature-review matrix builder: given 20 papers "
            "(title/abstract/method/result), emit a comparison table with gap "
            "analysis",
            2,
            "matrix covers all 20 papers, columns consistent, gap statement "
            "cites >= 3 supporting papers",
            "research", "literature", "synthesis", "gaps",
        ),
        # ---- deep (d=3) ----
        challenge(
            "Design a pre-registration template + checklist for an ML "
            "experiment and apply it to a real planned run",
            3,
            "template filled for a real experiment: hypotheses, metrics, "
            "stopping rules, analysis plan all specified",
            "research", "preregistration", "rigor", "planning",
        ),
        challenge(
            "Implement a multiple-comparison correction pipeline: run 50 "
            "hypothesis tests, apply Holm-Bonferroni, report adjusted "
            "significance",
            3,
            "test passes: on synthetic data with 5 true effects, exactly "
            "those 5 survive correction at alpha=0.05",
            "code", "multiple-testing", "holm", "fwer",
        ),
        challenge(
            "Build a replication study kit: take a published result, "
            "reimplement from the paper alone, document every deviation",
            3,
            "report: reimplementation matches claimed result within 5%, "
            "deviation log lists all 12+ judgment calls",
            "research", "replication", "documentation", "rigor",
        ),
        challenge(
            "Implement a causal-inference check: difference-in-differences on "
            "observational data with parallel-trends diagnostic",
            3,
            "test passes: recovers injected treatment effect within 10%, "
            "parallel-trends plot shows no pre-trend",
            "code", "causal", "did", "diagnostics",
        ),
    ],
    "source_trust": [
        # ---- easy (d=1) ----
        challenge(
            "Implement a URL provenance extractor: given a URL, record "
            "domain, fetch date, author/byline, and canonical link",
            1,
            "test passes: 30 URLs, provenance fields extracted >= 90%, "
            "canonical links resolved",
            "code", "provenance", "urls", "metadata",
        ),
        challenge(
            "Build a citation-format validator checking that citations in a "
            "document match a required schema (author, year, source, URL)",
            1,
            "test passes: 50 citations, flags all 8 malformed ones, zero "
            "false positives",
            "code", "citations", "validation", "schema",
        ),
        challenge(
            "Implement a freshness scorer rating a source 0-1 from publication "
            "date, update frequency, and last-verified timestamp",
            1,
            "test passes: scorer ranks 20 sources, stale sources (< 2023, "
            "unverified) score < 0.3",
            "code", "freshness", "scoring", "recency",
        ),
        # ---- medium (d=2) ----
        challenge(
            "Build a claim-to-source verifier: given a claim and a source "
            "text, output supported/contradicted/unclear with quoted evidence "
            "spans",
            2,
            "test passes: 60 claim-source pairs, >= 85% agreement with human "
            "labels, evidence spans quoted",
            "code", "verification", "evidence", "nli",
        ),
        challenge(
            "Implement conflict detection: ingest 5 sources on one topic, "
            "cluster agreeing/disagreeing claims, surface contradictions",
            2,
            "test passes: detects all 6 injected contradictions across 5 "
            "sources, clusters agree with labels",
            "code", "conflict", "clustering", "contradictions",
        ),
        challenge(
            "Build a domain-reputation scorer from blocklists, HTTPS status, "
            "and registration age, with an explainable score breakdown",
            2,
            "test passes: 100 domains scored, known-bad domains in bottom "
            "decile, breakdown sums to total",
            "code", "reputation", "domains", "scoring",
        ),
        challenge(
            "Implement quote verification: check whether a quoted passage "
            "appears verbatim (or near-verbatim) in the cited source",
            2,
            "test passes: 40 quotes, catches all 5 fabricated/misattributed "
            "ones, fuzzy match tolerance documented",
            "code", "quotes", "verification", "text-match",
        ),
        challenge(
            "Build a source-diversity auditor: given a research brief's "
            "citations, report domain concentration and single-source "
            "dependency",
            2,
            "audit flags briefs with > 50% citations from one domain, report "
            "lists concentration index",
            "code", "diversity", "citations", "audit",
        ),
        # ---- deep (d=3) ----
        challenge(
            "Implement a provenance chain builder tracing a fact through 3+ "
            "hops of citations back to the primary source, flagging broken "
            "links",
            3,
            "demo: 10 facts traced, primary source found for 8, broken-hop "
            "report lists the 2 failures",
            "build", "provenance", "chains", "tracing",
        ),
        challenge(
            "Build a misinformation-resilience test: inject 10 false claims "
            "into a corpus, measure whether the pipeline surfaces or "
            "amplifies them",
            3,
            "test passes: 9/10 injected claims flagged or quarantined, "
            "amplification score computed",
            "code", "misinformation", "red-team", "testing",
        ),
        challenge(
            "Design a trust-scoring rubric for AI-generated research briefs: "
            "weighted criteria, rater calibration, inter-rater reliability",
            3,
            "report: rubric applied to 15 briefs by 3 raters, ICC >= 0.7, "
            "criterion weights justified",
            "research", "rubric", "calibration", "reliability",
        ),
        challenge(
            "Implement a real-time source monitor watching 20 sources for "
            "retractions/corrections/updates and alerting on changes",
            3,
            "test passes: monitor detects 5/5 simulated retractions within "
            "one poll cycle, alerts logged",
            "code", "monitoring", "retractions", "alerts",
        ),
    ],
    "distillation": [
        # ---- easy (d=1) ----
        challenge(
            "Implement a teacher-logits collector: run a teacher model over "
            "500 prompts, save logits/probs to disk in a replayable format",
            1,
            "test passes: 500 prompts logged, reload reproduces tensors "
            "exactly, format documented",
            "code", "teacher", "logits", "data",
        ),
        challenge(
            "Build a distillation data filter dropping teacher outputs that "
            "are too short, repetitive, or low-confidence",
            1,
            "test passes: filter removes >= 90% of injected junk, keeps >= "
            "95% of good samples",
            "code", "filtering", "quality", "data",
        ),
        challenge(
            "Implement a KL-divergence training loss between student and "
            "teacher distributions with temperature scaling",
            1,
            "pytest passes: loss decreases on toy data, temperature=1 matches "
            "cross-entropy baseline",
            "code", "kl-loss", "temperature", "training",
        ),
        # ---- medium (d=2) ----
        challenge(
            "Build a teacher-student training loop distilling a small model on "
            "10k teacher-labeled examples, tracking eval loss",
            2,
            "loop completes on 10k examples, student eval loss within 15% of "
            "teacher's on held-out set",
            "code", "training", "pipeline", "student",
        ),
        challenge(
            "Implement an eval parity harness: same 200-prompt eval suite for "
            "teacher and student, report per-task deltas",
            2,
            "harness: teacher and student scored on 200 prompts, delta table "
            "per task, reproducible across runs",
            "code", "eval", "parity", "comparison",
        ),
        challenge(
            "Build a data-mixture tuner for distillation sweeping teacher-data "
            "vs ground-truth ratios to find the best blend",
            2,
            "sweep over 5 ratios: best blend beats pure-teacher by >= 3pp on "
            "eval, curve plotted",
            "code", "mixture", "tuning", "data",
        ),
        challenge(
            "Implement sequence-level distillation (SeqKD): generate teacher "
            "outputs with beam search, train student on them",
            2,
            "test passes: SeqKD student beats word-level KD by >= 2pp on "
            "500-example eval",
            "code", "seqkd", "beam-search", "training",
        ),
        challenge(
            "Build a small-model recipe card: architecture, data mix, training "
            "hyperparams, eval results - one reproducible config",
            2,
            "recipe card trains a student from scratch in < 4 GPU-hours, eval "
            "within 5% of claimed scores",
            "build", "recipe", "config", "reproducibility",
        ),
        # ---- deep (d=3) ----
        challenge(
            "Implement on-policy distillation: student generates, teacher "
            "corrects/reranks, iterate - measure compounding-error reduction",
            3,
            "test passes: on-policy student cuts compounding error >= 30% vs "
            "offline KD on long generations",
            "code", "on-policy", "dagger", "training",
        ),
        challenge(
            "Build a distillation scaling study: distill at 3 student sizes x "
            "3 data budgets, fit scaling curves",
            3,
            "report: 9 runs completed, scaling law fit R^2 >= 0.9, "
            "predicted-vs-actual within 10%",
            "research", "scaling", "study", "curves",
        ),
        challenge(
            "Implement task-specific distillation for structured output "
            "(JSON): teacher generates, schema-validator filters, student "
            "trained",
            3,
            "test passes: student emits schema-valid JSON >= 98% on 300 "
            "prompts, field accuracy within 3pp of teacher",
            "code", "structured", "json", "filtering",
        ),
        challenge(
            "Design a distillation data-contamination audit checking student "
            "eval gains are not from train/test overlap with teacher data",
            3,
            "report: n-gram overlap analysis on 10k eval items, contaminated "
            "items quarantined, clean-eval delta reported",
            "research", "contamination", "audit", "leakage",
        ),
    ],
}

register_topic_pack("chall_product", _PACK)
