# NoMorals — Module Map

Every production module and what it does. The system as one organism
(wave 87 Agent OS layering):

```
Core Mind (agents/coremind.py)        the mind — reads natural-language
                                      goals, routes to organs, keeps continuity
  ├── Orchestrator (agents/orchestrator.py)  the nervous system
  ├── Reasoning (agents/reasoning.py)        the visible thought
  ├── Agents = organs/specialists            (below)
  ├── Tools (tools/) = hands
  └── Memory / missions / continuity         long-term state

Games (games/) run under the Social Operator (agents/partner_runtime.py):
the GameEngine owns rules, turns, scoring and match state; the Core Mind
only decides when to route into it.

Chat commands (social/chat/control.py) are the manual overrides — 97 of
them, all in /list and /help.
```

Line counts are production code only (tests excluded).


## core  

- **barcode.py** (1,196 L) — Barcode decode + recovery — Code 128, EAN-13, UPC-A (wave 78).
- **cipher.py** (463 L) — Real cryptography, hermetic.
- **clock.py** (182 L) — Injectable time.
- **config.py** (1,221 L) — Layered configuration.
- **cookies.py** (533 L) — Cookie analysis & handling — the combined CookieLab (wave 76).
- **corpus.py** (415 L) — Bundled cracking corpus + rockyou-style mutation rules.
- **decoder.py** (1,698 L) — Universal Decoder — attempt to identify and decode almost anything.
- **errors.py** (235 L) — Typed error hierarchy.
- **events.py** (343 L) — Thread-safe event bus.
- **http.py** (461 L) — Minimal HTTP client on ``urllib``.
- **ids.py** (186 L) — ULID-style identifiers.
- **logging_setup.py** (261 L) — Structured logging with secret redaction.
- **midi.py** (371 L) — A real MIDI file writer — no dependencies.
- **observability.py** (349 L) — Metrics and tracing without a dependency on Prometheus/OpenTelemetry.
- **pdf.py** (960 L) — Pure-Python PDF writer — no dependencies, works on a phone.
- **policy.py** (533 L) — Capability policy.
- **ratelimit.py** (336 L) — Rate limiting: token buckets, sliding windows, and per-key registries.
- **result.py** (221 L) — The ``Outcome`` type: explicit, unavoidable error handling.
- **retry.py** (336 L) — Retry with backoff, jitter, and circuit breaking.
- **runtune.py** (293 L) — Profile-aware runtime tuning (wave 86).
- **text.py** (522 L) — Text primitives: tokenization, chunking, normalization, dedup.
- **trust.py** (245 L) — Source trust for web sources (wave 85) — core layer.

## llm  

- **base.py** (273 L) — Provider protocol and shared message types.
- **download.py** (298 L) — Model downloading from Hugging Face.
- **local_server.py** (480 L) — Local GGUF model manager: find, download, and *run* llama.cpp on this machine.
- **registry.py** (452 L) — Model registry and catalog.
- **router.py** (349 L) — Provider routing with fallback and hot-swap.

## llm/providers  

- **hf_serverless.py** (460 L) — Hugging Face provider.
- **llama_cpp.py** (116 L) — Local llama.cpp server provider.
- **mock.py** (258 L) — Deterministic offline provider.
- **ocr.py** (144 L) — OCR provider: real, offline "vision" through Tesseract.
- **openai_compat.py** (192 L) — OpenAI-compatible provider.

## agents  

- **autonomy.py** (398 L) — The autonomy agent: she initiates, on her own clock.
- **base.py** (291 L) — Agent lifecycle and budgets.
- **benchmark.py** (538 L) — Agent benchmark — measures the SYSTEM, not just the model.
- **blackboard.py** (213 L) — Shared scratch space for cooperating agents.
- **brief.py** (261 L) — Mission briefing — the prompt/mission structuring sub-agent (wave 68).
- **cipher.py** (391 L) — CipherAgent — real encryption at the tip of the tool system.
- **coding.py** (424 L) — The coding agent: draft -> run in the sandbox -> read the error -> fix.
- **cognition.py** (945 L) — The autonomous cascade — one heartbeat that keeps the whole intelligence
- **context.py** (474 L) — The dependency container.
- **coremind.py** (1,108 L) — Core Mind — the always-on natural-language layer of the Agent OS (wave 87).
- **decoder.py** (285 L) — DecoderAgent — the Universal Decoder as a living sub-agent.
- **devon.py** (1,704 L) — Devon — the autonomous development & investigation agent.
- **directives.py** (202 L) — Directives: direct instructions to the agent core.
- **evolution.py** (1,808 L) — The framework self-improvement agent (a.k.a. the Evolver) — god tier.
- **failure.py** (443 L) — Failure analysis agent — study failures, extract lessons, prevent repeats.
- **features.py** (86 L) — Feature flags: every major capability of the bot, on/off from a chat.
- **games.py** (463 L) — Games module: social games the companion plays IN the chat.
- **goals.py** (809 L) — Long-term goal system — persistent, multi-step objectives that survive
- **improvement.py** (478 L) — Closed-loop self-improvement — the system improves itself, verified.
- **investigate.py** (258 L) — InvestigateAgent — one entry point for ANY artifact.
- **kg.py** (769 L) — Knowledge graph memory — structured, relational, reason-over-stored-knowledge.
- **mission.py** (393 L) — Mission control (wave 62) — the portfolio view over the goal system.
- **monitor.py** (609 L) — MonitorAgent — watch URLs and files, alert on real change.
- **news.py** (206 L) — News system: a news sub-agent + a summarizer sub-agent.
- **notifier.py** (164 L) — Notifier: durable alerts, delivered to the owner on every live channel.
- **ops_alerts.py** (216 L) — Ops alerts (wave 82 + wave 83 escalation).
- **orchestrator.py** (1,397 L) — Master Orchestrator.
- **osint_graph.py** (929 L) — OSINT god tier: persistent identity correlation + campaign automation.
- **partner_runtime.py** (5,278 L) — The partner runtime: the brain, and the loop that keeps it alive.
- **power.py** (312 L) — Power mode: the owner's documented, audited expansion of capability.
- **projects.py** (1,238 L) — Autonomous Project Mode (wave 50) — complex multi-step projects with
- **reasoning.py** (1,584 L) — Reasoning engine — explicit, auditable, multi-strategy thought.
- **reflection.py** (401 L) — Reflective checkpointing (wave 64) — the loop that looks back.
- **research_swarm.py** (559 L) — Research swarm (wave 86): parallel specialized researchers.
- **researcher.py** (682 L) — Always-on research & suggestion engine.
- **router_select.py** (310 L) — Multi-model task router (wave 50) — route each task to the best model.
- **runtime.py** (497 L) — Hybrid execution runtime: threads + processes + asyncio, one scheduler.
- **scheduler.py** (425 L) — Scheduler: durable cron-style jobs, run in-process.
- **simulation.py** (330 L) — Advanced Simulation / Sandbox (wave 50) — test dangerous, complex, or
- **skills.py** (695 L) — Skill library — proven approaches the system reuses and improves.
- **structuring.py** (354 L) — Prompt / mission structuring sub-agent (wave 76).
- **supervisor.py** (279 L) — Supervisor: watchdog, restart policy, and budget enforcement.
- **swarm.py** (245 L) — Multi-agent swarm: parallel sub-investigators with a synthesis step.
- **task_type.py** (306 L) — Task-type classification (wave 65) — does this text want a BUILD?
- **tasks.py** (384 L) — Task graph: the unit of parallel work.
- **toolmaker.py** (550 L) — Tool creator — design, write, test, and register new tools when the
- **watcher.py** (116 L) — Watch loop (wave 83): the resident operator.

## agents/roles  

- **orchestrator_helpers.py** (33 L) — Shared helpers for role agents.

## agents/arena  

- **core.py** (591 L) — The self-improvement arena core.
- **topics.py** (126 L) — Topic sampling for the self-improvement arena.

## agents/search  

- **curate.py** (88 L) — Result curation: dedupe, junk filtering, and relevance scoring.
- **deep.py** (590 L) — Deep research: the /search deep pipeline.
- **engine.py** (468 L) — Dedicated search engine: search → read → crawl → curate → summarize.
- **summarize.py** (81 L) — Summarization: model-synthesized when a real model is answering,
- **trust.py** (9 L) — Wave 85 source trust — re-exported from the core layer.

## agents/trial  

- **flow.py** (191 L) — Single-account trial flow: research a platform, save ONE account's
- **vault.py** (164 L) — Encrypted credential vault for single-account trials.

## tools  

- **attacker.py** (564 L) — Network credential brute — the owner's attacker.py, in the repo.
- **audio.py** (550 L) — Voice I/O: text-to-speech and speech-to-text.
- **browser.py** (1,117 L) — Browser tool: a stateful web session built on the standard library.
- **compress.py** (124 L) — Compressor: make big files sendable.
- **connectors.py** (413 L) — API connector framework: plug in external services cleanly.
- **database.py** (174 L) — Database tools: structured storage, queried from the agent plane.
- **decoder.py** (335 L) — decoder — the Universal Decoder as a tool.
- **filesend.py** (408 L) — Research file creator + social sender.
- **filesystem.py** (258 L) — Filesystem tools.
- **giftcard.py** (393 L) — giftcard — barcode & card-number recovery (wave 78).
- **hashcrack.py** (962 L) — Hash cracker — offline password-hash verification engine.
- **imagedb.py** (307 L) — Image lookup + reverse image search.
- **macros.py** (355 L) — Automation recorder: record action sequences into reusable tools.
- **media.py** (296 L) — Media downloading.
- **metadata.py** (615 L) — Metadata extraction — offline file forensics, zero dependencies.
- **network.py** (600 L) — Network tools: DNS, port awareness, request crafting, whois, local services.
- **osint.py** (343 L) — OSINT toolkit: read-only intelligence from public sources.
- **osint_people.py** (622 L) — OSINT — people, identities, and correlation (the second OSINT unit).
- **parsers.py** (282 L) — Document parsers, stdlib only.
- **proxy.py** (251 L) — Proxy / tunnel manager: route the bot's own outbound traffic.
- **proxylab.py** (1,527 L) — God-tier proxy lab: scrape, test, rank, store, and serve working proxies.
- **proxysources.py** (315 L) — Proxy source catalog + health (wave 83).
- **registry.py** (412 L) — Tool registry: schema, dispatch, permission gating, and an audit trail.
- **sandbox_code.py** (291 L) — Sandboxed code interpreter: write Python, run it, see the result.
- **scriptgen.py** (407 L) — Script generator: render validated automation scripts into the workspace.
- **shell.py** (256 L) — Sandboxed shell execution.
- **ssh_socks.py** (511 L) — SSH → SOCKS5: turn any SSH server into a local SOCKS5 proxy.
- **traindata.py** (373 L) — Training-data tools: the conversation→training agent's control surface.
- **vision.py** (232 L) — Vision: image understanding.
- **web.py** (301 L) — Web tools: fetch, search, and HTML→text.
- **workspace.py** (107 L) — Workspace tools (wave 84): the main AI's window onto the virtual CPU

## tools/native  

- **loader.py** (157 L) — Build + load the native hash library (nmhash.so), with a pure fallback.

## tools/custom  


## social  

- **base.py** (178 L) — Platform adapters and the account model.
- **manager.py** (473 L) — Post scheduling and cross-platform fan-out.

## social/chat  

- **base.py** (235 L) — The chat adapter protocol: bidirectional messaging across platforms.
- **control.py** (1,105 L) — Control commands: the owner's hands on the machine, from inside a chat.
- **discord.py** (512 L) — Discord adapter — **your own account** (discord.py user client).
- **gateway.py** (406 L) — The chat gateway: every platform, one brain, at the same time.
- **local.py** (82 L) — Local console adapter: the companion, in your terminal.
- **telegram.py** (509 L) — Telegram adapter — userbot style, full control.
- **web.py** (86 L) — Web console adapter: the companion, in your browser.
- **whatsapp.py** (275 L) — WhatsApp adapter — Python client for the Node/Baileys bridge.

## games  

- **ai.py** (247 L) — GameMind: the AI's seat at every table.
- **economy.py** (154 L) — The game economy: coins, a persistent shop, and item ownership.
- **engine.py** (585 L) — The game engine: rooms, turns, timers, persistence, rewards.
- **players.py** (298 L) — Player identity, persistent profiles, and the leaderboard.

## games/games  

- **ambitious.py** (676 L) — The ambitious table: long arcs, persistent state, and stakes that
- **base.py** (233 L) — The multiplayer game contract.
- **easy.py** (1,100 L) — Easy / high-engagement games: quick to start, quick to finish,
- **medium.py** (1,422 L) — Medium-complexity games: state machines with real phases, roles, and

## memory  

- **base.py** (200 L) — Memory records and relevance scoring.
- **embeddings.py** (203 L) — Embedding with graceful degradation.
- **extract.py** (349 L) — Memory extraction: mining durable knowledge out of conversation.
- **manager.py** (505 L) — Memory manager: the unified façade.

## missions  

- **mission.py** (442 L) — Mission state and persistence.
- **runner.py** (881 L) — Mission execution: plan, run, checkpoint, resume, reflect.

## workspace  

- **profile.py** (199 L) — Environment profiles (wave 84): what kind of machine are we on, and
- **vcpu.py** (326 L) — Virtual CPUs (wave 84): independent execution engines.
- **workspace.py** (406 L) — The Workspace (wave 84): the machine's virtual CPU farm.

## training  

- **checkpoints.py** (259 L) — Checkpoint validation + selection for the free-tier training runs.
- **collect.py** (751 L) — Harvest safe training examples from the durable runtime stores.
- **dataset.py** (421 L) — Dataset codecs and the dataset registry.
- **evaluate.py** (253 L) — Evaluation for trained artifacts — one contract, every backend.
- **finetune.py** (1,721 L) — Persona fine-tune mix builder (wave 69).
- **free_datasets.py** (813 L) — Best free training datasets + one-command download/preparation.
- **policy.py** (256 L) — Offline policy for deciding whether the training loop should run.
- **preprocess.py** (169 L) — Corpus cleaning, deduplication, and splitting.
- **registry.py** (269 L) — Training run tracking, and the gate between a trained model and production.
- **tokenize.py** (288 L) — Byte-pair encoding, trained from scratch with the standard library.
- **trainer.py** (453 L) — A native pure-Python training loop.

## training/backends  

- **base.py** (64 L) — The common interface every training backend implements.
- **llama_factory.py** (277 L) — The LLaMA-Factory backend: run a finetune through the LLaMA-Factory CLI.
- **native.py** (78 L) — The native backend: pure-Python next-token MLP, zero dependencies.
- **unsloth.py** (377 L) — The Unsloth backend: QLoRA finetunes of 7B–8B models on a single GPU.

## voice  

- **tts.py** (545 L) — Universal TTS: swappable free/open-source neural backends, one interface.

## media  

- **music.py** (632 L) — MusicCreator — turn a topic into a real, performable song.
- **playback.py** (578 L) — PlaybackEngine — real music playback with full transport control.
- **video.py** (282 L) — VideoFinder — locate videos across the open web, with real metadata.

## partner  

- **background.py** (258 L) — Contextual background knowledge: her life in a Colorado mountain town.
- **context.py** (208 L) — Context assembly: what the model sees when she replies.
- **gating.py** (109 L) — Ownership-aware chat gating: who gets the full version of her.
- **mood.py** (539 L) — The mood engine: a persistent, event-driven emotional state.
- **persona.py** (294 L) — The partner's identity: who she is, how she talks, where her moods start.
- **presence.py** (190 L) — Human presence: typing pace and the busy/distracted gaps.
- **relationship.py** (227 L) — Relationship state: stages, milestones, fights, and what she knows about him.
- **responder.py** (467 L) — The reply pipeline: signals in, a real message out.
- **style.py** (419 L) — Texting-style enforcement: the hard layer under the model's soft layer.

## api  

- **console.py** (533 L) — Web console — the companion in your browser, key-gated.
- **server.py** (256 L) — HTTP API — stdlib only, so the service runs anywhere Python does.

## books  

- **forge.py** (335 L) — BookForge: the book pipeline.
- **model.py** (197 L) — Book model: chapters, status, and on-disk persistence.
- **outline.py** (207 L) — Outline generation: model-first, template fallback.
- **tools.py** (169 L) — BookForge registry tools — callable by the main AI and every sub-agent.
- **write.py** (248 L) — Chapter writing: model-first, template-composer floor.

## storage  

- **backup.py** (401 L) — Versioned backups with optional push to a Git repository.
- **blob.py** (394 L) — Content-addressed blob store.
- **db.py** (503 L) — SQLite access layer.
- **fts.py** (182 L) — FTS5 lexical search.
- **migrations.py** (1,252 L) — The schema, expressed as ordered migrations.
- **models.py** (296 L) — Typed dataclass mirrors of the core tables.
- **queue.py** (325 L) — Durable work queue in SQLite.
- **repository.py** (304 L) — Generic repository layer.
- **schema.py** (189 L) — Schema migration engine.
- **vectors.py** (387 L) — Vector similarity search on SQLite.

## .  

- **__main__.py** (15 L) — Console entry point: ``python -m nomorals``.
- **archives.py** (624 L) — Archive system — create and crack open any common archive.
- **builders.py** (1,331 L) — Builder system — scaffold real, runnable applications.
- **builders_proxy.py** (323 L) — A small, real HTTP reverse proxy used by ``AppBuilder.deploy``.
- **cli.py** (5,987 L) — Command-line interface: ``nm`` / ``python -m nomorals``.
- **compat.py** (196 L) — Optional-dependency detection.
- **execbox.py** (693 L) — Execution system — run code in many languages, safely, with real output.
- **exporter.py** (300 L) — Account-history export into the training pipeline.
- **self_improvement.py** (444 L) — Crash-resumable automatic self-improvement pipeline.
- **version.py** (28 L) — Version metadata.

## social/adapters  

- **bluesky.py** (136 L) — Bluesky adapter — the AT Protocol HTTP API.
- **mastodon.py** (140 L) — Mastodon adapter — the official REST API, no scraping.

## tui  

- **app.py** (189 L) — The curses driver. Deliberately thin.
- **model.py** (291 L) — TUI state and layout — pure, with no curses import.
---

## Where things run

| Surface | Entry |
|---|---|
| Phone / VPS / PC console | `python -m nomorals <command>` (`nm`) |
| Chat (Telegram / Discord / WhatsApp / local) | `nm chat` → `PartnerRuntime` |
| Games | the Social Operator → `GameEngine` (19 games, DM + group + channel) |
| Local model on this machine | `llm/local_server.py` (llama.cpp GGUF manager, phone-aware) |
| Training | `training/` (Kaggle/Colab notebooks, native or unsloth backend) |
