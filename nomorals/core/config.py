"""Layered configuration.

Precedence, lowest to highest:

1. Built-in defaults (``DEFAULTS`` below)
2. Profile preset (``workstation`` / ``laptop`` / ``termux``)
3. TOML config file (``$NM_HOME/config.toml`` or ``--config``)
4. ``.env`` file in the working directory
5. Environment variables (``NM_<SECTION>_<KEY>``)
6. Explicit overrides passed to :func:`Settings.load`

The result is an immutable-ish, typed dataclass tree. Reads are attribute access;
nothing reaches for ``os.environ`` after construction.
"""

from __future__ import annotations

import os
import threading
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar

from .errors import ConfigError

__all__ = [
    "Settings",
    "get_settings",
    "load_settings",
    "reset_settings",
    "set_settings",
    "env_var_path",
    "config_env_status",
    "dotted_path_exists",
    "is_direct_env_key",
    "direct_env_consumer",
    "diff_settings",
    "config_schema",
    "ConfigWatcher",
    "watch_settings",
    "start_global_watcher",
    "redacted_dict",
    "validate_settings",
]

T = TypeVar("T")


# ── Defaults ───────────────────────────────────────────────────────────────────


@dataclass
class LogSettings:
    level: str = "INFO"
    file: str = "logs/nomorals.log"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    json: bool = False
    redact: bool = True


@dataclass
class StorageSettings:
    path: str = "data/nomorals.db"
    wal: bool = True
    busy_timeout_ms: int = 5000
    synchronous: str = "NORMAL"
    blob_dir: str = "data/blobs"
    fts: bool = True


@dataclass
class BackupSettings:
    dir: str = "backups"
    keep: int = 14
    compress: bool = True
    git_repo: str = ""
    git_branch: str = "backups"
    git_author: str = "NoMorals Core <nm@localhost>"
    interval_seconds: int = 6 * 3600
    include_blobs: bool = True


@dataclass
class ConcurrencySettings:
    threads: int = 8
    processes: int = 0  # 0 -> default_process_workers(): NM_PROCESS_WORKERS env, else max(2, cpu_count - 1); termux preset forces 0 (moot: pools off there)
    use_processes: bool = True
    max_subagents: int = 64
    network_permits: int = 16
    disk_permits: int = 8
    gpu_permits: int = 1
    async_loop_threads: int = 2


@dataclass
class BudgetSettings:
    wall_seconds: float = 3600.0
    tokens: int = 2_000_000
    children: int = 256
    cost_usd: float = 0.0  # 0 = unlimited
    retries: int = 3


@dataclass
class AdaptiveBudgetSettings:
    """Cost-adaptive token budgeting (nomorals.llm.adaptive_budget).

    Daily + per-operation token/cost budgets with threshold-driven
    adaptation (downgrade tier / truncate context / defer non-urgent
    work) and an honest budget-exceeded block.  0 budgets == unlimited
    (pure allow).  Env: NM_ADAPTIVE_BUDGET_TOKENS,
    NM_ADAPTIVE_BUDGET_COST_USD, NM_ADAPTIVE_BUDGET_DOWNGRADE_AT,
    NM_ADAPTIVE_BUDGET_TRUNCATE_AT, NM_ADAPTIVE_BUDGET_DEFER_AT,
    NM_ADAPTIVE_BUDGET_MIN_CONTEXT_TOKENS.
    """
    daily_token_budget: int = 0
    daily_cost_usd: float = 0.0
    per_operation: dict[str, dict[str, float]] = field(default_factory=dict)
    warn_threshold: float = 0.5
    downgrade_threshold: float = 0.5
    truncate_threshold: float = 0.8
    defer_threshold: float = 0.8
    critical_truncate_threshold: float = 0.95
    full_context_tokens: int = 8192
    min_context_tokens: int = 1024
    cheap_price_cap_per_1m: float = 1.0
    background_operations: tuple[str, ...] = (
        "research", "background", "summarize", "judge", "index",
    )


@dataclass
class ProviderSettings:
    name: str = "mock"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: float = 120.0
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 4096
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMSettings:
    provider: str = "mock"
    fallback_chain: list[str] = field(default_factory=lambda: ["mock"])
    active_model: str = ""
    hf_token: str = ""
    hf_base_url: str = "https://router.huggingface.co/hf-inference"
    hf_endpoint_url: str = ""
    hf_model: str = "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2"
    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = ""
    openai_model: str = "dolphin-2.9-llama3-8b"
    llama_cpp_url: str = "http://localhost:8080"
    timeout: float = 120.0
    cache_dir: str = "models"
    system_prompt: str = ""
    local_model: str = ""  # path or name of a downloaded GGUF; drives the
                          # power-mode router cascade when present
    local_auto_start: bool = False  # NM_LLM_LOCAL_AUTO_START=1: boot starts
                                    # llama-server on local_model automatically
    local_host: str = "127.0.0.1"  # NM_LLM_LOCAL_HOST
    local_port: int = 8080  # NM_LLM_LOCAL_PORT; must match llama_cpp_url
    verify_at_startup: bool = True  # NM_LLM_VERIFY_AT_STARTUP=0: skip provider
                                   # probe at boot (faster startup, but bad
                                   # model names surface at chat time instead)
    #: Groq's model IDs churn (they retired llama-3.3-70b-versatile & co. on
    #: 2026-08-16) — the default must be a live ID, and NM_GROQ_MODEL exists
    #: so a drift event is a .env edit, not a re-release.
    groq_model: str = "openai/gpt-oss-120b"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    #: OpenRouter fallback. Model IDs (esp. :free ones) churn — set
    #: NM_OPENROUTER_MODEL explicitly, e.g. "qwen/qwen3-8b:free".
    openrouter_api_key: str = ""
    openrouter_model: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    local_lora: str = ""

    #: comma-separated GGUF LoRA adapter(s) the llama.cpp server stacks on
    #: the promoted base model (NM_LLM_LOCAL_LORA; written by
    #: nm models --promote-local <base> --lora <file>)
    #: opt-in ONLY: when no real provider configures, boot with the scripted
    #: mock instead of booting model-less. Default off — a silent mock pretending
    #: to answer is worse than an honest "no model configured".
    allow_mock_fallback: bool = False


@dataclass
class EmbeddingSettings:
    provider: str = "auto"
    model: str = ""
    dimensions: int = 512
    batch_size: int = 32
    cache: bool = True


@dataclass
class MemorySettings:
    recall_limit: int = 12
    context_budget_tokens: int = 6000
    decay_half_life_hours: float = 72.0
    forget_threshold: float = 0.02
    consolidation_interval_seconds: float = 3600.0
    consolidate_on_pressure: int = 5000
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "recency": 0.25,
            "importance": 0.30,
            "semantic": 0.30,
            "lexical": 0.15,
        }
    )
    #: automatic mining of durable facts from every turn (never the reply path)
    extract_enabled: bool = True
    #: opt-in LLM pass on top of the heuristics — one cheap completion,
    #: fails closed to the heuristic result on garbage
    extract_llm: bool = False
    #: fuzzy similarity above which a candidate counts as a duplicate
    extract_dedupe_ratio: float = 0.8
    #: two-tier memory (Khoj pattern, build-map #36): parallel opt-in layer
    #: alongside MemoryManager — events + distilled facts. OFF by default:
    #: the existing recall is trusted and stays exactly as it is.
    two_tier_enabled: bool = False


@dataclass
class TrainingSettings:
    base_model: str = ""
    backend: str = "native"  # native | unsloth | llama_factory
    data_dir: str = "data/training"
    output_dir: str = "artifacts"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    max_seq_len: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_accumulation: int = 4
    eval_split: float = 0.05
    dedup: bool = True
    promote_on_pass: bool = True
    regression_tolerance: float = 0.02


@dataclass
class ToolSettings:
    sandbox: bool = True
    sandbox_backend: str = "auto"  # auto | bwrap | unshare | rlimit | none
    shell_timeout: float = 120.0
    shell_max_output: int = 1_000_000
    allow_network: bool = True
    workspace: str = "workspace"
    max_upload_mb: int = 100
    user_agent: str = "NoMoralsCore/0.1 (+https://github.com/Oluwacutyp/No-morals-ai)"
    robots_txt: bool = True
    http_timeout: float = 30.0
    max_concurrent_downloads: int = 4


@dataclass
class SocialSettings:
    enabled: bool = True
    dry_run: bool = False
    rate_limit_per_hour: int = 20
    min_interval_seconds: float = 30.0
    audit: bool = True
    mastodon_domain: str = ""
    mastodon_token: str = ""
    bluesky_handle: str = ""
    bluesky_app_password: str = ""
    telegram_token: str = ""
    discord_webhook: str = ""
    x_api_key: str = ""
    x_api_secret: str = ""
    x_access_token: str = ""
    x_access_secret: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "nomorals-core/0.1"


@dataclass
class ChatSettings:
    local_enabled: bool = False
    max_history: int = 100
    media_in_groups: bool = False
    media_max_mb: float = 20.0
    telegram_enabled: bool = True
    telegram_session: str = ""
    telegram_api_id: str = ""
    telegram_api_hash: str = ""
    telegram_chats: str = ""
    threads_enabled: bool = True
    telegram_bot_enabled: bool = True
    telegram_bot_token: str = ""
    telegram_bot_chats: str = ""
    #: Partner (romantic) context on the gated bot. Default False:
    #: groups on the gated bot never carry partner context; an owner DM
    #: on it still does (brain rule — see brain._generate_and_persist).
    telegram_bot_partner_context: bool = False
    #: ── second BotFather bot: the personal bot ──────────────────────
    #: Owner-only, ungated (every chat on it is an owner chat), always
    #: carries partner context. NOT public: non-owners get one honest
    #: denial and the message never reaches the brain (no credit spend).
    #: Token per-adapter; empty/disabled = the bot simply doesn't run.
    telegram_personal_bot_enabled: bool = True
    telegram_personal_bot_token: str = ""
    #: Allowlist for the personal bot. The owner_only gate (owner sender
    #: IDs / owner_chats) governs regardless — this is an extra filter.
    telegram_personal_bot_chats: str = ""
    telegram_personal_bot_partner_context: bool = True
    discord_enabled: bool = True
    discord_token: str = ""
    discord_channels: str = ""
    discord_greet_new: bool = True
    whatsapp_enabled: bool = True
    whatsapp_host: str = "127.0.0.1"
    whatsapp_port: int = 8787
    # Persistent outbox for bridge outages — sends queued here flush on
    # reconnect instead of dying (whatsapp.py Outbox).
    whatsapp_outbox_dir: str = "data/chat/outbox"
    webhook_enabled: bool = True
    webhook_host: str = "127.0.0.1"
    webhook_port: int = 0
    webhook_token: str = ""
    webhook_reply_url: str = ""
    # SMS fallback surface (build-map #19): text a number, get Devon.
    # COST: Twilio numbers cost real money (~$1/mo + per-segment each way).
    # ON by default — but a missing Twilio number means the adapter fails
    # honestly ("needs sms_from_number") instead of silently not running;
    # NM_CHAT_SMS_ENABLED=0 turns it off entirely.
    sms_enabled: bool = True
    sms_from_number: str = ""
    sms_host: str = "127.0.0.1"
    sms_port: int = 0
    # Twilio auth token — enables X-Twilio-Signature verification on the
    # inbound webhook (forged/unsigned POSTs get 403). Empty = accepted
    # with a warning; don't expose the webhook without it.
    sms_auth_token: str = ""
    # The public URL Twilio is configured to POST to (needed for signature
    # verification behind a tunnel/reverse proxy).
    sms_public_url: str = ""
    max_per_hour: int = 60


@dataclass
class AutonomySettings:
    enabled: bool = True
    interval_hours: float = 6.0
    adaptive_cadence: bool = True
    daily_model_calls: int = 0  # 0 = unlimited; the loop meters regardless
    max_project_heals: int = 3
    tick_goals: bool = True
    replan_after_heals: int = 2
    tick_improvement: bool = True
    tick_projects: bool = True
    tick_train: bool = True
    reasoning_mode: str = "off"
    reflect_on_completion: bool = True
    
    def __post_init__(self):
        """Validate the autonomy dial before anything consumes it."""
        from nomorals.core.errors import ConfigError

        if self.interval_hours < 1.0:
            raise ConfigError(
                f"autonomy.interval_hours must be >= 1.0, got {self.interval_hours}")
        if self.max_project_heals < 0:
            raise ConfigError(
                f"autonomy.max_project_heals must be >= 0, got {self.max_project_heals}")
        if self.replan_after_heals < 0:
            raise ConfigError(
                f"autonomy.replan_after_heals must be >= 0, got {self.replan_after_heals}")
        if self.daily_model_calls < 0:
            raise ConfigError(
                "autonomy.daily_model_calls must be >= 0 "
                f"(0 = unlimited), got {self.daily_model_calls}")


@dataclass
class EvolutionSettings:
    benchmark: str = ""
    enabled: bool = True


@dataclass
class VisionSettings:
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    ocr_binary: str = ""
    ocr_language: str = "eng"
    enabled: bool = True
    # Prompt 09: vision tool knobs. Screenshots stay a privileged, explicit,
    # per-call-confirmed action — never scheduled, never background.
    allow_screenshot: bool = False
    max_dimension: int = 1568  # downscale images over this (px) before sending
    max_image_bytes: int = 25 * 1024 * 1024  # reject inputs over this
    log_calls: bool = True  # per-call audit log (timestamp/source/bytes, never pixels)


@dataclass
class ArenaSettings:
    enabled: bool = True
    build: bool = True
    research_pages: int = 3
    interval_hours: int = 24
    #: Topic anti-repeat window: no topic repeats within the last N
    #: served. Env: NM_ARENA_ANTI_REPEAT_WINDOW. A per-DB kv override
    #: (arena.anti_repeat_window) wins when set.
    anti_repeat_window: int = 10


@dataclass
class PartnerSettings:
    enabled: bool = True
    autonomy_mode: str = "auto"
    typing_cap_seconds: float = 0.0
    #: Which chat adapters to attempt at boot (comma-separated). ON by
    #: default for every surface: adapters with missing credentials fail
    #: honestly into the "chat adapters skipped" list instead of silently
    #: not running. NM_PARTNER_PLATFORMS=... overrides (explicit list wins).
    platforms: str = "local,telegram,telegram-bot,telegram-personal,discord,whatsapp,webhook,sms"
    #: Power mode default.  When True (the default), power mode auto-activates
    #: on first use without requiring the owner key — the owner's standing
    #: choice is unlimited/unrestricted operation.  An explicit `nm power
    #: lock` persists a locked state that survives reboots until unlocked
    #: again.  Env: NM_PARTNER_POWER_DEFAULT_ON=0 to require manual unlock.
    power_default_on: bool = True
    owner_chats: str = ""
    us_chats: str = ""
    personality: str = "helpful"
    memory_enabled: bool = True
    proactive: bool = True
    #: Master switch for proactive *push* sends (morning briefing delivery
    #: + watcher alerts).  Default ON.  Env: NM_PARTNER_PROACTIVE_ENABLED=0
    #: to make her speak only when spoken to.
    proactive_enabled: bool = True
    #: Per-kind toggles, all default ON.  Envs: NM_PARTNER_PROACTIVE_BRIEFING,
    #: NM_PARTNER_PROACTIVE_WATCHERS (0/1).
    proactive_briefing: bool = True
    proactive_watchers: bool = True
    persona_name: str = ""
    disclosure: str = ""
    background_gate: str = "us_or_romantic"
    #: Sender IDs (WhatsApp JIDs, Telegram numeric IDs) that belong to the
    #: owner. Used for message-level owner recognition in groups — the
    #: master gets recognized even when the chat itself isn't an owner
    #: chat. Comma-separated. Env: NM_PARTNER_OWNER_SENDER_IDS.
    owner_sender_ids: str = ""
    #: Owner key for `/power on <key>` (power-mode unlock). Read from
    #: NM_PARTNER_OWNER_KEY. Empty = no env-key unlock configured (the
    #: owner seal is the other unlock path).
    owner_key: str = ""
    #: Every owner conversation is a fine-tune pair waiting to happen; the
    #: runtime appends them to data/training/conversations.jsonl by default
    #: (NM_PARTNER_TRAIN_COLLECT=0 turns the recorder off).
    train_collect: bool = True
    gate_restricted_chats: bool = True
    history_window: int = 50
    reasoning: str = "auto"
    max_parallel_chats: int = 5
    max_proactive_dm_per_day: int = 10
    max_group_posts_per_day: int = 5
    #: comma-separated group chat keys the autonomy agent may post in
    group_chats: str = ""
    #: quiet hours for proactive sends (hour of day, 24h clock)
    quiet_start: int = 22
    quiet_end: int = 8
    typing_while_thinking: bool = True
    #: typing indicators are on for every chat kind — in groups they make
    #: the reply feel human-paced instead of telegraphed; the per-part
    #: typing run scales with chunk length
    typing_in_groups: bool = True
    typing_seconds: float = 1.0
    #: "typing…" refresh cadence — platforms expire the indicator (~5s on
    #: Telegram), so the keepalive tick must stay under that
    typing_keepalive_seconds: float = 4.0
    #: how long to keep holding "typing…" before giving up on the reply
    typing_keepalive_budget: float = 120.0
    #: after this much silence mid-reply, send one "still thinking" line
    slow_reply_notice_seconds: float = 45.0
    part_delay_seconds: float = 0.5
    #: Dynamic lexicon voice feed (nomorals/partner/lexicon_feed.py): blend
    #: scored lexicon terms into her phrasing — catchphrases, pet names,
    #: openers, transitions, acknowledgments, mood expressions, fallback
    #: lines — on top of the owner's own persona/style settings, which
    #: always win. Env: NM_PARTNER_LEXICON_VOICE=0 disables all dynamic
    #: lexical influence (pure static banks).
    lexicon_voice: bool = True
    #: Seed starter vocab for the partner lexicon on first boot, through
    #: the normal acquire -> score -> version pipeline (idempotent; never
    #: re-adds owner-retired terms). Env: NM_PARTNER_LEXICON_SEED=0 to skip.
    lexicon_seed: bool = True


@dataclass
class MissionSettings:
    checkpoint_dir: str = "data/missions"
    auto_reflect: bool = True
    max_iterations: int = 64
    tick_seconds: float = 5.0
    persist_checkpoints: bool = True


@dataclass
class APISettings:
    github_token: str = ""
    weather_latitude: float = 6.5244  # Enugu, the owner's coordinates
    weather_longitude: float = 7.5186
    request_timeout: float = 10.0
    host: str = "0.0.0.0"
    port: int = 8731
    token: str = ""
    cors_origins: str = "*"
    max_body_mb: int = 64
    workers: int = 8



@dataclass
class ImprovementSettings:
    enabled: bool = True
    # "approval" by default: the loop benchmarks, proposes, and WAITS for the
    # owner — it never auto-applies code changes. "autonomous" applies
    # through the test + benchmark-regression gate; "off" disables the loop.
    mode: str = "approval"
    target: float = 0.8
    benchmark: str = ""
    auto_tick: bool = True
    # |delta| above this counts as a real improvement/regression
    # (run_cycle autonomous path + apply_approval). Matches the training
    # regression_tolerance convention.
    tolerance: float = 0.02
    # Max cycles per tick() call (stops early on meets_target).
    max_cycles: int = 1

@dataclass
class TradingSettings:
    """FinancialExpert / trading gates. Live trading is inert by default."""
    # NM_TRADING_ENABLED master switch: tools/trading refuses every
    # trading action (except doctor/kill) when this is false.
    enabled: bool = True
    # NM_TRADING_MODE: default desk mode for exness actions when the
    # caller does not pass one ("paper" unless the owner says "live").
    mode: str = "paper"
    live_enabled: bool = False
    max_daily_loss_pct: float = 3.0
    min_sharpe: float = 1.0
    max_drawdown_pct: float = 20.0
    min_profit_factor: float = 1.3
    unlock_ttl_hours: float = 24.0
    followed_symbols: list = field(default_factory=list)

@dataclass
class OsintSettings:
    enabled: bool = True
    sources: list = field(default_factory=list)
    hibp_key: str = ""
    abuseipdb_key: str = ""
    shodan_key: str = ""
    request_timeout: float = 10.0
    crtsh_days: int = 90


@dataclass
class MindSettings:
    #: Max background jobs the Core Mind may have in flight at once
    #: (``_send_async``). A new send that cannot acquire a slot within
    #: ``inflight_acquire_timeout_s`` is shed: logged, counted, and the
    #: job is marked failed with an explicit note — never silently
    #: queued, never blocking the chat thread. Env: NM_MIND_MAX_INFLIGHT.
    max_inflight: int = 8
    #: Seconds a new send waits for an in-flight slot before it is shed.
    #: Env: NM_MIND_INFLIGHT_TIMEOUT.
    inflight_acquire_timeout_s: float = 30.0

@dataclass
class AudioSettings:
    enabled: bool = True
    model: str = ""
    tts_engine: str = "auto"
    tts_voice: str = ""
    stt_provider: str = "auto"
    stt_base_url: str = ""
    stt_model: str = ""
    stt_api_key: str = ""
    audio_dir: str = "audio"
    # live voice session (Prompt 10): silence that ends an utterance,
    # per-device data dir, raw-audio retention (off by default), and the
    # spoken-summary length cap. Env: NM_AUDIO_VOICE_SILENCE_MS etc.
    voice_silence_ms: int = 800
    voice_keep_audio: bool = False
    voice_speech_cap_secs: int = 60
    voice_data_dir: str = "voice_data"

@dataclass
class SchedulerSettings:
    enabled: bool = True
    interval_hours: float = 6.0
    tick_seconds: float = 60.0
    max_concurrent: int = 2
    wall_seconds: float = 300.0

@dataclass
class NewsSettings:
    enabled: bool = True
    sources: list = field(default_factory=list)

@dataclass
class NetSettings:
    enabled: bool = True
    proxy: str = ""
    probe_timeout: float = 10.0
    allowed_targets: str = ""
    default_ports: str = ""
    max_probe_ports: int = 100
    banner: bool = False

@dataclass
class ProxySettings:
    active: str = ""
    known: str = ""
    url: str = ""

@dataclass
class RuntimeSettings:
    enabled: bool = True
    profile: str = ""
    # 0 = auto: build_tune() derives threads from CPU/RAM. Any nonzero value
    # is treated as an explicit override and wins over auto-tuning, so the
    # default must stay 0 — a nonzero default would silently disable
    # auto-tuning for every deployment that never sets this knob.
    threads: int = 0

@dataclass
class TakeoverSettings:
    """Human-in-the-loop browser takeover (see browser/takeover.py).

    ``base_url`` is the PUBLIC prefix for takeover links — a tunnel URL
    or ``http://host:PORT``. Empty = fall back to the bind address
    (host-network-only). No hostnames/IPs/ports are hardcoded: every
    value here is operator-configured.
    Env: NM_TAKEOVER_ENABLED, NM_TAKEOVER_BASE_URL,
    NM_TAKEOVER_BIND_HOST, NM_TAKEOVER_PORT, NM_TAKEOVER_LINK_TTL_S,
    NM_TAKEOVER_WAIT_TIMEOUT_S, NM_TAKEOVER_POLL_S.
    """
    enabled: bool = True
    base_url: str = ""
    bind_host: str = "127.0.0.1"
    port: int = 0
    link_ttl_s: int = 900
    wait_timeout_s: int = 900
    poll_s: float = 2.0


@dataclass
class HubSettings:
    """Remote device hub (mesh + sync over HTTP).

    ``url`` points a device at its hub, e.g. ``http://192.168.1.5:8861``
    (empty = no hub; mesh/sync stay local). ``token`` is the shared bearer
    secret the hub was started with. ``bind_host``/``port`` are the hub
    server's own listen address when THIS machine runs the hub
    (``nm mesh serve``); a non-loopback bind without a token is refused.
    Env: NM_HUB_URL, NM_HUB_TOKEN, NM_HUB_HOST, NM_HUB_PORT, NM_HUB_TIMEOUT.
    """
    url: str = ""
    token: str = ""
    bind_host: str = "127.0.0.1"
    port: int = 8861
    request_timeout: float = 15.0

@dataclass
class LocationSettings:
    """Where the owner is — weather places + locale.

    Consumed by ``agency.adapters.places_from_settings`` (which reads
    ``settings.location.places`` / ``home_place`` / ``work_place``).
    Env: NM_PLACES (comma-separated), NM_HOME_PLACE, NM_WORK_PLACE,
    NM_LOCALE. These keys were documented in ``.env.example`` and the
    EC2 env template for a long time but mapped to nothing — a user's
    setting was silently ignored and the weather places derivation
    always came up empty on a real Settings tree.
    """
    places: list = field(default_factory=list)
    home_place: str = ""
    work_place: str = ""
    locale: str = ""


@dataclass
class Settings:
    """Root configuration object."""

    home: str = "~/.nomorals"
    profile: str = "workstation"
    debug: bool = False
    offline: bool = False
    config_file: str = ""

    log: LogSettings = field(default_factory=LogSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    backup: BackupSettings = field(default_factory=BackupSettings)
    concurrency: ConcurrencySettings = field(default_factory=ConcurrencySettings)
    budget: BudgetSettings = field(default_factory=BudgetSettings)
    adaptive_budget: AdaptiveBudgetSettings = field(
        default_factory=AdaptiveBudgetSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    tools: ToolSettings = field(default_factory=ToolSettings)
    social: SocialSettings = field(default_factory=SocialSettings)
    mission: MissionSettings = field(default_factory=MissionSettings)
    api: APISettings = field(default_factory=APISettings)
    partner: PartnerSettings = field(default_factory=PartnerSettings)
    chat: ChatSettings = field(default_factory=ChatSettings)
    arena: ArenaSettings = field(default_factory=ArenaSettings)
    reasoning_mode: str = "always"  # pre-flight permanently on (wave 85);
    # every hook is budget-capped, and "auto"|"off" still opt down/out
    reasoning_knowledge: str = "on"
    autonomy: AutonomySettings = field(default_factory=AutonomySettings)
    evolution: EvolutionSettings = field(default_factory=EvolutionSettings)
    vision: VisionSettings = field(default_factory=VisionSettings)
    router_intelligent: str = "off"
    
    def __post_init__(self):
        """Validate router_intelligent and reasoning_knowledge values."""
        if self.router_intelligent not in ("on", "off"):
            from nomorals.core.errors import ConfigError
            raise ConfigError(f"router_intelligent must be 'on' or 'off', got '{self.router_intelligent}'")
        if self.reasoning_knowledge not in ("on", "off"):
            from nomorals.core.errors import ConfigError
            raise ConfigError(f"reasoning_knowledge must be 'on' or 'off', got '{self.reasoning_knowledge}'")
    runtime: "RuntimeSettings" = field(default_factory=lambda: RuntimeSettings())
    net: "NetSettings" = field(default_factory=lambda: NetSettings())
    proxy: "ProxySettings" = field(default_factory=lambda: ProxySettings())
    news: "NewsSettings" = field(default_factory=lambda: NewsSettings())
    scheduler: "SchedulerSettings" = field(default_factory=lambda: SchedulerSettings())
    audio: "AudioSettings" = field(default_factory=lambda: AudioSettings())
    osint: "OsintSettings" = field(default_factory=lambda: OsintSettings())
    improvement: "ImprovementSettings" = field(default_factory=lambda: ImprovementSettings())
    trading: "TradingSettings" = field(default_factory=lambda: TradingSettings())
    mind: "MindSettings" = field(default_factory=lambda: MindSettings())
    hub: "HubSettings" = field(default_factory=lambda: HubSettings())
    takeover: "TakeoverSettings" = field(
        default_factory=lambda: TakeoverSettings())
    location: "LocationSettings" = field(
        default_factory=lambda: LocationSettings())

    # -- path helpers --------------------------------------------------------
    @property
    def home_path(self) -> Path:
        return Path(os.path.expanduser(self.home)).resolve()

    def resolve(self, relative: str) -> Path:
        """Resolve a config path against the data home."""
        p = Path(os.path.expanduser(relative))
        return p if p.is_absolute() else (self.home_path / p).resolve()

    @property
    def db_path(self) -> Path:
        return self.resolve(self.storage.path)

    @property
    def blob_dir(self) -> Path:
        return self.resolve(self.storage.blob_dir)

    @property
    def backup_dir(self) -> Path:
        return self.resolve(self.backup.dir)

    @property
    def log_path(self) -> Path:
        return self.resolve(self.log.file)

    @property
    def model_cache_dir(self) -> Path:
        return self.resolve(self.llm.cache_dir)

    @property
    def workspace_dir(self) -> Path:
        return self.resolve(self.tools.workspace)

    def ensure_dirs(self) -> list[Path]:
        """Create every directory the system expects to exist."""
        created: list[Path] = []
        for path in (
            self.home_path,
            self.db_path.parent,
            self.blob_dir,
            self.backup_dir,
            self.log_path.parent,
            self.model_cache_dir,
            self.workspace_dir,
            self.resolve(self.training.data_dir),
            self.resolve(self.training.output_dir),
            self.resolve(self.mission.checkpoint_dir),
        ):
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
        return created

    # -- serialisation -------------------------------------------------------
    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        return _asdict(self, redact=redact)

    def replace(self, **changes: Any) -> "Settings":
        return replace(self, **changes)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Read a nested value by dotted path, e.g. ``llm.provider``."""
        node: Any = self
        for part in dotted.split("."):
            if not is_dataclass(node):
                return default
            if not hasattr(node, part):
                return default
            node = getattr(node, part)
        return node

    def set(self, dotted: str, value: Any) -> "Settings":
        """Return a NEW Settings with ``dotted`` path set to ``value``.

        The value is coerced to the declared field type (strings from the
        outside world are parsed like env vars), then the whole tree is
        rebuilt and validated — a bad value raises :class:`ConfigError` and
        the original Settings is untouched. Unknown paths also raise.
        """
        data = _settings_to_dict(self)
        ftype = _find_field_type(Settings, dotted)
        if ftype is None:
            # let _build raise the canonical "unknown config key" error
            _apply_dotted(data, dotted, value)
            return _build(Settings, data)
        if isinstance(value, str):
            target = ftype if isinstance(ftype, type) else str
            try:
                value = _coerce(value, target)
            except (ValueError, TypeError) as exc:
                raise ConfigError(
                    f"invalid value for {dotted!r}: {value!r} ({exc})") from exc
            if value is _EMPTY:
                raise ConfigError(f"empty value for {dotted!r}: not set")
        _apply_dotted(data, dotted, value)
        new = _build(Settings, data)
        _heal_retired_urls(new)
        _validate(new)
        return new

    def with_overrides(self, **overrides: Any) -> "Settings":
        """Return a NEW Settings with several dotted-path overrides applied.

        Keys are dotted paths (``llm.timeout=30``); values go through the
        same coerce-and-validate path as :meth:`set`. Applied in sorted key
        order so the result is deterministic.
        """
        result = self
        for dotted in sorted(overrides):
            result = result.set(dotted, overrides[dotted])
        return result


# ── Profiles ───────────────────────────────────────────────────────────────────

#: Presets layered over the defaults. Kept as plain dicts so they can be merged
#: with the same code path as TOML/env overrides.
PROFILES: dict[str, dict[str, Any]] = {
    "workstation": {
        "concurrency": {"threads": 16, "processes": 0, "use_processes": True},
        "memory": {"context_budget_tokens": 12000},
        "training": {"max_seq_len": 4096, "batch_size": 8},
    },
    "laptop": {
        "concurrency": {"threads": 8, "processes": 2, "use_processes": True},
        "memory": {"context_budget_tokens": 8000},
        "training": {"max_seq_len": 2048, "batch_size": 2},
    },
    "termux": {
        # fork() on Android is unreliable; stay on threads.
        "concurrency": {"threads": 4, "processes": 0, "use_processes": False},
        "memory": {"context_budget_tokens": 4000, "recall_limit": 8},
        "training": {"backend": "native", "max_seq_len": 1024, "batch_size": 1},
        "tools": {"max_concurrent_downloads": 1, "http_timeout": 45.0},
        "backup": {"keep": 5, "include_blobs": False},
    },
}


# ── Loading ────────────────────────────────────────────────────────────────────

_ENV_PREFIX = "NM_"

#: Map of env var -> dotted config path, for the non-obvious ones.
_ENV_MAP: dict[str, str] = {
    "NM_HOME": "home",
    "NM_PROFILE": "profile",
    "NM_DEBUG": "debug",
    "NM_OFFLINE": "offline",
    "NM_LOG_LEVEL": "log.level",
    "NM_LOG_FILE": "log.file",
    "NM_DB_PATH": "storage.path",
    "NM_DB_WAL": "storage.wal",
    "NM_BACKUP_DIR": "backup.dir",
    "NM_BACKUP_KEEP": "backup.keep",
    "NM_BACKUP_GIT_REPO": "backup.git_repo",
    "NM_THREADS": "concurrency.threads",
    "NM_PROCESSES": "concurrency.processes",
    "NM_MAX_SUBAGENTS": "concurrency.max_subagents",
    "NM_BUDGET_WALL_SECONDS": "budget.wall_seconds",
    "NM_BUDGET_TOKENS": "budget.tokens",
    "NM_BUDGET_CHILDREN": "budget.children",
    "NM_ADAPTIVE_BUDGET_TOKENS": "adaptive_budget.daily_token_budget",
    "NM_ADAPTIVE_BUDGET_COST_USD": "adaptive_budget.daily_cost_usd",
    "NM_ADAPTIVE_BUDGET_DOWNGRADE_AT": "adaptive_budget.downgrade_threshold",
    "NM_ADAPTIVE_BUDGET_TRUNCATE_AT": "adaptive_budget.truncate_threshold",
    "NM_ADAPTIVE_BUDGET_DEFER_AT": "adaptive_budget.defer_threshold",
    "NM_ADAPTIVE_BUDGET_MIN_CONTEXT_TOKENS": "adaptive_budget.min_context_tokens",
    "NM_LLM_PROVIDER": "llm.provider",
    "NM_LLM_FALLBACK_CHAIN": "llm.fallback_chain",
    "NM_GROQ_API_KEY": "llm.groq_api_key",
    "NM_GROQ_MODEL": "llm.groq_model",
    "NM_GROQ_BASE_URL": "llm.groq_base_url",
    "NM_OPENROUTER_API_KEY": "llm.openrouter_api_key",
    "NM_OPENROUTER_MODEL": "llm.openrouter_model",
    "NM_OPENROUTER_BASE_URL": "llm.openrouter_base_url",
    "NM_LLM_ACTIVE_MODEL": "llm.active_model",
    "NM_LLM_TIMEOUT": "llm.timeout",
    "NM_LLM_CACHE_DIR": "llm.cache_dir",
    "NM_LLM_SYSTEM_PROMPT": "llm.system_prompt",
    "NM_LLM_LOCAL_MODEL": "llm.local_model",
    "NM_LLM_LOCAL_AUTO_START": "llm.local_auto_start",
    "NM_LLM_LOCAL_HOST": "llm.local_host",
    "NM_LLM_LOCAL_PORT": "llm.local_port",
    "NM_LLM_VERIFY_AT_STARTUP": "llm.verify_at_startup",
    "NM_LLAMA_CPP_URL": "llm.llama_cpp_url",
    # Legacy alias: the setup wizard wrote this key for GGUF installs
    # for a long time while nothing read it — those installs sat on
    # the mock. It names the same thing NM_LLM_LOCAL_MODEL names.
    "NM_LLAMA_CPP_MODEL": "llm.local_model",
    # Audio STT/TTS: documented in the env template as "Config key read by
    # Devon code" but previously unmapped, so a user's .env setting
    # (e.g. NM_VOICE_STT_PROVIDER=api) was silently ignored.
    "NM_AUDIO_STT_API_KEY": "audio.stt_api_key",
    "NM_AUDIO_STT_BASE_URL": "audio.stt_base_url",
    "NM_AUDIO_TTS_VOICE": "audio.tts_voice",
    "NM_AUDIO_VOICE_SILENCE_MS": "audio.voice_silence_ms",
    "NM_VOICE_STT_MODEL": "audio.stt_model",
    "NM_VOICE_STT_PROVIDER": "audio.stt_provider",
    "NM_VOICE_TTS_ENGINE": "audio.tts_engine",
    # Documented alias for the TTS engine (env template + the Windows
    # installer's sample .env): previously unmapped, so an owner's
    # NM_VOICE_TTS_PROVIDER=api was silently ignored.
    "NM_VOICE_TTS_PROVIDER": "audio.tts_engine",
    # Trading: documented in the env template (NM_TRADING_ENABLED /
    # NM_TRADING_ALLOW_LIVE / NM_TRADING_MODE) and cited in the trading
    # tool's own refusal messages, but previously unmapped — the env
    # values were silently ignored (the master switch did nothing and
    # ALLOW_LIVE never reached live_enabled).
    "NM_TRADING_ENABLED": "trading.enabled",
    "NM_TRADING_ALLOW_LIVE": "trading.live_enabled",
    "NM_TRADING_MODE": "trading.mode",
    "HF_TOKEN": "llm.hf_token",
    "NM_HF_TOKEN": "llm.hf_token",
    "NM_HF_BASE_URL": "llm.hf_base_url",
    "NM_HF_ENDPOINT_URL": "llm.hf_endpoint_url",
    "NM_HF_MODEL": "llm.hf_model",
    "NM_OPENAI_BASE_URL": "llm.openai_base_url",
    "NM_OPENAI_API_KEY": "llm.openai_api_key",
    "NM_OPENAI_MODEL": "llm.openai_model",
    "NM_EMBED_PROVIDER": "embedding.provider",
    "NM_EMBED_MODEL": "embedding.model",
    "NM_MEMORY_TWO_TIER": "memory.two_tier_enabled",
    "NM_API_HOST": "api.host",
    "NM_API_PORT": "api.port",
    "NM_API_TOKEN": "api.token",
    "NM_HUB_URL": "hub.url",
    "NM_HUB_TOKEN": "hub.token",
    "NM_HUB_HOST": "hub.bind_host",
    "NM_HUB_PORT": "hub.port",
    "NM_HUB_TIMEOUT": "hub.request_timeout",
    "NM_TAKEOVER_ENABLED": "takeover.enabled",
    "NM_TAKEOVER_BASE_URL": "takeover.base_url",
    "NM_TAKEOVER_BIND_HOST": "takeover.bind_host",
    "NM_TAKEOVER_PORT": "takeover.port",
    "NM_TAKEOVER_LINK_TTL_S": "takeover.link_ttl_s",
    "NM_TAKEOVER_WAIT_TIMEOUT_S": "takeover.wait_timeout_s",
    "NM_TAKEOVER_POLL_S": "takeover.poll_s",
    "NM_CHAT_MEDIA_IN_GROUPS": "chat.media_in_groups",
    "NM_CHAT_MEDIA_MAX_MB": "chat.media_max_mb",
    "NM_CHAT_TELEGRAM_ENABLED": "chat.telegram_enabled",
    "NM_CHAT_TELEGRAM_SESSION": "chat.telegram_session",
    "NM_CHAT_TELEGRAM_API_ID": "chat.telegram_api_id",
    "NM_CHAT_TELEGRAM_API_HASH": "chat.telegram_api_hash",
    "NM_CHAT_TELEGRAM_CHATS": "chat.telegram_chats",
    "NM_CHAT_TELEGRAM_BOT_ENABLED": "chat.telegram_bot_enabled",
    "NM_CHAT_TELEGRAM_BOT_TOKEN": "chat.telegram_bot_token",
    "NM_CHAT_TELEGRAM_BOT_CHATS": "chat.telegram_bot_chats",
    "NM_CHAT_TELEGRAM_BOT_PARTNER_CONTEXT": "chat.telegram_bot_partner_context",
    "NM_CHAT_TELEGRAM_PERSONAL_BOT_ENABLED": "chat.telegram_personal_bot_enabled",
    "NM_CHAT_TELEGRAM_PERSONAL_BOT_TOKEN": "chat.telegram_personal_bot_token",
    "NM_CHAT_TELEGRAM_PERSONAL_BOT_CHATS": "chat.telegram_personal_bot_chats",
    "NM_CHAT_TELEGRAM_PERSONAL_BOT_PARTNER_CONTEXT": "chat.telegram_personal_bot_partner_context",
    "NM_CHAT_DISCORD_ENABLED": "chat.discord_enabled",
    "NM_CHAT_DISCORD_TOKEN": "chat.discord_token",
    "NM_CHAT_DISCORD_WEBHOOK": "social.discord_webhook",
    "NM_CHAT_DISCORD_CHANNELS": "chat.discord_channels",
    "NM_CHAT_WEBHOOK_ENABLED": "chat.webhook_enabled",
    "NM_CHAT_WEBHOOK_HOST": "chat.webhook_host",
    "NM_CHAT_WEBHOOK_PORT": "chat.webhook_port",
    "NM_CHAT_WEBHOOK_TOKEN": "chat.webhook_token",
    "NM_CHAT_WEBHOOK_REPLY_URL": "chat.webhook_reply_url",
    "NM_CHAT_SMS_ENABLED": "chat.sms_enabled",
    "NM_CHAT_SMS_FROM_NUMBER": "chat.sms_from_number",
    "NM_CHAT_SMS_HOST": "chat.sms_host",
    "NM_CHAT_SMS_PORT": "chat.sms_port",
    "NM_CHAT_SMS_AUTH_TOKEN": "chat.sms_auth_token",
    "NM_CHAT_SMS_PUBLIC_URL": "chat.sms_public_url",
    "NM_CHAT_WHATSAPP_OUTBOX_DIR": "chat.whatsapp_outbox_dir",
    "NM_ARENA_ENABLED": "arena.enabled",
    "NM_ARENA_BUILD": "arena.build",
    "NM_ARENA_INTERVAL_HOURS": "arena.interval_hours",
    "NM_ARENA_RESEARCH_PAGES": "arena.research_pages",
    "NM_ARENA_ANTI_REPEAT_WINDOW": "arena.anti_repeat_window",
    "NM_PARTNER_PLATFORMS": "partner.platforms",
    "NM_PARTNER_OWNER_CHATS": "partner.owner_chats",
    "NM_PARTNER_OWNER_KEY": "partner.owner_key",
    "NM_PARTNER_PERSONALITY": "partner.personality",
    "NM_PARTNER_MEMORY_ENABLED": "partner.memory_enabled",
    "NM_PARTNER_PROACTIVE": "partner.proactive",
    "NM_PARTNER_PROACTIVE_ENABLED": "partner.proactive_enabled",
    "NM_PARTNER_PROACTIVE_BRIEFING": "partner.proactive_briefing",
    "NM_PARTNER_PROACTIVE_WATCHERS": "partner.proactive_watchers",
    "NM_PARTNER_QUIET_START": "partner.quiet_start",
    "NM_PARTNER_QUIET_END": "partner.quiet_end",
    "NM_PARTNER_LEXICON_VOICE": "partner.lexicon_voice",
    "NM_PARTNER_LEXICON_SEED": "partner.lexicon_seed",
    "NM_MIND_MAX_INFLIGHT": "mind.max_inflight",
    "NM_MIND_INFLIGHT_TIMEOUT": "mind.inflight_acquire_timeout_s",
    "NM_NET_ENABLED": "net.enabled",
    "NM_NET_PROXY": "net.proxy",
    "NM_NET_PROBE_TIMEOUT": "net.probe_timeout",
    "NM_NET_ALLOWED_TARGETS": "net.allowed_targets",
    "NM_NET_DEFAULT_PORTS": "net.default_ports",
    "NM_NET_MAX_PROBE_PORTS": "net.max_probe_ports",
    "NM_NET_BANNER": "net.banner",
    "NM_PROXY_ACTIVE": "proxy.active",
    "NM_PROXY_KNOWN": "proxy.known",
    "NM_PROXY_URL": "proxy.url",
    "NM_OSINT_ENABLED": "osint.enabled",
    "NM_OSINT_HIBP_KEY": "osint.hibp_key",
    "NM_OSINT_ABUSEIPDB_KEY": "osint.abuseipdb_key",
    "NM_OSINT_SHODAN_KEY": "osint.shodan_key",
    "NM_OSINT_REQUEST_TIMEOUT": "osint.request_timeout",
    "NM_OSINT_CRTSH_DAYS": "osint.crtsh_days",
    # Location: documented in .env.example + the EC2 template, consumed
    # via settings.location by the weather places derivation.
    "NM_PLACES": "location.places",
    "NM_HOME_PLACE": "location.home_place",
    "NM_WORK_PLACE": "location.work_place",
    "NM_LOCALE": "location.locale",
    # Direct-read keys: no Settings field — the named module reads the
    # variable itself at runtime. load_settings skips "direct:" entries;
    # they exist so the doctor's config audit can account for the key
    # instead of reporting it as ignored.
    "NM_CAPTCHA_SOLVER": "direct:nomorals/tools/captcha.py",
    # ── Explicit entries for keys that previously resolved only
    # via the generic field search (same target load_settings
    # already applied — registering them makes the mapping
    # auditable in one place; behavior is unchanged).
    "NM_API_GITHUB_TOKEN": "api.github_token",
    "NM_AUTONOMY_DAILY_MODEL_CALLS": "autonomy.daily_model_calls",
    "NM_AUTONOMY_ENABLED": "autonomy.enabled",
    "NM_CHAT_WHATSAPP_ENABLED": "chat.whatsapp_enabled",
    "NM_GITHUB_TOKEN": "api.github_token",
    "NM_LLM_ALLOW_MOCK_FALLBACK": "llm.allow_mock_fallback",
    "NM_LLM_LOCAL_LORA": "llm.local_lora",
    "NM_MASTODON_TOKEN": "social.mastodon_token",
    "NM_MEDIA_MAX_MB": "chat.media_max_mb",
    "NM_OCR_BINARY": "vision.ocr_binary",
    "NM_PARTNER_OWNER_SENDER_IDS": "partner.owner_sender_ids",
    "NM_PARTNER_POWER_DEFAULT_ON": "partner.power_default_on",
    "NM_PARTNER_TRAIN_COLLECT": "partner.train_collect",
    "NM_PARTNER_TYPING_CAP_SECONDS": "partner.typing_cap_seconds",
    "NM_REASONING_MODE": "reasoning_mode",
    "NM_SOCIAL_BLUESKY_APP_PASSWORD": "social.bluesky_app_password",
    "NM_SOCIAL_BLUESKY_HANDLE": "social.bluesky_handle",
    "NM_SOCIAL_DISCORD_WEBHOOK": "social.discord_webhook",
    "NM_SOCIAL_MASTODON_DOMAIN": "social.mastodon_domain",
    "NM_SOCIAL_MASTODON_TOKEN": "social.mastodon_token",
    "NM_SOCIAL_REDDIT_CLIENT_ID": "social.reddit_client_id",
    "NM_SOCIAL_REDDIT_CLIENT_SECRET": "social.reddit_client_secret",
    "NM_SOCIAL_TELEGRAM_TOKEN": "social.telegram_token",
    "NM_SOCIAL_X_ACCESS_SECRET": "social.x_access_secret",
    "NM_SOCIAL_X_ACCESS_TOKEN": "social.x_access_token",
    "NM_SOCIAL_X_API_KEY": "social.x_api_key",
    "NM_SOCIAL_X_API_SECRET": "social.x_api_secret",
    "NM_TELEGRAM_API_HASH": "chat.telegram_api_hash",
    "NM_TELEGRAM_API_ID": "chat.telegram_api_id",
    "NM_TELEGRAM_BOT_TOKEN": "chat.telegram_bot_token",
    "NM_TRAINING_BASE_MODEL": "training.base_model",
    "NM_VISION_OCR_BINARY": "vision.ocr_binary",
    "NM_WORKERS": "api.workers",
    # ── Direct-read registry (generated from a verified AST scan of
    # the package: each key is read by the named module itself, so
    # load_settings skips it and the doctor accounts for it as
    # direct-read instead of ignored. Guarded by
    # test_config_mapping.py::test_direct_entries_have_real_consumers.
    "NM_AGENTIC_MODE": "direct:nomorals/agents/orchestration/bridge.py",
    "NM_AGENT_MODES_STORE": "direct:nomorals/agents/modes.py",
    "NM_ANTHROPIC_API_KEY": "direct:nomorals/agents/context.py",
    "NM_ARCHIVE_MAX_EXTRACT_FILES": "direct:nomorals/tools/archive.py",
    "NM_ARCHIVE_MAX_EXTRACT_MB": "direct:nomorals/tools/archive.py",
    "NM_BACKUP_KEY_B64": "direct:nomorals/backup/__main__.py",
    "NM_BRAVE_SEARCH_API_KEY": "direct:nomorals/search/web.py",
    "NM_CAPTCHA_API_MODE": "direct:nomorals/tools/captcha.py",
    "NM_CAPTCHA_DAILY_CAP_USD": "direct:nomorals/tools/captcha.py",
    "NM_CAPTCHA_ERROR_COOLDOWN": "direct:nomorals/tools/captcha.py",
    "NM_CAPTCHA_MONTHLY_CAP_USD": "direct:nomorals/tools/captcha.py",
    "NM_CAPTCHA_PER_DAY": "direct:nomorals/tools/captcha.py",
    "NM_CAPTCHA_PER_MINUTE": "direct:nomorals/tools/captcha.py",
    "NM_CLOUD": "direct:nomorals/core/profile.py",
    "NM_CODEBEAST_MODEL": "direct:nomorals/llm/defaults.py",
    "NM_CONSOLE_THEME": "direct:nomorals/agents/partner/runtime.py",
    "NM_COST_LOG": "direct:nomorals/llm/router.py",
    "NM_DDGS_BACKEND": "direct:nomorals/search/web.py",
    "NM_DDGS_KIND": "direct:nomorals/search/web.py",
    "NM_DDGS_REGION": "direct:nomorals/search/web.py",
    "NM_DEEPSEEK_API_KEY": "direct:nomorals/agents/context.py",
    "NM_DISTILL": "direct:nomorals/agents/skill_distillation.py",
    "NM_EXA_API_KEY": "direct:nomorals/search/web.py",
    "NM_GEMINI_API_KEY": "direct:nomorals/agents/context.py",
    "NM_GH_ASKPASS_TOKEN": "direct:nomorals/connectors/github.py",
    "NM_HF_BASE": "direct:nomorals/training/free_datasets.py",
    "NM_HF_SERVER": "direct:nomorals/training/free_datasets.py",
    "NM_HUB_CERT": "direct:nomorals/hub/server.py",
    "NM_HUB_CORS_ORIGINS": "direct:nomorals/hub/server.py",
    "NM_HUB_KEY": "direct:nomorals/hub/server.py",
    "NM_HUB_RATE_LIMIT": "direct:nomorals/hub/server.py",
    "NM_LLM_CASSETTE": "direct:nomorals/llm/cassette.py",
    "NM_LLM_CASSETTE_DIR": "direct:nomorals/llm/cassette.py",
    "NM_LLM_REPLAY": "direct:nomorals/llm/cassette.py",
    "NM_MEDIA_AUDIO_BITRATE": "direct:nomorals/tools/media_pipeline.py",
    "NM_MEDIA_IMAGE_MAX_WIDTH": "direct:nomorals/tools/media_pipeline.py",
    "NM_MEDIA_IMAGE_QUALITY": "direct:nomorals/tools/media_pipeline.py",
    "NM_MEDIA_VIDEO_CRF": "direct:nomorals/tools/media_pipeline.py",
    "NM_MEDIA_VIDEO_PRESET": "direct:nomorals/tools/media_pipeline.py",
    "NM_MISSION_STALE_HEARTBEAT_S": "direct:nomorals/missions/mission.py",
    "NM_MODELS_DIR": "direct:nomorals/media_edit/_model_cache.py",
    "NM_NO_COLOR": "direct:nomorals/cmdline/style.py",
    "NM_OWNER_TZ": "direct:nomorals/pulses/timezone.py",
    "NM_PLAN_STORE": "direct:nomorals/agents/plan_mode.py",
    "NM_PLAY_UNLICENSED": "direct:nomorals/media/play_service.py",
    "NM_PROCESS_WORKERS": "direct:nomorals/agents/runtime.py",
    "NM_PULSE_TIMEZONE": "direct:nomorals/pulses/timezone.py",
    "NM_REPO_DIR": "direct:nomorals/cmdline/commands/update.py",
    "NM_RESEARCH_LOOP_DEDUPE_DAYS": "direct:nomorals/agents/research_loop.py",
    "NM_RESEARCH_LOOP_HOURS": "direct:nomorals/agents/research_loop.py",
    "NM_RESEARCH_LOOP_IGNORE_DAYS": "direct:nomorals/agents/research_loop.py",
    "NM_RESEARCH_LOOP_MAX_PROPOSALS": "direct:nomorals/agents/research_loop.py",
    "NM_RESOURCE_BATTERY_MIN": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_BATTERY_THROTTLE": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_CALM_SECONDS": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_CPU_CRITICAL": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_CPU_THROTTLE": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_DISK_CRITICAL": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_DISK_THROTTLE": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_HISTORY": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_MAX_PRESSURE": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_MEM_CRITICAL": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_MEM_THROTTLE": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_METERED": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_PSI_MEM_FULL_CRIT": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_PSI_MEM_WARN": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_RESERVE_MIN_MB": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_RESERVE_PCT": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_STEAL_CRITICAL": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_STEAL_WARN": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_THERMAL_HOT_C": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_THERMAL_WARM_C": "direct:nomorals/os/resources.py",
    "NM_RESOURCE_UP_TICKS": "direct:nomorals/os/resources.py",
    "NM_SANDBOX_ROOT": "direct:nomorals/tools/sandbox_code.py",
    "NM_SEARXNG_CATEGORY": "direct:nomorals/search/web.py",
    "NM_SEARXNG_ENGINES": "direct:nomorals/search/web.py",
    "NM_SEARXNG_LANGUAGE": "direct:nomorals/search/web.py",
    "NM_SEARXNG_TIME_RANGE": "direct:nomorals/search/web.py",
    "NM_SEARXNG_URL": "direct:nomorals/search/web.py",
    "NM_SERPER_API_KEY": "direct:nomorals/search/web.py",
    "NM_SOUNDFONT": "direct:nomorals/media/synth_backend.py",
    "NM_SPEC_STORE": "direct:nomorals/agents/plan_spec.py",
    "NM_SVS_DIR": "direct:nomorals/voice/singing.py",
    "NM_SYNTH_BACKEND": "direct:nomorals/media/synth_backend.py",
    "NM_TAVILY_API_KEY": "direct:nomorals/search/web.py",
    "NM_TERMUX_NOTIFY": "direct:nomorals/agents/notifier.py",
    "NM_TIMEZONE": "direct:nomorals/agents/coremind.py",
    "NM_VAULT_KEY": "direct:nomorals/agents/cipher.py",
    "NM_VAULT_PASSPHRASE": "direct:nomorals/agents/coremind.py",
    "NM_VOICE_PRIVATE": "direct:nomorals/voice/tts.py",
    "NM_VOICE_PUBLIC": "direct:nomorals/voice/tts.py",
    "NM_WEB_RERANK": "direct:nomorals/search/web.py",
    "NM_WEB_TIMEOUT": "direct:nomorals/search/web.py",
    "NM_WHATSAPP_BRIDGE_HOST": "direct:nomorals/core/verify.py",
    "NM_WHATSAPP_BRIDGE_PORT": "direct:nomorals/core/verify.py",
    "NM_WHISPER_CPP_BIN": "direct:nomorals/tools/audio.py",
    "NM_WHISPER_CPP_MODEL": "direct:nomorals/tools/audio.py",
}


# Sentinel for "env var was set but empty" — callers treat this as "not set"
# and fall back to defaults instead of crashing on template placeholders.
_EMPTY: Any = object()


# Field name fragments that indicate a path value — empty strings for these
# mean "not set" (template placeholder), not a valid empty path.
_PATH_HINTS = ("path", "dir", "file", "home")


def _is_path_field(dotted: str) -> bool:
    """Does this config field hold a filesystem path?"""
    leaf = dotted.rsplit(".", 1)[-1].lower()
    return any(h in leaf for h in _PATH_HINTS)


def _coerce(value: str, target: type) -> Any:
    """Convert a string from env/TOML into the declared field type.

    Empty/whitespace-only strings mean "not set" for non-string types —
    returns the _EMPTY sentinel so callers can fall back to defaults
    instead of crashing on template placeholders like ``NM_FLOAT=``.
    String fields keep empty strings as-is (valid value, no crash).
    """
    if target is not str and not value.strip():
        return _EMPTY
    if target is bool:
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    if target is int:
        return int(value)
    if target is float:
        return float(value)
    if target is list or getattr(target, "__origin__", None) is list:
        return [v.strip() for v in value.split(",") if v.strip()]
    if target is dict or getattr(target, "__origin__", None) is dict:
        import json

        return json.loads(value)
    return value


@lru_cache(maxsize=None)
def _resolved_types(cls: type) -> dict[str, Any]:
    """Resolve a dataclass's annotations to real objects.

    ``from __future__ import annotations`` turns every annotation into a string, so
    ``fields(cls)[i].type`` is ``"LogSettings"`` rather than the class. Without this
    resolution, nested sections silently stay plain dicts.
    """
    import typing

    try:
        return typing.get_type_hints(cls)
    except Exception:  # pragma: no cover - unresolvable forward ref  # noqa: E104 - expected fallback
        return {f.name: f.type for f in fields(cls)}  # type: ignore[arg-type]


def _apply_dotted(root: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = root
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"cannot set {dotted!r}: {part!r} is not a section")
    node[parts[-1]] = value


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = dict(value) if isinstance(value, Mapping) else value
    return out


def _asdict(obj: Any, *, redact: bool = True) -> Any:
    """dataclasses.asdict with secret redaction."""
    if is_dataclass(obj):
        out: dict[str, Any] = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            if redact and _looks_secret(f.name) and isinstance(value, str) and value:
                out[f.name] = _mask(value)
            else:
                out[f.name] = _asdict(value, redact=redact)
        return out
    if isinstance(obj, dict):
        return {
            k: (_mask(v) if redact and _looks_secret(k) and isinstance(v, str) and v else _asdict(v, redact=redact))
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_asdict(v, redact=redact) for v in obj]
    return obj


_SECRET_HINTS = ("token", "secret", "password", "api_key", "apikey", "credential")


def _looks_secret(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _SECRET_HINTS)


def _mask(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-4:]}"


def _env_file_candidates(environ: Mapping[str, str],
                         default_home: str) -> list[Path]:
    """The .env files load_settings consults, in order. First existing wins."""
    env_home = environ.get("NM_HOME") or default_home
    return [Path.cwd() / ".env",
            Path(os.path.expanduser(env_home)) / ".env"]


def _read_env_file(environ: Mapping[str, str], default_home: str) -> dict[str, str]:
    """Parse the first existing .env candidate; {} when none exists."""
    for env_path in _env_file_candidates(environ, default_home):
        if env_path.is_file():
            return _parse_env_file(env_path)
    return {}


def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal .env parser: KEY=VALUE, # comments, optional quotes, `export` prefix."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for lineno, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif value.startswith("#"):
            # `KEY=   # note` — the value IS the comment: empty, not "#" text
            value = ""
        elif " #" in value:
            value = value.split(" #", 1)[0].strip()
        if key:
            values[key] = value
    return values


def _load_toml(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as fh:
        return tomllib.load(fh)


def _build(cls: type[T], data: Mapping[str, Any], path: str = "") -> T:
    """Construct a dataclass tree from a dict, coercing scalars and rejecting typos."""
    hints = _resolved_types(cls)
    kwargs: dict[str, Any] = {}
    valid = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    for key, value in data.items():
        if key not in valid:
            raise ConfigError(f"unknown config key: {path}{key!r} (section {cls.__name__})")
        declared = hints.get(key, str)
        origin = getattr(declared, "__origin__", None)
        if isinstance(value, Mapping) and isinstance(declared, type) and is_dataclass(declared):
            kwargs[key] = _build(declared, value, path=f"{path}{key}.")
        elif isinstance(value, str) and origin not in (dict, list):
            target = declared if isinstance(declared, type) else str
            try:
                kwargs[key] = _coerce(value, target)
            except (ValueError, TypeError) as exc:
                raise ConfigError(f"invalid value for {path}{key!r}: {value!r} ({exc})") from exc
        else:
            kwargs[key] = value
    return cls(**kwargs)  # type: ignore[call-arg]


def _settings_to_dict(settings: Settings) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(settings)


def load_settings(
    config_file: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
    use_env_file: bool = True,
) -> Settings:
    """Build a :class:`Settings` following the documented precedence."""
    environ = dict(os.environ if env is None else env)

    # 1. defaults
    merged: dict[str, Any] = _settings_to_dict(Settings())

    # 2. .env file — parsed FIRST so NM_PROFILE from an env file can select
    # the profile preset. The values themselves merge at step 4 (below) with
    # setdefault semantics: real environment wins over the file.
    if use_env_file:
        for key, value in _read_env_file(environ, merged["home"]).items():
            environ.setdefault(key, value)

    # 3. profile preset — layered over the defaults and UNDER the TOML file,
    # exactly as the module docstring promises. (A previous revision resolved
    # the profile after the TOML, letting the preset silently win over the
    # config file — the precedence lie the docstring always denied.)
    # Empty string = not set (fresh template placeholder) → default.
    profile = environ.get("NM_PROFILE") or merged.get("profile", "workstation")
    if profile not in PROFILES:
        raise ConfigError(
            f"unknown profile {profile!r}; expected one of {sorted(PROFILES)}"
        )
    merged = _deep_merge(merged, PROFILES[profile])
    merged["profile"] = profile

    # 4. TOML config file
    candidates: list[Path] = []
    if config_file:
        candidates.append(Path(os.path.expanduser(str(config_file))))
    else:
        home = Path(os.path.expanduser(environ.get("NM_HOME", merged["home"])))
        candidates += [home / "config.toml", Path.cwd() / "nomorals.toml"]
    for candidate in candidates:
        if candidate.is_file():
            merged = _deep_merge(merged, _load_toml(candidate))
            merged["config_file"] = str(candidate)
            break

    # 5. environment variables (NM_PROFILE already resolved; skip it here)
    for key, value in environ.items():
        if key == "NM_PROFILE":
            continue
        if key in _ENV_MAP:
            dotted = _ENV_MAP[key]
            if dotted.startswith("direct:"):
                # Consumed directly by the named module at runtime; the
                # settings tree has no field for it. Applying it as a
                # dotted path would inject a bogus key and fail the build.
                continue
            # Empty path fields = not set (template placeholder).
            if _is_path_field(dotted) and not value.strip():
                continue
            # Coerce value to proper type based on field definition
            field_type = _find_field_type(Settings, dotted)
            if field_type is not None:
                value = _coerce(value, field_type)
                if value is _EMPTY:
                    continue  # empty env var = not set, keep default
            _apply_dotted(merged, dotted, value)
            continue
        if not key.startswith(_ENV_PREFIX):
            continue
        dotted = key[len(_ENV_PREFIX) :].lower()
        if "." not in dotted:
            # NM_THREADS style without an explicit map entry: search sections.
            target = _find_field_path(Settings, dotted)
            if target is None:
                continue
            dotted = target
        # Empty path fields = not set (template placeholder).
        if _is_path_field(dotted) and not value.strip():
            continue
        # Coerce value to proper type based on field definition
        field_type = _find_field_type(Settings, dotted)
        if field_type is not None:
            value = _coerce(value, field_type)
            if value is _EMPTY:
                continue  # empty env var = not set, keep default
        _apply_dotted(merged, dotted, value)

    # 6. explicit overrides
    if overrides:
        for key, value in overrides.items():
            _apply_dotted(merged, key, value)

    # 6b. Belt-and-suspenders: home must never be empty (would resolve to cwd).
    if not merged.get("home", "").strip():
        merged["home"] = "~/.nomorals"

    settings = _build(Settings, merged)
    _heal_retired_urls(settings)
    _validate(settings)
    return settings


# Retired endpoints map onto their live successors so an old .env keeps
# working instead of failing at runtime with 410s.
_RETIRED_URLS = {
    "https://api-inference.huggingface.co": "https://router.huggingface.co/hf-inference",
    "http://api-inference.huggingface.co": "https://router.huggingface.co/hf-inference",
    "https://api-inference.huggingface.co/": "https://router.huggingface.co/hf-inference",
}


def _heal_retired_urls(settings: "Settings") -> None:
    """Rewrite decommissioned endpoints in place (config-hygiene pass)."""
    llm = settings.llm
    raw = (llm.hf_base_url or "").strip().rstrip("/")
    if not raw:
        llm.hf_base_url = _RETIRED_URLS["https://api-inference.huggingface.co"]
        return
    try:
        from urllib.parse import urlparse

        host = (urlparse(raw).hostname or "").lower()
    except Exception:  # noqa: BLE001
        host = ""
    if host == "api-inference.huggingface.co":
        llm.hf_base_url = _RETIRED_URLS["https://api-inference.huggingface.co"]


def _find_field_path(root: type, name: str) -> str | None:
    """Locate a bare field name inside the settings tree; returns a dotted path.

    Accepts both the bare name (``daily_model_calls``) and a section-prefixed
    spelling (``autonomy_daily_model_calls`` — how flattened env vars arrive,
    e.g. ``NM_AUTONOMY_DAILY_MODEL_CALLS``), since section and field names use
    underscores too.
    """
    for f in fields(root):  # type: ignore[arg-type]
        if f.name == name:
            return name
    hints = _resolved_types(root)
    for f in fields(root):  # type: ignore[arg-type]
        ftype = hints.get(f.name)
        if isinstance(ftype, type) and is_dataclass(ftype):
            sub = _find_field_path(ftype, name)
            if sub:
                return f"{f.name}.{sub}"
            # section-prefix form: "<section>_<rest>"
            if name.startswith(f.name + "_"):
                rest = _find_field_path(ftype, name[len(f.name) + 1:])
                if rest:
                    return f"{f.name}.{rest}"
    return None


def _find_field_type(root: type, dotted: str) -> type | None:
    """Find the type of a field given its dotted path."""
    parts = dotted.split(".")
    current = root
    hints = _resolved_types(current)
    
    for i, part in enumerate(parts):
        # Find the field in current dataclass
        found = False
        for f in fields(current):  # type: ignore[arg-type]
            if f.name == part:
                ftype = hints.get(part)
                if ftype is None:
                    return None
                
                # If this is the last part, return the type
                if i == len(parts) - 1:
                    return ftype
                
                # Otherwise, descend into the nested dataclass
                if isinstance(ftype, type) and is_dataclass(ftype):
                    current = ftype
                    hints = _resolved_types(current)
                    found = True
                    break
                else:
                    return None  # Not a dataclass, can't descend
        
        if not found:
            return None
    
    return None


def _validate(settings: Settings) -> None:
    # 0 = auto (template: "Thread pool size (0 = cpu_count)"). Resolve to a
    # real positive count BEFORE the >= 1 check below, so 0 genuinely means
    # auto exactly like concurrency.processes == 0 does — it must never
    # survive as 0, and negatives are still rejected by the check.
    if settings.concurrency.threads == 0:
        settings.concurrency.threads = max(1, os.cpu_count() or 1)
    problems: list[str] = []
    if settings.storage.synchronous not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
        problems.append(f"storage.synchronous must be OFF|NORMAL|FULL|EXTRA")
    if settings.backup.keep < 1:
        problems.append("backup.keep must be >= 1")
    if settings.concurrency.threads < 1:
        problems.append("concurrency.threads must be >= 1")
    if settings.concurrency.max_subagents < 1:
        problems.append("concurrency.max_subagents must be >= 1")
    if settings.concurrency.network_permits < 1:
        problems.append("concurrency.network_permits must be >= 1")
    if settings.concurrency.disk_permits < 1:
        problems.append("concurrency.disk_permits must be >= 1")
    if settings.budget.wall_seconds <= 0:
        problems.append("budget.wall_seconds must be > 0")
    if settings.budget.tokens < 0:
        problems.append("budget.tokens must be >= 0")
    if settings.budget.retries < 0:
        problems.append("budget.retries must be >= 0")
    if settings.storage.busy_timeout_ms < 0:
        problems.append("storage.busy_timeout_ms must be >= 0")
    if settings.backup.interval_seconds < 60:
        problems.append("backup.interval_seconds must be >= 60")
    if settings.memory.recall_limit < 1:
        problems.append("memory.recall_limit must be >= 1")
    if settings.embedding.dimensions < 16:
        problems.append("embedding.dimensions must be >= 16")
    if settings.memory.context_budget_tokens < 256:
        problems.append("memory.context_budget_tokens must be >= 256")
    if settings.api.port < 1 or settings.api.port > 65535:
        problems.append("api.port out of range")
    # LLM provider settings: catch typos and nonsense values at load time
    # instead of failing silently at runtime with "no model is answering".
    _KNOWN_PROVIDERS = {
        "mock", "offline", "test",
        "ollama", "ollama_native",
        "groq", "openrouter",
        "openai", "openai_compat", "vllm", "lmstudio", "ollama_compat",
        "hf", "hf_serverless", "huggingface", "hf_endpoint",
        "llama_cpp", "llamacpp", "gguf",
        "ocr", "tesseract",
    }
    provider = (settings.llm.provider or "").strip().lower()
    if not provider:
        problems.append("llm.provider must not be empty")
    elif provider not in _KNOWN_PROVIDERS:
        problems.append(
            f"llm.provider {provider!r} is not a known provider "
            f"(known: {', '.join(sorted(_KNOWN_PROVIDERS))})"
        )
    for fb in settings.llm.fallback_chain or []:
        if (fb or "").strip().lower() not in _KNOWN_PROVIDERS:
            problems.append(f"llm.fallback_chain contains unknown provider {fb!r}")
    if settings.llm.timeout <= 0:
        problems.append("llm.timeout must be > 0")
    if settings.chat.webhook_port < 0 or settings.chat.webhook_port > 65535:
        problems.append("chat.webhook_port out of range")
    if settings.chat.telegram_bot_token and not settings.chat.telegram_bot_token.strip():
        problems.append("chat.telegram_bot_token is blank (set a real token or leave it unset)")
    if settings.chat.telegram_personal_bot_token and not settings.chat.telegram_personal_bot_token.strip():
        problems.append("chat.telegram_personal_bot_token is blank (set a real token or leave it unset)")
    weight_sum = sum(settings.memory.weights.values())
    if weight_sum <= 0:
        problems.append("memory.weights must sum to a positive value")
    if any(w < 0 for w in settings.memory.weights.values()):
        problems.append("memory.weights must not contain negative values")
    if problems:
        raise ConfigError("invalid configuration: " + "; ".join(problems))


# ── Diff & schema ──────────────────────────────────────────────────────────────

def diff_settings(old: Settings, new: Settings,
                  *, redact: bool = True) -> list[dict[str, Any]]:
    """Dotted-path diff between two Settings trees.

    Returns ``[{"path": "llm.timeout", "old": 120.0, "new": 30.0}, …]``,
    sorted by path. Lists and scalars compare wholesale; nested sections
    recurse. Secret-looking values are masked when ``redact`` (the default),
    so the diff is safe to log and show to operators.
    """
    old_d = _asdict(old, redact=False)
    new_d = _asdict(new, redact=False)
    out: list[dict[str, Any]] = []

    def _show(path: str, value: Any) -> Any:
        if redact and _looks_secret(path.rsplit(".", 1)[-1]) and isinstance(value, str) and value:
            return _mask(value)
        return value

    def _walk(o: Any, n: Any, prefix: str) -> None:
        if isinstance(o, dict) and isinstance(n, dict):
            for key in sorted(set(o) | set(n)):
                _walk(o.get(key), n.get(key),
                      f"{prefix}.{key}" if prefix else key)
            return
        if o != n:
            out.append({"path": prefix, "old": _show(prefix, o),
                        "new": _show(prefix, n)})

    _walk(old_d, new_d, "")
    return out


def _type_label(tp: Any) -> str:
    """Human-readable label for a resolved field type."""
    if tp is None:
        return "any"
    origin = getattr(tp, "__origin__", None)
    if origin in (list,):
        args = getattr(tp, "__args__", ())
        inner = _type_label(args[0]) if args else "any"
        return f"list[{inner}]"
    if origin in (dict,):
        return "dict"
    if isinstance(tp, type):
        if is_dataclass(tp):
            return tp.__name__
        return tp.__name__
    return str(tp)


def config_schema() -> dict[str, Any]:
    """JSON-serializable schema of every settings section and field.

    Powers ``nm config schema`` / generated docs: each section maps to its
    fields, each field carrying its type label, default, and whether it
    looks like a secret. No values from a live Settings are included.
    """
    blank = _settings_to_dict(Settings())
    hints = _resolved_types(Settings)
    sections: dict[str, Any] = {}
    for f in fields(Settings):
        ftype = hints.get(f.name)
        if isinstance(ftype, type) and is_dataclass(ftype):
            sub_hints = _resolved_types(ftype)
            fd: dict[str, Any] = {}
            for sf in fields(ftype):
                if sf.default is not MISSING:
                    default = sf.default
                elif sf.default_factory is not MISSING:  # type: ignore[misc]
                    try:
                        default = sf.default_factory()  # type: ignore[misc]
                    except Exception:  # noqa: BLE001
                        default = None
                else:
                    default = None
                try:
                    import json as _json
                    _json.dumps(default)
                except (TypeError, ValueError):
                    default = str(default)
                fd[sf.name] = {
                    "type": _type_label(sub_hints.get(sf.name)),
                    "default": default,
                    "secret": _looks_secret(sf.name),
                }
            sections[f.name] = {"fields": fd}
        else:
            default = blank.get(f.name)
            sections[f.name] = {
                "type": _type_label(ftype),
                "default": default,
                "secret": _looks_secret(f.name),
            }
    return {"sections": sections,
            "profiles": sorted(PROFILES),
            "env_prefix": _ENV_PREFIX}


# ── Hot reload ─────────────────────────────────────────────────────────────────

class ConfigWatcher:
    """Hot-reload Settings when config sources change on disk.

    Uses ``watchfiles`` when installed (efficient, Rust-backed); otherwise
    falls back to an mtime-polling daemon thread — the same degrade path
    uvicorn uses. Change bursts are debounced; the loader runs, and the new
    Settings only replaces the old one when it builds AND validates cleanly
    (validate-before-swap: a broken edit is logged and the running config
    stays live). Reloads are atomic-write safe: a half-written TOML is
    retried once before being treated as an error.

    ``loader`` is a zero-arg callable returning Settings (usually a
    ``functools.partial(load_settings, ...)``). ``on_change(old, new,
    changed)`` receives the diff list from :func:`diff_settings`.
    """

    def __init__(
        self,
        loader: Any,
        paths: Iterable[str | os.PathLike[str]],
        *,
        on_change: Any = None,
        on_error: Any = None,
        debounce_s: float = 0.75,
        poll_interval_s: float = 2.0,
        use_watchfiles: bool = True,
        logger: Any = None,
    ) -> None:
        from pathlib import Path as _Path
        self._loader = loader
        self._paths = [_Path(p).expanduser() for p in paths]
        self._on_change = on_change
        self._on_error = on_error
        self._debounce_s = max(0.05, debounce_s)
        self._poll_interval_s = max(0.25, poll_interval_s)
        self._use_watchfiles = use_watchfiles
        self._log = logger
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._current: Settings | None = None
        self._reloads = 0
        self._failures = 0
        self._last_changed: list[dict[str, Any]] = []

    # -- state -----------------------------------------------------------------
    @property
    def current(self) -> Settings | None:
        with self._lock:
            return self._current

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def watched_paths(self) -> list[str]:
        return [str(p) for p in self._paths]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "paths": self.watched_paths,
                "reloads": self._reloads,
                "failures": self._failures,
                "last_changed": list(self._last_changed),
            }

    # -- lifecycle ---------------------------------------------------------------
    def start(self) -> "ConfigWatcher":
        """Start the watcher thread. Idempotent; primes the baseline first."""
        if self.running:
            return self
        self._prime()
        self._stop.clear()
        import threading as _threading
        self._thread = _threading.Thread(
            target=self._run, name="config-watcher", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the watcher thread. Never raises."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=timeout)
            except Exception:  # noqa: BLE001
                pass

    def _prime(self) -> None:
        try:
            with self._lock:
                self._current = self._loader()
        except Exception:  # noqa: BLE001 - loader errors surface on reload
            pass

    # -- watching ------------------------------------------------------------------
    def _run(self) -> None:
        if self._use_watchfiles:
            try:
                from watchfiles import watch as _watch
            except Exception:  # noqa: BLE001 - degrade to polling
                _watch = None
            if _watch is not None:
                try:
                    self._run_watchfiles(_watch)
                    return
                except Exception:  # noqa: BLE001 - degrade to polling
                    pass
        self._run_poll()

    def _run_watchfiles(self, _watch: Any) -> None:
        existing = [str(p) for p in self._paths if p.exists()]
        if not existing:
            self._run_poll()
            return
        for _changes in _watch(*existing, stop_event=self._stop,
                               debounce=int(self._debounce_s * 1000),
                               raise_interrupt=False):
            if self._stop.is_set():
                break
            self._maybe_reload()

    def _run_poll(self) -> None:
        import time as _time
        mtimes = self._mtimes()
        while not self._stop.wait(self._poll_interval_s):
            now = self._mtimes()
            if now != mtimes:
                mtimes = now
                _time.sleep(self._debounce_s)  # coalesce save bursts
                if self._stop.is_set():
                    break
                mtimes = self._mtimes()  # re-read after debounce
                self._maybe_reload()

    def _mtimes(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in self._paths:
            try:
                out[str(p)] = p.stat().st_mtime_ns if p.exists() else -1
            except OSError:
                out[str(p)] = -1
        return out

    # -- reload --------------------------------------------------------------------
    def _maybe_reload(self) -> bool:
        """Reload, validate, and swap. True when the live config changed."""
        import time as _time
        new: Settings | None = None
        last_exc: Exception | None = None
        for _ in range(2):  # retry once: tolerate a half-written TOML
            try:
                new = self._loader()
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 - validate-before-swap
                last_exc = exc
                _time.sleep(0.2)
        with self._lock:
            old = self._current
        if new is None:
            with self._lock:
                self._failures += 1
            self._emit_error(last_exc)
            return False
        if old is not None and not diff_settings(old, new, redact=False):
            return False  # mtime moved, content identical
        changed = diff_settings(old, new) if old is not None else []
        with self._lock:
            self._current = new
            self._reloads += 1
            self._last_changed = changed
        self._emit_change(old, new, changed)
        return True

    def reload_now(self) -> list[dict[str, Any]]:
        """Force a reload check immediately; returns the diff (maybe empty)."""
        self._maybe_reload()
        with self._lock:
            return list(self._last_changed)

    def _emit_change(self, old: Settings | None, new: Settings,
                     changed: list[dict[str, Any]]) -> None:
        if self._on_change is not None:
            try:
                self._on_change(old, new, changed)
            except Exception:  # noqa: BLE001 - callbacks never break reloads
                if self._log:
                    self._log.exception("config on_change callback failed")
        elif self._log:
            paths = ", ".join(c["path"] for c in changed) or "(no changes)"
            self._log.info("config reloaded: %s", paths)

    def _emit_error(self, exc: Exception | None) -> None:
        if self._on_error is not None:
            try:
                self._on_error(exc)
            except Exception:  # noqa: BLE001
                pass
        elif self._log and exc is not None:
            self._log.warning("config reload failed (keeping old config): %s", exc)


def watch_settings(*, config_file: str | os.PathLike[str] | None = None,
                   env: Mapping[str, str] | None = None,
                   overrides: Mapping[str, Any] | None = None,
                   on_change: Any = None, on_error: Any = None,
                   logger: Any = None, **watch_kw: Any) -> tuple[Settings, ConfigWatcher]:
    """Load settings once and return ``(settings, watcher)``.

    The watcher tracks the TOML file that was actually loaded plus the .env
    candidates, and hot-reloads on change. Call ``watcher.start()`` to arm
    it; ``watcher.current`` always holds the latest validated Settings.
    """
    from functools import partial
    settings = load_settings(config_file=config_file, env=env,
                             overrides=overrides)
    paths: list[str] = []
    if settings.config_file:
        paths.append(settings.config_file)
    else:
        base_env = dict(os.environ if env is None else env)
        paths.extend(str(p) for p in _env_file_candidates(
            base_env, settings.home))
    paths.extend(str(p) for p in _env_file_candidates(
        dict(os.environ if env is None else env), settings.home))
    # de-dupe, keep order
    seen: set[str] = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]
    loader = partial(load_settings, config_file=config_file, env=env,
                     overrides=overrides)
    watcher = ConfigWatcher(loader, paths, on_change=on_change,
                            on_error=on_error, logger=logger, **watch_kw)
    return settings, watcher


def start_global_watcher(*, on_change: Any = None, on_error: Any = None,
                         logger: Any = None, **kwargs: Any) -> ConfigWatcher:
    """Load, publish process-wide, and hot-reload the global settings.

    ``kwargs`` pass through to :func:`load_settings` (``config_file``,
    ``env``, ``overrides``). The default ``on_change`` swaps the global via
    :func:`set_settings` so :func:`get_settings` always returns the latest
    validated config.
    """
    def _default_on_change(_old: Settings, new: Settings,
                           changed: list[dict[str, Any]]) -> None:
        set_settings(new)

    settings, watcher = watch_settings(
        on_change=on_change or _default_on_change,
        on_error=on_error, logger=logger, **kwargs)
    set_settings(settings)
    watcher.start()
    return watcher


# ── Process-wide accessor ──────────────────────────────────────────────────────

_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first use."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def set_settings(settings: Settings) -> None:
    global _settings
    _settings = settings


def reset_settings() -> None:
    """Forget the cached settings (used by tests)."""
    global _settings
    _settings = None


def env_var_path(env_var: str) -> str | None:
    """Convert an environment variable name to its dotted config path.

    Returns None when the variable maps to no real config field.
    
    Examples:
        env_var_path("NM_HOME") -> "home"
        env_var_path("NM_CHAT_MEDIA_IN_GROUPS") -> "chat.media_in_groups"
        env_var_path("HF_TOKEN") -> "llm.hf_token"
    """
    # Check explicit map first
    if env_var in _ENV_MAP:
        return _ENV_MAP[env_var]
    
    # Strip NM_ prefix if present
    if env_var.startswith("NM_"):
        remainder = env_var[3:]  # Strip "NM_"
    else:
        remainder = env_var
    
    # Validate against the real settings tree: an NM_FOO_BAR that matches no
    # field must resolve to None (silently-ignored env vars were the bug
    # this guard exists to catch).
    return _find_field_path(Settings, remainder.lower())


# ── Env-key audit (doctor) ─────────────────────────────────────────────────
#
# The doctor's config audit (infra spec §8.1) answers one question for
# every NM_* key the operator has set: does it DO anything? Three honest
# outcomes exist:
#
# - mapped:  the key lands on a Settings field (via _ENV_MAP or the
#            generic field search) and load_settings applies it.
# - direct:  no Settings field — a module reads the variable itself at
#            runtime. Either registered explicitly in _ENV_MAP as
#            "direct:<module>", or discovered by scanning the package
#            for the literal key in a consumption position.
# - ignored: set, but nothing reads it. Always a bug (owner typo, a
#            renamed key, or a documented key that was never wired).

_DIRECT_PREFIX = "direct:"


def dotted_path_exists(dotted: str) -> bool:
    """True when ``dotted`` resolves to a real field on the Settings tree."""
    node: type = Settings
    parts = dotted.split(".")
    for i, part in enumerate(parts):
        names = {f.name for f in fields(node)}  # type: ignore[arg-type]
        if part not in names:
            return False
        if i == len(parts) - 1:
            return True
        ftype = _resolved_types(node).get(part)
        if not (isinstance(ftype, type) and is_dataclass(ftype)):
            return False
        node = ftype
    return False


@lru_cache(maxsize=1)
def _direct_env_readers() -> dict[str, str]:
    """Map env key -> first module (repo-relative path) that reads it.

    AST scan of the ``nomorals`` package for key-shaped string constants
    in consumption positions: call arguments, Load subscripts, dict
    keys/values, tuple/list/set literals, assignments, and comparisons.
    Comments and docstrings never appear in those positions, so a key
    that is only *mentioned* in prose does not count as read. This
    module itself is excluded — it is the registry, not a consumer.
    """
    import ast
    import re

    key_re = re.compile(r"^(?:NM|HF)_[A-Z0-9_]+$")
    package = Path(__file__).resolve().parent.parent
    repo_root = package.parent
    self_path = Path(__file__).resolve()
    readers: dict[str, str] = {}

    def collect(node: Any, into: set[str]) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                    and key_re.match(sub.value):
                into.add(sub.value)

    for path in sorted(package.rglob("*.py")):
        if "__pycache__" in path.parts or path.resolve() == self_path:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8",
                                            errors="replace"))
        except (OSError, SyntaxError):
            continue
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    collect(arg, found)
            elif isinstance(node, ast.Subscript) \
                    and isinstance(node.ctx, ast.Load):
                collect(node.slice, found)
            elif isinstance(node, ast.Dict):
                for item in list(node.keys) + list(node.values):
                    if item is not None:
                        collect(item, found)
            elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                for item in node.elts:
                    collect(item, found)
            elif isinstance(node, ast.Assign):
                collect(node.value, found)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                collect(node.value, found)
            elif isinstance(node, ast.Compare):
                collect(node.left, found)
                for comparator in node.comparators:
                    collect(comparator, found)
        module = path.relative_to(repo_root).as_posix()
        for key in sorted(found):
            readers.setdefault(key, module)
    return readers


def _registered_direct(key: str) -> str | None:
    """The module an explicit ``direct:`` _ENV_MAP entry names, or None."""
    target = _ENV_MAP.get(key)
    if target is not None and target.startswith(_DIRECT_PREFIX):
        return target[len(_DIRECT_PREFIX):]
    return None


def is_direct_env_key(key: str) -> bool:
    """True when ``key`` is consumed directly by a module (no Settings field)."""
    if key in _ENV_MAP:
        return _ENV_MAP[key].startswith(_DIRECT_PREFIX)
    return key in _direct_env_readers()


def direct_env_consumer(key: str) -> str | None:
    """The module that reads ``key`` directly, or None when nothing does."""
    registered = _registered_direct(key)
    if registered is not None:
        return registered
    if key in _ENV_MAP:
        return None
    return _direct_env_readers().get(key)


def config_env_status(
        env: Mapping[str, str] | None = None) -> dict[str, list[str]]:
    """Classify every set NM_* key: mapped / direct-read / ignored.

    With no argument, audits the EFFECTIVE environment the runtime
    boots with — ``os.environ`` merged over the boot ``.env`` file,
    exactly as :func:`load_settings` resolves it — so the doctor never
    audits a different configuration than the one that is running.

    Returns ``{"set_mapped": [...], "set_direct": [...],
    "unset_mapped": [...], "ignored": [...]}`` (all sorted lists).
    ``unset_mapped`` lists the explicitly-mapped keys that are not set
    (they fall back to defaults).
    """
    if env is None:
        effective: dict[str, str] = dict(os.environ)
        for key, value in _read_env_file(effective, Settings().home).items():
            effective.setdefault(key, value)
    else:
        effective = dict(env)
    readers = _direct_env_readers()
    set_mapped: list[str] = []
    set_direct: list[str] = []
    ignored: list[str] = []
    for key in sorted(effective):
        if key not in _ENV_MAP and not key.startswith(_ENV_PREFIX):
            continue
        target = _ENV_MAP.get(key)
        if target is not None:
            (set_direct if target.startswith(_DIRECT_PREFIX)
             else set_mapped).append(key)
        elif env_var_path(key) is not None:
            set_mapped.append(key)
        elif key in readers:
            set_direct.append(key)
        else:
            ignored.append(key)
    unset_mapped = sorted(k for k in _ENV_MAP if k not in effective)
    return {"set_mapped": set_mapped, "set_direct": set_direct,
            "unset_mapped": unset_mapped, "ignored": ignored}


# ── redaction + validation ───────────────────────────────────────────────────

#: field-name fragments that mark a value as secret (case-insensitive).
_SECRET_HINTS = ("token", "secret", "password", "passwd", "api_key",
                 "apikey", "private_key", "auth", "credential", "bearer")


def _is_secret_key(key: str) -> bool:
    low = key.lower()
    return any(h in low for h in _SECRET_HINTS)


def redacted_dict(settings: "Settings | None" = None,
                  *, placeholder: str = "***") -> dict[str, Any]:
    """Dump settings to a plain dict with secret values redacted.

    Any field whose name smells like a credential (token/secret/password/
    api_key/…) is replaced by ``placeholder`` when set, ``""`` when empty —
    so the dump shows *which* secrets are configured without leaking them.
    Safe to log, paste into bug reports, or show in ``/config`` output.
    """
    data = _settings_to_dict(settings or get_settings())

    def scrub(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: (placeholder if v else "")
                    if _is_secret_key(k) else scrub(v)
                    for k, v in node.items()}
        if isinstance(node, list):
            return [scrub(v) for v in node]
        return node

    return scrub(data)


def validate_settings(settings: "Settings | None" = None) -> list[str]:
    """Structural validation of a Settings tree.

    Returns a list of human-readable problems (empty = valid). Checks:
    every leaf value matches its declared dataclass type, numeric budgets
    are non-negative, and string enums hold allowed values where the
    codebase constrains them.
    """
    from dataclasses import fields, is_dataclass

    problems: list[str] = []
    root = settings or get_settings()

    def check(node: Any, path: str) -> None:
        if not is_dataclass(node):
            return
        for f in fields(node):
            val = getattr(node, f.name)
            p = f"{path}.{f.name}" if path else f.name
            ftype = f.type
            # unwrap Optional[X] / X | None
            optional = False
            origin = getattr(ftype, "__origin__", None)
            args = getattr(ftype, "__args__", ())
            if str(ftype).startswith("typing.Optional") or (
                    origin is not None and type(None) in args):
                optional = True
                non_none = [a for a in args if a is not type(None)]
                ftype = non_none[0] if non_none else Any
            if val is None and optional:
                continue
            if is_dataclass(val):
                check(val, p)
                continue
            want = ftype if isinstance(ftype, type) else None
            if want is not None and not isinstance(val, want):
                # bool is an int subclass; int where bool declared is wrong
                if not (want is int and isinstance(val, bool) is False):
                    problems.append(
                        f"{p}: expected {want.__name__}, "
                        f"got {type(val).__name__}")
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if "budget" in f.name or f.name in (
                        "tokens", "max_tokens", "context_budget_tokens",
                        "max_parallel", "workers"):
                    if val < 0:
                        problems.append(f"{p}: negative value {val}")
        # enum-ish string fields the codebase constrains
        mode = getattr(node, "reasoning_mode", None)
        if isinstance(node, Settings):
            if mode not in ("always", "auto", "off"):
                problems.append(f"reasoning_mode: invalid {mode!r}")

    check(root, "")
    return problems
