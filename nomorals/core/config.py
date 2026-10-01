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
from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar

from .errors import ConfigError

__all__ = ["Settings", "get_settings", "load_settings", "reset_settings"]

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
    processes: int = 0  # 0 -> cpu_count // 2; forced to 0 on termux
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
    provider: str = "hashing"
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
    telegram_enabled: bool = False
    telegram_session: str = ""
    telegram_api_id: str = ""
    telegram_api_hash: str = ""
    telegram_chats: str = ""
    threads_enabled: bool = True
    telegram_bot_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_bot_chats: str = ""
    discord_enabled: bool = False
    discord_token: str = ""
    whatsapp_enabled: bool = False
    whatsapp_host: str = "127.0.0.1"
    whatsapp_port: int = 8787
    webhook_enabled: bool = False
    webhook_host: str = "127.0.0.1"
    webhook_port: int = 0
    webhook_token: str = ""
    webhook_reply_url: str = ""
    max_per_hour: int = 60


@dataclass
class AutonomySettings:
    enabled: bool = False
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
    enabled: bool = False


@dataclass
class VisionSettings:
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    ocr_binary: str = ""
    ocr_language: str = "eng"
    enabled: bool = False
    # Prompt 09: vision tool knobs. Screenshots stay a privileged, explicit,
    # per-call-confirmed action — never scheduled, never background.
    allow_screenshot: bool = False
    max_dimension: int = 1568  # downscale images over this (px) before sending
    max_image_bytes: int = 25 * 1024 * 1024  # reject inputs over this
    log_calls: bool = True  # per-call audit log (timestamp/source/bytes, never pixels)


@dataclass
class ArenaSettings:
    enabled: bool = False
    build: bool = False
    research_pages: int = 3
    interval_hours: int = 24
    #: Topic anti-repeat window: no topic repeats within the last N
    #: served. Env: NM_ARENA_ANTI_REPEAT_WINDOW. A per-DB kv override
    #: (arena.anti_repeat_window) wins when set.
    anti_repeat_window: int = 10


@dataclass
class PartnerSettings:
    enabled: bool = True
    autonomy_mode: str = "suggest"
    typing_cap_seconds: float = 0.0
    platforms: str = "local"
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
    enabled: bool = False
    mode: str = "off"
    target: float = 0.8
    benchmark: str = ""
    auto_tick: bool = False

@dataclass
class TradingSettings:
    """FinancialExpert / trading gates. Live trading is inert by default."""
    live_enabled: bool = False
    max_daily_loss_pct: float = 3.0
    min_sharpe: float = 1.0
    max_drawdown_pct: float = 20.0
    min_profit_factor: float = 1.3
    unlock_ttl_hours: float = 24.0
    followed_symbols: list = field(default_factory=list)

@dataclass
class OsintSettings:
    enabled: bool = False
    sources: list = field(default_factory=list)
    hibp_key: str = ""
    abuseipdb_key: str = ""
    shodan_key: str = ""
    request_timeout: float = 10.0
    crtsh_days: int = 90

@dataclass
class AudioSettings:
    enabled: bool = False
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
    enabled: bool = False
    sources: list = field(default_factory=list)

@dataclass
class NetSettings:
    enabled: bool = False
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
    enabled: bool = False
    profile: str = ""
    threads: int = 4

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
    "NM_API_HOST": "api.host",
    "NM_API_PORT": "api.port",
    "NM_API_TOKEN": "api.token",
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
    "NM_CHAT_WEBHOOK_ENABLED": "chat.webhook_enabled",
    "NM_CHAT_WEBHOOK_HOST": "chat.webhook_host",
    "NM_CHAT_WEBHOOK_PORT": "chat.webhook_port",
    "NM_CHAT_WEBHOOK_TOKEN": "chat.webhook_token",
    "NM_CHAT_WEBHOOK_REPLY_URL": "chat.webhook_reply_url",
    "NM_ARENA_ENABLED": "arena.enabled",
    "NM_ARENA_BUILD": "arena.build",
    "NM_ARENA_INTERVAL_HOURS": "arena.interval_hours",
    "NM_ARENA_RESEARCH_PAGES": "arena.research_pages",
    "NM_ARENA_ANTI_REPEAT_WINDOW": "arena.anti_repeat_window",
    "NM_PARTNER_PLATFORMS": "partner.platforms",
    "NM_PARTNER_OWNER_CHATS": "partner.owner_chats",
    "NM_PARTNER_PERSONALITY": "partner.personality",
    "NM_PARTNER_MEMORY_ENABLED": "partner.memory_enabled",
    "NM_PARTNER_PROACTIVE": "partner.proactive",
    "NM_PARTNER_PROACTIVE_ENABLED": "partner.proactive_enabled",
    "NM_PARTNER_PROACTIVE_BRIEFING": "partner.proactive_briefing",
    "NM_PARTNER_PROACTIVE_WATCHERS": "partner.proactive_watchers",
    "NM_PARTNER_QUIET_START": "partner.quiet_start",
    "NM_PARTNER_QUIET_END": "partner.quiet_end",
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
}


def _coerce(value: str, target: type) -> Any:
    """Convert a string from env/TOML into the declared field type."""
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

    # 2. profile (may itself be overridden by env, so read profile first)
    profile = environ.get("NM_PROFILE", merged.get("profile", "workstation"))
    if profile not in PROFILES:
        raise ConfigError(
            f"unknown profile {profile!r}; expected one of {sorted(PROFILES)}"
        )
    merged = _deep_merge(merged, PROFILES[profile])
    merged["profile"] = profile

    # 3. TOML config file
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

    # 4. .env file
    if use_env_file:
        env_home = environ.get("NM_HOME") or merged["home"]
        for env_path in (Path.cwd() / ".env",
                         Path(os.path.expanduser(env_home)) / ".env"):
            for key, value in _parse_env_file(env_path).items():
                environ.setdefault(key, value)
            if env_path.is_file():
                break

    # 5. environment variables
    for key, value in environ.items():
        if key in _ENV_MAP:
            dotted = _ENV_MAP[key]
            # Coerce value to proper type based on field definition
            field_type = _find_field_type(Settings, dotted)
            if field_type is not None:
                value = _coerce(value, field_type)
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
        # Coerce value to proper type based on field definition
        field_type = _find_field_type(Settings, dotted)
        if field_type is not None:
            value = _coerce(value, field_type)
        _apply_dotted(merged, dotted, value)

    # 6. explicit overrides
    if overrides:
        for key, value in overrides.items():
            _apply_dotted(merged, key, value)

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
    problems: list[str] = []
    if settings.storage.synchronous not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
        problems.append(f"storage.synchronous must be OFF|NORMAL|FULL|EXTRA")
    if settings.backup.keep < 1:
        problems.append("backup.keep must be >= 1")
    if settings.concurrency.threads < 1:
        problems.append("concurrency.threads must be >= 1")
    if settings.embedding.dimensions < 16:
        problems.append("embedding.dimensions must be >= 16")
    if settings.memory.context_budget_tokens < 256:
        problems.append("memory.context_budget_tokens must be >= 256")
    if settings.api.port < 1 or settings.api.port > 65535:
        problems.append("api.port out of range")
    weight_sum = sum(settings.memory.weights.values())
    if weight_sum <= 0:
        problems.append("memory.weights must sum to a positive value")
    if problems:
        raise ConfigError("invalid configuration: " + "; ".join(problems))


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
