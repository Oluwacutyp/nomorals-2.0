"""The schema, expressed as ordered migrations.

Every table the system persists to lives here. Adding a table means adding a
migration at the end of :data:`MIGRATIONS` — never editing an earlier one, which
:class:`~nomorals.storage.schema.MigrationRunner` will detect and refuse.
"""

from __future__ import annotations

from .schema import Migration

__all__ = ["MIGRATIONS", "latest_version"]

# ── 0001: core state ───────────────────────────────────────────────────────────

_V1 = """
-- Key/value store for settings, runtime state, and pointers.
CREATE TABLE IF NOT EXISTS kv_store (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'json',
    updated_at  REAL NOT NULL
);

-- Conversations and their messages: the raw material for memory and training.
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    agent       TEXT NOT NULL DEFAULT '',
    channel     TEXT NOT NULL DEFAULT 'cli',
    summary     TEXT NOT NULL DEFAULT '',
    tokens      INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '',
    name            TEXT NOT NULL DEFAULT '',
    tokens          INTEGER NOT NULL DEFAULT 0,
    model           TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL,
    metadata        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);

-- Episodic + semantic memory. `kind` discriminates; scoring columns are shared.
CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,              -- episode | fact | preference | skill | lesson
    content      TEXT NOT NULL,
    importance   REAL NOT NULL DEFAULT 0.5,
    salience     REAL NOT NULL DEFAULT 0.5,
    decay        REAL NOT NULL DEFAULT 1.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_access  REAL NOT NULL DEFAULT 0,
    embedding_id TEXT NOT NULL DEFAULT '',
    source       TEXT NOT NULL DEFAULT '',
    agent        TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    expires_at   REAL,
    metadata     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_salience ON memories(salience DESC);

-- Typed relations between memories (supersedes, derives-from, contradicts…).
CREATE TABLE IF NOT EXISTS memory_links (
    id         TEXT PRIMARY KEY,
    src        TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    dst        TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation   TEXT NOT NULL,
    weight     REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    UNIQUE(src, dst, relation)
);
CREATE INDEX IF NOT EXISTS idx_memory_links_dst ON memory_links(dst, relation);

-- Subject-predicate-object claims with provenance and supersession.
CREATE TABLE IF NOT EXISTS facts (
    id            TEXT PRIMARY KEY,
    subject       TEXT NOT NULL,
    predicate     TEXT NOT NULL,
    object        TEXT NOT NULL,
    confidence    REAL NOT NULL DEFAULT 0.8,
    provenance    TEXT NOT NULL DEFAULT '',
    superseded_by TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL,
    metadata      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject, predicate);

-- Embeddings live apart from rows so they can be rebuilt without touching data.
CREATE TABLE IF NOT EXISTS embeddings (
    id         TEXT PRIMARY KEY,
    owner_type TEXT NOT NULL,
    owner_id   TEXT NOT NULL,
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    norm       REAL NOT NULL DEFAULT 0,
    vector     BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_owner ON embeddings(owner_type, owner_id);
"""

# ── 0002: agents, tasks, missions ─────────────────────────────────────────────

_V2 = """
CREATE TABLE IF NOT EXISTS agents (
    id           TEXT PRIMARY KEY,
    role         TEXT NOT NULL,
    name         TEXT NOT NULL DEFAULT '',
    parent_id    TEXT NOT NULL DEFAULT '',
    mission_id   TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'idle',   -- idle|running|done|failed|cancelled
    capabilities TEXT NOT NULL DEFAULT '[]',
    spawned_at   REAL NOT NULL,
    finished_at  REAL,
    metadata     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_agents_parent ON agents(parent_id);
CREATE INDEX IF NOT EXISTS idx_agents_mission ON agents(mission_id, status);

CREATE TABLE IF NOT EXISTS missions (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    goal           TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'pending', -- pending|running|paused|done|failed|cancelled
    state          TEXT NOT NULL DEFAULT '{}',
    budget_wall    REAL NOT NULL DEFAULT 0,
    budget_tokens  INTEGER NOT NULL DEFAULT 0,
    spent_wall     REAL NOT NULL DEFAULT 0,
    spent_tokens   INTEGER NOT NULL DEFAULT 0,
    iterations     INTEGER NOT NULL DEFAULT 0,
    success        REAL,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    finished_at    REAL,
    metadata       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_missions_status ON missions(status, updated_at DESC);

CREATE TABLE IF NOT EXISTS mission_checkpoints (
    id         TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
    label      TEXT NOT NULL DEFAULT '',
    state      TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_mission ON mission_checkpoints(mission_id, created_at DESC);

-- Task DAG. `deps` is a JSON array of task ids.
CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    mission_id  TEXT NOT NULL DEFAULT '',
    parent_id   TEXT NOT NULL DEFAULT '',
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'io',      -- io | cpu | async
    agent_role  TEXT NOT NULL DEFAULT '',
    payload     TEXT NOT NULL DEFAULT '{}',
    deps        TEXT NOT NULL DEFAULT '[]',
    status      TEXT NOT NULL DEFAULT 'pending', -- pending|ready|running|done|failed|cancelled
    priority    INTEGER NOT NULL DEFAULT 0,
    attempts    INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    result      TEXT NOT NULL DEFAULT '',
    error       TEXT NOT NULL DEFAULT '',
    tokens      INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    started_at  REAL,
    finished_at REAL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_tasks_mission_status ON tasks(mission_id, status);
CREATE INDEX IF NOT EXISTS idx_tasks_status_priority ON tasks(status, priority DESC, created_at);

-- Self-reflection output: what the system learned from each mission.
CREATE TABLE IF NOT EXISTS reflections (
    id         TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL DEFAULT '',
    score      REAL NOT NULL DEFAULT 0,
    summary    TEXT NOT NULL DEFAULT '',
    lessons    TEXT NOT NULL DEFAULT '[]',
    weights    TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reflections_mission ON reflections(mission_id, created_at DESC);
"""

# ── 0003: tools, models, training ─────────────────────────────────────────────

_V3 = """
-- Append-only record of every tool invocation and its policy decision.
CREATE TABLE IF NOT EXISTS tool_calls (
    id             TEXT PRIMARY KEY,
    actor          TEXT NOT NULL DEFAULT '',
    tool           TEXT NOT NULL,
    capability     TEXT NOT NULL DEFAULT '',
    decision       TEXT NOT NULL DEFAULT '',   -- allow | deny | confirm
    args_digest    TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'pending', -- pending|ok|error|denied
    duration_ms    REAL NOT NULL DEFAULT 0,
    result_digest  TEXT NOT NULL DEFAULT '',
    error          TEXT NOT NULL DEFAULT '',
    created_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls(tool, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tool_calls_actor ON tool_calls(actor, created_at DESC);

CREATE TABLE IF NOT EXISTS audit_log (
    id         TEXT PRIMARY KEY,
    ts         REAL NOT NULL,
    actor      TEXT NOT NULL DEFAULT '',
    action     TEXT NOT NULL,
    capability TEXT NOT NULL DEFAULT '',
    decision   TEXT NOT NULL DEFAULT '',
    detail     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts DESC);

-- Model registry: foundation models and personal fine-tunes, hot-swappable.
CREATE TABLE IF NOT EXISTS models (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    family         TEXT NOT NULL DEFAULT '',
    kind           TEXT NOT NULL DEFAULT 'foundation', -- foundation|finetune|lora|merged
    source         TEXT NOT NULL DEFAULT '',           -- hf repo id, local path, url
    revision       TEXT NOT NULL DEFAULT '',
    params         INTEGER NOT NULL DEFAULT 0,
    context_length INTEGER NOT NULL DEFAULT 0,
    quantization   TEXT NOT NULL DEFAULT '',
    license        TEXT NOT NULL DEFAULT '',
    sha256         TEXT NOT NULL DEFAULT '',
    path           TEXT NOT NULL DEFAULT '',
    size_bytes     INTEGER NOT NULL DEFAULT 0,
    active         INTEGER NOT NULL DEFAULT 0,
    base_model     TEXT NOT NULL DEFAULT '',
    eval_scores    TEXT NOT NULL DEFAULT '{}',
    created_at     REAL NOT NULL,
    metadata       TEXT NOT NULL DEFAULT '{}',
    UNIQUE(name, revision)
);
CREATE INDEX IF NOT EXISTS idx_models_active ON models(active, created_at DESC);

CREATE TABLE IF NOT EXISTS datasets (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'chat',   -- chat|alpaca|sharegpt|pretrain|eval
    path       TEXT NOT NULL DEFAULT '',
    rows       INTEGER NOT NULL DEFAULT 0,
    bytes      INTEGER NOT NULL DEFAULT 0,
    tokens     INTEGER NOT NULL DEFAULT 0,
    checksum   TEXT NOT NULL DEFAULT '',
    schema_    TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    metadata   TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name, created_at DESC);

CREATE TABLE IF NOT EXISTS dataset_items (
    id          TEXT PRIMARY KEY,
    dataset_id  TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL DEFAULT 'sample',
    payload     TEXT NOT NULL,
    tokens      INTEGER NOT NULL DEFAULT 0,
    quality     REAL NOT NULL DEFAULT 0.5,
    content_hash TEXT NOT NULL DEFAULT '',
    source      TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dataset_items_dataset ON dataset_items(dataset_id, quality DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_items_hash ON dataset_items(dataset_id, content_hash);

CREATE TABLE IF NOT EXISTS training_runs (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL DEFAULT '',
    base_model   TEXT NOT NULL DEFAULT '',
    output_model TEXT NOT NULL DEFAULT '',
    dataset_id   TEXT NOT NULL DEFAULT '',
    backend      TEXT NOT NULL DEFAULT 'native',  -- native|unsloth|llama_factory
    status       TEXT NOT NULL DEFAULT 'pending', -- pending|running|done|failed|promoted|rejected
    config       TEXT NOT NULL DEFAULT '{}',
    metrics      TEXT NOT NULL DEFAULT '{}',
    steps        INTEGER NOT NULL DEFAULT 0,
    epochs       REAL NOT NULL DEFAULT 0,
    output_path  TEXT NOT NULL DEFAULT '',
    gate_passed  INTEGER,
    created_at   REAL NOT NULL,
    started_at   REAL,
    finished_at  REAL,
    error        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status, created_at DESC);

-- Standing capability probes: the regression gate for self-improvement.
CREATE TABLE IF NOT EXISTS eval_probes (
    id         TEXT PRIMARY KEY,
    suite      TEXT NOT NULL DEFAULT 'core',
    prompt     TEXT NOT NULL,
    reference  TEXT NOT NULL DEFAULT '',
    metric     TEXT NOT NULL DEFAULT 'contains',
    weight     REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_probes_suite ON eval_probes(suite);

CREATE TABLE IF NOT EXISTS eval_results (
    id           TEXT PRIMARY KEY,
    model        TEXT NOT NULL,
    suite        TEXT NOT NULL DEFAULT 'core',
    run_id       TEXT NOT NULL DEFAULT '',
    score        REAL NOT NULL DEFAULT 0,
    details      TEXT NOT NULL DEFAULT '{}',
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_results_model ON eval_results(model, created_at DESC);
"""

# ── 0004: files, media, blobs, social ─────────────────────────────────────────

_V4 = """
-- Content-addressed binary store.
CREATE TABLE IF NOT EXISTS blobs (
    sha256     TEXT PRIMARY KEY,
    size       INTEGER NOT NULL,
    mime       TEXT NOT NULL DEFAULT '',
    compressed INTEGER NOT NULL DEFAULT 0,
    stored     INTEGER NOT NULL DEFAULT 0,   -- bytes actually on disk
    refcount   INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id         TEXT PRIMARY KEY,
    path       TEXT NOT NULL DEFAULT '',
    kind       TEXT NOT NULL DEFAULT 'text', -- text|pdf|docx|xlsx|odt|epub|html|image|audio|video
    title      TEXT NOT NULL DEFAULT '',
    blob_sha   TEXT NOT NULL DEFAULT '',
    chars      INTEGER NOT NULL DEFAULT 0,
    chunks     INTEGER NOT NULL DEFAULT 0,
    language   TEXT NOT NULL DEFAULT '',
    parsed_at  REAL,
    created_at REAL NOT NULL,
    metadata   TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_documents_kind ON documents(kind, created_at DESC);

CREATE TABLE IF NOT EXISTS media_items (
    id            TEXT PRIMARY KEY,
    url           TEXT NOT NULL,
    extractor     TEXT NOT NULL DEFAULT '',
    kind          TEXT NOT NULL DEFAULT 'video', -- video|audio|image|subtitle
    title         TEXT NOT NULL DEFAULT '',
    uploader      TEXT NOT NULL DEFAULT '',
    duration      REAL NOT NULL DEFAULT 0,
    path          TEXT NOT NULL DEFAULT '',
    blob_sha      TEXT NOT NULL DEFAULT '',
    size_bytes    INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'pending', -- pending|done|failed
    error         TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL,
    metadata      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_media_status ON media_items(status, created_at DESC);

CREATE TABLE IF NOT EXISTS web_pages (
    id          TEXT PRIMARY KEY,
    url         TEXT NOT NULL,
    status      INTEGER NOT NULL DEFAULT 0,
    title       TEXT NOT NULL DEFAULT '',
    text        TEXT NOT NULL DEFAULT '',
    chars       INTEGER NOT NULL DEFAULT 0,
    fetched_at  REAL NOT NULL,
    etag        TEXT NOT NULL DEFAULT '',
    metadata    TEXT NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_web_pages_url ON web_pages(url);

CREATE TABLE IF NOT EXISTS social_accounts (
    id           TEXT PRIMARY KEY,
    platform     TEXT NOT NULL,
    handle       TEXT NOT NULL,
    active       INTEGER NOT NULL DEFAULT 1,
    credentials  TEXT NOT NULL DEFAULT '',  -- reference into the secret store, never the secret
    limits       TEXT NOT NULL DEFAULT '{}',
    created_at   REAL NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}',
    UNIQUE(platform, handle)
);

CREATE TABLE IF NOT EXISTS social_posts (
    id           TEXT PRIMARY KEY,
    platform     TEXT NOT NULL,
    account_id   TEXT NOT NULL DEFAULT '',
    external_id  TEXT NOT NULL DEFAULT '',
    content      TEXT NOT NULL,
    media_paths  TEXT NOT NULL DEFAULT '[]',
    status       TEXT NOT NULL DEFAULT 'queued', -- queued|scheduled|posting|posted|failed|cancelled
    scheduled_at REAL,
    posted_at    REAL,
    reply_to     TEXT NOT NULL DEFAULT '',
    metrics      TEXT NOT NULL DEFAULT '{}',
    attempts     INTEGER NOT NULL DEFAULT 0,
    error        TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_social_posts_status ON social_posts(status, scheduled_at);
CREATE INDEX IF NOT EXISTS idx_social_posts_platform ON social_posts(platform, created_at DESC);
"""

# ── 0005: durable queue ───────────────────────────────────────────────────────

_V5 = """
-- Durable at-least-once work queue with leases and visibility timeouts.
CREATE TABLE IF NOT EXISTS work_queue (
    id           TEXT PRIMARY KEY,
    topic        TEXT NOT NULL,
    payload      TEXT NOT NULL DEFAULT '{}',
    priority     INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'ready', -- ready|leased|done|failed|dead
    attempts     INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5,
    available_at REAL NOT NULL DEFAULT 0,
    lease_until  REAL NOT NULL DEFAULT 0,
    lease_owner  TEXT NOT NULL DEFAULT '',
    result       TEXT NOT NULL DEFAULT '',
    error        TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_work_queue_ready ON work_queue(topic, status, priority DESC, available_at);
CREATE INDEX IF NOT EXISTS idx_work_queue_lease ON work_queue(status, lease_until);
"""

# ── 0006: full-text search ────────────────────────────────────────────────────


def _create_fts(db: object) -> None:
    """Create FTS5 indexes.

    Done in Python rather than raw SQL so a build without FTS5 degrades to no
    lexical search instead of failing to start.
    """
    import sqlite3
    from ..core.logging_setup import get_logger

    log = get_logger(__name__)
    definitions = {
        "memories_fts": "CREATE VIRTUAL TABLE memories_fts USING fts5(content, tokenize='porter unicode61')",
        "messages_fts": "CREATE VIRTUAL TABLE messages_fts USING fts5(content, tokenize='porter unicode61')",
        "documents_fts": "CREATE VIRTUAL TABLE documents_fts USING fts5(title, body, tokenize='porter unicode61')",
        "web_fts": "CREATE VIRTUAL TABLE web_fts USING fts5(title, body, tokenize='porter unicode61')",
    }
    for name, ddl in definitions.items():
        try:
            db.execute(ddl)  # type: ignore[attr-defined]
        except sqlite3.OperationalError as exc:
            log.warning("FTS5 unavailable (%s); lexical search disabled for %s", exc, name)


_V6_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_facts_created ON facts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);
"""

_V8_MISSING_TABLES = """
-- Migration 8: Add 10 missing tables that various modules query but were never created
CREATE TABLE IF NOT EXISTS relationship (
    id TEXT PRIMARY KEY,
    stage TEXT NOT NULL DEFAULT 'getting_to_know',
    stage_since REAL NOT NULL DEFAULT 0,
    trust REAL NOT NULL DEFAULT 0.5,
    milestones TEXT NOT NULL DEFAULT '[]',
    fights TEXT NOT NULL DEFAULT '[]',
    user_profile TEXT NOT NULL DEFAULT '{}',
    updated_at REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    objective TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    goal_id TEXT NOT NULL DEFAULT '',
    budget_wall REAL NOT NULL DEFAULT 0,
    budget_tokens INTEGER NOT NULL DEFAULT 0,
    progress REAL NOT NULL DEFAULT 0,
    report TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0,
    finished_at REAL NOT NULL DEFAULT 0,
    task_kind TEXT NOT NULL DEFAULT '',
    artifact TEXT NOT NULL DEFAULT '',
    verify_cmd TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);
CREATE INDEX IF NOT EXISTS idx_projects_goal ON projects(goal_id);
CREATE TABLE IF NOT EXISTS monitors (
    id TEXT PRIMARY KEY,
    target TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT '',
    watch TEXT NOT NULL DEFAULT '',
    interval_s REAL NOT NULL DEFAULT 3600,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL DEFAULT 0,
    last_ts REAL NOT NULL DEFAULT 0,
    webhook_url TEXT NOT NULL DEFAULT '',
    webhook_secret TEXT NOT NULL DEFAULT '',
    min_alert_gap_s REAL NOT NULL DEFAULT 0,
    auto_decode INTEGER NOT NULL DEFAULT 1,
    volatile INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_monitors_target ON monitors(target);
CREATE INDEX IF NOT EXISTS idx_monitors_enabled ON monitors(enabled);
CREATE TABLE IF NOT EXISTS kg_nodes (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL DEFAULT 'entity',
    properties TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL DEFAULT 0,
    last_access REAL NOT NULL DEFAULT 0,
    access_count INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_kg_label ON kg_nodes(label);
CREATE INDEX IF NOT EXISTS idx_kg_type ON kg_nodes(type);
CREATE TABLE IF NOT EXISTS kg_edges (
    id TEXT PRIMARY KEY,
    src TEXT NOT NULL DEFAULT '',
    dst TEXT NOT NULL DEFAULT '',
    relation TEXT NOT NULL DEFAULT '',
    weight REAL NOT NULL DEFAULT 1.0,
    properties TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_kg_src ON kg_edges(src);
CREATE INDEX IF NOT EXISTS idx_kg_dst ON kg_edges(dst);
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    platform TEXT NOT NULL DEFAULT '',
    chat_id TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    peer TEXT NOT NULL DEFAULT '',
    is_owner INTEGER NOT NULL DEFAULT 0,
    in_us INTEGER NOT NULL DEFAULT 0,
    last_active REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chats_platform ON chats(platform);
CREATE INDEX IF NOT EXISTS idx_chats_chat_id ON chats(chat_id);
CREATE TABLE IF NOT EXISTS research_log (
    id TEXT PRIMARY KEY,
    domain TEXT NOT NULL DEFAULT '',
    topic TEXT NOT NULL DEFAULT '',
    digest TEXT NOT NULL DEFAULT '',
    suggestion TEXT NOT NULL DEFAULT '',
    sources TEXT NOT NULL DEFAULT '[]',
    delivered INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0,
    score_detail TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_research_domain ON research_log(domain);
CREATE INDEX IF NOT EXISTS idx_research_created ON research_log(created_at);
CREATE TABLE IF NOT EXISTS cognition_log (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL DEFAULT 0,
    stages TEXT NOT NULL DEFAULT '{}',
    seconds REAL NOT NULL DEFAULT 0,
    note TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_cognition_ts ON cognition_log(ts);
CREATE TABLE IF NOT EXISTS corpus_words (
    word TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT '',
    ts REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_corpus_ts ON corpus_words(ts);
CREATE TABLE IF NOT EXISTS arena_knowledge (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    digest TEXT NOT NULL DEFAULT '',
    sources TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_arena_topic ON arena_knowledge(topic);
CREATE INDEX IF NOT EXISTS idx_arena_category ON arena_knowledge(category);
CREATE TABLE IF NOT EXISTS schedule_jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT '',
    spec TEXT NOT NULL DEFAULT '',
    payload_kind TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL DEFAULT '{}',
    enabled INTEGER NOT NULL DEFAULT 1,
    next_run REAL DEFAULT 0,
    last_run REAL DEFAULT 0,
    last_result TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_schedule_enabled ON schedule_jobs(enabled);
CREATE INDEX IF NOT EXISTS idx_schedule_next ON schedule_jobs(next_run);
"""


_V9_AGENT_TABLES = """
-- Migration 9: Tables for agents/goals.py and agents/skills.py
-- These use agent_goals/agent_goal_steps and agent_skills to avoid conflicts
-- with the wired-in goals/ and skills/ modules that use different schemas.

CREATE TABLE IF NOT EXISTS agent_goals (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    progress REAL NOT NULL DEFAULT 0.0,
    next_action TEXT NOT NULL DEFAULT '',
    strategy TEXT NOT NULL DEFAULT '',
    project_id TEXT NOT NULL DEFAULT '',
    priority INTEGER NOT NULL DEFAULT 0,
    depends_on TEXT NOT NULL DEFAULT '[]',
    heals INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0,
    finished_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_agent_goals_status ON agent_goals(status);
CREATE INDEX IF NOT EXISTS idx_agent_goals_priority ON agent_goals(priority);

CREATE TABLE IF NOT EXISTS agent_goal_steps (
    id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL,
    position INTEGER NOT NULL DEFAULT 0,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT NOT NULL DEFAULT '',
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_agent_goal_steps_goal ON agent_goal_steps(goal_id);

CREATE TABLE IF NOT EXISTS agent_skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT 'strategy',
    body TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    uses INTEGER NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0,
    last_used REAL NOT NULL DEFAULT 0,
    pruned INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_agent_skills_name ON agent_skills(name);
CREATE INDEX IF NOT EXISTS idx_agent_skills_kind ON agent_skills(kind);

CREATE TABLE IF NOT EXISTS mood_state (
    id TEXT PRIMARY KEY DEFAULT 'default',
    dims TEXT NOT NULL DEFAULT '{}',
    label TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS mood_history (
    id TEXT PRIMARY KEY,
    dims TEXT NOT NULL DEFAULT '{}',
    label TEXT NOT NULL DEFAULT '',
    recorded_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_mood_history_recorded ON mood_history(recorded_at);

CREATE TABLE IF NOT EXISTS goals (
    goal_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    progress REAL NOT NULL DEFAULT 0.0,
    priority INTEGER NOT NULL DEFAULT 0,
    target_date REAL,
    heals INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0,
    completed_at REAL DEFAULT 0,
    tags TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT '',
    workspace TEXT NOT NULL DEFAULT 'default'
);
CREATE INDEX IF NOT EXISTS idx_goals_user ON goals(user_id, status);

CREATE TABLE IF NOT EXISTS subgoals (
    subgoal_id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    is_completed INTEGER NOT NULL DEFAULT 0,
    completed_at REAL DEFAULT 0,
    sort_order INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
);
CREATE INDEX IF NOT EXISTS idx_subgoals_goal ON subgoals(goal_id);

CREATE TABLE IF NOT EXISTS skill_uses (
    id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL,
    task TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL DEFAULT '',
    used_at REAL NOT NULL DEFAULT 0,
    success INTEGER NOT NULL DEFAULT 0,
    context TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_skill_uses_skill ON skill_uses(skill_id);
CREATE INDEX IF NOT EXISTS idx_skill_uses_used ON skill_uses(used_at);

CREATE TABLE IF NOT EXISTS macros (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    steps TEXT NOT NULL DEFAULT '[]',
    runs INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0,
    run_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_macros_name ON macros(name);

CREATE TABLE IF NOT EXISTS failures (
    id TEXT PRIMARY KEY,
    error TEXT NOT NULL DEFAULT '',
    error_type TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    context TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL DEFAULT 0,
    resolved INTEGER NOT NULL DEFAULT 0,
    resolution TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_failures_type ON failures(error_type);
CREATE INDEX IF NOT EXISTS idx_failures_created ON failures(created_at);

-- Additional tables
CREATE TABLE IF NOT EXISTS proactive_log (
    id TEXT PRIMARY KEY,
    trigger_type TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL DEFAULT '',
    result TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_proactive_log_created ON proactive_log(created_at);

CREATE TABLE IF NOT EXISTS media_queue (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL DEFAULT '',
    path TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 0,
    position INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL DEFAULT 0,
    completed_at REAL DEFAULT 0,
    result TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_media_queue_status ON media_queue(status);

CREATE TABLE IF NOT EXISTS coding_log (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL DEFAULT '',
    filename TEXT NOT NULL DEFAULT '',
    code TEXT NOT NULL DEFAULT '',
    output TEXT NOT NULL DEFAULT '',
    success INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_coding_log_created ON coding_log(created_at);

CREATE TABLE IF NOT EXISTS side_chats (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    created_at REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0,
    messages TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_side_chats_status ON side_chats(status);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(type);

-- More missing tables
CREATE TABLE IF NOT EXISTS lessons (
    id TEXT PRIMARY KEY,
    pattern TEXT NOT NULL DEFAULT '',
    prevention TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_lessons_pattern ON lessons(pattern);

CREATE TABLE IF NOT EXISTS achievements (
    id TEXT PRIMARY KEY,
    achievement_id TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    unlocked_at REAL DEFAULT 0,
    user_id TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_achievements_user ON achievements(user_id);

CREATE TABLE IF NOT EXISTS game_sessions (
    id TEXT PRIMARY KEY,
    game_type TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    players TEXT NOT NULL DEFAULT '[]',
    state TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_game_sessions_status ON game_sessions(status);

CREATE TABLE IF NOT EXISTS search_log (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL DEFAULT '',
    results_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_search_log_created ON search_log(created_at);

CREATE TABLE IF NOT EXISTS notifications (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL DEFAULT '',
    message TEXT NOT NULL DEFAULT '',
    read INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_notifications_read ON notifications(read);

CREATE TABLE IF NOT EXISTS improvement_runs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    dataset TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    dimension TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL DEFAULT 0,
    completed_at REAL DEFAULT 0,
    result TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_improvement_runs_status ON improvement_runs(status);

CREATE TABLE IF NOT EXISTS search_leads (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    snippet TEXT NOT NULL DEFAULT '',
    score REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS goal_steps (
    id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL DEFAULT 0,
    completed_at REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_goal_steps_goal ON goal_steps(goal_id);

CREATE TABLE IF NOT EXISTS acceptance_runs (
    id TEXT PRIMARY KEY,
    build_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL DEFAULT 0,
    result TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS evolution_outcomes (
    id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL DEFAULT 0,
    fitness REAL NOT NULL DEFAULT 0,
    genome TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS directives (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL DEFAULT '',
    priority INTEGER NOT NULL DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_directives_active ON directives(active);

CREATE TABLE IF NOT EXISTS game_stats (
    player_key TEXT NOT NULL,
    game_name TEXT NOT NULL,
    games_played INTEGER NOT NULL DEFAULT 0,
    games_won INTEGER NOT NULL DEFAULT 0,
    total_score INTEGER NOT NULL DEFAULT 0,
    best_score INTEGER NOT NULL DEFAULT 0,
    total_time REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (player_key, game_name)
);
CREATE INDEX IF NOT EXISTS idx_game_stats_player ON game_stats(player_key);

"""

_V10_CONNECTORS = """
-- Proxy pool
CREATE TABLE IF NOT EXISTS proxies (
    ip TEXT NOT NULL,
    port INTEGER NOT NULL,
    protocol TEXT NOT NULL DEFAULT 'http',
    country TEXT DEFAULT '',
    anonymity TEXT DEFAULT '',
    source TEXT DEFAULT '',
    score REAL DEFAULT 0.0,
    working INTEGER DEFAULT 0,
    latency_ms REAL DEFAULT 0.0,
    last_check REAL DEFAULT 0.0,
    fail_streak INTEGER DEFAULT 0,
    uptime_pct REAL DEFAULT 0.0,
    PRIMARY KEY (ip, port)
);

CREATE TABLE IF NOT EXISTS proxy_checks (
    id TEXT PRIMARY KEY,
    ip TEXT NOT NULL,
    port INTEGER NOT NULL,
    working INTEGER NOT NULL,
    latency_ms REAL DEFAULT 0.0,
    checked_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_proxies_score ON proxies(score);
CREATE INDEX IF NOT EXISTS idx_proxies_working ON proxies(working);
CREATE INDEX IF NOT EXISTS idx_proxy_checks_ip ON proxy_checks(ip, port);

-- Virtual cards
CREATE TABLE IF NOT EXISTS cards (
    token TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    type TEXT NOT NULL,
    last_four TEXT,
    state TEXT,
    spend_limit INTEGER DEFAULT 0,
    memo TEXT,
    created_at REAL NOT NULL
);

-- Finance accounts
CREATE TABLE IF NOT EXISTS finance_accounts (
    account_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    institution TEXT,
    mask TEXT,
    linked_at REAL NOT NULL
);

-- Price tracking
CREATE TABLE IF NOT EXISTS price_watches (
    watch_id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    user_id TEXT NOT NULL,
    target_price REAL DEFAULT 0.0,
    current_price REAL DEFAULT 0.0,
    last_check REAL DEFAULT 0.0,
    triggered INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS price_snapshots (
    id TEXT PRIMARY KEY,
    listing_id TEXT NOT NULL,
    marketplace TEXT NOT NULL,
    url TEXT NOT NULL,
    price_ngn REAL NOT NULL,
    title TEXT NOT NULL,
    snapshot_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_price_watches_user ON price_watches(user_id);
CREATE INDEX IF NOT EXISTS idx_price_snapshots_listing ON price_snapshots(listing_id);
CREATE INDEX IF NOT EXISTS idx_price_snapshots_time ON price_snapshots(snapshot_at);

-- Connector credentials (labels only, secrets in vault)
CREATE TABLE IF NOT EXISTS connector_credentials (
    connector TEXT NOT NULL,
    label TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (connector, label)
);
"""

_V11_FIX_ACHIEVEMENTS = """
-- Fix achievements table schema to match code expectations
-- Old schema had: id, achievement_id, name, description, unlocked_at, user_id
-- New schema has: player_key, achievement_id, unlocked_at (composite PK)

DROP TABLE IF EXISTS achievements;

CREATE TABLE achievements (
    player_key TEXT NOT NULL,
    achievement_id TEXT NOT NULL,
    unlocked_at REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (player_key, achievement_id)
);
CREATE INDEX IF NOT EXISTS idx_achievements_player ON achievements(player_key);

-- Add leaderboards table (missing from migration 0008)
CREATE TABLE IF NOT EXISTS leaderboards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_name TEXT NOT NULL,
    player_key TEXT NOT NULL,
    player_name TEXT NOT NULL DEFAULT '',
    score INTEGER NOT NULL DEFAULT 0,
    played_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_leaderboards_game ON leaderboards(game_name);
CREATE INDEX IF NOT EXISTS idx_leaderboards_score ON leaderboards(game_name, score DESC);

-- Fix game_stats table (missing created_at column)
DROP TABLE IF EXISTS game_stats;

CREATE TABLE game_stats (
    player_key TEXT NOT NULL,
    game_name TEXT NOT NULL,
    games_played INTEGER NOT NULL DEFAULT 0,
    games_won INTEGER NOT NULL DEFAULT 0,
    total_score INTEGER NOT NULL DEFAULT 0,
    best_score INTEGER NOT NULL DEFAULT 0,
    total_time REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (player_key, game_name)
);
CREATE INDEX IF NOT EXISTS idx_game_stats_player ON game_stats(player_key);
"""

_V12_GOALS_PROJECT_ID = """
-- Add project_id column to goals table for goal-project cascade
ALTER TABLE goals ADD COLUMN project_id TEXT NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_goals_project ON goals(project_id);
"""

_V13_PLACEHOLDER = "-- Placeholder migration 13"
_V14_PLACEHOLDER = "-- Placeholder migration 14"
_V15_PLACEHOLDER = "-- Placeholder migration 15"
_V16_PLACEHOLDER = "-- Placeholder migration 16"
_V17_PLACEHOLDER = "-- Placeholder migration 17"
_V18_PLACEHOLDER = "-- Placeholder migration 18"

_V19_ARENA_BUILDS = """
-- Add arena_builds table for content arena
CREATE TABLE IF NOT EXISTS arena_builds (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    purpose TEXT NOT NULL DEFAULT '',
    topic TEXT NOT NULL DEFAULT '',
    files TEXT NOT NULL DEFAULT '[]',
    dir TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    syntax TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0,
    decided_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_arena_builds_status ON arena_builds(status);

-- Add arena_stream table for arena event streaming
CREATE TABLE IF NOT EXISTS arena_stream (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL DEFAULT '',
    ts REAL NOT NULL DEFAULT 0,
    payload TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_arena_stream_kind ON arena_stream(kind);

-- Add arena_knowledge table for arena research knowledge
CREATE TABLE IF NOT EXISTS arena_knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    digest TEXT NOT NULL DEFAULT '',
    sources TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_arena_knowledge_topic ON arena_knowledge(topic);
"""

_V20_CODING_LOG_COLUMNS = """
-- Add missing columns to coding_log table
ALTER TABLE coding_log ADD COLUMN attempt INTEGER NOT NULL DEFAULT 1;
ALTER TABLE coding_log ADD COLUMN exit_code INTEGER NOT NULL DEFAULT 0;
ALTER TABLE coding_log ADD COLUMN timed_out INTEGER NOT NULL DEFAULT 0;
ALTER TABLE coding_log ADD COLUMN stdout TEXT NOT NULL DEFAULT '';
ALTER TABLE coding_log ADD COLUMN stderr TEXT NOT NULL DEFAULT '';
"""

_V21_MISSING_COLUMNS = """
-- Add missing columns to various tables
ALTER TABLE macros ADD COLUMN last_run REAL NOT NULL DEFAULT 0;
ALTER TABLE macros ADD COLUMN last_result TEXT NOT NULL DEFAULT '';

-- Add missing columns to notifications table
ALTER TABLE notifications ADD COLUMN kind TEXT NOT NULL DEFAULT '';
ALTER TABLE notifications ADD COLUMN title TEXT NOT NULL DEFAULT '';
ALTER TABLE notifications ADD COLUMN body TEXT NOT NULL DEFAULT '';
ALTER TABLE notifications ADD COLUMN delivered INTEGER NOT NULL DEFAULT 0;
ALTER TABLE notifications ADD COLUMN pruned_at REAL NOT NULL DEFAULT 0;

-- Add missing columns to improvement_runs table
ALTER TABLE improvement_runs ADD COLUMN before_score REAL NOT NULL DEFAULT 0;
ALTER TABLE improvement_runs ADD COLUMN after_score REAL NOT NULL DEFAULT 0;
ALTER TABLE improvement_runs ADD COLUMN category TEXT NOT NULL DEFAULT '';

-- Add missing columns to goals table
ALTER TABLE goals ADD COLUMN depends_on TEXT NOT NULL DEFAULT '[]';
"""


_V22_FAILURES_COLUMNS = """
-- Add missing columns to failures table
ALTER TABLE failures ADD COLUMN times_seen INTEGER NOT NULL DEFAULT 1;
ALTER TABLE failures ADD COLUMN root_cause TEXT NOT NULL DEFAULT '';
"""

_V23_LESSONS_COLUMNS = """
-- Add missing columns to lessons table
ALTER TABLE lessons ADD COLUMN times_seen INTEGER NOT NULL DEFAULT 1;
ALTER TABLE lessons ADD COLUMN updated_at REAL NOT NULL DEFAULT 0;
"""

_V24_ACCEPTANCE_RUNS_COLUMNS = """
-- Add missing columns to acceptance_runs table
ALTER TABLE acceptance_runs ADD COLUMN project_id TEXT NOT NULL DEFAULT '';
ALTER TABLE acceptance_runs ADD COLUMN step_id TEXT NOT NULL DEFAULT '';
ALTER TABLE acceptance_runs ADD COLUMN cmd TEXT NOT NULL DEFAULT '';
ALTER TABLE acceptance_runs ADD COLUMN output_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE acceptance_runs ADD COLUMN ok INTEGER NOT NULL DEFAULT 0;
ALTER TABLE acceptance_runs ADD COLUMN ts REAL NOT NULL DEFAULT 0;
"""

_V25_MISSING_TABLES_AND_COLUMNS = """
-- Create missing skills table
CREATE TABLE IF NOT EXISTS skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    code TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0
);

-- Create missing devon_memory table
CREATE TABLE IF NOT EXISTS devon_memory (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL DEFAULT '',
    key TEXT NOT NULL DEFAULT '',
    value TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL DEFAULT 0
);

-- Add missing columns to media_queue table
ALTER TABLE media_queue ADD COLUMN title TEXT NOT NULL DEFAULT '';
ALTER TABLE media_queue ADD COLUMN artist TEXT NOT NULL DEFAULT '';
ALTER TABLE media_queue ADD COLUMN duration REAL NOT NULL DEFAULT 0;

-- Add missing columns to failures table
ALTER TABLE failures ADD COLUMN family TEXT NOT NULL DEFAULT '';
ALTER TABLE failures ADD COLUMN fingerprint TEXT NOT NULL DEFAULT '';
"""
_V26_MISSING_COLUMNS = """
-- Add missing columns to media_queue
ALTER TABLE media_queue ADD COLUMN kind TEXT NOT NULL DEFAULT '';

-- Add missing columns to failures
ALTER TABLE failures ADD COLUMN lesson TEXT NOT NULL DEFAULT '';
"""


_V27_MEDIA_QUEUE_COLUMNS = """
-- Add missing columns to media_queue
ALTER TABLE media_queue ADD COLUMN added_at REAL NOT NULL DEFAULT 0;
ALTER TABLE media_queue ADD COLUMN played_at REAL NOT NULL DEFAULT 0;
ALTER TABLE media_queue ADD COLUMN skipped INTEGER NOT NULL DEFAULT 0;
"""


_V28_MISSING_COLUMNS_V2 = """
-- Add missing columns to failures table
ALTER TABLE failures ADD COLUMN ts REAL NOT NULL DEFAULT 0;
"""


_V29_MONITOR_COLUMNS = """
-- Add missing columns to monitors table
ALTER TABLE monitors ADD COLUMN last_size INTEGER NOT NULL DEFAULT 0;
ALTER TABLE monitors ADD COLUMN last_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE monitors ADD COLUMN error_streak INTEGER NOT NULL DEFAULT 0;
"""


_V30_MONITOR_COLUMNS_V2 = """
-- Add more missing columns to monitors table
ALTER TABLE monitors ADD COLUMN last_change_ts REAL NOT NULL DEFAULT 0;
ALTER TABLE monitors ADD COLUMN last_ok INTEGER NOT NULL DEFAULT 1;
"""


_V31_MORE_MISSING_COLUMNS = """
-- Add missing columns to monitors table
ALTER TABLE monitors ADD COLUMN last_content TEXT NOT NULL DEFAULT '';

-- Add missing columns to improvement_runs
ALTER TABLE improvement_runs ADD COLUMN delta REAL NOT NULL DEFAULT 0;
"""


_V32_MONITOR_ALERT_COLUMNS = """
-- Add missing columns to monitors table
ALTER TABLE monitors ADD COLUMN last_alert_ts REAL NOT NULL DEFAULT 0;
"""


_V33_STATUS_AND_OTHER_COLUMNS = """
-- Add missing columns to improvement_runs
ALTER TABLE improvement_runs ADD COLUMN proposal_id TEXT NOT NULL DEFAULT '';
"""


_V69_GAME_GEAR = """
-- Arena gear: durable equipment. One row per owned piece — wear only
-- ever decreases durability; breakage keeps the row so it can be
-- repaired. Never silently deleted.
CREATE TABLE IF NOT EXISTS game_gear (
    id             TEXT PRIMARY KEY,
    player_key     TEXT NOT NULL,
    slug           TEXT NOT NULL,
    durability     INTEGER NOT NULL,
    max_durability INTEGER NOT NULL,
    equipped       INTEGER NOT NULL DEFAULT 0,
    created_at     REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_game_gear_player ON game_gear(player_key);
"""


_V71_TRIAL_ASSIST_RUNS = """
-- Trial assist runs: durable background-signup state. A restart can
-- never silently swallow a run — rows not in a terminal state are
-- marked interrupted + reported on the next boot.
CREATE TABLE IF NOT EXISTS trial_assist_runs (
    run_id     TEXT PRIMARY KEY,
    platform   TEXT NOT NULL DEFAULT '',
    chat_key   TEXT NOT NULL DEFAULT '',
    started    REAL NOT NULL DEFAULT 0,
    state      TEXT NOT NULL DEFAULT '',
    note       TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_trial_assist_runs_state
    ON trial_assist_runs(state);
"""


_V72_TRIAL_SMS_WATCHES = """
-- Trial SMS watches: durable background verification-code watch state.
-- /trial sms code returns immediately and polls in the background; a
-- restart must not silently swallow the watch. Rows still in
-- 'watching' when the process died are resumed (deadline not passed)
-- or closed as timed-out (deadline passed) on the next boot, and the
-- owner is told either way. number_info pins the exact temp number the
-- watch was started for, so a newer grabbed number can't hijack it.
CREATE TABLE IF NOT EXISTS trial_sms_watches (
    watch_id    TEXT PRIMARY KEY,
    number      TEXT NOT NULL DEFAULT '',
    number_info TEXT NOT NULL DEFAULT '',
    chat_key    TEXT NOT NULL DEFAULT '',
    started     REAL NOT NULL DEFAULT 0,
    deadline    REAL NOT NULL DEFAULT 0,
    timeout     REAL NOT NULL DEFAULT 0,
    state       TEXT NOT NULL DEFAULT '',
    code        TEXT NOT NULL DEFAULT '',
    note        TEXT NOT NULL DEFAULT '',
    updated_at  REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_trial_sms_watches_state
    ON trial_sms_watches(state);
"""


_V73_GAME_SKILLS = """
-- Learned battle skills: martial arts for the arena. One row per
-- learned skill per player; learning is permanent (skills are never
-- lost, only gear breaks).
CREATE TABLE IF NOT EXISTS game_skills (
    id          TEXT PRIMARY KEY,
    player_key  TEXT NOT NULL,
    slug        TEXT NOT NULL,
    learned_at  REAL NOT NULL DEFAULT 0,
    UNIQUE(player_key, slug)
);
CREATE INDEX IF NOT EXISTS idx_game_skills_player
    ON game_skills(player_key);
"""

_V74_GAME_TITLES = """
-- Earnable titles: flair the player wears next to their name.
-- One row per unlocked title per player; exactly one active at a time.
CREATE TABLE IF NOT EXISTS game_titles (
    id          TEXT PRIMARY KEY,
    player_key  TEXT NOT NULL,
    title_id    TEXT NOT NULL,
    unlocked_at REAL NOT NULL DEFAULT 0,
    active      INTEGER NOT NULL DEFAULT 0,
    UNIQUE(player_key, title_id)
);
CREATE INDEX IF NOT EXISTS idx_game_titles_player
    ON game_titles(player_key);
"""


def _apply_game_attributes_rename(db: object) -> None:
    """Fix the game_stats table-name collision.

    ``nomorals/games/stats.py`` (RPG attributes) and the achievements
    per-game stats both used the table name ``game_stats`` with
    incompatible schemas — whichever CREATE ran first broke the other
    module.  The RPG table is now ``game_attributes``; this migration
    moves any existing RPG rows over and ensures the per-game
    ``game_stats`` table exists with its canonical schema.
    """
    ex = db.execute
    # Does game_stats currently hold the RPG schema? (strength column)
    cols = set()
    try:
        rows = ex("PRAGMA table_info(game_stats)").fetchall()
        cols = {str(r[1]) for r in rows}
    except Exception as exc:
        from ..core.logging_setup import get_logger
        get_logger(__name__).debug("game_stats schema probe failed: %s", exc)
    if "strength" in cols:
        # RPG table won the race — move it aside.
        ex("ALTER TABLE game_stats RENAME TO game_attributes")
    # Canonical per-game stats table (achievements/mastery).
    ex(
        "CREATE TABLE IF NOT EXISTS game_stats ("
        "player_key TEXT NOT NULL, game_name TEXT NOT NULL, "
        "games_played INTEGER NOT NULL DEFAULT 0, "
        "games_won INTEGER NOT NULL DEFAULT 0, "
        "total_score INTEGER NOT NULL DEFAULT 0, "
        "best_score INTEGER NOT NULL DEFAULT 0, "
        "total_time REAL NOT NULL DEFAULT 0, "
        "created_at REAL NOT NULL DEFAULT 0, "
        "updated_at REAL NOT NULL DEFAULT 0, "
        "PRIMARY KEY (player_key, game_name))"
    )
    ex(
        "CREATE INDEX IF NOT EXISTS idx_game_stats_player "
        "ON game_stats(player_key)"
    )


def _apply_game_identity_alias_merge(db: object) -> None:
    """Merge ``telegram-bot:`` player rows into ``telegram:``.

    ``Player.from_sender`` now canonicalizes ``telegram-bot`` to
    ``telegram`` for game keys (both endpoints see the same Telegram
    user IDs), so one human has one profile.  Pre-existing rows under
    the old ``telegram-bot:<sender>`` keys are merged into their
    canonical ``telegram:<sender>`` twins — or renamed outright when
    no twin exists — across every per-player game table.  Nothing is
    dropped: counters are summed, bests take the max, JSON ledgers
    union with the canonical side winning conflicts.
    """
    import json as _json

    ex = db.execute

    def tables() -> set[str]:
        try:
            rows = ex(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        except Exception:
            return set()
        return {str(r[0]) for r in rows}

    have = tables()
    if "game_players" not in have:
        return
    # alias senders can surface in any per-player table (gear earned
    # before a profile row was ever read is possible in theory) —
    # collect from all of them.
    alias_keys: set[str] = set()
    for t in ("game_players", "game_gear", "game_skills",
              "game_attributes", "game_titles", "game_stats"):
        if t not in have:
            continue
        try:
            for (k,) in ex(
                    f"SELECT DISTINCT player_key FROM {t} "
                    "WHERE player_key LIKE 'telegram-bot:%'").fetchall():
                alias_keys.add(str(k))
        except Exception:
            pass
    if not alias_keys:
        return

    def cols(table: str) -> list[str]:
        try:
            return [str(r[1]) for r in
                    ex(f"PRAGMA table_info({table})").fetchall()]
        except Exception:
            return []

    player_col_list = cols("game_players")
    player_cols = set(player_col_list)

    def q1(sql: str, args: tuple = ()) -> object:
        try:
            rows = ex(sql, args).fetchall()
        except Exception:
            return None
        return rows[0] if rows else None

    def qall(sql: str, args: tuple = ()) -> list:
        try:
            return ex(sql, args).fetchall()
        except Exception:
            return []

    for alias_key in sorted(alias_keys):
        sender = alias_key.split(":", 1)[1] if ":" in alias_key else ""
        if not sender:
            continue
        canon = f"telegram:{sender}"
        if alias_key == canon:
            continue

        alias = q1("SELECT * FROM game_players WHERE player_key = ?",
                   (alias_key,))
        a = dict(zip(player_col_list, alias)) if (player_col_list and alias) else {}
        c_row = q1("SELECT * FROM game_players WHERE player_key = ?",
                   (canon,))
        c = dict(zip(player_col_list, c_row)) if (player_col_list and c_row) else None

        def rename(table: str) -> None:
            if table in have:
                try:
                    ex("UPDATE " + table + " SET player_key = ? "
                       "WHERE player_key = ?", (canon, alias_key))
                except Exception:
                    pass

        if c is None:
            # No twin — pure rename across every per-player table.
            for t in ("game_players", "game_gear", "game_skills",
                      "game_attributes", "game_titles", "game_stats"):
                rename(t)
            continue

        # ── both exist: merge game_players ──────────────────────────
        def num(key: str) -> int:
            try:
                return int(a.get(key, 0) or 0) + int(c.get(key, 0) or 0)
            except Exception:
                return 0

        def jmerge(key: str) -> str:
            try:
                da = _json.loads(a.get(key) or "{}")
            except Exception:
                da = {}
            try:
                dc = _json.loads(c.get(key) or "{}")
            except Exception:
                dc = {}
            if not isinstance(da, dict):
                da = {}
            if not isinstance(dc, dict):
                dc = {}
            merged = dict(da)
            merged.update(dc)  # canonical side wins conflicts
            return _json.dumps(merged)

        a_streak = int(a.get("streak", 0) or 0)
        c_streak = int(c.get("streak", 0) or 0)
        streak = a_streak if abs(a_streak) > abs(c_streak) else c_streak
        display = c.get("display") or a.get("display") or ""
        created = min(float(a.get("created_at", 0) or 0),
                      float(c.get("created_at", 0) or 0))
        updated = max(float(a.get("updated_at", 0) or 0),
                      float(c.get("updated_at", 0) or 0))
        sets = {
            "coins": num("coins"), "points": num("points"),
            "wins": num("wins"), "losses": num("losses"),
            "draws": num("draws"), "games_played": num("games_played"),
            "streak": streak,
            "best_streak": max(int(a.get("best_streak", 0) or 0),
                               int(c.get("best_streak", 0) or 0)),
            "display": display, "created_at": created,
            "updated_at": updated,
            "per_game": jmerge("per_game"), "items": jmerge("items"),
        }
        if "xp" in player_cols:
            sets["xp"] = num("xp")
        set_sql = ", ".join(f"{k} = ?" for k in sets)
        try:
            ex(f"UPDATE game_players SET {set_sql} WHERE player_key = ?",
               (*sets.values(), canon))
            ex("DELETE FROM game_players WHERE player_key = ?",
               (alias_key,))
        except Exception:
            pass

        # ── gear: instance ids are globally unique — plain rename ────
        rename("game_gear")

        # ── skills: drop alias dupes, then rename ────────────────────
        if "game_skills" in have:
            try:
                canon_slugs = {str(r[0]) for r in qall(
                    "SELECT slug FROM game_skills WHERE player_key = ?",
                    (canon,))}
                for (slug,) in qall(
                        "SELECT slug FROM game_skills WHERE player_key = ?",
                        (alias_key,)):
                    if str(slug) in canon_slugs:
                        ex("DELETE FROM game_skills WHERE player_key = ? "
                           "AND slug = ?", (alias_key, str(slug)))
                rename("game_skills")
            except Exception:
                pass

        # ── attributes: UNIQUE(player_key) — merge, keep max ─────────
        if "game_attributes" in have:
            try:
                ar = q1("SELECT strength, stamina, mana, intelligence, "
                        "unspent, level_applied FROM game_attributes "
                        "WHERE player_key = ?", (alias_key,))
                cr = q1("SELECT strength, stamina, mana, intelligence, "
                        "unspent, level_applied FROM game_attributes "
                        "WHERE player_key = ?", (canon,))
                if ar and cr:
                    merged = [max(int(ar[i] or 0), int(cr[i] or 0))
                              for i in range(6)]
                    ex("UPDATE game_attributes SET strength = ?, "
                       "stamina = ?, mana = ?, intelligence = ?, "
                       "unspent = ?, level_applied = ? "
                       "WHERE player_key = ?",
                       (*merged, canon))
                    ex("DELETE FROM game_attributes WHERE player_key = ?",
                       (alias_key,))
                else:
                    rename("game_attributes")
            except Exception:
                pass

        # ── titles: UNIQUE(player_key, title_id) — dedupe, rename ────
        if "game_titles" in have:
            try:
                canon_titles = {str(r[0]) for r in qall(
                    "SELECT title_id FROM game_titles WHERE player_key = ?",
                    (canon,))}
                alias_active = [str(r[0]) for r in qall(
                    "SELECT title_id FROM game_titles WHERE player_key = ? "
                    "AND active = 1", (alias_key,))]
                canon_has_active = bool(q1(
                    "SELECT 1 FROM game_titles WHERE player_key = ? "
                    "AND active = 1", (canon,)))
                for (tid,) in qall(
                        "SELECT title_id FROM game_titles WHERE player_key = ?",
                        (alias_key,)):
                    if str(tid) in canon_titles:
                        ex("DELETE FROM game_titles WHERE player_key = ? "
                           "AND title_id = ?", (alias_key, str(tid)))
                rename("game_titles")
                if alias_active and not canon_has_active:
                    ex("UPDATE game_titles SET active = 1 WHERE player_key = ? "
                       "AND title_id = ?", (canon, alias_active[0]))
                    ex("UPDATE game_titles SET active = 0 WHERE player_key = ? "
                       "AND title_id != ?", (canon, alias_active[0]))
            except Exception:
                pass

        # ── per-game stats: PK(player_key, game_name) — sum, max best ─
        if "game_stats" in have:
            try:
                for row in qall(
                        "SELECT game_name, games_played, games_won, "
                        "total_score, best_score, total_time, created_at, "
                        "updated_at FROM game_stats WHERE player_key = ?",
                        (alias_key,)):
                    gname = str(row[0])
                    cr2 = q1(
                        "SELECT games_played, games_won, total_score, "
                        "best_score, total_time, created_at, updated_at "
                        "FROM game_stats WHERE player_key = ? AND "
                        "game_name = ?", (canon, gname))
                    if cr2:
                        ex("UPDATE game_stats SET games_played = ?, "
                           "games_won = ?, total_score = ?, best_score = ?, "
                           "total_time = ?, created_at = ?, updated_at = ? "
                           "WHERE player_key = ? AND game_name = ?",
                           (int(row[1] or 0) + int(cr2[0] or 0),
                            int(row[2] or 0) + int(cr2[1] or 0),
                            int(row[3] or 0) + int(cr2[2] or 0),
                            max(int(row[4] or 0), int(cr2[3] or 0)),
                            float(row[5] or 0) + float(cr2[4] or 0),
                            min(float(row[6] or 0), float(cr2[5] or 0)),
                            max(float(row[7] or 0), float(cr2[6] or 0)),
                            canon, gname))
                        ex("DELETE FROM game_stats WHERE player_key = ? "
                           "AND game_name = ?", (alias_key, gname))
                rename("game_stats")
            except Exception:
                pass


_V67_PLAYER_LIBRARY = """
-- Player library: named playlists, play history, favorites.
CREATE TABLE IF NOT EXISTS media_playlists (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS media_playlist_items (
    id TEXT PRIMARY KEY,
    playlist_id TEXT NOT NULL,
    path TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    artist TEXT NOT NULL DEFAULT '',
    album TEXT NOT NULL DEFAULT '',
    duration REAL NOT NULL DEFAULT 0,
    kind TEXT NOT NULL DEFAULT 'file',
    position INTEGER NOT NULL DEFAULT 0,
    added_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_playlist_items_pl
    ON media_playlist_items(playlist_id, position);
CREATE TABLE IF NOT EXISTS media_history (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    artist TEXT NOT NULL DEFAULT '',
    album TEXT NOT NULL DEFAULT '',
    duration REAL NOT NULL DEFAULT 0,
    played_at REAL NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_media_history_played
    ON media_history(played_at);
CREATE INDEX IF NOT EXISTS idx_media_history_path
    ON media_history(path);
CREATE TABLE IF NOT EXISTS media_favorites (
    path TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    artist TEXT NOT NULL DEFAULT '',
    album TEXT NOT NULL DEFAULT '',
    duration REAL NOT NULL DEFAULT 0,
    liked_at REAL NOT NULL DEFAULT 0
);
"""


_V34_MORE_MISSING_COLUMNS_V2 = """
-- Add missing columns to improvement_runs
ALTER TABLE improvement_runs ADD COLUMN action TEXT NOT NULL DEFAULT '';
"""


_V35_IMPROVEMENT_RUNS_COLUMNS = """
-- Add missing columns to improvement_runs
ALTER TABLE improvement_runs ADD COLUMN rationale TEXT NOT NULL DEFAULT '';
"""


_V36_IMPROVEMENT_RUNS_COLUMNS_V2 = """
-- Add missing columns to improvement_runs
ALTER TABLE improvement_runs ADD COLUMN edit_summary TEXT NOT NULL DEFAULT '';
"""


_V37_NOTIFICATIONS_COLUMNS = """
-- Notifications table already has all required columns
"""


_V38_STATUS_COLUMNS_V2 = """
-- Status columns already exist in missions and agent_tasks tables
"""


_V39_NOTIFICATIONS_COLUMNS_V2 = """
-- Notifications table already has pruned_at, chat_key, and category columns
"""


_V40_STATUS_COLUMNS_V3 = """
-- All columns already exist in missions, agent_tasks, and notifications tables
"""


_V41_FINAL_MISSING_COLUMNS = """
-- All columns already exist in missions, agent_tasks, and notifications tables
"""


_V42_MISSING_STATUS_COLUMNS = """
-- All columns already exist in missions, agent_tasks, and notifications tables
"""


_V46_GAME_TABLES = """
CREATE TABLE IF NOT EXISTS game_players (
    player_key    TEXT PRIMARY KEY,
    platform      TEXT NOT NULL DEFAULT '',
    display       TEXT NOT NULL DEFAULT '',
    coins         INTEGER NOT NULL DEFAULT 0,
    points        INTEGER NOT NULL DEFAULT 0,
    wins          INTEGER NOT NULL DEFAULT 0,
    losses        INTEGER NOT NULL DEFAULT 0,
    draws         INTEGER NOT NULL DEFAULT 0,
    streak        INTEGER NOT NULL DEFAULT 0,
    best_streak   INTEGER NOT NULL DEFAULT 0,
    games_played  INTEGER NOT NULL DEFAULT 0,
    per_game      TEXT NOT NULL DEFAULT '{}',
    items         TEXT NOT NULL DEFAULT '{}',
    created_at    REAL NOT NULL DEFAULT 0,
    updated_at    REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_game_players_points ON game_players(points DESC);

CREATE TABLE IF NOT EXISTS game_wallet (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    player_key TEXT NOT NULL,
    amount     INTEGER NOT NULL DEFAULT 0,
    reason     TEXT NOT NULL DEFAULT '',
    at         REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_game_wallet_player ON game_wallet(player_key, at DESC);

CREATE TABLE IF NOT EXISTS game_rooms (
    id         TEXT PRIMARY KEY,
    game       TEXT NOT NULL DEFAULT '',
    chat_key   TEXT NOT NULL DEFAULT '',
    platform   TEXT NOT NULL DEFAULT '',
    kind       TEXT NOT NULL DEFAULT '',
    players    TEXT NOT NULL DEFAULT '[]',
    turn       INTEGER NOT NULL DEFAULT 0,
    state      TEXT NOT NULL DEFAULT '{}',
    status     TEXT NOT NULL DEFAULT 'active',
    started_at REAL NOT NULL DEFAULT 0,
    seed       INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_game_rooms_chat ON game_rooms(chat_key, status, updated_at DESC);
"""

def _apply_skills_lessons_columns(db: object) -> None:
    """Skills pruning bookkeeping (wave 50) and durable lesson columns used
    by the failure-analysis learning loop.

    Guarded per column: some builds already carry these on the base table,
    and a raw ALTER would abort the migration on the first duplicate.
    """
    wanted = {
        "agent_skills": {
            "pruned_at": "REAL NOT NULL DEFAULT 0",
            "pruned_reason": "TEXT NOT NULL DEFAULT ''",
        },
        "lessons": {
            "category": "TEXT NOT NULL DEFAULT ''",
            "root_cause": "TEXT NOT NULL DEFAULT ''",
            "lesson": "TEXT NOT NULL DEFAULT ''",
            "fix": "TEXT NOT NULL DEFAULT ''",
            "prevention": "TEXT NOT NULL DEFAULT ''",
            "skill_id": "TEXT NOT NULL DEFAULT ''",
        },
    }
    for table, columns in wanted.items():
        try:
            have = {r["name"] for r in db.query(
                f"PRAGMA table_info({table})")}  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — table missing entirely: skip
            continue
        for column, decl in columns.items():
            if column in have:
                continue
            db.execute(  # type: ignore[attr-defined]
                f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _apply_evolution_outcome_columns(db: object) -> None:
    """Closed-loop evolution accounting (wave 76): per-cycle outcome rows.

    ``evolution_outcomes`` was originally a genetic-programming table
    (generation/fitness/genome); the self-improvement loop needs per-
    proposal outcome accounting instead — instruction, applied/reverted,
    test and line-count deltas, commit/tag.  Guarded per column so builds
    that already carry them (or the raw-ALTER duplicate-column abort) do
    not stall migration.
    """
    wanted = {
        "evolution_outcomes": {
            "proposal_id": "TEXT NOT NULL DEFAULT ''",
            "instruction": "TEXT NOT NULL DEFAULT ''",
            "source": "TEXT NOT NULL DEFAULT ''",
            "applied": "INTEGER NOT NULL DEFAULT 0",
            "reverted": "INTEGER NOT NULL DEFAULT 0",
            "reason": "TEXT NOT NULL DEFAULT ''",
            "tests_before": "INTEGER NOT NULL DEFAULT 0",
            "tests_after": "INTEGER NOT NULL DEFAULT 0",
            "lines_before": "INTEGER NOT NULL DEFAULT 0",
            "lines_after": "INTEGER NOT NULL DEFAULT 0",
            "commit_id": "TEXT NOT NULL DEFAULT ''",
            "tag": "TEXT NOT NULL DEFAULT ''",
            "ts": "REAL NOT NULL DEFAULT 0",
        },
    }
    for table, columns in wanted.items():
        try:
            have = {r["name"] for r in db.query(
                f"PRAGMA table_info({table})")}  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — table missing entirely: skip
            continue
        for column, decl in columns.items():
            if column in have:
                continue
            db.execute(  # type: ignore[attr-defined]
                f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

def _apply_proactive_mood_columns(db: object) -> None:
    """Autonomy proposals + mood journaling columns (partner runtime).

    ``proactive_log`` was drafted as a trigger/result ledger; the autonomy
    loop proposes *sends* — kind, target chat, content, approval status and
    timestamps.  ``mood_history`` journals under ``ts``/``event`` while the
    base table named them ``recorded_at`` with no event column.  Both writes
    were silently failing into their best-effort excepts; add the columns.
    """
    wanted = {
        "proactive_log": {
            "kind": "TEXT NOT NULL DEFAULT ''",
            "platform": "TEXT NOT NULL DEFAULT ''",
            "chat_id": "TEXT NOT NULL DEFAULT ''",
            "content": "TEXT NOT NULL DEFAULT ''",
            "status": "TEXT NOT NULL DEFAULT 'pending'",
            "reason": "TEXT NOT NULL DEFAULT ''",
            "decided_at": "REAL",
            "acted_at": "REAL",
        },
        "mood_history": {
            "ts": "REAL NOT NULL DEFAULT 0",
            "event": "TEXT NOT NULL DEFAULT ''",
        },
    }
    for table, columns in wanted.items():
        try:
            have = {r["name"] for r in db.query(
                f"PRAGMA table_info({table})")}  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — table missing entirely: skip
            continue
        for column, decl in columns.items():
            if column in have:
                continue
            db.execute(  # type: ignore[attr-defined]
                f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _apply_memory_tags_origin(db: object) -> None:
    """Memory v2: curated memories gained tags and an origin pointer
    (which chat the fact came from).  Guarded per column."""
    wanted = {
        "memories": {
            "tags": "TEXT NOT NULL DEFAULT ''",
            "origin": "TEXT NOT NULL DEFAULT ''",
        },
    }
    for table, columns in wanted.items():
        try:
            have = {r["name"] for r in db.query(
                f"PRAGMA table_info({table})")}  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — table missing entirely: skip
            continue
        for column, decl in columns.items():
            if column in have:
                continue
            db.execute(  # type: ignore[attr-defined]
                f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _apply_search_engine_schema(db: object) -> None:
    """The search engine's durable half: page cache, richer journal, lead
    notes. Idempotent by construction (IF NOT EXISTS + guarded columns),
    because the engine treated all of it as best-effort and the tables were
    simply never created."""
    db.execute(  # type: ignore[attr-defined]
        """CREATE TABLE IF NOT EXISTS page_cache (
            url        TEXT PRIMARY KEY,
            title      TEXT NOT NULL DEFAULT '',
            text       TEXT NOT NULL DEFAULT '',
            fetched_at REAL NOT NULL DEFAULT 0
        )"""
    )
    wanted = {
        "search_log": {
            "mode": "TEXT NOT NULL DEFAULT 'quick'",
            "results": "TEXT NOT NULL DEFAULT '[]'",
            "summary": "TEXT NOT NULL DEFAULT ''",
            "seconds": "REAL NOT NULL DEFAULT 0",
        },
        "search_leads": {
            "note": "TEXT NOT NULL DEFAULT ''",
            "domain": "TEXT NOT NULL DEFAULT ''",
        },
    }
    for table, columns in wanted.items():
        try:
            have = {r["name"] for r in db.query(
                f"PRAGMA table_info({table})")}  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — table missing entirely: skip
            continue
        for column, decl in columns.items():
            if column in have:
                continue
            db.execute(  # type: ignore[attr-defined]
                f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _apply_skills_table_unify(db: object) -> None:
    """Wave 77: ONE skills table.

    Two stores had grown up: the plain ``skills`` table (id, name,
    description, code, timestamps) and the SkillLibrary's ``agent_skills``
    twin with the usage/pruning columns.  The library now lives directly on
    ``skills`` (external tooling and schedulers update its rows by name),
    so extend ``skills`` to the full schema, copy any twin rows over, and
    index what the library queries.
    """
    wanted = {
        "kind": "TEXT NOT NULL DEFAULT 'strategy'",
        "body": "TEXT NOT NULL DEFAULT ''",
        "tags": "TEXT NOT NULL DEFAULT ''",
        "source": "TEXT NOT NULL DEFAULT ''",
        "uses": "INTEGER NOT NULL DEFAULT 0",
        "success_count": "INTEGER NOT NULL DEFAULT 0",
        "failure_count": "INTEGER NOT NULL DEFAULT 0",
        "last_used": "REAL NOT NULL DEFAULT 0",
        "version": "INTEGER NOT NULL DEFAULT 1",
        "pruned": "INTEGER NOT NULL DEFAULT 0",
        "pruned_at": "REAL NOT NULL DEFAULT 0",
        "pruned_reason": "TEXT NOT NULL DEFAULT ''",
    }
    try:  # the plain table always exists from migration 25; be paranoid
        have = {r["name"] for r in db.query("PRAGMA table_info(skills)")}  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - no table at all: create it first
        db.execute(  # type: ignore[attr-defined]
            "CREATE TABLE IF NOT EXISTS skills ("
            "id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '', "
            "description TEXT NOT NULL DEFAULT '', code TEXT NOT NULL DEFAULT '', "
            "created_at REAL NOT NULL DEFAULT 0, updated_at REAL NOT NULL DEFAULT 0)"
        )
        have = set()
    for col, ddl in wanted.items():
        if col not in have:
            db.execute(f"ALTER TABLE skills ADD COLUMN {col} {ddl}")  # type: ignore[attr-defined]
    try:  # carry the twin's rows forward (idempotent by PRIMARY KEY)
        keys = sorted(set(wanted) | {
            "id", "name", "description", "created_at", "updated_at"})
        cols = ", ".join(keys)
        marks = ", ".join("?" for _ in keys)
        for r in db.query("SELECT * FROM agent_skills"):  # type: ignore[attr-defined]
            vals = tuple(r[k] if k in r.keys() else None for k in keys)
            db.execute(  # type: ignore[attr-defined]
                f"INSERT OR IGNORE INTO skills ({cols}) VALUES ({marks})", vals)
    except Exception:  # noqa: BLE001 - no twin table: nothing to carry
        pass
    db.execute("CREATE INDEX IF NOT EXISTS idx_skills_name "  # type: ignore[attr-defined]
               "ON skills(name)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_skills_kind "  # type: ignore[attr-defined]
               "ON skills(kind)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_skills_pruned "  # type: ignore[attr-defined]
               "ON skills(pruned, updated_at DESC)")


def _apply_companion_schema(db: object) -> None:
    """Expansion-era writes the base schema only half-covers (wave 42+).

    The games relay, directives engine, news feed runner, image ledger and
    Devon's memory box all INSERT columns their tables were never given.
    Guarded per column: databases that already carry them skip the ALTER.
    """
    wanted = {
        "directives": {
            "status": "TEXT NOT NULL DEFAULT 'pending'",
            "result": "TEXT NOT NULL DEFAULT ''",
            "error": "TEXT NOT NULL DEFAULT ''",
            "finished_at": "REAL NOT NULL DEFAULT 0",
            "updated_at": "REAL NOT NULL DEFAULT 0",
        },
        "game_sessions": {
            "game": "TEXT NOT NULL DEFAULT ''",
            "chat_key": "TEXT NOT NULL DEFAULT ''",
        },
        "devon_memory": {
            "ts": "REAL NOT NULL DEFAULT 0",
            "chat_key": "TEXT NOT NULL DEFAULT ''",
            "run_id": "TEXT NOT NULL DEFAULT ''",
            "task": "TEXT NOT NULL DEFAULT ''",
            "step": "INTEGER NOT NULL DEFAULT 0",
            "tool": "TEXT NOT NULL DEFAULT ''",
            "args": "TEXT NOT NULL DEFAULT ''",
            "observation": "TEXT NOT NULL DEFAULT ''",
            "digest": "TEXT NOT NULL DEFAULT ''",
            "status": "TEXT NOT NULL DEFAULT 'step'",
        },
    }
    for table, cols in wanted.items():
        try:
            have = {r["name"] for r in db.query(f"PRAGMA table_info({table})")}  # noqa: UP031
        except Exception:  # noqa: BLE001 - table absent: nothing to alter
            continue
        for col, ddl in cols.items():
            if col not in have:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")

    # full-text feeds + the face of every photo she has ever seen
    db.execute(
        "CREATE TABLE IF NOT EXISTS news_items ("
        "id TEXT PRIMARY KEY, source TEXT NOT NULL DEFAULT '', "
        "title TEXT NOT NULL DEFAULT '', url TEXT NOT NULL DEFAULT '', "
        "summary TEXT NOT NULL DEFAULT '', published TEXT NOT NULL DEFAULT '', "
        "created_at REAL NOT NULL DEFAULT 0, UNIQUE(url))"
    )
    db.execute("CREATE INDEX IF NOT EXISTS idx_news_items_created ON news_items(created_at)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS image_index ("
        "hash TEXT PRIMARY KEY, path TEXT NOT NULL DEFAULT '', "
        "size INTEGER NOT NULL DEFAULT 0, mime TEXT NOT NULL DEFAULT '', "
        "first_seen REAL NOT NULL DEFAULT 0, last_seen REAL NOT NULL DEFAULT 0, "
        "seen_in TEXT NOT NULL DEFAULT '[]')"
    )
    try:
        db.execute("CREATE INDEX IF NOT EXISTS idx_game_sessions_chat "
                   "ON game_sessions(chat_key, status)")
    except Exception:  # noqa: BLE001
        pass



def _apply_cipher_vault(db: object) -> None:
    """The named-secrets vault table (mirrors CipherAgent._VAULT_DDL)."""
    db.execute(
        "CREATE TABLE IF NOT EXISTS cipher_vault ("
        "name TEXT PRIMARY KEY, blob TEXT NOT NULL, "
        "key_scheme TEXT NOT NULL DEFAULT 'pass', "
        "created_at REAL NOT NULL DEFAULT 0, updated_at REAL NOT NULL DEFAULT 0)"
    )


def _apply_game_relay_tables(db: object) -> None:
    """Game relay persistence: invites and DM-to-DM relay rooms.

    The relay previously kept everything in memory (lost on restart) and
    never reaped expired invites. These tables let a relay survive a
    reboot; the relay module also creates them IF NOT EXISTS defensively,
    so this migration is belt-and-braces for fresh installs.
    """
    db.execute_statements(  # type: ignore[attr-defined]
        """
        CREATE TABLE IF NOT EXISTS game_invites (
            code                TEXT PRIMARY KEY,
            game_name           TEXT NOT NULL,
            from_chat           TEXT NOT NULL,
            from_player_key     TEXT NOT NULL,
            from_player_name    TEXT NOT NULL DEFAULT '',
            from_player_platform TEXT NOT NULL DEFAULT '',
            to_label            TEXT NOT NULL DEFAULT '',
            created_at          REAL NOT NULL DEFAULT 0,
            expires_at          REAL NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS game_relays (
            room_id             TEXT PRIMARY KEY,
            game_name           TEXT NOT NULL,
            chat_a              TEXT NOT NULL,
            chat_b              TEXT NOT NULL,
            player_a_key        TEXT NOT NULL,
            player_a_name       TEXT NOT NULL DEFAULT '',
            player_a_platform   TEXT NOT NULL DEFAULT '',
            player_b_key        TEXT NOT NULL,
            player_b_name       TEXT NOT NULL DEFAULT '',
            player_b_platform   TEXT NOT NULL DEFAULT '',
            active              INTEGER NOT NULL DEFAULT 1,
            last_activity       REAL NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_game_invites_expires
            ON game_invites(expires_at);
        CREATE INDEX IF NOT EXISTS idx_game_relays_chats
            ON game_relays(chat_a, chat_b);
        """
    )


def _apply_research_organs(db: object) -> None:
    """Wave C research organs: upgrade-proposal approval queue and the
    dynamic lexicon store.

    upgrade_proposals — research findings become concrete module-upgrade
    proposals with patch plans. The owner approves or denies each one
    explicitly (deny requires a reason); approval dispatches to the
    evolution/coding stack, never straight to the tree.
    lexicon_terms / lexicon_versions — scored vocabulary acquired from
    research (personas, partner phrasing, domain terminology), versioned
    per consuming module so updates are reviewable and reloadable
    without a restart (the store is the database, not the code).
    """
    db.execute_statements(  # type: ignore[attr-defined]
        """
        CREATE TABLE IF NOT EXISTS upgrade_proposals (
            id                  TEXT PRIMARY KEY,
            title               TEXT NOT NULL DEFAULT '',
            rationale           TEXT NOT NULL DEFAULT '',
            patch_plan          TEXT NOT NULL DEFAULT '{}',
            files               TEXT NOT NULL DEFAULT '[]',
            tests               TEXT NOT NULL DEFAULT '[]',
            claim_ids           TEXT NOT NULL DEFAULT '[]',
            status              TEXT NOT NULL DEFAULT 'proposed',
            reason              TEXT NOT NULL DEFAULT '',
            created_at          REAL NOT NULL DEFAULT 0,
            decided_at          REAL NOT NULL DEFAULT 0,
            decided_by          TEXT NOT NULL DEFAULT '',
            evolution_proposal_id TEXT NOT NULL DEFAULT '',
            applied_result      TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_upgrade_proposals_status
            ON upgrade_proposals(status, created_at);
        CREATE TABLE IF NOT EXISTS lexicon_terms (
            id          TEXT PRIMARY KEY,
            term        TEXT NOT NULL,
            category    TEXT NOT NULL DEFAULT 'general',
            module      TEXT NOT NULL DEFAULT '',
            score       REAL NOT NULL DEFAULT 0,
            source      TEXT NOT NULL DEFAULT '',
            version     INTEGER NOT NULL DEFAULT 1,
            status      TEXT NOT NULL DEFAULT 'active',
            created_at  REAL NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_lexicon_terms_module
            ON lexicon_terms(module, category, status);
        CREATE INDEX IF NOT EXISTS idx_lexicon_terms_term
            ON lexicon_terms(term, module);
        CREATE TABLE IF NOT EXISTS lexicon_versions (
            module      TEXT PRIMARY KEY,
            version     INTEGER NOT NULL DEFAULT 0,
            term_count  INTEGER NOT NULL DEFAULT 0,
            updated_at  REAL NOT NULL DEFAULT 0
        );
        """
    )


def _apply_room_tables(db: object) -> None:
    """Prompt 05 (project rooms): per-goal workspace index.

    The room's directory (rooms/<slug>/) is the source of truth for
    human state (ROOM.md carries the rehydration contract); these tables
    are the queryable index — listing, linked-goal/project lookups, and
    cross-room links.  RoomManager also creates them IF NOT EXISTS
    defensively, so this migration is belt-and-braces for fresh installs.
    """
    db.execute_statements(  # type: ignore[attr-defined]
        """
        CREATE TABLE IF NOT EXISTS rooms (
            id              TEXT PRIMARY KEY,
            slug            TEXT NOT NULL UNIQUE,
            kind            TEXT NOT NULL DEFAULT 'ad_hoc',
            linked_id       TEXT NOT NULL DEFAULT '',
            title           TEXT NOT NULL DEFAULT '',
            status          TEXT NOT NULL DEFAULT 'active',
            created_at      REAL NOT NULL DEFAULT 0,
            last_entered_at REAL NOT NULL DEFAULT 0,
            updated_at      REAL NOT NULL DEFAULT 0,
            state_json      TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_rooms_linked
            ON rooms(kind, linked_id);
        CREATE INDEX IF NOT EXISTS idx_rooms_status ON rooms(status);
        CREATE TABLE IF NOT EXISTS room_links (
            slug_a     TEXT NOT NULL,
            slug_b     TEXT NOT NULL,
            created_at REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (slug_a, slug_b)
        );
        """
    )


def _apply_briefing_tables(db: object) -> None:
    """Prompt 04 (morning briefing): stored briefings for ``nm briefing
    today`` + cross-session follow-ups, and per-section engagement counts
    (views/followups/pinned — counts only, no content).

    Tables are created IF NOT EXISTS so re-runs are safe.
    """
    db.execute_statements(  # type: ignore[attr-defined]
        """
        CREATE TABLE IF NOT EXISTS briefings (
            id            TEXT PRIMARY KEY,
            date          TEXT NOT NULL,          -- YYYY-MM-DD (owner tz)
            sections_json TEXT NOT NULL DEFAULT '[]',
            generated_at  REAL NOT NULL DEFAULT 0,
            generation_ms REAL NOT NULL DEFAULT 0,
            late          INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_briefings_date
            ON briefings (date, generated_at DESC);
        CREATE TABLE IF NOT EXISTS briefing_engagement (
            section   TEXT PRIMARY KEY,  -- alerts|calendar|markets|repos|news|rooms|devon
            views     INTEGER NOT NULL DEFAULT 0,
            followups INTEGER NOT NULL DEFAULT 0,
            pinned    INTEGER NOT NULL DEFAULT 0
        );
        """
    )


def _apply_memory_archive(db: object) -> None:
    """Prompt 11 (persona depth): 30-day undo archive for memory curation.

    ``forget()`` / ``forget_below()`` / dedup merges copy the record here
    BEFORE deletion — nothing vanishes surprisingly.  ``MemoryCurator.restore()``
    can bring a record back within 30 days.  IF NOT EXISTS: re-runs are safe.
    """
    db.execute_statements(  # type: ignore[attr-defined]
        """
        CREATE TABLE IF NOT EXISTS memory_archive (
            id            TEXT PRIMARY KEY,
            kind          TEXT NOT NULL DEFAULT '',
            content       TEXT NOT NULL DEFAULT '',
            importance    REAL NOT NULL DEFAULT 0.5,
            metadata      TEXT NOT NULL DEFAULT '{}',
            archived_at   REAL NOT NULL DEFAULT 0,
            reason        TEXT NOT NULL DEFAULT '',
            superseded_by TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_memory_archive_at
            ON memory_archive (archived_at DESC);
        """
    )


def _apply_watchers_tables(db: object) -> None:
    """Prompt 03 (watchers): the general watcher model, check history, and
    the alert audit log.

    A NEW ``watchers`` table rather than an extension of ``monitors``:
    ``monitors`` is narrowly shaped for MonitorAgent's URL/file/page model
    (webhook_url, volatile, auto_decode, content-vs-size); six watcher kinds
    plus structured conditions, severity, channels, cooldowns and flap state
    would leave a dozen nullable columns and confuse both systems.

    Tables are created IF NOT EXISTS so re-runs and partial builds are safe.
    """
    db.execute_statements(  # type: ignore[attr-defined]
        """
        CREATE TABLE IF NOT EXISTS watchers (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL DEFAULT '',
            kind        TEXT NOT NULL DEFAULT '',   -- url|file|price|repo|keyword|condition
            target      TEXT NOT NULL DEFAULT '{}', -- JSON kind-specific params
            condition   TEXT NOT NULL DEFAULT '{}', -- JSON structured predicate
            interval_s  REAL NOT NULL DEFAULT 3600,
            cooldown_s  REAL NOT NULL DEFAULT 0,    -- 0 = interval_s * 6
            severity    TEXT NOT NULL DEFAULT 'info',
            channels    TEXT NOT NULL DEFAULT '[]', -- JSON subset of notifier channels
            quiet_hours TEXT NOT NULL DEFAULT '',   -- JSON {start,end,tz} or ''
            expires_at  REAL NOT NULL DEFAULT 0,
            state       TEXT NOT NULL DEFAULT 'active',
            last_check  REAL NOT NULL DEFAULT 0,
            last_value  TEXT NOT NULL DEFAULT '',   -- JSON baseline value
            last_alert_ts REAL NOT NULL DEFAULT 0,
            last_alert_severity TEXT NOT NULL DEFAULT '',
            error_streak INTEGER NOT NULL DEFAULT 0,
            last_changed INTEGER NOT NULL DEFAULT 0,
            flip_times  TEXT NOT NULL DEFAULT '[]', -- JSON flap timestamps
            created_at  REAL NOT NULL DEFAULT 0,
            updated_at  REAL NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_watchers_state
            ON watchers(state);
        CREATE INDEX IF NOT EXISTS idx_watchers_due
            ON watchers(state, last_check);
        -- last N check results per watcher (pruned to 50 by the store)
        CREATE TABLE IF NOT EXISTS watcher_checks (
            id          TEXT PRIMARY KEY,
            watcher_id  TEXT NOT NULL,
            checked_at  REAL NOT NULL DEFAULT 0,
            changed     INTEGER NOT NULL DEFAULT 0,
            summary     TEXT NOT NULL DEFAULT '',
            value_hash  TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_watcher_checks_watcher
            ON watcher_checks(watcher_id, checked_at);
        -- audit log: every sent/held/digested/suppressed alert
        CREATE TABLE IF NOT EXISTS watcher_alerts (
            id          TEXT PRIMARY KEY,
            watcher_id  TEXT NOT NULL,
            severity    TEXT NOT NULL DEFAULT '',
            channel     TEXT NOT NULL DEFAULT '',
            status      TEXT NOT NULL DEFAULT '',   -- sent|held|digested|suppressed
            title       TEXT NOT NULL DEFAULT '',
            body        TEXT NOT NULL DEFAULT '',
            created_at  REAL NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_watcher_alerts_watcher
            ON watcher_alerts(watcher_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_watcher_alerts_status
            ON watcher_alerts(status, created_at);
        -- sweeper-level state (digest bookkeeping)
        CREATE TABLE IF NOT EXISTS watcher_state (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL DEFAULT ''
        );
        """
    )


def _apply_trading_tables(db: object) -> None:
    """Prompt 07 (FinancialExpert): paper-trading sessions, the append-only
    live/paper trade journal, and live-unlock grants.

    Tables are created IF NOT EXISTS so re-runs and partial builds are safe.
    """
    db.execute_statements(  # type: ignore[attr-defined]
        """
        CREATE TABLE IF NOT EXISTS paper_sessions (
            id          TEXT PRIMARY KEY,
            symbol      TEXT NOT NULL,
            market      TEXT NOT NULL DEFAULT 'crypto',
            capital     REAL NOT NULL,
            status      TEXT NOT NULL DEFAULT 'open',   -- open | closed
            state_json  TEXT NOT NULL DEFAULT '{}',     -- broker state
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trade_journal (
            id          TEXT PRIMARY KEY,
            ts          REAL NOT NULL,
            kind        TEXT NOT NULL,                  -- paper | live
            symbol      TEXT NOT NULL DEFAULT '',
            side        TEXT NOT NULL DEFAULT '',       -- buy | sell | kill
            size        REAL NOT NULL DEFAULT 0,
            price       REAL NOT NULL DEFAULT 0,
            order_id    TEXT NOT NULL DEFAULT '',
            reason      TEXT NOT NULL DEFAULT '',
            meta_json   TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_trade_journal_ts
            ON trade_journal(ts DESC);
        CREATE TABLE IF NOT EXISTS live_unlocks (
            id          TEXT PRIMARY KEY,
            granted_at  REAL NOT NULL,
            expires_at  REAL NOT NULL,
            note        TEXT NOT NULL DEFAULT ''
        );
        """
    )


def _apply_self_improvement_v2(db: object) -> None:
    """Self-improvement engine v2 (prompt 01): lesson usefulness scoring,
    skill self-rewrite records, canary rollouts, and surfacing ledger.

    Guarded per column like the earlier _apply_* migrations so re-runs and
    partial builds never stall on a duplicate column.
    """
    wanted = {
        "lessons": {
            "times_surfaced": "INTEGER NOT NULL DEFAULT 0",
            "times_prevented": "INTEGER NOT NULL DEFAULT 0",
            "fingerprint": "TEXT NOT NULL DEFAULT ''",
            "demoted": "INTEGER NOT NULL DEFAULT 0",
        },
        "failures": {
            "fingerprint": "TEXT NOT NULL DEFAULT ''",
            "skill": "TEXT NOT NULL DEFAULT ''",
        },
    }
    for table, columns in wanted.items():
        try:
            have = {r["name"] for r in db.query(  # type: ignore[attr-defined]
                f"PRAGMA table_info({table})")}
        except Exception:  # noqa: BLE001 — table missing entirely: skip
            continue
        for column, decl in columns.items():
            if column in have:
                continue
            db.execute(  # type: ignore[attr-defined]
                f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    db.execute(  # type: ignore[attr-defined]
        "CREATE TABLE IF NOT EXISTS lessons_archive ("
        "id TEXT PRIMARY KEY, source TEXT NOT NULL DEFAULT '', "
        "category TEXT NOT NULL DEFAULT '', root_cause TEXT NOT NULL DEFAULT '', "
        "lesson TEXT NOT NULL DEFAULT '', fix TEXT NOT NULL DEFAULT '', "
        "prevention TEXT NOT NULL DEFAULT '', skill_id TEXT NOT NULL DEFAULT '', "
        "times_seen INTEGER NOT NULL DEFAULT 1, "
        "times_surfaced INTEGER NOT NULL DEFAULT 0, "
        "times_prevented INTEGER NOT NULL DEFAULT 0, "
        "fingerprint TEXT NOT NULL DEFAULT '', "
        "archived_at REAL NOT NULL DEFAULT 0, archive_reason TEXT NOT NULL DEFAULT '')"
    )
    db.execute(  # type: ignore[attr-defined]
        "CREATE TABLE IF NOT EXISTS lesson_surfacings ("
        "id TEXT PRIMARY KEY, lesson_id TEXT NOT NULL DEFAULT '', "
        "fingerprint TEXT NOT NULL DEFAULT '', surfaced_at REAL NOT NULL DEFAULT 0, "
        "evaluated INTEGER NOT NULL DEFAULT 0)"
    )
    db.execute(  # type: ignore[attr-defined]
        "CREATE INDEX IF NOT EXISTS idx_surfacings_lesson "
        "ON lesson_surfacings(lesson_id, evaluated)"
    )
    db.execute(  # type: ignore[attr-defined]
        "CREATE TABLE IF NOT EXISTS skill_edits ("
        "id TEXT PRIMARY KEY, skill_name TEXT NOT NULL DEFAULT '', "
        "target_kind TEXT NOT NULL DEFAULT '', target_ref TEXT NOT NULL DEFAULT '', "
        "before_hash TEXT NOT NULL DEFAULT '', after_hash TEXT NOT NULL DEFAULT '', "
        "diff TEXT NOT NULL DEFAULT '', triggering_failures TEXT NOT NULL DEFAULT '[]', "
        "gate_results TEXT NOT NULL DEFAULT '{}', mode TEXT NOT NULL DEFAULT '', "
        "status TEXT NOT NULL DEFAULT 'proposed', fingerprint TEXT NOT NULL DEFAULT '', "
        "created_at REAL NOT NULL DEFAULT 0, decided_at REAL NOT NULL DEFAULT 0)"
    )
    db.execute(  # type: ignore[attr-defined]
        "CREATE INDEX IF NOT EXISTS idx_skill_edits_skill "
        "ON skill_edits(skill_name, created_at DESC)"
    )
    db.execute(  # type: ignore[attr-defined]
        "CREATE TABLE IF NOT EXISTS skill_versions ("
        "id TEXT PRIMARY KEY, skill_name TEXT NOT NULL DEFAULT '', "
        "version_hash TEXT NOT NULL DEFAULT '', parent_hash TEXT NOT NULL DEFAULT '', "
        "body TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT '', "
        "created_at REAL NOT NULL DEFAULT 0)"
    )
    db.execute(  # type: ignore[attr-defined]
        "CREATE INDEX IF NOT EXISTS idx_skill_versions_skill "
        "ON skill_versions(skill_name, created_at DESC)"
    )
    db.execute(  # type: ignore[attr-defined]
        "CREATE TABLE IF NOT EXISTS canary_runs ("
        "id TEXT PRIMARY KEY, skill_name TEXT NOT NULL DEFAULT '', "
        "canary_hash TEXT NOT NULL DEFAULT '', baseline_hash TEXT NOT NULL DEFAULT '', "
        "fraction REAL NOT NULL DEFAULT 0.2, status TEXT NOT NULL DEFAULT 'running', "
        "decision TEXT NOT NULL DEFAULT '', decision_detail TEXT NOT NULL DEFAULT '{}', "
        "started_at REAL NOT NULL DEFAULT 0, decided_at REAL NOT NULL DEFAULT 0)"
    )
    db.execute(  # type: ignore[attr-defined]
        "CREATE TABLE IF NOT EXISTS canary_observations ("
        "id TEXT PRIMARY KEY, canary_id TEXT NOT NULL DEFAULT '', "
        "version_hash TEXT NOT NULL DEFAULT '', success INTEGER NOT NULL DEFAULT 0, "
        "observed_at REAL NOT NULL DEFAULT 0)"
    )


# ── 0060: artifacts ──────────────────────────────────────────────────────────

_V60_ARTIFACTS = """
-- Evolve the primitive artifacts table (migration 8: inline content) into
-- first-class artifacts: content-addressed through the blob store, with
-- creator/mission/task links and provenance.  The old `name`/`content`
-- columns stay for history; nothing in prod code wrote them.
ALTER TABLE artifacts ADD COLUMN content_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE artifacts ADD COLUMN size INTEGER NOT NULL DEFAULT 0;
ALTER TABLE artifacts ADD COLUMN mime TEXT NOT NULL DEFAULT '';
ALTER TABLE artifacts ADD COLUMN creator TEXT NOT NULL DEFAULT '';
ALTER TABLE artifacts ADD COLUMN mission_id TEXT NOT NULL DEFAULT '';
ALTER TABLE artifacts ADD COLUMN task_id TEXT NOT NULL DEFAULT '';
ALTER TABLE artifacts ADD COLUMN provenance TEXT NOT NULL DEFAULT '{}';
CREATE INDEX IF NOT EXISTS idx_artifacts_mission ON artifacts(mission_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_task ON artifacts(task_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_hash ON artifacts(content_hash);
"""

_V61_ARENA_SCORES = """
-- Arena build scoring (the scoring worker records per-build verdicts here)
CREATE TABLE IF NOT EXISTS arena_scores (id TEXT PRIMARY KEY, ts REAL NOT NULL, topic TEXT NOT NULL, category TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'code', research_usefulness REAL, build_compiled INTEGER, tests_passed INTEGER, edit_precision REAL, latency_s REAL, notes TEXT DEFAULT '');
CREATE INDEX IF NOT EXISTS idx_arena_scores_category ON arena_scores(category);
CREATE INDEX IF NOT EXISTS idx_arena_scores_ts ON arena_scores(ts);
"""


_V63_BENCHMARK_RUNS = """
-- K3 scoreboard: persisted benchmark runs (the "beat K3" scoreboard).
-- One row per run; per-dimension detail lives in ``dimensions`` as JSON so
-- the schema never has to change when suites are added. ``mode`` is either
-- 'model-scored' (a live LLM was measured) or 'harness-self-test' (the
-- harness verified its own mechanics with no model) — the two are never
-- mixed in comparisons.
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id          TEXT PRIMARY KEY,
    ts          REAL NOT NULL,
    scoreboard  TEXT NOT NULL DEFAULT 'k3',
    suite       TEXT NOT NULL DEFAULT 'all',
    mode        TEXT NOT NULL DEFAULT 'harness-self-test',
    provider    TEXT NOT NULL DEFAULT '',
    measurable  INTEGER NOT NULL DEFAULT 1,
    overall     REAL,
    passed      INTEGER NOT NULL DEFAULT 0,
    total       INTEGER NOT NULL DEFAULT 0,
    dimensions  TEXT NOT NULL DEFAULT '{}',
    seconds     REAL NOT NULL DEFAULT 0,
    notes       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_ts ON benchmark_runs(ts);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_suite ON benchmark_runs(suite);
"""

_V65_COREMIND_TELEMETRY = """
-- Core Mind router telemetry (wave E): persisted across processes so
-- ``nm mind`` can show live router behavior. One row per key:
--   route:<kind>   — router decision counts (kind = intent route/organ,
--                    e.g. 'research_swarm', 'coding', 'brain')
--   model_consults — model-check calls (kept in addition to per-route counts)
--   model_timeouts — model-check deadline hits
--   last_plan_error — JSON {error, route, at} of the most recent plan failure
CREATE TABLE IF NOT EXISTS coremind_telemetry (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL DEFAULT '',
    updated_at  REAL NOT NULL DEFAULT 0
);
"""


def _apply_research_loop_tables(db: object) -> None:
    """Wave E always-on research loop: per-cycle run history.

    research_loop_runs — one row per cycle (tick or on-demand run):
    gate-skip reason when deferred (quiet hours / feature off /
    proactive master off), topics covered, findings/claims/proposal
    counts, the queued proposal ids, whether the owner was notified.
    The ``nm`` status surface reads the latest row.
    """
    db.execute_statements(  # type: ignore[attr-defined]
        """
        CREATE TABLE IF NOT EXISTS research_loop_runs (
            id                  TEXT PRIMARY KEY,
            started_at          REAL NOT NULL DEFAULT 0,
            finished_at         REAL NOT NULL DEFAULT 0,
            ok                  INTEGER NOT NULL DEFAULT 0,
            skipped_reason      TEXT NOT NULL DEFAULT '',
            topics              TEXT NOT NULL DEFAULT '[]',
            findings_count      INTEGER NOT NULL DEFAULT 0,
            claims_count        INTEGER NOT NULL DEFAULT 0,
            proposals_created   INTEGER NOT NULL DEFAULT 0,
            proposal_ids        TEXT NOT NULL DEFAULT '[]',
            notified            INTEGER NOT NULL DEFAULT 0,
            error               TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_research_loop_runs_started
            ON research_loop_runs(started_at DESC);
        """
    )


def _apply_research_loop_g2(db: object) -> None:
    """Wave G2: rate-limit noise + honest usefulness signals.

    ``research_loop_runs`` gains per-cycle counters — ``proposals_deduped``
    (tickets dropped as cross-cycle duplicates), ``proposals_capped``
    (qualifying tickets dropped by the per-cycle proposal cap) — and a
    ``signals`` JSON snapshot of the aggregate usefulness counts
    (proposed/approved/denied/ignored/applied/tests_passed) taken when the
    cycle finished. The live aggregate is recomputed from
    ``upgrade_proposals`` by ``usefulness_signals()``; the snapshot is
    history.
    """
    db.execute_statements(  # type: ignore[attr-defined]
        """
        ALTER TABLE research_loop_runs
            ADD COLUMN proposals_deduped INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE research_loop_runs
            ADD COLUMN proposals_capped INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE research_loop_runs
            ADD COLUMN signals TEXT NOT NULL DEFAULT '{}';
        """
    )


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "core_state", sql=_V1),
    Migration(2, "agents_tasks_missions", sql=_V2),
    Migration(3, "tools_models_training", sql=_V3),
    Migration(4, "files_media_social", sql=_V4),
    Migration(5, "work_queue", sql=_V5),
    Migration(6, "full_text_search", fn=_create_fts, down="DROP TABLE IF EXISTS web_fts;"),
    Migration(7, "secondary_indexes", sql=_V6_INDEXES),
    Migration(8, "missing_tables", sql=_V8_MISSING_TABLES),
    Migration(9, "agent_tables", sql=_V9_AGENT_TABLES),
    Migration(10, "connectors", sql=_V10_CONNECTORS),
    Migration(11, "fix_achievements", sql=_V11_FIX_ACHIEVEMENTS),
    Migration(12, "goals_project_id", sql=_V12_GOALS_PROJECT_ID),
    Migration(13, "placeholder_13", sql=_V13_PLACEHOLDER),
    Migration(14, "placeholder_14", sql=_V14_PLACEHOLDER),
    Migration(15, "placeholder_15", sql=_V15_PLACEHOLDER),
    Migration(16, "placeholder_16", sql=_V16_PLACEHOLDER),
    Migration(17, "placeholder_17", sql=_V17_PLACEHOLDER),
    Migration(18, "placeholder_18", sql=_V18_PLACEHOLDER),
    Migration(19, "arena_builds", sql=_V19_ARENA_BUILDS),
    Migration(20, "coding_log_columns", sql=_V20_CODING_LOG_COLUMNS),
    Migration(21, "missing_columns", sql=_V21_MISSING_COLUMNS),
    Migration(22, "failures_columns", sql=_V22_FAILURES_COLUMNS),
    Migration(23, "lessons_columns", sql=_V23_LESSONS_COLUMNS),
    Migration(24, "acceptance_runs_columns", sql=_V24_ACCEPTANCE_RUNS_COLUMNS),
    Migration(25, "missing_tables_and_columns", sql=_V25_MISSING_TABLES_AND_COLUMNS),
    Migration(26, "missing_columns_v26", sql=_V26_MISSING_COLUMNS),
    Migration(27, "media_queue_columns", sql=_V27_MEDIA_QUEUE_COLUMNS),
    Migration(28, "missing_columns_v2", sql=_V28_MISSING_COLUMNS_V2),
    Migration(29, "monitor_columns", sql=_V29_MONITOR_COLUMNS),
    Migration(30, "monitor_columns_v2", sql=_V30_MONITOR_COLUMNS_V2),
    Migration(31, "more_missing_columns", sql=_V31_MORE_MISSING_COLUMNS),
    Migration(32, "monitor_alert_columns", sql=_V32_MONITOR_ALERT_COLUMNS),
    Migration(33, "status_and_other_columns", sql=_V33_STATUS_AND_OTHER_COLUMNS),
    Migration(34, "more_missing_columns_v2", sql=_V34_MORE_MISSING_COLUMNS_V2),
    Migration(35, "improvement_runs_columns", sql=_V35_IMPROVEMENT_RUNS_COLUMNS),
    Migration(36, "improvement_runs_columns_v2", sql=_V36_IMPROVEMENT_RUNS_COLUMNS_V2),
    Migration(37, "notifications_columns", sql=_V37_NOTIFICATIONS_COLUMNS),
    Migration(38, "status_columns_v2", sql=_V38_STATUS_COLUMNS_V2),
    Migration(39, "notifications_columns_v2", sql=_V39_NOTIFICATIONS_COLUMNS_V2),
    Migration(40, "status_columns_v3", sql=_V40_STATUS_COLUMNS_V3),
    Migration(41, "final_missing_columns", sql=_V41_FINAL_MISSING_COLUMNS),
    Migration(42, "missing_status_columns", sql=_V42_MISSING_STATUS_COLUMNS),
    Migration(43, "skills_lessons_columns", fn=_apply_skills_lessons_columns),
    Migration(44, "evolution_outcome_columns",
              fn=_apply_evolution_outcome_columns),
    Migration(45, "proactive_and_mood_columns",
              fn=_apply_proactive_mood_columns),
    Migration(46, "game_tables", sql=_V46_GAME_TABLES),
    Migration(47, "memory_tags_origin",
              fn=_apply_memory_tags_origin),
    Migration(48, "search_engine_schema",
              fn=_apply_search_engine_schema),
    Migration(49, "skills_table_unify", fn=_apply_skills_table_unify),
    Migration(50, "companion_schema", fn=_apply_companion_schema),
    Migration(51, "cipher_vault", fn=_apply_cipher_vault),
    Migration(52, "self_improvement_v2", fn=_apply_self_improvement_v2),
    Migration(53, "trading_tables", fn=_apply_trading_tables),
    Migration(54, "watchers_tables", fn=_apply_watchers_tables),
    Migration(55, "game_relay_tables", fn=_apply_game_relay_tables),
    Migration(56, "room_tables", fn=_apply_room_tables),
    Migration(57, "briefing_tables", fn=_apply_briefing_tables),
    Migration(58, "memory_archive", fn=_apply_memory_archive),
    Migration(59, "notifications_delivery_state",
              sql="ALTER TABLE notifications ADD COLUMN delivery_state "
                  "TEXT NOT NULL DEFAULT '';"),
    Migration(60, "artifacts", sql=_V60_ARTIFACTS),
    Migration(61, "arena_scores", sql=_V61_ARENA_SCORES),
    Migration(62, "research_organs", fn=_apply_research_organs),
    Migration(63, "benchmark_runs", sql=_V63_BENCHMARK_RUNS),
    Migration(64, "research_loop_runs", fn=_apply_research_loop_tables),
    Migration(65, "coremind_telemetry", sql=_V65_COREMIND_TELEMETRY),
    Migration(66, "research_loop_g2", fn=_apply_research_loop_g2),
    Migration(67, "player_library", sql=_V67_PLAYER_LIBRARY),
    Migration(68, "media_queue_album_column",
              sql="ALTER TABLE media_queue ADD COLUMN album "
                  "TEXT NOT NULL DEFAULT '';"),
    Migration(69, "game_gear", sql=_V69_GAME_GEAR),
    Migration(70, "game_players_xp_column",
              sql="ALTER TABLE game_players ADD COLUMN xp "
                  "INTEGER NOT NULL DEFAULT 0;"),
    Migration(71, "trial_assist_runs", sql=_V71_TRIAL_ASSIST_RUNS),
    Migration(72, "trial_sms_watches", sql=_V72_TRIAL_SMS_WATCHES),
    Migration(73, "game_skills", sql=_V73_GAME_SKILLS),
    Migration(74, "game_titles", sql=_V74_GAME_TITLES),
    Migration(75, "game_attributes_rename", fn=_apply_game_attributes_rename),
    Migration(76, "game_identity_alias_merge",
              fn=_apply_game_identity_alias_merge),
)


def latest_version() -> int:
    return max(m.version for m in MIGRATIONS)
