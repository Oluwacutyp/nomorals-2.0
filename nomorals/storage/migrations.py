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

-- Add missing status columns
ALTER TABLE missions ADD COLUMN status TEXT NOT NULL DEFAULT 'active';
ALTER TABLE agent_tasks ADD COLUMN status TEXT NOT NULL DEFAULT 'pending';
"""

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
)


def latest_version() -> int:
    return max(m.version for m in MIGRATIONS)
