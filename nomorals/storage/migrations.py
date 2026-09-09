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

MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "core_state", sql=_V1),
    Migration(2, "agents_tasks_missions", sql=_V2),
    Migration(3, "tools_models_training", sql=_V3),
    Migration(4, "files_media_social", sql=_V4),
    Migration(5, "work_queue", sql=_V5),
    Migration(6, "full_text_search", fn=_create_fts, down="DROP TABLE IF EXISTS web_fts;"),
    Migration(7, "secondary_indexes", sql=_V6_INDEXES),
)


def latest_version() -> int:
    return max(m.version for m in MIGRATIONS)
