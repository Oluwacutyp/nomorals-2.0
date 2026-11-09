"""L2 persistence: SQLite substrate, migrations, search, vectors, blobs, backups."""

from __future__ import annotations

from .analytics import AnalyticsBackend, DuckDBAnalytics, SQLiteAnalytics, open_analytics
from .backup import BackupInfo, BackupManager, RetentionPolicy
from .blob import BlobInfo, BlobStore
from .db import AsyncDatabase, Database, Row, transaction
from .docstore import (
    BackendHealth,
    DocumentStore,
    MongoDocumentStore,
    SQLiteDocumentStore,
    ShadowDocumentStore,
    StoreResolution,
    attach_document_store,
    open_document_store,
    resolve_document_store,
    storage_backend_status,
)
from .parity import ParityReport, check_document_parity, run_kv_parity_housekeeping
from .failures import (Failure, clear_failures, failure_count,
                       format_failures, recent_failures, record_failure)
from .fts import FTSIndex, SearchHit, build_match_query, escape_fts, near_query, phrase_query
from .kv import KVStore
from .queue import (
    CronNeverMatchesError,
    CronTimezoneError,
    Job,
    QueueFull,
    WorkQueue,
    next_cron_fire,
    parse_cron,
)
from .replication import SnapshotInfo, SqliteReplicator
from .s3blob import S3BlobStore, S3Config, open_blob_store, presign_url
from .schema import Migration, MigrationRunner, SchemaError
from .vectors import VectorRecord, VectorSearchHit, VectorStore, cosine, normalize

__all__ = [
    "AnalyticsBackend",
    "AsyncDatabase",
    "BackendHealth",
    "BackupInfo",
    "BackupManager",
    "BlobInfo",
    "BlobStore",
    "CronNeverMatchesError",
    "CronTimezoneError",
    "Database",
    "DocumentStore",
    "DuckDBAnalytics",
    "FTSIndex",
    "Failure",
    "Job",
    "KVStore",
    "Migration",
    "MigrationRunner",
    "MongoDocumentStore",
    "ParityReport",
    "QueueFull",
    "RetentionPolicy",
    "Row",
    "S3BlobStore",
    "S3Config",
    "SQLiteAnalytics",
    "SQLiteDocumentStore",
    "SchemaError",
    "SearchHit",
    "ShadowDocumentStore",
    "SnapshotInfo",
    "SqliteReplicator",
    "StoreResolution",
    "VectorRecord",
    "VectorSearchHit",
    "VectorStore",
    "attach_document_store",
    "build_match_query",
    "check_document_parity",
    "clear_failures",
    "cosine",
    "escape_fts",
    "failure_count",
    "format_failures",
    "near_query",
    "next_cron_fire",
    "normalize",
    "open_analytics",
    "open_blob_store",
    "open_document_store",
    "parse_cron",
    "phrase_query",
    "presign_url",
    "recent_failures",
    "record_failure",
    "resolve_document_store",
    "run_kv_parity_housekeeping",
    "storage_backend_status",
    "transaction",
]
