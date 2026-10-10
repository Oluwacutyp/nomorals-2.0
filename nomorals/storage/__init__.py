"""L2 persistence: SQLite substrate, migrations, search, vectors, blobs, backups."""

from __future__ import annotations

from .analytics import AnalyticsBackend, DuckDBAnalytics, SQLiteAnalytics, open_analytics
from .backup import BackupInfo, BackupManager, RetentionPolicy
from .blob import BlobInfo, BlobStore
from .db import AsyncDatabase, Database, Row, transaction
from .fts import FTSIndex, SearchHit, build_match_query, escape_fts, near_query, phrase_query
from .kv import KVStore
from .queue import Job, QueueFull, WorkQueue, next_cron_fire, parse_cron
from .replication import SnapshotInfo, SqliteReplicator
from .s3blob import S3BlobStore, S3Config, open_blob_store, presign_url
from .schema import Migration, MigrationRunner, SchemaError
from .vectors import VectorRecord, VectorSearchHit, VectorStore, cosine, normalize

__all__ = [
    "AnalyticsBackend",
    "AsyncDatabase",
    "BackupInfo",
    "BackupManager",
    "BlobInfo",
    "BlobStore",
    "Database",
    "DuckDBAnalytics",
    "FTSIndex",
    "Job",
    "KVStore",
    "Migration",
    "MigrationRunner",
    "QueueFull",
    "RetentionPolicy",
    "Row",
    "S3BlobStore",
    "S3Config",
    "SQLiteAnalytics",
    "SchemaError",
    "SearchHit",
    "SnapshotInfo",
    "SqliteReplicator",
    "VectorRecord",
    "VectorSearchHit",
    "VectorStore",
    "build_match_query",
    "cosine",
    "escape_fts",
    "near_query",
    "next_cron_fire",
    "normalize",
    "open_analytics",
    "open_blob_store",
    "parse_cron",
    "phrase_query",
    "presign_url",
    "transaction",
]
