"""L2 persistence: SQLite substrate, migrations, search, vectors, blobs, backups."""

from __future__ import annotations

from .analytics import AnalyticsBackend, DuckDBAnalytics, SQLiteAnalytics, open_analytics
from .db import Database, Row, transaction
from .kv import KVStore
from .replication import SnapshotInfo, SqliteReplicator
from .s3blob import S3BlobStore, S3Config, open_blob_store
from .schema import Migration, MigrationRunner, SchemaError

__all__ = [
    "AnalyticsBackend",
    "Database",
    "DuckDBAnalytics",
    "KVStore",
    "Migration",
    "MigrationRunner",
    "Row",
    "S3BlobStore",
    "S3Config",
    "SQLiteAnalytics",
    "SchemaError",
    "SnapshotInfo",
    "SqliteReplicator",
    "open_analytics",
    "open_blob_store",
    "transaction",
]
