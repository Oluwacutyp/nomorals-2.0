"""Schema migration engine.

Properties that matter for a long-lived personal database:

* **Versioned and ordered.** Migrations are integers; they apply in order, once.
* **Checksummed.** The SHA-256 of each applied migration is stored. If someone
  edits an already-applied migration, the runner refuses to start rather than
  silently diverging — because a silently diverged schema is unrecoverable.
* **Atomic.** Each migration runs inside its own transaction. SQLite DDL is
  transactional, so a failed migration leaves the database exactly as it was.
* **Forward-only by default.** Down-migrations are supported but never automatic;
  dropping data without an explicit operator action is not a default.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..core.errors import MigrationError, StorageError
from ..core.logging_setup import get_logger

__all__ = ["Migration", "MigrationRunner", "SchemaError"]

_log = get_logger(__name__)


class SchemaError(StorageError):
    code = "storage.schema"
    retryable = False


@dataclass(frozen=True)
class Migration:
    """One schema change.

    Provide ``sql`` for declarative DDL, or ``fn`` for anything that needs logic
    (backfilling a column, rebuilding a table). ``down`` is optional and only ever
    run explicitly.
    """

    version: int
    name: str
    sql: str = ""
    fn: Callable[[Any], None] | None = None
    down: str = ""

    def __post_init__(self) -> None:
        if not self.sql and self.fn is None:
            raise SchemaError(f"migration {self.version} ({self.name}) has neither sql nor fn")
        if self.version <= 0:
            raise SchemaError("migration versions must be positive")

    @property
    def checksum(self) -> str:
        body = self.sql or (self.fn.__doc__ or self.name)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]

    @property
    def label(self) -> str:
        return f"{self.version:04d}_{self.name}"


class MigrationRunner:
    """Applies migrations to a :class:`~nomorals.storage.db.Database`."""

    TABLE = "schema_migrations"

    def __init__(self, db: Any) -> None:
        self.db = db
        self._ensure_table()

    def _ensure_table(self) -> None:
        self.db.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE} (
                version     INTEGER PRIMARY KEY,
                name        TEXT    NOT NULL,
                checksum    TEXT    NOT NULL,
                applied_at  REAL    NOT NULL,
                duration_ms REAL    NOT NULL DEFAULT 0
            )
            """
        )

    # ── introspection ────────────────────────────────────────────────────────
    def applied(self) -> dict[int, dict[str, Any]]:
        rows = self.db.query(f"SELECT * FROM {self.TABLE} ORDER BY version")
        return {int(row["version"]): row for row in rows}

    def current_version(self) -> int:
        return int(self.db.scalar(f"SELECT COALESCE(MAX(version), 0) FROM {self.TABLE}", default=0))

    def pending(self, migrations: Sequence[Migration]) -> list[Migration]:
        done = self.applied()
        return [m for m in sorted(migrations, key=lambda m: m.version) if m.version not in done]

    def history(self) -> list[dict[str, Any]]:
        return self.db.query(f"SELECT * FROM {self.TABLE} ORDER BY version")

    # ── application ──────────────────────────────────────────────────────────
    def validate(self, migrations: Sequence[Migration]) -> list[str]:
        """Detect edits to already-applied migrations. Returns a list of problems."""
        done = self.applied()
        problems: list[str] = []
        seen: set[int] = set()
        for migration in migrations:
            if migration.version in seen:
                problems.append(f"duplicate migration version {migration.version}")
            seen.add(migration.version)
            record = done.get(migration.version)
            if record and record["checksum"] != migration.checksum:
                problems.append(
                    f"migration {migration.label} was modified after being applied "
                    f"(stored {record['checksum']}, now {migration.checksum})"
                )
        return problems

    def apply(self, migration: Migration) -> float:
        """Apply a single migration inside a transaction. Returns duration in ms."""
        started = time.perf_counter()
        with self.db.transaction():
            if migration.sql:
                # execute_statements, not executescript: the latter implicitly
                # commits, which would silently defeat the atomicity below.
                self.db.execute_statements(migration.sql)
            if migration.fn is not None:
                migration.fn(self.db)
            self.db.execute(
                f"INSERT INTO {self.TABLE} (version, name, checksum, applied_at, duration_ms) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    migration.version,
                    migration.name,
                    migration.checksum,
                    time.time(),
                    round((time.perf_counter() - started) * 1000, 3),
                ),
            )
        duration = (time.perf_counter() - started) * 1000
        _log.debug("applied migration %s in %.1fms", migration.label, duration)
        return duration

    def apply_all(self, migrations: Sequence[Migration], *, strict: bool = True) -> list[str]:
        """Apply every pending migration in version order.

        Raises :class:`MigrationError` if an applied migration has been edited.
        """
        problems = self.validate(migrations)
        if problems and strict:
            raise MigrationError(
                "refusing to migrate: " + "; ".join(problems),
                details={"problems": problems},
            )
        if problems:
            _log.warning("migration checksum problems ignored: %s", problems)

        applied: list[str] = []
        for migration in self.pending(migrations):
            try:
                self.apply(migration)
            except StorageError:
                raise
            except Exception as exc:  # noqa: BLE001 - wrap foreign errors
                raise MigrationError(
                    f"migration {migration.label} failed: {exc}"
                ) from exc
            applied.append(migration.label)
        return applied

    def rollback(self, migrations: Sequence[Migration], to_version: int) -> list[str]:
        """Explicitly roll back migrations above ``to_version``. Never automatic."""
        rolled: list[str] = []
        by_version = {m.version: m for m in migrations}
        for version in sorted(self.applied(), reverse=True):
            if version <= to_version:
                break
            migration = by_version.get(version)
            if migration is None or not migration.down:
                raise MigrationError(
                    f"migration {version} has no down-migration; cannot roll back safely"
                )
            with self.db.transaction():
                self.db.execute_statements(migration.down)
                self.db.execute(f"DELETE FROM {self.TABLE} WHERE version = ?", (version,))
            rolled.append(f"{version:04d}")
        return rolled
