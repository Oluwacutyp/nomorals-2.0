"""Schema migration engine.

Properties that matter for a long-lived personal database:

* **Versioned and ordered.** Migrations are integers; they apply in order, once.
* **Checksummed.** The SHA-256 of each applied migration is stored. If someone
  edits an already-applied migration, the runner refuses to start rather than
  silently diverging — because a silently diverged schema is unrecoverable.
* **Source-hashed for code migrations.** SQL migrations hash their text; Python
  (``fn``) migrations hash their *source code*, not their docstring — a doc
  edit is not a schema change and must not trip the guard.
* **Atomic.** Each migration runs inside its own transaction. SQLite DDL is
  transactional, so a failed migration leaves the database exactly as it was.
* **Cross-process safe.** ``apply_all`` holds an ``flock`` file lock while it
  works, then re-checks pending migrations *inside* the lock, so two processes
  (bot + CLI, bot + cron) can never double-apply a data migration.
* **Forward-only by default.** Down-migrations are supported but never automatic;
  dropping data without an explicit operator action is not a default.
* **Operator escape hatches, explicit only.** :meth:`MigrationRunner.repair`
  accepts an edited migration's new checksum (logged, never silent) and
  :meth:`MigrationRunner.baseline` stamps an existing database — both require
  an explicit call; nothing here ever runs on its own.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from ..core.errors import MigrationError, StorageError
from ..core.logging_setup import get_logger
from ..core.style import active_theme, header, kv_lines, styled_table

try:  # POSIX-only; the lock degrades to a no-op elsewhere
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

__all__ = ["Migration", "MigrationRunner", "SchemaError"]

_log = get_logger(__name__)


def _fn_fingerprint(fn: Callable[..., Any]) -> str:
    """AST-normalized fingerprint of a migration function's behavior.

    Comments, blank lines, and docstrings are stripped before hashing, so
    cosmetic edits to an already-applied migration don't trip the tamper
    guard. Anything that changes what the function *does* changes the
    fingerprint. Falls back to raw source, then to the docstring, when the
    AST route is unavailable.
    """
    try:
        raw = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return fn.__doc__ or fn.__name__
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return raw
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:]
    return ast.dump(tree)


class SchemaError(StorageError):
    code = "storage.schema"
    retryable = False


@dataclass(frozen=True)
class Migration:
    """One schema change.

    Provide ``sql`` for declarative DDL, or ``fn`` for anything that needs logic
    (backfilling a column, rebuilding a table). ``down`` is optional and only ever
    run explicitly.

    ``repeatable=True`` marks a Flyway-``R__``-style migration: it has no
    version ordering, runs *after* all pending versioned migrations, and is
    re-applied whenever its source checksum changes — the right semantic for
    idempotent objects like views and indexes, where "the latest definition
    wins". Repeatable migrations use ``version=0`` and are keyed by name.
    """

    version: int
    name: str
    sql: str = ""
    fn: Callable[[Any], None] | None = None
    down: str = ""
    repeatable: bool = False

    def __post_init__(self) -> None:
        if not self.sql and self.fn is None:
            raise SchemaError(f"migration {self.version} ({self.name}) has neither sql nor fn")
        if not self.repeatable and self.version <= 0:
            raise SchemaError("migration versions must be positive")
        if self.repeatable and self.version != 0:
            raise SchemaError("repeatable migrations must use version=0")

    @property
    def checksum(self) -> str:
        """Legacy checksum: SQL text, or the fn's docstring for code migrations.

        Kept byte-identical so every already-applied row keeps validating.
        New code paths prefer :attr:`source_checksum`.
        """
        body = self.sql or (self.fn.__doc__ or self.name)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]

    @property
    def source_checksum(self) -> str:
        """Checksum of what the migration actually *does*.

        SQL migrations hash their text. Python (``fn``) migrations hash an
        AST-normalized fingerprint of their code — comments, whitespace, and
        docstrings are stripped, so a doc edit no longer trips the tamper
        guard while any behavioral change does. Falls back to :attr:`checksum`
        when the source is unavailable.
        """
        if self.fn is None:
            return self.checksum
        return hashlib.sha256(_fn_fingerprint(self.fn).encode("utf-8")).hexdigest()[:16]

    @property
    def label(self) -> str:
        if self.repeatable:
            return f"R__{self.name}"
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
        # ``source_checksum`` arrived after the table did; add it in place on
        # databases created by older builds. NULL = legacy row, validated with
        # the doc-based checksum for backward compatibility.
        columns = {row["name"] for row in self.db.query(f"PRAGMA table_info({self.TABLE})")}
        if "source_checksum" not in columns:
            self.db.execute(f"ALTER TABLE {self.TABLE} ADD COLUMN source_checksum TEXT")
        # Repeatable migrations (Flyway R__) are keyed by name, not version —
        # the versioned table's INTEGER PRIMARY KEY can't hold them.
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations_repeatable (
                name            TEXT PRIMARY KEY,
                checksum        TEXT NOT NULL,
                source_checksum TEXT,
                applied_at      REAL NOT NULL,
                duration_ms     REAL NOT NULL DEFAULT 0
            )
            """
        )

    # ── introspection ────────────────────────────────────────────────────────
    def applied(self) -> dict[int, dict[str, Any]]:
        rows = self.db.query(f"SELECT * FROM {self.TABLE} ORDER BY version")
        return {int(row["version"]): row for row in rows}

    def applied_repeatable(self) -> dict[str, dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM schema_migrations_repeatable ORDER BY name"
        )
        return {str(row["name"]): row for row in rows}

    def current_version(self) -> int:
        return int(self.db.scalar(f"SELECT COALESCE(MAX(version), 0) FROM {self.TABLE}", default=0))

    def pending(self, migrations: Sequence[Migration]) -> list[Migration]:
        done = self.applied()
        return [m for m in sorted(migrations, key=lambda m: m.version)
                if not m.repeatable and m.version not in done]

    def pending_repeatable(
        self, migrations: Sequence[Migration]
    ) -> list[Migration]:
        """Repeatables whose checksum changed (or never applied)."""
        done = self.applied_repeatable()
        out: list[Migration] = []
        for migration in migrations:
            if not migration.repeatable:
                continue
            record = done.get(migration.name)
            if record is None:
                out.append(migration)
                continue
            stored = record.get("source_checksum") or record.get("checksum")
            current = (migration.source_checksum if record.get("source_checksum")
                       else migration.checksum)
            if stored != current:
                out.append(migration)
        return out

    def history(self) -> list[dict[str, Any]]:
        return self.db.query(f"SELECT * FROM {self.TABLE} ORDER BY version")

    # ── application ──────────────────────────────────────────────────────────
    def validate(self, migrations: Sequence[Migration]) -> list[str]:
        """Detect edits to already-applied migrations. Returns a list of problems.

        Rows that carry a ``source_checksum`` are compared on source (behavior);
        legacy rows without one fall back to the doc-based checksum so old
        databases keep validating unchanged. Also reports applied versions
        with no matching migration object (a migration was *removed* — Flyway's
        "resolved vs applied" check).
        """
        done = self.applied()
        problems: list[str] = []
        seen: set[int] = set()
        defined_versions = {m.version for m in migrations if not m.repeatable}
        for migration in migrations:
            if migration.repeatable:
                continue
            if migration.version in seen:
                problems.append(f"duplicate migration version {migration.version}")
            seen.add(migration.version)
            record = done.get(migration.version)
            if not record:
                continue
            stored_source = record.get("source_checksum")
            if stored_source:
                if stored_source != migration.source_checksum:
                    problems.append(
                        f"migration {migration.label} was modified after being applied "
                        f"(stored source {stored_source}, now {migration.source_checksum})"
                    )
            elif record["checksum"] != migration.checksum:
                problems.append(
                    f"migration {migration.label} was modified after being applied "
                    f"(stored {record['checksum']}, now {migration.checksum})"
                )
        for version in sorted(done):
            if version not in defined_versions:
                problems.append(
                    f"applied migration version {version} has no matching migration "
                    f"(it was removed or renamed; repair() or a new migration is needed)"
                )
        return problems

    def apply(self, migration: Migration) -> float:
        """Apply a single migration inside a transaction. Returns duration in ms.

        Repeatable migrations are recorded in ``schema_migrations_repeatable``
        keyed by name (re-applied whenever their checksum changes).
        """
        started = time.perf_counter()
        with self.db.transaction():
            if migration.sql:
                # execute_statements, not executescript: the latter implicitly
                # commits, which would silently defeat the atomicity below.
                self.db.execute_statements(migration.sql)
            if migration.fn is not None:
                migration.fn(self.db)
            if migration.repeatable:
                self.db.execute(
                    "INSERT INTO schema_migrations_repeatable "
                    "(name, checksum, source_checksum, applied_at, duration_ms) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(name) DO UPDATE SET "
                    "checksum = excluded.checksum, "
                    "source_checksum = excluded.source_checksum, "
                    "applied_at = excluded.applied_at, "
                    "duration_ms = excluded.duration_ms",
                    (
                        migration.name,
                        migration.checksum,
                        migration.source_checksum,
                        time.time(),
                        round((time.perf_counter() - started) * 1000, 3),
                    ),
                )
            else:
                self.db.execute(
                    f"INSERT INTO {self.TABLE} "
                    "(version, name, checksum, source_checksum, applied_at, duration_ms) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        migration.version,
                        migration.name,
                        migration.checksum,
                        migration.source_checksum,
                        time.time(),
                        round((time.perf_counter() - started) * 1000, 3),
                    ),
                )
        duration = (time.perf_counter() - started) * 1000
        _log.debug("applied migration %s in %.1fms", migration.label, duration)
        return duration

    @contextmanager
    def _migration_lock(self, timeout_s: float = 60.0) -> Iterator[None]:
        """Cross-process mutex so two ``migrate()`` callers can't interleave.

        ``flock`` releases automatically if the holder crashes, so there is no
        stale-lock state to clean up. Skipped for ``:memory:`` databases (no
        file to contend on) and on platforms without ``fcntl``.
        """
        path = getattr(self.db, "path", None)
        if path is None or fcntl is None:
            yield
            return
        lock_path = Path(str(path) + ".migrate.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout_s
        with open(lock_path, "a+b") as handle:
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise MigrationError(
                            f"could not acquire migration lock on {lock_path} "
                            f"within {timeout_s}s; another migrate is probably running"
                        )
                    time.sleep(0.2)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def apply_all(
        self,
        migrations: Sequence[Migration],
        *,
        strict: bool = True,
        lock_timeout_s: float = 60.0,
        target: int | None = None,
        out_of_order: bool = True,
    ) -> list[str]:
        """Apply every pending migration in version order, then repeatables.

        Holds the cross-process migration lock for the whole run and re-checks
        the pending set *inside* the lock, so a concurrent starter that lost
        the race observes history instead of re-applying.

        ``target`` caps the versioned run at a version (deploy staging).
        ``out_of_order=False`` refuses to apply a pending migration whose
        version is below the current one (Flyway-strict branchy workflows);
        the default True preserves the historical apply-everything behavior.
        Repeatable migrations (``repeatable=True``) always run last and are
        re-applied whenever their checksum changed.

        Raises :class:`MigrationError` if an applied migration has been edited
        (see :meth:`repair` for the explicit operator escape hatch).
        """
        with self._migration_lock(timeout_s=lock_timeout_s):
            problems = self.validate(migrations)
            if problems and strict:
                raise MigrationError(
                    "refusing to migrate: " + "; ".join(problems),
                    details={"problems": problems},
                )
            if problems:
                _log.warning("migration checksum problems ignored: %s", problems)

            applied: list[str] = []
            pending = self.pending(migrations)
            if target is not None:
                pending = [m for m in pending if m.version <= target]
            if not out_of_order:
                current = self.current_version()
                behind = [m for m in pending if m.version < current]
                if behind:
                    raise MigrationError(
                        "out-of-order migrations blocked: "
                        + ", ".join(m.label for m in behind)
                        + " (pass out_of_order=True to apply anyway)"
                    )
            for migration in pending:
                try:
                    self.apply(migration)
                except StorageError:
                    raise
                except Exception as exc:  # noqa: BLE001 - wrap foreign errors
                    raise MigrationError(
                        f"migration {migration.label} failed: {exc}"
                    ) from exc
                applied.append(migration.label)
            for migration in self.pending_repeatable(migrations):
                try:
                    self.apply(migration)
                except StorageError:
                    raise
                except Exception as exc:  # noqa: BLE001 - wrap foreign errors
                    raise MigrationError(
                        f"repeatable migration {migration.label} failed: {exc}"
                    ) from exc
                applied.append(migration.label)
        return applied

    # ── operator tools (explicit only, never automatic) ──────────────────────
    def repair(self, migrations: Sequence[Migration]) -> list[str]:
        """Accept the current definitions of edited migrations.

        This is the escape hatch for the checksum incident class (e.g. the
        migration-84 mismatch): instead of hand-editing ``schema_migrations``,
        the operator calls this, the stored checksums are updated to the
        current ones (with the old values in the warning log), and the source
        checksum is backfilled on legacy rows. Never called automatically.

        Returns the labels that were repaired.
        """
        done = self.applied()
        by_version = {m.version: m for m in migrations}
        repaired: list[str] = []
        with self.db.transaction():
            for version, record in done.items():
                migration = by_version.get(version)
                if migration is None:
                    continue
                stored_source = record.get("source_checksum")
                mismatch = (
                    stored_source != migration.source_checksum
                    if stored_source
                    else record["checksum"] != migration.checksum
                )
                if not mismatch:
                    continue
                _log.warning(
                    "repair: migration %s checksum accepted (stored checksum=%s "
                    "source=%s -> checksum=%s source=%s)",
                    migration.label,
                    record["checksum"],
                    stored_source,
                    migration.checksum,
                    migration.source_checksum,
                )
                self.db.execute(
                    f"UPDATE {self.TABLE} SET checksum = ?, source_checksum = ? "
                    f"WHERE version = ?",
                    (migration.checksum, migration.source_checksum, version),
                )
                repaired.append(migration.label)
        return repaired

    def baseline(
        self,
        migrations: Sequence[Migration],
        version: int,
        *,
        force: bool = False,
    ) -> list[str]:
        """Stamp migrations as applied without running them.

        For pointing fresh code at a database whose schema was built another
        way (manual DDL, an older tool). Refuses when the database already has
        applied migrations unless ``force=True``. Never automatic.
        """
        current = self.current_version()
        if current > 0 and not force:
            raise MigrationError(
                f"database already at version {current}; pass force=True to re-stamp"
            )
        done = set(self.applied())
        stamped: list[str] = []
        with self.db.transaction():
            for migration in sorted(migrations, key=lambda m: m.version):
                if migration.version > version or migration.version in done:
                    continue
                self.db.execute(
                    f"INSERT OR IGNORE INTO {self.TABLE} "
                    "(version, name, checksum, source_checksum, applied_at, duration_ms) "
                    "VALUES (?, ?, ?, ?, ?, 0)",
                    (
                        migration.version,
                        migration.name,
                        migration.checksum,
                        migration.source_checksum,
                        time.time(),
                    ),
                )
                stamped.append(migration.label)
        _log.warning("baseline: stamped %d migration(s) up to version %d", len(stamped), version)
        return stamped

    def dry_run(self, migrations: Sequence[Migration]) -> list[dict[str, Any]]:
        """Return the pending migration plan without executing anything."""
        from .db import split_sql

        plan: list[dict[str, Any]] = []
        for migration in [*self.pending(migrations),
                          *self.pending_repeatable(migrations)]:
            if migration.sql:
                statements: list[str] = split_sql(migration.sql)
            elif migration.fn is not None:
                statements = [f"<python: {migration.fn.__module__}.{migration.fn.__qualname__}>"]
            else:  # pragma: no cover - constructor forbids this
                statements = []
            plan.append(
                {
                    "version": migration.version,
                    "label": migration.label,
                    "repeatable": migration.repeatable,
                    "statements": statements,
                    "statement_count": len(statements),
                }
            )
        return plan

    def status(self, migrations: Sequence[Migration]) -> dict[str, Any]:
        """One-dict health overview: version, pending plan, checksum problems."""
        pending = self.pending(migrations)
        versions = [m.version for m in migrations]
        return {
            "current_version": self.current_version(),
            "latest_version": max(versions) if versions else 0,
            "applied_count": len(self.applied()),
            "pending": [m.label for m in pending],
            "pending_count": len(pending),
            "problems": self.validate(migrations),
        }

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

    def clean(self, confirm: str = "") -> list[str]:
        """Drop every user table — the Flyway ``clean`` command, explicit only.

        ``confirm`` must be the literal string ``"CLEAN"``; anything else
        raises. Migration history is wiped too, so the next ``apply_all``
        rebuilds from scratch. This is a development/test tool — never call
        it on a database with data you care about.
        """
        if confirm != "CLEAN":
            raise MigrationError(
                'clean() requires confirm="CLEAN" — refusing to wipe the database'
            )
        tables = [
            r["name"] for r in self.db.query(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        ]
        with self.db.transaction():
            for table in tables:
                self.db.execute(f'DROP TABLE "{table}"')
        _log.warning("clean: dropped %d tables", len(tables))
        return tables

    def status(self, migrations: Sequence[Migration]) -> dict[str, Any]:
        """One-dict health overview: version, pending plan, checksum problems."""
        pending = self.pending(migrations)
        versions = [m.version for m in migrations if not m.repeatable]
        return {
            "current_version": self.current_version(),
            "latest_version": max(versions) if versions else 0,
            "applied_count": len(self.applied()),
            "pending": [m.label for m in pending],
            "pending_count": len(pending),
            "repeatable_pending": [
                m.label for m in self.pending_repeatable(migrations)
            ],
            "repeatable_applied": len(self.applied_repeatable()),
            "problems": self.validate(migrations),
        }

    def format_status(self, migrations: Sequence[Migration],
                      theme: Any = None) -> str:
        """Human-readable migration overview through the shared style layer."""
        theme = theme or active_theme()
        status = self.status(migrations)
        problems = status["problems"]
        rows = [
            (f"v{status['current_version']}", f"latest v{status['latest_version']}",
             str(status["applied_count"]), str(status["pending_count"]),
             str(len(status["repeatable_pending"]))),
        ]
        table = styled_table(
            ["current", "latest", "applied", "pending", "repeatable↻"],
            rows, theme=theme,
        )
        lines = [header("migrations", theme=theme), table]
        if status["pending"]:
            lines.append("pending: " + ", ".join(status["pending"][:8]))
        if status["repeatable_pending"]:
            lines.append("repeatable↻: " + ", ".join(status["repeatable_pending"][:8]))
        if problems:
            from ..core.style import paint as _paint

            lines.append(_paint(
                "problems: " + "; ".join(problems[:5]), "error", theme=theme))
        return "\n".join(lines)
