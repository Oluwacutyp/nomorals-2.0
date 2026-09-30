"""Semantic code indexer - index repos by function/class for intelligent search.

Uses AST parsing to extract code structure (functions, classes, methods) and
indexes them with embeddings for semantic search. Much better than grep for
"find me the function that handles authentication" type queries.

Features:
- Parse Python/JS/TS files into AST
- Extract functions, classes, methods with docstrings
- Generate embeddings for each code unit
- Store in vector DB for semantic search
- Track file changes and re-index incrementally

Usage:
    indexer = CodeIndexer(db, embedding_model)
    
    # Index a repository
    await indexer.index_repo("/path/to/repo")
    
    # Semantic search
    results = await indexer.search("function that validates JWT tokens", limit=5)
    for hit in results:
        print(f"{hit.file}:{hit.line} - {hit.name}")
        print(hit.code)
    
    # Get function by name
    func = await indexer.get_function("auth.py", "validate_token")
"""

from __future__ import annotations

import ast
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..llm.base import LLMProvider
from ..storage.db import Database
from ..storage.vectors import VectorStore

__all__ = ["CodeIndexer", "CodeUnit", "CodeSearchResult"]

_log = get_logger(__name__)

# File extensions to index
SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
}

# Directories to skip
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", "coverage",
}


@dataclass
class CodeUnit:
    """A unit of code (function, class, method)."""
    
    unit_id: str
    file_path: str
    name: str
    unit_type: str  # function, class, method
    language: str
    code: str
    docstring: str = ""
    line_start: int = 0
    line_end: int = 0
    parent: str = ""  # Class name for methods
    parameters: list[str] = field(default_factory=list)
    return_type: str = ""
    imports: list[str] = field(default_factory=list)
    file_hash: str = ""
    indexed_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "file_path": self.file_path,
            "name": self.name,
            "unit_type": self.unit_type,
            "language": self.language,
            "docstring": self.docstring,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "parent": self.parent,
        }
    
    def to_embedding_text(self) -> str:
        """Generate text for embedding (combines name, docstring, signature)."""
        parts = [f"{self.unit_type}: {self.name}"]
        
        if self.parent:
            parts[0] = f"{self.unit_type}: {self.parent}.{self.name}"
        
        if self.parameters:
            parts.append(f"params: {', '.join(self.parameters)}")
        
        if self.return_type:
            parts.append(f"returns: {self.return_type}")
        
        if self.docstring:
            parts.append(f"description: {self.docstring}")
        
        return "\n".join(parts)


@dataclass
class CodeSearchResult:
    """Result from semantic code search."""
    
    unit: CodeUnit
    score: float
    snippet: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.unit.file_path,
            "name": self.unit.name,
            "type": self.unit.unit_type,
            "line": self.unit.line_start,
            "score": self.score,
            "docstring": self.unit.docstring[:200] if self.unit.docstring else "",
        }


class CodeIndexer:
    """Semantic code indexer and search engine."""
    
    def __init__(
        self,
        db: Database,
        embedding_model: LLMProvider,
        vector_store: VectorStore,
    ) -> None:
        self.db = db
        self.embedding_model = embedding_model
        self.vectors = vector_store
        self._ensure_schema()
        _log.info("Code indexer initialized")
    
    def _ensure_schema(self) -> None:
        """Create code indexer tables."""
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS code_units (
                    unit_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    name TEXT NOT NULL,
                    unit_type TEXT NOT NULL,
                    language TEXT NOT NULL,
                    code TEXT NOT NULL,
                    docstring TEXT NOT NULL DEFAULT '',
                    line_start INTEGER NOT NULL DEFAULT 0,
                    line_end INTEGER NOT NULL DEFAULT 0,
                    parent TEXT NOT NULL DEFAULT '',
                    parameters TEXT NOT NULL DEFAULT '[]',
                    return_type TEXT NOT NULL DEFAULT '',
                    imports TEXT NOT NULL DEFAULT '[]',
                    file_hash TEXT NOT NULL DEFAULT '',
                    indexed_at REAL NOT NULL
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS indexed_files (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    indexed_at REAL NOT NULL,
                    unit_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_units_file ON code_units(file_path)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_units_name ON code_units(name)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_units_type ON code_units(unit_type)")
    
    async def index_repo(
        self,
        repo_path: str,
        *,
        incremental: bool = True,
    ) -> dict[str, int]:
        """Index an entire repository.
        
        Args:
            repo_path: Path to repository root
            incremental: If True, only re-index changed files
            
        Returns:
            Stats dict with counts
        """
        repo = Path(repo_path).resolve()
        if not repo.exists():
            raise FileNotFoundError(f"Repository not found: {repo_path}")
        
        stats = {"files_scanned": 0, "files_indexed": 0, "units_indexed": 0, "skipped": 0}
        
        # Walk repository
        for file_path in repo.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Skip ignored directories
            if any(skip in file_path.parts for skip in SKIP_DIRS):
                stats["skipped"] += 1
                continue
            
            # Check extension
            if file_path.suffix not in SUPPORTED_EXTENSIONS:
                continue
            
            stats["files_scanned"] += 1
            
            try:
                # Check if file changed
                file_hash = self._hash_file(file_path)
                
                if incremental:
                    existing = self.db.query_one(
                        "SELECT file_hash FROM indexed_files WHERE file_path = ?",
                        (str(file_path),)
                    )
                    if existing and existing["file_hash"] == file_hash:
                        continue  # File unchanged
                
                # Index file
                units = await self.index_file(str(file_path))
                stats["files_indexed"] += 1
                stats["units_indexed"] += len(units)
                
                _log.info(f"Indexed {file_path}: {len(units)} units")
                
            except Exception as e:
                _log.warning(f"Failed to index {file_path}: {e}")
                stats["skipped"] += 1
        
        _log.info(f"Repository indexed: {stats}")
        return stats
    
    async def index_file(self, file_path: str) -> list[CodeUnit]:
        """Index a single file.
        
        Args:
            file_path: Path to source file
            
        Returns:
            List of indexed CodeUnit objects
        """
        path = Path(file_path)
        language = SUPPORTED_EXTENSIONS.get(path.suffix)
        
        if not language:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        code = path.read_text(encoding="utf-8", errors="ignore")
        file_hash = hashlib.sha256(code.encode()).hexdigest()
        
        # Parse based on language
        if language == "python":
            units = self._parse_python(code, file_path, file_hash)
        else:
            # For JS/TS, use regex-based extraction (AST parsing requires Node.js)
            units = self._parse_javascript(code, file_path, language, file_hash)
        
        # Remove old units for this file
        with self.db.transaction():
            self.db.execute("DELETE FROM code_units WHERE file_path = ?", (file_path,))
        
        # Store new units
        for unit in units:
            with self.db.transaction():
                self.db.execute("""
                    INSERT INTO code_units
                    (unit_id, file_path, name, unit_type, language, code, docstring,
                     line_start, line_end, parent, parameters, return_type, imports,
                     file_hash, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    unit.unit_id, unit.file_path, unit.name, unit.unit_type,
                    unit.language, unit.code, unit.docstring, unit.line_start,
                    unit.line_end, unit.parent, json.dumps(unit.parameters),
                    unit.return_type, json.dumps(unit.imports), unit.file_hash,
                    unit.indexed_at,
                ))
        
        # Update file index
        with self.db.transaction():
            self.db.execute("""
                INSERT OR REPLACE INTO indexed_files (file_path, file_hash, indexed_at, unit_count)
                VALUES (?, ?, ?, ?)
            """, (file_path, file_hash, time.time(), len(units)))
        
        # Generate embeddings
        for unit in units:
            text = unit.to_embedding_text()
            embeddings = self.embedding_model.embed([text])
            embedding = embeddings[0] if embeddings else []
            if embedding:
                self.vectors.put(
                    embedding,
                    owner_type="code_unit",
                    owner_id=unit.unit_id,
                )
        
        return units
    
    def _parse_python(self, code: str, file_path: str, file_hash: str) -> list[CodeUnit]:
        """Parse Python file using AST."""
        units = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            _log.warning(f"Failed to parse {file_path}: {e}")
            return units
        
        # Extract imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        # Extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                unit = self._extract_python_function(node, code, file_path, file_hash, imports)
                if unit:
                    units.append(unit)
            
            elif isinstance(node, ast.ClassDef):
                # Extract class
                class_unit = self._extract_python_class(node, code, file_path, file_hash, imports)
                if class_unit:
                    units.append(class_unit)
                
                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_unit = self._extract_python_function(
                            item, code, file_path, file_hash, imports, parent=node.name
                        )
                        if method_unit:
                            units.append(method_unit)
        
        return units
    
    def _extract_python_function(
        self,
        node: ast.FunctionDef,
        code: str,
        file_path: str,
        file_hash: str,
        imports: list[str],
        parent: str = "",
    ) -> Optional[CodeUnit]:
        """Extract a Python function from AST node."""
        # Get function code
        lines = code.splitlines()
        start_line = node.lineno - 1
        end_line = node.end_lineno or start_line + 1
        func_code = "\n".join(lines[start_line:end_line])
        
        # Extract docstring
        docstring = ast.get_docstring(node) or ""
        
        # Extract parameters
        params = [arg.arg for arg in node.args.args if arg.arg != "self"]
        
        # Extract return type
        return_type = ""
        if node.returns:
            return_type = ast.unparse(node.returns)
        
        return CodeUnit(
            unit_id=new_id("unit"),
            file_path=file_path,
            name=node.name,
            unit_type="method" if parent else "function",
            language="python",
            code=func_code,
            docstring=docstring,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            parent=parent,
            parameters=params,
            return_type=return_type,
            imports=imports,
            file_hash=file_hash,
        )
    
    def _extract_python_class(
        self,
        node: ast.ClassDef,
        code: str,
        file_path: str,
        file_hash: str,
        imports: list[str],
    ) -> Optional[CodeUnit]:
        """Extract a Python class from AST node."""
        lines = code.splitlines()
        start_line = node.lineno - 1
        end_line = node.end_lineno or start_line + 1
        class_code = "\n".join(lines[start_line:end_line])
        
        docstring = ast.get_docstring(node) or ""
        
        return CodeUnit(
            unit_id=new_id("unit"),
            file_path=file_path,
            name=node.name,
            unit_type="class",
            language="python",
            code=class_code,
            docstring=docstring,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            imports=imports,
            file_hash=file_hash,
        )
    
    def _parse_javascript(
        self,
        code: str,
        file_path: str,
        language: str,
        file_hash: str,
    ) -> list[CodeUnit]:
        """Parse JavaScript/TypeScript file using regex (basic extraction)."""
        import re
        units = []
        
        # Extract functions
        func_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)'
        for match in re.finditer(func_pattern, code):
            name = match.group(1)
            params = match.group(2).split(",") if match.group(2) else []
            
            # Find function body
            start = match.start()
            brace_count = 0
            end = start
            for i, char in enumerate(code[start:], start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            func_code = code[start:end]
            line_start = code[:start].count("\n") + 1
            line_end = code[:end].count("\n") + 1
            
            units.append(CodeUnit(
                unit_id=new_id("unit"),
                file_path=file_path,
                name=name,
                unit_type="function",
                language=language,
                code=func_code,
                line_start=line_start,
                line_end=line_end,
                parameters=[p.strip() for p in params if p.strip()],
                file_hash=file_hash,
            ))
        
        # Extract classes
        class_pattern = r'(?:export\s+)?class\s+(\w+)'
        for match in re.finditer(class_pattern, code):
            name = match.group(1)
            start = match.start()
            
            # Find class body
            brace_count = 0
            end = start
            for i, char in enumerate(code[start:], start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            class_code = code[start:end]
            line_start = code[:start].count("\n") + 1
            line_end = code[:end].count("\n") + 1
            
            units.append(CodeUnit(
                unit_id=new_id("unit"),
                file_path=file_path,
                name=name,
                unit_type="class",
                language=language,
                code=class_code,
                line_start=line_start,
                line_end=line_end,
                file_hash=file_hash,
            ))
        
        return units
    
    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        file_filter: str = "",
        unit_type: str = "",
    ) -> list[CodeSearchResult]:
        """Semantic search for code units.
        
        Args:
            query: Natural language query
            limit: Maximum results
            file_filter: Filter by file path (substring match)
            unit_type: Filter by unit type (function, class, method)
            
        Returns:
            List of CodeSearchResult objects
        """
        # Generate query embedding
        embeddings = self.embedding_model.embed([query])
        query_embedding = embeddings[0] if embeddings else []
        
        # Search vector store
        hits = self.vectors.search(query_embedding, limit=limit * 2, owner_type="code_unit")  # Get more to filter
        
        results = []
        for hit in hits:
            unit_id = hit.owner_id
            score = hit.score
            
            # Fetch full unit from DB
            row = self.db.query_one(
                "SELECT * FROM code_units WHERE unit_id = ?",
                (unit_id,)
            )
            
            if not row:
                continue
            
            # Apply filters
            if file_filter and file_filter not in row["file_path"]:
                continue
            if unit_type and row["unit_type"] != unit_type:
                continue
            
            unit = CodeUnit(
                unit_id=row["unit_id"],
                file_path=row["file_path"],
                name=row["name"],
                unit_type=row["unit_type"],
                language=row["language"],
                code=row["code"],
                docstring=row["docstring"],
                line_start=row["line_start"],
                line_end=row["line_end"],
                parent=row["parent"],
                parameters=json.loads(row["parameters"]),
                return_type=row["return_type"],
                imports=json.loads(row["imports"]),
                file_hash=row["file_hash"],
            )
            
            results.append(CodeSearchResult(unit=unit, score=score))
            
            if len(results) >= limit:
                break
        
        return results
    
    async def get_function(self, file_path: str, name: str) -> Optional[CodeUnit]:
        """Get a specific function by file and name."""
        row = self.db.query_one("""
            SELECT * FROM code_units
            WHERE file_path = ? AND name = ? AND unit_type IN ('function', 'method')
        """, (file_path, name))
        
        if not row:
            return None
        
        return CodeUnit(
            unit_id=row["unit_id"],
            file_path=row["file_path"],
            name=row["name"],
            unit_type=row["unit_type"],
            language=row["language"],
            code=row["code"],
            docstring=row["docstring"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            parent=row["parent"],
            parameters=json.loads(row["parameters"]),
            return_type=row["return_type"],
            imports=json.loads(row["imports"]),
            file_hash=row["file_hash"],
        )
    
    def _hash_file(self, file_path: Path) -> str:
        """Generate hash of file contents."""
        code = file_path.read_text(encoding="utf-8", errors="ignore")
        return hashlib.sha256(code.encode()).hexdigest()
    



# ── registry hook ──────────────────────────────────────────────────────────


def _run_async(coro: Any) -> Any:
    """Drive a coroutine from sync tool code, even inside a running loop."""
    import asyncio
    import concurrent.futures

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class _EmbedAdapter:
    """Adapt the hashing Embedder to the LLMProvider-style embed([texts])."""

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_many(texts)


def _get_indexer(context: Any) -> "CodeIndexer":
    from ..core.errors import ToolError
    from ..memory.embeddings import Embedder
    from ..storage.vectors import VectorStore

    db = getattr(context, "db", None) if context is not None else None
    if db is None:
        raise ToolError("code indexer needs a context with a database")
    # Hashing embedder: no model download, no router call — works offline.
    embedder = Embedder(provider="hashing")
    return CodeIndexer(db, _EmbedAdapter(embedder), VectorStore(db))  # type: ignore[arg-type]


def register(registry: Any) -> None:
    """Attach the code index tools to a registry."""
    from ..core.errors import ToolError
    from ..core.policy import Capability
    from .filesystem import safe_path

    context = registry.context

    @registry.register(
        "index_repo",
        description="Index a repository's functions/classes for semantic code search.",
        capability=Capability.FS_READ,
    )
    def index_repo(path: str = ".") -> dict[str, Any]:
        target = safe_path(context, path)
        if not target.is_dir():
            raise ToolError(f"not a directory: {path}")
        indexer = _get_indexer(context)
        return _run_async(indexer.index_repo(str(target)))

    @registry.register(
        "search_code",
        description="Semantic search over the indexed code: find functions/classes by description.",
        capability=Capability.MEM_READ,
    )
    def search_code(query: str, limit: int = 10) -> dict[str, Any]:
        indexer = _get_indexer(context)
        results = _run_async(indexer.search(query, limit=limit))
        return {"query": query,
                "results": [r.to_dict() for r in results]}

    @registry.register(
        "get_symbol",
        description="Fetch a specific function/class by file path and name.",
        capability=Capability.MEM_READ,
    )
    def get_symbol(path: str, name: str) -> dict[str, Any]:
        indexer = _get_indexer(context)
        unit = _run_async(indexer.get_function(path, name))
        if unit is None:
            raise ToolError(f"symbol {name!r} not found in {path}")
        return unit.to_dict()
