"""Voice cache — cached models and cached conversions.

The mined shape is storytold/discord-vocodes (census §3B.14): a
233-line Rust bot — a prefix command picks a fixed speaker
(``!donald`` → donald-trump, ``!snake`` → solid-snake, …), the text
goes to a TTS endpoint as ``{"speaker", "text"}``, and the returned
audio is attached back to the channel. The bot re-fetches every
time: the same (speaker, text) twice costs two endpoint calls, and
the model behind the endpoint is reloaded per request. There is no
cache anywhere in the clone — no cache dir, no key, no memo. That
missing piece is what this module builds, in Devon's voice organ,
as Devon's own improvement over the upstream.

Two caches, one eviction story each:

- :class:`ConversionCache` — converted audio on disk. A repeat
  conversion of the same input through the same model with the
  same settings never re-runs the pipeline: the cached wav is
  copied out instead. Keyed by a SHA-256 of (model identity, input
  file bytes, input rate, every output-affecting conversion
  setting). Entries are ``<key>.wav`` + a ``<key>.json`` sidecar
  carrying the output's own SHA-256, verified before any entry is
  served — a flipped byte refuses the entry by name instead of
  serving corrupt audio. Eviction is LRU by last use, under two
  caps (200 entries / 2 GiB by default, env-overridable); the entry
  being stored is never evicted to make room for itself.
- :class:`ModelLoadCache` — loaded models in memory. The conversion
  path used to build a fresh executor (and a fresh
  :class:`~nomorals.voice.lifecycle.ModelLifecycleManager`) per
  file, so ``torch.jit.load`` ran on every conversion even for the
  same model. One shared manager now owns TorchScript loads
  (keyed by path + size + mtime, so a replaced file reloads) and
  one shared :class:`~nomorals.voice.services.ModelServer` per
  services dir owns registry loads — a repeat load returns the
  live handle.

Both caches account for themselves: hits, misses, stores,
evictions, corrupt refusals, bytes served — visible in
:meth:`ConversionCache.status`, ``nm voice cache stats`` and
``/voice cache stats``. The result cache's counters persist in
``stats.json`` (atomic writes), so the accounting survives a
restart; the model cache's counters live on the lifecycle
manager's entries.

Stdlib only. Thread-safe (one lock per cache). A cache failure
never fails a conversion that otherwise succeeded: a store that
cannot write is recorded on the receipt, not raised.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from typing import Any

__all__ = [
    "ConversionCache",
    "CorruptCacheEntry",
    "ModelLoadCache",
    "cache_status_text",
    "default_cache_dir",
    "default_conversion_cache",
    "executor_identity",
    "shared_model_cache",
]

DEFAULT_MAX_ENTRIES = 200
DEFAULT_MAX_BYTES = 2 * 1024 * 1024 * 1024
_STATS_FILENAME = "stats.json"


class CorruptCacheEntry(ValueError):
    """A cached entry failed its integrity check.

    Raised by :meth:`ConversionCache.lookup`, naming the entry key
    — a corrupt entry is refused, never served. The entry's files
    are already removed when this raises, so the next lookup is a
    clean miss and the conversion re-runs.
    """


def default_cache_dir() -> str:
    """Where converted audio is cached.

    Env override ``NM_VOICE_CACHE_DIR``, else
    ``~/.config/nomorals/voice/cache`` — a sibling of the service
    store (``.../voice/services``), under the same parent.
    """
    override = os.environ.get("NM_VOICE_CACHE_DIR", "").strip()
    if override:
        return os.path.expanduser(override)
    return os.path.join(os.path.expanduser("~"), ".config",
                        "nomorals", "voice", "cache")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    tmp = path + f".tmp-{os.getpid()}-{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _atomic_copy(src: str, dest: str) -> None:
    tmp = dest + f".tmp-{os.getpid()}-{threading.get_ident()}"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# -- conversion result cache (disk) -----------------------------------------


class ConversionCache:
    """Converted wavs on disk, keyed by input + model + settings.

    Eviction rule (stated, enforced on every store): least-recently-
    used first, under ``max_entries`` entries and ``max_bytes`` total
    bytes. Caps come from the constructor, else the env
    (``NM_VOICE_CACHE_MAX_ENTRIES`` / ``NM_VOICE_CACHE_MAX_BYTES``),
    else the defaults (200 entries / 2 GiB).
    """

    def __init__(self, cache_dir: str = "", *,
                 max_entries: int | None = None,
                 max_bytes: int | None = None) -> None:
        self.cache_dir = cache_dir or default_cache_dir()
        self.max_entries = (
            int(max_entries) if max_entries is not None
            else _env_int("NM_VOICE_CACHE_MAX_ENTRIES",
                          DEFAULT_MAX_ENTRIES))
        self.max_bytes = (
            int(max_bytes) if max_bytes is not None
            else _env_int("NM_VOICE_CACHE_MAX_BYTES",
                          DEFAULT_MAX_BYTES))
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "stores": 0,
                       "evictions": 0, "corrupt": 0, "bytes_served": 0}
        self._load_stats()

    # -- keys -----------------------------------------------------------

    def key_for(self, *, model_identity: str, input_sha256: str,
                input_rate: int, params: Any) -> str:
        """The cache key for one conversion.

        ``params`` is a :class:`ConversionParams` or a plain dict of
        its fields. Every output-affecting setting is in the key, so
        two conversions share an entry only when they would produce
        byte-identical audio — a key that is too wide costs a miss,
        never a wrong hit.
        """
        if isinstance(params, dict):
            fields = params
        else:
            fields = {
                "model_rate": getattr(params, "model_rate", 0),
                "output_rate": getattr(params, "output_rate", 0),
                "buffer_size_minimum": getattr(
                    params, "buffer_size_minimum", 0),
                "direction": getattr(params, "direction", "A2B"),
                "pitch_shift_semitones": getattr(
                    params, "pitch_shift_semitones", 0.0),
                "f0_mean_log_src": getattr(params, "f0_mean_log_src", None),
                "f0_std_log_src": getattr(params, "f0_std_log_src", None),
                "f0_mean_log_target": getattr(
                    params, "f0_mean_log_target", None),
                "f0_std_log_target": getattr(
                    params, "f0_std_log_target", None),
                "skip_vocode": bool(getattr(params, "skip_vocode", False)),
                "discard_vocoded_audio": bool(
                    getattr(params, "discard_vocoded_audio", False)),
                "n_fft": getattr(params, "n_fft", 1024),
                "hop_length": getattr(params, "hop_length", 256),
                "griffin_iters": getattr(params, "griffin_iters", 0),
            }
        canonical = {
            "model": str(model_identity or "none"),
            "input_sha256": str(input_sha256),
            "input_rate": int(input_rate),
            "params": fields,
        }
        blob = json.dumps(canonical, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # -- paths ----------------------------------------------------------

    def _wav_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key + ".wav")

    def _meta_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key + ".json")

    def _stats_path(self) -> str:
        return os.path.join(self.cache_dir, _STATS_FILENAME)

    # -- entries --------------------------------------------------------

    def _read_meta(self, key: str) -> dict[str, Any] | None:
        try:
            with open(self._meta_path(key), encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            return None
        return meta if isinstance(meta, dict) else None

    def _scan(self) -> list[dict[str, Any]]:
        """Live entries, most-recently-used first."""
        entries = []
        try:
            names = os.listdir(self.cache_dir)
        except OSError:
            return []
        for name in names:
            if not name.endswith(".json") or name == _STATS_FILENAME:
                continue
            key = name[:-len(".json")]
            meta = self._read_meta(key)
            wav = self._wav_path(key)
            if meta is None or not os.path.isfile(wav):
                continue
            try:
                size = os.path.getsize(wav)
            except OSError:
                continue
            entries.append({
                "key": key,
                "bytes": size,
                "last_used": float(meta.get("last_used", 0.0)),
                "model": str(meta.get("model", "")),
                "output_rate": meta.get("output_rate"),
                "output_samples": meta.get("output_samples"),
                "output_sha256": str(meta.get("output_sha256", "")),
            })
        entries.sort(key=lambda e: e["last_used"], reverse=True)
        return entries

    def lookup(self, key: str) -> dict[str, Any] | None:
        """The cached entry for ``key``, or None on a miss.

        A hit verifies the stored wav against the sidecar's SHA-256
        before returning it, and refreshes its recency. A corrupt
        entry raises :class:`CorruptCacheEntry` naming the key (its
        files are already removed, so the next lookup is a miss).
        """
        with self._lock:
            meta = self._read_meta(key)
            wav = self._wav_path(key)
            if meta is None or not os.path.isfile(wav):
                self._stats["misses"] += 1
                self._save_stats()
                return None
            actual = _sha256_file(wav)
            if actual != str(meta.get("output_sha256", "")):
                self._remove_entry(key)
                self._stats["corrupt"] += 1
                self._save_stats()
                raise CorruptCacheEntry(
                    f"cached conversion {key} is corrupt: stored wav "
                    f"sha256 {actual[:12]} != recorded "
                    f"{str(meta.get('output_sha256', ''))[:12]} — "
                    "entry removed, re-convert to replace it")
            meta["last_used"] = time.time()
            _atomic_write_json(self._meta_path(key), meta)
            self._stats["hits"] += 1
            self._stats["bytes_served"] += int(
                meta.get("output_bytes", 0) or 0)
            self._save_stats()
            return {"key": key, "path": wav,
                    "model": meta.get("model", ""),
                    "input_sha256": meta.get("input_sha256", ""),
                    "input_rate": meta.get("input_rate"),
                    "output_rate": meta.get("output_rate"),
                    "output_samples": meta.get("output_samples"),
                    "output_sha256": meta.get("output_sha256", ""),
                    "output_bytes": meta.get("output_bytes", 0)}

    def store(self, key: str, wav_path: str, *,
              model_identity: str, input_sha256: str, input_rate: int,
              params: Any, input_samples: int, output_samples: int,
              output_rate: int) -> dict[str, Any]:
        """Store a converted wav under ``key``; returns its receipt.

        Writes are atomic (tmp + replace). Storing enforces the
        eviction rule immediately, so the cache never grows past
        its caps.
        """
        with self._lock:
            os.makedirs(self.cache_dir, exist_ok=True)
            dest = self._wav_path(key)
            _atomic_copy(wav_path, dest)
            meta = {
                "key": key,
                "model": str(model_identity or "none"),
                "input_sha256": str(input_sha256),
                "input_rate": int(input_rate),
                "params": (params if isinstance(params, dict) else {
                    "model_rate": getattr(params, "model_rate", 0),
                    "output_rate": getattr(params, "output_rate", 0),
                    "direction": getattr(params, "direction", "A2B"),
                    "pitch_shift_semitones": getattr(
                        params, "pitch_shift_semitones", 0.0),
                    "skip_vocode": bool(
                        getattr(params, "skip_vocode", False)),
                    "discard_vocoded_audio": bool(
                        getattr(params, "discard_vocoded_audio", False)),
                    "n_fft": getattr(params, "n_fft", 1024),
                    "hop_length": getattr(params, "hop_length", 256),
                    "griffin_iters": getattr(params, "griffin_iters", 0),
                }),
                "input_samples": int(input_samples),
                "output_samples": int(output_samples),
                "output_rate": int(output_rate),
                "output_bytes": int(os.path.getsize(dest)),
                "output_sha256": _sha256_file(dest),
                "last_used": time.time(),
            }
            _atomic_write_json(self._meta_path(key), meta)
            self._stats["stores"] += 1
            self._enforce_caps_locked(keep=key)
            self._save_stats()
            return {"key": key, "path": dest,
                    "output_sha256": meta["output_sha256"],
                    "output_bytes": meta["output_bytes"]}

    def _remove_entry(self, key: str) -> bool:
        removed = False
        for path in (self._wav_path(key), self._meta_path(key)):
            try:
                os.remove(path)
                removed = True
            except OSError:
                pass
        return removed

    def _enforce_caps_locked(self, *, keep: str = "") -> int:
        """Evict least-recently-used entries past the caps.

        Returns the eviction count. The entry named by ``keep``
        (the one just stored) is never evicted to make room for
        itself.
        """
        evicted = 0
        while True:
            entries = self._scan()
            total = sum(e["bytes"] for e in entries)
            over_count = (self.max_entries > 0
                          and len(entries) > self.max_entries)
            over_bytes = (self.max_bytes > 0
                          and total > self.max_bytes)
            if not over_count and not over_bytes:
                return evicted
            victim = next((e for e in reversed(entries)
                           if e["key"] != keep), None)
            if victim is None:
                return evicted
            self._remove_entry(victim["key"])
            self._stats["evictions"] += 1
            evicted += 1

    # -- accounting -----------------------------------------------------

    def _load_stats(self) -> None:
        try:
            with open(self._stats_path(), encoding="utf-8") as fh:
                saved = json.load(fh)
        except (OSError, ValueError):
            return
        if isinstance(saved, dict):
            for name in self._stats:
                if isinstance(saved.get(name), int):
                    self._stats[name] = saved[name]

    def _save_stats(self) -> None:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            _atomic_write_json(self._stats_path(), dict(self._stats))
        except OSError:
            pass

    def status(self) -> dict[str, Any]:
        """The cache as data: caps, entries, bytes, accounting."""
        with self._lock:
            entries = self._scan()
            return {
                "dir": self.cache_dir,
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes,
                "entries": len(entries),
                "bytes": sum(e["bytes"] for e in entries),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "stores": self._stats["stores"],
                "evictions": self._stats["evictions"],
                "corrupt": self._stats["corrupt"],
                "bytes_served": self._stats["bytes_served"],
                "newest": [e["key"][:12] for e in entries[:5]],
            }

    def clear(self) -> int:
        """Drop every cached entry; reset the counters. Returns the
        entry count removed."""
        with self._lock:
            entries = self._scan()
            count = 0
            for entry in entries:
                if self._remove_entry(entry["key"]):
                    count += 1
            for name in ("hits", "misses", "stores", "evictions",
                         "corrupt", "bytes_served"):
                self._stats[name] = 0
            self._save_stats()
            return count


def default_conversion_cache() -> ConversionCache:
    """The shared result cache at the default dir."""
    return ConversionCache()


def cache_status_text(result: dict[str, Any] | None = None,
                      models: dict[str, Any] | None = None) -> str:
    """The cache report, as human-readable text.

    Both halves: converted audio on disk (entries, bytes, hits /
    misses, evictions, corrupt refusals, bytes served) and loaded
    models in memory (how many sit loaded, under what policy).
    """
    if result is None:
        result = default_conversion_cache().status()
    if models is None:
        models = shared_model_cache().status()
    lines = ["voice cache:"]
    lines.append(f"  dir: {result['dir']}")
    lines.append(f"  conversions: {result['entries']} entries, "
                 f"{result['bytes']} bytes "
                 f"(cap {result['max_entries']} entries / "
                 f"{result['max_bytes']} bytes)")
    lines.append(f"  hits: {result['hits']}  misses: {result['misses']}  "
                 f"stores: {result['stores']}  "
                 f"evicted: {result['evictions']}  "
                 f"corrupt: {result['corrupt']}")
    lines.append(f"  bytes served from cache: {result['bytes_served']}")
    torchscript = models.get("torchscript", {})
    lines.append(f"  loaded models: {torchscript.get('loaded', 0)} "
                 f"(max {torchscript.get('policy', {}).get('max_loaded', '?')})")
    for entry in torchscript.get("entries", []):
        lines.append(f"    {entry['key'][:60]} [{entry['state']}] "
                     f"loads={entry['loads']} "
                     f"evictions={entry['evictions']}")
    return "\n".join(lines)


# -- model identity + model-load cache (memory) ---------------------------


def executor_identity(executor: Any) -> str:
    """The cache identity of a conversion executor.

    TorchScript by file path + size + mtime (a replaced file is a
    new identity, so a stale model is never served); registry by
    service ref (+ services dir, so two stores never collide);
    no executor (skip-vocode passthrough) is ``none``.
    """
    if executor is None:
        return "none"
    path = getattr(executor, "path", "")
    if path:
        try:
            stat = os.stat(path)
            return (f"torchscript:{os.path.abspath(path)}:"
                    f"{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            return f"torchscript:{os.path.abspath(path)}:missing"
    ref = getattr(executor, "ref", "")
    if ref:
        services_dir = str(getattr(executor, "services_dir", "") or "")
        if not services_dir:
            from .services import default_services_dir
            services_dir = default_services_dir()
        return f"service:{services_dir}:{ref}"
    return f"unknown:{type(executor).__name__}"


class ModelLoadCache:
    """Loaded conversion models, shared across conversions.

    One :class:`~nomorals.voice.lifecycle.ModelLifecycleManager`
    owns TorchScript loads (LRU, failure-as-state, retry on the
    next acquire); one shared
    :class:`~nomorals.voice.services.ModelServer` per services dir
    owns registry loads, so its own LRU actually persists between
    conversions instead of being rebuilt per executor.
    """

    def __init__(self, *, max_loaded: int = 4) -> None:
        from .lifecycle import LifecyclePolicy, ModelLifecycleManager
        self._manager = ModelLifecycleManager(
            LifecyclePolicy(max_loaded=max_loaded))
        self._servers: dict[str, Any] = {}
        self._lock = threading.RLock()

    def load_torchscript(self, path: str, *, entry: str = "forward") -> Any:
        """The loaded TorchScript module for ``path`` (cached)."""
        from .vits._torch import require_torch
        stat = os.stat(path)
        key = (f"torchscript:{os.path.abspath(path)}:"
               f"{stat.st_size}:{stat.st_mtime_ns}")

        def _loader() -> Any:
            torch = require_torch(
                "loading a TorchScript conversion model")
            module = torch.jit.load(path, map_location="cpu")
            module.eval()
            return module

        return self._manager.acquire(
            key, _loader, size_bytes=int(stat.st_size))

    def server(self, services_dir: str = "") -> Any:
        """The shared ModelServer for a services dir."""
        from .services import ModelServer, ServiceRegistry
        with self._lock:
            key = services_dir or ""
            server = self._servers.get(key)
            if server is None:
                registry = (ServiceRegistry(services_dir)
                            if services_dir else ServiceRegistry())
                server = ModelServer(registry)
                self._servers[key] = server
            return server

    def load_service(self, ref: str, *, services_dir: str = "") -> Any:
        """The loaded module for a registered service (cached)."""
        return self.server(services_dir).load(ref)

    def status(self) -> dict[str, Any]:
        """Loaded models + per-store server state, as data."""
        with self._lock:
            return {
                "torchscript": self._manager.status(),
                "services": {d: s.status()
                             for d, s in self._servers.items()},
            }

    def evict_all(self) -> int:
        """Evict every loaded TorchScript model. Returns the count."""
        return self._manager.evict_all(reason="cache-clear")


_MODEL_CACHE: ModelLoadCache | None = None
_MODEL_CACHE_LOCK = threading.Lock()


def shared_model_cache() -> ModelLoadCache:
    """The process-wide model-load cache (one manager, one server
    per services dir)."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        with _MODEL_CACHE_LOCK:
            if _MODEL_CACHE is None:
                _MODEL_CACHE = ModelLoadCache()
    return _MODEL_CACHE
