"""Voice efficiency core — caching, latency tracking, smart routing.

1. **Segment cache**: never regenerate the same phrase+voice+backend twice.
   Keyed on (text_hash, voice_id, backend, emotion_hash). LRU with size cap.
2. **Latency table**: measured time-to-first-chunk per backend, persisted.
   Routing prefers the fastest backend that meets the quality bar.
3. **Resource budgets**: per-backend VRAM/RAM notes, quantized preferences.
"""

import hashlib
import json
import os
import time
from collections import OrderedDict
from array import array


def _cache_dir() -> str:
    d = os.path.expanduser("~/.nomorals/voice_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _latency_path() -> str:
    return os.path.join(_cache_dir(), "latency.json")


class SegmentCache:
    """LRU cache for synthesized segments.

    Key: sha256(text | voice_id | backend | emotion_params).
    Value: (samples_array, sample_rate). Persisted to disk.
    """

    def __init__(self, max_entries: int = 500, ttl_s: float = 0.0):
        self.max_entries = max_entries
        #: disk entries older than ttl_s are treated as misses (0 = forever)
        self.ttl_s = ttl_s
        self._mem: OrderedDict[str, tuple[array, int]] = OrderedDict()
        self.dir = os.path.join(_cache_dir(), "segments")
        os.makedirs(self.dir, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(text: str, voice_id: str, backend: str,
            emotion: str = "") -> str:
        h = hashlib.sha256()
        h.update(text.encode("utf-8"))
        h.update(b"\x00" + voice_id.encode("utf-8"))
        h.update(b"\x00" + backend.encode("utf-8"))
        h.update(b"\x00" + emotion.encode("utf-8"))
        return h.hexdigest()[:32]

    def _disk_path(self, key: str) -> str:
        return os.path.join(self.dir, f"{key}.wav")

    def get(self, key: str) -> tuple[array, int] | None:
        # Memory first
        if key in self._mem:
            self._mem.move_to_end(key)
            self.hits += 1
            return self._mem[key]
        # Disk fallback
        path = self._disk_path(key)
        if os.path.exists(path):
            if self.ttl_s > 0:
                try:
                    age = time.time() - os.path.getmtime(path)
                except OSError:
                    age = 0.0
                if age > self.ttl_s:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    self.misses += 1
                    return None
            try:
                import wave
                with wave.open(path, "rb") as f:
                    n = f.getnframes()
                    data = f.readframes(n)
                    import struct
                    samples = array("h", struct.unpack(f"<{n}h", data))
                    sr = f.getframerate()
                self._mem[key] = (samples, sr)
                if len(self._mem) > self.max_entries:
                    self._mem.popitem(last=False)
                self.hits += 1
                return samples, sr
            except Exception:
                pass
        self.misses += 1
        return None

    def put(self, key: str, samples: array, sample_rate: int) -> None:
        self._mem[key] = (samples, sample_rate)
        self._mem.move_to_end(key)
        if len(self._mem) > self.max_entries:
            self._mem.popitem(last=False)
        # Persist to disk (best-effort)
        try:
            import wave
            import struct
            with wave.open(self._disk_path(key), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(sample_rate)
                f.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        except Exception:
            pass

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "mem_entries": len(self._mem),
        }

    def invalidate(self, prefix: str = "") -> int:
        """Drop cached entries whose key starts with ``prefix``.

        Empty prefix clears the in-memory tier only (disk stays — it is
        content-addressed and cheap). Returns entries dropped.
        """
        if not prefix:
            n = len(self._mem)
            self._mem.clear()
            return n
        doomed = [k for k in self._mem if k.startswith(prefix)]
        for k in doomed:
            del self._mem[k]
            try:
                os.remove(self._disk_path(k))
            except OSError:
                pass
        return len(doomed)

    def clear(self) -> dict[str, int]:
        """Drop everything: memory + disk. Returns counts."""
        mem = len(self._mem)
        self._mem.clear()
        disk = 0
        try:
            for entry in os.listdir(self.dir):
                if entry.endswith(".wav"):
                    try:
                        os.remove(os.path.join(self.dir, entry))
                        disk += 1
                    except OSError:
                        pass
        except OSError:
            pass
        return {"mem_cleared": mem, "disk_cleared": disk}

    def warm(self, entries: list[tuple[str, "array", int]]) -> int:
        """Preload (key, samples, sample_rate) triples into memory."""
        n = 0
        for key, samples, sr in entries:
            self._mem[key] = (samples, sr)
            self._mem.move_to_end(key)
            n += 1
        while len(self._mem) > self.max_entries:
            self._mem.popitem(last=False)
        return n

    def size_report(self) -> dict:
        """Memory entries + disk footprint."""
        disk_files, disk_bytes = 0, 0
        try:
            for entry in os.listdir(self.dir):
                if entry.endswith(".wav"):
                    disk_files += 1
                    try:
                        disk_bytes += os.path.getsize(
                            os.path.join(self.dir, entry))
                    except OSError:
                        pass
        except OSError:
            pass
        return {
            "mem_entries": len(self._mem),
            "disk_files": disk_files,
            "disk_bytes": disk_bytes,
            "disk_mb": round(disk_bytes / 1048576, 2),
        }


class LatencyTable:
    """Measured time-to-first-chunk per backend.

    Persisted to disk. Routing uses this to pick the fastest
    backend that meets the quality bar for the task.
    """

    def __init__(self):
        self.path = _latency_path()
        self.table: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path) as f:
                self.table = json.load(f)
        except Exception:
            self.table = {}

    def _save(self) -> None:
        try:
            with open(self.path, "w") as f:
                json.dump(self.table, f)
        except Exception:
            pass

    def record(self, backend: str, ttfc_ms: float,
               realtime_factor: float = 0.0) -> None:
        """Record a measurement. Keeps rolling average of last 20."""
        entry = self.table.get(backend, {"samples": []})
        samples = entry["samples"][-19:] + [ttfc_ms]
        entry["samples"] = samples
        entry["avg_ms"] = sum(samples) / len(samples)
        entry["rtf"] = realtime_factor
        entry["updated"] = time.time()
        self.table[backend] = entry
        self._save()

    def get_avg(self, backend: str) -> float | None:
        entry = self.table.get(backend)
        return entry["avg_ms"] if entry else None

    def fastest(self, candidates: list[str],
                max_ms: float = 300.0) -> str | None:
        """Pick the fastest backend under the latency budget.

        Returns None if no candidate has measurements — caller
        should measure first.
        """
        best, best_ms = None, float("inf")
        for c in candidates:
            avg = self.get_avg(c)
            if avg is not None and avg < best_ms and avg <= max_ms:
                best, best_ms = c, avg
        return best

    def all(self) -> dict:
        return {k: {"avg_ms": v["avg_ms"], "rtf": v.get("rtf", 0)}
                for k, v in self.table.items()}


# Resource budgets per backend (honest, from mining + docs)
RESOURCE_BUDGETS = {
    "chatterbox-nano": {"vram_gb": 0.5, "ram_gb": 1.0,
                        "phone": True, "quantized": True},
    "chatterbox-turbo": {"vram_gb": 2.0, "ram_gb": 2.0,
                         "phone": False, "quantized": False},
    "chatterbox": {"vram_gb": 2.5, "ram_gb": 2.5,
                   "phone": False, "quantized": False},
    "piper": {"vram_gb": 0.0, "ram_gb": 0.3,
              "phone": True, "quantized": True},
    "kitten": {"vram_gb": 0.0, "ram_gb": 0.3,
               "phone": True, "quantized": True},
    "spark": {"vram_gb": 2.0, "ram_gb": 2.0,
              "phone": False, "quantized": False},
    "zonos": {"vram_gb": 6.0, "ram_gb": 8.0,
              "phone": False, "quantized": False},
    "kokoro": {"vram_gb": 0.5, "ram_gb": 0.5,
               "phone": True, "quantized": True},
    "dia": {"vram_gb": 6.2, "ram_gb": 8.0,
            "phone": False, "quantized": False},
    "cosyvoice3": {"vram_gb": 4.5, "ram_gb": 6.0,
                   "phone": False, "quantized": False},
    "xtts": {"vram_gb": 4.0, "ram_gb": 6.0,
             "phone": False, "quantized": False},
    "fish-s2": {"vram_gb": 4.0, "ram_gb": 6.0,
                "phone": False, "quantized": False},
    "bark": {"vram_gb": 4.0, "ram_gb": 8.0,
             "phone": False, "quantized": False},
}


def phone_viable_backends() -> list[str]:
    """Backends that genuinely work on the phone profile."""
    return [k for k, v in RESOURCE_BUDGETS.items() if v["phone"]]
