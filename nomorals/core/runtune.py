"""Profile-aware runtime tuning (wave 86).

One question, answered once, applied everywhere: *what kind of machine is
this, and what does that mean for every resource knob?*

``nomorals.core.profile`` (moved from nomorals.workspace.profile, wave 84) detects the environment —
termux/mobile, embedded, pc, vps, workstation — and carries a VCPU
envelope. This module is the *runtime* half of that: it turns the detected
profile (plus the actual CPU/RAM) into a complete, inspectable set of
effective values for

* concurrency            — thread pool, process pools
* workspace              — VCPU min/target/max
* downloads              — parallelism, per-file size cap, HTTP timeout
* memory pressure        — context budget, autoscaler posture
* social runtime         — parallel chat workers
* background missions    — concurrency + autonomy aggressiveness (0..1)
* model/provider pref    — small local | HF router | full local | any

Resolution order per knob (first non-auto wins):

1. explicit ``runtime.*`` override in settings/env/config
2. the NAMED config profile (``profile = workstation|laptop|termux``) when
   it sets that knob
3. the DETECTED environment profile (auto-tuned by CPU/RAM)

Everything is a pure function of ``settings`` so it is trivially testable,
and the result carries its provenance (``notes``) for ``nm profile``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .profile import EnvironmentProfile, resolve_profile
from .config import _settings_to_dict

__all__ = ["RuntimeTune", "build_tune", "KNOWN_KINDS"]

KNOWN_KINDS = ("termux", "mobile", "embedded", "vps", "pc", "workstation")

# kind → base knobs. CPU/RAM scaling happens on top of these in build_tune.
_BASE: dict[str, dict[str, Any]] = {
    # phone: RAM is precious, fork() is unreliable, screens are small
    "termux": {
        "threads": 4, "use_processes": False,
        "max_parallel_chats": 2,
        "max_concurrent_downloads": 1, "max_download_mb": 25.0,
        "max_upload_mb": 25.0, "http_timeout": 45.0,
        "context_budget_tokens": 4000, "memory_pressure": "aggressive",
        "mission_max_concurrent": 1, "mission_autonomy": 0.5,
        "model_pref": "small_local",
    },
    "mobile": {
        "threads": 4, "use_processes": False,
        "max_parallel_chats": 2,
        "max_concurrent_downloads": 1, "max_download_mb": 25.0,
        "max_upload_mb": 25.0, "http_timeout": 45.0,
        "context_budget_tokens": 4000, "memory_pressure": "aggressive",
        "mission_max_concurrent": 1, "mission_autonomy": 0.5,
        "model_pref": "small_local",
    },
    # Pi-class / very constrained
    "embedded": {
        "threads": 2, "use_processes": False,
        "max_parallel_chats": 1,
        "max_concurrent_downloads": 1, "max_download_mb": 10.0,
        "max_upload_mb": 10.0, "http_timeout": 60.0,
        "context_budget_tokens": 2500, "memory_pressure": "aggressive",
        "mission_max_concurrent": 1, "mission_autonomy": 0.25,
        "model_pref": "none",
    },
    # small cloud box
    "vps": {
        "threads": 8, "use_processes": True,
        "max_parallel_chats": 3,
        "max_concurrent_downloads": 2, "max_download_mb": 150.0,
        "max_upload_mb": 100.0, "http_timeout": 30.0,
        "context_budget_tokens": 8000, "memory_pressure": "moderate",
        "mission_max_concurrent": 2, "mission_autonomy": 0.75,
        "model_pref": "hf_router",
    },
    # ordinary laptop/desktop
    "pc": {
        "threads": 16, "use_processes": True,
        "max_parallel_chats": 4,
        "max_concurrent_downloads": 4, "max_download_mb": 300.0,
        "max_upload_mb": 100.0, "http_timeout": 30.0,
        "context_budget_tokens": 8000, "memory_pressure": "moderate",
        "mission_max_concurrent": 2, "mission_autonomy": 1.0,
        "model_pref": "local_full",
    },
    # bare metal / big cloud
    "workstation": {
        "threads": 32, "use_processes": True,
        "max_parallel_chats": 4,
        "max_concurrent_downloads": 8, "max_download_mb": 1000.0,
        "max_upload_mb": 500.0, "http_timeout": 30.0,
        "context_budget_tokens": 12000, "memory_pressure": "relaxed",
        "mission_max_concurrent": 4, "mission_autonomy": 1.0,
        "model_pref": "local_full",
    },
}

#: knob → (dotted path in settings, LIBRARY DEFAULT). A value in settings
#: that differs from the default means *someone* set it deliberately
#: (config.toml, .env, or a named profile — load_settings has already
#: merged those in), so it wins over auto-tuning.
_CONFIG_SOURCES: dict[str, tuple[str, Any]] = {
    "threads": ("concurrency.threads", 8),
    "use_processes": ("concurrency.use_processes", True),
    "vcpu_min": ("workspace.min_vcpus", 0),
    "vcpu_target": ("workspace.vcpus", 0),
    "vcpu_max": ("workspace.max_vcpus", 0),
    "context_budget_tokens": ("memory.context_budget_tokens", 6000),
    "max_parallel_chats": ("partner.max_parallel_chats", 4),
    "max_concurrent_downloads": ("tools.max_concurrent_downloads", 4),
    "max_upload_mb": ("tools.max_upload_mb", 100),
    "http_timeout": ("tools.http_timeout", 30.0),
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _resolve_named(named: dict[str, Any], dotted: tuple[str, ...]) -> Any:
    """Walk a dotted path inside a nested dict; None when not set."""
    node: Any = named
    for part in dotted:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


@dataclass(frozen=True)
class RuntimeTune:
    """The effective runtime configuration for this environment."""

    profile: EnvironmentProfile
    # concurrency / workspace
    threads: int
    use_processes: bool
    vcpu_min: int
    vcpu_target: int
    vcpu_max: int
    # downloads
    max_concurrent_downloads: int
    max_download_mb: float
    max_upload_mb: float
    http_timeout: float
    # memory
    context_budget_tokens: int
    memory_pressure: str
    # social
    max_parallel_chats: int
    # background missions
    mission_max_concurrent: int
    mission_autonomy: float
    # model/provider preference (advisory)
    model_pref: str
    #: provenance: "knob = value (source)" lines, for `nm profile`
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "threads": self.threads,
            "use_processes": self.use_processes,
            "vcpu": {"min": self.vcpu_min, "target": self.vcpu_target, "max": self.vcpu_max},
            "downloads": {
                "max_concurrent": self.max_concurrent_downloads,
                "max_mb": self.max_download_mb,
                "max_upload_mb": self.max_upload_mb,
                "http_timeout": self.http_timeout,
            },
            "memory": {
                "context_budget_tokens": self.context_budget_tokens,
                "pressure": self.memory_pressure,
            },
            "social": {"max_parallel_chats": self.max_parallel_chats},
            "missions": {
                "max_concurrent": self.mission_max_concurrent,
                "autonomy": self.mission_autonomy,
            },
            "model_pref": self.model_pref,
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        """One-line human summary for ``nm profile``."""
        return (
            f"{self.profile.kind} (cloud: {self.profile.cloud}): "
            f"{self.threads} threads, "
            f"vcpus {self.vcpu_min}/{self.vcpu_target}/{self.vcpu_max}, "
            f"{self.max_concurrent_downloads} parallel downloads, "
            f"ctx {self.context_budget_tokens} tokens, "
            f"pressure {self.memory_pressure}, "
            f"missions {self.mission_max_concurrent}x @ "
            f"autonomy {self.mission_autonomy:.2f}, "
            f"model_pref {self.model_pref}"
        )

    def describe(self, *, color: bool | None = None,
                 theme: Any = None) -> str:
        """A styled multi-section runtime report for ``nm profile``."""
        from .style import paint, supports_color, kv_lines, header

        if color is None:
            color = supports_color()
        lines = [header(f"runtime tune — {self.profile.kind}", theme,
                        color=color), ""]
        sections = [
            ("concurrency", {
                "threads": self.threads,
                "process pool": "yes" if self.use_processes else "no",
                "vcpus (min/target/max)":
                    f"{self.vcpu_min}/{self.vcpu_target}/{self.vcpu_max}",
            }),
            ("downloads", {
                "parallel": self.max_concurrent_downloads,
                "max download": f"{self.max_download_mb:g} MB",
                "max upload": f"{self.max_upload_mb:g} MB",
                "http timeout": f"{self.http_timeout:g}s",
            }),
            ("memory", {
                "context budget": f"{self.context_budget_tokens:,} tokens",
                "pressure": self.memory_pressure,
            }),
            ("workload", {
                "parallel chats": self.max_parallel_chats,
                "missions": f"{self.mission_max_concurrent}x",
                "mission autonomy": f"{self.mission_autonomy:.2f}",
                "model preference": self.model_pref,
            }),
        ]
        for title, kv in sections:
            lines.append(paint(title.upper(), "label", theme, color=color))
            lines.extend(kv_lines(kv, theme, color=color))
            lines.append("")
        if self.notes:
            lines.append(paint("PROVENANCE", "label", theme, color=color))
            lines.extend(f"  {paint(n, 'muted', theme, color=color)}"
                         for n in self.notes)
        return "\n".join(lines).rstrip()


def build_tune(settings: Any, *, profile: EnvironmentProfile | None = None) -> RuntimeTune:
    """Compute the effective runtime tuning for ``settings``.

    Pure: same settings → same tune. ``profile`` may be pinned for tests.
    """
    rt = getattr(settings, "runtime", None)
    prof = profile or resolve_profile(getattr(rt, "profile", "") if rt else "")
    base = dict(_BASE.get(prof.kind, _BASE["pc"]))
    notes: list[str] = []
    notes.append(f"environment: {prof.detail} (cloud: {prof.cloud})")

    def resolve(knob: str, scale: Any | None = None) -> Any:
        """explicit runtime.* → deliberate settings value → auto (CPU/RAM)."""
        explicit = getattr(rt, knob, None) if rt is not None else None
        if explicit not in (None, "", 0, 0.0):
            notes.append(f"{knob} = {explicit} (explicit runtime.{knob})")
            return explicit
        source = _CONFIG_SOURCES.get(knob)
        if source is not None:
            path, default = source
            value = _resolve_named(_settings_to_dict(settings), path.split("."))
            if value is not None and value != default:
                notes.append(f"{knob} = {value} (settings {path})")
                return value
        value = scale(base[knob], cpu, mem_mb) if scale else base[knob]
        notes.append(f"{knob} = {value} (auto: {prof.detail})")
        return value

    cpu = max(1, prof.cpu or os.cpu_count() or 2)
    mem_mb = prof.memory_mb

    def _scale_threads(base_threads: float, cpu: int, mem: int) -> int:
        # Scale by the LIMITING resource: a 64-core box with 4 GB of RAM
        # must not get the thread pool of a 64-core/256 GB workstation.
        cpu_f = max(1.0, cpu / 8.0)
        ram_f = max(1.0, (mem or 8192) / 8192.0)
        return int(_clamp(base_threads * min(cpu_f, ram_f), 2, 32))

    threads = int(resolve("threads", _scale_threads))
    use_processes = bool(resolve("use_processes"))

    def _vcpu(knob: str, env_value: int) -> int:
        """VCPUs: explicit → settings → the profile's own envelope."""
        explicit = getattr(rt, knob, None) if rt is not None else None
        if explicit not in (None, "", 0, 0.0):
            notes.append(f"{knob} = {int(explicit)} (explicit runtime.{knob})")
            return int(explicit)
        path, _default = _CONFIG_SOURCES[knob]
        value = _resolve_named(_settings_to_dict(settings), path.split("."))
        if value:  # 0 = unset
            notes.append(f"{knob} = {int(value)} (settings {path})")
            return int(value)
        notes.append(f"{knob} = {env_value} (auto: profile envelope)")
        return int(env_value)

    # the detected profile's envelope is the authority for VCPUs: a
    # deliberately-pinned value still has to fit the machine
    vcpu_min, vcpu_target, vcpu_max = (
        _vcpu("vcpu_min", prof.min_vcpus),
        _vcpu("vcpu_target", prof.target_vcpus),
        _vcpu("vcpu_max", prof.max_vcpus),
    )
    vcpu_min, vcpu_target, vcpu_max = (
        max(1, min(vcpu_min, prof.min_vcpus)),
        max(vcpu_min, min(vcpu_target, prof.target_vcpus)),
        max(vcpu_target, min(vcpu_max, prof.max_vcpus)),
    )

    max_concurrent_downloads = int(resolve("max_concurrent_downloads"))
    max_download_mb = float(resolve("max_download_mb"))
    max_upload_mb = float(resolve("max_upload_mb"))
    http_timeout = float(resolve("http_timeout"))
    context_budget_tokens = int(resolve("context_budget_tokens"))
    max_parallel_chats = int(resolve("max_parallel_chats"))

    # memory pressure: explicit → RAM-based (phones are always tight)
    pressure = str(resolve("memory_pressure"))
    if pressure == base.get("memory_pressure") and mem_mb:
        if mem_mb <= 2 * 1024 or prof.kind in ("termux", "mobile", "embedded"):
            pressure = "aggressive"
        elif mem_mb <= 8 * 1024:
            pressure = "moderate"
        else:
            pressure = "relaxed"
        if pressure != base.get("memory_pressure"):
            notes.append(f"memory_pressure = {pressure} (auto: {mem_mb // 1024} GB RAM)")

    mission_max_concurrent = int(resolve("mission_max_concurrent"))
    mission_autonomy = float(resolve("mission_autonomy"))
    model_pref = str(resolve("model_pref"))

    # sanity floors/ceilings
    threads = int(_clamp(threads, 1, 128))
    max_concurrent_downloads = int(_clamp(max_concurrent_downloads, 1, 32))
    max_parallel_chats = int(_clamp(max_parallel_chats, 1, 16))
    mission_max_concurrent = int(_clamp(mission_max_concurrent, 1, 16))
    mission_autonomy = float(_clamp(mission_autonomy, 0.0, 1.0))

    return RuntimeTune(
        profile=prof,
        threads=threads,
        use_processes=use_processes,
        vcpu_min=vcpu_min,
        vcpu_target=vcpu_target,
        vcpu_max=vcpu_max,
        max_concurrent_downloads=max_concurrent_downloads,
        max_download_mb=max_download_mb,
        max_upload_mb=max_upload_mb,
        http_timeout=http_timeout,
        context_budget_tokens=context_budget_tokens,
        memory_pressure=str(pressure),
        max_parallel_chats=max_parallel_chats,
        mission_max_concurrent=mission_max_concurrent,
        mission_autonomy=mission_autonomy,
        model_pref=str(model_pref),
        notes=tuple(notes),
    )



