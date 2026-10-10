# Infrastructure Audit — October 10, 2026

System-wide infrastructure improvement.
CORRECTED TARGET: build everything at FULL capability. Never design down to any
deployment constraint (phone, free tier, AWS, anything). The profile-aware
runtime (resource_profile: workstation, laptop, termux) gates what runs where —
that is its job, not the builder's. Max out every class.

## Surface catalog

### Core runtime (nomorals/core/)
| File | Size | Classes | Role |
|---|---|---|---|
| config.py | 1306L | 32 settings classes | All configuration; env-var mapping |
| policy.py | 1091L | Capability, Policy, PermissionGradient | Capability gating, execution policy |
| profile.py / profiles.py | 214L/160L | EnvironmentProfile | Runtime profiles (workstation/laptop/termux) |
| shutdown.py | 63L | — | Graceful shutdown |
| runtune.py | 293L | 1 | Runtime tuning |

### Tool registry
| File | Role |
|---|---|
| nomorals/tools/registry.py | ToolRegistry — registration, capability checks |

### Scheduler
| File | Role |
|---|---|
| nomorals/agents/scheduler.py | Agent-internal scheduler (JUST upgraded: resource-aware, execution policies) |
| nomorals/scheduler/scheduler.py | User-facing scheduler (cron, reminders, hooks) |

### Resource management
| File | Role |
|---|---|
| nomorals/os/resources.py | ResourceManager — CPU/memory/thermal/battery sampling |
| nomorals/power/ | monitor, scheduler, power-mode |

### Persistence
| File | Role |
|---|---|
| nomorals/storage/db.py | Database class, migrations, kv_store |

### Events + error systems (nomorals/core/)
| File | Size | Role |
|---|---|---|
| events.py | 343L | EventBus |
| error_intelligence.py | 653L | Error analysis, knowledge base |
| error_doctor.py | 610L | Error diagnosis/repair suggestions |
| errors.py | 370L | 27 error classes |
| self_heal.py | 690L | Self-healing |
| retry.py | 336L | CircuitBreaker, backoff |
| ratelimit.py | 336L | Token bucket, sliding window, semaphores |

### Observability (nomorals/core/)
| File | Size | Role |
|---|---|---|
| logging_setup.py | 283L | Logging, redaction |
| observability.py | 349L | Metrics, tracing, health |
| tasks.py | 607L | TaskGraph |
| clock.py | 182L | Clock abstractions |

## Work split (no file overlap)

- **A/core-runtime:** config.py, policy.py, profile.py, profiles.py, shutdown.py, runtune.py, tools/registry.py
- **B/scheduler:** agents/scheduler.py, scheduler/scheduler.py (verify upgrade, fill gaps)
- **C/resources:** os/resources.py, power/ (all)
- **D/persistence:** storage/db.py + migrations + kv_store
- **E/events-errors:** events.py, error_intelligence.py, error_doctor.py, errors.py, self_heal.py, retry.py, ratelimit.py
- **F/observability:** logging_setup.py, observability.py, tasks.py, clock.py

## Rules for all sections
1. Mine first — write mining findings before build commits.
2. Full capability — no "for later."
3. Commit section-by-section LOCALLY. Do NOT push (parent coordinates).
4. AWS t3.small is the target; profile-gate for phone.
5. Report: what improved, what's real, what's weak.
