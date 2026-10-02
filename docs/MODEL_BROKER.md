# Model broker, lifecycle & benchmarks

`nm models list|add|remove|benchmark|use|select` — pick models by **capability**,
not by name. All state lives in the normal `nomorals.db` (tables
`model_lifecycle`, `lifecycle_events`, `lifecycle_kv`, `model_benchmarks`).

## Pointing a local GGUF at the broker as the primary mind

Your own weights are first-class broker candidates. Example with the
codebeast 3.8B Q4_K_M (~2.3GB, the phone brain) and dolphin-8b-merged (PC/VPS):

```bash
# 1. register the file (provider=llama_cpp, capabilities={chat, code})
nm models add ~/models/codebeast-3.8b.Q4_K_M.gguf --quant Q4_K_M --context-len 131072
nm models add ~/models/dolphin-8b-merged.Q4_K_M.gguf --quant Q4_K_M --context-len 8192

# 2. walk it to verified (download is a no-op for local files; verify = sha256)
#    (happens automatically inside `use`, shown here for transparency)

# 3. take real local measurements (stored as source='synthetic')
nm models benchmark codebeast-3.8b.Q4_K_M

# 4. promote: persisted primary + boot contract, rollbackable
nm models use codebeast-3.8b.Q4_K_M
```

`use` does three things:

1. Walks the model to `verified` if it isn't there yet
   (`registered → downloaded → verified`).
2. `lifecycle.promote(id)` — persists it as the primary mind and pushes the
   previous primary onto a history stack. `ModelLifecycle.rollback()`
   restores it (survives restarts; `nm models list` shows `*` on the primary).
3. For local GGUFs, writes the same boot contract as
   `nm models --promote-local`: `NM_LLM_PROVIDER=llama_cpp`,
   `NM_LLM_LOCAL_MODEL=<path>`, `NM_LLM_LOCAL_AUTO_START=1` in the home
   `.env`, so the next boot actually serves the file via llama-server.

The broker's operator override then pins that card: every broker-routed call
for a capability it serves starts on the primary first, with the router's
existing `hf_serverless → groq → openrouter → local` failover chain as the
safety net underneath.

## How selection works

`ModelBroker.select(capability, task_kind="", constraints=None)`:

1. **Hard filter** — only cards that serve the capability (CODE/JUDGE fall
   back to any CHAT card).
2. **Operator override** — `promote(id)` / `nm models use <id>` always wins
   for capabilities the pinned card serves.
3. **Benchmark score** — 70% recent success rate + 30% latency consistency,
   plus a cross-candidate latency rank (fastest measured median wins).
4. **Trajectory success rate** — from the cognition trajectory store, if one
   is injected (`ModelBroker(trajectories=store)`); purely advisory, never
   required, never imported.
5. Soft prefs — `task_kind` specialisation (`code` prefers CODE cards),
   `local_only`/`prefer_local`, `min_context`, cost caps.

`select()` returns `None` when nothing qualifies — the router then keeps its
existing name-based chain; the broker never reduces what the router could do.

## Wiring the router

```python
from nomorals.llm.broker import ModelBroker
from nomorals.llm.router import LLMRouter

broker = ModelBroker()
broker.build_from_router(router)   # cards from live providers
router.set_broker(broker)          # chat/complete/vision consult the broker
router.set_broker(None)            # detach → original behaviour
```

The consult is best-effort and exception-proof: any broker failure leaves the
router's active provider untouched.

## Benchmarks

`BenchmarkDB.record(model_id, capability, latency_s, success, …)` stores real
measurements; `score(model_id, capability)` → 0..1 (0.5 when no data).
`seed_synthetic(db, model_id, capability, provider, prompts)` runs genuine
round-trips through a provider and stores them labelled `source='synthetic'`
— the only sanctioned offline seed. Live traffic should be recorded with
`source='live'` by the serving path (not yet wired — see weaknesses below).

## Python API

```python
from nomorals.llm.lifecycle import ModelLifecycle
from nomorals.llm.broker import ModelBroker
from nomorals.llm.capabilities import Capability, ModelCard

lc = ModelLifecycle("data/nomorals.db")
m = lc.add_gguf("~/models/codebeast-3.8b.Q4_K_M.gguf", context_len=131072)
lc.download(m.id); lc.verify(m.id); lc.load(m.id); lc.warm(m.id)
lc.promote(m.id)          # primary mind (persisted)
lc.rollback()             # restore previous primary

broker = ModelBroker()
broker.register(ModelCard(id="codebeast", provider="llama_cpp",
                          model_id="codebeast-3.8b.Q4_K_M.gguf",
                          capabilities={Capability.CHAT, Capability.CODE},
                          context_len=131072, local=True, quant="Q4_K_M"))
best = broker.select(Capability.CHAT, task_kind="code")
```
