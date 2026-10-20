"""``nm doctor`` / ``models`` / ``setup`` — environment and model health."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from pathlib import Path
from ...llm.brain import brain_for
from ...version import __version__
from ...compat import feature_report, native_section, report_as_text
from ..emit import _emit



def _cmd_doctor(args: argparse.Namespace, settings: Any) -> int:
    if getattr(args, "build_native", False):
        from ... import native as _native
        ok, msg = _native.build()
        print(f"native build: {'OK' if ok else 'FAILED'} — {msg}")
        print()
    report = feature_report()
    # Native accelerator status: built/missing/stale-arch per kernel.  The
    # import lives here (cli may import L2 native) rather than in compat
    # (L0) — see test_layering.  Never let a broken native package break
    # doctor: fall back to an honest "unknown".
    try:
        from ... import native as _native
        report["native"] = native_section(_native.info())
    except Exception:  # noqa: BLE001 - doctor must always report
        report["native"] = {"overall": "unknown", "kernels": {},
                            "compiler": None, "fallback": "pure-python"}
    from ...storage.db import Database

    # Executor worker count, surfaced so a "why is this slow?" question has
    # a measured answer. Pure helper: no pool is constructed. Effective is
    # 0 when pools are off (termux profile) regardless of the formula.
    from ...agents.runtime import default_process_workers

    db = Database(settings.db_path)
    summary = db.migrate()
    health = {
        "version": __version__,
        "profile": settings.profile,
        "database": {
            "path": str(settings.db_path),
            "schema_version": summary.version,
            "tables": len(db.tables()),
            "integrity": db.integrity_check(),
        },
        "cpu_count": __import__("os").cpu_count(),
        "executor": {
            "effective_processes": (
                default_process_workers(settings.concurrency.processes or None)
                if settings.concurrency.use_processes
                else 0
            ),
            "use_processes": settings.concurrency.use_processes,
            "env_override": os.environ.get("NM_PROCESS_WORKERS") or None,
        },
    }
    db.close()
    # Tool modules that failed to register + config-key status are
    # computed BEFORE the report branches: JSON consumers need them too,
    # and the exit code is a contract — the EC2 deploy script treats a
    # nonzero `nm doctor` as VERIFY FAILED. Hard failures: a database
    # that fails its integrity check, tool modules that did not
    # register, and set-but-ignored config keys (§8.1: ignored must be
    # zero — a set-but-ignored NM_* key is a bug).
    _failed: list = []
    try:
        from ...tools.registry import ToolRegistry
        _probe = ToolRegistry().register_builtins()
        _failed = _probe.failed_modules()
    except Exception:  # noqa: BLE001 — doctor must always report
        _failed = []
    _status = None
    try:
        from ...core.config import config_env_status
        _status = config_env_status()
    except Exception:  # noqa: BLE001 — doctor must always report
        _status = None
    _hard_fail = (
        health["database"]["integrity"] != "ok"
        or bool(_failed)
        or bool(_status and _status["ignored"])
    )
    payload = {**health, **report,
               "tool_modules": {"failed": _failed},
               "config_keys": _status}
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 1 if _hard_fail else 0
    print(f"NoMorals Core {__version__}  profile={settings.profile}")
    print(f"database: {health['database']['path']}")
    print(f"  schema v{summary.version}, {health['database']['tables']} tables, "
          f"integrity {health['database']['integrity']}")
    print(f"executor: {health['executor']['effective_processes']} process workers, "
          f"pools {'on' if health['executor']['use_processes'] else 'off'}"
          + (f" (NM_PROCESS_WORKERS={health['executor']['env_override']})"
             if health['executor']['env_override'] else ""))
    print()
    print(report_as_text(report))
    # Surface tool modules that failed to register — a broken tool must
    # never vanish invisibly.
    if _failed:
        print(f"\n{len(_failed)} tool module(s) failed to register:")
        for entry in _failed:
            print(f"  ✗ {entry['module']}: {entry['error']}")
    # Config-key status (infra spec §8.1): set / unset / direct / ignored.
    if _status is not None:
        _ignored = _status["ignored"]
        print("\nconfig keys:")
        print(f"  set (via Settings): {len(_status['set_mapped'])}   "
              f"set (direct-read): {len(_status['set_direct'])}   "
              f"unset (defaults): {len(_status['unset_mapped'])}")
        if _ignored:
            print(f"  ✗ IGNORED (set but mapping to nothing — bug): "
                  f"{', '.join(_ignored)}")
        else:
            print("  ✓ ignored: 0 — every set NM_* key maps to real behavior")
    return 1 if _hard_fail else 0


def _env_update_home(context: Any, pairs: dict[str, str]) -> "Path":
    """Set KEY=VALUE lines in the ``.env`` the next boot actually reads
    (replace-or-append, one line per key; every other line — comments,
    commented-out keys, unrelated settings — is preserved).

    Boot reads the FIRST existing candidate of ``cwd/.env`` then
    ``$NM_HOME/.env`` (see ``core.config._env_file_candidates``), so the
    write must land in that same file: writing the home file while a
    cwd ``.env`` exists would be silently shadowed forever while the
    caller reports success. Falls back to the home ``.env`` when no
    candidate exists yet. The durable contract the next process reads
    on boot."""
    import os
    from pathlib import Path

    from ...core.config import _env_file_candidates

    home = Path(os.path.expanduser(os.environ.get("NM_HOME")
                                    or getattr(context.settings, "home",
                                               "~/.nomorals")))
    env_path = None
    for candidate in _env_file_candidates(dict(os.environ), str(home)):
        if candidate.is_file():
            env_path = candidate
            break
    if env_path is None:
        env_path = home / ".env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    remaining = dict(pairs)
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if key in remaining:
            lines[i] = f"{key}={remaining.pop(key)}"
    for key, value in remaining.items():
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_path


def _cmd_models(args: argparse.Namespace, context: Any) -> int:
    from ...llm.registry import ModelRegistry, search_catalog

    registry = ModelRegistry(context.db)
    if getattr(args, "promote_local", ""):
        path = str(args.promote_local).strip()
        lora = str(getattr(args, "lora", "") or "").strip()
        # Promoting a path that does not exist writes a boot contract
        # the next boot cannot serve (llama_cpp primary, no server,
        # model-less chats) while reporting success — refuse it the
        # way the setup wizard's GGUF branch refuses a missing file.
        if not Path(path).expanduser().is_file():
            _emit(args, {"ok": False, "local_model": path,
                         "error": "no such GGUF file"},
                  f"cannot promote {path}: no GGUF file there — "
                  "download or copy the model first, then re-run. "
                  "Nothing was written.")
            return 1
        update = {
            "NM_LLM_PROVIDER": "llama_cpp",
            "NM_LLM_LOCAL_MODEL": path,
            "NM_LLM_LOCAL_AUTO_START": "1",
        }
        if lora:
            update["NM_LLM_LOCAL_LORA"] = lora
        written = _env_update_home(context, update)
        try:
            context.settings.llm.provider = "llama_cpp"
            context.settings.llm.local_model = path
            if lora:
                context.settings.llm.local_lora = lora
        except Exception:  # noqa: BLE001 — the file is the durable truth
            pass
        payload = {"ok": True, "local_model": path, "lora": lora,
                   "provider": "llama_cpp"}
        lora_line = f"  lora on top: {lora}\n" if lora else ""
        _emit(args, payload,
              f"local model promoted: {path}\n" + lora_line +
              f"  provider -> llama_cpp, auto-start on — written to "
              f"{written}; the next boot runs entirely on it")
        return 0
    if args.activate:
        record = registry.activate(args.activate)
        _emit(args, {"activated": record.name}, f"activated {record.name}")
        return 0
    if args.catalog or args.search or args.kind:
        entries = search_catalog(
            args.search,
            kind=args.kind,
            max_params=args.max_params or None,
        )
        payload = [
            {
                "repo_id": e.repo_id, "family": e.family, "kind": e.kind,
                "params": e.params, "context": e.context_length,
                "size_gb": e.size_hint_gb, "license": e.license, "notes": e.notes,
            }
            for e in entries
        ]
        if args.json:
            print(json.dumps(payload, indent=2))
            return 0
        for entry in entries:
            print(f"{entry.repo_id}")
            print(f"    {entry.kind:<9} {entry.params/1e9:>5.1f}B  ctx {entry.context_length:>6}  "
                  f"~{entry.size_hint_gb}GB  {entry.license}")
            print(f"    {entry.notes}")
        return 0
    stats = registry.stats()
    rows = registry.list(limit=50)
    if args.json:
        print(json.dumps({"stats": stats, "models": [r.__dict__ for r in rows]}, indent=2, default=str))
        return 0
    print(f"active: {stats['active'] or '(none)'}   registered: {stats['total']}   "
          f"finetunes: {stats['finetunes']}")
    for record in rows:
        mark = "*" if record.active else " "
        print(f" {mark} {record.name:<52} {record.kind:<10} {record.source}")
    return 0


def _cmd_models_doctor(args: argparse.Namespace, context: Any) -> int:
    """Ping every provider on the router with a real round-trip.

    'Configured' is not 'working': each chat-capable provider gets a tiny
    live chat, vision floors get their health check. rc 0 only when a real
    model answers — a chain of dead providers or a misconfigured provider
    name exits 1 with the actionable line.
    """
    from ...llm.base import Message, SamplingParams

    router = getattr(context, "router", None)
    if router is None:
        print("doctor: no model router — the context did not build", file=sys.stderr)
        return 1
    names = list(router.providers())
    rows: list[dict[str, Any]] = []
    chat_ok = False
    for name in names:
        provider = router.get(name)
        if provider is None:
            continue
        entry: dict[str, Any] = {"provider": name, "health": False,
                                 "chat": None, "error": ""}
        try:
            entry["health"] = bool(provider.health())
        except Exception as exc:  # noqa: BLE001 — a failing check is the answer
            entry["error"] = f"health: {type(exc).__name__}: {exc}"[:160]
        if "chat" in provider.capabilities:
            try:
                resp = provider.chat([Message.user("ping")],
                                     SamplingParams(max_tokens=8))
                entry["chat"] = bool(resp.ok)
                if not resp.ok:
                    entry["error"] = (resp.error or "chat failed")[:160]
            except Exception as exc:  # noqa: BLE001
                entry["chat"] = False
                entry["error"] = f"{type(exc).__name__}: {exc}"[:160]
            chat_ok = chat_ok or bool(entry["chat"])
        rows.append(entry)

    lines = ["models doctor:"]
    for e in rows:
        if e["chat"]:
            state = "chat ok"
        elif e["chat"] is None:
            state = "ready (no chat: capability-only)" if e["health"] \
                else f"unavailable — {e['error'] or 'health check failed'}"
        else:
            state = f"FAIL — {e['error'] or 'chat failed'}"
        lines.append(f"  {e['provider']}: {state}")

    llm_cfg = getattr(getattr(context, "settings", None), "llm", None)
    wanted = str(getattr(llm_cfg, "provider", "") or "")
    configured_ok = wanted in names
    if wanted and not configured_ok:
        lines.append(f"  configured provider {wanted!r} did not register — "
                     "check NM_LLM_PROVIDER (or run nm setup)")
    if not chat_ok:
        lines.append("  no provider answered a chat: NM_LLM_ALLOW_MOCK_FALLBACK=1 "
                     "boots a scripted mock for offline demos")
    rc = 0 if (chat_ok and configured_ok) else 1
    _emit(args, {"providers": rows, "ok": rc == 0,
                 "configured": wanted}, "\n".join(lines))
    return rc


class _SetupAborted(Exception):
    """The wizard's stdin closed (EOF / Ctrl-C) before a choice was made."""


def _cmd_setup(args: argparse.Namespace, context: Any) -> int:
    """Guided model setup wizard."""
    try:
        return _cmd_setup_interactive(args, context)
    except _SetupAborted:
        # Aborts can only arise before the write step (the test prompt
        # handles its own EOF), so nothing has been written yet.
        print("\nSetup aborted — no configuration was written.")
        return 1


def _cmd_setup_interactive(args: argparse.Namespace, context: Any) -> int:
    import os
    from pathlib import Path

    def _ask(prompt: str = "") -> str:
        """input() that turns EOF / Ctrl-C into a clean abort.

        A wizard whose stdin closes mid-flow (piped input, Ctrl-D)
        must stop with a sentence, not an EOFError traceback — and
        must never fall through to writing a configuration the user
        did not finish choosing.
        """
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            raise _SetupAborted from None

    print("🧙 NoMorals AI - Guided Model Setup\n")
    print("This wizard will help you configure an LLM provider.\n")
    
    # Show current status
    active = context.router.active_model
    if active and active != "mock":
        print(f"Current active model: {active}")
        choice = _ask("Reconfigure? [y/N] ").strip().lower()
        if choice != "y":
            print("Keeping current configuration.")
            return 0
    else:
        print("No model configured (using mock provider).\n")
    
    # Provider selection
    print("Choose a provider:")
    print("  1. OpenRouter (free tier available, many models)")
    print("  2. Hugging Face (requires token, serverless inference)")
    print("  3. Local GGUF model (offline, requires download)")
    print("  4. OpenAI (requires paid API key)")
    print("  5. Skip (keep mock provider)\n")
    
    choice = _ask("Select [1-5]: ").strip()

    if choice == "5" or not choice:
        print("Skipping setup. You can run `nm setup` again later.")
        return 0
    if choice not in ("1", "2", "3", "4"):
        # Falling through with an empty env_vars used to print
        # "✅ Configuration written" over a write of nothing.
        print(f"❌ Unknown selection {choice!r} — enter 1-5. "
              "Nothing was written.")
        return 1

    # Pairs are written via _env_update_home at the end: the file the
    # next boot actually reads (cwd .env wins over the home .env), with
    # every unrelated line of the operator's file preserved.
    env_vars: dict[str, str] = {}

    if choice == "1":
        # OpenRouter setup
        print("\n📡 OpenRouter Setup")
        print("Get your API key at: https://openrouter.ai/keys\n")
        api_key = _ask("Enter your OpenRouter API key: ").strip()

        if not api_key:
            print("❌ No API key entered — nothing was written. "
                  "Re-run `nm setup` when you have the key.")
            return 1
        if not api_key.startswith("sk-or-"):
            print("⚠️  Warning: OpenRouter keys should start with 'sk-or-'")
        
        env_vars["NM_LLM_PROVIDER"] = "openrouter"
        # build_router's openrouter branch reads the openrouter_*
        # settings and SKIPS the provider entirely when the key is
        # empty — writing NM_OPENAI_API_KEY here (the old behavior)
        # left the install on the mock while reporting success.
        env_vars["NM_OPENROUTER_API_KEY"] = api_key
        
        # Model selection
        print("\nRecommended free models:")
        print("  1. nvidia/nemotron-3-ultra-550b-a55b:free (1M context)")
        print("  2. openai/gpt-oss-120b:free (general + tool use)")
        print("  3. google/gemma-4-31b-it:free (vision)")
        print("  4. Enter custom model ID\n")
        
        model_choice = _ask("Select [1-4]: ").strip()
        model_map = {
            "1": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "2": "openai/gpt-oss-120b:free",
            "3": "google/gemma-4-31b-it:free",
        }

        if model_choice in model_map:
            model_id = model_map[model_choice]
        else:
            model_id = _ask("Enter model ID: ").strip()

        env_vars["NM_OPENROUTER_MODEL"] = model_id

    elif choice == "2":
        # Hugging Face setup
        print("\n🤗 Hugging Face Setup")
        print("Get your token at: https://huggingface.co/settings/tokens\n")
        token = _ask("Enter your HF token: ").strip()

        if not token:
            print("❌ No token entered — nothing was written. "
                  "Re-run `nm setup` when you have the token.")
            return 1

        env_vars["HF_TOKEN"] = token
        env_vars["NM_LLM_PROVIDER"] = "hf_serverless"
        
        print("\nRecommended models:")
        print("  1. Sao10K/L3-8B-Stheno-v3.2")
        print("  2. Qwen/Qwen3-8B")
        print("  3. Enter custom model ID\n")
        
        model_choice = _ask("Select [1-3]: ").strip()
        model_map = {
            "1": "Sao10K/L3-8B-Stheno-v3.2",
            "2": "Qwen/Qwen3-8B",
        }

        if model_choice in model_map:
            model_id = model_map[model_choice]
        else:
            model_id = _ask("Enter model ID: ").strip()
        
        env_vars["NM_HF_MODEL"] = model_id
        
    elif choice == "3":
        # Local GGUF setup
        print("\n💻 Local GGUF Model Setup")
        print("You'll need to download a GGUF model file.\n")
        print("Recommended models:")
        print("  1. Phi-3.5-mini-instruct (3.8B, ~2.3GB Q4_K_M)")
        print("  2. Llama-3.2-3B-Instruct (3B, ~2GB Q4_K_M)")
        print("  3. Enter path to existing GGUF file\n")
        
        model_choice = _ask("Select [1-3]: ").strip()

        if model_choice == "3":
            model_path = _ask("Enter path to GGUF file: ").strip()
        else:
            print("\nTo download a model:")
            print("  1. Go to https://huggingface.co/models?search=gguf")
            print("  2. Download a Q4_K_M quantized model")
            print("  3. Place it in ~/.nomorals/models/")
            model_path = _ask("\nEnter path to downloaded GGUF file: ").strip()
        
        if not Path(model_path).exists():
            print(f"⚠️  File not found: {model_path}")
            return 1
        
        env_vars["NM_LLM_PROVIDER"] = "llama_cpp"
        # The router promotes a local model from llm.local_model
        # (NM_LLM_LOCAL_MODEL) and auto-starts the server when
        # NM_LLM_LOCAL_AUTO_START=1 — the same keys `nm models
        # --promote-local` writes. NM_LLAMA_CPP_MODEL is read nowhere.
        env_vars["NM_LLM_LOCAL_MODEL"] = model_path
        env_vars["NM_LLM_LOCAL_AUTO_START"] = "1"
        
    elif choice == "4":
        # OpenAI setup
        print("\n🤖 OpenAI Setup")
        print("Get your API key at: https://platform.openai.com/api-keys\n")
        api_key = _ask("Enter your OpenAI API key: ").strip()

        if not api_key:
            print("❌ No API key entered — nothing was written. "
                  "Re-run `nm setup` when you have the key.")
            return 1
        if not api_key.startswith("sk-"):
            print("⚠️  Warning: OpenAI keys should start with 'sk-'")

        env_vars["NM_LLM_PROVIDER"] = "openai"
        # The router passes llm.openai_api_key (NM_OPENAI_API_KEY) to
        # the provider; a bare OPENAI_API_KEY is never read, so the old
        # write left the provider keyless.
        env_vars["NM_OPENAI_API_KEY"] = api_key
        env_vars["NM_OPENAI_MODEL"] = "gpt-4o-mini"
        # llm.openai_base_url DEFAULTS to the local Ollama shim
        # (http://localhost:11434/v1) — without this line the booted
        # "openai" provider sends the owner's OpenAI key to localhost
        # with an OpenAI model name, and every chat fails there.
        env_vars["NM_OPENAI_BASE_URL"] = "https://api.openai.com/v1"

    # Write to the .env the next boot reads, preserving the rest of
    # the operator's file (comments, commented-out keys, other settings).
    written = _env_update_home(context, env_vars)

    print(f"\n✅ Configuration written to {written}")
    print("\nTo activate, restart NoMorals AI — the bot reads its .env "
          "only at startup, so the change applies on the next launch.")
    
    # Offer to test
    print("\nTest the configuration now? [Y/n]")
    try:
        test_choice = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        # The configuration IS written at this point; a closed stdin
        # just means "don't test", never an abort and never a crash.
        test_choice = "n"
        print("(no input available — skipping the test)")
    
    if test_choice != "n":
        print("\n🧪 Testing configuration...")
        try:
            # Reload settings
            from ...core.config import load_settings
            context.settings = load_settings()

            # Reinitialize the router the same way the boot path does —
            # LLMRouter() takes keyword-only tuning args, not settings; the
            # provider chain is assembled by build_router().
            from ...agents.context import build_router
            context.router = build_router(
                context.settings, context.bus, db=getattr(context, "db", None)
            )
            
            # Test with a simple prompt
            from ...llm.base import Message, SamplingParams
            response = brain_for(context).chat(
                [Message.user("Say 'Hello, I'm working!' in one sentence.")],
                SamplingParams(max_tokens=50),
            task_kind="chat")
            
            active = str(getattr(context.router, "active_model", "")
                         or "")
            if response.ok and response.text and active \
                    and active != "mock":
                print(f"\n✅ Model responded: {response.text}")
                print(f"Active model: {active}")
            elif response.ok and response.text:
                # A mock fallback answers happily — reporting that as
                # success would bless a configuration that never
                # registered the chosen provider.
                print("\n❌ The configured provider did not come up — "
                      "the router fell back to the scripted mock, so "
                      "this response is not from your provider.")
                print("Check the API key and model ID you just entered.")
            else:
                print(f"\n❌ Model failed: {response.error}")
                print("Check your API key and model configuration.")
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            print("Check your configuration and try again.")
    
    return 0
