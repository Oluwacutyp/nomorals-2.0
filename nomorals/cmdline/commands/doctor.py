"""``nm doctor`` / ``models`` / ``setup`` — environment and model health."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from pathlib import Path
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
    }
    db.close()
    payload = {**health, **report}
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print(f"NoMorals Core {__version__}  profile={settings.profile}")
    print(f"database: {health['database']['path']}")
    print(f"  schema v{summary.version}, {health['database']['tables']} tables, "
          f"integrity {health['database']['integrity']}")
    print()
    print(report_as_text(report))
    return 0


def _env_update_home(context: Any, pairs: dict[str, str]) -> "Path":
    """Set KEY=VALUE lines in the home ``.env`` (replace-or-append, one line
    per key). The durable contract the next process reads on boot."""
    import os
    from pathlib import Path

    home = Path(os.path.expanduser(os.environ.get("NM_HOME")
                                    or getattr(context.settings, "home",
                                               "~/.nomorals")))
    home.mkdir(parents=True, exist_ok=True)
    env_path = home / ".env"
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
        update = {
            "NM_LLM_PROVIDER": "llama_cpp",
            "NM_LLM_LOCAL_MODEL": path,
            "NM_LLM_LOCAL_AUTO_START": "1",
        }
        if lora:
            update["NM_LLM_LOCAL_LORA"] = lora
        _env_update_home(context, update)
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
              "  provider -> llama_cpp, auto-start on — written to the home "
              ".env; the next boot runs entirely on it")
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


def _cmd_setup(args: argparse.Namespace, context: Any) -> int:
    """Guided model setup wizard."""
    import os
    from pathlib import Path
    
    print("🧙 NoMorals AI - Guided Model Setup\n")
    print("This wizard will help you configure an LLM provider.\n")
    
    # Show current status
    active = context.router.active_model
    if active and active != "mock":
        print(f"Current active model: {active}")
        choice = input("Reconfigure? [y/N] ").strip().lower()
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
    
    choice = input("Select [1-5]: ").strip()
    
    if choice == "5" or not choice:
        print("Skipping setup. You can run `nm setup` again later.")
        return 0
    
    env_file = Path.home() / ".nomorals/.env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing .env
    env_vars = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()
    
    if choice == "1":
        # OpenRouter setup
        print("\n📡 OpenRouter Setup")
        print("Get your API key at: https://openrouter.ai/keys\n")
        api_key = input("Enter your OpenRouter API key: ").strip()
        
        if not api_key.startswith("sk-or-"):
            print("⚠️  Warning: OpenRouter keys should start with 'sk-or-'")
        
        env_vars["NM_LLM_PROVIDER"] = "openrouter"
        env_vars["NM_OPENAI_API_KEY"] = api_key
        env_vars["NM_OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        
        # Model selection
        print("\nRecommended free models:")
        print("  1. nvidia/nemotron-3-ultra-550b-a55b:free (1M context)")
        print("  2. openai/gpt-oss-120b:free (general + tool use)")
        print("  3. google/gemma-4-31b-it:free (vision)")
        print("  4. Enter custom model ID\n")
        
        model_choice = input("Select [1-4]: ").strip()
        model_map = {
            "1": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "2": "openai/gpt-oss-120b:free",
            "3": "google/gemma-4-31b-it:free",
        }
        
        if model_choice in model_map:
            model_id = model_map[model_choice]
        else:
            model_id = input("Enter model ID: ").strip()
        
        env_vars["NM_OPENAI_MODEL"] = model_id
        
    elif choice == "2":
        # Hugging Face setup
        print("\n🤗 Hugging Face Setup")
        print("Get your token at: https://huggingface.co/settings/tokens\n")
        token = input("Enter your HF token: ").strip()
        
        env_vars["HF_TOKEN"] = token
        env_vars["NM_LLM_PROVIDER"] = "hf_serverless"
        
        print("\nRecommended models:")
        print("  1. Sao10K/L3-8B-Stheno-v3.2")
        print("  2. Qwen/Qwen3-8B")
        print("  3. Enter custom model ID\n")
        
        model_choice = input("Select [1-3]: ").strip()
        model_map = {
            "1": "Sao10K/L3-8B-Stheno-v3.2",
            "2": "Qwen/Qwen3-8B",
        }
        
        if model_choice in model_map:
            model_id = model_map[model_choice]
        else:
            model_id = input("Enter model ID: ").strip()
        
        env_vars["NM_HF_MODEL"] = model_id
        
    elif choice == "3":
        # Local GGUF setup
        print("\n💻 Local GGUF Model Setup")
        print("You'll need to download a GGUF model file.\n")
        print("Recommended models:")
        print("  1. Phi-3.5-mini-instruct (3.8B, ~2.3GB Q4_K_M)")
        print("  2. Llama-3.2-3B-Instruct (3B, ~2GB Q4_K_M)")
        print("  3. Enter path to existing GGUF file\n")
        
        model_choice = input("Select [1-3]: ").strip()
        
        if model_choice == "3":
            model_path = input("Enter path to GGUF file: ").strip()
        else:
            print("\nTo download a model:")
            print("  1. Go to https://huggingface.co/models?search=gguf")
            print("  2. Download a Q4_K_M quantized model")
            print("  3. Place it in ~/.nomorals/models/")
            model_path = input("\nEnter path to downloaded GGUF file: ").strip()
        
        if not Path(model_path).exists():
            print(f"⚠️  File not found: {model_path}")
            return 1
        
        env_vars["NM_LLM_PROVIDER"] = "llama_cpp"
        env_vars["NM_LLAMA_CPP_MODEL"] = model_path
        
    elif choice == "4":
        # OpenAI setup
        print("\n🤖 OpenAI Setup")
        print("Get your API key at: https://platform.openai.com/api-keys\n")
        api_key = input("Enter your OpenAI API key: ").strip()
        
        if not api_key.startswith("sk-"):
            print("⚠️  Warning: OpenAI keys should start with 'sk-'")
        
        env_vars["NM_LLM_PROVIDER"] = "openai"
        env_vars["OPENAI_API_KEY"] = api_key
        env_vars["NM_OPENAI_MODEL"] = "gpt-4o-mini"
    
    # Write .env file
    with open(env_file, "w") as f:
        for key, value in sorted(env_vars.items()):
            f.write(f"{key}={value}\n")
    
    print(f"\n✅ Configuration written to {env_file}")
    print("\nTo activate, restart NoMorals AI or run:")
    print("  source ~/.nomorals/.env")
    
    # Offer to test
    print("\nTest the configuration now? [Y/n]")
    test_choice = input().strip().lower()
    
    if test_choice != "n":
        print("\n🧪 Testing configuration...")
        try:
            # Reload settings
            from ...core.config import load_settings
            context.settings = load_settings()
            
            # Reinitialize router
            from ...llm.router import LLMRouter
            context.router = LLMRouter(context.settings)
            
            # Test with a simple prompt
            from ...llm.base import Message, SamplingParams
            response = context.router.chat(
                [Message.user("Say 'Hello, I'm working!' in one sentence.")],
                SamplingParams(max_tokens=50),
            )
            
            if response.ok and response.text:
                print(f"\n✅ Model responded: {response.text}")
                print(f"Active model: {context.router.active_model}")
            else:
                print(f"\n❌ Model failed: {response.error}")
                print("Check your API key and model configuration.")
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            print("Check your configuration and try again.")
    
    return 0
