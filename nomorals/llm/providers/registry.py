"""Declarative provider registry — providers as data (Echelon Wave 4).

``build_provider()`` used to be a hardcoded if-chain: kind names, aliases,
module paths and constructor classes were wired into control flow, so
adding a provider meant editing the chain, and nothing outside the chain
could answer "what providers exist, how do they authenticate, what can
they do, which models do they walk?" without importing every backend.

Now the chain is a registry walk over :class:`ProviderSpec` entries —
one frozen dataclass per provider carrying its canonical id, aliases,
deferred class path, transport, **auth shape** (how the credential is
presented, which env keys resolve it, which constructor kwarg carries
it, whether construction refuses without one), declared capabilities,
default model, the models-roster hook, and its verification record (the
data behind ``llm/STATUS.md``).

Credentials never live in a spec: a spec names *where* a secret is
resolved from, never the secret.  Providers are constructed with
already-resolved secrets passed as constructor kwargs — the same
contract the if-chain had, now stated as data.

Adding a provider is a data-only change: append a spec (or call
:func:`register_provider_spec` at runtime) and ``build_provider``,
the conformance suite, and ``STATUS.md`` all see it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..base import LLMProvider

__all__ = [
    "AuthSpec",
    "ProviderSpec",
    "Verification",
    "build_provider",
    "canonical_kind",
    "expected_kinds",
    "get_provider_spec",
    "list_provider_specs",
    "register_provider_spec",
    "render_status_markdown",
    "require_provider_spec",
    "unregister_provider_spec",
]


@dataclass(frozen=True)
class AuthSpec:
    """How a provider's credential is shaped — never the credential."""

    #: Presentation: "bearer" (Authorization: Bearer), "x-api-key",
    #: "token" (provider-specific token kwarg), or "none".
    kind: str = "none"
    #: Env keys the constructor resolves the secret from, in order.
    env_keys: tuple[str, ...] = ()
    #: Constructor kwarg that carries the resolved secret ("" = none).
    kwarg: str = ""
    #: Construction refuses without a credential (Anthropic is the one
    #: incumbent that raises in its constructor when no key resolves).
    required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "env_keys": list(self.env_keys),
            "kwarg": self.kwarg,
            "required": self.required,
        }


@dataclass(frozen=True)
class Verification:
    """When and how this provider was last verified, honestly stated.

    ``status`` is one of ``live`` (real traffic / live probes),
    ``docs`` (vendor docs only), ``conformance`` (the shared offline
    conformance suite), or ``unverified`` (no record — stated, never
    dressed up).
    """

    status: str = "unverified"
    when: str = ""
    how: str = "no live verification recorded"

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "when": self.when, "how": self.how}


@dataclass(frozen=True)
class ProviderSpec:
    """One provider, as data.  See the module docstring."""

    #: Canonical kind — also the provider's ``name`` attribute.
    id: str
    #: Module holding the provider class (imported lazily, on build).
    module: str
    #: Class name inside ``module``.
    class_name: str
    aliases: tuple[str, ...] = ()
    #: Wire shape: "http-openai" | "http-native" | "http-hf" |
    #: "in-process" | "subprocess".
    transport: str = "http-openai"
    auth: AuthSpec = field(default_factory=AuthSpec)
    #: The capability set a default-constructed instance advertises.
    capabilities: frozenset[str] = frozenset({"chat"})
    #: Constructor's default model id ("" = none / env-resolved).
    default_model: str = ""
    default_base_url: str = ""
    #: Where the model roster lives: "module:CONSTANT" (a module-level
    #: table of {"id": ...} dicts or plain strings), "instance:method"
    #: (a live discovery method — network), or "" (no roster: an
    #: arbitrary endpoint has no safe guess).
    models_hook: str = ""
    #: Runs without leaving the operator's machine.
    local: bool = False
    notes: str = ""
    verification: Verification = field(default_factory=Verification)

    @property
    def provider_class(self) -> str:
        return f"{self.module}:{self.class_name}"

    def load_class(self) -> type[LLMProvider]:
        """Import the provider class (deferred until first build)."""
        module = importlib.import_module(self.module)
        cls = getattr(module, self.class_name)
        if not (isinstance(cls, type) and issubclass(cls, LLMProvider)):
            raise TypeError(
                f"provider spec {self.id!r}: {self.provider_class} is not "
                "an LLMProvider subclass"
            )
        return cls

    def build(self, **kwargs: Any) -> LLMProvider:
        return self.load_class()(**kwargs)

    def roster(self) -> list[str]:
        """Model ids from the module-level roster table, offline.

        Empty when the roster is instance-discovered or doesn't exist.
        """
        if not self.models_hook.startswith("module:"):
            return []
        module = importlib.import_module(self.module)
        table = getattr(module, self.models_hook.split(":", 1)[1], ())
        ids: list[str] = []
        for entry in table or ():
            if isinstance(entry, dict):
                if entry.get("id"):
                    ids.append(str(entry["id"]))
            elif isinstance(entry, str):
                ids.append(entry)
        return ids

    def roster_for(self, provider: LLMProvider) -> list[str]:
        """The offline model walk for one instance.

        Module tables resolve directly; otherwise the provider's own
        ``_fallback_models()`` roster (the ladderfix walk list) when it
        defines one.  Live-discovery hooks (Ollama ``list_models``,
        OpenRouter ``catalog``, HF ``fetch_catalog``) are never called
        here — they need the network by design.
        """
        table = self.roster()
        if table:
            return table
        hook = getattr(provider, "_fallback_models", None)
        if callable(hook):
            return [str(m) for m in hook() if str(m).strip()]
        return []

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "aliases": list(self.aliases),
            "provider_class": self.provider_class,
            "transport": self.transport,
            "auth": self.auth.to_dict(),
            "capabilities": sorted(self.capabilities),
            "default_model": self.default_model,
            "default_base_url": self.default_base_url,
            "models_hook": self.models_hook,
            "local": self.local,
            "notes": self.notes,
            "verification": self.verification.to_dict(),
        }


# ── the registry ─────────────────────────────────────────────────────────────

_SPECS: dict[str, ProviderSpec] = {}
_ALIASES: dict[str, str] = {}


def _normalize(kind: Any) -> str:
    return kind.strip().lower() if isinstance(kind, str) else ""


def register_provider_spec(spec: ProviderSpec, *, replace: bool = False) -> ProviderSpec:
    """Add one spec to the registry.  Data-only provider addition.

    Raises :class:`ValueError` when the id or any alias is already
    claimed (pass ``replace=True`` to deliberately re-seat an id).
    """
    if not isinstance(spec, ProviderSpec):
        raise TypeError(f"expected ProviderSpec, got {type(spec).__name__}")
    if not spec.id or spec.id != _normalize(spec.id):
        raise ValueError(f"provider id must be a normalized kind, got {spec.id!r}")
    names = [spec.id, *(_normalize(a) for a in spec.aliases)]
    if len(set(names)) != len(names):
        raise ValueError(f"provider {spec.id!r}: aliases collide with its own id")
    for name in names:
        owner = _ALIASES.get(name)
        if owner is not None and owner != spec.id and not replace:
            raise ValueError(
                f"provider kind {name!r} already claimed by {owner!r}")
    if spec.id in _SPECS and not replace:
        raise ValueError(f"provider {spec.id!r} already registered")
    if spec.id in _SPECS:  # replace: drop the old spec's alias claims
        old = _SPECS[spec.id]
        for name in [old.id, *(_normalize(a) for a in old.aliases)]:
            _ALIASES.pop(name, None)
    _SPECS[spec.id] = spec
    for name in names:
        _ALIASES[name] = spec.id
    return spec


def unregister_provider_spec(kind: str) -> ProviderSpec | None:
    """Remove a spec (and its aliases).  Returns the removed spec."""
    spec = _SPECS.pop(_normalize(kind), None)
    if spec is None:
        return None
    for name in [spec.id, *(_normalize(a) for a in spec.aliases)]:
        _ALIASES.pop(name, None)
    return spec


def canonical_kind(kind: Any) -> str | None:
    """The canonical id for a kind or alias, or None when unknown."""
    return _ALIASES.get(_normalize(kind))


def get_provider_spec(kind: Any) -> ProviderSpec | None:
    """The spec for a kind or alias, or None when unknown."""
    canonical = canonical_kind(kind)
    return _SPECS.get(canonical) if canonical else None


def list_provider_specs() -> list[ProviderSpec]:
    """Every registered spec, in registration order."""
    return list(_SPECS.values())


def expected_kinds() -> list[str]:
    """Canonical ids in registration order (the error-message list)."""
    return list(_SPECS.keys())


def require_provider_spec(kind: Any) -> ProviderSpec:
    """Resolve a kind to its spec, with the legacy chain's errors.

    Raises :class:`ValueError` for a missing/blank/non-string kind or an
    unknown one — never an ``AttributeError`` off ``None``.
    """
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(
            f"provider kind must be a non-empty string, got {kind!r}"
        )
    spec = get_provider_spec(kind)
    if spec is None:
        raise ValueError(
            f"unknown provider {_normalize(kind)!r}; expected "
            + " | ".join(expected_kinds())
        )
    return spec


def build_provider(kind: str, **kwargs: Any) -> LLMProvider:
    """Instantiate a provider by kind or alias — a registry walk.

    The provider module is imported lazily at build time, so an unused
    backend never costs an import and a missing optional dependency in
    one backend cannot break the others.
    """
    return require_provider_spec(kind).build(**kwargs)


# ── STATUS.md ────────────────────────────────────────────────────────────────

_STATUS_LABELS = {
    "live": "verified live",
    "docs": "docs-verified only",
    "conformance": "conformance suite",
    "unverified": "not verified",
}


def render_status_markdown(specs: Iterable[ProviderSpec] | None = None) -> str:
    """Render ``llm/STATUS.md`` from the registry.  Deterministic.

    Comparable data, not prose: one row per registered provider with
    its construction data and its verification record — which providers
    were verified live, when, and how; the rest say "not verified"
    plainly.
    """
    rows = list(specs) if specs is not None else list_provider_specs()
    lines = [
        "# LLM provider status",
        "",
        "Generated from the declarative provider registry",
        "(`nomorals/llm/providers/registry.py`) — do not edit by hand.",
        "Regenerate: `python3 -c \"from nomorals.llm.providers.registry import",
        "render_status_markdown; print(render_status_markdown())\"`",
        "",
        "| Provider | Class | Transport | Auth | Default model | Capabilities | Verified | When | How |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for spec in rows:
        auth = spec.auth
        auth_text = auth.kind
        if auth.env_keys:
            auth_text += " (" + ", ".join(auth.env_keys) + ")"
        if auth.required:
            auth_text += " [required]"
        ver = spec.verification
        lines.append(
            "| {id} | `{cls}` | {transport} | {auth} | {model} | {caps} "
            "| {status} | {when} | {how} |".format(
                id=spec.id,
                cls=spec.provider_class,
                transport=spec.transport,
                auth=auth_text,
                model=spec.default_model or "—",
                caps=", ".join(sorted(spec.capabilities)) or "—",
                status=_STATUS_LABELS.get(ver.status, ver.status),
                when=ver.when or "—",
                how=ver.how,
            )
        )
    lines += [
        "",
        "## Wave 4 notes",
        "",
        "- Construction is data-driven: `build_provider()` walks this",
        "  registry; adding a provider is a data-only change (one",
        "  `ProviderSpec`), proven by the registry tests.",
        "- Every registered provider must pass the shared conformance",
        "  suite (`tests/test_llm_provider_conformance.py`): construction",
        "  from spec, capability/roster honesty, a completion round-trip",
        "  against a mocked transport, typed failure classification",
        "  (`llm/failures.py: FailureClass`), the in-provider model",
        "  failover walk, and no-credential-leak assertions.",
        "- Owner-data egress is governed by the consent ledger",
        "  (`nomorals/core/consent.py`, bound to providers in",
        "  `nomorals/llm/consent.py`): a provider may receive a data kind",
        "  only under a recorded, expirable grant; a denied check raises",
        "  `CapabilityDenied` (code `policy.denied`). Local providers",
        "  (mock, ollama, llama_cpp, ocr) never take data off the",
        "  machine and need no grant.",
        "- Aliases are registry data too: agents/context.py's settings",
        "  chain canonicalizes kinds through this registry instead of",
        "  carrying its own alias sets.",
        "",
    ]
    return "\n".join(lines)


# ── the incumbent providers, as data ─────────────────────────────────────────
# Registration order is the legacy chain's error-message order:
# mock | ollama | groq | anthropic | deepseek | gemini | openrouter |
# openai_compat | hf_serverless | llama_cpp | ocr

register_provider_spec(ProviderSpec(
    id="mock",
    module="nomorals.llm.providers.mock",
    class_name="MockProvider",
    aliases=("offline", "test"),
    transport="in-process",
    auth=AuthSpec(kind="none"),
    capabilities=frozenset({"chat", "complete", "embed"}),
    default_model="mock-7b",
    local=True,
    notes="Deterministic offline provider; the test/offline floor.",
    verification=Verification(
        status="conformance", when="2026-11-05",
        how="shared conformance suite (tests/test_llm_provider_conformance.py); offline by construction",
    ),
))

register_provider_spec(ProviderSpec(
    id="ollama",
    module="nomorals.llm.providers.ollama",
    class_name="OllamaProvider",
    aliases=("ollama_native",),
    transport="http-native",
    auth=AuthSpec(kind="none"),
    capabilities=frozenset({"chat", "complete", "embed", "vision"}),
    default_model="",
    default_base_url="http://localhost:11434",
    models_hook="instance:list_models",
    local=True,
    notes="Ollama native /api/* surface plus model management.",
))

register_provider_spec(ProviderSpec(
    id="groq",
    module="nomorals.llm.providers.groq",
    class_name="GroqProvider",
    transport="http-openai",
    auth=AuthSpec(kind="bearer", env_keys=("GROQ_API_KEY", "NM_GROQ_API_KEY"),
                  kwarg="api_key"),
    capabilities=frozenset({"chat", "complete", "vision"}),
    default_model="openai/gpt-oss-120b",
    default_base_url="https://api.groq.com/openai/v1",
    models_hook="module:GROQ_FREE_MODELS",
    notes="Free tier; the operator models env (GROQ_MODELS on the "
          "defaults chain, NM_GROQ_MODELS via settings) overrides the "
          "walk roster via the fallback_models constructor kwarg.",
    verification=Verification(
        status="live", when="2026-11-04",
        how="live production traffic (2026-11-04 failover incident, tests/test_llm_ladderfix.py); roster verified against live /v1/models probes (Oct 2026)",
    ),
))

register_provider_spec(ProviderSpec(
    id="anthropic",
    module="nomorals.llm.providers.anthropic",
    class_name="AnthropicProvider",
    aliases=("claude",),
    transport="http-native",
    auth=AuthSpec(kind="x-api-key",
                  env_keys=("ANTHROPIC_API_KEY", "NM_ANTHROPIC_API_KEY"),
                  kwarg="api_key", required=True),
    capabilities=frozenset({"chat", "complete", "vision"}),
    default_model="claude-sonnet-4-6",
    default_base_url="https://api.anthropic.com/v1/messages",
    models_hook="module:KNOWN_MODELS",
    notes="Native Messages API; the constructor raises without a key.",
    verification=Verification(
        status="docs", when="2026-10",
        how="model ids verified against Anthropic's model docs (providers/anthropic.py); no live call recorded",
    ),
))

register_provider_spec(ProviderSpec(
    id="deepseek",
    module="nomorals.llm.providers.deepseek",
    class_name="DeepSeekProvider",
    transport="http-openai",
    auth=AuthSpec(kind="bearer",
                  env_keys=("DEEPSEEK_API_KEY", "NM_DEEPSEEK_API_KEY"),
                  kwarg="api_key"),
    capabilities=frozenset({"chat", "complete", "vision"}),
    default_model="deepseek-chat",
    default_base_url="https://api.deepseek.com/v1",
    models_hook="module:DEEPSEEK_MODELS",
))

register_provider_spec(ProviderSpec(
    id="gemini",
    module="nomorals.llm.providers.gemini",
    class_name="GeminiProvider",
    transport="http-openai",
    auth=AuthSpec(kind="bearer",
                  env_keys=("GEMINI_API_KEY", "NM_GEMINI_API_KEY"),
                  kwarg="api_key"),
    capabilities=frozenset({"chat", "complete", "vision", "embed"}),
    default_model="gemini-2.5-flash",
    default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    models_hook="module:GEMINI_MODELS",
))

register_provider_spec(ProviderSpec(
    id="openrouter",
    module="nomorals.llm.providers.openrouter",
    class_name="OpenRouterProvider",
    transport="http-openai",
    auth=AuthSpec(kind="bearer",
                  env_keys=("OPENROUTER_API_KEY", "NM_OPENROUTER_API_KEY"),
                  kwarg="api_key"),
    capabilities=frozenset({"chat", "complete", "embed", "vision"}),
    default_model="openrouter/free",
    default_base_url="https://openrouter.ai/api/v1",
    models_hook="instance:catalog",
    notes="Free roster rotates; discovered live via catalog(). "
          "The operator models env (OPENROUTER_MODELS on the defaults "
          "chain, NM_OPENROUTER_MODELS via settings) overrides the walk "
          "roster via the fallback_models constructor kwarg.",
))

register_provider_spec(ProviderSpec(
    id="openai_compat",
    module="nomorals.llm.providers.openai_compat",
    class_name="OpenAICompatProvider",
    aliases=("openai", "vllm", "lmstudio", "ollama_compat"),
    transport="http-openai",
    auth=AuthSpec(kind="bearer", kwarg="api_key"),
    capabilities=frozenset({"chat", "complete", "embed", "vision"}),
    default_model="",
    default_base_url="http://localhost:11434/v1",
    notes="Generic /v1/chat/completions endpoint (vLLM, LM Studio, "
          "Ollama's shim, gateways). No roster: an arbitrary server "
          "has no safe model guess.",
))

register_provider_spec(ProviderSpec(
    id="hf_serverless",
    module="nomorals.llm.providers.hf_serverless",
    class_name="HFServerlessProvider",
    aliases=("hf", "huggingface", "hf_endpoint"),
    transport="http-hf",
    auth=AuthSpec(kind="token", env_keys=("HF_TOKEN", "NM_HF_TOKEN"),
                  kwarg="token"),
    capabilities=frozenset({"chat", "complete", "embed", "vision"}),
    default_model="meta-llama/Llama-3.1-8B-Instruct",
    default_base_url="https://api-inference.huggingface.co",
    models_hook="instance:fetch_catalog",
    notes="HF router / Inference Endpoints; heals across catalog "
          "models on failure.",
    verification=Verification(
        status="live", when="2026-10",
        how="default model verified live on the HF router catalog (llm/defaults.py)",
    ),
))

register_provider_spec(ProviderSpec(
    id="llama_cpp",
    module="nomorals.llm.providers.llama_cpp",
    class_name="LlamaCppProvider",
    aliases=("llamacpp", "gguf"),
    transport="http-openai",
    auth=AuthSpec(kind="bearer", kwarg="api_key"),
    capabilities=frozenset({"chat", "complete", "embed", "grammar"}),
    default_model="local",
    default_base_url="http://localhost:8080",
    local=True,
    notes="llama-server on the operator's own hardware; keyless by "
          "default, optional bearer passed through.",
))

register_provider_spec(ProviderSpec(
    id="ocr",
    module="nomorals.llm.providers.ocr",
    class_name="OCRProvider",
    aliases=("tesseract",),
    transport="subprocess",
    auth=AuthSpec(kind="none"),
    capabilities=frozenset({"ocr"}),
    default_model="tesseract",
    local=True,
    notes="Offline vision floor: tesseract over image bytes. chat() "
          "refuses by design (a typed ProviderError).",
    verification=Verification(
        status="unverified", when="",
        how="no live verification recorded; tesseract binary absent on EC2 (INTEGRATION_PLAN §6.3)",
    ),
))
