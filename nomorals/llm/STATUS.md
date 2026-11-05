# LLM provider status

Generated from the declarative provider registry
(`nomorals/llm/providers/registry.py`) — do not edit by hand.
Regenerate: `python3 -c "from nomorals.llm.providers.registry import
render_status_markdown; print(render_status_markdown())"`

| Provider | Class | Transport | Auth | Default model | Capabilities | Verified | When | How |
|---|---|---|---|---|---|---|---|---|
| mock | `nomorals.llm.providers.mock:MockProvider` | in-process | none | mock-7b | chat, complete, embed | conformance suite | 2026-11-05 | shared conformance suite (tests/test_llm_provider_conformance.py); offline by construction |
| ollama | `nomorals.llm.providers.ollama:OllamaProvider` | http-native | none | — | chat, complete, embed, vision | not verified | — | no live verification recorded |
| groq | `nomorals.llm.providers.groq:GroqProvider` | http-openai | bearer (GROQ_API_KEY, NM_GROQ_API_KEY) | openai/gpt-oss-120b | chat, complete, vision | verified live | 2026-11-04 | live production traffic (2026-11-04 failover incident, tests/test_llm_ladderfix.py); roster verified against live /v1/models probes (Oct 2026) |
| anthropic | `nomorals.llm.providers.anthropic:AnthropicProvider` | http-native | x-api-key (ANTHROPIC_API_KEY, NM_ANTHROPIC_API_KEY) [required] | claude-sonnet-4-6 | chat, complete, vision | docs-verified only | 2026-10 | model ids verified against Anthropic's model docs (providers/anthropic.py); no live call recorded |
| deepseek | `nomorals.llm.providers.deepseek:DeepSeekProvider` | http-openai | bearer (DEEPSEEK_API_KEY, NM_DEEPSEEK_API_KEY) | deepseek-chat | chat, complete, vision | not verified | — | no live verification recorded |
| gemini | `nomorals.llm.providers.gemini:GeminiProvider` | http-openai | bearer (GEMINI_API_KEY, NM_GEMINI_API_KEY) | gemini-2.5-flash | chat, complete, embed, vision | not verified | — | no live verification recorded |
| openrouter | `nomorals.llm.providers.openrouter:OpenRouterProvider` | http-openai | bearer (OPENROUTER_API_KEY, NM_OPENROUTER_API_KEY) | openrouter/free | chat, complete, embed, vision | not verified | — | no live verification recorded |
| openai_compat | `nomorals.llm.providers.openai_compat:OpenAICompatProvider` | http-openai | bearer | — | chat, complete, embed, vision | not verified | — | no live verification recorded |
| hf_serverless | `nomorals.llm.providers.hf_serverless:HFServerlessProvider` | http-hf | token (HF_TOKEN, NM_HF_TOKEN) | meta-llama/Llama-3.1-8B-Instruct | chat, complete, embed, vision | verified live | 2026-10 | default model verified live on the HF router catalog (llm/defaults.py) |
| llama_cpp | `nomorals.llm.providers.llama_cpp:LlamaCppProvider` | http-openai | bearer | local | chat, complete, embed, grammar | not verified | — | no live verification recorded |
| ocr | `nomorals.llm.providers.ocr:OCRProvider` | subprocess | none | tesseract | ocr | not verified | — | no live verification recorded; tesseract binary absent on EC2 (INTEGRATION_PLAN §6.3) |

## Wave 4 notes

- Construction is data-driven: `build_provider()` walks this
  registry; adding a provider is a data-only change (one
  `ProviderSpec`), proven by the registry tests.
- Every registered provider must pass the shared conformance
  suite (`tests/test_llm_provider_conformance.py`): construction
  from spec, capability/roster honesty, a completion round-trip
  against a mocked transport, typed failure classification
  (`llm/failures.py: FailureClass`), the in-provider model
  failover walk, and no-credential-leak assertions.
- Owner-data egress is governed by the consent ledger
  (`nomorals/core/consent.py`, bound to providers in
  `nomorals/llm/consent.py`): a provider may receive a data kind
  only under a recorded, expirable grant; a denied check raises
  `CapabilityDenied` (code `policy.denied`). Local providers
  (mock, ollama, llama_cpp, ocr) never take data off the
  machine and need no grant.
- Aliases are registry data too: agents/context.py's settings
  chain canonicalizes kinds through this registry instead of
  carrying its own alias sets.
