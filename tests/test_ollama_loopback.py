"""Failing-first tests: the Ollama provider must reach its own loopback server.

The reasoning tool failed outright because the Ollama provider's HTTP client
ran the SSRF guard against ``http://localhost:11434`` — its own,
operator-configured model server. The sibling llama.cpp provider already
opts into loopback deliberately ("This is the user's own box talking to
itself, not an SSRF attack"). Ollama must do the same: the SSRF guard's
blessed escape hatch for explicitly internal tooling, not a bypass of it.

With the opt-in, an Ollama server that is not running fails with a plain
connection error (failover-eligible), never with "SSRF block".
"""

from __future__ import annotations

import unittest

from nomorals.llm.base import Message, SamplingParams
from nomorals.llm.providers.ollama import OllamaProvider


class OllamaLoopbackTests(unittest.TestCase):
    def test_ollama_allows_loopback(self) -> None:
        provider = OllamaProvider()
        self.assertTrue(provider.http.allow_private_ips)

    def test_unreachable_ollama_is_connection_error_not_ssrf(self) -> None:
        # Closed loopback port: nothing listens, so the failure must be a
        # connection error — never an SSRF block on our own box.
        provider = OllamaProvider(base_url="http://127.0.0.1:9",
                                  max_retries=1)
        with self.assertRaises(Exception) as ctx:  # noqa: BLE001
            provider.chat([Message.user("ping")], SamplingParams(max_tokens=4))
        self.assertNotIn("SSRF", str(ctx.exception))

    def test_router_fails_over_from_dead_ollama_to_mock(self) -> None:
        from nomorals.llm.providers import build_provider
        from nomorals.llm.router import LLMRouter

        router = LLMRouter()
        router.add(OllamaProvider(base_url="http://127.0.0.1:9",
                                  max_retries=1),
                   primary=True, name="ollama")
        router.add(build_provider("mock", model="mock-7b"), name="mock")
        resp = router.chat([Message.user("ping")],
                           SamplingParams(max_tokens=4))
        self.assertTrue(resp.ok, resp.error)
        self.assertEqual(resp.provider, "mock")
        self.assertTrue(resp.degraded)


if __name__ == "__main__":
    unittest.main()
