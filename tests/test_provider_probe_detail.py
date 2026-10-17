"""Provider probe must surface the real failure reason.

Regression: /providers printed "probe failed (no calls yet)" for every
dead provider — the probe DID run and fail, but the reason was discarded
at two layers (provider health() returned a bare bool; the status
formatter labelled zero recorded calls as "no calls yet"). The owner
could not tell a bad key (401) from a rate limit (429) from a missing
binary. Now the probe carries the reason and the status prints it.
"""

import unittest
from types import SimpleNamespace
from unittest import mock


def _fake_router(providers):
    """Minimal router: providers() / get() / stats_snapshot()."""
    router = SimpleNamespace()
    router._providers = dict(providers)
    router.providers = lambda: list(router._providers)
    router.get = lambda name: router._providers.get(name)
    router.stats_snapshot = lambda: {"health": {}, "active": ""}
    return router


def _runtime_with_router(router):
    from nomorals.agents.partner.runtime import PartnerRuntime

    rt = object.__new__(PartnerRuntime)
    rt.context = SimpleNamespace(router=router)
    return rt


class FakeProvider:
    """Provider stub with a canned health_detail + model_id."""

    def __init__(self, name, model_id, ok, detail):
        self.name = name
        self.model_id = model_id
        self._ok = ok
        self._detail = detail

    def health_detail(self):
        return self._ok, self._detail

    def health(self):
        return self._ok


class HealthDetailTests(unittest.TestCase):
    def test_openai_compat_401_surfaces_status_and_body(self):
        from nomorals.llm.providers.openai_compat import OpenAICompatProvider

        prov = OpenAICompatProvider(base_url="https://api.groq.com/openai/v1",
                                    model="openai/gpt-oss-120b")
        resp = SimpleNamespace(ok=False, status_code=401,
                               text="Invalid API Key")
        prov.http = SimpleNamespace(get=lambda *a, **k: resp)
        ok, detail = prov.health_detail()
        self.assertFalse(ok)
        self.assertIn("401", detail)
        self.assertIn("Invalid API Key", detail)
        self.assertFalse(prov.health())

    def test_openai_compat_ok_is_live_with_no_detail(self):
        from nomorals.llm.providers.openai_compat import OpenAICompatProvider

        prov = OpenAICompatProvider(base_url="https://api.groq.com/openai/v1",
                                    model="openai/gpt-oss-120b")
        resp = SimpleNamespace(ok=True, status_code=200, text="{}")
        prov.http = SimpleNamespace(get=lambda *a, **k: resp)
        self.assertEqual(prov.health_detail(), (True, ""))
        self.assertTrue(prov.health())

    def test_openai_compat_exception_becomes_detail(self):
        from nomorals.llm.providers.openai_compat import OpenAICompatProvider

        prov = OpenAICompatProvider(base_url="https://api.groq.com/openai/v1",
                                    model="openai/gpt-oss-120b")

        def _boom(*a, **k):
            raise ConnectionError("connection timed out")

        prov.http = SimpleNamespace(get=_boom)
        ok, detail = prov.health_detail()
        self.assertFalse(ok)
        self.assertIn("timed out", detail)

    def test_ocr_missing_binary_says_so(self):
        from nomorals.llm.providers import ocr as ocr_mod

        with mock.patch.object(ocr_mod, "ocr_binary", return_value=None):
            prov = ocr_mod.OCRProvider()
            ok, detail = prov.health_detail()
        self.assertFalse(ok)
        self.assertIn("tesseract", detail.lower())
        self.assertIn("not found", detail.lower())


class ProvidersStatusTests(unittest.TestCase):
    def test_failed_probe_shows_real_reason_not_no_calls_yet(self):
        # 401 with zero recorded calls: the old output was
        # "probe failed (no calls yet)" with no reason at all.
        prov = FakeProvider("groq", "openai/gpt-oss-120b",
                            False, "401: Invalid API Key")
        rt = _runtime_with_router(_fake_router({"groq": prov}))
        out = rt._control_providers()
        self.assertIn("401", out)
        self.assertIn("Invalid API Key", out)
        self.assertIn("auth failed", out)  # actionable hint
        self.assertNotIn("no calls yet", out)

    def test_failed_probe_without_detail_says_no_detail(self):
        prov = FakeProvider("groq", "openai/gpt-oss-120b", False, "")
        rt = _runtime_with_router(_fake_router({"groq": prov}))
        out = rt._control_providers()
        self.assertIn("probe failed (no detail)", out)
        self.assertNotIn("no calls yet", out)

    def test_zero_calls_passing_probe_is_live(self):
        prov = FakeProvider("groq", "openai/gpt-oss-120b", True, "")
        rt = _runtime_with_router(_fake_router({"groq": prov}))
        out = rt._control_providers()
        self.assertIn("live", out)
        self.assertNotIn("probe failed", out)

    def test_ocr_missing_binary_in_status(self):
        from nomorals.llm.providers import ocr as ocr_mod

        with mock.patch.object(ocr_mod, "ocr_binary", return_value=None):
            prov = ocr_mod.OCRProvider()
            rt = _runtime_with_router(_fake_router({"ocr": prov}))
            out = rt._control_providers()
        self.assertIn("tesseract binary not found", out)
        self.assertNotIn("no calls yet", out)


if __name__ == "__main__":
    unittest.main()
