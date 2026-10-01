"""Captcha detection + solving: fixture-HTML detection, backend contracts,
env-key hygiene, takeover behavior, and audit logging.

No test here touches a real solving service — HTTP is stubbed.
"""

import json
import os
import tempfile
import unittest
import urllib.request
from types import SimpleNamespace
from unittest import mock

from nomorals.tools import captcha as cap


# ── fixture pages ────────────────────────────────────────────────────────────

V2_HTML = """
<html><head>
<script src="https://www.google.com/recaptcha/api.js" async defer></script>
</head><body>
<form><div class="g-recaptcha" data-sitekey="6Le-AAAAv2sitekey123"></div>
<button>submit</button></form></body></html>
"""

V3_HTML = """
<html><head>
<script src="https://www.google.com/recaptcha/api.js?render=6Le-BBBv3key456"></script>
<script>grecaptcha.ready(function(){
grecaptcha.execute('6Le-BBBv3key456', {action: 'login'});});</script>
</head><body>login form</body></html>
"""

ENTERPRISE_HTML = """
<html><head>
<script src="https://www.google.com/recaptcha/enterprise.js?render=6Le-CCCEnt789"></script>
</head><body>
<div class="g-recaptcha" data-enterprise="true" data-sitekey="6Le-CCCEnt789"></div>
</body></html>
"""

HCAPTCHA_HTML = """
<html><head><script src="https://js.hcaptcha.com/1/api.js" async defer></script>
</head><body><div class="h-captcha" data-sitekey="10000000-ffff-ffff-ffff-000000000001"></div>
</body></html>
"""

TURNSTILE_HTML = """
<html><head>
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async></script>
</head><body><div class="cf-turnstile" data-sitekey="0x4AAAAAAATurnstile1"></div>
</body></html>
"""

CLOUDFLARE_HTML = """
<html><head><title>Just a moment...</title></head><body>
<div id="cf-challenge"><p>Verifying you are human. This may take a few seconds.</p></div>
<form id="challenge-form"><input type="hidden" name="__cf_chl" value="1"></form>
</body></html>
"""

IMAGE_HTML = """
<html><body><form>
<img id="captcha-img" src="/captcha.php?x=1" alt="captcha image">
<input name="captcha_code">
<img src="/logo.png" alt="logo">
</form></body></html>
"""

CLEAN_HTML = "<html><body><p>nothing to see here</p></body></html>"


class DetectTests(unittest.TestCase):
    def test_v2_sitekey(self):
        found = cap.detect(V2_HTML, "https://example.com/login")
        kinds = {c.kind for c in found}
        self.assertIn(cap.CaptchaKind.RECAPTCHA_V2, kinds)
        v2 = [c for c in found if c.kind == cap.CaptchaKind.RECAPTCHA_V2][0]
        self.assertEqual("6Le-AAAAv2sitekey123", v2.sitekey)
        self.assertEqual("example.com", v2.domain)

    def test_v3_render_key(self):
        found = cap.detect(V3_HTML, "https://example.com/")
        kinds = {c.kind for c in found}
        self.assertIn(cap.CaptchaKind.RECAPTCHA_V3, kinds)
        v3 = [c for c in found if c.kind == cap.CaptchaKind.RECAPTCHA_V3][0]
        self.assertEqual("6Le-BBBv3key456", v3.sitekey)

    def test_enterprise(self):
        found = cap.detect(ENTERPRISE_HTML)
        kinds = {c.kind for c in found}
        self.assertIn(cap.CaptchaKind.RECAPTCHA_ENTERPRISE, kinds)

    def test_hcaptcha(self):
        found = cap.detect(HCAPTCHA_HTML)
        self.assertEqual(cap.CaptchaKind.HCAPTCHA, found[0].kind)
        self.assertEqual("10000000-ffff-ffff-ffff-000000000001",
                         found[0].sitekey)

    def test_turnstile(self):
        found = cap.detect(TURNSTILE_HTML)
        self.assertEqual(cap.CaptchaKind.TURNSTILE, found[0].kind)
        self.assertEqual("0x4AAAAAAATurnstile1", found[0].sitekey)

    def test_cloudflare_challenge(self):
        found = cap.detect(CLOUDFLARE_HTML, "https://shop.example/")
        self.assertEqual(1, len(found))
        self.assertEqual(cap.CaptchaKind.UNKNOWN, found[0].kind)
        self.assertEqual("cloudflare", found[0].metadata["provider"])

    def test_image_captcha(self):
        found = cap.detect(IMAGE_HTML, "https://example.com/form")
        imgs = [c for c in found
                if c.kind == cap.CaptchaKind.IMAGE_CAPTCHA]
        self.assertEqual(1, len(imgs))
        self.assertEqual("https://example.com/captcha.php?x=1",
                         imgs[0].image_url)

    def test_clean_page_finds_nothing(self):
        self.assertEqual([], cap.detect(CLEAN_HTML, "https://example.com/"))

    def test_no_duplicates(self):
        doubled = V2_HTML + V2_HTML
        found = cap.detect(doubled)
        v2 = [c for c in found if c.kind == cap.CaptchaKind.RECAPTCHA_V2]
        self.assertEqual(1, len(v2))

    def test_detect_in_session(self):
        sess = SimpleNamespace(url="https://example.com/l",
                               _raw=HCAPTCHA_HTML)
        found = cap.detect_in_session(sess)
        self.assertEqual(cap.CaptchaKind.HCAPTCHA, found[0].kind)


# ── backend contracts ────────────────────────────────────────────────────────

def _stub_http(create_resp: bytes, poll_resps: list[bytes]):
    """Fake urlopen: in.php then res.php polls."""
    calls = []

    class FakeResp:
        def __init__(self, body: bytes):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        calls.append(url)
        if "/in.php" in url:
            return FakeResp(create_resp)
        body = poll_resps.pop(0) if poll_resps else b'{"status":0,"request":"CAPCHA_NOT_READY"}'
        return FakeResp(body)

    return fake_urlopen, calls


class ServiceBackendTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ,
                                   {"CAPTCHA_API_KEY": "TESTKEY-123"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_available_with_key(self):
        self.assertTrue(cap.ServiceBackend().available())

    def test_unavailable_without_key(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CAPTCHA_API_KEY", None)
            self.assertFalse(cap.ServiceBackend().available())

    def test_solve_v2_full_flow(self):
        fake, calls = _stub_http(
            b'{"status":1,"request":"task-42"}',
            [b'{"status":0,"request":"CAPCHA_NOT_READY"}',
             b'{"status":1,"request":"TOKEN-ABC"}'])
        backend = cap.ServiceBackend(sleeper=lambda s: None)
        ch = cap.CaptchaChallenge(cap.CaptchaKind.RECAPTCHA_V2,
                                  sitekey="SITEKEY",
                                  page_url="https://example.com/login")
        with mock.patch.object(urllib.request, "urlopen", fake):
            result = backend.solve(ch)
        self.assertTrue(result.ok)
        self.assertEqual("TOKEN-ABC", result.token)
        # key went out on the wire (it must, to authenticate) …
        self.assertTrue(any("in.php" in u for u in calls))
        self.assertTrue(any("res.php" in u for u in calls))

    def test_solve_v3_sends_version_params(self):
        fake, calls = _stub_http(
            b'{"status":1,"request":"task-7"}',
            [b'{"status":1,"request":"V3TOKEN"}'])
        backend = cap.ServiceBackend(sleeper=lambda s: None)
        ch = cap.CaptchaChallenge(cap.CaptchaKind.RECAPTCHA_V3,
                                  sitekey="K", page_url="https://e.com/",
                                  action="login", min_score=0.7)
        with mock.patch.object(urllib.request, "urlopen", fake):
            result = backend.solve(ch)
        self.assertTrue(result.ok)
        create_url = next(u for u in calls if "/in.php" in u)
        self.assertIn("version=v3", create_url)
        self.assertIn("action=login", create_url)
        self.assertIn("min_score=0.7", create_url)

    def test_solve_image_uses_base64(self):
        fake, calls = _stub_http(
            b'{"status":1,"request":"task-9"}',
            [b'{"status":1,"request":"ab12"}'])
        backend = cap.ServiceBackend(sleeper=lambda s: None)
        ch = cap.CaptchaChallenge(cap.CaptchaKind.IMAGE_CAPTCHA,
                                  image_bytes=b"\x89PNG fake")
        with mock.patch.object(urllib.request, "urlopen", fake):
            result = backend.solve(ch)
        self.assertTrue(result.ok)
        self.assertEqual("ab12", result.text)
        self.assertEqual("", result.token)

    def test_api_error_raises(self):
        fake, _ = _stub_http(b'{"status":0,"request":"ERROR_KEY_DOES_NOT_EXIST"}',
                             [])
        backend = cap.ServiceBackend(sleeper=lambda s: None)
        ch = cap.CaptchaChallenge(cap.CaptchaKind.RECAPTCHA_V2,
                                  sitekey="K", page_url="https://e.com/")
        with mock.patch.object(urllib.request, "urlopen", fake):
            with self.assertRaises(cap.CaptchaError):
                backend.solve(ch)

    def test_timeout_raises(self):
        fake, _ = _stub_http(b'{"status":1,"request":"task-1"}',
                             [b'{"status":0,"request":"CAPCHA_NOT_READY"}'] * 30)
        backend = cap.ServiceBackend(sleeper=lambda s: None)
        ch = cap.CaptchaChallenge(cap.CaptchaKind.RECAPTCHA_V2,
                                  sitekey="K", page_url="https://e.com/")
        with mock.patch.object(urllib.request, "urlopen", fake):
            with self.assertRaises(cap.CaptchaError):
                backend.solve(ch)

    def test_no_key_raises_clean_error(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CAPTCHA_API_KEY", None)
            backend = cap.ServiceBackend(sleeper=lambda s: None)
            ch = cap.CaptchaChallenge(cap.CaptchaKind.RECAPTCHA_V2,
                                      sitekey="K")
            with self.assertRaises(cap.CaptchaError) as ctx:
                backend.solve(ch)
        self.assertIn("CAPTCHA_API_KEY", str(ctx.exception))


class TakeoverTests(unittest.TestCase):
    def test_takeover_flags_owner_handoff(self):
        backend = cap.TakeoverBackend()
        ch = cap.CaptchaChallenge(cap.CaptchaKind.RECAPTCHA_V2,
                                  sitekey="K",
                                  page_url="https://example.com/login")
        result = backend.solve(ch)
        self.assertFalse(result.ok)
        self.assertTrue(result.takeover)
        self.assertIn("example.com", result.detail)

    def test_auto_without_key_picks_takeover(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CAPTCHA_API_KEY", None)
            backend = cap.backend_for("auto")
        self.assertIsInstance(backend, cap.TakeoverBackend)

    def test_auto_with_key_picks_service(self):
        with mock.patch.dict(os.environ, {"CAPTCHA_API_KEY": "x"}):
            backend = cap.backend_for("auto")
        self.assertIsInstance(backend, cap.ServiceBackend)

    def test_unknown_backend_raises(self):
        with self.assertRaises(cap.CaptchaError):
            cap.backend_for("bogus")


class DetectOnlyTests(unittest.TestCase):
    def test_reports_without_solving(self):
        backend = cap.DetectOnlyBackend()
        ch = cap.CaptchaChallenge(cap.CaptchaKind.HCAPTCHA, sitekey="K",
                                  page_url="https://example.com/")
        result = backend.solve(ch)
        self.assertFalse(result.ok)
        self.assertFalse(result.takeover)
        self.assertIn("hcaptcha", result.detail)


# ── audit hygiene ────────────────────────────────────────────────────────────

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="captcha-audit-")
        self.env = mock.patch.dict(
            os.environ,
            {"CAPTCHA_API_KEY": "SUPER-SECRET-KEY-999",
             "NOMORALS_CAPTCHA_DIR": self.tmp})
        self.env.start()
        self.addCleanup(self.env.stop)

    def _entries(self):
        path = os.path.join(self.tmp, "audit.jsonl")
        if not os.path.exists(path):
            return []
        with open(path) as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def test_successful_solve_is_audited_without_key(self):
        fake, _ = _stub_http(b'{"status":1,"request":"t1"}',
                             [b'{"status":1,"request":"TOK"}'])
        backend = cap.ServiceBackend(sleeper=lambda s: None)
        ch = cap.CaptchaChallenge(cap.CaptchaKind.RECAPTCHA_V2,
                                  sitekey="SITEKEY",
                                  page_url="https://example.com/login")
        with mock.patch.object(urllib.request, "urlopen", fake):
            cap.solve(ch, backend="service")
        entries = self._entries()
        self.assertEqual(1, len(entries))
        e = entries[0]
        self.assertEqual("recaptcha_v2", e["kind"])
        self.assertEqual("example.com", e["domain"])
        self.assertEqual("SITEKEY", e["sitekey"])
        self.assertEqual("service", e["backend"])
        self.assertTrue(e["ok"])
        self.assertIn("ts", e)
        blob = json.dumps(entries)
        self.assertNotIn("SUPER-SECRET-KEY-999", blob)

    def test_failed_solve_is_audited(self):
        ch = cap.CaptchaChallenge(cap.CaptchaKind.TURNSTILE)  # no sitekey
        result = cap.solve(ch, backend="service")
        self.assertFalse(result.ok)
        entries = self._entries()
        self.assertEqual(1, len(entries))
        self.assertFalse(entries[0]["ok"])
        self.assertNotIn("SUPER-SECRET-KEY-999", json.dumps(entries))

    def test_takeover_is_audited(self):
        ch = cap.CaptchaChallenge(cap.CaptchaKind.RECAPTCHA_V2,
                                  sitekey="K",
                                  page_url="https://example.com/")
        result = cap.solve(ch, backend="takeover")
        self.assertTrue(result.takeover)
        entries = self._entries()
        self.assertTrue(entries[0]["takeover"])


# ── tool registration ────────────────────────────────────────────────────────

class RegistrationTests(unittest.TestCase):
    def test_captcha_tool_registers(self):
        from nomorals.tools.registry import ToolRegistry
        reg = ToolRegistry(context=SimpleNamespace(settings=None),
                           enforce=False)
        reg.register_builtins()
        self.assertIn("captcha", reg.names())

    def test_tool_detect_action(self):
        from nomorals.tools.registry import ToolRegistry
        reg = ToolRegistry(context=SimpleNamespace(settings=None),
                           enforce=False)
        reg.register_builtins()
        spec = reg._tools["captcha"]
        out = spec.fn(action="detect", html=V2_HTML,
                      url="https://example.com/")
        self.assertEqual(1, out["count"])
        self.assertEqual("recaptcha_v2", out["challenges"][0]["kind"])

    def test_tool_status_hides_key(self):
        from nomorals.tools.registry import ToolRegistry
        with mock.patch.dict(os.environ, {"CAPTCHA_API_KEY": "K3Y"}):
            reg = ToolRegistry(context=SimpleNamespace(settings=None),
                               enforce=False)
            reg.register_builtins()
            out = reg._tools["captcha"].fn(action="status")
        self.assertTrue(out["api_key_configured"])
        self.assertNotIn("K3Y", json.dumps(out))


if __name__ == "__main__":
    unittest.main()
