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


def _isolate_limiter(testcase: unittest.TestCase) -> None:
    """Point the captcha state dir at a fresh temp dir and drop the
    shared-limiter cache, so rate-limit state never leaks between tests
    (or into the real ~/.config)."""
    tmp = tempfile.mkdtemp(prefix="captcha-limiter-")
    testcase.addCleanup(
        lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
    env = mock.patch.dict(os.environ, {"NOMORALS_CAPTCHA_DIR": tmp})
    env.start()
    testcase.addCleanup(env.stop)
    cap._LIMITERS.clear()
    testcase.addCleanup(cap._LIMITERS.clear)


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
        _isolate_limiter(self)

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


# ── Round 19: inline data-URI images, audio challenges, byte fetching ─────────

import base64 as _b64  # noqa: E402


class InlineImageTests(unittest.TestCase):
    def test_data_uri_image_decoded_into_bytes(self):
        payload = _b64.b64encode(b"\x89PNG" + b"\x00" * 100).decode()
        html = (
            '<html><body><img class="captcha" '
            f'src="data:image/png;base64,{payload}" alt="captcha"></body></html>'
        )
        found = cap.detect(html, "https://example.com/")
        imgs = [c for c in found if c.kind == cap.CaptchaKind.IMAGE_CAPTCHA]
        self.assertEqual(1, len(imgs))
        self.assertTrue(imgs[0].image_url.startswith("data:"))
        self.assertEqual(b"\x89PNG" + b"\x00" * 100, imgs[0].image_bytes)

    def test_malformed_data_uri_yields_empty_bytes_not_crash(self):
        html = ('<html><body><img class="captcha" '
                'src="data:image/png;base64,!!!not-base64!!!" '
                'alt="captcha"></body></html>')
        found = cap.detect(html, "https://example.com/")
        imgs = [c for c in found if c.kind == cap.CaptchaKind.IMAGE_CAPTCHA]
        self.assertEqual(1, len(imgs))
        self.assertEqual(b"", imgs[0].image_bytes)

    def test_summary_truncates_inline_image(self):
        payload = _b64.b64encode(b"x" * 5000).decode()
        html = (
            '<html><body><img class="captcha" '
            f'src="data:image/png;base64,{payload}"></body></html>'
        )
        found = cap.detect(html, "https://example.com/")
        summary = found[0].summary()
        self.assertLess(len(summary["image_url"]), 200)
        self.assertEqual(5000, summary["image_bytes"])

    def test_fetch_bytes_downloads_remote_image(self):
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return fake_png

        with mock.patch.object(cap.urllib.request, "urlopen",
                               return_value=_Resp()):
            found = cap.detect(IMAGE_HTML, "https://example.com/form",
                               fetch_bytes=True)
        imgs = [c for c in found if c.kind == cap.CaptchaKind.IMAGE_CAPTCHA]
        self.assertEqual(1, len(imgs))
        self.assertEqual(fake_png, imgs[0].image_bytes)

    def test_fetch_bytes_failure_leaves_empty_bytes(self):
        with mock.patch.object(cap.urllib.request, "urlopen",
                               side_effect=OSError("down")):
            found = cap.detect(IMAGE_HTML, "https://example.com/form",
                               fetch_bytes=True)
        imgs = [c for c in found if c.kind == cap.CaptchaKind.IMAGE_CAPTCHA]
        self.assertEqual(1, len(imgs))
        self.assertEqual(b"", imgs[0].image_bytes)

    def test_default_detect_stays_pure_no_network(self):
        with mock.patch.object(cap.urllib.request, "urlopen",
                               side_effect=AssertionError("no net")):
            found = cap.detect(IMAGE_HTML, "https://example.com/form")
        imgs = [c for c in found if c.kind == cap.CaptchaKind.IMAGE_CAPTCHA]
        self.assertEqual(b"", imgs[0].image_bytes)


class AudioCaptchaTests(unittest.TestCase):
    def setUp(self):
        _isolate_limiter(self)

    AUDIO_HTML = """
    <html><body>
    <div class="g-recaptcha" data-sitekey="6Le-AAAAv2sitekey123"></div>
    <div id="rc-audio">Get an audio challenge</div>
    <audio src="https://www.google.com/recaptcha/api2/payload/audio?k=abc"></audio>
    </body></html>
    """

    def test_audio_challenge_detected(self):
        found = cap.detect(self.AUDIO_HTML, "https://example.com/")
        kinds = {c.kind for c in found}
        self.assertIn(cap.CaptchaKind.AUDIO_CAPTCHA, kinds)
        audio = [c for c in found
                 if c.kind == cap.CaptchaKind.AUDIO_CAPTCHA][0]
        self.assertIn("google.com", audio.image_url)

    def test_audio_service_solve_sends_audio_method(self):
        posted: dict = {}

        class _Resp:
            def __init__(self, body):
                self._body = body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return self._body

        def fake_urlopen(req, timeout=30):
            url = req.full_url
            if url.endswith("/in.php?") or "/in.php?" in url:
                posted["in"] = url
                return _Resp(b'{"status":1,"request":"task-1"}')
            posted["res"] = url
            return _Resp(b'{"status":1,"request":"spoken words"}')

        ch = cap.CaptchaChallenge(
            kind=cap.CaptchaKind.AUDIO_CAPTCHA,
            page_url="https://example.com/",
            image_bytes=b"FAKEAUDIO" * 100)
        backend = cap.ServiceBackend(api_key="K", sleeper=lambda s: None)
        with mock.patch.object(cap.urllib.request, "urlopen", fake_urlopen):
            result = backend.solve(ch)
        self.assertTrue(result.ok)
        self.assertEqual("spoken words", result.text)
        self.assertIn("method=audio", posted["in"])

    def test_audio_solve_without_bytes_raises_clean_error(self):
        ch = cap.CaptchaChallenge(kind=cap.CaptchaKind.AUDIO_CAPTCHA,
                                  page_url="https://example.com/")
        backend = cap.ServiceBackend(api_key="K", sleeper=lambda s: None)
        with self.assertRaises(cap.CaptchaError) as ctx:
            backend.solve(ch)
        self.assertIn("bytes", str(ctx.exception))


# ── new-provider detection ───────────────────────────────────────────────────

GEETEST_V4_HTML = """
<html><head>
<script src="https://static.geetest.com/v4/gt4.js"></script>
</head><body><div id="captcha"></div>
<script>
initGeetest4({captchaId: "54088bb07d2df3c46b79f80300b0abbe", product: "bind"},
  function(captchaObj){});
</script></body></html>
"""

GEETEST_V3_HTML = """
<html><head><script src="https://static.geetest.com/static/js/gt.js"></script>
</head><body><div id="geetest-box"></div>
<script>
window.initGeetest({
  gt: "81e9e6a9f8d7c6b5a4e3d2c1b0a99887",
  challenge: "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
  offline: false
}, handler);
</script></body></html>
"""

ARKOSE_HTML = """
<html><head>
<script src="https://client-api.arkoselabs.com/fc/api/?onload=fcCb"></script>
</head><body>
<div id="funcaptcha" data-pkey="3225A9B1-4C2D-4E8F-9A0B-1C2D3E4F5A6B"></div>
</body></html>
"""

AWSWAF_HTML = """
<html><head><title>AWS WAF</title></head><body>
<script src="https://b82d-11e8.us-east-1.awswaf.com/b82d-11e8/22e3/challenge.js"></script>
<p>awswaf captcha: complete the challenge below</p>
<div id="waf-captcha"></div>
</body></html>
"""

FRIENDLY_HTML = """
<html><head><script src="https://cdn.jsdelivr.net/npm/friendly-challenge@0.9.12/widget.min.js"></script>
</head><body>
<div class="frc-captcha" data-sitekey="FCMGEMM1G7V5K8Q9"></div>
</body></html>
"""

SMARTCAPTCHA_HTML = """
<html><head>
<script src="https://smart-captcha.yandexcloud.net/captcha.js" defer></script>
</head><body>
<div class="smartcaptcha" data-sitekey="ysc1_test00000000000000000000"></div>
</body></html>
"""

PERIMETERX_HTML = """
<html><head><title>Press & Hold</title></head><body>
<script src="https://www.perimeterx.net/px/abc123/main.min.js"></script>
<div class="px-captcha"><p>Press &amp; Hold to confirm you are a human</p></div>
</body></html>
"""


class NewProviderDetectTests(unittest.TestCase):
    def test_geetest_v4_captcha_id(self):
        found = cap.detect(GEETEST_V4_HTML, "https://example.com/")
        g = [c for c in found if c.kind == cap.CaptchaKind.GEETEST]
        self.assertEqual(1, len(g))
        self.assertEqual("54088bb07d2df3c46b79f80300b0abbe", g[0].sitekey)
        self.assertEqual("v4", g[0].metadata.get("variant"))

    def test_geetest_v3_gt_challenge(self):
        found = cap.detect(GEETEST_V3_HTML, "https://example.com/")
        g = [c for c in found if c.kind == cap.CaptchaKind.GEETEST]
        self.assertEqual(1, len(g))
        self.assertEqual("81e9e6a9f8d7c6b5a4e3d2c1b0a99887",
                         g[0].metadata.get("gt"))
        self.assertEqual("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
                         g[0].metadata.get("challenge"))

    def test_arkose_pkey(self):
        found = cap.detect(ARKOSE_HTML, "https://example.com/")
        a = [c for c in found if c.kind == cap.CaptchaKind.ARKOSE]
        self.assertEqual(1, len(a))
        self.assertEqual("3225A9B1-4C2D-4E8F-9A0B-1C2D3E4F5A6B", a[0].sitekey)
        self.assertEqual("fc-token",
                         cap.CaptchaKind.TOKEN_FIELD[cap.CaptchaKind.ARKOSE])

    def test_aws_waf(self):
        found = cap.detect(AWSWAF_HTML, "https://shop.example/")
        w = [c for c in found if c.kind == cap.CaptchaKind.AWS_WAF]
        self.assertEqual(1, len(w))

    def test_friendly(self):
        found = cap.detect(FRIENDLY_HTML, "https://example.com/")
        f = [c for c in found if c.kind == cap.CaptchaKind.FRIENDLY]
        self.assertEqual(1, len(f))
        self.assertEqual("FCMGEMM1G7V5K8Q9", f[0].sitekey)

    def test_smartcaptcha(self):
        found = cap.detect(SMARTCAPTCHA_HTML, "https://example.com/")
        s = [c for c in found if c.kind == cap.CaptchaKind.SMARTCAPTCHA]
        self.assertEqual(1, len(s))
        self.assertEqual("ysc1_test00000000000000000000", s[0].sitekey)

    def test_perimeterx(self):
        found = cap.detect(PERIMETERX_HTML, "https://example.com/")
        p = [c for c in found if c.kind == cap.CaptchaKind.PERIMETERX]
        self.assertEqual(1, len(p))

    def test_token_field_map_complete(self):
        for kind in cap.CaptchaKind.TOKEN_KINDS:
            self.assertIn(kind, cap.CaptchaKind.TOKEN_FIELD,
                          f"no token field for {kind}")


# ── service backend: new task types, proxy, error classes ────────────────────

class _FakeHTTP:
    """Minimal urlopen double: canned in.php / res.php bodies."""

    def __init__(self, in_body: bytes, res_bodies: list[bytes]):
        self.in_body = in_body
        self.res_bodies = list(res_bodies)
        self.calls: list[str] = []

    def __call__(self, req, timeout=30):
        url = req.full_url
        self.calls.append(url)
        body = self.in_body if "/in.php?" in url else (
            self.res_bodies.pop(0) if self.res_bodies
            else b'{"status":1,"request":"TOKEN"}')
        resp = mock.Mock()
        resp.read.return_value = body
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda s, *a: False
        return resp


def _service_backend(**kw):
    kw.setdefault("api_key", "K")
    kw.setdefault("sleeper", lambda s: None)
    return cap.ServiceBackend(**kw)


class NewTaskTypeTests(unittest.TestCase):
    def setUp(self):
        _isolate_limiter(self)

    def _in_url(self, fake: _FakeHTTP) -> str:
        return next(u for u in fake.calls if "/in.php?" in u)

    def test_funcaptcha_sends_publickey(self):
        fake = _FakeHTTP(b'{"status":1,"request":"t1"}', [])
        backend = _service_backend()
        ch = cap.CaptchaChallenge(
            kind=cap.CaptchaKind.ARKOSE, sitekey="PUBKEY-1",
            page_url="https://example.com/signup")
        with mock.patch.object(cap.urllib.request, "urlopen", fake):
            result = backend.solve(ch)
        self.assertTrue(result.ok)
        url = self._in_url(fake)
        self.assertIn("method=funcaptcha", url)
        self.assertIn("publickey=PUBKEY-1", url)
        self.assertIn("pageurl=", url)
        # arkose token goes into fc-token
        self.assertEqual("fc-token",
                         cap.CaptchaKind.TOKEN_FIELD[ch.kind])

    def test_funcaptcha_needs_pkey(self):
        backend = _service_backend()
        ch = cap.CaptchaChallenge(kind=cap.CaptchaKind.ARKOSE,
                                  page_url="https://e.com/")
        with self.assertRaises(cap.CaptchaError):
            backend.solve(ch)

    def test_geetest_v4_sends_captcha_id(self):
        fake = _FakeHTTP(b'{"status":1,"request":"t2"}', [])
        backend = _service_backend()
        ch = cap.CaptchaChallenge(
            kind=cap.CaptchaKind.GEETEST, sitekey="CAPTCHAID-1",
            page_url="https://example.com/",
            metadata={"variant": "v4"})
        with mock.patch.object(cap.urllib.request, "urlopen", fake):
            result = backend.solve(ch)
        self.assertTrue(result.ok)
        url = self._in_url(fake)
        self.assertIn("method=geetest_v4", url)
        self.assertIn("captcha_id=CAPTCHAID-1", url)

    def test_geetest_v3_sends_gt_challenge(self):
        fake = _FakeHTTP(b'{"status":1,"request":"t3"}', [])
        backend = _service_backend()
        ch = cap.CaptchaChallenge(
            kind=cap.CaptchaKind.GEETEST, sitekey="GT-1",
            page_url="https://example.com/",
            metadata={"variant": "v3", "gt": "GT-1",
                      "challenge": "CH-1"})
        with mock.patch.object(cap.urllib.request, "urlopen", fake):
            result = backend.solve(ch)
        self.assertTrue(result.ok)
        url = self._in_url(fake)
        self.assertIn("gt=GT-1", url)
        self.assertIn("challenge=CH-1", url)

    def test_proxy_passed_to_service(self):
        fake = _FakeHTTP(b'{"status":1,"request":"t4"}', [])
        backend = _service_backend(proxy="socks5://127.0.0.1:1080")
        ch = cap.CaptchaChallenge(
            kind=cap.CaptchaKind.RECAPTCHA_V2, sitekey="K",
            page_url="https://example.com/")
        with mock.patch.object(cap.urllib.request, "urlopen", fake):
            backend.solve(ch)
        url = self._in_url(fake)
        self.assertIn("proxytype=SOCKS5", url)
        self.assertIn("proxy=127.0.0.1%3A1080", url)

    def test_no_proxy_on_image_tasks(self):
        fake = _FakeHTTP(b'{"status":1,"request":"t5"}', [])
        backend = _service_backend(proxy="socks5://127.0.0.1:1080")
        ch = cap.CaptchaChallenge(kind=cap.CaptchaKind.IMAGE_CAPTCHA,
                                  image_bytes=b"\x89PNG fake")
        with mock.patch.object(cap.urllib.request, "urlopen", fake):
            backend.solve(ch)
        url = self._in_url(fake)
        self.assertNotIn("proxy=", url)


class SolverErrorClassTests(unittest.TestCase):
    def setUp(self):
        _isolate_limiter(self)

    def test_hard_errors_detected(self):
        for code in ("ERROR_ZERO_BALANCE", "ERROR_KEY_DOES_NOT_EXIST",
                     "ERROR_WRONG_USER_KEY", "ERROR_IP_NOT_ALLOWED",
                     "ERROR_IP_BLOCKED"):
            self.assertTrue(cap._hard_error(code), code)
        self.assertFalse(cap._hard_error("ERROR_NO_SLOT_AVAILABLE"))
        self.assertFalse(cap._hard_error(""))

    def test_hard_error_trips_long_cooldown(self):
        fake = _FakeHTTP(
            b'{"status":0,"request":"ERROR_ZERO_BALANCE"}', [])
        backend = _service_backend()
        ch = cap.CaptchaChallenge(kind=cap.CaptchaKind.RECAPTCHA_V2,
                                  sitekey="K",
                                  page_url="https://example.com/")
        with mock.patch.object(cap.urllib.request, "urlopen", fake):
            with self.assertRaises(cap.CaptchaError) as ctx:
                backend.solve(ch)
        self.assertIn("zero balance", str(ctx.exception).lower())
        # any further solve is now blocked by the 6h cooldown
        with self.assertRaises(cap.CaptchaError) as ctx2:
            backend.solve(ch)
        self.assertIn("cooling down", str(ctx2.exception))

    def test_soft_error_backoff_is_transient(self):
        fake = _FakeHTTP(
            b'{"status":0,"request":"ERROR_NO_SLOT_AVAILABLE"}', [])
        backend = _service_backend()
        ch = cap.CaptchaChallenge(kind=cap.CaptchaKind.RECAPTCHA_V2,
                                  sitekey="K",
                                  page_url="https://example.com/")
        with mock.patch.object(cap.urllib.request, "urlopen", fake):
            with self.assertRaises(cap.CaptchaError) as ctx:
                backend.solve(ch)
        self.assertIn("busy", str(ctx.exception).lower())

    def test_friendly_error_messages(self):
        self.assertIn("zero balance",
                      cap._friendly_error("ERROR_ZERO_BALANCE").lower())
        self.assertIn("check the key",
                      cap._friendly_error("ERROR_KEY_DOES_NOT_EXIST").lower())
        self.assertIn("ERROR_WEIRD",
                      cap._friendly_error("ERROR_WEIRD"))

    def test_unknown_kind_has_clean_message(self):
        backend = _service_backend()
        ch = cap.CaptchaChallenge(kind=cap.CaptchaKind.PERIMETERX,
                                  page_url="https://e.com/")
        with self.assertRaises(cap.CaptchaError) as ctx:
            backend.solve(ch)
        self.assertIn("takeover", str(ctx.exception))


class RateLimiterTests(unittest.TestCase):
    def _limiter(self, **kw) -> cap.SolverRateLimiter:
        tmp = tempfile.mkdtemp(prefix="cap-rl-")
        self.addCleanup(
            lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        env = mock.patch.dict(os.environ, {"NOMORALS_CAPTCHA_DIR": tmp})
        env.start()
        self.addCleanup(env.stop)
        return cap.SolverRateLimiter(settings=None, **kw)

    def test_check_passes_when_idle(self):
        lim = self._limiter()
        lim.check()  # no raise

    def test_per_minute_budget(self):
        lim = self._limiter(per_minute=2)
        lim.record_submit()
        lim.record_submit()
        with self.assertRaises(cap.CaptchaError) as ctx:
            lim.check()
        self.assertIn("rate too high", str(ctx.exception).lower())

    def test_per_day_budget(self):
        lim = self._limiter(per_day=1)
        lim.record_submit()
        with self.assertRaises(cap.CaptchaError) as ctx:
            lim.check()
        self.assertIn("daily", str(ctx.exception).lower())

    def test_error_backoff_then_recovery(self):
        lim = self._limiter(error_cooldown_s=0.05)
        lim.record_failure()
        with self.assertRaises(cap.CaptchaError):
            lim.check()  # cooling down
        import time as _t
        _t.sleep(0.06)
        lim.check()  # backoff expired
        lim.record_success()
        lim.check()  # failure count reset, no cooldown

    def test_state_survives_reload(self):
        tmp = tempfile.mkdtemp(prefix="cap-rl2-")
        self.addCleanup(
            lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        with mock.patch.dict(os.environ, {"NOMORALS_CAPTCHA_DIR": tmp}):
            cap._LIMITERS.clear()
            lim1 = cap.rate_limiter()
            lim1.record_failure(hard=True)
            cap._LIMITERS.clear()
            lim2 = cap.rate_limiter()
            with self.assertRaises(cap.CaptchaError):
                lim2.check()  # 6h cooldown persisted
            st = lim2.status()
            self.assertTrue(st["cooling_down"])
            self.assertTrue(st["hard_failure"])

    def test_status_snapshot(self):
        lim = self._limiter()
        lim.record_submit()
        st = lim.status()
        self.assertEqual(1, st["submits_last_24h"])
        self.assertFalse(st["cooling_down"])
        self.assertIn("per_minute", st)
        self.assertIn("per_day", st)


class RegisterBackendTests(unittest.TestCase):
    def test_register_custom_backend(self):
        class LocalOcr(cap.CaptchaBackend):
            name = "localocr"

            def solve(self, challenge):
                return cap.SolveResult(
                    ok=True, kind=challenge.kind, backend=self.name,
                    text="read-text")

        cap.register_backend("localocr", LocalOcr)
        self.addCleanup(lambda: cap._BACKENDS.pop("localocr", None))
        backend = cap.backend_for("localocr")
        self.assertIsInstance(backend, LocalOcr)
        ch = cap.CaptchaChallenge(kind=cap.CaptchaKind.IMAGE_CAPTCHA)
        self.assertTrue(backend.solve(ch).ok)

    def test_register_rejects_bad_name(self):
        with self.assertRaises(cap.CaptchaError):
            cap.register_backend("", cap.TakeoverBackend)

    def test_register_rejects_non_backend(self):
        class NotABackend:
            pass

        with self.assertRaises(cap.CaptchaError):
            cap.register_backend("nope", NotABackend)


class TakeoverNotifyTests(unittest.TestCase):
    def test_takeover_notifies_owner(self):
        seen = {}

        class _FakeNotifier:
            def notify(self, context, kind, title, body, **kw):
                seen.update(kind=kind, title=title, body=body)
                return {"delivered": True}

        class _FakeContext:
            notifier = _FakeNotifier()

        ch = cap.CaptchaChallenge(
            kind=cap.CaptchaKind.RECAPTCHA_V2, sitekey="K",
            page_url="https://example.com/login")
        with mock.patch.object(cap, "_notify_takeover",
                               wraps=cap._notify_takeover) as wrapped:
            result = cap.solve(ch, backend="takeover",
                               notify_owner=True, context=_FakeContext())
        self.assertTrue(result.takeover)
        self.assertTrue(wrapped.called)
        self.assertEqual("captcha", seen["kind"])
        self.assertIn("recaptcha_v2", seen["title"])
        self.assertIn("example.com", seen["title"])
        self.assertIn("https://example.com/login", seen["body"])

    def test_notify_can_be_disabled(self):
        ch = cap.CaptchaChallenge(kind=cap.CaptchaKind.RECAPTCHA_V2,
                                  sitekey="K",
                                  page_url="https://example.com/")
        with mock.patch.object(cap, "_notify_takeover") as fake:
            cap.solve(ch, backend="takeover", notify_owner=False)
        fake.assert_not_called()

    def test_takeover_detail_has_page_and_hint(self):
        result = cap.TakeoverBackend().solve(
            cap.CaptchaChallenge(kind=cap.CaptchaKind.GEETEST,
                                 page_url="https://example.com/game"))
        self.assertIn("https://example.com/game", result.detail)
        self.assertIn("slider", result.detail.lower())

    def test_notify_never_breaks_solve(self):
        ch = cap.CaptchaChallenge(kind=cap.CaptchaKind.RECAPTCHA_V2,
                                  sitekey="K")
        with mock.patch("nomorals.agents.notifier.notify",
                         side_effect=RuntimeError("boom")):
            result = cap.solve(ch, backend="takeover", notify_owner=True)
        self.assertTrue(result.takeover)


class BrowserCheckCaptchaTests(unittest.TestCase):
    def _session_with(self, html: str, url: str = "https://example.com/"):
        from nomorals.tools.browser import BrowserSession, parse_html
        sess = BrowserSession(name="test-captcha", session_dir="")
        sess.url = url
        sess._raw = html
        sess.dom = parse_html(html)
        return sess

    def test_check_captcha_stashes_token(self):
        from nomorals.tools import browser as _bmod
        html = ("<html><body><div class='g-recaptcha' "
                "data-sitekey='SITEKEY-1'></div></body></html>")
        sess = self._session_with(html)

        def _fake_solve(challenge, **kw):
            return cap.SolveResult(
                ok=True, kind=challenge.kind, backend="service",
                token="SOLVED-TOKEN")

        with mock.patch.object(cap, "solve", _fake_solve):
            out = sess.check_captcha()
        self.assertEqual(1, len(out["challenges"]))
        self.assertEqual("g-recaptcha-response",
                         out["challenges"][0]["injected_into"])
        self.assertEqual("SOLVED-TOKEN",
                         sess.captcha_tokens["g-recaptcha-response"])

    def test_check_captcha_detect_only(self):
        html = ("<html><body><div class='h-captcha' "
                "data-sitekey='HK-1'></div></body></html>")
        sess = self._session_with(html)
        with mock.patch.object(cap, "solve") as fake_solve:
            out = sess.check_captcha(solve=False)
        fake_solve.assert_not_called()
        self.assertFalse(out["challenges"][0]["solve_attempted"])

    def test_check_captcha_takeover_flagged(self):
        html = ("<html><body><div class='cf-turnstile' "
                "data-sitekey='TS-1'></div></body></html>")
        sess = self._session_with(html)

        def _fake_solve(challenge, **kw):
            return cap.SolveResult(
                ok=False, kind=challenge.kind, backend="takeover",
                takeover=True, detail="owner takeover needed")

        with mock.patch.object(cap, "solve", _fake_solve):
            out = sess.check_captcha()
        self.assertTrue(out["challenges"][0]["takeover"])
        self.assertEqual({}, sess.captcha_tokens)

    def test_check_captcha_needs_page(self):
        from nomorals.tools.browser import BrowserSession
        sess = BrowserSession(name="test-empty", session_dir="")
        from nomorals.core.errors import ToolError
        with self.assertRaises(ToolError):
            sess.check_captcha()

    def test_browser_action_registered(self):
        from nomorals.tools.browser import _ACTIONS
        self.assertIn("check_captcha", _ACTIONS)


if __name__ == "__main__":
    unittest.main()
