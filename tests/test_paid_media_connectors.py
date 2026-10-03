"""Paid media-generation connector tests. HTTP is fully mocked — no network.

Covers: nano_banana, google_flow, leonardo, stability_ai.
"""

from __future__ import annotations

import base64
import json
import os
import unittest
import urllib.parse
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.googleflow import GoogleFlowConnector, GoogleFlowError
from nomorals.connectors.leonardo import LeonardoConnector, LeonardoError
from nomorals.connectors.nanobanana import (
    MODEL_FLASH,
    NanoBananaConnector,
    NanoBananaError,
)
from nomorals.connectors.registry import get_connector, list_connectors
from nomorals.connectors.stabilityai import (
    StabilityAIConnector,
    StabilityAIError,
)
from nomorals.core.errors import NoMoralsError
from nomorals.storage.db import Database

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwAD"
    "hgGAWjR9awAAAABJRU5ErkJggg=="
)
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        payload: Any = None,
        raw: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self._payload = payload
        self.body = raw if raw is not None else b""
        self.headers = headers or {}
        self.url = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        if self.body:
            return self.body.decode("utf-8", errors="replace")
        return json.dumps(self._payload)

    def json(self) -> Any:
        if self.body and not self._payload:
            return json.loads(self.body.decode("utf-8"))
        return self._payload


class FakeHttp:
    """Scripted stand-in for HttpClient. No network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any, Any]] = []
        self.routes: list[tuple[str, str, FakeResponse]] = []

    def route(self, method: str, path: str, response: FakeResponse) -> None:
        self.routes.append((method.upper(), path, response))

    @staticmethod
    def _pattern_matches(pattern: str, url: str) -> bool:
        """Substring match that respects path boundaries.

        A pattern matches when it appears in the URL and is not immediately
        followed by ``/`` + more path — so a route for
        ``/models/veo-3.1-fast-generate-preview`` matches the model-lookup
        URL but never an operation poll URL like
        ``.../models/veo-3.1-fast-generate-preview/operations/op-9``.
        """
        start = 0
        while True:
            idx = url.find(pattern, start)
            if idx < 0:
                return False
            after = idx + len(pattern)
            if after >= len(url) or url[after] != "/":
                return True
            start = after

    def _dispatch(
        self, method: str, url: str, payload: Any = None, **kw: Any
    ) -> FakeResponse:
        self.calls.append((method.upper(), url, payload, kw.get("headers")))
        matches = [
            (i, rm, rp, resp)
            for i, (rm, rp, resp) in enumerate(self.routes)
            if rm == method.upper() and self._pattern_matches(rp, url)
        ]
        if not matches:
            return FakeResponse(404, {"error": {"message": "not mocked"}})
        # Longest pattern wins; ties (the same route registered twice, e.g.
        # pending-then-done) are consumed FIFO so a scripted sequence of
        # responses plays out in registration order.
        best_len = max(len(rp) for _, _, rp, _ in matches)
        tied = [m for m in matches if len(m[2]) == best_len]
        if len(tied) > 1:
            idx = tied[0][0]
            _, _, resp = self.routes.pop(idx)
            return resp
        return tied[0][3]

    def get(self, url: str, **kw: Any) -> FakeResponse:
        params = kw.pop("params", None) or {}
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"
        return self._dispatch("GET", url, None, **kw)

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, payload, **kw)

    def post_multipart(self, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch(
            "POST", url, {"fields": kw.get("fields"),
                          "files": kw.get("files")}, **kw
        )


class DeadHttp(FakeHttp):
    """A dead network: every request raises NoMoralsError.

    Simulates the offline case — connectors must surface a clean typed
    error, never a raw traceback.
    """

    def _dispatch(
        self, method: str, url: str, payload: Any = None, **kw: Any
    ) -> FakeResponse:
        raise NoMoralsError("network unreachable (simulated offline)")


# ── registration ─────────────────────────────────────────────────


class RegistrationTests(unittest.TestCase):
    def test_all_registered(self) -> None:
        self.assertIs(get_connector("nano_banana"), NanoBananaConnector)
        self.assertIs(get_connector("google_flow"), GoogleFlowConnector)
        self.assertIs(get_connector("leonardo"), LeonardoConnector)
        self.assertIs(get_connector("stability_ai"), StabilityAIConnector)

    def test_listed(self) -> None:
        ids = {info["id"] for info in list_connectors()}
        for cid in ("nano_banana", "google_flow", "leonardo", "stability_ai"):
            self.assertIn(cid, ids)

    def test_api_key_auth(self) -> None:
        for cls in (NanoBananaConnector, GoogleFlowConnector,
                    LeonardoConnector, StabilityAIConnector):
            methods = [m.value for m in cls.auth_methods]
            self.assertIn("api_key", methods)


# ── Nano Banana ──────────────────────────────────────────────────


def _nb_model_ok() -> FakeResponse:
    return FakeResponse(200, {"name": f"models/{MODEL_FLASH}"})


def _nb_image_ok(text: str = "done") -> FakeResponse:
    return FakeResponse(200, {
        "candidates": [{
            "content": {
                "parts": [
                    {"text": text},
                    {"inlineData": {
                        "mimeType": "image/png", "data": PNG_B64}},
                ]
            }
        }]
    })


def _nb_connected(http: FakeHttp | None = None,
                  env_key: str = "NANO_BANANA_API_KEY"):
    http = http or FakeHttp()
    conn = NanoBananaConnector(_vault(), http=http)
    http.route("GET", f"/models/{MODEL_FLASH}", _nb_model_ok())
    with mock.patch.dict(os.environ, {env_key: "nb-test-key"}):
        result = conn.connect()
    assert result.ok
    return conn, http


class NanoBananaTests(unittest.TestCase):
    def test_connect_missing_key_fails_fast(self) -> None:
        conn = NanoBananaConnector(_vault(), http=FakeHttp())
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(ConnectorError) as ctx:
            conn.connect()
        self.assertIn("NANO_BANANA_API_KEY", str(ctx.exception))

    def test_connect_validates_and_stores(self) -> None:
        conn, http = _nb_connected()
        self.assertTrue(conn.test_connection())
        status = conn.status()
        self.assertTrue(status.connected)

    def test_connect_rejected_key(self) -> None:
        http = FakeHttp()
        conn = NanoBananaConnector(_vault(), http=http)
        http.route("GET", f"/models/{MODEL_FLASH}",
                   FakeResponse(401, {"error": {"message": "API key not valid"}}))
        with mock.patch.dict(os.environ, {"NANO_BANANA_API_KEY": "bad"}), self.assertRaises(NanoBananaError) as ctx:
            conn.connect()
        self.assertIn("rejected", str(ctx.exception))

    def test_connect_already_connected(self) -> None:
        conn, _ = _nb_connected()
        with mock.patch.dict(os.environ, {"NANO_BANANA_API_KEY": "x"}), self.assertRaises(ConnectorError):
            conn.connect()

    def test_disconnect(self) -> None:
        conn, _ = _nb_connected()
        conn.disconnect()
        self.assertFalse(conn.test_connection())
        status = conn.status()
        self.assertFalse(status.connected)

    def test_generate_image(self) -> None:
        conn, http = _nb_connected()
        http.route("POST", ":generateContent", _nb_image_ok())
        images = conn.generate_image("a red balloon",
                                     aspect_ratio="16:9", confirmed=True)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["image_bytes"], PNG_BYTES)
        self.assertEqual(images[0]["mime_type"], "image/png")
        method, url, payload, headers = http.calls[-1]
        self.assertEqual(method, "POST")
        self.assertIn(":generateContent", url)
        self.assertEqual(headers["x-goog-api-key"], "nb-test-key")
        cfg = payload["generationConfig"]
        self.assertEqual(cfg["responseModalities"], ["TEXT", "IMAGE"])
        self.assertEqual(cfg["imageConfig"]["aspectRatio"], "16:9")

    def test_generate_image_needs_confirmation(self) -> None:
        conn, http = _nb_connected()
        http.route("POST", ":generateContent", _nb_image_ok())
        with self.assertRaises(ConnectorError):
            conn.generate_image("a red balloon")

    def test_generate_image_bad_aspect(self) -> None:
        conn, http = _nb_connected()
        with self.assertRaises(ConnectorError):
            conn.generate_image("x", aspect_ratio="99:99", confirmed=True)

    def test_generate_image_empty_prompt(self) -> None:
        conn, http = _nb_connected()
        with self.assertRaises(ConnectorError):
            conn.generate_image("", confirmed=True)

    def test_edit_image(self) -> None:
        conn, http = _nb_connected()
        http.route("POST", ":generateContent", _nb_image_ok())
        images = conn.edit_image(PNG_BYTES, "make it blue", confirmed=True)
        self.assertEqual(images[0]["image_bytes"], PNG_BYTES)
        payload = http.calls[-1][2]
        parts = payload["contents"][0]["parts"]
        self.assertEqual(parts[0]["text"], "make it blue")
        self.assertEqual(parts[1]["inlineData"]["mimeType"], "image/png")
        self.assertEqual(parts[1]["inlineData"]["data"], PNG_B64)

    def test_compose_images(self) -> None:
        conn, http = _nb_connected()
        http.route("POST", ":generateContent", _nb_image_ok())
        images = conn.compose_images([PNG_BYTES, PNG_BYTES],
                                     "blend them", confirmed=True)
        self.assertEqual(len(images), 1)
        payload = http.calls[-1][2]
        parts = payload["contents"][0]["parts"]
        self.assertEqual(len(parts), 3)  # text + 2 images

    def test_compose_too_many(self) -> None:
        conn, http = _nb_connected()
        with self.assertRaises(ConnectorError):
            conn.compose_images([PNG_BYTES] * 5, "x", confirmed=True)

    def test_no_image_in_response(self) -> None:
        conn, http = _nb_connected()
        http.route("POST", ":generateContent",
                   FakeResponse(200, {"candidates": []}))
        with self.assertRaises(NanoBananaError):
            conn.generate_image("x", confirmed=True)

    def test_rate_limit_maps(self) -> None:
        conn, http = _nb_connected()
        http.route("POST", ":generateContent", FakeResponse(429, {}))
        with self.assertRaises(NanoBananaError) as ctx:
            conn.generate_image("x", confirmed=True)
        self.assertEqual(ctx.exception.status_code, 429)

    def test_capabilities(self) -> None:
        conn = NanoBananaConnector(_vault(), http=FakeHttp())
        caps = conn.capabilities()
        self.assertIn("generate_image", caps["can"])
        self.assertIn("edit_image", caps["can"])

    def test_offline_network_failure_is_clean_error(self) -> None:
        conn, _ = _nb_connected()
        conn.http = DeadHttp()
        with self.assertRaises(NanoBananaError) as ctx:
            conn.generate_image("x", confirmed=True)
        self.assertIn("network unreachable", str(ctx.exception))


# ── Google Flow (Veo) ────────────────────────────────────────────


def _flow_model_ok() -> FakeResponse:
    return FakeResponse(200, {"name": "models/veo-3.1-fast-generate-preview"})


def _flow_connected(http: FakeHttp | None = None,
                    env_key: str = "GOOGLE_FLOW_API_KEY"):
    http = http or FakeHttp()
    conn = GoogleFlowConnector(_vault(), http=http)
    http.route("GET", "/models/veo-3.1-fast-generate-preview",
               _flow_model_ok())
    with mock.patch.dict(os.environ, {env_key: "flow-test-key"}):
        result = conn.connect()
    assert result.ok
    return conn, http


def _op_pending() -> FakeResponse:
    return FakeResponse(200, {
        "name": "models/veo-3.1-fast-generate-preview/operations/op-1",
        "done": False,
    })


def _op_done() -> FakeResponse:
    return FakeResponse(200, {
        "name": "models/veo-3.1-fast-generate-preview/operations/op-1",
        "done": True,
        "response": {
            "generateVideoResponse": {
                "generatedSamples": [
                    {"video": {"uri": "https://video.example/op-1.mp4"}}
                ]
            }
        },
    })


class GoogleFlowTests(unittest.TestCase):
    def test_connect_missing_key_fails_fast(self) -> None:
        conn = GoogleFlowConnector(_vault(), http=FakeHttp())
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(ConnectorError) as ctx:
            conn.connect()
        self.assertIn("GOOGLE_FLOW_API_KEY", str(ctx.exception))

    def test_connect_validates_and_stores(self) -> None:
        conn, http = _flow_connected()
        self.assertTrue(conn.test_connection())
        self.assertTrue(conn.status().connected)

    def test_connect_rejected_key(self) -> None:
        http = FakeHttp()
        conn = GoogleFlowConnector(_vault(), http=http)
        http.route("GET", "/models/veo-3.1-fast-generate-preview",
                   FakeResponse(401, {"error": {"message": "bad key"}}))
        with mock.patch.dict(os.environ, {"GOOGLE_FLOW_API_KEY": "bad"}), self.assertRaises(GoogleFlowError):
            conn.connect()

    def test_disconnect(self) -> None:
        conn, _ = _flow_connected()
        conn.disconnect()
        self.assertFalse(conn.test_connection())

    def test_generate_video_text_to_video(self) -> None:
        conn, http = _flow_connected()
        http.route("POST", ":predictLongRunning", FakeResponse(200, {
            "name": "models/veo-3.1-fast-generate-preview/operations/op-1"
        }))
        http.route("GET", "/operations/op-1", _op_pending())
        http.route("GET", "/operations/op-1", _op_done())
        http.route("GET", "video.example/op-1.mp4",
                   FakeResponse(200, raw=b"FAKEMP4", headers={
                       "content-type": "video/mp4"}))
        with mock.patch("time.sleep", return_value=None):
            result = conn.generate_video("a cat runs", duration_seconds=4,
                                         confirmed=True)
        self.assertEqual(result["video_bytes"], b"FAKEMP4")
        self.assertEqual(result["video_uri"], "https://video.example/op-1.mp4")
        submit = next(
            c for c in http.calls
            if c[0] == "POST" and ":predictLongRunning" in c[1]
        )
        self.assertEqual(submit[0], "POST")
        self.assertIn(":predictLongRunning", submit[1])
        self.assertEqual(submit[3]["x-goog-api-key"], "flow-test-key")
        self.assertEqual(submit[2]["instances"][0]["prompt"], "a cat runs")
        params = submit[2]["parameters"]
        self.assertEqual(params["aspectRatio"], "16:9")
        self.assertEqual(params["durationSeconds"], "4")

    def test_generate_video_with_first_frame(self) -> None:
        conn, http = _flow_connected()
        http.route("POST", ":predictLongRunning", FakeResponse(200, {
            "name": "models/veo-3.1-fast-generate-preview/operations/op-2"
        }))
        http.route("GET", "/operations/op-2", FakeResponse(200, {
            "done": True,
            "response": {"generateVideoResponse": {"generatedSamples": [
                {"video": {"uri": "https://video.example/op-2.mp4"}}]}},
        }))
        http.route("GET", "video.example/op-2.mp4",
                   FakeResponse(200, raw=b"FAKEMP4"))
        with mock.patch("time.sleep", return_value=None):
            result = conn.generate_video(
                "pan left", first_frame=PNG_BYTES,
                first_frame_mime="image/png", confirmed=True)
        self.assertEqual(result["video_bytes"], b"FAKEMP4")
        instance = next(
            c for c in http.calls
            if c[0] == "POST" and ":predictLongRunning" in c[1]
        )[2]["instances"][0]
        self.assertEqual(instance["image"]["inlineData"]["mimeType"],
                         "image/png")

    def test_generate_video_needs_confirmation(self) -> None:
        conn, http = _flow_connected()
        with self.assertRaises(ConnectorError):
            conn.generate_video("a cat runs")

    def test_generate_video_bad_duration(self) -> None:
        conn, http = _flow_connected()
        with self.assertRaises(ConnectorError):
            conn.generate_video("x", duration_seconds=5, confirmed=True)

    def test_generate_video_bad_aspect(self) -> None:
        conn, http = _flow_connected()
        with self.assertRaises(ConnectorError):
            conn.generate_video("x", aspect_ratio="1:1", confirmed=True)

    def test_operation_error_surfaces(self) -> None:
        conn, http = _flow_connected()
        http.route("POST", ":predictLongRunning", FakeResponse(200, {
            "name": "models/veo-3.1-fast-generate-preview/operations/op-9"
        }))
        http.route("GET", "/operations/op-9", FakeResponse(200, {
            "done": True,
            "error": {"code": 3, "message": "blocked by policy"},
        }))
        with mock.patch("time.sleep", return_value=None), self.assertRaises(GoogleFlowError) as ctx:
            conn.generate_video("x", confirmed=True,
                                timeout_seconds=30)
        self.assertIn("blocked by policy", str(ctx.exception))

    def test_timeout(self) -> None:
        conn, http = _flow_connected()
        http.route("POST", ":predictLongRunning", FakeResponse(200, {
            "name": "models/veo-3.1-fast-generate-preview/operations/op-7"
        }))
        http.route("GET", "/operations/op-7", _op_pending())
        with mock.patch("time.sleep", return_value=None), self.assertRaises(GoogleFlowError) as ctx:
            conn.generate_video("x", confirmed=True, timeout_seconds=1)
        self.assertIn("timed out", str(ctx.exception))

    def test_403_explains_paid_tier(self) -> None:
        conn, http = _flow_connected()
        http.route("POST", ":predictLongRunning", FakeResponse(403, {}))
        with self.assertRaises(GoogleFlowError) as ctx:
            conn.generate_video("x", confirmed=True)
        self.assertIn("billing", str(ctx.exception))

    def test_missing_video_uri(self) -> None:
        conn, http = _flow_connected()
        http.route("POST", ":predictLongRunning", FakeResponse(200, {
            "name": "models/veo-3.1-fast-generate-preview/operations/op-8"
        }))
        http.route("GET", "/operations/op-8",
                   FakeResponse(200, {"done": True, "response": {}}))
        with mock.patch("time.sleep", return_value=None), self.assertRaises(GoogleFlowError):
            conn.generate_video("x", confirmed=True,
                                timeout_seconds=30)

    def test_capabilities(self) -> None:
        conn = GoogleFlowConnector(_vault(), http=FakeHttp())
        caps = conn.capabilities()
        self.assertIn("generate_video", caps["can"])
        self.assertIn("image_to_video", caps["can"])

    def test_offline_network_failure_is_clean_error(self) -> None:
        conn, _ = _flow_connected()
        conn.http = DeadHttp()
        with self.assertRaises(GoogleFlowError) as ctx:
            conn.generate_video("x", confirmed=True)
        self.assertIn("network unreachable", str(ctx.exception))


# ── Leonardo ─────────────────────────────────────────────────────


def _leo_me() -> FakeResponse:
    return FakeResponse(200, {"id": "user-1", "username": "devon-owner"})


def _leo_connected(http: FakeHttp | None = None):
    http = http or FakeHttp()
    conn = LeonardoConnector(_vault(), http=http)
    http.route("GET", "/me", _leo_me())
    with mock.patch.dict(os.environ, {"LEONARDO_API_KEY": "leo-test-key"}):
        result = conn.connect()
    assert result.ok
    return conn, http


def _leo_gen_submit() -> FakeResponse:
    return FakeResponse(200, {
        "sdGenerationJob": {"generationId": "gen-123", "apiCreditCost": 25}
    })


def _leo_pending() -> FakeResponse:
    return FakeResponse(200, {
        "generations_by_pk": {"generated_images": [], "status": "PENDING"}
    })


def _leo_complete() -> FakeResponse:
    return FakeResponse(200, {
        "generations_by_pk": {
            "status": "COMPLETE",
            "generated_images": [
                {"id": "img-1", "url": "https://cdn.leonardo.ai/img-1.png"},
                {"id": "img-2", "url": "https://cdn.leonardo.ai/img-2.png"},
            ],
        }
    })


class LeonardoTests(unittest.TestCase):
    def test_connect_missing_key_fails_fast(self) -> None:
        conn = LeonardoConnector(_vault(), http=FakeHttp())
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(ConnectorError) as ctx:
            conn.connect()
        self.assertIn("LEONARDO_API_KEY", str(ctx.exception))

    def test_connect_validates_and_stores(self) -> None:
        conn, http = _leo_connected()
        self.assertTrue(conn.test_connection())
        status = conn.status()
        self.assertTrue(status.connected)
        self.assertEqual(status.account, "devon-owner")

    def test_connect_rejected_key(self) -> None:
        http = FakeHttp()
        conn = LeonardoConnector(_vault(), http=http)
        http.route("GET", "/me", FakeResponse(401, {"error": "bad"}))
        with mock.patch.dict(os.environ, {"LEONARDO_API_KEY": "bad"}), self.assertRaises(LeonardoError):
            conn.connect()

    def test_connect_already_connected(self) -> None:
        conn, _ = _leo_connected()
        with mock.patch.dict(os.environ, {"LEONARDO_API_KEY": "x"}), self.assertRaises(ConnectorError):
            conn.connect()

    def test_disconnect(self) -> None:
        conn, _ = _leo_connected()
        conn.disconnect()
        self.assertFalse(conn.test_connection())

    def test_generate_image(self) -> None:
        conn, http = _leo_connected()
        http.route("POST", "/generations", _leo_gen_submit())
        http.route("GET", "/generations/gen-123", _leo_pending())
        http.route("GET", "/generations/gen-123", _leo_complete())
        with mock.patch("time.sleep", return_value=None):
            images = conn.generate_image("a lighthouse", num_images=2,
                                         width=512, height=512,
                                         confirmed=True)
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0]["url"],
                         "https://cdn.leonardo.ai/img-1.png")
        method, url, payload, headers = next(
            c for c in http.calls
            if c[0] == "POST" and c[1].endswith("/generations")
        )
        self.assertEqual(method, "POST")
        self.assertTrue(url.endswith("/generations"))
        self.assertEqual(headers["Authorization"], "Bearer leo-test-key")
        self.assertEqual(payload["prompt"], "a lighthouse")
        self.assertEqual(payload["num_images"], 2)
        self.assertEqual(payload["width"], 512)

    def test_generate_image_needs_confirmation(self) -> None:
        conn, http = _leo_connected()
        with self.assertRaises(ConnectorError):
            conn.generate_image("a lighthouse")

    def test_generate_image_bad_size(self) -> None:
        conn, http = _leo_connected()
        with self.assertRaises(ConnectorError):
            conn.generate_image("x", width=16, confirmed=True)
        with self.assertRaises(ConnectorError):
            conn.generate_image("x", num_images=9, confirmed=True)

    def test_generate_image_empty_prompt(self) -> None:
        conn, http = _leo_connected()
        with self.assertRaises(ConnectorError):
            conn.generate_image("", confirmed=True)

    def test_generate_timeout(self) -> None:
        conn, http = _leo_connected()
        http.route("POST", "/generations", _leo_gen_submit())
        http.route("GET", "/generations/gen-123", _leo_pending())
        with mock.patch("time.sleep", return_value=None), self.assertRaises(LeonardoError) as ctx:
            conn.generate_image("x", confirmed=True, timeout_seconds=1)
        self.assertIn("timed out", str(ctx.exception))

    def test_403_explains_credits(self) -> None:
        conn, http = _leo_connected()
        http.route("POST", "/generations", FakeResponse(403, {}))
        with self.assertRaises(LeonardoError) as ctx:
            conn.generate_image("x", confirmed=True)
        self.assertIn("credits", str(ctx.exception))

    def test_get_generation(self) -> None:
        conn, http = _leo_connected()
        http.route("GET", "/generations/gen-123", _leo_complete())
        images = conn.get_generation("gen-123")
        self.assertEqual(len(images), 2)

    def test_edit_image(self) -> None:
        conn, http = _leo_connected()
        http.route("POST", "/init-image", FakeResponse(200, {
            "uploadInitImage": {
                "id": "init-9",
                "url": "https://s3.example/upload",
                "fields": json.dumps(
                    {"key": "users/1/images/init-9.png",
                     "policy": "POLICY123"}),
            }
        }))
        http.route("POST", "s3.example/upload", FakeResponse(204))
        http.route("POST", "/generations", _leo_gen_submit())
        http.route("GET", "/generations/gen-123", _leo_complete())
        with mock.patch("time.sleep", return_value=None):
            images = conn.edit_image(PNG_BYTES, "more sunset",
                                     confirmed=True)
        self.assertEqual(images[0]["id"], "img-1")
        upload_call = next(c for c in http.calls
                           if c[0] == "POST" and "s3.example" in c[1])
        self.assertEqual(upload_call[2]["fields"]["policy"], "POLICY123")
        self.assertEqual(upload_call[2]["files"][0][0], "file")
        submit = next(c for c in http.calls
                      if c[0] == "POST" and c[1].endswith("/generations"))
        self.assertEqual(submit[2]["initImageId"], "init-9")
        self.assertEqual(submit[2]["initStrength"], 0.5)

    def test_edit_image_bad_strength(self) -> None:
        conn, http = _leo_connected()
        with self.assertRaises(ConnectorError):
            conn.edit_image(PNG_BYTES, "x", init_strength=1.5,
                            confirmed=True)

    def test_capabilities(self) -> None:
        conn = LeonardoConnector(_vault(), http=FakeHttp())
        caps = conn.capabilities()
        self.assertIn("generate_image", caps["can"])
        self.assertIn("edit_image", caps["can"])

    def test_offline_network_failure_is_clean_error(self) -> None:
        conn, _ = _leo_connected()
        conn.http = DeadHttp()
        with self.assertRaises(LeonardoError) as ctx:
            conn.generate_image("x", confirmed=True, timeout_seconds=5)
        self.assertIn("network unreachable", str(ctx.exception))


# ── Stability AI ─────────────────────────────────────────────────


def _stab_balance() -> FakeResponse:
    return FakeResponse(200, {"credits": 123})


def _stab_connected(http: FakeHttp | None = None):
    http = http or FakeHttp()
    conn = StabilityAIConnector(_vault(), http=http)
    http.route("GET", "/v1/user/balance", _stab_balance())
    with mock.patch.dict(os.environ, {"STABILITY_API_KEY": "stab-test-key"}):
        result = conn.connect()
    assert result.ok
    return conn, http


def _stab_image_ok() -> FakeResponse:
    return FakeResponse(200, {
        "image": PNG_B64,
        "seed": 7,
        "finish_reason": "SUCCESS",
    }, headers={"content-type": "application/json"})


class StabilityAITests(unittest.TestCase):
    def test_connect_missing_key_fails_fast(self) -> None:
        conn = StabilityAIConnector(_vault(), http=FakeHttp())
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(ConnectorError) as ctx:
            conn.connect()
        self.assertIn("STABILITY_API_KEY", str(ctx.exception))

    def test_connect_validates_and_stores(self) -> None:
        conn, http = _stab_connected()
        self.assertTrue(conn.test_connection())
        status = conn.status()
        self.assertTrue(status.connected)
        self.assertIn("123", status.detail)

    def test_connect_rejected_key(self) -> None:
        http = FakeHttp()
        conn = StabilityAIConnector(_vault(), http=http)
        http.route("GET", "/v1/user/balance", FakeResponse(401, {}))
        with mock.patch.dict(os.environ, {"STABILITY_API_KEY": "bad"}), self.assertRaises(StabilityAIError):
            conn.connect()

    def test_connect_already_connected(self) -> None:
        conn, _ = _stab_connected()
        with mock.patch.dict(os.environ, {"STABILITY_API_KEY": "x"}), self.assertRaises(ConnectorError):
            conn.connect()

    def test_disconnect(self) -> None:
        conn, _ = _stab_connected()
        conn.disconnect()
        self.assertFalse(conn.test_connection())

    def test_generate_image_json_response(self) -> None:
        conn, http = _stab_connected()
        http.route("POST", "/v2beta/stable-image/generate/sd3",
                   _stab_image_ok())
        result = conn.generate_image("a robot", aspect_ratio="1:1",
                                     seed=7, confirmed=True)
        self.assertEqual(result["image_bytes"], PNG_BYTES)
        self.assertEqual(result["mime_type"], "image/png")
        self.assertEqual(result["seed"], 7)
        self.assertEqual(result["finish_reason"], "SUCCESS")
        method, url, payload, headers = http.calls[-1]
        self.assertEqual(method, "POST")
        self.assertIn("/v2beta/stable-image/generate/sd3", url)
        self.assertEqual(headers["Authorization"], "Bearer stab-test-key")
        self.assertEqual(headers["Accept"], "application/json")
        fields = payload["fields"]
        self.assertEqual(fields["prompt"], "a robot")
        self.assertEqual(fields["model"], "sd3.5-large")
        self.assertEqual(fields["aspect_ratio"], "1:1")
        self.assertFalse(payload["files"])

    def test_generate_image_raw_bytes_response(self) -> None:
        conn, http = _stab_connected()
        http.route("POST", "/v2beta/stable-image/generate/sd3",
                   FakeResponse(200, raw=PNG_BYTES,
                                headers={"content-type": "image/png"}))
        result = conn.generate_image("a robot", confirmed=True)
        self.assertEqual(result["image_bytes"], PNG_BYTES)
        self.assertEqual(result["mime_type"], "image/png")

    def test_generate_image_needs_confirmation(self) -> None:
        conn, http = _stab_connected()
        with self.assertRaises(ConnectorError):
            conn.generate_image("a robot")

    def test_generate_image_bad_model(self) -> None:
        conn, http = _stab_connected()
        with self.assertRaises(ConnectorError):
            conn.generate_image("x", model="nope", confirmed=True)

    def test_generate_image_bad_aspect(self) -> None:
        conn, http = _stab_connected()
        with self.assertRaises(ConnectorError):
            conn.generate_image("x", aspect_ratio="7:7", confirmed=True)

    def test_generate_image_empty_prompt(self) -> None:
        conn, http = _stab_connected()
        with self.assertRaises(ConnectorError):
            conn.generate_image("", confirmed=True)

    def test_402_explains_credits(self) -> None:
        conn, http = _stab_connected()
        http.route("POST", "/v2beta/stable-image/generate/sd3",
                   FakeResponse(402, {"message": "insufficient credits"}))
        with self.assertRaises(StabilityAIError) as ctx:
            conn.generate_image("x", confirmed=True)
        self.assertIn("credits", str(ctx.exception))

    def test_edit_image(self) -> None:
        conn, http = _stab_connected()
        http.route("POST", "/v2beta/stable-image/generate/sd3",
                   _stab_image_ok())
        result = conn.edit_image(PNG_BYTES, "older robot", strength=0.8,
                                 confirmed=True)
        self.assertEqual(result["image_bytes"], PNG_BYTES)
        payload = http.calls[-1][2]
        fields = payload["fields"]
        self.assertEqual(fields["mode"], "image-to-image")
        self.assertEqual(fields["strength"], "0.8")
        self.assertTrue(payload["files"])
        self.assertEqual(payload["files"][0][0], "image")

    def test_edit_image_bad_strength(self) -> None:
        conn, http = _stab_connected()
        with self.assertRaises(ConnectorError):
            conn.edit_image(PNG_BYTES, "x", strength=2.0, confirmed=True)

    def test_balance(self) -> None:
        conn, http = _stab_connected()
        http.route("GET", "/v1/user/balance", _stab_balance())
        self.assertEqual(conn.balance()["credits"], 123)

    def test_capabilities(self) -> None:
        conn = StabilityAIConnector(_vault(), http=FakeHttp())
        caps = conn.capabilities()
        self.assertIn("generate_image", caps["can"])
        self.assertIn("edit_image", caps["can"])
        self.assertIn("balance", caps["can"])

    def test_offline_network_failure_is_clean_error(self) -> None:
        conn, _ = _stab_connected()
        conn.http = DeadHttp()
        with self.assertRaises(StabilityAIError) as ctx:
            conn.generate_image("x", confirmed=True)
        self.assertIn("network unreachable", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
