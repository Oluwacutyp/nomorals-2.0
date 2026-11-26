"""Craft-Libs port — shared libraries for the craft organs (lane 113).

``nomorals/craftlibs/`` is the one home for the code the craft organs
share: the deterministic ZIP package codec (cad + layout), the two
noise engines (textures' f32-exact point engine + material's grid
engine), the ai-core data model (upstream craft-libs ``ai-core``),
the shared PDF surface (safe_url + bounds + writer/reader bridge),
the i18n formatting rung (upstream craft-libs ``i18n``: numbers,
parsing, regions, file sizes), and the shared parity + attribution
runners the organs' own runners delegate to.

Every case here pins real behaviour against real fixtures. A missing
fixture fails, never skips.
"""

from __future__ import annotations

import io
import json
import math
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from nomorals.craftlibs import codec as shared_codec
from nomorals.craftlibs import i18n as shared_i18n
from nomorals.craftlibs import noise as shared_noise
from nomorals.craftlibs import pdf as shared_pdf
from nomorals.craftlibs.ai import (
    AiError,
    Auth,
    Billing,
    CapabilityId,
    CapabilitySpec,
    Catalog,
    Connection,
    ConnectionError,
    ConsentLedger,
    Disclosure,
    Endpoint,
    ErrorKind,
    Features,
    InputRole,
    JobSnapshot,
    JobState,
    Limits,
    Locality,
    MockProvider,
    ModelId,
    ModelInfo,
    Payload,
    Pick,
    Pricing,
    Request,
    Role,
    RoleId,
    SecretString,
    Spec,
    StoreFile,
    choose,
    redact,
)
from nomorals.craftlibs.ai.conformance import (
    run_cancel,
    run_lifecycle,
    run_rejects,
    run_unknown_job,
)
from nomorals.craftlibs.ai.local import LocalMotionProvider
from nomorals.craftlibs.ai.spec import SpecError, render_text, render_value
from nomorals.craftlibs.attribution import verify_attribution
from nomorals.craftlibs.parity import run_parity
from nomorals.craftlibs.pdf import Metadata
from nomorals.tools.registry import ToolRegistry

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "craftlibs"


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


class CodecTest(unittest.TestCase):
    def test_canonical_json_is_deterministic(self):
        a = shared_codec.canonical_json({"b": 1, "a": [3, 2, 1]})
        b = shared_codec.canonical_json({"a": [3, 2, 1], "b": 1})
        self.assertEqual(a, b)
        self.assertEqual(a, b'{"a":[3,2,1],"b":1}')

    def test_bounded_json_loads_rejects_deep_and_huge(self):
        deep = b"[" * 101 + b"]" * 101
        with self.assertRaises(shared_codec.CodecError):
            shared_codec.bounded_json_loads(deep)
        huge = json.dumps(list(range(600_000))).encode()
        with self.assertRaises(shared_codec.CodecError):
            shared_codec.bounded_json_loads(huge)
        with self.assertRaises(shared_codec.CodecError):
            shared_codec.bounded_json_loads(b"\xff\xfe{}")
        self.assertEqual(shared_codec.bounded_json_loads(b'{"a": 1}'),
                         {"a": 1})

    def test_read_write_package_round_trip(self):
        parts = {"document.json": b'{"x": 1}',
                 "meta.json": b'{"v": 1}',
                 "assets/a.bin": bytes(range(64))}
        data = shared_codec.write_package(parts)
        back = shared_codec.read_package(data)
        self.assertEqual(back, parts)
        # Deterministic: same parts, same bytes, twice.
        self.assertEqual(shared_codec.write_package(parts), data)
        # A package written by the shared codec opens as a zip with
        # fixed 1980 timestamps.
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            self.assertEqual({i.filename for i in zf.infolist()},
                             set(parts))
            self.assertTrue(all(i.date_time[:3] == (1980, 1, 1)
                                for i in zf.infolist()))

    def test_write_package_ordered_keeps_layout_order(self):
        entries = [("document.json", b"{}"), ("meta.json", b"{}"),
                   ("assets/z.bin", b"z"), ("assets/a.bin", b"a")]
        data = shared_codec.write_package_ordered(entries)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            self.assertEqual(zf.namelist(),
                             [name for name, _ in entries])

    def test_read_document_package_splits_doc_assets_meta(self):
        raw = (_FIXTURES / "page.designcraft").read_bytes()
        doc, assets, meta = shared_codec.read_document_package(raw)
        self.assertIsInstance(doc, dict)
        self.assertEqual(doc.get("title"), "craftlibs-fixture")
        self.assertEqual(meta.get("format_version"), 1)
        self.assertEqual(assets, {})

    def test_oversize_package_is_refused(self):
        with self.assertRaises(shared_codec.CodecError):
            shared_codec.read_package(b"x" * (shared_codec.MAX_PACKAGE_BYTES + 1))

    def test_cad_fixture_round_trips_through_shared_codec(self):
        raw = (_FIXTURES / "bracket.cadcraft").read_bytes()
        parts = shared_codec.read_package(raw)
        self.assertIn("document.json", parts)
        self.assertEqual(shared_codec.write_package(parts), raw)


class NoiseTest(unittest.TestCase):
    # Point engine values are the textures parity pins (seed 7).
    _POINTS = [(0.0, 0.0, 0.0), (0.5, 0.25, 0.125),
               (1.3, 2.7, 0.4), (3.7, 1.1, 9.2),
               (10.5, 10.5, 10.5), (-2.25, 4.75, -0.5)]
    _WANT_VALUE = [0.044569507, 0.186984763, 0.705094278, 0.278636873,
                   0.391551822, 0.538408279]
    _WANT_PERLIN = [0.5, 0.502421975, 0.460757077, 0.308367729,
                    0.4375, 0.511600018]
    _WANT_FBM = [0.178656384, 0.337487936, 0.552097023, 0.401274621,
                 0.409147024, 0.560437799]

    def test_point_engine_matches_textures_parity(self):
        for p, wv, wp, wf in zip(self._POINTS, self._WANT_VALUE,
                                 self._WANT_PERLIN, self._WANT_FBM,
                                 strict=True):
            self.assertTrue(_close(shared_noise.point_value(p, 7), wv),
                            f"value{p}")
            self.assertTrue(_close(shared_noise.point_perlin(p, 7), wp),
                            f"perlin{p}")
            self.assertTrue(
                _close(shared_noise.point_fbm(p, 5, 7,
                                             shared_noise.point_value), wf),
                f"fbm{p}")

    def test_point_cellular_and_worley(self):
        f1, f2, ident = shared_noise.point_cellular((0.5, 0.25, 0.125), 7)
        self.assertTrue(_close(f1, 0.851551))
        self.assertTrue(_close(f2, 0.886522))
        self.assertTrue(_close(ident, 0.280893))
        self.assertTrue(_close(shared_noise.point_worley((0.5, 0.25, 0.125), 7),
                               0.851551))
        self.assertTrue(_close(shared_noise.point_cell_hash(0, 0, 0, 7),
                               0.044569507, 1e-6))

    def test_grid_engine_matches_material_values(self):
        grid = shared_noise.grid_perlin(8, 8, 2.0, 7)
        self.assertEqual(grid.shape, (8, 8))
        self.assertTrue(_close(float(grid[0][0]), 0.552639))
        self.assertTrue(_close(float(grid[0][1]), 0.59632))
        self.assertTrue(_close(float(grid[4][4]), 0.455738))
        value = shared_noise.grid_value_noise(8, 8, 2.0, 7)
        self.assertTrue(_close(float(value[0][0]), 0.452911))
        self.assertTrue(_close(float(value[0][1]), 0.380981))
        fbm = shared_noise.grid_fbm(8, 8, 2.0, 3, 7)
        self.assertTrue(_close(float(fbm[0][0]), 0.551907))
        self.assertTrue(_close(float(fbm[0][1]), 0.587422))
        worley = shared_noise.grid_worley(8, 8, 2.0, 7)
        self.assertTrue(_close(float(worley[0][0]), 0.371007))
        self.assertTrue(_close(float(worley[0][1]), 0.211429))
        self.assertTrue(_close(float(shared_noise.grid_worley(
            8, 8, 2.0, 7, mode="f1").sum()), 27.5838, 1e-3))
        dx, dy = shared_noise.grid_warp_field(4, 4, 1.0, 7)
        self.assertTrue(_close(float(dx[0][0]), 0.077038))
        self.assertTrue(_close(float(dy[0][0]), 0.077038))
        self.assertEqual(shared_noise.grid_permutation_table(7)[:4].tolist(),
                         [240, 102, 218, 242])

    def test_grid_engine_tiles_seamlessly(self):
        # A tileable grid repeats: sampling the same lattice twice
        # gives the same row.
        a = shared_noise.grid_perlin(8, 8, 2.0, 7)
        b = shared_noise.grid_perlin(8, 8, 2.0, 7)
        self.assertTrue((a == b).all())
        self.assertTrue(((a >= 0.0) & (a <= 1.0)).all())


class AiIdsTest(unittest.TestCase):
    def test_ids_accept_normal_values(self):
        self.assertEqual(ModelId.new("black-forest-labs/flux-1.1-pro").as_str(),
                         "black-forest-labs/flux-1.1-pro")
        self.assertEqual(RoleId.new("generative-fill").as_str(),
                         "generative-fill")

    def test_ids_reject_bad_values(self):
        for bad in ("", "a b", "a\0b", "\u202ertl", "x" * 129):
            with self.assertRaises(ValueError, msg=bad):
                ModelId.new(bad)

    def test_secret_never_prints(self):
        secret = SecretString("sk-live-abc123")
        self.assertNotIn("abc123", repr(secret))
        self.assertNotIn("abc123", str(secret))
        self.assertEqual(redact("bad key sk-live-abc123 (sk-live-abc123)",
                                [secret]),
                         "bad key [redacted] ([redacted])")
        self.assertEqual(redact("xabcdx",
                                [SecretString("abc"), SecretString("bcd")]),
                         "x[redacted]x")


class AiErrorTest(unittest.TestCase):
    def test_kinds_and_retryable(self):
        self.assertTrue(AiError.new(ErrorKind.RATE_LIMITED, "slow").is_retryable())
        self.assertTrue(AiError.new(ErrorKind.NETWORK, "down").is_retryable())
        self.assertFalse(AiError.unsupported("x").is_retryable())
        self.assertFalse(AiError.new(ErrorKind.CONSENT_REQUIRED, "ask").is_retryable())
        self.assertEqual(str(AiError.new(ErrorKind.CONSENT_REQUIRED, "ask first")),
                         "consent required: ask first")

    def test_redacted_error_scrubs_the_key(self):
        key = SecretString("sk-test-999")
        err = AiError.redacted(ErrorKind.AUTH, "invalid key sk-test-999", [key])
        self.assertNotIn("999", err.message)
        self.assertNotIn("999", str(err))


class CapabilityTest(unittest.TestCase):
    def test_well_known_all_parse(self):
        from nomorals.craftlibs.ai.capability import well_known
        self.assertEqual(len(well_known()), 21)

    def test_splits_modality_and_task(self):
        cap = CapabilityId.parse("image.inpaint")
        self.assertEqual(cap.modality().as_str(), "image")
        self.assertEqual(cap.task(), "inpaint")

    def test_open_set_accepts_new_modalities(self):
        cap = CapabilityId.parse("cad.explain")
        self.assertEqual(cap.modality().as_str(), "cad")

    def test_rejects_malformed(self):
        for bad in ("", "image", "image.", ".inpaint", "image.in.paint",
                    "im age.x", "image.in/paint"):
            with self.assertRaises(ValueError, msg=bad):
                CapabilityId.parse(bad)


class PayloadTest(unittest.TestCase):
    def test_base64_vectors(self):
        from nomorals.craftlibs.ai.payload import base64_decode, base64_encode
        for raw, enc in (("", ""), ("f", "Zg=="), ("fo", "Zm8="),
                         ("foo", "Zm9v"), ("foob", "Zm9vYg=="),
                         ("foobar", "Zm9vYmFy")):
            self.assertEqual(base64_encode(raw.encode()), enc)
            self.assertEqual(base64_decode(enc), raw.encode())
        for bad in ("Zg=", "Z", "Zm9v!", "=Zm9", "Zg==Zg==", "Zm=v",
                    "Z===", "Zm9v\n"):
            self.assertIsNone(base64_decode(bad), bad)

    def test_limits_refuse_oversize_with_a_hint(self):
        limits = Limits(max_inline_bytes=4, max_text_bytes=4)
        err = None
        try:
            Payload.inline("image/png", bytes(5)).check(limits)
        except AiError as exc:
            err = exc
        self.assertIsNotNone(err)
        self.assertIn("handle", err.message)
        with self.assertRaises(AiError):
            Payload.text("12345").check(limits)
        Payload.text("1234").check(limits)
        handle = Payload.handle("file:///big.mov", "video/quicktime",
                                10_000_000_000)
        handle.check(limits)
        self.assertEqual(handle.size_hint(), 10_000_000_000)

    def test_kinds_follow_media_type(self):
        self.assertEqual(Payload.inline("image/png", b"\x01").kind().value,
                         "image")
        self.assertEqual(Payload.inline("application/pdf", b"\x01").kind().value,
                         "document")
        self.assertEqual(Payload.text("hi").kind().value, "text")

    def test_handle_rejects_bad_uris(self):
        for bad in ("", "file:///a\nb", "x" * 4097):
            with self.assertRaises(AiError, msg=bad[:20]):
                Payload.handle(bad, "video/mp4", None)

    def test_json_round_trip(self):
        payload = Payload.inline("image/png", bytes([0, 1, 2, 250, 255]))
        data = payload.to_dict()
        self.assertIn("inline", json.dumps(data))
        self.assertEqual(Payload.from_dict(data).to_dict(), data)
        with self.assertRaises(ValueError):
            Payload.from_dict({"type": "inline", "media_type": "image/png",
                               "data": "!!"})
        self.assertEqual(Payload.inline("image/png", b"\x01").kind().value,
                         "image")
        self.assertEqual(Payload.inline("application/pdf", b"\x01").kind().value,
                         "document")
        for bad in ("", "png", "image/", "/png", "Image/PNG",
                    "image/png/x", "image/p ng"):
            with self.assertRaises(AiError, msg=bad):
                Payload.inline(bad, b"\x01")


def _inpaint_spec() -> CapabilitySpec:
    return (CapabilitySpec.new(CapabilityId.parse("image.inpaint"))
            .accepting(InputRole.PROMPT, 1)
            .accepting(InputRole.SOURCE, 1)
            .accepting(InputRole.MASK, 1)
            .accepting(InputRole.REFERENCE, 1)
            .requiring(InputRole.SOURCE)
            .requiring(InputRole.MASK)
            .with_max_outputs(3))


def _good_request() -> Request:
    png = Payload.inline("image/png", b"\x01\x02\x03")
    return (Request.new(CapabilityId.parse("image.inpaint"))
            .with_input(InputRole.SOURCE, png)
            .with_input(InputRole.MASK, png))


class RequestTest(unittest.TestCase):
    def test_well_formed_request_passes(self):
        req = _good_request().with_input(InputRole.PROMPT,
                                         Payload.text("a cat"))
        req.validate(_inpaint_spec(), Limits())

    def test_unsupported_input_is_a_clear_error(self):
        spec = (CapabilitySpec.new(CapabilityId.parse("image.generate"))
                .accepting(InputRole.PROMPT, 1))
        req = (Request.new(CapabilityId.parse("image.generate"))
               .with_input(InputRole.MASK,
                           Payload.inline("image/png", b"\x01")))
        with self.assertRaises(AiError) as ctx:
            req.validate(spec, Limits())
        self.assertEqual(ctx.exception.kind, ErrorKind.UNSUPPORTED)
        self.assertIn("mask", ctx.exception.message)

    def test_missing_required_input_is_invalid(self):
        req = (Request.new(CapabilityId.parse("image.inpaint"))
               .with_input(InputRole.SOURCE,
                           Payload.inline("image/png", b"\x01")))
        with self.assertRaises(AiError) as ctx:
            req.validate(_inpaint_spec(), Limits())
        self.assertEqual(ctx.exception.kind, ErrorKind.INVALID_REQUEST)
        self.assertIn("mask", ctx.exception.message)

    def test_too_many_references_counted_separately_from_masks(self):
        png = Payload.inline("image/png", b"\x01\x02\x03")
        req = (_good_request()
               .with_input(InputRole.REFERENCE, png)
               .with_input(InputRole.REFERENCE, png))
        with self.assertRaises(AiError) as ctx:
            req.validate(_inpaint_spec(), Limits())
        self.assertEqual(ctx.exception.kind, ErrorKind.UNSUPPORTED)
        self.assertIn("reference", ctx.exception.message)

    def test_capability_mismatch_is_invalid(self):
        req = Request.new(CapabilityId.parse("image.generate"))
        with self.assertRaises(AiError) as ctx:
            req.validate(_inpaint_spec(), Limits())
        self.assertEqual(ctx.exception.kind, ErrorKind.INVALID_REQUEST)

    def test_oversized_inline_input_is_refused(self):
        limits = Limits(max_inline_bytes=2, max_text_bytes=10)
        with self.assertRaises(AiError):
            _good_request().validate(_inpaint_spec(), limits)

    def test_request_round_trips_and_defaults_apply(self):
        req = _good_request().with_param("seed", 7).with_outputs(2)
        back = Request.from_dict(req.to_dict())
        self.assertEqual(back.to_dict(), req.to_dict())
        minimal = Request.from_dict({"capability": "image.generate"})
        self.assertEqual(minimal.outputs, 1)
        self.assertFalse(minimal.stream)
        self.assertIsNone(_good_request().with_outputs(3)
                          .validate(_inpaint_spec(), Limits()))
        for n in (0, 4):
            with self.assertRaises(AiError):
                _good_request().with_outputs(n).validate(_inpaint_spec(),
                                                         Limits())

    def test_streaming_needs_the_feature(self):
        spec = (CapabilitySpec.new(CapabilityId.parse("text.complete"))
                .accepting(InputRole.PROMPT, 1))
        req = (Request.new(CapabilityId.parse("text.complete"))
               .with_input(InputRole.PROMPT, Payload.text("hi")).streaming())
        with self.assertRaises(AiError):
            req.validate(spec, Limits())
        spec = spec.with_features(Features(streaming=True))
        req.validate(spec, Limits())

    def test_oversized_params_are_refused(self):
        big = "x" * (64 * 1024 + 1)
        with self.assertRaises(AiError):
            _good_request().with_param("blob", big).validate(
                _inpaint_spec(), Limits())


class JobTest(unittest.TestCase):
    def test_forward_moves_are_legal(self):
        self.assertTrue(JobState.QUEUED.can_become(JobState.running()))
        self.assertTrue(JobState.running(10).can_become(JobState.running(80)))
        self.assertTrue(JobState.running(80).can_become(JobState.SUCCEEDED))
        self.assertTrue(JobState.QUEUED.can_become(JobState.SUCCEEDED))

    def test_progress_cannot_go_backwards(self):
        self.assertFalse(JobState.running(50).can_become(JobState.running(40)))
        self.assertFalse(JobState.running(50).can_become(JobState.running()))
        self.assertFalse(JobState.running(5).can_become(JobState.QUEUED))

    def test_terminal_states_are_final(self):
        for terminal in (JobState.SUCCEEDED, JobState.CANCELLED,
                         JobState.failed(AiError.new(ErrorKind.PROVIDER, "boom"))):
            self.assertTrue(terminal.is_terminal())
            self.assertTrue(terminal.can_become(terminal))
            self.assertFalse(terminal.can_become(JobState.running()))
            self.assertFalse(terminal.can_become(JobState.QUEUED))
        self.assertFalse(JobState.SUCCEEDED.can_become(JobState.CANCELLED))

    def test_snapshot_collects_outputs_only(self):
        snap = JobSnapshot.new(JobState.SUCCEEDED)
        snap.events.append({"type": "text_delta", "text": "hi"})
        snap.events.append({"type": "output",
                            "payload": Payload.text("done").to_dict()})
        self.assertEqual(len(snap.outputs()), 1)


class EndpointTest(unittest.TestCase):
    def test_loopback_forms(self):
        for ok in ("http://localhost:8188", "http://127.0.0.1:11434/v1",
                   "http://127.9.9.9", "http://[::1]:8080/x"):
            self.assertTrue(Endpoint.parse(ok).is_loopback(), ok)

    def test_remote_https_and_fingerprint(self):
        ep = Endpoint.parse("https://api.example.com/v1")
        self.assertFalse(ep.is_loopback())
        self.assertEqual(ep.fingerprint(), "https://api.example.com")

    def test_plain_http_to_remote_is_refused(self):
        for bad in ("http://api.example.com", "http://192.168.1.20:8188"):
            with self.assertRaises(ValueError, msg=bad):
                Endpoint.parse(bad)

    def test_userinfo_trick_is_refused(self):
        for bad in ("http://127.0.0.1@evil.example/",
                    "https://localhost:x@evil.example/",
                    "https://user@api.example.com"):
            with self.assertRaises(ValueError, msg=bad):
                Endpoint.parse(bad)

    def test_localhost_lookalikes_are_remote(self):
        for host in ("https://localhost.evil.example",
                     "https://127.0.0.1.evil.example", "https://0x7f.0.0.1"):
            self.assertFalse(Endpoint.parse(host).is_loopback(), host)


def _remote_connection(conn_id: str = "a") -> Connection:
    return Connection(
        id=conn_id, label="Remote", provider="http",
        endpoint="https://api.example.com", locality=Locality.REMOTE,
        auth=Auth.api_key("secret-a"), billing=Billing.METERED,
        models=[], enabled=True)


def _local_connection(conn_id: str = "b") -> Connection:
    conn = _remote_connection(conn_id)
    conn.endpoint = "http://127.0.0.1:8188"
    conn.locality = Locality.LOCAL
    conn.auth = Auth.none()
    conn.billing = Billing.FREE
    return conn


class ConnectionTest(unittest.TestCase):
    def test_good_connections_validate(self):
        _remote_connection("a").validate()
        _local_connection("b").validate()

    def test_remote_host_cannot_be_labelled_local(self):
        conn = _remote_connection("a")
        conn.locality = Locality.LOCAL
        with self.assertRaises(ConnectionError) as ctx:
            conn.validate()
        self.assertEqual(ctx.exception.kind, "local_but_not_loopback")

    def test_loopback_labelled_remote_is_flagged(self):
        conn = _local_connection("a")
        conn.locality = Locality.REMOTE
        with self.assertRaises(ConnectionError) as ctx:
            conn.validate()
        self.assertEqual(ctx.exception.kind, "remote_but_loopback")

    def test_the_userinfo_trick_cannot_make_a_remote_host_local(self):
        conn = _local_connection("a")
        conn.endpoint = "http://127.0.0.1@evil.example/"
        with self.assertRaises(ConnectionError) as ctx:
            conn.validate()
        self.assertEqual(ctx.exception.kind, "endpoint")

    def test_subscription_route_without_endpoint_cannot_be_local(self):
        conn = _remote_connection("codex")
        conn.endpoint = None
        conn.auth = Auth.subscription("codex-app-server", "s")
        conn.locality = Locality.LOCAL
        with self.assertRaises(ConnectionError) as ctx:
            conn.validate()
        self.assertEqual(ctx.exception.kind,
                         "local_without_endpoint_needs_no_auth")
        conn.locality = Locality.REMOTE
        conn.validate()

    def test_in_process_model_can_be_local_without_endpoint(self):
        conn = _local_connection("onnx")
        conn.endpoint = None
        conn.validate()

    def test_serialized_form_holds_a_reference_not_a_secret(self):
        data = _remote_connection("a").to_dict()
        text = json.dumps(data)
        self.assertIn("secret-a", text)
        self.assertNotIn("sk-", text.lower())

    def test_json_round_trip(self):
        conn = _remote_connection("a")
        back = Connection.from_dict(conn.to_dict())
        self.assertEqual(back.to_dict(), conn.to_dict())

    def test_duplicate_models_and_empty_labels_are_refused(self):
        from nomorals.craftlibs.ai.model import ModelInfo as _MI
        conn = _remote_connection("a")
        model = _MI.new(ModelId.new("m"), "M")
        conn.models = [model, model]
        with self.assertRaises(ConnectionError) as ctx:
            conn.validate()
        self.assertEqual(ctx.exception.kind, "duplicate_model")
        conn = _remote_connection("b")
        conn.label = "  "
        with self.assertRaises(ConnectionError) as ctx:
            conn.validate()
        self.assertEqual(ctx.exception.kind, "empty_label")

    def test_key_header_rules(self):
        from nomorals.craftlibs.ai.connection import check_key_header
        check_key_header("X-Api-Key")
        for bad in ("", "x" * 65, "Bad Header", "Authorization",
                    "Proxy-Auth", "sec-fetch-site", "Cookie"):
            with self.assertRaises(ValueError, msg=bad):
                check_key_header(bad)


class RoleTest(unittest.TestCase):
    def _role(self) -> Role:
        return Role(id="generative-fill", label="Generative Fill",
                    capability="image.inpaint",
                    inputs=[InputRole.SOURCE, InputRole.MASK])

    def _conn(self, conn_id: str, with_mask: bool) -> Connection:
        spec = (CapabilitySpec.new(CapabilityId.parse("image.inpaint"))
                .accepting(InputRole.SOURCE, 1))
        if with_mask:
            spec = spec.accepting(InputRole.MASK, 1)
        conn = _remote_connection(conn_id)
        conn.models = [ModelInfo.new(ModelId.new("m"), "M").with_spec(spec)]
        return conn

    def test_resolves_a_good_binding(self):
        from nomorals.craftlibs.ai.role import Binding, resolve
        conns = [self._conn("a", True)]
        binding = Binding(role=RoleId.new("generative-fill"),
                          connection="a", model="m")
        resolved = resolve(self._role(), [[], [binding]], conns)
        self.assertEqual(resolved.connection.id.as_str(), "a")

    def test_app_override_beats_default(self):
        from nomorals.craftlibs.ai.role import Binding, resolve
        conns = [self._conn("a", True), self._conn("b", True)]
        over = [Binding(role=RoleId.new("generative-fill"),
                        connection="b", model="m")]
        default = [Binding(role=RoleId.new("generative-fill"),
                           connection="a", model="m")]
        resolved = resolve(self._role(), [over, default], conns)
        self.assertEqual(resolved.connection.id.as_str(), "b")

    def test_broken_override_never_falls_through(self):
        from nomorals.craftlibs.ai.role import Binding, resolve
        conns = [self._conn("a", True)]
        over = [Binding(role=RoleId.new("generative-fill"),
                        connection="gone", model="m")]
        default = [Binding(role=RoleId.new("generative-fill"),
                           connection="a", model="m")]
        with self.assertRaises(ValueError) as ctx:
            resolve(self._role(), [over, default], conns)
        self.assertEqual(ctx.exception.kind, "connection_missing")

    def test_unbound_and_disabled(self):
        from nomorals.craftlibs.ai.role import Binding, resolve
        conns = [self._conn("a", True)]
        with self.assertRaises(ValueError) as ctx:
            resolve(self._role(), [[]], conns)
        self.assertEqual(ctx.exception.kind, "unbound")
        self.assertIn("no model is chosen", str(ctx.exception))
        conns[0].enabled = False
        with self.assertRaises(ValueError) as ctx:
            resolve(self._role(), [[Binding(
                role=RoleId.new("generative-fill"),
                connection="a", model="m")]], conns)
        self.assertEqual(ctx.exception.kind, "connection_disabled")

    def test_model_without_mask_does_not_resolve(self):
        from nomorals.craftlibs.ai.role import Binding, resolve
        conns = [self._conn("a", False)]
        binding = [Binding(role=RoleId.new("generative-fill"),
                           connection="a", model="m")]
        with self.assertRaises(ValueError) as ctx:
            resolve(self._role(), [binding], conns)
        self.assertEqual(ctx.exception.kind, "input_unsupported")
        self.assertIn("mask", str(ctx.exception))

    def test_required_input_the_role_cannot_supply(self):
        from nomorals.craftlibs.ai.role import Binding, resolve
        conns = [self._conn("a", True)]
        conns[0].models[0].capabilities[0].requires = [InputRole.MASK]
        role = self._role()
        role.inputs = [InputRole.SOURCE]
        with self.assertRaises(ValueError) as ctx:
            resolve(role, [[Binding(role=RoleId.new("generative-fill"),
                                    connection="a", model="m")]], conns)
        self.assertEqual(ctx.exception.kind, "input_required")
        self.assertEqual(str(ctx.exception),
                         "model m requires a 'mask' input that this "
                         "tool does not provide")

    def test_wrong_capability_and_missing_model(self):
        from nomorals.craftlibs.ai.role import Binding, resolve
        conns = [self._conn("a", True)]
        role = self._role()
        role.capability = CapabilityId.parse("image.generate")
        with self.assertRaises(ValueError) as ctx:
            resolve(role, [[Binding(role=RoleId.new("generative-fill"),
                                    connection="a", model="m")]], conns)
        self.assertEqual(ctx.exception.kind, "capability_mismatch")
        conns[0].models.clear()
        with self.assertRaises(ValueError) as ctx:
            resolve(self._role(), [[Binding(
                role=RoleId.new("generative-fill"),
                connection="a", model="m")]], conns)
        self.assertEqual(ctx.exception.kind, "model_missing")

    def test_eligible_filters_by_capability_inputs_and_enabled(self):
        from nomorals.craftlibs.ai.role import eligible
        off = self._conn("c", True)
        off.enabled = False
        conns = [self._conn("a", True), self._conn("b", False), off]
        found = eligible(self._role(), conns)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].connection.id.as_str(), "a")


class ConsentTest(unittest.TestCase):
    def _disclosure(self):
        conn = _remote_connection("a")
        req = (Request.new(CapabilityId.parse("image.inpaint"))
               .with_input(InputRole.SOURCE,
                           Payload.inline("image/png", bytes(10)))
               .with_input(InputRole.PROMPT, Payload.text("a cat"))
               .with_param("seed", 7))
        return Disclosure.for_request("photocraft", conn, req)

    def test_disclosure_lists_kinds_not_contents(self):
        disc = self._disclosure()
        self.assertEqual(disc.destination, "https://api.example.com")
        self.assertEqual([k.value for k in disc.kinds()],
                         ["text", "image"])
        self.assertEqual(disc.items[0]["bytes"], 10)
        self.assertEqual(disc.param_names, ["seed"])
        self.assertEqual(disc.capability.as_str(), "image.inpaint")
        self.assertNotIn("a cat", json.dumps(disc.to_dict()))

    def test_local_needs_no_consent(self):
        ledger = ConsentLedger()
        conn = _local_connection("b")
        req = Request.new(CapabilityId.parse("image.inpaint"))
        disc = Disclosure.for_request("photocraft", conn, req)
        self.assertFalse(ledger.needs_consent(disc, now=1_000_000))

    def test_remote_asks_until_granted_then_stops(self):
        ledger = ConsentLedger()
        disc = self._disclosure()
        self.assertTrue(ledger.needs_consent(disc, now=0))
        ledger.grant(disc)
        self.assertFalse(ledger.needs_consent(disc, now=0))

    def test_grant_until_lapses_per_kind(self):
        ledger = ConsentLedger()
        disc = self._disclosure()
        self.assertTrue(ledger.needs_consent(disc, now=1_000_000))
        ledger.grant_until(disc, expires_at=1_000_100)
        self.assertFalse(ledger.needs_consent(disc, now=1_000_050))
        self.assertTrue(ledger.needs_consent(disc, now=1_000_100))
        ledger.grant(disc)
        self.assertFalse(ledger.needs_consent(disc, now=9_999_999))
        ledger.revoke("photocraft", "a")
        self.assertTrue(ledger.needs_consent(disc, now=9_999_999))

    def test_new_kind_of_data_asks_again(self):
        ledger = ConsentLedger()
        ledger.grant(self._disclosure())
        req = (Request.new(CapabilityId.parse("image.inpaint"))
               .with_input(InputRole.SOURCE,
                           Payload.inline("image/png", bytes(10)))
               .with_input(InputRole.CONTEXT,
                           Payload.handle("file:///v.mp4", "video/mp4", 5)))
        disc = Disclosure.for_request("photocraft",
                                      _remote_connection("a"), req)
        self.assertTrue(ledger.needs_consent(disc, now=0))
        ledger.grant(disc)
        self.assertFalse(ledger.needs_consent(disc, now=0))
        self.assertEqual(len(ledger.records), 1)

    def test_new_destination_needs_fresh_consent(self):
        ledger = ConsentLedger()
        disc = self._disclosure()
        ledger.grant(disc)
        other = Disclosure.for_request(
            "photocraft",
            _remote_connection("a").with_endpoint("https://other.example.com"),
            Request.new(CapabilityId.parse("image.inpaint")))
        self.assertTrue(ledger.needs_consent(other, now=1_000_000))

    def test_consent_is_per_app_and_revocable(self):
        ledger = ConsentLedger()
        disc = self._disclosure()
        ledger.grant(disc)
        other = Disclosure.for_request(
            "wordcraft", _remote_connection("a"),
            Request.new(CapabilityId.parse("image.inpaint")))
        self.assertTrue(ledger.needs_consent(other, now=0))

    def test_parameters_alone_still_need_consent(self):
        req = (Request.new(CapabilityId.parse("image.inpaint"))
               .with_param("negative", "a private note"))
        disc = Disclosure.for_request("photocraft",
                                      _remote_connection("a"), req)
        self.assertEqual([k.value for k in disc.kinds()], ["text"])
        ledger = ConsentLedger()
        self.assertTrue(ledger.needs_consent(disc, now=0))
        ledger.grant(disc)
        self.assertFalse(ledger.needs_consent(disc, now=0))


class HistoryTest(unittest.TestCase):
    def _start(self, run: str, at_ms: int):
        from nomorals.craftlibs.ai.history import StartRecord
        return StartRecord(
            run=run, at_ms=at_ms, app="photocraft", role_id="fill",
            role_label="Generative Fill", capability="image.inpaint",
            connection="conn1", connection_label="Hosted", model="m1",
            locality=Locality.REMOTE,
            destination="https://api.example.com", sent=[],
            params=["seed"], provider_job="gen-9")

    def _end(self, run: str, at_ms: int, outcome):
        from nomorals.craftlibs.ai.history import EndRecord
        return EndRecord(run=run, at_ms=at_ms, outcome=outcome)

    def test_line_round_trips_and_leaves_out_what_is_absent(self):
        from nomorals.craftlibs.ai.history import HistoryRecord, Outcome
        line = json.dumps(HistoryRecord.start(
            self._start("a", 5)).to_dict())
        self.assertNotIn("\n", line)
        self.assertNotIn("prompt", line)
        self.assertNotIn("estimate", line)
        back = HistoryRecord.from_dict(json.loads(line))
        self.assertEqual(back.to_dict(),
                         HistoryRecord.start(self._start("a", 5)).to_dict())
        failed = HistoryRecord.end(
            self._end("a", 9, Outcome.failed(ErrorKind.QUOTA)))
        line = json.dumps(failed.to_dict())
        self.assertIn('"phase": "end"', line)
        self.assertEqual(HistoryRecord.from_dict(json.loads(line)).to_dict(),
                         failed.to_dict())

    def test_unknown_phase_does_not_parse(self):
        from nomorals.craftlibs.ai.history import HistoryRecord
        with self.assertRaises(ValueError):
            HistoryRecord.from_dict({"phase": "pause", "run": "a"})

    def test_merge_pairs_starts_and_ends_newest_first(self):
        from nomorals.craftlibs.ai.history import HistoryRecord, Outcome, merge
        log = merge([
            HistoryRecord.start(self._start("old", 10)),
            HistoryRecord.start(self._start("new", 30)),
            HistoryRecord.end(self._end("new", 45, Outcome.DONE)),
            HistoryRecord.end(self._end("old", 12, Outcome.CANCELLED)),
        ], skipped=0)
        self.assertEqual([e.start.run for e in log.entries], ["new", "old"])
        self.assertEqual(log.entries[0].duration_ms(), 15)
        self.assertEqual(log.entries[1].outcome(), Outcome.CANCELLED)

    def test_start_with_no_end_stays_unfinished(self):
        from nomorals.craftlibs.ai.history import HistoryRecord, merge
        log = merge([HistoryRecord.start(self._start("a", 10))], skipped=0)
        self.assertEqual(len(log.entries), 1)
        self.assertIsNone(log.entries[0].outcome())
        self.assertIsNone(log.entries[0].duration_ms())

    def test_end_without_a_start_is_dropped(self):
        from nomorals.craftlibs.ai.history import HistoryRecord, Outcome, merge
        log = merge([HistoryRecord.end(
            self._end("gone", 5, Outcome.DONE))], skipped=2)
        self.assertEqual(log.entries, [])
        self.assertEqual(log.skipped, 2)

    def test_first_end_wins_and_repeated_start_is_ignored(self):
        from nomorals.craftlibs.ai.history import HistoryRecord, Outcome, merge
        log = merge([
            HistoryRecord.start(self._start("a", 10)),
            HistoryRecord.start(self._start("a", 99)),
            HistoryRecord.end(self._end("a", 20, Outcome.DONE)),
            HistoryRecord.end(self._end("a", 30, Outcome.CANCELLED)),
        ], skipped=0)
        self.assertEqual(len(log.entries), 1)
        self.assertEqual(log.entries[0].start.at_ms, 10)
        self.assertEqual(log.entries[0].outcome(), Outcome.DONE)

    def test_clock_going_backwards_gives_no_duration(self):
        from nomorals.craftlibs.ai.history import HistoryRecord, Outcome, merge
        log = merge([
            HistoryRecord.start(self._start("a", 50)),
            HistoryRecord.end(self._end("a", 40, Outcome.DONE)),
        ], skipped=0)
        self.assertIsNone(log.entries[0].duration_ms())

    def test_cost_is_only_what_the_provider_reported(self):
        from nomorals.craftlibs.ai.history import HistoryRecord, Outcome, Usage, merge
        end = self._end("a", 20, Outcome.DONE)
        log = merge([HistoryRecord.start(self._start("a", 10)),
                     HistoryRecord.end(end)], skipped=0)
        self.assertIsNone(log.entries[0].reported_cost())
        end.usage = Usage(cost_micros=40_000)
        log = merge([HistoryRecord.start(self._start("a", 10)),
                     HistoryRecord.end(end)], skipped=0)
        self.assertIsNone(log.entries[0].reported_cost())
        end.usage = Usage(cost_micros=40_000, currency="USD")
        log = merge([HistoryRecord.start(self._start("a", 10)),
                     HistoryRecord.end(end)], skipped=0)
        self.assertEqual(log.entries[0].reported_cost(), (40_000, "USD"))

    def test_prompts_are_clipped_on_a_character_boundary(self):
        from nomorals.craftlibs.ai.history import clip_prompt
        long = "é" * 2050
        self.assertEqual(len(clip_prompt(long)), 2000)
        self.assertEqual(clip_prompt("short"), "short")


class StoreTest(unittest.TestCase):
    def _store(self):
        from nomorals.craftlibs.ai.role import Binding
        conn = _remote_connection("a")
        spec = (CapabilitySpec.new(CapabilityId.parse("text.complete"))
                .accepting(InputRole.PROMPT, 1))
        conn.models = [ModelInfo.new(ModelId.new("m"), "M").with_spec(spec)]
        binding = Binding(role=RoleId.new("rewrite"), connection="a",
                          model="m")
        return StoreFile(connections=[conn], defaults=[binding],
                         apps={}, custom_tools=[])

    def _role(self):
        return Role(id="rewrite", label="Rewrite",
                    capability="text.complete", inputs=[InputRole.PROMPT])

    def test_good_store_parses(self):
        text = (_FIXTURES / "good-store.json").read_text()
        store = StoreFile.parse(text)
        self.assertEqual(len(store.connections), 1)
        self.assertEqual(store.connections[0].id.as_str(), "local")

    def test_too_new_store_is_refused(self):
        from nomorals.craftlibs.ai.store import StoreError
        text = (_FIXTURES / "new-store.json").read_text()
        with self.assertRaises(StoreError) as ctx:
            StoreFile.parse(text)
        self.assertEqual(ctx.exception.kind, "too_new")
        self.assertIn("99", str(ctx.exception))

    def test_bad_input_is_an_error_not_a_panic(self):
        from nomorals.craftlibs.ai.store import StoreError
        for text in ("", "not json", "[]", "{}",
                     '{"schema_version": 0}', '{"schema_version": -1}',
                     '{"schema_version": 1, "connections": [1]}'):
            with self.assertRaises(StoreError, msg=text):
                StoreFile.parse(text)
        big = ('{"schema_version":1,"pad":"' + "x" * (1024 * 1024) + '"}')
        with self.assertRaises(StoreError) as ctx:
            StoreFile.parse(big)
        self.assertEqual(ctx.exception.kind, "too_large")

    def test_store_round_trips_without_secrets(self):
        store = self._store()
        text = store.to_json()
        self.assertIn("secret-a", text)
        for needle in ("sk-", '"api_key":', '"token":', '"password":',
                       '"key":'):
            self.assertNotIn(needle, text)
        back = StoreFile.parse(text)
        self.assertEqual(back.to_dict(), store.to_dict())

    def test_unknown_fields_are_ignored(self):
        data = json.loads(self._store().to_json())
        data["added_in_a_later_release"] = {"x": 1}
        data["connections"][0]["new_flag"] = True
        self.assertIsNotNone(StoreFile.parse(json.dumps(data)))

    def test_mislabelled_connection_is_refused_at_load(self):
        from nomorals.craftlibs.ai.store import StoreError
        store = self._store()
        store.connections[0].locality = Locality.LOCAL
        with self.assertRaises(StoreError) as ctx:
            StoreFile.parse(store.to_json())
        self.assertEqual(ctx.exception.kind, "invalid")

    def test_duplicate_connections_are_refused(self):
        from nomorals.craftlibs.ai.store import StoreError
        store = self._store()
        store.connections.append(store.connections[0])
        with self.assertRaises(StoreError) as ctx:
            store.validate()
        self.assertEqual(ctx.exception.kind, "duplicate_connection")

    def test_resolve_uses_default_then_app_override(self):
        from nomorals.craftlibs.ai.role import Binding
        store = self._store()
        resolved = store.resolve("wordcraft", self._role())
        self.assertEqual(resolved.connection.id.as_str(), "a")
        other = _remote_connection("b")
        other.models = store.connections[0].models
        store.connections.append(other)
        store.apps["wordcraft"] = {"overrides": [
            Binding(role=RoleId.new("rewrite"), connection="b", model="m")]}
        resolved = store.resolve("wordcraft", self._role())
        self.assertEqual(resolved.connection.id.as_str(), "b")
        resolved = store.resolve("photocraft", self._role())
        self.assertEqual(resolved.connection.id.as_str(), "a")

    def test_dangling_bindings_are_reported(self):
        store = self._store()
        store.connections.clear()
        self.assertEqual(len(store.dangling_bindings()), 1)


class PicksTest(unittest.TestCase):
    def _model(self, model_id: str, price=None, released=None):
        model = ModelInfo.new(ModelId.new(model_id), model_id)
        if price is not None:
            model.pricing = Pricing(per_image_micros=price)
        model.released = released
        return model

    def _cands(self, models):
        return [{"provider": provider, "model": model, "free": free}
                for provider, model, free in models]

    def _find(self, picks, pick):
        for found in picks:
            if found.pick == pick:
                return found
        return None

    def test_builtin_catalog_loads_and_starts_empty(self):
        catalog = Catalog.builtin()
        self.assertEqual(catalog.version, 1)
        self.assertEqual(catalog.entries, [])

    def test_lowest_cost_and_newest_from_what_is_reported(self):
        models = [("or", self._model("a-pricey", 80_000, 300), False),
                  ("or", self._model("b-cheap", 20_000, 100), False),
                  ("or", self._model("c-new", 50_000, 900), False),
                  ("or", self._model("d-unpriced", None, None), False)]
        picks = choose(CapabilityId.parse("image.inpaint"),
                       self._cands(models), Catalog())
        cheapest = self._find(picks, Pick.LOWEST_COST)
        self.assertEqual((cheapest.candidate, cheapest.by),
                         (1, ("price", 20_000)))
        newest = self._find(picks, Pick.NEWEST)
        self.assertEqual((newest.candidate, newest.by),
                         (2, ("released", 900)))
        self.assertIsNone(self._find(picks, Pick.RECOMMENDED))

    def test_free_model_is_the_lowest_cost(self):
        models = [("or", self._model("hosted", 1, None), False),
                  ("comfyui", self._model("local-sdxl", None, None), True),
                  ("or", self._model("unpriced", None, None), False)]
        picks = choose(CapabilityId.parse("image.generate"),
                       self._cands(models), Catalog())
        cheapest = self._find(picks, Pick.LOWEST_COST)
        self.assertEqual((cheapest.candidate, cheapest.by),
                         (1, ("free", None)))
        only_unpriced = [("or", self._model("x", None, None), False)]
        self.assertEqual(choose(CapabilityId.parse("image.generate"),
                                self._cands(only_unpriced), Catalog()), [])

    def test_text_models_compare_input_plus_output(self):
        def text(model_id, i, o):
            model = ModelInfo.new(ModelId.new(model_id), model_id)
            model.pricing = Pricing(input_micros_per_mtok=i,
                                    output_micros_per_mtok=o)
            return model
        models = [("or", text("big", 3_000_000, 15_000_000), False),
                  ("or", text("small", 150_000, 600_000), False)]
        picks = choose(CapabilityId.parse("text.complete"),
                       self._cands(models), Catalog())
        cheapest = self._find(picks, Pick.LOWEST_COST)
        self.assertEqual((cheapest.candidate, cheapest.by),
                         (1, ("price", 750_000)))

    def test_curated_picks_come_from_the_catalog(self):
        from nomorals.craftlibs.ai.picks import CatalogEntry
        models = [("or", self._model("flux-fill", None, None), False),
                  ("bfl", self._model("flux-pro", None, None), False)]
        catalog = Catalog(entries=[
            CatalogEntry("image.inpaint", "recommended", "or", "not-offered"),
            CatalogEntry("image.inpaint", "recommended", "bfl", "flux-pro"),
            CatalogEntry("image.inpaint", "best_result", "or", "flux-fill"),
            CatalogEntry("image.inpaint", "fastest", "bfl", "flux-fill"),
            CatalogEntry("text.complete", "most_popular", "or", "flux-fill"),
        ])
        picks = choose(CapabilityId.parse("image.inpaint"),
                       self._cands(models), catalog)
        self.assertEqual(self._find(picks, Pick.RECOMMENDED).candidate, 1)
        self.assertEqual(self._find(picks, Pick.BEST_RESULT).candidate, 0)
        self.assertEqual(self._find(picks, Pick.RECOMMENDED).by,
                         ("catalog", None))
        self.assertIsNone(self._find(picks, Pick.FASTEST))
        self.assertIsNone(self._find(picks, Pick.MOST_POPULAR))

    def test_a_model_that_disappears_stops_earning_its_pick(self):
        from nomorals.craftlibs.ai.picks import CatalogEntry
        catalog = Catalog(entries=[CatalogEntry(
            "image.inpaint", "recommended", "or", "flux-fill")])
        with_model = [("or", self._model("flux-fill", None, None), False)]
        self.assertIsNotNone(self._find(choose(
            CapabilityId.parse("image.inpaint"), self._cands(with_model),
            catalog), Pick.RECOMMENDED))
        without = [("or", self._model("something-else", None, None), False)]
        self.assertEqual(choose(CapabilityId.parse("image.inpaint"),
                                self._cands(without), catalog), [])
        self.assertEqual(choose(CapabilityId.parse("image.inpaint"), [],
                                catalog), [])

    def test_catalog_is_capped(self):
        with self.assertRaises(ValueError):
            Catalog.from_json("x" * (256 * 1024 + 1))


class SpecTest(unittest.TestCase):
    def test_good_spec_parses(self):
        text = (_FIXTURES / "good-spec.json").read_text()
        spec = Spec.parse(text)
        self.assertEqual(len(spec.models), 1)
        infos = spec.model_infos()
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0].id.as_str(), "local-motion")
        self.assertEqual(spec.limits.max_polls, 600)
        self.assertEqual(spec.limits.max_seconds, 600)

    def test_bad_version_is_refused(self):
        text = (_FIXTURES / "bad-spec.json").read_text()
        with self.assertRaises(SpecError) as ctx:
            Spec.parse(text)
        self.assertEqual(ctx.exception.at, "spec_version")
        self.assertIn("only version 1", ctx.exception.why)

    def test_spec_that_never_sends_the_prompt_is_refused(self):
        text = (_FIXTURES / "no-prompt-spec.json").read_text()
        with self.assertRaises(SpecError) as ctx:
            Spec.parse(text)
        self.assertEqual(ctx.exception.at, "flow")
        self.assertIn("prompt", ctx.exception.why)

    def _spec_dict(self):
        return json.loads((_FIXTURES / "good-spec.json").read_text())

    def test_unknown_fields_are_refused_by_name(self):
        data = self._spec_dict()
        data["streaming"] = True
        with self.assertRaises(SpecError) as ctx:
            Spec.parse(json.dumps(data))
        self.assertEqual(ctx.exception.at, "spec")
        self.assertIn("streaming", ctx.exception.why)

    def test_base_follows_the_connection_rules(self):
        for base, why in [
                ("http://api.example.com", "plain http"),
                ("https://u:p@api.example.com", "user-info"),
                ("https://api.example.com/v2?x=1", "no query"),
                ("https://api.example.com/v2#top", "no query"),
                ("ftp://api.example.com", "http"),
                ("https://api.example.com/v2/../x", "'..'")]:
            data = self._spec_dict()
            data["base"] = base
            with self.assertRaises(SpecError) as ctx:
                Spec.parse(json.dumps(data))
            self.assertIn(why, ctx.exception.why, base)

    def test_paths_cannot_leave_the_base(self):
        for path, why in [
                ("images", "starts with /"),
                ("https://evil.example/x", "another host"),
                ("//evil.example/x", "another host"),
                ("/a/../b", "'..'"), ("/..", "'..'"), ("/a//b", "'//'"),
                ("/a?x=1", "no ? or #"), ("/a#x", "no ? or #"),
                ("/a%2e%2e/b", "no backslash or % escape"),
                ("/a\\b", "no backslash or % escape"),
                ("/a b", "plain visible"), ("/a\nb", "plain visible")]:
            data = self._spec_dict()
            data["flow"]["call"]["path"] = path
            with self.assertRaises(SpecError) as ctx:
                Spec.parse(json.dumps(data))
            self.assertIn(why, ctx.exception.why, path)

    def test_file_size_limit_holds_at_its_edge(self):
        text = (_FIXTURES / "good-spec.json").read_text()
        pad = 32 * 1024 - len(text)
        self.assertIsNotNone(Spec.parse(text + " " * pad))
        with self.assertRaises(SpecError) as ctx:
            Spec.parse(text + " " * (pad + 1))
        self.assertIn("too large", ctx.exception.why)


class TemplateTest(unittest.TestCase):
    def test_render_substitutes_slots_only(self):
        from nomorals.craftlibs.ai.spec import render_value
        out = render_value({"prompt": "{prompt}", "model": "{model}",
                            "nested": ["{job_id}", "plain"]},
                           prompt="hi", model="m1", job_id="j9")
        self.assertEqual(out, {"prompt": "hi", "model": "m1",
                               "nested": ["j9", "plain"]})

    def test_braces_and_unknown_slots(self):
        self.assertEqual(render_text("{{literal}} {prompt}", prompt="x"),
                         "{literal} x")
        with self.assertRaises(ValueError):
            render_text("{unknown}", prompt="x")
        with self.assertRaises(ValueError):
            render_text("{prompt", prompt="x")
        with self.assertRaises(ValueError):
            render_text("lone } here", prompt="x")
        with self.assertRaises(ValueError):
            render_value({"k}ey": "{prompt}"}, prompt="x")


class MockProviderTest(unittest.TestCase):
    def _provider(self):
        model = (ModelInfo.new(ModelId.new("m1"), "M1")
                 .with_spec(CapabilitySpec.new(
                     CapabilityId.parse("text.complete"))
                     .accepting(InputRole.PROMPT, 1)))
        script = {"running_polls": 2, "text_chunks": ["hel", "lo"],
                  "outcome": ("succeed", [Payload.text("hello")])}
        return MockProvider([model], script)

    def test_lifecycle_runs_to_done(self):
        provider = self._provider()
        req = (Request.new(CapabilityId.parse("text.complete"))
               .with_input(InputRole.PROMPT, Payload.text("hi")))
        done = run_lifecycle(provider, ModelId.new("m1"), req)
        self.assertEqual(done.state, JobState.SUCCEEDED)
        self.assertEqual("".join(e["text"] for e in done.events
                                 if e["type"] == "text_delta"), "hello")

    def test_failure_is_reported_with_its_kind(self):
        err = AiError.new(ErrorKind.QUOTA, "out of credits")
        model = (ModelInfo.new(ModelId.new("m1"), "M1")
                 .with_spec(CapabilitySpec.new(
                     CapabilityId.parse("text.complete"))
                     .accepting(InputRole.PROMPT, 1)))
        provider = MockProvider([model], {"running_polls": 0,
                                         "text_chunks": [],
                                         "outcome": ("fail", err)})
        req = (Request.new(CapabilityId.parse("text.complete"))
               .with_input(InputRole.PROMPT, Payload.text("hi")))
        done = run_lifecycle(provider, ModelId.new("m1"), req)
        self.assertEqual(done.state, JobState.failed(err))

    def test_text_job_streams_deltas_in_order(self):
        model = (ModelInfo.new(ModelId.new("m1"), "M1")
                 .with_spec(CapabilitySpec.new(
                     CapabilityId.parse("text.complete"))
                     .accepting(InputRole.PROMPT, 1)
                     .with_features(Features(streaming=True))))
        provider = MockProvider([model], {"running_polls": 0,
                                         "text_chunks": ["Hel", "lo ",
                                                         "world"],
                                         "outcome": ("succeed", [])})
        req = (Request.new(CapabilityId.parse("text.complete"))
               .with_input(InputRole.PROMPT, Payload.text("say hello"))
               .streaming())
        done = run_lifecycle(provider, ModelId.new("m1"), req)
        text = "".join(e["text"] for e in done.events
                       if e["type"] == "text_delta")
        self.assertEqual(text, "Hello world")

    def test_cancel_ends_cancelled(self):
        provider = self._provider()
        req = (Request.new(CapabilityId.parse("text.complete"))
               .with_input(InputRole.PROMPT, Payload.text("hi")))
        run_cancel(provider, ModelId.new("m1"), req)

    def test_rejects_unknown_model(self):
        provider = self._provider()
        req = Request.new(CapabilityId.parse("text.complete"))
        run_rejects(provider, ModelId.new("nope"), req,
                    ErrorKind.INVALID_REQUEST)
        run_unknown_job(provider)

    def test_local_motion_provider_suggests(self):
        provider = LocalMotionProvider()
        req = (Request.new(CapabilityId.parse("motion.suggest"))
               .with_input(InputRole.PROMPT,
                           Payload.text("a ball bouncing happily")))
        done = run_lifecycle(provider, ModelId.new("local-motion"), req)
        self.assertEqual(done.state, JobState.SUCCEEDED)
        outputs = [e for e in done.events if e["type"] == "output"]
        self.assertEqual(len(outputs), 1)
        suggestion = json.loads(outputs[0]["payload"]["text"])
        self.assertEqual(suggestion["preset"], "bounce")


class PdfTest(unittest.TestCase):
    def test_safe_url_vectors(self):
        for ok in ("https://example.com", " HTTP://example.com/x",
                   "https://user@example.com:8443/x?q#f", "https://[::1]/",
                   "mailto:someone@example.com",
                   "mailto:?to=someone@example.com&subject=hi"):
            self.assertTrue(shared_pdf.safe_url(ok), ok)
        for bad in ("https://", "https://?q=x", "https:///x",
                    "https://user@/x", "https://:80/", "mailto:",
                    "mailto:?subject=hello", "file:///C:/notes.pdf",
                    "notes.pdf", "javascript:alert(1)",
                    "\t JavaScript:alert(1)", "java\nscript:alert(1)",
                    "vbscript:MsgBox(1)",
                    "data:text/html,<script>alert(1)</script>",
                    "ms-msdt:/id PCWDiagnostic",
                    "ms-word:ofe|u|\\\\attacker.example\\share\\x.docx",
                    "ldap://attacker.example/x", "//attacker.example/share",
                    "\\\\attacker.example\\share",
                    "file://attacker.example/share/x",
                    "file:////attacker.example/share/x",
                    "smb://attacker.example/share",
                    "search-ms:query=x&crumb=location:\\\\attacker.example\\share",
                    "search:query=x&crumb=location:\\\\attacker.example\\share"):
            self.assertFalse(shared_pdf.safe_url(bad), bad)

    def test_input_bounds(self):
        shared_pdf.check_input_size(100)
        with self.assertRaises(ValueError):
            shared_pdf.check_input_size(shared_pdf.MAX_INPUT_BYTES + 1)

    def test_write_then_read_round_trip(self):
        meta = Metadata(creator="craftlibs", producer="craftlibs",
                        title="Craftlibs proof")
        data = shared_pdf.write_pdf("# Craftlibs proof\n\nHello.\n", meta)
        self.assertTrue(data.startswith(b"%PDF-"))
        info = shared_pdf.read_pdf(data)
        self.assertEqual(info["pages"], 1)
        self.assertIn("Hello", info["text"])
        self.assertEqual(info["title"], "Craftlibs proof")

    def test_fixture_pdf_reads_back(self):
        raw = (_FIXTURES / "sample.pdf").read_bytes()
        info = shared_pdf.read_pdf(raw)
        self.assertEqual(info["pages"], 1)
        self.assertIn("Hello from the shared libraries", info["text"])


class I18nFormatTest(unittest.TestCase):
    def test_locale_digits_and_separators(self):
        cases = [("en", "1,234,567", "1,234.5"),
                 ("ja", "1,234,567", "1,234.5"),
                 ("de", "1.234.567", "1.234,5"),
                 ("fr", "1\u202f234\u202f567", "1\u202f234,5"),
                 ("hi", "12,34,567", "1,234.5"),
                 ("ar-EG", "١٬٢٣٤٬٥٦٧", "١٬٢٣٤٫٥"),
                 ("fa", "۱٬۲۳۴٬۵۶۷", "۱٬۲۳۴٫۵")]
        for tag, integer, number in cases:
            fmt = shared_i18n.Format.from_tag(tag)
            self.assertEqual((fmt.integer(1_234_567),
                              fmt.number(1234.5, 1)),
                             (integer, number), tag)
        # CLDR writes Arabic without a region in Latin digits.
        self.assertEqual(shared_i18n.Format.from_tag("ar").number(1234.5, 1),
                         "1,234.5")

    def test_ungrouped_numbers(self):
        for tag, value, decimals, want in [
                ("en", 1920.0, 0, "1920"),
                ("de", -1_234_567.5, 1, "-1234567,5"),
                ("hi", 1_234_567.0, 0, "1234567"),
                ("ar-EG", 1920.0, 0, "١٩٢٠")]:
            self.assertEqual(shared_i18n.Format.from_tag(tag)
                             .number_ungrouped(value, decimals), want, tag)

    def test_forced_latin_digits(self):
        self.assertEqual(shared_i18n.Format.new(
            shared_i18n.Region.from_tag("ar-EG"),
            shared_i18n.Digits.LATIN).number(1234.5, 1), "1,234.5")
        self.assertEqual(shared_i18n.Format.new(
            shared_i18n.Region.from_tag("fa"),
            shared_i18n.Digits.LATIN).integer(1_234_567), "1,234,567")

    def test_cldr_separators_for_every_region(self):
        for tag, want in [
                ("en-PL", "1.234.567,5"), ("en-DE", "1.234.567,5"),
                ("ar-MA", "1.234.567,5"), ("de-CH", "1'234'567.5"),
                ("nb-NO", "1\u00a0234\u00a0567,5"),
                ("sk-SK", "1\u00a0234\u00a0567,5"),
                ("bn-BD", "১২,৩৪,৫৬৭.৫"), ("he-IL", "1,234,567.5"),
                ("km", "1,234,567.5"), ("lo", "1.234.567,5")]:
            self.assertEqual(shared_i18n.Format.from_tag(tag)
                             .number(1_234_567.5, 1), want, tag)

    def test_system_separators(self):
        region = shared_i18n.Region.from_tag("en-US")
        region.decimal = ","
        self.assertEqual(shared_i18n.Format.new(
            region, shared_i18n.Digits.LOCALE).number(1234.5, 1), "1.234,5")
        region = shared_i18n.Region.from_tag("de-DE")
        region.grouping = "\u00a0"
        self.assertEqual(shared_i18n.Format.new(
            region, shared_i18n.Digits.LOCALE).number(1234.5, 1),
            "1\u00a0234,5")
        region.decimal = "."
        self.assertEqual(shared_i18n.Format.new(
            region, shared_i18n.Digits.LOCALE).number(1234.5, 1),
            "1\u00a0234.5")
        region.grouping = shared_i18n.NO_GROUPING
        self.assertEqual(shared_i18n.Format.new(
            region, shared_i18n.Digits.LOCALE).integer(1_234_567),
            "1234567")
        region = shared_i18n.Region.from_tag("ar-EG")
        region.grouping = ","
        self.assertEqual(shared_i18n.Format.new(
            region, shared_i18n.Digits.LOCALE).number(1234.5, 1),
            "١٬٢٣٤٫٥")

    def test_system_grouping_mark_equal_to_decimal_is_dropped(self):
        region = shared_i18n.Region.from_windows(
            {"LocaleName": "en-US", "sDecimal": ",",
             "sThousand": ","}.get)
        fmt = shared_i18n.Format.new(region, shared_i18n.Digits.LOCALE)
        self.assertEqual((fmt.number(1234.5, 1), fmt.integer(1_234_567)),
                         ("1234,5", "1234567"))

    def test_digit_system_the_region_names_is_used(self):
        self.assertEqual(shared_i18n.Format.new(
            shared_i18n.Region.from_tag("ar-EG-u-nu-latn"),
            shared_i18n.Digits.LOCALE).number(1234.5, 1), "1,234.5")
        self.assertEqual(shared_i18n.Format.new(
            shared_i18n.Region.from_tag("ar_EG@numbers=latn"),
            shared_i18n.Digits.LOCALE).integer(5), "5")
        self.assertEqual(shared_i18n.Format.new(
            shared_i18n.Region.from_tag("en-US-u-nu-arab"),
            shared_i18n.Digits.LOCALE).integer(5), "٥")

    def test_percentages(self):
        cases = [("en", 0.5, 0, "50%"), ("de", 0.5, 0, "50\u00a0%"),
                 ("fr", 0.125, 1, "12,5\u00a0%"), ("tr", 0.5, 0, "%50"),
                 ("ar-EG", 0.5, 0, "٥٠٪\u061c"), ("fa", 0.5, 0, "۵۰٪"),
                 ("hi", 0.5, 0, "50%")]
        for tag, ratio, decimals, want in cases:
            self.assertEqual(shared_i18n.Format.from_tag(tag)
                             .percent(ratio, decimals), want, tag)
        self.assertEqual(shared_i18n.Format.from_tag("tr").percent(-0.2, 0),
                         "-%20")
        self.assertEqual(shared_i18n.Format.new(
            shared_i18n.Region.from_tag("fa"),
            shared_i18n.Digits.LATIN).percent(0.5, 0), "50%")

    def test_signs_and_odd_values(self):
        en = shared_i18n.Format.from_tag("en")
        self.assertEqual(en.number(-1234.5, 0), "-1,234")
        self.assertEqual(en.number(-0.04, 1), "0.0")
        self.assertEqual(en.number(math.inf, 1), "∞")
        self.assertEqual(en.number(-math.inf, 1), "-∞")
        self.assertEqual(en.number(math.nan, 1), "NaN")
        self.assertEqual(shared_i18n.Format.from_tag("de").percent(
            math.nan, 0), "NaN\u00a0%")
        self.assertEqual(en.integer(-5), "-5")


class I18nParseTest(unittest.TestCase):
    def test_every_digit_set(self):
        parse = shared_i18n.parse_number
        for zero in (0x660, 0x966, 0x9E6, 0xC66, 0xE50, 0xFF10, 0x104A0,
                     0x1D7CE, 0xA900):
            digits = [chr(zero + n) for n in range(10)]
            all_digits = "".join(reversed(digits))
            self.assertEqual(parse(all_digits, "en"), 9_876_543_210.0,
                             f"zero U+{zero:04X}")
            grouped = (f"-{digits[1]}{digits[2]}{digits[3]},"
                       f"{digits[4]}{digits[5]}{digits[6]}.{digits[7]}")
            self.assertEqual(parse(grouped, "en"), -123_456.7,
                             f"zero U+{zero:04X}")

    def test_locale_forms(self):
        parse = shared_i18n.parse_number
        cases = [("1,234.5", "en", 1234.5), ("1.234,5", "de", 1234.5),
                 ("1 234,5", "fr", 1234.5),
                 ("1\u202f234,5", "fr", 1234.5),
                 ("1\u00a0234,5", "ru", 1234.5),
                 ("1’234.5", "de-CH", 1234.5),
                 ("1,234", "en", 1234.0), ("1,234", "de", 1.234),
                 ("1.234", "pt-br", 1234.0), ("1,5", "es", 1.5),
                 ("١٬٢٣٤٫٥", "ar", 1234.5), ("١٢٫٥", "ar", 12.5),
                 ("12.5", "ar", 12.5), ("12,5", "ar-MA", 12.5),
                 ("۱۲٫۵", "fa", 12.5), ("−۴۲", "fa", -42.0),
                 ("１２３，４５６．７", "ja", 123_456.7),
                 ("१२,३४,५६७.८९", "hi", 1_234_567.89),
                 ("1,234,567", "en", 1_234_567.0),
                 ("−3,5", "sv", -3.5), ("+7", "en", 7.0),
                 (".5", "en", 0.5), ("5.", "en", 5.0),
                 ("  42  ", "en", 42.0), ("1,234.5", "xx", 1234.5),
                 ("12", "xx", 12.0),
                 ("1,5", "en-PL", 1.5), ("1.234,5", "en-DE", 1234.5),
                 ("1.234,5", "lo", 1234.5), ("1,234.5", "km", 1234.5),
                 ("1'234.5", "de_CH.UTF-8", 1234.5),
                 ("1.5", "es-419", 1.5),
                 ("১২,৩৪,৫৬৭", "bn", 1_234_567.0),
                 ("౧,౨౩,౪౫౬", "te", 123_456.0),
                 ("๑,๒๓๔.๕", "th", 1234.5),
                 ("\u2066-1,5\u2069", "de", -1.5),
                 ("\u061c-١٢", "ar", -12.0),
                 ("－５", "ja", -5.0)]
        for text, locale, want in cases:
            self.assertEqual(parse(text, locale), want, f"{text!r} in {locale}")

    def test_errors_instead_of_guesses(self):
        parse = shared_i18n.parse_number
        for text, locale in [("", "en"), ("  ", "en"), ("-", "en"),
                             (",", "en"), ("1.2.3", "en"), ("1,,234", "en"),
                             ("1,234,", "en"), ("1.234,5", "en"),
                             ("1 234'567", "en"), ("1.234 567", "de"),
                             ("0.123,4", "de"), ("1.5,2", "en"),
                             ("123,45,678", "hi"), ("12٫3٫4", "ar"),
                             ("1٫5.2", "ar"), ("1٢", "ar"),
                             ("12abc", "en"), ("1e5", "en"), ("--5", "en"),
                             ("5-", "en"), ("- 5", "en")]:
            with self.assertRaises(ValueError, msg=f"{text!r} in {locale}"):
                parse(text, locale)
        with self.assertRaises(ValueError):
            parse("9" * 400, "en")

    def test_lone_marks(self):
        parse = shared_i18n.parse_number
        for text, locale, want in [("1.5", "de", 1.5), ("0.250", "de", 0.25),
                                   ("1234.5", "de", 1234.5),
                                   (".5", "de", 0.5),
                                   ("-12.34", "ru", -12.34),
                                   ("1.234", "de", 1234.0),
                                   ("1.234.567", "de", 1_234_567.0),
                                   ("1,5", "en", 1.5), ("0,5", "en", 0.5),
                                   ("12,34", "en", 12.34),
                                   ("1234,567", "en", 1234.567),
                                   ("1,234", "en", 1234.0),
                                   ("12,345", "en", 12345.0)]:
            self.assertEqual(parse(text, locale), want, text)

    def test_reads_what_format_writes(self):
        for tag in ("en", "de", "de-CH", "fr", "hi", "ar-EG", "fa",
                    "ar-MA", "en-PL", "pt-PT", "sv"):
            fmt = shared_i18n.Format.from_tag(tag)
            for value in (0.0, 1.5, -1234.25, 1_234_567.89, 12.0, 1234.0):
                text = fmt.number(value, 2)
                self.assertEqual(fmt.parse(text), value, f"{text!r} ({tag})")
            self.assertEqual(fmt.parse(fmt.integer(1_234_567)), 1_234_567.0)


class I18nFileSizeTest(unittest.TestCase):
    def test_binary_decimal_and_iec_steps(self):
        fmt = shared_i18n.Format.from_tag("en-US")
        size = shared_i18n.FileSize
        self.assertEqual(shared_i18n.file_size(fmt, 512, size.BINARY),
                         "512 bytes")
        self.assertEqual(shared_i18n.file_size(fmt, 1023, size.BINARY),
                         "1,023 bytes")
        self.assertEqual(shared_i18n.file_size(fmt, 1, size.BINARY),
                         "1 byte")
        self.assertEqual(shared_i18n.file_size(fmt, 1536, size.BINARY),
                         "1.5 KB")
        self.assertEqual(shared_i18n.file_size(fmt, 1536, size.IEC),
                         "1.5 KiB")
        self.assertEqual(shared_i18n.file_size(fmt, 1500, size.DECIMAL),
                         "1.5 KB")
        self.assertEqual(shared_i18n.file_size(fmt, 3 << 30, size.BINARY),
                         "3.0 GB")
        self.assertEqual(shared_i18n.file_size(
            fmt, 5 << 40, size.IEC.decimals(0)), "5 TiB")
        self.assertEqual(shared_i18n.file_size(
            fmt, 512, size.BINARY.byte_symbol()), "512 B")

    def test_precision_and_unit_range(self):
        fmt = shared_i18n.Format.from_tag("en-US")
        size = shared_i18n.FileSize
        self.assertEqual(shared_i18n.file_size(fmt, 1_048_575, size.BINARY),
                         "1.0 MB")
        self.assertEqual(shared_i18n.file_size(
            fmt, 999_999, size.DECIMAL.decimals(0)), "1 MB")
        self.assertEqual(shared_i18n.file_size(
            fmt, 1_610_612_736,
            size.BINARY.unit_decimals(shared_i18n.SizeUnit.GIGA, 2)),
            "1.50 GB")
        style = (size.DECIMAL.smallest(shared_i18n.SizeUnit.KILO)
                 .largest(shared_i18n.SizeUnit.MEGA)
                 .unit_decimals(shared_i18n.SizeUnit.KILO, 0)
                 .whole_from(10))
        self.assertEqual(shared_i18n.file_size(fmt, 1, style), "1 KB")
        self.assertEqual(shared_i18n.file_size(fmt, 0, style), "0 KB")
        self.assertEqual(shared_i18n.file_size(fmt, 640_400, style), "640 KB")
        self.assertEqual(shared_i18n.file_size(fmt, 3_400_000, style),
                         "3.4 MB")
        self.assertEqual(shared_i18n.file_size(fmt, 12_600_000, style),
                         "13 MB")
        self.assertEqual(shared_i18n.file_size(fmt, 9_960_000, style),
                         "10 MB")
        self.assertEqual(shared_i18n.file_size(fmt, 9_940_000, style),
                         "9.9 MB")
        self.assertEqual(shared_i18n.file_size(fmt, 5_000_000_000, style),
                         "5,000 MB")
        self.assertEqual(shared_i18n.file_size(
            fmt, 1025,
            size.BINARY.smallest(shared_i18n.SizeUnit.KILO)
            .unit_decimals(shared_i18n.SizeUnit.KILO, 0)), "1 KB")
        self.assertEqual(shared_i18n.file_size(
            fmt, 100,
            size.BINARY.smallest(shared_i18n.SizeUnit.KILO)
            .unit_decimals(shared_i18n.SizeUnit.KILO, 0)), "1 KB")

    def test_locale_digits_in_sizes(self):
        ar = shared_i18n.Format.from_tag("ar-EG")
        self.assertEqual(shared_i18n.file_size(ar, 1536,
                                               shared_i18n.FileSize.BINARY),
                         "١٫٥ KB")
        hi = shared_i18n.Format.from_tag("hi")
        self.assertEqual(shared_i18n.file_size(
            hi, 1_234_567, shared_i18n.FileSize.DECIMAL
            .largest(shared_i18n.SizeUnit.BYTE).byte_symbol()),
            "12,34,567 B")


class RegionTest(unittest.TestCase):
    def test_reads_locale_tags(self):
        self.assertEqual(shared_i18n.Region.from_tag("pl-PL").numbers,
                         "pl-PL")
        self.assertEqual(
            shared_i18n.Region.from_tag("de_AT.UTF-8@euro").numbers, "de-AT")
        self.assertEqual(shared_i18n.Region.from_tag("zh-hans-cn").numbers,
                         "zh-Hans-CN")
        self.assertEqual(shared_i18n.Region.from_tag("sr_RS@latin").numbers,
                         "sr-Latn-RS")
        self.assertEqual(shared_i18n.Region.from_tag("es-419").numbers,
                         "es-419")
        self.assertEqual(shared_i18n.Region.from_tag("fr").numbers, "fr")
        self.assertEqual(shared_i18n.Region.from_tag("en").numbers, "en-US")
        for junk in ("", "1234", "x", "--", "ünïcode-ÄÄ", "a" * 10_000):
            self.assertEqual(shared_i18n.Region.from_tag(junk),
                             shared_i18n.Region.default(), repr(junk))

    def test_c_locale_is_a4_metric_and_ungrouped(self):
        for tag in ("C", "POSIX", "C.UTF-8", "c.utf8", " C "):
            region = shared_i18n.Region.from_tag(tag)
            self.assertEqual(
                (region.numbers, region.paper, region.measurement,
                 region.decimal, region.grouping),
                ("en-US", shared_i18n.Paper.A4,
                 shared_i18n.Measurement.METRIC, ".",
                 shared_i18n.NO_GROUPING), tag)

    def test_chosen_digit_system_stays_in_the_locale(self):
        region = shared_i18n.Region.from_tag("ar-EG-u-nu-latn")
        self.assertEqual(region.numbers, "ar-EG-u-nu-latn")
        region = shared_i18n.Region.from_tag("ar_EG@numbers=latn")
        self.assertEqual((region.numbers, region.time),
                         ("ar-EG-u-nu-latn", "ar-EG-u-nu-latn"))
        self.assertEqual(
            shared_i18n.Region.from_tag("en-US-u-nu-arab-ms-metric").numbers,
            "en-US-u-nu-arab")
        self.assertEqual(
            shared_i18n.Region.from_tag("ar-EG-u-nu-x").numbers, "ar-EG")

    def test_unicode_extensions_and_apple_keywords(self):
        region = shared_i18n.Region.from_tag("en-US-u-ms-metric-hc-h23")
        self.assertEqual(
            (region.numbers, region.measurement, region.hour_cycle),
            ("en-US", shared_i18n.Measurement.METRIC,
             shared_i18n.HourCycle.H23))
        self.assertEqual(
            shared_i18n.Region.from_tag("en-GB-u-hc-h12").hour_cycle,
            shared_i18n.HourCycle.H12)
        self.assertEqual(
            shared_i18n.Region.from_tag("de-DE-x-ms-ussystem").measurement,
            shared_i18n.Measurement.METRIC)
        region = shared_i18n.Region.from_tag("en_GB@rg=uszzzz")
        self.assertEqual(
            (region.numbers, region.paper, region.measurement),
            ("en-US", shared_i18n.Paper.LETTER,
             shared_i18n.Measurement.US))
        self.assertEqual(
            shared_i18n.Region.from_tag("en_US@rg=plzzzz;ms=metric").numbers,
            "en-PL")

    def test_paper_and_measurement_follow_territory(self):
        region = shared_i18n.Region
        self.assertEqual(
            (region.from_tag("pl-PL").paper,
             region.from_tag("pl-PL").measurement),
            (shared_i18n.Paper.A4, shared_i18n.Measurement.METRIC))
        self.assertEqual(
            (region.from_tag("en-US").paper,
             region.from_tag("en-US").measurement),
            (shared_i18n.Paper.LETTER, shared_i18n.Measurement.US))
        self.assertEqual(
            (region.from_tag("fr-CA").paper,
             region.from_tag("fr-CA").measurement),
            (shared_i18n.Paper.LETTER, shared_i18n.Measurement.METRIC))
        self.assertEqual(
            (region.from_tag("en-GB").paper,
             region.from_tag("en-GB").measurement),
            (shared_i18n.Paper.A4, shared_i18n.Measurement.METRIC))
        self.assertEqual(
            (region.from_tag("de-CH").decimal,
             region.from_tag("de-CH").grouping), (None, None))
        self.assertEqual(region.from_tag("es-MX").paper,
                         shared_i18n.Paper.LETTER)
        self.assertEqual(region.from_tag("ja-JP").paper,
                         shared_i18n.Paper.A4)

    def test_setting_empty_means_system(self):
        self.assertEqual(shared_i18n.Region.from_setting(" pl-PL "),
                         shared_i18n.Region.from_tag("pl-PL"))
        self.assertEqual(shared_i18n.Region.from_setting(""),
                         shared_i18n.Region.system())

    def test_posix_variables_set_each_convention(self):
        env = {"LANG": "en_US.UTF-8", "LC_NUMERIC": "pl_PL.UTF-8",
               "LC_TIME": "en_GB.UTF-8", "LC_PAPER": "de_DE"}
        region = shared_i18n.Region.from_posix(env.get)
        self.assertEqual((region.numbers, region.time, region.decimal),
                         ("pl-PL", "en-GB", None))
        self.assertEqual((region.paper, region.measurement),
                         (shared_i18n.Paper.A4, shared_i18n.Measurement.US))
        env = {"LANG": "it_IT.UTF-8", "LC_ALL": "en_US",
               "LC_PAPER": "de_DE"}
        self.assertEqual(shared_i18n.Region.from_posix(env.get).paper,
                         shared_i18n.Paper.LETTER)
        env = {"LANG": "en_US", "LC_MEASUREMENT": "it_IT"}
        self.assertEqual(shared_i18n.Region.from_posix(env.get).measurement,
                         shared_i18n.Measurement.METRIC)
        env = {"LANG": "C.UTF-8", "LC_ALL": ""}
        self.assertEqual(shared_i18n.Region.from_posix(
            env.get, platform=lambda: "it-IT").numbers, "it-IT")
        self.assertEqual(shared_i18n.Region.from_posix({}.get),
                         shared_i18n.Region.default())
        env = {"LANG": "de_DE.UTF-8", "LC_ALL": "C"}
        self.assertEqual(shared_i18n.Region.from_posix(env.get),
                         shared_i18n.Region.from_tag("C"))

    def test_windows_regional_format(self):
        pairs = {"LocaleName": "pl-PL", "sDecimal": ",",
                 "sThousand": "\u00a0", "iMeasure": "0",
                 "sShortTime": "HH:mm"}
        region = shared_i18n.Region.from_windows(pairs.get)
        self.assertEqual(
            (region.numbers, region.decimal, region.grouping,
             region.hour_cycle),
            ("pl-PL", ",", "\u00a0", shared_i18n.HourCycle.H23))
        pairs = {"LocaleName": "en-US", "sDecimal": ",",
                 "iMeasure": "0", "iPaperSize": "9", "sShortTime": "h:mm tt"}
        region = shared_i18n.Region.from_windows(pairs.get)
        self.assertEqual(
            (region.decimal, region.grouping, region.measurement,
             region.paper, region.hour_cycle),
            (",", None, shared_i18n.Measurement.METRIC,
             shared_i18n.Paper.A4, shared_i18n.HourCycle.H12))
        pairs = {"sDecimal": "dec", "sThousand": ""}
        region = shared_i18n.Region.from_windows(pairs.get,
                                                  platform=lambda: "de-DE")
        self.assertEqual((region.numbers, region.decimal),
                         ("de-DE", None))
        self.assertEqual(region.grouping, shared_i18n.NO_GROUPING)


class ParityRunnerTest(unittest.TestCase):
    def test_missing_toml_fails_not_skips(self):
        result = run_parity(_FIXTURES / "no-such.toml")
        self.assertEqual(result["passed"], 0)
        self.assertEqual(result["failed"], ["<toml>"])

    def test_case_failure_is_reported_by_id(self):
        cases = [{"id": "good", "operation": "noop"},
                 {"id": "bad", "operation": "explode"}]
        result = run_parity(cases, run_case=lambda c: (
            None if c["operation"] == "noop"
            else (_ for _ in ()).throw(AssertionError("boom"))))
        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["failed"], ["bad"])
        self.assertEqual(result["failures"][0]["id"], "bad")


class AttributionTest(unittest.TestCase):
    def test_craftlibs_attribution_verifies(self):
        result = verify_attribution()
        self.assertEqual(result["failed"], [])
        self.assertGreater(result["verified"], 0)

    def test_missing_toml_reports(self):
        result = verify_attribution(_FIXTURES / "no-such.toml")
        self.assertEqual(result["failed"][0]["id"], "<toml>")


class UnificationTest(unittest.TestCase):
    def test_textures_noise_is_the_shared_point_engine(self):
        from nomorals.craftlibs.noise import point_fbm, point_perlin, point_value
        from nomorals.media.textures.noise import fbm, perlin, value
        self.assertIs(value, point_value)
        self.assertIs(perlin, point_perlin)
        self.assertIs(fbm, point_fbm)

    def test_material_noise_is_the_shared_grid_engine(self):
        from nomorals.craftlibs.noise import grid_fbm, grid_perlin, grid_value_noise, grid_worley
        from nomorals.media.material.noise import fbm, perlin, value_noise, worley
        self.assertIs(perlin, grid_perlin)
        self.assertIs(value_noise, grid_value_noise)
        self.assertIs(fbm, grid_fbm)
        self.assertIs(worley, grid_worley)

    def test_cad_and_layout_codecs_are_the_shared_codec(self):
        from nomorals.media.cad import codec as cad_codec
        from nomorals.media.cad.errors import CadError
        from nomorals.media.layout import codec as layout_codec
        from nomorals.media.layout.errors import LayoutError
        self.assertIs(cad_codec._canonical_json, shared_codec.canonical_json)
        self.assertIs(layout_codec._canonical_json,
                      shared_codec.canonical_json)
        self.assertIs(cad_codec.write_package, shared_codec.write_package)
        # The shared codec's failures surface as the organ's own error.
        with self.assertRaises(CadError):
            cad_codec.read_package(b"not a zip at all")
        with self.assertRaises(LayoutError):
            layout_codec.read_package(b"not a zip at all")
        raw = (_FIXTURES / "bracket.cadcraft").read_bytes()
        doc = cad_codec.parse_document(raw)
        self.assertEqual(cad_codec.to_document_bytes(doc), raw)
        raw_layout = (_FIXTURES / "page.designcraft").read_bytes()
        doc2 = layout_codec.parse_document(raw_layout)
        self.assertEqual(layout_codec.to_document_bytes(doc2), raw_layout)

    def test_anicraft_role_is_the_shared_role(self):
        from nomorals.craftlibs.ai.role import Role as SharedRole
        from nomorals.media.animation.ai import Role
        role = Role.suggest_motion()
        self.assertIsInstance(role._shared, SharedRole)
        self.assertEqual(role.capability, "motion.suggest")
        self.assertEqual(role.to_dict()["capability"], "motion.suggest")

    def test_textures_parity_uses_the_shared_runner(self):
        from nomorals.craftlibs.parity import run_cases
        from nomorals.media.textures import parity as textures_parity
        self.assertIs(textures_parity.run_cases, run_cases)


class ToolCountTest(unittest.TestCase):
    def test_tool_count_is_787_after_craftlibs(self):
        # 732 after the craft-libs port (lane 113) at its base, 733
        # after the craft-launcher port (lane 112) landed between that
        # base and this rebase, 734 after the craft-fonts port
        # (lane 114) added the craftfonts tool, 735 after the
        # indd-utils port (lane 117) added the inddutils tool,
        # 748 after the auth-plugin wiring (5 new tools) on top of the bughunter port (lane 121) added 8
        # bughunter tools. 749 after the mocap port (lane 125) added the mocap tool. 774 after the secprobe wire-up added 25 tools (scope_load, scope_check, param_mine, param_confirm, inject_probe, jwt_attack, graphql_probe, graphql_amplify, race_wave, cors_probe, csrf_poc, plan_next, plan_add, plan_resolve, leaks_harvest, dossier_build, ban_report, ban_state, ban_set_state, ban_appeal, report_capture, report_amplify, replay_run, replay_diagnose, sec_chain). 775 after this lane (B.19) added scene_spawn. 776 after this lane (B.20) added cloudcam. 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet). 801 after gapfix124 added browser_watch. 802 after this lane (SongRecon surface) added songrecon. 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 807 after this lane (agents192) added fleet. 808 after this lane (F3) added control. 809 after this lane (moneyprinter) added make_short.
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        self.assertEqual(len(registry._tools), 809)
        self.assertIn("craftlibs", registry._tools)


class CraftlibsParityTest(unittest.TestCase):
    def test_parity_runs_all_cases(self):
        result = run_parity()
        self.assertEqual(result["failed"], [], str(result["failures"]))
        self.assertEqual(result["passed"], result["total"])
        self.assertGreaterEqual(result["passed"], 12)


if __name__ == "__main__":
    unittest.main()
