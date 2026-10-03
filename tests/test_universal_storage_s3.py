"""Tests for the S3-compatible blob backend.

A fake in-memory S3 endpoint (``FakeS3Handler``) exercises the full HTTP
round trip — PUT/GET/HEAD/DELETE/ListObjectsV2 — including an *independent*
SigV4 verification written straight from the AWS spec (no shared code with
the client), plus the AWS published test-suite vector for the signer itself.
No network, no MinIO, no credentials leave the machine.
"""

from __future__ import annotations

import gzip
import hashlib
import hmac
import io
import os
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from nomorals.core.errors import ConfigError, NotFound, StorageError
from nomorals.storage.s3blob import S3BlobStore, S3Config, open_blob_store, sign_request

TEST_KEY = "TESTACCESSKEY"
TEST_SECRET = "TESTSECRETKEY0000000000000000000000000000"
TEST_REGION = "us-east-1"
BUCKET = "test-bucket"


# ── independent SigV4 verifier (spec-derived, shares no code with client) ──

def _independent_verify(handler: BaseHTTPRequestHandler, secret: str) -> bool:
    """Recompute the SigV4 signature from the raw received request."""
    authz = handler.headers.get("Authorization", "")
    if not authz.startswith("AWS4-HMAC-SHA256 "):
        return False
    parts: dict[str, str] = {}
    for item in authz[len("AWS4-HMAC-SHA256 "):].split(", "):
        key, _, value = item.partition("=")
        parts[key] = value
    try:
        cred, signed_headers, signature = parts["Credential"], parts["SignedHeaders"], parts["Signature"]
        _, _, datestamp_region, _, _ = cred.split("/")
    except (KeyError, ValueError):
        return False
    datestamp = cred.split("/")[1]
    region = cred.split("/")[2]
    if region != TEST_REGION:
        return False

    raw_path = handler.path
    path, _, qs = raw_path.partition("?")
    query: dict[str, str] = {}
    if qs:
        for pair in qs.split("&"):
            key, _, value = pair.partition("=")
            query[urllib.parse.unquote(key)] = urllib.parse.unquote(value)

    def quote(value: str) -> str:
        return urllib.parse.quote(value, safe="~")

    canonical_uri = "/".join(quote(seg) for seg in path.split("/"))
    canonical_qs = "&".join(f"{quote(k)}={quote(v)}" for k, v in sorted(query.items()))
    names = signed_headers.split(";")
    header_lines = ""
    for name in sorted(names):
        value = handler.headers.get(name)
        if value is None:
            return False
        header_lines += f"{name}:{value.strip()}\n"
    length = int(handler.headers.get("Content-Length", "0") or 0)
    body = handler.rfile.read(length) if handler.command in ("PUT", "POST") else b""
    handler._saved_body = body  # noqa: SLF001 - hand the body to the handler
    payload_hash = handler.headers.get("x-amz-content-sha256") or hashlib.sha256(body).hexdigest()
    # The payload hash must match the actual body (S3 checks this too).
    if payload_hash != hashlib.sha256(body).hexdigest():
        return False
    canonical = (
        f"{handler.command}\n{canonical_uri}\n{canonical_qs}\n"
        f"{header_lines}\n{signed_headers}\n{payload_hash}"
    )
    amz_date = handler.headers.get("x-amz-date", "")
    scope = f"{datestamp}/{region}/s3/aws4_request"
    string_to_sign = (
        f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n"
        + hashlib.sha256(canonical.encode()).hexdigest()
    )
    key = ("AWS4" + secret).encode()
    for value in (datestamp, region, "s3", "aws4_request"):
        key = hmac.new(key, value.encode(), hashlib.sha256).digest()
    expected = hmac.new(key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


# ── fake S3 endpoint ─────────────────────────────────────────────────────────

class FakeS3Handler(BaseHTTPRequestHandler):
    server_version = "FakeS3/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: object) -> None:  # silence test output
        pass

    def _bucket_key(self) -> tuple[str, str]:
        path = self.path.split("?", 1)[0]
        parts = path.lstrip("/").split("/", 1)
        bucket = parts[0]
        key = urllib.parse.unquote(parts[1]) if len(parts) > 1 else ""
        return bucket, key

    def _checked(self) -> bool:
        if not _independent_verify(self, TEST_SECRET):
            self._xml(403, "SignatureDoesNotMatch", "The request signature we calculated does not match.")
            return False
        return True

    def _xml(self, code: int, error_code: str, message: str) -> None:
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<Error><Code>{error_code}</Code><Message>{message}</Message></Error>"
        ).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, code: int, data: bytes, meta: dict[str, str]) -> None:
        self.send_response(code)
        for key, value in meta.items():
            self.send_header(f"x-amz-meta-{key}", value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _meta_from_headers(self) -> dict[str, str]:
        return {
            key[11:]: value
            for key, value in self.headers.items()
            if key.lower().startswith("x-amz-meta-")
        }

    # PUT: upload, or metadata-rewrite copy when x-amz-copy-source is set.
    def do_PUT(self) -> None:
        if not self._checked():
            return
        bucket, key = self._bucket_key()
        if bucket != BUCKET:
            self._xml(404, "NoSuchBucket", "The specified bucket does not exist.")
            return
        body: bytes = getattr(self, "_saved_body", b"")
        copy_source = self.headers.get("x-amz-copy-source")
        if copy_source:
            src_key = urllib.parse.unquote(copy_source).lstrip("/").split("/", 1)[1]
            if src_key not in self.server.store:  # type: ignore[attr-defined]
                self._xml(404, "NoSuchKey", "The specified key does not exist.")
                return
            src_body, src_meta = self.server.store[src_key]  # type: ignore[attr-defined]
            if self.headers.get("x-amz-metadata-directive") == "REPLACE":
                meta = self._meta_from_headers()
            else:
                meta = dict(src_meta)
            self.server.store[key] = (src_body, meta)  # type: ignore[attr-defined]
            result = (
                b'<?xml version="1.0" encoding="UTF-8"?>'
                b'<CopyObjectResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                b"<LastModified>2025-01-01T00:00:00.000Z</LastModified>"
                b"</CopyObjectResult>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)
            return
        self.server.store[key] = (body, self._meta_from_headers())  # type: ignore[attr-defined]
        etag = hashlib.md5(body).hexdigest()
        self.send_response(200)
        self.send_header("ETag", f'"{etag}"')
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if not self._checked():
            return
        bucket, key = self._bucket_key()
        if bucket != BUCKET:
            self._xml(404, "NoSuchBucket", "The specified bucket does not exist.")
            return
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if query.get("list-type") == ["2"]:
            self._list(query)
            return
        store = self.server.store  # type: ignore[attr-defined]
        if key not in store:
            self._xml(404, "NoSuchKey", "The specified key does not exist.")
            return
        body, meta = store[key]
        self._send_bytes(200, body, meta)

    def _list(self, query: dict[str, list[str]]) -> None:
        prefix = (query.get("prefix") or [""])[0]
        store = self.server.store  # type: ignore[attr-defined]
        items = "".join(
            f"<Contents><Key>{k}</Key><Size>{len(v[0])}</Size></Contents>"
            for k, v in sorted(store.items())
            if k.startswith(prefix)
        )
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<Name>{BUCKET}</Name><Prefix>{prefix}</Prefix>"
            f"<KeyCount>{len(store)}</KeyCount><MaxKeys>1000</MaxKeys>"
            "<IsTruncated>false</IsTruncated>"
            f"{items}</ListBucketResult>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self) -> None:
        if not self._checked():
            return
        bucket, key = self._bucket_key()
        store = self.server.store  # type: ignore[attr-defined]
        if bucket != BUCKET or key not in store:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body, meta = store[key]
        self._send_bytes(200, body, meta)

    def do_DELETE(self) -> None:
        if not self._checked():
            return
        _, key = self._bucket_key()
        self.server.store.pop(key, None)  # type: ignore[attr-defined]
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()


class FakeS3Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), FakeS3Handler)
        self.store: dict[str, tuple[bytes, dict[str, str]]] = {}


# ── tests ────────────────────────────────────────────────────────────────────

class S3TestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.server = FakeS3Server()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        port = self.server.server_address[1]
        self.config = S3Config(
            endpoint=f"http://127.0.0.1:{port}",
            bucket=BUCKET,
            access_key=TEST_KEY,
            secret_key=TEST_SECRET,
            region=TEST_REGION,
            prefix="test-prefix/",
            path_style=True,
            timeout=10.0,
        )
        self.store = S3BlobStore(self.config)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def raw_key(self, sha256: str, compressed: bool = False) -> str:
        suffix = ".gz" if compressed else ""
        return f"test-prefix/{sha256[:2]}/{sha256[2:4]}/{sha256}{suffix}"


class TestSigV4KnownAnswer(unittest.TestCase):
    def test_aws_published_test_vector(self) -> None:
        """AWS Signature Version 4 test suite, get-vanilla vector.

        Expected signature cross-checked against botocore (AWS reference SDK):
        5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31
        """
        headers = sign_request(
            "GET",
            "https://example.amazonaws.com/",
            access_key="AKIDEXAMPLE",
            secret_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
            service="service",
            payload_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            timestamp="20150830T123600Z",
            include_content_sha256=False,
        )
        self.assertEqual(
            headers["Authorization"],
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, "
            "SignedHeaders=host;x-amz-date, "
            "Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31",
        )

    def test_signing_is_deterministic(self) -> None:
        kwargs = dict(
            access_key="AK", secret_key="SK", region="us-east-1",
            timestamp="20250101T000000Z",
        )
        first = sign_request("PUT", "https://s3.example.com/b/k", **kwargs)  # type: ignore[arg-type]
        second = sign_request("PUT", "https://s3.example.com/b/k", **kwargs)  # type: ignore[arg-type]
        self.assertEqual(first["Authorization"], second["Authorization"])

    def test_different_secret_gives_different_signature(self) -> None:
        one = sign_request("GET", "https://s3.example.com/", access_key="A",
                           secret_key="one", region="us-east-1", timestamp="20250101T000000Z")
        two = sign_request("GET", "https://s3.example.com/", access_key="A",
                           secret_key="two", region="us-east-1", timestamp="20250101T000000Z")
        self.assertNotEqual(one["Authorization"], two["Authorization"])


class TestS3BlobStoreRoundTrip(S3TestBase):
    def test_put_get_bytes_round_trip(self) -> None:
        data = b"\x89PNG\r\n\x1a\n" + os.urandom(5000)
        info = self.store.put_bytes(data)
        self.assertEqual(info.mime, "image/png")  # magic-byte sniffed
        self.assertFalse(info.compressed)  # media is stored verbatim
        self.assertEqual(self.store.get_bytes(info.sha256), data)
        self.assertEqual(self.store.stats["puts"], 1)
        self.assertEqual(self.store.stats["gets"], 1)

    def test_compression_for_text(self) -> None:
        data = b'{"hello": "world", "n": 42}\n' * 500
        info = self.store.put_bytes(data, mime="application/json")
        self.assertTrue(info.compressed)
        self.assertLess(info.ratio, 0.5)
        # The stored object really is gzipped bytes under the .gz key.
        raw, _ = self.server.store[self.raw_key(info.sha256, compressed=True)]
        self.assertEqual(gzip.decompress(raw), data)
        self.assertEqual(self.store.get_bytes(info.sha256), data)

    def test_dedup_bumps_refcount(self) -> None:
        data = os.urandom(100)
        first = self.store.put_bytes(data)
        second = self.store.put_bytes(data)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(self.store.stats["dedup_hits"], 1)
        self.assertEqual(self.store.info(first.sha256).refcount, 2)  # type: ignore[union-attr]

    def test_put_file_streams_large_file(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "big.bin"
            source.write_bytes(os.urandom(3 * 1024 * 1024))  # multi-chunk
            info = self.store.put_file(source, mime="application/octet-stream")
            self.assertEqual(info.size, 3 * 1024 * 1024)
            self.assertFalse(info.compressed)
            self.assertEqual(self.store.get_bytes(info.sha256), source.read_bytes())

    def test_put_file_move_deletes_source(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "move.bin"
            source.write_bytes(b"move me")
            self.store.put_file(source, move=True)
            self.assertFalse(source.exists())

    def test_put_file_missing_raises_not_found(self) -> None:
        with self.assertRaises(NotFound):
            self.store.put_file("/nonexistent/path/blob.bin")

    def test_put_stream(self) -> None:
        info = self.store.put_stream(io.BytesIO(b"streamed-bytes" * 1000), mime="text/plain")
        self.assertEqual(self.store.get_bytes(info.sha256), b"streamed-bytes" * 1000)

    def test_export(self) -> None:
        import tempfile

        data = b"export me"
        info = self.store.put_bytes(data)
        with tempfile.TemporaryDirectory() as tmp:
            target = self.store.export(info.sha256, Path(tmp) / "out.bin")
            self.assertEqual(target.read_bytes(), data)

    def test_open_returns_file_like(self) -> None:
        info = self.store.put_bytes(b"hello")
        with self.store.open(info.sha256) as handle:
            self.assertEqual(handle.read(), b"hello")

    def test_info_and_exists(self) -> None:
        info = self.store.put_bytes(b"meta-check", mime="text/plain")
        fetched = self.store.info(info.sha256)
        assert fetched is not None
        self.assertEqual(fetched.mime, "text/plain")
        self.assertEqual(fetched.size, 10)
        self.assertTrue(self.store.exists(info.sha256))
        missing = "0" * 64
        self.assertIsNone(self.store.info(missing))
        self.assertFalse(self.store.exists(missing))

    def test_get_missing_raises_not_found(self) -> None:
        with self.assertRaises(NotFound):
            self.store.get_bytes("0" * 64)

    def test_release_and_purge(self) -> None:
        info = self.store.put_bytes(b"refcounted")
        self.assertEqual(self.store.release(info.sha256), 0)
        self.assertTrue(self.store.exists(info.sha256))  # still there at zero
        self.assertTrue(self.store.purge(info.sha256))
        self.assertFalse(self.store.exists(info.sha256))
        self.assertFalse(self.store.purge(info.sha256))  # idempotent

    def test_release_delete_at_zero(self) -> None:
        info = self.store.put_bytes(b"ephemeral")
        self.assertEqual(self.store.release(info.sha256, delete_at_zero=True), 0)
        self.assertFalse(self.store.exists(info.sha256))

    def test_verify_clean(self) -> None:
        info = self.store.put_bytes(b"verify me" * 100)
        self.assertEqual(self.store.verify(info.sha256), [])
        self.assertEqual(self.store.verify(), [])

    def test_verify_detects_corruption(self) -> None:
        info = self.store.put_bytes(b"tamper me" * 100)
        key = self.raw_key(info.sha256)
        body, meta = self.server.store[key]
        self.server.store[key] = (b"corrupted!" + body[10:], meta)
        problems = self.store.verify(info.sha256)
        self.assertEqual(len(problems), 1)
        self.assertIn("does not match", problems[0])

    def test_prune_unlisted(self) -> None:
        keep = self.store.put_bytes(b"keep me")
        drop = self.store.put_bytes(b"drop me")
        removed = self.store.prune_unlisted({keep.sha256})
        self.assertEqual(len(removed), 1)
        self.assertTrue(self.store.exists(keep.sha256))
        self.assertFalse(self.store.exists(drop.sha256))

    def test_total_size_and_stats(self) -> None:
        one = self.store.put_bytes(b"a" * 100)
        two = self.store.put_bytes(b"b" * 200)
        self.assertEqual(self.store.total_size(), one.stored + two.stored)
        snap = self.store.stats_snapshot()
        self.assertEqual(snap["blobs"], 2)
        self.assertEqual(snap["bucket"], BUCKET)

    def test_bad_credentials_rejected(self) -> None:
        bad = S3Config(
            endpoint=self.config.endpoint, bucket=BUCKET,
            access_key=TEST_KEY, secret_key="WRONG" + TEST_SECRET,
            region=TEST_REGION,
        )
        store = S3BlobStore(bad)
        with self.assertRaises(StorageError) as ctx:
            store.put_bytes(b"nope")
        self.assertIn("SignatureDoesNotMatch", str(ctx.exception))


class TestS3ConfigAndFactory(S3TestBase):
    def test_from_env(self) -> None:
        env = {
            "S3_ENDPOINT": "https://minio.example.com:9000",
            "S3_BUCKET": "blobs",
            "S3_ACCESS_KEY": "ak",
            "S3_SECRET_KEY": "sk",
            "S3_REGION": "eu-west-1",
            "S3_PREFIX": "devon",
        }
        old = dict(os.environ)
        os.environ.update(env)
        try:
            config = S3Config.from_env()
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(config.endpoint, "https://minio.example.com:9000")
        self.assertEqual(config.bucket, "blobs")
        self.assertEqual(config.region, "eu-west-1")
        self.assertEqual(config.prefix, "devon/")  # trailing slash normalized

    def test_config_requires_credentials(self) -> None:
        with self.assertRaises(ConfigError):
            S3Config(endpoint="https://x.example", bucket="b")

    def test_open_blob_store_s3(self) -> None:
        store = open_blob_store("s3", s3_config=self.config)
        self.assertIsInstance(store, S3BlobStore)

    def test_open_blob_store_local(self) -> None:
        import tempfile

        from nomorals.storage.blob import BlobStore
        from nomorals.storage.db import Database

        with tempfile.TemporaryDirectory() as tmp:
            db = Database(":memory:")
            db.migrate()
            store = open_blob_store("local", db=db, root=tmp)
            self.assertIsInstance(store, BlobStore)
            info = store.put_bytes(b"local backend still works")
            self.assertEqual(store.get_bytes(info.sha256), b"local backend still works")
            db.close()

    def test_open_blob_store_unknown_kind(self) -> None:
        with self.assertRaises(ConfigError):
            open_blob_store("gcs")

    def test_open_blob_store_missing_args(self) -> None:
        with self.assertRaises(ConfigError):
            open_blob_store("s3")
        with self.assertRaises(ConfigError):
            open_blob_store("local")


if __name__ == "__main__":
    unittest.main()
