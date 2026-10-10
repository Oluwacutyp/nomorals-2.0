"""S3-compatible blob store: the same content-addressed API as
:class:`~nomorals.storage.blob.BlobStore`, backed by any S3-compatible object
storage (MinIO self-hosted, LocalStack, AWS S3, Wasabi, …) instead of local
files.

Why this exists alongside the local store
-----------------------------------------
Local files are right for a phone or a single VPS. An S3-compatible backend
adds what local files cannot: blobs reachable from *other* machines (phone +
server sharing one store), effectively unlimited capacity, and server-side
durability — without paying for a cloud database. MinIO is free and
self-hosted (AGPL); the client here is pure standard library, so there is no
new pip dependency either way.

Authentication is AWS Signature Version 4, implemented with ``hashlib`` /
``hmac`` only. Object keys reuse the local store's layout
(``{prefix}ab/cd/<sha256>[.gz]``) so the two backends stay conceptually
interchangeable, and all blob metadata (size, MIME, compression flag,
refcount, creation time) travels as S3 object metadata — no database needed.
"""

from __future__ import annotations

import gzip
import hashlib
import hmac
import http.client
import io
import mimetypes
import os
import shutil
import tempfile
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from ..core.errors import ConfigError, NotFound, StorageError
from ..core.logging_setup import get_logger
from .blob import BlobInfo, _ext_for

__all__ = [
    "S3BlobStore",
    "S3Config",
    "open_blob_store",
    "presign_url",
    "sign_request",
]

#: Part size for multipart uploads (boto3 ``Upload`` default is 8 MiB).
_MULTIPART_PART_SIZE = 8 * 1024 * 1024

_log = get_logger(__name__)

CHUNK = 1024 * 1024
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

#: Object-metadata keys (stored as ``x-amz-meta-*`` on the object).
_META_SIZE = "size"
_META_MIME = "mime"
_META_COMPRESSED = "compressed"
_META_REFCOUNT = "refcount"
_META_CREATED = "created"

_DEFAULT_COMPRESSIBLE = (
    "text/",
    "application/json",
    "application/javascript",
    "application/xml",
    "image/svg",
)


# ── AWS Signature Version 4 (stdlib only) ────────────────────────────────────

def _quote(value: str, *, path_segment: bool = False) -> str:
    """RFC 3986 percent-encoding for SigV4. ``~`` stays unreserved (AWS rule)."""
    return urllib.parse.quote(value, safe="/~" if path_segment else "~")


def _canonical_request(
    method: str,
    parsed: urllib.parse.ParseResult,
    query_params: dict[str, str],
    headers: dict[str, str],
    payload_hash: str,
) -> tuple[str, str]:
    """Build the canonical request; return (canonical_request, signed_headers)."""
    canonical_uri = "/".join(
        _quote(segment, path_segment=True) for segment in parsed.path.split("/")
    )
    if not canonical_uri.startswith("/"):
        canonical_uri = "/" + canonical_uri
    canonical_qs = _encode_query(query_params)
    lowered = {k.lower(): v.strip() for k, v in headers.items()}
    canonical_headers = "".join(f"{k}:{lowered[k]}\n" for k in sorted(lowered))
    signed_headers = ";".join(sorted(lowered))
    request = (
        f"{method}\n{canonical_uri}\n{canonical_qs}\n"
        f"{canonical_headers}\n{signed_headers}\n{payload_hash}"
    )
    return request, signed_headers


def _encode_query(params: dict[str, str]) -> str:
    """Percent-encode a query dict exactly the way SigV4 canonicalizes it."""
    return "&".join(f"{_quote(k)}={_quote(v)}" for k, v in sorted(params.items()))


def _signing_key(secret_key: str, datestamp: str, region: str, service: str) -> bytes:
    key = ("AWS4" + secret_key).encode("utf-8")
    for value in (datestamp, region, service, "aws4_request"):
        key = hmac.new(key, value.encode("utf-8"), hashlib.sha256).digest()
    return key


def sign_request(
    method: str,
    url: str,
    *,
    access_key: str,
    secret_key: str,
    region: str,
    service: str = "s3",
    payload_hash: str = _EMPTY_SHA256,
    query_params: dict[str, str] | None = None,
    extra_headers: dict[str, str] | None = None,
    timestamp: str | None = None,
    include_content_sha256: bool = True,
) -> dict[str, str]:
    """Return the signed headers for one S3 request (SigV4, ``Authorization``).

    ``timestamp`` is ``YYYYMMDD'T'HHMMSS'Z'``; pass it explicitly in tests for
    deterministic signatures, otherwise the current UTC time is used.
    ``include_content_sha256=False`` omits the ``x-amz-content-sha256`` header
    (matches the AWS published test vectors; S3 itself accepts either form).
    """
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    default_port = (parsed.scheme == "https" and parsed.port == 443) or (
        parsed.scheme == "http" and parsed.port == 80
    )
    if parsed.port and not default_port:
        host = f"{host}:{parsed.port}"
    amz_date = timestamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    datestamp = amz_date[:8]

    headers = {"host": host, "x-amz-date": amz_date}
    if include_content_sha256:
        headers["x-amz-content-sha256"] = payload_hash
    if extra_headers:
        headers.update({k.lower(): v for k, v in extra_headers.items()})

    canonical, signed = _canonical_request(
        method.upper(), parsed, query_params or {}, headers, payload_hash
    )
    scope = f"{datestamp}/{region}/{service}/aws4_request"
    string_to_sign = (
        f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n"
        + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )
    signing_key = _signing_key(secret_key, datestamp, region, service)
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return {
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
        "Authorization": (
            f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
            f"SignedHeaders={signed}, Signature={signature}"
        ),
    }


def presign_url(
    method: str,
    url: str,
    *,
    access_key: str,
    secret_key: str,
    region: str,
    service: str = "s3",
    expires_in: int = 3600,
    timestamp: str | None = None,
) -> str:
    """Return a SigV4 presigned URL for ``method`` on ``url`` (query auth).

    Presigned URLs let a phone or browser upload/download blobs directly
    against the object store without proxying bytes through the bot — the
    single most useful S3 feature for a personal agent (boto3
    ``generate_presigned_url`` semantics, stdlib only). ``expires_in`` is
    capped at 604800 (7 days, the SigV4 maximum). ``timestamp`` pins the
    signing time for deterministic tests.
    """
    if expires_in <= 0 or expires_in > 604800:
        raise ValueError("expires_in must be within 1..604800 seconds")
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    default_port = (parsed.scheme == "https" and parsed.port == 443) or (
        parsed.scheme == "http" and parsed.port == 80
    )
    if parsed.port and not default_port:
        host = f"{host}:{parsed.port}"
    amz_date = timestamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    datestamp = amz_date[:8]
    scope = f"{datestamp}/{region}/{service}/aws4_request"
    params = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{access_key}/{scope}",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": str(expires_in),
        "X-Amz-SignedHeaders": "host",
    }
    canonical, _signed = _canonical_request(
        method.upper(), parsed, params, {"host": host}, "UNSIGNED-PAYLOAD"
    )
    string_to_sign = (
        f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n"
        + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )
    signing_key = _signing_key(secret_key, datestamp, region, service)
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    params["X-Amz-Signature"] = signature
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return base + "?" + _encode_query(params)


# ── configuration ──────────────────────────────────────────────────────────
@dataclass
class S3Config:
    """Connection details for an S3-compatible endpoint.

    ``endpoint`` is the full base URL, e.g. ``https://minio.example.com:9000``
    or ``https://s3.amazonaws.com``. Credentials are read from here or via
    :meth:`from_env`; they are never logged.
    """

    endpoint: str
    bucket: str
    access_key: str = ""
    secret_key: str = ""
    region: str = "us-east-1"
    prefix: str = ""
    path_style: bool = True
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.endpoint = self.endpoint.rstrip("/")
        if self.prefix and not self.prefix.endswith("/"):
            self.prefix += "/"
        if not self.endpoint or not self.bucket:
            raise ConfigError("S3Config needs an endpoint and a bucket")
        if not self.access_key or not self.secret_key:
            raise ConfigError(
                "S3Config needs access_key and secret_key "
                "(see S3Config.from_env for S3_ACCESS_KEY / S3_SECRET_KEY)"
            )

    @classmethod
    def from_env(cls, prefix: str = "S3_") -> S3Config:
        """Build from ``S3_ENDPOINT`` / ``S3_BUCKET`` / ``S3_ACCESS_KEY`` / …."""
        get = lambda name, default="": os.environ.get(prefix + name, default)  # noqa: E731
        return cls(
            endpoint=get("ENDPOINT"),
            bucket=get("BUCKET"),
            access_key=get("ACCESS_KEY"),
            secret_key=get("SECRET_KEY"),
            region=get("REGION", "us-east-1"),
            prefix=get("PREFIX", ""),
            path_style=get("PATH_STYLE", "1") not in {"0", "false", "no"},
            timeout=float(get("TIMEOUT", "30") or 30),
        )


# ── store ────────────────────────────────────────────────────────────────────

@dataclass
class _S3Object:
    key: str
    metadata: dict[str, str]
    size: int = 0


class S3BlobStore:
    """Content-addressed blob storage on S3-compatible object storage.

    API-compatible with :class:`~nomorals.storage.blob.BlobStore`
    (``put_bytes`` / ``put_file`` / ``put_stream`` / ``get_bytes`` /
    ``export`` / ``open`` / ``info`` / ``exists`` / ``release`` / ``purge`` /
    ``verify``) so callers can swap backends without code changes. Blob
    metadata lives on the object itself (``x-amz-meta-*``), so no database is
    required.
    """

    def __init__(
        self,
        config: S3Config,
        *,
        compress_above: int = 4096,
        compressible: tuple[str, ...] = _DEFAULT_COMPRESSIBLE,
        multipart_threshold: int = 64 * 1024 * 1024,
        multipart_part_size: int = _MULTIPART_PART_SIZE,
    ) -> None:
        if not isinstance(config, S3Config):
            raise ConfigError("S3BlobStore needs an S3Config (see S3Config.from_env)")
        self.config = config
        self.compress_above = compress_above
        self.compressible = compressible
        #: Files at/above this size upload via multipart (boto3 ``Upload``
        #: behavior): parts stream from disk, a failed part retries without
        #: restarting the whole file.
        self.multipart_threshold = multipart_threshold
        self.multipart_part_size = multipart_part_size
        self.stats = {"puts": 0, "dedup_hits": 0, "gets": 0, "bytes_written": 0}

    # ── keys & policy ────────────────────────────────────────────────────────
    def _key(self, sha256: str, compressed: bool = False) -> str:
        suffix = ".gz" if compressed else ""
        return f"{self.config.prefix}{sha256[:2]}/{sha256[2:4]}/{sha256}{suffix}"

    def _is_compressible(self, mime: str) -> bool:
        if not mime:
            return False
        lowered = mime.lower()
        if lowered.startswith(("image/", "video/", "audio/")) and "svg" not in lowered:
            return False
        if lowered in {
            "application/zip", "application/gzip", "application/x-gzip",
            "application/x-7z-compressed", "application/x-rar-compressed",
            "application/x-xz", "application/octet-stream",
        }:
            return False
        return any(lowered.startswith(p) for p in self.compressible)

    # ── low-level HTTP ───────────────────────────────────────────────────────
    def _url(self, key: str, query_params: dict[str, str] | None = None) -> str:
        if self.config.path_style:
            path = f"/{self.config.bucket}/{key}"
        else:
            parsed = urllib.parse.urlparse(self.config.endpoint)
            host = f"{self.config.bucket}.{parsed.hostname}"
            if parsed.port:
                host += f":{parsed.port}"
            path = f"/{key}"
            base = f"{parsed.scheme}://{host}"
            url = base + path
            if query_params:
                url += "?" + _encode_query(query_params)
            return url
        url = self.config.endpoint + path
        if query_params:
            url += "?" + _encode_query(query_params)
        return url

    def _request(
        self,
        method: str,
        key: str,
        *,
        query_params: dict[str, str] | None = None,
        body: bytes | BinaryIO | None = None,
        content_length: int | None = None,
        extra_headers: dict[str, str] | None = None,
        payload_hash: str | None = None,
        retries: int = 2,
    ) -> tuple[int, dict[str, str], bytes]:
        """One signed S3 request. Returns (status, headers, body)."""
        url = self._url(key, query_params)
        parsed = urllib.parse.urlparse(url)

        if body is None:
            payload = b""
            content_length = 0
        elif isinstance(body, bytes):
            payload = body
            content_length = len(body)
        else:
            payload = body  # streaming file object; content_length must be set

        digest = payload_hash or (
            hashlib.sha256(payload).hexdigest() if isinstance(payload, bytes) else _EMPTY_SHA256
        )
        # The Host header on the wire must equal the `host` value that was
        # signed (hostname plus non-default port).
        host_header = parsed.hostname or ""
        default_port = (parsed.scheme == "https" and parsed.port == 443) or (
            parsed.scheme == "http" and parsed.port == 80
        )
        if parsed.port and not default_port:
            host_header = f"{host_header}:{parsed.port}"
        signed = sign_request(
            method,
            url,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            region=self.config.region,
            payload_hash=digest,
            query_params=query_params or {},
            extra_headers=extra_headers,
        )
        headers = {
            "Host": host_header,
            "Content-Length": str(content_length or 0),
            "x-amz-date": signed["x-amz-date"],
            "Authorization": signed["Authorization"],
        }
        if "x-amz-content-sha256" in signed:
            headers["x-amz-content-sha256"] = signed["x-amz-content-sha256"]
        if extra_headers:
            headers.update(extra_headers)

        # Canonical path for the wire: http.client wants the raw path+query.
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        # For virtual-hosted style the signature was computed over the
        # bucket-prefixed host; http.client connects to that same host.
        connect_host = parsed.hostname or ""
        connect_port = parsed.port

        attempt = 0
        while True:
            attempt += 1
            try:
                if parsed.scheme == "https":
                    conn = http.client.HTTPSConnection(
                        connect_host, connect_port, timeout=self.config.timeout
                    )
                else:
                    conn = http.client.HTTPConnection(
                        connect_host, connect_port, timeout=self.config.timeout
                    )
                try:
                    conn.request(method.upper(), path, body=payload, headers=headers)
                    response = conn.getresponse()
                    status = response.status
                    resp_headers = {
                        k.lower(): v for k, v in response.getheaders()
                    }
                    data = response.read()
                finally:
                    conn.close()
            except (OSError, http.client.HTTPException) as exc:
                raise StorageError(
                    f"S3 request {method} {self.config.bucket}/{key} failed: {exc}"
                ) from exc

            if status in (500, 503, 429) and attempt <= retries:
                time.sleep(min(2.0 ** attempt, 8.0))
                continue
            return status, resp_headers, data

    @staticmethod
    def _s3_error(status: int, data: bytes) -> str:
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            root = None
        if root is not None:
            code = root.findtext("Code") or root.findtext("{*}Code") or ""
            message = root.findtext("Message") or root.findtext("{*}Message") or ""
            if code or message:
                return f"S3 error {status}: {code} {message}".strip()
        text = data[:200].decode("utf-8", "replace").strip()
        return f"S3 error {status}: {text}" if text else f"S3 error {status}"

    def _check(
        self, method: str, key: str, status: int, data: bytes
    ) -> None:
        if 200 <= status < 300:
            return
        if status == 404:
            raise NotFound(f"blob object not found: {key}")
        raise StorageError(self._s3_error(status, data))

    # ── object primitives ──────────────────────────────────────────────────
    def _meta_headers(self, metadata: dict[str, str]) -> dict[str, str]:
        return {f"x-amz-meta-{k}": str(v) for k, v in metadata.items()}

    def _put_object(
        self, key: str, data: bytes, metadata: dict[str, str]
    ) -> None:
        """PUT a complete in-memory payload. Streaming uploads go through
        :meth:`_put_stream_hashed`, which needs the payload hash up front."""
        status, _, body = self._request(
            "PUT",
            key,
            body=data,
            extra_headers=self._meta_headers(metadata),
            payload_hash=hashlib.sha256(data).hexdigest(),
        )
        self._check("PUT", key, status, body)

    def _put_stream_hashed(
        self, key: str, stream: BinaryIO, length: int, payload_hash: str,
        metadata: dict[str, str],
    ) -> None:
        status, _, body = self._request(
            "PUT", key, body=stream, content_length=length,
            extra_headers=self._meta_headers(metadata), payload_hash=payload_hash,
        )
        self._check("PUT", key, status, body)

    # ── multipart upload ─────────────────────────────────────────────────
    #
    # boto3's ``Upload`` switches to multipart for large files: parts upload
    # independently (a failed part retries alone) and the server assembles
    # them. Implemented here with stdlib HTTP so model files and media stop
    # being single-shot 2GB PUTs.

    @staticmethod
    def _xml_text(data: bytes, *tags: str) -> str:
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            return ""
        for tag in tags:
            found = root.findtext(tag) or root.findtext("{*}" + tag)
            if found:
                return found
        return ""

    def _create_multipart(self, key: str, metadata: dict[str, str]) -> str:
        status, _, body = self._request(
            "POST", key, query_params={"uploads": ""},
            extra_headers=self._meta_headers(metadata),
        )
        self._check("POST (create multipart)", key, status, body)
        upload_id = self._xml_text(body, "UploadId")
        if not upload_id:
            raise StorageError(f"multipart create for {key} returned no UploadId")
        return upload_id

    def _upload_part(
        self, key: str, upload_id: str, part_number: int, data: bytes
    ) -> str:
        status, headers, body = self._request(
            "PUT", key,
            query_params={"partNumber": str(part_number), "uploadId": upload_id},
            body=data,
            payload_hash=hashlib.sha256(data).hexdigest(),
        )
        self._check(f"PUT (part {part_number})", key, status, body)
        etag = headers.get("etag", "")
        if not etag:
            raise StorageError(f"multipart part {part_number} for {key} returned no ETag")
        return etag

    def _complete_multipart(
        self, key: str, upload_id: str, etags: list[str]
    ) -> None:
        parts = "".join(
            f"<Part><PartNumber>{i + 1}</PartNumber><ETag>{etag}</ETag></Part>"
            for i, etag in enumerate(etags)
        )
        payload = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<CompleteMultipartUpload>{parts}</CompleteMultipartUpload>"
        ).encode("utf-8")
        status, _, body = self._request(
            "POST", key,
            query_params={"uploadId": upload_id},
            body=payload,
            payload_hash=hashlib.sha256(payload).hexdigest(),
            extra_headers={"Content-Type": "application/xml"},
        )
        self._check("POST (complete multipart)", key, status, body)

    def _abort_multipart(self, key: str, upload_id: str) -> None:
        try:
            status, _, body = self._request(
                "DELETE", key, query_params={"uploadId": upload_id}, retries=0,
            )
            if status not in (200, 204, 404):
                _log.warning("multipart abort for %s returned %s", key, status)
        except Exception as exc:  # noqa: BLE001 - abort is best-effort cleanup
            _log.debug("multipart abort for %s failed: %s", key, exc)

    def _put_file_multipart(
        self, key: str, path: Path, size: int, payload_hash: str,
        metadata: dict[str, str],
        progress: Any = None,
    ) -> None:
        """Stream ``path`` to ``key`` in parts. Aborts cleanly on failure."""
        upload_id = self._create_multipart(key, metadata)
        etags: list[str] = []
        sent = 0
        try:
            with path.open("rb") as handle:
                part_number = 1
                while True:
                    chunk = handle.read(self.multipart_part_size)
                    if not chunk:
                        break
                    etags.append(
                        self._upload_part(key, upload_id, part_number, chunk)
                    )
                    part_number += 1
                    sent += len(chunk)
                    if progress is not None:
                        progress(sent, size)
            if not etags:  # pragma: no cover - empty file never reaches here
                raise StorageError(f"cannot multipart-upload empty file {path}")
            self._complete_multipart(key, upload_id, etags)
        except Exception:
            self._abort_multipart(key, upload_id)
            raise

    def _head(self, key: str) -> dict[str, str] | None:
        status, headers, _ = self._request("HEAD", key)
        if status == 404:
            return None
        if not 200 <= status < 300:
            raise StorageError(self._head_error(key, status))
        return headers

    def _head_error(self, key: str, status: int) -> str:
        """Build the error message for a failed HEAD request.

        HEAD responses never carry a body, so the S3 error code would be lost
        (``S3 error 403`` tells the operator nothing). One GET on the error
        path recovers the machine-readable code — ``SignatureDoesNotMatch``,
        ``AccessDenied``, … — because S3 checks the signature before it looks
        the key up. The GET body is only trusted when the GET itself failed;
        anything else falls back to the bare status.
        """
        get_status, data = 0, b""
        try:
            get_status, _, data = self._request("GET", key, retries=0)
        except Exception as exc:  # noqa: BLE001 - error-path best effort
            _log.debug("error-detail GET failed for %s: %s", key, exc)
        if data and not 200 <= get_status < 300:
            return self._s3_error(status, data)
        return f"S3 error {status}"

    def _get(self, key: str) -> tuple[bytes, dict[str, str]]:
        status, headers, data = self._request("GET", key)
        if status == 404:
            raise NotFound(f"blob object not found: {key}")
        self._check("GET", key, status, data)
        return data, headers

    def _delete(self, key: str) -> None:
        status, _, data = self._request("DELETE", key)
        if status == 404:
            return
        self._check("DELETE", key, status, data)

    def _copy_in_place(self, key: str, metadata: dict[str, str]) -> None:
        """Rewrite object metadata without re-uploading bytes (S3 COPY)."""
        source = f"/{self.config.bucket}/{_quote(key, path_segment=True)}"
        status, _, data = self._request(
            "PUT", key,
            extra_headers={
                **self._meta_headers(metadata),
                "x-amz-copy-source": source,
                "x-amz-metadata-directive": "REPLACE",
            },
        )
        self._check("PUT (copy)", key, status, data)

    def _list(self, prefix: str = "") -> list[_S3Object]:
        """List every object under ``prefix`` (follows continuation tokens)."""
        objects: list[_S3Object] = []
        token = ""
        while True:
            params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if token:
                params["continuation-token"] = token
            status, _, data = self._request("GET", "", query_params=params)
            self._check("GET (list)", prefix or "/", status, data)
            try:
                root = ET.fromstring(data)
            except ET.ParseError as exc:
                raise StorageError(f"cannot parse S3 list response: {exc}") from exc
            ns = ""
            if root.tag.startswith("{"):
                ns = root.tag.split("}")[0] + "}"
            for content in root.findall(f"{ns}Contents"):
                name = content.findtext(f"{ns}Key") or ""
                size = int(content.findtext(f"{ns}Size") or 0)
                objects.append(_S3Object(key=name, metadata={}, size=size))
            truncated = (root.findtext(f"{ns}IsTruncated") or "").lower() == "true"
            if not truncated:
                break
            token = root.findtext(f"{ns}NextContinuationToken") or ""
            if not token:
                break
        return objects

    # ── metadata <-> BlobInfo ──────────────────────────────────────────────
    @staticmethod
    def _metadata_for(info: BlobInfo) -> dict[str, str]:
        return {
            _META_SIZE: str(info.size),
            _META_MIME: info.mime,
            _META_COMPRESSED: "1" if info.compressed else "0",
            _META_REFCOUNT: str(info.refcount),
            _META_CREATED: repr(info.created_at),
        }

    @staticmethod
    def _info_from(sha256: str, headers: dict[str, str]) -> BlobInfo | None:
        def meta(name: str) -> str:
            return headers.get(f"x-amz-meta-{name}", "")

        if not meta(_META_SIZE):
            return None
        try:
            return BlobInfo(
                sha256=sha256,
                size=int(meta(_META_SIZE)),
                mime=meta(_META_MIME),
                compressed=meta(_META_COMPRESSED) == "1",
                stored=int(headers.get("content-length", meta(_META_SIZE)) or 0),
                refcount=int(meta(_META_REFCOUNT) or 1),
                created_at=float(meta(_META_CREATED) or 0.0),
            )
        except (ValueError, TypeError):
            return None

    def _sha_from_key(self, key: str) -> str | None:
        name = key.rsplit("/", 1)[-1]
        if name.endswith(".gz"):
            name = name[:-3]
        return name if len(name) == 64 and all(c in "0123456789abcdef" for c in name) else None

    # ── writes ─────────────────────────────────────────────────────────────
    def put_bytes(self, data: bytes, *, mime: str = "", refcount: int = 1) -> BlobInfo:
        sha256 = hashlib.sha256(data).hexdigest()
        existing = self.info(sha256)
        if existing is not None:
            self._bump_refcount(sha256, existing, refcount)
            self.stats["dedup_hits"] += 1
            return self.info(sha256) or existing

        guessed = mime or mimetypes.guess_type("f" + _ext_for(data))[0] or "application/octet-stream"
        should_compress = len(data) >= self.compress_above and self._is_compressible(guessed)
        payload = gzip.compress(data, compresslevel=6) if should_compress else data
        key = self._key(sha256, should_compress)
        info = BlobInfo(sha256, len(data), guessed, should_compress, len(payload), refcount, time.time())
        self._put_object(key, payload, self._metadata_for(info))
        self.stats["puts"] += 1
        self.stats["bytes_written"] += len(payload)
        return info

    def put_file(
        self,
        source: str | os.PathLike[str],
        *,
        mime: str = "",
        refcount: int = 1,
        move: bool = False,
        progress: Any = None,
    ) -> BlobInfo:
        """Store a file, streaming the upload so large files stay off the heap.

        Files at/above ``multipart_threshold`` upload in parts (boto3
        ``Upload`` behavior); ``progress(done, total)`` reports both paths.
        """
        path = Path(source).expanduser()
        if not path.is_file():
            raise NotFound(f"file not found: {path}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(CHUNK):
                digest.update(chunk)
                size += len(chunk)
        sha256 = digest.hexdigest()

        existing = self.info(sha256)
        if existing is not None:
            self._bump_refcount(sha256, existing, refcount)
            self.stats["dedup_hits"] += 1
            if move:
                path.unlink()
            return self.info(sha256) or existing

        guessed = mime or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        should_compress = size >= self.compress_above and self._is_compressible(guessed)
        key = self._key(sha256, should_compress)

        tmp: Path | None = None
        try:
            if should_compress:
                tmp = Path(tempfile.mktemp(prefix="nm-s3blob-", suffix=".gz"))
                with path.open("rb") as src, gzip.open(tmp, "wb", compresslevel=6) as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
                upload_path, payload_hash, stored = tmp, _sha256_file(tmp), tmp.stat().st_size
            else:
                upload_path, stored = path, size
                payload_hash = sha256
            info = BlobInfo(sha256, size, guessed, should_compress, stored, refcount, time.time())
            metadata = self._metadata_for(info)
            if stored >= self.multipart_threshold:
                # Multipart needs a real file path (parts stream from disk).
                self._put_file_multipart(
                    key, upload_path, stored, payload_hash, metadata,
                    progress=progress,
                )
            else:
                with upload_path.open("rb") as handle:
                    self._put_stream_hashed(
                        key, handle, stored, payload_hash, metadata,
                    )
                if progress is not None:
                    progress(stored, stored)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
        if move:
            path.unlink()
        self.stats["puts"] += 1
        self.stats["bytes_written"] += stored
        return info

    def put_stream(
        self,
        stream: BinaryIO,
        *,
        mime: str = "",
        filename: str = "",
        refcount: int = 1,
    ) -> BlobInfo:
        """Store from a file-like object, spooling to a temp file first."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            tmp_path = Path(tmp.name)
            digest = hashlib.sha256()
            size = 0
            while chunk := stream.read(CHUNK):
                digest.update(chunk)
                size += len(chunk)
                tmp.write(chunk)
        try:
            return self.put_file(tmp_path, mime=mime, refcount=refcount, move=True)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    # ── reads ──────────────────────────────────────────────────────────────
    def get_bytes(self, sha256: str) -> bytes:
        info = self.info(sha256)
        if info is None:
            raise NotFound(f"blob {sha256} not in store")
        data, _ = self._get(self._key(sha256, info.compressed))
        self.stats["gets"] += 1
        if info.compressed:
            try:
                return gzip.decompress(data)
            except (gzip.BadGzipFile, EOFError, OSError) as exc:
                raise StorageError(f"blob {sha256} is corrupt (bad gzip): {exc}") from exc
        return data

    def export(self, sha256: str, destination: str | os.PathLike[str]) -> Path:
        data = self.get_bytes(sha256)
        target = Path(destination).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    def get_range(self, sha256: str, start: int, end: int) -> bytes:
        """Fetch bytes ``[start, end]`` (inclusive, HTTP Range semantics).

        Media streaming without downloading the whole object. Compressed
        blobs raise :class:`StorageError` — ranges address stored bytes, and
        a gzip member is not addressable; callers should store media
        uncompressed (the default policy already does).
        """
        if start < 0 or end < start:
            raise ValueError("invalid range")
        info = self.info(sha256)
        if info is None:
            raise NotFound(f"blob {sha256} not in store")
        if info.compressed:
            raise StorageError(
                f"blob {sha256} is gzip-compressed; byte ranges are not addressable"
            )
        key = self._key(sha256, False)
        status, _, data = self._request(
            "GET", key, extra_headers={"Range": f"bytes={start}-{end}"}
        )
        if status == 404:
            raise NotFound(f"blob object not found: {key}")
        if status not in (200, 206):
            raise StorageError(self._s3_error(status, data))
        self.stats["gets"] += 1
        return data

    def open(self, sha256: str) -> BinaryIO:
        return io.BytesIO(self.get_bytes(sha256))

    # ── presigned URLs & bucket setup ────────────────────────────────────

    def presigned_get(self, sha256: str, *, expires_in: int = 3600) -> str:
        """A shareable download URL for ``sha256`` (no credentials needed).

        Hand this to a phone, a browser, or another machine: it downloads the
        blob directly from object storage without proxying through the bot.
        """
        info = self.info(sha256)
        if info is None:
            raise NotFound(f"blob {sha256} not in store")
        url = self._url(self._key(sha256, info.compressed))
        return presign_url(
            "GET", url,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            region=self.config.region,
            expires_in=expires_in,
        )

    def presigned_put(self, key_hint: str = "", *, expires_in: int = 900,
                      content_type: str = "") -> dict[str, str]:
        """A one-shot upload URL: the holder PUTs bytes straight to storage.

        Returns ``{"url": ..., "key": ...}``. The object lands under this
        store's prefix but outside content addressing — call
        :meth:`put_file`/:meth:`put_bytes` afterwards (or a ``sync`` pass)
        to index it. ``content_type`` is advisory for the uploader.
        """
        import secrets

        hint = "".join(
            c for c in key_hint if c.isalnum() or c in "-_.")[:64] or "upload"
        key = f"{self.config.prefix}incoming/{int(time.time())}-{secrets.token_hex(8)}-{hint}"
        url = self._url(key)
        signed = presign_url(
            "PUT", url,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            region=self.config.region,
            expires_in=expires_in,
        )
        return {"url": signed, "key": key, "content_type": content_type}

    def ensure_bucket(self) -> bool:
        """Create the bucket if missing. Returns True when created."""
        url = self._url("")
        # _url("") with path_style gives "<endpoint>/<bucket>/"; strip the
        # trailing slash for the bucket-level PUT.
        bucket_url = url.rstrip("/") if self.config.path_style else url
        parsed = urllib.parse.urlparse(bucket_url)
        host = parsed.hostname or ""
        signed = sign_request(
            "PUT", bucket_url,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            region=self.config.region,
        )
        headers = {
            "Host": host,
            "Content-Length": "0",
            "x-amz-date": signed["x-amz-date"],
            "Authorization": signed["Authorization"],
        }
        path = parsed.path or "/"
        try:
            conn_cls = (http.client.HTTPSConnection
                        if parsed.scheme == "https" else http.client.HTTPConnection)
            conn = conn_cls(host, parsed.port, timeout=self.config.timeout)
            try:
                conn.request("PUT", path, body=b"", headers=headers)
                response = conn.getresponse()
                status, data = response.status, response.read()
            finally:
                conn.close()
        except (OSError, http.client.HTTPException) as exc:
            raise StorageError(f"ensure_bucket failed: {exc}") from exc
        if status in (200, 409):  # 409 BucketAlreadyOwnedByYou
            return status == 200
        raise StorageError(self._s3_error(status, data))

    # ── metadata ───────────────────────────────────────────────────────────
    def info(self, sha256: str) -> BlobInfo | None:
        for compressed in (False, True):
            headers = self._head(self._key(sha256, compressed))
            if headers is not None:
                return self._info_from(sha256, headers)
        return None

    def exists(self, sha256: str) -> bool:
        return self.info(sha256) is not None

    def _bump_refcount(self, sha256: str, info: BlobInfo, delta: int) -> None:
        new_count = max(0, info.refcount + delta)
        updated = BlobInfo(
            sha256, info.size, info.mime, info.compressed,
            info.stored, new_count, info.created_at,
        )
        self._copy_in_place(self._key(sha256, info.compressed), self._metadata_for(updated))

    def release(self, sha256: str, *, delete_at_zero: bool = False) -> int:
        info = self.info(sha256)
        if info is None:
            return 0
        self._bump_refcount(sha256, info, -1)
        if delete_at_zero and info.refcount - 1 <= 0:
            self.purge(sha256)
            return 0
        return max(0, info.refcount - 1)

    def purge(self, sha256: str) -> bool:
        info = self.info(sha256)
        if info is None:
            return False
        self._delete(self._key(sha256, info.compressed))
        # A stale twin (same sha, other compression flag) must not survive.
        self._delete(self._key(sha256, not info.compressed))
        return True

    # ── maintenance ────────────────────────────────────────────────────────
    def list_keys(self) -> list[str]:
        """Every object key under this store's prefix."""
        return [obj.key for obj in self._list(self.config.prefix)]

    def verify(self, sha256: str | None = None) -> list[str]:
        """Re-download and re-hash blobs; returns a list of corruption problems.

        When ``sha256`` is given, the object is resolved through :meth:`info`
        first so exactly the key that exists is verified — probing the
        ``.gz`` twin of an uncompressed blob (or vice versa) would report a
        phantom "HEAD failed" for a key that was never supposed to exist.
        """
        problems: list[str] = []
        if sha256 is not None:
            info = self.info(sha256)
            if info is None:
                return [f"{sha256}: blob not found in store"]
            keys = [self._key(sha256, info.compressed)]
        else:
            keys = self.list_keys()
        for key in keys:
            digest = self._sha_from_key(key)
            if digest is None:
                problems.append(f"{key}: not a content-addressed key")
                continue
            headers = self._head(key)
            if headers is None:
                problems.append(f"{digest}: listed but HEAD failed for {key}")
                continue
            info = self._info_from(digest, headers)
            if info is None:
                problems.append(f"{digest}: unreadable object metadata at {key}")
                continue
            data, _ = self._get(key)
            try:
                raw = gzip.decompress(data) if info.compressed else data
            except (gzip.BadGzipFile, EOFError, OSError) as exc:
                problems.append(f"{digest}: cannot decompress stored bytes: {exc}")
                continue
            if hashlib.sha256(raw).hexdigest() != digest:
                problems.append(f"{digest}: rehashed content does not match key")
            elif len(raw) != info.size:
                problems.append(f"{digest}: size {len(raw)} != recorded {info.size}")
        return problems

    def prune_unlisted(self, known: Iterable[str]) -> list[str]:
        """Delete objects whose SHA-256 is not in ``known``. Returns deleted keys."""
        keep = set(known)
        removed: list[str] = []
        for key in self.list_keys():
            digest = self._sha_from_key(key)
            if digest is None or digest not in keep:
                self._delete(key)
                removed.append(key)
        return removed

    def total_size(self) -> int:
        return sum(obj.size for obj in self._list(self.config.prefix))

    def stats_snapshot(self) -> dict[str, Any]:
        objects = self._list(self.config.prefix)
        return {
            **self.stats,
            "blobs": len(objects),
            "bytes": sum(o.size for o in objects),
            "endpoint": self.config.endpoint,
            "bucket": self.config.bucket,
            "prefix": self.config.prefix,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def open_blob_store(
    kind: str = "local",
    *,
    db: Any = None,
    root: str | os.PathLike[str] | None = None,
    s3_config: S3Config | None = None,
    **kwargs: Any,
) -> Any:
    """Open a blob store by backend name.

    ``kind="local"`` needs ``db`` (a :class:`~nomorals.storage.db.Database`)
    and ``root`` (directory). ``kind="s3"`` needs ``s3_config`` (an
    :class:`S3Config`). ``kind="auto"`` uses the S3 backend when the
    environment fully configures it (``S3_ENDPOINT`` / ``S3_BUCKET`` /
    ``S3_ACCESS_KEY`` / ``S3_SECRET_KEY``) and falls back to the local store
    otherwise — the zero-config default keeps working with nothing installed
    and nothing configured. All three return objects with the same blob API,
    so callers stay backend-agnostic.
    """
    if kind == "auto":
        config: S3Config | None = None
        try:
            config = S3Config.from_env()
        except ConfigError:
            config = None
        if config is not None:
            _log.info(
                "blob store: auto-selected s3 backend (%s/%s)",
                config.endpoint, config.bucket,
            )
            return S3BlobStore(config, **kwargs)
        _log.info("blob store: no S3 env config; auto-selected local backend")
        kind = "local"
    if kind == "local":
        from .blob import BlobStore

        if db is None or root is None:
            raise ConfigError("local blob store needs db= and root=")
        return BlobStore(db, root, **kwargs)
    if kind == "s3":
        if s3_config is None:
            raise ConfigError("s3 blob store needs s3_config= (see S3Config.from_env)")
        return S3BlobStore(s3_config, **kwargs)
    raise ConfigError(f"unknown blob backend: {kind!r} (expected 'local', 's3' or 'auto')")
