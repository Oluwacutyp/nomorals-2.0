"""AWS connector — EC2 inventory and S3 object storage via boto3.

Auth: IAM access key pair (``AuthMethod.API_KEY``). The access key id is
vault-stored as the credential username, the secret access key as the
password, and the default region in vault metadata. The connector passes
the vaulted credentials explicitly to every boto3 client — it never relies
on ambient ``~/.aws`` / environment credentials, so Devon's calls always
use the key the owner connected with.

Capabilities:
* EC2: list instances across regions (``describe_instances``)
* S3: list buckets, list objects (``list_objects_v2``), upload
  (``put_object``, confirmation-gated), download (``get_object``)

boto3 is a hard dependency of this module's AWS calls (SigV4 is never
hand-rolled). If boto3 is not importable the connector fails fast with a
clear install hint instead of half-working.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .auth import prompt_secret
from ._confirm import confirm_or_checkpoint
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = ["AWSConnector", "AWSError"]

_log = get_logger(__name__)

try:  # boto3 is optional at import time; required for any AWS call.
    import boto3  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised via missing-boto3 test
    boto3 = None  # type: ignore[assignment]

INSTALL_HINT = "boto3 is required for the AWS connector: pip install boto3"

DEFAULT_REGION = "us-east-1"

#: Hard cap for single-put uploads. S3 ``put_object`` tops out at 5 GiB;
#: past this the connector refuses rather than silently loading gigabytes
#: into memory — multipart upload is the documented gap (see s3_upload).
_PUT_OBJECT_MAX_BYTES = 5 * 1024 * 1024 * 1024


class AWSError(ConnectorError):
    """An AWS API call failed (auth, permissions, or service error)."""

    def __init__(
        self, message: str, *, code: str = "", status_code: int = 0
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _require_boto3() -> Any:
    if boto3 is None:
        raise AWSError(
            f"{INSTALL_HINT} — the AWS connector cannot sign requests "
            "without it"
        )
    return boto3


@register_connector
class AWSConnector(Connector):
    """Devon's AWS adapter: EC2 inventory + S3 objects over boto3."""

    id = "aws"
    name = "AWS"
    description = (
        "Amazon Web Services: list EC2 instances, list S3 buckets, "
        "upload/download S3 objects. Authenticates with an IAM access "
        "key pair (requires boto3)."
    )
    auth_methods = (AuthMethod.API_KEY,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        region: str = "",
    ) -> ConnectResult:
        """Validate an IAM key pair against STS and vault-store it."""
        _require_boto3()
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "aws is already connected — one account per service. "
                "Disconnect first to switch keys."
            )
        key_id = ((access_key_id or "").strip() or prompt_secret(
            "AWS access key id", env_var="AWS_ACCESS_KEY_ID"
        ))
        secret = ((secret_access_key or "").strip() or prompt_secret(
            "AWS secret access key", env_var="AWS_SECRET_ACCESS_KEY"
        ))
        if not key_id or not secret:
            raise ConnectorError(
                "empty access key id or secret — nothing to connect with"
            )
        region = (region or "").strip() or DEFAULT_REGION
        identity = self._wrap(
            "sts get_caller_identity",
            self._sts(key_id, secret, region).get_caller_identity,
        )
        arn = str(identity.get("Arn", ""))
        account = str(identity.get("Account", ""))
        cred = self._store_credential(
            key_id,
            secret,
            credential_type="api_key",
            metadata={
                "region": region,
                "account_id": account,
                "arn": arn,
            },
        )
        _log.info("aws connected: account %s (%s)", account, region)
        return ConnectResult(
            ok=True,
            account=arn or key_id,
            message=(
                f"connected to AWS account {account} in {region}. The key "
                "pair is in the encrypted vault; rotate it in the IAM "
                "console any time."
            ),
            credential_id=cred.id,
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect "
                       "--name aws`",
            )
        region = str((cred.metadata or {}).get("region", DEFAULT_REGION))
        try:
            identity = self._wrap(
                "sts get_caller_identity",
                self._sts(
                    cred.username, cred.password, region
                ).get_caller_identity,
            )
        except AWSError as exc:
            return ConnectorStatus(
                connected=False,
                account=cred.username,
                last_checked=time.time(),
                detail=f"key rejected ({exc}): reconnect with a fresh key",
            )
        return ConnectorStatus(
            connected=True,
            account=str(identity.get("Arn", cred.username)),
            last_checked=time.time(),
            detail=f"key valid (account {identity.get('Account', '?')})",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._wrap(
                "sts get_caller_identity",
                self._sts(
                    cred.username, cred.password,
                    str((cred.metadata or {}).get("region", DEFAULT_REGION)),
                ).get_caller_identity,
            )
            return True
        except ConnectorError:
            return False

    # ── EC2 ──────────────────────────────────────────────────────

    def list_ec2_instances(
        self, *, region: str = ""
    ) -> list[dict[str, Any]]:
        """Every EC2 instance in ``region`` (``describe_instances``).

        Returns a flat list of instance dicts with the fields operators
        actually scan: ``instance_id``, ``state``, ``instance_type``,
        ``public_ip``, ``private_ip``, ``launch_time`` (ISO), ``tags``.
        """
        client = self._client("ec2", region)
        paginator = client.get_paginator("describe_instances")
        instances: list[dict[str, Any]] = []
        for page in self._wrap(
            "describe_instances", paginator.paginate, PaginationConfig={}
        ):
            for reservation in page.get("Reservations", []):
                for raw in reservation.get("Instances", []):
                    instances.append(self._summarize_instance(raw))
        return instances

    @staticmethod
    def _summarize_instance(raw: dict[str, Any]) -> dict[str, Any]:
        tags = {t.get("Key", ""): t.get("Value", "")
                for t in raw.get("Tags", [])}
        launch = raw.get("LaunchTime")
        return {
            "instance_id": raw.get("InstanceId", ""),
            "state": (raw.get("State") or {}).get("Name", ""),
            "instance_type": raw.get("InstanceType", ""),
            "public_ip": raw.get("PublicIpAddress", ""),
            "private_ip": raw.get("PrivateIpAddress", ""),
            "launch_time": (
                launch.isoformat() if hasattr(launch, "isoformat")
                else str(launch or "")
            ),
            "name": tags.get("Name", ""),
            "tags": tags,
        }

    # ── S3 ───────────────────────────────────────────────────────

    def list_s3_buckets(self) -> list[dict[str, Any]]:
        """All buckets the key can see (``list_buckets``)."""
        client = self._client("s3")
        data = self._wrap("list_buckets", client.list_buckets)
        out = []
        for bucket in data.get("Buckets", []):
            created = bucket.get("CreationDate")
            out.append({
                "name": bucket.get("Name", ""),
                "created": (
                    created.isoformat() if hasattr(created, "isoformat")
                    else str(created or "")
                ),
            })
        return out

    def s3_list(
        self, bucket: str, *, prefix: str = "", limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Objects under ``prefix`` in ``bucket`` (``list_objects_v2``)."""
        bucket = (bucket or "").strip()
        if not bucket:
            raise ConnectorError("empty bucket name")
        client = self._client("s3")
        paginator = client.get_paginator("list_objects_v2")
        objects: list[dict[str, Any]] = []
        remaining = max(1, limit)
        for page in self._wrap(
            "list_objects_v2",
            paginator.paginate,
            Bucket=bucket,
            Prefix=prefix,
            PaginationConfig={"PageSize": min(remaining, 1000)},
        ):
            for raw in page.get("Contents", []):
                objects.append({
                    "key": raw.get("Key", ""),
                    "size": raw.get("Size", 0),
                    "last_modified": (
                        raw["LastModified"].isoformat()
                        if hasattr(raw.get("LastModified"), "isoformat")
                        else str(raw.get("LastModified", ""))
                    ),
                    "etag": (raw.get("ETag") or "").strip('"'),
                    "storage_class": raw.get("StorageClass", ""),
                })
                if len(objects) >= limit:
                    return objects
            remaining = limit - len(objects)
            if remaining <= 0:
                break
        return objects

    def s3_upload(
        self,
        bucket: str,
        key: str,
        source: str | Path | bytes,
        *,
        content_type: str = "",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Upload bytes to ``s3://bucket/key`` (``put_object``).

        Uploading moves data into the owner's cloud account, so it never
        runs on implied consent: pass ``confirmed=True`` after the owner
        approved the exact source/destination, or ``db`` to park the exact
        payload on a human checkpoint.

        Honest limit: this is a single ``put_object`` — files larger than
        5 GiB are refused; multipart upload is not implemented yet (use the
        AWS CLI for multi-gigabyte files).
        """
        bucket = (bucket or "").strip()
        key = (key or "").strip()
        if not bucket or not key:
            raise ConnectorError("bucket and key are both required")
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise ConnectorError(f"no such file: {path}")
            size = path.stat().st_size
            if size > _PUT_OBJECT_MAX_BYTES:
                raise ConnectorError(
                    f"{path} is {size} bytes — over the 5 GiB single-put "
                    "limit; multipart upload is not implemented, use the "
                    "AWS CLI for this file"
                )
            data = path.read_bytes()
            label = str(path)
        elif isinstance(source, bytes):
            if len(source) > _PUT_OBJECT_MAX_BYTES:
                raise ConnectorError(
                    "payload exceeds the 5 GiB single-put limit; "
                    "multipart upload is not implemented"
                )
            data, label, size = source, "<bytes>", len(source)
        else:
            raise ConnectorError(
                "source must be a file path or bytes, "
                f"got {type(source).__name__}"
            )
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="s3_upload",
            title=f"Upload to s3://{bucket}/{key}",
            instructions="\n".join([
                "Devon wants to upload this object to your S3 bucket.",
                f"Source: {label} ({size} bytes)",
                f"Destination: s3://{bucket}/{key}",
                "The upload overwrites any object already at that key.",
            ]),
            resume_state={
                "bucket": bucket, "key": key, "source": label,
                "size": size,
            },
        )
        client = self._client("s3")
        kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key, "Body": data}
        if content_type:
            kwargs["ContentType"] = content_type
        resp = self._wrap("put_object", client.put_object, **kwargs)
        _log.info("aws s3 upload ok: s3://%s/%s (%d bytes)",
                  bucket, key, size)
        return {
            "bucket": bucket,
            "key": key,
            "size": size,
            "etag": (resp.get("ETag") or "").strip('"'),
        }

    def s3_download(
        self,
        bucket: str,
        key: str,
        dest: str | Path | None = None,
    ) -> dict[str, Any]:
        """Download ``s3://bucket/key`` (``get_object``).

        Returns the bytes in ``data`` when ``dest`` is omitted; otherwise
        streams to ``dest`` and returns the path. The whole object is read
        into memory first — very large objects should go through the AWS
        CLI instead.
        """
        bucket = (bucket or "").strip()
        key = (key or "").strip()
        if not bucket or not key:
            raise ConnectorError("bucket and key are both required")
        client = self._client("s3")
        resp = self._wrap(
            "get_object", client.get_object, Bucket=bucket, Key=key
        )
        body = resp.get("Body")
        if body is None or not hasattr(body, "read"):
            raise AWSError("s3 get_object returned no readable body")
        data = body.read()
        if hasattr(body, "close"):
            body.close()
        if dest is not None:
            path = Path(dest)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            _log.info("aws s3 download ok: s3://%s/%s -> %s (%d bytes)",
                      bucket, key, path, len(data))
            return {
                "bucket": bucket, "key": key, "path": str(path),
                "size": len(data),
            }
        return {
            "bucket": bucket, "key": key, "size": len(data), "data": data,
        }

    # ── boto3 plumbing ───────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "aws is not connected — run "
                "`nm connectors connect --name aws` first"
            )
        return cred

    def _default_region(self) -> str:
        cred = self._load_credential()
        if cred is not None:
            return str(
                (cred.metadata or {}).get("region", DEFAULT_REGION)
            )
        return DEFAULT_REGION

    def _sts(self, key_id: str, secret: str, region: str) -> Any:
        mod = _require_boto3()
        return mod.client(
            "sts",
            region_name=region,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
        )

    def _client(self, service: str, region: str = "") -> Any:
        """A boto3 client bound to the vaulted key pair (never ambient)."""
        mod = _require_boto3()
        cred = self._require_credential()
        return mod.client(
            service,
            region_name=(region or "").strip() or self._default_region(),
            aws_access_key_id=cred.username,
            aws_secret_access_key=cred.password,
        )

    def _wrap(self, op: str, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Call boto3; service/auth failures become AWSError."""
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - boto3 error shapes vary
            err = getattr(exc, "response", {}) or {}
            info = err.get("Error", {}) if isinstance(err, dict) else {}
            code = str(info.get("Code", "") or type(exc).__name__)
            message = str(info.get("Message", "") or exc)
            status = 0
            http_resp = err.get("ResponseMetadata", {}) if isinstance(
                err, dict) else {}
            if isinstance(http_resp, dict):
                status = int(http_resp.get("HTTPStatusCode", 0) or 0)
            hint = ""
            if code in ("InvalidClientTokenId", "SignatureDoesNotMatch",
                        "UnrecognizedClientException"):
                hint = " — the key pair is invalid or revoked; reconnect"
            elif code in ("AccessDenied", "AccessDeniedException",
                          "UnauthorizedOperation"):
                hint = (" — the key lacks permission for this call; widen "
                        "the IAM policy")
            elif status == 429 or "throttl" in code.lower():
                hint = " — throttled; back off and retry"
            raise AWSError(
                f"aws {op} failed ({code or 'error'}): {message}{hint}",
                code=code,
                status_code=status,
            ) from exc
