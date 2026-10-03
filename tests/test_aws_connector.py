"""AWS connector tests. boto3 is fully mocked — no network, no AWS."""

from __future__ import annotations

import json
import unittest
import urllib.parse
from typing import Any
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.aws import AWSConnector, AWSError
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.registry import get_connector
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


# ── fake boto3 ───────────────────────────────────────────────────


class FakeClientError(Exception):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.response = {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class FakeSTS:
    def __init__(self, fail: FakeClientError | None = None) -> None:
        self.fail = fail

    def get_caller_identity(self) -> dict[str, Any]:
        if self.fail:
            raise self.fail
        return {
            "UserId": "AIDATEST",
            "Account": "123456789012",
            "Arn": "arn:aws:iam::123456789012:user/devon",
        }


class FakePaginator:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.kwargs: dict[str, Any] = {}

    def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.kwargs = kwargs
        return self.pages


INSTANCE = {
    "InstanceId": "i-0123",
    "State": {"Name": "running"},
    "InstanceType": "t3.micro",
    "PublicIpAddress": "1.2.3.4",
    "PrivateIpAddress": "10.0.0.5",
    "Tags": [{"Key": "Name", "Value": "web"}],
}


class FakeEC2:
    def __init__(self, pages: list[dict[str, Any]] | None = None) -> None:
        self.paginator = FakePaginator(
            pages if pages is not None
            else [{"Reservations": [{"Instances": [INSTANCE]}]}]
        )

    def get_paginator(self, name: str) -> FakePaginator:
        assert name == "describe_instances"
        return self.paginator


class FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.closed = False

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        self.closed = True


class FakeS3:
    def __init__(self) -> None:
        self.puts: list[dict[str, Any]] = []
        self.pages: list[dict[str, Any]] = [
            {"Contents": [
                {"Key": "a.txt", "Size": 11, "ETag": '"abc"',
                 "StorageClass": "STANDARD"},
            ]}
        ]
        self.fail: FakeClientError | None = None

    def list_buckets(self) -> dict[str, Any]:
        if self.fail:
            raise self.fail
        return {"Buckets": [{"Name": "my-bucket"}]}

    def get_paginator(self, name: str) -> FakePaginator:
        assert name == "list_objects_v2"
        return FakePaginator(self.pages)

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.puts.append(kwargs)
        return {"ETag": '"def456"'}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["Bucket"] == "my-bucket"
        assert kwargs["Key"] == "a.txt"
        return {"Body": FakeBody(b"hello world"),
                "ContentLength": 11}


class FakeBoto3:
    def __init__(self) -> None:
        self.sts = FakeSTS()
        self.ec2 = FakeEC2()
        self.s3 = FakeS3()
        self.client_kwargs: dict[str, dict[str, Any]] = {}

    def client(self, service: str, **kwargs: Any) -> Any:
        self.client_kwargs[service] = kwargs
        return {"sts": self.sts, "ec2": self.ec2, "s3": self.s3}[service]


def _aws(fake: FakeBoto3 | None = None) -> tuple[AWSConnector, FakeBoto3]:
    fake = fake or FakeBoto3()
    patcher = mock.patch("nomorals.connectors.aws.boto3", fake)
    patcher.start()
    conn = AWSConnector(_vault())
    conn._boto3_patcher = patcher  # type: ignore[attr-defined]
    return conn, fake


def _connected(fake: FakeBoto3 | None = None) -> tuple[AWSConnector, FakeBoto3]:
    conn, fake = _aws(fake)
    result = conn.connect(access_key_id="AKIAIOSFODNN7EXAMPLE",
                          secret_access_key="wJalrXUtnFEMI",
                          region="eu-west-1")
    assert result.ok
    return conn, fake


class RegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        self.assertIs(get_connector("aws"), AWSConnector)

    def test_metadata(self) -> None:
        self.assertEqual(AWSConnector.id, "aws")
        self.assertIn("api_key", [m.value for m in AWSConnector.auth_methods])


class MissingBoto3Tests(unittest.TestCase):
    def test_missing_boto3_fails_fast(self) -> None:
        with mock.patch("nomorals.connectors.aws.boto3", None):
            conn = AWSConnector(_vault())
            with self.assertRaises(AWSError) as ctx:
                conn.connect(access_key_id="x", secret_access_key="y")
            self.assertIn("pip install boto3", str(ctx.exception))
            with self.assertRaises(AWSError):
                conn.list_s3_buckets()


class ConnectTests(unittest.TestCase):
    def tearDown(self) -> None:
        for conn_holder in getattr(self, "_conns", []):
            conn_holder._boto3_patcher.stop()

    def _track(self, conn: AWSConnector) -> AWSConnector:
        self._conns = getattr(self, "_conns", []) + [conn]
        return conn

    def test_connect_validates_and_stores(self) -> None:
        conn, fake = _aws()
        conn = self._track(conn)
        result = conn.connect(access_key_id="AKIAIOSFODNN7EXAMPLE",
                              secret_access_key="wJalrXUtnFEMI",
                              region="eu-west-1")
        self.assertTrue(result.ok)
        self.assertIn("123456789012", result.account)
        cred = conn.vault.get("connector:aws", "AKIAIOSFODNN7EXAMPLE")
        self.assertEqual(cred.password, "wJalrXUtnFEMI")
        self.assertEqual((cred.metadata or {}).get("region"), "eu-west-1")
        self.assertEqual((cred.metadata or {}).get("account_id"),
                         "123456789012")
        # the vaulted key pair (not ambient creds) reaches boto3
        self.assertEqual(
            fake.client_kwargs["sts"]["aws_access_key_id"],
            "AKIAIOSFODNN7EXAMPLE")
        self.assertEqual(fake.client_kwargs["sts"]["region_name"],
                         "eu-west-1")

    def test_connect_default_region(self) -> None:
        conn, _fake = _aws()
        conn = self._track(conn)
        conn.connect(access_key_id="AKIA1", secret_access_key="s1")
        cred = conn.vault.get("connector:aws", "AKIA1")
        self.assertEqual((cred.metadata or {}).get("region"), "us-east-1")

    def test_connect_rejects_second_key(self) -> None:
        conn, _fake = _connected()
        conn = self._track(conn)
        with self.assertRaises(ConnectorError) as ctx:
            conn.connect(access_key_id="OTHER", secret_access_key="x")
        self.assertIn("already connected", str(ctx.exception))

    def test_connect_bad_key_fails_fast(self) -> None:
        fake = FakeBoto3()
        fake.sts = FakeSTS(FakeClientError("InvalidClientTokenId",
                                           "bad key", 403))
        conn, _f = _aws(fake)
        conn = self._track(conn)
        with self.assertRaises(AWSError) as ctx:
            conn.connect(access_key_id="BAD", secret_access_key="BAD")
        self.assertIn("InvalidClientTokenId", str(ctx.exception))
        self.assertIsNone(conn._load_credential())

    def test_connect_empty_key_raises(self) -> None:
        conn, _fake = _aws()
        conn = self._track(conn)
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ConnectorError):
                conn.connect(access_key_id="", secret_access_key="")

    def test_disconnect_clears(self) -> None:
        conn, _fake = _connected()
        conn = self._track(conn)
        conn.disconnect()
        self.assertIsNone(conn._load_credential())
        conn.disconnect()  # idempotent


class StatusTests(unittest.TestCase):
    def tearDown(self) -> None:
        for conn_holder in getattr(self, "_conns", []):
            conn_holder._boto3_patcher.stop()

    def test_status_not_connected(self) -> None:
        conn, _fake = _aws()
        self._conns = [conn]
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertFalse(conn.test_connection())

    def test_status_connected(self) -> None:
        conn, _fake = _connected()
        self._conns = [conn]
        st = conn.status()
        self.assertTrue(st.connected)
        self.assertIn("123456789012", st.account or "")
        self.assertTrue(conn.test_connection())

    def test_status_rejected_key(self) -> None:
        conn, fake = _connected()
        self._conns = [conn]
        fake.sts = FakeSTS(FakeClientError("SignatureDoesNotMatch",
                                           "bad sig", 403))
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("rejected", st.detail)
        self.assertFalse(conn.test_connection())


class Ec2Tests(unittest.TestCase):
    def tearDown(self) -> None:
        for conn_holder in getattr(self, "_conns", []):
            conn_holder._boto3_patcher.stop()

    def test_list_ec2_instances(self) -> None:
        conn, fake = _connected()
        self._conns = [conn]
        instances = conn.list_ec2_instances()
        self.assertEqual(len(instances), 1)
        inst = instances[0]
        self.assertEqual(inst["instance_id"], "i-0123")
        self.assertEqual(inst["state"], "running")
        self.assertEqual(inst["instance_type"], "t3.micro")
        self.assertEqual(inst["public_ip"], "1.2.3.4")
        self.assertEqual(inst["name"], "web")
        # region override reaches the client
        conn.list_ec2_instances(region="ap-south-1")
        self.assertEqual(fake.client_kwargs["ec2"]["region_name"],
                         "ap-south-1")

    def test_list_ec2_not_connected(self) -> None:
        conn, _fake = _aws()
        self._conns = [conn]
        with self.assertRaises(ConnectorError) as ctx:
            conn.list_ec2_instances()
        self.assertIn("not connected", str(ctx.exception))

    def test_ec2_access_denied_hint(self) -> None:
        fake = FakeBoto3()
        conn, _f = _connected(fake)
        self._conns = [conn]
        def _boom(**kw: Any) -> Any:
            raise FakeClientError("UnauthorizedOperation",
                                  "not authorized", 403)
        fake.ec2.paginator.paginate = _boom  # type: ignore[method-assign]
        with self.assertRaises(AWSError) as ctx:
            conn.list_ec2_instances()
        self.assertIn("IAM policy", str(ctx.exception))


class S3Tests(unittest.TestCase):
    def tearDown(self) -> None:
        for conn_holder in getattr(self, "_conns", []):
            conn_holder._boto3_patcher.stop()

    def test_list_s3_buckets(self) -> None:
        conn, _fake = _connected()
        self._conns = [conn]
        buckets = conn.list_s3_buckets()
        self.assertEqual([b["name"] for b in buckets], ["my-bucket"])

    def test_s3_list(self) -> None:
        conn, _fake = _connected()
        self._conns = [conn]
        objects = conn.s3_list("my-bucket", prefix="a")
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["key"], "a.txt")
        self.assertEqual(objects[0]["size"], 11)
        self.assertEqual(objects[0]["etag"], "abc")

    def test_s3_list_empty_bucket_name(self) -> None:
        conn, _fake = _connected()
        self._conns = [conn]
        with self.assertRaises(ConnectorError):
            conn.s3_list("")

    def test_s3_upload_confirmed(self) -> None:
        import tempfile
        conn, fake = _connected()
        self._conns = [conn]
        with tempfile.TemporaryDirectory() as tmp:
            src = f"{tmp}/up.txt"
            with open(src, "w") as fh:
                fh.write("payload")
            result = conn.s3_upload("my-bucket", "up.txt", src,
                                    confirmed=True)
        self.assertEqual(result["bucket"], "my-bucket")
        self.assertEqual(result["key"], "up.txt")
        self.assertEqual(result["etag"], "def456")
        self.assertEqual(fake.s3.puts[0]["Bucket"], "my-bucket")
        self.assertEqual(fake.s3.puts[0]["Body"], b"payload")

    def test_s3_upload_bytes(self) -> None:
        conn, fake = _connected()
        self._conns = [conn]
        result = conn.s3_upload("my-bucket", "b.bin", b"\x00\x01",
                                confirmed=True,
                                content_type="application/octet-stream")
        self.assertEqual(result["size"], 2)
        self.assertEqual(fake.s3.puts[0]["ContentType"],
                         "application/octet-stream")

    def test_s3_upload_needs_confirmation(self) -> None:
        conn, fake = _connected()
        self._conns = [conn]
        with self.assertRaises(ConnectorError) as ctx:
            conn.s3_upload("my-bucket", "up.txt", b"data")
        self.assertIn("confirmation", str(ctx.exception))
        self.assertEqual(fake.s3.puts, [])

    def test_s3_upload_missing_file(self) -> None:
        conn, _fake = _connected()
        self._conns = [conn]
        with self.assertRaises(ConnectorError):
            conn.s3_upload("my-bucket", "x", "/no/such/file",
                           confirmed=True)

    def test_s3_upload_empty_bucket_or_key(self) -> None:
        conn, _fake = _connected()
        self._conns = [conn]
        with self.assertRaises(ConnectorError):
            conn.s3_upload("", "k", b"d", confirmed=True)

    def test_s3_upload_too_big(self) -> None:
        conn, _fake = _connected()
        self._conns = [conn]
        with mock.patch("nomorals.connectors.aws._PUT_OBJECT_MAX_BYTES", 4):
            with self.assertRaises(ConnectorError) as ctx:
                conn.s3_upload("my-bucket", "big", b"12345", confirmed=True)
            self.assertIn("multipart", str(ctx.exception))

    def test_s3_upload_service_error(self) -> None:
        fake = FakeBoto3()
        conn, _f = _connected(fake)
        self._conns = [conn]
        def _boom(**kw: Any) -> Any:
            raise FakeClientError("AccessDenied", "denied", 403)
        fake.s3.put_object = _boom  # type: ignore[method-assign]
        with self.assertRaises(AWSError) as ctx:
            conn.s3_upload("my-bucket", "up.txt", b"data", confirmed=True)
        self.assertIn("AccessDenied", str(ctx.exception))

    def test_s3_download_bytes(self) -> None:
        conn, _fake = _connected()
        self._conns = [conn]
        result = conn.s3_download("my-bucket", "a.txt")
        self.assertEqual(result["data"], b"hello world")
        self.assertEqual(result["size"], 11)

    def test_s3_download_to_file(self) -> None:
        import tempfile
        conn, _fake = _connected()
        self._conns = [conn]
        with tempfile.TemporaryDirectory() as tmp:
            dest = f"{tmp}/sub/a.txt"
            result = conn.s3_download("my-bucket", "a.txt", dest=dest)
            with open(dest, "rb") as fh:
                self.assertEqual(fh.read(), b"hello world")
            self.assertEqual(result["path"], dest)

    def test_s3_download_missing(self) -> None:
        fake = FakeBoto3()
        conn, _f = _connected(fake)
        self._conns = [conn]
        def _boom(**kw: Any) -> Any:
            raise FakeClientError("NoSuchKey", "no key", 404)
        fake.s3.get_object = _boom  # type: ignore[method-assign]
        with self.assertRaises(AWSError) as ctx:
            conn.s3_download("my-bucket", "nope.txt")
        self.assertIn("NoSuchKey", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
