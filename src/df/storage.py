"""Object storage: presigned uploads in, TTL deletes out.

Everything is behind the `Storage` protocol so the retention path can be tested
against a real, inspectable backend (InMemoryStorage) instead of a mock that
would happily report success without deleting anything. See
tests/test_retention_ttl.py.
"""
from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

from df.config import settings


@runtime_checkable
class Storage(Protocol):
    def presign_put(self, key: str, content_type: str, max_bytes: int) -> str: ...
    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None: ...
    def get_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def list_prefix(self, prefix: str) -> list[str]: ...
    def copy_object(self, src: str, dst: str) -> bool: ...
    def delete_object(self, key: str) -> bool: ...
    def delete_prefix(self, prefix: str) -> int: ...


class InMemoryStorage:
    """Test/dev backend. Thread-safe because workers share one instance in tests."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def presign_put(self, key: str, content_type: str, max_bytes: int) -> str:
        return f"memory://{key}?content_type={content_type}&max_bytes={max_bytes}"

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        with self._lock:
            self._objects[key] = data

    def get_bytes(self, key: str) -> bytes:
        with self._lock:
            return self._objects[key]

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._objects

    def list_prefix(self, prefix: str) -> list[str]:
        with self._lock:
            return sorted(k for k in self._objects if k.startswith(prefix))

    def copy_object(self, src: str, dst: str) -> bool:
        with self._lock:
            if src not in self._objects:
                return False
            self._objects[dst] = self._objects[src]
            return True

    def delete_object(self, key: str) -> bool:
        with self._lock:
            return self._objects.pop(key, None) is not None

    def delete_prefix(self, prefix: str) -> int:
        with self._lock:
            keys = [k for k in self._objects if k.startswith(prefix)]
            for k in keys:
                del self._objects[k]
            return len(keys)


class S3Storage:
    """boto3-backed. Works against MinIO locally and S3 in deployed envs."""

    def __init__(self, *, bucket: str | None = None, endpoint: str | None = None) -> None:
        import boto3
        from botocore.config import Config

        self.bucket = bucket or settings.s3_bucket
        cfg = Config(
            signature_version="s3v4",
            s3={"addressing_style": "path" if settings.s3_force_path_style else "auto"},
            retries={"max_attempts": 3, "mode": "standard"},
        )
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint or settings.s3_endpoint,
            region_name=settings.s3_region,
            config=cfg,
        )
        # Separate client signing against the browser-reachable hostname. Inside
        # compose the gateway talks to `minio` but the user's browser cannot
        # resolve that, and the host is part of the signature.
        self._public_client = boto3.client(
            "s3",
            endpoint_url=settings.s3_public_endpoint,
            region_name=settings.s3_region,
            config=cfg,
        )

    def presign_put(self, key: str, content_type: str, max_bytes: int) -> str:
        # Tier 1: large files bypass the API entirely and go straight to S3.
        return self._public_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=settings.presign_ttl_seconds,
        )

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def get_bytes(self, key: str) -> bytes:
        return self._client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def list_prefix(self, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return keys

    def copy_object(self, src: str, dst: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.copy_object(
                Bucket=self.bucket, Key=dst, CopySource={"Bucket": self.bucket, "Key": src}
            )
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def delete_object(self, key: str) -> bool:
        if not self.exists(key):
            return False
        self._client.delete_object(Bucket=self.bucket, Key=key)
        return True

    def delete_prefix(self, prefix: str) -> int:
        keys = self.list_prefix(prefix)
        deleted = 0
        for i in range(0, len(keys), 1000):
            batch = keys[i : i + 1000]
            self._client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
            )
            deleted += len(batch)
        return deleted


def build_storage() -> Storage:
    if settings.s3_endpoint.startswith("memory://"):
        return InMemoryStorage()
    return S3Storage()


# --- key layout -------------------------------------------------------------
# raw/<job_id>/original          the upload; deleted on inference completion
# derived/<job_id>/...           face crops, aligned faces, spectrograms; same TTL
# cold/<job_id>/...              extended retention window copies (Tier 2)


def raw_key(job_id: str) -> str:
    return f"raw/{job_id}/original"


def derived_prefix(job_id: str) -> str:
    return f"derived/{job_id}/"


def cold_prefix(job_id: str) -> str:
    return f"cold/{job_id}/"
