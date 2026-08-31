"""Object storage: presigned uploads in, TTL deletes out.

Everything is behind the `Storage` protocol so the retention path can be tested
against a real, inspectable backend (InMemoryStorage) instead of a mock that
would happily report success without deleting anything. See
tests/test_retention_ttl.py.
"""
from __future__ import annotations

import pathlib
import threading
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from df.config import settings


@dataclass(frozen=True)
class PresignedUpload:
    """A browser-usable upload grant.

    POST rather than PUT because a presigned PUT cannot express a size limit:
    the signature covers the URL, not the body length, so a PUT grant is an
    unbounded write to the bucket. A POST policy carries a
    content-length-range condition that the storage backend itself enforces
    and rejects, which matters because Tier 1 sends large uploads straight
    past the API -- ingress rate limiting never sees these bytes.
    """

    url: str
    fields: dict[str, str] = field(default_factory=dict)
    method: str = "POST"


@runtime_checkable
class Storage(Protocol):
    def presign_upload(
        self, key: str, content_type: str, max_bytes: int
    ) -> PresignedUpload: ...
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

    def presign_upload(self, key: str, content_type: str, max_bytes: int) -> PresignedUpload:
        return PresignedUpload(
            url=f"memory://{key}",
            fields={"key": key, "Content-Type": content_type, "x-max-bytes": str(max_bytes)},
        )

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

    def presign_upload(self, key: str, content_type: str, max_bytes: int) -> PresignedUpload:
        # Tier 1: large files bypass the API entirely and go straight to S3.
        # Signed on the PUBLIC client -- the host is part of the signature, so a
        # URL signed against the internal `minio` name is unusable by a browser.
        #
        # content-length-range is the only thing standing between an issued
        # grant and an unbounded write: these bytes never traverse the gateway,
        # so DF_MAX_UPLOAD_BYTES cannot be enforced anywhere else.
        post = self._public_client.generate_presigned_post(
            Bucket=self.bucket,
            Key=key,
            Fields={"Content-Type": content_type},
            Conditions=[
                {"Content-Type": content_type},
                ["content-length-range", 1, max_bytes],
            ],
            ExpiresIn=settings.presign_ttl_seconds,
        )
        return PresignedUpload(url=post["url"], fields=post["fields"])

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


class LocalDiskStorage:
    """Filesystem backend for a single-container deploy.

    Exists because presigned upload needs object storage, and running MinIO as
    an extra service on a demo platform is a moving part with no payoff when
    every process already shares one filesystem.

    THE UPLOAD PATH CHANGES SHAPE HERE, and CLAUDE.md is explicit about the
    property being traded. With S3/MinIO the browser POSTs straight to storage,
    so the bytes never traverse the gateway -- which is why the
    `content-length-range` condition in the POST policy is described there as
    the ONLY place DF_MAX_UPLOAD_BYTES can be enforced.

    On local disk there is no third party to POST to, so the grant points back
    at this service's own `/v1/uploads/{key}` endpoint. The client flow is
    unchanged -- still "POST to upload_url with upload_fields" -- but the bytes
    now go through the gateway. The size cap is therefore enforced directly by
    that endpoint rather than by a signed policy, and ingress rate limiting sees
    the upload for the first time. Different mechanism, and not weaker; but it
    IS different from what the compose deployment does, so do not read the two
    as interchangeable.

    EPHEMERAL. A redeploy wipes this directory while the Postgres rows survive,
    so job rows will reference media that no longer exists. That is survivable
    only because the evidence display reads scores and per-face detail from
    Postgres, never re-reading the image -- `media_deleted` already models media
    disappearing, since Tier 1 deletes it on completion anyway.
    """

    def __init__(self, root: str | None = None) -> None:
        self.root = pathlib.Path(root or settings.local_storage_root)
        # Created at startup rather than on first write: a permission error on
        # the first upload is indistinguishable from a broken API.
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> pathlib.Path:
        # Keys are service-generated (raw/<uuid>/original, derived/<uuid>/...),
        # never client-supplied, but resolve anyway so a future caller cannot
        # traverse out of the root.
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError(f"key escapes storage root: {key!r}")
        return p

    def presign_upload(self, key: str, content_type: str, max_bytes: int) -> PresignedUpload:
        return PresignedUpload(
            url=f"{settings.public_base_url}/v1/uploads",
            fields={
                "key": key,
                "Content-Type": content_type,
                # Advisory only. The endpoint enforces the real limit; a client
                # that lies simply gets a 413 instead of skipping the check.
                "x-max-bytes": str(max_bytes),
            },
        )

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a reader must never see a half-written crop, and
        # the workers poll for these across processes.
        tmp = p.with_suffix(p.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(p)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list_prefix(self, prefix: str) -> list[str]:
        base = self.root.resolve()
        start = self._path(prefix) if not prefix.endswith("/") else self._path(prefix.rstrip("/"))
        if not start.exists():
            # Prefix may be a partial path rather than a directory; fall back to
            # scanning, which is fine at demo scale.
            return sorted(
                str(p.relative_to(base)).replace("\\", "/")
                for p in base.rglob("*")
                if p.is_file() and str(p.relative_to(base)).replace("\\", "/").startswith(prefix)
            )
        if start.is_file():
            return [prefix]
        return sorted(
            str(p.relative_to(base)).replace("\\", "/")
            for p in start.rglob("*") if p.is_file()
        )

    def copy_object(self, src: str, dst: str) -> bool:
        s = self._path(src)
        if not s.is_file():
            return False
        self.put_bytes(dst, s.read_bytes())
        return True

    def delete_object(self, key: str) -> bool:
        p = self._path(key)
        if not p.is_file():
            return False
        p.unlink()
        return True

    def delete_prefix(self, prefix: str) -> int:
        keys = self.list_prefix(prefix)
        for k in keys:
            self.delete_object(k)
        return len(keys)


def build_storage() -> Storage:
    if settings.s3_endpoint.startswith("memory://"):
        return InMemoryStorage()
    if settings.s3_endpoint.startswith("file://"):
        return LocalDiskStorage()
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
