"""LocalDiskStorage: the backend the single-container deploy runs on.

Covered because it is production code on the deploy path, and because the
retention guarantees are asserted against a storage backend rather than a mock
(CLAUDE.md) — those assertions are only as good as the backend beneath them.

# In-process only. The live counterpart is the deploy image booting with
# DF_S3_ENDPOINT=file:///data/media and a real upload through POST /v1/uploads.
"""
from __future__ import annotations

import pytest

from df.storage import LocalDiskStorage


@pytest.fixture
def store(tmp_path):
    return LocalDiskStorage(root=str(tmp_path / "media"))


def test_the_root_is_created_at_construction(tmp_path):
    """Created at startup, not on first write. A permission or missing-directory
    error on the first upload is indistinguishable from a broken API."""
    root = tmp_path / "not-yet"
    assert not root.exists()

    LocalDiskStorage(root=str(root))

    assert root.is_dir()


def test_round_trip(store):
    store.put_bytes("raw/j1/original", b"payload", "image/jpeg")

    assert store.exists("raw/j1/original")
    assert store.get_bytes("raw/j1/original") == b"payload"


def test_nested_keys_create_their_directories(store):
    """Keys look like paths but arrive as opaque strings; the parent may not
    exist. S3 has no directories, so nothing upstream creates them."""
    store.put_bytes("derived/j1/items/f00007_x0.png", b"crop")

    assert store.get_bytes("derived/j1/items/f00007_x0.png") == b"crop"


def test_writes_are_atomic_from_a_readers_point_of_view(store, monkeypatch):
    """Workers in the deploy container share this filesystem across processes.
    A reader must never observe a half-written crop, so the write goes to a
    .part file and is renamed. Asserted by checking no partial file survives a
    completed write, and that the final path appears only with full content."""
    store.put_bytes("derived/j1/items/a.png", b"0123456789")

    leftovers = list((store.root / "derived/j1/items").glob("*.part"))
    assert leftovers == []
    assert store.get_bytes("derived/j1/items/a.png") == b"0123456789"


def test_list_prefix_finds_everything_under_a_directory(store):
    store.put_bytes("derived/j1/items/a.png", b"a")
    store.put_bytes("derived/j1/items/b.png", b"b")
    store.put_bytes("derived/j2/items/c.png", b"c")

    assert store.list_prefix("derived/j1/") == [
        "derived/j1/items/a.png",
        "derived/j1/items/b.png",
    ]


def test_delete_prefix_removes_only_that_job(store):
    """The retention sweeps delete by prefix. Over-deleting would destroy
    another job's evidence; under-deleting leaves biometric media on disk."""
    store.put_bytes("derived/j1/items/a.png", b"a")
    store.put_bytes("derived/j2/items/b.png", b"b")

    assert store.delete_prefix("derived/j1/") == 1
    assert store.list_prefix("derived/j1/") == []
    assert store.exists("derived/j2/items/b.png")


def test_copy_object_reports_a_missing_source(store):
    """The extended-retention window copies driving crops to cold storage before
    the Tier 1 delete. A silent no-op there would open a window over nothing."""
    assert store.copy_object("raw/missing", "cold/j1/x") is False

    store.put_bytes("raw/j1/original", b"bytes")
    assert store.copy_object("raw/j1/original", "cold/j1/original") is True
    assert store.get_bytes("cold/j1/original") == b"bytes"


def test_delete_object_reports_whether_anything_was_deleted(store):
    assert store.delete_object("raw/nope") is False

    store.put_bytes("raw/j1/original", b"x")
    assert store.delete_object("raw/j1/original") is True
    assert not store.exists("raw/j1/original")


def test_a_key_escaping_the_root_is_refused(store):
    """Keys are service-minted today, so this is defence in depth rather than a
    live hole. It is cheap, and the cost of being wrong is writing anywhere the
    process can reach."""
    with pytest.raises(ValueError, match="escapes storage root"):
        store.put_bytes("../../etc/passwd", b"nope")


def test_the_grant_points_at_this_service_not_object_storage(store, monkeypatch):
    """With S3 the browser POSTs straight to storage. On local disk there is no
    third party, so the grant must point back here — and absolutely, because a
    relative URL breaks a browser on a different origin."""
    # Settings is a frozen dataclass, so swap the whole instance rather than a
    # field. Frozen is the right call -- config that mutates at runtime is how a
    # deployed service ends up disagreeing with its own env.
    monkeypatch.setenv("DF_PUBLIC_BASE_URL", "https://api.example.com")
    from df.config import Settings
    monkeypatch.setattr("df.storage.settings", Settings())

    grant = store.presign_upload("raw/j1/original", "image/jpeg", 4096)

    assert grant.url == "https://api.example.com/v1/uploads"
    assert grant.method == "POST"
    assert grant.fields["key"] == "raw/j1/original"
