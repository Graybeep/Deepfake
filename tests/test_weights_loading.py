"""The torch-backend paths that can be checked without torch installed.

Two of them, and both were real bugs that only surfaced when a real checkpoint
ran through the real stack for the first time:

  * the stub extractors emitted a bare sha256 digest as `data`, which the CPU
    worker stored under a `.png` key with `content_type="image/png"`. Bytes
    labelled as an image, named as an image, and not an image. Every job
    dead-lettered on `cv2.imdecode` returning None the moment a detector that
    actually decodes its input was wired in.
  * the checkpoint is a training checkpoint with DataParallel-prefixed keys,
    so it needs unwrapping and de-prefixing before it will load at all.

# In-process only for the PNG shape and the key rewriting. The live counterpart
# is scripts/smoke_compose.py against the docker-compose.weights.yml overlay,
# which is what proved the real B7 checkpoint loads (0 missing / 0 unexpected
# keys) and scores a job end to end at validation=research-checkpoint.
# `measured: yes` 2026-08-29.
"""
from __future__ import annotations

import hashlib

from df.inference.efficientnet import _state_dict_from
from df.pipelines.extract import (
    StubAudioChunker,
    StubFaceExtractor,
    StubFrameSampler,
    _stub_png,
)

PNG_SIGNATURE = bytes([137, 80, 78, 71, 13, 10, 26, 10])


# --- the stub extractors must emit decodable images -------------------------


def test_stub_png_is_a_real_png_not_a_hash_digest():
    """The regression. A 32-byte sha256 digest is not a PNG, and storing one
    under a .png key with content_type image/png is a claim, not a filename."""
    png = _stub_png(b"anything")

    assert png[:8] == PNG_SIGNATURE
    assert png[-8:-4] == b"IEND"
    # A digest is 32 bytes; anything near that length is the old behaviour back.
    assert len(png) > 100


def test_stub_png_stays_deterministic__with_a_positive_control():
    """Determinism is the whole reason the stubs exist: the same upload must
    produce the same score run after run. The sibling assertion matters as much
    -- a generator that returned one constant image would also be deterministic
    and would make every job score identically."""
    assert _stub_png(b"same") == _stub_png(b"same")
    assert _stub_png(b"one") != _stub_png(b"two")


def test_every_stub_extractor_emits_png_bytes():
    """All three feed storage keys ending .png. Frame.data's own type comment
    says "encoded still (PNG/JPEG bytes)", so the contract was always right and
    the stubs were the side violating it."""
    frames = StubFrameSampler().sample(b"video")
    crops = StubFaceExtractor(force_faces=2).extract(b"image")
    chunks = StubAudioChunker().chunk(b"audio")

    assert frames and crops and chunks
    assert all(f.data[:8] == PNG_SIGNATURE for f in frames)
    assert all(c.data[:8] == PNG_SIGNATURE for c in crops)
    assert all(c.spectrogram[:8] == PNG_SIGNATURE for c in chunks)


def test_stub_face_crops_still_differ_per_face():
    """Worst-case rollup needs the faces on one frame to score differently. If
    the PNG change had made every crop identical, the multi-face path would
    still "work" while testing nothing."""
    crops = StubFaceExtractor(force_faces=3).extract(b"image", frame_index=7)

    assert len({c.data for c in crops}) == 3


# --- checkpoint key rewriting -----------------------------------------------


def test_training_checkpoint_is_unwrapped_and_de_prefixed():
    """The published file is {epoch, state_dict, bce_best} with every key under
    a `module.` prefix from nn.DataParallel. Loading it raw matches nothing."""
    checkpoint = {
        "epoch": 37,
        "bce_best": 0.1639973577717397,
        "state_dict": {
            "module.encoder.conv_stem.weight": 1,
            "module.fc.weight": 2,
            "module.fc.bias": 3,
        },
    }

    sd = _state_dict_from(checkpoint)

    assert sd == {"encoder.conv_stem.weight": 1, "fc.weight": 2, "fc.bias": 3}
    assert not any(k.startswith("module.") for k in sd)
    assert "epoch" not in sd and "bce_best" not in sd


def test_a_bare_state_dict_passes_through_unharmed():
    """Not every checkpoint is wrapped. Unwrapping unconditionally would drop
    the tensors of a plain state dict, and stripping a prefix that is not there
    must be a no-op rather than a truncation."""
    bare = {"encoder.conv_stem.weight": 1, "fc.weight": 2}

    assert _state_dict_from(bare) == bare


def test_a_key_merely_containing_module_is_not_mangled():
    """Only a leading `module.` is the DataParallel wrapper. Stripping the
    substring anywhere would corrupt a legitimate nested name."""
    sd = {"encoder.module.block.weight": 1, "module.fc.weight": 2}

    assert _state_dict_from(sd) == {"encoder.module.block.weight": 1, "fc.weight": 2}


def test_digest_length_guard_documents_the_old_shape():
    """Pins what the bug looked like, so the regression is legible later: the
    old stubs returned exactly this, and it is exactly what cv2 could not
    decode."""
    old_shape = hashlib.sha256(b"image" + b"face" + bytes([0])).digest()

    assert len(old_shape) == 32
    assert old_shape[:8] != PNG_SIGNATURE
