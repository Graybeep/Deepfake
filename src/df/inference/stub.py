"""Deterministic placeholder scorer. NOT A DETECTOR.

It exists so the ingest -> preprocess -> score -> aggregate -> route pipeline can
be built, wired, and tested end to end before real weights land. It derives a
score from a hash of the input bytes: stable for the same input, uncorrelated
with whether the input is actually manipulated.

Its model_version_id carries `stub` and its ModelVersion carries
is_real_detector=False so no stored result can be mistaken for a real one.
"""
from __future__ import annotations

import hashlib

from df.inference.base import Detector, ModelVersion, Prediction

_FACE_VERSION = ModelVersion(
    model_version_id="face-stub-v0",
    architecture="none (hash placeholder standing in for EfficientNet)",
    modality="face",
    weights_sha256=None,
    calibration="none",
    is_real_detector=False,
)

_AUDIO_VERSION = ModelVersion(
    model_version_id="audio-stub-v0",
    architecture="none (hash placeholder standing in for EfficientNet)",
    modality="audio",
    weights_sha256=None,
    calibration="none",
    is_real_detector=False,
)


def _hash_score(data: bytes, salt: bytes) -> float:
    digest = hashlib.sha256(salt + data).digest()
    return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF * 100.0


class StubDetector(Detector):
    def __init__(self, version: ModelVersion, salt: bytes) -> None:
        self._version = version
        self._salt = salt

    @property
    def version(self) -> ModelVersion:
        return self._version

    def predict_batch(self, inputs: list[bytes]) -> list[Prediction]:
        out = []
        for blob in inputs:
            score = _hash_score(blob, self._salt)
            # Fixed mid-high confidence: the stub has no real detection quality
            # signal, and pretending otherwise would let confidence weighting in
            # aggregation look meaningful when it isn't.
            out.append(Prediction(score=round(score, 4), confidence=0.75))
        return out


def face_stub() -> StubDetector:
    return StubDetector(_FACE_VERSION, b"face")


def audio_stub() -> StubDetector:
    return StubDetector(_AUDIO_VERSION, b"audio")
