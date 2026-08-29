"""The scoring loop that turns a labelled video set into calibration input.

Covers `rows_for_video` against injected fakes. The container smoke run covers
the wiring around it -- the real model loading, metadata parsing, and the skip
paths for unreadable and unlabelled videos -- but it cannot cover the happy path
without a video containing a face a Haar cascade will actually find, and this
project has no such file.

# In-process only. The real end-to-end run is blocked on the DFDC validation
# split, which needs an AWS account and accepted terms; when that lands, the
# first real invocation is itself the live probe.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from df.inference.base import Prediction  # noqa: E402
from df.pipelines.extract import FaceCrop, Frame  # noqa: E402


def _load_rows_for_video():
    """Import the script without executing its CLI.

    It lives in scripts/ rather than src/, so it is imported by path. Kept in a
    helper so the import error is legible if the file moves.
    """
    from extract_logits import rows_for_video

    return rows_for_video


class FakeSampler:
    def __init__(self, n: int = 4) -> None:
        self.n = n

    def sample(self, blob: bytes) -> list[Frame]:
        return [Frame(index=i, data=b"frame-%d" % i, timestamp_s=float(i)) for i in range(self.n)]


class FakeExtractor:
    """`per_frame` faces on every frame, with distinguishable confidences."""

    def __init__(self, per_frame: int = 1) -> None:
        self.per_frame = per_frame

    def extract(self, image_bytes: bytes, frame_index: int = 0) -> list[FaceCrop]:
        return [
            FaceCrop(
                frame_index=frame_index,
                face_index=i,
                data=b"crop-%d-%d" % (frame_index, i),
                confidence=0.6 + 0.1 * i,
            )
            for i in range(self.per_frame)
        ]


class FakeDetector:
    def __init__(self, logit: float | None = 2.5) -> None:
        self.logit = logit
        self.batches: list[list[bytes]] = []

    def predict_batch(self, inputs: list[bytes]) -> list[Prediction]:
        self.batches.append(list(inputs))
        return [Prediction(score=90.0, confidence=1.0, logit=self.logit) for _ in inputs]


def test_one_row_per_crop_carrying_the_videos_label():
    rows_for_video = _load_rows_for_video()

    rows = rows_for_video(
        b"video", 1, "a.mp4",
        sampler=FakeSampler(3), extractor=FakeExtractor(1),
        detector=FakeDetector(), max_faces=8,
    )

    assert len(rows) == 3
    assert {r["label"] for r in rows} == {1}
    assert [r["frame_index"] for r in rows] == [0, 1, 2]
    assert all(r["video"] == "a.mp4" for r in rows)


def test_the_logit_is_what_is_recorded_not_the_score():
    """A temperature can only be fitted from logits. `score` has already had a
    sigmoid and a temperature applied and cannot be un-applied, so recording it
    here would silently make the fit meaningless."""
    rows_for_video = _load_rows_for_video()

    rows = rows_for_video(
        b"v", 0, "a.mp4",
        sampler=FakeSampler(1), extractor=FakeExtractor(1),
        detector=FakeDetector(logit=-1.75), max_faces=8,
    )

    assert rows[0]["logit"] == -1.75
    assert "score" not in rows[0]


def test_a_detector_with_no_logit_is_refused():
    """The stub reports logit=None because its score is a hash. Fitting on that
    would calibrate a hash function, so it must fail loudly rather than write
    nulls into the calibration set."""
    rows_for_video = _load_rows_for_video()

    with pytest.raises(ValueError, match="no logit"):
        rows_for_video(
            b"v", 1, "a.mp4",
            sampler=FakeSampler(1), extractor=FakeExtractor(1),
            detector=FakeDetector(logit=None), max_faces=8,
        )


def test_a_video_with_no_detected_face_yields_nothing():
    """0 faces routes to `undetermined` in production rather than a score, so it
    has no place in a set used to calibrate scores."""
    rows_for_video = _load_rows_for_video()

    assert rows_for_video(
        b"v", 1, "a.mp4",
        sampler=FakeSampler(3), extractor=FakeExtractor(0),
        detector=FakeDetector(), max_faces=8,
    ) == []


def test_the_face_cap_bounds_one_videos_influence__with_a_positive_control():
    """Without a cap, a long clip with a reliably-detected face contributes far
    more rows than a short one and dominates the fit for no principled reason.

    The uncapped case is the control: it shows the cap is what limits the count,
    not the fake only ever producing two crops.
    """
    rows_for_video = _load_rows_for_video()
    kwargs = dict(sampler=FakeSampler(10), extractor=FakeExtractor(2),
                  detector=FakeDetector())

    capped = rows_for_video(b"v", 1, "a.mp4", max_faces=5, **kwargs)
    uncapped = rows_for_video(b"v", 1, "a.mp4", max_faces=100, **kwargs)

    assert len(capped) == 5
    assert len(uncapped) == 20


def test_every_crop_is_scored_in_one_batch():
    """Scoring crop-by-crop would work and would be far slower over 4,000
    clips. Pinning the batching also pins that no crop is silently dropped
    between extraction and scoring."""
    rows_for_video = _load_rows_for_video()
    detector = FakeDetector()

    rows = rows_for_video(
        b"v", 1, "a.mp4",
        sampler=FakeSampler(4), extractor=FakeExtractor(1),
        detector=detector, max_faces=8,
    )

    assert len(detector.batches) == 1
    assert len(detector.batches[0]) == len(rows) == 4


def test_detection_confidence_travels_with_the_row():
    """Kept so a later pass can weight or filter the calibration set by
    detection quality. The aggregation weights already use it, and a calibration
    fitted over crops the pipeline would have dropped describes a distribution
    production never scores."""
    rows_for_video = _load_rows_for_video()

    rows = rows_for_video(
        b"v", 1, "a.mp4",
        sampler=FakeSampler(1), extractor=FakeExtractor(2),
        detector=FakeDetector(), max_faces=8,
    )

    assert [r["detection_confidence"] for r in rows] == [0.6, 0.7]
