"""The real (OpenCV) face extractor's crop path.

Nothing covered this branch, which is how it came to raise on every detected
face: `FACE_INPUT_SIZE` changed from a `(224, 224)` tuple to the int `380` when
the torch backend was rewritten, and `cv2.resize` needs a sequence. The CPU
worker runs the stub extractor, the weights overlay switches only the GPU
worker, so the real path never ran and a constant changing type across a module
boundary went unnoticed.

The cascade is stubbed rather than fed a photograph: this project has no image
containing a face a Haar cascade will actually find, and the detection quality
of Haar is not what these tests are about. What they pin is what the extractor
does with a detection once it has one.

# In-process. The live counterpart is a real video through compose with
# DF_INFERENCE_BACKEND=torch on the CPU worker; that covers sampling and the
# no-face route, and cannot cover the crop path for the same missing-photo
# reason.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from df.pipelines.extract import OpenCVFaceExtractor, _haar_confidence

# Imported hard, not via importorskip. opencv is a declared dev dependency
# precisely so these run; skipping on a missing import would turn the module
# green-by-absence, which is the failure mode that let the crop path stay broken
# in the first place. If cv2 is missing, that is a broken environment and this
# should say so loudly.


def _png(h: int = 300, w: int = 400) -> bytes:
    img = np.random.default_rng(0).integers(0, 255, (h, w, 3)).astype(np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


class _Cascade:
    def __init__(self, boxes, weights):
        self._boxes, self._weights = np.array(boxes), np.array(weights)

    def detectMultiScale3(self, *a, **k):
        return (self._boxes, None, self._weights)


@pytest.fixture
def stub_cascade(monkeypatch):
    def _install(boxes, weights):
        monkeypatch.setattr(cv2, "CascadeClassifier", lambda *a, **k: _Cascade(boxes, weights))
    return _install


def test_a_detected_face_produces_a_decodable_crop(stub_cascade):
    """The regression. This raised for every detection."""
    stub_cascade([[10, 20, 120, 60]], [9.0])

    crops = OpenCVFaceExtractor().extract(_png(), frame_index=7)

    assert len(crops) == 1
    assert cv2.imdecode(np.frombuffer(crops[0].data, np.uint8), cv2.IMREAD_COLOR) is not None
    assert crops[0].frame_index == 7
    assert crops[0].bbox == (10, 20, 120, 60)


def test_the_crop_keeps_its_native_aspect_ratio(stub_cascade):
    """Geometry belongs to the detector, which does an isotropic resize and pad
    to match upstream preprocessing. Squaring the crop here would hand that step
    an already-distorted image and silently make it a no-op."""
    stub_cascade([[10, 20, 120, 60]], [9.0])

    crop = OpenCVFaceExtractor().extract(_png())[0]
    decoded = cv2.imdecode(np.frombuffer(crop.data, np.uint8), cv2.IMREAD_COLOR)

    assert decoded.shape[:2] == (60, 120)  # h, w -- as detected, not squared


def test_geometry_is_recorded_for_every_crop(stub_cascade):
    """migration 006 stores face_w/face_h so a size-bucketed threshold can be
    fitted later. The bbox is the source of those, so it must be the DETECTED
    box in source-frame pixels, not the crop's own dimensions after any
    resizing."""
    stub_cascade([[5, 6, 70, 90], [100, 110, 48, 52]], [9.0, 4.0])

    crops = OpenCVFaceExtractor().extract(_png())

    assert [c.bbox for c in crops] == [(5, 6, 70, 90), (100, 110, 48, 52)]
    assert [c.face_index for c in crops] == [0, 1]


def test_low_confidence_faces_are_kept_not_silently_dropped(stub_cascade):
    """This used to filter below 0.3 inside the extractor, so weak detections
    never became items at all.

    CLAUDE.md's rule is that low-confidence items are "still recorded in
    job_items -- dropped, not deleted, so the audit trail shows what was
    ignored". A filter here broke that invisibly: an extractor-level drop leaves
    no row, no count, and nothing for a dispute to look at. The drop now happens
    in aggregation, where it is recorded.
    """
    stub_cascade([[10, 10, 50, 50], [80, 80, 50, 50]], [0.5, 9.0])

    crops = OpenCVFaceExtractor().extract(_png())

    assert len(crops) == 2
    assert crops[0].confidence == pytest.approx(0.05)
    assert crops[1].confidence == pytest.approx(0.9)


def test_an_undecodable_image_yields_no_faces_rather_than_raising(stub_cascade):
    """The CPU worker parses untrusted media. A malformed frame must return
    nothing and let the job route to `undetermined`, not take the worker down."""
    stub_cascade([[10, 10, 50, 50]], [9.0])

    assert OpenCVFaceExtractor().extract(b"not an image") == []


def test_a_zero_area_box_is_skipped(stub_cascade):
    """A degenerate box slices to an empty array, which cv2.imencode cannot
    encode. Skipping beats raising on one bad detection in a long video."""
    stub_cascade([[10, 10, 0, 40], [50, 50, 60, 60]], [9.0, 9.0])

    assert len(OpenCVFaceExtractor().extract(_png())) == 1


# --- the confidence scale ---------------------------------------------------


def test_haar_confidence_is_bounded_and_monotone():
    """It is an arbitrary squash of an unbounded cascade reject level, not a
    probability. Monotonicity is the only property aggregation relies on, and
    the bounds are what stop one detection dominating a weighted mean."""
    assert _haar_confidence(-5.0) == 0.0
    assert _haar_confidence(0.0) == 0.0
    assert _haar_confidence(1000.0) == 1.0
    assert _haar_confidence(3.0) < _haar_confidence(7.0)


def test_haar_confidence_stays_within_the_aggregation_weight_range():
    """Aggregation multiplies scores by this. A value outside 0-1 would not
    fail, it would silently reweight the mean."""
    for w in (-100.0, -1.0, 0.5, 2.0, 9.99, 10.0, 10.01, 1e6):
        assert 0.0 <= _haar_confidence(w) <= 1.0
