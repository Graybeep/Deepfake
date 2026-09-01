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


# --- detection is bounded, cropping is not -----------------------------------
#
# The bug: a 12.2 MP phone photo (4032x3024) sat in `preprocessing` for 85+
# seconds on the deployed service and then took the container down with SIGKILL,
# returning `undetermined`. `measured: yes` 2026-09-01, reproduced deliberately
# against the deploy. Haar ran on the full-resolution image, and the peak RSS of
# a single extract() on an 8.4 MP photo was +143.9 MB -- next to a resident B7,
# which is why the kernel killed the inference worker and made a preprocessing
# problem look like an inference one.
#
# `measured: yes` for the fix on the same 8.4 MP photo: +52.7 MB peak (2.7x
# less), 1.05s -> 0.35s, same three faces, boxes within ~2% of the full-res ones.
#
# # In-process. Live counterpart: upload a >8 MP image to a running stack and
# # watch /healthz stay up -- scripts/smoke_compose.py does not cover image size.


def _photo(width: int, height: int) -> bytes:
    """A synthetic image with real high-frequency detail at a known size.

    Not an upscaled small image: that was the first fixture here and it made the
    cap look like it destroyed detection (0 faces at 1600 against 2 junk boxes at
    full size), because upscaling leaves no genuine detail for a texture cascade
    at any scale. The conclusion would have been exactly backwards.
    """
    import cv2
    import numpy as np

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_detection_runs_on_a_bounded_image(monkeypatch):
    """The cap must actually reach cv2, not merely exist in config."""
    import cv2

    from df.config import Settings
    from df.pipelines import extract as ex

    monkeypatch.setenv("DF_DETECT_MAX_SIDE", "800")
    monkeypatch.setattr(ex, "settings", Settings())

    seen: list[tuple[int, int]] = []
    real = cv2.CascadeClassifier.detectMultiScale3

    def spy(self, image, **kwargs):
        seen.append((image.shape[1], image.shape[0]))
        return real(self, image, **kwargs)

    monkeypatch.setattr(cv2.CascadeClassifier, "detectMultiScale3", spy)
    ex.OpenCVFaceExtractor().extract(_photo(3200, 2400))

    assert seen, "detectMultiScale3 was never called"
    assert max(seen[0]) == 800, f"detection ran at {seen[0]}, not capped to 800"


def test_the_cap_can_be_switched_off(monkeypatch):
    """0 restores full-resolution detection, so the change is reversible on a
    running deployment without a rebuild."""
    import cv2

    from df.config import Settings
    from df.pipelines import extract as ex

    monkeypatch.setenv("DF_DETECT_MAX_SIDE", "0")
    monkeypatch.setattr(ex, "settings", Settings())

    seen: list[tuple[int, int]] = []
    real = cv2.CascadeClassifier.detectMultiScale3

    def spy(self, image, **kwargs):
        seen.append((image.shape[1], image.shape[0]))
        return real(self, image, **kwargs)

    monkeypatch.setattr(cv2.CascadeClassifier, "detectMultiScale3", spy)
    ex.OpenCVFaceExtractor().extract(_photo(2000, 1500))

    assert seen[0] == (2000, 1500)


def test_a_small_image_is_never_upscaled(monkeypatch):
    """The cap is a ceiling, not a target. Enlarging a small image would invent
    detail and slow down the common case for nothing."""
    import cv2

    from df.config import Settings
    from df.pipelines import extract as ex

    monkeypatch.setenv("DF_DETECT_MAX_SIDE", "1600")
    monkeypatch.setattr(ex, "settings", Settings())

    seen: list[tuple[int, int]] = []
    real = cv2.CascadeClassifier.detectMultiScale3

    def spy(self, image, **kwargs):
        seen.append((image.shape[1], image.shape[0]))
        return real(self, image, **kwargs)

    monkeypatch.setattr(cv2.CascadeClassifier, "detectMultiScale3", spy)
    ex.OpenCVFaceExtractor().extract(_photo(640, 480))

    assert seen[0] == (640, 480)


def test_boxes_are_mapped_back_to_native_pixels(monkeypatch):
    """The geometry seam, which this file has already got wrong twice.

    Detection happens on a downscaled copy, so every box comes back in
    downscaled coordinates and MUST be multiplied out before cropping. If it is
    not, the crop is taken from the wrong part of the image -- and nothing
    downstream can tell, because a wrong crop is still a valid image that still
    scores. The detection would be silently reading the wrong face.

    The cascade is stubbed to return one known box so the arithmetic is checked
    exactly rather than through whatever a real detector happens to find.
    """
    import cv2
    import numpy as np

    from df.config import Settings
    from df.pipelines import extract as ex

    monkeypatch.setenv("DF_DETECT_MAX_SIDE", "800")
    monkeypatch.setattr(ex, "settings", Settings())

    # 3200x2400 capped at 800 -> scale 0.25, so a box at 100,50,60x60 on the
    # detection image describes 400,200,240x240 in the original.
    def fake(self, image, **kwargs):
        assert max(image.shape[:2]) == 800
        return (np.array([[100, 50, 60, 60]]), None, np.array([5.0]))

    monkeypatch.setattr(cv2.CascadeClassifier, "detectMultiScale3", fake)
    crops = ex.OpenCVFaceExtractor().extract(_photo(3200, 2400))

    assert len(crops) == 1
    assert crops[0].bbox == (400, 200, 240, 240)

    # and the pixels handed to the model must match the box that was recorded
    decoded = cv2.imdecode(np.frombuffer(crops[0].data, np.uint8), cv2.IMREAD_COLOR)
    assert (decoded.shape[1], decoded.shape[0]) == (240, 240)


def test_a_box_at_the_frame_edge_stays_inside_the_image(monkeypatch):
    """Rounding outward can push x+w past the width. numpy slicing does not
    raise for that -- it returns a short crop -- so the bbox on the audit row
    would describe pixels that were never scored."""
    import cv2
    import numpy as np

    from df.config import Settings
    from df.pipelines import extract as ex

    monkeypatch.setenv("DF_DETECT_MAX_SIDE", "800")
    monkeypatch.setattr(ex, "settings", Settings())

    def fake(self, image, **kwargs):
        h, w = image.shape[:2]
        # flush against the bottom-right corner of the detection image
        return (np.array([[w - 31, h - 31, 31, 31]]), None, np.array([5.0]))

    monkeypatch.setattr(cv2.CascadeClassifier, "detectMultiScale3", fake)
    # 801x840, NOT 3200x2400. The first version of this test used 3200x2400,
    # where 3200/800 divides evenly: the flush box maps back to exactly the
    # boundary, the clamp is never reached, and the test passed against code
    # with no clamp at all. The mutation harness reported NO-OP and that is how
    # it was found. Sweeping for real overflow cases gives 801x840 with a 31 px
    # box, which lands 1 px outside.
    crops = ex.OpenCVFaceExtractor().extract(_photo(801, 840))

    x, y, w, h = crops[0].bbox
    assert x + w <= 801 and y + h <= 840, f"box {crops[0].bbox} escapes the image"

    decoded = cv2.imdecode(np.frombuffer(crops[0].data, np.uint8), cv2.IMREAD_COLOR)
    assert (decoded.shape[1], decoded.shape[0]) == (w, h), "crop does not match its bbox"


# --- detection fallback: glasses and bad lighting ---------------------------
#
# Prompted by a real report from a phone: normal photos worked, photos with
# glasses and photos in bad lighting returned "could not analyse this". Glasses
# occlude the eye region, which is a primary Haar feature; bad lighting flattens
# the local contrast the cascade measures.
#
# `measured: yes` 2026-09-01 over 23 hard cases (public-domain portraits with
# glasses, plus controlled side-lit, low-light-with-noise and harsh-shadow
# variants):
#     plain only ..... 20/23
#     CLAHE always ... 22/23   fixes 3, LOSES 1
#     cascading ...... 23/23   fixes 3, loses 0
#
# The middle row is why the order matters: applying CLAHE unconditionally breaks
# a dark noisy portrait that the plain pass handles (1 face -> 0). Trying the
# plain image FIRST makes the gain non-regressive by construction.
#
# # In-process. Live counterpart: upload a backlit or bespectacled photo to a
# # running stack and check it does not come back `undetermined`.


def test_the_plain_pass_is_tried_first(monkeypatch):
    """Non-regression by construction. If an enhanced pass ran first, it could
    replace a detection the shipped path already made -- measured to happen on
    at least one real image."""
    import cv2

    from df.pipelines import extract as ex

    # Recording the cascade FILENAME is not enough: the plain and the enhanced
    # stage use the same file, so a mutation that reorders them leaves the
    # filename sequence identical. The mutation harness reported NO-OP for
    # exactly that, on this test. What distinguishes them is the PIXELS.
    seen: list[object] = []

    class Recording:
        def __init__(self, path):
            self._name = path.replace("\\", "/").rsplit("/", 1)[-1]

        def detectMultiScale3(self, image, **kw):
            seen.append((self._name, image.copy()))
            return (np.array([[1, 2, 30, 30]]), None, np.array([5.0]))

    monkeypatch.setattr(cv2, "CascadeClassifier", Recording)
    blob = _png()
    ex.OpenCVFaceExtractor().extract(blob)

    assert seen, "no detection attempt was made"
    name, first_image = seen[0]
    assert name == "haarcascade_frontalface_default.xml", f"first cascade was {name}"

    plain = cv2.cvtColor(cv2.imdecode(np.frombuffer(blob, np.uint8),
                                      cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    assert np.array_equal(first_image, plain), (
        "the first detection attempt received a CONTRAST-ENHANCED image; the "
        "plain image must be tried first or the fallback can lose a detection "
        "the shipped path already made"
    )


def test_later_stages_run_only_when_earlier_ones_find_nothing(monkeypatch):
    """The fallback must cost nothing on the common path, and must actually
    engage when the primary is empty."""
    import cv2

    from df.pipelines import extract as ex

    calls: list[str] = []
    real = cv2.CascadeClassifier

    class Empty:
        def __init__(self, path):
            self._name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

        def detectMultiScale3(self, image, **kw):
            calls.append(self._name)
            return (np.empty((0, 4), dtype=int), None, np.empty((0,)))

    monkeypatch.setattr(cv2, "CascadeClassifier", Empty)
    crops = ex.OpenCVFaceExtractor().extract(_png())

    assert crops == []
    assert len(calls) == len(ex._DETECT_STAGES), (
        f"expected every stage to be tried when all are empty, got {calls}"
    )
    assert "haarcascade_frontalface_alt.xml" in calls


def test_the_fallback_can_be_switched_off(monkeypatch):
    """So the single-pass behaviour can be measured against it."""
    import cv2

    from df.config import Settings
    from df.pipelines import extract as ex

    monkeypatch.setenv("DF_DETECT_FALLBACK", "0")
    monkeypatch.setattr(ex, "settings", Settings())

    calls: list[str] = []
    real = cv2.CascadeClassifier

    class Empty:
        def __init__(self, path):
            calls.append(path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1])

        def detectMultiScale3(self, image, **kw):
            return (np.empty((0, 4), dtype=int), None, np.empty((0,)))

    monkeypatch.setattr(cv2, "CascadeClassifier", Empty)
    ex.OpenCVFaceExtractor().extract(_png())

    assert len(calls) == 1, f"fallback disabled but {len(calls)} cascades were tried"


def test_all_faces_in_a_frame_come_from_one_cascade(monkeypatch):
    """The confidence gate is RELATIVE, and levelWeights are unbounded cascade
    reject levels whose scale differs between cascades. Mixing detections from
    two cascades would make the ratio compare incomparable numbers. The first
    successful stage returning immediately is what guarantees it cannot happen.
    """
    import cv2

    from df.pipelines import extract as ex

    used: list[str] = []
    real = cv2.CascadeClassifier

    class OnlyAltFinds:
        def __init__(self, path):
            self._name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

        def detectMultiScale3(self, image, **kw):
            if self._name == "haarcascade_frontalface_alt.xml":
                used.append(self._name)
                return (np.array([[1, 1, 40, 40], [50, 50, 40, 40]]), None,
                        np.array([6.0, 3.0]))
            return (np.empty((0, 4), dtype=int), None, np.empty((0,)))

    monkeypatch.setattr(cv2, "CascadeClassifier", OnlyAltFinds)
    crops = ex.OpenCVFaceExtractor().extract(_png())

    assert len(crops) == 2
    assert used == ["haarcascade_frontalface_alt.xml"], "more than one cascade contributed"


def test_cascades_are_not_cached_across_calls(monkeypatch):
    """A module-level cascade cache is the obvious optimisation and it is wrong:
    the suite patches cv2.CascadeClassifier itself, so a cache hands one test the
    stub installed by another. Six tests passed alone and failed together when
    that cache existed."""
    import cv2

    from df.pipelines import extract as ex

    constructed: list[int] = []

    def make_stub(boxes):
        class Stub:
            def __init__(self, path):
                constructed.append(1)

            def detectMultiScale3(self, image, **kw):
                return (np.array(boxes), None, np.array([5.0] * len(boxes)))
        return Stub

    monkeypatch.setattr(cv2, "CascadeClassifier", make_stub([[1, 1, 30, 30]]))
    first = ex.OpenCVFaceExtractor().extract(_png())
    monkeypatch.setattr(cv2, "CascadeClassifier", make_stub([[2, 2, 40, 40], [9, 9, 40, 40]]))
    second = ex.OpenCVFaceExtractor().extract(_png())

    assert len(first) == 1 and len(second) == 2, (
        "the second call reused the first call's cascade -- cross-call state"
    )
