"""The detection-confidence gate: drop junk before the model, and record it.

Why gate rather than reweight, since the reweighting option was seriously
considered and rejected:

A non-face region entering the model produces an arbitrary number. There is no
principled way to combine an arbitrary number with a real one -- averaging it is
not better than taking the max of it, only less alarming. So the fix belongs
where the junk enters, not three layers downstream.

And worst-case rollup over survivors is deliberately preserved. One manipulated
face is what makes an image manipulated; a confidence-weighted mean across faces
would drag a swapped face's score toward the crowd in a group photo, trading a
visible false positive for invisible false negatives on exactly the case this
detector exists for. Nobody would notice in a demo, which is what makes it worse.

Measured 2026-08-31 on a public-domain portrait, which is the whole evidence base
for the threshold: Haar returned the real face at confidence 0.97 (scoring 0.54,
authentic) plus artefacts at 0.32 and 0.08. The 0.32 artefact scored 55.79 and
set the verdict for the entire image to `uncertain` through worst-case rollup.

# In-process. Live counterpart: the same portrait through the real pipeline under
# docker-compose.realpipeline.yml, which is how the artefact was found and how
# the gate + recording were confirmed (floor 0.5 -> discarded and recorded,
# verdict corrected to likely_authentic at 0.54).
"""
from __future__ import annotations

import pytest

from df.pipelines.extract import FaceCrop
from df.workers.cpu_preprocess import _gate_detections


def _crop(face_index: int, confidence: float) -> FaceCrop:
    return FaceCrop(
        frame_index=0,
        face_index=face_index,
        data=b"crop",
        confidence=confidence,
        bbox=(10, 10, 120, 120),
    )


def test_sub_threshold_detections_never_reach_the_model(monkeypatch):
    """The point of gating: junk is excluded before inference, not after."""
    monkeypatch.setenv("DF_MIN_DETECTION_CONFIDENCE", "0.5")
    from df.config import Settings
    monkeypatch.setattr("df.workers.cpu_preprocess.settings", Settings())

    discarded: list[dict] = []
    kept = _gate_detections(
        [_crop(0, 0.97), _crop(1, 0.32), _crop(2, 0.08)], discarded, frame_index=0
    )

    assert [c.face_index for c in kept] == [0]
    assert [d["face_index"] for d in discarded] == [1, 2]


def test_what_is_gated_is_recorded_not_silently_dropped(monkeypatch):
    """This gate existed before, inside the extractor, and was removed on
    2026-08-30 precisely because an extraction-time drop left no row, no count
    and nothing for a dispute to read. It comes back only with a record.

    job_items cannot hold these: `score` is NOT NULL and these were never
    scored, so a row would have to invent a score for a crop the model never
    saw. The preprocess.complete event is the permanent trace.
    """
    monkeypatch.setenv("DF_MIN_DETECTION_CONFIDENCE", "0.5")
    from df.config import Settings
    monkeypatch.setattr("df.workers.cpu_preprocess.settings", Settings())

    discarded: list[dict] = []
    _gate_detections([_crop(1, 0.316)], discarded, frame_index=7)

    assert len(discarded) == 1
    rec = discarded[0]
    assert rec["face_index"] == 1
    assert rec["frame_index"] == 7
    assert rec["confidence"] == pytest.approx(0.316)
    # Geometry travels too, so a size-bucketed threshold can later be fitted
    # against what was rejected as well as what was kept.
    assert (rec["face_w"], rec["face_h"]) == (120, 120)


def test_a_confident_detection_is_kept__the_positive_control(monkeypatch):
    """Without this, a gate that rejected everything would pass the checks
    above and quietly turn every job into `undetermined`."""
    monkeypatch.setenv("DF_MIN_DETECTION_CONFIDENCE", "0.5")
    from df.config import Settings
    monkeypatch.setattr("df.workers.cpu_preprocess.settings", Settings())

    discarded: list[dict] = []
    kept = _gate_detections([_crop(0, 0.97)], discarded, frame_index=0)

    assert len(kept) == 1
    assert discarded == []


def test_the_shipped_floor_does_not_catch_the_observed_artefact(monkeypatch):
    """Pins an uncomfortable fact rather than hiding it.

    The artefact that set the verdict measured 0.316, and the shipped default is
    0.3 -- so it survives the gate. Raising the default to catch it would be
    fitting a threshold to a single example, which is how an untuned constant
    acquires the appearance of a tuned one.

    The real repair is a detector that returns an actual detection probability
    (RetinaFace/SCRFD) instead of Haar's unbounded cascade reject level. Until
    then this test documents the gap, and `face_evidence` surfaces the per-face
    confidences so a reader can see a marginal detection for what it is.

    If the default changes, change this test in the same commit and say what
    evidence justified it.
    """
    monkeypatch.delenv("DF_MIN_DETECTION_CONFIDENCE", raising=False)
    from df.config import Settings
    settings = Settings()
    monkeypatch.setattr("df.workers.cpu_preprocess.settings", settings)

    assert settings.min_detection_confidence == 0.3

    discarded: list[dict] = []
    kept = _gate_detections([_crop(0, 0.97), _crop(1, 0.316)], discarded, frame_index=0)

    assert len(kept) == 2, "0.316 is above the 0.3 floor -- it is NOT gated"
    assert discarded == []
