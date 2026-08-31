"""The detection gate: drop junk before the model, record it, never empty the set.

Why gate rather than reweight, since reweighting was seriously proposed and
rejected: a non-face region entering the model produces an arbitrary number, and
there is no principled way to combine an arbitrary number with a real one --
averaging it is not better than maxing it, only less alarming. The fix belongs
where the junk enters.

And worst-case rollup over survivors is preserved. One manipulated face is what
makes an image manipulated; a confidence-weighted mean across faces would drag a
swapped face's score toward the crowd in a group photo, trading a visible false
positive for invisible false negatives on exactly the case the tool exists for.

Why RELATIVE rather than an absolute floor: the confidence is a squashed
`levelWeights` cascade reject level, not a probability, so no absolute threshold
is more justified than any other. A ratio is invariant to that untrusted scale,
and it cannot empty a non-empty set -- the best detection is always ratio 1.0.

Evidence base, in full: one public-domain portrait, 2026-08-31. Real face 0.968
(B7 scored it 0.54, authentic); artefacts 0.316 and 0.075. The 0.316 artefact
scored 55.79 and set the whole image to `uncertain` through worst-case rollup.

# In-process. Live counterpart: the same portrait through
# docker-compose.realpipeline.yml, which is how the artefact was found.
"""
from __future__ import annotations

import pytest

from df.config import Settings
from df.pipelines.extract import FaceCrop
from df.workers.cpu_preprocess import _gate_detections

# Both regimes are exercised explicitly. A test that pins only the shipped
# default documents behaviour nobody may run; a test that pins only an override
# documents behaviour nobody ships. Parametrising says which is which.
RATIOS = [0.4, 0.5]


@pytest.fixture
def gate(monkeypatch):
    def _configure(ratio: float | None = None):
        if ratio is None:
            monkeypatch.delenv("DF_DETECTION_CONFIDENCE_RATIO", raising=False)
        else:
            monkeypatch.setenv("DF_DETECTION_CONFIDENCE_RATIO", str(ratio))
        settings = Settings()
        monkeypatch.setattr("df.workers.cpu_preprocess.settings", settings)
        return settings
    return _configure


def _crop(face_index: int, confidence: float) -> FaceCrop:
    return FaceCrop(
        frame_index=0, face_index=face_index, data=b"crop",
        confidence=confidence, bbox=(10, 10, 120, 120),
    )


# --- the property that removes the zero-survivor failure mode ---------------


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("confidences", [
    [0.44],                       # a lone marginal detection
    [0.02],                       # a lone terrible detection
    [0.31, 0.30],                 # all weak, none stands out
    [0.968, 0.316, 0.075],        # the measured portrait
    [0.5, 0.5, 0.5],              # a tie
])
def test_gating_never_empties_a_non_empty_detection_set(gate, ratio, confidences):
    """The failure mode this design removes by construction.

    A judge in bad light, slightly turned, wearing glasses: Haar's weight drops,
    one face is detected at 0.44, and an absolute floor of 0.5 would gate it,
    leaving zero survivors and an `undetermined` verdict on a face that is
    plainly there. With a ratio the best detection is always 1.0, so at least one
    always survives -- a degraded answer instead of no answer, guaranteed
    structurally rather than by a fallback branch that might itself be wrong.
    """
    gate(ratio)
    discarded: list[dict] = []

    kept = _gate_detections(
        [_crop(i, c) for i, c in enumerate(confidences)], discarded, frame_index=0
    )

    assert kept, f"gating emptied the set at ratio={ratio} for {confidences}"
    # And the strongest detection is always among the survivors.
    assert max(c.confidence for c in kept) == max(confidences)


@pytest.mark.parametrize("ratio", RATIOS)
def test_an_empty_detection_set_stays_empty(gate, ratio):
    """0 faces routes to `undetermined` in production. The gate must not invent
    a detection to avoid that, only refrain from creating the case itself."""
    gate(ratio)
    discarded: list[dict] = []

    assert _gate_detections([], discarded, frame_index=0) == []
    assert discarded == []


# --- the measured case, at both regimes ------------------------------------


@pytest.mark.parametrize("ratio", RATIOS)
def test_the_measured_artefact_is_gated_at_both_shipped_ratios(gate, ratio):
    """0.316 / 0.968 = 0.33, below both 0.4 and 0.5, so it gates either way.

    This is the case that made a clean portrait come back `uncertain`. Unlike an
    absolute floor -- where 0.3 kept it and 0.5 dropped it, and neither number
    had a justification -- the ratio gives the same answer across the plausible
    range, which is the point of using a scale-invariant comparison.
    """
    gate(ratio)
    discarded: list[dict] = []

    kept = _gate_detections(
        [_crop(0, 0.968), _crop(1, 0.316), _crop(2, 0.075)], discarded, frame_index=0
    )

    assert [c.face_index for c in kept] == [0]
    assert [d["face_index"] for d in discarded] == [1, 2]


@pytest.mark.parametrize("ratio", RATIOS)
def test_a_second_strong_detection_survives__the_positive_control(gate, ratio):
    """Without this, a gate that kept only the single best detection would pass
    every check above -- and would silently discard the second face of a real
    two-person photo, which is the case worst-case rollup exists for."""
    gate(ratio)
    discarded: list[dict] = []

    kept = _gate_detections([_crop(0, 0.95), _crop(1, 0.80)], discarded, frame_index=0)

    assert [c.face_index for c in kept] == [0, 1]
    assert discarded == []


# --- what is gated must be recorded ----------------------------------------


def test_what_is_gated_is_recorded_not_silently_dropped(gate):
    """This gate lived inside the extractor until 2026-08-30 and was removed
    because an extraction-time drop left no row, no count, and nothing for a
    dispute to read. It returns only with a record.

    job_items cannot hold these: `score` is NOT NULL and they were never scored,
    so a row would have to invent a score for a crop the model never saw. The
    preprocess.complete event is the permanent trace.
    """
    gate(0.4)
    discarded: list[dict] = []

    _gate_detections([_crop(0, 0.968), _crop(1, 0.316)], discarded, frame_index=7)

    assert len(discarded) == 1
    rec = discarded[0]
    assert rec["face_index"] == 1
    assert rec["frame_index"] == 7
    assert rec["confidence"] == pytest.approx(0.316)
    # The ratio and the frame's best are recorded too. Without them the raw
    # confidence is uninterpretable later -- 0.316 means nothing unless you know
    # what it was compared against.
    assert rec["relative_to_best"] == pytest.approx(0.316 / 0.968, rel=1e-3)
    assert rec["best_in_frame"] == pytest.approx(0.968)
    # Geometry travels, so a size-bucketed threshold can later be fitted against
    # what was rejected as well as what was kept.
    assert (rec["face_w"], rec["face_h"]) == (120, 120)


def test_zero_confidence_everywhere_keeps_everything(gate):
    """A ratio is undefined when the best detection is 0. Keeping all is the
    honest degradation; inventing an ordering would be worse, and dividing by
    zero would take the worker down on a frame it should merely handle badly."""
    gate(0.4)
    discarded: list[dict] = []

    kept = _gate_detections([_crop(0, 0.0), _crop(1, 0.0)], discarded, frame_index=0)

    assert len(kept) == 2
    assert discarded == []


def test_the_shipped_default_is_the_ratio_not_an_absolute_floor(gate):
    """Guards against a revert to an absolute threshold. If this changes, say in
    the same commit what evidence justified a number on an uncalibrated scale."""
    settings = gate(None)

    assert settings.detection_confidence_ratio == 0.4
    assert not hasattr(settings, "min_detection_confidence")
