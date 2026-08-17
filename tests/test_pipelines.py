"""Pipeline behaviour around the two rules that must never be relaxed:
0 faces => undetermined, and >1 face => worst-case rollup.
"""
from __future__ import annotations

from df.bands import ResultClass
from df.inference.base import VALIDATION_PLACEHOLDER, ModelVersion, Prediction
from df.inference.stub import audio_stub, face_stub
from df.pipelines import audio as audio_pipeline
from df.pipelines import image as image_pipeline
from df.pipelines import video as video_pipeline
from df.pipelines.extract import StubAudioChunker, StubFaceExtractor, StubFrameSampler


class FixedDetector:
    """Returns a preset score per call, so rollup/aggregation can be asserted."""

    def __init__(self, scores: list[float], confidence: float = 1.0) -> None:
        self.scores = scores
        self.confidence = confidence
        self.calls = 0

    @property
    def version(self) -> ModelVersion:
        return ModelVersion("test-fixed-v0", "test", "face", None, "none", False,
                            VALIDATION_PLACEHOLDER)

    def predict_batch(self, inputs: list) -> list[Prediction]:
        out = []
        for _ in inputs:
            out.append(Prediction(self.scores[self.calls % len(self.scores)], self.confidence))
            self.calls += 1
        return out


# --- 0 faces => undetermined ------------------------------------------------


def test_video_with_no_faces_is_undetermined():
    result = video_pipeline.run(
        b"video",
        sampler=StubFrameSampler(n_frames=10),
        extractor=StubFaceExtractor(force_faces=0),
        detector=face_stub(),
    )

    assert result.routing.result_class is ResultClass.UNDETERMINED
    assert result.score is None
    assert result.face_count == 0


def test_image_with_no_faces_is_undetermined():
    result = image_pipeline.run(
        b"image", extractor=StubFaceExtractor(force_faces=0), detector=face_stub()
    )

    assert result.routing.result_class is ResultClass.UNDETERMINED
    assert result.score is None


def test_video_with_no_decodable_frames_is_undetermined():
    result = video_pipeline.run(
        b"video",
        sampler=StubFrameSampler(n_frames=0),
        extractor=StubFaceExtractor(force_faces=2),
        detector=face_stub(),
    )

    assert result.routing.result_class is ResultClass.UNDETERMINED


def test_undetermined_never_becomes_authentic_or_manipulated():
    """Guards the specific failure CLAUDE.md calls out: silently defaulting."""
    result = image_pipeline.run(
        b"image", extractor=StubFaceExtractor(force_faces=0), detector=face_stub()
    )

    assert result.routing.result_class not in {ResultClass.AUTHENTIC, ResultClass.MANIPULATED}
    assert result.score != 0.0


# --- >1 face => worst-case rollup -------------------------------------------


def test_image_with_two_faces_rolls_up_to_the_worse_one():
    detector = FixedDetector([5.0, 95.0])

    result = image_pipeline.run(
        b"image", extractor=StubFaceExtractor(force_faces=2), detector=detector
    )

    assert result.score == 95.0
    assert result.face_count == 2
    assert result.routing.result_class is ResultClass.MANIPULATED
    # Both faces are still recorded individually for the audit trail.
    assert len(result.items) == 2


def test_video_frame_with_multiple_faces_rolls_up_per_frame():
    detector = FixedDetector([10.0, 90.0])

    result = video_pipeline.run(
        b"video",
        sampler=StubFrameSampler(n_frames=8),
        extractor=StubFaceExtractor(force_faces=2),
        detector=detector,
    )

    assert result.face_count == 2
    # Every frame rolled up to its worse face, so the aggregate sits at 90.
    assert result.score == 90.0
    assert any("rolled up worst-case" in n for n in result.notes)


def test_single_face_needs_no_rollup():
    detector = FixedDetector([42.0])

    result = image_pipeline.run(
        b"image", extractor=StubFaceExtractor(force_faces=1), detector=detector
    )

    assert result.score == 42.0
    assert result.face_count == 1


# --- audio ------------------------------------------------------------------


def test_audio_uses_a_separate_model_from_the_face_pipelines():
    face = face_stub().version.model_version_id
    aud = audio_stub().version.model_version_id

    assert face != aud


def test_audio_has_no_face_count():
    result = audio_pipeline.run(
        b"audio", chunker=StubAudioChunker(n_chunks=8), detector=audio_stub()
    )

    # None, not 0 -- 0 would collide with the "0 faces => undetermined" rule.
    assert result.face_count is None
    assert result.score is not None


def test_audio_with_no_chunks_is_undetermined():
    result = audio_pipeline.run(
        b"audio", chunker=StubAudioChunker(n_chunks=0), detector=audio_stub()
    )

    assert result.routing.result_class is ResultClass.UNDETERMINED


# --- model identity ---------------------------------------------------------


def test_video_and_image_share_one_face_model_version_id():
    """CLAUDE.md: same weights, one model_version_id across video and image."""
    v = video_pipeline.run(
        b"clip",
        sampler=StubFrameSampler(n_frames=6),
        extractor=StubFaceExtractor(force_faces=1),
        detector=face_stub(),
    )
    i = image_pipeline.run(
        b"still", extractor=StubFaceExtractor(force_faces=1), detector=face_stub()
    )

    assert v.model_version.model_version_id == i.model_version.model_version_id


def test_stub_backend_is_marked_as_not_a_real_detector():
    """Nothing may present a stub score as a detection result."""
    assert face_stub().version.is_real_detector is False
    assert audio_stub().version.is_real_detector is False
    assert "stub" in face_stub().version.model_version_id
