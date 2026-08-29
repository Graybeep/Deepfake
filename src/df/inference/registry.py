"""Model registry: one face detector, one audio detector, cached per process.

The face detector instance is shared by the video and image pipelines on
purpose -- same weights, one model_version_id (CLAUDE.md).
"""
from __future__ import annotations

import functools
import logging

from df.config import settings
from df.inference.base import Detector

log = logging.getLogger("df.inference")


@functools.lru_cache(maxsize=1)
def get_face_model() -> Detector:
    if settings.inference_backend == "torch":
        from df.inference.efficientnet import build_face_detector

        return build_face_detector(settings.face_weights)

    from df.inference.stub import face_stub

    log.warning(
        "face model: STUB backend active -- scores are placeholders, not detections"
    )
    return face_stub()


@functools.lru_cache(maxsize=1)
def get_audio_model() -> Detector:
    # The mixed state CLAUDE.md predicts, made real rather than left to crash.
    #
    # The chosen checkpoint (DFDC-winner EfficientNet) is a FACE model. There is
    # no audio checkpoint under that decision, so switching the backend to torch
    # must not drag audio along with it: video and image go to
    # research-checkpoint while audio stays at placeholder, and the advisory is
    # derived per model so it reports that correctly without special-casing.
    #
    # Falling back is fail-CLOSED. The stub declares validation=placeholder,
    # which produces the strongest caveat on every audio result -- the opposite
    # of quietly presenting an unvalidated score as a detection.
    if settings.inference_backend == "torch" and settings.audio_weights:
        from df.inference.efficientnet import build_audio_detector

        return build_audio_detector(settings.audio_weights)

    if settings.inference_backend == "torch":
        log.warning(
            "audio model: torch backend requested but DF_AUDIO_WEIGHTS is unset; "
            "audio stays on the STUB scorer (validation=placeholder) while "
            "video/image use real weights -- see CLAUDE.md, 'Weights'"
        )

    from df.inference.stub import audio_stub

    log.warning(
        "audio model: STUB backend active -- scores are placeholders, not detections"
    )
    return audio_stub()
