"""EfficientNet backend (torch). Loaded only when DF_INFERENCE_BACKEND=torch.

Two separate models, per CLAUDE.md:
  * face  -- shared by the video and image pipelines. Same weights, ONE
             model_version_id, so a video result and an image result are
             directly comparable.
  * audio -- separate model over spectrograms, its own model_version_id and its
             own calibration temperature.

Imports of torch are deferred into __init__ so the gateway and CPU worker images
never need the ML stack.
"""
from __future__ import annotations

import hashlib
import os
import pathlib

from df.inference.base import (
    VALIDATION_LEVELS,
    VALIDATION_PRODUCTION,
    VALIDATION_RESEARCH,
    Detector,
    ModelVersion,
    Prediction,
)


def _validation_level() -> str:
    """How far these weights may be trusted. Defaults to research-checkpoint.

    Loading real weights is not validation. A public checkpoint is trained on
    someone else's distribution, thresholded for someone else's task, and has
    never been measured against this pipeline's bands -- so the honest default
    for any torch backend is `research-checkpoint`, and it must stay that way
    without a deliberate act.

    Claiming production-validated therefore takes two keys: the level AND a
    non-empty DF_MODEL_VALIDATION_SIGNOFF naming who signed it off, which is
    recorded. CLAUDE.md forbids the claim outright today; this makes reaching
    it an explicit, attributable act rather than a one-character env change.
    """
    level = os.environ.get("DF_MODEL_VALIDATION", VALIDATION_RESEARCH).strip()
    if level not in VALIDATION_LEVELS:
        raise ValueError(
            f"DF_MODEL_VALIDATION={level!r} is not one of {sorted(VALIDATION_LEVELS)}"
        )
    if level == VALIDATION_PRODUCTION and not os.environ.get(
        "DF_MODEL_VALIDATION_SIGNOFF", ""
    ).strip():
        raise ValueError(
            "DF_MODEL_VALIDATION=production-validated requires "
            "DF_MODEL_VALIDATION_SIGNOFF naming who validated these weights "
            "against this pipeline. CLAUDE.md does not permit the claim yet."
        )
    return level
from df.inference.calibration import (
    AUDIO_TEMPERATURE,
    CALIBRATION_SCHEME,
    FACE_TEMPERATURE,
    Temperature,
)

FACE_INPUT_SIZE = (224, 224)
AUDIO_INPUT_SIZE = (224, 224)  # spectrogram rendered to the same input geometry


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class EfficientNetDetector(Detector):
    def __init__(
        self,
        weights_path: str,
        *,
        modality: str,
        temperature: Temperature,
        arch: str = "efficientnet_b0",
        device: str | None = None,
    ) -> None:
        import torch
        import torchvision

        path = pathlib.Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{modality} weights not found at {weights_path}. Set DF_FACE_WEIGHTS / "
                f"DF_AUDIO_WEIGHTS, or run with DF_INFERENCE_BACKEND=stub."
            )

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model = getattr(torchvision.models, arch)(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, 1)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval().to(self.device)
        self._model = model
        self._temperature = temperature

        self._version = ModelVersion(
            model_version_id=f"{modality}-{arch}-{_sha256_file(path)[:12]}",
            architecture=arch,
            modality=modality,
            weights_sha256=_sha256_file(path),
            calibration=CALIBRATION_SCHEME,
            is_real_detector=True,
            # Never defaults to production-validated. A checkpoint being
            # loadable says nothing about whether it was validated for this
            # data, this threshold set, or this decision.
            validation=_validation_level(),
        )

    @property
    def version(self) -> ModelVersion:
        return self._version

    def predict_batch(self, inputs: list) -> list[Prediction]:
        """inputs: list of CHW float tensors already resized and normalised."""
        torch = self._torch
        if not inputs:
            return []
        batch = torch.stack(inputs).to(self.device)
        with torch.no_grad():
            logits = self._model(batch).squeeze(-1)
        return [
            # Confidence is supplied by the face detector / aligner upstream and
            # overwritten by the caller; 1.0 here means "model had no objection".
            Prediction(score=self._temperature.apply(float(logit)), confidence=1.0)
            for logit in logits.cpu()
        ]


def build_face_detector(weights_path: str) -> EfficientNetDetector:
    return EfficientNetDetector(weights_path, modality="face", temperature=FACE_TEMPERATURE)


def build_audio_detector(weights_path: str) -> EfficientNetDetector:
    return EfficientNetDetector(weights_path, modality="audio", temperature=AUDIO_TEMPERATURE)
