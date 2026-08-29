"""Torch backend: the DFDC-winner EfficientNet checkpoint.

CLAUDE.md chose `selimsef/dfdc_deepfake_challenge` as the placeholder detector.
This module loads it. Everything asserted here about the checkpoint was
`measured: yes` on 2026-08-29 by reading the file itself, not from the paper or
the README -- the previous version of this module was written from an assumption
about the architecture and was wrong in four independent ways at once:

  1. the encoder is a **timm** model (`tf_efficientnet_b7_ns`), not torchvision.
     torchvision's EfficientNet uses `features.*` / `classifier.*` keys; this
     checkpoint uses `encoder.*` / `fc.*`. The two do not overlap in a single
     key, so `load_state_dict` would have failed on every parameter.
  2. it is **B7**, not B0/B4 -- 66,661,404 parameters, `conv_stem` at 64
     channels, 7 block groups, 2560 encoder features.
  3. the file is a **training checkpoint**, `{epoch, state_dict, bce_best}`,
     not a bare state dict, and every key inside carries a `module.` prefix
     from `nn.DataParallel`.
  4. the input is **380x380**, not 224x224.

Verified end state: `load_state_dict(strict=True)` reports 0 missing and 0
unexpected keys against the model built below.
[epoch 37, bce_best 0.1639973577717397, sha256 9db77ab9…]

Sources for the two facts that cannot be read off the tensor file --
`measured: no (source)`, both read 2026-08-29:
  * architecture and 380px input:
    <https://github.com/selimsef/dfdc_deepfake_challenge>
  * inference preprocessing (ImageNet mean/std, isotropic resize then
    zero-padded centring, INTER_CUBIC up / INTER_AREA down):
    <https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/kernel_utils.py>

Note the normalisation, because it is a trap: these are the **ImageNet**
constants. timm's own `pretrained_cfg` for a `tf_`-prefixed model reports the
Inception constants (0.5/0.5/0.5), so resolving mean/std from the model -- the
obvious thing to do -- silently produces differently-scaled inputs and a score
that looks plausible and is wrong.

LICENCE, unchanged and still load-bearing: the repository code is MIT, but these
weights are trained on Meta's DFDC dataset, whose terms are not published and
whose flow-through to derived weights is unsettled. Do not ship commercially
without legal review. Loading them does NOT make the detector validated: the
default level for any torch backend is `research-checkpoint`, and CLAUDE.md
forbids claiming more.
"""
from __future__ import annotations

import hashlib
import logging
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
from df.inference.calibration import (
    AUDIO_TEMPERATURE,
    FACE_TEMPERATURE,
    Temperature,
)

log = logging.getLogger("df.inference")

# The encoder the checkpoint was trained with. timm 1.0.11 still resolves the
# deprecated name to `tf_efficientnet_b7.ns_jft_in1k` (with a warning), so the
# original name is kept here: it is what the weight file is named after and what
# the upstream repo refers to.
FACE_ARCH = "tf_efficientnet_b7_ns"
FACE_INPUT_SIZE = 380

# ImageNet, per kernel_utils.py. NOT timm's cfg for this model -- see docstring.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_module(arch: str, dropout: float = 0.0):
    """Reconstruct selimsef's `DeepFakeClassifier`.

    Structure is dictated by the checkpoint, not chosen: encoder + a parameterless
    AdaptiveAvgPool2d + dropout + Linear(features, 1). The checkpoint contains no
    `avg_pool.*`, `gwap.*` or `srm_conv.*` tensors, which is what identifies it as
    the plain variant rather than the GWAP or SRM ones in the same file upstream.

    The encoder keeps its own 1000-class ImageNet `classifier` head: it is present
    in the checkpoint and unused at inference. Dropping it would make a strict
    load fail, and loading non-strict to paper over that would hide a genuine
    mismatch -- so it is built and simply not called.
    """
    import timm
    import torch.nn as nn

    class DeepFakeClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = timm.create_model(arch, pretrained=False)
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(self.encoder.num_features, 1)

        def forward(self, x):
            x = self.encoder.forward_features(x)
            x = self.avg_pool(x).flatten(1)
            return self.fc(self.dropout(x))

    return DeepFakeClassifier()


def _state_dict_from(checkpoint) -> dict:
    """Unwrap `{epoch, state_dict, bce_best}` and strip the DataParallel prefix.

    Both steps are required by the file as published. Neither is guesswork: the
    wrapper keys and the `module.` prefix were read off the checkpoint.
    """
    sd = checkpoint
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }


class EfficientNetDetector(Detector):
    def __init__(
        self,
        weights_path: str,
        *,
        modality: str,
        temperature: Temperature,
        arch: str = FACE_ARCH,
        input_size: int = FACE_INPUT_SIZE,
        device: str | None = None,
    ) -> None:
        import torch

        path = pathlib.Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{modality} weights not found at {weights_path}. Set DF_FACE_WEIGHTS / "
                f"DF_AUDIO_WEIGHTS, or run with DF_INFERENCE_BACKEND=stub."
            )

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        model = _build_module(arch)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # strict=True on purpose. A silently partial load is a model with random
        # weights in whichever layers did not match, which still returns
        # confident-looking numbers -- the exact failure this file's history is
        # a record of.
        model.load_state_dict(_state_dict_from(checkpoint), strict=True)
        model.eval().to(self.device)
        self._model = model
        self._temperature = temperature

        digest = _sha256_file(path)
        self._version = ModelVersion(
            model_version_id=f"{modality}-{arch}-{digest[:12]}",
            architecture=arch,
            modality=modality,
            weights_sha256=digest,
            # Derived from whether a fit actually happened, not a constant.
            # This field previously asserted 'launch-snapshot' on every
            # result while T was 1.0 and nothing had been fitted.
            calibration=temperature.scheme,
            is_real_detector=True,
            # Never defaults to production-validated. A checkpoint being
            # loadable says nothing about whether it was validated for this
            # data, this threshold set, or this decision.
            validation=_validation_level(),
        )
        log.info(
            "loaded %s detector arch=%s device=%s sha=%s validation=%s",
            modality, arch, self.device, digest[:12], self._version.validation,
        )

    @property
    def version(self) -> ModelVersion:
        return self._version

    def _to_tensor(self, blob: bytes):
        """Encoded image bytes -> normalised CHW tensor.

        Takes BYTES, deliberately. The stub detector's `predict_batch` takes
        encoded bytes and the GPU worker hands it exactly that
        (`storage.get_bytes(...)`), while this class previously documented
        "CHW float tensors already resized and normalised" -- so the two
        backends had incompatible contracts and switching to torch would have
        fed PNG bytes into `torch.stack`. Nothing caught it because nothing had
        ever run this path.

        Isotropic resize then zero-padded centring, matching kernel_utils.py:
        a plain square resize would distort the aspect ratio of every non-square
        crop, which is a different image from the one the model was trained on.
        """
        import cv2
        import numpy as np
        import torch

        arr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("could not decode item bytes as an image")
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

        size = self.input_size
        h, w = arr.shape[:2]
        scale = size / max(h, w)
        # Upsampling and downsampling take different interpolations upstream;
        # INTER_AREA is what avoids aliasing when shrinking.
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        resized = cv2.resize(arr, (max(1, round(w * scale)), max(1, round(h * scale))),
                             interpolation=interp)

        rh, rw = resized.shape[:2]
        canvas = np.zeros((size, size, 3), dtype=resized.dtype)
        top, left = (size - rh) // 2, (size - rw) // 2
        canvas[top:top + rh, left:left + rw] = resized

        x = torch.from_numpy(canvas).float().div_(255.0).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (x - mean) / std

    def predict_batch(self, inputs: list[bytes]) -> list[Prediction]:
        torch = self._torch
        if not inputs:
            return []
        batch = torch.stack([self._to_tensor(b) for b in inputs]).to(self.device)
        with torch.no_grad():
            logits = self._model(batch).squeeze(-1)
        # squeeze(-1) collapses a batch of one to a 0-d tensor; keep it iterable.
        if logits.ndim == 0:
            logits = logits.unsqueeze(0)
        return [
            # Confidence is supplied by the face detector / aligner upstream and
            # overwritten by the caller; 1.0 here means "model had no objection".
            Prediction(score=self._temperature.apply(float(logit)), confidence=1.0)
            for logit in logits.cpu()
        ]


def build_face_detector(weights_path: str) -> EfficientNetDetector:
    return EfficientNetDetector(
        weights_path, modality="face", temperature=FACE_TEMPERATURE, arch=FACE_ARCH
    )


def build_audio_detector(weights_path: str) -> EfficientNetDetector:
    """There is no audio checkpoint under the current decision.

    CLAUDE.md: "Audio has no checkpoint under this decision and stays on the
    stub. Expect a mixed state: video/image at research-checkpoint, audio at
    placeholder." The registry enforces that by not calling this without
    explicit weights; reaching it means someone configured DF_AUDIO_WEIGHTS, so
    the architecture below is a guess about a file this project has never seen.
    """
    raise NotImplementedError(
        "no audio checkpoint has been chosen for this project; DF_AUDIO_WEIGHTS "
        f"is set to {weights_path!r} but the architecture of that file is "
        "unknown here. Audio stays on the stub backend (validation=placeholder) "
        "per CLAUDE.md. Wire a real audio model deliberately, with its own "
        "architecture and its own calibration temperature."
    )
