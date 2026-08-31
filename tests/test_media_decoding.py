"""Decoding the media a judge actually uploads, and saying so when it fails.

Two failures live here, both invisible to every other test because the suite
feeds the pipeline correctly-oriented JPEGs it generated itself.

1. HEIC. A phone camera roll is HEIC by default and OpenCV has no HEIF codec at
   all -- `measured: yes` 2026-08-31, its build lists JPEG, PNG and WEBP only.
   `cv2.imdecode` returned None, `extract()` returned [], and the job completed
   as `undetermined`: the service told someone no face was found in a photo of
   their own face. A confidently wrong answer, and nothing about it looked
   broken.

2. Reporting. `undetermined` is the right verdict for a photo with nobody in it
   and the wrong verdict for a photo that could not be opened, and the two were
   indistinguishable from outside.

EXIF orientation was checked at the same time and needs no fix: `cv2.imdecode`
honours it (`measured: yes` -- a 300x100 image tagged orientation=6 decodes as
100x300), so a sideways phone photo arrives upright on the OpenCV path. The
pillow-heif path does NOT get that for free, which is why `decode_image` calls
`exif_transpose` on that branch specifically.

# In-process. Live counterpart: a real HEIC built with pillow-heif and uploaded
# through docker-compose.realpipeline.yml -- decoded, face found, scored 0.55,
# `likely_authentic`.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from df.gateway.app import _decode_advisories
from df.pipelines.extract import (
    DECODABLE_IMAGE_FORMATS,
    decode_image,
    sniff_format,
)


def _ftyp(brand: bytes) -> bytes:
    return bytes(4) + b"ftyp" + brand + bytes(16)


# --- format sniffing --------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    (bytes([0xFF, 0xD8, 0xFF, 0xE0]) + bytes(16), "jpeg"),
    (bytes([137, 80, 78, 71, 13, 10, 26, 10]) + bytes(16), "png"),
    (_ftyp(b"heic"), "heic"),
    (_ftyp(b"mif1"), "heic"),      # what iOS actually writes for stills
    (_ftyp(b"avif"), "avif"),
    (_ftyp(b"isom"), "mp4"),
    (_ftyp(b"zzzz"), "iso-bmff"),  # unknown brand, still identifiably ISO-BMFF
    (b"not an image at all", "unknown"),
    (b"", "unknown"),              # must not raise on a truncated upload
])
def test_sniff_identifies_what_a_phone_sends(raw, expected):
    assert sniff_format(raw) == expected


def test_heic_is_not_in_the_opencv_decodable_set():
    """Pins why the pillow-heif fallback exists. If HEIC ever appears here
    because OpenCV gained a codec, the fallback can go -- but that is a change
    to make deliberately, not to discover."""
    assert "heic" not in DECODABLE_IMAGE_FORMATS
    assert "avif" not in DECODABLE_IMAGE_FORMATS
    assert "jpeg" in DECODABLE_IMAGE_FORMATS


# --- decoding ---------------------------------------------------------------


def test_a_normal_jpeg_decodes():
    img = np.random.default_rng(0).integers(0, 255, (60, 80, 3)).astype(np.uint8)
    raw = cv2.imencode(".jpg", img)[1].tobytes()

    out = decode_image(raw)

    assert out is not None
    assert out.shape[:2] == (60, 80)


def test_undecodable_bytes_return_none_rather_than_raising():
    """The CPU worker parses untrusted media. Garbage must produce a reportable
    None, not take the worker down -- and not be mistaken for an empty photo."""
    assert decode_image(b"definitely not an image") is None
    assert decode_image(b"") is None


def test_a_truncated_heic_returns_none_not_a_crash():
    """A real HEIC header with no payload behind it. pillow-heif must fail
    closed inside decode_image rather than propagating."""
    assert decode_image(_ftyp(b"heic")) is None


# --- saying WHY nothing was examined ---------------------------------------


def test_an_undecodable_upload_is_reported_as_not_analysed():
    """The distinction that was missing entirely. This must not read as
    'no face present', because that is a different and wrong claim."""
    advice = _decode_advisories({"decodable": False, "media_format": "heic"})

    assert len(advice) == 1
    assert "MEDIA NOT DECODED" in advice[0]
    assert "heic" in advice[0]
    assert "NOT 'no face present'" in advice[0]


def test_a_decodable_upload_gets_no_decode_advisory__the_positive_control():
    """Without this, an advisory builder that fired unconditionally would pass
    the check above and put a scary warning on every healthy result."""
    assert _decode_advisories({"decodable": True, "media_format": "jpeg"}) == []


def test_a_pre_gate_job_with_no_recorded_decodability_stays_quiet():
    """Absent is not False. A job from before this was recorded must not be
    retroactively labelled undecodable -- that would be inventing a failure."""
    assert _decode_advisories({}) == []
    assert _decode_advisories({"media_format": "jpeg"}) == []
