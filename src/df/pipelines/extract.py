"""Media decoding: frame sampling, face extraction/alignment, audio chunking.

Every step is a Protocol with a deterministic stub implementation, so the
pipelines can be exercised in tests without OpenCV/librosa or real media. The
real implementations import their heavy deps lazily.

These run in the CPU-preprocess worker, which parses untrusted media. Per
CLAUDE.md that worker must be network-isolated and locked down IN THE SAME PHASE
it is built -- there is no AV scanning in front of it, and an unisolated parser
of attacker-supplied video is an open compromise window.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from df.config import settings


@dataclass(frozen=True)
class Frame:
    index: int
    data: bytes           # encoded still (PNG/JPEG bytes)
    timestamp_s: float


@dataclass(frozen=True)
class FaceCrop:
    """One aligned face. `confidence` combines detection and alignment quality
    and becomes the item's aggregation weight."""

    frame_index: int
    face_index: int
    data: bytes
    confidence: float
    bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class AudioChunk:
    index: int
    spectrogram: bytes
    start_s: float
    duration_s: float
    confidence: float = 1.0


class FrameSampler(Protocol):
    def sample(self, video_bytes: bytes) -> list[Frame]: ...


class FaceExtractor(Protocol):
    def extract(self, image_bytes: bytes, frame_index: int = 0) -> list[FaceCrop]: ...


class AudioChunker(Protocol):
    def chunk(self, audio_bytes: bytes) -> list[AudioChunk]: ...


# --- deterministic stubs ---------------------------------------------------


def _det(data: bytes, salt: bytes, mod: int) -> int:
    return int.from_bytes(hashlib.sha256(salt + data).digest()[:4], "big") % mod


def _stub_png(payload: bytes, size: int = 64) -> bytes:
    """A real, decodable PNG whose pixels are derived deterministically from
    `payload`.

    These stubs used to emit a bare 32-byte sha256 digest as `data`. The CPU
    worker then stored it under a `.png` key with `content_type="image/png"` --
    so the bytes were labelled as an image, named as an image, and were not an
    image. Nothing noticed for the life of the project because the only consumer
    was the stub scorer, which hashes the bytes rather than decoding them.

    It surfaced the first time a real detector ran: `cv2.imdecode` returned None
    and every job dead-lettered with "could not decode item bytes as an image".
    That is the seam this file's `Frame.data` docstring already described
    ("encoded still (PNG/JPEG bytes)"), so the contract was right and the stub
    was violating it.

    Determinism is preserved -- the same input still produces the same bytes and
    therefore the same stub score -- and no dependency is added: an 8-bit
    greyscale PNG is a signature plus three chunks, and zlib/struct are stdlib.
    Doing this with numpy or cv2 would put a heavy import in the path that exists
    precisely so tests need no infrastructure.
    """
    import struct
    import zlib

    stream = bytearray()
    seed = payload
    while len(stream) < size * size:
        seed = hashlib.sha256(seed).digest()
        stream.extend(seed)
    pixels = bytes(stream[: size * size])
    # Each scanline is prefixed with filter type 0 (None).
    raw = b"".join(
        b"\x00" + pixels[row * size : (row + 1) * size] for row in range(size)
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        bytes([137, 80, 78, 71, 13, 10, 26, 10])  # PNG signature
        # width, height, bit depth 8, colour type 0 (greyscale), default
        # compression/filter/interlace.
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


class StubFrameSampler:
    """Emits a fixed number of synthetic frames derived from the input bytes."""

    def __init__(self, n_frames: int = 12) -> None:
        self.n_frames = n_frames

    def sample(self, video_bytes: bytes) -> list[Frame]:
        fps = settings.video_fps_sample or 1.0
        return [
            Frame(
                index=i,
                data=_stub_png(hashlib.sha256(video_bytes + bytes([i])).digest()),
                timestamp_s=i / fps,
            )
            for i in range(self.n_frames)
        ]


class StubFaceExtractor:
    """Returns 0, 1, or 2 faces deterministically from the input bytes.

    Configure `force_faces` to drive the 0-face (undetermined) and multi-face
    (worst-case rollup) paths in tests.
    """

    def __init__(self, force_faces: int | None = None) -> None:
        self.force_faces = force_faces

    def extract(self, image_bytes: bytes, frame_index: int = 0) -> list[FaceCrop]:
        n = self.force_faces if self.force_faces is not None else _det(image_bytes, b"nfaces", 3)
        return [
            FaceCrop(
                frame_index=frame_index,
                face_index=i,
                data=_stub_png(hashlib.sha256(image_bytes + b"face" + bytes([i])).digest()),
                confidence=0.6 + 0.1 * (i % 3),
            )
            for i in range(n)
        ]


class StubAudioChunker:
    def __init__(self, n_chunks: int = 8) -> None:
        self.n_chunks = n_chunks

    def chunk(self, audio_bytes: bytes) -> list[AudioChunk]:
        dur = settings.audio_chunk_seconds
        return [
            AudioChunk(
                index=i,
                spectrogram=_stub_png(
                    hashlib.sha256(audio_bytes + b"chunk" + bytes([i])).digest()
                ),
                start_s=i * dur,
                duration_s=dur,
            )
            for i in range(self.n_chunks)
        ]


# --- real implementations (lazy heavy imports) -----------------------------


class OpenCVFrameSampler:
    """Uniform time sampling at DF_VIDEO_FPS_SAMPLE, capped at DF_VIDEO_MAX_FRAMES.

    Capped because cost per job has to be bounded -- an hour of 60fps video would
    otherwise turn one upload into six figures of GPU inferences.
    """

    def sample(self, video_bytes: bytes) -> list[Frame]:
        import tempfile

        import cv2

        frames: list[Frame] = []
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        try:
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, int(round(src_fps / max(settings.video_fps_sample, 0.01))))
            i = out_i = 0
            while out_i < settings.video_max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % step == 0:
                    ok_enc, buf = cv2.imencode(".png", frame)
                    if ok_enc:
                        frames.append(
                            Frame(index=out_i, data=buf.tobytes(), timestamp_s=i / src_fps)
                        )
                        out_i += 1
                i += 1
        finally:
            cap.release()
            import os

            os.unlink(tmp_path)
        return frames


class OpenCVFaceExtractor:
    """Haar cascade detect + similarity-align to the model input size.

    Haar is a placeholder for a proper detector (RetinaFace/SCRFD) -- it is
    fast and dependency-free but misses profile and small faces, which shows up
    downstream as 0-face `undetermined` results rather than wrong verdicts.
    """

    def __init__(self, min_confidence: float = 0.3) -> None:
        self.min_confidence = min_confidence

    def extract(self, image_bytes: bytes, frame_index: int = 0) -> list[FaceCrop]:
        import cv2
        import numpy as np

        from df.inference.efficientnet import FACE_INPUT_SIZE

        arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return []

        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        boxes, _, weights = cascade.detectMultiScale3(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48), outputRejectLevels=True
        )

        crops: list[FaceCrop] = []
        for face_index, ((x, y, w, h), weight) in enumerate(zip(boxes, weights)):
            # Haar's reject-level weight is unbounded; squash to 0-1 so it can be
            # used as an aggregation weight alongside other detectors.
            confidence = float(min(1.0, max(0.0, weight / 10.0)))
            if confidence < self.min_confidence:
                continue
            crop = cv2.resize(arr[y : y + h, x : x + w], FACE_INPUT_SIZE)
            ok, buf = cv2.imencode(".png", crop)
            if not ok:
                continue
            crops.append(
                FaceCrop(
                    frame_index=frame_index,
                    face_index=face_index,
                    data=buf.tobytes(),
                    confidence=confidence,
                    bbox=(int(x), int(y), int(w), int(h)),
                )
            )
        return crops


class LibrosaAudioChunker:
    """Fixed-length chunks -> log-mel spectrogram PNGs."""

    def chunk(self, audio_bytes: bytes) -> list[AudioChunk]:
        import io

        import librosa
        import numpy as np

        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        dur = settings.audio_chunk_seconds
        samples_per_chunk = int(dur * sr)
        chunks: list[AudioChunk] = []

        for i in range(0, max(1, len(y) // samples_per_chunk)):
            seg = y[i * samples_per_chunk : (i + 1) * samples_per_chunk]
            if len(seg) < samples_per_chunk * 0.5:
                continue  # trailing partial chunk carries too little signal
            mel = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=128)
            db = librosa.power_to_db(mel, ref=np.max)
            norm = ((db - db.min()) / max(1e-6, db.max() - db.min()) * 255).astype("uint8")
            import cv2

            ok, buf = cv2.imencode(".png", norm)
            if not ok:
                continue
            chunks.append(
                AudioChunk(
                    index=len(chunks),
                    spectrogram=buf.tobytes(),
                    start_s=i * dur,
                    duration_s=dur,
                )
            )
        return chunks


def build_frame_sampler() -> FrameSampler:
    return StubFrameSampler() if settings.inference_backend == "stub" else OpenCVFrameSampler()


def build_face_extractor() -> FaceExtractor:
    return StubFaceExtractor() if settings.inference_backend == "stub" else OpenCVFaceExtractor()


def build_audio_chunker() -> AudioChunker:
    return StubAudioChunker() if settings.inference_backend == "stub" else LibrosaAudioChunker()
