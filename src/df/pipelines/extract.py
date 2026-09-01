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
from typing import Iterable, Iterator, Protocol

from df.config import settings


@dataclass(frozen=True, eq=False)
class Frame:
    index: int
    # Encoded still (PNG/JPEG bytes) OR an already-decoded BGR array.
    #
    # The real sampler yields ARRAYS. It used to PNG-encode every frame and the
    # extractor immediately decoded it again, in the same process: a lossless
    # round trip whose only effect was cost. Per 1080p frame that was ~6.2 MB
    # decoded + 3.4 MB encoded + 6.2 MB decoded again, plus two codec passes;
    # now it is one 6.2 MB array and no codec work.
    #
    # `extract()` accepts either, so the stub sampler still emits real PNGs and
    # every existing caller that passes encoded bytes is unaffected.
    data: "bytes | object"
    timestamp_s: float

    # eq=False: a numpy field would make the generated __eq__ return an array,
    # and bool() of that raises. Nothing compares Frames, and this keeps it
    # that way rather than leaving a landmine.


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
    # Iterable, NOT list. The real sampler yields one frame at a time so peak
    # memory does not scale with clip length -- see OpenCVFrameSampler. A list
    # still satisfies this, so the stub is unchanged.
    #
    # CALLER CONTRACT: you cannot ask an Iterable whether it is empty without
    # consuming it. `bool(sample(...))` is always True for a generator, and the
    # one caller relied on exactly that to decide whether the video was
    # decodable. Count what you consume instead.
    def sample(self, video_bytes: bytes) -> Iterable[Frame]: ...


class FaceExtractor(Protocol):
    # Accepts encoded bytes OR a decoded BGR array. The video path passes
    # arrays to avoid an encode/decode round trip that exists only to satisfy a
    # type; the image path passes the raw upload so decode_image still owns
    # HEIC handling and the "could not decode" verdict.
    def extract(self, image: "bytes | object", frame_index: int = 0) -> list[FaceCrop]: ...


class AudioChunker(Protocol):
    def chunk(self, audio_bytes: bytes) -> list[AudioChunk]: ...


# --- deterministic stubs ---------------------------------------------------


def _is_array(image: object) -> bool:
    """True for a decoded image array, without importing numpy at module scope.

    extract.py keeps cv2/numpy inside functions so it can be imported in
    environments that lack them; a module-level `import numpy` for an
    isinstance check would undo that.
    """
    return hasattr(image, "shape") and hasattr(image, "dtype")


def _hashable_bytes(image: object) -> bytes:
    """Deterministic bytes for the stubs, which hash whatever they are given."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)
    return image.tobytes()          # type: ignore[union-attr]


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

    def extract(self, image: "bytes | object", frame_index: int = 0) -> list[FaceCrop]:
        image_bytes = _hashable_bytes(image)
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


# --- format sniffing -------------------------------------------------------

# Magic bytes, because the client's declared content type is its own word and a
# phone upload frequently mislabels. Only what we can act on.
_MAGIC = (
    (bytes([0xFF, 0xD8, 0xFF]), "jpeg"),
    (bytes([137, 80, 78, 71, 13, 10, 26, 10]), "png"),
    (b"RIFF", "webp-or-wav"),
    (b"GIF8", "gif"),
    (b"BM", "bmp"),
)
# ISO-BMFF brands live at offset 4 after 'ftyp'. HEIC and AVIF are the ones a
# phone actually produces.
_FTYP_BRANDS = {
    b"heic": "heic", b"heix": "heic", b"hevc": "heic", b"heim": "heic",
    b"heis": "heic", b"hevm": "heic", b"mif1": "heic", b"msf1": "heic",
    b"avif": "avif", b"avis": "avif",
    b"mp42": "mp4", b"isom": "mp4", b"iso2": "mp4", b"qt  ": "mov",
}


def sniff_format(raw: bytes) -> str:
    """Identify the container from magic bytes. "unknown" when unrecognised.

    Exists so a failure to DECODE can be reported as what it is. Without it an
    undecodable upload returns zero faces, which the pipeline then reports as
    `undetermined` -- indistinguishable from a photo with nobody in it. A judge
    uploading a HEIC straight off an iPhone would be told no face was found in a
    picture of their own face, which is a confidently wrong answer rather than a
    visible failure. Worse than a crash, because nothing looks broken.
    """
    if len(raw) < 12:
        return "unknown"
    if raw[4:8] == b"ftyp":
        return _FTYP_BRANDS.get(raw[8:12].lower(), "iso-bmff")
    for magic, name in _MAGIC:
        if raw.startswith(magic):
            return name
    return "unknown"


# Formats this service can decode. HEIC/AVIF are absent from OpenCV's codec
# list (`measured: yes` 2026-08-31: JPEG, PNG, WEBP only) and are handled by
# the pillow-heif fallback in `decode_image` when it is installed.
DECODABLE_IMAGE_FORMATS = frozenset({"jpeg", "png", "webp-or-wav", "gif", "bmp"})


def decode_image(raw: bytes):
    """Decode to a BGR array, or None. Tries OpenCV, then pillow-heif.

    OpenCV honours EXIF orientation for JPEG (`measured: yes` 2026-08-31: a
    300x100 image tagged orientation=6 decodes as 100x300), so no separate
    transpose is needed on that path -- a phone photo stored sideways comes back
    upright and Haar sees a face the right way up.

    The pillow-heif branch exists because a phone camera roll is HEIC by default
    and OpenCV cannot read it at all. PIL does not decode HEIC unless
    pillow_heif registers its opener, so that registration is the whole point of
    the import.
    """
    import cv2
    import numpy as np

    try:
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        # cv2.imdecode ASSERTS on an empty buffer ("!buf.empty()") rather than
        # returning None, so a zero-byte upload would take the worker down. It
        # returns None for merely malformed bytes, so this handler is the only
        # thing covering the empty case.
        #
        # There was an explicit `if not raw: return None` here too. It was
        # removed as redundant: the handler covers empty AND malformed, and with
        # both present no single mutation could change the behaviour, so the
        # mutation harness withheld a verdict and the invariant went untested.
        # Two mechanisms guarding one property meant neither was checkable.
        arr = None
    if arr is not None:
        return arr

    try:
        import io

        import pillow_heif
        from PIL import Image, ImageOps

        pillow_heif.register_heif_opener()
        img = Image.open(io.BytesIO(raw))
        # PIL does NOT apply orientation on open, unlike cv2, so this transpose
        # is required on this branch specifically.
        img = ImageOps.exif_transpose(img).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception:
        # Undecodable by every route available. The caller reports the sniffed
        # format so the reason is legible instead of being flattened into
        # "no faces found".
        return None


# --- real implementations (lazy heavy imports) -----------------------------


class OpenCVFrameSampler:
    """Uniform time sampling at DF_VIDEO_FPS_SAMPLE, capped at DF_VIDEO_MAX_FRAMES.

    Capped because cost per job has to be bounded -- an hour of 60fps video would
    otherwise turn one upload into six figures of GPU inferences.
    """

    def sample(self, video_bytes: bytes) -> Iterator[Frame]:
        """Yields frames one at a time. Deliberately a generator.

        It used to build the whole list before returning it, and every entry is
        a PNG-encoded FULL FRAME -- 3.4 MB each at 1080p (`measured: yes`). So
        peak memory scaled with resolution AND length before a single frame was
        scored, next to a resident 66M-parameter model.

        What that cost, `measured: yes` 2026-09-01 on the deployed service: 40
        frames at 720p killed the inference worker with SIGKILL, and the SAME 20s
        clip at an 8-frame cap gave 201s-never-finished / 53s-crashed-recovered /
        4.2s-clean across three consecutive runs. Identical input, three
        outcomes, because the binding constraint was the container's remaining
        memory rather than anything about the clip.

        Streaming makes one decoded frame plus one crop the working set, so cost
        stops scaling with length. `video_max_frames` still bounds total work.

        The temp file is cleaned in `finally`, which for a generator runs when it
        is exhausted OR closed -- so an abandoned iteration still cleans up, and
        GeneratorExit is not swallowed.
        """
        import os
        import tempfile

        import cv2

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
                    # The decoded array goes straight through. cap.read()
                    # allocates a fresh array per call, so yielding it is safe
                    # -- the consumer is not handed a buffer we then overwrite.
                    yield Frame(index=out_i, data=frame, timestamp_s=i / src_fps)
                    out_i += 1
                i += 1
        finally:
            cap.release()
            os.unlink(tmp_path)


class OpenCVFaceExtractor:
    """Haar cascade detect, emitting the face crop at its NATIVE resolution.

    Haar is a placeholder for a proper detector (RetinaFace/SCRFD) -- it is fast
    and dependency-free but misses profile and small faces, which shows up
    downstream as 0-face `undetermined` results rather than as wrong verdicts.

    THIS DOES NOT RESIZE TO THE MODEL INPUT SIZE, and that is deliberate.

    It used to do `cv2.resize(box, FACE_INPUT_SIZE)`, which caused two problems
    at once, both found 2026-08-30 the first time this path was exercised:

      * It broke outright. FACE_INPUT_SIZE became the int 380 when the torch
        backend was rewritten for the B7 checkpoint, and cv2.resize needs a
        (w, h) sequence, so every detected face raised. Nothing caught it: the
        CPU worker runs the stub extractor, the weights overlay switches only
        the GPU worker, and no test covered this branch. A constant changing
        type across a module boundary, invisible because the path never ran.
      * More quietly, it was wrong even when it worked. A square resize of a
        non-square face box destroys the aspect ratio -- and it did so BEFORE
        the detector's isotropic-resize-and-pad, which was written to match
        upstream's preprocessing exactly. The careful step was being handed an
        already-distorted square and became a no-op. Two resizes, and the wrong
        one won.

    So geometry belongs to the detector, which is the component that knows what
    its model wants. This one detects and crops; `EfficientNetDetector._to_tensor`
    does the isotropic resize, the zero-padded centring and the normalisation.

    KNOWN DEVIATION, untuned: upstream crops with a margin around the detected
    box, and this takes the box exactly. Adding a margin without a labelled set
    to measure it against would be inventing a number, so it is named here
    instead of guessed.
    """

    def extract(self, image: "bytes | object", frame_index: int = 0) -> list[FaceCrop]:
        import cv2
        import numpy as np

        # An already-decoded frame skips the decode entirely. The video path
        # passes arrays; the image path still passes the raw upload bytes, and
        # those must go through decode_image so HEIC and the undecodable case
        # keep behaving as documented.
        arr = image if _is_array(image) else decode_image(image)
        if arr is None:
            return []

        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

        # Detect on a bounded copy; crop from the original. See
        # Settings.detect_max_side for why -- a 12 MP phone photo killed the
        # container and produced 52x52 junk detections at the same time.
        #
        # The scale factor is inverted below to map boxes back to native
        # coordinates. This is the geometry seam this file has already got wrong
        # twice, so it is deliberately the ONLY transform: one scalar, applied
        # once, to a box -- not a resize of any pixels that reach the model.
        full_h, full_w = gray.shape[:2]
        cap = settings.detect_max_side
        scale = 1.0
        if cap > 0 and max(full_h, full_w) > cap:
            scale = cap / max(full_h, full_w)
            # INTER_AREA is the correct filter for downscaling; INTER_LINEAR
            # aliases, which invents high-frequency detail for a texture-based
            # cascade to trip on.
            gray = cv2.resize(
                gray,
                (max(1, round(full_w * scale)), max(1, round(full_h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        boxes, _, weights = cascade.detectMultiScale3(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48), outputRejectLevels=True
        )

        crops: list[FaceCrop] = []
        for face_index, ((x, y, w, h), weight) in enumerate(zip(boxes, weights)):
            if scale != 1.0:
                # Back to native pixels, then clamped INTO the image. Rounding
                # outward on a box at the frame edge can put x+w past the width,
                # and numpy slicing would silently return a short crop rather
                # than raise -- so the bbox recorded on the row would not
                # describe the pixels that were scored.
                x = min(max(0, round(x / scale)), full_w - 1)
                y = min(max(0, round(y / scale)), full_h - 1)
                w = max(1, min(round(w / scale), full_w - x))
                h = max(1, min(round(h / scale), full_h - y))
            crop = arr[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            ok, buf = cv2.imencode(".png", crop)
            if not ok:
                continue
            crops.append(
                FaceCrop(
                    frame_index=frame_index,
                    face_index=face_index,
                    data=buf.tobytes(),
                    confidence=_haar_confidence(weight),
                    bbox=(int(x), int(y), int(w), int(h)),
                )
            )
        return crops


def _haar_confidence(weight: float) -> float:
    """Haar reject-level -> 0-1 aggregation weight. UNCALIBRATED, and arbitrary.

    Be clear about what this number is, because it is load-bearing: it becomes
    the item's weight in the confidence-weighted mean, and CLAUDE.md notes that
    the weighting is the one robustness mechanism in aggregation that actually
    fires.

    `detectMultiScale3`'s reject level is an unbounded internal cascade score.
    It is NOT a probability, NOT calibrated, and dividing it by 10 is a squash
    chosen to land typical detections in a usable range -- nothing more. It is
    monotone in Haar's own confidence, which is the only property relied on.

    Every measurement of the confidence distribution in this repo so far came
    from the STUB extractor, which emits 0.6/0.7/0.8 by construction. So
    "the weights genuinely differ" has never been observed on this path, and
    `min_confidence=0.3` has never been tuned against it. Both wait on the same
    labelled set the calibration does.

    A real detector (RetinaFace/SCRFD) returns an actual detection probability
    and would replace this outright.
    """
    return float(min(1.0, max(0.0, weight / 10.0)))


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
