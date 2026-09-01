"""The frame sampler streams, and the decodability signal survives it.

Two things are under test and the second is the reason this file exists.

STREAMING. `OpenCVFrameSampler.sample()` used to build a list of every sampled
frame before returning it, and each entry is a PNG-encoded FULL frame -- 3.4 MB
at 1080p. Peak memory scaled with resolution and length before anything was
scored, beside a resident 66M-parameter model. `measured: yes` 2026-09-01 on the
deployed service: 40 frames at 720p killed the inference worker with SIGKILL,
and the SAME 20 s clip at an 8-frame cap produced 201 s-never-finished,
53 s-crashed-then-recovered and 4.2 s-clean across three consecutive runs.

THE TRAP. The one caller decided whether the video was readable with
`sampled_any = bool(frames)`. A generator is ALWAYS truthy, so converting the
sampler without touching that line would report every video as decodable --
including a file that yielded nothing. `sampled_any` feeds `decodable`, which is
what raises `MEDIA NOT DECODED`, so the visible effect would be telling someone
"no face found" about a file that was never read. CLAUDE.md records that exact
confusion as fixed on 2026-08-31. It would have come back silently, and no
existing test covered it, because with a list the line was correct.

# In-process. Live counterpart: upload a video to a running stack and watch
# /healthz -- scripts/ has no video probe, and the crash this fixes was only
# ever visible against the deployed container.
"""
from __future__ import annotations

import inspect
import pathlib

import pytest

from df.pipelines.extract import Frame, OpenCVFrameSampler
from df.storage import InMemoryStorage
from df.workers import cpu_preprocess
from tests.fakes import FakeDb, FakeJobStatus

pytest.importorskip("cv2")


# --- the sampler streams ----------------------------------------------------


def test_sample_is_a_generator_not_a_list():
    """The whole point. If this reverts to returning a list, peak memory goes
    back to scaling with clip length and the caller's truthiness check silently
    starts working again -- masking the bug below."""
    assert inspect.isgeneratorfunction(OpenCVFrameSampler.sample)


def test_frames_arrive_one_at_a_time(tmp_path, monkeypatch):
    """Consuming one frame must not have decoded the rest.

    Asserted by counting `cv2.imencode` calls: after pulling a single frame,
    exactly one frame has been encoded. A list-returning sampler would have
    encoded every frame before the first `next()` returned.
    """
    import cv2

    from df.config import Settings
    from df.pipelines import extract as ex

    monkeypatch.setenv("DF_VIDEO_MAX_FRAMES", "10")
    monkeypatch.setenv("DF_VIDEO_FPS_SAMPLE", "30")     # every frame
    monkeypatch.setattr(ex, "settings", Settings())

    video = _make_video(tmp_path, frames=10)

    encodes = {"n": 0}
    real = cv2.imencode

    def counting(ext, img, *a, **k):
        encodes["n"] += 1
        return real(ext, img, *a, **k)

    monkeypatch.setattr(cv2, "imencode", counting)

    gen = ex.OpenCVFrameSampler().sample(video)
    first = next(gen)

    assert isinstance(first, Frame)
    assert encodes["n"] == 1, f"{encodes['n']} frames encoded to produce the first"

    gen.close()


def test_the_frame_cap_still_bounds_total_work(tmp_path, monkeypatch):
    """Streaming removes the memory ceiling, not the work ceiling."""
    from df.config import Settings
    from df.pipelines import extract as ex

    monkeypatch.setenv("DF_VIDEO_MAX_FRAMES", "3")
    monkeypatch.setenv("DF_VIDEO_FPS_SAMPLE", "30")
    monkeypatch.setattr(ex, "settings", Settings())

    video = _make_video(tmp_path, frames=20)

    assert len(list(ex.OpenCVFrameSampler().sample(video))) == 3


def test_the_temp_file_is_removed_even_if_iteration_is_abandoned(tmp_path, monkeypatch):
    """`finally` in a generator runs on close as well as on exhaustion. Without
    that, every abandoned video would leak a full copy of the upload to disk."""
    import tempfile

    from df.config import Settings
    from df.pipelines import extract as ex

    monkeypatch.setenv("DF_VIDEO_FPS_SAMPLE", "30")
    monkeypatch.setattr(ex, "settings", Settings())

    created: list[str] = []
    real_ntf = tempfile.NamedTemporaryFile

    def tracking(*a, **k):
        fh = real_ntf(*a, **k)
        created.append(fh.name)
        return fh

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", tracking)

    gen = ex.OpenCVFrameSampler().sample(_make_video(tmp_path, frames=10))
    next(gen)                    # start it, do not finish it
    assert created, "no temp file was created"
    gen.close()

    assert not pathlib.Path(created[0]).exists(), "temp file leaked on abandon"


# --- the trap: decodability must survive streaming --------------------------


class _Sampler:
    """Yields a fixed number of frames, streaming, like the real one."""

    def __init__(self, n: int) -> None:
        self.n = n

    def sample(self, video_bytes: bytes):
        for i in range(self.n):
            yield Frame(index=i, data=_png(), timestamp_s=float(i))


def _run(sampler, monkeypatch) -> list[tuple[str, str, dict]]:
    from df.queue import TOPIC_PREPROCESS, InMemoryQueue

    monkeypatch.setattr(cpu_preprocess, "build_frame_sampler", lambda: sampler)

    db, storage = FakeDb(), InMemoryStorage()
    queue, status = InMemoryQueue(), FakeJobStatus()
    db.add_job("job-v", "video")
    storage.put_bytes("raw/job-v/original", b"pretend-video")
    queue.push(TOPIC_PREPROCESS, {"job_id": "job-v", "media_type": "video"})

    cpu_preprocess.handle(
        queue.pop(TOPIC_PREPROCESS), db=db, storage=storage, queue=queue, status=status
    )
    # FakeDb.events is a flat list of (job_id, event, detail) tuples
    return [(j, e, d) for j, e, d in db.events if j == "job-v"]


def test_a_video_that_yields_no_frames_is_reported_undecodable(monkeypatch):
    """THE regression this file exists for.

    `bool(<generator>)` is True even when the generator yields nothing, so the
    old line would have marked an unreadable video as decodable and the result
    would have said "no face found" instead of "not analysed".
    """
    events = _run(_Sampler(0), monkeypatch)

    complete = [d for _, e, d in events if e == "preprocess.complete"]
    assert complete, f"no preprocess.complete; got {[e for _, e, _ in events]}"
    assert complete[-1]["decodable"] is False, (
        "a video yielding zero frames was reported decodable -- MEDIA NOT "
        "DECODED will not fire and the user is told 'no face found' about a "
        "file that was never read"
    )


def test_a_video_that_yields_frames_is_reported_decodable(monkeypatch):
    """The positive control, and not optional: a mutation that hardcoded
    `decodable = False` would satisfy the case above on its own."""
    events = _run(_Sampler(4), monkeypatch)

    complete = [d for _, e, d in events if e == "preprocess.complete"]
    assert complete, f"no preprocess.complete; got {[e for _, e, _ in events]}"
    assert complete[-1]["decodable"] is True


# --- helpers ----------------------------------------------------------------


def _png() -> bytes:
    import cv2
    import numpy as np

    rng = np.random.default_rng(0)
    return cv2.imencode(".png", rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))[1].tobytes()


def _make_video(tmp_path: pathlib.Path, *, frames: int, size=(160, 120)) -> bytes:
    import cv2
    import numpy as np

    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    if not writer.isOpened():
        pytest.skip("no mp4v encoder available in this OpenCV build")
    rng = np.random.default_rng(1)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()
