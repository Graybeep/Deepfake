"""Score a labelled video set with the REAL detector and emit {logit, label} JSONL.

This is the missing link between "we have a labelled held-out set" and
"scripts/fit_calibration.py can fit a temperature". It runs inside the GPU
container, because that is where torch and the weights are:

    docker compose -f docker-compose.yml -f docker-compose.weights.yml \\
        run --rm -v /path/to/dfdc_validation:/data:ro gpu-inference \\
        python scripts/extract_logits.py --dir /data --out /tmp/heldout.jsonl

Expects a directory of videos plus a `metadata.json` in DFDC's own format:

    {"abcdef.mp4": {"label": "FAKE", ...}, "ghijkl.mp4": {"label": "REAL", ...}}

WHICH DFDC SPLIT
----------------
The **validation split (4,000 clips)**, not the Kaggle `test_videos` folder.
Kaggle shipped 400 unlabelled videos because withholding ground truth is how the
leaderboard worked; there is nothing to fit against there. The official
validation split from the AWS portal has `metadata.json`, is 50/50 real/fake,
and -- the part that matters -- uses 214 subjects **none of which appear in the
training set**. Subject-disjoint is a stronger guarantee than clip-disjoint: a
different clip of the same actor would leak identity the model has already seen.

Do NOT point this at the training split. The weights were fitted on it, and a
temperature fitted on memorised data produces a confident, well-calibrated-
looking number that is wrong in deployment, with nothing downstream to catch it.

WHY IT REUSES THE PRODUCTION PIPELINE
-------------------------------------
Frame sampling, face extraction and the detector all come from `df.*` rather
than being reimplemented here. A temperature is only valid for the distribution
it was fitted on, and that includes preprocessing: fit on differently-cropped or
differently-resized faces and the calibration describes a pipeline that never
runs.

A LIMITATION TO READ BEFORE TRUSTING THE RESULT
------------------------------------------------
This emits ONE ROW PER FACE CROP, labelled with its video's label, because
`Temperature.apply` runs per item inside `predict_batch` -- so per-item is the
level the temperature actually acts on.

But the score bands in `df/bands.py` apply to the **aggregated** score, and a
confidence-weighted trimmed mean of calibrated probabilities is not itself
guaranteed calibrated. Fitting per item makes per-item scores well calibrated
and leaves the aggregate approximately so, not provably so. Calibrating the
aggregate directly would need an aggregate-level logit, which does not exist:
aggregation happens on scores, after the sigmoid.

That gap is real, it is not closed by this script, and it should be recorded
against any temperature this produces.

Two more honest caveats:
  * every crop from a FAKE video is labelled fake. DFDC manipulates the whole
    clip so this holds there; it would not hold for a set with partial
    manipulation, and using one would poison the fit.
  * videos contributing many crops dominate the fit. `--max-faces-per-video`
    caps that; without it a long clip with a well-detected face counts for more
    than a short one for no principled reason.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, "/app/src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from df.config import settings  # noqa: E402
from df.inference.registry import get_face_model  # noqa: E402
from df.pipelines.extract import (  # noqa: E402
    build_face_extractor,
    build_frame_sampler,
)

LABELS = {"FAKE": 1, "REAL": 0}


def rows_for_video(
    blob: bytes,
    label: int,
    name: str,
    *,
    sampler,
    extractor,
    detector,
    max_faces: int,
) -> list[dict]:
    """Sample, extract faces, score, and return one row per surviving crop.

    Separated from main() so the scoring path can be tested without a video
    file containing a real face. The container smoke run covers the wiring --
    model loads, metadata parses, unreadable and unlabelled videos skipped --
    and this function is where the rows are actually shaped.
    """
    crops = []
    for frame in sampler.sample(blob):
        crops.extend(extractor.extract(frame.data, frame_index=frame.index))
        if len(crops) >= max_faces:
            break
    crops = crops[:max_faces]
    if not crops:
        # 0 faces is `undetermined` in production, not a score, so these rows
        # have no place in a calibration either.
        return []

    preds = detector.predict_batch([c.data for c in crops])
    rows = []
    for crop, pred in zip(crops, preds):
        if pred.logit is None:
            raise ValueError(
                "detector returned no logit; a temperature cannot be fitted "
                "from already-scaled scores"
            )
        rows.append({
            "logit": pred.logit,
            "label": label,
            "video": name,
            "frame_index": crop.frame_index,
            "face_index": crop.face_index,
            "detection_confidence": crop.confidence,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, type=pathlib.Path,
                    help="directory of videos containing metadata.json")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--max-faces-per-video", type=int, default=8,
                    help="cap so long clips do not dominate the fit")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N videos (for a smoke run)")
    args = ap.parse_args()

    if settings.inference_backend != "torch":
        raise SystemExit(
            "DF_INFERENCE_BACKEND is not 'torch'. The stub scores by hashing its "
            "input and reports no logit, so there is nothing to calibrate. Run "
            "this under docker-compose.weights.yml."
        )

    meta_path = args.dir / "metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"no metadata.json in {args.dir} -- is this the labelled split?")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    detector = get_face_model()
    if not detector.version.is_real_detector:
        raise SystemExit("the loaded detector is not a real one; refusing to fit on it")
    sampler, extractor = build_frame_sampler(), build_face_extractor()

    print(f"model     : {detector.version.model_version_id}")
    print(f"videos    : {len(metadata)} in {args.dir}")

    written = skipped_no_face = skipped_unreadable = 0
    unknown_labels: set[str] = set()

    with args.out.open("w", encoding="utf-8") as out:
        for n, (name, entry) in enumerate(sorted(metadata.items()), 1):
            if args.limit and n > args.limit:
                break
            raw_label = (entry or {}).get("label", "")
            if raw_label not in LABELS:
                # Never guess. An unrecognised label silently mapped to 0 would
                # poison the fit in the direction of "everything is authentic".
                unknown_labels.add(str(raw_label))
                continue
            label = LABELS[raw_label]

            path = args.dir / name
            if not path.exists():
                skipped_unreadable += 1
                continue

            try:
                rows = rows_for_video(
                    path.read_bytes(), label, name,
                    sampler=sampler, extractor=extractor, detector=detector,
                    max_faces=args.max_faces_per_video,
                )
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad file must not end the run
                print(f"  skip {name}: {type(exc).__name__}: {exc}")
                skipped_unreadable += 1
                continue

            if not rows:
                skipped_no_face += 1
                continue

            for row in rows:
                out.write(json.dumps(row) + "\n")
                written += 1

            if n % 100 == 0:
                print(f"  {n}/{len(metadata)} videos, {written} rows")

    print(f"\nwrote {written} rows to {args.out}")
    print(f"skipped: {skipped_no_face} with no detected face, "
          f"{skipped_unreadable} unreadable")
    if unknown_labels:
        print(f"WARNING: {len(unknown_labels)} unrecognised label value(s) "
              f"{sorted(unknown_labels)} -- those videos were skipped, not guessed")
    if written:
        print(f"\nNext:  python scripts/fit_calibration.py --scores {args.out} "
              f"--model face --describe 'DFDC validation split'")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
