"""Fit the launch-snapshot temperature for one model, from a labelled set.

    python scripts/fit_calibration.py --scores held_out.jsonl --model face

Input is JSON Lines, one record per scored item:

    {"logit": -1.83, "label": 0}
    {"logit":  4.20, "label": 1}

`logit` is the model's RAW pre-sigmoid output. It is not the 0-100 score this
service reports: that has already had a sigmoid applied, and it cannot be
un-applied without knowing the temperature you are trying to find. Collect
logits at inference time.

`label` is ground truth -- 1 manipulated, 0 authentic -- and there is no way
around needing it. Temperature scaling minimises negative log-likelihood against
labels; with no labels there is no loss surface and nothing to minimise.

--------------------------------------------------------------------------
THIS HAS NEVER BEEN RUN ON REAL DATA, because no labelled held-out set exists
in this project. Real weights landed 2026-08-29 and removed one of the two
blockers CLAUDE.md named for calibration; the labelled set is the other, and it
is still missing. The evaluation sets already assessed for licensing (FF++,
DeepfakeBench, DFDC) are gated, non-commercial, or both.

The fitter itself is verified against synthetic data whose true temperature is
known by construction -- see tests/test_calibration_fit.py, which distorts a
calibrated set by a known factor and checks the fit recovers it. That tests the
optimiser, and says nothing about any real model.

Do NOT run this on unlabelled data, on the training set, or on scores from the
stub backend, and do not hand-pick a T that "looks right". An invented
temperature is worse than the current 1.0: 1.0 is visibly the identity and reads
as "nothing applied", while a plausible 1.7 reads as measured and nothing in
this system could contradict it.
--------------------------------------------------------------------------

On success it prints the fitted temperature and the before/after diagnostics,
and shows the exact edit to make in src/df/inference/calibration.py. It does not
write that file itself: changing how every score in the service is computed is a
reviewed commit, not a side effect of running a script.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from df.inference.calibration import (  # noqa: E402
    Temperature,
    expected_calibration_error,
    fit_temperature,
)


def load(path: pathlib.Path) -> tuple[list[float], list[int]]:
    logits: list[float] = []
    labels: list[int] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rec = json.loads(line)
            logits.append(float(rec["logit"]))
            labels.append(int(rec["label"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"{path}:{n}: {exc}") from exc
    return logits, labels


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True, type=pathlib.Path,
                    help="JSONL of {logit, label} from a HELD-OUT labelled set")
    ap.add_argument("--model", required=True, choices=("face", "audio"),
                    help="which temperature this fit is for; they are separate")
    ap.add_argument("--describe", default="",
                    help="what the held-out set is -- goes in the audit trail")
    args = ap.parse_args()

    logits, labels = load(args.scores)
    described = args.describe or f"{args.scores.name} ({len(logits)} items)"

    n_pos = sum(labels)
    print(f"held-out set : {len(logits)} items, {n_pos} manipulated, "
          f"{len(labels) - n_pos} authentic")
    if len(logits) < 200:
        # Not a hard failure: a small set is still better than none. But a
        # temperature fitted on a handful of items is noise wearing a decimal
        # point, and the audit trail should say so out loud.
        print(f"WARNING: {len(logits)} items is a thin basis for a calibration; "
              "the fitted value will carry more variance than it appears to")

    before = Temperature(1.0, fitted_on="identity", fitted=False)
    fitted = fit_temperature(logits, labels, fitted_on=described)

    ece_before = expected_calibration_error([_sigmoid(z) for z in logits], labels)
    ece_after = expected_calibration_error(
        [_sigmoid(z / fitted.value) for z in logits], labels
    )

    print(f"\nfitted T     : {fitted.value:.6f}")
    print(f"scheme       : {fitted.scheme}  (was {before.scheme})")
    print(f"ECE before   : {ece_before:.6f}")
    print(f"ECE after    : {ece_after:.6f}")
    if ece_after > ece_before:
        # Worth saying rather than burying: NLL is the objective and ECE is the
        # diagnostic, so they can disagree. When they do, the fit is not
        # obviously an improvement and someone should look before shipping it.
        print("NOTE: ECE got worse even though NLL improved. That can happen -- "
              "they are different objectives -- but do not ship this without "
              "understanding why.")

    const = "FACE_TEMPERATURE" if args.model == "face" else "AUDIO_TEMPERATURE"
    print(f"\nTo adopt, edit src/df/inference/calibration.py:\n")
    print(f"    {const} = Temperature(")
    print(f"        value={fitted.value:.6f},")
    print(f"        fitted_on={described!r},")
    print(f"        fitted=True,")
    print(f"    )")
    print("\nThen re-read CLAUDE.md: this is a LAUNCH SNAPSHOT, it is not tied "
          "to any\nrecalibration pipeline, and it is still not "
          "'production-validated calibration'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
