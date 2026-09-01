"""Characterise the detector's behaviour on a set, instead of on anecdotes.

WHAT THIS CAN AND CANNOT MEASURE, because the difference decides what the output
is allowed to say.

CAN:
  * the **undetermined rate** -- how often a real photograph of a real face comes
    back with nothing scored. This is the demo risk and the deployment risk, and
    it needs no labels: every input here is known to contain a face.
  * the **score distribution on known-authentic images**. Every source is a
    public-domain photograph, so a high score is a false positive by
    construction. That bounds the false-positive behaviour from below.
  * the **effect of compression and resampling in isolation**. Same face, same
    framing, only the encoding varies. This is the one claim in DEMO-NOTES that
    rested on a single sample ("a screenshot scored 69.53"), and a controlled
    sweep either supports it with n>1 or kills it.
  * **detection stability** -- how many faces Haar reports in a single-person
    portrait, which is where the confidence gate earns or wastes its place.

CANNOT:
  * a true-positive rate, precision, recall or accuracy. There are no real
    deepfakes here. Every "manipulated" sample is a feathered composite this
    script builds, which is a different artefact class from a GAN or diffusion
    face swap. A number computed against it would describe this script, not the
    world.
  * anything about calibration. The temperature is unfitted; the scores are not
    probabilities and no amount of unlabelled data changes that.

So the output is a characterisation, not a benchmark, and it says so.

Usage (needs a running service; the deployed URL is fine):

    python scripts/evaluate.py --corpus <dir-of-jpegs> --url https://host
    python scripts/evaluate.py --corpus <dir> --out docs/EVALUATION.md

Paced for the ingress limiter (30 capacity, 0.5/s refill by default) and it
backs off on 429 rather than pretending the run succeeded.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.request

# --- variants ---------------------------------------------------------------
#
# Derived from each source so the ONLY difference is the encoding. Comparing
# different photographs would confound the thing under test with the subject.

VARIANTS = [
    ("original", None),
    ("jpeg_q70", ("jpeg", 70)),
    ("jpeg_q40", ("jpeg", 40)),
    ("resize_50", ("resize", 0.5)),
    ("screenshot_sim", ("screenshot", None)),
]


def build_variant(img, spec):
    import cv2

    if spec is None:
        return img
    kind, arg = spec
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, arg])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img
    if kind == "resize":
        h, w = img.shape[:2]
        return cv2.resize(img, (max(1, int(w * arg)), max(1, int(h * arg))),
                          interpolation=cv2.INTER_AREA)
    if kind == "screenshot":
        # What a phone screenshot of a photo actually does: resample to screen
        # size, then re-encode lossily. This is the case DEMO-NOTES says scores
        # like a manipulation.
        h, w = img.shape[:2]
        scale = 900 / max(h, w)
        small = build_variant(img, ("resize", scale)) if scale < 1 else img
        return build_variant(small, ("jpeg", 60))
    raise ValueError(kind)


def composite(a, b):
    """A feathered-ellipse face splice: two public-domain portraits blended.

    Deliberately NOT called a deepfake. It is a crude local manipulation, and
    the point of including it is that the pipeline should react to *something*
    -- not to claim detection accuracy against generative models.
    """
    import cv2
    import numpy as np

    h, w = a.shape[:2]
    b = cv2.resize(b, (w, h))
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (w // 2, int(h * 0.42)), (int(w * 0.20), int(h * 0.17)),
                0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=w * 0.03)[:, :, None]
    return (a.astype(np.float32) * (1 - mask) + b.astype(np.float32) * mask).astype("uint8")


# --- the service ------------------------------------------------------------


class Client:
    def __init__(self, base: str, pace: float):
        self.base = base.rstrip("/")
        self.pace = pace
        self.last = 0.0

    def _wait(self):
        gap = time.time() - self.last
        if gap < self.pace:
            time.sleep(self.pace - gap)
        self.last = time.time()

    def _post(self, path, payload, retries=4):
        for attempt in range(retries):
            self._wait()
            req = urllib.request.Request(
                self.base + path, data=json.dumps(payload).encode(), method="POST",
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = r.read()
                    return json.loads(body) if body else None
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    back = int(e.headers.get("Retry-After") or 2) + 2 * attempt
                    print(f"    429, backing off {back}s", file=sys.stderr)
                    time.sleep(back)
                    continue
                raise
        raise RuntimeError(f"gave up on {path} after {retries} attempts (rate limited)")

    def analyse(self, jpeg: bytes, timeout: float = 120.0) -> dict:
        job = self._post("/v1/jobs", {"media_type": "image", "content_type": "image/jpeg"})
        jid = job["job_id"]
        boundary = "----dfeval"
        head = "".join(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
            for k, v in job["upload_fields"].items()
        ) + (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
             f'filename="e.jpg"\r\nContent-Type: image/jpeg\r\n\r\n')
        body = head.encode() + jpeg + f"\r\n--{boundary}--\r\n".encode()
        urllib.request.urlopen(urllib.request.Request(
            job["upload_url"], data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        ), timeout=180).read()

        t0 = time.time()
        self._post(f"/v1/jobs/{jid}/uploaded", {})
        while time.time() - t0 < timeout:
            try:
                with urllib.request.urlopen(self.base + f"/v1/jobs/{jid}", timeout=30) as r:
                    res = json.loads(r.read())
                if res["status"] in ("complete", "failed"):
                    res["_seconds"] = round(time.time() - t0, 2)
                    return res
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    raise
            # 0.25s, not 2s: a coarse poll quantises the reported duration into
            # multi-second buckets and makes the timing column meaningless. Measured
            # the hard way -- a 3s poll elsewhere made a 2.5s job look like 8.3s and
            # I reported a latency regression that did not exist.
            time.sleep(0.25)
        return {"job_id": jid, "status": "timeout", "_seconds": round(time.time() - t0, 2)}


# --- reporting --------------------------------------------------------------


def summarise(rows: list[dict]) -> str:
    out: list[str] = []
    scored = [r for r in rows if r["score"] is not None]
    undet = [r for r in rows if r["band"] == "undetermined"]

    out.append("## What was measured\n")
    out.append(f"- **{len(rows)} runs** over {len({r['source'] for r in rows})} "
               f"public-domain source photographs and their derived variants.")
    out.append(f"- **Undetermined: {len(undet)}/{len(rows)} "
               f"({100 * len(undet) / max(1, len(rows)):.0f}%)** — a real photograph of a "
               f"real face that came back with nothing scored.")
    if scored:
        vals = [r["score"] for r in scored]
        out.append(f"- Scores on the {len(scored)} that were scored: "
                   f"min {min(vals):.2f}, median {statistics.median(vals):.2f}, "
                   f"max {max(vals):.2f}.")
    out.append("")

    out.append("### Undetermined rate by variant\n")
    out.append("| variant | runs | undetermined | median score |")
    out.append("|---|---|---|---|")
    by_variant: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)
    for name, _ in VARIANTS + [("composite", None)]:
        group = by_variant.get(name, [])
        if not group:
            continue
        gs = [r["score"] for r in group if r["score"] is not None]
        nd = sum(1 for r in group if r["band"] == "undetermined")
        med = f"{statistics.median(gs):.2f}" if gs else "—"
        out.append(f"| {name} | {len(group)} | {nd} | {med} |")
    out.append("")

    authentic = [r for r in rows if r["variant"] != "composite" and r["score"] is not None]
    if authentic:
        high = [r for r in authentic if r["score"] > 60]
        out.append("### False positives on known-authentic images\n")
        out.append(f"Every non-composite source here is a public-domain photograph, so any "
                   f"high score is a false positive by construction. "
                   f"**{len(high)}/{len(authentic)} scored above 60** "
                   f"({100 * len(high) / len(authentic):.0f}%).")
        if high:
            out.append("")
            for r in sorted(high, key=lambda r: -r["score"])[:8]:
                out.append(f"- `{r['source']}` / {r['variant']}: **{r['score']:.2f}** "
                           f"({r['band']})")
        out.append("")

    out.append("### Every run\n")
    out.append("| source | variant | band | score | faces | coverage | s |")
    out.append("|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["source"], r["variant"])):
        score = f"{r['score']:.2f}" if r["score"] is not None else "—"
        cov = f"{r['coverage']:.2f}" if r["coverage"] is not None else "—"
        out.append(f"| {r['source']} | {r['variant']} | {r['band']} | {score} | "
                   f"{r['faces_total']} | {cov} | {r['_seconds']} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=pathlib.Path)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--pace", type=float, default=4.0,
                    help="seconds between rate-limited calls (limiter refills 0.5/s)")
    ap.add_argument("--limit", type=int, default=0, help="cap runs, for a smoke test")
    args = ap.parse_args()

    import cv2

    sources = sorted(p for p in args.corpus.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not sources:
        print(f"no images in {args.corpus}")
        return 2

    client = Client(args.url, args.pace)
    jobs: list[tuple[str, str, bytes]] = []
    loaded = {p.stem: cv2.imread(str(p)) for p in sources}
    loaded = {k: v for k, v in loaded.items() if v is not None}

    for name, img in loaded.items():
        for vname, spec in VARIANTS:
            variant = build_variant(img, spec)
            ok, buf = cv2.imencode(".jpg", variant, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                jobs.append((name, vname, buf.tobytes()))

    names = list(loaded)
    for i in range(len(names) - 1):
        blended = composite(loaded[names[i]], loaded[names[i + 1]])
        ok, buf = cv2.imencode(".jpg", blended, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            jobs.append((f"{names[i]}+{names[i+1]}", "composite", buf.tobytes()))

    if args.limit:
        jobs = jobs[: args.limit]

    print(f"{len(jobs)} runs against {args.url} (pace {args.pace}s)")
    rows: list[dict] = []
    for i, (source, variant, blob) in enumerate(jobs, 1):
        try:
            res = client.analyse(blob)
        except Exception as e:
            print(f"  [{i}/{len(jobs)}] {source}/{variant}: {type(e).__name__}: {e}")
            continue
        ev = res.get("face_evidence") or {}
        rows.append({
            "source": source, "variant": variant,
            "band": res.get("band") or res.get("status"),
            "score": res.get("aggregate_score"),
            "faces_total": ev.get("faces_total", 0),
            "discarded": ev.get("detections_discarded", 0),
            "coverage": res.get("coverage"),
            "_seconds": res.get("_seconds"),
        })
        r = rows[-1]
        print(f"  [{i}/{len(jobs)}] {source:22} {variant:15} "
              f"{r['band']:20} {r['score'] if r['score'] is not None else '—'}")

    report = summarise(rows)
    if args.out:
        header = (
            "# Detector characterisation\n\n"
            "Generated by `scripts/evaluate.py` against the deployed service.\n\n"
            "**This is a characterisation, not a benchmark.** Every source is a\n"
            "public-domain photograph, so there are no labelled deepfakes here and no\n"
            "accuracy, precision or recall can be computed. What it does measure: how\n"
            "often a real face yields no verdict, how known-authentic images score, and\n"
            "what compression and resampling do in isolation. The `composite` rows are\n"
            "feathered ellipse blends this script builds — a crude local manipulation,\n"
            "a different artefact class from a generative face swap.\n\n"
            "Calibration is unfitted, so no score here is a probability.\n\n"
        )
        args.out.write_text(header + report + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
        (args.out.with_suffix(".json")).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    else:
        print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
