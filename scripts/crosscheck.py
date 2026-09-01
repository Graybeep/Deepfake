"""End-to-end cross-check of a DEPLOYED service. Fails loudly, exits non-zero.

The live probes in this directory each cover one seam -- the queue, retention,
attribution, the presign policy. This covers the whole surface a user or a
reviewer actually touches, against a real deployment: routing, content
negotiation, the claims on both pages, a real analysis with its advisories, the
many-face cap, and the undecodable path.

Why it exists as a script rather than a pytest file: every assertion here needs
a running service with real weights. pytest covers the logic; this covers the
deployment, and a passing test run has never been evidence that the deployed
thing works.

    python scripts/crosscheck.py --url https://host

NEGATION MATTERS, and it is the reason this file is not a list of substrings.
Some phrases are forbidden outright ("legal hold"). Others are REQUIRED but only
under negation: the landing page must say "Not production-validated", and a flat
`"production-validated" not in page` check fails on the caveat while the page is
correct. That mistake was made three separate times in this project before it
was written down here.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

# Phrases that must never appear in user-facing copy, in any form.
FORBIDDEN = ("legal hold", "adversarially robust", "GDPR", "BIPA",
             "state of the art", "100% accurate")

# Phrases that may appear ONLY when negated. The claim is forbidden; the caveat
# is required. Checked by looking for a negation immediately before the phrase.
NEGATED_ONLY = ("production-validated", "robust to adversarial")

NEGATION = re.compile(r"\b(not|no|never)\b[^.]{0,25}$", re.I)


class Check:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def __call__(self, name: str, ok: object, detail: str = "") -> bool:
        ok = bool(ok)
        self.rows.append((name, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
        return ok

    def report(self) -> int:
        bad = [n for n, ok, _ in self.rows if not ok]
        print(f"\n  {len(self.rows) - len(bad)}/{len(self.rows)} checks passed")
        if bad:
            print("  FAILURES:")
            for n in bad:
                print(f"    - {n}")
        return 1 if bad else 0


def visible_text(html: str) -> str:
    """Copy only: comments, CSS and script bodies removed.

    Correct for the LANDING page, whose copy is in markup. The analyzer keeps
    its user-facing sentences inside <script>, so structural checks on that page
    read the raw source instead -- see the analyzer section below.
    """
    stripped = re.sub(r"<!--.*?-->|<style.*?</style>|<script.*?</script>", " ", html, flags=re.S)
    return re.sub(r"<[^>]+>", " ", stripped)


class Service:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def get(self, path: str, accept: str | None = None) -> tuple[str, str, int]:
        headers = {"Accept": accept} if accept else {}
        req = urllib.request.Request(self.base + path, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode("utf-8", "replace"), r.headers.get("Content-Type", ""), r.status

    def post(self, path: str, payload: dict) -> dict | None:
        req = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
            return json.loads(body) if body else None

    def analyse(self, blob: bytes, content_type: str = "image/jpeg",
                media_type: str = "image", timeout: float = 150.0) -> dict:
        job = self.post("/v1/jobs", {"media_type": media_type, "content_type": content_type})
        assert job is not None
        boundary = "----dfcross"
        head = "".join(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
            for k, v in job["upload_fields"].items()
        ) + (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
             f'filename="f.bin"\r\nContent-Type: {content_type}\r\n\r\n')
        body = head.encode() + blob + f"\r\n--{boundary}--\r\n".encode()
        urllib.request.urlopen(urllib.request.Request(
            job["upload_url"], data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        ), timeout=180).read()

        started = time.time()
        self.post(f"/v1/jobs/{job['job_id']}/uploaded", {})
        while time.time() - started < timeout:
            with urllib.request.urlopen(
                    self.base + f"/v1/jobs/{job['job_id']}", timeout=30) as r:
                res = json.loads(r.read())
            if res["status"] in ("complete", "failed"):
                res["_seconds"] = round(time.time() - started, 2)
                return res
            # Fine-grained on purpose: a 3s poll quantised a 2.5s job into an
            # 8.3s reading and produced a reported "latency regression" that did
            # not exist.
            time.sleep(0.25)
        return {"status": "timeout", "_seconds": round(time.time() - started, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--portrait", type=pathlib.Path,
                    help="a single-face JPEG (skips the analysis checks if absent)")
    ap.add_argument("--group", type=pathlib.Path,
                    help="a many-face JPEG, to exercise the face cap")
    args = ap.parse_args()

    svc = Service(args.url)
    check = Check()

    print("routing and content negotiation")
    landing, ctype, _ = svc.get("/", accept="text/html")
    check("/ serves the landing page to a browser",
          "text/html" in ctype and "trust it" in landing.lower(), ctype.split(";")[0])
    as_json, ctype_json, _ = svc.get("/", accept="*/*")
    check("/ still serves JSON to a script (*/*)",
          "application/json" in ctype_json and "Deepfake Detection API" in as_json)
    service_json, _, _ = svc.get("/v1/service", accept="text/html")
    check("/v1/service is JSON even for a browser",
          json.loads(service_json)["ui"] == "/app")
    analyzer, ctype_app, _ = svc.get("/app", accept="text/html")
    check("/app serves the analyzer", "text/html" in ctype_app and "const PLAIN" in analyzer)
    _, _, docs_status = svc.get("/docs", accept="text/html")
    check("/docs reachable", docs_status == 200)
    health = json.loads(svc.get("/healthz")[0])
    check("/healthz reports every worker up",
          health["ok"] and all(health["workers"].values()), str(health["workers"]))
    whoami = json.loads(svc.get("/v1/whoami")[0])
    check("/v1/whoami resolves a client identity",
          whoami["identity"].startswith("ip:") or whoami["identity"].startswith("key:"),
          whoami["identity"])

    print("\nlanding page claims")
    copy = visible_text(landing)
    check("audio is not advertised", "audio" not in copy.lower())
    check("the pipeline count is not stale",
          "Two routes in" not in landing and "Three routes in" not in landing)
    check("Image is marked live in the app", "Live in the app" in landing)
    check("Video is marked API only", "API only" in landing)
    for phrase in FORBIDDEN:
        check(f"no forbidden claim: {phrase!r}", phrase.lower() not in copy.lower())
    for phrase in NEGATED_ONLY:
        hits = [m.start() for m in re.finditer(re.escape(phrase), copy, re.I)]
        negated = all(NEGATION.search(copy[max(0, i - 45):i]) for i in hits)
        check(f"{phrase!r} appears only under negation", negated,
              f"{len(hits)} occurrence(s)" if hits else "absent")
    check("the caveats are present",
          "not a probability" in copy.lower() and "research checkpoint" in copy.lower())

    print("\nanalyzer claims")
    check("the picker accepts images only", 'accept="image/*"' in analyzer)
    check("only media_type=image can be sent",
          analyzer.count("media_type:") == 1 and "media_type: 'image'" in analyzer)
    check("video is not promised", "not available in this build" in analyzer)
    check("advisories are rendered and reachable",
          "(d.advisories || []).map" in analyzer and "<details>" in analyzer)
    check("the score scale carries 0/100 end labels",
          bool(re.search(r'class="ends"><span><b>0</b>.*?<b>100</b>', analyzer, re.S)))

    if args.portrait and args.portrait.is_file():
        print("\na real single-face analysis")
        res = svc.analyse(args.portrait.read_bytes())
        check("analysis completes", res["status"] == "complete", f"{res.get('_seconds')}s")
        check("a score was produced", res.get("aggregate_score") is not None,
              str(res.get("aggregate_score")))
        # Retried, because a false here is EXPECTED for a moment. The router
        # records the decision (status=complete) before deleting the media, so a
        # crash between them loses the photo and keeps the verdict rather than
        # the reverse. A fast poll can therefore see complete with the delete
        # still pending -- which is correct, not a failure.
        deleted = res.get("media_deleted")
        if not deleted:
            for _ in range(20):
                time.sleep(0.25)
                with urllib.request.urlopen(
                        svc.base + f"/v1/jobs/{res['job_id']}", timeout=30) as r:
                    deleted = json.loads(r.read()).get("media_deleted")
                if deleted:
                    break
        check("media deleted on completion (retried; the delete follows the "
              "decision by design)", deleted is True)
        check("validation level recorded on the row",
              res.get("model_validation") in {"placeholder", "research-checkpoint",
                                              "production-validated"},
              str(res.get("model_validation")))
        check("calibration provenance recorded", bool(res.get("calibration")),
              str(res.get("calibration")))
        advisories = " ".join(res.get("advisories") or [])
        check("adversarial advisory attached", "adversarial" in advisories.lower())
        check("model-trust advisory attached",
              "RESEARCH CHECKPOINT" in advisories or "PLACEHOLDER MODEL" in advisories)
        check("uncalibrated advisory attached", "UNCALIBRATED" in advisories)

    if args.group and args.group.is_file():
        print("\nthe many-face cost bound")
        res = svc.analyse(args.group.read_bytes())
        ev = res.get("face_evidence") or {}
        check("a many-face photo completes", res["status"] == "complete",
              f"{res.get('_seconds')}s")
        check("the cap is applied and reported",
              (ev.get("faces_capped") or 0) > 0 and ev.get("max_faces_scored"),
              f"scored={ev.get('faces_total')} capped={ev.get('faces_capped')}")
        check("capped is reported apart from discarded",
              ev.get("faces_capped") != ev.get("detections_discarded"),
              f"capped={ev.get('faces_capped')} discarded={ev.get('detections_discarded')}")

    print("\nthe undecodable path")
    res = svc.analyse(b"this is definitively not an image")
    check("undecodable input routes to undetermined", res.get("band") == "undetermined")
    check("undecodable input invents no score", res.get("aggregate_score") is None)
    check("undecodable input raises MEDIA NOT DECODED",
          any("MEDIA NOT DECODED" in a for a in (res.get("advisories") or [])))

    print("\nafter the whole pass")
    health = json.loads(svc.get("/healthz")[0])
    check("workers still up", health["ok"] and all(health["workers"].values()))

    return check.report()


if __name__ == "__main__":
    raise SystemExit(main())
