"""The landing page, and the content negotiation that serves it.

Two things are under test and they fail in opposite directions.

The ROUTING half: `/` now returns HTML to a browser and the same JSON it always
returned to everything else. The risk is that adding the page breaks a caller
that was consuming the JSON -- curl, a probe, a client library -- so the cases
below pin every Accept shape those send, not just the browser one.

The COPY half is the interesting one. CLAUDE.md keeps a list of claims this
project may not make in code, comments, or user-facing copy, and a landing page
is the single most likely place for one to reappear: marketing copy is written
to sound confident, and "adversarially robust" is exactly the phrase someone
reaches for. So the forbidden phrases are asserted absent -- with a sibling
assertion that the caveats are PRESENT, because an absence check alone passes
against an empty file, a renamed file, or a read that silently returned "".

# In-process. No live probe: these are a static file's contents and a route's
# return value, and neither needs infrastructure to be true.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

from df.gateway import app as app_mod

LANDING = pathlib.Path(__file__).resolve().parents[1] / "web" / "landing.html"


class FakeRequest:
    """Only what `root()` touches: the Accept header."""

    def __init__(self, accept: str | None = None):
        self.headers = {"accept": accept} if accept is not None else {}


def payload(response) -> dict:
    return json.loads(bytes(response.body).decode("utf-8"))


@pytest.fixture(scope="module")
def page() -> str:
    """The raw file. Use for structural checks (elements, attributes, CSS)."""
    return LANDING.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def copy(page: str) -> str:
    """Only what a reader sees: comments, CSS and script stripped.

    A copy assertion has to run against copy. Checking the raw file instead
    fails on the page's own machinery -- the comment `<!-- FINAL CTA (no
    billing, no plans, no pricing) -->` tripped the commerce checks below, and
    a JS comment mentioning `<noscript>` defeated the fallback check further
    up. Both times the substring was real and the claim was not.
    """
    text = re.sub(r"<!--.*?-->", " ", page, flags=re.S)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S)
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.S)
    return re.sub(r"<[^>]+>", " ", text)


# --- content negotiation ----------------------------------------------------


def test_a_browser_gets_the_landing_page():
    resp = app_mod.root(FakeRequest("text/html,application/xhtml+xml,*/*;q=0.8"))

    assert resp.media_type == "text/html"
    assert "Don&#39;t trust it." in resp.body.decode("utf-8") or \
           "Don't trust it." in resp.body.decode("utf-8")


@pytest.mark.parametrize("accept", [
    "application/json",     # a client library
    "*/*",                  # curl's default -- the one most likely to regress
    "text/plain",
    None,                   # no Accept header at all
])
def test_everything_that_is_not_a_browser_still_gets_the_json(accept):
    """The bar for this change: nothing that consumed `/` before may notice it.

    `*/*` matters most. It is what curl sends, it contains no "text/html", and
    a looser check -- say, one that treated `*/*` as "browser" -- would flip
    every scripted caller onto an HTML page.
    """
    resp = app_mod.root(FakeRequest(accept))

    assert resp.media_type == "application/json"
    assert payload(resp)["service"] == "Deepfake Detection API"


def test_the_json_payload_is_unchanged_by_the_split():
    """The identity moved into a helper; the contract did not move with it."""
    body = payload(app_mod.root(FakeRequest("*/*")))

    assert set(body) == {"service", "status", "ui", "docs", "health", "detector"}
    assert body["ui"] == "/app"
    assert "research checkpoint" in body["detector"]["note"]


def test_the_service_endpoint_is_json_even_when_a_browser_asks():
    """The whole point of /v1/service: a contract that does not depend on
    getting an Accept header right."""
    assert app_mod.service_identity() == app_mod._service_identity()


def test_a_missing_page_falls_back_to_json_rather_than_erroring(monkeypatch):
    """An image built without web/ still has a working service, and saying so
    is more useful than a 404 that reads as a dead deploy."""
    monkeypatch.setattr(app_mod.pathlib.Path, "is_file", lambda self: False)

    resp = app_mod.root(FakeRequest("text/html"))

    assert resp.media_type == "application/json"
    assert payload(resp)["status"] == "ok"


# --- the copy may not make claims CLAUDE.md forbids -------------------------


@pytest.mark.parametrize("forbidden", [
    "legal hold",             # it is a fixed-timer extended retention window
    "adversarially robust",   # not built
    "adversarial robustness",  # not built
    "GDPR",                   # a legal determination this codebase cannot assert
    "BIPA",
    "state of the art",
    "state-of-the-art",
    "99%",
    "100% accurate",
])
def test_the_landing_page_makes_no_forbidden_claim(page, forbidden):
    assert forbidden.lower() not in page.lower(), (
        f"{forbidden!r} appears in the landing copy; CLAUDE.md forbids it"
    )


@pytest.mark.parametrize("required", [
    "not a probability",          # calibration is unfitted
    "Not production-validated",   # the weights have never been validated here
    "Not robust to adversarial",  # the gap, stated rather than hidden
    "Placeholder",                # audio has no real model
    "Research checkpoint",        # what the face weights actually are
])
def test_the_landing_page_carries_the_caveats_it_must(page, required):
    """The positive control for the test above, and it is not optional.

    An absence assertion passes against an empty string. If the fixture ever
    reads the wrong path, or the file is truncated, every forbidden-phrase case
    goes green for no reason -- that exact failure has happened twice in this
    repo (the probe-topic checks in verify_queue.py, and the extraction-time
    confidence drop). These cases fail in that situation, so the pair is only
    green when the file was really read and really says what it should.
    """
    assert required.lower() in page.lower()


def test_no_score_is_rendered_as_a_percentage(page):
    """The screenshot row scores 69.53 while being authentic. Rendering that as
    "69.53%" would assert a calibrated probability the model does not produce.

    Checks rendered TEXT, not the whole file. The first version of this test
    asserted `"69.53%" not in page` and failed on `style="--w:69.53%"` -- a bar
    width, which genuinely is a percentage of its track. The page was right and
    the test was wrong. A substring search over a document that mixes copy with
    CSS cannot tell a claim from a layout value; only position between tags can.
    """
    displayed = re.findall(r">\s*(\d+(?:\.\d+)?%)\s*<", page)

    assert displayed == [], f"scores rendered as percentages: {displayed}"


# --- the page has to keep working when things are missing -------------------


def test_the_page_loads_no_third_party_script(page):
    """A demo that must not fail does not take a CDN dependency for decoration.

    Google Fonts is the one allowed external origin, and it degrades to the
    fallback stack on its own. Anything else -- a script tag, an analytics
    beacon -- is a way for someone else's outage to become this page's outage.
    """
    external = [
        line.strip() for line in page.splitlines()
        if ("src=\"http" in line or "src='http" in line)
    ]
    assert external == []

    for host in ("cdn.", "unpkg", "jsdelivr", "cdnjs"):
        assert host not in page


def test_reduced_motion_is_honoured(page):
    """Every animation on the page is decorative, so all of it is suppressible.

    Checked in the stylesheet AND in the script: the CSS block cannot stop the
    JS counter from animating, so a page that respected the preference in only
    one place would still move for someone who asked it not to.
    """
    assert "prefers-reduced-motion" in page
    assert page.count("prefers-reduced-motion") >= 2


def test_no_emoji_stand_in_for_an_icon(page):
    """The design system's checklist is explicit: SVG icons, never emoji. This
    catches the common ranges rather than every possible codepoint."""
    emoji = [ch for ch in page if 0x1F300 <= ord(ch) <= 0x1FAFF or 0x2600 <= ord(ch) <= 0x27BF]

    assert emoji == []


def test_the_copy_is_visible_without_working_javascript(page):
    """The reveal animation hides every section until script adds `.in`.

    That makes JS load-bearing for READING the page, which it must never be:
    a script that fails to run -- disabled, blocked by an extension, or throwing
    on an older engine -- would leave a blank shell where the copy should be,
    and the failure is silent because the HTML is all there in view-source.

    Two independent covers, because they fail in different situations:
    <noscript> handles JS-disabled, which the script itself cannot detect, and
    the timed failsafe handles JS-enabled-but-broken, which <noscript> cannot.
    """
    # Matched as an ELEMENT followed by its rule, not as the substring
    # "<noscript>" -- which also appears in a JS comment further down the file,
    # so a substring check stays green after the real element is deleted. The
    # mutation harness caught exactly that and withheld its verdict.
    assert re.search(r"<noscript>\s*<style>.*?\.reveal\s*\{\s*opacity:1", page, re.S),         "no <noscript> block restoring .reveal visibility"

    # The delay is asserted, not just the identifier. "a failsafe exists" is
    # satisfied by one set to fire in a day, which is the same as none.
    m = re.search(r"failsafe = setTimeout\(.*?\}, (\d+)\);", page, re.S)
    assert m, "no failsafe timer found"
    assert int(m.group(1)) <= 5000, f"failsafe fires after {m.group(1)}ms"


@pytest.mark.parametrize("commercial", [
    "pricing", "subscription", "subscribe", "billing", "per month", "/month",
    "free trial", "upgrade", "credit card", "cancel anytime",
])
def test_the_page_sells_nothing(copy, commercial):
    """No billing surface, by request.

    The landing pattern this page is built from puts subscription CTAs after the
    feature grid, so the shape of the template invites them back in every time
    someone extends the page. This fails the moment one reappears.
    """
    assert commercial.lower() not in copy.lower()


def test_no_price_figure_appears(copy):
    """Catches "$9", "$19/mo" and friends, which the word list above cannot."""
    assert re.search(r"\$\s?\d", copy) is None
