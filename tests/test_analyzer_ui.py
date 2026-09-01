"""The analyzer page at /app: does it say what the API said, in plain words.

The result display was simplified because a non-specialist could not read it: a
bare `0.79 / 100` with nothing indicating which end was bad, then ~150 words of
advisory prose, then six boxes including `temperature.v1:unfitted` and a weights
hash. Simplifying that is the easy half. The hard half is that "simpler" must
not become "more confident than the model earns", so the checks below pull in
both directions on purpose:

  * every advisory the API attaches is still rendered, verbatim;
  * every Band the server can emit has plain-language text, because a MISSING
    one silently falls back to the `undetermined` copy -- which would tell
    someone "could not analyse this" about a photo that was analysed fine;
  * the scale says which end is which;
  * the copy does not assert what the image IS, only what was observed.

These are source-level checks. The rendering is JavaScript, so pytest cannot
execute it -- what a browser actually produced is recorded in the commit that
introduced this file, verified against real API payloads for the single-face,
multi-face-with-gating, screenshot and undecodable cases.

# In-process, static. No live probe: this asserts the contents of a shipped
# file, and the browser-side behaviour it cannot see is checked by hand.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from df.bands import Band

ANALYZER = pathlib.Path(__file__).resolve().parents[1] / "web" / "index.html"


@pytest.fixture(scope="module")
def page() -> str:
    return ANALYZER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def copy(page: str) -> str:
    """Everything a reader can see, with the commentary removed.

    NOT the landing page's fixture, and the difference matters. There, the copy
    was in the markup and `<script>` was machinery worth stripping. Here the
    user-facing sentences live INSIDE the script -- `BANDS`, `PLAIN`, and the
    template literals in `render()` -- so stripping script bodies removes
    exactly the text under test. Written the landing page's way first, and three
    checks failed against copy that was plainly there.

    So: strip HTML comments, CSS, and code comments (which discuss the very
    words these tests forbid), and keep the rest.
    """
    text = re.sub(r"<!--.*?-->", " ", page, flags=re.S)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.S)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    # Line comments only where the line starts with them, so "https://" and the
    # regexes in this file's own source survive.
    text = re.sub(r"(?m)^\s*//.*$", " ", text)
    return re.sub(r"<[^>]+>", " ", text)


# --- the UI must cover every verdict the server can produce ------------------


@pytest.mark.parametrize("band", [b.value for b in Band])
def test_every_band_has_a_headline(page, band):
    """A band absent from BANDS falls through to `BANDS.undetermined`.

    That failure is silent and it lies in the worst direction: a real verdict
    would be shown as "Could not analyse this". Parametrised off the server's
    own enum, so adding a Band in bands.py fails here until the UI catches up
    rather than shipping a mislabelled result.
    """
    assert re.search(rf"\b{band}:\s*\[", page), f"{band} missing from BANDS"


@pytest.mark.parametrize("band", [b.value for b in Band])
def test_every_band_has_plain_language(page, band):
    """Same argument for the sentence under the headline."""
    plain = page.split("const PLAIN = {", 1)[1].split("};", 1)[0]

    assert re.search(rf"\b{band}:", plain), f"{band} missing from PLAIN"


# --- simplifying must not delete the caveats --------------------------------


def test_every_advisory_from_the_api_is_rendered(page):
    """The advisories are the fail-closed mechanism CLAUDE.md describes. They
    moved into a collapsed block; they were not dropped, and nothing filters or
    truncates the array on the way."""
    assert "(d.advisories || []).map" in page

    rendered = page.split("(d.advisories || []).map", 1)[1][:200]
    assert "esc(a)" in rendered, "advisory text is not escaped"
    assert "slice" not in rendered, "advisory text is being truncated"


def test_the_advisories_are_reachable_not_hidden(page):
    """Collapsed behind a disclosure is fine; removed from the document is not,
    and neither is `display:none` with no way to open it."""
    assert "<details>" in page and "<summary>" in page
    assert 'class="adv"' in page


def test_the_scale_says_which_end_is_which(page):
    """The bug that prompted this: `0.79 / 100` with no direction. A number
    whose polarity the reader has to guess is worse than no number.

    Asserted against the SCALE MARKUP, not the phrases. The first version
    checked `"no signs" in copy` and `"strong signs" in copy` -- both of which
    also occur in the PLAIN sentences ("No signs of face manipulation were
    found", "Strong signs consistent with manipulation"), so it passed with the
    labels deleted. The mutation harness reported NO-OP and that is how it was
    caught; nothing about the test's name or its green tick would have shown it.
    """
    assert re.search(r'class="ends"><span><b>0</b>.*?<b>100</b>', page, re.S), \
        "the meter has no 0/100 end labels, so the score has no stated polarity"


def test_the_score_is_not_called_a_percentage(copy):
    """Calibration is unfitted, so the score is not a probability. The old
    display implied a percentage by showing `/ 100` alone."""
    assert "not a percentage" in copy.lower()


# --- the copy describes observations, not the world -------------------------


@pytest.mark.parametrize("overclaim", [
    "this photo is real",
    "this photo is fake",
    "is genuine",
    "is authentic",       # "Looks authentic" is fine; "is authentic" is not
    "confirmed",
    "proof that",
    "guaranteed",
    "100% accurate",
    "adversarially robust",
    "production-validated",
    "legal hold",
])
def test_the_copy_does_not_assert_what_the_image_is(copy, overclaim):
    """`Looks authentic` and `No signs were found` are claims about the
    detector. `This photo is real` is a claim about the world, and an
    uncalibrated research checkpoint cannot make it."""
    assert overclaim.lower() not in copy.lower()


@pytest.mark.parametrize("required", [
    "not proof",             # the standing caveat
    "research checkpoint",   # what the weights actually are
    "signs",                 # the hedge that makes the headline an observation
])
def test_the_hedges_are_actually_present(copy, required):
    """Positive control. Every check above is an absence assertion, and those
    all pass against an empty string -- so if the fixture ever reads the wrong
    file, these fail instead of the suite going quietly green.
    """
    assert required.lower() in copy.lower()


def test_undetermined_is_not_described_as_a_clean_result(page):
    """CLAUDE.md is emphatic that "could not decode" is not "nothing wrong".
    The plain-language line for `undetermined` has to carry that distinction,
    because the headline alone reads like a mild negative result."""
    plain = page.split("const PLAIN = {", 1)[1].split("};", 1)[0]
    undetermined = plain.split("undetermined:", 1)[1]

    assert "never analysed" in undetermined or "not the" in undetermined
