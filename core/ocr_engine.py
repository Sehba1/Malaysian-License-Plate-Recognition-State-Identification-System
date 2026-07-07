"""
core.ocr_engine
===============

Optical-character-recognition backend for the Malaysian LPR & SIS project.

This module is the single owner of three concerns that all ten vehicle
processors share:

1.  Backend lifecycle.  A lazy module-level singleton wraps PaddleOCR so the
    expensive model load happens exactly once per Python interpreter, even
    when ten different processors call into this layer.  We deliberately
    refuse to expose the underlying reader handle to callers; everything goes
    through the public functions defined here.  That isolation keeps the
    vehicle processors backend-agnostic — swapping PaddleOCR for another
    classical OCR engine in future would touch this file only.

2.  Lexical contract.  Malaysian civilian plates obey a tightly bounded
    grammar (alpha-prefix, numeric body, optional alpha suffix).  We compile
    this grammar into ``re.Pattern`` objects at import time and expose a
    single ``validate_plate_syntax`` function returning the canonical
    ``(is_valid, confidence, format_label)`` triple consumed by every
    processor.

3.  Recognition robustness.  Raw OCR output is noisy: 'O' becomes '0' in the
    numeric body, '1' becomes 'I' in the alpha prefix, two-row motorcycle
    plates arrive as two separate detections that must be fused vertically.
    We centralise the corrective post-processing here so that every
    processor benefits from identical sanitisation behaviour.

Design rationale: by funnelling all OCR concerns through this module we
preserve the architecture invariant that the ``processors/`` package is
state-agnostic and engine-agnostic — its sole job is computer vision.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section 1 — Lazy PaddleOCR singleton
# ---------------------------------------------------------------------------

# A module-level handle, populated on first ``get_ocr_reader`` call.  We
# refuse to load PaddleOCR at import time because it is a multi-hundred-MB
# initialisation that would make CLI smoke tests of individual processors
# painfully slow.  ``None`` here means "not yet loaded".
_OCR_READER: Any | None = None
_OCR_INIT_FAILED: bool = False


def get_ocr_reader() -> Any | None:
    """Return the cached PaddleOCR reader, instantiating it on first call.

    The reader is configured for vehicle-plate scenarios specifically: text
    line orientation classifier is enabled because Malaysian motorcycle
    plates are two-row, but document-level orientation and unwarping are
    disabled because they assume page-shaped inputs and slow each call by an
    order of magnitude with no quality gain on cropped plate ROIs.

    Returns ``None`` if PaddleOCR cannot be imported or constructed, allowing
    upstream callers to degrade gracefully rather than crash.
    """
    global _OCR_READER, _OCR_INIT_FAILED

    if _OCR_READER is not None:
        return _OCR_READER
    if _OCR_INIT_FAILED:
        return None

    try:
        from paddleocr import PaddleOCR
        _OCR_READER = PaddleOCR(
            use_angle_cls=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            show_log=False,
        )
        LOG.info("PaddleOCR backend initialised.")
        return _OCR_READER
    except Exception as exc:  # noqa: BLE001 — we genuinely want a broad catch here
        LOG.warning("PaddleOCR initialisation failed: %s", exc)
        _OCR_INIT_FAILED = True
        return None


# ---------------------------------------------------------------------------
# Section 2 — Malaysian plate grammar
# ---------------------------------------------------------------------------

# The two civilian-grammar patterns capture roughly 95 percent of standard
# plates.  An "alpha prefix" runs 1-3 letters (state code + serial letters),
# the numeric body is 1-4 digits, and a single trailing alpha character is
# the optional check letter introduced in modern series.
_PATTERN_WITH_SUFFIX: re.Pattern[str] = re.compile(r"^[A-Z]+\d{1,4}[A-Z]$")
_PATTERN_STANDARD: re.Pattern[str] = re.compile(r"^[A-Z]+\d{1,4}$")

# Single-letter state codes recognised by JPJ.  Note that 'E', 'G', 'I', 'O',
# 'U', 'X', 'Y' are reserved or unused on the Peninsula and so are excluded
# from the bonus heuristic — a plate beginning with one of those is almost
# certainly an OCR misread of a neighbour ('B' read as '8', etc.).
_KNOWN_STATE_LETTERS: frozenset[str] = frozenset("ABCDFJKLMNPQRSTVWZH")

# Confusion tables used by ``apply_position_corrections``.  These map *into*
# the canonical glyph: when a digit zone contains a stray letter that is a
# common 7-segment misread, push it back to its digit; when an alpha zone
# contains a stray digit, do the inverse.
_LETTER_TO_DIGIT_FIXES: dict[str, str] = {
    "O": "0", "D": "0", "Q": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}
_DIGIT_TO_LETTER_FIXES: dict[str, str] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}


# ---------------------------------------------------------------------------
# Section 3 — Public sanitisation and validation API
# ---------------------------------------------------------------------------

def strip_to_alphanumeric(raw_text: str) -> str:
    """Collapse arbitrary OCR output to a pure ``[A-Z0-9]+`` token.

    This is the first stage of sanitisation: punctuation, whitespace,
    lowercase letters, and language-specific marks are discarded.  The result
    is suitable as the input to ``apply_position_corrections``.
    """
    if not raw_text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", raw_text.upper().strip())


def _segment_alpha_numeric_zones(token: str) -> tuple[str, str, str]:
    """Split an alphanumeric token into (prefix, body, suffix) by position.

    The Malaysian grammar guarantees a contiguous alpha prefix, a contiguous
    numeric body, and at most one trailing alpha suffix.  We walk the token
    forward to find the longest leading run of letters, then walk backward
    from the tail to find the trailing single letter (if any).  The body is
    whatever sits between.

    This zone-aware split is what enables position-aware confusion fixes:
    once we know which characters *ought* to be letters and which *ought* to
    be digits, the corrections become unambiguous.
    """
    if not token:
        return "", "", ""

    n = len(token)
    # Forward walk: longest leading alphabetic prefix
    prefix_end = 0
    while prefix_end < n and token[prefix_end].isalpha():
        prefix_end += 1

    # Backward walk: a single trailing alpha if present, otherwise none
    suffix_start = n
    if n > prefix_end and token[-1].isalpha():
        suffix_start = n - 1

    prefix = token[:prefix_end]
    body = token[prefix_end:suffix_start]
    suffix = token[suffix_start:]
    return prefix, body, suffix


def apply_position_corrections(raw_text: str) -> str:
    """Apply position-aware character substitutions to a sanitised plate.

    The function is idempotent: calling it twice yields the same output.  It
    presumes ``strip_to_alphanumeric`` has already run.

    Strategy:
        * Alpha zones (prefix and optional suffix) get any stray digits
          mapped to their letter look-alikes via ``_DIGIT_TO_LETTER_FIXES``.
        * The numeric body gets any stray letters mapped to their digit
          look-alikes via ``_LETTER_TO_DIGIT_FIXES``.

    Characters that have no entry in the relevant confusion table are passed
    through unchanged — we never invent corrections we are not confident in.
    """
    token = strip_to_alphanumeric(raw_text)
    if len(token) < 3:
        return token

    prefix, body, suffix = _segment_alpha_numeric_zones(token)

    # The body may legitimately be empty if the upstream OCR fragmented; in
    # that case there is nothing position-aware to correct.
    if not body:
        return token

    fixed_prefix = "".join(_DIGIT_TO_LETTER_FIXES.get(ch, ch) for ch in prefix)
    fixed_body = "".join(_LETTER_TO_DIGIT_FIXES.get(ch, ch) for ch in body)
    fixed_suffix = "".join(_DIGIT_TO_LETTER_FIXES.get(ch, ch) for ch in suffix)

    return fixed_prefix + fixed_body + fixed_suffix


def validate_plate_syntax(plate_text: str) -> tuple[bool, float, str]:
    """Test a sanitised token against the Malaysian civilian grammar.

    Returns a ``(is_valid, confidence, format_label)`` triple where
    ``confidence`` is a heuristic between 0.0 and 1.0 derived from how well
    the token matches the canonical patterns and whether its first character
    is a recognised state-code letter.

    A plate that ends in a letter is given a small confidence edge over the
    suffix-less form because modern Malaysian issuances overwhelmingly carry
    a check letter — a token that does not have one is statistically more
    likely to be missing its tail to OCR error than to be genuinely
    suffix-less.
    """
    token = strip_to_alphanumeric(plate_text)
    if len(token) < 3:
        return False, 0.0, "too_short"

    if _PATTERN_WITH_SUFFIX.match(token):
        bonus = 0.1 if token[0] in _KNOWN_STATE_LETTERS else 0.0
        return True, min(1.0, 0.95 + bonus), "standard_with_suffix"

    if _PATTERN_STANDARD.match(token):
        bonus = 0.1 if token[0] in _KNOWN_STATE_LETTERS else 0.0
        return True, min(1.0, 0.90 + bonus), "standard"

    return False, 0.0, "invalid"


# ---------------------------------------------------------------------------
# Section 4 — Structured OCR detection record
# ---------------------------------------------------------------------------

@dataclass
class _PaddleDetection:
    """In-memory shape of a single PaddleOCR text detection.

    Encapsulating the four-corner box and confidence into a small record
    rather than a tuple makes the two-row fusion logic later in this module
    far more readable than the donor's positional-index access patterns.
    """
    text_raw: str
    confidence: float
    text_clean: str
    bbox: list[list[float]] = field(default_factory=list)
    center_x: float = 0.0
    center_y: float = 0.0
    box_width: float = 0.0
    box_height: float = 0.0


def _parse_paddle_result(raw_result: Any) -> list[_PaddleDetection]:
    """Convert PaddleOCR's nested list output into our structured detections.

    PaddleOCR's return shape has shifted between minor versions (sometimes
    the outermost element is the per-image list, sometimes the per-image
    detections themselves), so we defensively handle both layouts.
    """
    if not raw_result or not isinstance(raw_result, list):
        return []

    primary = raw_result[0] if (raw_result and isinstance(raw_result[0], list)) else raw_result
    if primary is None:
        return []

    detections: list[_PaddleDetection] = []
    for entry in primary:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        try:
            bbox, payload = entry[0], entry[1]
            text_value, conf_value = payload[0], float(payload[1])
        except (TypeError, ValueError, IndexError):
            continue

        cleaned = strip_to_alphanumeric(text_value)
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        detections.append(
            _PaddleDetection(
                text_raw=text_value,
                confidence=conf_value,
                text_clean=cleaned,
                bbox=list(bbox),
                center_x=sum(xs) / 4.0,
                center_y=sum(ys) / 4.0,
                box_width=max(xs) - min(xs),
                box_height=max(ys) - min(ys),
            )
        )
    return detections


# ---------------------------------------------------------------------------
# Section 5 — Single-line and two-row recognition
# ---------------------------------------------------------------------------

def _attempt_single_line_extraction(
    detections: list[_PaddleDetection],
    min_confidence: float = 0.6,
    min_token_length: int = 4,
) -> str | None:
    """Return the strongest detection whose token alone matches the grammar.

    We sort by raw OCR confidence descending and accept the first match that
    passes ``validate_plate_syntax``.  This handles the common case of a
    well-localised single-row plate where PaddleOCR has produced the full
    string in one detection.
    """
    ranked = sorted(detections, key=lambda d: d.confidence, reverse=True)
    for detection in ranked:
        if detection.confidence < min_confidence:
            continue
        if len(detection.text_clean) < min_token_length:
            continue
        is_valid, _, _ = validate_plate_syntax(detection.text_clean)
        if is_valid:
            return detection.text_clean
    return None


def _attempt_two_row_fusion(
    detections: list[_PaddleDetection],
    canvas_width: int,
) -> str | None:
    """Stitch a two-row Malaysian plate back together from independent rows.

    Motorcycle plates carry the prefix on the upper row and the body on the
    lower row.  PaddleOCR detects these as two separate boxes; we fuse them
    by requiring (a) at least two detections of reasonable confidence,
    (b) a vertical centre-to-centre separation above a small floor, and
    (c) sufficient horizontal overlap that the rows belong to the same plate
    rather than two unrelated text blocks in the frame.

    For every viable pair, we try both row orderings, validate the resulting
    concatenation, and keep the variant with the highest combined
    OCR×grammar score.
    """
    eligible = [
        d for d in detections
        if d.confidence > 0.2 and len(d.text_clean) >= 1
    ]
    if len(eligible) < 2:
        return None

    # Sort top-to-bottom so the first element is the upper row by default.
    eligible.sort(key=lambda d: d.center_y)

    best_token: str | None = None
    best_score: float = 0.0
    horizontal_overlap_ceiling = canvas_width * 0.5

    for i, upper in enumerate(eligible[:-1]):
        for lower in eligible[i + 1 : i + 3]:  # only consider near neighbours
            vertical_gap = abs(lower.center_y - upper.center_y)
            horizontal_gap = abs(upper.center_x - lower.center_x)
            if vertical_gap <= 5:
                continue
            if horizontal_gap >= horizontal_overlap_ceiling:
                continue

            # Both orderings — the donor's geometry sort was buggy when rows
            # had near-identical centre_y, so we check both and let grammar
            # arbitrate.
            ordering_a = upper.text_clean + lower.text_clean
            ordering_b = lower.text_clean + upper.text_clean

            for combined in (ordering_a, ordering_b):
                is_valid, grammar_conf, _ = validate_plate_syntax(combined)
                if not is_valid:
                    continue
                combined_score = ((upper.confidence + lower.confidence) / 2.0) * grammar_conf
                if combined_score > best_score:
                    best_token = combined
                    best_score = combined_score

    if best_token and best_score > 0.3:
        return best_token

    # Permissive fallback: even if no fused pair passed the grammar, return
    # the simple top-bottom concatenation of the two most-confident rows so
    # the caller can still display something to the operator.
    top_two = sorted(eligible, key=lambda d: d.confidence, reverse=True)[:2]
    if len(top_two) == 2:
        top_two.sort(key=lambda d: d.center_y)
        naive_combined = top_two[0].text_clean + top_two[1].text_clean
        average_confidence = (top_two[0].confidence + top_two[1].confidence) / 2.0
        if len(naive_combined) >= 3 and average_confidence > 0.3:
            return naive_combined

    return None


def _attempt_best_single_fallback(
    detections: list[_PaddleDetection],
    min_token_length: int = 3,
) -> str | None:
    """Last-resort fallback: return the highest-confidence non-trivial token.

    Used when neither the grammar-strict single-line path nor the two-row
    fusion path produced a result.  This deliberately bypasses the grammar
    check so that partial or unusual plates (military Z-series, special
    series tokens like PUTRAJAYA) can still surface for the orchestrator's
    state-identification step.
    """
    candidates = [d for d in detections if len(d.text_clean) >= 2]
    if not candidates:
        return None
    best = max(candidates, key=lambda d: d.confidence)
    return best.text_clean if len(best.text_clean) >= min_token_length else None


def read_plate_text(plate_image: np.ndarray) -> str | None:
    """Run OCR on a cropped plate image and return a canonical token.

    The function attempts three strategies in priority order:
        1. Single-line extraction (grammar-strict, high-confidence).
        2. Two-row fusion (for motorcycle plates).
        3. Single-line fallback (grammar-lax, for special/military series).

    Returns ``None`` only if every strategy failed or the OCR backend is
    unavailable.
    """
    reader = get_ocr_reader()
    if reader is None or plate_image is None or plate_image.size == 0:
        return None

    # PaddleOCR expects BGR; our pipeline operates in RGB internally.
    if plate_image.ndim == 3:
        ocr_input = cv2.cvtColor(plate_image, cv2.COLOR_RGB2BGR)
    else:
        ocr_input = plate_image

    try:
        # PaddleOCR's ``cls`` keyword was removed in some versions; tolerate
        # both signatures.
        try:
            raw_result = reader.ocr(ocr_input, cls=True)
        except TypeError:
            raw_result = reader.ocr(ocr_input)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("PaddleOCR inference failed: %s", exc)
        return None

    detections = _parse_paddle_result(raw_result)
    if not detections:
        return None

    canvas_width = plate_image.shape[1] if plate_image.ndim >= 2 else 100

    primary = _attempt_single_line_extraction(detections)
    if primary:
        return primary

    fused = _attempt_two_row_fusion(detections, canvas_width)
    if fused:
        return fused

    return _attempt_best_single_fallback(detections)


# ---------------------------------------------------------------------------
# Section 6 — Multi-phase arbitration for verification scoring
# ---------------------------------------------------------------------------

@dataclass
class _OcrAttempt:
    """Outcome of running OCR on one phase image for one candidate."""
    text: str
    confidence: float
    phase_label: str
    is_valid: bool


def _score_attempt(text: str, format_confidence: float, is_valid: bool) -> float:
    """Combine length-normalised and grammar-confidence signals."""
    length_score = min(1.0, len(text) / 6.0)
    base = length_score * 0.5 + format_confidence * 0.5
    return base * 1.5 if is_valid else base


def arbitrate_multi_phase_ocr(
    phase_images: dict[str, np.ndarray],
    candidate_bbox: tuple[int, int, int, int],
    roi_prep_fn: Any,
    phase_priority: tuple[tuple[str, str], ...] = (
        ("restored", "Phase 3 — bilateral filtered"),
        ("enhanced", "Phase 2 — histogram equalised"),
        ("color_processed", "Phase 4 — HSV V channel"),
    ),
) -> tuple[str, float]:
    """Run OCR across several pre-processed views and take a majority vote.

    For every phase image we crop the same bounding box, hand it through the
    caller-supplied ROI preparation function (typically
    ``core.image_pipeline.prepare_roi_for_recognition``), and submit the
    result to ``read_plate_text``.  Each attempt is scored by length and
    grammar conformance.

    Attempts whose composite scores fall within 0.1 of one another are
    treated as a tie and resolved by simple majority of the recognised text;
    if a tie persists, the longest token wins because it is more likely to
    be a complete read than a truncated one.

    Returns ``(text, composite_confidence)``.  Empty string and zero are
    returned when no phase produced any output.
    """
    x, y, w, h = candidate_bbox
    attempts: list[_OcrAttempt] = []

    for phase_key, phase_label in phase_priority:
        if phase_key not in phase_images:
            continue
        source = phase_images[phase_key]
        try:
            roi = roi_prep_fn(source, (x, y, w, h))
        except Exception:  # noqa: BLE001
            continue
        if roi is None or roi.size == 0:
            continue

        extracted = read_plate_text(roi)
        if not extracted:
            continue

        valid, fmt_conf, _ = validate_plate_syntax(extracted)
        attempts.append(
            _OcrAttempt(
                text=extracted,
                confidence=_score_attempt(extracted, fmt_conf, valid),
                phase_label=phase_label,
                is_valid=valid,
            )
        )

    if not attempts:
        return "", 0.0

    # Bucket by score rounded to 0.1 — within-bucket items are treated as a
    # statistical tie and majority-voted.
    buckets: dict[float, list[_OcrAttempt]] = {}
    for attempt in attempts:
        key = round(attempt.confidence, 1)
        buckets.setdefault(key, []).append(attempt)

    top_bucket = buckets[max(buckets.keys())]

    if len(top_bucket) == 1:
        winner = top_bucket[0]
        return winner.text, winner.confidence

    vote_tally: dict[str, list[_OcrAttempt]] = {}
    for attempt in top_bucket:
        vote_tally.setdefault(attempt.text, []).append(attempt)

    winners = sorted(
        vote_tally.items(),
        key=lambda kv: (len(kv[1]), len(kv[0])),
        reverse=True,
    )
    winning_text, winning_attempts = winners[0]
    average_conf = sum(a.confidence for a in winning_attempts) / len(winning_attempts)
    return winning_text, average_conf