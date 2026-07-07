"""
processors.motorcycle_processor
===============================

Plate localiser for **motorcycles** (scooters, sport bikes, mopeds).

Contribution: Sehba
Categories handled by this processor:
    * Two-Row   — civilian two-row stacked plate (the default for all
                  Malaysian motorcycle plates)

Strategy
--------
Motorcycle plates differ from every other vehicle type in this project in
two fundamental ways:

    1.  Geometry — they are arranged as two stacked rows (alpha prefix on
        top, numeric body underneath) inside a near-square bounding box.
        The aspect ratio sits in the 0.7 – 1.5 band rather than the
        wide-rectangle 3.0 – 5.0 band of every other vehicle.

    2.  Distance — at typical traffic-camera distances a motorcycle plate
        occupies far fewer pixels than a car plate, so robustness depends
        on multi-scale detection rather than a single best mask.

Six binarisation routes are run in parallel:

    Route A — Vertical-connection morphology after adaptive thresholding
              with the donor's (3, 8) → (8, 3) → (5, 12) cascade.
    Route B — Square-kernel morphology using a (10, 3) row linker followed
              by a (12, 12) block kernel whose isotropic closing pressure
              bridges the inter-row gap more reliably than asymmetric
              tall-thin kernels.  Uses a focused near-square aspect profile.
    Route C — Mean-adaptive thresholding with micro kernels (3, 1) and
              (1, 4) for distant or low-resolution motorcycle plates.
    Route D — Mean-adaptive thresholding with the donor's regular (5, 2)
              / (2, 5) kernel pair for typical mid-distance scenes.
    Route E — Multi-scale closing with progressive (2, 1)/(3, 2)/(4, 3)
              kernels to handle plates at varying scales in a single image.
    Route F — Gamma-corrected adaptive threshold for indoor / artificial-
              lighting conditions (parking garages, urban tunnels).

Two-row OCR strategy (the critical fix)
----------------------------------------
Morphological merging is not guaranteed to fuse both rows into a single
bounding box at every shooting distance and lighting condition.  When the
detector produces a single-row candidate, the crop fed to PaddleOCR
contains only one text line, so ``_attempt_two_row_fusion`` in
``core.ocr_engine`` has nothing to fuse.

This mirrors how the donor resolved the same problem: instead of relying
on detection alone, the OCR loop now makes a SECOND attempt on a
*vertically expanded* crop (``_expand_bbox_vertically``) for every
candidate.  The expanded crop gives PaddleOCR enough vertical context to
see both rows simultaneously, enabling the existing two-row fusion path to
concatenate ``ABC + 86`` or ``WBU + 3333`` into the full plate token.

The two attempts (original bbox, expanded bbox) are arbitrated by
composite confidence; the higher-scoring result is kept.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import cv2
import numpy as np

from core.image_pipeline import (
    ASPECT_BAND_MOTORCYCLE,
    ASPECT_BAND_SQUARE,
    ASPECT_BAND_TWO_LINE,
    CandidateRegion,
    deduplicate_overlapping,
    filter_contours_geometric,
    load_rgb_frame,
    position_aware_score,
    prepare_roi_for_recognition,
    run_preprocessing_pipeline,
)
from core.ocr_engine import (
    apply_position_corrections,
    arbitrate_multi_phase_ocr,
    validate_plate_syntax,
)

LOG = logging.getLogger(__name__)

VEHICLE_TYPE: str = "motorcycle"

# The motorcycle aspect profile is the inverse of every other processor:
# square and near-square dominate, with a small allowance for sport bikes
# whose plates are slightly wider than tall.
ASPECT_PROFILE: tuple[tuple[float, float], ...] = (
    ASPECT_BAND_SQUARE,
    ASPECT_BAND_TWO_LINE,
    ASPECT_BAND_MOTORCYCLE,
)

# Focused aspect envelope used exclusively by the square-stacked route.
# The donor's geometry section places two-row motorcycle plates in a 1.1–1.8
# w/h band at typical shooting distances.  We widen to SQUARE ∪ MOTORCYCLE
# ([0.5, 1.5] ∪ [1.0, 2.8]) to absorb mild perspective foreshortening while
# still excluding the elongated single-row candidates that dominate the
# general ASPECT_PROFILE.
_SQUARE_STACKED_ASPECT_PROFILE: tuple[tuple[float, float], ...] = (
    ASPECT_BAND_SQUARE,      # (0.5, 1.5) — near-square bounding boxes
    ASPECT_BAND_MOTORCYCLE,  # (1.0, 2.8) — slightly wider two-row variants
)

# Route weights — the square-stacked route is weighted above the generic
# two-row vertical route because its isotropic kernel is specifically
# engineered for the near-square plate geometry observed in the test corpus.
_ROUTE_WEIGHTS: dict[str, float] = {
    "two_row_vertical":   4.5,
    "square_stacked":     5.8,
    "micro_kernel":       6.0,
    "regular_motorcycle": 5.0,
    "multi_scale":        5.5,
    "gamma_indoor":       4.0,
}

# Expansion factor for the two-row OCR attempt.  The expanded crop pads the
# detected bbox by this multiple of its height on each vertical side so that
# a single-row detection can still expose both rows to PaddleOCR in one
# inference call.  1.0 → total crop height becomes 3× the original row
# height, which is generous enough to span a second row at all typical
# shooting distances while the frame-boundary clamp prevents out-of-bounds
# slicing.
_TWO_ROW_VERTICAL_EXPANSION: float = 1.0


# ---------------------------------------------------------------------------
# Bbox utility — vertical expansion for two-row OCR
# ---------------------------------------------------------------------------

def _expand_bbox_vertically(
    bbox: tuple[int, int, int, int],
    frame_height: int,
    frame_width: int,
    expansion_ratio: float = _TWO_ROW_VERTICAL_EXPANSION,
) -> tuple[int, int, int, int]:
    """Return a vertically expanded bounding box clamped to the frame boundary.

    When the detector localises only one row of a two-row motorcycle plate the
    crop passed to ``arbitrate_multi_phase_ocr`` contains only that row.
    PaddleOCR therefore returns a single detection and the
    ``_attempt_two_row_fusion`` path in ``core.ocr_engine`` has nothing to
    merge, so only half the plate token is ever assembled.

    Padding the crop by ``expansion_ratio × h`` above and below widens the
    vertical field of view so PaddleOCR can detect both text rows in one
    inference pass.  The existing fusion logic then concatenates them in the
    correct top-to-bottom order (e.g. ``ABC`` + ``86``  → ``ABC86``,
    ``WBU`` + ``3333`` → ``WBU3333``).

    This is the same strategy the donor employed: pass a deliberately larger
    crop to OCR and let the engine's internal text-line detection find the
    rows; only then can the Python-level fusion assemble a complete token.

    A 10 % horizontal pad is added on each side to avoid clipping characters
    whose contours were slightly cut by the morphological detection step.
    All coordinates are clamped to [0, frame_dimension].

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        Original ``(x, y, w, h)`` candidate bounding box from detection.
    frame_height : int
        Pixel height of the source frame (for boundary clamp).
    frame_width : int
        Pixel width of the source frame (for boundary clamp).
    expansion_ratio : float
        Fraction of ``h`` added above and below.  Default 1.0 → total
        crop height = 3 × original row height.

    Returns
    -------
    tuple[int, int, int, int]
        Expanded ``(x, y, w, h)`` bounding box, safe to use as a slice index.
    """
    x, y, w, h = bbox
    v_pad = int(h * expansion_ratio)
    h_pad = int(w * 0.10)

    new_x  = max(0, x - h_pad)
    new_y  = max(0, y - v_pad)
    new_x2 = min(frame_width,  x + w + h_pad)
    new_y2 = min(frame_height, y + h + v_pad)

    return (new_x, new_y, new_x2 - new_x, new_y2 - new_y)


# ---------------------------------------------------------------------------
# Mask generation — six parallel routes
# ---------------------------------------------------------------------------

def _two_row_vertical_mask(luma_grid: np.ndarray) -> np.ndarray:
    """Build the primary two-row fusion mask.

    The four-kernel cascade (vertical-close → horizontal-close → merge-close
    → noise-open) is preserved from the donor's empirical tuning.  The
    initial adaptive threshold uses a 15-pixel window — wider than the
    11-pixel general window in ``core`` — to better accommodate the larger
    character heights typical of motorcycle plates relative to their frame
    width.
    """
    binary_base = cv2.adaptiveThreshold(
        luma_grid, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 3,
    )

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 8))
    closed = cv2.morphologyEx(binary_base, cv2.MORPH_CLOSE, vertical_kernel)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 3))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, horizontal_kernel)

    merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 12))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, merge_kernel)

    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, cleanup_kernel)


def _square_stacked_plate_mask(luma_grid: np.ndarray) -> np.ndarray:
    """Build a dedicated mask for near-square, two-row stacked motorcycle plates.

    Malaysian motorcycle plates arrange an alphabetic prefix on the upper row
    and a numeric body on the lower row inside a near-square frame (w/h ≈ 1.1
    – 1.8, per the donor's geometry notes).  The inter-row gap — typically
    4–12 px at common shooting distances — resists the primary two-row
    vertical route's (5, 12) merge kernel because that kernel's asymmetry
    directs closing energy upward into the character bodies before bridging
    across the gap in the perpendicular direction.

    A blockier square kernel — (12, 12) — applies equal closing pressure in
    both spatial axes simultaneously, reliably collapsing the inter-row gap
    and producing a single filled blob whose bounding rectangle falls within
    the target 1.1–1.8 aspect band.

    The wider adaptive window (19 px) prevents the binarisation from choking
    on the taller character heights that occupy a full two-row crop.
    """
    binary_base = cv2.adaptiveThreshold(
        luma_grid, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        19, 4,
    )

    # Pass 1 — horizontal sweep to link characters within each row.
    row_linker = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    stage = cv2.morphologyEx(binary_base, cv2.MORPH_CLOSE, row_linker)

    # Pass 2 — blockier square kernel fuses the vertical inter-row gap.
    block_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    stage = cv2.morphologyEx(stage, cv2.MORPH_CLOSE, block_kernel)

    # Pass 3 — moderate opening removes residual bleed from rear reflectors.
    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(stage, cv2.MORPH_OPEN, cleanup_kernel)


def _micro_kernel_mask(motorcycle_thresh: np.ndarray) -> np.ndarray:
    """Build the micro-kernel mask for distant or low-resolution plates.

    Kernel sizes are deliberately tiny — (3, 1), (1, 4), (2, 5) — because
    fattening the kernels at this scale would blur adjacent plate text
    together with neighbouring scooter components (helmet straps, exhaust
    pipes, rider body) and ruin the contour shape.
    """
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    closed = cv2.morphologyEx(motorcycle_thresh, cv2.MORPH_CLOSE, horizontal_kernel)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, vertical_kernel)

    finishing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
    return cv2.morphologyEx(closed, cv2.MORPH_CLOSE, finishing_kernel)


def _regular_motorcycle_mask(motorcycle_thresh: np.ndarray) -> np.ndarray:
    """Build the standard-distance motorcycle mask using the donor's tuning."""
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    closed = cv2.morphologyEx(motorcycle_thresh, cv2.MORPH_CLOSE, h_kernel)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, v_kernel)

    final_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 6))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, final_kernel)

    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, cleanup_kernel)


def _multi_scale_masks(luma_grid: np.ndarray) -> list[np.ndarray]:
    """Yield three masks at progressive kernel scales for varying distances.

    Each scale uses a kernel of size ``(s, s-1)`` horizontally and
    ``(1, s+1)`` vertically.  Smaller kernels keep distant plates intact,
    larger kernels recover mid-distance plates whose characters are too
    thick for the tiny kernel to bridge.
    """
    scale1_thresh = cv2.adaptiveThreshold(
        luma_grid, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        7, 2,
    )
    masks: list[np.ndarray] = []
    for kw, kh in ((2, 1), (3, 2), (4, 3)):
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        stage = cv2.morphologyEx(scale1_thresh, cv2.MORPH_CLOSE, h_kernel)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh + 2))
        stage = cv2.morphologyEx(stage, cv2.MORPH_CLOSE, v_kernel)
        masks.append(stage)
    return masks


def _gamma_indoor_mask(luma_grid: np.ndarray, gamma: float = 1.3) -> np.ndarray:
    """Mask tuned for indoor / artificial lighting conditions.

    A modest positive gamma compresses the highlight half of the dynamic
    range, which helps when overhead fluorescent lighting has driven the
    plate body into the upper bright tail of the histogram.
    """
    normalised = luma_grid.astype(np.float32) / 255.0
    gamma_balanced = np.clip(
        255.0 * np.power(normalised, gamma), 0, 255,
    ).astype(np.uint8)

    indoor_thresh = cv2.adaptiveThreshold(
        gamma_balanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        5, 1,
    )
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    closed = cv2.morphologyEx(indoor_thresh, cv2.MORPH_CLOSE, h_kernel)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    return cv2.morphologyEx(closed, cv2.MORPH_CLOSE, v_kernel)


def _build_candidate_masks(luma_grid: np.ndarray) -> list[tuple[np.ndarray, str]]:
    """Run every binarisation route in sequence and emit ``(mask, tag)`` pairs.

    The square-stacked route is inserted immediately after the primary two-row
    vertical route so that it operates on the same ``luma_grid`` input without
    any additional preprocessing overhead.
    """
    masks: list[tuple[np.ndarray, str]] = []

    try:
        masks.append((_two_row_vertical_mask(luma_grid), "two_row_vertical"))
    except cv2.error as exc:
        LOG.debug("Two-row-vertical route failed: %s", exc)

    try:
        masks.append((_square_stacked_plate_mask(luma_grid), "square_stacked"))
    except cv2.error as exc:
        LOG.debug("Square-stacked route failed: %s", exc)

    try:
        # The micro and regular routes share the same initial binarisation
        # so we compute it once and feed both downstream cascades.
        motorcycle_base = cv2.adaptiveThreshold(
            luma_grid, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9, 3,
        )
        masks.append((_micro_kernel_mask(motorcycle_base), "micro_kernel"))
        masks.append((_regular_motorcycle_mask(motorcycle_base), "regular_motorcycle"))
    except cv2.error as exc:
        LOG.debug("Micro/regular routes failed: %s", exc)

    try:
        for scale_mask in _multi_scale_masks(luma_grid):
            masks.append((scale_mask, "multi_scale"))
    except cv2.error as exc:
        LOG.debug("Multi-scale route failed: %s", exc)

    try:
        masks.append((_gamma_indoor_mask(luma_grid), "gamma_indoor"))
    except cv2.error as exc:
        LOG.debug("Gamma-indoor route failed: %s", exc)

    return masks


# ---------------------------------------------------------------------------
# Candidate ranking
# ---------------------------------------------------------------------------

def _rank_candidate(region: CandidateRegion, frame_shape: tuple[int, ...]) -> float:
    """Motorcycle ranker — very small floor for the small-plate prior.

    The floor of 800 px² (rather than the car-processor's 1500 px²) reflects
    the fact that distant motorcycle plates are *intended* to be small —
    they are not a sign of misdetection, they are the typical case.
    """
    base = region.score
    spatial = position_aware_score(region, frame_shape, small_plate_area_floor=800)
    route_weight = _ROUTE_WEIGHTS.get(region.method_tag, 1.0)
    return base * spatial * route_weight


# ---------------------------------------------------------------------------
# OCR helper — two-pass attempt (tight bbox + expanded bbox)
# ---------------------------------------------------------------------------

def _ocr_with_expansion(
    debug_phases: dict[str, np.ndarray],
    original_bbox: tuple[int, int, int, int],
    frame_height: int,
    frame_width: int,
) -> tuple[str, float]:
    """Run OCR twice — on the original bbox and on a vertically expanded crop.

    This is the core fix for the two-row recognition failure observed in both
    test cases (``ABC 86``, ``WBU 3333``).

    Root cause recap
    ~~~~~~~~~~~~~~~~
    ``arbitrate_multi_phase_ocr`` crops exactly the supplied bounding box
    before calling PaddleOCR.  If the detector localised only one text row,
    that crop contains a single line, PaddleOCR returns a single detection,
    and ``_attempt_two_row_fusion`` in ``core.ocr_engine`` cannot fire —
    so only half the plate is ever read.

    Fix mechanism
    ~~~~~~~~~~~~~
    After the standard (tight bbox) attempt, a second attempt is made on a
    vertically expanded copy of the same bounding box.  The padded crop
    exposes both text rows to PaddleOCR in one inference call.  PaddleOCR
    then returns two detections whose Y-centres differ, and the pre-existing
    ``_attempt_two_row_fusion`` logic sorts them top-to-bottom and
    concatenates the tokens into the full plate string.

    This is identical to the donor's strategy of deliberately using larger
    plate crops and letting the OCR engine's own text-line detection handle
    the row splitting.

    Arbitration
    ~~~~~~~~~~~
    Both attempts are run independently.  The expanded result is preferred
    when it is either more confident OR it produced a longer token (more
    complete plate read) and its confidence is within 15 % of the tight
    bbox result.  This avoids noisy background text from the padded margins
    winning purely on text length.

    Parameters
    ----------
    debug_phases : dict[str, np.ndarray]
        Full nine-phase image dictionary from ``run_preprocessing_pipeline``.
    original_bbox : tuple[int, int, int, int]
        The detector's candidate bounding box ``(x, y, w, h)``.
    frame_height : int
        Height of the source frame (used for clamp arithmetic).
    frame_width : int
        Width of the source frame (used for clamp arithmetic).

    Returns
    -------
    tuple[str, float]
        ``(best_recognised_text, best_composite_confidence)``.
    """
    # --- Pass 1: tight detector bbox ---------------------------------------
    text_tight, conf_tight = arbitrate_multi_phase_ocr(
        debug_phases,
        original_bbox,
        roi_prep_fn=prepare_roi_for_recognition,
    )

    # --- Pass 2: vertically expanded crop ----------------------------------
    expanded_bbox = _expand_bbox_vertically(
        original_bbox, frame_height, frame_width,
    )

    text_expanded, conf_expanded = "", 0.0
    if expanded_bbox != original_bbox:
        text_expanded, conf_expanded = arbitrate_multi_phase_ocr(
            debug_phases,
            expanded_bbox,
            roi_prep_fn=prepare_roi_for_recognition,
        )

    # --- Arbitrate ---------------------------------------------------------
    if text_tight and text_expanded:
        expanded_is_longer   = len(text_expanded) > len(text_tight)
        confidence_is_close  = conf_expanded >= conf_tight * 0.85
        expanded_wins = conf_expanded > conf_tight or (
            expanded_is_longer and confidence_is_close
        )
        if expanded_wins:
            LOG.debug(
                "Two-row expansion preferred: %r (%.3f) over %r (%.3f)",
                text_expanded, conf_expanded, text_tight, conf_tight,
            )
            return text_expanded, conf_expanded
        return text_tight, conf_tight

    if text_expanded:
        return text_expanded, conf_expanded
    return text_tight, conf_tight


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process(image_path: str) -> dict[str, Any]:
    """Detect and recognise a motorcycle plate in the supplied image.

    For each ranked candidate the OCR loop delegates to
    ``_ocr_with_expansion``, which internally runs two arbitration attempts:

    1. Tight bbox attempt — standard crop of the detected region.
    2. Expanded bbox attempt — the same bbox padded vertically by
       ``_TWO_ROW_VERTICAL_EXPANSION × h`` on each side.

    The expanded crop is what allows the pre-existing
    ``_attempt_two_row_fusion`` logic in ``core.ocr_engine`` to combine both
    text rows of a stacked plate even when morphological detection only
    localised one of them.

    The candidate-harvest loop also dispatches a focused near-square aspect
    profile to the ``square_stacked`` route to prevent its blockier
    morphology from being evaluated against the wider general envelope.
    """
    debug_phases: dict[str, np.ndarray] = {}

    try:
        if not isinstance(image_path, str) or not os.path.isfile(image_path):
            return _failure(debug_phases, f"Image not readable at {image_path!r}.")

        source_frame  = load_rgb_frame(image_path)
        debug_phases  = run_preprocessing_pipeline(source_frame)
        luma_grid     = debug_phases["grayscale"]
        frame_h, frame_w = source_frame.shape[:2]

        candidate_pool: list[CandidateRegion] = []
        for binary_mask, route_tag in _build_candidate_masks(luma_grid):
            active_profile = (
                _SQUARE_STACKED_ASPECT_PROFILE
                if route_tag == "square_stacked"
                else ASPECT_PROFILE
            )
            harvested = filter_contours_geometric(
                binary_mask,
                source_frame=source_frame,
                aspect_bands=active_profile,
                method_tag=route_tag,
            )
            candidate_pool.extend(harvested)

        if not candidate_pool:
            return _failure(debug_phases, "No plate-shaped contours found.")

        unique_regions = deduplicate_overlapping(
            candidate_pool, centre_distance_threshold=20,
        )
        unique_regions.sort(
            key=lambda region: _rank_candidate(region, source_frame.shape),
            reverse=True,
        )

        top_n = unique_regions[: min(10, len(unique_regions))]
        chosen_region: CandidateRegion | None = None
        winning_token: str = ""
        winning_confidence: float = 0.0

        for region in top_n:
            # Two-pass OCR: tight bbox first, expanded bbox second.
            # The expanded pass is the mechanism that recovers the second
            # text row when detection only found one.
            recognised_text, attempt_confidence = _ocr_with_expansion(
                debug_phases,
                region.bbox,
                frame_h,
                frame_w,
            )

            if not recognised_text:
                continue

            if attempt_confidence > winning_confidence or chosen_region is None:
                chosen_region      = region
                winning_token      = recognised_text
                winning_confidence = attempt_confidence
                is_valid, _, _     = validate_plate_syntax(recognised_text)
                if is_valid and attempt_confidence > 0.80:
                    break

        if chosen_region is None or not winning_token:
            return _failure(
                debug_phases,
                "OCR produced no readable text on any candidate.",
            )

        cleaned_token = apply_position_corrections(winning_token)
        if len(cleaned_token) < 3:
            return _failure(
                debug_phases,
                f"OCR result {winning_token!r} too short after sanitisation.",
            )

        overlay = source_frame.copy()
        cv2.rectangle(
            overlay,
            (chosen_region.x, chosen_region.y),
            (chosen_region.x + chosen_region.w, chosen_region.y + chosen_region.h),
            (0, 0, 255), 3,
        )
        cv2.putText(
            overlay, cleaned_token,
            (chosen_region.x, max(0, chosen_region.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )
        debug_phases["detection_result"] = overlay

        plate_crop = source_frame[
            chosen_region.y: chosen_region.y + chosen_region.h,
            chosen_region.x: chosen_region.x + chosen_region.w,
        ].copy()

        return {
            "success":        True,
            "vehicle_type":   VEHICLE_TYPE,
            "plate_category": "Two-Row",
            "plate_bbox":     chosen_region.bbox,
            "plate_image":    plate_crop,
            "raw_ocr_text":   winning_token,
            "cleaned_text":   cleaned_token,
            "confidence":     min(1.0, winning_confidence),
            "debug_stages":   debug_phases,
            "error_message":  "",
        }

    except Exception as exc:  # noqa: BLE001
        LOG.exception("Motorcycle processor crashed.")
        return _failure(debug_phases, f"Unhandled exception: {exc}")


def _failure(debug_phases: dict[str, np.ndarray], message: str) -> dict[str, Any]:
    """Failure envelope for ``process``."""
    return {
        "success":        False,
        "vehicle_type":   VEHICLE_TYPE,
        "plate_category": "Unknown",
        "plate_bbox":     None,
        "plate_image":    None,
        "raw_ocr_text":   "",
        "cleaned_text":   "",
        "confidence":     0.0,
        "debug_stages":   debug_phases or {},
        "error_message":  message,
    }


# ---------------------------------------------------------------------------
# Standalone CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sample = sys.argv[1] if len(sys.argv) > 1 else "test_images/motorcycles/motorcycle1.jpg"
    outcome = process(sample)
    print(f"[motorcycle_processor] success={outcome['success']}")
    print(f"  category={outcome['plate_category']}")
    print(f"  cleaned ={outcome['cleaned_text']!r}")
    print(f"  bbox    ={outcome['plate_bbox']}")
    print(f"  conf    ={outcome['confidence']:.3f}")
    if outcome["error_message"]:
        print(f"  error   ={outcome['error_message']}")