"""
processors.minibus_processor
============================

Plate localiser for **minibuses** (passenger vans, 14- to 22-seat shuttles,
hotel courtesy vans, midi-buses).

Contribution: Rudra
Categories handled by this processor:
    * Standard         — civilian state-issued single-row plate

Strategy
--------
A minibus is the classification gap between a panel van and a full
single-decker bus.  Its plate-mounting geometry inherits from both
parent classes: the plate sits low on a flat front panel (van-like)
but the panel itself is rendered in a fleet livery (bus-like).  We
therefore combine one mask route from each genealogical branch with
the donor's general-purpose adaptive route as a fallback.

Mask routes:

    Route A — Adaptive Gaussian threshold (window 11, C = 2).  Donor M1.
              Acts as the van-like baseline for minibuses photographed
              under uniform daylight; works on the white personal-use
              plates that hotel courtesy vans typically carry.

    Route B — Bright-text mean-relative mask (intensity > 1.1 × mean)
              closed with (25, 6), dilated with (15, 3), opened with
              (10, 5).  Donor M5.  Targets the bright-character-on-
              dark-livery polarity characteristic of commercial fleet
              vans where the plate panel is painted in route colours
              (red, blue, green) but the characters themselves are
              white reflective vinyl.

    Route C — Lower-2/3 inverted Otsu, closed with (30, 8), dilated
              with (20, 5), opened with (8, 6).  Donor M6.  A spatial
              prior identical to the bus processor's lower-Otsu route
              because minibuses obey bus mounting geometry — the plate
              is always in the bottom two-thirds of the frame.  The
              y-offset is rebased after harvesting so downstream stages
              treat the coordinates as full-frame.

All kernel sizes and threshold multipliers are donor-preserved.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import cv2
import numpy as np

from core.image_pipeline import (
    ASPECT_BAND_SINGLE_LINE,
    ASPECT_BAND_VERY_WIDE,
    ASPECT_BAND_WIDE,
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

VEHICLE_TYPE: str = "minibus"

# Minibus plates can render in any of three aspect regimes depending on
# capture distance: standard single-line at close range, wide for
# typical CCTV captures, and occasionally very-wide when the panel is
# photographed at an oblique angle.  We admit all three.
ASPECT_PROFILE: tuple[tuple[float, float], ...] = (
    ASPECT_BAND_SINGLE_LINE,
    ASPECT_BAND_WIDE,
    ASPECT_BAND_VERY_WIDE,
)

# Trust weights.  The lower-Otsu route inherits its high weight from the
# bus processor's empirical findings — when it fires it is almost always
# correct.  The bright-text route is similarly trusted because the
# minibus fleet livery regime is the one in which it is most reliable.
_ROUTE_WEIGHTS: dict[str, float] = {
    "adaptive": 2.0,
    "bright_text": 2.5,
    "lower_otsu": 3.0,
}


# ---------------------------------------------------------------------------
# Mask generators
# ---------------------------------------------------------------------------

def _adaptive_threshold_mask(luma_grid: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian threshold (donor M1).

    The window of 11 px is the donor's empirically chosen size, broadly
    matching the height of a Malaysian plate character at our typical
    capture resolution; the constant subtraction of 2 produces a
    threshold safely below the local mean for darker character pixels.
    """
    return cv2.adaptiveThreshold(
        luma_grid, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2,
    )


def _bright_text_mask(luma_grid: np.ndarray) -> np.ndarray:
    """Mean-relative bright-text isolation with bus-grade closure (donor M5).

    A commercial minibus often presents the plate as bright characters
    on a coloured fleet panel; thresholding at 110 % of the global mean
    captures those characters.  The closure cascade — wide (25, 6) close,
    (15, 3) dilation, (10, 5) opening — first connects adjacent
    characters into a single blob, then thickens the blob for tolerance
    against motion blur, and finally removes speckle noise that escaped
    the first two stages.
    """
    mean_intensity = float(np.mean(luma_grid))
    bright_mask = (luma_grid > mean_intensity * 1.1).astype(np.uint8) * 255

    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 6))
    bridged = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, bridge_kernel)

    spread_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    spread = cv2.dilate(bridged, spread_kernel, iterations=1)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    return cv2.morphologyEx(spread, cv2.MORPH_OPEN, clean_kernel)


def _lower_two_thirds_otsu_mask(luma_grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Inverted Otsu over the lower two-thirds of the frame (donor M6).

    Returning the y-offset lets the caller translate harvested bounding
    boxes back into full-frame coordinates.  The wide (30, 8) closing is
    the donor's bus-grade kernel; we keep it verbatim because minibuses
    sit in the same physical size class as small buses for plate-mount
    geometry.
    """
    full_height = luma_grid.shape[0]
    y_offset = full_height // 3
    lower_segment = luma_grid[y_offset:, :]

    _, otsu_inverted = cv2.threshold(
        lower_segment, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 8))
    bridged = cv2.morphologyEx(otsu_inverted, cv2.MORPH_CLOSE, bridge_kernel)

    spread_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    spread = cv2.dilate(bridged, spread_kernel, iterations=2)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 6))
    return cv2.morphologyEx(spread, cv2.MORPH_OPEN, clean_kernel), y_offset


# ---------------------------------------------------------------------------
# Lower-ROI candidate harvesting (with y-rebasing)
# ---------------------------------------------------------------------------

def _harvest_lower_region(
    mask: np.ndarray,
    y_offset: int,
    source_frame: np.ndarray,
) -> list[CandidateRegion]:
    """Run the geometric filter on a cropped lower-ROI mask and rebase y-coords.

    We hand the geometric filter a *cropped* RGB frame so its colour and
    texture descriptors evaluate the same pixels that produced the mask.
    Each surviving candidate's y-coordinate is then translated downward
    by ``y_offset`` so downstream NMS, scoring and OCR arbitration all
    operate in full-frame space.
    """
    cropped_source = source_frame[y_offset:, :]
    harvested = filter_contours_geometric(
        mask,
        source_frame=cropped_source,
        aspect_bands=ASPECT_PROFILE,
        method_tag="lower_otsu",
    )
    rebased: list[CandidateRegion] = []
    for region in harvested:
        x, y, w, h = region.bbox
        rebased.append(
            CandidateRegion(
                bbox=(x, y + y_offset, w, h),
                score=region.score,
                method_tag=region.method_tag,
            )
        )
    return rebased


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def _rank_candidate(region: CandidateRegion, frame_shape: tuple[int, ...]) -> float:
    """Minibus-specific ranker — small-plate floor raised to bus levels.

    A minibus plate is large enough relative to the frame that anything
    below 2500 pixels is almost certainly noise (a road sign reflection
    or a vehicle livery character).  The floor is set slightly below
    the bus value (3000) because minibuses are photographed closer than
    full-size buses on average and so their plates fill a slightly
    smaller fraction of the frame when capture occurs at the same
    physical distance.
    """
    base_score = region.score
    spatial_score = position_aware_score(region, frame_shape, small_plate_area_floor=2500)
    route_weight = _ROUTE_WEIGHTS.get(region.method_tag, 1.0)
    return base_score * spatial_score * route_weight


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process(image_path: str) -> dict[str, Any]:
    """Detect and recognise a minibus plate in the supplied image."""
    debug_phases: dict[str, np.ndarray] = {}

    try:
        if not isinstance(image_path, str) or not os.path.isfile(image_path):
            return _failure(debug_phases, f"Image not readable at {image_path!r}.")

        source_frame = load_rgb_frame(image_path)
        debug_phases = run_preprocessing_pipeline(source_frame)
        luma_grid = debug_phases["grayscale"]

        candidate_pool: list[CandidateRegion] = []

        # --- Route A: van-like adaptive threshold --------------------------
        try:
            adaptive_mask = _adaptive_threshold_mask(luma_grid)
            candidate_pool.extend(
                filter_contours_geometric(
                    adaptive_mask,
                    source_frame=source_frame,
                    aspect_bands=ASPECT_PROFILE,
                    method_tag="adaptive",
                )
            )
        except cv2.error as exc:
            LOG.debug("Adaptive route failed: %s", exc)

        # --- Route B: bus-like bright-text mask ----------------------------
        try:
            bright_mask = _bright_text_mask(luma_grid)
            candidate_pool.extend(
                filter_contours_geometric(
                    bright_mask,
                    source_frame=source_frame,
                    aspect_bands=ASPECT_PROFILE,
                    method_tag="bright_text",
                )
            )
        except cv2.error as exc:
            LOG.debug("Bright-text route failed: %s", exc)

        # --- Route C: bus-like lower-2/3 inverted Otsu ---------------------
        try:
            lower_mask, y_offset = _lower_two_thirds_otsu_mask(luma_grid)
            candidate_pool.extend(_harvest_lower_region(lower_mask, y_offset, source_frame))
        except cv2.error as exc:
            LOG.debug("Lower-Otsu route failed: %s", exc)

        if not candidate_pool:
            return _failure(debug_phases, "No plate-shaped contours found.")

        unique_regions = deduplicate_overlapping(candidate_pool, centre_distance_threshold=40)
        unique_regions.sort(
            key=lambda region: _rank_candidate(region, source_frame.shape),
            reverse=True,
        )

        # --- OCR arbitration on the top-N candidates -----------------------
        top_n = unique_regions[: min(10, len(unique_regions))]
        chosen_region: CandidateRegion | None = None
        winning_token: str = ""
        winning_confidence: float = 0.0

        for region in top_n:
            recognised_text, attempt_confidence = arbitrate_multi_phase_ocr(
                debug_phases,
                region.bbox,
                roi_prep_fn=prepare_roi_for_recognition,
            )
            if not recognised_text:
                continue
            if attempt_confidence > winning_confidence or chosen_region is None:
                chosen_region = region
                winning_token = recognised_text
                winning_confidence = attempt_confidence
                is_valid, _, _ = validate_plate_syntax(recognised_text)
                if is_valid and attempt_confidence > 0.85:
                    break

        if chosen_region is None or not winning_token:
            return _failure(debug_phases, "OCR produced no readable text on any candidate.")

        cleaned_token = apply_position_corrections(winning_token)
        if len(cleaned_token) < 3:
            return _failure(debug_phases, f"OCR result {winning_token!r} too short after sanitisation.")

        # --- Overlay & crop --------------------------------------------------
        overlay = source_frame.copy()
        cv2.rectangle(
            overlay,
            (chosen_region.x, chosen_region.y),
            (chosen_region.x + chosen_region.w, chosen_region.y + chosen_region.h),
            (0, 255, 0), 3,
        )
        cv2.putText(
            overlay, cleaned_token,
            (chosen_region.x, max(0, chosen_region.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        debug_phases["detection_result"] = overlay

        plate_crop = source_frame[
            chosen_region.y: chosen_region.y + chosen_region.h,
            chosen_region.x: chosen_region.x + chosen_region.w,
        ].copy()

        return {
            "success": True,
            "vehicle_type": VEHICLE_TYPE,
            "plate_category": "Standard",
            "plate_bbox": chosen_region.bbox,
            "plate_image": plate_crop,
            "raw_ocr_text": winning_token,
            "cleaned_text": cleaned_token,
            "confidence": min(1.0, winning_confidence),
            "debug_stages": debug_phases,
            "error_message": "",
        }

    except Exception as exc:  # noqa: BLE001
        LOG.exception("Minibus processor crashed.")
        return _failure(debug_phases, f"Unhandled exception: {exc}")


def _failure(debug_phases: dict[str, np.ndarray], message: str) -> dict[str, Any]:
    """Construct the canonical failure envelope for ``process``."""
    return {
        "success": False,
        "vehicle_type": VEHICLE_TYPE,
        "plate_category": "Unknown",
        "plate_bbox": None,
        "plate_image": None,
        "raw_ocr_text": "",
        "cleaned_text": "",
        "confidence": 0.0,
        "debug_stages": debug_phases or {},
        "error_message": message,
    }


# ---------------------------------------------------------------------------
# Standalone CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sample = sys.argv[1] if len(sys.argv) > 1 else "test_images/minibuses/minibus1.jpg"
    outcome = process(sample)
    print(f"[minibus_processor] success={outcome['success']}")
    print(f"  category={outcome['plate_category']}")
    print(f"  cleaned ={outcome['cleaned_text']!r}")
    print(f"  bbox    ={outcome['plate_bbox']}")
    print(f"  conf    ={outcome['confidence']:.3f}")
    if outcome["error_message"]:
        print(f"  error   ={outcome['error_message']}")