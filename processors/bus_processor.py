"""
processors.bus_processor
========================

Plate localiser for **buses** (single-decker, double-decker, coach, tour bus).

Contribution: Manreen
Categories handled by this processor:
    * Standard         — civilian state-issued single-row plate

Strategy
--------
Bus plates sit on a tall flat front panel that is usually painted in a
fleet livery (bold red, blue or yellow on white).  Two characteristics
exploit this geometry:

    Route A — Bright-text mask.  Bus fleets typically render plate
              characters in light paint against a darker chassis bumper,
              so a mask that retains pixels above 110 % of the mean grey
              level isolates the character body directly.  Wide
              morphological closing connects adjacent characters into a
              single plate blob.

    Route B — Lower-ROI Otsu mask.  Bus plates are by physics in the
              bottom half of the camera frame.  Cropping to the lower
              2/3 of the image and running an inverted Otsu threshold on
              that subregion suppresses windshield, signage and tree-line
              clutter that confuses the global threshold.  Coordinates
              are rebased to the full frame after harvesting.

Both routes use the donor's empirically tuned wide rectangular kernels
((25, 6), (30, 8)) which are essential because a bus plate is roughly
twice as wide as a car plate at our typical capture distances.
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

VEHICLE_TYPE: str = "bus"

# Bus plates are heavily skewed toward the wide / very-wide aspect bands
# because they are mounted on a flat front panel with no width constraint.
ASPECT_PROFILE: tuple[tuple[float, float], ...] = (
    ASPECT_BAND_SINGLE_LINE,
    ASPECT_BAND_WIDE,
    ASPECT_BAND_VERY_WIDE,
)

_ROUTE_WEIGHTS: dict[str, float] = {
    "bright_text": 2.5,
    "lower_otsu": 3.0,
}


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def _bright_text_mask(luma_grid: np.ndarray) -> np.ndarray:
    """Isolate bright-on-dark plate text via mean-relative thresholding."""
    mean_intensity = float(np.mean(luma_grid))
    threshold_value = mean_intensity * 1.1
    bright_mask = (luma_grid > threshold_value).astype(np.uint8) * 255

    # Wide horizontal closing to connect adjacent character glyphs into one
    # continuous plate blob — the (25, 6) kernel is sized to bridge typical
    # inter-character gaps in bus signage at our capture distances.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 6))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, close_kernel)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    bright_mask = cv2.dilate(bright_mask, dilate_kernel, iterations=1)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    return cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, open_kernel)


def _lower_roi_otsu_mask(luma_grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Run inverted Otsu on the lower 2/3 of the frame, returning the y-offset.

    The y-offset is needed by the caller to translate harvested bounding
    boxes back into full-frame coordinates.
    """
    full_height = luma_grid.shape[0]
    y_offset = full_height // 3
    lower_segment = luma_grid[y_offset:, :]

    _, otsu_inverted = cv2.threshold(
        lower_segment, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    wide_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 8))
    closed_pass = cv2.morphologyEx(otsu_inverted, cv2.MORPH_CLOSE, wide_kernel)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    dilated = cv2.dilate(closed_pass, dilate_kernel, iterations=2)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 6))
    return cv2.morphologyEx(dilated, cv2.MORPH_OPEN, clean_kernel), y_offset


# ---------------------------------------------------------------------------
# Candidate harvesting wrappers
# ---------------------------------------------------------------------------

def _harvest_lower_roi(
    mask: np.ndarray,
    y_offset: int,
    source_frame: np.ndarray,
) -> list[CandidateRegion]:
    """Run the shared geometric filter on a lower-ROI mask and rebase y-coords.

    Because the underlying mask was computed on a cropped frame, every
    bounding box returned by the filter must be translated downward by
    ``y_offset`` to express it in full-frame coordinates.
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


def _rank_candidate(region: CandidateRegion, frame_shape: tuple[int, ...]) -> float:
    """Bus-specific candidate ranker — larger floor for the small-plate prior."""
    base = region.score
    spatial = position_aware_score(region, frame_shape, small_plate_area_floor=3000)
    route_weight = _ROUTE_WEIGHTS.get(region.method_tag, 1.0)
    return base * spatial * route_weight


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process(image_path: str) -> dict[str, Any]:
    """Detect and recognise a bus plate in the supplied image."""
    debug_phases: dict[str, np.ndarray] = {}

    try:
        if not isinstance(image_path, str) or not os.path.isfile(image_path):
            return _failure(debug_phases, f"Image not readable at {image_path!r}.")

        source_frame = load_rgb_frame(image_path)
        debug_phases = run_preprocessing_pipeline(source_frame)
        luma_grid = debug_phases["grayscale"]

        # --- Route A: bright text on dark fleet livery --------------------
        candidate_pool: list[CandidateRegion] = []
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

        # --- Route B: inverted Otsu on lower 2/3 of the frame -------------
        try:
            lower_mask, y_offset = _lower_roi_otsu_mask(luma_grid)
            candidate_pool.extend(_harvest_lower_roi(lower_mask, y_offset, source_frame))
        except cv2.error as exc:
            LOG.debug("Lower-ROI route failed: %s", exc)

        if not candidate_pool:
            return _failure(debug_phases, "No plate-shaped contours found.")

        unique_regions = deduplicate_overlapping(candidate_pool, centre_distance_threshold=40)
        unique_regions.sort(
            key=lambda region: _rank_candidate(region, source_frame.shape),
            reverse=True,
        )

        # --- OCR arbitration on the top-N candidates ---------------------
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

        # --- Detection overlay --------------------------------------------
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
        LOG.exception("Bus processor crashed.")
        return _failure(debug_phases, f"Unhandled exception: {exc}")


def _failure(debug_phases: dict[str, np.ndarray], message: str) -> dict[str, Any]:
    """Failure envelope for ``process``."""
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
    sample = sys.argv[1] if len(sys.argv) > 1 else "test_images/buses/bus1.jpg"
    outcome = process(sample)
    print(f"[bus_processor] success={outcome['success']}")
    print(f"  category={outcome['plate_category']}")
    print(f"  cleaned ={outcome['cleaned_text']!r}")
    print(f"  bbox    ={outcome['plate_bbox']}")
    print(f"  conf    ={outcome['confidence']:.3f}")
    if outcome["error_message"]:
        print(f"  error   ={outcome['error_message']}")