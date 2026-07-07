"""
processors.car_processor
========================

Plate localiser for **passenger cars (sedans, hatchbacks, coupes)**.

Contribution: Manreen
Categories handled by this processor:
    * Standard         — civilian state-issued single-row plate
    * Taxi             — H-family commercial transport prefix
    * Diplomatic       — CC / CD / DC consular plate

Strategy
--------
Cars carry a single-row, near-rectangular plate of typical aspect ratio
3.0 – 4.5.  Three mask-generation routes are run in parallel to maximise
recall under varying lighting:

    Route A — Adaptive Gaussian threshold (best for cleanly lit plates).
    Route B — Bilateral + Canny edge cascade (best for shadow-occluded
              plates where the character outlines remain visible even
              though the body intensity is washed out).
    Route C — Dark-mask thresholding around 70 % of the mean grey level
              (best for white-on-black or black-on-white high-contrast
              plates where the body intensity differs sharply from the
              surrounding bumper).

The three mask sets are concatenated, run through the shared geometric
filter, deduplicated by centre-distance NMS, then ranked by a route-aware
combination of geometric area, position prior and route confidence.
"""

from __future__ import annotations

import logging
import os
import re
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
    strip_to_alphanumeric,
    validate_plate_syntax,
)

LOG = logging.getLogger(__name__)

VEHICLE_TYPE: str = "car"

# Aspect bands the car processor considers viable for a sedan plate.
# Square or motorcycle aspect ratios are deliberately excluded — a car plate
# in this aspect band would be physically impossible at our camera
# distances.
ASPECT_PROFILE: tuple[tuple[float, float], ...] = (
    ASPECT_BAND_SINGLE_LINE,
    ASPECT_BAND_WIDE,
    ASPECT_BAND_VERY_WIDE,
)

# Category-detection regexes applied to the cleaned OCR token.
_TAXI_LEADING = re.compile(r"^(HW|HBA|HBB|HBC|HC|HKL|TX|KX|LM)")
_DIPLOMATIC_LEADING = re.compile(r"^\d{0,3}(CC|CD|DC)\d{1,4}")

# Per-route quality weights used by ``_rank_candidate``.  These multipliers
# preserve the donor's empirical bias toward dark-contrast detections when
# multiple routes simultaneously fire on the same physical plate.
_ROUTE_WEIGHTS: dict[str, float] = {
    "adaptive": 1.8,
    "edge": 1.6,
    "dark_contrast": 2.0,
}


# ---------------------------------------------------------------------------
# Mask generation — three parallel routes
# ---------------------------------------------------------------------------

def _build_candidate_masks(
    luma_grid: np.ndarray,
    denoised_frame: np.ndarray,
) -> list[tuple[np.ndarray, str]]:
    """Produce the three binary masks from which contour candidates are drawn.

    The function operates on the already-denoised luminance image — the
    bilateral filter in ``core.run_preprocessing_pipeline`` is therefore not
    re-applied here, in accordance with the architecture's anti-duplication
    rule for shared preprocessing.
    """
    masks: list[tuple[np.ndarray, str]] = []

    # --- Route A: adaptive Gaussian threshold -----------------------------
    try:
        adaptive_mask = cv2.adaptiveThreshold(
            luma_grid, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2,
        )
        masks.append((adaptive_mask, "adaptive"))
    except cv2.error as exc:
        LOG.debug("Adaptive route failed: %s", exc)

    # --- Route B: Canny edge cascade --------------------------------------
    try:
        edge_layer = cv2.Canny(denoised_frame, 30, 100)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
        connected = cv2.morphologyEx(edge_layer, cv2.MORPH_CLOSE, close_kernel)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))
        connected = cv2.dilate(connected, dilate_kernel, iterations=2)
        masks.append((connected, "edge"))
    except cv2.error as exc:
        LOG.debug("Edge route failed: %s", exc)

    # --- Route C: dark-region mask ----------------------------------------
    try:
        mean_intensity = float(np.mean(luma_grid))
        threshold_value = mean_intensity * 0.7
        dark_mask = (luma_grid < threshold_value).astype(np.uint8) * 255

        connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, connect_kernel)
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, clean_kernel)
        masks.append((dark_mask, "dark_contrast"))
    except cv2.error as exc:
        LOG.debug("Dark-contrast route failed: %s", exc)

    return masks


# ---------------------------------------------------------------------------
# Category classification post-OCR
# ---------------------------------------------------------------------------

def _classify_plate_category(cleaned_token: str) -> str:
    """Map a sanitised OCR token to its plate category label.

    The priority order matters: a token like ``HW1234`` is unambiguously a
    taxi, never a diplomatic plate, so taxi is tested first.  Diplomatic is
    tested second because its grammar is the most distinctive (the
    ``CC``/``CD``/``DC`` substring is otherwise extremely rare in civilian
    plates).  Anything that fails both tests is reported as ``Standard``.
    """
    if not cleaned_token:
        return "Unknown"
    if _TAXI_LEADING.match(cleaned_token):
        return "Taxi"
    if _DIPLOMATIC_LEADING.match(cleaned_token):
        return "Diplomatic"
    return "Standard"


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def _rank_candidate(region: CandidateRegion, frame_shape: tuple[int, ...]) -> float:
    """Convert a CandidateRegion into a single ranking scalar.

    Combines the geometric base score (which already incorporates colour
    and texture confidence from the core filter), the shared position
    prior, and a route-specific multiplier reflecting which binarisation
    strategy produced the candidate.
    """
    base = region.score
    spatial = position_aware_score(region, frame_shape, small_plate_area_floor=1500)
    route_weight = _ROUTE_WEIGHTS.get(region.method_tag, 1.0)
    return base * spatial * route_weight


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process(image_path: str) -> dict[str, Any]:
    """Detect and recognise a sedan plate in the supplied image.

    The function never raises — every error path collapses into a
    well-formed failure dictionary that the GUI can render gracefully.
    """
    debug_phases: dict[str, np.ndarray] = {}

    try:
        if not isinstance(image_path, str) or not os.path.isfile(image_path):
            return _failure(debug_phases, f"Image not readable at {image_path!r}.")

        source_frame = load_rgb_frame(image_path)
        debug_phases = run_preprocessing_pipeline(source_frame)
        luma_grid = debug_phases["grayscale"]
        denoised_frame = debug_phases["restored"]

        # --- Harvest candidates from every binarisation route --------------
        candidate_pool: list[CandidateRegion] = []
        for binary_mask, route_tag in _build_candidate_masks(luma_grid, denoised_frame):
            harvested = filter_contours_geometric(
                binary_mask,
                source_frame=source_frame,
                aspect_bands=ASPECT_PROFILE,
                method_tag=route_tag,
            )
            candidate_pool.extend(harvested)

        if not candidate_pool:
            return _failure(debug_phases, "No plate-shaped contours found.")

        unique_regions = deduplicate_overlapping(candidate_pool, centre_distance_threshold=30)
        unique_regions.sort(
            key=lambda region: _rank_candidate(region, source_frame.shape),
            reverse=True,
        )

        # --- OCR arbitration on the top-N candidates ----------------------
        top_n = unique_regions[: min(10, len(unique_regions))]
        chosen_region: CandidateRegion | None = None
        winning_token: str = ""
        winning_confidence: float = 0.0
        best_raw_text: str = ""

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
                best_raw_text = recognised_text
                # Early exit on a high-confidence civilian-grammar hit.
                is_valid, _, _ = validate_plate_syntax(recognised_text)
                if is_valid and attempt_confidence > 0.85:
                    break

        if chosen_region is None or not winning_token:
            return _failure(debug_phases, "OCR produced no readable text on any candidate.")

        cleaned_token = apply_position_corrections(winning_token)
        if len(cleaned_token) < 3:
            return _failure(debug_phases, f"OCR result {winning_token!r} too short after sanitisation.")

        plate_category = _classify_plate_category(cleaned_token)

        # --- Build the detection-overlay debug image ----------------------
        overlay = source_frame.copy()
        cv2.rectangle(
            overlay,
            (chosen_region.x, chosen_region.y),
            (chosen_region.x + chosen_region.w, chosen_region.y + chosen_region.h),
            (255, 0, 0), 3,
        )
        cv2.putText(
            overlay, cleaned_token,
            (chosen_region.x, max(0, chosen_region.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
        )
        debug_phases["detection_result"] = overlay

        plate_crop = source_frame[
            chosen_region.y: chosen_region.y + chosen_region.h,
            chosen_region.x: chosen_region.x + chosen_region.w,
        ].copy()

        return {
            "success": True,
            "vehicle_type": VEHICLE_TYPE,
            "plate_category": plate_category,
            "plate_bbox": chosen_region.bbox,
            "plate_image": plate_crop,
            "raw_ocr_text": best_raw_text,
            "cleaned_text": cleaned_token,
            "confidence": min(1.0, winning_confidence),
            "debug_stages": debug_phases,
            "error_message": "",
        }

    except Exception as exc:  # noqa: BLE001 — architecture mandates zero-crash
        LOG.exception("Car processor crashed.")
        return _failure(debug_phases, f"Unhandled exception: {exc}")


def _failure(debug_phases: dict[str, np.ndarray], message: str) -> dict[str, Any]:
    """Construct a well-formed failure envelope for ``process``."""
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
    sample = sys.argv[1] if len(sys.argv) > 1 else "test_images/cars/car1.jpg"
    outcome = process(sample)
    print(f"[car_processor] success={outcome['success']}")
    print(f"  category={outcome['plate_category']}")
    print(f"  cleaned ={outcome['cleaned_text']!r}")
    print(f"  bbox    ={outcome['plate_bbox']}")
    print(f"  conf    ={outcome['confidence']:.3f}")
    if outcome["error_message"]:
        print(f"  error   ={outcome['error_message']}")