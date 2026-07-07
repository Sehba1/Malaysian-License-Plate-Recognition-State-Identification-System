"""
processors.truck_processor
==========================

Plate localiser for **trucks** (cargo trucks, lorries, semi-trailers).

Contribution: Sehba
Categories handled by this processor:
    * Standard         — civilian state-issued single-row plate

Strategy
--------
Trucks present the OCR pipeline with a particular combination of
challenges: the plate sits below a tall, often dark cab, frequently in
heavy shadow, and the plate panel is typically large enough that
sub-character grime visibly degrades OCR confidence.  We address these
with four routes:

    Route A — Adaptive Gaussian threshold over the full frame (catches
              well-lit truck plates at typical distances).
    Route B — Bilateral-prepared Canny edge cascade with slightly wider
              connecting kernels than the car processor uses, because
              trucks tend to be photographed at greater stand-off.
    Route C — Lower-frame ROI Otsu thresholding, modelled after the bus
              processor.  Truck plates are constrained by physics to the
              lower half of the captured frame.
    Route D — Dark-region mask (the donor's Method 4 dark-contrast route)
              for white-on-black plate styles.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import cv2
import numpy as np

from core.image_pipeline import (
    ASPECT_BAND_SINGLE_LINE,
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

VEHICLE_TYPE: str = "truck"

ASPECT_PROFILE: tuple[tuple[float, float], ...] = (
    ASPECT_BAND_SINGLE_LINE,
    ASPECT_BAND_WIDE,
)

_ROUTE_WEIGHTS: dict[str, float] = {
    "adaptive": 1.8,
    "edge": 1.7,
    "lower_otsu": 2.6,
    "dark_contrast": 2.0,
}


# ---------------------------------------------------------------------------
# Route helpers
# ---------------------------------------------------------------------------

def _adaptive_route(luma_grid: np.ndarray) -> np.ndarray:
    """Standard adaptive Gaussian threshold mask."""
    return cv2.adaptiveThreshold(
        luma_grid, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2,
    )


def _edge_route(denoised_frame: np.ndarray) -> np.ndarray:
    """Canny → wide-close → dilate cascade tailored for stand-off captures."""
    edge_layer = cv2.Canny(denoised_frame, 30, 100)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
    connected = cv2.morphologyEx(edge_layer, cv2.MORPH_CLOSE, close_kernel)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))
    return cv2.dilate(connected, dilate_kernel, iterations=2)


def _dark_contrast_route(luma_grid: np.ndarray) -> np.ndarray:
    """Mean-relative dark mask for white-on-black truck plate styles."""
    mean_intensity = float(np.mean(luma_grid))
    dark_mask = (luma_grid < mean_intensity * 0.7).astype(np.uint8) * 255

    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, connect_kernel)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    return cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, clean_kernel)


def _lower_otsu_route(luma_grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Otsu-inverted mask on the lower half of the frame; returns y-offset."""
    full_height = luma_grid.shape[0]
    y_offset = full_height // 2
    lower_segment = luma_grid[y_offset:, :]

    _, otsu_inverted = cv2.threshold(
        lower_segment, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 6))
    closed = cv2.morphologyEx(otsu_inverted, cv2.MORPH_CLOSE, close_kernel)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 4))
    dilated = cv2.dilate(closed, dilate_kernel, iterations=1)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 4))
    return cv2.morphologyEx(dilated, cv2.MORPH_OPEN, clean_kernel), y_offset


def _harvest_lower_roi(
    mask: np.ndarray,
    y_offset: int,
    source_frame: np.ndarray,
) -> list[CandidateRegion]:
    """Harvest candidates from a lower-frame mask, rebasing y-coords."""
    cropped_source = source_frame[y_offset:, :]
    harvested = filter_contours_geometric(
        mask,
        source_frame=cropped_source,
        aspect_bands=ASPECT_PROFILE,
        method_tag="lower_otsu",
    )
    return [
        CandidateRegion(
            bbox=(region.x, region.y + y_offset, region.w, region.h),
            score=region.score,
            method_tag=region.method_tag,
        )
        for region in harvested
    ]


def _rank_candidate(region: CandidateRegion, frame_shape: tuple[int, ...]) -> float:
    """Truck ranker — moderate small-plate floor."""
    base = region.score
    spatial = position_aware_score(region, frame_shape, small_plate_area_floor=2000)
    route_weight = _ROUTE_WEIGHTS.get(region.method_tag, 1.0)
    return base * spatial * route_weight


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process(image_path: str) -> dict[str, Any]:
    """Detect and recognise a truck plate in the supplied image."""
    debug_phases: dict[str, np.ndarray] = {}

    try:
        if not isinstance(image_path, str) or not os.path.isfile(image_path):
            return _failure(debug_phases, f"Image not readable at {image_path!r}.")

        source_frame = load_rgb_frame(image_path)
        debug_phases = run_preprocessing_pipeline(source_frame)
        luma_grid = debug_phases["grayscale"]
        denoised_frame = debug_phases["restored"]

        candidate_pool: list[CandidateRegion] = []

        # Full-frame routes
        for builder, route_tag in (
            (_adaptive_route, "adaptive"),
            (_dark_contrast_route, "dark_contrast"),
        ):
            try:
                mask = builder(luma_grid)
                candidate_pool.extend(
                    filter_contours_geometric(
                        mask,
                        source_frame=source_frame,
                        aspect_bands=ASPECT_PROFILE,
                        method_tag=route_tag,
                    )
                )
            except cv2.error as exc:
                LOG.debug("Route %s failed: %s", route_tag, exc)

        # Edge route uses the denoised frame
        try:
            edge_mask = _edge_route(denoised_frame)
            candidate_pool.extend(
                filter_contours_geometric(
                    edge_mask,
                    source_frame=source_frame,
                    aspect_bands=ASPECT_PROFILE,
                    method_tag="edge",
                )
            )
        except cv2.error as exc:
            LOG.debug("Edge route failed: %s", exc)

        # Lower-ROI route with coordinate rebase
        try:
            lower_mask, y_offset = _lower_otsu_route(luma_grid)
            candidate_pool.extend(_harvest_lower_roi(lower_mask, y_offset, source_frame))
        except cv2.error as exc:
            LOG.debug("Lower-Otsu route failed: %s", exc)

        if not candidate_pool:
            return _failure(debug_phases, "No plate-shaped contours found.")

        unique_regions = deduplicate_overlapping(candidate_pool, centre_distance_threshold=35)
        unique_regions.sort(
            key=lambda region: _rank_candidate(region, source_frame.shape),
            reverse=True,
        )

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

        # Overlay drawn in orange (255, 128, 0) for visual distinction
        overlay = source_frame.copy()
        cv2.rectangle(
            overlay,
            (chosen_region.x, chosen_region.y),
            (chosen_region.x + chosen_region.w, chosen_region.y + chosen_region.h),
            (255, 128, 0), 3,
        )
        cv2.putText(
            overlay, cleaned_token,
            (chosen_region.x, max(0, chosen_region.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2,
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
        LOG.exception("Truck processor crashed.")
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
    sample = sys.argv[1] if len(sys.argv) > 1 else "test_images/trucks/truck01.jpg"
    outcome = process(sample)
    print(f"[truck_processor] success={outcome['success']}")
    print(f"  category={outcome['plate_category']}")
    print(f"  cleaned ={outcome['cleaned_text']!r}")
    print(f"  bbox    ={outcome['plate_bbox']}")
    print(f"  conf    ={outcome['confidence']:.3f}")
    if outcome["error_message"]:
        print(f"  error   ={outcome['error_message']}")