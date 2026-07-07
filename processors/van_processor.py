"""
processors.van_processor
========================

Plate localiser for **vans** (passenger vans, PVs, cargo vans).

Contribution: Egor
Categories handled by this processor:
    * Standard         — civilian state-issued single-row plate

Strategy
--------
Vans carry the same plate format as cars but are typically photographed
at a slightly larger pixel footprint due to their bigger silhouette,
which shifts the optimal candidate size brackets upward.  Four parallel
binarisation routes are used:

    Route A — Adaptive Gaussian threshold on the raw luminance plane.
    Route B — Otsu threshold on the *rphological gradient* output of
              the shared preprocessing pipeline.  The gradient image is
              already an edge-rich representation, so a global Otsu is
              well-defined; this route is particularly robust when the
              plate body and the surrounding bodywork have similar
              luminance.
    Route C — Canny edge cascade for shadow-occluded captures.
    Route D — Mean-relative dark mask for white-on-black plate styles.
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

VEHICLE_TYPE: str = "van"

ASPECT_PROFILE: tuple[tuple[float, float], ...] = (
    ASPECT_BAND_SINGLE_LINE,
    ASPECT_BAND_WIDE,
    ASPECT_BAND_VERY_WIDE,
)

_ROUTE_WEIGHTS: dict[str, float] = {
    "adaptive": 1.8,
    "morph_gradient": 2.2,
    "edge": 1.6,
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


def _morph_gradient_route(gradient_image: np.ndarray) -> np.ndarray:
    """Otsu-binarise the precomputed morphological gradient.

    The gradient image already concentrates information at character
    outlines, so a global Otsu produces a clean character-edge skeleton.
    A horizontal-close kernel then bridges adjacent character outlines
    into single plate blobs.
    """
    _, otsu_mask = cv2.threshold(
        gradient_image, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))
    return cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, close_kernel)


def _edge_route(denoised_frame: np.ndarray) -> np.ndarray:
    """Canny + wide-close + dilate cascade."""
    edge_layer = cv2.Canny(denoised_frame, 30, 100)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
    connected = cv2.morphologyEx(edge_layer, cv2.MORPH_CLOSE, close_kernel)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))
    return cv2.dilate(connected, dilate_kernel, iterations=2)


def _dark_contrast_route(luma_grid: np.ndarray) -> np.ndarray:
    """Mean-relative dark mask for white-on-black plate styles."""
    mean_intensity = float(np.mean(luma_grid))
    dark_mask = (luma_grid < mean_intensity * 0.7).astype(np.uint8) * 255

    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, connect_kernel)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    return cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, clean_kernel)


def _rank_candidate(region: CandidateRegion, frame_shape: tuple[int, ...]) -> float:
    """Van ranker — slightly higher small-plate floor than cars."""
    base = region.score
    spatial = position_aware_score(region, frame_shape, small_plate_area_floor=1800)
    route_weight = _ROUTE_WEIGHTS.get(region.method_tag, 1.0)
    return base * spatial * route_weight


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process(image_path: str) -> dict[str, Any]:
    """Detect and recognise a van plate in the supplied image."""
    debug_phases: dict[str, np.ndarray] = {}

    try:
        if not isinstance(image_path, str) or not os.path.isfile(image_path):
            return _failure(debug_phases, f"Image not readable at {image_path!r}.")

        source_frame = load_rgb_frame(image_path)
        debug_phases = run_preprocessing_pipeline(source_frame)
        luma_grid = debug_phases["grayscale"]
        denoised_frame = debug_phases["restored"]
        gradient_image = debug_phases["morphological"]

        candidate_pool: list[CandidateRegion] = []

        route_dispatch: tuple[tuple[Any, str, np.ndarray], ...] = (
            (_adaptive_route,        "adaptive",       luma_grid),
            (_morph_gradient_route,  "morph_gradient", gradient_image),
            (_edge_route,            "edge",           denoised_frame),
            (_dark_contrast_route,   "dark_contrast",  luma_grid),
        )
        for builder, route_tag, route_input in route_dispatch:
            try:
                mask = builder(route_input)
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

        if not candidate_pool:
            return _failure(debug_phases, "No plate-shaped contours found.")

        unique_regions = deduplicate_overlapping(candidate_pool, centre_distance_threshold=30)
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

        # Overlay in cyan (0, 200, 200) for visual distinction
        overlay = source_frame.copy()
        cv2.rectangle(
            overlay,
            (chosen_region.x, chosen_region.y),
            (chosen_region.x + chosen_region.w, chosen_region.y + chosen_region.h),
            (0, 200, 200), 3,
        )
        cv2.putText(
            overlay, cleaned_token,
            (chosen_region.x, max(0, chosen_region.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2,
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
        LOG.exception("Van processor crashed.")
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
    sample = sys.argv[1] if len(sys.argv) > 1 else "test_images/vans/van1.jpg"
    outcome = process(sample)
    print(f"[van_processor] success={outcome['success']}")
    print(f"  category={outcome['plate_category']}")
    print(f"  cleaned ={outcome['cleaned_text']!r}")
    print(f"  bbox    ={outcome['plate_bbox']}")
    print(f"  conf    ={outcome['confidence']:.3f}")
    if outcome["error_message"]:
        print(f"  error   ={outcome['error_message']}")