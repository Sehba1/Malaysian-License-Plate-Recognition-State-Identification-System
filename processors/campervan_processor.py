"""
processors.campervan_processor
==============================

Plate localiser for **camper vans** (motorhomes, converted-vans, RVs).

Contribution: Muqri
Categories handled by this processor:
    * Standard         — civilian state-issued single-row plate

Strategy
--------
Camper vans straddle the boundary between sedans and small trucks: they
carry a standard single-row JPJ plate mounted on a tall, painted rear or
front panel.  Two photographic regularities follow from that geometry:

    1.  The plate frequently photographs *wider* than on a car at the same
        physical capture distance, because the dorsal RV panel is mounted
        further forward than a sedan boot lid and the camera-to-plate
        perspective therefore foreshortens the plate less aggressively.
        We accommodate this by widening the aspect-ratio envelope to admit
        the very-wide band (4.0 - 7.0) in addition to the conventional
        single-line and wide bands.

    2.  The panel paint is typically a solid pastel colour (white, beige,
        cream) with no chrome trim around the plate, which means simple
        adaptive thresholding picks out plate characters cleanly.  Dark-
        contrast and Canny-edge mask routes are added as redundant
        validators — at least two of the three routes usually agree.

Mask routes (terminology aligned with the donor's M1 / M3 / M4 labels):

    Route A — Adaptive Gaussian threshold (window 11, C = 2).  General
              purpose; handles uniform lighting well.
    Route B — Bilateral-smoothed Canny edges (30 / 100) closed with a
              (12, 3) rectangular kernel and dilated with (8, 2).  Robust
              to plate paint variations.
    Route C — Dark-region mask (pixels below 0.7 × mean intensity) closed
              with a wide (15, 3) kernel.  Targets the high-contrast
              character/background polarity directly.

All numerical constants are inherited verbatim from the donor's empirically
calibrated pipeline.
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

VEHICLE_TYPE: str = "campervan"

# Camper-van aspect envelope: in addition to the canonical single-line band
# we admit both wide and very-wide brackets to absorb the elongated rear-
# panel framing characteristic of RV photographs.
ASPECT_PROFILE: tuple[tuple[float, float], ...] = (
    ASPECT_BAND_SINGLE_LINE,
    ASPECT_BAND_WIDE,
    ASPECT_BAND_VERY_WIDE,
)

# Per-route trust weights.  M3 (Canny) is the most informative route for
# camper bodywork because it does not rely on global intensity statistics
# the way the contrast mask does — paint colour varies enormously across
# the camper-van segment.
_ROUTE_WEIGHTS: dict[str, float] = {
    "adaptive": 2.0,
    "canny_edge": 2.5,
    "dark_contrast": 2.2,
}


# ---------------------------------------------------------------------------
# Mask generators — each isolates one binarisation strategy
# ---------------------------------------------------------------------------

def _adaptive_threshold_mask(luma_grid: np.ndarray) -> np.ndarray:
    """Adaptive-Gaussian binary with the donor's (11, 2) window.

    The 11-pixel window is broadly the height of a Malaysian plate
    character in our test corpus, so the local mean it computes
    approximates the plate-background level and the thresholding becomes
    locally polarity-correct regardless of global illumination.
    """
    return cv2.adaptiveThreshold(
        luma_grid, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2,
    )


def _canny_edge_mask(luma_grid: np.ndarray) -> np.ndarray:
    """Bilateral-smoothed Canny edges fused with horizontal closing.

    Bilateral filtering before Canny suppresses paint-grain noise on
    camper bodywork without smearing the plate boundary; the (30, 100)
    Canny thresholds are deliberately permissive at the low end because
    the closing pass that follows is responsible for stitching the
    fragmented edges of dimly-lit characters back into a single plate
    blob.
    """
    denoised = cv2.bilateralFilter(luma_grid, 11, 17, 17)
    edge_mask = cv2.Canny(denoised, 30, 100)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
    bridged = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, close_kernel)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))
    return cv2.dilate(bridged, dilate_kernel, iterations=2)


def _dark_contrast_mask(luma_grid: np.ndarray) -> np.ndarray:
    """Mean-relative dark-region mask, morphologically reconnected.

    Malaysian plates run dark characters on a light field roughly 70 %
    of the time in our corpus; pixels below 0.7 × mean intensity
    therefore capture the character body itself, and a wide (15, 3)
    closing reunites the disconnected glyphs into a single bounding
    rectangle.  A subsequent (5, 3) opening removes small noise blobs
    that survive thresholding.
    """
    mean_intensity = float(np.mean(luma_grid))
    polarity_mask = (luma_grid < mean_intensity * 0.7).astype(np.uint8) * 255

    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed_pass = cv2.morphologyEx(polarity_mask, cv2.MORPH_CLOSE, bridge_kernel)

    denoise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    return cv2.morphologyEx(closed_pass, cv2.MORPH_OPEN, denoise_kernel)


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def _rank_candidate(region: CandidateRegion, frame_shape: tuple[int, ...]) -> float:
    """Camper-specific final ranker — keeps the conventional 1000-px floor.

    Camper plates are roughly sedan-sized in the image, so we leave the
    small-plate floor at the universal 1000-pixel default and let the
    route-trust weights do the discrimination work between mask sources.
    """
    base_score = region.score
    spatial_score = position_aware_score(region, frame_shape, small_plate_area_floor=1000)
    route_weight = _ROUTE_WEIGHTS.get(region.method_tag, 1.0)
    return base_score * spatial_score * route_weight


# ---------------------------------------------------------------------------
# Public entry point — strict ten-key dict contract
# ---------------------------------------------------------------------------

def process(image_path: str) -> dict[str, Any]:
    """Detect and recognise a camper-van plate in the supplied image.

    The function honours the architecture's zero-crash policy: every
    failure mode is converted into a populated ``success=False`` dict so
    the GUI can render an explanation rather than confront an exception.
    """
    debug_phases: dict[str, np.ndarray] = {}

    try:
        if not isinstance(image_path, str) or not os.path.isfile(image_path):
            return _failure(debug_phases, f"Image not readable at {image_path!r}.")

        source_frame = load_rgb_frame(image_path)
        debug_phases = run_preprocessing_pipeline(source_frame)
        luma_grid = debug_phases["grayscale"]

        # --- Three parallel binarisation routes -----------------------------
        candidate_pool: list[CandidateRegion] = []

        for route_label, mask_builder in (
            ("adaptive", _adaptive_threshold_mask),
            ("canny_edge", _canny_edge_mask),
            ("dark_contrast", _dark_contrast_mask),
        ):
            try:
                binary_mask = mask_builder(luma_grid)
                candidate_pool.extend(
                    filter_contours_geometric(
                        binary_mask,
                        source_frame=source_frame,
                        aspect_bands=ASPECT_PROFILE,
                        method_tag=route_label,
                    )
                )
            except cv2.error as exc:
                LOG.debug("Route %s failed: %s", route_label, exc)

        if not candidate_pool:
            return _failure(debug_phases, "No plate-shaped contours found.")

        unique_regions = deduplicate_overlapping(candidate_pool, centre_distance_threshold=30)
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
        LOG.exception("Campervan processor crashed.")
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
    sample = sys.argv[1] if len(sys.argv) > 1 else "test_images/campervans/campervan1.jpg"
    outcome = process(sample)
    print(f"[campervan_processor] success={outcome['success']}")
    print(f"  category={outcome['plate_category']}")
    print(f"  cleaned ={outcome['cleaned_text']!r}")
    print(f"  bbox    ={outcome['plate_bbox']}")
    print(f"  conf    ={outcome['confidence']:.3f}")
    if outcome["error_message"]:
        print(f"  error   ={outcome['error_message']}")