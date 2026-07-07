"""
processors.pickup_processor
===========================

Plate localiser for **pickup trucks** (single-cab, double-cab, dual-cab
4×4 utilities).

Contribution: Rudra
Categories handled by this processor:
    * Standard         — civilian state-issued single-row plate

Strategy
--------
A pickup truck's plate is mechanically constrained to one of two
positions on the vehicle: the front bumper (when photographed head-on)
or the tailgate just above the rear bumper (when photographed from
behind).  Either way the plate ends up in the lower half of the frame
for a properly composed CCTV or smartphone shot — and the *upper* half
of such a shot is occupied by windscreen, roof rack, or cargo-bed
clutter that aggressively confuses any global binarisation.

We therefore run three routes:

    Route A — Bilateral-smoothed Canny edges (30 / 100) over the entire
              frame, closed with a (12, 3) rectangular kernel and
              dilated with (8, 2).  This is the donor's M3 method and
              acts as the general-purpose fallback.

    Route B — Sub-mean dark-contrast mask (intensity < 0.7 × mean)
              closed with (15, 3) and opened with (5, 3).  Donor M4,
              also full-frame.

    Route C — A *pickup-positional* re-run of Route A constrained to the
              lower half of the frame.  Coordinates of harvested
              candidates are rebased upward by the y-offset before they
              re-enter the global pool so that downstream stages (NMS,
              ranking, OCR arbitration) see them in full-frame space.
              This route is essentially the lower-ROI design from the
              donor's bus M6 method, but executed with Canny instead of
              inverted Otsu because pickup plates rarely have the high-
              contrast fleet livery that makes Otsu safe.

The route-trust weights below give Route C the highest base trust
because it has already paid the spatial-prior cost (clutter from above
is precluded by construction).
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

VEHICLE_TYPE: str = "pickup"

# Pickup plates obey civilian single-row JPJ dimensions; the very-wide
# band is excluded because the bumper-mount geometry of a pickup never
# produces a 4:1+ apparent aspect ratio at our capture distances.
ASPECT_PROFILE: tuple[tuple[float, float], ...] = (
    ASPECT_BAND_SINGLE_LINE,
    ASPECT_BAND_WIDE,
)

# Route trust weights.  ``lower_canny`` is preferred because the spatial
# prior is baked into the route by construction.  The full-frame routes
# remain as fallbacks for unusual compositions (e.g. drone shots).
_ROUTE_WEIGHTS: dict[str, float] = {
    "canny_edge": 2.0,
    "dark_contrast": 2.2,
    "lower_canny": 2.8,
}


# ---------------------------------------------------------------------------
# Mask generators
# ---------------------------------------------------------------------------

def _canny_edge_mask(luma_grid: np.ndarray) -> np.ndarray:
    """Bilateral + Canny + horizontal closing (donor M3).

    The bilateral pre-filter is essential before Canny on truck imagery:
    sheet-metal panels and tarpaulin straps produce ribbon-like edges
    that would otherwise dominate the output and starve the plate of
    contour pixels.  The (12, 3) closing then bridges fragmented
    character outlines into a single connected component and the (8, 2)
    dilation adds tolerance for slight motion blur.
    """
    denoised = cv2.bilateralFilter(luma_grid, 11, 17, 17)
    edge_mask = cv2.Canny(denoised, 30, 100)

    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
    bridged = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, bridge_kernel)

    spread_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))
    return cv2.dilate(bridged, spread_kernel, iterations=2)


def _dark_contrast_mask(luma_grid: np.ndarray) -> np.ndarray:
    """Sub-mean dark-region mask (donor M4).

    A pixel-wise polarity mask separating plate characters from their
    typically-lighter background.  Wider closing and opening kernels are
    used here than in the edge route because the polarity mask produces
    larger filled blobs that need both wider bridging and stronger noise
    suppression than thin edges do.
    """
    mean_intensity = float(np.mean(luma_grid))
    polarity_mask = (luma_grid < mean_intensity * 0.7).astype(np.uint8) * 255

    join_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed_pass = cv2.morphologyEx(polarity_mask, cv2.MORPH_CLOSE, join_kernel)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    return cv2.morphologyEx(closed_pass, cv2.MORPH_OPEN, clean_kernel)


def _lower_half_canny_mask(luma_grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Run the M3 Canny pipeline on only the lower half of the frame.

    The returned y-offset (always ``height // 2``) must be added back to
    every harvested ``y`` coordinate before the candidate enters the
    full-frame pool — see ``_harvest_lower_region``.

    Using the lower half rather than the bus processor's lower 2/3 is a
    deliberate tightening for pickup geometry: the plate on a pickup is
    notably *below* the centre-line because both mounting points
    (bumper, tailgate) sit at or below the line of the wheel arches.
    """
    full_height = luma_grid.shape[0]
    y_offset = full_height // 2
    lower_segment = luma_grid[y_offset:, :]

    denoised = cv2.bilateralFilter(lower_segment, 11, 17, 17)
    edge_mask = cv2.Canny(denoised, 30, 100)

    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
    closed_pass = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, bridge_kernel)

    spread_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))
    return cv2.dilate(closed_pass, spread_kernel, iterations=2), y_offset


# ---------------------------------------------------------------------------
# Lower-ROI candidate harvesting (with y-rebasing)
# ---------------------------------------------------------------------------

def _harvest_lower_region(
    mask: np.ndarray,
    y_offset: int,
    source_frame: np.ndarray,
) -> list[CandidateRegion]:
    """Run the geometric filter on a cropped mask and translate y-coords back.

    The geometric filter receives a *cropped* source frame so that its
    own colour/texture descriptors operate on the same pixels that
    produced the mask.  After filtering, every bounding box has its
    ``y`` coordinate shifted by ``y_offset`` so the rest of the
    pipeline can treat it identically to a full-frame candidate.
    """
    cropped_source = source_frame[y_offset:, :]
    harvested = filter_contours_geometric(
        mask,
        source_frame=cropped_source,
        aspect_bands=ASPECT_PROFILE,
        method_tag="lower_canny",
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
    """Pickup-specific final ranker.

    Pickup plates are slightly larger in the frame than sedan plates on
    average (larger vehicle, closer working distance for typical front-
    or rear-shot composition), so we raise the small-plate floor to 2000
    pixels rather than the universal 1000.  Below this floor the
    candidate is almost certainly debris.
    """
    base_score = region.score
    spatial_score = position_aware_score(region, frame_shape, small_plate_area_floor=2000)
    route_weight = _ROUTE_WEIGHTS.get(region.method_tag, 1.0)
    return base_score * spatial_score * route_weight


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process(image_path: str) -> dict[str, Any]:
    """Detect and recognise a pickup-truck plate in the supplied image."""
    debug_phases: dict[str, np.ndarray] = {}

    try:
        if not isinstance(image_path, str) or not os.path.isfile(image_path):
            return _failure(debug_phases, f"Image not readable at {image_path!r}.")

        source_frame = load_rgb_frame(image_path)
        debug_phases = run_preprocessing_pipeline(source_frame)
        luma_grid = debug_phases["grayscale"]

        candidate_pool: list[CandidateRegion] = []

        # --- Route A: full-frame Canny edges -------------------------------
        try:
            canny_mask = _canny_edge_mask(luma_grid)
            candidate_pool.extend(
                filter_contours_geometric(
                    canny_mask,
                    source_frame=source_frame,
                    aspect_bands=ASPECT_PROFILE,
                    method_tag="canny_edge",
                )
            )
        except cv2.error as exc:
            LOG.debug("Canny-edge route failed: %s", exc)

        # --- Route B: full-frame dark contrast -----------------------------
        try:
            dark_mask = _dark_contrast_mask(luma_grid)
            candidate_pool.extend(
                filter_contours_geometric(
                    dark_mask,
                    source_frame=source_frame,
                    aspect_bands=ASPECT_PROFILE,
                    method_tag="dark_contrast",
                )
            )
        except cv2.error as exc:
            LOG.debug("Dark-contrast route failed: %s", exc)

        # --- Route C: lower-half positional Canny --------------------------
        try:
            lower_mask, y_offset = _lower_half_canny_mask(luma_grid)
            candidate_pool.extend(_harvest_lower_region(lower_mask, y_offset, source_frame))
        except cv2.error as exc:
            LOG.debug("Lower-half Canny route failed: %s", exc)

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
        LOG.exception("Pickup processor crashed.")
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
    sample = sys.argv[1] if len(sys.argv) > 1 else "test_images/pickups/pickup1.jpg"
    outcome = process(sample)
    print(f"[pickup_processor] success={outcome['success']}")
    print(f"  category={outcome['plate_category']}")
    print(f"  cleaned ={outcome['cleaned_text']!r}")
    print(f"  bbox    ={outcome['plate_bbox']}")
    print(f"  conf    ={outcome['confidence']:.3f}")
    if outcome["error_message"]:
        print(f"  error   ={outcome['error_message']}")