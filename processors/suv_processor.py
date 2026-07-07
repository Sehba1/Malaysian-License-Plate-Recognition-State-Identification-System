"""
processors.suv_processor
========================

Plate localiser for **SUVs and crossovers** (sport-utility vehicles, CUVs).

Contribution: Egor
Categories handled by this processor:
    * Standard         — civilian state-issued single-row plate
    * Special Series   — vanity / federal series tokens such as
                         PUTRAJAYA, PATRIOT, MADANI, EV, VIP, GOLD, LIMO

Strategy
--------
SUVs carry a standard single-row plate whose canonical aspect ratio is
approximately 4.7 : 1 (520 mm × 110 mm).  What distinguishes the SUV
detection context from a saloon car is the physical proximity of the
plate to the front or rear bumper chrome trim and the radiator grille —
both of which are dark or highly reflective regions that can be confused
with the plate in coarse morphological passes.

Five parallel mask routes are run against complementary image
representations.  Route ordering is deliberately chosen so that the
most discriminating routes (Otsu-inverse and adaptive) generate
candidates first, reducing the risk that high-scoring false positives
from the chrome route dominate the Top-10 window:

    Route A — Adaptive Gaussian threshold    (donor Method 1, block=11, C=2)
    Route B — Bilateral + Canny edge cascade (donor Method 3, kernel (12,3))
    Route C — Dark-region CLOSE              (donor Method 4, kernel (15,3))
    Route D — Otsu-inverse for black plates  (donor Method 2 — white-on-black)
    Route E — Bright-region for chrome/vanity plates

Post-route, all candidates pass through ``filter_contours_geometric``
which applies donor-reference aspect-ratio bands, area-fraction gates,
and colour/texture statistical descriptors before entering the OCR
arbitration loop.

References
----------
Gonzalez & Woods (2018), *Digital Image Processing* (4th ed.), §9.3
    (morphological reconstruction) and §10.2 (thresholding).
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
    ASPECT_BAND_WIDE,
    ASPECT_BAND_VERY_WIDE,
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

VEHICLE_TYPE: str = "suv"

# ---------------------------------------------------------------------------
# Aspect-ratio profile
# ---------------------------------------------------------------------------
# Standard Malaysian civilian plate: ~520 mm × 110 mm → aspect ≈ 4.7
# Special series plates can be shorter (e.g. "EV 1") → aspect ≈ 2.3
# Both regimes are covered by SINGLE_LINE (1.8–6.0).  WIDE and VERY_WIDE
# are retained because chrome vanity plates occasionally present at
# exaggerated aspect ratios when the photo is taken from a low angle.
ASPECT_PROFILE: tuple[tuple[float, float], ...] = (
    ASPECT_BAND_SINGLE_LINE,   # 1.8 – 6.0  (primary — covers ~4.7 standard)
    ASPECT_BAND_WIDE,          # 2.8 – 4.0  (overlapping sub-band, boosts score)
    ASPECT_BAND_VERY_WIDE,     # 4.0 – 7.0  (vanity/wide-format plates)
)

# Per-route confidence multipliers applied inside the candidate ranker.
# The Otsu-inverse route receives the highest weight because it is the most
# selective binarisation for high-contrast white-on-black plates (the
# dominant SUV plate style in the Malaysian corpus).
_ROUTE_WEIGHTS: dict[str, float] = {
    "adaptive":      1.8,
    "edge":          1.6,
    "dark_contrast": 2.0,
    "otsu_inverse":  2.2,   # Route D — new, highest specificity
    "bright_chrome": 1.4,   # Route E — reduced; chrome generates false positives
}

# Diplomatic plate pattern — retained for robustness against user mis-selection.
_DIPLOMATIC_LEADING: re.Pattern[str] = re.compile(r"^\d{0,3}(CC|CD|DC)\d{1,4}")


# ---------------------------------------------------------------------------
# Lazy import of the special-series registry
# ---------------------------------------------------------------------------

def _is_special_series_token(cleaned_token: str) -> bool:
    """Test whether a sanitised plate string begins with a recognised vanity token.

    The leading-substring + digit-boundary match policy mirrors the policy in
    ``main_processor.identify_state`` so that the processor's category
    annotation is always consistent with the orchestrator's final resolution.
    The token registry is imported lazily to avoid the import-time circular
    dependency between ``main_processor`` and the processor modules it owns.
    """
    if not cleaned_token:
        return False
    try:
        from main_processor import SPECIAL_SERIES_TOKENS
    except ImportError:
        return False

    for vanity_token in sorted(SPECIAL_SERIES_TOKENS, key=len, reverse=True):
        if cleaned_token.startswith(vanity_token):
            tail = cleaned_token[len(vanity_token):]
            if (not tail) or tail[0].isdigit():
                return True
    return False


# ---------------------------------------------------------------------------
# Route builders — each accepts one pre-computed image plane and returns
# a binary mask ready for ``filter_contours_geometric``.
# ---------------------------------------------------------------------------

def _adaptive_route(luma_grid: np.ndarray) -> np.ndarray:
    """Route A — Adaptive Gaussian threshold.

    The (11, 2) window/constant pair is the donor-validated configuration
    (``detect_license_plate_regions`` Method 1).  The 11-pixel block width
    approximates the height of plate characters at typical capture
    resolutions, making the threshold locally responsive to the
    character/background polarity without being fooled by global
    illumination gradients.
    """
    return cv2.adaptiveThreshold(
        luma_grid, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2,
    )


def _edge_route(denoised_frame: np.ndarray) -> np.ndarray:
    """Route B — Canny edge detection followed by a wide morphological close.

    Kernel dimensions are taken verbatim from the donor's Method 3:
        ``kernel_rect  = (12, 3)``  — connects horizontal character runs
        ``kernel_dilate = (8, 2)``  — widens the connected edge map
    Two dilation iterations are the donor's empirically chosen value.
    """
    edge_map = cv2.Canny(denoised_frame, 30, 100)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
    connected = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, close_kernel)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))
    return cv2.dilate(connected, dilate_kernel, iterations=2)


def _dark_contrast_route(luma_grid: np.ndarray) -> np.ndarray:
    """Route C — Mean-relative dark-region mask for white-on-black plates.

    The 0.7× mean threshold and the (15, 3) / (5, 3) kernel pair are
    carried from the donor's Method 4.  A critical SUV-specific hazard is
    that the radiator grille mesh, which sits immediately above or below
    the plate, is also a dark region.  The donor's (15, 3) CLOSE kernel
    can bridge the plate background into the grille via a thin horizontal
    dark corridor.

    Mitigation — after the standard donor OPEN pass, a one-column vertical
    ERODE with a (1, 3) kernel severs any residual narrow horizontal
    bridges without disturbing the plate's own compact dark mass.
    """
    mean_intensity = float(np.mean(luma_grid))
    raw_dark_mask = (luma_grid < mean_intensity * 0.7).astype(np.uint8) * 255

    # Donor-reference kernels (Method 4)
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    bridged = cv2.morphologyEx(raw_dark_mask, cv2.MORPH_CLOSE, connect_kernel)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    cleaned = cv2.morphologyEx(bridged, cv2.MORPH_OPEN, clean_kernel)

    # SUV-specific bridge-breaker: narrow vertical erosion severs thin
    # horizontal connections to the grille without collapsing the plate blob.
    bridge_breaker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    return cv2.erode(cleaned, bridge_breaker, iterations=1)


def _otsu_inverse_route(luma_grid: np.ndarray) -> np.ndarray:
    """Route D — Otsu binarisation on the inverted luma plane.

    This route is the direct translation of the donor's Method 2
    (``dark_thresh`` via Otsu on an inverted image) and is the most
    reliable path for the dominant Malaysian SUV plate style: white
    characters on a black background (e.g. "NDM 9414", "IM 18").

    Inverting before Otsu causes the bright characters to present as
    foreground (white in the binary), then a (13, 3) CLOSE bridges the
    individual character strokes into a compact plate-shaped rectangle,
    and a (5, 3) OPEN removes isolated noise specks.  The (13, 3) width
    is slightly narrower than the dark-contrast route's (15, 3) because
    the characters in this path are isolated bright islands rather than
    a continuous dark field, making over-bridging less likely.

    This route was absent from the original processor — its absence was
    the primary cause of missed detections on standard white-on-black SUV
    plates.
    """
    inverted_luma = cv2.bitwise_not(luma_grid)
    _, binary_mask = cv2.threshold(
        inverted_luma, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
    connected = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, connect_kernel)
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    return cv2.morphologyEx(connected, cv2.MORPH_OPEN, clean_kernel)


def _bright_chrome_route(luma_grid: np.ndarray) -> np.ndarray:
    """Route E — Bright-region mask for chrome-finish vanity plates.

    This route targets reflective chrome or metallic SUV vanity plates
    whose background saturates toward the sensor ceiling.  Two corrective
    adjustments relative to the original implementation:

    1.  Threshold raised from 1.2× to 1.4× mean intensity.  At 1.2× the
        mask indiscriminately captures SUV chrome grille surrounds, headlamp
        bezels, and chrome window trim, all of which are bright but are NOT
        plates.  The 1.4× threshold restricts candidates to regions that are
        genuinely over-exposed, which chrome plates tend to be in direct
        sunlight but ambient body trim is not.

    2.  Connect kernel reduced from (20, 4) to (15, 3) to match the donor's
        proven standard-plate bridge width (Method 4).  The original (20, 4)
        kernel was merging horizontally disparate chrome features into plate-
        sized blobs that then competed with — and often outranked — the
        genuine plate in the OCR arbitration window.
    """
    mean_intensity = float(np.mean(luma_grid))
    bright_mask = (luma_grid > mean_intensity * 1.4).astype(np.uint8) * 255

    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    connected = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, connect_kernel)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    return cv2.morphologyEx(connected, cv2.MORPH_OPEN, clean_kernel)


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------

def _classify_plate_category(cleaned_token: str) -> str:
    """Map a sanitised OCR token to its plate category label.

    Priority ordering: Special Series → Diplomatic → Standard.
    Military (Z-prefix) is intentionally excluded because military
    vehicles are classified under the 'jeep' taxonomy in our system.
    """
    if not cleaned_token:
        return "Unknown"
    if _is_special_series_token(cleaned_token):
        return "Special Series"
    if _DIPLOMATIC_LEADING.match(cleaned_token):
        return "Diplomatic"
    return "Standard"


# ---------------------------------------------------------------------------
# Candidate ranker
# ---------------------------------------------------------------------------

def _rank_candidate(region: CandidateRegion, frame_shape: tuple[int, ...]) -> float:
    """Compute a composite rank score for one plate candidate.

    The base geometric score from ``filter_contours_geometric`` is
    multiplied by a spatial position prior (plates live in the lower
    half of vehicle photographs) and by the per-route confidence weight
    defined in ``_ROUTE_WEIGHTS``.
    """
    base_score = region.score
    spatial_multiplier = position_aware_score(
        region, frame_shape, small_plate_area_floor=2000,
    )
    route_multiplier = _ROUTE_WEIGHTS.get(region.method_tag, 1.0)
    return base_score * spatial_multiplier * route_multiplier


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process(image_path: str) -> dict[str, Any]:
    """Detect and recognise an SUV plate from a single image file.

    Parameters
    ----------
    image_path:
        Absolute or relative path to the image.  Must resolve to a
        readable regular file; directory paths and missing files both
        produce an immediate ``success=False`` return.

    Returns
    -------
    dict
        Fully-populated processor data contract (ten keys).  ``success``
        is ``True`` only when a candidate was localised AND PaddleOCR
        produced at least three alphanumeric characters.
    """
    debug_phases: dict[str, np.ndarray] = {}

    try:
        if not isinstance(image_path, str) or not os.path.isfile(image_path):
            return _failure(debug_phases, f"Image not readable at {image_path!r}.")

        source_frame = load_rgb_frame(image_path)
        debug_phases = run_preprocessing_pipeline(source_frame)
        luma_grid: np.ndarray = debug_phases["grayscale"]
        denoised_frame: np.ndarray = debug_phases["restored"]

        candidate_pool: list[CandidateRegion] = []

        # Route dispatch table: (builder_fn, route_tag, input_plane)
        # Order: high-specificity routes first so their candidates enter
        # the deduplication pool before the noisier routes do.
        route_dispatch: tuple[tuple[Any, str, np.ndarray], ...] = (
            (_otsu_inverse_route,  "otsu_inverse",  luma_grid),      # Route D — primary
            (_adaptive_route,      "adaptive",      luma_grid),      # Route A
            (_edge_route,          "edge",          denoised_frame), # Route B
            (_dark_contrast_route, "dark_contrast", luma_grid),      # Route C
            (_bright_chrome_route, "bright_chrome", luma_grid),      # Route E
        )

        for builder_fn, route_tag, input_plane in route_dispatch:
            try:
                binary_mask = builder_fn(input_plane)
                candidate_pool.extend(
                    filter_contours_geometric(
                        binary_mask,
                        source_frame=source_frame,
                        aspect_bands=ASPECT_PROFILE,
                        method_tag=route_tag,
                    )
                )
            except cv2.error as exc:
                LOG.debug("SUV route '%s' raised a cv2.error: %s", route_tag, exc)

        if not candidate_pool:
            return _failure(debug_phases, "No plate-shaped contours survived geometric filtering.")

        # Deduplicate near-coincident candidates that originated from
        # different binarisation routes; keep highest-scoring duplicate.
        unique_regions = deduplicate_overlapping(candidate_pool, centre_distance_threshold=30)
        unique_regions.sort(
            key=lambda r: _rank_candidate(r, source_frame.shape),
            reverse=True,
        )

        # Probe the top-ten ranked candidates through the OCR arbitrator.
        top_candidates = unique_regions[: min(10, len(unique_regions))]
        chosen_region: CandidateRegion | None = None
        winning_raw_text: str = ""
        winning_confidence: float = 0.0

        for region in top_candidates:
            raw_text, attempt_confidence = arbitrate_multi_phase_ocr(
                debug_phases,
                region.bbox,
                roi_prep_fn=prepare_roi_for_recognition,
            )
            if not raw_text:
                continue
            if attempt_confidence > winning_confidence or chosen_region is None:
                chosen_region = region
                winning_raw_text = raw_text
                winning_confidence = attempt_confidence
                is_valid, _, _ = validate_plate_syntax(raw_text)
                if is_valid and attempt_confidence > 0.85:
                    break  # Early-exit: high-confidence syntactically valid hit

        if chosen_region is None or not winning_raw_text:
            return _failure(debug_phases, "OCR produced no readable text across all candidates.")

        cleaned_token = apply_position_corrections(winning_raw_text)
        if len(cleaned_token) < 3:
            return _failure(
                debug_phases,
                f"OCR result {winning_raw_text!r} too short after sanitisation ({len(cleaned_token)} chars).",
            )

        plate_category = _classify_plate_category(cleaned_token)

        # Annotate the detection result phase for GUI debug display.
        detection_overlay = source_frame.copy()
        cv2.rectangle(
            detection_overlay,
            (chosen_region.x, chosen_region.y),
            (chosen_region.x + chosen_region.w, chosen_region.y + chosen_region.h),
            (200, 0, 200), 3,
        )
        cv2.putText(
            detection_overlay, cleaned_token,
            (chosen_region.x, max(0, chosen_region.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2,
        )
        debug_phases["detection_result"] = detection_overlay

        plate_crop = source_frame[
            chosen_region.y: chosen_region.y + chosen_region.h,
            chosen_region.x: chosen_region.x + chosen_region.w,
        ].copy()

        return {
            "success":        True,
            "vehicle_type":   VEHICLE_TYPE,
            "plate_category": plate_category,
            "plate_bbox":     chosen_region.bbox,
            "plate_image":    plate_crop,
            "raw_ocr_text":   winning_raw_text,
            "cleaned_text":   cleaned_token,
            "confidence":     min(1.0, winning_confidence),
            "debug_stages":   debug_phases,
            "error_message":  "",
        }

    except Exception as exc:  # noqa: BLE001
        LOG.exception("SUV processor encountered an unhandled exception.")
        return _failure(debug_phases, f"Unhandled exception: {exc}")


# ---------------------------------------------------------------------------
# Failure envelope helper
# ---------------------------------------------------------------------------

def _failure(debug_phases: dict[str, np.ndarray], message: str) -> dict[str, Any]:
    """Construct a contract-compliant failure dictionary.

    Every key required by the processor data contract is present so that
    the GUI and orchestrator never encounter a ``KeyError`` on a failed run.
    """
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

    sample_path = sys.argv[1] if len(sys.argv) > 1 else "test_images/suvs/suv01.jpg"
    result = process(sample_path)
    print(f"[suv_processor] success   = {result['success']}")
    print(f"                category  = {result['plate_category']}")
    print(f"                cleaned   = {result['cleaned_text']!r}")
    print(f"                bbox      = {result['plate_bbox']}")
    print(f"                conf      = {result['confidence']:.3f}")
    if result["error_message"]:
        print(f"                error     = {result['error_message']}")