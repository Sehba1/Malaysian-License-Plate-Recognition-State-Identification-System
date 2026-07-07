"""
processors.jeep_processor
=========================

Plate localiser for **jeeps** (off-roaders, 4WDs, recreational utilities,
and — critically — Malaysian Armed Forces tactical vehicles).

Contribution: Muqri
Categories handled by this processor:
    * Standard         — civilian state-issued single-row plate
    * Military         — Z-prefix Armed Forces issuance (ZC = Maritime,
                         ZD = Air Force, ZB = Navy, ZA = Joint Forces,
                         ZL = Logistics, Z = Army)

Strategy
--------
A jeep is the only civilian vehicle class in the Malaysian taxonomy that
routinely shares its silhouette with an active military design (the
Weststar GK-M1 light tactical vehicle and successor platforms are
visually indistinguishable from a civilian Wrangler at typical CCTV
ranges).  Consequently this processor must do two jobs:

    (i)  Localise the plate using five independent classical routes.
         Three of the five routes target the standard civilian polarity
         (dark characters on a light or yellow background).  The two new
         routes — bright-polarity masking and CLAHE-enhanced adaptive
         thresholding — are aimed specifically at the Malaysian Armed
         Forces plate phenotype, which is a *white raised-letter plate on
         a jet-black background*.  Without these routes the system reliably
         detects only the numeric sub-group ("415") while discarding the
         service-branch prefix ("ZC"), causing the category to be
         misclassified as "Standard".

    (ii) After OCR, inspect the recognised token's prefix.  A leading
         ``Z`` followed by an optional second alpha and at least one
         digit is the canonical military grammar (``ZC415``).  If the
         token matches, the processor labels ``plate_category="Military"``
         so the orchestrator's state cascade can short-circuit to the
         ``Malaysian Armed Forces`` resolution.

         Additionally, when the best OCR result is purely numeric (e.g.
         "415" — only the number sub-group was detected), the processor
         performs an expanded-bbox retry that pads the candidate region
         leftward by 50 % and re-submits to OCR.  This compensates for
         the frequent scenario where the two character groups on a
         military plate are only partially bridged by morphological
         closing, resulting in two separate contours of which only the
         larger (numeric) group survived the area filter.

The state-name lookup itself is performed exclusively inside
``main_processor.identify_state`` per the architecture's single-source-of-
truth rule.  This processor never sets ``state_code`` or ``state_name``.

Mask routes
-----------
    Route A — Adaptive Gaussian threshold, window 11, C = 2.
               (donor M1 — civilian plate standard route)

    Route B — Bilateral-smoothed Canny edges (30 / 100) closed with
              (18, 3) and dilated with (8, 2).
               Gap-bridging width widened from (12, 3) to (18, 3) to
               reliably span the wide inter-group gap on military plates
               (≈ 20–30 px at typical CCTV capture resolution).
               (donor M3 — edge detection route)

    Route C — Sub-mean dark mask (intensity < 0.7 × mean) closed with
              (15, 3) and opened with (5, 3).
               Effective for civilian plates; contributes less signal for
               military plates but is retained for civilian jeep coverage.
               (donor M4 — dark contrast route)

    Route D — Super-mean bright mask (intensity > 1.1 × mean) closed with
              (25, 6) and dilated with (15, 3).
               Isolates white / silver character strokes on dark
               backgrounds.  The (25, 6) closing kernel is wide enough to
               bridge the inter-group gap on Malaysian Armed Forces plates
               and merge "ZC" and "415" into a single plate-shaped blob.
               (donor M5 — bright polarity route, newly ported)

    Route E — CLAHE-equalised luma, adaptive threshold window 15, C = 3.
               CLAHE (clipLimit 4.0, tile 4×4) dramatically improves local
               contrast on underlit military vehicles, recovering the
               plate-edge boundaries that the uniform bilateral smoothing
               in the standard pipeline flattens out.
               (new route — no donor equivalent)
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
    ASPECT_BAND_SQUARE,
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

VEHICLE_TYPE: str = "jeep"

# ---------------------------------------------------------------------------
# Aspect-ratio profile
# ---------------------------------------------------------------------------

# Standard jeep plates mirror sedan plates (single-row JPJ format).  The
# SQUARE band is included here — unlike other vehicle processors — because
# military tactical vehicles are sometimes photographed at steep angles where
# the perspective foreshortening pushes a nominally wide plate toward 1:1.
# The WIDE band handles long-body jeep plates at shallow depression angles.
ASPECT_PROFILE: tuple[tuple[float, float], ...] = (
    ASPECT_BAND_SINGLE_LINE,   # 1.8 – 6.0  standard civilian / military single-row
    ASPECT_BAND_WIDE,          # 2.8 – 4.0  wide single-row at close range
    ASPECT_BAND_SQUARE,        # 0.5 – 1.5  perspective-foreshortened military plate
)

# ---------------------------------------------------------------------------
# Compile-time constants
# ---------------------------------------------------------------------------

# Military prefix detector.  Intentionally permissive (prefix + one digit)
# because the orchestrator's MILITARY_REGEX owns final validation.
_MILITARY_PREFIX: re.Pattern[str] = re.compile(r"^Z[A-Z]?\d")

# Purely-numeric result detector — signals that detection captured only the
# number sub-group of a military plate, triggering the expanded-bbox retry.
_PURELY_NUMERIC: re.Pattern[str] = re.compile(r"^\d+$")

# Per-route scoring multipliers.  Routes D and E are upweighted because they
# are specifically engineered for the military plate phenotype; on a civilian
# jeep image they will simply produce fewer candidates than the adaptive and
# Canny routes (which remain the primary workhorses), so up-weighting them
# does not degrade civilian performance.
_ROUTE_WEIGHTS: dict[str, float] = {
    "adaptive":       2.0,   # Route A — civilian primary
    "canny_edge":     2.5,   # Route B — edge-shape primary
    "dark_contrast":  2.2,   # Route C — civilian dark plates
    "bright_polarity": 3.0,  # Route D — military white-on-dark  ← boosted
    "clahe_adaptive": 2.8,   # Route E — military underlit scenes ← boosted
}

# Fraction by which the candidate bbox is expanded leftward when the OCR
# fallback detects a purely-numeric result on a jeep image.  A 50 % leftward
# pad is empirically sufficient to capture the one-to-two-letter military
# service-branch prefix without pulling in competing background detail.
_MILITARY_BBOX_LEFT_EXPANSION: float = 0.50


# ---------------------------------------------------------------------------
# Mask generators — five independent binarisation strategies
# ---------------------------------------------------------------------------

def _adaptive_threshold_mask(luma_grid: np.ndarray) -> np.ndarray:
    """Locally-adaptive binarisation tolerant of cross-frame illumination shifts.

    Adaptive Gaussian thresholding computes a separate threshold for each
    pixel's local neighbourhood, which is exactly what is needed for
    outdoor jeep imagery where one half of the bonnet may be sunlit and
    the other in deep shade.  Window 11 / C 2 are the donor-validated
    parameters (donor Method 1).
    """
    return cv2.adaptiveThreshold(
        luma_grid, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2,
    )


def _canny_edge_mask(luma_grid: np.ndarray) -> np.ndarray:
    """Canny edges post-bridged into plate-shaped blobs.

    Jeeps photograph against complex backgrounds (foliage, off-road
    terrain) that overwhelm naïve edge detection; the prior bilateral
    smooth (with the donor's 11 / 17 / 17 triple) collapses leaf-edge
    micro-detail before Canny runs, leaving the plate boundary as one of
    the strongest survivors.

    **Fix applied (2025-11):** The horizontal closing kernel was widened
    from the original (12, 3) to (18, 3).  On a Malaysian Armed Forces
    plate the gap between the service-branch alpha group ("ZC") and the
    numeric body ("415") spans approximately 20–30 px at CCTV resolution;
    the former 12 px kernel consistently failed to bridge this gap,
    producing two separate sub-contours that each failed the single-plate
    aspect-ratio filter.  The (18, 3) kernel bridges gaps up to 18 px,
    and the subsequent (8, 2) dilation (2 iterations) adds a further
    ≈ 16 px, yielding a total bridging reach of roughly 34 px.
    (donor Method 3 — parameters updated)
    """
    denoised = cv2.bilateralFilter(luma_grid, 11, 17, 17)
    edge_mask = cv2.Canny(denoised, 30, 100)

    # Widened from (12, 3) to (18, 3) to bridge the military plate inter-group gap.
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 3))
    closed_pass = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, bridge_kernel)

    spread_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))
    return cv2.dilate(closed_pass, spread_kernel, iterations=2)


def _dark_contrast_mask(luma_grid: np.ndarray) -> np.ndarray:
    """Sub-mean dark-region mask, reconnected by wide horizontal closing.

    Most Malaysian civilian plates carry dark characters on a light or
    yellow background; pixels below 70 % of the frame's mean intensity
    therefore overlap the character strokes themselves.  We then bridge
    inter-character gaps with a (15, 3) closing and remove speckle noise
    with a (5, 3) opening — both kernel sizes are donor-preserved
    (donor Method 4).

    Note: this route is *not* effective for military plates (white text on
    black background) because the white characters reside above the mean,
    not below it.  Route D (_bright_polarity_mask) handles that polarity.
    """
    mean_intensity = float(np.mean(luma_grid))
    polarity_mask = (luma_grid < mean_intensity * 0.7).astype(np.uint8) * 255

    join_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    rejoined = cv2.morphologyEx(polarity_mask, cv2.MORPH_CLOSE, join_kernel)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    return cv2.morphologyEx(rejoined, cv2.MORPH_OPEN, clean_kernel)


def _bright_polarity_mask(luma_grid: np.ndarray) -> np.ndarray:
    """Super-mean bright-region mask for white-text-on-dark-background plates.

    Malaysian Armed Forces plates (ZC-, ZD-, ZB-, ZA-, ZL-, Z-prefix)
    carry white or silver raised characters on a jet-black background —
    the opposite polarity from all civilian plate types.  The three
    existing mask routes (adaptive, Canny, dark-contrast) were designed
    for the civilian polarity and produce either a filled dark blob (the
    plate body) or disconnected micro-contours on military plates, neither
    of which passes the shape-quality filter inside
    ``filter_contours_geometric``.

    This route isolates the *character strokes* directly by selecting
    pixels that exceed 110 % of the frame mean — a threshold borrowed from
    the donor's Method 5 ("bright_mask", used for bus plates which share
    the light-on-dark polarity).  The (25, 6) horizontal closing kernel is
    the donor's bus-route parameter and is wide enough to bridge the
    inter-group gap between the service-branch alpha prefix and the numeric
    body on a military plate without merging unrelated bright features
    (vehicle lights, sky patches).  The subsequent (15, 3) dilation adds
    robustness for plates at greater shooting distances.
    """
    mean_intensity = float(np.mean(luma_grid))
    bright_mask = (luma_grid > mean_intensity * 1.1).astype(np.uint8) * 255

    # Wide closing bridges "ZC" ↔ "415" character groups (donor M5 kernel).
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 6))
    bridged = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, bridge_kernel)

    spread_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(bridged, spread_kernel, iterations=1)

    # Light opening removes isolated bright specks (reflective bolts, rivets).
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    return cv2.morphologyEx(dilated, cv2.MORPH_OPEN, clean_kernel)


def _clahe_adaptive_mask(luma_grid: np.ndarray) -> np.ndarray:
    """CLAHE-equalised luma followed by wider adaptive thresholding.

    Military tactical vehicles are frequently photographed in shaded
    outdoor environments (covered vehicle bays, jungle approach roads,
    parade grounds under overcast skies).  The standard pipeline's
    bilateral-denoised luma tends to flatten local contrast in these
    conditions, burying the plate-edge gradient below the Canny threshold
    and making adaptive binarisation produce a salt-and-pepper background
    rather than clean character blobs.

    Contrast-Limited Adaptive Histogram Equalisation (CLAHE) with a 4×4
    tile grid restores local contrast on a per-region basis without
    globally amplifying noise.  The clipLimit = 4.0 matches the donor's
    CLAHE configuration used for its "dark image" detection case.  The
    subsequent adaptive threshold uses a 15-pixel window (wider than the
    civilian 11-pixel default) to accommodate the larger character
    dimensions typical of military embossed plates.
    """
    equaliser = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    enhanced_luma = equaliser.apply(luma_grid)

    return cv2.adaptiveThreshold(
        enhanced_luma, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 3,
    )


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def _rank_candidate(region: CandidateRegion, frame_shape: tuple[int, ...]) -> float:
    """Jeep-specific final ranker using the universal position-prior helper.

    The ``small_plate_area_floor`` is deliberately set at 1 200 px² rather
    than the default 1 000 px² because military tactical vehicles tend to
    be photographed at a distance from security cameras, making the plate
    subtend a smaller solid angle than on a civilian close-up shot.
    """
    base_score = region.score
    spatial_score = position_aware_score(
        region, frame_shape, small_plate_area_floor=1200,
    )
    route_weight = _ROUTE_WEIGHTS.get(region.method_tag, 1.0)
    return base_score * spatial_score * route_weight


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------

def _classify_plate_category(cleaned_token: str) -> str:
    """Decide between ``"Military"`` and ``"Standard"`` for a sanitised token.

    A Malaysian Armed Forces plate always begins with ``Z`` (optionally
    followed by one further service-branch letter, e.g. ``ZC`` for
    Maritime) and then a numeric body.  We intentionally accept the
    minimal prefix here rather than a fully validated grammar because the
    orchestrator's ``identify_state`` will re-run the strict regex anyway
    — premature filtering here would risk a real military plate being
    labelled ``"Standard"`` just because OCR dropped its trailing check
    letter.
    """
    if cleaned_token and _MILITARY_PREFIX.match(cleaned_token):
        return "Military"
    return "Standard"


# ---------------------------------------------------------------------------
# Military-specific OCR helpers
# ---------------------------------------------------------------------------

def _expand_bbox_leftward(
    bbox: tuple[int, int, int, int],
    expansion_fraction: float,
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Return a new bbox expanded leftward by ``expansion_fraction`` of its width.

    The expansion is clamped to the left image border so the returned bbox
    is always a valid slice of a frame with ``frame_shape``.

    Parameters
    ----------
    bbox:
        Original ``(x, y, w, h)`` bounding box.
    expansion_fraction:
        Fraction of ``w`` to add on the left side.  0.50 means "add half
        the current width to the left edge".
    frame_shape:
        ``(height, width[, channels])`` of the source frame, used for
        clamping.
    """
    x, y, w, h = bbox
    delta = int(w * expansion_fraction)
    new_x = max(0, x - delta)
    new_w = w + (x - new_x)   # preserve the right edge
    return new_x, y, new_w, h


def _retry_ocr_with_expanded_bbox(
    phase_images: dict[str, np.ndarray],
    original_bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
) -> tuple[str, float]:
    """Re-run multi-phase OCR on a left-expanded bounding box.

    Called when the primary OCR loop returns a purely-numeric token on a
    jeep image.  The intuition: if detection found only the numeric group
    ("415") of a military plate, the alpha prefix ("ZC") sits immediately
    to the left in the source frame.  Expanding the crop leftward by 50 %
    of the detected width includes that prefix region and lets OCR read the
    full token ("ZC415").
    """
    expanded_bbox = _expand_bbox_leftward(
        original_bbox, _MILITARY_BBOX_LEFT_EXPANSION, frame_shape,
    )
    LOG.debug(
        "Military prefix retry: original bbox %s → expanded bbox %s",
        original_bbox, expanded_bbox,
    )
    return arbitrate_multi_phase_ocr(
        phase_images,
        expanded_bbox,
        roi_prep_fn=prepare_roi_for_recognition,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process(image_path: str) -> dict[str, Any]:
    """Detect and recognise a jeep plate, classifying military vs. civilian.

    Parameters
    ----------
    image_path:
        Absolute or relative path to the input image file.

    Returns
    -------
    dict
        Canonical result envelope as defined in ``ARCHITECTURE.md``.
    """
    debug_phases: dict[str, np.ndarray] = {}

    try:
        if not isinstance(image_path, str) or not os.path.isfile(image_path):
            return _failure(debug_phases, f"Image not readable at {image_path!r}.")

        source_frame = load_rgb_frame(image_path)
        debug_phases = run_preprocessing_pipeline(source_frame)
        luma_grid = debug_phases["grayscale"]

        # --- Five independent mask routes -----------------------------------
        # Routes A–C cover the civilian polarity (dark-on-light / yellow).
        # Routes D–E cover the military polarity (white-on-dark / black).
        candidate_pool: list[CandidateRegion] = []

        for route_label, mask_builder in (
            ("adaptive",        _adaptive_threshold_mask),  # Route A
            ("canny_edge",      _canny_edge_mask),           # Route B
            ("dark_contrast",   _dark_contrast_mask),        # Route C
            ("bright_polarity", _bright_polarity_mask),      # Route D — military
            ("clahe_adaptive",  _clahe_adaptive_mask),       # Route E — military
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

        # Centre-distance NMS — suppress near-duplicate bboxes from different
        # routes that resolved to the same physical plate.
        unique_regions = deduplicate_overlapping(candidate_pool, centre_distance_threshold=30)
        unique_regions.sort(
            key=lambda region: _rank_candidate(region, source_frame.shape),
            reverse=True,
        )

        # --- OCR arbitration on top-N candidates ----------------------------
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

        # --- Military prefix recovery (expanded-bbox retry) -----------------
        # If the best OCR result is purely numeric (e.g. "415"), the detector
        # most likely captured only the number sub-group of a military plate.
        # Re-run OCR with a 50 %-leftward-expanded bounding box to include the
        # service-branch alpha prefix that sits immediately to the left.
        cleaned_for_check = apply_position_corrections(winning_token)
        if _PURELY_NUMERIC.match(cleaned_for_check):
            LOG.debug(
                "Numeric-only OCR result %r — attempting military prefix recovery.",
                cleaned_for_check,
            )
            retry_text, retry_confidence = _retry_ocr_with_expanded_bbox(
                debug_phases, chosen_region.bbox, source_frame.shape,
            )
            if retry_text:
                retry_clean = apply_position_corrections(retry_text)
                # Accept the expanded-bbox result if it adds an alpha prefix
                # (i.e. the new token is longer) and the confidence is not
                # catastrophically worse than the original.
                if (
                    len(retry_clean) > len(cleaned_for_check)
                    and retry_confidence >= winning_confidence * 0.65
                ):
                    LOG.info(
                        "Military prefix recovery succeeded: %r → %r",
                        cleaned_for_check, retry_clean,
                    )
                    winning_token = retry_text
                    winning_confidence = retry_confidence

        cleaned_token = apply_position_corrections(winning_token)
        if len(cleaned_token) < 3:
            return _failure(
                debug_phases,
                f"OCR result {winning_token!r} too short after sanitisation.",
            )

        # --- Category classification ----------------------------------------
        plate_category = _classify_plate_category(cleaned_token)

        # --- Overlay & crop --------------------------------------------------
        # Military plates receive an amber overlay so the operator can
        # immediately distinguish them from civilian detections at a glance.
        # This is a presentation-layer decision only; the orchestrator's
        # ``identify_state`` performs all authoritative state attribution.
        overlay_color = (255, 165, 0) if plate_category == "Military" else (0, 255, 0)
        overlay = source_frame.copy()
        cv2.rectangle(
            overlay,
            (chosen_region.x, chosen_region.y),
            (chosen_region.x + chosen_region.w, chosen_region.y + chosen_region.h),
            overlay_color, 3,
        )
        cv2.putText(
            overlay, cleaned_token,
            (chosen_region.x, max(0, chosen_region.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, overlay_color, 2,
        )
        debug_phases["detection_result"] = overlay

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
            "raw_ocr_text":   winning_token,
            "cleaned_text":   cleaned_token,
            "confidence":     min(1.0, winning_confidence),
            "debug_stages":   debug_phases,
            "error_message":  "",
        }

    except Exception as exc:  # noqa: BLE001
        LOG.exception("Jeep processor encountered an unhandled exception.")
        return _failure(debug_phases, f"Unhandled exception: {exc}")


def _failure(debug_phases: dict[str, np.ndarray], message: str) -> dict[str, Any]:
    """Construct the canonical failure envelope for ``process``."""
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
    sample = sys.argv[1] if len(sys.argv) > 1 else "test_images/jeeps/jeep1.jpg"
    outcome = process(sample)
    print(f"[jeep_processor] success={outcome['success']}")
    print(f"  category={outcome['plate_category']}")
    print(f"  cleaned ={outcome['cleaned_text']!r}")
    print(f"  bbox    ={outcome['plate_bbox']}")
    print(f"  conf    ={outcome['confidence']:.3f}")
    if outcome["error_message"]:
        print(f"  error   ={outcome['error_message']}")