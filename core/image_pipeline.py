"""
core.image_pipeline
===================

Vehicle-agnostic image-processing primitives used by every processor in the
``processors/`` package.

The module is organised as five layered concerns:

    Layer A — Acquisition & enhancement   : RGB load, grayscale conversion,
                                            histogram + gamma rebalancing.
    Layer B — Restoration & multi-domain  : bilateral denoise, HSV value
                                            channel, ``db4`` wavelet detail
                                            magnitude, compression simulation.
    Layer C — Morphology & segmentation   : structural gradient, adaptive
                                            Gaussian threshold.
    Layer D — Candidate geometry          : contour harvesting under a
                                            tunable aspect/area envelope.
    Layer E — Recognition preparation     : ROI extraction with size-aware
                                            upscaling for small motorcycle
                                            plates and unsharp masking for
                                            larger automobile plates.

Layers A through C together implement the academic nine-phase pipeline that
the assignment requires us to demonstrate in the GUI debug panel.  Layer D
is invoked by every vehicle processor after it has produced its own per-
vehicle binary masks (rule §3 of the architecture document forbids
centralising those masks).  Layer E is invoked by the OCR arbitration
routine in ``core.ocr_engine``.

The cardinal rule for this module: nothing here may know which vehicle type
is being processed.  Vehicle-specific kernels and aspect bands stay in the
respective processor file.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import pywt

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateRegion:
    """A single rectangular plate candidate harvested from a binary mask.

    Carrying the data as a frozen dataclass rather than a positional tuple
    eliminates an entire class of indexing bugs that plagued the donor's
    six-element tuple ``(x, y, w, h, area, method)`` style — every consumer
    site here can refer to ``region.bbox`` or ``region.score`` by name.
    """
    bbox: tuple[int, int, int, int]
    score: float
    method_tag: str = "generic"

    @property
    def x(self) -> int: return self.bbox[0]

    @property
    def y(self) -> int: return self.bbox[1]

    @property
    def w(self) -> int: return self.bbox[2]

    @property
    def h(self) -> int: return self.bbox[3]

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


# ---------------------------------------------------------------------------
# Layer A — Acquisition & enhancement
# ---------------------------------------------------------------------------

def load_rgb_frame(image_path: str) -> np.ndarray:
    """Read an image from disk and return it in canonical RGB byte order.

    OpenCV's native order is BGR which is a frequent source of subtle
    colour-channel bugs downstream; converting here once means every later
    stage can safely assume RGB.
    """
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image at {image_path!r}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_luminance(source_frame: np.ndarray) -> np.ndarray:
    """Convert an RGB or already-grayscale frame to a single-channel luma image."""
    if source_frame.ndim == 2:
        return source_frame.copy()
    return cv2.cvtColor(source_frame, cv2.COLOR_RGB2GRAY)


def _equalise_then_gamma(luma_grid: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Histogram-equalise the luma plane then apply a mild gamma rebalance.

    Equalisation alone tends to over-flatten night-time and back-lit scenes;
    a small positive gamma after the fact brings highlights back toward the
    middle of the dynamic range without undoing the contrast win.  The
    1.2 exponent is the donor's empirically chosen value and is retained.
    """
    equalised = cv2.equalizeHist(luma_grid)
    # In-place power transform via float buffer avoids the silent integer
    # underflow that ``equalised ** gamma`` would suffer on a uint8 array.
    normalised = equalised.astype(np.float32) / 255.0
    return np.clip(255.0 * np.power(normalised, gamma), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Layer B — Restoration & multi-domain transforms
# ---------------------------------------------------------------------------

def _bilateral_restore(tonemap_output: np.ndarray) -> np.ndarray:
    """Edge-preserving denoise tuned for plate-character preservation.

    The ``(d=11, sigmaColor=17, sigmaSpace=17)`` triple keeps the high-
    contrast plate borders crisp while flattening textured backgrounds (car
    paint, road tarmac).  These values are the donor's and have been
    preserved verbatim because they have been validated across the entire
    test corpus.
    """
    return cv2.bilateralFilter(tonemap_output, d=11, sigmaColor=17, sigmaSpace=17)


def _extract_value_channel(source_frame: np.ndarray, fallback_luma: np.ndarray) -> np.ndarray:
    """Return the V channel from the HSV decomposition.

    The V plane is brightness-only and discards hue/saturation noise, which
    makes it useful as an alternative OCR substrate when the original RGB
    image has unusual colour casts (yellow plates on yellow taxis, white
    plates on white SUVs, etc.).  Falls back to a luma copy for already-
    grayscale inputs.
    """
    if source_frame.ndim != 3:
        return fallback_luma.copy()
    hsv_space = cv2.cvtColor(source_frame, cv2.COLOR_RGB2HSV)
    return hsv_space[:, :, 2]


def _wavelet_detail_magnitude(denoised_frame: np.ndarray) -> np.ndarray:
    """Compute the Euclidean magnitude of the three ``db4`` detail bands.

    ``LH + HL + HH`` quadrature highlights edge energy at the wavelet's
    natural scale — this is more selective than a raw Sobel because the
    ``db4`` filter bank is approximately tuned to character-stroke width on
    typical plate crops.
    """
    try:
        approx, (h_detail, v_detail, d_detail) = pywt.dwt2(denoised_frame, "db4")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Wavelet decomposition failed (%s); falling back to denoised image.", exc)
        return denoised_frame.copy()

    detail_magnitude = np.sqrt(h_detail ** 2 + v_detail ** 2 + d_detail ** 2)
    lo, hi = float(detail_magnitude.min()), float(detail_magnitude.max())
    if hi <= lo:
        return np.zeros_like(detail_magnitude, dtype=np.uint8)
    rescaled = (detail_magnitude - lo) * (255.0 / (hi - lo))
    return rescaled.astype(np.uint8)


def _simulate_compression(denoised_frame: np.ndarray) -> np.ndarray:
    """Down-sample by four then up-sample to model JPEG-style compression.

    The point is not to compress for storage — it is to verify the pipeline
    is tolerant of the loss-of-detail conditions typical of CCTV uploads.
    """
    height, width = denoised_frame.shape[:2]
    small_w = max(1, width // 4)
    small_h = max(1, height // 4)
    downsampled = cv2.resize(denoised_frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(downsampled, (width, height), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# Layer C — Morphology & segmentation
# ---------------------------------------------------------------------------

def _morphological_gradient(denoised_frame: np.ndarray) -> np.ndarray:
    """Close-then-gradient cascade producing an edge mask.

    The closing fills small gaps inside characters; the subsequent gradient
    surfaces the now-continuous outlines of those characters.  The
    ``(3, 3)`` rectangle is the donor-preserved kernel size.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_pass = cv2.morphologyEx(denoised_frame, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(closed_pass, cv2.MORPH_GRADIENT, kernel)


def _adaptive_gaussian_segment(denoised_frame: np.ndarray) -> np.ndarray:
    """Adaptive-threshold the denoised image with the donor's ``(11, 2)`` window.

    The 11-pixel window is roughly the height of plate characters at our
    test resolutions, which makes the threshold respond locally to the
    character/background polarity instead of being fooled by global
    illumination gradients.
    """
    return cv2.adaptiveThreshold(
        denoised_frame, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2,
    )


# ---------------------------------------------------------------------------
# Public preprocessing entry point
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(source_frame: np.ndarray) -> dict[str, np.ndarray]:
    """Execute the nine-phase academic pipeline and return all intermediates.

    The returned dictionary is the substrate from which every downstream
    consumer draws:
        * vehicle processors read ``enhanced`` / ``restored`` / ``morphological``
          to build their own binary masks for contour harvesting,
        * the OCR arbitrator reads ``restored`` / ``enhanced`` / ``color_processed``
          to run multi-phase OCR,
        * the Tkinter GUI displays the entire grid in its debug panel.

    Key
    ---
    ``original``         RGB acquisition.
    ``grayscale``        Single-plane luma.
    ``enhanced``         Equalised + gamma-rebalanced luma (Phase 2).
    ``restored``         Bilateral-denoised enhancement (Phase 3).
    ``color_processed``  HSV value channel (Phase 4).
    ``wavelet``          ``db4`` detail magnitude (Phase 5).
    ``compressed``       Quarter-rate compression simulation (Phase 6).
    ``morphological``    Closed-then-gradient edge mask (Phase 7).
    ``segmented``        Adaptive-Gaussian binary (Phase 8).

    Phase 9 (representation/description) lives inside each processor — that
    is where vehicle-specific contour harvesting and labelling occurs.
    """
    luma_grid = to_luminance(source_frame)
    tonemap_output = _equalise_then_gamma(luma_grid)
    denoised_frame = _bilateral_restore(tonemap_output)

    phase_images: dict[str, np.ndarray] = {
        "original": source_frame,
        "grayscale": luma_grid,
        "enhanced": tonemap_output,
        "restored": denoised_frame,
        "color_processed": _extract_value_channel(source_frame, luma_grid),
        "wavelet": _wavelet_detail_magnitude(denoised_frame),
        "compressed": _simulate_compression(denoised_frame),
        "morphological": _morphological_gradient(denoised_frame),
        "segmented": _adaptive_gaussian_segment(denoised_frame),
    }
    return phase_images


# ---------------------------------------------------------------------------
# Layer D — Geometric candidate filter
# ---------------------------------------------------------------------------

# Aspect-ratio bands.  Each processor will declare which subset of these
# applies to its vehicle type.  They are exposed as module-level constants
# so downstream code can name them rather than hard-coding magic numbers.
ASPECT_BAND_SINGLE_LINE: tuple[float, float] = (1.8, 6.0)
ASPECT_BAND_TWO_LINE:    tuple[float, float] = (0.7, 3.0)
ASPECT_BAND_SQUARE:      tuple[float, float] = (0.5, 1.5)
ASPECT_BAND_MOTORCYCLE:  tuple[float, float] = (1.0, 2.8)
ASPECT_BAND_WIDE:        tuple[float, float] = (2.8, 4.0)
ASPECT_BAND_VERY_WIDE:   tuple[float, float] = (4.0, 7.0)

# The full union, used by processors that have no opinion about plate
# geometry and want to harvest everything plate-shaped.
DEFAULT_ASPECT_BANDS: tuple[tuple[float, float], ...] = (
    ASPECT_BAND_SINGLE_LINE,
    ASPECT_BAND_TWO_LINE,
    ASPECT_BAND_SQUARE,
    ASPECT_BAND_MOTORCYCLE,
    ASPECT_BAND_WIDE,
    ASPECT_BAND_VERY_WIDE,
)


def _aspect_in_any_band(aspect: float, bands: tuple[tuple[float, float], ...]) -> bool:
    """Membership test across a union of half-open aspect bands."""
    return any(lo <= aspect <= hi for lo, hi in bands)


def _passes_shape_quality(
    region_area: float,
    perimeter_length: float,
    contour_points: np.ndarray,
    bbox_w: int,
    bbox_h: int,
) -> bool:
    """Test convexity, rectangularity and fill-extent against donor envelopes.

    These four shape descriptors filter out elongated road markings, glass
    reflections and bumper trims that incidentally pass the aspect-ratio
    test but are not actually rectangles.
    """
    bbox_area = bbox_w * bbox_h
    if bbox_area <= 0 or perimeter_length <= 0:
        return False

    fill_extent = region_area / bbox_area
    if fill_extent <= 0.20:
        return False

    convex_hull = cv2.convexHull(contour_points)
    hull_area = cv2.contourArea(convex_hull)
    if hull_area <= 0:
        return False
    convex_solidity = region_area / hull_area
    if convex_solidity <= 0.4:
        return False

    roundness_index = 4.0 * np.pi * region_area / (perimeter_length * perimeter_length)
    if not (0.02 <= roundness_index <= 0.98):
        return False

    arc_tolerance = 0.02 * perimeter_length
    polygon_approx = cv2.approxPolyDP(contour_points, arc_tolerance, True)
    vertex_count = len(polygon_approx)
    return 3 <= vertex_count <= 12


def score_color_signature(roi_rgb: np.ndarray) -> tuple[bool, float]:
    """Score how plate-like a colour region looks via its V-channel statistics.

    A Malaysian civilian plate is either bright-on-dark or dark-on-bright;
    the V channel of the HSV decomposition contains both regimes in nearly
    bimodal form.  Three exclusive sub-patterns are scored and the strongest
    is taken as the confidence in this colour signature.

    Returns ``(is_plate_like, confidence)`` where confidence is in [0, 1].
    """
    if roi_rgb is None or roi_rgb.size == 0 or roi_rgb.ndim != 3:
        return False, 0.0

    try:
        hsv_space = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
        value_layer = hsv_space[:, :, 2]
    except cv2.error:
        return False, 0.0

    total_pixels = roi_rgb.shape[0] * roi_rgb.shape[1]
    if total_pixels == 0:
        return False, 0.0

    bright_fraction = float(np.sum(value_layer > 180)) / total_pixels
    dark_fraction = float(np.sum(value_layer < 75)) / total_pixels

    pattern_bright_plate = bright_fraction > 0.6 and dark_fraction > 0.05
    pattern_dark_plate = dark_fraction > 0.5 and bright_fraction > 0.05
    pattern_contrast_mix = (0.2 <= bright_fraction <= 0.8) and (0.1 <= dark_fraction <= 0.6)

    background_sample = roi_rgb[value_layer > np.median(value_layer)]
    if background_sample.size > 30:
        background_std = float(np.std(background_sample))
        uniformity = 1.0 / (1.0 + background_std / 50.0)
    else:
        uniformity = 0.0

    is_plate_like = pattern_bright_plate or pattern_dark_plate or pattern_contrast_mix
    pattern_weight = 0.8 if pattern_bright_plate else (0.6 if pattern_dark_plate else 0.4)
    composite_confidence = uniformity * pattern_weight
    return is_plate_like, composite_confidence


def score_texture_signature(roi_luma: np.ndarray) -> tuple[bool, float]:
    """Score how plate-like a region's gradient texture is.

    The horizontal projection of the Sobel-magnitude image carries one peak
    per character stroke; regular peak spacing therefore correlates with
    "this region contains evenly spaced glyphs".  The texture variance is
    also scored against a heuristic target of 1500 — empirically the band in
    which plate ROIs cluster on our dataset.
    """
    if roi_luma is None or roi_luma.size == 0:
        return False, 0.0

    grad_x = cv2.Sobel(roi_luma, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(roi_luma, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    horizontal_profile = np.mean(gradient_magnitude, axis=0)

    threshold = float(np.mean(horizontal_profile)) * 0.5
    peak_indices: list[int] = []
    for i in range(1, len(horizontal_profile) - 1):
        sample = horizontal_profile[i]
        if (
            sample > threshold
            and sample > horizontal_profile[i - 1]
            and sample > horizontal_profile[i + 1]
        ):
            peak_indices.append(i)

    if len(peak_indices) >= 2:
        spacings = np.diff(peak_indices)
        mean_spacing = float(np.mean(spacings))
        if mean_spacing > 0:
            spacing_irregularity = float(np.std(spacings)) / mean_spacing
            spacing_regularity = max(0.0, min(1.0, 1.0 - spacing_irregularity))
        else:
            spacing_regularity = 0.0
    else:
        spacing_regularity = 0.0

    texture_variance = float(np.var(roi_luma))
    texture_score = 1.0 / (1.0 + abs(texture_variance - 1500.0) / 1000.0)

    is_plate_like = (spacing_regularity > 0.3) or (texture_score > 0.5)
    confidence = spacing_regularity * 0.6 + texture_score * 0.4
    return is_plate_like, confidence


def filter_contours_geometric(
    binary_mask: np.ndarray,
    source_frame: np.ndarray | None = None,
    aspect_bands: tuple[tuple[float, float], ...] = DEFAULT_ASPECT_BANDS,
    method_tag: str = "generic",
) -> list[CandidateRegion]:
    """Harvest plausible plate rectangles from a binary mask.

    Every contour is screened against five geometric envelopes before it can
    enter the candidate list:

        1. Aspect ratio falls inside at least one supplied band.
        2. Area is between 0.001 % and 20 % of frame area.
        3. Pixel-space dimensions exceed (12, 4).
        4. Pixel-space dimensions stay under (0.8 W, 0.6 H).
        5. The shape passes the convexity / rectangularity / fill-extent
           envelope defined by ``_passes_shape_quality``.

    Surviving candidates have their area amplified by the colour and
    texture confidence of their interior pixels, so that visually plate-
    like rectangles are ranked above shape-only matches that happen to have
    correct dimensions but look nothing like a plate.

    Notes
    -----
    The donor module also ran an unconditional ``cv2.matchTemplate`` against
    synthetic single-line and two-line glyph templates as a third heuristic.
    That stage was deliberately deleted to comply with the assignment's
    explicit ban on pattern-matching methods; we lean entirely on contour
    geometry plus the colour/texture statistical descriptors instead.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []

    image_area = binary_mask.shape[0] * binary_mask.shape[1]
    max_w_allowed = binary_mask.shape[1] * 0.8
    max_h_allowed = binary_mask.shape[0] * 0.6

    accepted: list[CandidateRegion] = []
    for contour in contours:
        region_area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        aspect_value = w / float(h)
        if not _aspect_in_any_band(aspect_value, aspect_bands):
            continue

        area_fraction = (w * h) / image_area
        if not (0.00001 <= area_fraction <= 0.20):
            continue

        if w <= 12 or h <= 4:
            continue
        if w > max_w_allowed or h > max_h_allowed:
            continue

        perimeter_length = cv2.arcLength(contour, True)
        if not _passes_shape_quality(region_area, perimeter_length, contour, w, h):
            continue

        composite_score = float(region_area)
        if source_frame is not None and source_frame.size > 0:
            roi_rgb = source_frame[y:y + h, x:x + w]
            roi_luma = to_luminance(roi_rgb) if roi_rgb.ndim == 3 else roi_rgb

            color_match, color_conf = score_color_signature(roi_rgb if roi_rgb.ndim == 3 else cv2.cvtColor(roi_rgb, cv2.COLOR_GRAY2RGB))
            texture_match, texture_conf = score_texture_signature(roi_luma)

            statistical_confidence = color_conf * 0.4 + texture_conf * 0.6
            no_descriptor_agreed = (not color_match) and (not texture_match)
            if statistical_confidence < 0.1 and no_descriptor_agreed:
                continue

            composite_score *= (1.0 + statistical_confidence)

        accepted.append(
            CandidateRegion(
                bbox=(int(x), int(y), int(w), int(h)),
                score=composite_score,
                method_tag=method_tag,
            )
        )

    return accepted


# ---------------------------------------------------------------------------
# Center-distance NMS deduplication
# ---------------------------------------------------------------------------

def deduplicate_overlapping(
    regions: list[CandidateRegion],
    centre_distance_threshold: int = 30,
) -> list[CandidateRegion]:
    """Suppress near-duplicate candidates that originate from different masks.

    Because each vehicle processor runs several binarisation strategies in
    parallel, the same physical plate is frequently detected several times
    with marginally different bounding boxes.  We use a centre-distance
    proximity test rather than IoU because the boxes can differ
    significantly in tightness while still corresponding to the same plate
    — IoU is too pessimistic in that regime.

    For each near-duplicate cluster the highest-scoring member is kept.
    """
    survivors: list[CandidateRegion] = []
    for candidate in regions:
        cx, cy = candidate.center
        merged = False
        for index, existing in enumerate(survivors):
            ex_cx, ex_cy = existing.center
            if abs(cx - ex_cx) < centre_distance_threshold and abs(cy - ex_cy) < centre_distance_threshold:
                merged = True
                if candidate.score > existing.score:
                    survivors[index] = candidate
                break
        if not merged:
            survivors.append(candidate)
    return survivors


# ---------------------------------------------------------------------------
# Layer E — ROI preparation for OCR
# ---------------------------------------------------------------------------

def _clamp_bbox_to_frame(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Clip a bounding box to the valid pixel range of its parent frame."""
    height, width = frame_shape[:2]
    x, y, w, h = bbox
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    x_end = min(x + w, width)
    y_end = min(y + h, height)
    return x, y, max(1, x_end - x), max(1, y_end - y)


def prepare_roi_for_recognition(
    phase_image: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop a plate bounding box and condition it for OCR.

    Two distinct conditioning regimes are applied based on ROI area:

        * Small ROIs (< 3000 px², characteristic of motorcycle plates at
          mid-to-far distance) are isotropically upscaled to at least
          60 pixels tall and aggressively contrast-stretched.  PaddleOCR's
          recognition stage has a minimum-height threshold around 32 px;
          undercutting that threshold collapses recall sharply, and
          upscaling before recognition is empirically more effective than
          letting PaddleOCR resize internally.

        * Larger ROIs receive a gentler contrast-stretch (only fired when
          the histogram dynamic range is below 100/255) and an unsharp-mask
          finishing pass.

    Both regimes preserve the donor's empirically chosen thresholds.
    """
    if phase_image is None or phase_image.size == 0:
        return np.zeros((50, 100), dtype=np.uint8)

    x, y, w, h = _clamp_bbox_to_frame(bbox, phase_image.shape)
    roi = phase_image[y:y + h, x:x + w]
    if roi.size == 0:
        return np.zeros((50, 100), dtype=np.uint8)

    conditioned = roi.copy()
    pixel_count = conditioned.shape[0] * conditioned.shape[1]

    if pixel_count < 3000:
        shortest_side = min(conditioned.shape[:2])
        scale_factor = max(2, int(60 / max(1, shortest_side)))
        if scale_factor > 1:
            new_size = (conditioned.shape[1] * scale_factor, conditioned.shape[0] * scale_factor)
            conditioned = cv2.resize(conditioned, new_size, interpolation=cv2.INTER_CUBIC)

        # Aggressive contrast stretch for small text
        low, high = int(np.min(conditioned)), int(np.max(conditioned))
        dynamic_range = high - low
        if 0 < dynamic_range < 150:
            conditioned = ((conditioned.astype(np.float32) - low) * (255.0 / dynamic_range))
            conditioned = np.clip(conditioned, 0, 255).astype(np.uint8)

        if conditioned.ndim == 2:
            conditioned = cv2.medianBlur(conditioned, 3)

    else:
        # Gentle stretch only fires when contrast is genuinely poor.
        low, high = int(np.min(conditioned)), int(np.max(conditioned))
        dynamic_range = high - low
        if 0 < dynamic_range < 100:
            conditioned = ((conditioned.astype(np.float32) - low) * (255.0 / dynamic_range))
            conditioned = np.clip(conditioned, 0, 255).astype(np.uint8)

        # Unsharp mask finishing pass for plates large enough not to suffer
        # from the inevitable noise amplification.
        if conditioned.shape[0] > 30 and conditioned.shape[1] > 60:
            blurred = cv2.GaussianBlur(conditioned, (3, 3), 1.0)
            conditioned = cv2.addWeighted(conditioned, 1.2, blurred, -0.2, 0)

    return conditioned


# ---------------------------------------------------------------------------
# Position-aware scoring used by every processor (shared but tunable)
# ---------------------------------------------------------------------------

def position_aware_score(
    region: CandidateRegion,
    frame_shape: tuple[int, ...],
    small_plate_area_floor: int = 1000,
) -> float:
    """Compute a position- and size-prior score multiplier for a candidate.

    The scalar returned here is intended to multiply the geometric base
    score so that plates with sensible image positions (lower-half, away
    from frame edges) and sensible sizes (in the empirical 200 - 15000 px²
    band) rank above outliers.

    Each vehicle processor calls this same function but supplies its own
    geometric base score and post-multipliers — the position prior is
    universal, the size prior is universal in shape but each processor can
    adjust ``small_plate_area_floor`` to better match its vehicle class
    (motorcycle processors will leave it at 1000; bus processors might
    raise it dramatically).
    """
    img_h, img_w = frame_shape[:2]
    x, y, w, h = region.bbox
    area = float(w) * float(h)
    is_small = area < small_plate_area_floor

    # --- Edge penalty -------------------------------------------------------
    if y <= 2 or x <= 2 or (y + h) >= (img_h - 2) or (x + w) >= (img_w - 2):
        edge_factor = 0.001
    elif (not is_small) and (
        y < 20 or x < 20 or (y + h) > (img_h - 20) or (x + w) > (img_w - 20)
    ):
        edge_factor = 0.1
    elif is_small and (
        y < 10 or x < 10 or (y + h) > (img_h - 10) or (x + w) > (img_w - 10)
    ):
        edge_factor = 0.5
    else:
        edge_factor = 1.0

    # --- Vertical position bonus ------------------------------------------
    centre_y = y + h / 2.0
    if centre_y > img_h * 0.7:
        position_factor = 1.5
    elif centre_y > img_h * 0.3:
        position_factor = 2.0
    else:
        position_factor = 1.0

    # --- Size bracket bonus ------------------------------------------------
    if area > 25000:
        size_factor = 0.2
    elif area > 15000:
        size_factor = 0.4
    elif 5000 <= area <= 15000:
        size_factor = 1.5
    elif 2000 <= area <= 5000:
        size_factor = 1.8
    elif 800 <= area <= 2000:
        size_factor = 2.5
    elif 400 <= area <= 800:
        size_factor = 2.8
    elif 200 <= area <= 400:
        size_factor = 2.2
    elif 100 <= area <= 200:
        size_factor = 1.8
    elif 50 <= area <= 100:
        size_factor = 1.2
    else:
        size_factor = 0.3

    # --- Aspect ratio bonus ------------------------------------------------
    aspect = w / float(h) if h > 0 else 0.0
    if 3.0 <= aspect <= 4.5:
        aspect_factor = 1.2
    elif 1.4 <= aspect <= 2.2:
        aspect_factor = 1.5
    elif 1.0 <= aspect <= 1.4:
        aspect_factor = 1.3
    elif 2.2 <= aspect <= 3.0:
        aspect_factor = 1.1
    elif 0.7 <= aspect <= 1.0:
        aspect_factor = 1.2
    else:
        aspect_factor = 1.0

    return edge_factor * position_factor * size_factor * aspect_factor