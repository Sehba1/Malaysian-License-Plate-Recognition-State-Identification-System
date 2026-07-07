"""
main_processor
==============

Orchestrator and Single Source of Truth for State Identification.

This module fulfils three contractual obligations defined by the project's
architecture document:

    1.  It hosts every lookup table and compiled regular expression that
        encodes the Malaysian numbering system.  Vehicle processors stay
        completely state-agnostic by design — duplication of these tables
        across ten processor files would create a maintenance and grading
        hazard that we expressly avoid.

    2.  It exposes ``run(image_path, vehicle_type=None)`` as the single
        public function the Tkinter GUI is allowed to call.  No GUI code
        reaches past this boundary; no processor reaches past it in the
        other direction.

        When ``vehicle_type`` is supplied the directed-dispatch path is
        used (original behaviour, fully preserved for backward
        compatibility).  When ``vehicle_type`` is ``None`` (the new
        default after the GUI checkbox was removed) the module runs a
        **Chain-of-Responsibility** pass: every installed processor is
        invoked silently and the single result that achieved the highest
        OCR confidence is returned.

    3.  It enriches the dictionary returned by each processor with two
        derived keys (``state_code`` and ``state_name``) and never permits
        a processor to write those keys directly.

The state-identification logic implements a strict priority cascade:

        Military regex   →   Diplomatic regex   →   Taxi prefix map
                         →   Special-series tokens  →   State-code lookup

The cascade matters because Malaysian special-purpose plates often share
prefix letters with civilian series — a Z-prefix military plate must be
recognised as military rather than mis-attributed to a civilian state that
happens to share a leading letter; an EV vanity series must not be confused
with a hypothetical "E" state.
"""

from __future__ import annotations

import logging
import os
import re
from importlib import import_module
from typing import Any, Callable

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


# ---------------------------------------------------------------------------
# State-identification reference data
# ---------------------------------------------------------------------------

# Canonical single- and multi-letter prefix → state lookup for civilian
# plates.  Multi-letter prefixes appear here so that ``identify_state`` can
# resolve them by longest-match before falling back to single letters
# (e.g. ``KV`` for Langkawi must outrank ``K`` for Kedah).
STATE_MAP: dict[str, str] = {
    # --- Peninsula ---
    "A":  "Perak",
    "B":  "Selangor",
    "C":  "Pahang",
    "D":  "Kelantan",
    "J":  "Johor",
    "K":  "Kedah",
    "M":  "Melaka",
    "N":  "Negeri Sembilan",
    "P":  "Pulau Pinang",
    "R":  "Perlis",
    "T":  "Terengganu",
    # --- East Malaysia ---
    "Q":  "Sarawak",
    "S":  "Sabah",
    # --- Federal Territories ---
    "F":  "Putrajaya",
    "L":  "Labuan",
    "V":  "Kuala Lumpur",
    "W":  "Kuala Lumpur",
    # --- Special tourism prefix ---
    "KV": "Langkawi",
}

# Taxi service prefix mapping.  Malaysian taxis carry an "H" family prefix
# tied to the issuing region.  We surface those region tags here so that
# taxi plates display a sensible "state" even though they are not civilian
# state issuances strictly speaking.
TAXI_PREFIX_MAP: dict[str, str] = {
    "HW":  "Kuala Lumpur (Taxi)",
    "HBA": "Selangor (Taxi)",
    "HBB": "Selangor (Taxi)",
    "HBC": "Selangor (Taxi)",
    "HC":  "Selangor (Taxi)",
    "HKL": "Kuala Lumpur (Taxi)",
    "KX":  "Sabah (Taxi)",
    "LM":  "Limousine",
    "TX":  "Terengganu (Taxi)",
}

# Vanity / federal special-issue tokens.  These appear as full prefixes
# rather than letter-by-letter codes, so the matching policy is exact rather
# than longest-prefix.  Tokens are stored uppercase to match the canonical
# form produced by ``core.ocr_engine.strip_to_alphanumeric``.
SPECIAL_SERIES_TOKENS: frozenset[str] = frozenset({
    "PUTRAJAYA",
    "PATRIOT",
    "MERDEKA",
    "MADANI",
    "PETRA",
    "JAGUH",
    "BIJAK",
    "PERMATA",
    "PROTON",
    "PERODUA",
    "MALAYSIA",
    "MYVI",
    "VIP",
    "GOLD",
    "LIMO",
    "BIRU",
    "EMAS",
    "FFF",
    "EV",
    "G1M",
    "1M4U",
    "NAZA",
    "WAJA",
    "U",
    "X",
    "Y",
})

# Military plates begin with Z, optionally followed by one further letter
# (Army Z, Navy ZB, Air-Force ZD, etc.), then up to four digits and an
# optional check letter.
MILITARY_REGEX: re.Pattern[str] = re.compile(r"^Z[A-Z]?\d{1,4}[A-Z]?$")

# Diplomatic plates in Malaysia follow a ``[country-code]-CC-[serial]`` or
# ``[country-code]-CD-[serial]`` form.  After sanitisation the dashes
# disappear, so we test for the presence of ``CC`` or ``CD`` at the front of
# the alpha section.
DIPLOMATIC_REGEX: re.Pattern[str] = re.compile(r"^\d{0,3}(CC|CD|DC)\d{1,4}[A-Z]?$")


# ---------------------------------------------------------------------------
# Vehicle dispatch table — populated at import time
# ---------------------------------------------------------------------------

# The valid taxonomy strings.  Each must map to a processor module whose
# public ``process(image_path) -> dict`` callable is reachable.
_SUPPORTED_VEHICLES: tuple[str, ...] = (
    "car", "bus", "motorcycle", "truck", "van",
    "suv", "campervan", "jeep", "pickup", "minibus",
)


def _try_import_processor(vehicle_token: str) -> Callable[[str], dict[str, Any]] | None:
    """Resolve a vehicle taxonomy token to its processor's ``process`` callable.

    The dispatcher is permissive at import time: a missing processor logs a
    warning and disables that vehicle category rather than aborting the
    entire program.  This is a deliberate concession to parallel team
    development — if a teammate has not yet committed their module, the
    other processors and the GUI still work.
    """
    module_name = f"processors.{vehicle_token}_processor"
    try:
        module = import_module(module_name)
    except ImportError as exc:
        LOG.warning("Processor module %s could not be imported: %s", module_name, exc)
        return None

    process_fn = getattr(module, "process", None)
    if not callable(process_fn):
        LOG.warning(
            "Processor module %s has no callable 'process' entry point.", module_name,
        )
        return None
    return process_fn


VEHICLE_DISPATCH: dict[str, Callable[[str], dict[str, Any]] | None] = {
    vehicle: _try_import_processor(vehicle) for vehicle in _SUPPORTED_VEHICLES
}


# ---------------------------------------------------------------------------
# State-identification cascade
# ---------------------------------------------------------------------------

def _match_military(token: str) -> tuple[str, str] | None:
    """Return a (code, name) pair for Z-prefix military plates, or ``None``."""
    if MILITARY_REGEX.match(token):
        leading_alpha = re.match(r"^Z[A-Z]?", token)
        code = leading_alpha.group(0) if leading_alpha else "Z"
        return code, "Malaysian Armed Forces"
    return None


def _match_diplomatic(token: str) -> tuple[str, str] | None:
    """Return a (code, name) pair for CC/CD/DC diplomatic plates, or ``None``."""
    if DIPLOMATIC_REGEX.match(token):
        body = re.search(r"(CC|CD|DC)", token)
        code = body.group(0) if body else "CC"
        return code, "Diplomatic Corps"
    return None


def _match_taxi(token: str) -> tuple[str, str] | None:
    """Resolve a multi-letter taxi prefix by longest-match policy."""
    for prefix in sorted(TAXI_PREFIX_MAP.keys(), key=len, reverse=True):
        if token.startswith(prefix):
            return prefix, TAXI_PREFIX_MAP[prefix]
    return None


def _match_special_series(token: str) -> tuple[str, str] | None:
    """Resolve a vanity / federal series token by leading-substring match."""
    for vanity_token in sorted(SPECIAL_SERIES_TOKENS, key=len, reverse=True):
        if token.startswith(vanity_token) and (
            len(token) == len(vanity_token)
            or token[len(vanity_token)].isdigit()
        ):
            return vanity_token, "Special Series"
    return None


def _match_state(token: str) -> tuple[str, str] | None:
    """Resolve a civilian state-code prefix by longest-match policy."""
    for prefix in sorted(STATE_MAP.keys(), key=len, reverse=True):
        if token.startswith(prefix):
            return prefix, STATE_MAP[prefix]
    return None


def identify_state(cleaned_text: str, plate_category: str = "Standard") -> tuple[str, str]:
    """Resolve a sanitised plate token to a (state_code, state_name) pair.

    The function implements the priority cascade documented at the top of
    this module.  The ``plate_category`` argument acts as a hint from the
    processor — if a processor has already classified its plate as
    ``"Military"`` we honour that classification without re-running the
    regex.  Conversely, ``"Standard"`` does not preclude the cascade from
    discovering that the plate is in fact one of the special categories.

    Returns ``("", "Unknown")`` if no rule fires.
    """
    if not cleaned_text:
        return "", "Unknown"

    token = cleaned_text.upper()

    # If the caller has already pre-classified, honour and short-circuit.
    if plate_category == "Military":
        outcome = _match_military(token)
        if outcome:
            return outcome
    if plate_category == "Diplomatic":
        outcome = _match_diplomatic(token)
        if outcome:
            return outcome
    if plate_category == "Taxi":
        outcome = _match_taxi(token)
        if outcome:
            return outcome
    if plate_category == "Special Series":
        outcome = _match_special_series(token)
        if outcome:
            return outcome

    # Full cascade — even a processor that returned ``"Standard"`` can be
    # overridden if the regex discovers an obviously special plate.
    for matcher in (
        _match_military, _match_diplomatic, _match_taxi,
        _match_special_series, _match_state,
    ):
        outcome = matcher(token)
        if outcome:
            return outcome

    return "", "Unknown"


# ---------------------------------------------------------------------------
# Output validation and enrichment helpers
# ---------------------------------------------------------------------------

_REQUIRED_PROCESSOR_KEYS: tuple[str, ...] = (
    "success", "vehicle_type", "plate_category", "plate_bbox", "plate_image",
    "raw_ocr_text", "cleaned_text", "confidence", "debug_stages", "error_message",
)


def _validate_processor_output(result: dict[str, Any], vehicle_type: str) -> dict[str, Any]:
    """Defensively complete a processor's output dictionary.

    Even though every processor is contractually obliged to return all ten
    keys, a partially-implemented or buggy processor might omit one — and
    the GUI would then crash on the missing key.  We fill any absent keys
    with sentinel defaults so the GUI can always render something.
    """
    completed = dict(result) if isinstance(result, dict) else {}
    completed.setdefault("success", False)
    completed.setdefault("vehicle_type", vehicle_type)
    completed.setdefault("plate_category", "Unknown")
    completed.setdefault("plate_bbox", None)
    completed.setdefault("plate_image", None)
    completed.setdefault("raw_ocr_text", "")
    completed.setdefault("cleaned_text", "")
    completed.setdefault("confidence", 0.0)
    completed.setdefault("debug_stages", {})
    completed.setdefault("error_message", "")

    # Guard rail: the architecture forbids processors from setting these.
    if completed.get("state_code") or completed.get("state_name"):
        LOG.warning(
            "Processor for %s attempted to set state_* keys; values will be overridden.",
            vehicle_type,
        )
    return completed


def _enrich_with_state(enriched: dict[str, Any]) -> dict[str, Any]:
    """Inject ``state_code`` and ``state_name`` into an enriched result dict.

    This is the single place in the codebase where those two keys are
    written.  Processors must never set them directly.
    """
    state_code, state_name = identify_state(
        enriched.get("cleaned_text", ""),
        enriched.get("plate_category", "Standard"),
    )
    enriched["state_code"] = state_code
    enriched["state_name"] = state_name
    return enriched


# ---------------------------------------------------------------------------
# Chain-of-Responsibility engine (internal)
# ---------------------------------------------------------------------------

def _run_chain_of_responsibility(image_path: str) -> dict[str, Any]:
    """Run every installed processor and return the highest-confidence result.

    Design rationale
    ----------------
    With the vehicle-type dropdown removed from the GUI, the system must
    self-select the best processor autonomously.  We implement the
    *Chain of Responsibility* pattern: each processor in ``VEHICLE_DISPATCH``
    is invoked in turn (in an unspecified order that does not matter because
    we sort by confidence at the end).  The processor is the handler; if it
    "accepts" the image (``success=True``), its confidence score enters a
    running tournament.  The winner is the handler whose confidence is
    greatest across all successful runs.

    Fallback policy
    ---------------
    If no processor produces ``success=True`` — for instance, when the image
    quality is too poor for any route to find a plate — we return the failure
    result that has the *highest* confidence among all failed attempts.  This
    gives the GUI a best-effort ``debug_stages`` dict to display in its
    nine-phase panel, which is more useful for diagnosis than an empty dict.

    Processor isolation
    -------------------
    Every processor call is wrapped in its own try/except so that a crash
    in one processor never prevents the remaining processors from running.
    This is an extension of the architecture's Zero-Crash Policy to the
    orchestration layer.

    Logging
    -------
    Each processor's outcome is logged at DEBUG level so the terminal output
    during a grading demo stays clean, but the information is available when
    ``--log-level DEBUG`` is passed.
    """
    best_success: dict[str, Any] | None = None
    best_failure: dict[str, Any] | None = None

    for vehicle_token, process_fn in VEHICLE_DISPATCH.items():
        if process_fn is None:
            LOG.debug("Chain: skipping %r — processor not installed.", vehicle_token)
            continue

        try:
            raw_result = process_fn(image_path)
        except Exception as exc:  # noqa: BLE001
            LOG.debug(
                "Chain: processor %r raised an unhandled exception: %s",
                vehicle_token, exc,
            )
            raw_result = _build_error_envelope(
                vehicle_token, f"Unhandled processor exception: {exc}",
            )

        candidate = _validate_processor_output(raw_result, vehicle_token)
        candidate_confidence = float(candidate.get("confidence", 0.0))

        LOG.debug(
            "Chain: %r → success=%s  conf=%.3f  text=%r",
            vehicle_token,
            candidate.get("success"),
            candidate_confidence,
            candidate.get("cleaned_text", ""),
        )

        if candidate.get("success"):
            if (
                best_success is None
                or candidate_confidence > float(best_success.get("confidence", 0.0))
            ):
                best_success = candidate
        else:
            if (
                best_failure is None
                or candidate_confidence > float(best_failure.get("confidence", 0.0))
            ):
                best_failure = candidate

    if best_success is not None:
        LOG.info(
            "Chain-of-Responsibility winner: vehicle_type=%r  conf=%.3f  text=%r",
            best_success.get("vehicle_type"),
            best_success.get("confidence", 0.0),
            best_success.get("cleaned_text", ""),
        )
        return best_success

    # No processor succeeded — return the best failure for GUI debug display.
    LOG.info(
        "Chain-of-Responsibility: no processor succeeded. Returning best failure from %r.",
        best_failure.get("vehicle_type") if best_failure else "none",
    )
    return best_failure or _build_error_envelope(
        "unknown", "No installed processor could localise a plate in this image.",
    )


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def run(
    image_path: str,
    vehicle_type: str | None = None,
) -> dict[str, Any]:
    """Orchestrate a single LPR-and-SIS pass on one image.

    Parameters
    ----------
    image_path:
        Absolute or relative path to the image to process.
    vehicle_type:
        One of the ten supported taxonomy strings, or ``None``.

        * When a string is provided the original directed-dispatch
          behaviour is used — the named processor is called directly.
          This path is fully preserved for backward compatibility.

        * When ``None`` (the default after the GUI vehicle-type dropdown
          was removed) the Chain-of-Responsibility engine is activated:
          every installed processor is invoked silently and the result
          with the highest OCR confidence is returned.

    Returns
    -------
    dict
        The enriched processor output dictionary, guaranteed to contain
        every key in ``_REQUIRED_PROCESSOR_KEYS`` plus ``state_code`` and
        ``state_name``.  ``success`` is ``False`` whenever any precondition
        or processor stage fails; ``error_message`` then explains why.
    """
    # --- Input validation ---------------------------------------------------
    if not isinstance(image_path, str) or not image_path.strip():
        return _build_error_envelope(vehicle_type or "unknown", "Empty image path supplied.")
    if not os.path.isfile(image_path):
        return _build_error_envelope(
            vehicle_type or "unknown", f"File not found: {image_path}",
        )

    # --- Routing decision ---------------------------------------------------
    if vehicle_type is None:
        # Automatic mode: Chain of Responsibility across all processors.
        raw_result = _run_chain_of_responsibility(image_path)
    else:
        # Directed mode: use the explicitly requested processor.
        if vehicle_type not in VEHICLE_DISPATCH:
            return _build_error_envelope(
                vehicle_type, f"Unsupported vehicle type {vehicle_type!r}.",
            )
        process_fn = VEHICLE_DISPATCH[vehicle_type]
        if process_fn is None:
            return _build_error_envelope(
                vehicle_type, f"Processor for {vehicle_type!r} is not installed.",
            )
        try:
            raw_result = process_fn(image_path)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Processor for %s raised unexpectedly.", vehicle_type)
            return _build_error_envelope(vehicle_type, f"Processor exception: {exc}")

    enriched = _validate_processor_output(
        raw_result, raw_result.get("vehicle_type", vehicle_type or "unknown"),
    )
    return _enrich_with_state(enriched)


def _build_error_envelope(vehicle_type: str, message: str) -> dict[str, Any]:
    """Construct a fully-populated failure dictionary for upstream callers."""
    return {
        "success":        False,
        "vehicle_type":   vehicle_type,
        "plate_category": "Unknown",
        "plate_bbox":     None,
        "plate_image":    None,
        "raw_ocr_text":   "",
        "cleaned_text":   "",
        "confidence":     0.0,
        "debug_stages":   {},
        "error_message":  message,
        "state_code":     "",
        "state_name":     "Unknown",
    }


# ---------------------------------------------------------------------------
# Standalone CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main_processor.py <image_path> [vehicle_type]")
        print(f"Supported types: {', '.join(_SUPPORTED_VEHICLES)}")
        print("Omit vehicle_type to activate Chain-of-Responsibility auto-routing.")
        sys.exit(1)

    image_arg = sys.argv[1]
    vehicle_arg = sys.argv[2] if len(sys.argv) >= 3 else None

    outcome = run(image_arg, vehicle_arg)
    print("=" * 70)
    print(f"Image:             {image_arg}")
    print(f"Vehicle mode:      {'auto (chain)' if vehicle_arg is None else vehicle_arg!r}")
    print("-" * 70)
    print(f"Success:           {outcome['success']}")
    print(f"Vehicle (winner):  {outcome['vehicle_type']}")
    print(f"Plate category:    {outcome['plate_category']}")
    print(f"Raw OCR text:      {outcome['raw_ocr_text']!r}")
    print(f"Cleaned text:      {outcome['cleaned_text']!r}")
    print(f"State code:        {outcome['state_code']!r}")
    print(f"State name:        {outcome['state_name']}")
    print(f"Confidence:        {outcome['confidence']:.3f}")
    print(f"Plate bbox:        {outcome['plate_bbox']}")
    if outcome["error_message"]:
        print(f"Error:             {outcome['error_message']}")
    print("=" * 70)