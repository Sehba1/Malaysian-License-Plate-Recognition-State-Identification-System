"""
app_gui
=======

Tkinter front-end for the Malaysian Vehicle LPR & SIS group project.

The architectural rule for this file is non-negotiable: it imports *only*
``main_processor`` and the standard library / Pillow.  No processor is
addressed directly; no ``core`` module is touched.  Every interaction
with the computer-vision pipeline travels through ``main_processor.run``,
which returns the enriched 12-key dictionary defined in the architecture
document.

Layout overview (updated — vehicle-type dropdown removed)
----------------------------------------------------------
The window is a single-screen, single-column design — no tabs, no
notebooks, no modal dialogs except the native file picker:

    Row 0 — Title bar.
    Row 1 — Browse-image button + selected-file label.
    Row 2 — Run-detection button.
    Row 3 — Two side-by-side image canvases:
               left  = original frame with the detected bounding box,
               right = cropped plate ROI.
    Row 4 — Five result labels (plate number, plate type, detected vehicle,
            state, confidence).
    Row 5 — Status / error line.
    Row 6 — Collapsible "Show debug stages" expander revealing a
            scrollable 3×3 grid of the nine pipeline phase images.

Design change notes
-------------------
Vehicle-type dropdown
    Removed in favour of the Chain-of-Responsibility auto-routing
    implemented in ``main_processor.run``.  ``run()`` now accepts
    ``vehicle_type=None`` by default which activates the automatic path.
    The "Detected As" result row exposes whichever vehicle type the
    winning processor reported.

Scrollable debug panel
    The nine-phase thumbnail grid is wrapped in a ``tk.Canvas`` +
    ``ttk.Scrollbar`` scrollable-frame pattern so that all phase images
    remain accessible regardless of the application window height.
    Mouse-wheel scrolling is activated only while the cursor is inside
    the canvas, preventing wheel events from hijacking other widgets.

Rationale for using Tkinter rather than PyQt: Tkinter ships with the
standard CPython distribution, so the marker can run the GUI on any
freshly-installed lab machine without ``pip install`` steps — a hard
requirement of the assignment's "minimal-install" implicit constraint.
Pillow is the only non-standard runtime dependency.

This file deliberately contains *no* image-processing logic.  It is a
view-controller; the model is ``main_processor``.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageTk

# The architecture's single-entry-point rule: ONLY this import.
import main_processor


# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

WINDOW_TITLE: str = "LPR & SIS — CT036-3-IPPR Group Project"
WINDOW_GEOMETRY: str = "1180x880"

ORIGINAL_CANVAS_SIZE: tuple[int, int] = (540, 360)
PLATE_CANVAS_SIZE: tuple[int, int] = (540, 360)
DEBUG_THUMB_SIZE: tuple[int, int] = (240, 160)

# The nine canonical preprocessing phases plus the detection overlay,
# laid out row-major into a 3×3 + 1 grid.
DEBUG_PHASE_LAYOUT: tuple[tuple[str, str], ...] = (
    ("original",         "Phase 1 — Original"),
    ("enhanced",         "Phase 2 — Equalised + Gamma"),
    ("restored",         "Phase 3 — Bilateral Restored"),
    ("color_processed",  "Phase 4 — HSV V-Channel"),
    ("wavelet",          "Phase 5 — db4 Wavelet Detail"),
    ("compressed",       "Phase 6 — Compression Sim"),
    ("morphological",    "Phase 7 — Morph Gradient"),
    ("segmented",        "Phase 8 — Adaptive Threshold"),
    ("detection_result", "Phase 9 — Detection Overlay"),
)


# ---------------------------------------------------------------------------
# Helper: numpy frame → Tkinter-displayable PhotoImage
# ---------------------------------------------------------------------------

def _ndarray_to_pil(frame: np.ndarray | None) -> Image.Image | None:
    """Convert a numpy frame from the pipeline into a Pillow image.

    Accepts both 2-D grayscale arrays and 3-D RGB arrays of dtype uint8.
    Returns ``None`` when the input is empty or malformed.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None

    if frame.dtype != np.uint8:
        rescaled = frame.astype(np.float32)
        lo, hi = float(rescaled.min()), float(rescaled.max())
        if hi <= lo:
            rescaled = np.zeros_like(rescaled)
        else:
            rescaled = (rescaled - lo) * (255.0 / (hi - lo))
        frame = rescaled.clip(0, 255).astype(np.uint8)

    if frame.ndim == 2:
        return Image.fromarray(frame, mode="L")
    if frame.ndim == 3 and frame.shape[2] == 3:
        return Image.fromarray(frame, mode="RGB")
    return None


def _fit_into_box(pil_image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Letterbox-resize a Pillow image into ``target_size`` preserving aspect."""
    target_w, target_h = target_size
    original_w, original_h = pil_image.size
    if original_w <= 0 or original_h <= 0:
        return Image.new("RGB", target_size, color=(32, 32, 32))

    scale = min(target_w / original_w, target_h / original_h)
    new_w = max(1, int(original_w * scale))
    new_h = max(1, int(original_h * scale))
    resized = pil_image.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", target_size, color=(32, 32, 32))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    if resized.mode != "RGB":
        resized = resized.convert("RGB")
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def _draw_bbox_overlay(
    pil_image: Image.Image,
    bbox: tuple[int, int, int, int] | None,
    label: str,
) -> Image.Image:
    """Render a green bounding box and label onto a copy of ``pil_image``."""
    if bbox is None:
        return pil_image
    annotated = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    x, y, w, h = bbox
    draw.rectangle((x, y, x + w, y + h), outline=(0, 255, 0), width=4)
    if label:
        text_y = max(0, y - 18)
        draw.rectangle(
            (x, text_y, x + max(60, 9 * len(label)), text_y + 18),
            fill=(0, 0, 0),
        )
        draw.text((x + 2, text_y + 2), label, fill=(0, 255, 0))
    return annotated


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------

class LicensePlateApp:
    """Top-level Tkinter controller for the LPR & SIS demonstrator.

    The class deliberately owns no domain logic.  Its only responsibilities
    are widget construction, event wiring, and translation between the
    dictionary returned by ``main_processor.run`` and the visible widgets.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_GEOMETRY)
        self.root.minsize(1080, 760)

        # Persistent PhotoImage references — Tk garbage-collects them
        # aggressively unless something Python-side holds a reference.
        self._photo_originals: list[ImageTk.PhotoImage] = []
        self._photo_plate: ImageTk.PhotoImage | None = None
        self._debug_photos: list[ImageTk.PhotoImage] = []

        # State variables
        self.selected_path: tk.StringVar = tk.StringVar(value="(no image selected)")

        self.plate_text: tk.StringVar    = tk.StringVar(value="—")
        self.plate_type: tk.StringVar    = tk.StringVar(value="—")
        self.detected_as: tk.StringVar   = tk.StringVar(value="—")
        self.state_name: tk.StringVar    = tk.StringVar(value="—")
        self.confidence: tk.StringVar    = tk.StringVar(value="—")
        self.status_message: tk.StringVar = tk.StringVar(value="Ready.")

        self.image_path: str | None = None
        self.debug_panel_visible: bool = False

        self._build_widgets()

    # ----- Widget construction ---------------------------------------------

    def _build_widgets(self) -> None:
        """Construct the full static widget tree.

        The debug-stages area is implemented as a Canvas-based scrollable
        frame rather than a plain LabelFrame so that all nine pipeline phase
        thumbnails remain accessible when the application window is shorter
        than the full grid height.
        """
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        # Header
        ttk.Label(
            outer,
            text="Malaysian License-Plate Recognition & State-Identification System",
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        # Browse row
        browse_row = ttk.Frame(outer)
        browse_row.pack(fill=tk.X, pady=4)
        ttk.Button(browse_row, text="Browse Image…", command=self.on_browse).pack(
            side=tk.LEFT,
        )
        ttk.Label(
            browse_row, textvariable=self.selected_path, foreground="#444",
        ).pack(side=tk.LEFT, padx=10)

        # Run button row — vehicle dropdown deliberately absent; auto-routing
        # is handled by main_processor.run(vehicle_type=None).
        run_row = ttk.Frame(outer)
        run_row.pack(fill=tk.X, pady=6)
        ttk.Button(
            run_row, text="Run Detection  (Auto-Route)", command=self.on_run,
        ).pack(side=tk.LEFT)

        ttk.Separator(outer, orient="horizontal").pack(fill=tk.X, pady=8)

        # Image preview row
        preview_row = ttk.Frame(outer)
        preview_row.pack(fill=tk.X, pady=4)

        left_box = ttk.LabelFrame(
            preview_row, text="Original Image (with detection)", padding=4,
        )
        left_box.pack(side=tk.LEFT, padx=(0, 8))
        self.original_canvas = tk.Canvas(
            left_box,
            width=ORIGINAL_CANVAS_SIZE[0],
            height=ORIGINAL_CANVAS_SIZE[1],
            bg="#202020",
            highlightthickness=0,
        )
        self.original_canvas.pack()

        right_box = ttk.LabelFrame(
            preview_row, text="Detected Plate (cropped)", padding=4,
        )
        right_box.pack(side=tk.LEFT)
        self.plate_canvas = tk.Canvas(
            right_box,
            width=PLATE_CANVAS_SIZE[0],
            height=PLATE_CANVAS_SIZE[1],
            bg="#202020",
            highlightthickness=0,
        )
        self.plate_canvas.pack()

        # Result fields
        results_frame = ttk.LabelFrame(outer, text="Recognition Result", padding=10)
        results_frame.pack(fill=tk.X, pady=8)

        self._add_result_row(results_frame, "Plate Number:",  self.plate_text,  0)
        self._add_result_row(results_frame, "Plate Type:",    self.plate_type,  1)
        self._add_result_row(results_frame, "State:",         self.state_name,  3)
        self._add_result_row(results_frame, "Confidence:",    self.confidence,  4)

        # Status line
        ttk.Label(
            outer, textvariable=self.status_message, foreground="#666",
        ).pack(fill=tk.X, pady=(4, 0))

        # ------------------------------------------------------------------ #
        # Debug expander — scrollable nine-phase pipeline grid                #
        # ------------------------------------------------------------------ #
        self.debug_toggle_button = ttk.Button(
            outer, text="Show Debug Stages ▼", command=self.on_toggle_debug,
        )
        self.debug_toggle_button.pack(anchor="w", pady=(8, 4))

        # Outer label-frame — not packed yet; revealed by on_toggle_debug.
        self._debug_outer = ttk.LabelFrame(
            outer, text="Nine-phase Pipeline", padding=6,
        )

        # The scrollable interior pairs a Canvas (viewport) with a vertical
        # Scrollbar.  A ttk.Frame acts as the scroll host so the Scrollbar
        # hugs the canvas right edge cleanly.
        scroll_host = ttk.Frame(self._debug_outer)
        scroll_host.pack(fill=tk.BOTH, expand=True)

        self._debug_canvas = tk.Canvas(
            scroll_host, bg="#1a1a1a", highlightthickness=0, height=420,
        )
        self._debug_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        _debug_vscroll = ttk.Scrollbar(
            scroll_host, orient=tk.VERTICAL, command=self._debug_canvas.yview,
        )
        _debug_vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._debug_canvas.configure(yscrollcommand=_debug_vscroll.set)

        # The actual 3-column grid lives in a ttk.Frame embedded as a canvas
        # window so the grid participates in the canvas viewport.
        self._debug_inner_frame = ttk.Frame(self._debug_canvas)
        self._debug_canvas_window = self._debug_canvas.create_window(
            (0, 0), window=self._debug_inner_frame, anchor="nw",
        )

        # Synchronise scroll region and inner-frame width with canvas geometry.
        self._debug_inner_frame.bind("<Configure>", self._on_debug_frame_configure)
        self._debug_canvas.bind("<Configure>", self._on_debug_canvas_configure)

        # Mouse-wheel scrolling is active only while the cursor is inside the
        # canvas, preventing the canvas from hijacking wheel events on other
        # widgets when the debug panel is open.
        self._debug_canvas.bind("<Enter>", lambda _e: self._bind_mousewheel())
        self._debug_canvas.bind("<Leave>", lambda _e: self._unbind_mousewheel())

    def _add_result_row(
        self,
        parent: ttk.LabelFrame,
        label_text: str,
        value_var: tk.StringVar,
        row_index: int,
    ) -> None:
        """Place one (label, value) pair into the result frame at ``row_index``."""
        ttk.Label(
            parent, text=label_text, font=("Segoe UI", 10, "bold"),
        ).grid(row=row_index, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(
            parent, textvariable=value_var, font=("Consolas", 11),
        ).grid(row=row_index, column=1, sticky="w", padx=10, pady=2)

    # ----- Scrollable debug panel helpers ----------------------------------

    def _on_debug_frame_configure(self, _event: tk.Event) -> None:
        """Synchronise the canvas scroll region with the inner grid frame's size.

        Tk fires this binding each time the inner frame's geometry changes,
        i.e. immediately after ``_populate_debug_grid`` inserts or removes
        phase cells.  Without this update the scrollbar thumb would not
        reflect the new content height and the lower phase thumbnails would
        remain inaccessible.
        """
        self._debug_canvas.configure(
            scrollregion=self._debug_canvas.bbox("all"),
        )

    def _on_debug_canvas_configure(self, event: tk.Event) -> None:
        """Stretch the embedded frame to fill the full canvas width on resize.

        Without this binding the inner frame retains its natural (minimum)
        width and leaves an empty strip on the right whenever the application
        window is widened beyond the grid's intrinsic width.
        """
        self._debug_canvas.itemconfigure(
            self._debug_canvas_window, width=event.width,
        )

    def _on_mousewheel(self, event: tk.Event) -> None:
        """Translate platform mouse-wheel events into canvas vertical scroll steps.

        Platform encoding differs across operating systems:

        * **Windows / macOS** — ``event.delta`` carries a signed integer that
          is a multiple of ±120; positive values scroll upward.
        * **Linux (X11)**     — the wheel fires discrete ``Button-4`` (scroll
          up) and ``Button-5`` (scroll down) synthetic events; ``event.delta``
          is always zero.

        Both encodings are normalised here to single-unit scroll steps so the
        behaviour is consistent across all target environments.
        """
        if event.num == 4:                      # Linux — scroll up
            self._debug_canvas.yview_scroll(-1, "units")
        elif event.num == 5:                    # Linux — scroll down
            self._debug_canvas.yview_scroll(1, "units")
        elif event.delta:                       # Windows / macOS
            direction = -1 if event.delta > 0 else 1
            self._debug_canvas.yview_scroll(direction, "units")

    def _bind_mousewheel(self) -> None:
        """Activate canvas wheel scrolling while the cursor is inside the panel.

        Binds ``<MouseWheel>`` (Windows/macOS) and ``<Button-4>`` /
        ``<Button-5>`` (Linux) globally so scroll events reach the canvas
        even when a thumbnail sub-canvas has keyboard focus.
        """
        self._debug_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self._debug_canvas.bind_all("<Button-4>",   self._on_mousewheel)
        self._debug_canvas.bind_all("<Button-5>",   self._on_mousewheel)

    def _unbind_mousewheel(self) -> None:
        """Deactivate canvas wheel scrolling when the cursor leaves the panel.

        Removing the global bindings ensures that scrolling the debug grid
        does not interfere with other scrollable widgets in the application
        once the cursor moves away from the debug area.
        """
        self._debug_canvas.unbind_all("<MouseWheel>")
        self._debug_canvas.unbind_all("<Button-4>")
        self._debug_canvas.unbind_all("<Button-5>")

    # ----- Event handlers --------------------------------------------------

    def on_browse(self) -> None:
        """Open the native file picker and remember the chosen path."""
        chosen = filedialog.askopenfilename(
            title="Select a vehicle image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
                ("All files",   "*.*"),
            ],
        )
        if not chosen:
            return
        self.image_path = chosen
        self.selected_path.set(os.path.basename(chosen))
        self.status_message.set(f"Loaded: {os.path.basename(chosen)}")
        self._preview_unprocessed_image(chosen)

    def _preview_unprocessed_image(self, image_path: str) -> None:
        """Show the user's selection on the left canvas before pressing Run."""
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Cannot open image", str(exc))
            return

        fitted = _fit_into_box(pil_image, ORIGINAL_CANVAS_SIZE)
        photo = ImageTk.PhotoImage(fitted)
        self._photo_originals = [photo]
        self.original_canvas.delete("all")
        self.original_canvas.create_image(0, 0, anchor="nw", image=photo)

        self.plate_canvas.delete("all")
        self._photo_plate = None
        for var in (
            self.plate_text, self.plate_type, self.detected_as,
            self.state_name, self.confidence,
        ):
            var.set("—")

    def on_run(self) -> None:
        """Invoke the auto-routing orchestrator and dispatch to ``_render_outcome``.

        ``main_processor.run`` is called without a ``vehicle_type`` argument,
        which activates the Chain-of-Responsibility engine — every installed
        processor is tried and the highest-confidence success is returned.
        The winning processor's ``vehicle_type`` is surfaced in the
        "Detected As" result row so the operator can see which path won.

        Long-running pipeline work runs inline on the Tk main thread —
        acceptable for the assignment's single-image-per-button-press
        evaluation pattern.
        """
        if not self.image_path:
            messagebox.showinfo("No image", "Please browse for an image first.")
            return

        self.status_message.set("Auto-routing across all processors…")
        self.root.config(cursor="watch")
        self.root.update_idletasks()

        try:
            # vehicle_type omitted → Chain-of-Responsibility auto-routing
            outcome = main_processor.run(self.image_path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Pipeline error", str(exc))
            self.status_message.set("Error during processing.")
            return
        finally:
            self.root.config(cursor="")

        self._render_outcome(outcome)

    def on_toggle_debug(self) -> None:
        """Show or hide the scrollable nine-phase debug grid.

        Packing the outer LabelFrame rather than an inner widget ensures the
        Scrollbar is also hidden when the panel collapses, keeping the layout
        clean when no image has been processed yet.
        """
        if self.debug_panel_visible:
            self._debug_outer.pack_forget()
            self.debug_toggle_button.config(text="Show Debug Stages ▼")
            self.debug_panel_visible = False
        else:
            self._debug_outer.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
            self.debug_toggle_button.config(text="Hide Debug Stages ▲")
            self.debug_panel_visible = True

    # ----- Rendering -------------------------------------------------------

    def _render_outcome(self, outcome: dict[str, Any]) -> None:
        """Translate one orchestrator dictionary into visible widgets."""
        success = bool(outcome.get("success"))
        cleaned = outcome.get("cleaned_text", "") or "—"
        category = outcome.get("plate_category", "Unknown") or "—"
        state = outcome.get("state_name", "Unknown") or "—"
        code = outcome.get("state_code", "") or ""
        vehicle_winner = outcome.get("vehicle_type", "—") or "—"
        confidence_value = float(outcome.get("confidence", 0.0) or 0.0)

        self.plate_text.set(cleaned if success else "—")
        self.plate_type.set(category)
        # "Detected As" shows the winning processor's vehicle type even on
        # failure — useful for understanding which route the image ended up in.
        self.detected_as.set(vehicle_winner.title())
        state_display = f"{state} ({code})" if (success and code) else state
        self.state_name.set(state_display)
        self.confidence.set(f"{confidence_value:.2f}" if success else "—")

        if success:
            self.status_message.set(
                f"Detection succeeded via '{vehicle_winner}' processor.",
            )
        else:
            err = outcome.get("error_message") or "Detection failed."
            self.status_message.set(err)

        # Original canvas (with bbox overlay)
        debug_stages = outcome.get("debug_stages") or {}
        original = debug_stages.get("original")
        bbox = outcome.get("plate_bbox") if success else None
        original_pil = _ndarray_to_pil(original)
        if original_pil is None and self.image_path is not None:
            try:
                original_pil = Image.open(self.image_path).convert("RGB")
            except Exception:  # noqa: BLE001
                original_pil = None

        if original_pil is not None:
            annotated = _draw_bbox_overlay(
                original_pil, bbox, cleaned if success else "",
            )
            fitted = _fit_into_box(annotated, ORIGINAL_CANVAS_SIZE)
            photo = ImageTk.PhotoImage(fitted)
            self._photo_originals = [photo]
            self.original_canvas.delete("all")
            self.original_canvas.create_image(0, 0, anchor="nw", image=photo)

        # Plate crop canvas
        plate_image = outcome.get("plate_image") if success else None
        plate_pil = _ndarray_to_pil(plate_image)
        self.plate_canvas.delete("all")
        if plate_pil is not None:
            fitted_plate = _fit_into_box(plate_pil, PLATE_CANVAS_SIZE)
            self._photo_plate = ImageTk.PhotoImage(fitted_plate)
            self.plate_canvas.create_image(
                0, 0, anchor="nw", image=self._photo_plate,
            )
        else:
            self.plate_canvas.create_text(
                PLATE_CANVAS_SIZE[0] // 2,
                PLATE_CANVAS_SIZE[1] // 2,
                text="(no plate detected)",
                fill="#888",
                font=("Segoe UI", 11, "italic"),
            )

        self._populate_debug_grid(debug_stages)

    def _populate_debug_grid(self, debug_stages: dict[str, np.ndarray]) -> None:
        """Render the nine-phase debug grid into the scrollable inner frame.

        Each pipeline phase occupies one cell of a 3-column grid.  After all
        cells are built, ``update_idletasks`` forces the geometry manager to
        finalise widget sizes before the scroll region is recalculated,
        ensuring the scrollbar thumb accurately represents the full content
        height even when the panel has not yet been expanded by the user.

        Parameters
        ----------
        debug_stages : dict[str, np.ndarray]
            The ``debug_stages`` sub-dictionary returned by the active
            processor, keyed by the phase names in ``DEBUG_PHASE_LAYOUT``.
        """
        for child in self._debug_inner_frame.winfo_children():
            child.destroy()
        self._debug_photos = []

        for index, (phase_key, phase_label) in enumerate(DEBUG_PHASE_LAYOUT):
            row = index // 3
            column = index % 3

            cell = ttk.Frame(self._debug_inner_frame, padding=4, relief="ridge")
            cell.grid(row=row, column=column, padx=4, pady=4, sticky="nsew")

            ttk.Label(
                cell, text=phase_label, font=("Segoe UI", 9, "bold"),
            ).pack(anchor="w")

            phase_image = debug_stages.get(phase_key)
            phase_pil = _ndarray_to_pil(phase_image)
            if phase_pil is None:
                ttk.Label(cell, text="(unavailable)", foreground="#888").pack(pady=20)
                continue

            fitted = _fit_into_box(phase_pil, DEBUG_THUMB_SIZE)
            photo = ImageTk.PhotoImage(fitted)
            self._debug_photos.append(photo)
            thumb_canvas = tk.Canvas(
                cell,
                width=DEBUG_THUMB_SIZE[0],
                height=DEBUG_THUMB_SIZE[1],
                bg="#202020",
                highlightthickness=0,
            )
            thumb_canvas.pack()
            thumb_canvas.create_image(0, 0, anchor="nw", image=photo)

        # Flush the geometry manager then refresh the scroll region so the
        # scrollbar thumb immediately reflects the freshly populated grid height.
        self._debug_inner_frame.update_idletasks()
        self._debug_canvas.configure(
            scrollregion=self._debug_canvas.bbox("all"),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Construct the root window, the application controller and run the loop."""
    root = tk.Tk()
    LicensePlateApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()