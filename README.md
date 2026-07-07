```markdown
# Malaysian Vehicle LPR & SIS — Definitive Technical Architecture
**Stack:** Python · OpenCV · PaddleOCR · NumPy · PyWavelets · Tkinter · Pillow (PIL) · `re` (regex) · `logging` · etc.


# 0. How to Execute

### Standard Startup
Ensure you have **Python 3.13.4** installed on your system. 

1. Open your terminal in the project root directory and install the dependencies:
   ```bash
   pip install -r requirements.txt

2. Launch the graphical interface:

python app_gui.py

Troubleshooting: How to Recreate the Virtual Environment
If the application fails to run or dependencies conflict (e.g., OpenCV or PaddleOCR throwing DLL/import errors), your virtual environment might be corrupted. Follow these steps to completely rebuild it.

For Windows:
# 1. Deactivate the current environment (if it is currently active)
deactivate

# 2. Delete the old virtual environment folder (assuming it's named 'venv')
rmdir /s /q venv

# 3. Create a fresh virtual environment using Python 3.13.4
python -m venv venv

# 4. Activate the new environment
venv\Scripts\activate

# 5. Upgrade pip to avoid installation issues
python -m pip install --upgrade pip

# 6. Reinstall all required packages
pip install -r requirements.txt

# 7. Run the application
python app_gui.py



For macOS / Linux:
# 1. Deactivate the current environment (if it is currently active)
deactivate

# 2. Delete the old virtual environment folder
rm -rf venv

# 3. Create a fresh virtual environment using Python 3.13.4
python3.13 -m venv venv

# 4. Activate the new environment
source venv/bin/activate

# 5. Upgrade pip 
pip install --upgrade pip

# 6. Reinstall all required packages
pip install -r requirements.txt

# 7. Run the application
python3 app_gui.py


---

## 1. Directory Structure

```
LPR_SIS_System/
│
├── main_processor.py          ← Orchestrator + STATE_MAP (Routing Core)
├── app_gui.py                 ← Minimal Tkinter GUI, Starting Point of the program.
├── generate_debug_grids.py    ← Batch testing utility: generates phase grids for the report
│
├── core/                      ← Shared Infrastructure
│   ├── image_pipeline.py      ← Base preprocessing
│   └── ocr_engine.py          ← PaddleOCR initialization, Regex validation, and text sanitization
│
├── processors/                ← Isolated Business Logic (Individual Contributions)
│   │
│   │  ── Manreen ─────────────────────────────────────────
│   ├── car_processor.py       ← Standard Plate + TAXI + DIPLOMATIC
│   ├── bus_processor.py       ← Standard Plate
│   │
│   │  ── Sehba ───────────────────────────────────────────
│   ├── pickup_processor.py    ← Standard plate
│   ├── truck_processor.py     ← Standard Plate
│   │
│   │  ── Egor ────────────────────────────────────────────
│   ├── van_processor.py       ← Standard Plate
│   ├── suv_processor.py       ← Standard Plate
│   │
│   │  ── Muqri ───────────────────────────────────────────
│   ├── campervan_processor.py ← Standard Plate
│   ├── jeep_processor.py      ← Standard Plate + MILITARY (Z-prefix)
│   │
│   │  ── Rudra ───────────────────────────────────────────
│   ├── motorcycle_processor.py← TWO-ROW square plate geometry
│   └── minibus_processor.py   ← Standard plates
│
├── debug_output/              ← Auto-generated visual artifacts
│   ├── cars/        ├── trucks/      ├── campervans/   ├── pickups/
│   ├── buses/       ├── vans/        ├── jeeps/        └── minibuses/
│   └── motorcycles/ └── suvs/
│
└── test_images/               ← Datasets separated by vehicle taxonomy
    ├── cars/        ├── trucks/      ├── campervans/   ├── pickups/
    ├── buses/       ├── vans/        ├── jeeps/        └── minibuses/
    └── motorcycles/ └── suvs/
```


---

## 2. Unified Return Interface & SIS Strategy

### 2.1 The Processor Data Contract (Worker Output)
Every `process(image_path)` in the 10 vehicle processors **must** return exactly this dictionary shape, on success or failure. The processors are strictly responsible for localization and OCR, nothing else.

| Key | Type | Meaning |
|---|---|---|
| `success` | `bool` | `True` only if a plate was localized AND OCR yielded ≥3 alnum chars |
| `vehicle_type` | `str` | `"car"`, `"bus"`, `"motorcycle"`, … (hardcoded inside each module) |
| `plate_category` | `str` | `"Standard"` / `"Taxi"` / `"Military"` / `"Diplomatic"` / `"Special Series"` / `"Two-Row"` / `"Unknown"` |
| `plate_bbox` | `tuple\|None` | `(x, y, w, h)` in **original** (un-resized) image coordinates |
| `plate_image` | `ndarray\|None` | Cropped, deskewed plate (BGR), used for GUI display |
| `raw_ocr_text` | `str` | Untouched OCR output (best-of multi-phase) |
| `cleaned_text` | `str` | Uppercase, alnum-only, position-aware confusion-fixes applied |
| `confidence` | `float` | Composite 0.0–1.0 score |
| `debug_stages` | `dict` | Phase-keyed dict of intermediate images for GUI inspection |
| `error_message` | `str` | `""` on success; otherwise human-readable cause |

### 2.2 The Enriched Data Contract (Orchestrator Output)
Once the `main_processor.py` receives the dictionary from a vehicle processor, it acts as the orchestrator. It performs the state lookup based on the `cleaned_text` and `plate_category`, and injects two new keys before sending the final dictionary to the GUI (`app_gui.py`):

| Injected Key | Type | Owner | Meaning |
|---|---|---|---|
| `state_code` | `str` | **main_processor** | E.g. `"P"`, `"KV"` |
| `state_name` | `str` | **main_processor** | E.g. `"Penang"` |

### 2.3 Why state identification lives only in `main_processor.py`
- **Single Source of Truth:** Prevents maintaining and duplicating lookup dictionaries across 10 different vehicle processors.
- **Strict Separation of Concerns:** Vehicle processors stay completely **state-agnostic**. Their only job is computer vision (localization) and text extraction (OCR).
- **Clean Overrides:** Special categories (Military / Diplomatic / Taxi / Special Series) override prefix lookup cleanly via a single centralized function.


---

## 3. GUI Flow & Integration

### 3.1 Files
- **`app_gui.py`** — single GUI file, **Tkinter** 
- The GUI imports **only `main_processor`** — never any vehicle processor directly

### 3.2 GUI Layout (minimalist, single window)
```
┌────────────────────────────────────────────────────────┐
│  LPR & SIS — CT036-3-IPPR Group Project                │
├────────────────────────────────────────────────────────┤
│  [ Browse Image... ]      selected: car03.jpg          │
│                                                        │
│  Vehicle Type:  ( Car ▼ )                              │
│   Car · Bus · Motorcycle · Truck · Van ·               │
│   SUV · Campervan · Jeep · Pickup · Minibus            │
│                                                        │
│  [ Run Detection ]                                     │
├────────────────────────────────────────────────────────┤
│  ┌───────────────────┐    ┌───────────────────┐        │
│  │  Original Image   │    │  Detected Plate   │        │
│  │  (with bbox)      │    │  (cropped)        │        │
│  └───────────────────┘    └───────────────────┘        │
│                                                        │
│  Plate Number :  PJC 1234                              │
│  Plate Type   :  Standard                              │
│  State        :  Penang                                │
│  Confidence   :  0.87                                  │
│                                                        │
│                                                        │
└────────────────────────────────────────────────────────┘
```
