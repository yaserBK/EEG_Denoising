# eeg_denoising

A Python library for loading, annotating, and preprocessing EEG data from labelled EDF corpora, with a focus on artifact characterisation and noise removal while preserving seizure activity.

---

## Project Structure

```
EEG_Denoising/                         
├── data/                           # All data (not committed)
│   ├── annotated/
│   │   └── <dataset>/
│   │       └── <montage>/
│   │           ├── edf_format/     # Processed EDF with embedded annotations
│   │           └── fif_format/     # Processed FIF with embedded annotations
│   ├── json/
│   │   └── <dataset>/
│   │       └── <montage>/          # master.json and per-patient <id>.json files
│   └── raw/                        # Raw EDF and CSV files
├── eeg_denoising/
│   ├── eeg_denoising/
│   │   ├── __init__.py
│   │   ├── config.py               # Path definitions and other misc config
│   │   ├── data.py                 # JSON loading utilities
│   │   ├── eeg_utils.py            # EDF loading, segment extraction, annotations
│   │   ├── montage.py              # Montage lazy-loading and application
│   │   ├── montage_util.py         # NEDC montage file parser
│   │   └── utils.py                # General utilities
│   └── build.sh
├── notebooks/
│   ├── archive/
│   │   └── test_visualization.ipynb
│   └── UsingTheUtils.ipynb    
├── other_data/
├── scripts/
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yourname/EEG_Denoising.git
cd EEG_Denoising
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or using the build script:

```bash
sh eeg_denoising/build.sh
```

### Dependencies

```bash
pip install mne numpy matplotlib mne-qt-browser pyqt5 ipympl
```

---

## Configuration

All paths are defined in `eeg_denoising/config.py` and resolve relative to the project root. Update these to point to wherever your data lives — nothing else in the library needs to change.

```python
from pathlib import Path

PATH_TO_ROOT                          = Path(__file__).parent.parent.parent
PATH_TO_DATA                          = PATH_TO_ROOT / 'data'
PATH_TO_RAW_DATA                      = PATH_TO_DATA / 'raw'
PATH_TO_ANNOTATED_DATA                = PATH_TO_DATA / 'annotated_data'
PATH_TO_JSON_DATA                     = PATH_TO_DATA / 'json'

# Raw EDF data
PATH_TO_EDF_DATASET                   = PATH_TO_RAW_DATA / '<dataset>/<montage>'
PATH_TO_EDF_FILES                     = PATH_TO_EDF_DATASET / 'edf'
PATH_TO_ANNOTATION_FILES              = PATH_TO_EDF_DATASET / 'annotations'

# Processed data
PATH_TO_ANNOTATED_EDF                 = PATH_TO_ANNOTATED_DATA / 'edf_format'
PATH_TO_ANNOTATED_FIF                 = PATH_TO_ANNOTATED_DATA / 'fif_format'

# Metadata
PATH_TO_JSON                          = PATH_TO_JSON_DATA / '<dataset>/<montage>'
PATH_TO_CUSTOM_MONTAGE                = PATH_TO_EDF_DATASET / 'montage.txt'
```

> The `data/` directory is excluded from version control. Place your EDF files here following the structure above.

---

## Modules

### `config.py`
Centralised path definitions for all data directories. Import paths directly from here rather than hardcoding them anywhere in your code.

---

### `data.py`
Utilities for loading patient and recording metadata from JSON.

The expected JSON structure is:
- `master.json` — dataset-level summary and index of all patients
- `<patient_id>.json` — full recording and segment metadata for a single patient

```python
from eeg_denoising.data import load_master, load_patient, iter_patients

# load dataset summary
master = load_master()

# load a single patient on demand
patient = load_patient("patient_001")

# iterate all patients one at a time
for patient_id, patient in iter_patients(master):
    ...
```

---

### `montage.py`
Lazy-loads montage definitions from disk, caching them on first access. The montage file is only read once — subsequent accesses return the cached result.

```python
from eeg_denoising import montage

channels = montage.TCP_AR   # parsed and cached on first access
```

Montage names map to `.txt` files defined in `config.py`. To add a new montage, add an entry to `_MONTAGE_FILES` in `montage.py`.

---

### `montage_util.py`
Parser for NEDC-style montage `.txt` files. Each file defines a set of bipolar channel derivations (anode `--` cathode pairs).

```
montage = 0, FP1-F7: EEG FP1-REF -- EEG F7-REF
montage = 1, F7-T3:  EEG F7-REF  -- EEG T3-REF
...
```

Returns a list of dicts with keys `index`, `name`, `anode`, `cathode`.

---

### `eeg_utils.py`
Core utilities for the data pipeline. All functions accept paths and montage channels as arguments — nothing is hardcoded.

#### Loading

```python
from eeg_denoising.eeg_utils import load_edf, iter_edfs

# load a single recording — returns (data, ch_names, sfreq)
# data shape: (n_channels, n_samples) in µV
data, ch_names, sfreq = load_edf(recording, edf_dir, montage_channels)

# iterate all recordings for a patient one at a time
for recording_id, data, ch_names, sfreq, recording in iter_edfs(patient, edf_dir, montage_channels):
    ...
```

`load_edf` automatically applies the bipolar montage and returns a raw NumPy array. The MNE `Raw` object is discarded immediately to free memory.

#### Segment Extraction

```python
from eeg_denoising.eeg_utils import get_segments, extract_signal, extract_all_signals

# get segment metadata as a list of dicts
segments = get_segments(recording)
segments = get_segments(recording, labels="musc")
segments = get_segments(recording, labels=["musc", "eyem"])
segments = get_segments(recording, channels="FP1-F7")
segments = get_segments(recording, labels="musc", channels="FP1-F7")
segments = get_segments(recording, include_seiz=True)   # includes seizure labels

# each segment dict contains:
# { start, stop, duration, channel, label, confidence }

# extract the raw signal array for a single segment
signal = extract_signal(data, ch_names, seg, sfreq)   # shape: (n_samples,)

# extract all signals and labels at once (for ML)
X, y = extract_all_signals(data, ch_names, sfreq, recording, include_seiz=True)
```

#### MNE Annotations (for visualisation)

```python
from eeg_denoising.eeg_utils import apply_annotations
import mne

raw = mne.io.read_raw_edf(fpath, preload=True)
apply_annotations(raw, recording, include_seiz=True)
raw.plot(duration=10, n_channels=22, scalings="auto", block=True)
```

Annotation descriptions are encoded as `label:channel` (e.g. `eyem:FP1-F7`) so they can be filtered directly in the MNE browser.

---

## Data Format

### JSON Structure

**`master.json`** — top-level index:
```json
{
  "summary": { "total_edf_files": 231, "total_duration": 154494.0 },
  "Patients": {
    "patient_001": { "..." }
  }
}
```

**`<patient_id>.json`** — per-patient detail:
```json
{
  "id": "patient_001",
  "has_seiz": false,
  "recordings": {
    "patient_001_s001_t000": {
      "edf": "path/to/file.edf",
      "has_seiz": false,
      "duration": 1442.0,
      "labels": {
        "eyem": {
          "count": 24,
          "segments": [
            { "start": 22.97, "stop": 30.07, "channel": "FP1-F7", "confidence": 1.0 }
          ]
        }
      },
      "seiz_labels": {}
    }
  }
}
```

### Label Types

The library supports any string labels defined in the JSON. Common artifact and seizure labels include:

| Label | Description |
|-------|-------------|
| `elec` | Electrode artefact |
| `elpp` | Electrode pop |
| `eyem` | Eye movement |
| `eyem_musc` | Eye movement + muscle |
| `eyem_elec` | Eye movement + electrode |
| `musc` | Muscle artefact |
| `musc_elec` | Muscle + electrode |
| `chew` | Chewing |
| `shiv` | Shivering |
| `bckg` | Background (non-seizure) |
| `tcsz` | Tonic-clonic seizure |

### Montage

The default montage is the bipolar Temporal Central Parasagittal (TCP) Averaged Reference with 22 channels:

```
FP1-F7, F7-T3, T3-T5, T5-O1,
FP2-F8, F8-T4, T4-T6, T6-O2,
A1-T3,  T3-C3, C3-CZ, CZ-C4, C4-T4, T4-A2,
FP1-F3, F3-C3, C3-P3, P3-O1,
FP2-F4, F4-C4, C4-P4, P4-O2
```

Additional montages can be added by placing a `.txt` file in the appropriate directory and registering it in `montage.py`.

---

## Memory Management

EDF files are large — load them one at a time using the generator pattern:

| Object | Size | Strategy |
|--------|------|----------|
| `master.json` | ~14 MB | Load once, keep in memory |
| `<id>.json` | small | Load per patient, discard after |
| Montage definition | ~few KB | Lazy load, cache permanently |
| EDF file | ~150 MB | Load one at a time, `del` after use |

```python
# never do this
all_data = [load_edf(r, edf_dir, montage_channels) for r in recordings]

# do this instead
for recording_id, data, ch_names, sfreq, recording in iter_edfs(patient, edf_dir, montage_channels):
    # process one file at a time
    del data
```

---

## Visualisation

**In Jupyter / PyCharm / VSCode notebooks:**
```python
%matplotlib inline
```


**Interactive scrollable browser with annotation toggling:**
```python
mne.viz.set_browser_backend("pyqtgraph")
raw.plot(duration=10, n_channels=22, scalings="auto", block=True)
```

In the MNE Qt browser, press `A` to toggle annotation mode. Individual label types can be shown or hidden via the annotation panel on the right.

---

## Goal

The aim of this library is to support the development of an EEG denoising pipeline that removes artifacts (`musc`, `eyem`, `elec` etc.) without impacting seizure activity. The labelled segment data provides ground truth for both artifact characterisation and post-denoising evaluation.