import numpy as np
import pandas as pd
from scipy.signal import welch
from bokeh.plotting import figure, output_file
from bokeh.models import (HoverTool, ColorBar, LinearColorMapper,
                          BasicTicker, ColumnDataSource)
from bokeh.io import save
from bokeh.transform import transform
from bokeh.palettes import RdYlGn11
from tqdm import tqdm
from eeg_denoising import config, data, montage, eeg_utils
from eeg_denoising.eeg_utils import get_segments
from collections import defaultdict
import logging
import argparse

# ── argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
args, _ = parser.parse_known_args()

log_level = logging.DEBUG if args.debug else logging.ERROR
logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

import mne
mne.set_log_level("DEBUG" if args.debug else "ERROR")

# ── setup ─────────────────────────────────────────────────────────────────────

SEIZURE_LABELS = {"tcsz", "fnsz", "cpsz", "gnsz", "absz", "tnsz", "spsz", "bckg"}

master     = data.load_master()
n_patients = len(master["Patients"])

plots_dir    = config.PATH_TO_ROOT / "plots" / "spectral_overlap"
analysis_dir = config.PATH_TO_DATA / "analysis" / "spectral_overlap"
plots_dir.mkdir(parents=True, exist_ok=True)
analysis_dir.mkdir(parents=True, exist_ok=True)

# ── accumulate PSD per label ──────────────────────────────────────────────────

label_psds       = defaultdict(list)
label_seg_counts = defaultdict(int)
sfreq            = None

patients     = list(master["Patients"].keys())
patient_pbar = tqdm(patients, desc="Loading patients", unit="patient")

for patient_id in patient_pbar:
    patient = data.load_patient(patient_id)
    patient_pbar.set_postfix(patient=patient_id)

    recordings = {k: v for k, v in patient["recordings"].items() if v["edf"] is not None}
    rec_pbar   = tqdm(recordings.items(), desc="  Recordings", unit="rec", leave=False)

    for recording_id, recording in rec_pbar:
        rec_pbar.set_postfix(recording=recording_id)
        segments = get_segments(recording, include_seiz=True)
        if not segments:
            continue

        try:
            edf_data, ch_names, sfreq = eeg_utils.load_edf(
                recording, config.PATH_TO_TUAR_EDF_FILES, montage.TUAR_TCP_AR
            )
        except Exception as e:
            logging.error(f"Failed to load {recording_id}: {e}")
            continue

        for seg in segments:
            signal = eeg_utils.extract_signal(edf_data, ch_names, seg, sfreq)

            if len(signal) < int(sfreq * 2):
                continue

            freqs, psd = welch(signal, fs=sfreq, nperseg=int(sfreq * 2))
            mask       = freqs <= 70
            psd_norm   = psd[mask] / psd[mask].sum()

            label_psds[seg["label"]].append(psd_norm)
            label_seg_counts[seg["label"]] += 1

        del edf_data

# ── average PSD per label ─────────────────────────────────────────────────────

label_avg_psd = {}

for label, psds in label_psds.items():
    min_len = min(len(p) for p in psds)
    stacked = np.array([p[:min_len] for p in psds])
    label_avg_psd[label] = np.mean(stacked, axis=0)

freqs_plot = np.linspace(0, 70, min_len)

# ── separate labels into artifact vs seizure+bckg ─────────────────────────────

all_labels      = sorted(label_avg_psd.keys())
seiz_labels     = sorted([l for l in all_labels if l in SEIZURE_LABELS])
artifact_labels = sorted([l for l in all_labels if l not in SEIZURE_LABELS])

if not seiz_labels:
    tqdm.write("No seizure labels found in dataset — exiting.")
    exit()

# ── compute spectral overlap ──────────────────────────────────────────────────

def spectral_overlap(psd_a, psd_b):
    """Bhattacharyya coefficient. 1.0 = identical, 0.0 = no overlap."""
    min_len = min(len(psd_a), len(psd_b))
    return float(np.sum(np.sqrt(psd_a[:min_len] * psd_b[:min_len])))


# rows = artifact labels, cols = seizure+bckg labels
overlap_matrix = np.zeros((len(artifact_labels), len(seiz_labels)))

for i, art_label in enumerate(artifact_labels):
    for j, sz_label in enumerate(seiz_labels):
        overlap_matrix[i, j] = spectral_overlap(
            label_avg_psd[art_label],
            label_avg_psd[sz_label]
        )

# ── save to CSV ───────────────────────────────────────────────────────────────

df_overlap = pd.DataFrame(overlap_matrix, index=artifact_labels, columns=seiz_labels)
df_overlap.index.name = "artifact_label"
df_overlap.to_csv(analysis_dir / "artifact_vs_seizure_overlap.csv")

df_psds = pd.DataFrame(
    {label: label_avg_psd[label][:min_len] for label in all_labels},
    index=freqs_plot
)
df_psds.index.name = "frequency_hz"
df_psds.to_csv(analysis_dir / "average_psd_per_label.csv")

# ── plot heatmap ──────────────────────────────────────────────────────────────

x_coords, y_coords, values, art_tips, sz_tips = [], [], [], [], []

for i, art_label in enumerate(artifact_labels):
    for j, sz_label in enumerate(seiz_labels):
        x_coords.append(sz_label)
        y_coords.append(art_label)
        values.append(overlap_matrix[i, j])
        art_tips.append(art_label)
        sz_tips.append(sz_label)

source = ColumnDataSource(dict(
    x=x_coords,
    y=y_coords,
    values=values,
    artifact=art_tips,
    seizure=sz_tips,
))

mapper = LinearColorMapper(
    palette=RdYlGn11,
    low=0,
    high=1,
)

output_file(
    filename=str(plots_dir / f"artifact_vs_seizure_overlap_{n_patients}patients.html"),
    title="Artifact vs Seizure Spectral Overlap"
)

p = figure(
    title=f"Artifact vs Seizure/Background Spectral Overlap — Bhattacharyya Coefficient "
          f"({n_patients} patients)",
    x_range=seiz_labels,
    y_range=list(reversed(artifact_labels)),
    width=600,
    height=max(400, len(artifact_labels) * 40),
    tools="pan,box_zoom,wheel_zoom,reset,save",
)

p.rect(
    x="x", y="y",
    width=1, height=1,
    source=source,
    fill_color=transform("values", mapper),
    line_color=None,
)

p.add_tools(HoverTool(tooltips=[
    ("artifact",  "@artifact"),
    ("seizure",   "@seizure"),
    ("overlap",   "@values{0.3f}"),
]))

color_bar = ColorBar(
    color_mapper=mapper,
    ticker=BasicTicker(desired_num_ticks=10),
    label_standoff=8,
    width=12,
    location=(0, 0),
    title="Overlap (0–1)",
)

p.add_layout(color_bar, "right")
p.xaxis.axis_label = "Seizure / Background Label"
p.yaxis.axis_label = "Artifact Label"
p.xaxis.major_label_orientation = 1.0
p.axis.major_label_text_font_size = "11px"

save(p)

# ── print ranked risk summary ─────────────────────────────────────────────────

tqdm.write("\n── Artifact overlap with seizure/background labels ──────────────")

for j, sz_label in enumerate(seiz_labels):
    tqdm.write(f"\n  vs {sz_label} ({label_seg_counts.get(sz_label, 0)} segments):")
    ranked = sorted(
        [(artifact_labels[i], overlap_matrix[i, j]) for i in range(len(artifact_labels))],
        key=lambda x: x[1], reverse=True
    )
    for art_label, score in ranked:
        risk = "HIGH" if score > 0.8 else "MED" if score > 0.6 else "LOW"
        n    = label_seg_counts.get(art_label, 0)
        tqdm.write(f"    {art_label:15} overlap: {score:.3f}  [{risk}]  ({n} segments)")

tqdm.write(f"\nplot saved to:  {plots_dir}")
tqdm.write(f"CSV saved to:   {analysis_dir}")