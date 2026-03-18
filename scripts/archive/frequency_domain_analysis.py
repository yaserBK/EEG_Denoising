import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from bokeh.plotting import figure, output_file
from bokeh.layouts import column
from bokeh.models import HoverTool, Div
from bokeh.io import save
from eeg_denoising import config, data, montage, eeg_utils
from eeg_denoising.eeg_utils import get_segments
from collections import defaultdict


# ── signal processing functions ───────────────────────────────────────────────

def compute_magnitude(signal, sfreq):
    magnitude = np.abs(np.fft.rfft(signal))
    freqs     = np.fft.rfftfreq(len(signal), 1 / sfreq)
    return freqs, magnitude


def compute_phase(signal, sfreq):
    phase = np.angle(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1 / sfreq)
    return freqs, phase


def compute_frequency_response(signal, sfreq):
    fft_vals = np.fft.rfft(signal)
    response = 20 * np.log10(np.abs(fft_vals) + 1e-10)
    return response


def resample_to(arr, length):
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, length)
    return interp1d(x_old, arr)(x_new)


# ── plotting functions ────────────────────────────────────────────────────────

def plot_magnitude(avg_magnitude, label, output_dir, approach_name, approach_description, n_patients, label_segment_counts):
    n_segs = sum(label_segment_counts[label].values())
    output_file(
        filename=str(output_dir / f"magnitude_avg_{label}_{n_segs}segments.html"),
        title=f"[{approach_name}] Average Magnitude — {label}"
    )
    plots = [Div(text=f"<h2>Average Magnitude — {label}</h2>"
                      f"<p><b>Approach:</b> {approach_name} — {approach_description}</p>"
                      f"<p>{n_segs} segments averaged across {n_patients} patients</p>")]

    for channel, mag in avg_magnitude.items():
        n_ch_segs = label_segment_counts[label][channel]
        ch_freqs  = np.linspace(0, sfreq / 2, len(mag))
        ch_mask   = ch_freqs <= 70
        p = figure(
            title=f"{channel}  ({n_ch_segs} segments)",
            width=900, height=150,
            x_range=(0, 70), y_range=(0, 1e4),
            tools="pan,box_zoom,wheel_zoom,reset,save",
        )
        p.add_tools(HoverTool(tooltips=[("freq", "@x{0.1f} Hz"), ("magnitude", "@y{0.0f}")]))
        p.line(ch_freqs[ch_mask], mag[ch_mask], line_width=0.8, color="steelblue")
        p.xaxis.axis_label = "Frequency (Hz)" if channel == list(avg_magnitude.keys())[-1] else ""
        p.yaxis.axis_label = "Magnitude"
        p.title.text_font_size = "10px"
        plots.append(p)
    save(column(*plots))


def plot_phase(avg_phase, label, output_dir, approach_name, approach_description, n_patients, label_segment_counts):
    n_segs = sum(label_segment_counts[label].values())
    output_file(
        filename=str(output_dir / f"phase_avg_{label}_{n_segs}segments.html"),
        title=f"[{approach_name}] Average Phase — {label}"
    )
    plots = [Div(text=f"<h2>Average Phase — {label}</h2>"
                      f"<p><b>Approach:</b> {approach_name} — {approach_description}</p>"
                      f"<p>{n_segs} segments averaged across {n_patients} patients</p>")]

    for channel, ph in avg_phase.items():
        n_ch_segs = label_segment_counts[label][channel]
        ch_freqs  = np.linspace(0, sfreq / 2, len(ph))
        ch_mask   = ch_freqs <= 70
        p = figure(
            title=f"{channel}  ({n_ch_segs} segments)",
            width=900, height=150,
            x_range=(0, 70), y_range=(-np.pi, np.pi),
            tools="pan,box_zoom,wheel_zoom,reset,save",
        )
        p.add_tools(HoverTool(tooltips=[("freq", "@x{0.1f} Hz"), ("phase", "@y{0.3f} rad")]))
        p.line(ch_freqs[ch_mask], ph[ch_mask], line_width=0.8, color="tomato")
        p.xaxis.axis_label = "Frequency (Hz)" if channel == list(avg_phase.keys())[-1] else ""
        p.yaxis.axis_label = "Phase (rad)"
        p.title.text_font_size = "10px"
        plots.append(p)
    save(column(*plots))


def plot_frequency_response(avg_response, label, output_dir, approach_name, approach_description, n_patients, label_segment_counts):
    n_segs = sum(label_segment_counts[label].values())
    output_file(
        filename=str(output_dir / f"freq_response_avg_{label}_{n_segs}segments.html"),
        title=f"[{approach_name}] Average Frequency Response — {label}"
    )
    plots = [Div(text=f"<h2>Average Frequency Response (dB) — {label}</h2>"
                      f"<p><b>Approach:</b> {approach_name} — {approach_description}</p>"
                      f"<p>{n_segs} segments averaged across {n_patients} patients | "
                      f"Normalised to 0 dB peak</p>")]

    for channel, resp in avg_response.items():
        n_ch_segs = label_segment_counts[label][channel]
        ch_freqs  = np.linspace(0, sfreq / 2, len(resp))
        ch_mask   = ch_freqs <= 70
        p = figure(
            title=f"{channel}  ({n_ch_segs} segments)",
            width=900, height=150,
            x_range=(0, 70), y_range=(-60, 0),
            tools="pan,box_zoom,wheel_zoom,reset,save",
        )
        p.add_tools(HoverTool(tooltips=[("freq", "@x{0.1f} Hz"), ("response", "@y{0.1f} dB")]))
        p.line(ch_freqs[ch_mask], resp[ch_mask], line_width=0.8, color="mediumseagreen")
        p.xaxis.axis_label = "Frequency (Hz)" if channel == list(avg_response.keys())[-1] else ""
        p.yaxis.axis_label = "Response (dB)"
        p.title.text_font_size = "10px"
        plots.append(p)
    save(column(*plots))

def save_csv(avg_data, label, metric_name, output_dir, approach_name, label_segment_counts):
    n_segs = sum(label_segment_counts[label].values())
    rows   = {}

    for channel, values in avg_data.items():
        ch_freqs          = np.linspace(0, sfreq / 2, len(values))
        ch_mask           = ch_freqs <= 70
        rows[channel]     = pd.Series(values[ch_mask], index=ch_freqs[ch_mask])

    # align all channels to a common frequency index via outer join
    df = pd.DataFrame(rows)
    df.index.name = "frequency_hz"

    fname = output_dir / f"{metric_name}_avg_{label}_{n_segs}segments_{approach_name}.csv"
    df.to_csv(fname)


# ── setup ─────────────────────────────────────────────────────────────────────

FIXED_LEN  = 1024
master     = data.load_master()
n_patients = len(master["Patients"])

plots_dir       = config.PATH_TO_ROOT / "plots"
trim_dir        = plots_dir / "01_trim_to_minimum"
pad_dir         = plots_dir / "02_zero_pad_to_maximum"
interpolate_dir = plots_dir / "03_interpolate_to_fixed_length"

analysis_dir        = config.PATH_TO_DATA / "analysis"
trim_csv_dir        = analysis_dir / "01_trim_to_minimum"
pad_csv_dir         = analysis_dir / "02_zero_pad_to_maximum"
interpolate_csv_dir = analysis_dir / "03_interpolate_to_fixed_length"

for d in [trim_dir, pad_dir, interpolate_dir,
          trim_csv_dir, pad_csv_dir, interpolate_csv_dir]:
    d.mkdir(parents=True, exist_ok=True)

# ── data accumulation ─────────────────────────────────────────────────────────

label_segments       = defaultdict(lambda: defaultdict(list))
label_segment_counts = defaultdict(lambda: defaultdict(int))

for patient_id, patient in data.iter_patients(master):
    for recording_id, recording in patient["recordings"].items():
        if recording["edf"] is None:
            continue
        segments = get_segments(recording, include_seiz=True)
        if not segments:
            continue

        edf_data, ch_names, sfreq = eeg_utils.load_edf(
            recording, config.PATH_TO_TUAR_EDF_FILES, montage.TUAR_TCP_AR
        )

        for seg in segments:
            label   = seg["label"]
            channel = seg["channel"]
            signal  = eeg_utils.extract_signal(edf_data, ch_names, seg, sfreq)

            if len(signal) < int(sfreq * 2):
                continue

            _, magnitude = compute_magnitude(signal, sfreq)
            _, phase     = compute_phase(signal, sfreq)
            response     = compute_frequency_response(signal, sfreq)

            label_segments[label][channel].append({
                "magnitude": magnitude / magnitude.max(),
                "phase":     phase,
                "response":  response - response.max(),
            })
            label_segment_counts[label][channel] += 1

        del edf_data

# ── averaging, plotting, and csv export ───────────────────────────────────────

for label, channel_data in label_segments.items():
    if not channel_data:
        continue

    mag_trim,    phase_trim,    freq_trim    = {}, {}, {}
    mag_pad,     phase_pad,     freq_pad     = {}, {}, {}
    mag_interp,  phase_interp,  freq_interp  = {}, {}, {}

    for channel, segs in channel_data.items():
        mags      = [s["magnitude"] for s in segs]
        phases    = [s["phase"]     for s in segs]
        responses = [s["response"]  for s in segs]

        min_len = min(len(s) for s in mags)
        max_len = max(len(s) for s in mags)

        mag_trim[channel]   = np.mean(np.array([s[:min_len] for s in mags]),      axis=0) * 1e4
        phase_trim[channel] = np.mean(np.array([s[:min_len] for s in phases]),    axis=0)
        freq_trim[channel]  = np.mean(np.array([s[:min_len] for s in responses]), axis=0)

        mag_pad[channel]   = np.mean(np.array([np.pad(s, (0, max_len - len(s))) for s in mags]),      axis=0) * 1e4
        phase_pad[channel] = np.mean(np.array([np.pad(s, (0, max_len - len(s))) for s in phases]),    axis=0)
        freq_pad[channel]  = np.mean(np.array([np.pad(s, (0, max_len - len(s))) for s in responses]), axis=0)

        mag_interp[channel]   = np.mean(np.array([resample_to(s, FIXED_LEN) for s in mags]),      axis=0) * 1e4
        phase_interp[channel] = np.mean(np.array([resample_to(s, FIXED_LEN) for s in phases]),    axis=0)
        freq_interp[channel]  = np.mean(np.array([resample_to(s, FIXED_LEN) for s in responses]), axis=0)

    # create per-label subdirectories
    for approach_dir in [trim_dir, pad_dir, interpolate_dir]:
        (approach_dir / label).mkdir(parents=True, exist_ok=True)
    for csv_dir in [trim_csv_dir, pad_csv_dir, interpolate_csv_dir]:
        (csv_dir / label).mkdir(parents=True, exist_ok=True)

    # ── trim ─────────────────────────────────────────────────────────────────
    plot_magnitude(mag_trim,   label, trim_dir / label,
                   "Trim to Minimum", "All segments trimmed to the shortest segment length",
                   n_patients, label_segment_counts)
    plot_phase(phase_trim,     label, trim_dir / label,
               "Trim to Minimum", "All segments trimmed to the shortest segment length",
               n_patients, label_segment_counts)
    plot_frequency_response(freq_trim, label, trim_dir / label,
                            "Trim to Minimum", "All segments trimmed to the shortest segment length",
                            n_patients, label_segment_counts)
    save_csv(mag_trim,   label, "magnitude",         trim_csv_dir / label, "trim_to_minimum",        label_segment_counts)
    save_csv(phase_trim, label, "phase",              trim_csv_dir / label, "trim_to_minimum",        label_segment_counts)
    save_csv(freq_trim,  label, "frequency_response", trim_csv_dir / label, "trim_to_minimum",        label_segment_counts)

    # ── pad ──────────────────────────────────────────────────────────────────
    plot_magnitude(mag_pad,    label, pad_dir / label,
                   "Zero Pad to Maximum", "Shorter segments zero padded to the longest segment length",
                   n_patients, label_segment_counts)
    plot_phase(phase_pad,      label, pad_dir / label,
               "Zero Pad to Maximum", "Shorter segments zero padded to the longest segment length",
               n_patients, label_segment_counts)
    plot_frequency_response(freq_pad, label, pad_dir / label,
                            "Zero Pad to Maximum", "Shorter segments zero padded to the longest segment length",
                            n_patients, label_segment_counts)
    save_csv(mag_pad,   label, "magnitude",         pad_csv_dir / label, "zero_pad_to_maximum",    label_segment_counts)
    save_csv(phase_pad, label, "phase",              pad_csv_dir / label, "zero_pad_to_maximum",    label_segment_counts)
    save_csv(freq_pad,  label, "frequency_response", pad_csv_dir / label, "zero_pad_to_maximum",    label_segment_counts)

    # ── interpolate ───────────────────────────────────────────────────────────
    plot_magnitude(mag_interp,   label, interpolate_dir / label,
                   "Interpolate to Fixed Length", f"All segments resampled to {FIXED_LEN} points via cubic interpolation",
                   n_patients, label_segment_counts)
    plot_phase(phase_interp,     label, interpolate_dir / label,
               "Interpolate to Fixed Length", f"All segments resampled to {FIXED_LEN} points via cubic interpolation",
               n_patients, label_segment_counts)
    plot_frequency_response(freq_interp, label, interpolate_dir / label,
                            "Interpolate to Fixed Length", f"All segments resampled to {FIXED_LEN} points via cubic interpolation",
                            n_patients, label_segment_counts)
    save_csv(mag_interp,   label, "magnitude",         interpolate_csv_dir / label, "interpolate_to_fixed_length", label_segment_counts)
    save_csv(phase_interp, label, "phase",              interpolate_csv_dir / label, "interpolate_to_fixed_length", label_segment_counts)
    save_csv(freq_interp,  label, "frequency_response", interpolate_csv_dir / label, "interpolate_to_fixed_length", label_segment_counts)

    print(f"saved plots and CSVs for: {label}")

print(f"\nall plots saved to:  {plots_dir}")
print(f"all CSVs saved to:   {analysis_dir}")