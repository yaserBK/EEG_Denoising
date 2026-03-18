import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from bokeh.plotting import figure, output_file
from bokeh.layouts import column
from bokeh.models import HoverTool, Div
from bokeh.io import save
from eeg_denoising import config, data, montage, eeg_utils
from eeg_denoising.eeg_utils import get_segments
from collections import defaultdict


def compute_frequency_response(signal, sfreq):
    fft_vals = np.fft.rfft(signal)
    response = 20 * np.log10(np.abs(fft_vals) + 1e-10)
    return response


def resample_to(arr, length):
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, length)
    return interp1d(x_old, arr)(x_new)


FIXED_LEN = 1024

master     = data.load_master()
n_patients = len(master["Patients"])

# create output directories
plots_dir       = config.PATH_TO_ROOT / "plots"
freq_trim_dir   = plots_dir / "01_trim_to_minimum"
freq_pad_dir    = plots_dir / "02_zero_pad_to_maximum"
freq_interp_dir = plots_dir / "03_interpolate_to_fixed_length"

for d in [freq_trim_dir, freq_pad_dir, freq_interp_dir]:
    d.mkdir(parents=True, exist_ok=True)

# accumulate raw frequency responses per label per channel
label_freq_accum     = defaultdict(lambda: defaultdict(list))
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

            response      = compute_frequency_response(signal, sfreq)
            response_norm = response - response.max()   # normalise to 0 dB peak

            label_freq_accum[label][channel].append(response_norm)
            label_segment_counts[label][channel] += 1

        del edf_data


def plot_frequency_response(avg_response, label, output_dir, approach_name, approach_description):
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
        p.add_tools(HoverTool(tooltips=[
            ("freq",     "@x{0.1f} Hz"),
            ("response", "@y{0.1f} dB")
        ]))
        p.line(ch_freqs[ch_mask], resp[ch_mask], line_width=0.8, color="mediumseagreen")
        p.xaxis.axis_label = "Frequency (Hz)" if channel == list(avg_response.keys())[-1] else ""
        p.yaxis.axis_label = "Response (dB)"
        p.title.text_font_size = "10px"
        plots.append(p)

    save(column(*plots))


for label, channel_data in label_freq_accum.items():
    if not channel_data:
        continue

    freq_trim   = {}
    freq_pad    = {}
    freq_interp = {}

    for channel, responses in channel_data.items():

        # ── approach 1: trim to minimum length ───────────────────────────────
        min_len = min(len(r) for r in responses)
        freq_trim[channel] = np.mean(np.array([r[:min_len] for r in responses]), axis=0)

        # ── approach 2: zero pad to maximum length ────────────────────────────
        max_len = max(len(r) for r in responses)
        freq_pad[channel] = np.mean(np.array([np.pad(r, (0, max_len - len(r))) for r in responses]), axis=0)

        # ── approach 3: interpolate to fixed length ───────────────────────────
        freq_interp[channel] = np.mean(np.array([resample_to(r, FIXED_LEN) for r in responses]), axis=0)

    plot_frequency_response(
        freq_trim, label, freq_trim_dir,
        "Trim to Minimum",
        "All segments trimmed to the shortest segment length"
    )

    plot_frequency_response(
        freq_pad, label, freq_pad_dir,
        "Zero Pad to Maximum",
        "Shorter segments zero padded to the longest segment length"
    )

    plot_frequency_response(
        freq_interp, label, freq_interp_dir,
        "Interpolate to Fixed Length",
        f"All segments resampled to {FIXED_LEN} points via cubic interpolation"
    )

print(f"Saved frequency response plots to {plots_dir}")