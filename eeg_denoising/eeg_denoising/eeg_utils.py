import mne
import numpy as np
from pathlib import Path

def load_edf(recording: dict, edf_dir: Path, montage_channels: list[dict]) -> tuple[np.ndarray, list[str], float]:
    from .montage import apply_nedc_montage
    fpath = edf_dir / Path(recording["edf"]).name
    raw = mne.io.read_raw_edf(fpath, preload=True)
    raw = apply_nedc_montage(raw, montage_channels)

    data     = raw.get_data() * 1e6
    ch_names = raw.ch_names
    sfreq    = raw.info["sfreq"]

    del raw
    return data, ch_names, sfreq


def iter_edfs(patient: dict, edf_dir: Path, montage_channels: list[dict]):
    for recording_id, recording in patient["recordings"].items():
        if recording["edf"] is None:
            continue
        data, ch_names, sfreq = load_edf(recording, edf_dir, montage_channels)
        yield recording_id, data, ch_names, sfreq, recording

def get_segments(
    recording: dict,
    labels: str | list[str] | None = None,
    channels: str | list[str] | None = None,
    include_seiz: bool = False,
) -> list[dict]:

    if isinstance(labels, str):
        labels = [labels]
    if isinstance(channels, str):
        channels = [channels]

    sources = [recording["labels"]]
    if include_seiz:
        sources.append(recording.get("seiz_labels", {}))

    segments = []
    for source in sources:
        for label, data in source.items():
            if labels and label not in labels:
                continue
            for seg in data["segments"]:
                if channels and seg["channel"] not in channels:
                    continue
                segments.append({
                    "start":      seg["start"],
                    "stop":       seg["stop"],
                    "duration":   seg["stop"] - seg["start"],
                    "channel":    seg["channel"],
                    "label":      label,
                    "confidence": seg["confidence"],
                })

    return segments


def extract_signal(data: np.ndarray, ch_names: list[str], seg: dict, sfreq: float) -> np.ndarray:
    ch_idx = ch_names.index(seg["channel"])
    start  = int(seg["start"] * sfreq)
    stop   = int(seg["stop"]  * sfreq)
    return data[ch_idx, start:stop]


def extract_all_signals(
    data: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    recording: dict,
    labels: str | list[str] | None = None,
    channels: str | list[str] | None = None,
    include_seiz: bool = False,
) -> tuple[list[np.ndarray], list[str]]:
    segments = get_segments(recording, labels=labels, channels=channels, include_seiz=include_seiz)
    X = [extract_signal(data, ch_names, seg, sfreq) for seg in segments]
    y = [seg["label"] for seg in segments]
    return X, y


def apply_annotations(
    raw: mne.io.BaseRaw,
    recording: dict,
    labels: str | list[str] | None = None,
    channels: str | list[str] | None = None,
    include_seiz: bool = False,
) -> mne.io.BaseRaw:
    segments = get_segments(recording, labels=labels, channels=channels, include_seiz=include_seiz)
    annotations = mne.Annotations(
        onset       = [s["start"]                      for s in segments],
        duration    = [s["duration"]                   for s in segments],
        description = [f"{s['label']}:{s['channel']}" for s in segments],
    )
    raw.set_annotations(annotations)
    return raw