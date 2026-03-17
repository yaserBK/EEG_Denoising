"""
Batch-append CSV annotations to matching EDF files.

Usage:
    python batch_annotate_v3.py /path/to/data/dir [--output-dir /path/to/output]

Requires:
    pip install mne pandas tqdm
"""

import argparse
from pathlib import Path

import pandas as pd
import mne
from tqdm import tqdm

mne.set_log_level("ERROR")


def load_annotations_from_csv(csv_path):
    df = pd.read_csv(csv_path, comment="#")
    onsets = df["start_time"].values
    durations = df["stop_time"].values - df["start_time"].values
    descriptions = (df["label"] + "/" + df["channel"]).values
    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)


def process_pair(edf_path, csv_path, output_dir):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    annotations = load_annotations_from_csv(csv_path)
    raw.set_annotations(raw.annotations + annotations)

    fif_out = output_dir / (edf_path.stem + "_annotated-raw.fif")
    raw.save(str(fif_out), overwrite=True, verbose=False)

    try:
        edf_out = output_dir / (edf_path.stem + "_annotated.edf")
        mne.export.export_raw(str(edf_out), raw, overwrite=True, verbose=False)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "annotated"
    output_dir.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(data_dir.glob("*.edf"))
    pairs = [(e, e.with_suffix(".csv")) for e in edf_files if e.with_suffix(".csv").exists()]
    no_csv = len(edf_files) - len(pairs)

    errors = []
    for edf_path, csv_path in tqdm(pairs, desc="Annotating", unit="file"):
        try:
            process_pair(edf_path, csv_path, output_dir)
        except Exception as e:
            errors.append((edf_path.name, str(e)))

    print(f"\n{len(pairs) - len(errors)} annotated, "
          f"{len(errors)} errors, {no_csv} skipped (no csv)")
    if errors:
        print(f"\nErrors:")
        for name, msg in errors:
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
