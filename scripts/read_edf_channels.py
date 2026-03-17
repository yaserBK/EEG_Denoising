"""
Read and display all channel names from an EDF file.

Usage:
    python read_edf_channels.py <path_to_edf_file>

Requires:
    pip install pyedflib
"""

import sys
import pyedflib


def get_channel_names(edf_path: str) -> list[str]:
    """Extract all channel names from an EDF file."""
    f = pyedflib.EdfReader(edf_path)
    try:
        n_channels = f.signals_in_file
        channel_names = f.getSignalLabels()
        return channel_names
    finally:
        f.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python read_edf_channels.py <path_to_edf_file>")
        sys.exit(1)

    edf_path = sys.argv[1]
    channels = get_channel_names(edf_path)

    print(f"Found {len(channels)} channels in '{edf_path}':\n")
    for i, name in enumerate(channels, start=1):
        print(f"  {i:3d}. {name}")


if __name__ == "__main__":
    main()
