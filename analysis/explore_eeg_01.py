from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data/raw/eeg"
PROCESSED_PATH = PROJECT_ROOT / "data/processed/eeg"

# small std = small variance = dead EEG channel, set threshold req
FLAT_STD = 1e-10


def _decode(name: bytes | str) -> str:
    """Normalize HDF5 byte and string metadata to Python strings."""
    return name.decode() if isinstance(name, bytes) else str(name)


def inventory(path: Path) -> list[dict[str, Any]]:
    """Collect recording metadata and basic quality metrics for one HDF5 file."""
    rows: list[dict[str, Any]] = []

    with h5py.File(path, "r") as file: # autoclose fyi
        channels = [_decode(channel) for channel in file["ch_names"][:]] # decode each channel name
        sampling_frequency = int(np.ravel(file["srate"][:])[0]) # read sample rate, flatten to 1d, take 1st value and cast to int
        # since we know Time Period = 1/frequency, we can find sample seperation
        # ^^ assuming single value anyway, install H5Web extension to view .h5 files and check

        # fix to remove plain names ch_names and sampling frequency, keep only conditions
        conditions = [key for key in file.keys() if isinstance(file[key], h5py.Group)]

        for condition in conditions:
            # e.g. part_1, ... part_n, we grab this number and sort by it
            parts = sorted(file[condition].keys(), key = lambda s: int(s.split("_")[1]))

            for part in parts:
                x = file[condition][part][:].astype(np.float64) # shape of (number_of_channels, number_of_samples)


                # verify matrix shape
                orientation_state = x.shape[0] == len(channels)
                if not orientation_state and x.shape[1] == len(channels):
                    x = x.T # transpose
                    orientation_state = True
                if not orientation_state:
                    raise ValueError(
                        f"{path}: {condition}/{part} shape {x.shape} does not match "
                        f"{len(channels)} channel names"
                    )

                std = x.std(axis=1, ddof=1) # calc sample std of every channel across time
                # how strongly each row's average stands out relative to row variation
                dc_ratio = np.abs(x.mean(axis=1)) / np.maximum(std, FLAT_STD)

                rows.append({
                    "participant" : path.stem, # filename without extension
                    "condition" : condition,
                    "part" : part,
                    "number_of_channels" : x.shape[0],
                    "number_of_samples" : x.shape[1],
                    "sampling_frequency" : sampling_frequency,
                    "seconds" : round(x.shape[1]/sampling_frequency, 2), # samples/frequency
                    "orientation_state" : orientation_state,
                    "median_std" : float(np.median(std)),
                    "max_abs" : float(np.abs(x).max()),
                    "dead_channels" : int((std < FLAT_STD).sum()),
                    "nan_count" : int(np.isnan(x).sum()),
                    "median_dc_to_std" : float(np.median(dc_ratio))
                })

    return rows
                 

def main() -> None:
    all_raw_files = RAW_PATH.iterdir()

    # keep only files starting with P and has .h5 extension
    eeg_files = sorted(item for item in all_raw_files if item.is_file() and item.suffix == ".h5" and item.name.startswith("P"))

    if not eeg_files:
        raise FileNotFoundError(f"No EEG files found in {RAW_PATH.resolve()}. Please verify that file name starts with P and has .h5 extension\n")

    print(f"Found {len(eeg_files)} EEG file(s) in {RAW_PATH.resolve()}.\n")

    all_parts = []
    for eeg_file in eeg_files:
        print(f"Reading {eeg_file.name}")

        try:
            part_info = inventory(eeg_file)
        except Exception as exception:
            print(f"Could not read {eeg_file.name} : {exception}")
            continue

        all_parts.extend(part_info)

    if not all_parts:
        raise IOError("No artifacts from the EEG file can be read. Check files.\n")

    manifest = pd.DataFrame(all_parts)
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    manifest_path = PROCESSED_PATH / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("\nParts per participant and condition")
    
    part_counts = manifest.pivot_table(
        index="participant",
        columns="condition",
        values="part",
        aggfunc="count",
        fill_value=0,
    )
    print(part_counts.to_string())

    print("\nRecording duration (seconds) by condition:")
    durations = manifest.groupby("condition")["seconds"].agg(
        ["min", "median", "max"]
    )
    print(durations.to_string())

    print("\nData-quality checks:")

    print(f"  Sampling rates: {sorted(manifest['sampling_frequency'].unique())}")
    print(f"  Channel counts: {sorted(manifest['number_of_channels'].unique())}")
    print(f"  Parts with orientation problems: {(~manifest['orientation_state']).sum()}")
    print(f"  Parts with dead channels: {(manifest['dead_channels'] > 0).sum()}")
    print(f"  Parts containing NaNs: {(manifest['nan_count'] > 0).sum()}")
 
    print(
        f"  Median channel standard deviation: {manifest['median_std'].median():.3g} "
    )
    print(
        f"  Median |mean| / standard deviation: {manifest['median_dc_to_std'].median():.2f} "
        "(much greater than 1 suggests a DC offset; high-pass filter)"
    )
 
    print(f"\nManifest saved to: {manifest_path}")



if __name__ == "__main__":
    main()