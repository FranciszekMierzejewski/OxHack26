from pathlib import Path
from typing import TypeAlias

import h5py
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos, welch

WindowFeatures: TypeAlias = tuple[float, np.ndarray]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data/raw/eeg"
PROCESSED_PATH = PROJECT_ROOT / "data/processed/eeg"

FS_IN, DECIMATE = 1000, 4 # 1000 samples per s, keeping 4th sample
FS = FS_IN // DECIMATE   # 250 Hz after decimation
BAND_HZ = (1.0, 30.0)   # band-pass edges to use between 1-30Hz
NOTCH_HZ = 50.0         # UK mains electricity interference
WINDOW_S, HOP_S = 4, 1  # 4 s windows, new window every 1s
SETTLE_S = 10     # discard first seconds  for filter settling
REJECT_V = 1.5e-4    # volts, peak-to-peak after filtering (150 uV)
MAX_BAD_FRACTION = 0.10  # reject window if >10% of channels exceed REJECT_V
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)} # standard freq bands for waves


CHANNELS_KEPT = None # since dataset has 63 channels, custom to your headset


def build_filter() -> np.ndarray:
    """Build the band-pass and mains-notch filter as second-order sections."""
    bandpass = butter(4, BAND_HZ, btype = "bandpass", fs = FS_IN, output = "sos")
    notch = tf2sos(*iirnotch(NOTCH_HZ, Q=30, fs=FS_IN)) # surpress mains at band mains with sharpness 30
 
    return np.vstack([bandpass, notch]) # vstack both matrices


def filter_causal(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Filter channels-by-samples data while initializing state from each channel."""
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("EEG data must be a non-empty channels-by-samples array")
    zi = sosfilt_zi(sos)[:, None, :] * x[None, :, :1]
    y, _ = sosfilt(sos, x, axis=1, zi=zi)
    return y  # i.e. return filtered version of each channel

def band_power(window: np.ndarray) -> np.ndarray:
    """Return log10 band power for each frequency band and EEG channel."""
    freqs, psd = welch(window, fs=FS, nperseg=2 * FS, noverlap=FS, axis=1,
                       detrend="constant")
    out = []
    for low, high in BANDS.values():
        use = (freqs >= low) & (freqs <= high)
        out.append(trapezoid(psd[:, use], freqs[use], axis=1))
    return np.log10(np.asarray(out) + 1e-30)


def process_part(x: np.ndarray, sos: np.ndarray) -> tuple[list[WindowFeatures], int]:
    """Filter one channels-by-samples recording and extract artifact-screened windows."""
    y = filter_causal(x, sos)[:, ::DECIMATE]
 
    size, hop, start0 = WINDOW_S * FS, HOP_S * FS, SETTLE_S * FS
    kept, n_rejected = [], 0
    for start in range(start0, y.shape[1] - size + 1, hop):
        window = y[:, start:start + size]
        finite_channels = np.all(np.isfinite(window), axis=1)
        bad_channels = np.ones(window.shape[0], dtype=bool)
        if np.any(finite_channels):
            finite_window = window[finite_channels]
            ptp = finite_window.max(axis=1) - finite_window.min(axis=1)
            bad_channels[finite_channels] = ptp > REJECT_V
        if not finite_channels.all() or bad_channels.mean() > MAX_BAD_FRACTION:
            n_rejected += 1
            continue
        kept.append((start / FS, band_power(window)))
    return kept, n_rejected

def main() -> None:
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in RAW_PATH.iterdir()
                   if p.is_file() and p.suffix == ".h5" and p.name.startswith("P"))
    if not files:
        raise FileNotFoundError(f"No P*.h5 files in {RAW_PATH.resolve()}")
 
    sos = build_filter()
    summary = []
 
    for path in files:
        print(f"Processing {path.name}")
        rows = []
        with h5py.File(path, "r") as f:
            channels = [c.decode() if isinstance(c, bytes) else str(c) for c in f["ch_names"][:]]
            fs = int(np.ravel(f["srate"][:])[0])
            if fs != FS_IN:
                raise ValueError(f"{path.name}: expected {FS_IN} Hz, got {fs}")
 
            keep = [i for i, c in enumerate(channels)
                    if CHANNELS_KEPT is None or c in CHANNELS_KEPT]
            kept_names = [channels[i] for i in keep]
            names = [f"{band}_{ch}" for band in BANDS for ch in kept_names]
 
            for condition in [k for k in f.keys() if isinstance(f[k], h5py.Group)]:
                for part in sorted(f[condition].keys(), key=lambda s: int(s.split("_")[1])):
                    x = f[condition][part][:].astype(np.float64)
                    if x.shape[0] != len(channels):
                        if x.shape[1] == len(channels):
                            x = x.T
                        else:
                            raise ValueError(
                                f"{path.name}: {condition}/{part} shape {x.shape} "
                                f"does not match {len(channels)} channel names"
                            )
                    x = x[keep]
                    windows, n_rej = process_part(x, sos)
                    summary.append({"participant": path.stem, "condition": condition,
                                    "part": part, "windows_kept": len(windows),
                                    "windows_rejected": n_rej})
                    for start_s, feats in windows:
                        rows.append([path.stem, condition, part, start_s, *feats.ravel()])
 
        df = pd.DataFrame(rows, columns=["participant", "condition", "part", "start_s", *names])
        df.to_csv(PROCESSED_PATH / f"features_{path.stem}.csv", index=False)
        print(f"  {len(df)} windows, {len(names)} features")
 
    summary = pd.DataFrame(summary)
    summary["reject_rate"] = summary.windows_rejected / (summary.windows_kept + summary.windows_rejected)
    summary.to_csv(PROCESSED_PATH / "preprocess_summary.csv", index=False)
    print("\nRejection rate by condition:")
    print(summary.groupby("condition")["reject_rate"].mean().round(3).to_string())
 
 
if __name__ == "__main__":
    main()