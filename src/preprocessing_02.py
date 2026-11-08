from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos, welch

from explore_eeg_01 import RAW_PATH, PROCESSED_PATH

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


def build_filter():
    """
    Combined digital filter that passes freq band, removes interfering frequency
    returns coeff. as second order sections array
    """
    bandpass = butter(4, BAND_HZ, btype = "bandpass", fs = FS_IN, output = "sos")
    notch = tf2sos(*iirnotch(NOTCH_HZ, Q=30, fs=FS_IN)) # surpress mains at band mains with sharpness 30
 
    return np.vstack([bandpass, notch]) # vstack both matrices


def filter_causal(x, sos):
    # x: (channels, samples). Initialise state for a constant input equal to x[:, 0].
    zi = sosfilt_zi(sos)[:, None, :] * x[None, :, :1]
    y, _ = sosfilt(sos, x, axis=1, zi=zi)
    return y  # i.e. return filtered version of each channel

def band_power(window):
    # window: (channels, samples) at frequency sampling. Returns (n_bands, channels) of log10 power.
    freqs, psd = welch(window, fs=FS, nperseg=2 * FS, noverlap=FS, axis=1,
                       detrend="constant")
    out = []
    for low, high in BANDS.values():
        use = (freqs >= low) & (freqs <= high)
        out.append(trapezoid(psd[:, use], freqs[use], axis=1))
    return np.log10(np.asarray(out) + 1e-30)


def process_part(x, sos):
    # filter at the original rate, THEN decimate (the 30 Hz low-pass is the anti-alias filter)
    y = filter_causal(x, sos)[:, ::DECIMATE]
 
    size, hop, start0 = WINDOW_S * FS, HOP_S * FS, SETTLE_S * FS
    kept, n_rejected = [], 0
    for start in range(start0, y.shape[1] - size + 1, hop):
        window = y[:, start:start + size]
        ptp = window.max(axis=1) - window.min(axis=1)
        if (ptp > REJECT_V).mean() > MAX_BAD_FRACTION:
            n_rejected += 1
            continue
        kept.append((start / FS, band_power(window)))
    return kept, n_rejected

def main():
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
                    x = f[condition][part][:].astype(np.float64)[keep]
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