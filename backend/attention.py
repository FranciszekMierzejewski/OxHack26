"""Turn windows of EEG into an attention estimate and a playback speed factor.

The estimate is the engagement index beta / (alpha + theta), compared with the
listener's own calibration baseline. It is a published heuristic, not a model
trained on attention labels, so treat it as a relative signal for one listener.
"""
import math

import numpy as np
from scipy.signal import detrend, welch

THETA = (4.0, 8.0)
ALPHA = (8.0, 13.0)
BETA = (13.0, 30.0)

# Central/parietal sites used for the engagement index. They also sit further
# from the eyes than frontal sites, so blinks affect them less.
PREFERRED_CHANNELS = ("Cz", "Pz", "P3", "P4")


def pick_channels(names):
    """Return indices of the preferred channels, or all channels if none match."""
    wanted = {name.lower() for name in PREFERRED_CHANNELS}
    picked = [i for i, name in enumerate(names) if name.strip().lower() in wanted]
    return picked or list(range(len(names)))


def _band_power(freqs, psd, band):
    mask = (freqs >= band[0]) & (freqs < band[1])
    return psd[:, mask].sum(axis=1)


def extract_features(window, fs):
    """Return (log engagement index, peak-to-peak size), or None if unusable.

    window has shape (channels, samples). Welch's linear detrend removes the
    large DC offset and slow drift seen in the recordings.
    """
    usable = np.all(np.isfinite(window), axis=1) & (np.ptp(window, axis=1) > 0)
    x = window[usable]
    if x.shape[0] == 0:
        return None

    freqs, psd = welch(x, fs=fs, nperseg=min(int(fs), x.shape[1]), detrend="linear", axis=1)
    theta = _band_power(freqs, psd, THETA)
    alpha = _band_power(freqs, psd, ALPHA)
    beta = _band_power(freqs, psd, BETA)
    if np.any(alpha + theta <= 0) or np.any(beta <= 0):
        return None

    log_index = float(np.median(np.log(beta / (alpha + theta))))
    peak_to_peak = float(np.median(np.ptp(detrend(x, axis=1), axis=1)))
    return log_index, peak_to_peak


class AttentionTracker:
    """Learn a personal baseline, then score each window against it."""

    MIN_SCALE = 0.05  # stops a very quiet baseline from exaggerating z-scores
    ARTIFACT_RATIO = 4.0  # reject windows this many times the normal amplitude

    def __init__(self, fs, calibration_s=30.0, step_s=1.0, smoothing_s=5.0):
        self.fs = fs
        self.step_s = step_s
        self.calibration_windows = max(5, round(calibration_s / step_s))
        self.smoothing = 1.0 - math.exp(-step_s / smoothing_s)
        self.reset()

    def reset(self):
        self._log_indices = []
        self._peak_to_peaks = []
        self.median = None
        self.scale = None
        self.peak_limit = None
        self.z = None

    @property
    def calibrated(self):
        return self.median is not None

    @property
    def progress(self):
        if self.calibrated:
            return 1.0
        return len(self._log_indices) / self.calibration_windows

    @property
    def attention(self):
        if self.z is None:
            return None
        return 1.0 / (1.0 + math.exp(-self.z))

    def update(self, window):
        """Process one window. Returns True if it was rejected as an artifact."""
        features = extract_features(window, self.fs)
        if features is None:
            return True
        log_index, peak_to_peak = features

        if not self.calibrated:
            self._log_indices.append(log_index)
            self._peak_to_peaks.append(peak_to_peak)
            if len(self._log_indices) >= self.calibration_windows:
                self._finish_calibration()
            return False

        if peak_to_peak > self.peak_limit:
            return True

        z = (log_index - self.median) / self.scale
        self.z = z if self.z is None else self.z + self.smoothing * (z - self.z)
        return False

    def _finish_calibration(self):
        values = np.asarray(self._log_indices)
        self.median = float(np.median(values))
        mad = float(np.median(np.abs(values - self.median)))
        self.scale = max(1.4826 * mad, self.MIN_SCALE)
        self.peak_limit = self.ARTIFACT_RATIO * float(np.median(self._peak_to_peaks))


class SpeedPolicy:
    """Map attention to a speed factor, moving one calm step at a time."""

    FACTORS = (1.0, 0.9, 0.8)
    THRESHOLDS = (0.42, 0.30)  # below these, move to the next slower level
    MARGIN = 0.04  # attention must clear the threshold by this much to speed up
    DWELL_S = 4.0  # a change must be wanted for this long before it happens

    def __init__(self):
        self.reset()

    def reset(self):
        self.level = 0
        self._pending = 0
        self._held_s = 0.0

    @property
    def factor(self):
        return self.FACTORS[self.level]

    def _wanted_direction(self, attention):
        if self.level < len(self.THRESHOLDS) and attention < self.THRESHOLDS[self.level]:
            return 1
        if self.level > 0 and attention >= self.THRESHOLDS[self.level - 1] + self.MARGIN:
            return -1
        return 0

    def update(self, attention, dt):
        direction = self._wanted_direction(attention)
        if direction == 0 or direction != self._pending:
            self._pending = direction
            self._held_s = 0.0
        if direction != 0:
            self._held_s += dt
            if self._held_s >= self.DWELL_S:
                self.level += direction
                self._pending = 0
                self._held_s = 0.0
        return self.factor
