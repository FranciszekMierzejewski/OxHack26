"""EEG inputs. Each source has fs, channel_names and read() -> (channels, samples)."""
import time
from pathlib import Path
from typing import Protocol

import h5py
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class EEGSource(Protocol):
    """Common interface for live and replay EEG sources."""

    fs: float
    channel_names: list[str]

    def read(self) -> FloatArray:
        """Return the next chunk in channels-by-samples order."""


class _Pacer:
    """Sleep so that chunks are delivered at the rate they were recorded."""

    def __init__(self, chunk_s: float, pace: float = 1.0) -> None:
        if chunk_s <= 0 or pace < 0:
            raise ValueError("chunk duration must be positive and pace non-negative")
        self.chunk_s = chunk_s / pace if pace else 0.0
        self.next_time = time.monotonic()

    def wait(self) -> None:
        if not self.chunk_s:
            return
        self.next_time += self.chunk_s
        delay = self.next_time - time.monotonic()
        if delay > 0:
            time.sleep(delay)


class SimulatedSource:
    """Synthetic EEG that alternates focused and drifting spells, for demos."""

    channel_names = ["Fz", "Cz", "Pz", "P3", "P4", "O1", "O2", "Oz"]
    FOCUSED = 0.75
    DRIFTING = 0.10
    NEUTRAL = 0.5

    def __init__(self, calibration_s: float = 30.0, spell_s: float = 40.0,
                 fs: int = 250, pace: float = 1.0, seed: int | None = None) -> None:
        if fs < 4 or calibration_s <= 0 or spell_s <= 0:
            raise ValueError("sampling rate must be at least 4 Hz and durations positive")
        self.fs = fs
        self.calibration_s = calibration_s
        self.spell_s = spell_s
        self.rng = np.random.default_rng(seed)
        self.sample = 0
        self.chunk = fs // 4
        self.pacer = _Pacer(self.chunk / fs, pace)
        n = len(self.channel_names)
        self.offsets = self.rng.normal(0.02, 0.002, n)  # volts, like the recordings
        self.phases = self.rng.uniform(0, 2 * np.pi, (3, n))

    def true_attention(self, t: float) -> float:
        if t < self.calibration_s:
            return self.NEUTRAL
        spell = int((t - self.calibration_s) // self.spell_s)
        return self.FOCUSED if spell % 2 == 0 else self.DRIFTING

    def read(self) -> FloatArray:
        self.pacer.wait()
        t = (self.sample + np.arange(self.chunk)) / self.fs
        focus = self.true_attention(t[0])
        amplitudes = {
            6.0: 4e-6 + 12e-6 * (1 - focus),
            10.0: 5e-6 + 28e-6 * (1 - focus),
            20.0: 3e-6 + 14e-6 * focus,
        }
        n = len(self.channel_names)
        data = self.rng.normal(0, 6e-6, (n, self.chunk)) + self.offsets[:, None]
        for phase, (freq, amplitude) in zip(self.phases, amplitudes.items()):
            data += amplitude * np.sin(2 * np.pi * freq * t + phase[:, None])
        self.sample += self.chunk
        return data


class H5ReplaySource:
    """Replay one part of a recorded .h5 file in real time, looping at the end."""

    def __init__(self, path: str | Path, condition: str = "clean",
                 part: str = "part_1", pace: float = 1.0) -> None:
        with h5py.File(path, "r") as file:
            names = file["ch_names"][:]
            self.channel_names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
            self.fs = float(np.ravel(file["srate"][:])[0])
            if self.fs <= 0:
                raise ValueError(f"{path}: sampling rate must be positive")
            data = file[condition][part][:].astype(np.float64)
        if data.shape[0] != len(self.channel_names):
            data = data.T
        if data.shape[0] != len(self.channel_names):
            raise ValueError(f"{path}: data shape does not match the channel list")
        if data.shape[1] == 0:
            raise ValueError(f"{path}: recording part contains no samples")
        self.data = data
        self.position = 0
        self.chunk = int(self.fs / 4)
        self.pacer = _Pacer(self.chunk / self.fs, pace)

    def read(self) -> FloatArray:
        self.pacer.wait()
        end = self.position + self.chunk
        chunk = self.data[:, self.position:end]
        self.position = end if end < self.data.shape[1] else 0
        return chunk


class LSLSource:
    """Read a live headset through Lab Streaming Layer (stream type EEG)."""

    def __init__(self, timeout: float = 5.0) -> None:
        from pylsl import StreamInlet, resolve_byprop

        streams = resolve_byprop("type", "EEG", timeout=timeout)
        if not streams:
            raise ConnectionError("No LSL stream of type EEG was found")
        self.inlet = StreamInlet(streams[0], max_buflen=10)
        info = self.inlet.info()
        self.fs = float(info.nominal_srate())
        if self.fs <= 0:
            raise ValueError("The LSL stream does not declare a sampling rate")
        self.channel_names = self._read_labels(info)

    @staticmethod
    def _read_labels(info: object) -> list[str]:
        labels = []
        channel = info.desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
            labels.append(channel.child_value("label"))
            channel = channel.next_sibling()
        if not all(labels):
            return [f"ch{i}" for i in range(info.channel_count())]
        return labels

    def read(self) -> FloatArray:
        samples, _ = self.inlet.pull_chunk(timeout=0.25, max_samples=int(self.fs))
        if not samples:
            return np.empty((len(self.channel_names), 0))
        return np.asarray(samples, dtype=np.float64).T
