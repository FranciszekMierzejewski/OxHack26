"""NeuroTune attention service for the Chrome side panel.

Run from the repository root with ``python -m src.server --source sim``.
Supported sources are ``sim``, ``replay``, and ``lsl``.
"""
import argparse
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from src.attention import AttentionTracker, SpeedPolicy, pick_channels
from src.eeg_sources import EEGSource, H5ReplaySource, LSLSource, SimulatedSource

WINDOW_S = 2.0
STEP_S = 1.0
STALE_AFTER_S = 3.0
RETRY_S = 2.0


class Engine(threading.Thread):
    """Read EEG in the background and keep the latest result available."""

    def __init__(self, make_source: Callable[[], EEGSource], source_name: str,
                 calibration_s: float) -> None:
        super().__init__(daemon=True)
        self.make_source = make_source
        self.source_name = source_name
        self.calibration_s = calibration_s
        self.lock = threading.Lock()
        self.recalibrate_requested = threading.Event()
        self.stop_requested = threading.Event()
        self.error: str | None = None
        self.last_data = 0.0
        self.result: dict[str, Any] = {
            "calibration_progress": 0.0,
            "attention": None,
            "speed_factor": 1.0,
            "artifact": False,
            "calibrated": False,
        }

    def run(self) -> None:
        while not self.stop_requested.is_set():
            try:
                self._stream(self.make_source())
            except Exception as exception:  # The headset or stream may not be ready yet.
                self.error = str(exception)
                self.stop_requested.wait(RETRY_S)

    def _stream(self, source: EEGSource) -> None:
        self.error = None
        channels = pick_channels(source.channel_names)
        if not channels:
            raise ValueError("The EEG source has no channels")

        window_samples = int(WINDOW_S * source.fs)
        step_samples = int(STEP_S * source.fs)
        tracker = AttentionTracker(source.fs, self.calibration_s, STEP_S)
        policy = SpeedPolicy()
        self._publish(tracker, policy, False)
        buffer = np.empty((len(channels), 0))
        since_step = 0

        while not self.stop_requested.is_set():
            chunk = source.read()[channels]
            if chunk.shape[1] == 0:
                continue
            with self.lock:
                self.last_data = time.monotonic()
            buffer = np.concatenate((buffer, chunk), axis=1)[:, -window_samples:]
            since_step += chunk.shape[1]

            if self.recalibrate_requested.is_set():
                self.recalibrate_requested.clear()
                tracker.reset()
                policy.reset()
                self._publish(tracker, policy, False)

            if buffer.shape[1] < window_samples or since_step < step_samples:
                continue
            since_step = 0
            artifact = tracker.update(buffer)
            if tracker.attention is not None and not artifact:
                policy.update(tracker.attention, STEP_S)
            self._publish(tracker, policy, artifact)

    def _publish(self, tracker: AttentionTracker, policy: SpeedPolicy,
                 artifact: bool) -> None:
        with self.lock:
            self.result = {
                "calibration_progress": round(tracker.progress, 3),
                "attention": None if tracker.attention is None else round(tracker.attention, 3),
                "speed_factor": policy.factor if tracker.calibrated else 1.0,
                "artifact": artifact,
                "calibrated": tracker.calibrated,
            }

    def state(self) -> dict[str, Any]:
        """Return the latest score and a status derived from signal freshness."""
        with self.lock:
            result = dict(self.result)
            fresh = time.monotonic() - self.last_data < STALE_AFTER_S
        calibrated = result.pop("calibrated")
        if self.error:
            status = "error"
        elif not fresh:
            status = "no_signal"
        elif calibrated:
            status = "tracking"
        else:
            status = "calibrating"
        if status != "tracking":
            result["attention"] = None
            result["speed_factor"] = 1.0
        remaining = (1.0 - result["calibration_progress"]) * self.calibration_s
        return {**result, "status": status, "source": self.source_name,
                "calibration_remaining_s": round(remaining), "error": self.error}


def create_app(engine: Engine) -> FastAPI:
    """Create the local API and manage the EEG worker with application lifespan."""
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        engine.start()
        yield
        engine.stop_requested.set()

    app = FastAPI(title="NeuroTune", lifespan=lifespan)

    @app.get("/state")
    def state() -> dict[str, Any]:
        return engine.state()

    @app.post("/calibrate")
    def calibrate(request: Request) -> dict[str, bool]:
        # Web pages can send requests to localhost, so accept only the extension.
        origin = request.headers.get("origin")
        if origin and not origin.startswith("chrome-extension://"):
            raise HTTPException(status_code=403, detail="Origin not allowed")
        engine.recalibrate_requested.set()
        return {"ok": True}

    return app


def main() -> None:
    """Parse command-line options and start the local attention service."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--source", choices=("sim", "replay", "lsl"), default="sim")
    parser.add_argument("--file", help="recording to replay, e.g. data/raw/eeg/P00.h5")
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--part", default="part_1")
    parser.add_argument("--calibration", type=float, default=30.0, help="baseline length in seconds")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.source == "replay" and not args.file:
        parser.error("--source replay needs --file")

    makers: dict[str, Callable[[], EEGSource]] = {
        "sim": lambda: SimulatedSource(calibration_s=args.calibration),
        "replay": lambda: H5ReplaySource(args.file, args.condition, args.part),
        "lsl": LSLSource,
    }
    engine = Engine(makers[args.source], args.source, args.calibration)
    uvicorn.run(create_app(engine), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
