const API_URL = "http://127.0.0.1:8000";
const POLL_MS = 1000;

const audioFileInput = document.getElementById("audioFile");
const audiobook = document.getElementById("audiobook");
const speedSelect = document.getElementById("speedSelect");
const speedValue = document.getElementById("speedValue");
const fileStatus = document.getElementById("fileStatus");
const speedStatus = document.getElementById("speedStatus");
const attentionStatus = document.getElementById("attentionStatus");
const attentionMeter = document.getElementById("attentionMeter");
const adaptiveToggle = document.getElementById("adaptiveToggle");
const recalibrateButton = document.getElementById("recalibrateButton");

let currentAudioUrl;
let speedFactor = 1;

function applySpeed() {
  const preferred = Number(speedSelect.value);
  const factor = adaptiveToggle.checked ? speedFactor : 1;
  const rate = Math.round(preferred * factor * 100) / 100;

  // load() resets playbackRate to the default, so keep both in step.
  audiobook.defaultPlaybackRate = rate;
  if (audiobook.playbackRate !== rate) {
    audiobook.playbackRate = rate;
  }

  speedValue.textContent = `${preferred}x`;
  if (factor < 1) {
    speedStatus.textContent = `Slowed to ${rate}x to help you refocus`;
  } else if (rate === 1) {
    speedStatus.textContent = "Normal listening speed";
  } else {
    speedStatus.textContent = `Playback speed: ${rate}x`;
  }
}

function describeState(state) {
  switch (state.status) {
    case "calibrating":
      return `Calibrating: keep listening normally (${state.calibration_remaining_s}s left)`;
    case "tracking":
      if (state.artifact) {
        return "Noisy EEG signal, holding the current speed";
      }
      return `Attention: ${Math.round(state.attention * 100)}%`;
    case "no_signal":
      return "No EEG signal, using your preferred speed";
    default:
      return `EEG problem: ${state.error}`;
  }
}

function showState(state) {
  speedFactor = state.speed_factor;
  attentionStatus.textContent = describeState(state);
  attentionMeter.value = state.attention ?? 0;
  attentionMeter.hidden = state.status !== "tracking";
  recalibrateButton.disabled = state.status === "calibrating";
  applySpeed();
}

function showDisconnected() {
  speedFactor = 1;
  attentionStatus.textContent = "Waiting for EEG connection";
  attentionMeter.hidden = true;
  recalibrateButton.disabled = true;
  applySpeed();
}

async function poll() {
  try {
    const response = await fetch(`${API_URL}/state`, { signal: AbortSignal.timeout(POLL_MS) });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    showState(await response.json());
  } catch {
    showDisconnected();
  }
  setTimeout(poll, POLL_MS);
}

audioFileInput.addEventListener("change", () => {
  const file = audioFileInput.files[0];
  if (!file) {
    return;
  }

  if (currentAudioUrl) {
    URL.revokeObjectURL(currentAudioUrl);
  }

  currentAudioUrl = URL.createObjectURL(file);
  audiobook.src = currentAudioUrl;
  audiobook.load();
  applySpeed();
  fileStatus.textContent = file.name;
});

speedSelect.addEventListener("input", applySpeed);
adaptiveToggle.addEventListener("change", applySpeed);

recalibrateButton.addEventListener("click", async () => {
  try {
    await fetch(`${API_URL}/calibrate`, { method: "POST" });
  } catch {
    showDisconnected();
  }
});

applySpeed();
poll();
