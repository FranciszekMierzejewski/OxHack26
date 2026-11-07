const audioFileInput = document.getElementById("audioFile");
const audiobook = document.getElementById("audiobook");
const speedSelect = document.getElementById("speedSelect");
const fileStatus = document.getElementById("fileStatus");
const speedStatus = document.getElementById("speedStatus");

let currentAudioUrl;

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
  fileStatus.textContent = file.name;
});

speedSelect.addEventListener("change", () => {
  const playbackRate = Number(speedSelect.value);
  audiobook.playbackRate = playbackRate;
  speedStatus.textContent = playbackRate === 1
    ? "Normal listening speed"
    : `Playback speed: ${playbackRate}x`;
});

window.addEventListener("unload", () => {
  if (currentAudioUrl) {
    URL.revokeObjectURL(currentAudioUrl);
  }
});
