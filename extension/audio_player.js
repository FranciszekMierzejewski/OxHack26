const speedSelect = document.getElementById("speedSelect");
const mediaStatus = document.getElementById("mediaStatus");
const speedStatus = document.getElementById("speedStatus");

let updateInProgress = false;
let applySpeedPending = false;

async function updatePageMedia(applySpeed = false) {
  if (updateInProgress) {
    applySpeedPending ||= applySpeed;
    return;
  }

  updateInProgress = true;
  const playbackRate = Number(speedSelect.value);

  try {
    if (typeof chrome.scripting?.executeScript !== "function") {
      mediaStatus.textContent = "The extension scripting API is unavailable.";
      speedStatus.textContent = "Reload NeuroTune from edge://extensions or chrome://extensions and confirm it has the scripting permission.";
      return;
    }

    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true
    });

    if (!tab || tab.id === undefined) {
      mediaStatus.textContent = "No active tab is available.";
      return;
    }

    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: (rate, shouldApplySpeed) => {
        const controllerKey = "__neuroTuneMediaController";
        let controller = globalThis[controllerKey];
        const isNewController = !controller;

        if (isNewController) {
          controller = { rate };
          controller.applyRate = (media) => {
            media.playbackRate = controller.rate;
          };
          controller.observeAddedMedia = (node) => {
            if (node.nodeType !== Node.ELEMENT_NODE) {
              return;
            }

            if (node.matches("audio, video")) {
              controller.applyRate(node);
            }

            node.querySelectorAll("audio, video").forEach(controller.applyRate);
          };
          globalThis[controllerKey] = controller;

          new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
              mutation.addedNodes.forEach(controller.observeAddedMedia);
            });
          }).observe(document.documentElement, {
            childList: true,
            subtree: true
          });
        }

        if (shouldApplySpeed) {
          controller.rate = rate;
        }

        const media = document.querySelectorAll("audio, video");
        if (isNewController || shouldApplySpeed) {
          media.forEach(controller.applyRate);
        }

        return { count: media.length };
      },
      args: [playbackRate, applySpeed]
    });

    const count = results[0]?.result?.count;
    if (count === undefined) {
      throw new Error("The page did not return its media status.");
    }

    mediaStatus.textContent = count === 0
      ? "No audio or video found on this page yet."
      : `Controlling ${count} media element${count === 1 ? "" : "s"} on this page.`;
    speedStatus.textContent = count === 0
      ? `Selected speed: ${playbackRate}x. It will apply when media is detected.`
      : `Playback speed: ${playbackRate}x`;
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    mediaStatus.textContent = `Can't access media on this page: ${detail}`;
    speedStatus.textContent = "Try a regular webpage and open NeuroTune from its toolbar icon.";
  } finally {
    updateInProgress = false;
    if (applySpeedPending) {
      applySpeedPending = false;
      updatePageMedia(true);
    }
  }
}

speedSelect.addEventListener("change", () => updatePageMedia(true));
updatePageMedia(true);
window.setInterval(() => updatePageMedia(), 1200);
