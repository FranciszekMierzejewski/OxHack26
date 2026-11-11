# NeuroTune

An audiobook player that adapts playback speed using a listener-specific EEG attention estimate. Built for Oxford University's Neurotech Hackathon 2026.

The live Python service reads EEG through Lab Streaming Layer (LSL), learns a 30-second baseline, and serves attention and playback-speed state to a Chrome side panel. A simulated source is available for demos without a headset. The attention ratio is a heuristic for relative within-listener changes, not a clinical measure or a trained attention classifier.

## Project layout

- `src/`: live attention scoring, EEG sources, and the local HTTP service.
- `extension/`: Chrome Manifest V3 side panel and local audio player.
- `analysis/`: offline recording inventory, preprocessing, and LSL diagnostics.
- `data/raw/eeg/`: input HDF5 EEG recordings.
- `data/processed/eeg/`: generated manifests, features, and summaries.

## Setup

Use Python 3.10 or later. From the repository root, install the service and analysis dependencies:

```powershell
python -m pip install -r requirements.txt
```

For live EEG, install and launch an LSL-compatible headset stream before starting the service.

## Run

Start the simulated service for a demo:

```powershell
python -m src.server --source sim
```

For a live LSL stream:

```powershell
python -m src.server --source lsl
```

For a recording replay, provide an HDF5 file and, if needed, its condition and part:

```powershell
python -m src.server --source replay --file data/raw/eeg/P00.h5 --condition clean --part part_1
```

The service listens on `http://127.0.0.1:8000`. In Chrome, open `chrome://extensions`, enable Developer mode, choose **Load unpacked**, and select `extension/`. Open the NeuroTune side panel and choose an audio file.

## Offline analysis

Run the inventory and preprocessing tools from any working directory; their data paths are based on the repository location:

```powershell
python analysis/explore_eeg_01.py
python analysis/preprocessing_02.py
```

To inspect available LSL streams and read one sample:

```powershell
python analysis/check_lsl.py
```

The offline preprocessing pipeline is for dataset analysis and is not used by the live attention service.

## Collaborators
- Kiran Kacha
- Georgia Rushton
- Gamz Duman
- Salaar Muhammad
- Franciszek Mierzejewski

## License
 
MIT