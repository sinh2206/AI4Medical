# EEG HMM Pipeline — Quick Start and Run Instructions

Purpose: The scripts in the `src/` folder implement an EEG analysis pipeline:
preprocessing → source reconstruction → train AE-HMM / TDE-HMM → extract state time-series and features.

Prerequisites
- Python 3.9–3.11 (3.10 recommended)
- Operating system: Windows (PowerShell examples). Linux/macOS use equivalent shell commands.

Installation (PowerShell)
```powershell
# Create and activate a virtual environment in the workspace
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes: `osl` and `osl-dynamics` may require additional system dependencies (and optionally a specific `torch` build for GPU). If you plan to train with GPU, install a CUDA-compatible `torch` first, then install `osl-dynamics`.

Configuration
- Edit `src/pipeline_config.py` and set `BIDS_ROOT` to the directory containing your BIDS dataset (e.g. `D:/data/nencki_symfonia`).
- Output folders are created automatically by the scripts; other pipeline parameters can be tuned in `pipeline_config.py`.

Main scripts (run from the repository root)
- Preprocessing (EEG preprocessing):

```powershell
python src/01_preprocessing.py
```

- Source reconstruction (parcellated source time-series; uses OSL when available):

```powershell
python src/02_source_reconstruction.py
```

- Train AE-HMM (Amplitude Envelope HMM):

```powershell
python src/03_train_ae_hmm.py
```

- Train TDE-HMM (Time-Delay Embedded HMM):

```powershell
python src/04_train_tde_hmm.py
```

- Extract state time-series and features (fractional occupancy, event-locked dynamics, Viterbi path):

```powershell
python src/05_extract_state_timeseries.py
```

Operational notes
- If `osl` / `osl-dynamics` are not installed, the scripts include fallbacks: preprocessing and source reconstruction fall back to `mne`; training can fall back to `hmmlearn` (reduced capabilities).
- Some steps (PCA, model training) are memory intensive. If you encounter memory issues, reduce values in `src/pipeline_config.py` such as `max_duration_sec` and `n_pca_components`.
- Tweak training parameters (number of states, batch size, epochs, regularization) in `pipeline_config.py`.

Quick install verification
```powershell
python -c "import numpy, mne, scipy, sklearn, pandas; print('ok')"
```
If any import fails, install the missing package with `pip install <package>`.

---
If you want, I can:
- Pin package versions in `requirements.txt`.
- Add an example `run.sh` / `run.ps1` that executes a minimal end-to-end demo (if you have a small test dataset).
Which would you prefer? 
