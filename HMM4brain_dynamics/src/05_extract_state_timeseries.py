# =============================================================================
# scripts/05_extract_state_timeseries.py
# Trích xuất State Time Series (alpha) từ cả AE-HMM và TDE-HMM
# Tính fractional occupancy, event-locked dynamics, Viterbi path
# Chạy trong: venv_osl_dynamics
# =============================================================================

import os
import sys
import glob
import numpy as np
import pickle
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.pipeline_config import (
    SOURCE_DIR, HMM_DIR, STATE_TS_DIR,
    AE_HMM, TDE_HMM, FEATURE_EXTRACTION, 
    EXCLUDED_SUBJECTS
)

try:
    from osl_dynamics.data import Data
    from osl_dynamics.models.hmm import Model
    USE_OSL_DYN = True
except ImportError:
    USE_OSL_DYN = False

# Scripts từ training
sys.path.insert(0, str(Path(__file__).parent))
from scripts_03_train_ae_hmm import prepare_ae_hmm_data
from scripts_04_train_tde_hmm import prepare_tde_hmm_data

# Import tên module đúng
import importlib
preproc_ae  = importlib.import_module("03_train_ae_hmm") if False else None  # handled below
preproc_tde = importlib.import_module("04_train_tde_hmm") if False else None


# =============================================================================
# LOAD HMM MODEL
# =============================================================================
def load_hmm_model(model_name, n_states, n_channels):
    """Load trained HMM model từ disk."""
    model_dir = os.path.join(HMM_DIR, model_name)
    
    if USE_OSL_DYN:
        try:
            from osl_dynamics.models.hmm import Config, Model as HMMModel
            
            # Đọc config từ model dir
            config_file = os.path.join(model_dir, "config.pkl")
            if os.path.exists(config_file):
                with open(config_file, "rb") as f:
                    config = pickle.load(f)
                model = HMMModel(config)
            else:
                # Reconstruct config
                model_config = Config(
                    n_states=n_states,
                    n_channels=n_channels,
                    sequence_length=1000,
                    learn_means=("ae" in model_name),
                    learn_covariances=True,
                )
                model = HMMModel(model_config)
            
            model.load_weights(model_dir)
            print(f"  [OK] Loaded osl-dynamics model: {model_name}")
            return model
        except Exception as e:
            print(f"  [WARN] osl-dynamics load failed: {e}")
    
    # Fallback: hmmlearn
    model_file = os.path.join(model_dir, "model.pkl")
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        print(f"  [OK] Loaded hmmlearn model: {model_name}")
        return model
    
    raise FileNotFoundError(f"Không tìm thấy model: {model_dir}")


# =============================================================================
# TÍNH STATE PROBABILITIES (ALPHA)
# alpha[t, k] = P(state=k tại thời điểm t | data)
# =============================================================================
def compute_state_probabilities(model, prepared_data, model_type="osl"):
    """
    Tính soft state probabilities cho mỗi subject.
    
    Parameters
    ----------
    model : HMM model
    prepared_data : list of np.ndarray (n_times, n_channels)
    model_type : "osl" hoặc "hmmlearn"
    
    Returns
    -------
    alphas : list of np.ndarray (n_times, n_states)
    """
    alphas = []
    
    if model_type == "osl" and USE_OSL_DYN:
        osl_data = Data(prepared_data)
        alphas = model.get_alpha(osl_data)
        
    else:
        # hmmlearn: dùng predict_proba
        for data in prepared_data:
            try:
                _, posteriors = model.score_samples(data)
                alphas.append(posteriors)
            except Exception as e:
                # Fallback: forward-backward manually
                print(f"    [WARN] predict_proba failed: {e}")
                # Dùng Viterbi hard assignment
                states = model.predict(data)
                n_states = model.n_components
                alpha = np.zeros((len(data), n_states))
                for t, s in enumerate(states):
                    alpha[t, s] = 1.0
                alphas.append(alpha)
    
    return alphas


# =============================================================================
# VITERBI PATH (HARD STATE ASSIGNMENT)
# =============================================================================
def compute_viterbi_path(model, prepared_data, model_type="osl"):
    """
    Tính hard state assignments dùng Viterbi algorithm.
    
    Returns
    -------
    viterbi_paths : list of np.ndarray (n_times,) với giá trị 0..K-1
    """
    viterbi_paths = []
    
    if model_type == "osl" and USE_OSL_DYN:
        osl_data = Data(prepared_data)
        alphas = model.get_alpha(osl_data)
        viterbi_paths = [np.argmax(a, axis=1) for a in alphas]
    else:
        for data in prepared_data:
            try:
                path = model.predict(data)
                viterbi_paths.append(path)
            except Exception as e:
                print(f"    [WARN] Viterbi failed: {e}")
                viterbi_paths.append(np.zeros(len(data), dtype=int))
    
    return viterbi_paths


# =============================================================================
# FRACTIONAL OCCUPANCY
# =============================================================================
def compute_fractional_occupancy(alphas):
    """
    Tính fractional occupancy (%) cho mỗi subject và mỗi state.
    
    Returns
    -------
    fo_matrix : np.ndarray (n_subjects, n_states)
    """
    fo_list = []
    for alpha in alphas:
        # Mean probability của mỗi state qua thời gian
        fo = np.mean(alpha, axis=0)  # (n_states,)
        fo_list.append(fo)
    
    fo_matrix = np.array(fo_list)  # (n_subjects, n_states)
    return fo_matrix


# =============================================================================
# EVENT-LOCKED DYNAMICS
# Tính average state probabilities locked to stimulus onset
# =============================================================================
def compute_event_locked_dynamics(
    alphas, 
    event_files,
    subject_ids,
    sfreq=250,
    tmin=-0.2, 
    tmax=0.8
):
    """
    Tính average state probability trajectories locked to stimulus onset.
    
    Parameters
    ----------
    alphas : list of np.ndarray (n_times, n_states)
    event_files : list of str  (đường dẫn file events .csv hoặc .tsv)
    subject_ids : list of str
    sfreq : float (Hz)
    tmin, tmax : float (seconds, relative to stimulus onset)
    
    Returns
    -------
    event_locked : dict với keys "standard", "target"
        Mỗi key là np.ndarray (n_times_epoch, n_states)
    """
    from configs.pipeline_config import FEATURE_EXTRACTION
    
    event_ids = FEATURE_EXTRACTION["event_ids"]
    
    n_pre  = int(abs(tmin) * sfreq)   # số samples trước onset
    n_post = int(tmax * sfreq)         # số samples sau onset
    n_epoch = n_pre + n_post
    
    epochs_per_condition = {
        "standard": [],
        "target": [],
    }
    
    for subj_idx, (alpha, event_file, subj_id) in enumerate(
        zip(alphas, event_files, subject_ids)
    ):
        if not os.path.exists(event_file):
            print(f"    [WARN] Event file không tìm thấy: {event_file}")
            continue
        
        # Load events
        try:
            events_df = pd.read_csv(event_file, sep="\t")
        except Exception:
            try:
                events_df = pd.read_csv(event_file)
            except Exception as e:
                print(f"    [WARN] Không đọc được event file: {e}")
                continue
        
        # Tìm columns onset và trial_type
        onset_col = next((c for c in events_df.columns 
                          if "onset" in c.lower()), None)
        type_col  = next((c for c in events_df.columns 
                          if "trial_type" in c.lower() or 
                             "stim_type" in c.lower() or
                             "value" in c.lower()), None)
        
        if onset_col is None or type_col is None:
            print(f"    [WARN] Không tìm thấy onset/trial_type columns: "
                  f"{events_df.columns.tolist()}")
            continue
        
        # Extract epochs
        for condition, event_code in [
            ("standard", event_ids["standard"]),
            ("target",   event_ids["target"]),
        ]:
            mask = events_df[type_col].astype(str) == str(event_code)
            onset_times = events_df.loc[mask, onset_col].values
            
            for onset_sec in onset_times:
                # Tính sample index (offset vì Auto-Trimmer cắt 60s đầu)
                onset_sample = int((onset_sec - 60) * sfreq)
                
                t_start = onset_sample - n_pre
                t_end   = onset_sample + n_post
                
                # Kiểm tra boundary
                if t_start < 0 or t_end > len(alpha):
                    continue
                
                epoch = alpha[t_start:t_end]  # (n_epoch, n_states)
                
                if epoch.shape[0] == n_epoch:
                    epochs_per_condition[condition].append(epoch)
    
    # Average qua tất cả epochs
    result = {}
    times = np.linspace(tmin, tmax, n_epoch)
    
    for condition, epoch_list in epochs_per_condition.items():
        if epoch_list:
            stacked = np.stack(epoch_list, axis=0)  # (n_epochs, n_times, n_states)
            result[condition] = {
                "mean": np.mean(stacked, axis=0),  # (n_times, n_states)
                "sem":  np.std(stacked, axis=0) / np.sqrt(len(epoch_list)),
                "n_epochs": len(epoch_list),
            }
            print(f"    {condition}: {len(epoch_list)} epochs")
        else:
            print(f"    [WARN] Không có epochs cho {condition}")
    
    result["times"] = times
    
    return result


# =============================================================================
# EXTRACT FEATURES CHO CLASSIFICATION
# =============================================================================
def extract_classification_features(
    alphas, 
    event_files,
    subject_ids,
    sfreq=250,
):
    """
    Trích xuất 32 features (4 windows × 8 states) cho mỗi trial.
    
    Returns
    -------
    features_df : pd.DataFrame
        Columns: subject_id, trial_idx, condition, feat_W0_S0, ..., feat_W3_S7
    """
    from configs.pipeline_config import FEATURE_EXTRACTION
    
    event_ids = FEATURE_EXTRACTION["event_ids"]
    time_windows = FEATURE_EXTRACTION["time_windows"]
    
    all_rows = []
    
    for alpha, event_file, subj_id in zip(alphas, event_files, subject_ids):
        if not os.path.exists(event_file):
            continue
        
        try:
            events_df = pd.read_csv(event_file, sep="\t")
        except Exception:
            events_df = pd.read_csv(event_file)
        
        onset_col = next((c for c in events_df.columns if "onset" in c.lower()), None)
        type_col  = next((c for c in events_df.columns 
                          if any(k in c.lower() for k in 
                                 ["trial_type", "stim_type", "value"])), None)
        
        if onset_col is None or type_col is None:
            continue
        
        tmin = FEATURE_EXTRACTION["tmin"]  # -0.2s
        tmax = FEATURE_EXTRACTION["tmax"]  # +0.8s
        
        for _, row in events_df.iterrows():
            event_code = str(row[type_col])
            
            # Chỉ lấy standard và target
            if event_code == str(event_ids["standard"]):
                condition = 0  # Standard
            elif event_code == str(event_ids["target"]):
                condition = 1  # Target
            else:
                continue  # Bỏ qua distractor
            
            onset_sec = float(row[onset_col])
            onset_sample = int((onset_sec - 60) * sfreq)
            
            # Trích xuất features cho mỗi time window
            feat_row = {
                "subject_id": subj_id,
                "condition": condition,
                "onset_sec": onset_sec,
            }
            
            for win_name, (win_start, win_end) in time_windows.items():
                t_start = onset_sample + int(win_start * sfreq)
                t_end   = onset_sample + int(win_end   * sfreq)
                
                if t_start < 0 or t_end > len(alpha):
                    # Epoch ngoài boundary -> skip toàn bộ trial
                    feat_row = None
                    break
                
                # Mean state probabilities trong window
                window_alpha = alpha[t_start:t_end]  # (n_samples, n_states)
                mean_probs = np.mean(window_alpha, axis=0)  # (n_states,)
                
                for state_idx, prob in enumerate(mean_probs):
                    feat_row[f"feat_{win_name}_S{state_idx}"] = prob
            
            if feat_row is not None:
                all_rows.append(feat_row)
    
    features_df = pd.DataFrame(all_rows)
    return features_df


# =============================================================================
# MAIN
# =============================================================================
def run_state_timeseries_extraction():
    print("\n" + "="*60)
    print(" BƯỚC 5: EXTRACT STATE TIME SERIES")
    print("="*60)
    
    os.makedirs(STATE_TS_DIR, exist_ok=True)
    
    # Load source files
    source_files = sorted(glob.glob(
        os.path.join(SOURCE_DIR, "sub-*_source_parc.npy")
    ))
    source_files = [
        f for f in source_files
        if not any(excl in f for excl in EXCLUDED_SUBJECTS)
    ]
    
    subject_ids = [
        os.path.basename(f).split("_")[0] for f in source_files
    ]
    
    print(f"  Subjects: {len(source_files)}")
    
    # Load event files (BIDS format)
    from configs.pipeline_config import BIDS_ROOT
    event_files = []
    for subj_id in subject_ids:
        # Nencki: sub-XX/eeg/sub-XX_task-oddball_events.tsv
        ev_file = glob.glob(os.path.join(
            BIDS_ROOT, subj_id, "eeg", f"*task-oddball*events.tsv"
        ))
        if ev_file:
            event_files.append(ev_file[0])
        else:
            event_files.append("")
    
    sfreq = 250  # Hz sau downsampling
    
    # ---------------------------------------------------------------
    # Xử lý cả 2 models
    # ---------------------------------------------------------------
    for model_tag, model_config, prep_fn in [
        ("ae_hmm",  AE_HMM,  "ae"),
        ("tde_hmm", TDE_HMM, "tde"),
    ]:
        print(f"\n  {'='*50}")
        print(f"  Model: {model_tag.upper()}")
        print(f"  {'='*50}")
        
        # Chuẩn bị data (phải dùng cùng pipeline như lúc training)
        print(f"  Chuẩn bị dữ liệu...")
        
        # Load PCA model
        pca_file = os.path.join(HMM_DIR, f"{model_tag}_pca.pkl")
        if not os.path.exists(pca_file):
            print(f"  [ERROR] Không tìm thấy PCA: {pca_file}")
            print(f"  Hãy chạy training script trước!")
            continue
        
        with open(pca_file, "rb") as f:
            pca = pickle.load(f)
        
        n_components = pca.n_components_
        
        # Tái tạo prepared data
        if prep_fn == "ae":
            # Import hàm từ training script
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "train_ae", os.path.join(os.path.dirname(__file__), "03_train_ae_hmm.py")
            )
            train_ae_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_ae_mod)
            prepared_data, _ = train_ae_mod.prepare_ae_hmm_data(source_files, model_config)
        else:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "train_tde", os.path.join(os.path.dirname(__file__), "04_train_tde_hmm.py")
            )
            train_tde_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_tde_mod)
            prepared_data, _ = train_tde_mod.prepare_tde_hmm_data(source_files, model_config)
        
        # Load model
        try:
            model = load_hmm_model(
                model_config["model_name"],
                n_states=model_config["n_states"],
                n_channels=n_components
            )
        except Exception as e:
            print(f"  [ERROR] Không load được model: {e}")
            continue
        
        model_type = "osl" if USE_OSL_DYN else "hmmlearn"
        
        # --- Tính state probabilities ---
        print(f"  Tính state probabilities (alpha)...")
        alphas = compute_state_probabilities(model, prepared_data, model_type)
        
        # Lưu alphas
        alpha_file = os.path.join(STATE_TS_DIR, f"{model_tag}_alphas.pkl")
        with open(alpha_file, "wb") as f:
            pickle.dump({
                "alphas": alphas,
                "subject_ids": subject_ids,
                "n_states": model_config["n_states"],
                "sfreq": sfreq,
            }, f)
        print(f"  [SAVED] {alpha_file}")
        
        # --- Fractional Occupancy ---
        print(f"  Tính fractional occupancy...")
        fo_matrix = compute_fractional_occupancy(alphas)
        
        fo_file = os.path.join(STATE_TS_DIR, f"{model_tag}_fractional_occupancy.npy")
        np.save(fo_file, fo_matrix)
        
        # Print summary
        fo_mean = np.mean(fo_matrix, axis=0)
        fo_std  = np.std(fo_matrix,  axis=0)
        print(f"  Fractional Occupancy (group mean ± std):")
        for k in range(model_config["n_states"]):
            flag = " <-- COLLAPSED" if fo_mean[k] > 0.7 else ""
            print(f"    S{k}: {fo_mean[k]*100:.1f} ± {fo_std[k]*100:.1f}%{flag}")
        
        # --- Viterbi path ---
        print(f"  Tính Viterbi paths...")
        viterbi_paths = compute_viterbi_path(model, prepared_data, model_type)
        
        viterbi_file = os.path.join(STATE_TS_DIR, f"{model_tag}_viterbi.pkl")
        with open(viterbi_file, "wb") as f:
            pickle.dump(viterbi_paths, f)
        print(f"  [SAVED] {viterbi_file}")
        
        # --- Event-locked dynamics ---
        valid_events = [f for f in event_files if f]
        if valid_events:
            print(f"  Tính event-locked dynamics...")
            event_locked = compute_event_locked_dynamics(
                alphas, event_files, subject_ids, sfreq=sfreq
            )
            
            el_file = os.path.join(STATE_TS_DIR, f"{model_tag}_event_locked.pkl")
            with open(el_file, "wb") as f:
                pickle.dump(event_locked, f)
            print(f"  [SAVED] {el_file}")
        
        # --- Classification features ---
        if valid_events:
            print(f"  Trích xuất classification features...")
            features_df = extract_classification_features(
                alphas, event_files, subject_ids, sfreq=sfreq
            )
            
            feat_file = os.path.join(STATE_TS_DIR, f"{model_tag}_features.csv")
            features_df.to_csv(feat_file, index=False)
            print(f"  [SAVED] {feat_file} ({len(features_df)} trials, "
                  f"{len(features_df.columns)-3} features)")
    
    # ---------------------------------------------------------------
    # Visualize Fractional Occupancy
    # ---------------------------------------------------------------
    _visualize_fractional_occupancy(STATE_TS_DIR)
    
    print(f"\n{'='*60}")
    print(f" STATE TIME SERIES EXTRACTION HOÀN TẤT")
    print(f"  Output: {STATE_TS_DIR}")
    print(f"{'='*60}")


def _visualize_fractional_occupancy(state_ts_dir):
    """Tạo biểu đồ fractional occupancy như Figure 7."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, model_tag, title in [
        (ax1, "ae_hmm",  "AE-HMM"),
        (ax2, "tde_hmm", "TDE-HMM"),
    ]:
        fo_file = os.path.join(state_ts_dir, f"{model_tag}_fractional_occupancy.npy")
        if not os.path.exists(fo_file):
            continue
        
        fo = np.load(fo_file)  # (n_subjects, n_states)
        n_states = fo.shape[1]
        
        fo_mean = np.mean(fo, axis=0) * 100
        fo_std  = np.std(fo,  axis=0) * 100
        
        x = np.arange(n_states)
        bars = ax.bar(x, fo_mean, yerr=fo_std, capsize=4,
                      color="steelblue", alpha=0.8, edgecolor="black")
        
        # Uniform reference line
        ax.axhline(100 / n_states, color="red", linestyle="--", 
                   linewidth=1.5, label=f"Uniform ({100/n_states:.1f}%)")
        
        ax.set_xlabel("State", fontsize=12)
        ax.set_ylabel("Fractional Occupancy (%)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{i}" for i in range(n_states)])
        ax.legend()
        ax.set_ylim(0, max(fo_mean) * 1.3)
        ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    out_file = os.path.join(state_ts_dir, "fractional_occupancy.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_file}")


if __name__ == "__main__":
    run_state_timeseries_extraction()
