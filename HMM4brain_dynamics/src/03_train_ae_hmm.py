# =============================================================================
# scripts/03_train_ae_hmm.py
# Training Amplitude Envelope HMM (AE-HMM)
# Chạy trong: venv_osl_dynamics
# =============================================================================

import os
import sys
import glob
import numpy as np
import pickle
from pathlib import Path

sys.path.insert(0, str(Path("D:\AI y tế").parent.parent))
from configs.pipeline_config import SOURCE_DIR, HMM_DIR, AE_HMM, EXCLUDED_SUBJECTS

try:
    from osl_dynamics.data import Data
    from osl_dynamics.models.hmm import Config, Model
    USE_OSL_DYN = True
    print(f"[OK] osl-dynamics loaded")
except ImportError:
    USE_OSL_DYN = False
    print("[WARN] osl-dynamics không available, dùng hmmlearn fallback")

try:
    from hmmlearn.hmm import GaussianHMM
    USE_HMMLEARN = True
except ImportError:
    USE_HMMLEARN = False


# =============================================================================
# DATA PREPARATION CHO AE-HMM
# =============================================================================
def prepare_ae_hmm_data(source_files, config):
    """
    Pipeline chuẩn bị dữ liệu cho AE-HMM:
    1. Bandpass filter (1-45 Hz)
    2. Hilbert transform -> amplitude envelope
    3. Smoothing (moving average)
    4. Per-channel standardization
    5. PCA dimensionality reduction
    
    Parameters
    ----------
    source_files : list of str
        Đường dẫn đến các file .npy (parcellated source data)
    config : dict
        AE_HMM config từ pipeline_config.py
    
    Returns
    -------
    prepared_data : list of np.ndarray
        Danh sách arrays (n_times, n_components) cho mỗi subject
    pca_model : sklearn PCA
    """
    from scipy.signal import butter, filtfilt, hilbert
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    print(f"\n  [AE-HMM] Chuẩn bị dữ liệu cho {len(source_files)} subjects...")
    
    all_envelopes = []
    
    for i, fpath in enumerate(source_files):
        print(f"    [{i+1}/{len(source_files)}] {os.path.basename(fpath)}")
        
        # Load parcellated data: (n_parcels, n_times)
        data = np.load(fpath).astype(np.float32)
        n_parcels, n_times = data.shape
        
        # --- Bước 1: Bandpass filter 1-45 Hz ---
        sfreq = 250  # Hz
        b, a = butter(
            4,
            [config["bandpass"][0] / (sfreq / 2), 
             config["bandpass"][1] / (sfreq / 2)],
            btype="band"
        )
        data_filtered = filtfilt(b, a, data, axis=1)
        
        # --- Bước 2: Hilbert transform -> amplitude envelope ---
        analytic_signal = hilbert(data_filtered, axis=1)
        envelope = np.abs(analytic_signal)  # (n_parcels, n_times)
        
        # --- Bước 3: Smoothing với moving average ---
        window = config["smoothing_window"]
        kernel = np.ones(window) / window
        envelope_smooth = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"),
            axis=1,
            arr=envelope
        )
        
        # --- Bước 4: Per-channel standardization ---
        if config["standardize"]:
            mean = np.mean(envelope_smooth, axis=1, keepdims=True)
            std  = np.std(envelope_smooth,  axis=1, keepdims=True) + 1e-8
            envelope_std = (envelope_smooth - mean) / std
        else:
            envelope_std = envelope_smooth
        
        # Transpose: (n_times, n_parcels) cho PCA
        all_envelopes.append(envelope_std.T)
    
    # --- Bước 5: PCA qua tất cả subjects ---
    print(f"\n  [AE-HMM] PCA: {all_envelopes[0].shape[1]} channels -> "
          f"{config['n_pca_components']} components")
    
    # Stack tất cả data để fit PCA
    # (Để tránh memory overflow, fit trên subset nếu cần)
    concat_data = np.vstack(all_envelopes)  # (total_times, n_parcels)
    
    pca = PCA(
        n_components=config["n_pca_components"],
        whiten=config["whiten"],
        random_state=42
    )
    
    # Fit PCA trên toàn bộ data
    try:
        pca.fit(concat_data)
    except MemoryError:
        print("  [WARN] Memory overflow, fit PCA trên 10% data...")
        n_samples = len(concat_data)
        idx = np.random.choice(n_samples, n_samples // 10, replace=False)
        pca.fit(concat_data[idx])
    
    explained_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"  [AE-HMM] PCA giải thích {explained_var[config['n_pca_components']-1]*100:.1f}% variance")
    
    # Transform mỗi subject
    prepared_data = [pca.transform(env) for env in all_envelopes]
    
    # Lưu PCA model
    pca_file = os.path.join(HMM_DIR, "ae_hmm_pca.pkl")
    os.makedirs(HMM_DIR, exist_ok=True)
    with open(pca_file, "wb") as f:
        pickle.dump(pca, f)
    print(f"  [SAVED] PCA model: {pca_file}")
    
    return prepared_data, pca


# =============================================================================
# TRAINING AE-HMM VỚI OSL-DYNAMICS
# =============================================================================
def train_ae_hmm_osl(prepared_data, config):
    """
    Train AE-HMM dùng osl-dynamics với Variational Bayes optimization.
    """
    print(f"\n  [AE-HMM] Training với osl-dynamics...")
    print(f"  Config: K={config['n_states']}, "
          f"seq_len={config['sequence_length']}, "
          f"batch={config['batch_size']}, "
          f"epochs={config['n_epochs']}")
    
    # Tạo osl-dynamics Data object
    osl_data = Data(prepared_data)
    
    # Config model
    hmm_config = Config(
        n_states=config["n_states"],
        n_channels=prepared_data[0].shape[1],
        sequence_length=config["sequence_length"],
        learn_means=config["learn_means"],
        learn_covariances=config["learn_covariances"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        n_epochs=config["n_epochs"],
        covariances_regularization=config["covariance_regularization"],
    )
    
    # Multi-start initialization
    print(f"\n  [AE-HMM] Multi-start initialization "
          f"(n_init={config['n_init']})...")
    
    best_loss = np.inf
    best_model = None
    
    for init_run in range(config["n_init"]):
        print(f"    Init {init_run+1}/{config['n_init']}...", end=" ", flush=True)
        
        model = Model(hmm_config)
        model.random_state_time_course_initialization(
            osl_data,
            n_epochs=config["n_init_epochs"],
            n_init=1,
            verbose=0
        )
        
        # Nhanh train 5 epochs trên 25% data
        n_samples = len(prepared_data)
        subset_idx = np.random.choice(n_samples, max(1, n_samples // 4), replace=False)
        subset_data = Data([prepared_data[i] for i in subset_idx])
        
        history = model.fit(subset_data, verbose=0)
        final_loss = history.history["loss"][-1]
        print(f"loss={final_loss:.3f}")
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_model = model
    
    print(f"  [AE-HMM] Best init loss: {best_loss:.3f}")
    
    # Full training với best initialization
    print(f"\n  [AE-HMM] Full training ({config['n_epochs']} epochs)...")
    history = best_model.fit(osl_data, verbose=1)
    
    # Lưu model
    model_dir = os.path.join(HMM_DIR, config["model_name"])
    os.makedirs(model_dir, exist_ok=True)
    best_model.save(model_dir)
    
    # Lưu training history
    history_file = os.path.join(model_dir, "training_history.pkl")
    with open(history_file, "wb") as f:
        pickle.dump(history.history, f)
    
    print(f"  [SAVED] AE-HMM model: {model_dir}")
    
    return best_model, history


# =============================================================================
# TRAINING AE-HMM VỚI HMMLEARN (fallback)
# =============================================================================
def train_ae_hmm_hmmlearn(prepared_data, config):
    """
    Train AE-HMM dùng hmmlearn (fallback khi không có osl-dynamics).
    """
    print(f"\n  [AE-HMM] Training với hmmlearn (fallback)...")
    
    # Concatenate tất cả subjects
    X = np.vstack(prepared_data)
    lengths = [d.shape[0] for d in prepared_data]
    
    best_model = None
    best_score = -np.inf
    
    for init_run in range(config["n_init"]):
        print(f"    Init {init_run+1}/{config['n_init']}...", end=" ", flush=True)
        
        model = GaussianHMM(
            n_components=config["n_states"],
            covariance_type="full",
            n_iter=config["n_epochs"],
            random_state=init_run,
            verbose=False
        )
        
        try:
            model.fit(X, lengths)
            score = model.score(X, lengths)
            print(f"score={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_model = model
        except Exception as e:
            print(f"FAILED: {e}")
    
    # Lưu model
    model_dir = os.path.join(HMM_DIR, config["model_name"])
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, "aehmm\model.weights.h5")
    with open(model_file, "wb") as f:
        pickle.dump(best_model, f)
    
    print(f"  [SAVED] AE-HMM (hmmlearn): {model_file}")
    
    return best_model, None


# =============================================================================
# MAIN
# =============================================================================
def run_ae_hmm_training():
    print("\n" + "="*60)
    print(" BƯỚC 3: TRAINING AE-HMM")
    print("="*60)
    
    # Load source files
    source_files = sorted(glob.glob(
        os.path.join(SOURCE_DIR, "sub-*_source_parc.npy")
    ))
    
    # Loại excluded subjects
    source_files = [
        f for f in source_files
        if not any(excl in f for excl in EXCLUDED_SUBJECTS)
    ]
    
    if not source_files:
        print(f"[ERROR] Không tìm thấy source data trong {SOURCE_DIR}")
        print("  Hãy chạy 02_source_reconstruction.py trước!")
        return
    
    print(f"  Subjects: {len(source_files)}")
    
    # Kiểm tra xem model đã tồn tại chưa
    model_dir = os.path.join(HMM_DIR, AE_HMM["aehmm\model.weights.h5"])
    if os.path.exists(model_dir):
        print(f"  [INFO] Model đã tồn tại: {model_dir}")
        response = input("  Retrain? (y/n): ").strip().lower()
        if response != "y":
            print("  [SKIP] Bỏ qua training")
            return
    
    # Chuẩn bị data
    prepared_data, pca_model = prepare_ae_hmm_data(source_files, AE_HMM)
    
    print(f"\n  Data shape per subject: {prepared_data[0].shape}")
    print(f"  Total time points: {sum(d.shape[0] for d in prepared_data)}")
    
    # Train model
    os.makedirs(HMM_DIR, exist_ok=True)
    
    if USE_OSL_DYN:
        model, history = train_ae_hmm_osl(prepared_data, AE_HMM)
    elif USE_HMMLEARN:
        model, history = train_ae_hmm_hmmlearn(prepared_data, AE_HMM)
    else:
        print("[ERROR] Cần osl-dynamics hoặc hmmlearn!")
        print("  pip install osl-dynamics  hoặc  pip install hmmlearn")
        return
    
    print(f"\n{'='*60}")
    print(f" AE-HMM TRAINING HOÀN TẤT")
    print(f"{'='*60}")
    print(f"  Model: {os.path.join(HMM_DIR, AE_HMM['aehmm\model.weights.h5'])}")
    
    return model


if __name__ == "__main__":
    run_ae_hmm_training()
