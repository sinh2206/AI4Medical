# =============================================================================
# scripts/04_train_tde_hmm.py
# Training Time-Delay Embedded HMM (TDE-HMM)
# Chạy trong: venv_osl_dynamics
# =============================================================================

import os
import sys
import glob
import numpy as np
import pickle
from pathlib import Path

sys.path.insert(0, str(Path("D:\AI y tế").parent.parent))
from configs.pipeline_config import SOURCE_DIR, HMM_DIR, TDE_HMM, EXCLUDED_SUBJECTS

try:
    from osl_dynamics.data import Data
    from osl_dynamics.models.hmm import Config, Model
    from osl_dynamics.analysis import modes, spectral
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
# TIME-DELAY EMBEDDING
# =============================================================================
def time_delay_embedding(X, n_embeddings, center=0):
    """
    Time-Delay Embedding: augment signal với shifted copies.
    
    Với n_embeddings=15 và center=0:
    Lags = [-7, -6, ..., -1, 0, 1, ..., 7] (15 lags)
    
    Parameters
    ----------
    X : np.ndarray, shape (n_times, n_channels)
    n_embeddings : int
        Số lags (phải là số lẻ)
    center : int
        Điểm giữa (thường là 0)
    
    Returns
    -------
    X_tde : np.ndarray, shape (n_valid_times, n_channels * n_embeddings)
    """
    n_times, n_channels = X.shape
    
    # Tính các lags
    half = n_embeddings // 2
    lags = list(range(-half, half + 1))  # -7 to +7
    
    assert len(lags) == n_embeddings, \
        f"n_embeddings={n_embeddings} không khớp với số lags={len(lags)}"
    
    # Trim dữ liệu để tránh boundary effects
    t_start = half
    t_end   = n_times - half
    n_valid = t_end - t_start
    
    # Build embedded matrix
    X_tde = np.zeros((n_valid, n_channels * n_embeddings), dtype=X.dtype)
    
    for i, lag in enumerate(lags):
        start = t_start - lag
        end   = t_end - lag
        X_tde[:, i * n_channels:(i + 1) * n_channels] = X[start:end]
    
    return X_tde


# =============================================================================
# DATA PREPARATION CHO TDE-HMM
# =============================================================================
def prepare_tde_hmm_data(source_files, config):
    """
    Pipeline chuẩn bị dữ liệu cho TDE-HMM:
    1. Time-delay embedding (15 lags = ±7)
    2. PCA (1845 dims -> 80 components)
    3. Per-channel standardization
    
    Parameters
    ----------
    source_files : list of str
    config : dict  (TDE_HMM config)
    
    Returns
    -------
    prepared_data : list of np.ndarray (n_times, n_components)
    pca_model : sklearn PCA
    """
    from sklearn.decomposition import PCA
    
    n_embeddings = config["n_embeddings"]
    
    print(f"\n  [TDE-HMM] Chuẩn bị dữ liệu cho {len(source_files)} subjects...")
    print(f"  Time-delay embedding: {n_embeddings} lags (±{n_embeddings//2})")
    
    all_tde = []
    
    for i, fpath in enumerate(source_files):
        print(f"    [{i+1}/{len(source_files)}] {os.path.basename(fpath)}")
        
        # Load: (n_parcels, n_times)
        data = np.load(fpath).astype(np.float32)
        
        # Transpose: (n_times, n_parcels)
        X = data.T
        n_times, n_channels = X.shape
        
        # --- Bước 1: Time-Delay Embedding ---
        # (n_times, n_channels * n_embeddings)
        X_tde = time_delay_embedding(X, n_embeddings)
        
        all_tde.append(X_tde)
        
        if i == 0:
            print(f"    Original: ({n_times}, {n_channels}) -> "
                  f"TDE: {X_tde.shape} "
                  f"= ({n_times - n_embeddings//2*2}, "
                  f"{n_channels} × {n_embeddings})")
    
    # --- Bước 2: PCA trên concatenated TDE data ---
    expected_dim = all_tde[0].shape[1]
    print(f"\n  [TDE-HMM] PCA: {expected_dim} dims -> "
          f"{config['n_pca_components']} components...")
    
    # Sample để fit PCA (tiết kiệm memory)
    concat_tde = np.vstack(all_tde)
    total_samples = len(concat_tde)
    
    # Giới hạn 500K samples để fit PCA (đủ statistical power)
    max_fit_samples = 500_000
    if total_samples > max_fit_samples:
        idx = np.random.choice(total_samples, max_fit_samples, replace=False)
        fit_data = concat_tde[idx]
    else:
        fit_data = concat_tde
    
    pca = PCA(
        n_components=config["n_pca_components"],
        whiten=config["whiten"],
        random_state=42
    )
    pca.fit(fit_data)
    
    explained_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"  [TDE-HMM] PCA giải thích "
          f"{explained_var[config['n_pca_components']-1]*100:.1f}% variance")
    
    # Transform mỗi subject
    # --- Bước 3: Per-channel standardization (sau PCA) ---
    prepared_data = []
    for tde in all_tde:
        transformed = pca.transform(tde)  # (n_times, n_pca_components)
        
        if config["standardize"]:
            mean = np.mean(transformed, axis=0, keepdims=True)
            std  = np.std(transformed,  axis=0, keepdims=True) + 1e-8
            transformed = (transformed - mean) / std
        
        prepared_data.append(transformed)
    
    # Kiểm tra tính Gaussian (TDE data nên gần Gaussian)
    sample = np.vstack(prepared_data[:3])
    sample_flat = sample.flatten()
    skewness = _compute_skewness(sample_flat)
    kurtosis = _compute_kurtosis(sample_flat)
    print(f"\n  [TDE-HMM] Kiểm tra phân phối (3 subjects đầu):")
    print(f"    Skewness: {skewness:.4f} (Gaussian = 0)")
    print(f"    Excess Kurtosis: {kurtosis:.4f} (Gaussian = 0)")
    
    # Lưu PCA model
    pca_file = os.path.join(HMM_DIR, "tde_hmm_pca.pkl")
    os.makedirs(HMM_DIR, exist_ok=True)
    with open(pca_file, "wb") as f:
        pickle.dump(pca, f)
    print(f"  [SAVED] PCA model: {pca_file}")
    
    return prepared_data, pca


def _compute_skewness(x):
    """Tính skewness thủ công."""
    mean = np.mean(x)
    std  = np.std(x)
    return np.mean(((x - mean) / (std + 1e-10)) ** 3)

def _compute_kurtosis(x):
    """Tính excess kurtosis thủ công."""
    mean = np.mean(x)
    std  = np.std(x)
    return np.mean(((x - mean) / (std + 1e-10)) ** 4) - 3


# =============================================================================
# TRAINING TDE-HMM VỚI OSL-DYNAMICS
# =============================================================================
def train_tde_hmm_osl(prepared_data, config):
    """
    Train TDE-HMM dùng osl-dynamics.
    
    Điểm khác biệt so với AE-HMM:
    - learn_means = False (TDE data is zero-mean)
    - Phân phối data gần Gaussian hơn -> convergence tốt hơn
    """
    print(f"\n  [TDE-HMM] Training với osl-dynamics...")
    print(f"  Config: K={config['n_states']}, "
          f"learn_means={config['learn_means']}, "
          f"epochs={config['n_epochs']}")
    
    osl_data = Data(prepared_data)
    
    hmm_config = Config(
        n_states=config["n_states"],
        n_channels=prepared_data[0].shape[1],
        sequence_length=config["sequence_length"],
        learn_means=config["learn_means"],     # False cho TDE-HMM
        learn_covariances=config["learn_covariances"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        n_epochs=config["n_epochs"],
        covariances_regularization=config["covariance_regularization"],
    )
    
    # Multi-start initialization
    print(f"\n  [TDE-HMM] Multi-start initialization "
          f"(n_init={config['n_init']})...")
    
    best_loss = np.inf
    best_model = None
    
    for init_run in range(config["n_init"]):
        print(f"    Init {init_run+1}/{config['n_init']}...", end=" ", flush=True)
        
        model = Model(hmm_config)
        
        # Khởi tạo ngẫu nhiên
        model.random_state_time_course_initialization(
            osl_data,
            n_epochs=config["n_init_epochs"],
            n_init=1,
            verbose=0
        )
        
        # Quick train trên 25% data
        n_subjects = len(prepared_data)
        n_subset = max(1, n_subjects // 4)
        idx_subset = np.random.choice(n_subjects, n_subset, replace=False)
        subset_data = Data([prepared_data[i] for i in idx_subset])
        
        try:
            history = model.fit(subset_data, verbose=0)
            final_loss = history.history["loss"][-1]
            print(f"loss={final_loss:.4f}")
            
            if final_loss < best_loss:
                best_loss = final_loss
                best_model = model
        except Exception as e:
            print(f"FAILED: {e}")
    
    if best_model is None:
        raise RuntimeError("Tất cả initialization đều thất bại!")
    
    print(f"  [TDE-HMM] Best init loss: {best_loss:.4f}")
    
    # Full training
    print(f"\n  [TDE-HMM] Full training ({config['n_epochs']} epochs)...")
    history = best_model.fit(osl_data, verbose=1)
    
    # Post-training: Kiểm tra state collapse
    alpha = best_model.get_alpha(osl_data)
    state_means = np.mean(np.vstack(alpha), axis=0)
    
    print(f"\n  [TDE-HMM] State occupancy:")
    for k, occ in enumerate(state_means):
        flag = " <-- COLLAPSED" if occ > 0.9 else ""
        print(f"    S{k}: {occ*100:.1f}%{flag}")
    
    # Cảnh báo state collapse
    collapsed = [k for k, occ in enumerate(state_means) if occ > 0.9]
    if collapsed:
        print(f"\n  [WARN] State collapse phát hiện tại states: {collapsed}")
        print(f"  [HINT] Thử tăng n_states hoặc covariance_regularization")
    
    # Lưu model
    model_dir = os.path.join(HMM_DIR, config["model_name"])
    os.makedirs(model_dir, exist_ok=True)
    best_model.save(model_dir)
    
    history_file = os.path.join(model_dir, "training_history.pkl")
    with open(history_file, "wb") as f:
        pickle.dump(history.history, f)
    
    print(f"  [SAVED] TDE-HMM model: {model_dir}")
    
    return best_model, history


# =============================================================================
# TRAINING TDE-HMM VỚI HMMLEARN (fallback)
# =============================================================================
def train_tde_hmm_hmmlearn(prepared_data, config):
    """Fallback training dùng hmmlearn."""
    print(f"\n  [TDE-HMM] Training với hmmlearn (fallback)...")
    
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
    
    model_dir = os.path.join(HMM_DIR, config["model_name"])
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, "model.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(best_model, f)
    
    print(f"  [SAVED] TDE-HMM (hmmlearn): {model_file}")
    return best_model, None


# =============================================================================
# VISUALIZE TRAINING LOSS
# =============================================================================
def plot_training_loss(ae_history_file, tde_history_file, output_dir):
    """Tạo biểu đồ training loss như Figure 6 trong báo cáo."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, history_file, title in [
        (ax1, ae_history_file, "AE-HMM Training Loss"),
        (ax2, tde_history_file, "TDE-HMM Training Loss"),
    ]:
        if os.path.exists(history_file):
            with open(history_file, "rb") as f:
                history = pickle.load(f)
            
            losses = history.get("loss", [])
            epochs = range(1, len(losses) + 1)
            
            ax.plot(epochs, losses, "r.-" if "AE" in title else "b.-", 
                    linewidth=2, markersize=8)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Negative Log Likelihood")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "training_loss.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Training loss plot: {out_file}")


# =============================================================================
# MAIN
# =============================================================================
def run_tde_hmm_training():
    print("\n" + "="*60)
    print(" BƯỚC 4: TRAINING TDE-HMM")
    print("="*60)
    
    # Load source files
    source_files = sorted(glob.glob(
        os.path.join(SOURCE_DIR, "sub-*_source_parc.npy")
    ))
    source_files = [
        f for f in source_files
        if not any(excl in f for excl in EXCLUDED_SUBJECTS)
    ]
    
    if not source_files:
        print(f"[ERROR] Không tìm thấy source data trong {SOURCE_DIR}")
        return
    
    print(f"  Subjects: {len(source_files)}")
    
    # Chuẩn bị data
    prepared_data, pca_model = prepare_tde_hmm_data(source_files, TDE_HMM)
    print(f"\n  Data shape per subject: {prepared_data[0].shape}")
    
    # Train
    os.makedirs(HMM_DIR, exist_ok=True)
    
    if USE_OSL_DYN:
        model, history = train_tde_hmm_osl(prepared_data, TDE_HMM)
    elif USE_HMMLEARN:
        model, history = train_tde_hmm_hmmlearn(prepared_data, TDE_HMM)
    else:
        print("[ERROR] Cần osl-dynamics hoặc hmmlearn!")
        return
    
    # Plot training curves
    ae_history = os.path.join(HMM_DIR, "ae_hmm_k8", "training_history.pkl")
    tde_history = os.path.join(HMM_DIR, "tde_hmm_k8", "training_history.pkl")
    plot_training_loss(ae_history, tde_history, os.path.join(HMM_DIR, "plots"))
    
    print(f"\n{'='*60}")
    print(f" TDE-HMM TRAINING HOÀN TẤT")
    print(f"{'='*60}")
    print(f"  Model: {os.path.join(HMM_DIR, TDE_HMM['tdehmm\model.weights.h5'])}")
    
    return model


if __name__ == "__main__":
    run_tde_hmm_training()
