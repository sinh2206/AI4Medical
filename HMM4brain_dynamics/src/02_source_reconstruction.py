# =============================================================================
# scripts/02_source_reconstruction.py
# Source Reconstruction: RHINO coregistration + LCMV Beamforming + Parcellation
# Chạy trong: venv_osl_ephys
# =============================================================================

import os
import sys
import glob
import numpy as np
import mne
from pathlib import Path

sys.path.insert(0, str(Path("D:\AI y tế").parent.parent))
from configs.pipeline_config import (
    BIDS_ROOT, PREPROC_DIR, SOURCE_DIR, 
    SOURCE_RECON, EXCLUDED_SUBJECTS
)

try:
    import osl
    from osl.source_recon import (
        run_src_recon, 
        find_template_trans,
        rhino,
        beamforming,
        parcellation
    )
    USE_OSL_SRC = True
    print(f"[OK] osl-ephys source recon loaded")
except ImportError:
    USE_OSL_SRC = False
    print("[WARN] osl source_recon không available, dùng MNE trực tiếp")


# =============================================================================
# VIRTUAL FIDUCIALS
# Kỹ thuật khi không có digitized head points
# =============================================================================
def get_virtual_fiducials():
    """
    Trả về fiducials dựa trên landmarks giải phẫu chuẩn (MNI space).
    Dùng khi raw data không có digitized head points.
    """
    fids = SOURCE_RECON["fiducials"]
    nasion = np.array(fids["nasion"]) / 1000.0   # mm -> m
    lpa    = np.array(fids["lpa"])    / 1000.0
    rpa    = np.array(fids["rpa"])    / 1000.0
    return nasion, lpa, rpa


# =============================================================================
# SOURCE RECONSTRUCTION VỚI MNE THUẦN TÚY
# (fallback khi osl không available)
# =============================================================================
def source_recon_mne(subject_id, preproc_file, output_dir):
    """
    Source reconstruction dùng MNE: BEM + LCMV + Parcellation
    
    Workflow:
      1. Load preprocessed raw
      2. Setup forward model (BEM với MNI template)
      3. Compute covariance matrices
      4. LCMV beamforming -> source estimates
      5. Parcellate với dk_cortical atlas (68 ROIs)
      6. Symmetric orthogonalization
    """
    print(f"\n  Source reconstruction: {subject_id}")
    
    output_file = os.path.join(output_dir, f"{subject_id}_source_parc.fif")
    if os.path.exists(output_file):
        print(f"  [SKIP] Đã có: {output_file}")
        return mne.io.read_raw_fif(output_file, preload=True)
    
    # --- 1. Load preprocessed data ---
    print(f"    [1/6] Loading preprocessed data...")
    raw = mne.io.read_raw_fif(preproc_file, preload=True, verbose=False)
    
    # --- 2. Setup info với virtual fiducials ---
    print(f"    [2/6] Coregistration với MNI template...")
    nasion, lpa, rpa = get_virtual_fiducials()
    
    # Tạo digitization points từ virtual fiducials
    dig_points = [
        mne.channels.make_dig_montage(
            nasion=nasion,
            lpa=lpa,
            rpa=rpa,
        )
    ]
    
    # Load MNI fsaverage subject (dùng làm source space)
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = os.path.dirname(fs_dir)
    subject = "fsaverage"
    
    # --- 3. Source space ---
    print(f"    [3/6] Building source space ({SOURCE_RECON['spacing']})...")
    src = mne.setup_source_space(
        subject=subject,
        spacing=SOURCE_RECON["spacing"],
        subjects_dir=subjects_dir,
        verbose=False
    )
    
    # --- 4. Forward model (BEM) ---
    print(f"    [4/6] Computing forward model (BEM)...")
    bem_model = mne.make_bem_model(
        subject=subject,
        subjects_dir=subjects_dir,
        conductivity=SOURCE_RECON["conductivity"],
        verbose=False
    )
    bem_solution = mne.make_bem_solution(bem_model, verbose=False)
    
    # Transformation matrix (identity nếu dùng MNI template trực tiếp)
    trans = "fsaverage"
    
    fwd = mne.make_forward_solution(
        raw.info,
        trans=trans,
        src=src,
        bem=bem_solution,
        meg=False,
        eeg=True,
        verbose=False
    )
    print(f"    [4/6] Forward model: {fwd['nsource']} sources")
    
    # --- 5. LCMV Beamforming ---
    print(f"    [5/6] LCMV Beamforming...")
    
    # Compute data covariance (từ toàn bộ recording)
    data_cov = mne.compute_raw_covariance(
        raw, 
        tmin=0, 
        tmax=None,
        method="empirical",
        verbose=False
    )
    
    # Compute noise covariance (từ baseline hoặc identity)
    noise_cov = mne.make_ad_hoc_cov(raw.info, verbose=False)
    
    # LCMV filters
    filters = mne.beamformer.make_lcmv(
        raw.info,
        fwd,
        data_cov,
        reg=SOURCE_RECON["reg"],
        noise_cov=noise_cov,
        pick_ori="max-power",
        weight_norm="unit-noise-gain",
        rank=None,
        verbose=False
    )
    
    # Apply filters -> source time series
    stc = mne.beamformer.apply_lcmv_raw(raw, filters, verbose=False)
    print(f"    [5/6] Source time series: {stc.data.shape}")
    
    # --- 6. Parcellation -> 68 ROIs ---
    print(f"    [6/6] Parcellating ({SOURCE_RECON['n_parcels']} ROIs)...")
    
    # Load parcellation labels (Desikan-Killiany)
    labels = mne.read_labels_from_annot(
        subject=subject,
        parc="aparc",   # Desikan-Killiany = aparc
        subjects_dir=subjects_dir,
        verbose=False
    )
    
    # Loại bỏ unknown labels
    labels = [l for l in labels if "unknown" not in l.name.lower()]
    print(f"    [6/6] {len(labels)} parcels sau khi lọc")
    
    # Extract parcel time series bằng mean
    parcel_ts = mne.extract_label_time_course(
        stc,
        labels,
        src,
        mode="mean_flip",
        allow_empty=True,
        verbose=False
    )  # Shape: (n_labels, n_times)
    
    # --- Symmetric Orthogonalization ---
    print(f"    [6/6] Symmetric orthogonalization (giảm spatial leakage)...")
    parcel_ts_orth = symmetric_orthogonalization(parcel_ts)
    
    # --- Lưu kết quả ---
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu dưới dạng numpy
    np_output = os.path.join(output_dir, f"{subject_id}_source_parc.npy")
    np.save(np_output, parcel_ts_orth)
    
    # Lưu info
    info_output = os.path.join(output_dir, f"{subject_id}_info.pkl")
    import pickle
    info_dict = {
        "subject_id": subject_id,
        "n_parcels": parcel_ts_orth.shape[0],
        "n_times": parcel_ts_orth.shape[1],
        "sfreq": raw.info["sfreq"],
        "parcel_names": [l.name for l in labels],
        "shape": parcel_ts_orth.shape,
    }
    with open(info_output, "wb") as f:
        pickle.dump(info_dict, f)
    
    print(f"    [SAVED] {np_output}")
    print(f"    Shape: {parcel_ts_orth.shape} "
          f"(parcels × time)")
    
    return parcel_ts_orth


# =============================================================================
# SYMMETRIC ORTHOGONALIZATION
# Loại bỏ spatial leakage giữa các ROIs
# =============================================================================
def symmetric_orthogonalization(X):
    """
    Symmetric orthogonalization của Löwdin.
    Đảm bảo các parcel signals độc lập về mặt không gian.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_parcels, n_times)
    
    Returns
    -------
    X_orth : np.ndarray, shape (n_parcels, n_times)
    """
    # Tính correlation matrix
    X_norm = X / (np.std(X, axis=1, keepdims=True) + 1e-10)
    C = X_norm @ X_norm.T / X.shape[1]
    
    # Löwdin orthogonalization: C^(-1/2)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    eigenvalues = np.maximum(eigenvalues, 1e-10)  # Đảm bảo positive
    W = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T
    
    X_orth = W @ X
    
    return X_orth


# =============================================================================
# SOURCE RECONSTRUCTION VỚI OSL
# =============================================================================
def source_recon_osl(subject_id, preproc_file, output_dir, bids_root):
    """
    Source reconstruction dùng osl-ephys (RHINO pipeline).
    Ưu tiên dùng hàm này nếu osl available.
    """
    print(f"\n  [OSL] Source reconstruction: {subject_id}")
    
    output_file = os.path.join(output_dir, f"{subject_id}_source_parc.npy")
    if os.path.exists(output_file):
        print(f"  [SKIP] Đã có: {output_file}")
        return np.load(output_file)
    
    # OSL config cho source reconstruction
    config = {
        "source_recon": [
            {
                "extract_fiducials_from_fif": {
                    "filepath": None,  # Sẽ dùng virtual fiducials
                    "use_mne_head_pos": True,
                }
            },
            {
                "compute_surfaces": {
                    "include_nose": False,
                    "use_qform": True,
                }
            },
            {
                "coregister": {
                    "use_nose": False,
                    "use_headshape": False,  # Không có headshape -> virtual fiducials
                }
            },
            {
                "forward_model": {
                    "model": "Single Layer",
                }
            },
            {
                "beamform_and_parcellate": {
                    "freq_range": [1, 45],
                    "chantypes": ["eeg"],
                    "rank": {"eeg": 60},
                    "parcellation_file": SOURCE_RECON["atlas"],
                    "method": "spatial_basis",
                    "orthogonalisation": SOURCE_RECON["orthogonalization"],
                }
            }
        ]
    }
    
    # Chạy OSL source reconstruction
    src_dir = os.path.join(bids_root, subject_id, "src")
    os.makedirs(src_dir, exist_ok=True)
    
    try:
        run_src_recon(
            config,
            preproc_file,
            src_dir,
            verbose=True
        )
        
        # Load kết quả parcellated data
        parc_file = os.path.join(src_dir, "rhino", "parc-raw.fif")
        raw_parc = mne.io.read_raw_fif(parc_file, preload=True)
        parcel_ts = raw_parc.get_data()
        
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_file, parcel_ts)
        print(f"  [SAVED] {output_file}, shape: {parcel_ts.shape}")
        
        return parcel_ts
        
    except Exception as e:
        print(f"  [WARN] OSL source recon thất bại: {e}")
        print(f"  [INFO] Chuyển sang MNE fallback...")
        return source_recon_mne(subject_id, preproc_file, output_dir)


# =============================================================================
# VISUALIZE SOURCE RECONSTRUCTION OUTCOMES
# (Power Spectral Density, Power Maps, Connectivity)
# =============================================================================
def visualize_source_outcomes(source_dir, output_dir):
    """
    Tạo các visualization như trong báo cáo:
    - Group-average PSD
    - Power maps theo frequency bands
    - Coherence networks
    - AEC networks
    """
    import matplotlib.pyplot as plt
    
    print("\n  Visualizing source reconstruction outcomes...")
    
    # Load tất cả parcellated time series
    parcel_files = sorted(glob.glob(os.path.join(source_dir, "*_source_parc.npy")))
    
    if not parcel_files:
        print("  [WARN] Không tìm thấy parcellated data")
        return
    
    all_psd = []
    sfreq = 250  # Hz (sau downsampling)
    
    for f in parcel_files:
        data = np.load(f)  # (n_parcels, n_times)
        
        # Tính PSD cho mỗi parcel
        from scipy.signal import welch
        freqs, psd = welch(
            data,
            fs=sfreq,
            nperseg=sfreq * 4,   # 4-second segments
            noverlap=sfreq * 2,  # 50% overlap
            axis=-1
        )
        all_psd.append(np.mean(psd, axis=0))  # Average qua parcels
    
    # Group-average PSD
    group_psd = np.mean(all_psd, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    freq_bands = SOURCE_RECON.get("freq_bands", {
        "delta": (1, 4),
        "theta": (4, 7),
        "alpha": (7, 13),
        "beta":  (13, 30),
    })
    colors = {"delta": "blue", "theta": "orange", "alpha": "green", "beta": "red"}
    
    ax.plot(freqs, group_psd, "k-", linewidth=2)
    
    for band_name, (fmin, fmax) in freq_bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        ax.fill_between(
            freqs[mask], 0, group_psd[mask],
            alpha=0.3,
            color=colors.get(band_name, "gray"),
            label=f"{band_name.capitalize()} ({fmin}-{fmax} Hz)"
        )
    
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (a.u.)")
    ax.set_title("Group-Average Power Spectral Density (Source Space)")
    ax.set_xlim([1, 30])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    psd_file = os.path.join(output_dir, "group_avg_psd.png")
    plt.savefig(psd_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {psd_file}")


# =============================================================================
# MAIN
# =============================================================================
def run_source_reconstruction():
    print("\n" + "="*60)
    print(" BƯỚC 2: SOURCE RECONSTRUCTION PIPELINE")
    print("="*60)
    
    preproc_files = sorted(glob.glob(
        os.path.join(PREPROC_DIR, "sub-*_preproc_raw.fif")
    ))
    
    if not preproc_files:
        print(f"[ERROR] Không tìm thấy preprocessed files trong {PREPROC_DIR}")
        print("  Hãy chạy 01_preprocessing.py trước!")
        return
    
    print(f"  Tìm thấy {len(preproc_files)} preprocessed subjects")
    
    results = {"success": [], "failed": []}
    
    for preproc_file in preproc_files:
        # Trích xuất subject ID
        filename = os.path.basename(preproc_file)
        subject_id = filename.split("_")[0]
        
        if subject_id in EXCLUDED_SUBJECTS:
            print(f"  [SKIP] {subject_id} excluded")
            continue
        
        try:
            if USE_OSL_SRC:
                parcel_ts = source_recon_osl(
                    subject_id, preproc_file, SOURCE_DIR, BIDS_ROOT
                )
            else:
                parcel_ts = source_recon_mne(
                    subject_id, preproc_file, SOURCE_DIR
                )
            
            if parcel_ts is not None:
                results["success"].append(subject_id)
        except Exception as e:
            print(f"  [ERROR] {subject_id}: {e}")
            results["failed"].append(subject_id)
    
    # Visualize outcomes
    vis_dir = os.path.join(SOURCE_DIR, "visualizations")
    visualize_source_outcomes(SOURCE_DIR, vis_dir)
    
    print(f"\n{'='*60}")
    print(f" SOURCE RECONSTRUCTION HOÀN TẤT")
    print(f"{'='*60}")
    print(f"  Thành công: {len(results['success'])} subjects")
    print(f"  Thất bại:   {len(results['failed'])} subjects")
    print(f"  Output:     {SOURCE_DIR}")
    
    return results


if __name__ == "__main__":
    run_source_reconstruction()
