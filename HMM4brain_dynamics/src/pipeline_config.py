# =============================================================================
# configs/pipeline_config.py
# Cấu hình tập trung cho toàn bộ pipeline
# =============================================================================

import os

# ---------------------------------------------------------------
# ĐƯỜNG DẪN DỮ LIỆU
# ---------------------------------------------------------------
# Thư mục gốc của dataset Nencki-Symfonia
BIDS_ROOT = "/data/nencki_symfonia"

# Thư mục lưu kết quả preprocessing
PREPROC_DIR = "./outputs/01_preprocessed"

# Thư mục lưu kết quả source reconstruction
SOURCE_DIR = "./outputs/02_source_reconstructed"

# Thư mục lưu HMM models
HMM_DIR = "./outputs/03_hmm_models"

# Thư mục lưu state timeseries
STATE_TS_DIR = "./outputs/04_state_timeseries"

# Thư mục lưu connectivity matrices
CONNECTIVITY_DIR = "./outputs/05_connectivity"

# ---------------------------------------------------------------
# CẤU HÌNH SUBJECTS
# ---------------------------------------------------------------
# Danh sách subjects (bài báo dùng 30 subjects, loại 4)
N_SUBJECTS = 30
EXCLUDED_SUBJECTS = ["sub-01", "sub-22"]  # Loại trước HMM (data quality)
# sub-25 và sub-28 loại SAU training (state collapse TDE-HMM)

# ---------------------------------------------------------------
# CẤU HÌNH PREPROCESSING
# ---------------------------------------------------------------
PREPROC = {
    # Bandpass filter
    "l_freq": 0.5,          # Hz - loại bỏ slow drift
    "h_freq": 45.0,         # Hz - loại bỏ high-freq noise
    "notch_freq": 50.0,     # Hz - loại bỏ line noise (EU: 50Hz)
    
    # Downsampling
    "sfreq_original": 1000, # Hz - native sampling rate
    "sfreq_target": 250,    # Hz - sau downsampling (đủ cho 45Hz)
    
    # Auto-Trimmer (giải quyết RAM constraint)
    "trim_start_sec": 60,   # Cắt 1 phút đầu (artifacts chuẩn bị)
    "max_duration_sec": 900,# Giới hạn 15 phút / subject
    
    # ICA
    "n_ica_components": 64, # Số ICA components
    "ica_method": "fastica",
    "eog_channel": "EOG",   # Kênh EOG để auto-detect eye artifacts
    "ecg_channel": "ECG",   # Kênh ECG để auto-detect cardiac artifacts
    
    # Bad channel interpolation
    "interpolation_method": "spline",
    
    # EEG reference
    "reference": "average", # Re-reference sang average reference
    
    # Số kênh
    "n_channels_original": 128,
}

# ---------------------------------------------------------------
# CẤU HÌNH SOURCE RECONSTRUCTION
# ---------------------------------------------------------------
SOURCE_RECON = {
    # MRI template (không có digitized points -> dùng MNI)
    "mri_template": "MNI152_T1_2mm",
    
    # Fiducials ảo (Virtual Fiducials technique)
    "use_virtual_fiducials": True,
    "fiducials": {
        "nasion": [0.0, 85.0, -40.0],   # MNI coordinates (mm)
        "lpa":    [-83.0, -20.0, -35.0],
        "rpa":    [83.0, -20.0, -35.0],
    },
    
    # Beamforming
    "method": "lcmv",       # Linearly Constrained Minimum Variance
    "reg": 0.05,            # Regularization parameter
    
    # Parcellation
    "atlas": "dk_cortical", # Desikan-Killiany atlas
    "n_parcels": 68,        # 68 Regions of Interest
    
    # Orthogonalization (giảm spatial leakage)
    "orthogonalization": "symmetric",
    
    # Forward model
    "conductivity": (0.3, 0.006, 0.3),  # EEG 3-shell model
    "spacing": "oct6",      # Source grid
}

# ---------------------------------------------------------------
# CẤU HÌNH AE-HMM
# ---------------------------------------------------------------
AE_HMM = {
    # Số hidden states
    "n_states": 8,
    
    # Dữ liệu đầu vào
    "bandpass": (1.0, 45.0),    # Hz
    "hilbert_envelope": True,    # Lấy amplitude envelope
    "smoothing_window": 5,       # Samples (= 20ms tại 250Hz)
    
    # PCA dimensionality reduction
    "use_pca": True,
    "n_pca_components": 30,      # 123 channels -> 30 components
    "whiten": True,
    
    # Per-channel standardization
    "standardize": True,
    
    # Training
    "learn_means": True,         # AE-HMM cần learn means
    "learn_covariances": True,
    "sequence_length": 1000,     # 4 giây tại 250Hz
    "batch_size": 16,
    "n_epochs": 20,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "covariance_regularization": 1e-6,
    
    # Multi-start initialization
    "n_init": 10,
    "n_init_epochs": 5,
    "init_frac": 0.25,           # 25% dữ liệu cho initialization
    
    # Output
    "model_name": "ae_hmm_k8",
}

# ---------------------------------------------------------------
# CẤU HÌNH TDE-HMM
# ---------------------------------------------------------------
TDE_HMM = {
    # Số hidden states
    "n_states": 8,
    
    # Time-Delay Embedding
    "n_embeddings": 15,          # 15 lags = ±7 samples
    "embeddings_center": 0,
    
    # PCA dimensionality reduction
    "use_pca": True,
    "n_pca_components": 80,      # 123*15=1845 -> 80 components
    "whiten": True,
    
    # Per-channel standardization (sau PCA)
    "standardize": True,
    
    # Training (TDE không learn means vì zero-mean signals)
    "learn_means": False,
    "learn_covariances": True,
    "sequence_length": 1000,
    "batch_size": 16,
    "n_epochs": 20,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "covariance_regularization": 1e-6,
    
    # Multi-start initialization
    "n_init": 10,
    "n_init_epochs": 5,
    "init_frac": 0.25,
    
    # Output
    "model_name": "tde_hmm_k8",
}

# ---------------------------------------------------------------
# CẤU HÌNH FEATURE EXTRACTION (cho classification)
# ---------------------------------------------------------------
FEATURE_EXTRACTION = {
    # Epoch xung quanh stimulus onset
    "tmin": -0.2,   # -200 ms
    "tmax": 0.8,    # +800 ms
    
    # Event codes từ Nencki-Symfonia
    "event_ids": {
        "standard": "S5",   # 85% trials
        "target":   "S6",   # 15% trials
        "distractor": "S7", # 15% trials (loại khỏi classification)
    },
    
    # Time windows (aligned với ERP components)
    "time_windows": {
        "baseline": (-0.2, 0.0),    # Pre-stimulus
        "P200":     (0.15, 0.25),   # Early sensory
        "P300":     (0.25, 0.45),   # Target detection
        "late":     (0.5, 0.8),     # Post-decision
    },
    
    # Tổng features: 4 windows × 8 states = 32 features/trial
}

# ---------------------------------------------------------------
# CẤU HÌNH CONNECTIVITY
# ---------------------------------------------------------------
CONNECTIVITY = {
    # Frequency bands
    "freq_bands": {
        "delta": (1, 4),
        "theta": (4, 7),
        "alpha": (7, 13),
        "beta":  (13, 30),
        "gamma": (30, 45),
    },
    
    # Connectivity metrics
    "metrics": ["aec", "coherence", "pli", "wpli"],
    
    # Amplitude Envelope Correlation
    "aec_orth": True,           # Orthogonalise trước khi tính AEC
    
    # Per-state connectivity
    "compute_per_state": True,
    "min_occupancy_threshold": 0.01,  # Bỏ qua states < 1% occupancy
}

# ---------------------------------------------------------------
# TẠO CÁC THƯ MỤC OUTPUT
# ---------------------------------------------------------------
for directory in [PREPROC_DIR, SOURCE_DIR, HMM_DIR, STATE_TS_DIR, CONNECTIVITY_DIR]:
    os.makedirs(directory, exist_ok=True)
