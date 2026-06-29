import os
import sys
import glob
import numpy as np
import mne
from pathlib import Path

# Thêm thư mục gốc vào path để import config
sys.path.insert(0, str(Path("D:\AI y tế").parent.parent))
from configs.pipeline_config import BIDS_ROOT, PREPROC_DIR, PREPROC, EXCLUDED_SUBJECTS

try:
    import osl
    from osl import preprocessing
    USE_OSL = True
    print(f"[OK] osl-ephys {osl.__version__} loaded")
except ImportError:
    USE_OSL = False
    print("[WARN] osl-ephys không tìm thấy, dùng MNE thuần túy")


def auto_trimmer(raw, trim_start_sec=60, max_duration_sec=900):
    """
    Cắt bỏ phần đầu (thường có artifacts chuẩn bị) và
    giới hạn tổng thời gian để tiết kiệm RAM.
    
    Parameters
    ----------
    raw : mne.io.Raw
    trim_start_sec : float
        Số giây đầu bị cắt (mặc định 60s = 1 phút)
    max_duration_sec : float
        Giới hạn thời gian phân tích (mặc định 900s = 15 phút)
    
    Returns
    -------
    raw_trimmed : mne.io.Raw
    """
    original_duration = raw.times[-1]
    
    tmin = trim_start_sec
    tmax = min(trim_start_sec + max_duration_sec, original_duration)
    
    raw_trimmed = raw.copy().crop(tmin=tmin, tmax=tmax)
    
    trimmed_duration = raw_trimmed.times[-1]
    print(f"  [AutoTrimmer] {original_duration:.1f}s -> {trimmed_duration:.1f}s "
          f"(cắt {trim_start_sec}s đầu, giới hạn {max_duration_sec}s)")
    
    return raw_trimmed


def preprocess_subject(subject_id, raw_file_path, output_dir):
    """
    Pipeline preprocessing cho một subject.
    
    Parameters
    ----------
    subject_id : str  (vd: "sub-02")
    raw_file_path : str  (đường dẫn đến file .vhdr hoặc .fif)
    output_dir : str
    
    Returns
    -------
    raw_preproc : mne.io.Raw  hoặc None nếu lỗi
    """
    print(f"\n{'='*60}")
    print(f" Processing: {subject_id}")
    print(f"{'='*60}")
    
    # --- Kiểm tra excluded ---
    if subject_id in EXCLUDED_SUBJECTS:
        print(f"  [SKIP] {subject_id} nằm trong danh sách excluded.")
        return None
    
    output_file = os.path.join(output_dir, f"{subject_id}_preproc_raw.fif")
    if os.path.exists(output_file):
        print(f"  [SKIP] File đã tồn tại: {output_file}")
        return mne.io.read_raw_fif(output_file, preload=True)
    
    # BƯỚC 1: Load dữ liệu thô

    print(f"  [1/7] Loading: {raw_file_path}")
    try:
        if raw_file_path.endswith(".vhdr"):
            raw = mne.io.read_raw_brainvision(raw_file_path, preload=True, verbose=False)
        elif raw_file_path.endswith(".fif"):
            raw = mne.io.read_raw_fif(raw_file_path, preload=True, verbose=False)
        elif raw_file_path.endswith(".set"):
            raw = mne.io.read_raw_eeglab(raw_file_path, preload=True, verbose=False)
        else:
            raise ValueError(f"Định dạng file không hỗ trợ: {raw_file_path}")
    except Exception as e:
        print(f"  [ERROR] Không load được: {e}")
        return None
    
    print(f"  Duration: {raw.times[-1]:.1f}s | "
          f"Channels: {len(raw.ch_names)} | "
          f"Sfreq: {raw.info['sfreq']} Hz")
    
    # BƯỚC 2: Auto-Trimmer
    
    print(f"  [2/7] Auto-Trimmer...")
    raw = auto_trimmer(
        raw,
        trim_start_sec=PREPROC["trim_start_sec"],
        max_duration_sec=PREPROC["max_duration_sec"]
    )
    

    # BƯỚC 3: Set EEG reference và channel types
    print(f"  [3/7] Setting montage và reference...")
    
    # Set standard 10-20 montage (Nencki dùng 128-ch)
    try:
        montage = mne.channels.make_standard_montage("biosemi128")
        # Chỉ set channels có trong montage
        raw.set_montage(montage, on_missing="warn", verbose=False)
    except Exception:
        print("  [WARN] Không set được montage chuẩn, thử EasyCap128...")
        try:
            montage = mne.channels.make_standard_montage("easycap-M1")
            raw.set_montage(montage, on_missing="warn", verbose=False)
        except Exception as e:
            print(f"  [WARN] Montage error: {e}")
    

    # BƯỚC 4: Bandpass + Notch filtering
    print(f"  [4/7] Filtering: "
          f"{PREPROC['l_freq']}-{PREPROC['h_freq']} Hz, "
          f"Notch {PREPROC['notch_freq']} Hz...")
    
    raw.filter(
        l_freq=PREPROC["l_freq"],
        h_freq=PREPROC["h_freq"],
        method="iir",
        iir_params={"order": 4, "ftype": "butter"},
        verbose=False
    )
    raw.notch_filter(
        freqs=PREPROC["notch_freq"],
        verbose=False
    )
    

    # BƯỚC 5: Downsampling
    print(f"  [5/7] Downsampling: "
          f"{PREPROC['sfreq_original']} Hz -> "
          f"{PREPROC['sfreq_target']} Hz...")
    raw.resample(PREPROC["sfreq_target"], verbose=False)
    
    # BƯỚC 6: Bad channel detection và interpolation
    print(f"  [6/7] Bad channel detection...")
    raw.set_eeg_reference("average", projection=True, verbose=False)
    raw.apply_proj(verbose=False)
    
    # Tự động phát hiện bad channels bằng z-score
    eeg_data = raw.get_data(picks="eeg")
    channel_std = np.std(eeg_data, axis=1)
    z_scores = np.abs((channel_std - np.median(channel_std)) / 
                      (np.std(channel_std) + 1e-10))
    bad_channels = [raw.ch_names[i] for i, z in enumerate(z_scores) if z > 3.5]
    
    if bad_channels:
        raw.info["bads"] = bad_channels
        raw.interpolate_bads(reset_bads=True, verbose=False)
        pct_bad = len(bad_channels) / len(raw.ch_names) * 100
        print(f"  [6/7] Interpolated {len(bad_channels)} bad channels ({pct_bad:.1f}%): "
              f"{bad_channels[:5]}{'...' if len(bad_channels) > 5 else ''}")
    else:
        print(f"  [6/7] Không phát hiện bad channels")
    
    # BƯỚC 7: ICA để loại bỏ ocular và cardiac artifacts
    print(f"  [7/7] ICA artifact removal "
          f"({PREPROC['n_ica_components']} components)...")
    
    ica = mne.preprocessing.ICA(
        n_components=PREPROC["n_ica_components"],
        method=PREPROC["ica_method"],
        random_state=42,
        verbose=False
    )
    ica.fit(raw, picks="eeg", verbose=False)
    
    # Auto-detect artifacts bằng EOG/ECG correlations
    exclude_idx = []
    
    # EOG (ocular)
    eog_indices, eog_scores = ica.find_bads_eog(
        raw,
        threshold=3.0,
        verbose=False
    )
    exclude_idx.extend(eog_indices)
    
    # ECG (cardiac)
    try:
        ecg_indices, ecg_scores = ica.find_bads_ecg(
            raw,
            threshold=0.3,
            verbose=False
        )
        exclude_idx.extend(ecg_indices)
    except Exception:
        pass  # Không có ECG channel
    
    exclude_idx = list(set(exclude_idx))
    print(f"  [ICA] Loại {len(exclude_idx)} components: {exclude_idx}")
    
    ica.exclude = exclude_idx
    raw = ica.apply(raw, verbose=False)
    
    # LƯU KẾT QUẢ
    os.makedirs(output_dir, exist_ok=True)
    raw.save(output_file, overwrite=True, verbose=False)
    print(f"  [SAVED] {output_file}")
    
    return raw


# =============================================================================
# MAIN - chạy cho tất cả subjects
# =============================================================================
def run_preprocessing():
    print("\n" + "="*60)
    print(" BƯỚC 1: EEG PREPROCESSING PIPELINE")
    print("="*60)
    print(f"  BIDS Root:   {BIDS_ROOT}")
    print(f"  Output Dir:  {PREPROC_DIR}")
    print(f"  Excluded:    {EXCLUDED_SUBJECTS}")
    
    # Tìm tất cả file raw EEG
    # Nencki-Symfonia: BIDS format, task oddball
    raw_files = sorted(glob.glob(
        os.path.join(BIDS_ROOT, "sub-*", "eeg", "*task-oddball*_eeg.vhdr")
    ))
    
    if not raw_files:
        # Fallback: tìm .fif hoặc .set
        raw_files = sorted(glob.glob(os.path.join(BIDS_ROOT, "sub-*", "eeg", "*.fif")))
    
    if not raw_files:
        print(f"\n[ERROR] Không tìm thấy file EEG trong {BIDS_ROOT}")
        print("  Kiểm tra cấu trúc BIDS: {BIDS_ROOT}/sub-XX/eeg/*.vhdr")
        return
    
    print(f"\n  Tìm thấy {len(raw_files)} subjects")
    
    results = {
        "success": [],
        "failed": [],
        "skipped": [],
    }
    
    for raw_file in raw_files:
        # Trích xuất subject ID từ đường dẫn
        parts = Path(raw_file).parts
        subject_id = [p for p in parts if p.startswith("sub-")][0]
        
        if subject_id in EXCLUDED_SUBJECTS:
            results["skipped"].append(subject_id)
            continue
        
        try:
            raw_preproc = preprocess_subject(subject_id, raw_file, PREPROC_DIR)
            if raw_preproc is not None:
                results["success"].append(subject_id)
            else:
                results["skipped"].append(subject_id)
        except Exception as e:
            print(f"  [ERROR] {subject_id}: {e}")
            results["failed"].append(subject_id)
    
    # Summary
    print(f"\n{'='*60}")
    print(f" PREPROCESSING HOÀN TẤT")
    print(f"{'='*60}")
    print(f"  Thành công:  {len(results['success'])} subjects")
    print(f"  Thất bại:    {len(results['failed'])} subjects: {results['failed']}")
    print(f"  Bỏ qua:      {len(results['skipped'])} subjects: {results['skipped']}")
    print(f"  Output:      {PREPROC_DIR}")
    
    return results


if __name__ == "__main__":
    run_preprocessing()
