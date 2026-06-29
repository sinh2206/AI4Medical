"""
Generate all figures for the report
Outputs: F:\train_HMM\figures\*.png (300 DPI, publication quality)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from glob import glob

# ==================== STYLE ====================
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.dpi'] = 100

# ==================== PATHS ====================
TDE_DIR = Path(r"F:\train_HMM\derivatives\tde_hmm")
AE_DIR = Path(r"F:\train_HMM\derivatives\ae_hmm_v2")
FEAT_DIR = Path(r"F:\train_HMM\derivatives\features")
HMM_INPUT = Path(r"F:\train_HMM\derivatives\hmm_input")
FIG_DIR = Path(r"F:\train_HMM\figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE = ["sub-01", "sub-22", "sub-25", "sub-28"]
N_STATES = 8
SFREQ = 250

# Color palette
state_colors = plt.cm.tab10(np.linspace(0, 1, N_STATES))
method_colors = {'AE-HMM': '#E74C3C', 'TDE-HMM': '#3498DB'}

print("="*70)
print("Generating figures for report...")
print("="*70)

# ==================== FIG 2: LOSS CURVES (FIXED) ====================
print("\n[2/6] Loss curves...")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ae_hist_file = AE_DIR / "history.npy"
tde_hist_file = TDE_DIR / "history.npy"

if ae_hist_file.exists():
    ae_hist = np.load(ae_hist_file, allow_pickle=True).item()
    if 'loss' in ae_hist:
        loss = ae_hist['loss']
        axes[0].plot(loss, color=method_colors['AE-HMM'], linewidth=2, marker='o', markersize=4)
axes[0].set_title('AE-HMM Training Loss', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Negative Log-Likelihood')
axes[0].grid(alpha=0.3)
axes[0].axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
# Auto y-limits

if tde_hist_file.exists():
    tde_hist = np.load(tde_hist_file, allow_pickle=True).item()
    if 'loss' in tde_hist:
        loss = tde_hist['loss']
        axes[1].plot(loss, color=method_colors['TDE-HMM'], linewidth=2, marker='o', markersize=4)
axes[1].set_title('TDE-HMM Training Loss', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Negative Log-Likelihood')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig2_loss_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ saved fig2_loss_curves.png (FIXED)")

# ==================== FIG 3: FRACTIONAL OCCUPANCY ====================
print("\n[3/6] Fractional Occupancy comparison...")

def compute_fo_from_alphas(hmm_dir):
    """Tính FO từ tất cả alpha files"""
    alpha_files = sorted(glob(str(hmm_dir / "*_alpha.npy")))
    fos = []
    for f in alpha_files:
        sub_id = Path(f).stem.replace("_task-oddball_alpha", "")
        if sub_id in EXCLUDE:
            continue
        alpha = np.load(f)
        stc = np.argmax(alpha, axis=1)
        fo = np.array([np.mean(stc == s) for s in range(N_STATES)])
        fos.append(fo)
    return np.array(fos)

ae_fos = compute_fo_from_alphas(AE_DIR)
tde_fos = compute_fo_from_alphas(TDE_DIR)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

x = np.arange(N_STATES)

# AE-HMM
ae_mean = ae_fos.mean(axis=0) * 100
ae_std = ae_fos.std(axis=0) * 100
bars1 = axes[0].bar(x, ae_mean, yerr=ae_std, color=method_colors['AE-HMM'],
                     edgecolor='black', linewidth=0.5, capsize=4, alpha=0.8)
axes[0].set_title('AE-HMM: All subjects collapsed to S7', fontweight='bold')
axes[0].set_xlabel('State')
axes[0].set_ylabel('Fractional Occupancy (%)')
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'S{i}' for i in range(N_STATES)])
axes[0].set_ylim(0, 110)
axes[0].grid(axis='y', alpha=0.3)
axes[0].axhline(12.5, color='gray', linestyle='--', linewidth=1, label='Uniform (12.5%)')
axes[0].legend()

# TDE-HMM
tde_mean = tde_fos.mean(axis=0) * 100
tde_std = tde_fos.std(axis=0) * 100
bars2 = axes[1].bar(x, tde_mean, yerr=tde_std, color=method_colors['TDE-HMM'],
                     edgecolor='black', linewidth=0.5, capsize=4, alpha=0.8)
axes[1].set_title('TDE-HMM: Diverse state activation', fontweight='bold')
axes[1].set_xlabel('State')
axes[1].set_ylabel('Fractional Occupancy (%)')
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'S{i}' for i in range(N_STATES)])
axes[1].set_ylim(0, 110)
axes[1].grid(axis='y', alpha=0.3)
axes[1].axhline(12.5, color='gray', linestyle='--', linewidth=1, label='Uniform (12.5%)')
axes[1].legend()

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig3_fractional_occupancy.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ saved fig3_fractional_occupancy.png")

# ==================== FIG 4: STATE TIME COURSE AROUND STIMULUS ====================
print("\n[4/6] State time courses around stimulus...")

# Compute mean alpha quanh stimulus (TDE-HMM only — vì AE collapse)
def compute_event_locked_alpha(hmm_dir, code, tmin=-0.2, tmax=0.8):
    """Trung bình alpha across trials cho 1 event code"""
    alpha_files = sorted(glob(str(hmm_dir / "*_alpha.npy")))
    all_epochs = []
    
    for f in alpha_files:
        sub_id = Path(f).stem.replace("_task-oddball_alpha", "")
        if sub_id in EXCLUDE:
            continue
        alpha = np.load(f)
        events_file = HMM_INPUT / f"{sub_id}_task-oddball_events.npy"
        events = np.load(events_file)
        events = events[events[:, 2] == code]
        
        # TDE offset
        offset = 7 if 'tde' in str(hmm_dir).lower() else 0
        t1, t2 = int(tmin*SFREQ), int(tmax*SFREQ)
        
        for ev in events:
            idx = ev[0] - offset
            start = idx + t1
            end = idx + t2
            if 0 <= start and end <= alpha.shape[0]:
                all_epochs.append(alpha[start:end])
    
    return np.array(all_epochs)   # (n_trials, n_times, n_states)

print("  Computing for standard...")
std_epochs = compute_event_locked_alpha(TDE_DIR, code=5)
print("  Computing for target...")
tgt_epochs = compute_event_locked_alpha(TDE_DIR, code=6)

print(f"  std: {std_epochs.shape}, tgt: {tgt_epochs.shape}")

t_axis = np.linspace(-0.2, 0.8, std_epochs.shape[1])

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

# Standard
mean_std = std_epochs.mean(axis=0)   # (n_times, n_states)
for s in range(N_STATES):
    axes[0].plot(t_axis * 1000, mean_std[:, s] * 100, 
                 label=f'S{s}', color=state_colors[s], linewidth=1.5)
axes[0].set_title('TDE-HMM: State probabilities locked to STANDARD stimulus', fontweight='bold')
axes[0].set_ylabel('State probability (%)')
axes[0].axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
axes[0].grid(alpha=0.3)
axes[0].legend(loc='upper right', ncol=4, fontsize=9)

# Target
mean_tgt = tgt_epochs.mean(axis=0)
for s in range(N_STATES):
    axes[1].plot(t_axis * 1000, mean_tgt[:, s] * 100,
                 label=f'S{s}', color=state_colors[s], linewidth=1.5)
axes[1].set_title('TDE-HMM: State probabilities locked to TARGET stimulus', fontweight='bold')
axes[1].set_ylabel('State probability (%)')
axes[1].set_xlabel('Time (ms, 0 = stimulus onset)')
axes[1].axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
axes[1].axvspan(250, 450, alpha=0.1, color='red', label='P300 window')
axes[1].grid(alpha=0.3)
axes[1].legend(loc='upper right', ncol=4, fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig4_state_timecourses.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ saved fig4_state_timecourses.png")

# ==================== FIG 5: CLASSIFICATION COMPARISON ====================
print("\n[5/6] Classification accuracy bars...")

# Hardcoded results — paste từ output bạn đã có
results = {
    'AE-HMM v2': {
        'LogReg':      (50.00, 0.00),
        'LogReg L1':   (50.00, 0.00),
        'SVM-RBF':     (50.00, 0.00),
        'RandomForest':(50.00, 0.00),
        'GradBoost':   (50.00, 0.00),
    },
    'TDE-HMM': {
        'LogReg':      (61.33, 1.84),
        'LogReg L1':   (61.37, 1.80),
        'SVM-RBF':     (59.90, 1.05),
        'RandomForest':(58.32, 1.36),
        'GradBoost':   (53.76, 1.64),
    }
}

fig, ax = plt.subplots(figsize=(10, 5))

methods = list(results['AE-HMM v2'].keys())
x = np.arange(len(methods))
width = 0.35

ae_means = [results['AE-HMM v2'][m][0] for m in methods]
ae_stds  = [results['AE-HMM v2'][m][1] for m in methods]
tde_means = [results['TDE-HMM'][m][0] for m in methods]
tde_stds  = [results['TDE-HMM'][m][1] for m in methods]

bars1 = ax.bar(x - width/2, ae_means, width, yerr=ae_stds, label='AE-HMM v2',
                color=method_colors['AE-HMM'], edgecolor='black', linewidth=0.5,
                capsize=4, alpha=0.8)
bars2 = ax.bar(x + width/2, tde_means, width, yerr=tde_stds, label='TDE-HMM',
                color=method_colors['TDE-HMM'], edgecolor='black', linewidth=0.5,
                capsize=4, alpha=0.8)

ax.axhline(50, color='gray', linestyle='--', linewidth=1.5, label='Chance (50%)', zorder=0)

ax.set_xlabel('Classifier')
ax.set_ylabel('Balanced Accuracy (%)')
ax.set_title('Classification: Standard vs Target — AE-HMM vs TDE-HMM',
              fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15)
ax.set_ylim(45, 70)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars2, tde_means):
    height = bar.get_height()
    ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig5_classification.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ saved fig5_classification.png")

# ==================== FIG 6: CONFUSION MATRIX ====================
print("\n[6/6] Confusion matrices...")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import confusion_matrix

X_tde = np.load(FEAT_DIR / "X_tde_v2.npy")
y_tde = np.load(FEAT_DIR / "y_tde_v2.npy")
sub_tde = np.load(FEAT_DIR / "subjects_tde_v2.npy")

X_ae = np.load(FEAT_DIR / "X_ae.npy")
y_ae = np.load(FEAT_DIR / "y_ae.npy")
sub_ae = np.load(FEAT_DIR / "subjects_ae.npy")

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=2000)),
])

gkf = GroupKFold(n_splits=5)

print("  Computing predictions...")
y_pred_ae = cross_val_predict(clf, X_ae, y_ae, groups=sub_ae, cv=gkf, n_jobs=-1)
y_pred_tde = cross_val_predict(clf, X_tde, y_tde, groups=sub_tde, cv=gkf, n_jobs=-1)

cm_ae = confusion_matrix(y_ae, y_pred_ae, normalize='true')
cm_tde = confusion_matrix(y_tde, y_pred_tde, normalize='true')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, cm, title in zip(axes, [cm_ae, cm_tde], ['AE-HMM', 'TDE-HMM']):
    im = ax.imshow(cm * 100, cmap='Blues', vmin=0, vmax=100)
    ax.set_title(f'{title} Confusion Matrix', fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Standard', 'Target'])
    ax.set_yticklabels(['Standard', 'Target'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i,j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i,j]*100:.1f}%', ha='center', va='center',
                     color=color, fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig6_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ saved fig6_confusion_matrix.png")

print("\n" + "="*70)
print(f"✅ DONE! Tất cả figures saved to {FIG_DIR}")
print("="*70)
print("\nFigures generated:")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"  - {f.name}")