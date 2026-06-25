import time
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pygame
import torch
import torch.nn as nn
from scipy.signal import welch
from scipy.linalg import eigh

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    CHECKPOINT_PATH: str = "Motor Imagery Classification/Model/braindance.joblib"

    BOARD_ID: int = BoardIds.CYTON_BOARD.value
    SERIAL_PORT: str = "COM3"

    MAC_ADDRESS: str = ""
    IP_ADDRESS: str = ""
    IP_PORT: int = 0
    OTHER_INFO: str = ""
    TIMEOUT: int = 0
    SAMPLING_RATE: int = 250

    # Live acquisition:
    # collect 1300, filter, crop 50 front + 50 back -> 1200 model samples
    WINDOW_SAMPLES: int = 1300
    CROP_START: int = 50
    CROP_END_FROM_LAST: int = 50
    EXPECTED_FINAL_SAMPLES: int = 1200

    LOW_CUT: float = 8.0
    HIGH_CUT: float = 13.0
    FILTER_ORDER: int = 4
    ENV_NOISE_MODE: int = 1

    # Same bad channels as your acquisition notebook.
    BAD_CHANNELS: Tuple[int, ...] = (0, 1, 2, 6, 7)

    # ========================================================
    # Live EEG window quality rejection
    # ========================================================
    # Bad windows are rejected BEFORE model inference.
    # Rejected windows do not update the decision stabilizer and do not move the car.
    ENABLE_WINDOW_QUALITY_REJECTION: bool = True

    # Raw-window checks are intentionally lenient because raw OpenBCI EEG can have
    # large DC offsets. These mostly catch disconnected/frozen/saturated data.
    RAW_MAX_ABS_UV: float = 1_000_000.0
    RAW_MAX_PTP_UV: float = 50_000.0
    RAW_MIN_STD_UV: float = 1e-6

    # Filtered-window checks happen after bad-channel removal, detrend, 8-13 Hz
    # bandpass, notch/environmental-noise removal, and edge cropping.
    # These are the important checks for rejecting motion/muscle/electrode noise.
    FILTERED_MAX_ABS_UV: float = 250.0
    FILTERED_MAX_PTP_UV: float = 350.0
    FILTERED_MIN_STD_UV: float = 0.02
    FILTERED_MAX_STD_UV: float = 80.0

    # With only C3/Cz/C4 after BAD_CHANNELS, be strict: reject if any channel is bad.
    MAX_BAD_CHANNELS_PER_WINDOW: int = 0

    # Reject frozen windows: too many consecutive near-identical samples.
    FLAT_DIFF_EPS: float = 1e-9
    MAX_FLATLINE_RUN_SAMPLES: int = 50

    # Optional debug print for why windows were rejected.
    PRINT_REJECTED_WINDOWS: bool = True

    # Inference stabilization
    INFERENCE_INTERVAL_SECONDS: float = 0.25
    CONFIDENCE_THRESHOLD: float = 0.65
    REQUIRED_CONSECUTIVE_PREDICTIONS: int = 2
    DECISION_COOLDOWN_SECONDS: float = 5

    CLASS_TO_COMMAND: Dict[int, str] = None

    # Pygame
    SCREEN_WIDTH: int = 1440
    SCREEN_HEIGHT: int = 720
    FPS: int = 60

    ROAD_WIDTH: int = 360
    LANE_WIDTH: int = 180

    CAR_WIDTH: int = 58
    CAR_HEIGHT: int = 95

    CAR_Y: int = 500
    CAR_LANE_MOVE_SPEED: float = 12.0

    OBSTACLE_WIDTH: int = 70
    OBSTACLE_HEIGHT: int = 90

    # Step-based game logic.
    # Obstacles move down by this amount only after a valid EEG decision.
    STEP_PIXELS: int = 110
    OBSTACLE_LOOKAHEAD_STEPS: int = 8
    OBSTACLE_SPAWN_EVERY_STEPS: int = 10
    STARTING_OBSTACLES: int = 1

    KEYBOARD_DEBUG_CONTROL: bool = True

    def __post_init__(self):
        if self.CLASS_TO_COMMAND is None:
            self.CLASS_TO_COMMAND = {
                0: "left",
                1: "right",
            }


CFG = Config()

class BraindanceCNNLSTMFusion(nn.Module):

    def __init__(
        self,
        seq_input_size,
        feature_input_size,
        num_classes=2,
        bidirectional=False
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.cnn = nn.Sequential(

            nn.Conv1d(
                in_channels=seq_input_size,
                out_channels=128,
                kernel_size=15,
                padding=7
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=9,
                padding=4
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(kernel_size=4),
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=0.45,
            bidirectional=bidirectional
        )

        lstm_output_size = 128 if bidirectional else 64

        self.lstm_projector = nn.Sequential(
            nn.Linear(lstm_output_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.feature_mlp = nn.Sequential(

            nn.Linear(feature_input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(

            nn.Linear(16+8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(16, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(4, num_classes)
        )

        self.gate = nn.Linear(24, 2)

    def forward(self, x_seq, x_features):

        # (B,T,C) -> (B,C,T)
        x_seq = x_seq.transpose(1, 2)

        cnn_out = self.cnn(x_seq)

        # (B,C,T) -> (B,T,C)
        cnn_out = cnn_out.transpose(1, 2)

        _, (h_n, _) = self.lstm(cnn_out)

        if self.bidirectional:
            lstm_final = torch.cat(
                [h_n[-2], h_n[-1]],
                dim=1
            )
        else:
            lstm_final = h_n[-1]

        lstm_features = self.lstm_projector(lstm_final)
        mlp_features = self.feature_mlp(x_features)

        combined = torch.cat(
            [
                lstm_features,
                mlp_features
            ],
            dim=1
        )

        gate_logits = self.gate(combined)
        weights = torch.softmax(gate_logits, dim=1)

        lstm_features = (
            lstm_features *
            weights[:, 0:1]
        )

        mlp_features = (
            mlp_features *
            weights[:, 1:2]
        )

        fused = torch.cat(
            [
                lstm_features,
                mlp_features
            ],
            dim=1
        )

        logits = self.classifier(fused)

        self.last_lstm_weight = weights[:, 0].mean().detach()
        self.last_mlp_weight = weights[:, 1].mean().detach()

        return logits
    

def add_motor_imagery_virtual_channels(X, c3_idx, cz_idx, c4_idx):
    """
    Full Pipeline-4-compatible virtual channel generator.

    This generates the large discovery-mode channel set used by your older
    Pipeline 4 checkpoint. The live classifier later selects/reorders generated
    channels to exactly match checkpoint["extractor_state"]["channel_names"].
    """

    if X.ndim != 3:
        raise ValueError(f"Expected X shape (trials, channels, samples), got {X.shape}")

    n_trials, n_channels, n_samples = X.shape

    X_parts = [X]
    channel_names = [f"ch{idx}" for idx in range(n_channels)]

    valid_c3 = c3_idx >= 0 and c3_idx < n_channels
    valid_cz = cz_idx >= 0 and cz_idx < n_channels
    valid_c4 = c4_idx >= 0 and c4_idx < n_channels

    if valid_c3:
        channel_names[c3_idx] = "C3"
    if valid_cz:
        channel_names[cz_idx] = "Cz"
    if valid_c4:
        channel_names[c4_idx] = "C4"

    if not (valid_c3 and valid_cz and valid_c4):
        print("Warning: C3/Cz/C4 not all valid. Returning original X only.")
        return X, channel_names

    c3 = X[:, c3_idx, :]
    cz = X[:, cz_idx, :]
    c4 = X[:, c4_idx, :]

    def add_channel(name, sig):
        X_parts.append(sig[:, None, :])
        channel_names.append(name)

    # 1. Clean / reference-subtracted channels
    c3_clean_025 = c3 - 0.25 * cz
    c3_clean_050 = c3 - 0.50 * cz
    c3_clean_075 = c3 - 0.75 * cz
    c3_clean_100 = c3 - 1.00 * cz

    c4_clean_025 = c4 - 0.25 * cz
    c4_clean_050 = c4 - 0.50 * cz
    c4_clean_075 = c4 - 0.75 * cz
    c4_clean_100 = c4 - 1.00 * cz

    add_channel("C3_clean_025Cz", c3_clean_025)
    add_channel("C3_clean_050Cz", c3_clean_050)
    add_channel("C3_clean_075Cz", c3_clean_075)
    add_channel("C3_clean_100Cz", c3_clean_100)

    add_channel("C4_clean_025Cz", c4_clean_025)
    add_channel("C4_clean_050Cz", c4_clean_050)
    add_channel("C4_clean_075Cz", c4_clean_075)
    add_channel("C4_clean_100Cz", c4_clean_100)

    add_channel("C3_clean", c3_clean_050)
    add_channel("C4_clean", c4_clean_050)

    # 2. Left-right contrast / asymmetry channels
    c3_minus_c4 = c3 - c4
    c4_minus_c3 = c4 - c3

    add_channel("C3_minus_C4", c3_minus_c4)
    add_channel("C4_minus_C3", c4_minus_c3)

    add_channel("C3clean025_minus_C4clean025", c3_clean_025 - c4_clean_025)
    add_channel("C3clean050_minus_C4clean050", c3_clean_050 - c4_clean_050)
    add_channel("C3clean075_minus_C4clean075", c3_clean_075 - c4_clean_075)
    add_channel("C3clean100_minus_C4clean100", c3_clean_100 - c4_clean_100)
    add_channel("C3clean_minus_C4clean", c3_clean_050 - c4_clean_050)

    # 3. Average/common-mode channels
    add_channel("C3_C4_mean", 0.5 * c3 + 0.5 * c4)
    add_channel("C3_Cz_mean", 0.5 * c3 + 0.5 * cz)
    add_channel("C4_Cz_mean", 0.5 * c4 + 0.5 * cz)
    add_channel("C3_Cz_C4_mean", (c3 + cz + c4) / 3.0)

    # 4. Smoothed spatial channels
    c3_smooth_421 = (4.0 / 7.0) * c3 + (2.0 / 7.0) * cz + (1.0 / 7.0) * c4
    c4_smooth_421 = (4.0 / 7.0) * c4 + (2.0 / 7.0) * cz + (1.0 / 7.0) * c3
    cz_smooth_211 = (2.0 / 4.0) * cz + (1.0 / 4.0) * c3 + (1.0 / 4.0) * c4

    add_channel("C3_smooth_421", c3_smooth_421)
    add_channel("Cz_smooth_211", cz_smooth_211)
    add_channel("C4_smooth_421", c4_smooth_421)

    add_channel("C3_smooth_611", (6.0 / 8.0) * c3 + (1.0 / 8.0) * cz + (1.0 / 8.0) * c4)
    add_channel("C4_smooth_611", (6.0 / 8.0) * c4 + (1.0 / 8.0) * cz + (1.0 / 8.0) * c3)

    add_channel("C3_smooth_532", (5.0 / 10.0) * c3 + (3.0 / 10.0) * cz + (2.0 / 10.0) * c4)
    add_channel("C4_smooth_532", (5.0 / 10.0) * c4 + (3.0 / 10.0) * cz + (2.0 / 10.0) * c3)

    # 5. Laplacian-ish channels
    add_channel("C3_laplacian_simple", c3 - 0.5 * (cz + c4))
    add_channel("C4_laplacian_simple", c4 - 0.5 * (cz + c3))
    add_channel("Cz_laplacian_simple", cz - 0.5 * (c3 + c4))

    add_channel("C3_laplacian_strong", c3 - (0.75 * cz + 0.25 * c4))
    add_channel("C4_laplacian_strong", c4 - (0.75 * cz + 0.25 * c3))

    # 6. Ratio-like normalized asymmetry channels
    eps = 1e-8

    add_channel(
        "C3_minus_C4_over_abs_sum",
        (c3 - c4) / (np.abs(c3) + np.abs(c4) + eps)
    )

    add_channel(
        "C3clean_minus_C4clean_over_abs_sum",
        (c3_clean_050 - c4_clean_050) / (
            np.abs(c3_clean_050) + np.abs(c4_clean_050) + eps
        )
    )

    # 7. Cz-centered asymmetries
    add_channel("C3_minus_Cz", c3 - cz)
    add_channel("C4_minus_Cz", c4 - cz)
    add_channel("Cz_minus_C3", cz - c3)
    add_channel("Cz_minus_C4", cz - c4)
    add_channel("C3_minus_Cz_minus_C4_minus_Cz", (c3 - cz) - (c4 - cz))

    # 8. Weighted asymmetry blends
    add_channel("C3_dominant_asym_70_30", 0.7 * c3 - 0.3 * c4)
    add_channel("C4_dominant_asym_70_30", 0.7 * c4 - 0.3 * c3)
    add_channel("C3clean_dominant_asym_70_30", 0.7 * c3_clean_050 - 0.3 * c4_clean_050)
    add_channel("C4clean_dominant_asym_70_30", 0.7 * c4_clean_050 - 0.3 * c3_clean_050)

    X_aug = np.concatenate(X_parts, axis=1)

    return X_aug, channel_names

def safe_log(x, eps=1e-12):
    return np.log(np.maximum(x, eps))

def hjorth_mobility_complexity(sig):
    eps = 1e-12

    d1 = np.diff(sig)
    d2 = np.diff(d1)

    var0 = np.var(sig) + eps
    var1 = np.var(d1) + eps
    var2 = np.var(d2) + eps

    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / (mobility + eps)

    return mobility, complexity

def bandpower(freqs, psd, fmin, fmax):
    idx = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(idx):
        return 0.0

    return np.trapezoid(psd[idx], freqs[idx])

def clean_corr(sig_a, sig_b):
    corr = np.corrcoef(sig_a, sig_b)[0, 1]

    if np.isnan(corr) or np.isinf(corr):
        corr = 0.0

    return corr

def extract_curated_mi_features(X, channel_names, fs, bands):
    """
    Lean MI-focused feature set.

    Per channel:
    - log variance
    - Hjorth mobility
    - Hjorth complexity
    - relative bandpower for mu/alpha bands

    Selected pairwise:
    - C3 vs C4
    - C3_clean vs C4_clean
    - C3_minus_C4 vs C3clean_minus_C4clean
    - C3_clean vs C3_minus_C4
    - C4_clean vs C3_minus_C4

    This avoids feature explosion from all-to-all pairwise combinations.
    """

    n_trials, n_channels, n_samples = X.shape

    name_to_idx = {
        name: idx
        for idx, name in enumerate(channel_names)
    }

    useful_pair_names = [
        ("C3", "C4"),
        ("C3_clean", "C4_clean"),
        ("C3_minus_C4", "C3clean_minus_C4clean"),
        ("C3_clean", "C3_minus_C4"),
        ("C4_clean", "C3_minus_C4"),
    ]

    rows = []

    for trial_idx in range(n_trials):
        trial = X[trial_idx]
        row = {}

        for ch_idx in range(n_channels):
            sig = trial[ch_idx].astype(np.float64)
            ch_name = channel_names[ch_idx]

            sig_var = np.var(sig) + 1e-12

            mobility, complexity = hjorth_mobility_complexity(sig)

            row[f"{ch_name}_logvar"] = safe_log(sig_var)
            row[f"{ch_name}_hjorth_mobility"] = mobility
            row[f"{ch_name}_hjorth_complexity"] = complexity

            freqs, psd = welch(
                sig,
                fs=fs,
                nperseg=min(256, len(sig))
            )

            total_power = np.trapezoid(psd, freqs) + 1e-12

            for band_name, (fmin, fmax) in bands.items():
                bp = bandpower(freqs, psd, fmin, fmax)
                row[f"{ch_name}_{band_name}_relpower"] = bp / total_power

        for name_a, name_b in useful_pair_names:
            if name_a not in name_to_idx or name_b not in name_to_idx:
                continue

            idx_a = name_to_idx[name_a]
            idx_b = name_to_idx[name_b]

            sig_a = trial[idx_a].astype(np.float64)
            sig_b = trial[idx_b].astype(np.float64)

            var_a = np.var(sig_a) + 1e-12
            var_b = np.var(sig_b) + 1e-12

            row[f"{name_a}_minus_{name_b}_logvar_diff"] = safe_log(var_a) - safe_log(var_b)
            row[f"{name_a}_minus_{name_b}_corr"] = clean_corr(sig_a, sig_b)

        rows.append(row)

    features_df = pd.DataFrame(rows)
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(0.0)

    return features_df

class SimpleCSP:
    """
    Binary CSP.

    Fit only on training data.
    Transform train/val/test after fitting.

    Input shape:
        X = (trials, channels, samples)
    """

    def __init__(self, n_components=6, reg=1e-6):
        self.n_components = n_components
        self.reg = reg
        self.filters_ = None

    def _covariance(self, trial):
        cov = trial @ trial.T
        cov = cov / (np.trace(cov) + 1e-12)
        return cov

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        classes = np.unique(y)

        if len(classes) != 2:
            raise ValueError(f"CSP needs exactly 2 classes, got {classes}")

        n_trials, n_channels, n_samples = X.shape

        covs = []

        for cls in classes:
            X_cls = X[y == cls]

            cls_covs = []

            for trial in X_cls:
                cls_covs.append(self._covariance(trial))

            covs.append(np.mean(cls_covs, axis=0))

        cov_0 = covs[0]
        cov_1 = covs[1]

        composite_cov = cov_0 + cov_1

        cov_0 = cov_0 + self.reg * np.eye(n_channels)
        composite_cov = composite_cov + self.reg * np.eye(n_channels)

        eigenvalues, eigenvectors = eigh(cov_0, composite_cov)

        sorted_indices = np.argsort(eigenvalues)

        n_components = min(self.n_components, n_channels)

        half = n_components // 2

        if n_components % 2 == 0:
            selected_indices = np.concatenate([
                sorted_indices[:half],
                sorted_indices[-half:]
            ])
        else:
            selected_indices = np.concatenate([
                sorted_indices[:half + 1],
                sorted_indices[-half:]
            ])

        self.filters_ = eigenvectors[:, selected_indices].T

        return self

    def transform(self, X):
        if self.filters_ is None:
            raise RuntimeError("CSP must be fitted before transform.")

        X = np.asarray(X, dtype=np.float64)

        out = []

        for trial in X:
            projected = self.filters_ @ trial

            var = np.var(projected, axis=1)
            var_norm = var / (np.sum(var) + 1e-12)

            features = np.log(var_norm + 1e-12)
            out.append(features)

        return np.asarray(out)


def longest_flatline_run(sig: np.ndarray, eps: float = 1e-9) -> int:
    """Return longest run of near-identical consecutive samples."""
    sig = np.asarray(sig, dtype=np.float64)

    if sig.size < 2:
        return sig.size

    diffs = np.abs(np.diff(sig)) <= eps

    longest = 0
    current = 0

    for is_flat in diffs:
        if is_flat:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0

    # diffs count transitions, sample run is transitions + 1
    return int(longest + 1) if longest > 0 else 0


def assess_eeg_window_quality(X: np.ndarray, cfg: Config, stage: str) -> Tuple[bool, str, Dict]:
    """
    Returns:
        is_good, reason, metrics

    X shape must be (channels, samples).
    stage must be "raw" or "filtered".
    """

    if not cfg.ENABLE_WINDOW_QUALITY_REJECTION:
        return True, "quality rejection disabled", {}

    if X.ndim != 2:
        return False, f"{stage}: bad shape {X.shape}", {}

    if not np.all(np.isfinite(X)):
        return False, f"{stage}: NaN/Inf detected", {}

    if stage == "raw":
        max_abs_limit = cfg.RAW_MAX_ABS_UV
        max_ptp_limit = cfg.RAW_MAX_PTP_UV
        min_std_limit = cfg.RAW_MIN_STD_UV
        max_std_limit = float("inf")
    elif stage == "filtered":
        max_abs_limit = cfg.FILTERED_MAX_ABS_UV
        max_ptp_limit = cfg.FILTERED_MAX_PTP_UV
        min_std_limit = cfg.FILTERED_MIN_STD_UV
        max_std_limit = cfg.FILTERED_MAX_STD_UV
    else:
        raise ValueError(f"Unknown quality stage: {stage}")

    bad_reasons = []
    per_channel = []

    for ch in range(X.shape[0]):
        sig = np.asarray(X[ch], dtype=np.float64)

        max_abs = float(np.max(np.abs(sig)))
        ptp = float(np.ptp(sig))
        std = float(np.std(sig))
        rms = float(np.sqrt(np.mean(sig ** 2)))
        flat_run = longest_flatline_run(sig, eps=cfg.FLAT_DIFF_EPS)

        reasons = []

        if max_abs > max_abs_limit:
            reasons.append(f"abs {max_abs:.1f}>{max_abs_limit:.1f}")

        if ptp > max_ptp_limit:
            reasons.append(f"ptp {ptp:.1f}>{max_ptp_limit:.1f}")

        if std < min_std_limit:
            reasons.append(f"std {std:.6f}<{min_std_limit:.6f}")

        if std > max_std_limit:
            reasons.append(f"std {std:.1f}>{max_std_limit:.1f}")

        if flat_run >= cfg.MAX_FLATLINE_RUN_SAMPLES:
            reasons.append(f"flatline {flat_run}>={cfg.MAX_FLATLINE_RUN_SAMPLES}")

        per_channel.append({
            "channel": ch,
            "max_abs": max_abs,
            "ptp": ptp,
            "std": std,
            "rms": rms,
            "flat_run": flat_run,
            "bad": len(reasons) > 0,
            "reasons": reasons,
        })

        if reasons:
            bad_reasons.append(f"ch{ch}: " + ", ".join(reasons))

    bad_channel_count = sum(1 for item in per_channel if item["bad"])

    metrics = {
        "stage": stage,
        "bad_channel_count": bad_channel_count,
        "per_channel": per_channel,
    }

    if bad_channel_count > cfg.MAX_BAD_CHANNELS_PER_WINDOW:
        reason = f"{stage}: {bad_channel_count} bad channel(s): " + " | ".join(bad_reasons)
        return False, reason, metrics

    return True, f"{stage}: clean", metrics


def preprocess_live_window(raw_eeg: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Input:
        raw_eeg shape = (brainflow_eeg_channels, 1300)

    Output:
        X shape = (1, kept_channels, 1200)
    """

    if raw_eeg.ndim != 2:
        raise ValueError(f"raw_eeg must be (channels, samples), got {raw_eeg.shape}")

    if raw_eeg.shape[1] != cfg.WINDOW_SAMPLES:
        raise ValueError(
            f"Expected {cfg.WINDOW_SAMPLES} samples, got {raw_eeg.shape[1]}"
        )

    X = raw_eeg.copy().astype(np.float64)

    if len(cfg.BAD_CHANNELS) > 0:
        X = np.delete(X, cfg.BAD_CHANNELS, axis=0)

    raw_good, raw_reason, raw_metrics = assess_eeg_window_quality(
        X,
        cfg,
        stage="raw",
    )

    if not raw_good:
        raise ValueError(raw_reason)

    for ch in range(X.shape[0]):
        signal = np.ascontiguousarray(X[ch], dtype=np.float64)

        DataFilter.detrend(
            signal,
            DetrendOperations.CONSTANT.value,
        )

        DataFilter.perform_bandpass(
            signal,
            cfg.SAMPLING_RATE,
            cfg.LOW_CUT,
            cfg.HIGH_CUT,
            cfg.FILTER_ORDER,
            FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
            0,
        )

        DataFilter.remove_environmental_noise(
            signal,
            cfg.SAMPLING_RATE,
            cfg.ENV_NOISE_MODE,
        )

        X[ch] = signal

    X = X[:, cfg.CROP_START:-cfg.CROP_END_FROM_LAST]

    if X.shape[1] != cfg.EXPECTED_FINAL_SAMPLES:
        raise ValueError(
            f"Final sample count mismatch. Expected {cfg.EXPECTED_FINAL_SAMPLES}, got {X.shape[1]}"
        )

    filtered_good, filtered_reason, filtered_metrics = assess_eeg_window_quality(
        X,
        cfg,
        stage="filtered",
    )

    if not filtered_good:
        raise ValueError(filtered_reason)

    return X[np.newaxis, :, :]


class LiveEEGClassifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cpu")

        checkpoint = joblib.load(cfg.CHECKPOINT_PATH)

        self.checkpoint = checkpoint
        self.model_params = dict(checkpoint["model_params"])

        self.seq_scaler = checkpoint["seq_scaler"]
        self.feature_scaler = checkpoint["feature_scaler"]
        self.csp = checkpoint.get("csp", None)

        extractor_params = checkpoint.get("extractor_params", {})
        extractor_state = checkpoint.get("extractor_state", {})

        self.c3_idx = extractor_params.get("c3_idx", 0)
        self.cz_idx = extractor_params.get("cz_idx", 1)
        self.c4_idx = extractor_params.get("c4_idx", 2)
        self.fs = extractor_params.get("fs", cfg.SAMPLING_RATE)
        self.bands = extractor_params.get("bands", None)

        if self.bands is None:
            # Fallback only. Normally loaded from Pipeline 4 checkpoint.
            self.bands = {
                "mu": (8, 12),
                "alpha": (8, 13),
                "low_alpha": (8, 10),
                "high_alpha": (10, 13),
                "beta": (13, 30),
            }

        self.expected_channel_names = extractor_state.get("channel_names", None)
        self.feature_names = extractor_state.get("feature_names", None)

        if self.feature_names is None:
            raise RuntimeError("Checkpoint missing extractor_state['feature_names'].")

        self.hand_feature_names = [
            name for name in self.feature_names
            if not str(name).startswith("csp_")
        ]

        self.csp_feature_names = [
            name for name in self.feature_names
            if str(name).startswith("csp_")
        ]

        self.model = BraindanceCNNLSTMFusion(
            seq_input_size=self.model_params["seq_input_size"],
            feature_input_size=self.model_params["feature_input_size"],
            num_classes=self.model_params["num_classes"],
            bidirectional=True,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.class_to_command = checkpoint.get("class_to_command", cfg.CLASS_TO_COMMAND)

        print("\nLoaded fused Pipeline 4 checkpoint:", cfg.CHECKPOINT_PATH)
        print("Checkpoint:", checkpoint.get("checkpoint_name", "unknown"))
        print("Model params:", self.model_params)
        print("C3/Cz/C4 idx:", self.c3_idx, self.cz_idx, self.c4_idx)
        print("Bands:", self.bands)
        print("Seq input size:", self.model_params["seq_input_size"])
        print("Feature input size:", self.model_params["feature_input_size"])
        print("Feature count:", len(self.feature_names))
        print("CSP features:", len(self.csp_feature_names))
        print("Class mapping:", self.class_to_command)
        print()

    def _build_pipeline4_inputs(self, X_window: np.ndarray):
        """
        X_window:
            (1, raw_kept_channels, samples)

        Returns:
            X_seq_scaled:  (1, samples, augmented_channels)
            X_feat_scaled: (1, feature_input_size)
        """

        X_aug, channel_names = add_motor_imagery_virtual_channels(
            X_window,
            self.c3_idx,
            self.cz_idx,
            self.c4_idx,
        )

        if self.expected_channel_names is not None:
            # The live generator creates a superset. Select/reorder exactly as the checkpoint was trained.
            name_to_idx = {name: idx for idx, name in enumerate(channel_names)}
            missing_channels = [name for name in self.expected_channel_names if name not in name_to_idx]

            if missing_channels:
                raise RuntimeError(
                    "Live virtual channel generator cannot reproduce checkpoint channels.\n"
                    f"Missing channels: {missing_channels}\n"
                    f"Generated channels: {channel_names}\n"
                    f"Checkpoint channels: {self.expected_channel_names}"
                )

            keep_indices = [name_to_idx[name] for name in self.expected_channel_names]
            X_aug = X_aug[:, keep_indices, :]
            channel_names = list(self.expected_channel_names)

        # Raw LSTM branch: (trials, channels, samples) -> (trials, samples, channels)
        X_seq = np.transpose(X_aug, (0, 2, 1))
        n_trials, n_samples, n_channels = X_seq.shape

        if n_channels != self.model_params["seq_input_size"]:
            raise RuntimeError(
                f"seq_input_size mismatch. Live has {n_channels}, "
                f"checkpoint expects {self.model_params['seq_input_size']}"
            )

        X_seq_scaled = self.seq_scaler.transform(
            X_seq.reshape(-1, n_channels)
        ).reshape(X_seq.shape)

        # Handcrafted features
        hand_df = extract_curated_mi_features(
            X_aug,
            channel_names=channel_names,
            fs=self.fs,
            bands=self.bands,
        )

        # Align handcrafted columns exactly to training order.
        missing_hand = [name for name in self.hand_feature_names if name not in hand_df.columns]
        if missing_hand:
            raise RuntimeError(
                "Missing handcrafted features during inference:\n"
                + "\n".join(missing_hand[:30])
            )

        hand_df = hand_df[self.hand_feature_names].reset_index(drop=True)

        # CSP features
        if self.csp is not None and len(self.csp_feature_names) > 0:
            X_csp = self.csp.transform(X_aug)
            csp_df = pd.DataFrame(X_csp, columns=self.csp_feature_names)
            full_df = pd.concat([hand_df, csp_df], axis=1)
        else:
            full_df = hand_df

        # Align final feature order exactly.
        missing_final = [name for name in self.feature_names if name not in full_df.columns]
        if missing_final:
            raise RuntimeError(
                "Missing final features during inference:\n"
                + "\n".join(missing_final[:30])
            )

        full_df = full_df[self.feature_names]

        if full_df.shape[1] != self.model_params["feature_input_size"]:
            raise RuntimeError(
                f"feature_input_size mismatch. Live has {full_df.shape[1]}, "
                f"checkpoint expects {self.model_params['feature_input_size']}"
            )

        X_feat_scaled = self.feature_scaler.transform(full_df)

        X_seq_scaled = np.nan_to_num(X_seq_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_feat_scaled = np.nan_to_num(X_feat_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        return X_seq_scaled, X_feat_scaled

    def predict_from_raw_window(self, raw_eeg: np.ndarray) -> Dict:
        X_window = preprocess_live_window(raw_eeg, self.cfg)

        X_seq_scaled, X_feat_scaled = self._build_pipeline4_inputs(X_window)

        X_seq_t = torch.tensor(X_seq_scaled, dtype=torch.float32).to(self.device)
        X_feat_t = torch.tensor(X_feat_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X_seq_t, X_feat_t)
            lstm_gate = float(self.model.last_lstm_weight.cpu())
            mlp_gate = float(self.model.last_mlp_weight.cpu())
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])

        command = self.class_to_command.get(pred_class, "none")

        return {
            "class": pred_class,
            "command": command,
            "confidence": confidence,
            "probabilities": probs,
            "lstm_gate": lstm_gate,
            "mlp_gate": mlp_gate,
        }



class DecisionStabilizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.candidate_class = None
        self.candidate_command = None
        self.candidate_count = 0

        self.last_accepted_time = 0.0
        self.last_accepted_command = None

        self.status = "waiting"

    def update(self, result: Dict) -> Tuple[Optional[str], str]:
        now = time.time()

        pred_class = result.get("class", None)
        command = result.get("command", "none")
        confidence = float(result.get("confidence", 0.0))

        if pred_class is None or command not in ("left", "right"):
            self.status = "no valid prediction"
            return None, self.status

        if confidence < self.cfg.CONFIDENCE_THRESHOLD:
            self.status = f"low confidence {confidence:.2f}"
            self.candidate_class = None
            self.candidate_command = None
            self.candidate_count = 0
            return None, self.status

        if pred_class == self.candidate_class:
            self.candidate_count += 1
        else:
            self.candidate_class = pred_class
            self.candidate_command = command
            self.candidate_count = 1

        if self.candidate_count < self.cfg.REQUIRED_CONSECUTIVE_PREDICTIONS:
            self.status = (
                f"candidate {command} "
                f"{self.candidate_count}/{self.cfg.REQUIRED_CONSECUTIVE_PREDICTIONS}"
            )
            return None, self.status

        cooldown_left = self.cfg.DECISION_COOLDOWN_SECONDS - (now - self.last_accepted_time)

        if cooldown_left > 0:
            self.status = f"cooldown {cooldown_left:.2f}s"
            return None, self.status

        self.last_accepted_time = now
        self.last_accepted_command = command

        self.candidate_class = None
        self.candidate_command = None
        self.candidate_count = 0

        self.status = f"accepted {command} -> step"
        return command, self.status


# ============================================================
# BRAINFLOW
# ============================================================

def make_brainflow_params(cfg: Config) -> BrainFlowInputParams:
    params = BrainFlowInputParams()
    params.serial_port = cfg.SERIAL_PORT
    params.mac_address = cfg.MAC_ADDRESS
    params.ip_address = cfg.IP_ADDRESS
    params.ip_port = cfg.IP_PORT
    params.other_info = cfg.OTHER_INFO
    params.timeout = cfg.TIMEOUT
    return params


class EEGStream:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.board = None
        self.eeg_channels = None

    def start(self):
        BoardShim.enable_dev_board_logger()

        params = make_brainflow_params(self.cfg)
        self.board = BoardShim(self.cfg.BOARD_ID, params)

        self.board.prepare_session()
        self.board.start_stream(450000)

        self.eeg_channels = BoardShim.get_eeg_channels(self.cfg.BOARD_ID)

        print("Board started.")
        print("Board ID:", self.cfg.BOARD_ID)
        print("EEG channels:", self.eeg_channels)
        print("Waiting for buffer to fill...")
        print()

    def stop(self):
        if self.board is not None:
            try:
                self.board.stop_stream()
            except Exception:
                pass

            try:
                self.board.release_session()
            except Exception:
                pass

            print("Board stopped.")

    def get_window(self) -> Optional[np.ndarray]:
        if self.board is None:
            return None

        data = self.board.get_current_board_data(self.cfg.WINDOW_SAMPLES)

        if data.shape[1] < self.cfg.WINDOW_SAMPLES:
            return None

        eeg_data = data[self.eeg_channels, :]

        if eeg_data.shape[1] != self.cfg.WINDOW_SAMPLES:
            return None

        return eeg_data


# ============================================================
# STEP-BASED CAR GAME
# ============================================================

class TopDownCarGame:
    def __init__(self, cfg: Config, classifier: LiveEEGClassifier, stream: EEGStream):
        self.cfg = cfg
        self.classifier = classifier
        self.stream = stream
        self.stabilizer = DecisionStabilizer(cfg)

        pygame.init()
        pygame.display.set_caption("BrainDance EEG Step Car Game")

        self.screen = pygame.display.set_mode((cfg.SCREEN_WIDTH, cfg.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Arial", 24)
        self.small_font = pygame.font.SysFont("Arial", 19)
        self.big_font = pygame.font.SysFont("Arial", 42, bold=True)

        self.road_x = (cfg.SCREEN_WIDTH - cfg.ROAD_WIDTH) // 2
        self.left_lane_x = self.road_x + cfg.LANE_WIDTH // 2
        self.right_lane_x = self.road_x + cfg.LANE_WIDTH + cfg.LANE_WIDTH // 2

        self.logical_lane = "left"
        self.target_lane = "left"
        self.car_x = float(self.left_lane_x)
        self.car_y = cfg.CAR_Y

        self.obstacles = []
        self.steps_taken = 0

        self.last_inference_time = 0.0

        self.latest_result = {
            "class": None,
            "command": "collecting",
            "confidence": 0.0,
            "probabilities": np.array([0.0, 0.0]),
        }

        self.decision_status = "waiting"

        self.score = 0
        self.game_over = False

        self.seed_starting_obstacles()

    def lane_center(self, lane: str) -> int:
        if lane == "left":
            return self.left_lane_x
        if lane == "right":
            return self.right_lane_x
        return int(self.car_x)

    def make_obstacle(self, lane: str, y: int):
        x = self.lane_center(lane)
        rect = pygame.Rect(
            x - self.cfg.OBSTACLE_WIDTH // 2,
            y,
            self.cfg.OBSTACLE_WIDTH,
            self.cfg.OBSTACLE_HEIGHT,
        )
        return {
            "lane": lane,
            "rect": rect,
        }

    def spawn_obstacle_at_top(self):
        lane = random.choice(["left", "right"])
        y = self.car_y - self.cfg.OBSTACLE_LOOKAHEAD_STEPS * self.cfg.STEP_PIXELS
        self.obstacles.append(self.make_obstacle(lane, y))

    def seed_starting_obstacles(self):
        self.obstacles = []
        used_rows = set()

        for _ in range(self.cfg.STARTING_OBSTACLES):
            row = random.randint(3, self.cfg.OBSTACLE_LOOKAHEAD_STEPS + 2)

            # Avoid stacking too many in the same row at start.
            tries = 0
            while row in used_rows and tries < 10:
                row = random.randint(3, self.cfg.OBSTACLE_LOOKAHEAD_STEPS + 2)
                tries += 1

            used_rows.add(row)

            lane = random.choice(["left", "right"])
            y = self.car_y - row * self.cfg.STEP_PIXELS
            self.obstacles.append(self.make_obstacle(lane, y))

    def apply_decision_step(self, command: str):
        if self.game_over:
            return

        if command not in ("left", "right"):
            return

        # The prediction chooses the lane and advances one step.
        self.logical_lane = command
        self.target_lane = command

        for obstacle in self.obstacles:
            obstacle["rect"].y += self.cfg.STEP_PIXELS

        self.steps_taken += 1
        self.score += 1

        self.obstacles = [
            obs for obs in self.obstacles
            if obs["rect"].y < self.cfg.SCREEN_HEIGHT + 100
        ]

        if self.steps_taken % self.cfg.OBSTACLE_SPAWN_EVERY_STEPS == 0:
            self.spawn_obstacle_at_top()

        self.check_collision()

    def update_inference(self):
        now = time.time()

        if now - self.last_inference_time < self.cfg.INFERENCE_INTERVAL_SECONDS:
            return

        self.last_inference_time = now

        raw_window = self.stream.get_window()

        if raw_window is None:
            self.latest_result = {
                "class": None,
                "command": "collecting",
                "confidence": 0.0,
                "probabilities": np.array([0.0, 0.0]),
            }
            self.decision_status = "collecting samples"
            return

        try:
            result = self.classifier.predict_from_raw_window(raw_window)
            self.latest_result = result

            accepted_command, status = self.stabilizer.update(result)
            self.decision_status = status

            if accepted_command in ("left", "right"):
                self.apply_decision_step(accepted_command)

        except ValueError as e:
            # Most ValueErrors here are deliberate quality-rejection cases.
            # Do NOT infer and do NOT move the car.
            reason = str(e)

            self.latest_result = {
                "class": None,
                "command": "rejected",
                "confidence": 0.0,
                "probabilities": np.array([0.0, 0.0]),
            }

            self.decision_status = "bad window rejected"

            if self.cfg.PRINT_REJECTED_WINDOWS:
                print("Rejected EEG window:", reason)

        except Exception as e:
            self.latest_result = {
                "class": None,
                "command": f"error: {type(e).__name__}",
                "confidence": 0.0,
                "probabilities": np.array([0.0, 0.0]),
            }
            self.decision_status = "inference error"
            print("Inference error:", repr(e))

    def handle_debug_keydown(self, key):
        if not self.cfg.KEYBOARD_DEBUG_CONTROL:
            return

        if key in (pygame.K_LEFT, pygame.K_a):
            self.apply_decision_step("left")
            self.decision_status = "keyboard left -> step"

        if key in (pygame.K_RIGHT, pygame.K_d):
            self.apply_decision_step("right")
            self.decision_status = "keyboard right -> step"

    def update_game_state(self):
        if self.game_over:
            return

        # Only visual lane interpolation. Forward progress is NOT automatic.
        target_x = self.lane_center(self.target_lane)

        if abs(self.car_x - target_x) < self.cfg.CAR_LANE_MOVE_SPEED:
            self.car_x = float(target_x)
        elif self.car_x < target_x:
            self.car_x += self.cfg.CAR_LANE_MOVE_SPEED
        else:
            self.car_x -= self.cfg.CAR_LANE_MOVE_SPEED

        self.check_collision()

    def check_collision(self):
        # Use logical lane for collision so the model's accepted command decides instantly,
        # even while the car sprite is still sliding visually.
        car_x = self.lane_center(self.logical_lane)

        car_rect = pygame.Rect(
            int(car_x) - self.cfg.CAR_WIDTH // 2,
            self.car_y,
            self.cfg.CAR_WIDTH,
            self.cfg.CAR_HEIGHT,
        )

        for obstacle in self.obstacles:
            if car_rect.colliderect(obstacle["rect"]):
                self.game_over = True
                break

    def draw_road(self):
        self.screen.fill((40, 150, 60))

        road_rect = pygame.Rect(
            self.road_x,
            0,
            self.cfg.ROAD_WIDTH,
            self.cfg.SCREEN_HEIGHT,
        )
        pygame.draw.rect(self.screen, (45, 45, 45), road_rect)

        center_x = self.road_x + self.cfg.ROAD_WIDTH // 2

        dash_height = 45
        gap = 30

        # Dashes are fixed now; the road does not auto-scroll.
        for y in range(-dash_height, self.cfg.SCREEN_HEIGHT + dash_height, dash_height + gap):
            pygame.draw.rect(
                self.screen,
                (230, 230, 230),
                pygame.Rect(center_x - 5, y, 10, dash_height),
            )

        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (self.road_x, 0),
            (self.road_x, self.cfg.SCREEN_HEIGHT),
            5,
        )
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (self.road_x + self.cfg.ROAD_WIDTH, 0),
            (self.road_x + self.cfg.ROAD_WIDTH, self.cfg.SCREEN_HEIGHT),
            5,
        )

    def draw_car(self):
        car_rect = pygame.Rect(
            int(self.car_x) - self.cfg.CAR_WIDTH // 2,
            self.car_y,
            self.cfg.CAR_WIDTH,
            self.cfg.CAR_HEIGHT,
        )

        pygame.draw.rect(self.screen, (40, 120, 250), car_rect, border_radius=10)

        windshield = pygame.Rect(
            car_rect.x + 10,
            car_rect.y + 12,
            car_rect.width - 20,
            25,
        )
        pygame.draw.rect(self.screen, (170, 220, 255), windshield, border_radius=5)

        hood = pygame.Rect(
            car_rect.x + 11,
            car_rect.y + 48,
            car_rect.width - 22,
            30,
        )
        pygame.draw.rect(self.screen, (25, 90, 210), hood, border_radius=5)

    def draw_obstacles(self):
        for obstacle in self.obstacles:
            pygame.draw.rect(
                self.screen,
                (210, 40, 40),
                obstacle["rect"],
                border_radius=8,
            )

            # Small lane label/debug mark
            label = self.small_font.render(obstacle["lane"][0].upper(), True, (255, 255, 255))
            self.screen.blit(
                label,
                (
                    obstacle["rect"].centerx - label.get_width() // 2,
                    obstacle["rect"].centery - label.get_height() // 2,
                ),
            )

    def draw_hud(self):
        result = self.latest_result

        command_text = str(result["command"]).upper()
        class_text = f"Class: {result['class']}"
        conf_text = f"Conf: {result['confidence']:.2f}"
        lane_text = f"Lane: {self.logical_lane.upper()}"
        score_text = f"Steps: {self.score}"
        status_text = f"Decision: {self.decision_status}"

        left_lines = [
            f"Prediction: {command_text}",
            class_text,
            conf_text,
            status_text,
        ]

        right_lines = [
            lane_text,
            score_text,
            f"Threshold: {self.cfg.CONFIDENCE_THRESHOLD:.2f}",
            f"Repeat: {self.cfg.REQUIRED_CONSECUTIVE_PREDICTIONS}x",
            f"Step px: {self.cfg.STEP_PIXELS}",
            "A/D debug = step" if self.cfg.KEYBOARD_DEBUG_CONTROL else "",
        ]

        y = 15
        for line in left_lines:
            surf = self.small_font.render(line, True, (255, 255, 255))
            self.screen.blit(surf, (15, y))
            y += 25

        y = 15
        for line in right_lines:
            if not line:
                continue
            surf = self.small_font.render(line, True, (255, 255, 255))
            self.screen.blit(surf, (self.cfg.SCREEN_WIDTH - surf.get_width() - 15, y))
            y += 25

        probs = result.get("probabilities", np.array([]))

        gate_text = (
        f"LSTM Gate: {result.get('lstm_gate',0):.3f} | "
        f"MLP Gate: {result.get('mlp_gate',0):.3f}"
        )

        surf = self.small_font.render(
            gate_text,
            True,
            (255, 255, 255)
        )

        self.screen.blit(
            surf,
            (15, self.cfg.SCREEN_HEIGHT - 65)
        )

        if probs is not None and len(probs) >= 2:
            p_text = f"P[left/class0]={probs[0]:.2f}    P[right/class1]={probs[1]:.2f}"
            surf = self.small_font.render(p_text, True, (255, 255, 255))
            self.screen.blit(surf, (15, self.cfg.SCREEN_HEIGHT - 36))

        help_text = "Each accepted LEFT/RIGHT prediction changes lane AND advances one step. No automatic forward motion."
        surf = self.small_font.render(help_text, True, (255, 255, 255))
        self.screen.blit(
            surf,
            (
                self.cfg.SCREEN_WIDTH // 2 - surf.get_width() // 2,
                self.cfg.SCREEN_HEIGHT - 36,
            ),
        )

        if self.game_over:
            overlay = pygame.Surface((self.cfg.SCREEN_WIDTH, self.cfg.SCREEN_HEIGHT))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            self.screen.blit(overlay, (0, 0))

            text = self.big_font.render("GAME OVER", True, (255, 70, 70))
            self.screen.blit(
                text,
                (
                    self.cfg.SCREEN_WIDTH // 2 - text.get_width() // 2,
                    self.cfg.SCREEN_HEIGHT // 2 - 60,
                ),
            )

            restart = self.font.render("Press R to restart or ESC to quit", True, (255, 255, 255))
            self.screen.blit(
                restart,
                (
                    self.cfg.SCREEN_WIDTH // 2 - restart.get_width() // 2,
                    self.cfg.SCREEN_HEIGHT // 2,
                ),
            )

    def restart(self):
        self.logical_lane = "left"
        self.target_lane = "left"
        self.car_x = float(self.left_lane_x)
        self.steps_taken = 0
        self.score = 0
        self.game_over = False
        self.stabilizer = DecisionStabilizer(self.cfg)
        self.decision_status = "waiting"
        self.seed_starting_obstacles()

    def run(self):
        running = True

        while running:
            self.clock.tick(self.cfg.FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    if event.key == pygame.K_r and self.game_over:
                        self.restart()

                    if not self.game_over:
                        self.handle_debug_keydown(event.key)

            if not self.game_over:
                self.update_inference()
                self.update_game_state()

            self.draw_road()
            self.draw_obstacles()
            self.draw_car()
            self.draw_hud()

            pygame.display.flip()

        pygame.quit()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nBrainDance Live Inference Step Car Game")
    print("---------------------------------------")
    print(f"Checkpoint: {CFG.CHECKPOINT_PATH}")
    print(f"Window samples: {CFG.WINDOW_SAMPLES}")
    print(f"Crop: {CFG.CROP_START} front, {CFG.CROP_END_FROM_LAST} rear")
    print(f"Final samples: {CFG.WINDOW_SAMPLES - CFG.CROP_START - CFG.CROP_END_FROM_LAST}")
    print(f"Inference interval: {CFG.INFERENCE_INTERVAL_SECONDS}s")
    print(f"Confidence threshold: {CFG.CONFIDENCE_THRESHOLD}")
    print(f"Required repeats: {CFG.REQUIRED_CONSECUTIVE_PREDICTIONS}")
    print(f"Cooldown: {CFG.DECISION_COOLDOWN_SECONDS}s")
    print(f"Step pixels: {CFG.STEP_PIXELS}")
    print()

    classifier = LiveEEGClassifier(CFG)
    stream = EEGStream(CFG)

    try:
        stream.start()
        game = TopDownCarGame(CFG, classifier, stream)
        game.run()

    finally:
        stream.stop()


if __name__ == "__main__":
    main()
