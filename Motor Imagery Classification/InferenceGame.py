import time
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pygame
import torch
import torch.nn as nn
from scipy.linalg import eigh
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations



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

    WINDOW_SAMPLES: int = 1300
    CROP_START: int = 50
    CROP_END_FROM_LAST: int = 50
    EXPECTED_FINAL_SAMPLES: int = 1200

    LOW_CUT: float = 8.0
    HIGH_CUT: float = 13.0
    FILTER_ORDER: int = 4
    ENV_NOISE_MODE: int = 1
    BAD_CHANNELS: Tuple[int, ...] = (0, 1, 2, 6, 7)
    INFERENCE_INTERVAL_SECONDS: float = 0.25
    CONFIDENCE_THRESHOLD: float = 0.5
    REQUIRED_CONSECUTIVE_PREDICTIONS: int = 3
    DECISION_COOLDOWN_SECONDS: float = 0.5
    HOLD_LANE_ON_LOW_CONFIDENCE: bool = True
    CLASS_TO_COMMAND: Dict[int, str] = None

    SCREEN_WIDTH: int = 1440
    SCREEN_HEIGHT: int = 720
    FPS: int = 60

    ROAD_WIDTH: int = 360
    LANE_WIDTH: int = 180

    CAR_WIDTH: int = 58
    CAR_HEIGHT: int = 95

    CAR_Y: int = 500
    CAR_LANE_MOVE_SPEED: float = 1.0

    OBSTACLE_WIDTH: int = 70
    OBSTACLE_HEIGHT: int = 90
    OBSTACLE_SPEED: float = 5.5
    OBSTACLE_SPAWN_INTERVAL_SECONDS: float = 900
    KEYBOARD_DEBUG_CONTROL: bool = True

    def __post_init__(self):
        if self.CLASS_TO_COMMAND is None:
            self.CLASS_TO_COMMAND = {
                0: "left",
                1: "right",
            }


CFG = Config()

BANDS = {
    "theta": (4, 8),
    "mu": (8, 13),
    "beta": (13, 30),
    "mi": (8, 30),
}

RATIO_PAIRS = [
    ("mu", "beta"),
    ("mu", "theta"),
    ("beta", "theta"),
    ("mi", "theta"),
    ("mu", "mi"),
    ("beta", "mi"),
]

class EEGMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def safe_array(x):
    return np.nan_to_num(
        np.asarray(x, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def safe_value(x):
    try:
        if np.isnan(x) or np.isinf(x):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def safe_log(x):
    return np.log(float(x) + 1e-12)


def welch_psd(signal, fs):
    signal = safe_array(signal)

    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=min(256, len(signal)),
        noverlap=min(128, len(signal) // 2),
    )

    psd = np.nan_to_num(
        psd,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return freqs, psd + 1e-12


def bandpower_welch(signal, fs, low, high):
    freqs, psd = welch_psd(signal, fs)
    mask = (freqs >= low) & (freqs <= high)

    if not np.any(mask):
        return 0.0

    return safe_value(np.mean(psd[mask]))


def covariance_matrix(trial):
    trial = safe_array(trial)
    cov = trial @ trial.T
    cov /= np.trace(cov) + 1e-12
    return cov


def fit_csp(X_train_selected, y_train, n_components=4):
    classes = np.unique(y_train)

    if len(classes) != 2:
        raise ValueError("CSP expects binary labels.")

    n_channels = X_train_selected.shape[1]

    if n_channels < 2:
        return None

    n_components = min(n_components, n_channels)

    if n_components % 2 != 0:
        n_components -= 1

    if n_components < 2:
        return None

    cov_0 = np.mean(
        [covariance_matrix(x) for x in X_train_selected[y_train == classes[0]]],
        axis=0,
    )

    cov_1 = np.mean(
        [covariance_matrix(x) for x in X_train_selected[y_train == classes[1]]],
        axis=0,
    )

    eigvals, eigvecs = eigh(cov_0, cov_0 + cov_1)

    ix = np.argsort(eigvals)
    eigvecs = eigvecs[:, ix]

    half = n_components // 2

    filters = np.concatenate(
        [
            eigvecs[:, :half],
            eigvecs[:, -half:],
        ],
        axis=1,
    )

    return filters.T


def csp_logvar_features(trial_selected, csp_filters):
    trial_selected = safe_array(trial_selected)
    projected = csp_filters @ trial_selected

    var = np.var(projected, axis=1)
    norm_var = var / (np.sum(var) + 1e-12)

    return np.log(norm_var + 1e-12)


def extract_trial_features(trial_selected, selected_labels, fs):
    trial_selected = safe_array(trial_selected)

    row = []
    names = []

    power_cache = {}

    for ch_i, label in enumerate(selected_labels):
        sig = trial_selected[ch_i]
        power_cache[label] = {}

        for band_name, (low, high) in BANDS.items():
            p = bandpower_welch(sig, fs, low, high)
            power_cache[label][band_name] = p

            row.append(p)
            names.append(f"{label}_{band_name}_power")

            row.append(safe_log(p))
            names.append(f"{label}_log_{band_name}_power")

        var = np.var(sig)
        std = np.std(sig)
        ptp = np.ptp(sig)
        rms = np.sqrt(np.mean(sig ** 2))
        mean_abs = np.mean(np.abs(sig))

        row.extend([
            safe_log(var),
            std,
            ptp,
            rms,
            mean_abs,
        ])

        names.extend([
            f"{label}_logvar",
            f"{label}_std",
            f"{label}_ptp",
            f"{label}_rms",
            f"{label}_mean_abs",
        ])

    for label in selected_labels:
        for a, b in RATIO_PAIRS:
            pa = power_cache[label][a]
            pb = power_cache[label][b]

            row.append(pa / (pb + 1e-12))
            names.append(f"{label}_{a}_over_{b}")

            row.append(safe_log(pa) - safe_log(pb))
            names.append(f"{label}_log_{a}_minus_log_{b}")

    from itertools import combinations

    for i, j in combinations(range(len(selected_labels)), 2):
        label_a = selected_labels[i]
        label_b = selected_labels[j]

        sig_a = trial_selected[i]
        sig_b = trial_selected[j]

        bipolar = sig_a - sig_b
        pair_label = f"{label_a}_minus_{label_b}"

        for band_name, (low, high) in BANDS.items():
            pa = power_cache[label_a][band_name]
            pb = power_cache[label_b][band_name]

            diff = pa - pb
            logdiff = safe_log(pa) - safe_log(pb)
            lateralization = (pa - pb) / (pa + pb + 1e-12)

            row.extend([
                diff,
                logdiff,
                lateralization,
            ])

            names.extend([
                f"{label_a}_{label_b}_{band_name}_power_diff",
                f"{label_a}_{label_b}_{band_name}_log_power_diff",
                f"{label_a}_{label_b}_{band_name}_lateralization",
            ])

            bp = bandpower_welch(bipolar, fs, low, high)

            row.append(bp)
            names.append(f"{pair_label}_{band_name}_power")

            row.append(safe_log(bp))
            names.append(f"{pair_label}_log_{band_name}_power")

        row.append(safe_log(np.var(bipolar)))
        names.append(f"{pair_label}_logvar")

        row.append(np.std(bipolar))
        names.append(f"{pair_label}_std")

        row.append(np.ptp(bipolar))
        names.append(f"{pair_label}_ptp")

    row = [safe_value(v) for v in row]
    return row, names


def covariance_geometry_features(trial_selected):
    trial_selected = safe_array(trial_selected)

    if trial_selected.shape[0] < 2:
        return [], []

    cov = np.cov(trial_selected)

    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.abs(eigvals))[::-1]
    eigvals_norm = eigvals / (np.sum(eigvals) + 1e-12)

    vals = [
        eigvals_norm[0],
        eigvals_norm[1] if len(eigvals_norm) > 1 else 0.0,
        eigvals_norm[0] / (eigvals_norm[-1] + 1e-12),
        -np.sum(eigvals_norm * np.log(eigvals_norm + 1e-12)),
    ]

    names = [
        "cov_eigen_1",
        "cov_eigen_2",
        "cov_condition_ratio",
        "cov_eigen_entropy",
    ]

    vals = [safe_value(v) for v in vals]
    return vals, names


class MotorChannelFeatureExtractor:
    def __init__(
        self,
        c3_idx,
        cz_idx,
        c4_idx,
        fs=250,
        use_csp=True,
        csp_components=4,
        use_covariance_geometry=True,
    ):
        self.c3_idx = c3_idx
        self.cz_idx = cz_idx
        self.c4_idx = c4_idx

        self.fs = fs
        self.use_csp = use_csp
        self.csp_components = csp_components
        self.use_covariance_geometry = use_covariance_geometry

        self.selected_indices = None
        self.selected_labels = None

        self.csp_filters = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def _select_motor_channels(self, X):
        return X[:, self.selected_indices, :]

    def set_fitted_state(
        self,
        selected_indices,
        selected_labels,
        csp_filters,
        feature_scaler,
        feature_names,
    ):
        self.selected_indices = list(selected_indices)
        self.selected_labels = list(selected_labels)
        self.csp_filters = csp_filters
        self.scaler = feature_scaler
        self.feature_names = list(feature_names)

    def transform(self, X):
        if self.selected_indices is None:
            raise RuntimeError("Extractor has no fitted state loaded.")

        if X.ndim != 3:
            raise ValueError(f"X must have shape (trials, channels, samples), got {X.shape}")

        X_selected = self._select_motor_channels(X)

        raw_features, _ = self._extract_raw(X_selected)

        raw_features = np.nan_to_num(
            raw_features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        scaled_features = self.scaler.transform(raw_features)

        scaled_features = np.nan_to_num(
            scaled_features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return scaled_features

    def _extract_raw(self, X_selected):
        all_features = []
        final_names = None

        for trial_i in range(X_selected.shape[0]):
            trial = X_selected[trial_i]

            trial_features = []
            names = []

            f, n = extract_trial_features(
                trial_selected=trial,
                selected_labels=self.selected_labels,
                fs=self.fs,
            )

            trial_features.extend(f)
            names.extend(n)

            if self.csp_filters is not None:
                csp_feats = csp_logvar_features(trial, self.csp_filters)

                for i, v in enumerate(csp_feats):
                    trial_features.append(safe_value(v))
                    names.append(f"csp_{i}_log_normalized_variance")

            if self.use_covariance_geometry:
                cov_feats, cov_names = covariance_geometry_features(trial)
                trial_features.extend(cov_feats)
                names.extend(cov_names)

            trial_features = [safe_value(v) for v in trial_features]
            all_features.append(trial_features)

            if final_names is None:
                final_names = names

        features = np.array(all_features, dtype=np.float32)

        features = np.nan_to_num(
            features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return features, final_names

def preprocess_live_window(raw_eeg: np.ndarray, cfg: Config) -> np.ndarray:

    if raw_eeg.ndim != 2:
        raise ValueError(f"raw_eeg must be (channels, samples), got {raw_eeg.shape}")

    if raw_eeg.shape[1] != cfg.WINDOW_SAMPLES:
        raise ValueError(
            f"Expected {cfg.WINDOW_SAMPLES} samples, got {raw_eeg.shape[1]}"
        )

    X = raw_eeg.copy().astype(np.float64)

    # Drop same bad channels as notebook.
    if len(cfg.BAD_CHANNELS) > 0:
        X = np.delete(X, cfg.BAD_CHANNELS, axis=0)

    # Process each channel independently.
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

    # Remove bandpass edge artifacts from both sides.
    X = X[:, cfg.CROP_START:-cfg.CROP_END_FROM_LAST]

    if X.shape[1] != cfg.EXPECTED_FINAL_SAMPLES:
        raise ValueError(
            f"Final sample count mismatch. Expected {cfg.EXPECTED_FINAL_SAMPLES}, got {X.shape[1]}"
        )

    return X[np.newaxis, :, :]

class LiveEEGClassifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cpu")

        checkpoint = joblib.load(cfg.CHECKPOINT_PATH)

        self.selected_features = np.asarray(checkpoint["selected_features"], dtype=int)
        self.final_scaler = checkpoint["final_scaler"]

        extractor_state = checkpoint["extractor_state"]
        extractor_params = checkpoint["extractor_params"]

        self.extractor = MotorChannelFeatureExtractor(
            c3_idx=extractor_params["c3_idx"],
            cz_idx=extractor_params["cz_idx"],
            c4_idx=extractor_params["c4_idx"],
            fs=extractor_params["fs"],
            use_csp=extractor_params["use_csp"],
            csp_components=extractor_params["csp_components"],
            use_covariance_geometry=extractor_params["use_covariance_geometry"],
        )

        self.extractor.set_fitted_state(
            selected_indices=extractor_state["selected_indices"],
            selected_labels=extractor_state["selected_labels"],
            csp_filters=extractor_state["csp_filters"],
            feature_scaler=extractor_state["feature_scaler"],
            feature_names=extractor_state["feature_names"],
        )

        input_dim = checkpoint["input_dim"]
        num_classes = checkpoint["num_classes"]

        self.model = EEGMLP(
            input_dim=input_dim,
            num_classes=num_classes,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.class_to_command = checkpoint.get("class_to_command", cfg.CLASS_TO_COMMAND)

        print("\nLoaded checkpoint:", cfg.CHECKPOINT_PATH)
        print("Input dim:", input_dim)
        print("Num classes:", num_classes)
        print("Selected features:", self.selected_features)
        print("Class mapping:", self.class_to_command)
        print()

    def predict_from_raw_window(self, raw_eeg: np.ndarray) -> Dict:
        X_window = preprocess_live_window(raw_eeg, self.cfg)

        X_features = self.extractor.transform(X_window)
        X_selected = X_features[:, self.selected_features]
        X_scaled = self.final_scaler.transform(X_selected)

        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])

        command = self.class_to_command.get(pred_class, "none")

        return {
            "class": pred_class,
            "command": command,
            "confidence": confidence,
            "probabilities": probs,
        }

class DecisionStabilizer:

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.candidate_class = None
        self.candidate_command = None
        self.candidate_count = 0

        self.last_accepted_time = 0.0
        self.last_accepted_command = "left"

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

        # Same confident class as before.
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

        # Accept decision.
        self.last_accepted_time = now
        self.last_accepted_command = command

        # Reset candidate so a new repeated command is required next time.
        self.candidate_class = None
        self.candidate_command = None
        self.candidate_count = 0

        self.status = f"accepted {command}"
        return command, self.status

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

        # Keep buffer comfortably larger than window.
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

class TopDownCarGame:
    def __init__(self, cfg: Config, classifier: LiveEEGClassifier, stream: EEGStream):
        self.cfg = cfg
        self.classifier = classifier
        self.stream = stream
        self.stabilizer = DecisionStabilizer(cfg)

        pygame.init()
        pygame.display.set_caption("BrainDance EEG Car Game")

        self.screen = pygame.display.set_mode((cfg.SCREEN_WIDTH, cfg.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Arial", 24)
        self.small_font = pygame.font.SysFont("Arial", 19)
        self.big_font = pygame.font.SysFont("Arial", 42, bold=True)

        self.road_x = (cfg.SCREEN_WIDTH - cfg.ROAD_WIDTH) // 2
        self.left_lane_x = self.road_x + cfg.LANE_WIDTH // 2
        self.right_lane_x = self.road_x + cfg.LANE_WIDTH + cfg.LANE_WIDTH // 2

        self.target_lane = "left"
        self.car_x = float(self.left_lane_x)
        self.car_y = cfg.CAR_Y

        self.obstacles = []
        self.last_spawn_time = time.time()

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

    def lane_center(self, lane: str) -> int:
        if lane == "left":
            return self.left_lane_x
        if lane == "right":
            return self.right_lane_x
        return int(self.car_x)

    def spawn_obstacle(self):
        lane = random.choice(["left", "right"])
        x = self.lane_center(lane)
        y = -self.cfg.OBSTACLE_HEIGHT

        rect = pygame.Rect(
            x - self.cfg.OBSTACLE_WIDTH // 2,
            y,
            self.cfg.OBSTACLE_WIDTH,
            self.cfg.OBSTACLE_HEIGHT,
        )

        self.obstacles.append({
            "lane": lane,
            "rect": rect,
        })

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
                self.target_lane = accepted_command

        except Exception as e:
            self.latest_result = {
                "class": None,
                "command": f"error: {type(e).__name__}",
                "confidence": 0.0,
                "probabilities": np.array([0.0, 0.0]),
            }
            self.decision_status = "inference error"
            print("Inference error:", repr(e))

    def handle_keyboard_debug(self):
        if not self.cfg.KEYBOARD_DEBUG_CONTROL:
            return

        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.target_lane = "left"

        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.target_lane = "right"

    def update_game_state(self):
        if self.game_over:
            return

        target_x = self.lane_center(self.target_lane)

        if abs(self.car_x - target_x) < self.cfg.CAR_LANE_MOVE_SPEED:
            self.car_x = float(target_x)
        elif self.car_x < target_x:
            self.car_x += self.cfg.CAR_LANE_MOVE_SPEED
        else:
            self.car_x -= self.cfg.CAR_LANE_MOVE_SPEED

        now = time.time()

        if now - self.last_spawn_time >= self.cfg.OBSTACLE_SPAWN_INTERVAL_SECONDS:
            self.spawn_obstacle()
            self.last_spawn_time = now

        for obstacle in self.obstacles:
            obstacle["rect"].y += int(self.cfg.OBSTACLE_SPEED)

        self.obstacles = [
            obs for obs in self.obstacles
            if obs["rect"].y < self.cfg.SCREEN_HEIGHT + 100
        ]

        self.score += 1

        car_rect = pygame.Rect(
            int(self.car_x) - self.cfg.CAR_WIDTH // 2,
            self.car_y,
            self.cfg.CAR_WIDTH,
            self.cfg.CAR_HEIGHT,
        )

        for obstacle in self.obstacles:
            if car_rect.colliderect(obstacle["rect"]):
                self.game_over = True
                break

    def draw_road(self):
        # Green ground
        self.screen.fill((40, 150, 60))

        # Road
        road_rect = pygame.Rect(
            self.road_x,
            0,
            self.cfg.ROAD_WIDTH,
            self.cfg.SCREEN_HEIGHT,
        )
        pygame.draw.rect(self.screen, (45, 45, 45), road_rect)

        # Lane divider
        center_x = self.road_x + self.cfg.ROAD_WIDTH // 2

        dash_height = 45
        gap = 30

        for y in range(-dash_height, self.cfg.SCREEN_HEIGHT + dash_height, dash_height + gap):
            pygame.draw.rect(
                self.screen,
                (230, 230, 230),
                pygame.Rect(center_x - 5, y, 10, dash_height),
            )

        # Road borders
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

    def draw_hud(self):
        result = self.latest_result

        command_text = str(result["command"]).upper()
        class_text = f"Class: {result['class']}"
        conf_text = f"Conf: {result['confidence']:.2f}"
        lane_text = f"Lane: {self.target_lane.upper()}"
        score_text = f"Score: {self.score}"
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
            "A/D debug" if self.cfg.KEYBOARD_DEBUG_CONTROL else "",
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

        if probs is not None and len(probs) >= 2:
            p_text = f"P[left/class0]={probs[0]:.2f}    P[right/class1]={probs[1]:.2f}"
            surf = self.small_font.render(p_text, True, (255, 255, 255))
            self.screen.blit(surf, (15, self.cfg.SCREEN_HEIGHT - 36))

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
        self.target_lane = "left"
        self.car_x = float(self.left_lane_x)
        self.obstacles = []
        self.last_spawn_time = time.time()
        self.score = 0
        self.game_over = False
        self.stabilizer = DecisionStabilizer(self.cfg)
        self.decision_status = "waiting"

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
                self.update_inference()
                self.handle_keyboard_debug()
                self.update_game_state()

            self.draw_road()
            self.draw_obstacles()
            self.draw_car()
            self.draw_hud()

            pygame.display.flip()

        pygame.quit()

def main():
    print("\nBrainDance Live Inference Car Game")
    print("----------------------------------")
    print(f"Window samples: {CFG.WINDOW_SAMPLES}")
    print(f"Crop: {CFG.CROP_START} front, {CFG.CROP_END_FROM_LAST} rear")
    print(f"Final samples: {CFG.WINDOW_SAMPLES - CFG.CROP_START - CFG.CROP_END_FROM_LAST}")
    print(f"Inference interval: {CFG.INFERENCE_INTERVAL_SECONDS}s")
    print(f"Confidence threshold: {CFG.CONFIDENCE_THRESHOLD}")
    print(f"Required repeats: {CFG.REQUIRED_CONSECUTIVE_PREDICTIONS}")
    print(f"Cooldown: {CFG.DECISION_COOLDOWN_SECONDS}s")
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
