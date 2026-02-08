import time
import cv2
import numpy as np
import torch
import joblib
from collections import deque
from scipy.signal import welch

from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes

import torch.nn as nn

# =========================
# CONFIG
# =========================
BOARD_ID = 0
SERIAL_PORT = "COM3"
FS = 250
WINDOW_SIZE = FS
SMOOTHING = 10
CAMERA_INDEX = 0

# =========================
# MODEL DEFINITION (MATCH TRAINING)
# =========================
class EyeClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# FEATURE EXTRACTION (MATCH TRAINING)
# =========================
def extract_features(signal, fs=250):
    f, pxx = welch(signal, fs=fs, nperseg=fs)

    theta_idx = (f >= 4) & (f < 8)
    alpha_idx = (f >= 8) & (f < 13)
    beta_idx  = (f >= 13) & (f < 30)

    theta = np.trapezoid(pxx[theta_idx], f[theta_idx])
    alpha = np.trapezoid(pxx[alpha_idx], f[alpha_idx])
    beta  = np.trapezoid(pxx[beta_idx],  f[beta_idx])

    total = theta + alpha + beta + 1e-10

    rel_theta = theta / total
    rel_alpha = alpha / total
    rel_beta  = beta  / total

    alpha_beta_ratio = alpha / (beta + 1e-10)

    psd_norm = pxx / (np.sum(pxx) + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

    var = np.var(signal)
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)

    mobility = np.sqrt(np.var(diff1) / (var + 1e-10))
    complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)

    # New Test Features
    mobility_theta = np.sqrt(np.var(np.diff(signal)) / (theta + 1e-10))
    complexity_theta = np.sqrt(np.var(np.diff(np.diff(signal))) / (theta + 1e-10)) / (mobility_theta + 1e-10)

    mobility_alpha = np.sqrt(np.var(np.diff(signal)) / (alpha + 1e-10))
    complexity_alpha = np.sqrt(np.var(np.diff(np.diff(signal))) / (alpha + 1e-10)) / (mobility_alpha + 1e-10)

    mobility_beta = np.sqrt(np.var(np.diff(signal)) / (beta + 1e-10))
    complexity_beta = np.sqrt(np.var(np.diff(np.diff(signal))) / (beta + 1e-10)) / (mobility_beta + 1e-10)

    #Extra Features
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    max_signal = np.max(signal)
    min_signal = np.min(signal)

    return [
        np.log(theta + 1e-10),
        np.log(alpha + 1e-10),
        np.log(beta + 1e-10),
        np.log(rel_theta + 1e-10),
        np.log(rel_alpha + 1e-10),
        np.log(rel_beta + 1e-10),
        np.log(alpha_beta_ratio + 1e-10),
        np.log(var + 1e-10),
        mobility,
        complexity,
        spectral_entropy,
        mobility_theta,
        complexity_theta,
        mobility_alpha,
        complexity_alpha,
        mobility_beta,
        complexity_beta,
        mean_signal,
        std_signal,
        max_signal,
        min_signal
    ]

# =========================
# LOAD MODEL + SCALER
# =========================
scaler = joblib.load("Eye Classifier/feature_scaler.pkl")
input_dim = scaler.mean_.shape[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = EyeClassifier(input_dim).to(device)
model.load_state_dict(torch.load("Eye Classifier/eye_model.pth", map_location=device))
model.eval()

# =========================
# BRAINFLOW SETUP
# =========================
params = BrainFlowInputParams()
params.serial_port = SERIAL_PORT
board = BoardShim(BOARD_ID, params)

board.prepare_session()
board.start_stream()

eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)

# =========================
# OPENCV SETUP
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)
pred_buffer = deque(maxlen=SMOOTHING)

print("Live EEG + Camera classification started (press Q to quit)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data = board.get_current_board_data(WINDOW_SIZE)
        if data.shape[1] < WINDOW_SIZE:
            continue

        eeg = data[eeg_channels, :].copy()

        # --- FILTER ---
        for ch in range(eeg.shape[0]):
            DataFilter.perform_bandpass(eeg[ch], FS, 0.5, 40, 4, FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(eeg[ch], FS, 48, 52, 4, FilterTypes.BUTTERWORTH.value, 0)

        # --- FEATURE EXTRACTION ---
        features = []
        for ch in range(eeg.shape[0]):
            features.extend(extract_features(eeg[ch]))

        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)

        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
            logits = model(features_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        logits_np = logits.cpu().numpy()[0]
        probs_np = probs.cpu().numpy()[0]

        pred_buffer.append(pred)

        if len(pred_buffer) == SMOOTHING:
            final_pred = max(set(pred_buffer), key=pred_buffer.count)
            label = "EYES OPEN" if final_pred == 0 else "EYES CLOSED"
            color = (0, 255, 0) if final_pred == 0 else (0, 0, 255)

            cv2.putText(frame, f"Thought: {label}", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

            cv2.putText(frame,
                        f"Logits: [{logits_np[0]:.1f}, {logits_np[1]:.1f}]",
                        (40, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.putText(frame,
                        f"Probs: Open={probs_np[0]:.2f} Closed={probs_np[1]:.2f}",
                        (40, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("EEG Live Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    board.stop_stream()
    board.release_session()