import time
import cv2
import numpy as np
import torch
from collections import deque

from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes

import torch.nn as nn

# =========================
# CONFIG
# =========================
BOARD_ID = 0
SERIAL_PORT = "COM3"
FS = 250
WINDOW_SIZE = FS      # 4 seconds
SEQ_LEN = 5                # MUST match training
SMOOTHING = 5
CAMERA_INDEX = 0

# =========================
# LSTM MODEL (MATCH TRAINING)
# =========================
class EyeStateLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)
    

# =========================
#Helper Functions
def draw_text_box(img, text, pos, font, scale, text_color, bg_color, thickness=2, padding=6):
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos

    cv2.rectangle(
        img,
        (x - padding, y - h - padding),
        (x + w + padding, y + baseline + padding),
        bg_color,
        -1
    )

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )

# =========================
# LOAD MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

checkpoint = torch.load("Eye Classifier/assets/eye_model.pth", map_location=device)

FEATURE_DIM = 2000
SEQ_LEN = 5

model = EyeStateLSTM(FEATURE_DIM).to(device)
model.load_state_dict(checkpoint["model_state"])
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

sequence_buffer = deque(maxlen=SEQ_LEN)
pred_buffer = deque(maxlen=SMOOTHING)

print("Live LSTM EEG inference started (press Q to quit)")

# =========================
# REALTIME LOOP
# =========================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data = board.get_current_board_data(WINDOW_SIZE)

        if data.shape[1] < WINDOW_SIZE:
            continue

        # Explicitly select EEG channels only (ignore marker channel)
        eeg = data[eeg_channels, :].copy()

        # FILTER ONLY ONCE PER WINDOW (safe)
        for ch in range(eeg.shape[0]):
            DataFilter.perform_bandpass(
                eeg[ch],
                FS,
                0.5,
                40.0,
                4,
                FilterTypes.BUTTERWORTH.value,
                0
            )
            DataFilter.perform_bandstop(
                eeg[ch],
                FS,
                48.0,
                52.0,
                4,
                FilterTypes.BUTTERWORTH.value,
                0
            )

        # --- FLATTEN WINDOW ---
        window_flat = eeg.reshape(-1)  # (8*250 = 2000,)
        sequence_buffer.append(window_flat)

        if len(sequence_buffer) < SEQ_LEN:
            cv2.imshow("EEG Live LSTM Classification", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # --- LSTM INFERENCE ---
        seq = np.array(sequence_buffer).reshape(1, SEQ_LEN, -1)

        with torch.no_grad():
            X = torch.tensor(seq, dtype=torch.float32).to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        pred_buffer.append(pred)

        if len(pred_buffer) == SMOOTHING:
            final_pred = max(set(pred_buffer), key=pred_buffer.count)

            label = "EYES OPEN" if final_pred == 0 else "EYES CLOSED"
            color = (0, 255, 0) if final_pred == 0 else (0, 0, 255)

            logits_np = logits.cpu().numpy()[0]
            probs_np = probs.cpu().numpy()[0]

            draw_text_box(
                frame,
                label,
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                color,
                (0, 0, 0),
                thickness=3
            )

            draw_text_box(
                frame,
                f"Logits: [{logits_np[0]:.1f}, {logits_np[1]:.1f}]",
                (40, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                (0, 0, 0),
                thickness=2
            )

            draw_text_box(
                frame,
                f"Probs: Open={probs_np[0]:.2f} Closed={probs_np[1]:.2f}",
                (40, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                (0, 0, 0),
                thickness=2
            )

        cv2.imshow("EEG Live LSTM Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    board.stop_stream()
    board.release_session()