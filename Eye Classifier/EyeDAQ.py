import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import random
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes

# ===================== CV SETUP =====================
model_path = "Eye Classifier/assets/face_landmarker.task"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def ear(landmarks, idx, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in idx]
    v1 = np.linalg.norm(np.subtract(pts[1], pts[5]))
    v2 = np.linalg.norm(np.subtract(pts[2], pts[4]))
    hdist = np.linalg.norm(np.subtract(pts[0], pts[3]))
    return (v1 + v2) / (2.0 * hdist + 1e-6)

options = vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

# ===================== EEG SETUP =====================
params = BrainFlowInputParams()
params.serial_port = "COM3"      # adjust if needed
board_id = 0                     # OpenBCI Cyton

board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream()

# ===================== RECORDING =====================
cap = cv2.VideoCapture(0)
cv_log = []

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    start_time = time.time()
    while time.time() - start_time < 1500:   # 5 minutes
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        state = -1
        if result.face_landmarks:
            lm = result.face_landmarks[0]
            ear_val = (ear(lm, LEFT_EYE, w, h) + ear(lm, RIGHT_EYE, w, h)) / 2.0

            if ear_val < 0.16:
                state = 1      # CLOSED
            elif ear_val > 0.20:
                state = 0      # OPEN
            else:
                state = -1     # AMBIGUOUS

        cv_log.append({
            "t_ns": ts_ms * 1_000_000,   # CV timestamp in ns
            "state": state
        })

        label_txt = "CLOSED" if state == 1 else "OPEN" if state == 0 else "UNK"
        cv2.putText(frame, label_txt, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("DAQ", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# ===================== STOP EEG =====================
data = board.get_board_data()
board.stop_stream()
board.release_session()

# ===================== EEG PROCESSING =====================
fs = BoardShim.get_sampling_rate(board_id)

eeg = data[BoardShim.get_eeg_channels(board_id)]
eeg_ts = data[BoardShim.get_timestamp_channel(board_id)]

for ch in range(eeg.shape[0]):
    DataFilter.perform_bandpass(
        eeg[ch], fs, 0.5, 40, 4, FilterTypes.BUTTERWORTH.value, 0
    )
    DataFilter.perform_bandstop(
        eeg[ch], fs, 48, 52, 4, FilterTypes.BUTTERWORTH.value, 0
    )

cut = fs * 30  # discard first 30 seconds
eeg = eeg[:, cut:]
eeg_ts = eeg_ts[cut:]

# ===================== ALIGN CV → EEG =====================
cv_times = np.array([x["t_ns"] for x in cv_log])
cv_states = np.array([x["state"] for x in cv_log])

window_size = fs
X, y = [], []

for i in range(0, eeg.shape[1] - window_size, window_size):
    t_center_ns = int(
        (eeg_ts[i] + eeg_ts[i + window_size - 1]) * 0.5 * 1e9
    )

    idx = np.argmin(np.abs(cv_times - t_center_ns))
    label = cv_states[idx]

 hjo    
    if label == -1:
        continue

    X.append(eeg[:, i:i + window_size])
    y.append(label)

X = np.array(X)
y = np.array(y)

randomName = random.randint(10000000, 99999999)

np.savez(
    f"Eye Classifier/raw/raw_eye_data_{randomName}.npz",
    X=X,
    y=y,
    fs=fs,
    window_size=window_size,
    step_size=window_size
)

print("Saved raw_eye_data.npz")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Label distribution:", dict(zip(*np.unique(y, return_counts=True))))