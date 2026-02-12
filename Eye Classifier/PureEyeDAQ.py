import time
import numpy as np
import random
from tqdm import tqdm

from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes

# =====================
# CONFIG
# =====================
BOARD_ID = 0                  # OpenBCI Cyton
SERIAL_PORT = "COM3"
FS = 250

TOTAL_RECORD_SEC = 15
WARMUP_SEC = 5

WINDOW_SEC = 1
LABEL_NAME = "eyes_closed"          # offline label only
LABEL_MARKER = 2             # offline label only

# =====================
# SETUP BOARD
# =====================
params = BrainFlowInputParams()
params.serial_port = SERIAL_PORT

BoardShim.enable_dev_board_logger()
board = BoardShim(BOARD_ID, params)

print("Preparing session...")
board.prepare_session()
board.start_stream()

# =====================
# RECORD WITH PROGRESS BAR
# =====================
print("Recording EEG...")
for _ in tqdm(range(TOTAL_RECORD_SEC), desc="Recording", unit="s"):
    time.sleep(1)

data = board.get_board_data()
board.stop_stream()
board.release_session()

print("Recording stopped.")
print("Raw data shape:", data.shape)

# =====================
# EXTRACT EEG + TIMESTAMPS
# =====================
eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
ts_channel = BoardShim.get_timestamp_channel(BOARD_ID)

eeg = data[eeg_channels]          # (C, T)
eeg_ts = data[ts_channel]         # (T,)

# =====================
# BRAINFLOW FILTERING
# =====================
print("Filtering EEG...")
for ch in tqdm(range(eeg.shape[0]), desc="Filtering channels"):
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

# =====================
# DISCARD WARMUP
# =====================
cut = FS * WARMUP_SEC
eeg = eeg[:, cut:]
eeg_ts = eeg_ts[cut:]

# =====================
# WINDOWING
# =====================
T = FS * WINDOW_SEC
X, y = [], []

for i in range(0, eeg.shape[1] - T, T):
    X.append(eeg[:, i:i + T])
    y.append(LABEL_MARKER)

X = np.array(X)
y = np.array(y)

# =====================
# SAVE
# =====================
random_name = random.randint(10000000, 99999999)

np.savez(
    f"Eye Classifier/raw/{LABEL_NAME}_{random_name}.npz",
    X=X,
    y=y,
    fs=FS,
    window_sec=WINDOW_SEC,
    bandpass_low=0.5,
    bandpass_high=40.0,
    bandstop_low=48.0,
    bandstop_high=52.0,
    warmup_sec=WARMUP_SEC
)

print("Done.")
print("X shape:", X.shape)
print("y shape:", y.shape)