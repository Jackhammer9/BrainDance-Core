import time
import numpy as np
import random

from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes

# =====================
# CONFIG
# =====================
BOARD_ID = 0
SERIAL_PORT = "COM3"
FS = 250
WIN_LEN = 2.5
SAMPLES_PER_WINDOW = int(FS * WIN_LEN)

TRIALS_PER_CLASS = 3

MARKERS = {
    "idle": 1,
    "attack": 2,
    "parry": 3,
}

SUBJECT_NAME = "ArnavBajaj"

# =====================
# SETUP BOARD
# =====================
params = BrainFlowInputParams()
params.serial_port = SERIAL_PORT

board = BoardShim(BOARD_ID, params)
print("Preparing session...")
board.prepare_session()

print("Starting stream...")
board.start_stream()

print("Warming up...")
time.sleep(5)

# =====================
# TRIAL LIST
# =====================
trial_list = (
    ["idle"] * TRIALS_PER_CLASS +
    ["attack"] * TRIALS_PER_CLASS +
    ["parry"] * TRIALS_PER_CLASS
)

random.shuffle(trial_list)

# =====================
# RUN TRIALS
# =====================
for i, trial in enumerate(trial_list):

    print(f"\nTrial {i+1}/{len(trial_list)}")
    print("Baseline...")
    time.sleep(2)

    print(f"THINK: {trial.upper()}")
    board.insert_marker(MARKERS[trial])
    time.sleep(WIN_LEN)

    print("Rest...")
    time.sleep(2)

print("\nCollection complete.")

# =====================
# GET DATA
# =====================
data = board.get_board_data()

board.stop_stream()
board.release_session()

print("Raw data shape:", data.shape)

# =====================
# EXTRACT CHANNELS
# =====================
eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
marker_channel = BoardShim.get_marker_channel(BOARD_ID)

eeg_data = data[eeg_channels, :]
markers = data[marker_channel, :]

# =====================
# FILTER CONTINUOUS
# =====================
print("Filtering continuous EEG...")

for ch in range(eeg_data.shape[0]):

    DataFilter.perform_bandpass(
        eeg_data[ch],
        FS,
        0.5,
        40.0,
        4,
        FilterTypes.BUTTERWORTH.value,
        0
    )

    DataFilter.perform_bandstop(
        eeg_data[ch],
        FS,
        48.0,
        52.0,
        4,
        FilterTypes.BUTTERWORTH.value,
        0
    )

print("Filtering complete.")

# =====================
# WINDOW USING MARKERS
# =====================
marker_indices = np.where(markers != 0)[0]

X = []
y = []

for idx in marker_indices:
    start = idx
    end = idx + SAMPLES_PER_WINDOW

    if end <= eeg_data.shape[1]:
        window = eeg_data[:, start:end]
        label_value = int(markers[idx])

        if window.shape[1] == SAMPLES_PER_WINDOW:
            X.append(window)
            y.append(label_value)

X = np.array(X)
y = np.array(y)

print("Total markers found:", len(marker_indices))
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Label counts:", np.unique(y, return_counts=True))

# =====================
# SAVE
# =====================
random_name = random.randint(10000000, 99999999)

np.savez(
    f"{SUBJECT_NAME}_{random_name}.npz",
    X=X,
    y=y,
    fs=FS
)

print("Saved dataset.")