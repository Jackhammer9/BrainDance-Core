import time
import numpy as np
import serial

from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import (
    DataFilter,
    DetrendOperations,
    WindowOperations,
    NoiseTypes,
)

MIN_SCORE = 0.200
MAX_SCORE = 0.400

BOARD_ID = 0
SERIAL_PORT = "COM3"
FS = 250

arduino = serial.Serial("COM5", 115200, timeout=1)  # change COM5
time.sleep(2)

CH_C3 = 3   # left motor cortex -> RIGHT hand
CH_C4 = 5   # right motor cortex -> LEFT hand

WINDOW_SECONDS = 4
WINDOW_SAMPLES = WINDOW_SECONDS * FS

BASELINE_SECONDS = 25
UPDATE_SECONDS = 0.25

MU_LOW = 8.0
MU_HIGH = 13.0

params = BrainFlowInputParams()
params.serial_port = SERIAL_PORT

board = BoardShim(BOARD_ID, params)
board.prepare_session()

EEG_CHANNELS = BoardShim.get_eeg_channels(BOARD_ID)

print("EEG rows:", EEG_CHANNELS)
print("C3 EEG index:", CH_C3, "raw row:", EEG_CHANNELS[CH_C3])
print("C4 EEG index:", CH_C4, "raw row:", EEG_CHANNELS[CH_C4])

board.start_stream()
time.sleep(5)


def compute_mu_power(sig):
    sig = np.array(sig, dtype=np.float64).copy()

    DataFilter.detrend(sig, DetrendOperations.LINEAR.value)
    DataFilter.remove_environmental_noise(sig, FS, NoiseTypes.FIFTY.value)

    nfft = 512

    psd = DataFilter.get_psd_welch(
        sig,
        nfft,
        nfft // 2,
        FS,
        WindowOperations.BLACKMAN_HARRIS.value
    )

    return DataFilter.get_band_power(psd, MU_LOW, MU_HIGH)


def get_window():
    data = board.get_current_board_data(WINDOW_SAMPLES)

    if data.shape[1] < WINDOW_SAMPLES:
        return None

    return data[EEG_CHANNELS, :]


def collect_baseline():
    print(f"Collecting {BASELINE_SECONDS}s baseline. Stay relaxed.")

    c3_powers = []
    c4_powers = []

    t0 = time.time()

    while time.time() - t0 < BASELINE_SECONDS:
        eeg = get_window()

        if eeg is None:
            time.sleep(0.05)
            continue

        c3_power = compute_mu_power(eeg[CH_C3])
        c4_power = compute_mu_power(eeg[CH_C4])

        c3_powers.append(c3_power)
        c4_powers.append(c4_power)

        print(f"baseline C3={c3_power:.6f}, C4={c4_power:.6f}")

        time.sleep(UPDATE_SECONDS)

    baseline_c3 = float(np.median(c3_powers))
    baseline_c4 = float(np.median(c4_powers))

    print("\nBaseline ready:")
    print(f"C3 baseline: {baseline_c3:.6f}")
    print(f"C4 baseline: {baseline_c4:.6f}")

    return baseline_c3, baseline_c4


def classify_forced_binary(c3_power, c4_power, baseline_c3, baseline_c4):
    erd_c3 = (c3_power - baseline_c3) / (baseline_c3)
    erd_c4 = (c4_power - baseline_c4) / (baseline_c4)

    score = erd_c3 - erd_c4

    candidate = "IGNORE"

    # accept only clean score range: 0.200 to 0.400
    if MIN_SCORE <= abs(score) <= MAX_SCORE:
        if score < 0:
            candidate = "RIGHT"
        else:
            candidate = "LEFT"

    return candidate, erd_c3, erd_c4, score


try:
    baseline_c3, baseline_c4 = collect_baseline()

    print("\nStarting forced binary ERD asymmetry detection.")
    print("It will always classify either LEFT or RIGHT.")
    print("Only state changes will be printed.\n")

    last_output = None

    while True:
        eeg = get_window()

        if eeg is None:
            continue

        c3_power = compute_mu_power(eeg[CH_C3])
        c4_power = compute_mu_power(eeg[CH_C4])

        candidate, erd_c3, erd_c4, score = classify_forced_binary(
            c3_power,
            c4_power,
            baseline_c3,
            baseline_c4
        )

        if candidate != last_output:
            print(
                f"STATE CHANGE -> {candidate} | "
                f"ERD_C3={erd_c3:+.3f} "
                f"ERD_C4={erd_c4:+.3f} | "
                f"score={score:+.3f}"
            )

            if candidate == "LEFT":
                arduino.write(b'1')
            else:
                arduino.write(b'0')

            last_output = candidate

        time.sleep(UPDATE_SECONDS)

finally:
    print("Stopping...")
    try:
        board.stop_stream()
    except Exception:
        pass

    try:
        board.release_session()
    except Exception:
        pass