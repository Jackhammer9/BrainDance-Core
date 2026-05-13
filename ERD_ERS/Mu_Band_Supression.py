import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import (
    DataFilter,
    DetrendOperations,
    WindowOperations,
    NoiseTypes,
)

# =========================
# CONFIG
# =========================
BOARD_ID = 0
SERIAL_PORT = "COM3"
FS = 250

CH_LEFT = 3   
CH_RIGHT = 5 

# Signal settings
WINDOW_SIZE = 1500          
UPDATE_MS = 50            # update every 200 ms
PLOT_WINDOW = 10           # show last 10 sec
MU_LOW = 8.0
MU_HIGH = 13
SMOOTH_ALPHA = 0.2         # lower = smoother

# Baseline
BASELINE_SECONDS = 40

# Detector thresholds
SUPPRESSION_THRESHOLD = -0.05  
DECISION_THRESHOLD = 0.02     
PERSISTENCE_STEPS = 3        

# If labels are flipped, set this True
FLIP_LEFT_RIGHT = False

# =========================
# GLOBALS
# =========================
running = True

times = []
left_powers = []
right_powers = []
score_values = []

baseline_left = None
baseline_right = None

decision_history = []
stable_state = "CALIBRATING"

start_time = None
calibration_done = False

# =========================
# BRAINFLOW SETUP
# =========================
params = BrainFlowInputParams()
params.serial_port = SERIAL_PORT

BoardShim.enable_dev_board_logger()
board = BoardShim(BOARD_ID, params)

print("Preparing session...")
board.prepare_session()
board.start_stream()

# =========================
# PLOT SETUP
# =========================
fig, ax = plt.subplots(figsize=(12, 6))

line_left, = ax.plot([], [], label="Ch4 mu power")
line_right, = ax.plot([], [], label="Ch6 mu power")
line_score, = ax.plot([], [], label="ERD asymmetry score")

txt_left = ax.text(0, 0, "", fontsize=9, ha="left", va="bottom")
txt_right = ax.text(0, 0, "", fontsize=9, ha="left", va="top")
txt_score = ax.text(0, 0, "", fontsize=9, ha="left", va="center")

status_text = fig.text(
    0.02, 0.95, "State: CALIBRATING", fontsize=14, weight="bold"
)
info_text = fig.text(
    0.02, 0.91, f"Baseline: collecting {BASELINE_SECONDS}s rest...", fontsize=10
)

ax.set_xlim(0, PLOT_WINDOW)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Value")
ax.set_title("Primitive ERD Detector")
ax.legend()

# =========================
# HELPERS
# =========================
def compute_mu_power(signal):
    x = np.array(signal, dtype=np.float64)

    DataFilter.detrend(x, DetrendOperations.LINEAR.value)
    DataFilter.remove_environmental_noise(x, FS, NoiseTypes.FIFTY.value)

    nfft = DataFilter.get_nearest_power_of_two(FS)
    psd = DataFilter.get_psd_welch(
        x,
        nfft,
        nfft // 2,
        FS,
        WindowOperations.BLACKMAN_HARRIS.value
    )

    return DataFilter.get_band_power(psd, MU_LOW, MU_HIGH)

def smooth_value(new_value, history, alpha=SMOOTH_ALPHA):
    if len(history) == 0:
        return new_value
    return alpha * new_value + (1 - alpha) * history[-1]

def classify(current_left, current_right, baseline_left, baseline_right):
    """
    Assumes:
      CH_LEFT  = left hemisphere motor area
      CH_RIGHT = right hemisphere motor area

    Right hand imagery -> stronger suppression on left hemisphere
    Left hand imagery  -> stronger suppression on right hemisphere
    """

    erd_left = (current_left - baseline_left) / (baseline_left + 1e-12)
    erd_right = (current_right - baseline_right) / (baseline_right + 1e-12)

    # More negative = more suppression
    score = erd_left - erd_right

    candidate = "NONE"

    if (erd_left < SUPPRESSION_THRESHOLD) or (erd_right < SUPPRESSION_THRESHOLD):
        if score < -DECISION_THRESHOLD:
            candidate = "RIGHT"
        elif score > DECISION_THRESHOLD:
            candidate = "LEFT"

    if FLIP_LEFT_RIGHT:
        if candidate == "LEFT":
            candidate = "RIGHT"
        elif candidate == "RIGHT":
            candidate = "LEFT"

    return candidate, erd_left, erd_right, score

def persistent_decision(candidate):
    global decision_history

    decision_history.append(candidate)
    if len(decision_history) > PERSISTENCE_STEPS:
        decision_history.pop(0)

    if len(decision_history) < PERSISTENCE_STEPS:
        return "NONE"

    if all(x == decision_history[0] for x in decision_history):
        return decision_history[0]

    return "NONE"

# =========================
# BASELINE CALIBRATION
# =========================
def collect_baseline():
    """
    Collect resting baseline for BASELINE_SECONDS seconds.
    User should stay relaxed, no movement, no imagery.
    """
    print(f"Collecting {BASELINE_SECONDS}s baseline... stay relaxed.")

    baseline_left_vals = []
    baseline_right_vals = []

    t0 = time.time()

    while time.time() - t0 < BASELINE_SECONDS:
        data = board.get_current_board_data(WINDOW_SIZE)
        data = data[board.get_eeg_channels(0), :]

        if data.shape[1] < WINDOW_SIZE:
            time.sleep(0.05)
            continue

        left_sig = data[CH_LEFT, :]
        right_sig = data[CH_RIGHT, :]

        p_left = compute_mu_power(left_sig)
        p_right = compute_mu_power(right_sig)

        baseline_left_vals.append(p_left)
        baseline_right_vals.append(p_right)

        time.sleep(UPDATE_MS / 1000.0)

    baseline_left = float(np.mean(baseline_left_vals))
    baseline_right = float(np.mean(baseline_right_vals))

    print(f"Baseline left : {baseline_left:.6f}")
    print(f"Baseline right: {baseline_right:.6f}")

    return baseline_left, baseline_right

# =========================
# UPDATE LOOP
# =========================
def update(frame):
    global running
    global stable_state
    global calibration_done

    if not running:
        return (
            line_left, line_right, line_score,
            txt_left, txt_right, txt_score,
            status_text, info_text
        )

    try:
        now = time.time() - start_time

        data = board.get_current_board_data(WINDOW_SIZE)
        data = data[board.get_eeg_channels(0), :]
        if data.shape[1] < WINDOW_SIZE:
            return (
                line_left, line_right, line_score,
                txt_left, txt_right, txt_score,
                status_text, info_text
            )

        sig_left = data[CH_LEFT, :]
        sig_right = data[CH_RIGHT, :]

        p_left = compute_mu_power(sig_left)
        p_right = compute_mu_power(sig_right)

        p_left = smooth_value(p_left, left_powers)
        p_right = smooth_value(p_right, right_powers)

        candidate, erd_left, erd_right, score = classify(
            p_left, p_right, baseline_left, baseline_right
        )

        stable_candidate = persistent_decision(candidate)
        stable_state = stable_candidate

        times.append(now)
        left_powers.append(p_left)
        right_powers.append(p_right)
        score_values.append(score)

        while times and (times[-1] - times[0] > PLOT_WINDOW):
            times.pop(0)
            left_powers.pop(0)
            right_powers.pop(0)
            score_values.pop(0)

        shifted_time = [t - times[0] for t in times]

        line_left.set_data(shifted_time, left_powers)
        line_right.set_data(shifted_time, right_powers)
        line_score.set_data(shifted_time, score_values)

        ax.set_xlim(0, PLOT_WINDOW)

        all_vals = left_powers + right_powers + score_values
        ymin = min(all_vals)
        ymax = max(all_vals)
        if ymin == ymax:
            ymax = ymin + 1e-12
        pad = 0.1 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

        x_last = shifted_time[-1]
        y_left = left_powers[-1]
        y_right = right_powers[-1]
        y_score = score_values[-1]

        txt_left.set_position((x_last + 0.05, y_left))
        txt_left.set_text(f"Ch4: {y_left:.3f}")

        txt_right.set_position((x_last + 0.05, y_right))
        txt_right.set_text(f"Ch6: {y_right:.3f}")

        txt_score.set_position((x_last + 0.05, y_score))
        txt_score.set_text(f"Score: {y_score:.3f}")

        status_text.set_text(f"State: {stable_state}")
        info_text.set_text(
            f"ERD_L={erd_left:.3f}   ERD_R={erd_right:.3f}   "
            f"cand={candidate}   stable={stable_state}"
        )

        print(
            f"Lpow={p_left:.4f}  Rpow={p_right:.4f}  "
            f"ERD_L={erd_left:.3f}  ERD_R={erd_right:.3f}  "
            f"Score={score:.3f}  cand={candidate}  stable={stable_state}"
        )

        return (
            line_left, line_right, line_score,
            txt_left, txt_right, txt_score,
            status_text, info_text
        )

    except Exception as e:
        print("Update error:", e)
        running = False
        return (
            line_left, line_right, line_score,
            txt_left, txt_right, txt_score,
            status_text, info_text
        )

def on_close(event):
    global running
    running = False
    print("Closing plot...")

fig.canvas.mpl_connect("close_event", on_close)

# =========================
# MAIN
# =========================
try:
    # let buffer fill a bit
    time.sleep(3)

    baseline_left, baseline_right = collect_baseline()
    calibration_done = True
    start_time = time.time()

    status_text.set_text("State: NONE")
    info_text.set_text(
        f"Baseline ready. L={baseline_left:.3f}, R={baseline_right:.3f}"
    )

    ani = animation.FuncAnimation(
        fig,
        update,
        interval=UPDATE_MS,
        blit=False,
        cache_frame_data=False
    )

    plt.show()

finally:
    running = False
    print("Stopping stream...")
    try:
        board.stop_stream()
    except Exception:
        pass
    try:
        board.release_session()
    except Exception:
        pass