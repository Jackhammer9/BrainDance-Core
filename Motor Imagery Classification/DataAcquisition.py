import os
import time
import random
import threading
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
from tqdm import tqdm

import tkinter as tk
from tkinter import ttk, messagebox

from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes

BOARD_ID = 0
SERIAL_PORT = "COM3"
FS = 250

NUM_TRIALS_PER_CLASS = 15

FIXATION_SEC = 2
IMAGERY_SEC = 5
REST_SEC = 3
WARMUP_SEC = 5


CSP_START_AFTER_CUE = 1.0
CSP_END_AFTER_CUE = 4.0

SAVE_DIR = "Motor Imagery Classification/Datasets/Raw"

# Full-screen task display is better for cognitive load.
FULLSCREEN = True

# If True, shows only icons/cross/countdown during trials.
MINIMAL_TRIAL_TEXT = True

MARKERS = {
    "LEFT": 1,
    "RIGHT": 2,
}

def clean_name(name: str) -> str:
    cleaned = "".join(c for c in name.strip().upper() if c.isalnum() or c in "_-")
    return cleaned if cleaned else "SUBJECT"


def get_next_session_name(subject_folder: str, subject_code: str) -> str:
    existing = [
        f for f in os.listdir(subject_folder)
        if f.startswith(subject_code + "_") and f.endswith(".npz")
    ]

    nums = []
    for f in existing:
        try:
            nums.append(int(f.replace(".npz", "").split("_")[-1]))
        except ValueError:
            pass

    next_num = max(nums, default=0) + 1
    return f"{subject_code}_{next_num:02d}"


def label_to_name(label: int) -> str:
    return "LEFT" if label == MARKERS["LEFT"] else "RIGHT"


@dataclass
class SessionInfo:
    subject_name: str
    subject_code: str
    age: str
    gender: str
    mood: str
    session_name: str
    date_time: str
    task: str
    fixation_sec: float
    imagery_sec: float
    rest_sec: float
    warmup_sec: float
    num_trials_per_class: int
    fs: int
    board_id: int
    serial_port: str
    csp_start_after_cue: float
    csp_end_after_cue: float


# ============================================================
# GUI
# ============================================================

class MotorImageryDAQGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Motor Imagery DAQ")
        self.root.configure(bg="#050505")

        if FULLSCREEN:
            self.root.attributes("-fullscreen", True)

        self.root.bind("<Escape>", lambda event: self.exit_fullscreen())
        self.root.bind("<F11>", lambda event: self.toggle_fullscreen())

        self.board = None
        self.is_running = False
        self.abort_requested = False
        self.worker_thread = None

        self.trial_labels = []
        self.session_info = None
        self.subject_folder = None

        self.setup_style()
        self.build_start_screen()

    def setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TButton",
            font=("Segoe UI", 14),
            padding=10,
        )
        style.configure(
            "TEntry",
            font=("Segoe UI", 13),
            padding=6,
        )
        style.configure(
            "TLabel",
            background="#050505",
            foreground="#f0f0f0",
            font=("Segoe UI", 13),
        )

    def clear(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def exit_fullscreen(self):
        self.root.attributes("-fullscreen", False)

    def toggle_fullscreen(self):
        current = bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", not current)

    def build_start_screen(self):
        self.clear()

        frame = tk.Frame(self.root, bg="#050505")
        frame.pack(expand=True, fill="both")

        title = tk.Label(
            frame,
            text="BrainDance Motor Imagery DAQ",
            bg="#050505",
            fg="#ffffff",
            font=("Segoe UI", 30, "bold"),
        )
        title.pack(pady=(60, 10))

        subtitle = tk.Label(
            frame,
            text="Visual cue recording: fixation → imagine → rest",
            bg="#050505",
            fg="#aaaaaa",
            font=("Segoe UI", 15),
        )
        subtitle.pack(pady=(0, 35))

        form = tk.Frame(frame, bg="#050505")
        form.pack()

        self.subject_var = tk.StringVar()
        self.age_var = tk.StringVar()
        self.gender_var = tk.StringVar()
        self.mood_var = tk.StringVar()

        self.add_form_row(form, "Subject name", self.subject_var, 0)
        self.add_form_row(form, "Age", self.age_var, 1)
        self.add_form_row(form, "Gender", self.gender_var, 2)
        self.add_form_row(form, "Mood / condition", self.mood_var, 3)

        config_text = (
            f"Port: {SERIAL_PORT}    Board: {BOARD_ID}    "
            f"Trials: {NUM_TRIALS_PER_CLASS * 2}    "
            f"Fixation: {FIXATION_SEC}s    Imagery: {IMAGERY_SEC}s    Rest: {REST_SEC}s"
        )

        config = tk.Label(
            frame,
            text=config_text,
            bg="#050505",
            fg="#777777",
            font=("Consolas", 12),
        )
        config.pack(pady=25)

        button_row = tk.Frame(frame, bg="#050505")
        button_row.pack(pady=10)

        start_button = ttk.Button(
            button_row,
            text="Start Recording",
            command=self.start_clicked,
        )
        start_button.grid(row=0, column=0, padx=10)

        quit_button = ttk.Button(
            button_row,
            text="Quit",
            command=self.root.destroy,
        )
        quit_button.grid(row=0, column=1, padx=10)

        hint = tk.Label(
            frame,
            text="During trials: keep eyes on crosshair, do not move physically, imagine the shown direction.",
            bg="#050505",
            fg="#999999",
            font=("Segoe UI", 13),
        )
        hint.pack(pady=(30, 0))

    def add_form_row(self, parent, label, var, row):
        lbl = tk.Label(
            parent,
            text=label,
            bg="#050505",
            fg="#eeeeee",
            font=("Segoe UI", 13),
            width=18,
            anchor="e",
        )
        lbl.grid(row=row, column=0, padx=10, pady=8)

        entry = ttk.Entry(parent, textvariable=var, width=32)
        entry.grid(row=row, column=1, padx=10, pady=8)

    def start_clicked(self):
        subject_name = self.subject_var.get().strip()

        if not subject_name:
            messagebox.showerror("Missing subject", "Enter subject name.")
            return

        subject_code = clean_name(subject_name)
        self.subject_folder = os.path.join(SAVE_DIR, subject_code)
        os.makedirs(self.subject_folder, exist_ok=True)

        session_name = get_next_session_name(self.subject_folder, subject_code)

        self.session_info = SessionInfo(
            subject_name=subject_name,
            subject_code=subject_code,
            age=self.age_var.get().strip(),
            gender=self.gender_var.get().strip(),
            mood=self.mood_var.get().strip(),
            session_name=session_name,
            date_time=datetime.now().isoformat(),
            task="left_right_motor_imagery",
            fixation_sec=FIXATION_SEC,
            imagery_sec=IMAGERY_SEC,
            rest_sec=REST_SEC,
            warmup_sec=WARMUP_SEC,
            num_trials_per_class=NUM_TRIALS_PER_CLASS,
            fs=FS,
            board_id=BOARD_ID,
            serial_port=SERIAL_PORT,
            csp_start_after_cue=CSP_START_AFTER_CUE,
            csp_end_after_cue=CSP_END_AFTER_CUE,
        )

        self.trial_labels = (
            [MARKERS["LEFT"]] * NUM_TRIALS_PER_CLASS
            + [MARKERS["RIGHT"]] * NUM_TRIALS_PER_CLASS
        )
        random.shuffle(self.trial_labels)

        self.is_running = True
        self.abort_requested = False

        self.build_trial_screen()

        self.worker_thread = threading.Thread(target=self.run_daq_session, daemon=True)
        self.worker_thread.start()

    def build_trial_screen(self):
        self.clear()

        self.canvas = tk.Canvas(self.root, bg="#050505", highlightthickness=0)
        self.canvas.pack(expand=True, fill="both")

        self.status_var = tk.StringVar(value="Preparing...")
        self.trial_var = tk.StringVar(value="")
        self.countdown_var = tk.StringVar(value="")
        self.phase_var = tk.StringVar(value="")

        top_frame = tk.Frame(self.root, bg="#050505")
        top_frame.place(relx=0.5, rely=0.04, anchor="n")

        self.status_label = tk.Label(
            top_frame,
            textvariable=self.status_var,
            bg="#050505",
            fg="#bbbbbb",
            font=("Segoe UI", 16),
        )
        self.status_label.pack()

        self.trial_label = tk.Label(
            top_frame,
            textvariable=self.trial_var,
            bg="#050505",
            fg="#777777",
            font=("Segoe UI", 13),
        )
        self.trial_label.pack()

        bottom_frame = tk.Frame(self.root, bg="#050505")
        bottom_frame.place(relx=0.5, rely=0.93, anchor="s")

        self.phase_label = tk.Label(
            bottom_frame,
            textvariable=self.phase_var,
            bg="#050505",
            fg="#777777",
            font=("Segoe UI", 14),
        )
        self.phase_label.pack()

        self.countdown_label = tk.Label(
            bottom_frame,
            textvariable=self.countdown_var,
            bg="#050505",
            fg="#ffffff",
            font=("Segoe UI", 26, "bold"),
        )
        self.countdown_label.pack()

        abort_button = ttk.Button(
            self.root,
            text="Abort & Save",
            command=self.request_abort,
        )
        abort_button.place(relx=0.98, rely=0.03, anchor="ne")

    def request_abort(self):
        self.abort_requested = True
        self.status_var.set("Abort requested. Saving collected data...")

    def update_canvas(self, phase: str, direction: str | None = None):
        self.canvas.delete("all")

        width = max(self.canvas.winfo_width(), 800)
        height = max(self.canvas.winfo_height(), 600)
        cx, cy = width // 2, height // 2

        if phase == "fixation":
            self.draw_crosshair(cx, cy)
        elif phase == "left":
            self.draw_arrow(cx, cy, "LEFT")
        elif phase == "right":
            self.draw_arrow(cx, cy, "RIGHT")
        elif phase == "rest":
            self.draw_rest_icon(cx, cy)
        elif phase == "warmup":
            self.draw_breathe_icon(cx, cy)
        elif phase == "saving":
            self.draw_save_icon(cx, cy)
        elif phase == "done":
            self.draw_done_icon(cx, cy)
        else:
            self.draw_crosshair(cx, cy)

    def draw_crosshair(self, cx, cy):
        size = 70
        self.canvas.create_line(cx - size, cy, cx + size, cy, fill="#ffffff", width=6)
        self.canvas.create_line(cx, cy - size, cx, cy + size, fill="#ffffff", width=6)
        self.canvas.create_oval(cx - 10, cy - 10, cx + 10, cy + 10, outline="#ffffff", width=3)

    def draw_arrow(self, cx, cy, direction):
        # Big simple icon. No dependency on image files.
        if direction == "LEFT":
            points = [
                cx - 230, cy,
                cx - 70, cy - 130,
                cx - 70, cy - 55,
                cx + 220, cy - 55,
                cx + 220, cy + 55,
                cx - 70, cy + 55,
                cx - 70, cy + 130,
            ]
            label = "←"
        else:
            points = [
                cx + 230, cy,
                cx + 70, cy - 130,
                cx + 70, cy - 55,
                cx - 220, cy - 55,
                cx - 220, cy + 55,
                cx + 70, cy + 55,
                cx + 70, cy + 130,
            ]
            label = "→"

        self.canvas.create_polygon(points, fill="#ffffff", outline="#ffffff")
        if not MINIMAL_TRIAL_TEXT:
            self.canvas.create_text(
                cx,
                cy + 210,
                text=f"IMAGINE {direction}",
                fill="#ffffff",
                font=("Segoe UI", 42, "bold"),
            )
        else:
            self.canvas.create_text(
                cx,
                cy + 210,
                text=label,
                fill="#777777",
                font=("Segoe UI", 44, "bold"),
            )

    def draw_rest_icon(self, cx, cy):
        r = 90
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#ffffff", width=8)
        self.canvas.create_line(cx, cy, cx, cy - 55, fill="#ffffff", width=8)
        self.canvas.create_line(cx, cy, cx + 42, cy + 35, fill="#ffffff", width=8)

    def draw_breathe_icon(self, cx, cy):
        for r in [55, 100, 145]:
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#ffffff", width=3)
        if not MINIMAL_TRIAL_TEXT:
            self.canvas.create_text(cx, cy + 220, text="SETTLE", fill="#ffffff", font=("Segoe UI", 36, "bold"))

    def draw_save_icon(self, cx, cy):
        self.canvas.create_rectangle(cx - 110, cy - 120, cx + 110, cy + 120, outline="#ffffff", width=8)
        self.canvas.create_rectangle(cx - 65, cy - 85, cx + 65, cy - 15, fill="#ffffff", outline="#ffffff")
        self.canvas.create_rectangle(cx - 65, cy + 35, cx + 65, cy + 85, outline="#ffffff", width=6)

    def draw_done_icon(self, cx, cy):
        self.canvas.create_line(cx - 140, cy, cx - 35, cy + 100, fill="#ffffff", width=18)
        self.canvas.create_line(cx - 35, cy + 100, cx + 160, cy - 130, fill="#ffffff", width=18)

    def sleep_with_countdown(self, seconds: float, phase_name: str):
        start = time.time()
        while True:
            elapsed = time.time() - start
            remaining = seconds - elapsed
            if remaining <= 0:
                break
            if self.abort_requested:
                break

            self.countdown_var.set(f"{remaining:0.1f}s")
            self.phase_var.set(phase_name)
            time.sleep(0.05)

        self.countdown_var.set("")

    def run_daq_session(self):
        data = None

        try:
            self.root.after(0, lambda: self.status_var.set("Preparing OpenBCI session..."))
            self.root.after(0, lambda: self.update_canvas("warmup"))

            params = BrainFlowInputParams()
            params.serial_port = SERIAL_PORT

            BoardShim.enable_dev_board_logger()
            self.board = BoardShim(BOARD_ID, params)

            self.board.prepare_session()
            self.board.start_stream()

            self.root.after(0, lambda: self.status_var.set("Warmup: relax and keep still"))
            self.root.after(0, lambda: self.update_canvas("warmup"))
            self.sleep_with_countdown(WARMUP_SEC, "Warmup")

            total_trials = len(self.trial_labels)

            for trial_idx, label in enumerate(self.trial_labels, start=1):
                if self.abort_requested:
                    break

                label_name = label_to_name(label)

                self.root.after(0, lambda i=trial_idx, t=total_trials: self.trial_var.set(f"Trial {i}/{t}"))
                self.root.after(0, lambda: self.status_var.set(""))
                self.root.after(0, lambda: self.update_canvas("fixation"))
                self.sleep_with_countdown(FIXATION_SEC, "Fixation")

                if self.abort_requested:
                    break

                # Marker inserted exactly at cue onset.
                self.board.insert_marker(label)

                cue_phase = "left" if label_name == "LEFT" else "right"
                self.root.after(0, lambda p=cue_phase: self.update_canvas(p))
                self.sleep_with_countdown(IMAGERY_SEC, "Imagine")

                if self.abort_requested:
                    break

                self.root.after(0, lambda: self.update_canvas("rest"))
                self.sleep_with_countdown(REST_SEC, "Rest")

            time.sleep(1)
            self.root.after(0, lambda: self.status_var.set("Stopping stream..."))
            data = self.board.get_board_data()

        except Exception as exc:
            self.root.after(0, lambda e=exc: messagebox.showerror("DAQ error", str(e)))
            try:
                if self.board is not None:
                    data = self.board.get_board_data()
            except Exception:
                data = None

        finally:
            try:
                if self.board is not None:
                    self.board.stop_stream()
                    self.board.release_session()
            except Exception:
                pass

        if data is not None and data.size > 0:
            self.root.after(0, lambda: self.update_canvas("saving"))
            self.root.after(0, lambda: self.status_var.set("Filtering and saving..."))

            try:
                file_path, summary = self.process_and_save(data)
                self.root.after(0, lambda: self.show_done_screen(file_path, summary))
            except Exception as exc:
                self.root.after(0, lambda e=exc: messagebox.showerror("Save error", str(e)))
                self.root.after(0, self.build_start_screen)
        else:
            self.root.after(0, lambda: messagebox.showwarning("No data", "No data was collected."))
            self.root.after(0, self.build_start_screen)

    def process_and_save(self, data):
        eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
        marker_channel = BoardShim.get_marker_channel(BOARD_ID)

        eeg_raw = data[eeg_channels]
        markers = data[marker_channel]

        eeg_filtered = eeg_raw.copy()

        print("\nFiltering...")
        for ch in tqdm(range(eeg_filtered.shape[0])):
            DataFilter.perform_bandpass(
                eeg_filtered[ch],
                FS,
                0.5,
                45.0,
                4,
                FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
                0,
            )

            DataFilter.remove_environmental_noise(
                eeg_filtered[ch],
                FS,
                NoiseTypes.FIFTY.value,
            )

        marker_indices = np.where(markers > 0)[0]
        marker_values = markers[marker_indices].astype(int)

        pre_samp = int(FIXATION_SEC * FS)
        post_samp = int(IMAGERY_SEC * FS)

        X_raw = []
        X_filtered = []
        y = []
        kept_marker_indices = []

        for idx, label in zip(marker_indices, marker_values):
            if label not in [MARKERS["LEFT"], MARKERS["RIGHT"]]:
                continue

            start_idx = idx - pre_samp
            end_idx = idx + post_samp

            if start_idx < 0 or end_idx > eeg_raw.shape[1]:
                continue

            X_raw.append(eeg_raw[:, start_idx:end_idx].copy())
            X_filtered.append(eeg_filtered[:, start_idx:end_idx].copy())
            y.append(label)
            kept_marker_indices.append(idx)

        X_raw = np.array(X_raw)
        X_filtered = np.array(X_filtered)
        y = np.array(y)
        kept_marker_indices = np.array(kept_marker_indices)

        csp_start = int((FIXATION_SEC + CSP_START_AFTER_CUE) * FS)
        csp_end = int((FIXATION_SEC + CSP_END_AFTER_CUE) * FS)

        if len(X_filtered) > 0:
            X_csp = X_filtered[:, :, csp_start:csp_end]
        else:
            X_csp = np.empty((0, len(eeg_channels), csp_end - csp_start))

        metadata = asdict(self.session_info)

        file_path = os.path.join(
            self.subject_folder,
            f"{self.session_info.session_name}.npz",
        )

        np.savez(
            file_path,
            continuous_eeg_raw=eeg_raw,
            continuous_eeg_filtered=eeg_filtered,
            continuous_markers=markers,
            X_raw=X_raw,
            X_filtered=X_filtered,
            X_csp=X_csp,
            y=y,
            fs=FS,
            eeg_channels=np.array(eeg_channels),
            marker_channel=marker_channel,
            marker_indices=kept_marker_indices,
            trial_order=np.array(self.trial_labels),
            fixation_sec=FIXATION_SEC,
            imagery_sec=IMAGERY_SEC,
            rest_sec=REST_SEC,
            warmup_sec=WARMUP_SEC,
            csp_start_after_cue=CSP_START_AFTER_CUE,
            csp_end_after_cue=CSP_END_AFTER_CUE,
            metadata=np.array(metadata, dtype=object),
        )

        summary = {
            "continuous_raw_shape": eeg_raw.shape,
            "continuous_filtered_shape": eeg_filtered.shape,
            "markers_shape": markers.shape,
            "X_raw_shape": X_raw.shape,
            "X_filtered_shape": X_filtered.shape,
            "X_csp_shape": X_csp.shape,
            "y_shape": y.shape,
            "left_trials": int((y == MARKERS["LEFT"]).sum()),
            "right_trials": int((y == MARKERS["RIGHT"]).sum()),
        }

        print("\nFinal shapes:")
        for key, value in summary.items():
            print(f"{key}: {value}")

        print("\nSaved:")
        print(file_path)

        return file_path, summary

    def show_done_screen(self, file_path, summary):
        self.clear()

        frame = tk.Frame(self.root, bg="#050505")
        frame.pack(expand=True, fill="both")

        canvas = tk.Canvas(frame, bg="#050505", highlightthickness=0, height=260)
        canvas.pack(fill="x", pady=(50, 10))
        canvas.update()
        cx = self.root.winfo_width() // 2
        cy = 130
        canvas.create_line(cx - 100, cy, cx - 25, cy + 75, fill="#ffffff", width=14)
        canvas.create_line(cx - 25, cy + 75, cx + 120, cy - 95, fill="#ffffff", width=14)

        title = tk.Label(
            frame,
            text="Session saved",
            bg="#050505",
            fg="#ffffff",
            font=("Segoe UI", 30, "bold"),
        )
        title.pack(pady=10)

        path_label = tk.Label(
            frame,
            text=file_path,
            bg="#050505",
            fg="#aaaaaa",
            font=("Consolas", 12),
        )
        path_label.pack(pady=10)

        summary_text = (
            f"X_raw: {summary['X_raw_shape']}    "
            f"X_filtered: {summary['X_filtered_shape']}    "
            f"X_csp: {summary['X_csp_shape']}\n"
            f"Left: {summary['left_trials']}    Right: {summary['right_trials']}"
        )

        summary_label = tk.Label(
            frame,
            text=summary_text,
            bg="#050505",
            fg="#dddddd",
            font=("Consolas", 14),
        )
        summary_label.pack(pady=15)

        button_row = tk.Frame(frame, bg="#050505")
        button_row.pack(pady=25)

        new_button = ttk.Button(
            button_row,
            text="New Session",
            command=self.build_start_screen,
        )
        new_button.grid(row=0, column=0, padx=10)

        quit_button = ttk.Button(
            button_row,
            text="Quit",
            command=self.root.destroy,
        )
        quit_button.grid(row=0, column=1, padx=10)


def main():
    root = tk.Tk()
    app = MotorImageryDAQGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
