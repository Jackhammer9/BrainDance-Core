import numpy as np
import torch
import torch.nn as nn
from scipy.signal import welch
import cv2
from brainflow import DataFilter, FilterTypes, AggOperations
from brainflow.board_shim import BoardShim, BrainFlowInputParams
import time
def noise_augmentation(X):
    N, C, T = X.shape # n windows, n channels, n timepoints

    rms = np.zeros((N, C))
    peak = np.zeros((N, C))
    spike = np.zeros(N)
    corr_mean = np.zeros(N)

    for i in range(N):
        w = X[i]

        rms[i] = np.sqrt(np.mean(w**2, axis=1))
        peak[i] = np.max(np.abs(w), axis=1)

        diff = np.diff(w, axis=1)
        spike[i] = np.max(np.abs(diff))

        corr = np.corrcoef(w)
        upper = corr[np.triu_indices_from(corr, k=1)]
        corr_mean[i] = np.mean(np.abs(upper))

    rms_mu, rms_std = rms.mean(), rms.std()
    peak_mu, peak_std = peak.mean(), peak.std()
    spike_mu, spike_std = spike.mean(), spike.std()

    bad_rms   = np.any(rms > rms_mu + 6 * rms_std, axis=1)
    bad_peak  = np.any(peak > peak_mu + 10 * peak_std, axis=1)
    bad_spike = spike > spike_mu + 10 * spike_std
    bad_corr  = corr_mean > 0.95

    bad = bad_rms | bad_peak | bad_spike | bad_corr
    good = ~bad

    X_clean = X[good]

    X_clean = np.clip(X_clean, -300, 300)  # µV

    return X_clean
def extract_band_features_per_channel(X, fs=250):
    bands = {"delta": (0.5, 4),"theta": (4, 8),"alpha": (8, 13),"beta":  (13, 30),"gamma": (30, 45),}
    N, C, T = X.shape
    features = []
    feature_names = []

    # ---------- FEATURE NAMES ----------
    for ch in range(C):
        prefix = f"ch{ch}"

        # spectral powers
        for band in bands:
            feature_names.append(f"{prefix}_{band}_power")
        for band in bands:
            feature_names.append(f"{prefix}_log_{band}_power")

        # ratios + complexity
        feature_names += [f"{prefix}_alpha_relative",f"{prefix}_alpha_beta_ratio",f"{prefix}_theta_alpha_ratio",f"{prefix}_beta_alpha_ratio",f"{prefix}_spectral_entropy",f"{prefix}_log_total_power"]

        # time-domain stats (NEW)
        feature_names += [f"{prefix}_time_mean",f"{prefix}_time_std",f"{prefix}_time_var",]

    # ---------- FEATURE EXTRACTION ----------
    for i in range(N):
        feat_vec = []

        for ch in range(C):
            signal = X[i, ch]

            # ----- PSD -----
            f, Pxx = welch(signal, fs=fs, nperseg=fs // 2)

            band_powers = {}
            total_power = 0.0

            for band, (fmin, fmax) in bands.items():
                mask = (f >= fmin) & (f <= fmax)
                power = np.trapezoid(Pxx[mask], f[mask])
                band_powers[band] = power
                total_power += power

            # ----- raw band powers -----
            for band in bands:
                feat_vec.append(band_powers[band])

            # ----- log band powers -----
            for band in bands:
                feat_vec.append(np.log(band_powers[band] + 1e-8))

            # ----- ratios -----
            alpha_rel   = band_powers["alpha"] / (total_power + 1e-8)
            alpha_beta  = band_powers["alpha"] / (band_powers["beta"] + 1e-8)
            theta_alpha = band_powers["theta"] / (band_powers["alpha"] + 1e-8)
            beta_alpha  = band_powers["beta"]  / (band_powers["alpha"] + 1e-8)

            # ----- spectral entropy -----
            Pxx_norm = Pxx / (Pxx.sum() + 1e-8)
            spec_entropy = -np.sum(Pxx_norm * np.log(Pxx_norm + 1e-8))

            log_total_power = np.log(total_power + 1e-8)

            feat_vec += [
                alpha_rel,
                alpha_beta,
                theta_alpha,
                beta_alpha,
                spec_entropy,
                log_total_power,
            ]

            # ----- time-domain stats (NEW) -----
            feat_vec += [
                signal.mean(),
                signal.std(),
                signal.var(),
            ]

        features.append(feat_vec)

    return np.array(features), feature_names
def build_lstm_sequences_no_y(X, seq_len):
    N, C, T = X.shape

    X_seq = []

    for i in range(N - seq_len + 1):
        seq = X[i:i + seq_len]              # (seq_len, C, T)
        seq_flat = seq.reshape(seq_len, -1) # (seq_len, C*T)
        X_seq.append(seq_flat)

    return np.array(X_seq)
class EyeClassifier(nn.Module):
    def __init__(
        self,
        feat_dim,        # number of handcrafted features (e.g. 12)
        raw_dim,         # raw EEG per timestep (e.g. C*T or reduced)
        lstm_hidden=64
    ):
        super().__init__()

        # -------- Feature branch (band powers etc.) --------
        self.feature_mlp = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # -------- Raw EEG branch (temporal dynamics) --------
        self.lstm = nn.LSTM(
            input_size=raw_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )

        # -------- Fusion + classifier --------
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden + 16, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x_feat, x_raw):
        """
        x_feat: (B, T, feat_dim)   or (B, feat_dim) if already pooled
        x_raw:  (B, T, raw_dim)
        """

        # ---- Feature branch ----
        if x_feat.dim() == 3:
            feat_emb = self.feature_mlp(x_feat)   # (B, T, 32)
            feat_last = feat_emb[:, -1, :]        # (B, 32)
        else:
            feat_last = self.feature_mlp(x_feat)  # (B, 32)

        # ---- Raw EEG branch ----
        lstm_out, _ = self.lstm(x_raw)             # (B, T, H)
        lstm_last = lstm_out[:, -1, :]             # (B, H)

        # ---- Fusion ----
        fused = torch.cat([feat_last, lstm_last], dim=1)
        return self.classifier(fused)
    
BOARD_ID = 0
PORT = "COM3"
FS = 250
params = BrainFlowInputParams()
params.serial_port = PORT

BoardShim.enable_dev_board_logger()
board = BoardShim(BOARD_ID, params)
board.prepare_session()
board.start_stream()

print("Noise Rejection Under Process...")
time.sleep(5)


cam = cv2.VideoCapture(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = np.load("Eye Classifier/assets/model_scaler.npz")
feat_mean = scaler["feat_mean"]
feat_std = scaler["feat_std"]

ckpt = torch.load("Eye Classifier/assets/eye_model.pth", weights_only=False)

model = EyeClassifier(
    feat_dim=ckpt["feat_mean"].shape[-1],
    raw_dim=2000
).to(device)

model.load_state_dict(ckpt["model_state"])

old_state = ""

while True:

    #Brainflow Stuff
    data = board.get_current_board_data(750)

    if data.shape[1] < 750:
        continue

    eeg_data = data[board.get_eeg_channels(BOARD_ID)]

    #Filtering Data
    test_X = []
    for ch in range(eeg_data.shape[0]):
        DataFilter.perform_bandpass(eeg_data[ch], FS, 0.5, 40.0, 4, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(eeg_data[ch], FS, 48.0, 52.0, 4, FilterTypes.BUTTERWORTH.value, 0)
        for i in range(0, eeg_data.shape[1] - 250, 250):
            test_X.append(eeg_data[:, i:i + 250])

    test_X = np.array(test_X)  # (N, 8, 250)
    # Noise Augmentation

    clean_X = noise_augmentation(test_X)
    if clean_X.shape[0] >= 3:
        pass
    else:
        continue

    # Feature Extraction
    features, _ = extract_band_features_per_channel(clean_X, fs=FS)
    #Build sequences
    features_seq = features[np.newaxis, :, :]  # (1, 3, num_features)
    features_seq = (features_seq - feat_mean) / feat_std

    X_seq = build_lstm_sequences_no_y(clean_X, seq_len=3)[-1:] 
        
    # Pytorch Inference
    with torch.no_grad():
        X_feat_tensor = torch.tensor(features_seq, dtype=torch.float32).to(device)
        X_raw_tensor  = torch.tensor(X_seq, dtype=torch.float32).to(device)

        outputs = model(X_feat_tensor, X_raw_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        logits = outputs.cpu().numpy()

    #Show Results on web camera feed
    output_text = "Eyes Closed" if preds[-1] == 0 else "Eyes Open"

    if output_text != old_state:
        old_state = output_text
        print(outputs)
        print(output_text)