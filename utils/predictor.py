"""
utils/predictor.py
Model architecture definitions + load + predict logic.
Copy-paste dari training script agar konsisten.
"""

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLINICAL_FEATURES_FULL = [
    "Total skor IPSS", "Skor Storage", "Skor Voiding", "QoL",
    "Usia", "Volume Prostat", "IPP",
    "Diabetes", "Stroke", "Neurology abnormalities"
]
CLINICAL_FEATURES_REDUCED = [
    "Skor Voiding", "Skor Storage", "Usia", "Total skor IPSS", "Volume Prostat"
]
TABULAR_FEATURES = [
    "Qmax_ml_s", "Qave_ml_s", "Voided_Volume_ml",
    "PVR_ml", "Flow_Time_s", "Time_to_Max_Flow_s"
]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES (identik dengan training)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock1d(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + res)


class WaveEncoderResidual(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.stage1 = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            ResidualBlock1d(64, 3, dropout * 0.5), nn.MaxPool1d(2)
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            ResidualBlock1d(128, 3, dropout * 0.5),
            ResidualBlock1d(128, 3, dropout * 0.5), nn.MaxPool1d(2)
        )
        self.stage3 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            ResidualBlock1d(256, 3, dropout * 0.5),
        )
        self.attn = nn.Sequential(nn.Conv1d(256, 64, 1), nn.Tanh(), nn.Conv1d(64, 1, 1))
        self.proj = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x); x = self.stage1(x); x = self.stage2(x)
        fm = self.stage3(x)
        aw = torch.softmax(self.attn(fm), dim=-1)
        return self.proj((fm * aw).sum(dim=-1))


class ModelA(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.enc  = WaveEncoderResidual(dropout)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                   nn.Dropout(dropout * 0.5), nn.Linear(64, 2))
    def forward(self, w, t=None, c=None): return self.head(self.enc(w))


class ModelC(nn.Module):
    def __init__(self, dropout=0.3, n_cli=10):
        super().__init__()
        self.wave = WaveEncoderResidual(dropout)
        self.tab  = nn.Sequential(nn.Linear(6, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(32, 16), nn.ReLU())
        self.cli  = nn.Sequential(nn.Linear(n_cli, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(32, 16), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(160, 64), nn.ReLU(), nn.Dropout(dropout))
        self.head   = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, w, t, c):
        e = torch.cat([self.wave(w), self.tab(t), self.cli(c)], dim=1)
        return self.head(self.fusion(e))


class ModelD(nn.Module):
    def __init__(self, dropout=0.3, n_cli_red=5):
        super().__init__()
        self.wave = WaveEncoderResidual(dropout)
        self.tab  = nn.Sequential(nn.Linear(6, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(32, 16), nn.ReLU())
        self.cli  = nn.Sequential(nn.Linear(n_cli_red, 16), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(16, 8), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(152, 64), nn.ReLU(), nn.Dropout(dropout))
        self.head   = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, w, t, c):
        e = torch.cat([self.wave(w), self.tab(t), self.cli(c)], dim=1)
        return self.head(self.fusion(e))


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_key: str):
    """
    model_key: 'A', 'C', atau 'D'
    Return: (model, scalers, config) atau raise FileNotFoundError
    """
    key = model_key.lower()
    pt_path  = MODELS_DIR / f"model_{key}_locked_v2.pt"
    pkl_path = MODELS_DIR / f"model_{key}_scaler_v2.pkl"
    cfg_path = MODELS_DIR / f"model_{key}_config_v2.json"

    # Fallback ke v1 filename jika v2 tidak ada
    if not pt_path.exists():
        pt_path  = MODELS_DIR / f"model_{key}_locked.pt"
        pkl_path = MODELS_DIR / f"model_{key}_scaler.pkl"
        cfg_path = MODELS_DIR / f"model_{key}_config.json"

    if not pt_path.exists():
        raise FileNotFoundError(f"Model {model_key} tidak ditemukan di {MODELS_DIR}")

    with open(cfg_path) as f:
        config = json.load(f)

    with open(pkl_path, 'rb') as f:
        scalers = pickle.load(f)

    # Instantiate model
    if model_key == 'A':
        model = ModelA()
    elif model_key == 'C':
        model = ModelC(n_cli=len(CLINICAL_FEATURES_FULL))
    elif model_key == 'D':
        model = ModelD(n_cli_red=len(CLINICAL_FEATURES_REDUCED))
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    state = torch.load(pt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, scalers, config


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING + PREDIKSI
# ─────────────────────────────────────────────────────────────────────────────

def determine_model_route(ufm_params: dict, clinical_full: dict, clinical_reduced: dict) -> str:
    """
    Tentukan model mana yang dipakai berdasarkan kelengkapan data.
    
    Return: 'C', 'D', atau 'A'
    """
    # Cek apakah 10 fitur klinis tersedia (minimal 7/10 tidak NaN)
    full_vals = [clinical_full.get(f, np.nan) for f in CLINICAL_FEATURES_FULL]
    n_full_valid = sum(1 for v in full_vals if not np.isnan(float(v) if v is not None else np.nan))

    if n_full_valid >= 7:
        return 'C'

    # Cek apakah 5 fitur reduced tersedia (minimal 4/5)
    red_vals = [clinical_reduced.get(f, np.nan) for f in CLINICAL_FEATURES_REDUCED]
    n_red_valid = sum(1 for v in red_vals if not np.isnan(float(v) if v is not None else np.nan))

    if n_red_valid >= 4:
        return 'D'

    return 'A'


def predict(
    waveform: np.ndarray,
    ufm_params: dict,
    clinical_full: dict,
    model_key: str = None,
) -> dict:
    """
    Jalankan prediksi BOO.
    
    Args:
        waveform: array [280] float32
        ufm_params: dict dengan TABULAR_FEATURES keys
        clinical_full: dict dengan CLINICAL_FEATURES_FULL keys
        model_key: override model ('A','C','D'), None = auto-route
    
    Return: dict dengan keys:
        model_used, prob_boo, pred_label, confidence,
        threshold, youden_threshold, needs_urodynamics
    """
    # Build clinical reduced dari full
    clinical_reduced = {f: clinical_full.get(f, np.nan) for f in CLINICAL_FEATURES_REDUCED}

    # Auto-route jika tidak di-override
    if model_key is None:
        model_key = determine_model_route(ufm_params, clinical_full, clinical_reduced)

    # Load model
    model, scalers, config = load_model(model_key)
    threshold = config.get('youden_threshold', 0.5)

    # Prepare arrays
    Xw = waveform.reshape(1, -1).astype(np.float32)

    Xt = None
    if model_key in ('B', 'C', 'D'):
        tab_vals = np.array([
            ufm_params.get(f, np.nan) for f in TABULAR_FEATURES
        ], dtype=np.float32).reshape(1, -1)
        it, st = scalers.get('tab', (SimpleImputer(strategy='median'), StandardScaler()))
        # Jika scaler sudah fitted, transform langsung
        try:
            Xt = st.transform(it.transform(tab_vals)).astype(np.float32)
        except:
            Xt = tab_vals  # fallback raw jika scaler belum fitted

    Xc = None
    if model_key == 'C':
        cli_feats = CLINICAL_FEATURES_FULL
        cli_vals = np.array([
            clinical_full.get(f, np.nan) for f in cli_feats
        ], dtype=np.float32).reshape(1, -1)
        ic, sc = scalers.get('cli', (SimpleImputer(strategy='median'), StandardScaler()))
        try:
            Xc = sc.transform(ic.transform(cli_vals)).astype(np.float32)
        except:
            Xc = cli_vals

    elif model_key == 'D':
        cli_feats = CLINICAL_FEATURES_REDUCED
        cli_vals = np.array([
            clinical_reduced.get(f, np.nan) for f in cli_feats
        ], dtype=np.float32).reshape(1, -1)
        ic, sc = scalers.get('cli', (SimpleImputer(strategy='median'), StandardScaler()))
        try:
            Xc = sc.transform(ic.transform(cli_vals)).astype(np.float32)
        except:
            Xc = cli_vals

    # Inference
    with torch.no_grad():
        w_t = torch.tensor(Xw).to(device)
        t_t = torch.tensor(Xt).to(device) if Xt is not None else None
        c_t = torch.tensor(Xc).to(device) if Xc is not None else None
        logits = model(w_t, t_t, c_t)
        prob_boo = torch.softmax(logits, dim=1)[0, 1].item()

    pred_boo = prob_boo >= threshold

    # Confidence level
    if prob_boo >= 0.85 or prob_boo <= 0.15:
        confidence = "TINGGI"
        confidence_color = "#22c55e"
    elif prob_boo >= 0.70 or prob_boo <= 0.30:
        confidence = "SEDANG"
        confidence_color = "#f59e0b"
    else:
        confidence = "RENDAH"
        confidence_color = "#ef4444"

    # Saran UDS
    needs_urodynamics = (
        confidence == "RENDAH" or
        model_key == 'A' or
        (0.35 <= prob_boo <= 0.65)
    )

    # Clinical recommendation
    if pred_boo and confidence != "RENDAH":
        clinical_rec = "Alpha-Blocker Trial"
        clinical_detail = "Inisiasi terapi medikamentosa dapat dilakukan. Pertimbangkan pemeriksaan lanjutan jika tidak ada respons dalam 4–6 minggu."
    elif not pred_boo and confidence != "RENDAH":
        clinical_rec = "Evaluasi Detrusor Underactivity"
        clinical_detail = "Tidak ditemukan pola obstruksi signifikan. Pertimbangkan evaluasi kontraktilitas otot detrusor."
    else:
        clinical_rec = "Pemeriksaan Lanjutan Diperlukan"
        clinical_detail = "Hasil inkonklusif. Pemeriksaan urodinamika (UDS) direkomendasikan untuk konfirmasi."

    return {
        "model_used": f"Model {model_key}",
        "model_description": {
            'A': "Waveform Only",
            'C': "Waveform + UFM + Klinis Lengkap",
            'D': "Waveform + UFM + Klinis Ringkas (5 fitur)",
        }[model_key],
        "prob_boo": prob_boo,
        "pred_boo": pred_boo,
        "pred_label": "BOO" if pred_boo else "Non-BOO",
        "confidence": confidence,
        "confidence_color": confidence_color,
        "threshold": threshold,
        "needs_urodynamics": needs_urodynamics,
        "clinical_recommendation": clinical_rec,
        "clinical_detail": clinical_detail,
        "model_key": model_key,
        "model_auc_cv": {"A": 0.52, "C": 0.856, "D": 0.838}[model_key],
        "model_auc_ext": {"A": 0.54, "C": 0.812, "D": 0.667}[model_key],
    }