"""
utils/predictor.py — v2.2
Load scaler dari JSON (version-agnostic) bukan pickle.
Fix: tidak ada lagi sklearn version mismatch di production.
"""

import json, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predictor")

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

# Fitur yang nilai 0 dianggap belum diisi
_MUST_BE_POSITIVE = {"Usia", "Volume Prostat", "Total skor IPSS",
                     "Qmax_ml_s", "Voided_Volume_ml"}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES
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
        self.wave   = WaveEncoderResidual(dropout)
        self.tab    = nn.Sequential(nn.Linear(6, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(32, 16), nn.ReLU())
        self.cli    = nn.Sequential(nn.Linear(n_cli, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(32, 16), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(160, 64), nn.ReLU(), nn.Dropout(dropout))
        self.head   = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, w, t, c):
        return self.head(self.fusion(torch.cat([self.wave(w), self.tab(t), self.cli(c)], dim=1)))


class ModelD(nn.Module):
    def __init__(self, dropout=0.3, n_cli_red=5):
        super().__init__()
        self.wave   = WaveEncoderResidual(dropout)
        self.tab    = nn.Sequential(nn.Linear(6, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(32, 16), nn.ReLU())
        self.cli    = nn.Sequential(nn.Linear(n_cli_red, 16), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(16, 8), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(152, 64), nn.ReLU(), nn.Dropout(dropout))
        self.head   = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, w, t, c):
        return self.head(self.fusion(torch.cat([self.wave(w), self.tab(t), self.cli(c)], dim=1)))


# ─────────────────────────────────────────────────────────────────────────────
# SCALER — JSON based (version-agnostic, no pickle)
# ─────────────────────────────────────────────────────────────────────────────

class JsonScaler:
    """
    Minimal StandardScaler + SimpleImputer dari JSON params.
    Tidak tergantung versi scikit-learn sama sekali.
    """
    def __init__(self, imputer_statistics, scaler_mean, scaler_scale):
        self.statistics_ = np.array(imputer_statistics, dtype=np.float64)
        self.mean_        = np.array(scaler_mean,        dtype=np.float64)
        self.scale_       = np.array(scaler_scale,       dtype=np.float64)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute NaN → scale → return float32."""
        X = np.array(X, dtype=np.float64).copy()
        # Impute: ganti NaN dengan median dari training
        for j in range(X.shape[1]):
            nan_mask = np.isnan(X[:, j])
            if nan_mask.any():
                X[nan_mask, j] = self.statistics_[j]
        # Standardize
        X = (X - self.mean_) / (self.scale_ + 1e-8)
        return X.astype(np.float32)


def _load_scalers_from_json(model_key: str) -> dict:
    """
    Load scaler params dari JSON — tidak ada sklearn dependency.
    Return dict: {'tab': JsonScaler, 'cli': JsonScaler}
    """
    key      = model_key.lower()
    json_path = MODELS_DIR / f"model_{key}_scaler_params.json"

    if not json_path.exists():
        # Fallback: coba load dari pkl (untuk backward compat)
        return _load_scalers_from_pkl(model_key)

    with open(json_path) as f:
        data = json.load(f)

    scalers = {}
    for sname, params in data.items():
        scalers[sname] = JsonScaler(
            imputer_statistics = params['imputer_statistics'],
            scaler_mean        = params['scaler_mean'],
            scaler_scale       = params['scaler_scale'],
        )
    logger.info(f"Scalers loaded from JSON for Model {model_key}: {list(scalers.keys())}")
    return scalers


def _load_scalers_from_pkl(model_key: str) -> dict:
    """Fallback: load dari pickle (bisa error jika versi sklearn berbeda)."""
    import pickle
    key      = model_key.lower()
    pkl_path = MODELS_DIR / f"model_{key}_scaler_v2.pkl"

    if not pkl_path.exists():
        pkl_path = MODELS_DIR / f"model_{key}_scaler.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Scaler untuk Model {model_key} tidak ditemukan.\n"
            f"Cari: {MODELS_DIR}/model_{key}_scaler_params.json (JSON) atau _scaler_v2.pkl"
        )

    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)

    # Convert ke JsonScaler-compatible dict
    scalers = {}
    for sname, (imputer, scaler) in raw.items():
        scalers[sname] = JsonScaler(
            imputer_statistics = imputer.statistics_.tolist(),
            scaler_mean        = scaler.mean_.tolist(),
            scaler_scale       = scaler.scale_.tolist(),
        )
    logger.info(f"Scalers loaded from PKL (fallback) for Model {model_key}")
    return scalers


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_key: str):
    key      = model_key.lower()
    pt_path  = MODELS_DIR / f"model_{key}_locked_v2.pt"
    cfg_path = MODELS_DIR / f"model_{key}_config_v2.json"

    if not pt_path.exists():
        pt_path  = MODELS_DIR / f"model_{key}_locked.pt"
        cfg_path = MODELS_DIR / f"model_{key}_config.json"

    if not pt_path.exists():
        raise FileNotFoundError(
            f"Model {model_key} tidak ditemukan di {MODELS_DIR}\n"
            f"Pastikan model_{key}_locked_v2.pt ada di folder models/"
        )

    with open(cfg_path) as f:
        config = json.load(f)

    # Load scalers dari JSON (no sklearn version issue)
    scalers = _load_scalers_from_json(model_key)

    if   model_key == 'A': model = ModelA()
    elif model_key == 'C': model = ModelC(n_cli=len(CLINICAL_FEATURES_FULL))
    elif model_key == 'D': model = ModelD(n_cli_red=len(CLINICAL_FEATURES_REDUCED))
    else: raise ValueError(f"Unknown model key: {model_key}")

    state = torch.load(pt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    logger.info(f"Model {model_key} loaded | threshold={config.get('youden_threshold', 0.5):.3f}")
    return model, scalers, config


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────────────────────

def _is_valid_val(val, feat_name: str) -> bool:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return False
    if np.isnan(v):
        return False
    if feat_name in _MUST_BE_POSITIVE and v <= 0:
        return False
    return True


def determine_model_route(ufm_params: dict, clinical_full: dict, clinical_reduced: dict) -> str:
    n_full = sum(1 for f in CLINICAL_FEATURES_FULL
                 if _is_valid_val(clinical_full.get(f), f))
    if n_full >= 7:
        return 'C'

    n_red = sum(1 for f in CLINICAL_FEATURES_REDUCED
                if _is_valid_val(clinical_full.get(f, clinical_reduced.get(f)), f))
    if n_red >= 4:
        return 'D'

    return 'A'


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────

def predict(waveform, ufm_params, clinical_full, model_key=None):
    clinical_reduced = {f: clinical_full.get(f, np.nan) for f in CLINICAL_FEATURES_REDUCED}

    if model_key is None:
        model_key = determine_model_route(ufm_params, clinical_full, clinical_reduced)

    logger.info(f"Predict: model_key={model_key}")

    model, scalers, config = load_model(model_key)
    threshold = config.get('youden_threshold', 0.5)

    # Prepare inputs
    Xw = waveform.reshape(1, -1).astype(np.float32)

    Xt = None
    if model_key in ('B', 'C', 'D'):
        tab_arr = np.array([ufm_params.get(f, np.nan) for f in TABULAR_FEATURES],
                           dtype=np.float32).reshape(1, -1)
        Xt = scalers['tab'].transform(tab_arr)

    Xc = None
    if model_key == 'C':
        cli_arr = np.array([clinical_full.get(f, np.nan) for f in CLINICAL_FEATURES_FULL],
                           dtype=np.float32).reshape(1, -1)
        Xc = scalers['cli'].transform(cli_arr)
    elif model_key == 'D':
        cli_arr = np.array([clinical_reduced.get(f, np.nan) for f in CLINICAL_FEATURES_REDUCED],
                           dtype=np.float32).reshape(1, -1)
        Xc = scalers['cli'].transform(cli_arr)

    # Inference
    with torch.no_grad():
        w_t = torch.tensor(Xw).to(device)
        t_t = torch.tensor(Xt).to(device) if Xt is not None else None
        c_t = torch.tensor(Xc).to(device) if Xc is not None else None
        prob_boo = torch.softmax(model(w_t, t_t, c_t), dim=1)[0, 1].item()

    logger.info(f"prob_boo={prob_boo:.4f}, threshold={threshold:.3f}")

    pred_boo = prob_boo >= threshold

    # Safety net: IPSS voiding berat tapi model bilang Non-BOO
    skor_voiding = float(clinical_full.get("Skor Voiding", 0) or 0)
    ipss_total   = float(clinical_full.get("Total skor IPSS", 0) or 0)
    safety_net   = (not pred_boo and skor_voiding >= 10 and ipss_total >= 18)

    # Confidence
    if   prob_boo >= 0.85 or prob_boo <= 0.15: conf = "TINGGI"; conf_col = "#22c55e"
    elif prob_boo >= 0.70 or prob_boo <= 0.30: conf = "SEDANG"; conf_col = "#f59e0b"
    else:                                        conf = "RENDAH"; conf_col = "#ef4444"

    needs_uds = (
        conf == "RENDAH" or model_key == 'A' or
        (0.35 <= prob_boo <= 0.65) or safety_net
    )

    if pred_boo and conf != "RENDAH":
        rec = "Alpha-Blocker Trial"
        det = "Inisiasi terapi medikamentosa dapat dilakukan. Pertimbangkan pemeriksaan lanjutan jika tidak ada respons dalam 4–6 minggu."
    elif safety_net:
        rec = "UDS Direkomendasikan — Gejala Berat"
        det = f"Model memprediksi Non-BOO, namun skor voiding ({int(skor_voiding)}) dan IPSS ({int(ipss_total)}) tergolong berat. Konfirmasi UDS diperlukan."
    elif not pred_boo and conf != "RENDAH":
        rec = "Evaluasi Detrusor Underactivity"
        det = "Tidak ditemukan pola obstruksi signifikan. Pertimbangkan evaluasi kontraktilitas otot detrusor."
    else:
        rec = "Pemeriksaan Lanjutan Diperlukan"
        det = "Hasil inkonklusif. Pemeriksaan urodinamika (UDS) direkomendasikan untuk konfirmasi."

    return {
        "model_used":              f"Model {model_key}",
        "model_description":       {'A': "Waveform Only",
                                    'C': "Waveform + UFM + Klinis Lengkap",
                                    'D': "Waveform + UFM + Klinis Ringkas (5 fitur)"}[model_key],
        "prob_boo":                prob_boo,
        "pred_boo":                pred_boo,
        "pred_label":              "BOO" if pred_boo else "Non-BOO",
        "confidence":              conf,
        "confidence_color":        conf_col,
        "threshold":               threshold,
        "needs_urodynamics":       needs_uds,
        "safety_net_triggered":    safety_net,
        "clinical_recommendation": rec,
        "clinical_detail":         det,
        "model_key":               model_key,
        "model_auc_cv":            {"A": 0.52, "C": 0.836, "D": 0.873}[model_key],
        "model_auc_ext":           {"A": 0.44, "C": 0.828, "D": 0.703}[model_key],
    }