"""
utils/waveform_extractor.py
Ekstrak waveform uroflowmetry dari file .DTA atau PDF.
"""

import struct
import re
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pathlib import Path


TARGET_LENGTH = 280


# ─────────────────────────────────────────────────────────────────────────────
# DARI .DTA (binary Laborie format)
# ─────────────────────────────────────────────────────────────────────────────

def extract_from_dta(filepath) -> np.ndarray | None:
    """
    Baca file .DTA binary Laborie, ambil channel Flow terbaik.
    Return: normalized waveform array [280] atau None jika gagal.
    """
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()

        if b'UDS' not in raw[:10]:
            return None

        all_labels = [b'Flow\x00', b'Volume\x00', b'CALQ\x00',
                      b'Pdet\x00', b'Pves\x00', b'Pabd\x00']
        all_pos = sorted(set(
            m.start() for lb in all_labels
            for m in re.finditer(re.escape(lb), raw)
        ))

        flow_positions = [m.start() for m in re.finditer(re.escape(b'Flow\x00'), raw)]
        best_flow = None
        best_score = -1

        for fp in flow_positions:
            next_ch = next((p for p in all_pos if p > fp + 10), len(raw))
            pos = fp + 5
            while pos < len(raw) and raw[pos] != 0:
                pos += 1
            pos += 1

            data = raw[pos:next_ch]
            n = len(data) // 2
            if n < 10:
                continue

            vals = np.array(struct.unpack(f'<{n}H', data[:n * 2]), dtype=float)
            vals[vals > 600] = 0
            flow_mls = vals * 0.1

            runs = []
            in_run = False
            for i, v in enumerate(flow_mls):
                if v > 0.1 and not in_run:
                    in_run = True
                    s = i
                elif v <= 0.1 and in_run:
                    in_run = False
                    runs.append((s, i))
            if in_run:
                runs.append((s, len(flow_mls)))

            valid = [(s, e) for s, e in runs
                     if (e - s) >= 10 and 1.0 <= flow_mls[s:e].max() <= 60.0]
            if not valid:
                continue

            best_run = max(valid, key=lambda x: x[1] - x[0])
            score = best_run[1] - best_run[0]
            if score > best_score:
                best_score = score
                best_flow = flow_mls[best_run[0]:best_run[1]]

        return _preprocess_wave(best_flow) if best_flow is not None else None

    except Exception as e:
        print(f"[DTA] Error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DARI PDF (Laborie Golden Report — cari gambar plot atau tabel flow)
# ─────────────────────────────────────────────────────────────────────────────

def extract_from_pdf(filepath) -> tuple[np.ndarray | None, dict]:
    """
    Ekstrak waveform dari PDF uroflowmetry + UFM parameters.
    
    Strategy:
    1. Coba extract embedded DTA filename dari teks PDF → load .DTA
    2. Fallback: render halaman → crop area plot → ekstrak kurva dari gambar
    3. Extract UFM params (Qmax, Qave, VV, PVR, dll) dari teks PDF
    
    Return: (waveform_array or None, ufm_params dict)
    """
    import fitz  # PyMuPDF

    ufm_params = {
        'Qmax_ml_s': np.nan,
        'Qave_ml_s': np.nan,
        'Voided_Volume_ml': np.nan,
        'PVR_ml': np.nan,
        'Flow_Time_s': np.nan,
        'Time_to_Max_Flow_s': np.nan,
    }

    try:
        doc = fitz.open(str(filepath))
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        # ── Extract UFM parameters dari teks ─────────────────────────────
        patterns = {
            'Qmax_ml_s': [
                r'Qmax[^:=\n]*[:\s=]+(\d+\.?\d*)\s*ml',
                r'Maximum\s+flow[^:=\n]*[:\s=]+(\d+\.?\d*)',
            ],
            'Qave_ml_s': [
                r'Qave[^:=\n]*[:\s=]+(\d+\.?\d*)\s*ml',
                r'Average\s+flow[^:=\n]*[:\s=]+(\d+\.?\d*)',
            ],
            'Voided_Volume_ml': [
                r'V(?:oided)?\s*V(?:ol(?:ume)?)?[^:=\n]*[:\s=]+(\d+\.?\d*)\s*ml',
                r'VV[^:=\n]*[:\s=]+(\d+\.?\d*)',
            ],
            'PVR_ml': [
                r'PVR[^:=\n]*[:\s=]+(\d+\.?\d*)\s*ml',
                r'Post\s*void\s*resid[^:=\n]*[:\s=]+(\d+\.?\d*)',
            ],
            'Flow_Time_s': [
                r'Flow\s*time[^:=\n]*[:\s=]+(\d+:?\d*\.?\d*)\s*(?:s|mm)',
            ],
            'Time_to_Max_Flow_s': [
                r'Time\s*to\s*max[^:=\n]*[:\s=]+(\d+:?\d*\.?\d*)\s*(?:s|mm)',
            ],
        }

        for key, pats in patterns.items():
            for pat in pats:
                m = re.search(pat, full_text, re.IGNORECASE)
                if m:
                    raw = m.group(1).strip()
                    if ':' in raw:
                        parts = raw.split(':')
                        ufm_params[key] = float(parts[0]) * 60 + float(parts[1])
                    else:
                        try:
                            ufm_params[key] = float(raw)
                        except:
                            pass
                    break

        # ── Waveform: coba render + image processing ──────────────────────
        waveform = _extract_waveform_from_pdf_image(filepath)
        return waveform, ufm_params

    except Exception as e:
        print(f"[PDF] Error: {e}")
        return None, ufm_params


def _extract_waveform_from_pdf_image(filepath) -> np.ndarray | None:
    """
    Render PDF → crop area flow plot → ekstrak kurva via pixel scanning.
    Ini heuristik, works best untuk Laborie Golden Report standard format.
    """
    try:
        import fitz
        from PIL import Image
        import io

        doc = fitz.open(str(filepath))
        page = doc[0]

        # Render halaman dengan resolusi tinggi
        mat = fitz.Matrix(3.0, 3.0)  # 3x zoom
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()

        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = img.size
        img_arr = np.array(img)

        # Cari area plot: biasanya 30-70% height, 20-80% width
        # Deteksi area dengan kurva biru/hitam pada background putih
        crop_y1 = int(h * 0.30)
        crop_y2 = int(h * 0.70)
        crop_x1 = int(w * 0.15)
        crop_x2 = int(w * 0.85)

        plot_region = img_arr[crop_y1:crop_y2, crop_x1:crop_x2]

        # Deteksi piksel kurva (gelap pada background terang)
        # Threshold: piksel dengan brightness < 100 = bagian kurva
        gray = np.mean(plot_region, axis=2)
        ph, pw = gray.shape

        # Per kolom x, cari y terendah (terbawah) dari kurva
        flow_signal = []
        for col in range(pw):
            col_data = gray[:, col]
            dark_rows = np.where(col_data < 80)[0]
            if len(dark_rows) > 0:
                # Y dalam plot (0 = atas, ph = bawah) → invert untuk flow
                y_pos = dark_rows.mean()
                flow_val = (ph - y_pos) / ph  # normalized 0-1
                flow_signal.append(max(0, flow_val))
            else:
                flow_signal.append(0.0)

        flow_signal = np.array(flow_signal, dtype=np.float32)

        # Filter noise
        if len(flow_signal) > 11:
            flow_signal = savgol_filter(flow_signal, 11, 3)
            flow_signal = np.clip(flow_signal, 0, 1)

        # Cari bagian yang benar-benar ada flow (> noise floor)
        threshold = 0.05
        valid_mask = flow_signal > threshold
        if valid_mask.sum() < 10:
            return None

        # Crop ke bagian valid
        indices = np.where(valid_mask)[0]
        flow_cropped = flow_signal[indices[0]:indices[-1] + 1]

        # Normalize
        mx = flow_cropped.max()
        if mx <= 0:
            return None
        flow_norm = flow_cropped / mx

        return _preprocess_wave(flow_norm * 10)  # scale back untuk preprocess

    except Exception as e:
        print(f"[PDF Image] Error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING SHARED
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_wave(flow_mls, target=TARGET_LENGTH) -> np.ndarray | None:
    """Smooth, normalize, dan resample ke target length."""
    if flow_mls is None or len(flow_mls) < 10:
        return None

    # Savitzky-Golay smoothing
    win = min(11, len(flow_mls) if len(flow_mls) % 2 == 1 else len(flow_mls) - 1)
    if win >= 5:
        flow_s = savgol_filter(flow_mls, win, 3)
        flow_s = np.clip(flow_s, 0, None)
    else:
        flow_s = flow_mls.copy()

    mx = flow_s.max()
    if mx <= 0:
        return None

    flow_n = flow_s / mx

    # Resample ke target length
    f_interp = interp1d(
        np.linspace(0, 1, len(flow_n)), flow_n,
        kind='linear', fill_value=0, bounds_error=False
    )
    return f_interp(np.linspace(0, 1, target)).astype(np.float32)


def get_waveform_source_label(has_dta: bool) -> str:
    return "DTA Binary (Akurasi Tinggi)" if has_dta else "PDF Image Extraction (Estimasi)"